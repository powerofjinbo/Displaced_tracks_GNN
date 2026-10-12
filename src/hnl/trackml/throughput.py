"""TrackML throughput phase (Zenodo 4730157): the tar parts streamed over HTTP, their event manifest, the compact
tables and their structural verification (docs/trackml_plan.md sections 1.4 (D1, D2), 2.1, 2.4 and 6.0).

The parts training_part01 ... 09 are tar archives of gzipped csv members event<id>-{cells,hits,particles,truth}.csv.gz,
events from 21000 on. D1 = throughput and D2 = train_1-sized (the user's decisions): training_part01 and
training_part02 are used. No archive is stored anywhere: each part is read sequentially over HTTP (resumed with a
Range request after a connection error) and the md5 of every byte read is compared with the Zenodo listing
(hnl.trackml.split.TP_ARCHIVES).

  scan     pass 1: stream a part; record every member (event, kind, stream position, offsets, size, md5 of its bytes,
           gzip ISIZE) in <manifest>/<part>.members.tsv, and the archive's size and md5 verdict in
           manifest_tp_summary.json. The split file is made from these ids (hnl.trackml.split make-tp).
  convert  pass 2: stream the part again; check every member's md5 against pass 1; convert each complete event with
           hnl.trackml.convert.event_tables (the v1 columns, derived quantities and data-quality rule) in worker
           processes; write the shards as *.tmp; publish them (rename) only if the archive's md5 equals Zenodo's and
           at most MAX_FAIL_FRAC of the events failed. Otherwise the tmp files are removed and the job stops.
  verify   the published shards against the meta file, the manifest, the split and the converter's csv row counts:
           sha256, schema, one row group per event in increasing order, the stored events, row counts. No physics
           value is read, so it is safe for val and test events.

Differences from the accuracy-phase conversion (plan section 2.1; everything else is hnl.trackml.convert unchanged):
  * input: tar of csv.gz, streamed once per pass (no central directory, no stored archive); gzip.decompress checks
    each member's CRC-32 and length;
  * particles.csv of the throughput data has an extra column particle_type (PDG id). The tables keep the v1 schema
    (categories stay mass-hypothesis based, plan section 3.3), and particle_type goes to a side table `ptype` (event,
    particle_id, particle_type; one row per particles row, same order);
  * event sets from configs/trackml/split_tp_v1.json: score weights stored for the dev, calib, val and test events;
    truth kinematics for the calib events (the 100 throughput dev ids + the next 100 training ids, plan section 6.0);
  * score weights: in the throughput data part of the secondaries carry weight 0 (its score leaves secondaries out);
    they are stored as in the csv;
  * the beam spot (55 mm), the corrected electron scattering and the strip thickness change no conversion step.

Output: <root>/{hits,particles,truth,ptype}/<part>/shard_<first:05d>.parquet (first = event // 500 * 500, as v1),
meta_<part>.json and dq_<part>.tsv; read with hnl.trackml.convert.read_table(table, events, root=<root>).

    python -m hnl.trackml.throughput scan --part training_part01 [--manifest DIR]
    python -m hnl.trackml.throughput convert --part training_part01 [--root DIR] [--nproc N] [--overwrite]
    python -m hnl.trackml.throughput verify --part training_part01 [--root DIR]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import http.client
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hnl.data.convert import SHARD
from hnl.trackml import convert as C
from hnl.trackml.split import EXT, REPO, TP_ARCHIVES, TP_RECORD, tp_subsets

DERIVED_TP = os.path.join(EXT, "derived", "tp_v1")
MANIFEST = os.path.join(EXT, "manifest")
SUMMARY = "manifest_tp_summary.json"
SPLIT_TP = os.path.join(REPO, "configs", "trackml", "split_tp_v1.json")
MAX_FAIL_FRAC = 0.01                  # more than 1 % of the events failing or corrupted: stop (task rule)
TABLE_BYTES_PER_EVENT = 2.6e6         # storage projection per event (hits + particles + ptype + truth share)
MEMBER_COLUMNS = ["archive", "event", "kind", "member_index", "offset_header", "offset_data", "size", "md5", "isize",
                  "name"]
PTYPE_SCHEMA = pa.schema([("event", pa.int32()), ("particle_id", pa.int64()), ("particle_type", pa.int32())])
SCHEMAS = {**C.SCHEMAS, "ptype": PTYPE_SCHEMA}
TABLES = ("hits", "particles", "truth", "ptype")
WRITE_OPTS = {**C.WRITE_OPTS, "ptype": {"use_dictionary": ["event", "particle_type"],
                                        "column_encoding": {"particle_id": "DELTA_BINARY_PACKED"}}}
CSV_DTYPES = {**C.CSV_DTYPES,
              "particles": {"particle_id": np.int64, "particle_type": np.int64, "vx": np.float64, "vy": np.float64,
                            "vz": np.float64, "px": np.float64, "py": np.float64, "pz": np.float64, "q": np.int64,
                            "nhits": np.int64}}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------ HTTP stream
class HTTPStream(io.RawIOBase):
    """Sequential read-only view of a remote file over HTTP. Every byte returned is fed to an md5 in order. After a
    connection error or a premature end the transfer resumes at the current byte with a Range request (Zenodo
    answers 206)."""

    RESUME_SLEEP = 5.0                 # seconds per resume so far, at most 60

    def __init__(self, url: str, timeout: float = 120.0, max_resumes: int = 50, chunk: int = 4 << 20):
        super().__init__()
        self.url, self.timeout, self.max_resumes, self.chunk = url, timeout, max_resumes, chunk
        self.pos = 0
        self.size: int | None = None
        self.md5 = hashlib.md5()
        self.resumes = 0
        self.resp = None
        self._open()

    def _open(self):
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        resp = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout)
        if self.pos:
            cr = resp.headers.get("Content-Range", "")
            if resp.status != 206 or not cr.startswith(f"bytes {self.pos}-"):
                resp.close()
                raise RuntimeError(f"resume at byte {self.pos}: HTTP {resp.status}, Content-Range {cr!r}")
        else:
            self.size = int(resp.headers["Content-Length"])
        self.resp = resp

    def _resume(self, err: BaseException):
        while True:
            self.resumes += 1
            if self.resumes > self.max_resumes:
                raise RuntimeError(f"{self.url}: more than {self.max_resumes} resumes") from err
            print(f"[stream] {type(err).__name__}: {err}; resuming at byte {self.pos} (resume {self.resumes})",
                  flush=True)
            try:
                if self.resp is not None:
                    self.resp.close()
            except Exception:
                pass
            time.sleep(min(60.0, self.RESUME_SLEEP * self.resumes))
            try:
                self._open()
                return
            except (OSError, http.client.HTTPException, RuntimeError) as e:
                err = e

    def readable(self):
        return True

    def read(self, n=-1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        out, got = [], 0
        while got < n and self.pos < self.size:
            try:
                b = self.resp.read(min(n - got, self.chunk))
            except (OSError, http.client.HTTPException) as e:
                self._resume(e)
                continue
            if not b:
                self._resume(EOFError(f"connection closed at byte {self.pos} of {self.size}"))
                continue
            self.md5.update(b)
            self.pos += len(b)
            got += len(b)
            out.append(b)
        return b"".join(out)

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

    def drain(self):
        """Read (and hash) the rest of the file, e.g. the tar end blocks after the last member."""
        while self.pos < self.size:
            self.read(min(self.chunk, self.size - self.pos))

    def close(self):
        try:
            if self.resp is not None:
                self.resp.close()
        finally:
            super().close()


# ------------------------------------------------------------------------------------------------ tar members
def member_event_kind(name: str) -> tuple[int, str] | None:
    """(event, kind) of a member named [dir/]event<id>-<kind>.csv.gz, else None."""
    base = name.rsplit("/", 1)[-1]
    if not (base.startswith("event") and base.endswith(".csv.gz")):
        return None
    ev, _, kind = base[5:-7].partition("-")
    if not ev.isdigit() or kind not in C.KINDS:
        return None
    return int(ev), kind


def iter_members(fileobj):
    """(TarInfo, bytes or None) of every member of a tar read sequentially from fileobj (stream mode: no seek)."""
    tf = tarfile.open(fileobj=fileobj, mode="r|", bufsize=1 << 20)
    try:
        for ti in tf:
            yield ti, (tf.extractfile(ti).read() if ti.isreg() else None)
    finally:
        tf.close()


def gzip_isize(data: bytes) -> int:
    """ISIZE field of a gzip member (uncompressed length mod 2^32), -1 if too short."""
    return int.from_bytes(data[-4:], "little") if len(data) >= 18 else -1


# ------------------------------------------------------------------------------------------------ manifest
def manifest_paths(part: str, manifest_dir: str = MANIFEST) -> tuple[str, str]:
    return os.path.join(manifest_dir, f"{part}.members.tsv"), os.path.join(manifest_dir, SUMMARY)


def read_manifest(part: str, manifest_dir: str = MANIFEST) -> pd.DataFrame:
    tsv, _ = manifest_paths(part, manifest_dir)
    return pd.read_csv(tsv, sep="\t", dtype={"md5": str, "name": str})


def complete_events(man: pd.DataFrame) -> list[int]:
    k = man.groupby("event").kind.apply(lambda s: tuple(sorted(s)))
    return sorted(int(e) for e, v in k.items() if v == tuple(sorted(C.KINDS)))


def read_summary(part: str, manifest_dir: str = MANIFEST) -> dict:
    with open(manifest_paths(part, manifest_dir)[1]) as fh:
        return json.load(fh)["archives"][part + ".tar"]


def scan_part(part: str, manifest_dir: str = MANIFEST, url: str | None = None) -> dict:
    """Pass 1 (module docstring): the member table and the size / md5 verdict of one streamed part."""
    name = part + ".tar"
    size_z, md5_z = TP_ARCHIVES[name]
    url = url or TP_RECORD.format(name=name)
    tsv, summ_path = manifest_paths(part, manifest_dir)
    os.makedirs(manifest_dir, exist_ok=True)
    started, t0 = utc(), time.time()
    st = HTTPStream(url)
    print(f"[scan] {part}: {st.size} bytes at {url}", flush=True)
    rows, other, dup = [], [], []
    seen: dict[int, set] = {}
    order_events: list[int] = []
    for i, (ti, data) in enumerate(iter_members(st)):
        ek = member_event_kind(ti.name)
        if ek is None or data is None:
            other.append({"member_index": i, "name": ti.name, "size": int(ti.size)})
            continue
        ev, kind = ek
        if kind in seen.setdefault(ev, set()):
            dup.append(ti.name)
        seen[ev].add(kind)
        if not order_events or order_events[-1] != ev:
            order_events.append(ev)
        rows.append((part, ev, kind, i, int(ti.offset), int(ti.offset_data), int(ti.size),
                     hashlib.md5(data).hexdigest(), gzip_isize(data), ti.name))
        if len(rows) % 400 == 0:
            el = time.time() - t0
            print(f"[scan] {part}: {len(rows)} members, {len(seen)} events, {st.pos / 1e9:.2f} GB, "
                  f"{st.pos / 1e6 / max(el, 1e-9):.0f} MB/s", flush=True)
    st.drain()
    el = time.time() - t0
    md5_s = st.md5.hexdigest()
    complete = sorted(e for e, k in seen.items() if k == set(C.KINDS))
    grouped = len(order_events) == len(set(order_events))
    increasing = all(b > a for a, b in zip(order_events, order_events[1:]))
    res = {
        "archive": name, "record": "10.5281/zenodo.4730157", "url": url,
        "bytes_zenodo": size_z, "md5_zenodo": md5_z, "bytes_streamed": int(st.pos), "md5_streamed": md5_s,
        "size_match": int(st.pos) == size_z and st.size == size_z, "md5_match": md5_s == md5_z,
        "n_members": len(rows), "n_events": len(seen), "n_complete_events": len(complete),
        "first_event": min(seen) if seen else None, "last_event": max(seen) if seen else None,
        "incomplete_events": sorted(set(seen) - set(complete)), "duplicate_members": dup, "other_members": other,
        "members_grouped_by_event": grouped, "events_in_increasing_order": increasing,
        "compressed_bytes_by_kind": {k: int(sum(r[6] for r in rows if r[2] == k)) for k in C.KINDS},
        "uncompressed_bytes_by_kind_isize": {k: int(sum(r[8] for r in rows if r[2] == k)) for k in C.KINDS},
        "stream_seconds": round(el, 1), "mb_per_s": round(st.pos / 1e6 / max(el, 1e-9), 1), "resumes": st.resumes,
        "started_utc": started, "finished_utc": utc(), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "members_tsv": os.path.basename(tsv),
    }
    st.close()
    pd.DataFrame(rows, columns=MEMBER_COLUMNS).to_csv(tsv + ".tmp", sep="\t", index=False)
    os.replace(tsv + ".tmp", tsv)
    with open(tsv, "rb") as fh:
        res["members_tsv_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    summ = {"record": "10.5281/zenodo.4730157", "archives": {}}
    if os.path.exists(summ_path):
        with open(summ_path) as fh:
            summ = json.load(fh)
    summ["archives"][name] = res
    summ["archives"] = dict(sorted(summ["archives"].items()))
    with open(summ_path + ".tmp", "w") as fh:
        json.dump(summ, fh, indent=1)
    os.replace(summ_path + ".tmp", summ_path)
    return res


# ------------------------------------------------------------------------------------------------ split
def load_tp_split(path: str = SPLIT_TP) -> dict:
    """The throughput split file as sets (dev, val, valmon, test, train, calib, gbdt, merge_train), with its subsets
    recomputed by hnl.trackml.split.tp_subsets as a check."""
    with open(path) as fh:
        s = json.load(fh)
    if s.get("phase") != "throughput":
        raise ValueError(f"{path} is not a throughput split file")
    sub = tp_subsets(s["train"], s["dev"])
    for k, v in sub.items():
        if s[k] != v:
            raise RuntimeError(f"{path}: {k} differs from tp_subsets")
    held = set(s["val"]) | set(s["test"])
    if held & set(s["train"]) or set(s["val"]) & set(s["test"]):
        raise RuntimeError(f"{path}: val / test overlap train or each other")
    out = {k: set(int(e) for e in s[k]) for k in ("dev", "val", "valmon", "test", "train", "calib", "gbdt",
                                                   "merge_train")}
    out.update({"path": path, "sha256": C.sha256_file(path), "parts": list(s["parts"]), "archives": s["archives"]})
    return out


# ------------------------------------------------------------------------------------------------ conversion
_TW: dict = {}


def _tp_init(det_path: str):
    _TW["det"] = C.load_detectors(det_path)


def parse_members(members: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    """The four csv members of one event (gzip bytes) as DataFrames with the expected columns."""
    raw = {}
    for k in C.KINDS:
        df = pd.read_csv(io.BytesIO(gzip.decompress(members[k])), dtype=CSV_DTYPES[k], engine="c")
        if list(df.columns) != list(CSV_DTYPES[k]):
            raise ValueError(f"{k}: columns {list(df.columns)}")
        raw[k] = df
    return raw


def ptype_table(event: int, p: pd.DataFrame) -> pa.Table:
    t = p.particle_type.to_numpy()
    if len(t) and (t.min() < -2 ** 31 or t.max() >= 2 ** 31):
        raise ValueError("particle_type outside int32")
    return pa.Table.from_arrays([pa.array(np.full(len(p), event, dtype=np.int32)),
                                 pa.array(p.particle_id.to_numpy().astype(np.int64)),
                                 pa.array(t.astype(np.int32))], schema=PTYPE_SCHEMA)


def event_tables_tp(members: dict[str, bytes], event: int, det: dict, keep_weight: bool, keep_truth: bool):
    """(tables or None, data-quality record) of one throughput event: hnl.trackml.convert.event_tables on the parsed
    csv members plus the ptype side table; csv row counts recorded for the structural checks."""
    dq = {"event": int(event), "max_outside_mm": float("nan"), "problems": [], "csv_hits": -1, "csv_cells": -1,
          "csv_particles": -1, "csv_truth": -1, "csv_truth_nonnoise": -1}
    try:
        raw = parse_members(members)
        dq.update(csv_hits=len(raw["hits"]), csv_cells=len(raw["cells"]), csv_particles=len(raw["particles"]),
                  csv_truth=len(raw["truth"]))
        if keep_truth:
            dq["csv_truth_nonnoise"] = int((raw["truth"].particle_id.to_numpy() != 0).sum())
        tables, d = C.event_tables(raw, event, det, keep_weight, keep_truth)
        dq["max_outside_mm"], dq["problems"] = d["max_outside_mm"], list(d["problems"])
        if tables is not None:
            tables["ptype"] = ptype_table(event, raw["particles"])
    except Exception as e:                       # a corrupt member or an unexpected format: a failed event
        tables = None
        dq["problems"].append(f"exception {type(e).__name__}: {e}")
    return tables, dq


def _tp_convert(args):
    event, members, keep_weight, keep_truth = args
    t0 = time.time()
    tables, dq = event_tables_tp(members, event, _TW["det"], keep_weight, keep_truth)
    dq["seconds"] = round(time.time() - t0, 2)
    return event, tables, dq


def iter_events(stream, man: pd.DataFrame | None = None):
    """(event, {kind: gzip bytes}) of every complete event of a streamed part, in stream order. With the pass-1
    manifest, every member's position, size and md5 must agree (RuntimeError otherwise: corrupt stream)."""
    ref = None
    if man is not None:
        ref = {(int(r.event), r.kind): (int(r.member_index), int(r.size), r.md5) for r in man.itertuples()}
    buf: dict[int, dict[str, bytes]] = {}
    for i, (ti, data) in enumerate(iter_members(stream)):
        ek = member_event_kind(ti.name)
        if ek is None or data is None:
            continue
        ev, kind = ek
        if ref is not None:
            want = ref.get((ev, kind))
            got = (i, len(data), hashlib.md5(data).hexdigest())
            if want != got:
                raise RuntimeError(f"member {ti.name}: stream {got} differs from the pass-1 manifest {want}")
        buf.setdefault(ev, {})[kind] = data
        if len(buf[ev]) == len(C.KINDS):
            yield ev, buf.pop(ev)


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def convert_part(part: str, root: str = DERIVED_TP, nproc: int = 1, level: int = 9, overwrite: bool = False,
                 manifest_dir: str = MANIFEST, split_path: str = SPLIT_TP, det_path: str = C.DETECTORS,
                 url: str | None = None) -> dict:
    """Pass 2 (module docstring) of one part."""
    t0, started = time.time(), utc()
    name = part + ".tar"
    size_z, md5_z = TP_ARCHIVES[name]
    url = url or TP_RECORD.format(name=name)
    summ = read_summary(part, manifest_dir)
    if not (summ["size_match"] and summ["md5_match"]):
        raise RuntimeError(f"{part}: pass 1 did not verify size and md5 ({summ['md5_streamed']}); not converted")
    man = read_manifest(part, manifest_dir)
    complete = complete_events(man)
    split = load_tp_split(split_path)
    if part not in split["parts"]:
        raise RuntimeError(f"{part} is not a part of {split_path}")
    known = split["dev"] | split["val"] | split["test"] | split["train"]
    unknown = [e for e in complete if e not in known]
    if unknown:
        raise RuntimeError(f"{len(unknown)} events of {part} are in no set of {split_path}: {unknown[:10]}")
    firsts = sorted({e // SHARD * SHARD for e in complete})
    finals = [C.shard_path(root, k, part, f) for f in firsts for k in TABLES]
    if any(os.path.exists(f) for f in finals) and not overwrite:
        raise FileExistsError(f"shards of {part} exist under {root} (use --overwrite to redo them)")
    _rm([f + ".tmp" for f in finals])                       # stale tmp files of an interrupted run of this code
    st0 = C.storage_check(len(complete) * TABLE_BYTES_PER_EVENT)
    print(f"[convert] {part}: {len(complete)} events, shards {firsts}; storage {st0}", flush=True)
    weight_ev = split["calib"] | split["val"] | split["test"]
    writers: dict[str, tuple[int, pq.ParquetWriter, str]] = {}
    written: dict[tuple[int, str], list[int]] = {}
    tmp_files: list[str] = []
    dq_rows: list[dict] = []
    storage_log = [st0]
    nproc = max(int(nproc), 1)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(nproc, initializer=_tp_init, initargs=(det_path,))     # before the socket exists
    stream = None

    def handle(res):
        e, tables, dq = res.get()
        dq["archive"] = part
        dq["status"] = "ok" if tables is not None else (
            "failed" if any(p.startswith("exception") for p in dq["problems"]) else "dropped")
        dq_rows.append(dq)
        if tables is None:
            print(f"[convert] event {e} {dq['status'].upper()}: {dq['problems']}", flush=True)
            return
        first = e // SHARD * SHARD
        for k in TABLES:
            if k not in tables:
                continue
            w = writers.get(k)
            if w is None or w[0] != first:
                if w is not None:
                    w[1].close()
                if k == "hits":
                    storage_log.append(C.storage_check(SHARD * TABLE_BYTES_PER_EVENT))
                tmp = C.shard_path(root, k, part, first) + ".tmp"
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                tmp_files.append(tmp)
                writers[k] = (first, pq.ParquetWriter(tmp, SCHEMAS[k], compression="zstd", compression_level=level,
                                                      write_statistics=True, **WRITE_OPTS[k]), tmp)
            writers[k][1].write_table(tables[k], row_group_size=max(tables[k].num_rows, 1))
            written.setdefault((first, k), []).append(e)

    n_seen, last = 0, -1
    try:
        stream = HTTPStream(url)
        if stream.size != size_z:
            raise RuntimeError(f"{part}: {stream.size} bytes on Zenodo, {size_z} expected")
        pending = collections.deque()
        for ev, members in iter_events(stream, man):
            if ev <= last:
                raise RuntimeError(f"{part}: events not in increasing order ({last} then {ev})")
            last = ev
            n_seen += 1
            pending.append(pool.apply_async(_tp_convert, ((ev, members, ev in weight_ev, ev in split["calib"]),)))
            while len(pending) >= 3 * nproc:
                handle(pending.popleft())
            if n_seen % 100 == 0:
                el = time.time() - t0
                print(f"[convert] {part}: {n_seen} events streamed, {stream.pos / 1e9:.2f} GB, "
                      f"{stream.pos / 1e6 / max(el, 1e-9):.0f} MB/s", flush=True)
        while pending:
            handle(pending.popleft())
        stream.drain()
    except BaseException:
        for w in writers.values():
            w[1].close()
        _rm(tmp_files)
        pool.terminate()
        raise
    pool.close()
    pool.join()
    for w in writers.values():
        w[1].close()
    md5_s = stream.md5.hexdigest()
    n_bytes, resumes = int(stream.pos), stream.resumes
    stream.close()
    stream_ok = n_bytes == size_z and md5_s == md5_z
    seen = sorted(d["event"] for d in dq_rows)
    failed = [d for d in dq_rows if d["status"] != "ok"]
    verdict = {"md5_streamed": md5_s, "bytes_streamed": n_bytes, "md5_match": md5_s == md5_z,
               "size_match": n_bytes == size_z, "events_seen": len(seen), "events_expected": len(complete),
               "events_failed": len(failed), "max_fail_frac": MAX_FAIL_FRAC}
    print(f"[convert] {part}: {json.dumps(verdict)}", flush=True)
    stop = None
    if not stream_ok:
        stop = f"archive size / md5 mismatch in pass 2 ({n_bytes} B, {md5_s}; Zenodo {size_z} B, {md5_z})"
    elif seen != complete:
        stop = f"pass 2 saw {len(seen)} complete events, the manifest lists {len(complete)}"
    elif len(failed) > MAX_FAIL_FRAC * len(complete):
        stop = f"{len(failed)} of {len(complete)} events failed or were dropped (> {MAX_FAIL_FRAC:.0%})"
    if stop:
        _rm(tmp_files)
        raise RuntimeError(f"STOP {part}: {stop}; nothing published")
    shard_meta: dict[int, dict] = {}
    for (first, k), evs in sorted(written.items()):
        final = C.shard_path(root, k, part, first)
        os.replace(final + ".tmp", final)
        shard_meta.setdefault(first, {})[k] = {"file": os.path.relpath(final, root), "events": evs,
                                               "bytes": os.path.getsize(final), "sha256": C.sha256_file(final)}
    res = {
        "archive": part, "phase": "throughput", "source": url, "record": "10.5281/zenodo.4730157",
        "source_bytes": size_z, "source_md5_zenodo": md5_z, "source_md5_verified": md5_s,
        "pass1_md5": summ["md5_streamed"], "member_md5_checked_against_pass1": int(len(man)),
        "manifest": {"tsv": os.path.relpath(manifest_paths(part, manifest_dir)[0], EXT),
                     "sha256": summ.get("members_tsv_sha256")},
        "converted_utc": utc(), "started_utc": started, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "code_sha256": {"throughput.py": C.sha256_file(os.path.abspath(__file__)),
                        "convert.py": C.sha256_file(os.path.abspath(C.__file__))},
        "split": {"path": os.path.relpath(split["path"], REPO), "sha256": split["sha256"]},
        "detectors_sha256": C.sha256_file(det_path), "shard_rule": f"first = event // {SHARD} * {SHARD}",
        "zstd_level": level, "nproc": nproc,
        "n_events_in_archive": len(complete), "incomplete_events_in_archive": summ["incomplete_events"],
        "n_events_converted": sum(len(m["hits"]["events"]) for m in shard_meta.values() if "hits" in m),
        "dropped": [{"event": d["event"], "status": d["status"], "problems": d["problems"]} for d in failed],
        "weight_events": "dev + calib + val + test of the split file (plan section 6.0)",
        "truth_events": "calib of the split file: the 100 throughput dev ids + the next 100 training ids (plan 6.0)",
        "ptype": "particle_type (PDG id) of every particles row, in the particles row order (throughput csv only)",
        "stream": {"seconds": round(time.time() - t0, 1), "resumes": resumes},
        "storage_checks": storage_log,
        "shards": {str(k): v for k, v in sorted(shard_meta.items())},
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"meta_{part}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    cols = ["archive", "event", "status", "problems", "max_outside_mm", "seconds", "csv_hits", "csv_cells",
            "csv_particles", "csv_truth", "csv_truth_nonnoise"]
    dq = pd.DataFrame([{**d, "problems": "; ".join(d["problems"]),
                        "max_outside_mm": round(d["max_outside_mm"], 4)} for d in dq_rows])[cols]
    dq.sort_values("event").to_csv(os.path.join(root, f"dq_{part}.tsv"), sep="\t", index=False)
    return res


# ------------------------------------------------------------------------------------------------ verification
def verify_part(part: str, root: str = DERIVED_TP, manifest_dir: str = MANIFEST, split_path: str = SPLIT_TP) -> dict:
    """Structure of the stored shards (module docstring); reads parquet metadata and file bytes only."""
    with open(os.path.join(root, f"meta_{part}.json")) as fh:
        m = json.load(fh)
    split = load_tp_split(split_path)
    man = read_manifest(part, manifest_dir)
    complete = complete_events(man)
    dq = pd.read_csv(os.path.join(root, f"dq_{part}.tsv"), sep="\t").set_index("event")
    dropped = {d["event"] for d in m["dropped"]}
    problems = []
    seen = {k: [] for k in TABLES}
    rows = {k: {} for k in TABLES}
    for first, sh in m["shards"].items():
        for k in ("particles", "ptype"):
            if sh.get("hits", {}).get("events") != sh.get(k, {}).get("events"):
                problems.append(f"shard {first}: hits and {k} hold different events")
        for k, info in sh.items():
            path = os.path.join(root, info["file"])
            if C.sha256_file(path) != info["sha256"]:
                problems.append(f"{info['file']}: sha256 differs from meta")
            f = pq.ParquetFile(path)
            if not f.schema_arrow.equals(SCHEMAS[k], check_metadata=False):
                problems.append(f"{info['file']}: schema differs from the v1 schema")
            ic = f.schema_arrow.get_field_index("event")
            evs = []
            for i in range(f.metadata.num_row_groups):
                rg = f.metadata.row_group(i)
                s = rg.column(ic).statistics
                if s is None or s.min != s.max:
                    problems.append(f"{info['file']}: row group {i} holds several events")
                    continue
                evs.append(int(s.min))
                rows[k][int(s.min)] = int(rg.num_rows)
            if evs != info["events"]:
                problems.append(f"{info['file']}: row-group events differ from the listed events")
            if any(b <= a for a, b in zip(evs, evs[1:])):
                problems.append(f"{info['file']}: events not increasing")
            if any(e // SHARD * SHARD != int(first) for e in evs):
                problems.append(f"{info['file']}: event outside the shard range")
            seen[k] += evs
    expect = [e for e in complete if e not in dropped]
    for k in ("hits", "particles", "ptype"):
        if sorted(seen[k]) != expect:
            problems.append(f"{k}: {len(seen[k])} events stored, {len(expect)} expected")
    calib_expect = sorted(e for e in expect if e in split["calib"])
    if sorted(seen["truth"]) != calib_expect:
        problems.append(f"truth: {len(seen['truth'])} events stored, {len(calib_expect)} calibration events expected")
    n_bad = {k: 0 for k in TABLES}
    for e in seen["hits"]:
        if rows["hits"].get(e) != dq.at[e, "csv_hits"]:
            n_bad["hits"] += 1
        if rows["particles"].get(e) != dq.at[e, "csv_particles"]:
            n_bad["particles"] += 1
        if rows["ptype"].get(e) != dq.at[e, "csv_particles"]:
            n_bad["ptype"] += 1
    for e in seen["truth"]:
        if rows["truth"].get(e) != dq.at[e, "csv_truth_nonnoise"]:
            n_bad["truth"] += 1
    for k, v in n_bad.items():
        if v:
            problems.append(f"{k}: row counts differ from the csv counts in {v} events")
    status_ok = int((dq.status == "ok").sum())
    if status_ok != len(seen["hits"]):
        problems.append(f"dq: {status_ok} events ok, {len(seen['hits'])} stored")
    if not (dq.loc[dq.status == "ok", "csv_hits"] == dq.loc[dq.status == "ok", "csv_truth"]).all():
        problems.append("dq: csv truth rows differ from csv hit rows in some event")
    ev_set = set(complete)
    if part == "training_part01":
        for k in ("dev", "val", "test"):
            if not split[k] <= ev_set:
                problems.append(f"split {k} events missing from {part}")
        if ev_set - (split["dev"] | split["val"] | split["test"] | split["train"]):
            problems.append(f"{part} events outside the split")
    else:
        if ev_set & (split["val"] | split["test"] | split["dev"]):
            problems.append(f"{part}: dev / val / test events outside training_part01")
        if not ev_set <= split["train"]:
            problems.append(f"{part}: events outside train")
    by_set = {k: int(len(ev_set & split[k])) for k in ("dev", "val", "test", "train", "calib", "gbdt", "merge_train")}
    stored_by_set = {k: int(len(set(seen["hits"]) & split[k])) for k in by_set}
    return {"archive": part, "events_in_archive": len(complete), "events_stored": len(seen["hits"]),
            "events_dropped": len(dropped), "calibration_events_with_truth": len(seen["truth"]),
            "events_by_set": by_set, "stored_by_set": stored_by_set,
            "row_count_mismatches": n_bad, "problems": problems,
            "bytes": {k: int(sum(sh[k]["bytes"] for sh in m["shards"].values() if k in sh)) for k in TABLES}}


# ------------------------------------------------------------------------------------------------ random access
class TarMembers:
    """Members of a streamed part fetched by HTTP range requests at the manifest offsets (md5 checked) and gunzipped:
    usable as the `zf` argument of hnl.trackml.check_convert (open(name) -> csv file object)."""

    def __init__(self, part: str, manifest_dir: str = MANIFEST, url: str | None = None):
        self.part = part
        self.url = url or TP_RECORD.format(name=part + ".tar")
        man = read_manifest(part, manifest_dir)
        self.rows = {r.name: (int(r.offset_data), int(r.size), r.md5) for r in man.itertuples()}
        self.names: dict[int, dict[str, str]] = {}
        for r in man.itertuples():
            self.names.setdefault(int(r.event), {})[r.kind] = r.name
        self.nreq = 0
        self.nbytes = 0

    def read_gz(self, name: str, tries: int = 4) -> bytes:
        off, size, md5 = self.rows[name]
        err = None
        for k in range(tries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + size - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"HTTP {r.status} for a range request")
                    data = r.read()
                self.nreq += 1
                self.nbytes += len(data)
                if len(data) != size or hashlib.md5(data).hexdigest() != md5:
                    raise RuntimeError(f"{name}: range read differs from the manifest (size / md5)")
                return data
            except (OSError, http.client.HTTPException, RuntimeError) as e:
                err = e
                time.sleep(5 * (k + 1))
        raise RuntimeError(f"{name}: {tries} range reads failed") from err

    def open(self, name: str):
        return io.BytesIO(gzip.decompress(self.read_gz(name)))


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--part", required=True)
    s.add_argument("--manifest", default=MANIFEST)
    c = sub.add_parser("convert")
    c.add_argument("--part", required=True)
    c.add_argument("--root", default=DERIVED_TP)
    c.add_argument("--manifest", default=MANIFEST)
    c.add_argument("--split", default=SPLIT_TP)
    c.add_argument("--nproc", type=int, default=len(os.sched_getaffinity(0)))
    c.add_argument("--level", type=int, default=9, help="zstd level")
    c.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--part", required=True)
    v.add_argument("--root", default=DERIVED_TP)
    v.add_argument("--manifest", default=MANIFEST)
    v.add_argument("--split", default=SPLIT_TP)
    v.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        r = scan_part(a.part, a.manifest)
        print(json.dumps({k: v for k, v in r.items() if k not in ("other_members",)}, indent=1))
        if not (r["size_match"] and r["md5_match"]):
            print(f"[scan] {a.part}: SIZE / MD5 MISMATCH: STOP, do not convert from this archive", flush=True)
            sys.exit(2)
        if r["incomplete_events"] or r["duplicate_members"]:
            print(f"[scan] {a.part}: incomplete events {r['incomplete_events']}, duplicates {r['duplicate_members']}",
                  flush=True)
    elif a.cmd == "convert":
        r = convert_part(a.part, a.root, a.nproc, a.level, a.overwrite, a.manifest, a.split)
        print(json.dumps({k: r[k] for k in ("archive", "n_events_in_archive", "n_events_converted", "dropped",
                                            "source_md5_verified", "elapsed_s")}, indent=1))
    else:
        r = verify_part(a.part, a.root, a.manifest, a.split)
        print(json.dumps(r, indent=1))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(r, fh, indent=1)
        if r["problems"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
