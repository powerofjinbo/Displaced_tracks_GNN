"""hnl.trackml.throughput and the throughput split (hnl.trackml.split): member naming, tar streaming with md5 and
resume, event grouping, a synthetic part scanned, split, converted and verified end to end (with a local fake of
Zenodo's HTTP answers), and the split file's determinism and set definitions.

    pytest src/hnl/trackml/tests
"""

import gzip
import hashlib
import io
import json
import os
import tarfile

import numpy as np
import pandas as pd
import pytest

from hnl.trackml import convert as C
from hnl.trackml import split as S
from hnl.trackml import throughput as T

LAYERS = [(8, 2, 32.0), (8, 4, 72.0), (8, 6, 116.0), (8, 8, 172.0)]


# ------------------------------------------------------------------------------------------------ synthetic data
def _csv(rows, cols) -> bytes:
    return pd.DataFrame(rows, columns=cols).to_csv(index=False, float_format="%.6g").encode()


def event_members(event: int, n_tracks: int = 5, n_noise: int = 3, bad_nhits: bool = False) -> dict[str, bytes]:
    """csv.gz members of one synthetic throughput event: straight tracks from the origin through one barrel module
    per layer (detectors_csv), noise hits, particles.csv with the throughput column particle_type."""
    rng = np.random.default_rng(event)
    hits, truth, cells, parts = [], [], [], []
    hid = 0
    for j in range(n_tracks):
        pid = (1 << 52) | ((j + 1) << 20)
        phi, cot, pt, q = rng.uniform(-0.02, 0.02), rng.uniform(-0.15, 0.15), rng.uniform(1.5, 5.0), int(
            rng.choice([-1, 1]))
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * cot
        for v, lay, r in LAYERS:
            hid += 1
            x, y, z = r, r * np.tan(phi), r * cot
            hits.append((hid, x, y, z, v, lay, 1))
            truth.append((hid, pid, x, y, z, px, py, pz, 0.01))
            cells.append((hid, 10 + j, 20, 0.25))
            cells.append((hid, 11 + j, 21, 0.5))
        parts.append((pid, 211 * q, 0.0, 0.0, 0.0, px, py, pz, q, len(LAYERS) + (1 if bad_nhits and j == 0 else 0)))
    for k in range(n_noise):
        v, lay, r = LAYERS[k % len(LAYERS)]
        hid += 1
        y, z = rng.uniform(-5, 5), rng.uniform(-20, 20)
        hits.append((hid, r, y, z, v, lay, 1))
        truth.append((hid, 0, r, y, z, 0.0, 0.0, 0.0, 0.0))
        cells.append((hid, 1, 1, 0.1))
    raw = {"cells": _csv(cells, ["hit_id", "ch0", "ch1", "value"]),
           "hits": _csv(hits, ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"]),
           "particles": _csv(parts, ["particle_id", "particle_type", "vx", "vy", "vz", "px", "py", "pz", "q", "nhits"]),
           "truth": _csv(truth, ["hit_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"])}
    return {k: gzip.compress(v, mtime=0) for k, v in raw.items()}


def detectors_csv(path):
    rows = []
    for v, lay, r in LAYERS:
        rows.append({"volume_id": v, "layer_id": lay, "module_id": 1, "cx": r, "cy": 0.0, "cz": 0.0,
                     "rot_xu": 0.0, "rot_xv": 0.0, "rot_xw": 1.0, "rot_yu": 1.0, "rot_yv": 0.0, "rot_yw": 0.0,
                     "rot_zu": 0.0, "rot_zv": 1.0, "rot_zw": 0.0, "module_t": 0.15, "module_minhu": 8.4,
                     "module_maxhu": 8.4, "module_hv": 36.0, "pitch_u": 0.05, "pitch_v": 0.05})
    pd.DataFrame(rows).to_csv(path, index=False)


def make_tar(events, drop=None, extra_member=False) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        if extra_member:
            ti = tarfile.TarInfo("README.txt")
            ti.size = 5
            tf.addfile(ti, io.BytesIO(b"hello"))
        for e in events:
            m = event_members(e)
            for k in sorted(m):
                if drop == (e, k):
                    continue
                ti = tarfile.TarInfo(f"event{e:09d}-{k}.csv.gz")
                ti.size = len(m[k])
                ti.mtime = 0
                tf.addfile(ti, io.BytesIO(m[k]))
    return buf.getvalue()


class FakeResp:
    def __init__(self, data: bytes, start: int, fail_after: int | None, short: bool):
        self.status = 206 if start else 200
        self.headers = {"Content-Length": str(len(data) - start)}
        if start:
            self.headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        self.b = io.BytesIO(data[start:])
        self.fail_after, self.short, self.n = fail_after, short, 0

    def read(self, n=-1):
        if self.fail_after is not None and self.n >= self.fail_after:
            if self.short:
                return b""                              # a connection closed early
            raise ConnectionResetError("fake reset")
        if self.fail_after is not None and n is not None and n >= 0:
            n = min(n, self.fail_after - self.n)
        d = self.b.read(n)
        self.n += len(d)
        return d

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_zenodo(monkeypatch, blobs: dict[str, bytes], failures: list | None = None):
    """urllib.request.urlopen answering for {url: bytes}, with Range requests; failures: [(fail_after, short)] for
    the first plain (non-range) openings."""
    failures = list(failures or [])
    calls = []

    def urlopen(req, timeout=None):
        url = req.full_url
        rng = req.get_header("Range")
        data = blobs[url]
        calls.append((url, rng))
        if rng:
            a, b = rng.split("=")[1].split("-")
            a = int(a)
            if b:
                r = FakeResp(data[:int(b) + 1], a, None, False)
                return r
            return FakeResp(data, a, None, False)
        fa, short = failures.pop(0) if failures else (None, False)
        return FakeResp(data, 0, fa, short)

    monkeypatch.setattr(T.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(T.HTTPStream, "RESUME_SLEEP", 0.0)
    return calls


# ------------------------------------------------------------------------------------------------ unit tests
def test_member_event_kind():
    assert T.member_event_kind("event000021000-cells.csv.gz") == (21000, "cells")
    assert T.member_event_kind("dir/event000021001-truth.csv.gz") == (21001, "truth")
    assert T.member_event_kind("event000021000-cells.csv") is None
    assert T.member_event_kind("event000021000-foo.csv.gz") is None
    assert T.member_event_kind("README") is None


def test_stream_md5_and_resume(monkeypatch):
    data = os.urandom(3_000_001)
    url = "https://example.invalid/x.tar"
    calls = fake_zenodo(monkeypatch, {url: data}, failures=[(1_000_003, False)])
    st = T.HTTPStream(url, chunk=65536)
    got = st.read(2_000_000)
    st.drain()
    assert got == data[:2_000_000]
    assert st.pos == len(data) and st.md5.hexdigest() == hashlib.md5(data).hexdigest()
    assert st.resumes == 1 and calls[1][1] == "bytes=1000003-"
    calls = fake_zenodo(monkeypatch, {url: data}, failures=[(777, True)])      # premature end
    st = T.HTTPStream(url)
    st.drain()
    assert st.md5.hexdigest() == hashlib.md5(data).hexdigest() and st.resumes == 1


def test_iter_events_grouping_and_manifest_check():
    blob = make_tar([21000, 21001, 21002], drop=(21001, "particles"), extra_member=True)
    got = list(T.iter_events(io.BufferedReader(io.BytesIO(blob))))
    assert [e for e, _ in got] == [21000, 21002]
    assert set(got[0][1]) == set(C.KINDS)
    man = pd.DataFrame([{"event": 21000, "kind": "cells", "member_index": 1, "size": 1, "md5": "x"}])
    with pytest.raises(RuntimeError):
        list(T.iter_events(io.BytesIO(blob), man))


def test_event_tables_tp_columns_and_ptype(tmp_path):
    det_path = tmp_path / "detectors.csv"
    detectors_csv(det_path)
    det = C.load_detectors(str(det_path))
    m = event_members(21005)
    tables, dq = T.event_tables_tp(m, 21005, det, True, True)
    assert dq["problems"] == [] and tables is not None
    for k in ("hits", "particles", "truth"):
        assert tables[k].schema.equals(C.SCHEMAS[k])
    p = pd.read_csv(io.BytesIO(gzip.decompress(m["particles"])))
    pt = tables["ptype"].to_pandas()
    assert np.array_equal(pt.particle_id.to_numpy(), p.particle_id.to_numpy())
    assert np.array_equal(pt.particle_type.to_numpy(), p.particle_type.to_numpy())
    assert np.array_equal(tables["particles"].column("particle_id").to_numpy(), p.particle_id.to_numpy())
    assert dq["csv_hits"] == tables["hits"].num_rows == dq["csv_truth"]
    assert dq["csv_truth_nonnoise"] == tables["truth"].num_rows
    bad = event_members(21006, bad_nhits=True)
    t2, dq2 = T.event_tables_tp(bad, 21006, det, False, False)
    assert t2 is None and any("nhits" in x for x in dq2["problems"])
    broken = dict(m, hits=m["hits"][:-8])
    t3, dq3 = T.event_tables_tp(broken, 21005, det, False, False)
    assert t3 is None and dq3["problems"][0].startswith("exception")


# ------------------------------------------------------------------------------------------------ end to end
def _small_rule(monkeypatch):
    for k, v in {"N_TP_DEV": 2, "N_TEST": 2, "N_VAL": 2, "N_VALMON": 1, "N_CALIB_EXTRA": 1, "N_GBDT": 1,
                 "N_MERGE_TRAIN": 2}.items():
        monkeypatch.setattr(S, k, v)


def test_scan_split_convert_verify(tmp_path, monkeypatch):
    _small_rule(monkeypatch)
    ev1 = [21000, 21001, 21002, 21003, 21004, 21006, 21007, 21009]
    ev2 = [21010, 21011, 21012]
    blobs = {"training_part01.tar": make_tar(ev1), "training_part02.tar": make_tar(ev2)}
    urls = {T.TP_RECORD.format(name=k): v for k, v in blobs.items()}
    fake_zenodo(monkeypatch, urls)
    arch = {k: (len(v), hashlib.md5(v).hexdigest()) for k, v in blobs.items()}
    monkeypatch.setattr(T, "TP_ARCHIVES", arch)
    monkeypatch.setattr(C, "storage_check", lambda b: {"projected": int(b)})
    man = str(tmp_path / "manifest")
    for p in ("training_part01", "training_part02"):
        r = T.scan_part(p, man)
        assert r["md5_match"] and r["size_match"] and r["events_in_increasing_order"] and not r["incomplete_events"]
    # the split, twice: byte-identical
    out_a, out_b = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    S.make_tp(man, ["training_part01", "training_part02"], out_a)
    S.make_tp(man, ["training_part01", "training_part02"], out_b)
    assert open(out_a, "rb").read() == open(out_b, "rb").read()
    s = json.load(open(out_a))
    assert s["dev"] == [21000, 21001]
    assert set(s["test"]) | set(s["val"]) <= set(ev1[2:]) and len(s["test"]) == 2 and len(s["val"]) == 2
    assert s["train"] == sorted(set(ev1 + ev2) - set(s["test"]) - set(s["val"]))
    assert s["calib"] == sorted(s["dev"] + [e for e in s["train"] if e not in s["dev"]][:1])
    # conversion and structural verification
    det_path = tmp_path / "detectors.csv"
    detectors_csv(det_path)
    root = str(tmp_path / "derived")
    for p in ("training_part01", "training_part02"):
        res = T.convert_part(p, root, nproc=2, manifest_dir=man, split_path=out_a, det_path=str(det_path))
        assert res["dropped"] == [] and res["source_md5_verified"] == arch[p + ".tar"][1]
        v = T.verify_part(p, root, man, out_a)
        assert v["problems"] == [], v["problems"]
    h = C.read_table("hits", [21000, 21010], root=root)
    assert set(h.event) == {21000, 21010}
    tr = C.read_table("truth", s["calib"], root=root)
    assert sorted(set(tr.event)) == s["calib"]
    w = C.read_table("hits", [21010], root=root).weight.to_numpy()
    assert np.isnan(w).all() == (21010 not in set(s["calib"]) | set(s["val"]) | set(s["test"]))
    pt = C.read_table("ptype", [21000], root=root)
    assert len(pt) == len(C.read_table("particles", [21000], root=root))
    with pytest.raises(FileExistsError):
        T.convert_part("training_part01", root, nproc=1, manifest_dir=man, split_path=out_a, det_path=str(det_path))


def test_convert_refuses_md5_mismatch(tmp_path, monkeypatch):
    _small_rule(monkeypatch)
    blob = make_tar([21000, 21001, 21002, 21003, 21004, 21005])
    fake_zenodo(monkeypatch, {T.TP_RECORD.format(name="training_part01.tar"): blob})
    monkeypatch.setattr(T, "TP_ARCHIVES", {"training_part01.tar": (len(blob), "0" * 32)})
    man = str(tmp_path / "m")
    r = T.scan_part("training_part01", man)
    assert r["size_match"] and not r["md5_match"]
    with pytest.raises(RuntimeError):
        S.make_tp(man, ["training_part01"], str(tmp_path / "s.json"))


def test_tp_subsets_definition():
    train = list(range(21000, 21700))
    dev = list(range(21000, 21100))
    sub = S.tp_subsets(train, dev)
    assert sub["calib"] == list(range(21000, 21200))
    assert sub["gbdt"] == list(range(21200, 21300))
    assert sub["merge_train"] == list(range(21300, 21500))
