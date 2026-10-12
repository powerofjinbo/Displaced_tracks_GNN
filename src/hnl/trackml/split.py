"""TrackML accuracy-phase event manifest and the fixed train / val / test split (docs/trackml_plan.md section 2).

The labelled accuracy-phase data are the five archives train_1 ... train_5 of Zenodo record 4730167
(DOI 10.5281/zenodo.4730167, CC-BY-4.0). `manifest` reads their member lists from the zip central directories with
HTTP range requests (public, no login; no archive is downloaded): event ids, compressed and uncompressed sizes and the
CRC-32 of every member. `make` applies the split rule below and writes the explicit event lists.

Split rule (fixed in docs/trackml_plan.md before any TrackML result):
  * events 1000-1099 of train_1 (the Kaggle `train_sample`, inspected in docs/trackml_feasibility.md) are never
    val / test events; they are the development subset of the training set;
  * eligible for val / test: every other train_1 event; numpy.random.default_rng(SEED).permutation of the sorted
    eligible ids: the first N_TEST -> test, the next N_VAL -> val; valmon (checkpoint monitoring during training) =
    the first N_VALMON val ids in sorted order;
  * train (scope "all", the plan's default): every event of train_1 ... train_5 that is not val / test;
    train (scope "train_1"): the train_1 events that are not val / test.

Throughput-phase rule (Zenodo 4730157, training_part01 ... 09; used if the user picks that phase, and for the
cross-phase test set otherwise; `throughput_split`): the tar parts have no central directory, so the ids of
training_part01 are recorded when the part is first streamed, before any model sees them; dev = the 100 smallest
part01 ids (they include 21000-21004, inspected for the plan); eligible = the other part01 ids;
default_rng(SEED).permutation of the sorted eligible ids: first N_TEST -> test, next N_VAL -> val, valmon = the first
N_VALMON val ids in sorted order; train = every other event of the nine parts.

Throughput subsets (`tp_subsets`; plan sections 6.0, 6.2 and 6.6, with the dev ids of the throughput rule): calib =
dev + the next N_CALIB_EXTRA training ids in sorted order (truth kinematics and score weights stored); gbdt = the
N_GBDT smallest training ids outside calib; merge_train = the N_MERGE_TRAIN smallest training ids outside calib and
gbdt. hnl.trackml.config computes the same lists from the stored training events.

D1 = throughput with D2 = train_1-sized (the user's decisions, plan section 1.4): only training_part01 and
training_part02 are streamed (hnl.trackml.throughput scan), so `make-tp` applies the rule to their recorded ids: dev,
val, valmon and test depend on part01 alone, and train is "every other event" of the two streamed parts.

    python -m hnl.trackml.split manifest [--out DIR]          (about 20 range requests, a few MB)
    python -m hnl.trackml.split make [--manifest DIR] [--out configs/trackml/split_v1.json]
    python -m hnl.trackml.split make-tp [--manifest DIR] [--parts training_part01,training_part02]
        [--out configs/trackml/split_tp_v1.json]          (ids from the stream manifest; deterministic output)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import urllib.request
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXT = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml"
RECORD = "https://zenodo.org/api/records/4730167/files/{name}/content"
# Zenodo record 4730167 file listing (API, 2026-10-05 UTC): bytes and md5
ARCHIVES = {
    "train_1.zip": (15941486907, "c1aa5e944b2a07942e8f6c3613497478"),
    "train_2.zip": (15999064939, "75974545e4e9314e4b1722a24637bbea"),
    "train_3.zip": (15967434796, "5e46b1ac5af91c7ac214925df7c80eb1"),
    "train_4.zip": (15956023763, "4a836f0079845a4d9dcbbb6e35cbfff3"),
    "train_5.zip": (15959040049, "9ddd9d9445a4a78d4c2d45c66f097e0b"),
}
KINDS = ("cells", "hits", "particles", "truth")
SEED = 0
N_TEST = 300
N_VAL = 150
N_VALMON = 50
DEV = range(1000, 1100)
TP_DOI = "10.5281/zenodo.4730157"
TP_RECORD = "https://zenodo.org/api/records/4730157/files/{name}/content"
# Zenodo record 4730157 ("TrackML Throughput Phase", CC-BY-4.0) file listing (API, read 2026-10-05 UTC): bytes and md5
TP_ARCHIVES = {
    "training_part01.tar": (6962872320, "72380b0d17260cc1b450ac30dbd5c24f"),
    "training_part02.tar": (7342325760, "4ef4e1fc3fb4f2386c71bb88c5482a13"),
    "training_part03.tar": (6932060160, "b0c407c72e2753ed4e1e619c6c030e62"),
    "training_part04.tar": (7738347520, "6a3e8529a95be085b564f9391f8c4c87"),
    "training_part05.tar": (7717632000, "11180caac1a0b125ac13d65938c21354"),
    "training_part06.tar": (7726489600, "2f1df2e1480925466b69454ab4eba427"),
    "training_part07.tar": (7739555840, "d5fe471441dd450ede18ff3956ab72c0"),
    "training_part08.tar": (7746406400, "1274fb71105334e9915686fec5761b6c"),
    "training_part09.tar": (7738511360, "109dd519f313909fff98c71a2b2b96b6"),
}
TP_PARTS_D2 = ("training_part01", "training_part02")    # D2 (user, 2026-10-05): train_1-sized training scope
N_TP_DEV = 100
N_CALIB_EXTRA = 100
N_GBDT = 100
N_MERGE_TRAIN = 200


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests (Zenodo answers 206)."""

    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.nreq = 0
        self.nbytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.nreq += 1
        self.nbytes += len(data)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def read_members(name: str) -> tuple[list[tuple], dict]:
    """Rows (archive, event, kind, compress_size, file_size, crc32) of one archive, and request statistics."""
    f = HTTPRangeFile(RECORD.format(name=name))
    if f.size != ARCHIVES[name][0]:
        raise RuntimeError(f"{name}: {f.size} bytes on Zenodo, {ARCHIVES[name][0]} expected")
    zr = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    rows = []
    for i in zr.infolist():
        base = i.filename.split("/")[-1]
        if not base.startswith("event"):
            continue
        ev, kind = base[5:].split("-")
        rows.append((name, int(ev), kind.split(".")[0], i.compress_size, i.file_size, i.CRC))
    return rows, {"archive_bytes": f.size, "http_range_requests": f.nreq, "bytes_fetched": f.nbytes}


def manifest(out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {"record": "10.5281/zenodo.4730167", "read_utc": now, "archives": {}}
    for name in ARCHIVES:
        rows, st = read_members(name)
        ev = sorted({r[1] for r in rows})
        kinds = {}
        for r in rows:
            kinds.setdefault(r[1], set()).add(r[2])
        incomplete = [e for e in ev if kinds[e] != set(KINDS)]
        tsv = os.path.join(out, name.replace(".zip", ".members.tsv"))
        with open(tsv, "w") as fh:
            fh.write("archive\tevent\tkind\tcompress_size\tfile_size\tcrc32\n")
            for r in sorted(rows, key=lambda r: (r[1], r[2])):
                fh.write("\t".join(str(x) for x in r) + "\n")
        summary["archives"][name] = {**st, "md5": ARCHIVES[name][1], "n_members": len(rows), "n_events": len(ev),
                                     "first_event": ev[0], "last_event": ev[-1], "incomplete_events": incomplete,
                                     "compressed_bytes_by_kind": {k: int(sum(r[3] for r in rows if r[2] == k))
                                                                  for k in KINDS},
                                     "uncompressed_bytes_by_kind": {k: int(sum(r[4] for r in rows if r[2] == k))
                                                                    for k in KINDS}}
        print(name, summary["archives"][name]["n_events"], "events", flush=True)
    with open(os.path.join(out, "manifest_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def _events(manifest_dir: str) -> dict[str, list[int]]:
    out = {}
    for name in ARCHIVES:
        tsv = os.path.join(manifest_dir, name.replace(".zip", ".members.tsv"))
        ev = set()
        with open(tsv) as fh:
            next(fh)
            for line in fh:
                ev.add(int(line.split("\t")[1]))
        out[name] = sorted(ev)
    return out


def _sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def make(manifest_dir: str, out: str) -> dict:
    ev = _events(manifest_dir)
    allev = [e for name in ARCHIVES for e in ev[name]]
    if len(allev) != len(set(allev)):
        raise RuntimeError("an event id occurs in two archives")
    t1 = np.array(ev["train_1.zip"], dtype=np.int64)
    dev = np.array([e for e in t1 if e in DEV], dtype=np.int64)
    eligible = np.setdiff1d(t1, dev)
    perm = eligible[np.random.default_rng(SEED).permutation(len(eligible))]
    test = np.sort(perm[:N_TEST])
    val = np.sort(perm[N_TEST:N_TEST + N_VAL])
    valmon = val[:N_VALMON]
    held = np.union1d(test, val)
    train_t1 = np.setdiff1d(t1, held)
    train_all = np.setdiff1d(np.array(sorted(allev), dtype=np.int64), held)
    res = {
        "version": "v1",
        "fixed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rule": __doc__.split("Split rule")[1].split("Throughput-phase rule")[0].strip(),
        "seed": SEED, "n_test": N_TEST, "n_val": N_VAL, "n_valmon": N_VALMON,
        "archives": {name: {"bytes": ARCHIVES[name][0], "md5": ARCHIVES[name][1], "n_events": len(ev[name]),
                            "first_event": ev[name][0], "last_event": ev[name][-1]} for name in ARCHIVES},
        "counts": {"dev": int(len(dev)), "test": int(len(test)), "val": int(len(val)), "valmon": int(len(valmon)),
                   "train_train1": int(len(train_t1)), "train_all": int(len(train_all))},
        "sha256": {"test": _sha(test), "val": _sha(val), "train_train1": _sha(train_t1), "train_all": _sha(train_all)},
        "test": test.tolist(), "val": val.tolist(), "valmon": valmon.tolist(), "dev": dev.tolist(),
        "train_train1": train_t1.tolist(), "train_all": train_all.tolist(),
    }
    if os.path.exists(out):
        old = json.load(open(out))
        same = all(old.get(k) == res[k] for k in ("test", "val", "valmon", "train_train1", "train_all"))
        if not same:
            raise RuntimeError(f"{out} exists with a different split; refusing to overwrite")
        return old
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def throughput_split(part01_ids, all_ids) -> dict:
    """The throughput-phase rule of the module docstring, applied to the recorded event ids (lists of int)."""
    p1 = np.array(sorted(set(int(i) for i in part01_ids)), dtype=np.int64)
    allv = np.array(sorted(set(int(i) for i in all_ids)), dtype=np.int64)
    if not np.isin(p1, allv).all():
        raise RuntimeError("part01 ids missing from the full id list")
    dev = p1[:N_TP_DEV]
    eligible = p1[N_TP_DEV:]
    perm = eligible[np.random.default_rng(SEED).permutation(len(eligible))]
    test = np.sort(perm[:N_TEST])
    val = np.sort(perm[N_TEST:N_TEST + N_VAL])
    train = np.setdiff1d(allv, np.union1d(test, val))
    return {"version": "tp_v1", "seed": SEED, "rule": "throughput-phase rule of hnl.trackml.split",
            "counts": {"dev": int(len(dev)), "test": int(len(test)), "val": int(len(val)), "valmon": N_VALMON,
                       "train": int(len(train))},
            "sha256": {"test": _sha(test), "val": _sha(val), "train": _sha(train)},
            "test": test.tolist(), "val": val.tolist(), "valmon": val[:N_VALMON].tolist(), "dev": dev.tolist(),
            "train": train.tolist()}


def tp_subsets(train, dev) -> dict[str, list[int]]:
    """calib, gbdt and merge_train of the module docstring (plan sections 6.0, 6.2, 6.6) from the sorted training ids;
    the same expressions as hnl.trackml.config._sets."""
    train = sorted(int(e) for e in train)
    devs = {int(e) for e in dev}
    if not devs <= set(train):
        raise RuntimeError("dev events outside the training set")
    extra = [e for e in train if e not in devs][:N_CALIB_EXTRA]
    calib = sorted(devs | set(extra))
    cs = set(calib)
    gbdt = [e for e in train if e not in cs][:N_GBDT]
    gs = cs | set(gbdt)
    merge_train = [e for e in train if e not in gs][:N_MERGE_TRAIN]
    return {"calib": calib, "gbdt": gbdt, "merge_train": merge_train}


def tp_manifest(manifest_dir: str, parts) -> dict:
    """Recorded event ids of streamed throughput parts (hnl.trackml.throughput scan): {part: {...}}. Refuses a part
    whose streamed size or md5 differs from the Zenodo listing."""
    with open(os.path.join(manifest_dir, "manifest_tp_summary.json")) as fh:
        summ = json.load(fh)
    out = {}
    for part in parts:
        a = summ["archives"].get(part + ".tar")
        if a is None:
            raise RuntimeError(f"{part}: not in {manifest_dir}/manifest_tp_summary.json (stream it first)")
        if not (a["size_match"] and a["md5_match"]):
            raise RuntimeError(f"{part}: streamed size / md5 do not match the Zenodo listing: {a}")
        tsv = os.path.join(manifest_dir, f"{part}.members.tsv")
        kinds: dict[int, set] = {}
        with open(tsv) as fh:
            head = next(fh).rstrip("\n").split("\t")
            ie, ik = head.index("event"), head.index("kind")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                kinds.setdefault(int(f[ie]), set()).add(f[ik])
        complete = sorted(e for e, k in kinds.items() if k == set(KINDS))
        with open(tsv, "rb") as fh:
            tsv_sha = hashlib.sha256(fh.read()).hexdigest()
        out[part] = {"bytes": int(a["bytes_streamed"]), "md5": a["md5_streamed"], "n_events": len(complete),
                     "first_event": complete[0], "last_event": complete[-1],
                     "incomplete_events": sorted(set(kinds) - set(complete)),
                     "manifest_tsv": os.path.basename(tsv), "manifest_tsv_sha256": tsv_sha, "events": complete}
    return out


def make_tp(manifest_dir: str, parts, out: str) -> dict:
    """The throughput split file from the stream manifest. The content is a function of the manifest only (no time
    stamp), so regenerating it reproduces the file byte for byte. Refuses to overwrite a file with other ids."""
    parts = list(parts)
    if parts[0] != "training_part01":
        raise ValueError("the rule needs training_part01 first")
    man = tp_manifest(manifest_dir, parts)
    ids = [e for p in parts for e in man[p]["events"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("an event id occurs in two parts")
    r = throughput_split(man["training_part01"]["events"], ids)
    sub = tp_subsets(r["train"], r["dev"])
    doc_rule = __doc__.split("Throughput-phase rule")[1].split("Throughput subsets")[0]
    doc_sub = __doc__.split("Throughput subsets")[1].split("D1 = throughput with D2")[0]
    doc_scope = "D1 = throughput with D2" + __doc__.split("D1 = throughput with D2")[1].split("    python -m")[0]
    flat = lambda s: " ".join(s.split())
    res = {
        "version": "tp_v1", "phase": "throughput", "record": TP_DOI,
        "rule": flat("Throughput-phase rule" + doc_rule), "subset_rule": flat("Throughput subsets" + doc_sub),
        "scope": flat(doc_scope), "parts": parts,
        "seed": SEED, "n_dev": N_TP_DEV, "n_test": N_TEST, "n_val": N_VAL, "n_valmon": N_VALMON,
        "n_calib_extra": N_CALIB_EXTRA, "n_gbdt": N_GBDT, "n_merge_train": N_MERGE_TRAIN,
        "archives": {p: {k: v for k, v in man[p].items() if k != "events"} for p in parts},
        "counts": {**r["counts"], **{k: len(v) for k, v in sub.items()},
                   "train_excluding_dev": len(r["train"]) - len(r["dev"])},
        "sha256": {"dev": _sha(r["dev"]), "test": r["sha256"]["test"], "val": r["sha256"]["val"],
                   "valmon": _sha(r["valmon"]), "train": r["sha256"]["train"],
                   **{k: _sha(v) for k, v in sub.items()}},
        "test": r["test"], "val": r["val"], "valmon": r["valmon"], "dev": r["dev"], "train": r["train"], **sub,
    }
    keys = ("test", "val", "valmon", "dev", "train", "calib", "gbdt", "merge_train")
    if os.path.exists(out):
        with open(out) as fh:
            old = json.load(fh)
        if any(old.get(k) != res[k] for k in keys):
            raise RuntimeError(f"{out} exists with a different split; refusing to overwrite")
        return old
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(out + ".tmp", out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", default=os.path.join(EXT, "manifest"))
    k = sub.add_parser("make")
    k.add_argument("--manifest", default=os.path.join(EXT, "manifest"))
    k.add_argument("--out", default=os.path.join(REPO, "configs", "trackml", "split_v1.json"))
    t = sub.add_parser("make-tp")
    t.add_argument("--manifest", default=os.path.join(EXT, "manifest"),
                   help="directory with manifest_tp_summary.json and <part>.members.tsv")
    t.add_argument("--parts", default=",".join(TP_PARTS_D2), help="streamed parts, training_part01 first")
    t.add_argument("--out", default=os.path.join(REPO, "configs", "trackml", "split_tp_v1.json"))
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        s = manifest(a.out)
        short = {n: {k: v for k, v in d.items() if k != "incomplete_events"} for n, d in s["archives"].items()}
        print(json.dumps(short, indent=1))
    elif a.cmd == "make":
        r = make(a.manifest, a.out)
        print(json.dumps({k: r[k] for k in ("seed", "counts", "sha256", "archives")}, indent=1))
    else:
        r = make_tp(a.manifest, a.parts.split(","), a.out)
        print(json.dumps({k: r[k] for k in ("seed", "parts", "counts", "sha256")}, indent=1))


if __name__ == "__main__":
    main()
