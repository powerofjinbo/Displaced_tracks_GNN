"""Training arm "g" of preregistration addendum G, step T3 (docs/preregistration_new_sample.md section 5.2;
docs/addendum_g_t3_plan.md): Stage-0 arm (iii) with the HNL events replaced by the generic train list.

Event lists (sample, index):
  train      ttbar train (7,999) + splits/generic_train_g7999.txt (7,999: 4,005 generic_iso, 3,994 generic_dense; T1)
  valmon     the first 166 val events of ttbar, generic_iso and generic_dense each (checkpoint monitoring; the arm's own
             val mix, as arm (iii)'s 166 x 3 samples)
  val        ttbar val + generic_iso val + generic_dense val (the arm's own val mix: the metric-learning radius scan)
  val_all    the val splits of ttbar, hnl10_100, hnl15_200, generic_iso and generic_dense (scored for the read-outs)
No test event of any sample is listed here. The generated samples need HNL_GENERATED=1 (hnl.data.npz). Staging reads
the feature-store tars of derived/acorn_fs/v1 (ttbar, HNL) and derived/acorn_fs/generated_v1 (generated samples), or one
packed stage output, exactly as hnl.train.acorn_runner.stage_events does for arms (i)-(iii).

    python -m hnl.train.arm_g stage --what train,valmon --source fs --dest $LOCAL/fs
    python -m hnl.train.arm_g stage --what train,valmon --source tar --tar-dir RUN/ml_graphs --dest $LOCAL/in
    python -m hnl.train.arm_g list --what train                      (counts per sample; no event is read)
"""

from __future__ import annotations

import argparse
import os
import tarfile

import numpy as np

from hnl.train import acorn_runner as AR

GENERIC = ("generic_iso", "generic_dense")
LABELED = ("ttbar", "hnl10_100", "hnl15_200")
FS_GEN = os.path.join(AR.DERIVED, "acorn_fs", "generated_v1")
TRAIN_LIST = os.path.join(AR.REPO, "splits", "generic_train_g7999.txt")


def _require_generated():
    from hnl.data import acorn_store as AS
    if "generic_iso" not in AS.SAMPLE_CODE:
        raise SystemExit("arm g needs the generated samples: run with HNL_GENERATED=1")


def generic_train() -> list[tuple[str, int]]:
    out = []
    for line in open(TRAIN_LIST):
        if line.startswith("#") or not line.strip():
            continue
        s, i = line.split()
        if s not in GENERIC:
            raise SystemExit(f"{TRAIN_LIST}: unexpected sample {s}")
        out.append((s, int(i)))
    if len(out) != 7999 or len(set(out)) != len(out):
        raise SystemExit(f"{TRAIN_LIST}: {len(out)} events ({len(set(out))} distinct), expected 7,999")
    return out


def events(what: str) -> list[tuple[str, int]]:
    _require_generated()
    split = lambda s, sp: [(s, int(i)) for i in AR.read_split(s, sp)]
    if what == "train":
        tt = split("ttbar", "train")
        g = generic_train()
        allowed = {e for s in GENERIC for e in split(s, "train")}
        if not set(g) <= allowed:
            raise SystemExit("the generic training list holds events outside the generic train splits")
        return tt + g
    if what == "valmon":
        k = AR.N_VAL_MONITOR // 3
        return [e for s in ("ttbar", *GENERIC) for e in split(s, "val")[:k]]
    if what == "val":
        return [e for s in ("ttbar", *GENERIC) for e in split(s, "val")]
    if what == "val_all":
        return [e for s in (*LABELED, *GENERIC) for e in split(s, "val")]
    raise SystemExit(f"unknown event list {what}")


def stage(evs: list[tuple[str, int]], source: str, dest: str, tar_name: str | None = None) -> int:
    """Extract the events from the feature-store tars (source fs) or from one packed stage output (source tar) into
    dest (flat), as hnl.train.acorn_runner.stage_events."""
    os.makedirs(dest, exist_ok=True)
    want = {AR.event_id(s, i) for s, i in evs}
    if source == "fs":
        tars = []
        for s in sorted({s for s, _ in evs}):
            d = FS_GEN if s in GENERIC else AR.FS_DIR
            tars += [os.path.join(d, f"{s}_{sp}.tar") for sp in ("train", "val")]     # never a test tar
    else:
        tars = [tar_name]
    n = 0
    for p in tars:
        with tarfile.open(p, "r") as tar:
            for m in tar:
                eid = m.name.split("event")[-1].split("-")[0].split(".")[0]
                if eid in want:
                    m.name = os.path.basename(m.name)
                    tar.extract(m, dest, filter="data")
                    n += 1
    if n != len(want):
        raise RuntimeError(f"staged {n} of {len(want)} events into {dest}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--what", required=True, help="comma list of train,valmon,val,val_all")
    s.add_argument("--source", default="fs", choices=["fs", "tar"])
    s.add_argument("--tar-dir", default=None)
    s.add_argument("--dest", required=True)
    l_ = sub.add_parser("list")
    l_.add_argument("--what", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for w in a.what.split(","):
            ev = events(w)
            u, c = np.unique([s for s, _ in ev], return_counts=True)
            print(w, len(ev), dict(zip(u.tolist(), c.tolist())))
        return
    for w in a.what.split(","):
        d = os.path.join(a.dest, AR.ACORN_SPLIT[w])
        tar = os.path.join(a.tar_dir, f"{AR.ACORN_SPLIT[w]}.tar") if a.source == "tar" else None
        n = stage(events(w), a.source, d, tar)
        print(f"[stage] arm g {w}: {n} events -> {d}", flush=True)


if __name__ == "__main__":
    main()
