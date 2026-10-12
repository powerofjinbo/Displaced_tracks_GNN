"""Pipeline configuration of Canto-L on TrackML (configs/trackml/pipeline_v1.yaml) and its event sets.

Event sets: train, val, valmon, test, dev (from data.split, hnl.trackml.events.event_sets), t2 (the test events of
data.t2_split: the cross-phase test set), calib (dev + the next data.n_calibration_extra training ids in sorted order),
gbdt (the support.gbdt_events smallest training ids outside calib), merge_train (the scoring.merge_train_events smallest
training ids outside calib and gbdt), smoke (data.smoke_events), or an explicit comma list of event ids. test and t2
need allow_test (the one-time test chain only; plan section 2.3).

Environment: HNL_TML_CONFIG overrides the config path; HNL_TML_SPLIT overrides data.split (e.g. the accuracy-phase
split for development runs before the throughput phase is converted).

    python -m hnl.trackml.config get graph.densities            (scalars plain, lists / dicts as JSON)
    python -m hnl.trackml.config events valmon [--allow-test] [--count]
    python -m hnl.trackml.config describe                        (counts and sha256 of every non-test set)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import yaml

from hnl.trackml.split import REPO

DEFAULT = os.path.join(REPO, "configs", "trackml", "pipeline_v1.yaml")
TEST_SETS = ("test", "t2")


def load(path: str | None = None) -> dict:
    path = path or os.environ.get("HNL_TML_CONFIG", DEFAULT)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = path
    if os.environ.get("HNL_TML_SPLIT"):
        cfg["data"]["split"] = os.environ["HNL_TML_SPLIT"]
    return cfg


def threads() -> int:
    """Torch / OpenMP threads: OMP_NUM_THREADS when set (env/activate.sh sets the Slurm allocation, 4 on a login node),
    else the CPUs this process may run on."""
    import torch
    n = int(os.environ.get("OMP_NUM_THREADS") or len(os.sched_getaffinity(0)))
    torch.set_num_threads(n)
    return n


def repo_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def get(cfg: dict, dotted: str):
    v = cfg
    for k in dotted.split("."):
        v = v[k]
    return v


def _sets(cfg: dict) -> dict:
    from hnl.trackml.events import event_sets
    d = cfg["data"]
    s = event_sets(repo_path(d["split"]), d.get("train_archives"), d["tables"])
    train = s["train"]
    dev = set(s["dev"])
    extra = [e for e in train if e not in dev][:int(d.get("n_calibration_extra", 100))]
    calib = sorted(dev | set(extra))
    gb = [e for e in train if e not in set(calib)][:int(cfg["support"]["gbdt_events"])]
    mt = [e for e in train if e not in set(calib) | set(gb)][:int(cfg["scoring"]["merge_train_events"])]
    with open(repo_path(d["t2_split"])) as fh:
        t2 = sorted(int(e) for e in json.load(fh)["test"])
    return {**{k: s[k] for k in ("train", "val", "valmon", "test", "dev")}, "t2": t2, "calib": calib, "gbdt": gb,
            "merge_train": mt, "smoke": [int(e) for e in d.get("smoke_events", [])], "_phase": s["phase"],
            "_split": s["split"]}


def event_set(spec: str, cfg: dict | None = None, allow_test: bool = False) -> list[int]:
    """Event ids of a named set or an explicit comma list (see the module docstring)."""
    if spec.replace(",", "").strip().isdigit():
        return [int(x) for x in spec.split(",") if x.strip()]
    if spec in TEST_SETS and not allow_test:
        raise PermissionError(f"event set {spec!r} is a test set: only the one-time test chain reads it (allow_test)")
    cfg = cfg or load()
    if spec == "smoke":                        # needs no split file
        return [int(e) for e in cfg["data"].get("smoke_events", [])]
    return list(_sets(cfg)[spec])


def _sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("key")
    e = sub.add_parser("events")
    e.add_argument("set")
    e.add_argument("--allow-test", action="store_true")
    e.add_argument("--count", action="store_true")
    sub.add_parser("describe")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.cmd == "get":
        v = get(cfg, a.key)
        print(json.dumps(v) if isinstance(v, (list, dict)) else v)
    elif a.cmd == "events":
        ev = event_set(a.set, cfg, a.allow_test)
        print(len(ev) if a.count else ",".join(str(x) for x in ev))
    else:
        s = _sets(cfg)
        print(json.dumps({"config": cfg["_path"], "phase": s["_phase"], "split": s["_split"],
                          **{k: {"n": len(v), "sha256": _sha(v)[:8], "first": v[:3]} for k, v in s.items()
                             if not k.startswith("_") and k not in TEST_SETS},
                          **{k: {"n": len(s[k])} for k in TEST_SETS}}, indent=1))


if __name__ == "__main__":
    main()
