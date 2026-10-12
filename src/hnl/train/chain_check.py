"""Consistency checks of a baseline run before an evaluation (review fixes, DECISIONS.md 2026-10-03).

  python -m hnl.train.chain_check --run RUN [--stages ml,gnn] [--wp WORKING_POINTS_JSON]

--run: the best checkpoint of each finished training stage (the newest finished run_<job>.json, as used by the stage
scripts) must be the checkpoint, byte for byte (sha256), that produced the run's val inference (ml_graphs /
gnn_scores infer_*.json); otherwise val scores and working points would belong to another model.
--wp: the working points must come from all 3 arms x 3 seeds (no missing runs) before any test evaluation; with --run
also every run behind them must still have the val chain recorded in the working points (val_chain).
Exits non-zero with a message on any failure.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

ARMS, SEEDS = {"i", "ii", "iii"}, {"0", "1", "2"}
RECORD_DIR = {"ml": "ml_graphs", "gnn": "gnn_scores"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_ckpt(run: str, stage: str) -> str:
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(run, stage, "run_*.json"))]
    runs = [r for r in runs if r.get("state") == "done" and r.get("best_ckpt")]
    if not runs:
        raise SystemExit(f"no finished {stage} training in {run}")
    return max(runs, key=lambda r: r["end_utc"])["best_ckpt"]


def check_run(run: str, stages: list[str]) -> list[str]:
    errors = []
    for stage in stages:
        ck = best_ckpt(run, stage)
        recs = sorted(glob.glob(os.path.join(run, RECORD_DIR[stage], "infer_*.json")), key=os.path.getmtime)
        if not recs:
            errors.append(f"{stage}: no val inference record in {RECORD_DIR[stage]}")
            continue
        rec = json.load(open(recs[-1]))
        if rec.get("ckpt") != ck:
            errors.append(f"{stage}: best checkpoint {ck} != val inference checkpoint {rec.get('ckpt')}")
        elif not os.path.exists(ck) or rec.get("ckpt_sha256") != sha256(ck):
            errors.append(f"{stage}: checkpoint {ck} changed since the val inference (sha256 mismatch)")
    return errors


def newest_record(run: str, stage: str):
    recs = sorted(glob.glob(os.path.join(run, RECORD_DIR[stage], "infer_*.json")), key=os.path.getmtime)
    return json.load(open(recs[-1])) if recs else None


def val_chain(run: str) -> dict:
    """sha256 of the checkpoints behind the run's val inference (metric learning, GNN)."""
    return {st: (newest_record(run, st) or {}).get("ckpt_sha256") for st in RECORD_DIR}


def check_wp(path: str, runs_root: str | None = None) -> list[str]:
    """Shape of the working points; with runs_root also that every run behind them still has the same val chain."""
    wp = json.load(open(path))
    errors = []
    if set(wp.get("s_star", {})) != ARMS:
        errors.append(f"s_star arms {sorted(wp.get('s_star', {}))} != {sorted(ARMS)}")
    for key in ("cuts_at_FR_star", "cuts_at_e_star"):
        d = wp.get(key, {})
        if set(d) != ARMS or any(set(d[a]) != SEEDS for a in d):
            errors.append(f"{key} is not 3 arms x 3 seeds")
    if wp.get("missing_runs"):
        errors.append(f"missing runs: {wp['missing_runs']}")
    if "runs" not in wp:
        errors.append("no run list (working points written by an old version of baseline_val)")
    ref = wp.get("reference_arm")
    accepted = {tuple(map(str, x)) for x in wp.get("accepted_unreachable_reference", [])}
    if any(x[0] == ref and (str(x[0]), str(x[1])) not in accepted for x in wp.get("unreachable_at_FR_star", [])):
        errors.append("a seed of the reference arm cannot reach FR* (not reviewed by hand)")
    if runs_root is not None:
        chain = wp.get("val_chain")
        if not chain:
            errors.append("no val_chain record in the working points")
        else:
            for name in wp.get("runs", []):
                run = os.path.join(runs_root, name)
                errors += [f"{name}: {e}" for e in check_run(run, ["ml", "gnn"])]
                now = val_chain(run)
                if chain.get(name) != now:
                    errors.append(f"{name}: val inference {now} is not the one behind the working points "
                                  f"({chain.get(name)})")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--stages", default="ml,gnn")
    ap.add_argument("--wp", default=None)
    a = ap.parse_args(argv)
    if a.run is None and a.wp is None:
        sys.exit("chain check: nothing to check (give --run and/or --wp)")
    errors = []
    if a.run is not None:
        errors += check_run(a.run, a.stages.split(","))
    if a.wp is not None:
        root = os.path.dirname(os.path.normpath(a.run)) if a.run is not None else None
        errors += check_wp(a.wp, root)
    if errors:
        sys.exit("chain check failed:\n  " + "\n  ".join(errors))
    print("chain check passed", a.run or "", a.wp or "")


if __name__ == "__main__":
    main()
