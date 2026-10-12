"""The selection freeze of Canto-L on TrackML (docs/trackml_plan.md section 5, "Freeze"): every chosen configuration
(WP-A, WP-S, WP-R with their conditions), the checkpoints, models and calibration inputs with their sha256, written to
configs/trackml/selection_v1.json. The main session reviews and commits it before any test job; the test chain
(slurm/trackml/test_chain.sh) refuses a selection file that is not committed unchanged, and runs each frozen
configuration once.

    python -m hnl.trackml.freeze make --selection RUN/selection_v1.json --run RUN --out configs/trackml/selection_v1.json
    python -m hnl.trackml.freeze check --file configs/trackml/selection_v1.json
    python -m hnl.trackml.freeze configs --file configs/trackml/selection_v1.json --condition full   (frozen configs)
    python -m hnl.trackml.freeze verify --file configs/trackml/selection_v1.json --run RUN   (every input's sha256,
        the run directory and the classifier checkpoint that best_ckpt resolves to; the test chain runs it first)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

from hnl.trackml import config as CFG


def _sha(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _best(run: str, stage: str) -> str | None:
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(run, stage, "run_*.json"))]
    runs = [r for r in runs if r.get("state") == "done" and r.get("best_ckpt")]
    return max(runs, key=lambda r: r["end_utc"])["best_ckpt"] if runs else None


def make(selection: str, run: str, out: str) -> dict:
    cfg = CFG.load()
    sel = json.load(open(selection))
    gc = os.path.join(run, "graph", "graph_choice.json")
    choice = json.load(open(gc)) if os.path.exists(gc) else {}
    fm = open(os.path.join(run, "support", "field_map.txt")).read().strip() if os.path.exists(
        os.path.join(run, "support", "field_map.txt")) else None
    inputs = {
        "ml_checkpoint": _best(run, "ml"), "classifier_checkpoint": _best(run, "gnn"),
        "support_model": os.path.join(run, "support", "model.joblib"), "graph_choice": gc,
        "merge_model": os.path.join(run, "lmerge", "model.joblib"),
        "field_map": CFG.repo_path(fm) if fm and fm != "none" else None,
        "pipeline_config": cfg["_path"], "ml_config": CFG.repo_path(cfg["metric_learning"]["config"]),
        "classifier_config": CFG.repo_path(cfg["classifier"]["config"]), "split": CFG.repo_path(cfg["data"]["split"]),
        "t2_split": CFG.repo_path(cfg["data"]["t2_split"]),
        "detectors": os.path.join(cfg["data"]["tables"], "..", "..", "detectors.csv"),
        # calibration tables read by the TrackML stages (plan 6.5; res_v1 replaces the hit sigmas at import), the
        # resolved training configurations of this run, the environment lock, and the val grids behind the selection
        "res_table": CFG.repo_path("configs/trackml/res_v1.json"),
        "merge_cuts": CFG.repo_path(cfg["builders"]["merge_cuts"]["table"]),
        "slope_window": CFG.repo_path("configs/trackml/slope_window_v1.tsv"),
        "ml_train_config_run": os.path.join(run, "ml", "train_config.yaml"),
        "classifier_train_config_run": os.path.join(run, "gnn", "train_config.yaml"),
        "env_lock": CFG.repo_path("env/lock.txt"),
    }
    if os.path.basename(os.path.normpath(run)).endswith("_x1"):          # X1 (plan D3) trained with ml_train_x1.yaml
        inputs["ml_config_x1"] = CFG.repo_path("configs/trackml/ml_train_x1.yaml")
    for key, d in (("full", sel.get("grid")), ("nonoise", sel.get("grid_nonoise")), ("1gev", sel.get("grid_1gev"))):
        if d:
            inputs[f"val_grid_{key}_summary"] = os.path.join(d, "summary.csv")
            inputs[f"val_grid_{key}_configs"] = os.path.join(d, "configs.json")
    res = {"version": "v1", "plan": "docs/trackml_plan.md section 5", "run": run, "selection": sel,
           "graph_choice": choice,
           "inputs": {k: {"path": v, "sha256": _sha(v)} for k, v in inputs.items()},
           "git": subprocess.run(["git", "-C", CFG.REPO, "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()}
    missing = [k for k, v in res["inputs"].items() if v["path"] and v["sha256"] is None]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def verify(path: str, run: str) -> list[str]:
    """Problems (empty if none): an input whose sha256 differs from the frozen one, another run directory, or a
    classifier checkpoint resolved by best_ckpt (score_full's default) other than the frozen one."""
    d = json.load(open(path))
    bad = []
    if os.path.normpath(d["run"]) != os.path.normpath(run):
        bad.append(f"run {run} is not the frozen run {d['run']}")
    for k, v in d["inputs"].items():
        if v["path"] and _sha(v["path"]) != v["sha256"]:
            bad.append(f"{k}: sha256 of {v['path']} differs from the frozen value")
    if _best(run, "gnn") != d["inputs"]["classifier_checkpoint"]["path"]:
        bad.append(f"best_ckpt gnn resolves to {_best(run, 'gnn')}, not the frozen classifier checkpoint")
    return bad


def check(path: str) -> bool:
    """True if the file is tracked by git and identical to its committed version."""
    rel = os.path.relpath(os.path.abspath(path), CFG.REPO)
    tracked = subprocess.run(["git", "-C", CFG.REPO, "ls-files", "--error-unmatch", rel], capture_output=True)
    if tracked.returncode != 0:
        return False
    diff = subprocess.run(["git", "-C", CFG.REPO, "diff", "--quiet", "HEAD", "--", rel])
    return diff.returncode == 0


def frozen_configs(path: str, condition: str) -> list[dict]:
    """The frozen configurations to run in a condition: WP-A (and WP-A under FR_T,CC if another configuration) in every
    condition, since table 4.3 has a WP-A column for the C-nonoise and C-1GeV rows (plan sections 4.3 and 10); WP-S in
    C-full and C-nonoise (reported with the score and the P1 numbers, plan section 5; no score in C-1GeV); each WP-R
    in its reference's condition (P1 nonoise, P2 / P3 full, P4 1gev)."""
    d = json.load(open(path))["selection"]
    out = [{"name": "WP-A", **d["WP-A"]}]
    if d.get("WP-A-FRcc") and not d["WP-A-FRcc"]["same_as_WP-A"]:       # the FR_T,CC sensitivity, reported only
        out.append({"name": "WP-A-FRcc", **d["WP-A-FRcc"]})
    if condition in ("full", "nonoise") and d.get("WP-S"):
        out.append({"name": "WP-S", **d["WP-S"]})
    cond = {"P1": "nonoise", "P2": "full", "P3": "full", "P4": "1gev"}
    for ref, wp in (d.get("WP-R") or {}).items():
        if wp and wp.get("reached") and cond[ref] == condition:
            out.append({"name": f"WP-R-{ref}", **wp})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--selection", required=True)
    m.add_argument("--run", required=True)
    m.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--file", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--file", required=True)
    v.add_argument("--run", required=True)
    f = sub.add_parser("configs")
    f.add_argument("--file", required=True)
    f.add_argument("--condition", default="full")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        r = make(a.selection, a.run, a.out)
        print(json.dumps({k: r[k] for k in ("selection", "inputs")}, indent=1, default=float))
    elif a.cmd == "check":
        ok = check(a.file)
        print("committed and unchanged" if ok else "NOT committed (or changed since the commit)")
        sys.exit(0 if ok else 1)
    elif a.cmd == "verify":
        bad = verify(a.file, a.run)
        print("\n".join(bad) if bad else "every frozen input, the run and the classifier checkpoint verified")
        sys.exit(1 if bad else 0)
    else:
        print(json.dumps(frozen_configs(a.file, a.condition)))


if __name__ == "__main__":
    main()
