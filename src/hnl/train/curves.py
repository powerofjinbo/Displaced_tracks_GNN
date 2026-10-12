"""Training curves and the Stage-0 convergence rule (DECISIONS.md 2026-10-03, "Stage-0 epoch budget").

For a run's stage directory: the monitored val metric per epoch (metric learning: f1, max; GNN: val_loss, min) from
the Lightning CSV logs (all versions, i.e. across resumed jobs), and whether the run "still improves": best over the
last 5 epochs better than the best before them by more than 0.002 (f1) or 1 % relative (val loss).

    python -m hnl.train.curves RUN_DIR/ml RUN_DIR/gnn ...
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd

RULE = {"f1": ("max", 0.002, "abs"), "val_loss": ("min", 0.01, "rel")}


def epoch_metrics(stage_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(stage_dir, "lightning_logs", "version_*", "metrics.csv"))):
        d = pd.read_csv(f)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    cols = [c for c in d.columns if c in ("f1", "val_loss", "train_loss", "lr-AdamW", "auc", "total_eff", "total_pur")]
    d = d.dropna(subset=[c for c in ("f1", "val_loss") if c in d.columns], how="all")
    return d.groupby("epoch")[cols].last().reset_index()


def still_improving(m: pd.DataFrame, metric: str, window: int = 5) -> dict:
    mode, thr, kind = RULE[metric]
    s = m[metric].dropna()
    if len(s) <= window:
        return {"epochs": len(s), "decision": "too few epochs"}
    head, tail = s.iloc[:-window], s.iloc[-window:]
    if mode == "max":
        gain = tail.max() - head.max()
        imp = gain > thr
    else:
        gain = (head.min() - tail.min()) / abs(head.min())
        imp = gain > thr
    return {"epochs": len(s), "best_before": float(head.max() if mode == "max" else head.min()),
            "best_last5": float(tail.max() if mode == "max" else tail.min()), "gain": float(gain),
            "still_improving": bool(imp)}


def main(argv=None):
    for d in (argv or sys.argv[1:]):
        m = epoch_metrics(d)
        if m.empty:
            print(d, "no metrics")
            continue
        metric = "f1" if "f1" in m.columns else "val_loss"
        print(f"== {d} ({metric})")
        print(m.round(5).to_string(index=False))
        print(still_improving(m, metric))


if __name__ == "__main__":
    main()
