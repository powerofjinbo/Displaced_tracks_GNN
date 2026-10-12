"""Summary of the per-minute GPU utilisation logs written by the training jobs (slurm/baseline/_common.sh).

    python -m hnl.train.gpu_util RUN_DIR/gnn [...]
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd


def summary(stage_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(stage_dir, "gpu_util_*.csv"))):
        try:
            d = pd.read_csv(f, header=None, names=["t", "util", "mem"], skipinitialspace=True)
        except pd.errors.EmptyDataError:
            continue
        u = d.util.astype(str).str.replace("%", "").str.strip().astype(float)
        m = d.mem.astype(str).str.replace("MiB", "").str.strip().astype(float)
        rows.append({"file": os.path.basename(f), "minutes": len(d), "util_mean": u.mean(), "util_median": u.median(),
                     "mem_max_mib": m.max()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d)
        print(summary(d).round(1).to_string(index=False))
