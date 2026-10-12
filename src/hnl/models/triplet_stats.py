"""Triplet counts and feature distributions on saved metric-learning graphs (sizing the Stage-1 B triplet model).

A triplet is a pair of graph edges i -> j -> k (oriented outward in R = sqrt(r^2 + z^2)) sharing hit j; it is "true"
if both edges are chain-truth edges of one particle. Reports per event the number of edges, of triplets and of true
triplets, and quantiles of |turn|, |dcot|, |kappa| for true triplets, plus the fraction of all / true triplets kept by
simple geometric cuts.

    python -m hnl.models.triplet_stats --graphs DIR [--max-events 200]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from hnl.graphs import ceiling as C
from hnl.models import geometry_features as GF

CUTS = [(0.1, 0.2), (0.2, 0.5), (0.3, 1.0), (0.5, 2.0)]   # (|turn| max [rad], |dcot| max)


def main(argv=None):
    from acorn.utils.loading_utils import load_pyg
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--max-events", type=int, default=200)
    ap.add_argument("--pattern", default="event*.pyg.gz", help="e.g. event2*.pyg.gz for hnl10_100 events only")
    a = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(a.graphs, a.pattern)))[:a.max_events]
    rows, true_feats = [], []
    for p in paths:
        g = load_pyg(p[:-3], weights_only=False)
        r = (g.r if "r" in g else g.hit_r).double()
        phi = (g.phi if "phi" in g else g.hit_phi).double()
        z = (g.z if "z" in g else g.hit_z).double()
        n = r.shape[0]
        e = torch.from_numpy(C.unordered_unique(g.edge_index.numpy()))
        R = torch.sqrt(r * r + z * z)
        fwd = (R[e[0]] < R[e[1]]) | ((R[e[0]] == R[e[1]]) & (e[0] < e[1]))
        src, dst = torch.where(fwd, e[0], e[1]), torch.where(fwd, e[1], e[0])
        te = C.unordered_unique(g.track_edges.numpy())
        tcode = set((te[0].astype(np.int64) * n + te[1]).tolist())
        lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
        is_true = torch.tensor([(int(x) * n + int(y)) in tcode for x, y in zip(lo, hi)], dtype=torch.bool)
        pid = g.hit_particle_id
        ei, eo = GF.build_triplets(src, dst, n)
        f = GF.triplet_features(r, phi, z, src, dst, ei, eo)
        tt = is_true[ei] & is_true[eo] & (pid[src[ei]] == pid[dst[eo]])
        row = {"event": os.path.basename(p), "n_edges": len(src), "n_triplets": len(ei), "n_true_triplets": int(tt.sum())}
        for k in (2, 4, 8):
            keep = GF.select_triplets(ei, eo, GF.straightness(f), k)
            row[f"topk{k}_n"] = int(keep.sum())
            row[f"topk{k}_true"] = float(keep[tt].float().mean()) if tt.any() else np.nan
        for tmax, dmax in CUTS:
            keep = (f[:, 1].abs() < tmax) & (f[:, 2].abs() < dmax)
            row[f"keep_all_{tmax}_{dmax}"] = float(keep.float().mean()) if len(keep) else np.nan
            row[f"keep_true_{tmax}_{dmax}"] = float(keep[tt].float().mean()) if tt.any() else np.nan
        rows.append(row)
        true_feats.append(f[tt].numpy())
    d = pd.DataFrame(rows)
    tf = np.concatenate(true_feats)
    pd.set_option("display.width", 200)
    print(d.drop(columns="event").describe().round(4).to_string())
    for i, name in enumerate(["kappa_1_per_m", "turn", "dcot", "angle3d", "len_ratio"]):
        print(name, "true-triplet |x| quantiles 0.5/0.9/0.99/0.999:", np.quantile(np.abs(tf[:, i]), [0.5, 0.9, 0.99, 0.999]).round(4))


if __name__ == "__main__":
    main()
