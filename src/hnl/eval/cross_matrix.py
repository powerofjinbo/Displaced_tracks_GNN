"""Headroom crossed comparisons (brief §5g, M4), val only: the metric-learning graphs of arm G scored by the GNN of
arm N (same seed; slurm/baseline/cross_eval.sbatch -> RUNS/cross/graph<G>_gnn<N>_s<S>/eval_val), with the baseline
runs (G = N) on the diagonal. Every pair is evaluated at its own preregistered §4 cut for FR* (largest pooled-HNL val
fake rate <= FR* on the 0.01 grid). Reported per pair: the efficiencies (primary population, HNL daughters, W muons,
ttbar prompt), the pair efficiency, and the connectivity decomposition of the primary population (graph = 1 - chain
ceiling of graph G, classifier = ceiling - connected, track building = connected - actual; DECISIONS.md). Rows =
graph arm, columns = GNN arm: a column that wins in every row means the GNN's training set matters, a row that wins
in every column means the graph does.

    python -m hnl.eval.cross_matrix --runs RUNS/baseline_v1 (--wp WORKING_POINTS_JSON | --fr-star FR) --out OUT
        [--seeds 0]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import curves, cut_at_fake_rate, load_run, populations
from hnl.eval.headroom import run_table
from hnl.eval.particles import particle_table

ARMS = ("i", "ii", "iii")
COLS = ["eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt", "fake_rate_hnl", "fake_rate_ttbar",
        "pair_eff", "pair_merge_rate"]


def pair_dir(runs: str, g: str, n: str, s: int) -> str:
    return os.path.join(runs, f"arm{g}_s{s}") if g == n else os.path.join(runs, "cross", f"graph{g}_gnn{n}_s{s}")


def evaluate_pair(d: str, fr_star: float, info: pd.DataFrame | None) -> tuple[dict, pd.DataFrame]:
    pp, pe = load_run(d)
    if info is None:
        info = particle_table(np.unique(pe.event_id))
    c = curves(pp, pe, info)
    cut, reachable = cut_at_fake_rate(c, fr_star)
    t = run_table(d, cut, info)
    m = populations(t)["primary"]
    prim = t[m]
    row = {"cut": cut, "reachable": bool(reachable), **{k: float(c.loc[cut, k]) for k in COLS if k in c},
           "n_primary": int(m.sum()), "ceiling_primary": float(prim.ceiling.mean()),
           "connected_primary": float(prim.connected.mean()), "actual_primary": float(prim.actual.mean())}
    row["loss_graph"] = 1 - row["ceiling_primary"]
    row["loss_classifier"] = row["ceiling_primary"] - row["connected_primary"]
    row["loss_trackbuilding"] = row["connected_primary"] - row["actual_primary"]
    return row, info


def plot(tab: pd.DataFrame, out_name: str = "cross_matrix") -> list[str]:
    from hnl.plots import style
    import matplotlib.pyplot as plt
    mean = tab.groupby(["graph_arm", "gnn_arm"])[["eff_primary", "loss_classifier"]].mean()
    paths = []
    for col, title in (("eff_primary", "primary-population efficiency at FR*"),
                       ("loss_classifier", "classifier loss (primary population)")):
        mat = np.full((3, 3), np.nan)
        for i, g in enumerate(ARMS):
            for j, n in enumerate(ARMS):
                if (g, n) in mean.index:
                    mat[i, j] = mean.loc[(g, n), col]
        fig, ax = style.new(figsize=(7, 6))
        im = ax.imshow(mat, cmap="viridis", origin="upper")
        for i in range(3):
            for j in range(3):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="w", fontsize=14)
        ax.set_xticks(range(3), [f"GNN ({a})" for a in ARMS])
        ax.set_yticks(range(3), [f"graph ({a})" for a in ARMS])
        fig.colorbar(im, ax=ax, label=title)
        ax.set_title(f"{style.LABEL}\nval, seeds " + ",".join(map(str, sorted(tab.seed.unique()))), fontsize=12)
        paths += style.save(fig, f"{out_name}_{col}", "m4")
        plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--wp", default=None)
    ap.add_argument("--fr-star", type=float, default=None, help="previews only, without working points")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", type=int, default=1)
    a = ap.parse_args(argv)
    if a.wp:
        fr = float(json.load(open(a.wp))["FR_star"])
    elif a.fr_star is not None:
        fr = a.fr_star
    else:
        raise SystemExit("give --wp (or --fr-star for a preview)")
    os.makedirs(a.out, exist_ok=True)
    rows, missing, info = [], [], None
    for s in map(int, a.seeds.split(",")):
        for g in ARMS:
            for n in ARMS:
                d = pair_dir(a.runs, g, n, s)
                if not os.path.exists(os.path.join(d, "eval.done")):
                    missing.append(os.path.relpath(d, a.runs))
                    continue
                row, info = evaluate_pair(d, fr, info)
                rows.append({"seed": s, "graph_arm": g, "gnn_arm": n, **row})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "cross_matrix.csv"), index=False)
    with open(os.path.join(a.out, "cross_matrix.json"), "w") as f:
        json.dump({"FR_star": fr, "source": a.wp or "--fr-star (preview)", "missing": missing}, f, indent=1)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    if missing:
        print("missing:", missing)
    if a.figs and len(tab):
        for p in plot(tab):
            print(p)


if __name__ == "__main__":
    main()
