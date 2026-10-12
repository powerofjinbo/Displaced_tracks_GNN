"""M4 figures (headroom, val): the baseline's inefficiency split into graph, classifier and track-building losses per
bin (stacked bars per arm, seed means), and daughter-pair efficiency / merge rate vs opening angle and r_v.

    python -m hnl.plots.m4 --headroom DIR (hnl.eval.headroom output)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hnl.plots import style

XLAB = {"d0": "truth |d0| [mm]", "rv": "truth r_v [mm]", "r_first": "first-hit radius [mm]",
        "n_pixel_hits": "pixel hits", "pt": "truth pT [GeV]", "pair_angle": "daughter opening angle [rad]"}
COLORS = {"loss_graph": "#5790fc", "loss_classifier": "#f89c20", "loss_trackbuilding": "#e42536"}
NAMES = {"loss_graph": "graph stage", "loss_classifier": "classifier", "loss_trackbuilding": "track building"}
SUFFIX = "_conn"   # the connectivity decomposition (RESULTS.md M4); "_sp" same-particle, "" chain truth


def plot_losses(losses: pd.DataFrame, population: str, var: str) -> list[str]:
    d = losses[(losses.population == population) & (losses.variable == var)]
    arms = sorted(d.arm.unique())
    if d.empty:
        return []
    bins = list(dict.fromkeys(d.bin))
    fig, ax = style.new(figsize=(10, 6))
    width = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        a = d[d.arm == arm].set_index("bin").reindex(bins)
        x = np.arange(len(bins)) + (k - (len(arms) - 1) / 2) * width
        bottom = np.zeros(len(bins))
        for col in ("loss_graph", "loss_classifier", "loss_trackbuilding"):
            v = a[col + SUFFIX].fillna(0).to_numpy() if col + SUFFIX in a else a[col].fillna(0).to_numpy()
            ax.bar(x, v, width * 0.95, bottom=bottom, color=COLORS[col], edgecolor="black", linewidth=0.5,
                   label=NAMES[col] if k == 0 else None, hatch=["", "//", ".."][k % 3])
            bottom += v
        for xi, b in zip(x, bottom):
            ax.text(xi, b, arm, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(bins)), bins, rotation=45, ha="right", fontsize=10)
    style.axis_labels(ax, XLAB.get(var, var), "inefficiency (1 - DM efficiency)")
    ax.set_ylim(0, max(0.05, 1.3 * float((1 - d.actual).max())))
    ax.legend(fontsize=10, loc="upper right")
    style.label(ax, f"val, {population}, arms i / ii / iii at FR*, seed mean; connectivity decomposition")
    return style.save(fig, f"losses_{population}_vs_{var}", "m4")


def plot_pairs(pairs: pd.DataFrame) -> list[str]:
    figs = []
    for var in ("pair_angle", "rv"):
        d = pairs[pairs.variable == var].groupby(["arm", "bin"], sort=False)[["pair_eff", "merge_rate"]].mean()
        if d.empty:
            continue
        fig, ax = style.new()
        for arm, g in d.groupby(level=0):
            bins = [b for _, b in g.index]
            ax.plot(range(len(bins)), g.merge_rate.to_numpy(), marker="o", label=f"arm {arm}: merge rate")
            ax.plot(range(len(bins)), 1 - g.pair_eff.to_numpy(), marker="s", ls="--", label=f"arm {arm}: 1 - pair eff")
        ax.set_xticks(range(len(bins)), bins, rotation=45, ha="right", fontsize=10)
        style.axis_labels(ax, XLAB[var], "fraction of daughter pairs")
        ax.legend(fontsize=9)
        style.label(ax, "val, HNL events with two target daughters, FR*")
        figs += style.save(fig, f"pairs_vs_{var}", "m4")
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", required=True)
    a = ap.parse_args(argv)
    losses = pd.read_csv(os.path.join(a.headroom, "losses.csv"))
    figs = []
    for pop in ("primary", "hnl_targets", "w_muons"):
        for var in ("d0", "rv", "r_first", "n_pixel_hits", "pt", "pair_angle"):
            figs += plot_losses(losses, pop, var)
    figs += plot_pairs(pd.read_csv(os.path.join(a.headroom, "pairs_per_seed.csv")))
    print("\n".join(figs))


if __name__ == "__main__":
    main()
