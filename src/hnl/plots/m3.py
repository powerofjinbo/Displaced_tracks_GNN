"""M3 figures: baseline arms on val (efficiency vs fake rate over the cut grid) and on test (efficiency vs |d0|, r_v,
pT at FR*, with the preregistered uncertainties), and the paper comparison (arm ii, external set, Fig. 3 |d0| bins).

    python -m hnl.plots.m3 --val VAL_DIR (baseline_val output) [--test TEST_DIR (test_stats output)]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from hnl.plots import style

PAPER_D0 = [  # Fig. 3 |d0| panel, displaced (docs/paper_facts.md): (lo, hi, eff, err)
    (0, 50, 0.9622, 0.0058), (50, 100, 0.9023, 0.0117), (100, 150, 0.8575, 0.0185), (150, 200, 0.8769, 0.0201),
    (200, 250, 0.8871, 0.0232), (250, 300, 0.9320, 0.0208), (300, 350, 0.8817, 0.0335), (350, 400, 0.8537, 0.0552),
    (400, 450, 0.8235, 0.0654), (450, 500, 0.9615, 0.0377)]
ARM_LABEL = {"i": "acorn arm (i) ttbar", "ii": "acorn arm (ii) HNL", "iii": "acorn arm (iii) ttbar+HNL"}


def val_curves(val_dir: str):
    out = {}
    for f in sorted(glob.glob(os.path.join(val_dir, "curves_arm*_s*.csv"))):
        arm, seed = re.match(r"curves_arm(\w+)_s(\d+)\.csv", os.path.basename(f)).groups()
        out.setdefault(arm, []).append(pd.read_csv(f, index_col=0))
    return out


def plot_val(val_dir: str) -> list[str]:
    cv = val_curves(val_dir)
    figs = []
    for pop, lab in (("eff_primary", "HNL daughters, |d0| >= 10 mm or r_v >= 100 mm"), ("eff_w_muons", "W muons"),
                     ("eff_all_targets", "all targets (HNL events)")):
        fig, ax = style.new()
        for arm, frames in sorted(cv.items()):
            m = pd.concat(frames).groupby(level=0).mean()
            ax.plot(m.fake_rate_hnl, m[pop], marker=".", label=ARM_LABEL.get(arm, arm))
        ax.set_xscale("log")
        style.axis_labels(ax, "track fake rate (pooled HNL val, tracks >= 5 hits)", f"DM efficiency, {lab}", 16)
        ax.legend(fontsize=10, loc="lower right")
        style.label(ax, "val, seed mean, edge-score cut scanned")
        figs += style.save(fig, f"val_{pop}_vs_fake_rate", "m3")
    return figs


def plot_test(test_dir: str) -> list[str]:
    b = pd.read_csv(os.path.join(test_dir, "test_binned.csv"))
    figs = []
    for var, xl in (("absd0", "truth |d0| [mm]"), ("rv", "truth r_v [mm]"), ("pt", "truth pT [GeV]")):
        fig, ax = style.new()
        for i, (arm, d) in enumerate(b[(b.variable == var) & (b.population == "hnl_targets")
                                       & (b.model != "paper_comparison_arm_ii_s_star")].groupby("model")):
            x = np.arange(len(d)) + (i - 1) * 0.12
            ax.errorbar(x, d["mean"], yerr=d["sigma"], marker="o", ls="none", capsize=2, label=ARM_LABEL.get(arm, arm))
        lab = ["no vertex" if not np.isfinite(lo) else (f"[{lo:g},{hi:g})" if np.isfinite(hi) else f">={lo:g}")
               for lo, hi in zip(d.lo, d.hi)]
        ax.set_xticks(np.arange(len(d)), lab, rotation=45, ha="right", fontsize=11)
        style.axis_labels(ax, xl, "DM efficiency, HNL-daughter targets")
        ax.legend(fontsize=10, loc="lower left")
        style.label(ax, "test (HNL joint test + external), FR* working point, 3-seed mean")
        figs += style.save(fig, f"test_hnl_eff_vs_{var}", "m3")
    p = b[(b.variable == "absd0_paper") & np.isfinite(b.lo)]
    q = b[(b.variable == "rv_paper") & np.isfinite(b.lo)]
    if len(p):
        fig, ax = style.new()
        c = 0.5 * (p.lo + np.where(np.isfinite(p.hi), p.hi, p.lo + 50))
        ax.errorbar(c, p["mean"], yerr=p["sigma"], marker="o", ls="none", capsize=2,
                    label="acorn arm (ii), this work, vs truth |d0| (s*, external set)")
        if len(q):
            cq = 0.5 * (q.lo + np.where(np.isfinite(q.hi), q.hi, q.lo + 50))
            ax.errorbar(cq - 5, q["mean"], yerr=q["sigma"], marker="D", ls="none", capsize=2, mfc="none",
                        label="acorn arm (ii), this work, vs r_v")
        pc = np.array([0.5 * (lo + hi) for lo, hi, _, _ in PAPER_D0])
        ax.errorbar(pc + 5, [e for *_, e, _ in PAPER_D0], yerr=[s for *_, s in PAPER_D0], marker="s", ls="none",
                    capsize=2, label="Wang et al. 2022, Fig. 3 (displaced)")
        style.axis_labels(ax, "truth |d0| [mm] (paper: 'truth |d0|')", "track efficiency, HNL daughters")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=10, loc="lower left")
        style.label(ax, "15 GeV / 100 mm sample")
        figs += style.save(fig, "paper_comparison_d0", "m3")
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default=None)
    a = ap.parse_args(argv)
    figs = plot_val(a.val)
    if a.test:
        figs += plot_test(a.test)
    print("\n".join(figs))


if __name__ == "__main__":
    main()
