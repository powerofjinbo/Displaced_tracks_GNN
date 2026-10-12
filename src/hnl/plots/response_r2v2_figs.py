"""Figure of the revised response to the supervisor's second round (docs/feedback/response_andre_r2/response_v2.tex):
Canto-L trained on HNL events (arm ii, seed 0) against Canto-L trained on ttbar (arm i) and acorn (HNL), validation.

R2_trainsample.pdf:
  (a) track efficiency of HNL-daughter targets per |d0| bin, every model with the builder family FAMILY at its val
      working point (results/training_sample_val_<FAMILY>.csv, hnl.eval.training_sample_board): Canto-L (ttbar) and
      acorn (HNL) as the mean of three seeds, Canto-L (HNL) seed 0; error bars: binomial (one seed) or the seed spread
      combined with the binomial error of the three-seed mean;
  (b) chain-edge efficiency of HNL daughters per |d0| bin at the common edge working point e* (RUN/edge_tpr_estar.csv,
      hnl.eval.edge_tpr), Canto-L (ttbar) seed 0 and Canto-L (HNL) seed 0;
  (c) classifier-stage loss rate (graph ceiling DM-matched, score ceiling at s0 = 0.2 not) of the primary HNL
      daughters per |d0| / r_first bin (results/oblique_val_ref_v1_particles.csv, results/oblique_val_ii_v1_particles.csv,
      hnl.eval.oblique_val); three seeds pooled for Canto-L (ttbar) and acorn (HNL).

    source env/activate.sh && python src/hnl/plots/response_r2v2_figs.py [--family mwslmerge]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "paper", "figures"))
sys.dont_write_bytecode = True
import paperstyle as ps  # noqa: E402

OUT = os.path.join(REPO, "docs", "feedback", "response_andre_r2")
RES = os.path.join(REPO, "results")
RUNS = ps.RUNS
CANTO_HNL = ps.lighter(ps.CANTO_L, 0.45)
D0_TRACK = [("d0_10_30", "10–30"), ("d0_30_100", "30–100"), ("d0_100_300", "100–300"), ("d0_300_inf", "≥300")]
EDGE_BINS = ["[10.0, 30.0)", "[30.0, 100.0)", "[100.0, 300.0)", "[300.0, inf)"]
EDGE_LAB = ["10–30", "30–100", "100–300", "≥300"]
RATIO = [(0.0, 0.5, "< 0.5"), (0.5, 0.7, "0.5–0.7"), (0.7, 1.01, "≥ 0.7")]


def wilson(k: float, n: float, z: float = 1.0) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def track_panel(ax, family: str) -> dict:
    t = pd.read_csv(os.path.join(RES, f"training_sample_val_{family}.csv"))
    out = {}
    x = np.arange(len(D0_TRACK))
    for k, (model, col, mk, filled) in enumerate([("acorn (HNL)", ps.ACORN_HNL, "s", True),
                                                   ("Canto-L (ttbar)", ps.CANTO_L, "o", True),
                                                   ("Canto-L (HNL)", CANTO_HNL, "D", True)]):
        g = t[t.model == model]
        if g.empty:
            raise SystemExit(f"{family}: no rows for {model}")
        vals, lo, hi = [], [], []
        for q, _ in D0_TRACK:
            eff = g[q].to_numpy()
            kk, nn = g[q + "_k"].sum(), g[q + "_n"].sum()
            m = eff.mean()
            a, b = wilson(kk, nn)
            if len(g) > 1:     # seed spread of the mean added in quadrature to the binomial interval of the pool
                s = eff.std(ddof=1) / np.sqrt(len(g))
                a, b = m - np.hypot(m - a, s), m + np.hypot(b - m, s)
            vals.append(100 * m)
            lo.append(100 * (m - a))
            hi.append(100 * (b - m))
            out[(model, q)] = (int(kk), int(nn), len(g))
        dx = (k - 1) * 0.12
        ax.errorbar(x + dx, vals, yerr=[lo, hi], ls="-", lw=1.0, marker=mk, ms=4.0, capsize=0, elinewidth=0.8,
                    label=f"{model} ({len(g)} seed{'s' if len(g) > 1 else ''})", **ps.marker_kw(col, filled))
    ps.cat_axis(ax, [b for _, b in D0_TRACK], size=7.0)
    ax.set_xlim(-0.5, len(D0_TRACK) - 0.5)
    ax.set_ylim(70, 102)
    ax.set_xlabel("$|d_0|$ [mm]")
    ax.set_ylabel("Track efficiency [%]")
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1.6, borderaxespad=0.4)
    return out


def edge_panel(ax) -> dict:
    out = {}
    x = np.arange(len(EDGE_BINS))
    for k, (name, run, col, mk) in enumerate([
            ("Canto-L (ttbar) s0", os.path.join(RUNS, "stage1_final_learned", "l1tl", "armi_s0"), ps.CANTO_L, "o"),
            ("Canto-L (HNL) s0", os.path.join(RUNS, "stage1_final_learned_ts", "l1tl", "armii_s0"), CANTO_HNL, "D")]):
        e = pd.read_csv(os.path.join(run, "edge_tpr_estar.csv"))
        e = e[(e.population == "hnl_daughters") & (e.variable == "d0_bin")].set_index("bin")
        col_eff = [c for c in e.columns if c.startswith("edge_eff@")][0]
        v = [100 * float(e.loc[b, col_eff]) for b in EDGE_BINS]
        n = [int(e.loc[b, "n_true_edges"]) for b in EDGE_BINS]
        ax.plot(x + (k - 0.5) * 0.1, v, ls="-", lw=1.0, marker=mk, ms=4.0, label=f"{name}, cut {col_eff[9:]}",
                **ps.marker_kw(col))
        out[name] = (v, n)
    ps.cat_axis(ax, EDGE_LAB, size=7.0)
    ax.set_xlim(-0.5, len(EDGE_BINS) - 0.5)
    ax.set_ylim(70, 102)
    ax.set_xlabel("$|d_0|$ [mm]")
    ax.set_ylabel("Chain edges kept at $e^*$ [%]")
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1.6, borderaxespad=0.4)
    return out


def oblique_panel(ax) -> tuple[list, dict]:
    ref = pd.read_csv(os.path.join(RES, "oblique_val_ref_v1_particles.csv"))
    ii = pd.read_csv(os.path.join(RES, "oblique_val_ii_v1_particles.csv"))
    p = pd.concat([ref, ii], ignore_index=True)
    p["fam"] = p.label.str.replace(r" s\d$", "", regex=True)
    out, series = {}, []
    for fam, lab, col in [("acorn (HNL)", "acorn (HNL), 3 seeds", ps.ACORN_HNL),
                          ("Canto-L (i)", "Canto-L (ttbar), 3 seeds", ps.CANTO_L),
                          ("Canto-L (ii)", "Canto-L (HNL), seed 0", CANTO_HNL)]:
        g = p[(p.fam == fam) & (p.stage != "graph")]
        vals, errs = [], []
        for lo, hi, name in RATIO:
            h = g[(g.ratio >= lo) & (g.ratio < hi)]
            k, n = int((h.stage == "classifier").sum()), len(h)
            vals.append(100 * k / n)
            a, b = wilson(k, n)
            errs.append(100 * (b - a) / 2)
            out[(fam, name)] = (k, n)
        series.append({"label": lab, "color": col, "values": vals, "errors": errs})
    handles, _ = ps.grouped_bars(ax, [n for *_, n in RATIO], series, width=0.8, fmt="{:.1f}", fontsize=6.5)
    ps.group_axis(ax, [n for *_, n in RATIO], size=7.0)
    ps.bar_frame(ax)
    ax.set_ylim(0, 17)
    ax.set_xlabel("$|d_0|\\,/\\,r_\\mathrm{first}$")
    ax.set_ylabel("Classifier-stage losses [%]")
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, handlelength=1.1, borderaxespad=0.4)
    return handles, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", default="mwslmerge", choices=["mwslmerge", "mwslayermerge"])
    a = ap.parse_args(argv)
    fig = ps.figure(ps.TEXT_W, 2.55)
    gs = fig.add_gridspec(1, 3)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    tr = track_panel(axes[0], a.family)
    ed = edge_panel(axes[1])
    _, ob = oblique_panel(axes[2])
    for ax, (letter, text) in zip(axes, [("a", None), ("b", None), ("c", None)]):
        ps.tag(ax, letter, loc="upper right", text=text)
    for t in ps.header(axes[0], "Validation, HNL daughters", ax_last=axes[2]):
        if t is not None:
            t.set_in_layout(False)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.93))
    files = ps.save(fig, "R2_trainsample", outdir=OUT)
    rows = [{"panel": "a", "model": m, "bin": q, "k": k, "n": n, "seeds": s} for (m, q), (k, n, s) in tr.items()]
    rows += [{"panel": "b", "model": m, "bin": b, "edge_eff_pct": v, "n_edges": n}
             for m, (vs, ns) in ed.items() for b, v, n in zip(EDGE_BINS, vs, ns)]
    rows += [{"panel": "c", "model": m, "bin": b, "k": k, "n": n} for (m, b), (k, n) in ob.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"R2_trainsample_{a.family}.csv"), index=False)
    print("\n".join(files))


if __name__ == "__main__":
    main()
