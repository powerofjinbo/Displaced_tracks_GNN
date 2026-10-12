"""Figures of the response to the supervisor's second round of feedback (docs/feedback/response_andre_r2/).

R2_far.pdf: HNL daughters with |d0| >= 300 mm lost by each model and builder, (a) validation (19 daughters per seed,
57 daughter-seeds) and (b) test (36 per seed, 108; post-freeze), for MWS-layer + learned merging (LM) and MWS-layer +
helix merging (MWS+M); (c) where they are lost on validation (round-4 check 2: graph, classifier, track building).
Counts: results/strong_baselines_v3*.csv (k_n_d0_300), results/strong_baselines_v1.csv (Canto-L + MWS+M on test),
results/round4_checks_far_summary.csv. Uses the paper's plotting style (paper/figures/paperstyle.py).

    source env/activate.sh && python src/hnl/plots/response_r2_figs.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "paper", "figures"))
sys.dont_write_bytecode = True
import paperstyle as ps  # noqa: E402

OUT = os.path.join(REPO, "docs", "feedback", "response_andre_r2")
R = os.path.join(REPO, "results")


def found(split: str, label: str, src: str = "strong_baselines_v3.csv") -> tuple[int, int]:
    d = pd.read_csv(os.path.join(R, src))
    q = d[(d.set == split) & (d.label == label)]
    if len(q) != 1:
        raise SystemExit(f"{split} {label}: {len(q)} rows in {src}")
    k, n = (int(x) for x in q.k_n_d0_300.iloc[0].split("/"))
    return k, n


def main():
    rows = {
        ("Canto-L", "LM"): ("Canto-L (frozen: learned merging)", "strong_baselines_v3.csv"),
        ("Canto-L", "MWS+M"): (None, None),
        ("acorn (HNL)", "LM"): ("acorn (HNL) + MWS-layer + learned merging (Canto-L's builder)", "strong_baselines_v3.csv"),
        ("acorn (HNL)", "MWS+M"): ("acorn (HNL) + MWS-layer + merging (Canto builder)", "strong_baselines_v3.csv"),
        ("acorn (ttbar)", "LM"): ("acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder)", "strong_baselines_v3.csv"),
        ("acorn (ttbar)", "MWS+M"): ("acorn (ttbar) + MWS-layer + merging (Canto builder)", "strong_baselines_v3.csv"),
    }
    cnt = {}
    for split in ("val", "test"):
        for (m, b), (lab, src) in rows.items():
            if lab is None:                       # Canto-L + MWS+M: val in v3, test in v1
                if split == "val":
                    lab = "Canto-L + MWS-layer + merging (Canto builder), same builder (not the frozen builder)"
                    k, n = found("val", lab)
                else:
                    d = pd.read_csv(os.path.join(R, "strong_baselines_v1.csv"))
                    q = d[(d.set == "test") & d.label.str.startswith("Canto-L + MWS-layer + merging")]
                    k, n = (int(x) for x in q.k_n_d0_300.iloc[0].split("/"))
            else:
                k, n = found(split, lab, src)
            cnt[(split, m, b)] = (k, n)
    expect = {("val", "Canto-L", "LM"): (50, 57), ("val", "Canto-L", "MWS+M"): (49, 57),
              ("val", "acorn (HNL)", "LM"): (52, 57), ("val", "acorn (HNL)", "MWS+M"): (51, 57),
              ("test", "Canto-L", "LM"): (94, 108), ("test", "Canto-L", "MWS+M"): (101, 108),
              ("test", "acorn (HNL)", "LM"): (101, 108), ("test", "acorn (HNL)", "MWS+M"): (101, 108)}
    for k, v in expect.items():
        if cnt[k] != v:
            raise SystemExit(f"{k}: {cnt[k]} != {v}")
    models = [("acorn (ttbar)", ps.ACORN_TTBAR), ("acorn (HNL)", ps.ACORN_HNL), ("Canto-L", ps.CANTO_L)]
    groups = ["MWS-layer +\nlearned merging", "MWS-layer +\nhelix merging"]
    builders = ["LM", "MWS+M"]

    far = pd.read_csv(os.path.join(R, "round4_checks_far_summary.csv"))
    far = far[(far.seed == "pooled") & (far.d0_min_mm == 300.0)].set_index("model")

    fig = ps.figure(ps.TEXT_W, 2.7)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    axes = []
    for j, split in enumerate(("val", "test")):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        series = [{"label": m, "color": c,
                   "values": [cnt[(split, m, b)][1] - cnt[(split, m, b)][0] for b in builders],
                   "errors": None} for m, c in models]
        handles, _ = ps.grouped_bars(ax, groups, series, width=0.8, fmt="{:.0f}", fontsize=6.5)
        ps.group_axis(ax, groups, size=6.5)
        ps.bar_frame(ax)
        n = cnt[(split, "Canto-L", "LM")][1]
        ax.set_ylim(0, 21)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel(f"Daughter-seeds lost (of {n})")
        ps.tag(ax, "a" if j == 0 else "b", text="Validation" if j == 0 else "Test, post-freeze", fontsize=7.0)
    axes[1].legend(handles=handles, loc="upper right", fontsize=6.5, handlelength=1.1, borderaxespad=0.3,
                   bbox_to_anchor=(1.0, 0.88))
    # (c) loss classes on validation
    ax = fig.add_subplot(gs[0, 2])
    names = [("Canto-L (frozen, learned merging)", "Canto-L\nLM"), ("Canto-L + MWS-layer + merging", "Canto-L\nMWS+M"),
             ("acorn (HNL) val-best (MWS-layer + learned merging)", "acorn (HNL)\nLM"),
             ("acorn (HNL) + MWS-layer + merging", "acorn (HNL)\nMWS+M")]
    cls = [("graph", "graph", ps.INK), ("classifier", "classifier", ps.CANTO_H),
           ("builder", "track building", ps.lighter(ps.ACORN_HNL, 0.25))]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    rows_c = []
    for key, lab, col in cls:
        if key == "builder":
            v = np.array([far.loc[m, ["fragmented", "impure", "short"]].sum() for m, _ in names], float)
        else:
            v = np.array([far.loc[m, key] for m, _ in names], float)
        ax.bar(x, v, bottom=bottom, width=0.62, color=col, edgecolor=ps.darker(col, 0.25), lw=0.5, label=lab, zorder=3)
        for xi, b0, vi in zip(x, bottom, v):
            if vi > 0:
                ax.annotate(f"{vi:.0f}", (xi, b0 + vi / 2), ha="center", va="center", fontsize=6.5,
                            color="white" if key != "builder" else ps.INK, zorder=4)
        bottom += v
        rows_c.append({"class": lab, **{n: int(vv) for (_, n), vv in zip(names, v)}})
    ps.cat_axis(ax, [n for _, n in names], size=6.5)
    ax.tick_params(axis="x", which="major", length=0)
    ps.bar_frame(ax)
    ax.set_ylim(0, 10)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel("Daughter-seeds lost (of 57)")
    ps.tag(ax, "c", text="Validation: where", fontsize=7.0)
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.1, borderaxespad=0.3, bbox_to_anchor=(1.0, 0.86))
    for t in ps.header(axes[0], "HNL daughters, $|d_0|$ \u2265 300 mm", ax_last=ax):
        if t is not None:
            t.set_in_layout(False)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))
    os.makedirs(OUT, exist_ok=True)
    files = ps.save(fig, "R2_far", outdir=OUT)
    pd.DataFrame([{"split": s, "model": m, "builder": b, "found": k, "n": n}
                  for (s, m, b), (k, n) in cnt.items()]).to_csv(os.path.join(OUT, "R2_far_counts.csv"), index=False)
    pd.DataFrame(rows_c).to_csv(os.path.join(OUT, "R2_far_classes_val.csv"), index=False)
    print("\n".join(files))


if __name__ == "__main__":
    main()
