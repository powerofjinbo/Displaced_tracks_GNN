"""Presentation figures for the Canto slides (figures/slides; PNG at 200 dpi + PDF).

event_display, event_display_rz
    Transverse (x-y) and r-z views of one real HNL validation event: every hit of the event in light grey, the hits
    of the W muon (prompt) in blue and of the two HNL-daughter muons in orange (each muon's hits joined along its
    truth chain), a star at the HNL decay vertex, a cross at the origin and a dotted line for the flight of the
    neutral HNL from the primary vertex. The HNL itself is not in the particle table; its decay vertex is the
    two-track vertex of the daughters' truth-hit helix fits (docs/definitions.md section 3, truth_fits/v1
    vertices_<sample>.parquet), the primary vertex the W muon's point of closest approach to the beam line (truth-hit
    fit). Event choice (unless --event): val events of the sample with an HNL decay radius r_v in [150, 500] mm, two
    HNL-daughter targets with >= 6 hits and |eta| < 1 and a transverse opening angle >= 0.3 rad, one W-muon target
    with |eta| < 1; the largest r_v wins (the daughters start farthest out). The choice is written to
    event_display.json.

edge_tpr_vs_d0
    The pointing shortcut: edge true-positive rate of the in-graph chain-true edges of HNL-daughter targets vs truth
    |d0| (bins of the CSV) at one target edge false-positive rate, for the equal-budget acorn-architecture IGNN and the
    Canto classifier (L1 + truncation), both trained on ttbar only at the screening budget (hnl.eval.edge_tpr outputs
    edge_tpr_estar.csv / _efpr.json). Per-bin counts are recovered exactly from the CSV (in-graph edges = n_true_edges
    x graph_eff, kept = TPR x in-graph edges; checked to be integers); error bars are Clopper-Pearson 68 % intervals
    on edge counts. dD = TPR(|d0| 100-300 mm) - TPR(|d0| 0.1-1 mm) is recomputed from the CSV and must equal the
    facts-sheet value (docs/canto_facts.md section 5: -6.0 and +2.4 points) before it is printed on the figure. The
    plotted numbers are written to edge_tpr_vs_d0.csv.

    python -m hnl.plots.canto_figures [--which event_display,edge_tpr] [--sample hnl15_200] [--event N] [--unsigned-r]
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import beta

from hnl.plots import style  # noqa: I001  (sets the Agg backend and the mplhep style before pyplot is used)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SUBDIR = "slides"
DPI = 200
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
EVENTS = os.path.join(DATA, "derived", "events", "v1")
FITS = os.path.join(DATA, "derived", "truth_fits", "v1")
SCREEN = os.path.join(DATA, "runs", "stage1_screen_ttbar")
SAMPLE_CODE = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}     # hnl.data.acorn_store.SAMPLE_CODE
SAMPLE_TEXT = {"hnl15_200": r"HNL 15 GeV, $c\tau$ = 200 mm", "hnl10_100": r"HNL 10 GeV, $c\tau$ = 100 mm"}

GREY, BLUE, ORANGE, INK = "#c4c4c4", "#1f77b4", "#ff7f0e", "#1a1a1a"

# edge TPR vs |d0|: (legend label, run directory, colour, marker, line style, dD in points from docs/canto_facts.md)
TPR_SERIES = [
    ("IGNN, acorn architecture (ttbar, screening budget)", os.path.join(SCREEN, "ignn", "armi_s0"), "#7f7f7f", "o",
     "--", -6.0),
    ("Canto classifier (L1 + trunc., ttbar)", os.path.join(SCREEN, "symregion_trunc", "armi_s0"), "#ff7f0e", "D",
     "-", +2.4),
]
DD_LOW, DD_HIGH = (0.1, 1.0), (100.0, 300.0)                     # hnl.eval.edge_tpr.DD_LOW / DD_HIGH [mm]


def save(fig, name: str, subdir: str = SUBDIR) -> list[str]:
    """style.save with the PNG at DPI (style.save writes 150 dpi): same directory layout, PNG + PDF, tight bbox."""
    d = os.path.join(style.FIG_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(d, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=DPI if ext == "png" else None)
        out.append(p)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ event display

def candidates(sample: str, rv_range=(150.0, 500.0)) -> pd.DataFrame:
    """Val events of `sample` with an HNL decay radius in rv_range (truth two-track vertex), one row per event with the
    vertex, the daughters' hit counts / |eta| / transverse opening angle and the W muon's |eta|, best first."""
    from hnl.data.convert import read_table
    from hnl.data.splits import read_split
    ev = read_split(sample, "val")
    v = pd.read_parquet(os.path.join(FITS, f"vertices_{sample}.parquet"))
    v = v[v.ok & v.event.isin(ev) & (v.rv >= rv_range[0]) & (v.rv <= rv_range[1])]
    v = v.drop_duplicates("event").set_index("event")
    p = read_table(EVENTS, "particles", sample, events=v.index.to_numpy(),
                   columns=["event", "particle_id", "category", "is_target", "n_hits", "eta", "px", "py"])
    d, w = p[p.category == "hnl"], p[p.category == "w"]
    rows = []
    for e, g in d.groupby("event"):
        we = w[w.event == e]
        if len(g) != 2 or len(we) != 1:
            continue
        phi = np.arctan2(g.py.to_numpy(), g.px.to_numpy())
        rows.append({"event": int(e), "n_daughter_targets": int(g.is_target.sum()), "min_hits": int(g.n_hits.min()),
                     "max_abs_eta": float(g.eta.abs().max()),
                     "open_xy": float(abs(np.angle(np.exp(1j * (phi[0] - phi[1]))))),
                     "w_target": bool(we.is_target.iloc[0]), "w_abs_eta": float(abs(we.eta.iloc[0]))})
    c = pd.DataFrame(rows).set_index("event").join(v[["rv", "vx", "vy", "vz", "pid_a", "pid_b"]], how="inner")
    c["passes"] = ((c.n_daughter_targets == 2) & (c.min_hits >= 6) & (c.max_abs_eta < 1.0) & (c.open_xy >= 0.3)
                   & c.w_target & (c.w_abs_eta < 1.0))
    return c.sort_values(["passes", "rv"], ascending=[False, False])


def load_event(sample: str, event: int) -> dict:
    """Hits (row order, with each muon's hits in truth-chain order), the muons, the HNL decay vertex and the primary
    vertex of one event."""
    from hnl.data.convert import read_table
    from hnl.truth.helix import order_hits
    h = read_table(EVENTS, "hits", sample, events=[event], columns=["event", "row", "x", "y", "z", "pid"])
    h = h.sort_values("row").reset_index(drop=True)
    assert (h.row.to_numpy() == np.arange(len(h))).all(), (sample, event)
    e = read_table(EVENTS, "edges", sample, events=[event])
    te = np.vstack([e.src.to_numpy(np.int64), e.dst.to_numpy(np.int64)])
    pid = h.pid.to_numpy(np.int64)
    chain = h.iloc[order_hits(pid, te)]                     # grouped by particle, first hit first
    p = read_table(EVENTS, "particles", sample, events=[event],
                   columns=["event", "particle_id", "category", "is_target", "n_hits", "pt", "eta"])
    v = pd.read_parquet(os.path.join(FITS, f"vertices_{sample}.parquet"), filters=[("event", "==", int(event))])
    v = v[v.ok]
    if len(v) != 1:
        raise ValueError(f"{sample} event {event}: {len(v)} valid HNL two-track vertices")
    v = v.iloc[0]
    w_id = p.loc[p.category == "w", "particle_id"].to_numpy(np.int64)
    d_ids = p.loc[p.category == "hnl", "particle_id"].to_numpy(np.int64)
    if len(w_id) != 1 or len(d_ids) != 2:
        raise ValueError(f"{sample} event {event}: {len(w_id)} W muons, {len(d_ids)} HNL daughters")
    f = pd.read_parquet(os.path.join(FITS, f"particles_{sample}.parquet"),
                        columns=["event", "particle_id", "fix_xpca", "fix_ypca", "fix_z0"],
                        filters=[("event", "==", int(event)), ("particle_id", "==", int(w_id[0]))])
    if len(f) != 1 or not np.isfinite(f[["fix_xpca", "fix_ypca", "fix_z0"]].to_numpy()).all():
        raise ValueError(f"{sample} event {event}: no truth-hit fit of the W muon")
    muon = lambda i: chain[chain.pid == i][["x", "y", "z"]].to_numpy(np.float64)
    return {"sample": sample, "event": int(event), "event_id": SAMPLE_CODE[sample] * 100_000_000 + int(event),
            "hits": h[["x", "y", "z"]].to_numpy(np.float64), "w": muon(w_id[0]), "daughters": [muon(i) for i in d_ids],
            "w_id": int(w_id[0]), "daughter_ids": [int(i) for i in d_ids],
            "vertex": np.array([v.vx, v.vy, v.vz], dtype=np.float64), "rv": float(v.rv),
            "pv": f[["fix_xpca", "fix_ypca", "fix_z0"]].to_numpy(np.float64)[0],
            "particles": p[p.category != "other"].to_dict("records")}


def _legend_handles():
    return [Line2D([], [], ls="none", marker="o", ms=5, mfc=GREY, mec="none", label="All hits of the event"),
            Line2D([], [], color=BLUE, lw=1.4, marker="o", ms=7, mec="white", mew=0.8, label="W muon (prompt)"),
            Line2D([], [], color=ORANGE, lw=1.4, marker="s", ms=7, mec="white", mew=0.8,
                   label="HNL daughter muons"),
            Line2D([], [], color=INK, lw=1.6, ls=(0, (1.5, 2.0)), label="HNL flight (neutral, no hits)"),
            Line2D([], [], ls="none", marker="*", ms=19, mfc=INK, mec="white", mew=1.0, label="HNL decay vertex"),
            Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK, mew=2.0, label="Primary vertex"),
            Line2D([], [], ls="none", marker="x", ms=12, mew=2.6, color=INK, label="Origin")]


def _draw(ax, ev: dict, proj):
    """Draw one projection; proj maps an (n, 3) array of x, y, z to the 2D plot coordinates."""
    a = proj(ev["hits"])
    ax.scatter(a[:, 0], a[:, 1], s=6, c=GREY, lw=0, zorder=1, label="_hits")
    for trk, col, mk in [(ev["w"], BLUE, "o")] + [(d, ORANGE, "s") for d in ev["daughters"]]:
        t = proj(trk)
        ax.plot(t[:, 0], t[:, 1], "-", color=col, lw=1.4, zorder=3)
        ax.plot(t[:, 0], t[:, 1], ls="none", marker=mk, ms=7, mfc=col, mec="white", mew=0.8, zorder=4)
    fl = proj(np.vstack([ev["pv"], ev["vertex"]]))
    ax.plot(fl[:, 0], fl[:, 1], color=INK, lw=1.6, ls=(0, (1.5, 2.0)), zorder=2)
    vx = proj(ev["vertex"][None])[0]                       # below the hit markers, which it would otherwise hide
    ax.plot(*vx, ls="none", marker="*", ms=21, mfc=INK, mec="white", mew=1.2, zorder=3.5)
    pv = proj(ev["pv"][None])[0]                           # the W muon's point of closest approach (truth-hit fit)
    ax.plot(*pv, ls="none", marker="o", ms=12, mfc="none", mec=INK, mew=2.0, zorder=5)
    o = proj(np.zeros((1, 3)))[0]
    ax.plot(*o, ls="none", marker="x", ms=13, mew=2.8, color=INK, zorder=2.6)


def event_display(ev: dict, signed_r: bool = True) -> list[str]:
    """event_display (x-y) and event_display_rz. signed_r: r < 0 for hits on the far side of the beam line from the
    HNL (the sign of the hit's transverse position projected on the HNL flight direction), so that the W muon, which
    recoils against the HNL, does not overlap the daughters in the r-z view."""
    extra = (SAMPLE_TEXT[ev["sample"]] + ": one validation event\n"
             + r"$pp \to W \to \mu N,\ N \to \nu_\mu \mu^+ \mu^-$")
    legend = dict(handles=_legend_handles(), loc="center left", fontsize=16, frameon=False, handlelength=2.2,
                  labelspacing=1.0)
    files = []
    # transverse view, equal aspect; an empty band on top for the label
    rmax = float(np.hypot(ev["hits"][:, 0], ev["hits"][:, 1]).max())
    lim = 50.0 * np.ceil(rmax / 50.0) + 60.0
    fig, ax = style.new(figsize=(11.5, 8.5))
    _draw(ax, ev, lambda a: a[:, :2])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, 1.32 * lim)
    ax.set_aspect("equal")
    style.axis_labels(ax, "x [mm]", "y [mm]")
    style.label(ax, extra=extra)
    ax.legend(bbox_to_anchor=(1.02, 0.5), **legend)
    files += save(fig, "event_display")
    # r-z view
    fig, ax = style.new(figsize=(15, 7))
    if signed_r:
        u = ev["vertex"][:2] / np.hypot(*ev["vertex"][:2])          # + = the HNL's side of the beam line
        proj = lambda a: np.c_[a[:, 2], np.hypot(a[:, 0], a[:, 1]) * np.where(a[:, :2] @ u >= 0, 1.0, -1.0)]
    else:
        proj = lambda a: np.c_[a[:, 2], np.hypot(a[:, 0], a[:, 1])]
    _draw(ax, ev, proj)
    zmax = 100.0 * np.ceil(np.abs(ev["hits"][:, 2]).max() / 100.0) + 100.0
    ax.set_xlim(-zmax, zmax)
    if signed_r:
        ax.set_ylim(-lim, 1.50 * lim)
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        style.axis_labels(ax, "z [mm]", "signed r [mm]")
        style.label(ax, extra=extra + "\nr > 0: hits on the HNL side of the beam line")
    else:
        ax.set_ylim(-0.03 * lim, 1.40 * lim)
        style.axis_labels(ax, "z [mm]", "r [mm]")
        style.label(ax, extra=extra)
    ax.legend(bbox_to_anchor=(1.01, 0.5), **legend)
    files += save(fig, "event_display_rz")
    return files


# --------------------------------------------------------------------------------------------- edge TPR vs |d0|

def _parse_bin(s: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in re.fullmatch(r"\[\s*([^,]+),\s*([^)]+)\)", s.strip()).groups())
    return lo, hi


def tpr_table(run: str) -> tuple[pd.DataFrame, dict]:
    """Per |d0| bin of the HNL-daughter targets: in-graph chain-true edges n, kept k, TPR and its 68 % interval."""
    meta = json.load(open(os.path.join(run, "edge_tpr_estar_efpr.json")))
    c = pd.read_csv(os.path.join(run, "edge_tpr_estar.csv"))
    col = f"tpr_in_graph@{meta['cut']:g}"
    if col not in c.columns:
        raise ValueError(f"{run}: no column {col} (columns {list(c.columns)})")
    c = c[(c.population == "hnl_daughters") & (c.variable == "d0_bin")].copy()
    c[["lo", "hi"]] = [list(_parse_bin(b)) for b in c["bin"]]
    c = c.sort_values("lo").reset_index(drop=True)
    n_f = c.n_true_edges * c.graph_eff
    n = np.rint(n_f).astype(np.int64)
    k_f = c[col] * n
    k = np.rint(k_f).astype(np.int64)
    if np.abs(n_f - n).max() > 1e-6 or np.abs(k_f - k).max() > 1e-6:
        raise ValueError(f"{run}: edge counts are not recoverable from the CSV")
    lo = np.where(k > 0, beta.ppf(0.16, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(0.84, k + 1, n - k), 1.0)
    t = pd.DataFrame({"d0_lo": c.lo, "d0_hi": c.hi, "n_true_edges": c.n_true_edges, "n_in_graph": n, "n_kept": k,
                      "tpr": k / n, "err_lo": k / n - lo, "err_hi": hi - k / n})
    return t, meta


def delta_d_points(t: pd.DataFrame) -> float:
    sel = lambda r: t[(t.d0_lo == r[0]) & (t.d0_hi == r[1])]
    a, b = sel(DD_HIGH), sel(DD_LOW)
    return 100.0 * float(a.n_kept.sum() / a.n_in_graph.sum() - b.n_kept.sum() / b.n_in_graph.sum())


def edge_tpr_vs_d0() -> list[str]:
    tabs, metas = [], []
    for lab, run, *_rest, dd_fact in TPR_SERIES:
        t, m = tpr_table(run)
        dd = delta_d_points(t)
        if f"{dd:+.1f}" != f"{dd_fact:+.1f}" or f"{100 * m['dD']:+.1f}" != f"{dd_fact:+.1f}":
            raise ValueError(f"{lab}: dD {dd:+.2f} (CSV) / {100 * m['dD']:+.2f} (json) differs from the facts "
                             f"sheet {dd_fact:+.1f}")
        tabs.append(t.assign(label=lab, dD_points=dd))
        metas.append(m)
    if len({m["target_efpr"] for m in metas}) != 1:
        raise ValueError("the classifiers are not cut at the same target edge FPR")
    if len({tuple(t.d0_lo) for t in tabs}) != 1:
        raise ValueError("different |d0| bins")
    os.makedirs(os.path.join(style.FIG_DIR, SUBDIR), exist_ok=True)
    pd.concat(tabs, ignore_index=True).to_csv(os.path.join(style.FIG_DIR, SUBDIR, "edge_tpr_vs_d0.csv"), index=False)

    nb = len(tabs[0])
    names = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else rf"$\geq${lo:g}" for lo, hi in zip(tabs[0].d0_lo, tabs[0].d0_hi)]
    fig, ax = style.new(figsize=(11, 8))
    i_low = int(np.flatnonzero((tabs[0].d0_lo == DD_LOW[0]).to_numpy())[0])
    i_high = int(np.flatnonzero((tabs[0].d0_lo == DD_HIGH[0]).to_numpy())[0])
    for i in (i_low, i_high):                                       # the two bins of dD
        ax.axvspan(i - 0.5, i + 0.5, color="0.94", lw=0, zorder=0)
    lows = []
    for j, ((lab, _run, colr, mk, ls, dd_fact), t) in enumerate(zip(TPR_SERIES, tabs)):
        off = 0.12 * (2 * j - 1)
        x = np.arange(nb) + off
        ax.errorbar(x, t.tpr, yerr=[t.err_lo, t.err_hi], color=colr, marker=mk, ls=ls, lw=1.8, ms=9, capsize=3,
                    label=lab, zorder=3)
        lows.append(float((t.tpr - t.err_lo).min()))
        # dD: the low-|d0| reference level carried to the 100-300 mm bin, an arrow from it to the value there and the
        # facts-sheet value beside the arrow head (below a falling arrow, above a rising one)
        y0, y1 = float(t.tpr[i_low]), float(t.tpr[i_high])
        xa = i_high + off - 0.30                                    # left of the point: clear of the line to the next bin
        ax.plot([i_low + off, xa], [y0, y0], color=colr, lw=1.1, ls=":", zorder=2)
        ax.annotate("", xy=(xa, y1), xytext=(xa, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=2.2, mutation_scale=20, shrinkA=0, shrinkB=0))
        txt = rf"$\Delta D$ = {dd_fact:+.1f} points".replace("-", "\u2212")
        if y1 < y0:
            ax.text(xa + 0.05, y1 - 0.004, txt, ha="right", va="top", fontsize=16, zorder=5)
        else:
            ax.text(xa, y1 + float(t.err_hi[i_high]) + 0.006, txt, ha="center", va="bottom", fontsize=16, zorder=5)
    ax.set_xticks(np.arange(nb))
    ax.set_xticklabels(names, fontsize=15)
    ax.set_xlim(-0.6, nb - 0.4)
    ymin = max(0.0, np.floor((min(lows) - 0.02) * 20) / 20)
    ax.set_ylim(ymin, 1.0 + 0.42 * (1.0 - ymin))                   # headroom for the label
    ax.set_yticks(np.round(np.arange(ymin, 1.0001, 0.05), 2))      # no tick label above a rate of 1
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    style.axis_labels(ax, r"Truth $|d_0|$ of the HNL daughter [mm]", "Edge TPR, HNL-daughter targets")
    style.label(ax, extra="HNL val; both trained on ttbar only, same screening budget\n"
                          "in-graph true edges; cut at the same target edge false-positive rate\n"
                          "bars: 68 % binomial on edge counts (edges of one track are correlated)\n"
                          r"$\Delta D$ = TPR($|d_0|$ 100-300 mm) $-$ TPR($|d_0|$ 0.1-1 mm)")
    ax.legend(loc="lower left", fontsize=16, frameon=False)
    return save(fig, "edge_tpr_vs_d0")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="event_display,edge_tpr")
    ap.add_argument("--sample", default="hnl15_200", choices=sorted(SAMPLE_TEXT))
    ap.add_argument("--event", type=int, default=None, help="event index (default: the selection rule)")
    ap.add_argument("--unsigned-r", action="store_true", help="r-z view with r >= 0 (default: signed by the HNL side)")
    a = ap.parse_args(argv)
    files = []
    which = set(a.which.split(","))
    if "event_display" in which:
        c = candidates(a.sample)
        pd.set_option("display.width", 220)
        print(c.head(12).round(3).to_string())
        event = int(c.index[0]) if a.event is None else a.event
        if a.event is None and not c.passes.iloc[0]:
            raise SystemExit("no event passes the selection rule")
        ev = load_event(a.sample, event)
        files += event_display(ev, signed_r=not a.unsigned_r)
        info = {k: ev[k] for k in ("sample", "event", "event_id", "rv", "w_id", "daughter_ids", "particles")}
        info.update(vertex=ev["vertex"].tolist(), primary_vertex=ev["pv"].tolist(), chosen_by_rule=a.event is None,
                    signed_r=not a.unsigned_r)
        with open(os.path.join(style.FIG_DIR, SUBDIR, "event_display.json"), "w") as f:
            json.dump(info, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(json.dumps({k: info[k] for k in ("sample", "event", "event_id", "rv", "vertex", "primary_vertex")}))
    if "edge_tpr" in which:
        files += edge_tpr_vs_d0()
    print("\n".join(files))


if __name__ == "__main__":
    main()
