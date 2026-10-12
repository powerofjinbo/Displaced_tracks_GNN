"""M2 figures and tables: graph-stage ceilings on val (brief §5e).

Inputs: the physics-window study (derived/graph_ceiling/windows_v1) and, when available, the baseline runs'
ceilings on their own metric-learning val graphs (runs/baseline_v1/arm*_s*/ceiling_ml, else eval_val). Populations as in the preregistration: HNL-daughter targets on the pooled HNL val splits (all,
and the primary population |d0| >= 10 mm or r_v >= 100 mm), W-muon targets (HNL val), ttbar targets (ttbar val).

    python -m hnl.plots.m2 [--windows DIR] [--runs RUN_ROOT] [--out TABLE_DIR]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from hnl.plots import style
from hnl.truth.report import D0_BINS, RV_BINS

DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
HNL_SAMPLES = ("hnl10_100", "hnl15_200")


def pops(p: pd.DataFrame) -> dict[str, pd.Series]:
    hnl_ev = p["sample"].isin(HNL_SAMPLES)
    t = p.is_target.astype(bool)
    disp = (p.d0.abs() >= 10) | (p.rv >= 100)
    return {"hnl": hnl_ev & t & (p.category == "hnl"), "primary": hnl_ev & t & (p.category == "hnl") & disp,
            "w": hnl_ev & t & (p.category == "w"), "ttbar": (p["sample"] == "ttbar") & t}


def window_tables(wdir: str):
    pe = pd.read_parquet(os.path.join(wdir, "per_event.parquet"))
    pp = pd.read_parquet(os.path.join(wdir, "per_particle.parquet"))
    size = pe.assign(hnl=pe["sample"].isin(HNL_SAMPLES)).groupby(["config", "hnl"]).agg(
        pairs=("n_graph", "mean"), n_true=("n_true", "sum"), n_kept=("n_true_in_graph", "sum")).reset_index()
    size["edge_eff"] = size.n_kept / size.n_true
    rows = []
    for cfg, g in pp.groupby("config"):
        P = pops(g)
        r = {"source": f"window {cfg}"}
        for k, m in P.items():
            r[f"ceil_{k}"] = g.dm_matched[m].mean()
            r[f"ceil_samepid_{k}"] = g.dm_matched_samepid[m].mean()
            r[f"n_{k}"] = int(m.sum())
        s = size[(size.config == cfg)].set_index("hnl")
        r["pairs_hnl"], r["pairs_ttbar"] = s.loc[True, "pairs"], s.loc[False, "pairs"]
        r["edge_eff_hnl"], r["edge_eff_ttbar"] = s.loc[True, "edge_eff"], s.loc[False, "edge_eff"]
        rows.append(r)
    return pd.DataFrame(rows), pp


def ml_tables(runs: str):
    from hnl.eval.particles import particle_table, split_event_id
    rows, parts = [], []
    info = None
    for run in sorted(glob.glob(os.path.join(runs, "arm*_s*"))):
        d = next((os.path.join(run, x) for x in ("ceiling_ml", "eval_val")
                  if os.path.exists(os.path.join(run, x, "ceiling_particle.parquet"))), None)
        if d is None:
            continue
        ce = pd.read_parquet(os.path.join(d, "ceiling_particle.parquet"))
        pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
        pe = pe[pe.cut.isna()] if pe.cut.isna().any() else pe[pe.cut == pe.cut.min()]
        if info is None:
            info = particle_table(ce.event_id.unique())
        g = ce.merge(info, on=["event_id", "particle_id"], how="inner").rename(
            columns={"dm_ceiling": "dm_matched", "dm_ceiling_samepid": "dm_matched_samepid"})
        name = os.path.basename(run)
        P = pops(g)
        r = {"source": f"metric learning {name}"}
        for k, m in P.items():
            r[f"ceil_{k}"] = g.dm_matched[m].mean()
            r[f"ceil_samepid_{k}"] = g.dm_matched_samepid[m].mean()
            r[f"n_{k}"] = int(m.sum())
        code = split_event_id(pe.event_id.to_numpy())[0]
        for lab, m in (("hnl", np.isin(code, (2, 4))), ("ttbar", code == 1)):
            r[f"pairs_{lab}"] = pe[m].n_edges.mean()
            r[f"edge_eff_{lab}"] = pe[m].n_true_in_graph.sum() / pe[m].n_true.sum()
            if "n_true_mu" in pe:       # muon-target edges, and the clean variant without outlier-touching edges
                r[f"edge_eff_mu_{lab}"] = pe[m].n_true_in_graph_mu.sum() / pe[m].n_true_mu.sum()
                r[f"edge_eff_mu_clean_{lab}"] = pe[m].n_true_in_graph_mu_clean.sum() / pe[m].n_true_mu_clean.sum()
        rows.append(r)
        g["config"] = name
        parts.append(g)
    return pd.DataFrame(rows), (pd.concat(parts, ignore_index=True) if parts else None)


def _bin_labels(edges, var: str) -> list[str]:
    labels = [f"[{lo:g},{hi:g})" if np.isfinite(hi) else f">={lo:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    return labels + (["no vertex"] if var == "rv" else [])


def binned(pp: pd.DataFrame, var: str, edges) -> pd.DataFrame:
    """Ceiling per bin; for r_v also the preregistered "no vertex" category (daughters without a pair vertex)."""
    h = pp[pops(pp)["hnl"]]
    x = h.d0.abs() if var == "d0" else h.rv
    b = pd.cut(x, edges, right=False)
    lab = pd.Series(pd.Categorical.from_codes(b.cat.codes.where(b.cat.codes >= 0, len(edges) - 1),
                                              _bin_labels(edges, var) if var == "rv" else
                                              _bin_labels(edges, var) + ["nan"]), index=h.index)
    g = h.groupby([h.config, lab], observed=True).dm_matched.agg(["mean", "size"])
    out = g.reset_index()
    out.columns = ["config", "bin", "mean", "size"]
    return out[out.bin != "nan"]


def plot_bins(tab: pd.DataFrame, var: str, edges, name: str, configs: list[str]):
    fig, ax = style.new()
    labels = _bin_labels(edges, var)
    centers = np.arange(len(labels))
    for cfg in configs:
        t = tab[tab.config == cfg].set_index("bin")
        y = np.full(len(centers), np.nan)
        e = np.full(len(centers), np.nan)
        for i, lab in enumerate(labels):
            if lab in t.index:
                m, n = float(t.loc[lab, "mean"]), int(t.loc[lab, "size"])
                y[i], e[i] = m, np.sqrt(max(m * (1 - m), 1e-12) / n)
        ax.errorbar(centers, y, yerr=e, marker="o", label=cfg, capsize=2)
    ax.set_xticks(centers, labels, rotation=45, ha="right", fontsize=11)
    style.axis_labels(ax, {"d0": "truth |d0| [mm]", "rv": "truth r_v [mm]"}[var],
                      "graph ceiling: DM efficiency, HNL daughters")
    ax.set_ylim(0.0, 1.18)
    ax.legend(fontsize=9, loc="lower left")
    style.label(ax, "val, graph-stage ceiling")
    return style.save(fig, name, "m2")


def ml_scan_points(runs: str) -> pd.DataFrame:
    """Ceilings of the metric-learning graphs at the scan radii (each arm's own val mix; exact radius graph, no knn
    cap) from runs/*/ml_scan: HNL-daughter and W-muon targets vs pairs per event."""
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, "arm*_s0", "ml_scan"))):
        arm = os.path.basename(os.path.dirname(d))
        try:
            cat = pd.read_csv(os.path.join(d, "ceiling_by_category.csv"))
            curve = pd.read_csv(os.path.join(d, "radius_curve.csv"))
        except FileNotFoundError:
            continue
        for r, g in cat.groupby("r"):
            # pairs per event at r (log-interpolated between scan grid points when r is not on the grid)
            pairs = float(np.exp(np.interp(r, curve.r, np.log(curve.pairs_per_event))))
            row = {"source": f"ML {arm} r={r:g}", "r": r, "pairs": pairs}
            for c in ("hnl", "w"):
                m = g[g.category == c]
                row[f"ceil_{c}"] = float(m.eff.iloc[0]) if len(m) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_size(summary: pd.DataFrame, name: str, scan: pd.DataFrame | None = None):
    fig, ax = style.new(figsize=(9, 6.5))
    win = summary[summary.source.str.startswith("window")]
    ml = summary[summary.source.str.startswith("metric learning")]
    ax.scatter(win.pairs_hnl, win.ceil_primary, marker="o", color="tab:blue",
               label="windows: HNL daughters, |d0| >= 10 mm or r_v >= 100 mm")
    ax.scatter(win.pairs_hnl, win.ceil_w, marker="s", color="tab:orange", label="windows: W muons")
    for _, r in win.iterrows():
        ax.annotate(r.source.replace("window ", ""), (r.pairs_hnl, r.ceil_primary), fontsize=7, xytext=(3, -8),
                    textcoords="offset points")
    if len(ml):
        ax.scatter(ml.pairs_hnl, ml.ceil_primary, marker="*", s=160, color="tab:red",
                   label="metric learning at (r*, knn*): primary population")
        ax.scatter(ml.pairs_hnl, ml.ceil_w, marker="*", s=160, color="tab:brown", label="metric learning: W muons")
        for _, r in ml.drop_duplicates(subset=["source"]).iterrows():
            if r.source.endswith("_s0"):
                ax.annotate(r.source.replace("metric learning ", ""), (r.pairs_hnl, r.ceil_primary), fontsize=8,
                            xytext=(-60, -14), textcoords="offset points")
    if scan is not None and len(scan):
        ok = scan.dropna(subset=["ceil_hnl"])
        ok = ok[ok.ceil_hnl > 0.5]
        ax.plot(ok.pairs, ok.ceil_hnl, marker="^", ls=":", color="tab:green",
                label="metric-learning radius scan (all HNL daughters)")
    ax.set_xscale("log")
    style.axis_labels(ax, "graph size [unique hit pairs per HNL event]", "graph ceiling: DM efficiency")
    lo = min(summary.ceil_primary.min(), summary.ceil_w.min())
    ax.set_ylim(lo - 0.02, 1.0 + 0.3 * (1.0 - lo))
    ax.legend(fontsize=9, loc="lower right")
    style.label(ax, "val, graph-stage ceiling (perfect classifier + CC + DM)")
    return style.save(fig, name, "m2")


def layer_pair_figures(lp_dir: str) -> list[str]:
    """Heatmaps of true-edge efficiency per layer pair (HNL-daughter targets, pooled HNL val) for every source."""
    import matplotlib.pyplot as plt
    figs = []
    files = sorted(glob.glob(os.path.join(lp_dir, "layer_pairs_*.parquet")))
    if not files:
        return figs
    lp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sel = lp[lp.targets_only & lp["sample"].isin(HNL_SAMPLES) & (lp.category == "hnl")]
    rows = []
    for src, d in sel.groupby("source"):
        g = d.groupby(["lay_lo", "lay_hi"])[["n_true", "n_kept"]].sum()
        g = g[g.n_true >= 20]
        g["eff"] = g.n_kept / g.n_true
        layers = sorted(set(g.index.get_level_values(0)) | set(g.index.get_level_values(1)))
        pos = {l: i for i, l in enumerate(layers)}
        M = np.full((len(layers), len(layers)), np.nan)
        for (lo, hi), r in g.iterrows():
            M[pos[hi], pos[lo]] = r.eff
        fig, ax = style.new(figsize=(10, 8))
        im = ax.imshow(M, origin="lower", vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(layers)), [str(l) for l in layers], rotation=90, fontsize=7)
        ax.set_yticks(range(len(layers)), [str(l) for l in layers], fontsize=7)
        style.axis_labels(ax, "inner layer (volume*100 + layer)", "outer layer")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("true-edge efficiency, HNL-daughter targets", fontsize=16)
        style.label(ax, f"val, {src}")
        name = "layer_pairs_" + "".join(ch if ch.isalnum() else "_" for ch in src)
        figs += style.save(fig, name, "m2")
        worst = g.sort_values("eff").head(10).reset_index()
        worst["source"] = src
        rows.append(worst)
    if rows:
        pd.concat(rows).to_csv(os.path.join(lp_dir, "worst_layer_pairs_hnl.csv"), index=False)
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=os.path.join(DERIVED, "graph_ceiling", "windows_v1"))
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(DERIVED), "runs", "baseline_v1"))
    ap.add_argument("--out", default=os.path.join(DERIVED, "graph_ceiling", "m2_v1"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ws, wpp = window_tables(a.windows)
    ms, mpp = ml_tables(a.runs) if os.path.isdir(a.runs) else (pd.DataFrame(), None)
    summary = pd.concat([ws, ms], ignore_index=True)
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    pp = pd.concat([wpp, mpp], ignore_index=True) if mpp is not None else wpp
    d0 = binned(pp, "d0", D0_BINS)
    rv = binned(pp, "rv", RV_BINS)
    d0.astype({"bin": str}).to_csv(os.path.join(a.out, "hnl_ceiling_by_d0.csv"), index=False)
    rv.astype({"bin": str}).to_csv(os.path.join(a.out, "hnl_ceiling_by_rv.csv"), index=False)
    configs = sorted(pp.config.unique())
    sel = [c for c in configs if c.startswith("arm") and c.endswith("_s0")] + \
        [c for c in configs if c in ("d0=0,D=prompt,s=1", "d0=10,D=inf,s=1", "d0=100,D=inf,s=1", "d0=300,D=inf,s=1")]
    figs = plot_bins(d0, "d0", D0_BINS, "ceiling_hnl_vs_d0", sel or configs[:6])
    figs += plot_bins(rv, "rv", RV_BINS, "ceiling_hnl_vs_rv", sel or configs[:6])
    scan = ml_scan_points(a.runs) if os.path.isdir(a.runs) else None
    if scan is not None and len(scan):
        scan.to_csv(os.path.join(a.out, "ml_scan_points.csv"), index=False)
    figs += plot_size(summary, "ceiling_vs_graph_size", scan)
    figs += layer_pair_figures(os.path.join(DERIVED, "graph_ceiling", "layer_pairs_v1"))
    pd.set_option("display.width", 250)
    cols = ["source", "pairs_hnl", "pairs_ttbar", "edge_eff_hnl", "edge_eff_mu_hnl", "edge_eff_mu_clean_hnl",
            "ceil_hnl", "ceil_primary", "ceil_w", "ceil_ttbar", "ceil_samepid_primary"]
    print(summary[[c for c in cols if c in summary]].round(4).to_string())
    print("\n".join(figs))


if __name__ == "__main__":
    main()
