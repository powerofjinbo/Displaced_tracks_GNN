"""Truth statistics and validation from the §5b fits (brief §2.5, §5b): per-sample target counts in |d0| and r_v
bins, W-muon d0 validation, vertex-fit failures, the |d0| <= r_v constraint, and figures.

    python -m hnl.truth.report --fits <derived/truth_fits/v1> --out <dir> [--figdir docs/figures/truth]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.data import splits as S

D0_BINS = [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]
RV_BINS = [0, 10, 30, 100, 200, 300, np.inf]
HNL_SAMPLES = ("hnl10_100", "hnl15_200", "hnl15_100")
LABEL = "ACTS generic detector simulation, no pileup"


def truth_d0(df: pd.DataFrame) -> pd.Series:
    """Definition (docs/definitions.md §3): muons fixed-R, others free curvature."""
    return pd.Series(np.where(df.pdg.abs() == 13, df.fix_d0, df.free_d0), index=df.index)


def load(fits: str, sample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(os.path.join(fits, f"particles_{sample}.parquet"))
    vpath = os.path.join(fits, f"vertices_{sample}.parquet")
    v = pd.read_parquet(vpath) if os.path.exists(vpath) else pd.DataFrame()
    p["d0"] = truth_d0(p)
    p["is_target"] = (p.pt > 1.0) & (p.eta.abs() < 4.0) & (p.n_hits >= 5)
    lab = "label" if sample != "hnl15_100" else "rule"
    p["is_hnl"] = p[f"{lab}_hnl"]
    p["is_w"] = p[f"{lab}_w"]
    if len(v):
        rv = pd.concat([v[["event", "pid_a", "rv", "ok"]].rename(columns={"pid_a": "particle_id"}),
                        v[["event", "pid_b", "rv", "ok"]].rename(columns={"pid_b": "particle_id"})])
        p = p.merge(rv, on=["event", "particle_id"], how="left")
    else:
        p["rv"], p["ok"] = np.nan, np.nan
    p["rv_cat"] = np.where(p.ok.fillna(False).astype(bool), p.rv, np.nan)
    return p, v


def split_of(sample: str, events: np.ndarray) -> np.ndarray:
    out = np.full(len(events), "", dtype=object)
    names = ("external",) if sample == "hnl15_100" else ("train", "val", "test")
    for n in names:
        out[np.isin(events, S.read_split(sample, n))] = n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figdir", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rep: dict = {}
    tables = []
    for s in ("ttbar",) + HNL_SAMPLES:
        p, v = load(a.fits, s)
        n_ev = p.event.nunique()
        p["split"] = split_of(s, p.event.to_numpy())
        w = p[p.is_w & (p.n_hits >= 5)]
        r = {"events": int(n_ev),
             "w_muon_5hits_per_event": round(len(w) / n_ev, 4),
             "w_muon_d0_abs_median_mm": float(w.d0.abs().median()) if len(w) else None,
             "w_muon_d0_abs_p90_mm": float(w.d0.abs().quantile(0.9)) if len(w) else None,
             "w_muon_z0_robust_sigma_mm": float(1.4826 * np.median(np.abs(w.fix_z0 - w.fix_z0.median()))) if len(w) else None}
        if s != "ttbar":
            h = p[p.is_hnl]
            h5 = h[h.n_hits >= 5]
            ht = h[h.is_target]
            r.update({
                "hnl_daughters_with_3hits_per_event": round(len(h) / n_ev, 4),
                "hnl_daughters_5hits_per_event": round(len(h5) / n_ev, 4),
                "hnl_targets_per_event": round(len(ht) / n_ev, 4),
                "hnl_5hits_frac_ge3_pixel": round(float((h5.n_pixel_hits >= 3).mean()), 4),
                "hnl_5hits_median_first_hit_radius_mm": round(float(h5.r_first.median()), 1),
                "hnl_5hits_median_pt": round(float(h5.pt.median()), 2),
                "hnl_target_d0_abs_quantiles_mm": {str(q): round(float(ht.d0.abs().quantile(q)), 2) for q in (0.1, 0.5, 0.9)},
                "hnl_target_counts_d0_bins": pd.cut(ht.d0.abs(), D0_BINS, right=False).value_counts(sort=False)
                .rename(lambda iv: f"[{iv.left:g},{iv.right:g})").to_dict(),
                "hnl_target_counts_rv_bins": pd.cut(ht.rv_cat, RV_BINS, right=False).value_counts(sort=False)
                .rename(lambda iv: f"[{iv.left:g},{iv.right:g})").to_dict(),
                "hnl_target_no_vertex": int(ht.rv_cat.isna().sum()),
                "vertex_pairs": int(len(v)),
                "vertex_fail_frac": round(float(1 - v.ok.mean()), 4) if len(v) else None,
                "vertex_dz_median_mm": round(float(v.dz.median()), 3) if len(v) else None,
                "d0_gt_rv_frac_targets_with_vertex": round(float((ht.d0.abs() > ht.rv_cat + 1.0)[ht.rv_cat.notna()].mean()), 4),
            })
            for name in ("train", "val", "test", "external"):
                hs = ht[ht.split == name]
                if len(hs):
                    r[f"hnl_targets_{name}"] = int(len(hs))
                    r[f"hnl_targets_{name}_d0_ge30"] = int((hs.d0.abs() >= 30).sum())
                    r[f"hnl_targets_{name}_d0_ge100"] = int((hs.d0.abs() >= 100).sum())
            tables.append(ht.assign(sample=s))
        else:
            t = p[p.is_target]
            r["ttbar_targets_per_event"] = round(len(t) / n_ev, 2)
            r["ttbar_target_counts_d0_bins"] = pd.cut(t.d0.abs(), D0_BINS, right=False).value_counts(sort=False) \
                .rename(lambda iv: f"[{iv.left:g},{iv.right:g})").to_dict()
        rep[s] = r
    allh = pd.concat(tables)
    rep["all_hnl"] = {
        "targets": int(len(allh)),
        "targets_d0_30_100": int(((allh.d0.abs() >= 30) & (allh.d0.abs() < 100)).sum()),
        "targets_d0_ge100": int((allh.d0.abs() >= 100).sum()),
        "targets_rv_ge100": int((allh.rv_cat >= 100).sum()),
    }
    with open(os.path.join(a.out, "truth_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print(json.dumps(rep, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    if a.figdir:
        figures(a.fits, a.figdir)


def figures(fits: str, figdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.ROOT)
    os.makedirs(figdir, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    bins = np.logspace(-4, 3.5, 76)
    fig, ax = plt.subplots(figsize=(8, 6))
    for s, ls in (("hnl10_100", "-"), ("hnl15_200", "--"), ("hnl15_100", ":")):
        p, _ = load(fits, s)
        ax.hist(p[p.is_hnl & p.is_target].d0.abs(), bins=bins, histtype="step", ls=ls, lw=1.5,
                label=f"HNL daughters, {s}")
        if s == "hnl10_100":
            ax.hist(p[p.is_w & p.is_target].d0.abs(), bins=bins, histtype="step", lw=1.5, color="k",
                    label="W muons, hnl10_100")
    t, _ = load(fits, "ttbar")
    ax.hist(t[t.is_target].d0.abs(), bins=bins, histtype="step", lw=1.0, color="grey", label="ttbar targets")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|d0| from truth-hit fit [mm]")
    ax.set_ylabel("Targets / bin")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title(LABEL, fontsize=12, loc="right")
    save(fig, "truth_d0_targets")

    fig, ax = plt.subplots(figsize=(8, 6))
    for s, ls in (("hnl10_100", "-"), ("hnl15_200", "--"), ("hnl15_100", ":")):
        p, _ = load(fits, s)
        ax.hist(p[p.is_hnl & p.is_target].rv_cat.dropna(), bins=np.linspace(0, 1000, 51), histtype="step", ls=ls,
                lw=1.5, label=s)
    ax.set_xlabel("r_v of the HNL-daughter pair (truth-hit vertex) [mm]")
    ax.set_ylabel("Target daughters / 20 mm")
    ax.legend(fontsize=11)
    ax.set_title(LABEL, fontsize=12, loc="right")
    save(fig, "truth_rv_targets")

    fig, ax = plt.subplots(figsize=(7, 6))
    p, _ = load(fits, "hnl10_100")
    h = p[p.is_hnl & p.is_target & p.rv_cat.notna()]
    ax.scatter(h.rv_cat, h.d0.abs(), s=2, alpha=0.3)
    x = np.linspace(0, 1000, 10)
    ax.plot(x, x, "r--", lw=1, label="|d0| = r_v")
    ax.set_xlabel("r_v [mm]")
    ax.set_ylabel("|d0| [mm]")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.legend(fontsize=11)
    ax.set_title(LABEL + ", hnl10_100", fontsize=11, loc="right")
    save(fig, "truth_d0_vs_rv")


if __name__ == "__main__":
    main()
