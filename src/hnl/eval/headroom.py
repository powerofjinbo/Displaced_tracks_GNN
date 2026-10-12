"""Headroom report (brief §5g), val only: where the acorn baselines lose targets, and why.

For every baseline run at its working cut (per seed, the cut at FR*; docs/preregistration.md §4), per particle:
  ceiling  = DM after CC on every true edge of the run's metric-learning graph (perfect classifier),
  ta       = DM after CC on the true edges kept at the cut (truth-assisted builder on the same kept edges),
  actual   = DM after CC on all kept edges.
Losses: graph = 1 - ceiling, classifier = ceiling - ta, track building = ta - actual (DECISIONS.md 2026-10-03).
Same-particle variant (*_sp): ceiling_sp = CC on every graph edge joining two hits of one particle, ta_sp = CC on the
kept edges of that kind; the chain-truth ta drops same-particle skip edges that CC needs, so ta can fall below the
actual efficiency (RESULTS.md, M4). Connectivity variant (*_conn, shown in M4): connected = some CC track of the kept
edges holds more than half of the particle's hits (an oracle splitting tracks by truth would match it), so
track building = connected - actual >= 0 (merges / impurity) and classifier = ceiling - connected (missed edges).
Binned in |d0|, r_v, first-hit radius, number of pixel hits, pT and the daughters' opening angle, for HNL-daughter
targets (pooled HNL val), the primary population, and W muons; daughter-pair merges vs opening angle and r_v.

    python -m hnl.eval.headroom --runs RUN_ROOT --wp WORKING_POINTS_JSON --out OUT
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import HNL_CODES, load_run, populations
from hnl.eval.particles import particle_table, split_event_id

BINS = {
    "d0": ("abs_d0", [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]),
    "rv": ("rv", [0, 10, 30, 100, 200, 300, np.inf]),
    "r_first": ("r_first", [0, 40, 80, 150, 250, 400, 700, np.inf]),
    "n_pixel_hits": ("n_pixel_hits", [0, 1, 2, 3, 4, 5, 6, 8, np.inf]),
    "pt": ("pt", [1, 2, 3, 5, 10, 20, 50, np.inf]),
    "pair_angle": ("pair_angle", [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.pi + 1e-9]),
}


def run_table(run: str, cut: float, info: pd.DataFrame) -> pd.DataFrame:
    """Per particle of the val events: ceiling, ta, actual at `cut`, joined with the truth table."""
    pp, pe = load_run(run)
    ce = pd.read_parquet(os.path.join(run, "eval_val", "ceiling_particle.parquet"))
    cols = ["event_id", "particle_id", "dm_matched", "dm_nofp"] + [c for c in ("dm_nocross", "connected") if c in pp]
    at = pp[np.isclose(pp.cut, cut, atol=1e-6)][cols]
    t = ce.merge(at, on=["event_id", "particle_id"], how="inner")
    t = t.merge(info, on=["event_id", "particle_id"], how="inner")
    return t.rename(columns={"dm_ceiling": "ceiling", "dm_nofp": "ta", "dm_matched": "actual",
                             "dm_ceiling_samepid": "ceiling_sp", "dm_nocross": "ta_sp"})


def binned_losses(t: pd.DataFrame, pop_mask: np.ndarray, var: str) -> pd.DataFrame:
    col, edges = BINS[var] if var in BINS else (var, [0.0, 1.0])
    x = t[col] if col in t else None
    if col == "abs_d0":
        x = t.d0.abs()
    sel = t[pop_mask]
    b = pd.cut(x[pop_mask], edges, right=False)
    lab = {"rv": "no vertex", "pair_angle": "no pair"}.get(var)   # NaN categories (no pair vertex / no partner)
    if lab:
        b = b.cat.add_categories([lab])
        b[x[pop_mask].isna()] = lab
    g = sel.groupby(b, observed=True)
    out = pd.DataFrame({"n": g.size(), "ceiling": g.ceiling.mean(), "ta": g.ta.mean(), "actual": g.actual.mean()})
    out["loss_graph"] = 1 - out.ceiling
    out["loss_classifier"] = out.ceiling - out.ta
    out["loss_trackbuilding"] = out.ta - out.actual
    if "ta_sp" in sel and "ceiling_sp" in sel:
        out["ceiling_sp"], out["ta_sp"] = g.ceiling_sp.mean(), g.ta_sp.mean()
        out["loss_graph_sp"] = 1 - out.ceiling_sp
        out["loss_classifier_sp"] = out.ceiling_sp - out.ta_sp
        out["loss_trackbuilding_sp"] = out.ta_sp - out.actual
    if "connected" in sel:
        # connectivity decomposition (the one shown in M4): an oracle that splits every CC track into its particles
        # matches exactly the connected particles, so track building (merges) costs connected - actual >= 0 and the
        # classifier (missing kept edges) costs ceiling - connected
        out["connected"] = g.connected.mean()
        out["loss_graph_conn"] = 1 - out.ceiling
        out["loss_classifier_conn"] = out.ceiling - out.connected
        out["loss_trackbuilding_conn"] = out.connected - out.actual
    out.index = out.index.astype(str)
    return out


def merges(pe: pd.DataFrame, cut: float, info: pd.DataFrame) -> pd.DataFrame:
    """Pair efficiency and merge rate per event at the cut, with the pair's opening angle and r_v."""
    e = pe[np.isclose(pe.cut, cut, atol=1e-6) & np.isin(split_event_id(pe.event_id.to_numpy())[0], HNL_CODES)]
    e = e.dropna(subset=["pair_both"])
    h = info[info.category == "hnl"].groupby("event_id").agg(pair_angle=("pair_angle", "first"), rv=("rv", "first"))
    return e[["event_id", "pair_both", "pair_merged"]].merge(h, left_on="event_id", right_index=True, how="left")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--wp", required=True, help="working_points.json from hnl.eval.baseline_val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    wp = json.load(open(a.wp))
    tabs, mtabs = {}, {}
    info = None
    for arm, cuts in wp["cuts_at_FR_star"].items():
        for seed, cut in cuts.items():
            run = os.path.join(a.runs, f"arm{arm}_s{seed}")
            if info is None:
                ev = pd.read_parquet(os.path.join(run, "eval_val", "ceiling_particle.parquet"), columns=["event_id"])
                info = particle_table(ev.event_id.unique())
            tabs[(arm, int(seed))] = run_table(run, float(cut), info)
            mtabs[(arm, int(seed))] = merges(load_run(run)[1], float(cut), info)
    rows = []
    for (arm, seed), t in tabs.items():
        pops = populations(t)
        for pop in ("hnl_targets", "primary", "w_muons", "ttbar_prompt"):
            for var in BINS:
                if var in ("pair_angle", "rv") and pop not in ("hnl_targets", "primary"):
                    continue
                r = binned_losses(t, pops[pop], var)
                r["arm"], r["seed"], r["population"], r["variable"] = arm, seed, pop, var
                rows.append(r.reset_index(names="bin"))
            m = pops[pop]
            r = binned_losses(t.assign(_all=0.5), m, "_all")
            r["arm"], r["seed"], r["population"], r["variable"] = arm, seed, pop, "all"
            rows.append(r.reset_index(names="bin").assign(bin="all"))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(a.out, "losses_per_seed.csv"), index=False)
    num = [c for c in ["n", "ceiling", "ta", "actual", "loss_graph", "loss_classifier", "loss_trackbuilding",
                       "ceiling_sp", "ta_sp", "loss_graph_sp", "loss_classifier_sp", "loss_trackbuilding_sp",
                       "connected", "loss_graph_conn", "loss_classifier_conn", "loss_trackbuilding_conn"]
           if c in res]
    agg = res.groupby(["arm", "population", "variable", "bin"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" if s == "std" else c for c, s in agg.columns]
    agg.reset_index().to_csv(os.path.join(a.out, "losses.csv"), index=False)
    mm = []
    for (arm, seed), m in mtabs.items():
        for var, edges in (("pair_angle", BINS["pair_angle"][1]), ("rv", BINS["rv"][1])):
            b = pd.cut(m[var], edges, right=False)
            lab = {"rv": "no vertex", "pair_angle": "no pair"}.get(var)
            if lab:
                b = b.cat.add_categories([lab])
                b[m[var].isna()] = lab
            g = m.groupby(b, observed=True)
            d = pd.DataFrame({"n_pairs": g.size(), "pair_eff": g.pair_both.mean(), "merge_rate": g.pair_merged.mean()})
            d["arm"], d["seed"], d["variable"] = arm, seed, var
            mm.append(d.reset_index(names="bin").astype({"bin": str}))
        mm.append(pd.DataFrame([{"bin": "all", "n_pairs": len(m), "pair_eff": m.pair_both.mean(),
                                 "merge_rate": m.pair_merged.mean(), "arm": arm, "seed": seed, "variable": "all"}]))
    mres = pd.concat(mm, ignore_index=True)
    mres.to_csv(os.path.join(a.out, "pairs_per_seed.csv"), index=False)
    pd.set_option("display.width", 250)
    print(agg.loc[(slice(None), ["primary", "hnl_targets", "w_muons"], "all"), :].round(4).to_string())
    print(mres[mres.variable == "all"].round(4).to_string())


if __name__ == "__main__":
    main()
