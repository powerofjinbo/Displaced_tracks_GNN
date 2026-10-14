"""Expected primary population per event of candidate HNL points, from generator-level events only
(docs/hnl_points_spec.md section 4). No event of a candidate point is simulated before the spec is committed.

Primary population (docs/definitions.md, preregistration section 6): HNL-daughter targets (pT > 1 GeV, |eta| < 4,
>= 5 hits) with truth |d0| >= 10 mm or r_v >= 100 mm.

Method:
  1. From the full-simulation pilot of the existing points (hnl10_100, hnl15_200, our generator, closure of
     results/hnl_points_closure_v1.md): the probability that a daughter with pT > 1 GeV and |eta| < 4 has >= 5 hits,
     in bins of its true decay radius r_v and |eta| (the hit count is a property of the detector, not of the HNL mass
     or lifetime); and the fraction of targets that are in the primary population with fitted d0 and r_v among
     targets that are in it with the generator's d0 and r_v (the fitted r_v needs both daughters fitted).
  2. For each candidate point (generator level, hnl_census.py gen): expected primary daughters per event =
     sum over daughters with pT > 1 GeV, |eta| < 4 and true (|d0| >= 10 mm or r_v >= 100 mm) of the hit
     probability of their bin, times the fit factor of step 1.
  3. Check: the same estimate for the generator-level runs of the existing points against the primary population per
     event counted on the pilot and on the Zenodo splits.

    python scripts/acts/hnl_yield.py --pilot PREFIX[,PREFIX] --plan DIR --points p5_50,... --target 603 --out OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

RV_EDGES = np.array([0, 30, 60, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000, 1e9])
ETA_EDGES = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])


def pilot_tables(prefixes):
    D = []
    for pfx in prefixes:
        E = pd.read_parquet(pfx + "_events.parquet")
        S = pd.read_parquet(pfx + "_signal.parquet")
        d = S[S.role == "d"].merge(E[["event", "rv", "rv_true"]], on="event", how="left")
        d["sample"] = os.path.basename(pfx)
        d["n_events"] = len(E)
        D.append(d)
    return pd.concat(D, ignore_index=True)


def hit_map(d: pd.DataFrame):
    acc = d[(d.pt > 1) & (d.eta.abs() < 4)]
    i = np.clip(np.digitize(acc.rv_true, RV_EDGES) - 1, 0, len(RV_EDGES) - 2)
    j = np.clip(np.digitize(acc.eta.abs(), ETA_EDGES) - 1, 0, len(ETA_EDGES) - 2)
    num = np.zeros((len(RV_EDGES) - 1, len(ETA_EDGES) - 1))
    den = np.zeros_like(num)
    np.add.at(den, (i, j), 1)
    np.add.at(num, (i, j), (acc.n_hits >= 5).to_numpy(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = num / den
    # empty bins: the r_v-row average (they hold almost no daughters)
    for a in range(p.shape[0]):
        row = p[a]
        if np.isnan(row).any():
            fill = num[a].sum() / den[a].sum() if den[a].sum() else 0.0
            row[np.isnan(row)] = fill
    return p, den


def fit_factor(d: pd.DataFrame) -> float:
    t = d[(d.pt > 1) & (d.eta.abs() < 4) & (d.n_hits >= 5)]
    prim_fit = (t.d0.abs() >= 10) | (t.rv >= 100)
    prim_true = (t.d0_true.abs() >= 10) | (t.rv_true >= 100)
    return float(prim_fit.sum() / prim_true.sum())


def expected(S: pd.DataFrame, E: pd.DataFrame, p: np.ndarray, factor: float) -> dict:
    d = S[S.role == "d"].merge(E[["dir", "event", "rv_true"]], on=["dir", "event"], how="left")
    acc = d[(d.pt > 1) & (d.eta.abs() < 4)]
    prim = acc[(acc.d0_true.abs() >= 10) | (acc.rv_true >= 100)]
    i = np.clip(np.digitize(prim.rv_true, RV_EDGES) - 1, 0, len(RV_EDGES) - 2)
    j = np.clip(np.digitize(prim.eta.abs(), ETA_EDGES) - 1, 0, len(ETA_EDGES) - 2)
    w = p[i, j]
    i2 = np.clip(np.digitize(acc.rv_true, RV_EDGES) - 1, 0, len(RV_EDGES) - 2)
    j2 = np.clip(np.digitize(acc.eta.abs(), ETA_EDGES) - 1, 0, len(ETA_EDGES) - 2)
    return {"events": len(E), "targets_per_event": float(p[i2, j2].sum() / len(E)),
            "primary_per_event": float(w.sum() * factor / len(E)),
            "primary_r_v_ge_100_per_event": float(w[(prim.rv_true >= 100).to_numpy()].sum() * factor / len(E)),
            "primary_r_v_ge_300_per_event": float(w[(prim.rv_true >= 300).to_numpy()].sum() * factor / len(E)),
            "primary_d0_ge_100_per_event": float(w[(prim.d0_true.abs() >= 100).to_numpy()].sum() * factor / len(E))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="census prefixes of the full-simulation pilots, comma separated")
    ap.add_argument("--plan", required=True, help="directory of the generator-level runs p<m>_<ctau>/")
    ap.add_argument("--points", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = pilot_tables(a.pilot.split(","))
    p, den = hit_map(d)
    factor = fit_factor(d)
    # pilot truth: counted primary population per event (fitted d0 and r_v, as in the evaluation)
    counted = {}
    for s, g in d.groupby("sample"):
        t = g[(g.pt > 1) & (g.eta.abs() < 4) & (g.n_hits >= 5)]
        counted[s] = float((((t.d0.abs() >= 10) | (t.rv >= 100))).sum() / g.n_events.iloc[0])
    res = {"hit_map_rv_edges": RV_EDGES.tolist(), "hit_map_eta_edges": ETA_EDGES.tolist(),
           "hit_map": np.round(p, 4).tolist(), "hit_map_counts": den.astype(int).tolist(), "fit_factor": factor,
           "pilot_counted_primary_per_event": counted, "target": a.target, "cap": a.cap, "points": {}}
    for pt in a.points.split(","):
        files = sorted(glob.glob(os.path.join(a.plan, pt, "gen_*_events.parquet")))
        E = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        S = pd.concat([pd.read_parquet(f.replace("_events", "_signal")) for f in files], ignore_index=True)
        r = expected(S, E, p, factor)
        r["events_needed"] = int(np.ceil(a.target / r["primary_per_event"]))
        r["primary_at_cap"] = r["primary_per_event"] * a.cap
        res["points"][pt] = r
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("hit_map", "hit_map_counts")}, indent=1))


if __name__ == "__main__":
    main()
