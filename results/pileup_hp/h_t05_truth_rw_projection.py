"""Addendum H, step H-T0.5.1 (truth only): the P_P yield of the H-val overlay and the reweighted val projection of
section 6, repeated with the H-val gun truth after the emulation (docs/preregistration_pileup.md sections 6 and 10).

Gun side: the stored H-val overlay (derived/overlay_v1, set hval; particles and hits as the writer stored them, after
Fatras, emulation and digitisation). Gun targets, P_P, P_D, the soft-sibling flag and the regions of the forward
fallback come from hnl.overlay.h1.gun_targets, the H1 code's own truth selection (vertex field 1001-1010; P_P =
isolated target with |d0| >= 10 mm or r_v >= 100 mm; target = pT > 1 GeV, |eta| < 4, >= 5 stored hits). r_first and d0
are the converter's, the same columns the native diag tables carry. No model output of a gun target is read.

Native side and statistics: unchanged from results/pileup_hp/addendum_h_inputs/rw_projection.py (c0d689a): the stored
diag tables of the frozen Canto-L (TrackML) on the 150 native val events (C-full and C-1GeV, WP-A), targets with
|d0| >= 10 mm or r_v >= 100 mm, generation 0 and all generations; cells r_first [0, 200), [200, 700), [700, inf) mm by
|d0|/r_first [0, 0.05), [0.05, 0.2), [0.2, inf); the gun-weighted Delta, its event-bootstrap sigma (B = 500,
default_rng(12345)), and the projection to n_P targets with sigma scaled by sqrt(n_eff / n_P) and the outcome odds of
section 6 (H-test Delta normal around the val Delta, val and projected variances added).

Target counts of the projection: the H-val P_P count scaled to the 600 H-test hosts (x 4), and the 2,894 and 2,160 of
the addendum's tables for comparison.

    python results/pileup_hp/h_t05_truth_rw_projection.py --set hval --out-json results/pileup_hp/h_t05_truth_rw_projection_v1.json
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import h1 as H
from hnl.overlay import writer as W

PD = "/work/submit/jinboz1/hnl/pileup_design"
DIAG = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/diag/displaced"
REPO = "/work/submit/jinboz1/hnl/hnl-displaced"
VAL = set(json.load(open(f"{REPO}/configs/trackml/split_tp_v1.json"))["val"])
RF = [0, 200, 700, np.inf]
RAT = [0, 0.05, 0.2, np.inf]
PROBE = f"{REPO}/results/pileup_hp/addendum_h_inputs/rw_projection.json"
N_HTEST_HOSTS = 600


def stored(root: str, set_name: str, table: str, columns: list[str]) -> pd.DataFrame:
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    parts = [pd.read_parquet(os.path.join(root, sh[table]["file"]), columns=columns)
             for _, sh in sorted(meta["shards"].items())]
    return pd.concat(parts, ignore_index=True), meta


def gun_side(root: str, set_name: str) -> tuple[pd.DataFrame, dict]:
    p, meta = stored(root, set_name, "particles", ["event", "particle_id", "pt", "eta", "d0", "rv", "n_hits",
                                                   "is_target", "r_first", "px", "py"])
    h, _ = stored(root, set_name, "hits", ["event", "pid", "z"])
    g = H.gun_targets(p, h)
    n_ev = int(p.event.nunique())
    pp = g[g.pp].copy()
    pp["b_rf"] = np.digitize(pp.r_first.astype(float), RF) - 1
    pp["b_ra"] = np.digitize(pp.ratio, RAT) - 1
    n_pp, n_pd = int(len(pp)), int(g.pd.sum())
    scale = N_HTEST_HOSTS / n_ev
    yld = {"set": set_name, "events": n_ev, "overlay_hash": meta.get("overlay_hash"),
           "gun_particles": int(len(g)), "gun_targets": int(g.target.sum()),
           "P_P": n_pp, "P_D": n_pd,
           "P_P_per_isolated_llp": n_pp / (n_ev * H.N_ISO), "P_D_per_dense_llp": n_pd / (n_ev * H.N_DENSE),
           "P_P_by_region": {r: int((pp.region == r).sum()) for r in H.REGIONS},
           "P_P_central_share": float((pp.region == "central").mean()) if n_pp else None,
           "P_P_with_soft_sibling": int(pp.soft_sibling.sum()),
           "P_P_ratio_ge_0p2": float((pp.ratio >= 0.2).mean()) if n_pp else None,
           "P_P_abs_d0_ge_100": float((pp.abs_d0 >= 100).mean()) if n_pp else None,
           "projected_htest_P_P": round(n_pp * scale), "projected_htest_P_P_poisson_sigma": round(math.sqrt(n_pp) * scale),
           "projected_htest_central_P_P": round(int((pp.region == "central").sum()) * scale),
           "statistics_check": f"H1 is labelled underpowered if H-test holds fewer than {H.MIN_PP} P_P targets after any "
                               "restriction; the H-test count is committed at H-T1 (section 6)"}
    return pp, yld


def project(nat, name, gw, n_targets):
    """Unchanged from addendum_h_inputs/rw_projection.py (c0d689a)."""
    cell = nat.groupby(["b_rf", "b_ra"])
    n_cell = cell.size()
    common = [k for k in gw.index if k in n_cell.index]
    cov = float(sum(gw[k] for k in common))
    wcell = {k: gw[k] / cov for k in common}
    w = np.array([wcell.get((a, b), 0.0) / n_cell.get((a, b), 1) for a, b in zip(nat.b_rf, nat.b_ra)])
    d = nat.delta.to_numpy()
    est = 100 * float(np.sum(w * d))
    ev = nat.event.to_numpy()
    evs = np.unique(ev)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(evs), size=(500, len(evs)))
    pos = {e: np.flatnonzero(ev == e) for e in evs}
    bs = []
    for r in idx:
        ii = np.concatenate([pos[evs[j]] for j in r])
        sub = nat.iloc[ii]
        c = sub.groupby(["b_rf", "b_ra"]).delta.mean()
        bs.append(100 * sum(wcell[k] * c.get(k, 0.0) for k in common))
    s_val = float(np.std(bs, ddof=1))
    n_eff = 1.0 / float(np.sum(w ** 2))
    per_cell = {f"{a},{b}": {"n_native": int(n_cell.get((a, b), 0)), "gun_share": round(float(gw.get((a, b), 0)), 4),
                             "delta_pts": round(100 * float(nat[(nat.b_rf == a) & (nat.b_ra == b)].delta.mean()), 2)
                             if (a, b) in n_cell.index else None,
                             "eff_full": round(float(nat[(nat.b_rf == a) & (nat.b_ra == b)].dm_full.mean()), 3)
                             if (a, b) in n_cell.index else None} for (a, b) in sorted(set(gw.index) | set(n_cell.index))}
    proj = {}
    for nP in n_targets:
        s = s_val * math.sqrt(n_eff / nP)
        pred = math.sqrt(s_val ** 2 + s ** 2)
        phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        p_hold = 1 - phi((-1 + 2 * s - est) / pred)
        p_fail = phi((-1 - 2 * s - est) / pred)
        proj[nP] = {"sigma_proj": round(s, 3), "delta_minus_2sigma": round(est - 2 * s, 3),
                    "p_hold": round(p_hold, 2), "p_inconclusive": round(1 - p_hold - p_fail, 2), "p_fail": round(p_fail, 2)}
    return {"population": name, "n_native": int(len(nat)), "gun_coverage": round(cov, 4), "delta_rw_pts": round(est, 3),
            "sigma_val_pts": round(s_val, 3), "n_eff": round(n_eff, 1), "cells": per_cell, "projection": proj}


def natives() -> pd.DataFrame:
    """Unchanged from addendum_h_inputs/rw_projection.py (c0d689a)."""
    cols = ["event", "particle_id", "d0", "rv", "r_first", "dm"]
    f = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(glob.glob(f"{DIAG}/diag_full_b*.parquet"))])
    h1 = pd.concat([pd.read_parquet(p, columns=["event", "particle_id", "dm"])
                    for p in sorted(glob.glob(f"{DIAG}/diag_1gev_b*.parquet"))])
    assert set(f.event) <= VAL and set(h1.event) <= VAL
    m = f.merge(h1, on=["event", "particle_id"], suffixes=("_full", "_1gev"))
    m["gen"] = ((m.particle_id.values.astype(np.uint64) >> np.uint64(24)) & np.uint64(0xFFF)).astype(np.int64)
    m = m[(m.d0.abs() >= 10) | (m.rv >= 100)].copy()
    m["ratio"] = m.d0.abs() / m.r_first
    m["b_rf"] = np.digitize(m.r_first, RF) - 1
    m["b_ra"] = np.digitize(m.ratio, RAT) - 1
    m["delta"] = m.dm_full.astype(float) - m.dm_1gev.astype(float)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="hval", choices=["hval", "pilot"])
    ap.add_argument("--root", default=W.DERIVED_OVERLAY)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--gun-only", action="store_true", help="the yield and the gun cells only (code test)")
    a = ap.parse_args()
    t0 = time.time()
    pp, yld = gun_side(a.root, a.set)
    gw = pp.groupby(["b_rf", "b_ra"]).size() / len(pp)
    with open(PROBE) as fh:
        probe = json.load(fh)
    out = {"what": "addendum H step H-T0.5.1: H-val P_P yield and the reweighted val projection of section 6 with the "
                   "H-val gun truth after the emulation (truth on the gun side; stored val outputs of native targets)",
           "root": a.root, "yield": yld,
           "gun_bins": {f"{x},{y}": round(float(v), 4) for (x, y), v in gw.items()},
           "gun_probe_bins": probe["gun_bins"], "gun_probe_pp_like": probe["gun_pp_like"],
           "gun_probe_ratio_ge_0p2": probe["gun_ratio_ge_0p2"], "gun_probe_d0_ge_100": probe["gun_d0_ge_100"],
           "code": {"script": os.path.relpath(os.path.abspath(__file__), REPO), "native_side_from": "results/pileup_hp/addendum_h_inputs/rw_projection.py (c0d689a)",
                    "gun_probe_record": PROBE.replace(REPO + "/", "")}}
    if not a.gun_only:
        m = natives()
        n_targets = (int(yld["projected_htest_P_P"]), 2894, 2160)
        out["n_targets_projected"] = list(n_targets)
        out["reweighted"] = [project(m[m.gen == 0], "gen-0, |d0|>=10 or rv>=100", gw, n_targets),
                             project(m, "all gens, |d0|>=10 or rv>=100", gw, n_targets)]
        out["reweighted_with_probe_cells"] = {r["population"]: {"delta_rw_pts": r["delta_rw_pts"]}
                                              for r in probe["reweighted"]}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({k: v for k, v in out.items() if k not in ("reweighted",)}, indent=1,
                     default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    for r in out.get("reweighted", []):
        print(r["population"], r["n_native"], r["gun_coverage"], r["delta_rw_pts"], r["sigma_val_pts"], r["n_eff"],
              json.dumps(r["projection"]))


if __name__ == "__main__":
    main()
