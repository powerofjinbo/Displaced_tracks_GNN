"""Reweighted val projection of H1 (truth on the gun side, stored val outputs of native targets only).

Gun side: the gun probe of the patched build (hp_field/build-v15-zmirror/gun_sim.tar, Fatras sim hits without
emulation, inputs hp_field/build-v15/gun_in). P_P-like = isolated gun particles with pT > 1 GeV, |eta| < 4, at least
5 sim hits and |d0| >= 10 mm or r_v >= 100 mm (d0 of hnl.trackml.convert.d0_from_vertex). r_first = r of the sim hit
nearest to the production vertex (as the converter's rank 0).

Native side: the stored diag tables of the frozen Canto-L (TrackML) on the 150 val events (C-full and C-1GeV, WP-A),
the same tables as study_design/val_discordance.py. No gun target, no test event.

Bins: r_first [0, 200), [200, 700), [700, inf) mm; |d0|/r_first [0, 0.05), [0.05, 0.2), [0.2, inf). Native
candidates: targets with |d0| >= 10 mm or r_v >= 100 mm, (i) generation 0 only, (ii) all generations. Delta per bin
and the gun-weighted sum; sigma from an event bootstrap of the val events (B = 500, default_rng(12345)); the
projection to n_P targets scales the bootstrap sigma by sqrt(n_eff / n_P) with n_eff = 1 / sum(w_i^2) per target."""
import glob
import io
import json
import math
import sys
import tarfile

import numpy as np
import pandas as pd

from hnl.trackml.convert import d0_from_vertex

PD = "/work/submit/jinboz1/hnl/pileup_design"
DIAG = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/diag/displaced"
VAL = set(json.load(open("/work/submit/jinboz1/hnl/hnl-displaced/configs/trackml/split_tp_v1.json"))["val"])
RF = [0, 200, 700, np.inf]
RAT = [0, 0.05, 0.2, np.inf]

# ---------------------------------------------------------------- gun probe
truth = pd.read_csv(f"{PD}/hp_field/build-v15/gun_in/gun_probe_truth.csv")
truth = truth[truth.q != 0].copy()
truth["pt"] = np.hypot(truth.px, truth.py)
truth["eta"] = np.arcsinh(truth.pz / truth.pt)
truth["d0"] = d0_from_vertex(truth.vx.values, truth.vy.values, truth.px.values, truth.py.values, truth.q.values)
rows = []
with tarfile.open(f"{PD}/hp_field/build-v15-zmirror/gun_sim.tar") as tf:
    for m in tf.getmembers():
        if not m.name.endswith("-hits.csv"):
            continue
        ev = int(m.name.split("/event")[1].split("-")[0])
        h = pd.read_csv(io.BytesIO(tf.extractfile(m).read()), usecols=["particle_id", "tx", "ty", "tz"])
        h["event"] = ev
        rows.append(h)
hits = pd.concat(rows, ignore_index=True)
hits = hits.merge(truth[["event", "particle_id", "vx", "vy", "vz"]], on=["event", "particle_id"], how="inner")
hits["dist"] = np.sqrt((hits.tx - hits.vx) ** 2 + (hits.ty - hits.vy) ** 2 + (hits.tz - hits.vz) ** 2)
hits["r"] = np.hypot(hits.tx, hits.ty)
first = hits.sort_values(["event", "particle_id", "dist"]).groupby(["event", "particle_id"]).agg(
    r_first=("r", "first"), n_hits=("r", "size")).reset_index()
g = truth.merge(first, on=["event", "particle_id"], how="left")
g["n_hits"] = g.n_hits.fillna(0)
pp = g[(g.component == "iso") & (g.pt > 1) & (g.eta.abs() < 4) & (g.n_hits >= 5)
       & ((g.d0.abs() >= 10) | (g.r_v >= 100))].copy()
pp["ratio"] = pp.d0.abs() / pp.r_first
pp["b_rf"] = np.digitize(pp.r_first, RF) - 1
pp["b_ra"] = np.digitize(pp.ratio, RAT) - 1
gw = pp.groupby(["b_rf", "b_ra"]).size() / len(pp)
out = {"gun_pp_like": int(len(pp)), "gun_ratio_ge_0p2": float((pp.ratio >= 0.2).mean()),
       "gun_d0_ge_100": float((pp.d0.abs() >= 100).mean()), "gun_bins": {f"{a},{b}": round(float(v), 4) for (a, b), v in gw.items()}}

# ---------------------------------------------------------------- natives
cols = ["event", "particle_id", "d0", "rv", "r_first", "dm"]
f = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(glob.glob(f"{DIAG}/diag_full_b*.parquet"))])
h1 = pd.concat([pd.read_parquet(p, columns=["event", "particle_id", "dm"]) for p in sorted(glob.glob(f"{DIAG}/diag_1gev_b*.parquet"))])
assert set(f.event) <= VAL and set(h1.event) <= VAL
m = f.merge(h1, on=["event", "particle_id"], suffixes=("_full", "_1gev"))
m["gen"] = ((m.particle_id.values.astype(np.uint64) >> np.uint64(24)) & np.uint64(0xFFF)).astype(np.int64)
m = m[(m.d0.abs() >= 10) | (m.rv >= 100)].copy()
m["ratio"] = m.d0.abs() / m.r_first
m["b_rf"] = np.digitize(m.r_first, RF) - 1
m["b_ra"] = np.digitize(m.ratio, RAT) - 1
m["delta"] = m.dm_full.astype(float) - m.dm_1gev.astype(float)


def project(nat, name, n_targets=(2894, 2160, 1040)):
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


out["reweighted"] = [project(m[m.gen == 0], "gen-0, |d0|>=10 or rv>=100"), project(m, "all gens, |d0|>=10 or rv>=100")]
# unweighted rows with the two target counts
V = json.load(open(f"{PD}/study_design/val_discordance.json"))
rows = []
for r in V["paired"]:
    for nP in (2894, 2160, 1040):
        s = r["sigma_boot_pts"] * math.sqrt(r["n"] / nP)
        d = r["delta_pts"]
        pred = math.sqrt(r["sigma_boot_pts"] ** 2 + s ** 2)
        phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        ph = 1 - phi((-1 + 2 * s - d) / pred)
        pf = phi((-1 - 2 * s - d) / pred)
        rows.append({"population": r["population"], "n_P": nP, "sigma_proj": round(s, 3), "d_minus_2s": round(d - 2 * s, 3),
                     "hold": round(ph, 2), "inconclusive": round(1 - ph - pf, 2), "fail": round(pf, 2)})
out["unweighted"] = rows
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
