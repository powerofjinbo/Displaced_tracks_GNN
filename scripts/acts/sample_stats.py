"""Truth-level statistics of a set of npz events in the Zenodo format (src/hnl/data/npz.py), for comparing a sample
we simulate with ACTS v15 against the Zenodo sample (docs/new_samples_spec.md, results/acts_validation_v1.md).
No model is run.

Per event set: hits and particles per event; hits per particle; hits per (volume, layer); pT, eta, PDG of the
particles with hits; the secondary fraction (barcode generation > 0) and its PDG mix; the momentum of primaries
with hits near the simulation threshold; z residuals of long- and short-strip barrel hits from a straight r-z line
through the pixel-barrel hits of the same high-pT primary (tells smeared measurements from truth positions:
sigma_loc1 is 3.12 mm in the long strips and 0.35 mm in the short strips); and the label-exchange test of
hnl.eval.dense_jet_check (pairs of hits of two particles on one module, by transverse separation; control: pairs on
different modules of one layer).

    python scripts/acts/sample_stats.py --dir DIR --events 0-499 --out OUT.json [--workers N]
    python scripts/acts/sample_stats.py --dir DIR --event-file splits/ttbar_train.txt --first 500 --out OUT.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from hnl.data.npz import COL, NPZ_KEYS  # noqa: E402
from hnl.eval.dense_jet_check import SEP_EDGES, chain_orders, group_pairs, pair_test, reference_lines  # noqa: E402

PT_EDGES = np.logspace(-1, 2.5, 36)
ETA_EDGES = np.linspace(-4, 4, 33)
NHIT_EDGES = np.arange(0, 41)
P_EDGES = np.linspace(0, 2, 41)
RES_EDGES = np.linspace(-10, 10, 81)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        ev = {k: f[k] for k in f.files}
    got = {k: str(v.dtype) for k, v in ev.items()}
    if got != NPZ_KEYS:
        raise ValueError(f"{path}: keys/dtypes {got}")
    return ev


def one(path):
    ev = load(path)
    x = ev["x"].astype(np.float64)
    r, phi, z = x[:, 0], x[:, 1], x[:, 2]
    X, Y = r * np.cos(phi), r * np.sin(phi)
    pid, geo = ev["pid"], ev["layers"]
    P = ev["particles"]
    ids = P[:, COL["particle_id"]].astype(np.int64)
    vol = (geo >> 56).astype(np.int64)
    lay = ((geo >> 36) & 0xFFF).astype(np.int64)
    out = {"n_hits": len(x), "n_particles": len(P)}
    if len(np.setdiff1d(pid, ids)):
        raise ValueError(f"{path}: hit pid not in the particle table")
    if not np.array_equal(np.sort(ev["hid"]), np.arange(len(x))):
        raise ValueError(f"{path}: hid is not a permutation")
    up, cnt = np.unique(pid, return_counts=True)
    out["n_particles_with_hits"] = len(up)
    out["hits_per_particle"] = np.histogram(cnt, NHIT_EDGES)[0]
    vlu, vlc = np.unique(np.c_[vol, lay], axis=0, return_counts=True)
    out["vol_layer"] = {f"{v}_{l}": int(n) for (v, l), n in zip(vlu.tolist(), vlc.tolist())}
    sel = np.isin(ids, up)
    Ph = P[sel]
    pt = Ph[:, COL["pt"]]
    with np.errstate(invalid="ignore"):
        eta = Ph[:, COL["eta"]]
    gen = (ids[sel] >> 16) & 0xFF
    pdg = Ph[:, COL["particle_type"]].astype(np.int64)
    out["pt_all"] = np.histogram(pt, PT_EDGES)[0]
    out["pt_gen0"] = np.histogram(pt[gen == 0], PT_EDGES)[0]
    out["eta_all"] = np.histogram(eta[np.isfinite(eta)], ETA_EDGES)[0]
    out["n_gen0"], out["n_genpos"] = int((gen == 0).sum()), int((gen > 0).sum())
    out["pdg_gen0"] = {int(k): int(v) for k, v in zip(*np.unique(np.abs(pdg[gen == 0]), return_counts=True))}
    out["pdg_genpos"] = {int(k): int(v) for k, v in zip(*np.unique(np.abs(pdg[gen > 0]), return_counts=True))}
    proc = Ph[:, COL["process"]].astype(np.int64)
    out["process"] = {int(k): int(v) for k, v in zip(*np.unique(proc, return_counts=True))}
    p = np.sqrt(Ph[:, COL["px"]] ** 2 + Ph[:, COL["py"]] ** 2 + Ph[:, COL["pz"]] ** 2)
    out["p_gen0_low"] = np.histogram(p[gen == 0], P_EDGES)[0]
    # z residuals of barrel strip hits from the r-z line through the pixel-barrel hits of primaries with pT > 5 GeV
    res = {13: [], 17: []}
    hp = {int(i): k for k, i in enumerate(ids)}
    for q in up:
        row = P[hp[int(q)]]
        if row[COL["pt"]] < 5 or ((int(q) >> 16) & 0xFF) != 0:
            continue
        m = pid == q
        pix = m & (vol == 8)
        if pix.sum() < 3:
            continue
        rr, zz = r[pix], z[pix]                      # Theil-Sen line: robust to one hit of another particle
        ii, jj = np.triu_indices(len(rr), 1)
        ok = np.abs(rr[jj] - rr[ii]) > 20
        if not ok.any():
            continue
        a = np.median((zz[jj] - zz[ii])[ok] / (rr[jj] - rr[ii])[ok])
        b = np.median(zz - a * rr)
        for v in (13, 17):
            s = m & (vol == v)
            if s.any():
                res[v].extend((z[s] - (a * r[s] + b)).tolist())
    out["zres_vol13"] = np.histogram(res[13], RES_EDGES)[0]
    out["zres_vol17"] = np.histogram(res[17], RES_EDGES)[0]
    out["zres_vol13_mad"] = res[13]
    out["zres_vol17_mad"] = res[17]
    # true-edge chains: count, direction, repeated modules (as described in docs/brief.md section 2.3)
    e = ev["layerless_true_edges"]
    orders = chain_orders(pid, e[0], e[1])
    Rs = np.hypot(r, z)
    out["n_edges_minus_expected"] = int(e.shape[1] - (len(x) - len(up)))
    out["edges_out_r"] = int((r[e[1]] > r[e[0]]).sum())
    out["edges_out_R"] = int((Rs[e[1]] > Rs[e[0]]).sum())
    out["n_edges"] = int(e.shape[1])
    out["edges_same_module"] = int((geo[e[0]] == geo[e[1]]).sum())
    ch = [o for o in orders.values() if len(o) >= 2]
    out["chains"] = len(ch)
    out["chains_R_monotone"] = int(sum(np.all(np.diff(Rs[o]) >= 0) for o in ch))
    out["particles_repeated_module"] = int(sum(len(np.unique(geo[o])) < len(o) for o in orders.values()))
    # label-exchange test (as hnl.eval.dense_jet_check, work_event step 2)
    p1, p2 = reference_lines(geo, orders)
    i, j = group_pairs(geo)
    _, _, ssep, sx = pair_test(X, Y, pid, p1, p2, i, j)
    vl = vol * 1000 + lay
    ci, cj = group_pairs(vl)
    keep = (geo[ci] != geo[cj]) & (np.hypot(X[ci] - X[cj], Y[ci] - Y[cj]) <= 50.0)
    _, _, csep, cx = pair_test(X, Y, pid, p1, p2, ci[keep], cj[keep])
    nb = len(SEP_EDGES) - 1
    b1, b2 = np.digitize(ssep, SEP_EDGES) - 1, np.digitize(csep, SEP_EDGES) - 1
    out["swap_same_n"] = np.bincount(b1, minlength=nb)[:nb]
    out["swap_same_x"] = np.bincount(b1[sx], minlength=nb)[:nb]
    out["swap_ctrl_n"] = np.bincount(b2, minlength=nb)[:nb]
    out["swap_ctrl_x"] = np.bincount(b2[cx], minlength=nb)[:nb]
    return out


def merge(rows):
    tot = {}
    for k in rows[0]:
        v = rows[0][k]
        if isinstance(v, np.ndarray):
            tot[k] = np.sum([r[k] for r in rows], axis=0).tolist()
        elif isinstance(v, dict):
            d = {}
            for r in rows:
                for kk, vv in r[k].items():
                    d[kk] = d.get(kk, 0) + vv
            tot[k] = d
        elif isinstance(v, list):
            allv = np.concatenate([np.asarray(r[k], float) for r in rows]) if rows else np.zeros(0)
            tot[k] = {"n": int(len(allv)), "median_abs": float(np.median(np.abs(allv))) if len(allv) else None,
                      "sigma_mad": float(1.4826 * np.median(np.abs(allv - np.median(allv)))) if len(allv) else None}
        else:
            a = np.asarray([r[k] for r in rows], float)
            tot[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                      "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
                      "sum": float(a.sum())}
    tot["n_events"] = len(rows)
    tot["edges"] = {"pt": PT_EDGES.tolist(), "eta": ETA_EDGES.tolist(), "nhit": NHIT_EDGES.tolist(),
                    "p": P_EDGES.tolist(), "zres": RES_EDGES.tolist(), "sep": SEP_EDGES}
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--events", help="a-b (inclusive)")
    ap.add_argument("--event-file")
    ap.add_argument("--first", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if a.events:
        lo, hi = map(int, a.events.split("-"))
        idx = list(range(lo, hi + 1))
    else:
        idx = [int(s) for s in open(a.event_file) if s.strip() and not s.startswith("#")]
    if a.first:
        idx = idx[:a.first]
    paths = [os.path.join(a.dir, f"{i}.npz") for i in idx]
    paths = [p for p in paths if os.path.exists(p)]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, paths, chunksize=8))
    tot = merge(rows)
    tot["dir"], tot["events"] = a.dir, f"{len(paths)} files"
    with open(a.out, "w") as f:
        json.dump(tot, f, indent=1)
    print(f"{len(paths)} events -> {a.out}")


if __name__ == "__main__":
    main()
