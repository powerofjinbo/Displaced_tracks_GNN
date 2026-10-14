"""Robust trajectory test of the hit labels (companion of label_swap_verify.py); the authors' npz files, truth only.

For every "clean" particle (pT >= PTMIN, at least 6 hits), a circle in the transverse plane is fitted by RANSAC over
triplets of its labelled hits (inlier: |distance to centre - R| < TOL), then refitted to the inliers. The particle is
used if at least 5 hits and 60 % of its hits are inliers. A labelled hit that is an outlier is "displaced from its
trajectory"; it is "explained by an exchange" if another hit of the same module (same geometry id), labelled as another
particle, lies within TOL of the circle (the particle's own position carries another label). Controls: outliers on
modules that hold no other hit. Per module whose hits all belong to clean particles and where every particle has
exactly one position within TOL, the recovered assignment is compared with the labels.

    python scripts/diagnostics/label_swap_ransac.py --sample S --split SPLIT --out DIR [--limit N] [--ptmin 2]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

TOL = 1.0


def circle3(p):
    """Circles through point triplets p (k, 3, 2): centres (k, 2), radii (k,)."""
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    d = np.where(np.abs(d) < 1e-12, np.nan, d)
    sa, sb, sc = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
    ux = (sa * (b[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc * (a[:, 1] - b[:, 1])) / d
    uy = (sa * (c[:, 0] - b[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc * (b[:, 0] - a[:, 0])) / d
    return np.c_[ux, uy], np.hypot(a[:, 0] - ux, a[:, 1] - uy)


def fit_circle(px, py):
    """Algebraic circle fit, centred coordinates."""
    mx, my = px.mean(), py.mean()
    u, v = px - mx, py - my
    A = np.c_[u, v, np.ones_like(u)]
    c = np.linalg.lstsq(A, -(u ** 2 + v ** 2), rcond=None)[0]
    cx, cy = -c[0] / 2, -c[1] / 2
    return np.array([cx + mx, cy + my]), np.sqrt(max(cx ** 2 + cy ** 2 - c[2], 0.0))


def ransac(px, py, max_trip=600, seed=0):
    n = len(px)
    trip = np.array(list(combinations(range(n), 3)))
    if len(trip) > max_trip:
        trip = trip[np.random.default_rng(seed).choice(len(trip), max_trip, replace=False)]
    pts = np.stack([np.c_[px, py][trip[:, k]] for k in range(3)], 1)
    cen, R = circle3(pts)
    res = np.abs(np.hypot(px[None, :] - cen[:, 0:1], py[None, :] - cen[:, 1:2]) - R[:, None])
    cnt = np.where(np.isfinite(R)[:, None], res < TOL, False).sum(1)
    best = int(np.argmax(cnt))
    inl = res[best] < TOL
    if inl.sum() >= 3:
        c2, R2 = fit_circle(px[inl], py[inl])
        res2 = np.abs(np.hypot(px - c2[0], py - c2[1]) - R2)
        if (res2 < TOL).sum() >= inl.sum():
            return c2, R2, res2
    return cen[best], R[best], res[best]


def work(task):
    from hnl.data.npz import COL, load_event, particle_ids
    from hnl.geometry.ids import decode_geometry_id
    sample, e, ptmin = task["sample"], task["event"], task["ptmin"]
    ev = load_event(sample, e)
    xs = ev["x"].astype(np.float64)
    r, phi, z = xs[:, 0], xs[:, 1], xs[:, 2]
    x, y = r * np.cos(phi), r * np.sin(phi)
    pid, hid, geo = ev["pid"], ev["hid"], ev["layers"]
    vol = decode_geometry_id(geo)["volume"]
    P = ev["particles"]
    ids = particle_ids(P)
    pt = pd.Series(P[:, COL["pt"]], index=ids)
    pdg = pd.Series(P[:, COL["particle_type"]], index=ids)
    up, cnt = np.unique(pid, return_counts=True)
    nmod = pd.Series(geo).map(pd.Series(geo).value_counts()).to_numpy()
    rows, fits, parts = [], {}, []
    for p, c in zip(up, cnt):
        if c < 6 or pt.get(p, 0) < ptmin:
            continue
        h = np.flatnonzero(pid == p)
        cen, R, res = ransac(x[h], y[h], seed=int(p % 100000))
        ok = (res < TOL).sum() >= 5 and (res < TOL).mean() >= 0.6
        if not ok:
            continue
        fits[int(p)] = (cen, R)
        parts.append({"event": e, "pid": int(p), "pt": float(pt[p]), "n_hits": int(c), "n_inl": int((res < TOL).sum()),
                      "d0_clean_circle": float(abs(np.hypot(cen[0], cen[1]) - R)), "n_exch": 0})
        for k, hh in enumerate(h):
            same = np.flatnonzero((geo == geo[hh]) & (pid != p))
            best_row, best_res, best_sep = -1, np.inf, np.nan
            if len(same):
                rr = np.abs(np.hypot(x[same] - cen[0], y[same] - cen[1]) - R)
                m = int(np.argmin(rr))
                best_row, best_res = int(same[m]), float(rr[m])
                best_sep = float(np.hypot(x[same[m]] - x[hh], y[same[m]] - y[hh]))
            rows.append({"event": e, "pid": int(p), "pt": float(pt[p]), "pdg": int(pdg[p]), "row": int(hh),
                         "vol": int(vol[hh]), "n_on_module": int(nmod[hh]), "res": float(res[k]),
                         "other_row": best_row, "other_res": best_res, "other_sep": best_sep,
                         "other_pid": int(pid[best_row]) if best_row >= 0 else 0})
    rows_df = pd.DataFrame(rows)
    parts_df = pd.DataFrame(parts)
    if len(rows_df):
        ex = rows_df[(rows_df.n_on_module > 1) & (rows_df.res >= TOL) & (rows_df.other_res < TOL)]
        parts_df["n_exch"] = parts_df.pid.map(ex.groupby("pid").size()).fillna(0).astype(int)
    # modules whose hits all belong to clean particles: recovered assignment vs labels
    mods = []
    for gid in np.unique(geo):
        s = np.flatnonzero(geo == gid)
        if len(s) < 2 or len(np.unique(pid[s])) != len(s):
            continue
        if not all(int(q) in fits for q in pid[s]):
            continue
        s = s[np.argsort(hid[s])]
        M = np.array([[abs(np.hypot(x[k] - fits[int(q)][0][0], y[k] - fits[int(q)][0][1]) - fits[int(q)][1])
                       for k in s] for q in pid[s]])
        hit = M < TOL
        if not ((hit.sum(1) == 1).all() and (hit.sum(0) == 1).all()):
            mods.append({"event": e, "n": len(s), "resolved": False, "perm": ""})
            continue
        perm = hit.argmax(1)   # label (in hid order) -> position index (hid order) of its trajectory
        mods.append({"event": e, "n": len(s), "resolved": True, "perm": ",".join(map(str, perm)),
                     "identity": bool((perm == np.arange(len(s))).all()), "vol": int(vol[s[0]]),
                     "pids": ",".join(str(int(q)) for q in pid[s]),
                     "pts": ",".join(f"{pt[q]:.3f}" for q in pid[s]),
                     "rs": ",".join(f"{r[k]:.2f}" for k in s), "phis": ",".join(f"{phi[k]:.5f}" for k in s),
                     "zs": ",".join(f"{z[k]:.2f}" for k in s)})
    return rows_df, pd.DataFrame(mods), parts_df


def main(argv=None):
    from hnl.data.splits import read_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args(argv)
    evs = read_split(a.sample, a.split)
    if a.limit:
        evs = evs[:a.limit]
    tasks = [{"sample": a.sample, "event": int(e), "ptmin": a.ptmin} for e in evs]
    hits, mods, parts = [], [], []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for h, m, q in ex.map(work, tasks, chunksize=2):
            hits.append(h)
            mods.append(m)
            parts.append(q)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.sample}_{a.split}"
    H = pd.concat(hits, ignore_index=True)
    M = pd.concat(mods, ignore_index=True)
    H.to_parquet(os.path.join(a.out, f"ransac_hits_{tag}.parquet"), index=False)
    M.to_parquet(os.path.join(a.out, f"ransac_mods_{tag}.parquet"), index=False)
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(a.out, f"ransac_parts_{tag}.parquet"), index=False)
    sh = H.n_on_module > 1
    out_ = H.res >= TOL
    exch = out_ & (H.other_res < TOL)
    summ = {"sample": a.sample, "split": a.split, "events": len(evs), "ptmin": a.ptmin,
            "clean_particles": int(H.groupby(["event", "pid"]).ngroups), "hits": len(H),
            "hits_unshared": int((~sh).sum()), "outlier_unshared": int((out_ & ~sh).sum()),
            "hits_shared": int(sh.sum()), "outlier_shared": int((out_ & sh).sum()),
            "outlier_shared_explained": int((exch & sh).sum()),
            "modules_all_clean": int(len(M)), "modules_resolved": int(M.resolved.sum()) if len(M) else 0,
            "modules_resolved_nonidentity": int((M.resolved & ~M.identity.fillna(True).astype(bool)).sum())
            if len(M) else 0}
    with open(os.path.join(a.out, f"ransac_summary_{tag}.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
