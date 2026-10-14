"""Independent check of the hit-label exchange reported by hnl.eval.dense_jet_check (results/dense_jet_check_v1.md).

Reads the authors' npz files directly (hnl.data.npz.load_event: np.load of the unpacked Zenodo files, no conversion),
truth only, no model output. Per particle the hits are ordered along the npz chain (layerless_true_edges). For every
hit h of particle A with at least four hits, a local fit to the nearest chain hits of A that are not on the module of
h (up to two on each side, at least three in all):
  - transverse plane: quadratic y'(u) in the frame of the chord of the fit hits (a helix to second order);
  - (r, z) plane: straight line by total least squares.
res_A(h) = hypot of the two distances of h from these fits. For two hits i (particle A) and j (particle B) on one
module (same geometry id), cur = res_A(i) + res_B(j), swp = res_A(j) + res_B(i), the fits of A and B being those of i
and j. The pair is exchanged when swp < 0.5 cur and cur - swp > 1 mm (the criterion of dense_jet_check, applied to
these residuals). Control: hits of two particles on different modules of one layer, at most 50 mm apart. Also:
the residual of every hit w.r.t. its own particle's fit (a kink in the truth chain), split by shared / unshared
module; within modules holding exactly two hits of two particles, the order of hit ids against the order of particle
ids; and, with --derived, a row-by-row comparison of the derived hit tables (hnl.data.convert) with the npz arrays.

    python scripts/diagnostics/label_swap_verify.py --sample ttbar --split val --out DIR [--limit N] [--workers N]
        [--examples K] [--derived]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os

import numpy as np
import pandas as pd

SEP_EDGES = [0, 1, 2, 5, 10, 20, 50, 1e5]


def chain_orders(pid, src, dst):
    nxt = dict(zip(src.tolist(), dst.tolist()))
    has_prev = set(dst.tolist())
    out, bad = {}, 0
    for p in np.unique(pid):
        rows = np.flatnonzero(pid == p)
        heads = [r for r in rows.tolist() if r not in has_prev]
        if len(heads) != 1:
            bad += 1
            continue
        order, c = [], heads[0]
        while c is not None:
            order.append(c)
            c = nxt.get(c)
        if len(order) != len(rows):
            bad += 1
            continue
        out[int(p)] = np.asarray(order, dtype=np.int64)
    return out, bad


def local_sets(geo, orders):
    """Per hit: indices of up to two chain neighbours on each side that are not on its module (-1 padded, 4 slots)."""
    n = len(geo)
    nb = np.full((n, 4), -1, np.int64)
    for o in orders.values():
        if len(o) < 4:
            continue
        g = geo[o]
        for k, i in enumerate(o):
            prev = [o[m] for m in range(k - 1, -1, -1) if g[m] != g[k]][:2]
            nxt = [o[m] for m in range(k + 1, len(o)) if g[m] != g[k]][:2]
            pts = prev + nxt
            if len(pts) < 3:   # at a chain end: take more from the other side
                more = ([o[m] for m in range(k + 1, len(o)) if g[m] != g[k]][2:4] if len(prev) < 2 else
                        [o[m] for m in range(k - 1, -1, -1) if g[m] != g[k]][2:4])
                pts = (pts + more)[:4]
            if len(pts) >= 3:
                nb[i, :len(pts)] = pts
    return nb


def fit_residual(x, y, r, z, nbi, h):
    """Distance of hits h (array) from the local fits defined by the neighbour sets nbi (rows of 4, -1 padded)."""
    out = np.full(len(h), np.nan)
    for k in range(len(h)):
        s = nbi[k][nbi[k] >= 0]
        if len(s) < 3:
            continue
        px, py = x[s], y[s]
        # transverse: chord frame of the two fit points farthest apart, quadratic v(u)
        d2 = (px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2
        a, b = np.unravel_index(np.argmax(d2), d2.shape)
        ux, uy = px[b] - px[a], py[b] - py[a]
        L = np.hypot(ux, uy)
        if L < 1e-6:
            continue
        ux, uy = ux / L, uy / L
        u = (px - px[a]) * ux + (py - py[a]) * uy
        v = -(px - px[a]) * uy + (py - py[a]) * ux
        c = np.polyfit(u, v, 2)
        hu = (x[h[k]] - px[a]) * ux + (y[h[k]] - py[a]) * uy
        hv = -(x[h[k]] - px[a]) * uy + (y[h[k]] - py[a]) * ux
        dxy = abs(hv - np.polyval(c, hu))
        # (r, z): total-least-squares line
        P = np.c_[r[s], z[s]]
        m = P.mean(0)
        _, _, vt = np.linalg.svd(P - m)
        nrm = vt[1]
        drz = abs((np.array([r[h[k]], z[h[k]]]) - m) @ nrm)
        out[k] = np.hypot(dxy, drz)
    return out


def residual_matrix(x, y, r, z, nb, owner_hits, test_hits):
    """res of test_hits[k] w.r.t. the fit of owner_hits[k]."""
    return fit_residual(x, y, r, z, nb[owner_hits], test_hits)


def same_key_pairs(key):
    o = np.argsort(key, kind="stable")
    k = key[o]
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    en = np.r_[st[1:], len(k)]
    ii, jj = [], []
    for s, e in zip(st, en):
        if e - s < 2:
            continue
        a, b = np.triu_indices(e - s, 1)
        ii.append(o[s + a])
        jj.append(o[s + b])
    if not ii:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def work(task):
    from hnl.data.npz import COL, load_event, particle_ids
    from hnl.geometry.ids import decode_geometry_id
    sample, e = task["sample"], task["event"]
    ev = load_event(sample, e)
    xs = ev["x"].astype(np.float64)
    r, phi, z = xs[:, 0], xs[:, 1], xs[:, 2]
    x, y = r * np.cos(phi), r * np.sin(phi)
    pid, hid, geo = ev["pid"], ev["hid"], ev["layers"]
    g = decode_geometry_id(geo)
    vl = g["volume"] * 10000 + g["layer"]
    te = ev["layerless_true_edges"]
    orders, bad = chain_orders(pid, te[0], te[1])
    nb = local_sets(geo, orders)
    P = ev["particles"]
    pt = pd.Series(P[:, COL["pt"]], index=particle_ids(P))
    own = fit_residual(x, y, r, z, nb, np.arange(len(pid)))
    # same-module pairs of two particles
    i, j = same_key_pairs(geo)
    k = pid[i] != pid[j]
    i, j = i[k], j[k]
    shared = np.zeros(len(pid), bool)
    shared[i] = True
    shared[j] = True
    ok = (nb[i, 0] >= 0) & (nb[j, 0] >= 0)
    i, j = i[ok], j[ok]
    rAi, rBj = own[i], own[j]
    rAj = residual_matrix(x, y, r, z, nb, i, j)
    rBi = residual_matrix(x, y, r, z, nb, j, i)
    cur, swp = rAi + rBj, rAj + rBi
    ex = (swp < 0.5 * cur) & (cur - swp > 1.0)
    sepxy = np.hypot(x[i] - x[j], y[i] - y[j])
    sep3 = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
    pairs = pd.DataFrame({"event": e, "i": i, "j": j, "vol": g["volume"][i], "layer": g["layer"][i],
                          "pid_i": pid[i], "pid_j": pid[j], "hid_i": hid[i], "hid_j": hid[j],
                          "pt_i": pt.reindex(pid[i]).to_numpy(), "pt_j": pt.reindex(pid[j]).to_numpy(),
                          "sepxy": sepxy, "sep3": sep3, "cur": cur, "swp": swp, "exch": ex})
    # control: different modules of one layer, <= 50 mm apart (transverse)
    ci, cj = same_key_pairs(vl)
    kc = (pid[ci] != pid[cj]) & (geo[ci] != geo[cj]) & (np.hypot(x[ci] - x[cj], y[ci] - y[cj]) <= 50.0)
    ci, cj = ci[kc], cj[kc]
    okc = (nb[ci, 0] >= 0) & (nb[cj, 0] >= 0)
    ci, cj = ci[okc], cj[okc]
    if len(ci) > 4000:   # cap the control per event
        sel = np.random.default_rng(e).choice(len(ci), 4000, replace=False)
        ci, cj = ci[sel], cj[sel]
    ccur = own[ci] + own[cj]
    cswp = residual_matrix(x, y, r, z, nb, ci, cj) + residual_matrix(x, y, r, z, nb, cj, ci)
    cex = (cswp < 0.5 * ccur) & (ccur - cswp > 1.0)
    ctrl = pd.DataFrame({"event": e, "sepxy": np.hypot(x[ci] - x[cj], y[ci] - y[cj]), "exch": cex})
    # two-hit two-particle modules: hid order vs pid order
    o = np.argsort(geo, kind="stable")
    gs = geo[o]
    st = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    en = np.r_[st[1:], len(gs)]
    two = [(o[s], o[s + 1]) for s, t in zip(st, en) if t - s == 2 and pid[o[s]] != pid[o[s + 1]]]
    n_two = len(two)
    n_agree = sum(int((hid[a] < hid[b]) == (pid[a] < pid[b])) for a, b in two)
    hits = pd.DataFrame({"event": e, "own_res": own, "shared": shared, "vol": g["volume"],
                         "pt": pt.reindex(pid).to_numpy()})
    return {"pairs": pairs, "ctrl": ctrl, "hits": hits[np.isfinite(hits.own_res)],
            "two": (n_two, n_agree), "bad_chains": bad}


def derived_check(sample, events):
    """Row-by-row equality of the derived hit tables and the npz arrays."""
    from hnl.data.convert import read_table
    from hnl.data.npz import load_event
    root = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived", "events",
                        "v1")
    h = read_table(root, "hits", sample, events=events, columns=["event", "row", "hid", "r", "phi", "z", "geoid",
                                                                    "pid"])
    bad = 0
    for e, d in h.groupby("event"):
        d = d.sort_values("row")
        ev = load_event(sample, int(e))
        same = (np.array_equal(d.pid.to_numpy(), ev["pid"]) and np.array_equal(d.hid.to_numpy(), ev["hid"])
                and np.array_equal(d.geoid.to_numpy(), ev["layers"])
                and np.allclose(d.r.to_numpy(), ev["x"][:, 0]) and np.allclose(d.phi.to_numpy(), ev["x"][:, 1])
                and np.allclose(d.z.to_numpy(), ev["x"][:, 2].astype(np.float64)))
        bad += not same
    return {"events": int(h.event.nunique()), "mismatched_events": int(bad)}


def main(argv=None):
    from hnl.data.splits import read_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--derived", action="store_true")
    a = ap.parse_args(argv)
    evs = read_split(a.sample, a.split)
    if a.limit:
        evs = evs[:a.limit]
    tasks = [{"sample": a.sample, "event": int(e)} for e in evs]
    res = []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for rr in ex.map(work, tasks, chunksize=4):
            res.append(rr)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.sample}_{a.split}"
    pairs = pd.concat([q["pairs"] for q in res], ignore_index=True)
    ctrl = pd.concat([q["ctrl"] for q in res], ignore_index=True)
    hits = pd.concat([q["hits"] for q in res], ignore_index=True)
    pairs.to_parquet(os.path.join(a.out, f"pairs_{tag}.parquet"), index=False)
    ctrl.to_parquet(os.path.join(a.out, f"ctrl_{tag}.parquet"), index=False)
    hits.to_parquet(os.path.join(a.out, f"hits_{tag}.parquet"), index=False)
    n_two = sum(q["two"][0] for q in res)
    n_agree = sum(q["two"][1] for q in res)
    summ = {"sample": a.sample, "split": a.split, "events": len(evs), "pairs": len(pairs),
            "pairs_exch": int(pairs.exch.sum()), "ctrl": len(ctrl), "ctrl_exch": int(ctrl.exch.sum()),
            "two_hit_modules": n_two, "hid_order_eq_pid_order": n_agree,
            "bad_chains": int(sum(q["bad_chains"] for q in res))}
    if a.derived:
        summ["derived"] = derived_check(a.sample, evs)
    with open(os.path.join(a.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
