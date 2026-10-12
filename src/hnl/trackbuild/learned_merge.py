"""Learned fragment merging (Stage 1 C: the "merge" part of the design panel's HTA; replaces the hand cuts of
hnl.trackbuild.merge_fragments).

A missed true edge, or a repulsion in the Mutex Watershed, splits a particle's hits into fragments; under double
majority a 3 + 3 split loses the particle. After a first pass (the same-layer Mutex Watershed, s0 = 0.2), every pair of
tracks (fragments, >= 2 hits) whose local directions roughly agree is a candidate (a loose window, wider than
merge_fragments' cuts, so that the model and not the window decides). A gradient-boosted classifier trained on ttbar
TRAIN events gives p(same particle) from features that do not change under translations of the production point and
rotations about z: the weighted helix fit of the union (hnl.trackbuild.helix_fit: normalised transverse and z rms,
outlier fraction), the two fragments' own fits (|delta kappa|, |delta cot theta|), the gap or overlap of their
arc-length ranges on the union's fitted circle, the number of detector layers both use (a particle crosses a layer
once, up to module overlaps), the union's arc-length lever arm, the closest distance between the fragments, their sizes
and the subsystems at the junction. Decoding:
candidates in decreasing p, union-find over fragments, a merge if p > tau and the two clusters share at most one layer.

    python -m hnl.trackbuild.learned_merge data --graphs DIR --out SHARD.npz [--max-events N]   (scored TRAIN graphs)
    python -m hnl.trackbuild.learned_merge train --data 'DIR/*.npz' --out MODEL.joblib
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from hnl.trackbuild import helix_fit as HF

FEATURES = ["n_small", "n_large", "u_rms_n", "u_zrms_n", "u_out_frac", "frag_rms_max", "dkappa", "dcot", "gap_s",
            "shared_layers", "span_s", "d_min", "sub_junction_a", "sub_junction_b"]
FIRST_S0 = 0.2
MAX_DPHI, MAX_DZ = 0.3, 0.3            # loose candidate window (merge_fragments "any": 0.05 / 0.05)
MAX_PARTNERS = 6                        # per fragment, the closest candidates in the window
END_DIST = 450.0                        # [mm] end-to-end candidates: an end hit of each fragment within this distance
END_PARTNERS = 4                        # per fragment end, the nearest other fragments' ends


def _sub(volume):
    v = np.asarray(volume)
    return np.where(np.isin(v, (7, 8, 9)), 0, np.where(np.isin(v, (12, 13, 14)), 1, 2))


def candidates(labels, r, phi, z):
    """Pairs (a, b) of track labels (a < b, >= 2 hits each) whose local phi(R) and z(r) lines agree at the pair's
    median radius within the loose window (at most MAX_PARTNERS per fragment, closest first), together with the
    end-to-end candidates (end_candidates)."""
    labels = np.asarray(labels)
    R = np.hypot(r, z)
    ok = labels >= 0
    trk, inv, cnt = np.unique(labels[ok], return_inverse=True, return_counts=True)
    keep = cnt >= 2
    if keep.sum() < 2:
        return np.zeros((0, 2), dtype=np.int64)
    T = len(trk)
    idx = np.flatnonzero(ok)
    # per track least-squares lines phi(R) (unwrapped around the track's circular mean) and z(r)
    ph = phi[idx]
    c = np.bincount(inv, weights=np.cos(ph), minlength=T)
    s = np.bincount(inv, weights=np.sin(ph), minlength=T)
    mean_phi = np.arctan2(s, c)
    dph = np.remainder(ph - mean_phi[inv] + np.pi, 2 * np.pi) - np.pi
    Ri, ri, zi = R[idx], r[idx], z[idx]
    n = cnt.astype(np.float64)

    def line(xv, yv):
        sx, sy = np.bincount(inv, xv, T), np.bincount(inv, yv, T)
        sxx, sxy = np.bincount(inv, xv * xv, T), np.bincount(inv, xv * yv, T)
        det = n * sxx - sx * sx
        slope = np.where(np.abs(det) > 1e-9, (n * sxy - sx * sy) / np.where(np.abs(det) > 1e-9, det, 1), 0.0)
        icpt = (sy - slope * sx) / n
        return slope, icpt
    kp, bp = line(Ri, dph)
    kz, bz = line(ri, zi)
    Rm = np.bincount(inv, Ri, T) / n
    rm = np.bincount(inv, ri, T) / n
    sel = np.flatnonzero(keep)
    a_, b_ = np.meshgrid(sel, sel, indexing="ij")
    up = a_ < b_
    a_, b_ = a_[up], b_[up]
    Rp, rp = 0.5 * (Rm[a_] + Rm[b_]), 0.5 * (rm[a_] + rm[b_])
    pa = mean_phi[a_] + kp[a_] * Rp + bp[a_]
    pb = mean_phi[b_] + kp[b_] * Rp + bp[b_]
    dphi = np.abs(np.remainder(pa - pb + np.pi, 2 * np.pi) - np.pi)
    dz = np.abs(kz[a_] * rp + bz[a_] - kz[b_] * rp - bz[b_]) / np.maximum(rp, 1.0)
    win = (dphi < MAX_DPHI) & (dz < MAX_DZ)
    a_, b_, q = a_[win], b_[win], (dphi + dz)[win]
    if len(a_) == 0:
        return end_candidates(labels, r, phi, z)
    # at most MAX_PARTNERS per fragment, closest first (a pair stays if it is within either end's limit)
    rank = np.zeros(len(q), dtype=bool)
    for end in (a_, b_):
        o = np.lexsort((q, end))
        first = np.r_[True, end[o][1:] != end[o][:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
        rank[o[(np.arange(len(o)) - start) < MAX_PARTNERS]] = True
    a_, b_ = a_[rank], b_[rank]
    pairs = np.stack([trk[a_], trk[b_]], 1).astype(np.int64)
    return np.unique(np.concatenate([pairs, end_candidates(labels, r, phi, z)]), axis=0)


def end_candidates(labels, r, phi, z):
    """Pairs (a, b), a < b, of fragments (>= 2 hits) whose end hits (innermost / outermost in R) are within END_DIST
    in 3D: the sequential continuations that a missed edge or a repulsion cut (merge_fragments' sequential mode);
    at most END_PARTNERS per fragment end (nearest first)."""
    from scipy.spatial import cKDTree
    labels = np.asarray(labels)
    R = np.hypot(r, z)
    ok = np.flatnonzero(labels >= 0)
    trk, inv, cnt = np.unique(labels[ok], return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    ok, inv = ok[keep], inv[keep]
    if len(np.unique(inv)) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    o = np.lexsort((R[ok], inv))
    first = np.r_[True, inv[o][1:] != inv[o][:-1]]
    last = np.r_[inv[o][1:] != inv[o][:-1], True]
    ends = np.r_[ok[o][first], ok[o][last]]
    owner = np.r_[inv[o][first], inv[o][last]]
    xyz = np.stack([r[ends] * np.cos(phi[ends]), r[ends] * np.sin(phi[ends]), z[ends]], 1)
    tree = cKDTree(xyz)
    k = min(len(ends), 2 * END_PARTNERS + 2)
    d, j = tree.query(xyz, k=k, distance_upper_bound=END_DIST)
    out = set()
    for i in range(len(ends)):
        n = 0
        for dd, jj in zip(d[i], j[i]):
            if not np.isfinite(dd) or jj >= len(ends):
                break
            if owner[jj] == owner[i]:
                continue
            a, b = sorted((int(trk[owner[i]]), int(trk[owner[jj]])))
            out.add((a, b))
            n += 1
            if n >= END_PARTNERS:
                break
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


def features(labels, pairs, r, phi, z, geo):
    """(X) per candidate pair in FEATURES order; geo = hnl.geometry.resolution.hit_geometry(...)[event]."""
    labels = np.asarray(labels)
    x, y = r * np.cos(phi), r * np.sin(phi)
    vol, phi_c, lay = np.asarray(geo["volume"]), np.asarray(geo["phi_c"]), np.asarray(geo["layer_id"])
    P = len(pairs)
    X = np.full((P, len(FEATURES)), np.nan)
    if P == 0:
        return X
    col = {f: k for k, f in enumerate(FEATURES)}
    # every fragment's own fit (>= 3 hits) and every union's fit, vectorised by pseudo-labels
    trk = np.unique(pairs)
    own = HF.fit(x, y, z, labels, vol, phi_c, min_hits=3)
    own_i = {int(t): k for k, t in enumerate(own["track"])}
    rows, plab = [], []
    hits = {int(t): np.flatnonzero(labels == t) for t in trk}
    for k, (a, b) in enumerate(pairs):
        h = np.r_[hits[int(a)], hits[int(b)]]
        rows.append(h)
        plab.append(np.full(len(h), k))
    rows, plab = np.concatenate(rows), np.concatenate(plab)
    uni = HF.fit(x[rows], y[rows], z[rows], plab, vol[rows], phi_c[rows], min_hits=3)
    ui = {int(t): k for k, t in enumerate(uni["track"])}
    for k, (a, b) in enumerate(pairs):
        ha, hb = hits[int(a)], hits[int(b)]
        na, nb = len(ha), len(hb)
        X[k, col["n_small"]], X[k, col["n_large"]] = min(na, nb), max(na, nb)
        if k in ui:
            j = ui[k]
            X[k, col["u_rms_n"]] = uni["rms_n"][j]
            X[k, col["u_zrms_n"]] = uni["zrms_n"][j]
            X[k, col["u_out_frac"]] = uni["n_out"][j] / (na + nb)
            # arc length on the union's circle from its reference angle: gap (> 0) or overlap (< 0) and lever arm
            th = np.arctan2(np.r_[y[ha], y[hb]] - uni["yc"][j], np.r_[x[ha], x[hb]] - uni["xc"][j])
            sl = uni["R"][j] * (np.remainder(th - uni["theta_ref"][j] + np.pi, 2 * np.pi) - np.pi)
            sa, sb = sl[:na], sl[na:]
            X[k, col["gap_s"]] = max(sa.min(), sb.min()) - min(sa.max(), sb.max())
            X[k, col["span_s"]] = sl.max() - sl.min()
        fa, fb = own_i.get(int(a)), own_i.get(int(b))
        if fa is not None and fb is not None:
            X[k, col["frag_rms_max"]] = max(own["rms_n"][fa], own["rms_n"][fb])
            X[k, col["dkappa"]] = abs(1e3 / own["R"][fa] - 1e3 / own["R"][fb])
            X[k, col["dcot"]] = abs(own["cot"][fa] - own["cot"][fb])
        X[k, col["shared_layers"]] = len(np.intersect1d(lay[ha], lay[hb]))
        pa = np.stack([x[ha], y[ha], z[ha]], 1)
        pb = np.stack([x[hb], y[hb], z[hb]], 1)
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        X[k, col["d_min"]] = d[i, j]
        X[k, col["sub_junction_a"]] = _sub(vol[ha[i]])
        X[k, col["sub_junction_b"]] = _sub(vol[hb[j]])
    return X


def labels_for(labels, pairs, pid):
    """1 if both fragments' majority particle is the same (pid != 0) and each holds > 50 % of it."""
    labels, pid = np.asarray(labels), np.asarray(pid)
    maj = {}
    for t in np.unique(pairs):
        p, c = np.unique(pid[labels == t], return_counts=True)
        k = np.argmax(c)
        maj[int(t)] = int(p[k]) if 2 * c[k] > c.sum() else 0
    return np.array([maj[int(a)] != 0 and maj[int(a)] == maj[int(b)] for a, b in pairs], dtype=bool)


def apply(labels, pairs, p, tau: float, lay, max_shared: int = 1):
    """Merge fragments greedily in decreasing p (> tau), unless the two clusters share more than max_shared layers."""
    out = np.asarray(labels).copy()
    if len(pairs) == 0:
        return out
    parent, layers = {}, {}
    for t in np.unique(pairs):
        parent[int(t)] = int(t)
        layers[int(t)] = set(lay[out == t].tolist())

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    for k in np.argsort(-p, kind="stable"):
        if p[k] <= tau:
            break
        a, b = find(int(pairs[k, 0])), find(int(pairs[k, 1]))
        if a == b or len(layers[a] & layers[b]) > max_shared:
            continue
        if len(layers[a]) < len(layers[b]):
            a, b = b, a
        parent[b] = a
        layers[a] |= layers.pop(b)
    for t in list(parent):
        r_ = find(t)
        if r_ != t:
            out[out == t] = r_
    return out


def first_pass(n, a, b, sc, L):
    from hnl.trackbuild.mutex_ws import mws_layer
    return mws_layer(n, np.asarray(a), np.asarray(b), np.asarray(sc), np.asarray(L), FIRST_S0)


class Merger:
    def __init__(self, path: str):
        import joblib
        self.model = joblib.load(path)

    def prepare(self, labels, r, phi, z, geo):
        pairs = candidates(labels, r, phi, z)
        X = features(labels, pairs, r, phi, z, geo)
        p = self.model.predict_proba(X)[:, 1] if len(pairs) else np.zeros(0)
        return pairs, p


def _event_rows(path: str):
    from hnl.eval.trackbuild_val import load_scored
    from hnl.geometry.resolution import hit_geometry
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    geo = hit_geometry([int(eid)])[int(eid)]
    lab = first_pass(len(pid), a, b, sc, geo["layer_id"])
    pairs = candidates(lab, r, phi, z)
    X = features(lab, pairs, r, phi, z, geo)
    y = labels_for(lab, pairs, pid) if len(pairs) else np.zeros(0, dtype=bool)
    return X, y, np.full(len(y), int(eid), dtype=np.int64)


def _chunk(paths):
    res = [_event_rows(p) for p in paths]
    return [np.concatenate([x[i] for x in res]) for i in range(3)]


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--graphs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-events", type=int, default=None)
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "data":
        paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
        if a.max_events is not None and a.max_events < len(paths):
            pick = np.random.default_rng(0).choice(len(paths), a.max_events, replace=False)
            paths = [paths[k] for k in np.sort(pick)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            parts = list(ex.map(_chunk, [paths[k:k + 10] for k in range(0, len(paths), 10)]))
        X, y, ev = (np.concatenate([p[i] for p in parts]) for i in range(3))
        np.savez(a.out, X=X, y=y, event=ev, features=np.array(FEATURES))
        print(json.dumps({"events": len(paths), "pairs": int(len(y)), "positive_frac": float(y.mean()),
                          "elapsed_s": round(time.time() - t0, 1)}))
    else:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        Z = [np.load(f) for f in sorted(glob.glob(a.data))]
        X = np.concatenate([z["X"] for z in Z])
        y = np.concatenate([z["y"] for z in Z])
        ev = np.concatenate([z["event"] for z in Z])
        rng = np.random.default_rng(a.seed)
        uev = np.unique(ev)
        hold = np.isin(ev, rng.choice(uev, size=max(1, len(uev) // 10), replace=False))
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                             l2_regularization=1.0, random_state=a.seed)
        clf.fit(X[~hold], y[~hold])
        clf.hnl_features = FEATURES
        info = {"pairs_train": int((~hold).sum()), "holdout_auc": float(roc_auc_score(y[hold],
                clf.predict_proba(X[hold])[:, 1])), "positive_frac": float(y.mean()), "features": FEATURES,
                "first_pass": f"mwslayer {FIRST_S0}", "elapsed_s": round(time.time() - t0, 1)}
        joblib.dump(clf, a.out)
        with open(a.out + ".json", "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
