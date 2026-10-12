"""Learned signed-graph partitioning for track building: the Mutex Watershed (Wolf et al. 2018) on learned
same-particle log-odds of hit pairs (Stage 1 C; the design panel's top proposal, DECISIONS.md).

Every candidate pair of hits gets a signed weight w = logit(p) + beta: positive = attraction, negative = repulsion.
Connected components keep only score > cut and ignore all negative evidence; the Mutex Watershed also uses confident
"different particle" pairs to block weak links between two particles, which is the measured failure of CC (merges,
impure tracks) and of the walkthrough (paths switching onto a neighbouring particle).

The GNN's scores are not same-particle probabilities: its truth is the CHAIN edge (consecutive hits of one particle),
so a same-particle skip edge (i, k) over a hit j gets a low score and, used as a repulsion, would split its own track.
The learned pair head therefore re-calibrates every candidate pair into p(same particle) from production-point-
invariant features. Candidate pairs: every graph edge (direct) and every pair (i, k) of hits sharing a graph neighbour
j through two edges with score >= SCORE_FLOOR (lifted edges: continuations i - j - k and branches at a junction).
Features (NaN where a pair has no such evidence; the gradient-boosted head handles missing values natively): the
direct edge's GNN logit; the strongest 2-hop path's two logits and the number of 2-hop paths; the helix consistency
of that triplet (|kappa|, |turning angle|, |dcot|, 3D angle, segment length ratio); the pair's transverse and
longitudinal separation and |cot theta| of the segment i -> k; the number of confident edges (score > 0.5) at i and at
k (local competition); the subsystems of i, k (and j). No feature uses a hit's absolute position or the pointing to
the beam line. The partition has no threshold; beta is the only working point (builder rule on val).

Without a pair head (family "mws"), the signed weights are the raw GNN logits of the graph edges (the naive version).

Algorithm: all signed edges in descending |w|; union-find clusters with mutex sets (merged small into large); an
attractive edge merges its two clusters unless they carry a mutex, a repulsive edge adds a mutex. Tracks = clusters
with >= 2 hits (label -1 otherwise, the convention of hnl.trackbuild.builders).

    python -m hnl.trackbuild.mutex_ws data --graphs DIR --out SHARD.npz [--row-frac 0.25]  (scored TRAIN graphs)
    python -m hnl.trackbuild.mutex_ws train --data 'DIR/*.npz' --out MODEL.joblib
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from hnl.trackbuild.merge_fragments import _volume_from_radius

PAIR_FEATURES = ["direct_logit", "path_min", "path_max", "n_paths", "kappa", "turn", "dcot", "angle3d", "len_ratio",
                 "d_t", "d_z", "seg_cot", "deg_i", "deg_k", "sub_i", "sub_k", "sub_j", "same_layer", "same_volume"]
SCORE_FLOOR = 0.01                       # edges below this score do not seed lifted pairs


def _logit(s):
    s = np.clip(np.asarray(s, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(s) - np.log1p(-s)


def hit_layers(eids) -> dict:
    """volume * 1000 + layer of every hit, by event id (graph node order = row order of the hits table)."""
    from hnl.data.convert import read_table
    from hnl.eval.particles import CODE_SAMPLE, DERIVED
    eids = np.asarray(eids, dtype=np.int64)
    codes = eids // 100_000_000
    out = {}
    for c in np.unique(codes):
        ev = (eids[codes == c] % 100_000_000).tolist()
        h = read_table(os.path.join(DERIVED, "events", "v1"), "hits", CODE_SAMPLE[int(c)], events=ev,
                       columns=["event", "row", "volume", "layer"])
        for e, g in h.groupby("event"):
            g = g.sort_values("row")
            assert (g.row.to_numpy() == np.arange(len(g))).all(), e
            out[int(c) * 100_000_000 + int(e)] = g.volume.to_numpy().astype(np.int64) * 1000 + g.layer.to_numpy()
    return out


def mws_layer(n, a, b, sc, L, s0: float) -> np.ndarray:
    """Mutex Watershed on the graph edges with w = logit(s) - logit(s0), where only hits on the same detector layer
    repel (a particle crosses a layer once, up to module overlaps): every other edge below s0 is dropped, so the
    attractive set is CC's at s0 and a merge is refused only when it would put two hits of one layer joined by a
    low-score edge into one track (the design critique's pivot, DECISIONS.md)."""
    w = _logit(sc) - _logit(s0)
    keep = (w > 0) | (L[a] == L[b])
    return mutex_watershed(n, a[keep], b[keep], w[keep])


def mutex_watershed(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Labels of the Mutex Watershed partition of n nodes for signed edges (u, v, w)."""
    order = np.argsort(-np.abs(w), kind="stable")
    parent = np.arange(n)
    mutex: dict[int, set[int]] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in order:
        a, b = find(int(u[e])), find(int(v[e]))
        if a == b:
            continue
        ma, mb = mutex.get(a), mutex.get(b)
        if w[e] > 0:
            if ma is not None and b in ma:
                continue
            # union by mutex-set size: b into a
            if (len(ma) if ma else 0) < (len(mb) if mb else 0):
                a, b, ma, mb = b, a, mb, ma
            parent[b] = a
            if mb:
                for c in mb:
                    mc = mutex[c]
                    mc.discard(b)
                    mc.add(a)
                if ma is None:
                    mutex[a] = ma = set()
                ma |= mb
                del mutex[b]
        else:
            mutex.setdefault(a, set()).add(b)
            mutex.setdefault(b, set()).add(a)
    roots = np.array([find(i) for i in range(n)])
    _, lab, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    return np.where(cnt[lab] >= 2, lab, -1).astype(np.int64)


def two_hop(n, a, b, sc):
    """Pairs (i, k), i < k, of hits sharing a neighbour j through two edges with score >= SCORE_FLOOR. Returns
    (ik, vals) with vals = (lower score, higher score, j) of the strongest path (highest lower score) and the number
    of 2-hop paths."""
    keep = sc >= SCORE_FLOOR
    a, b, sc = a[keep], b[keep], sc[keep]
    if len(a) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 4))
    src = np.r_[a, b]
    dst = np.r_[b, a]
    s2 = np.r_[sc, sc]
    order = np.argsort(src, kind="stable")
    src, dst, s2 = src[order], dst[order], s2[order]
    start = np.searchsorted(src, np.arange(n + 1))
    best = {}
    for j in np.flatnonzero(np.diff(start) >= 2):
        lo, hi = start[j], start[j + 1]
        nb, ns = dst[lo:hi].tolist(), s2[lo:hi].tolist()
        for p in range(len(nb)):
            for q in range(p + 1, len(nb)):
                i, k = nb[p], nb[q]
                if i == k:
                    continue
                key = (i, k) if i < k else (k, i)
                smin, smax = (ns[p], ns[q]) if ns[p] < ns[q] else (ns[q], ns[p])
                cur = best.get(key)
                if cur is None:
                    best[key] = [smin, smax, j, 1]
                else:
                    cur[3] += 1
                    if smin > cur[0]:
                        cur[0], cur[1], cur[2] = smin, smax, j
    if not best:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 4))
    return np.array(list(best.keys()), dtype=np.int64), np.array(list(best.values()), dtype=np.float64)


def pair_candidates(n, a, b, sc, r, phi, z, L=None):
    """Candidate pairs (direct graph edges and 2-hop pairs), i < k, with the features PAIR_FEATURES. a < b and one
    entry per unordered edge are expected (hnl.eval.trackbuild_val.load_scored). Returns (ik, X)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    sc = np.asarray(sc, dtype=np.float64)
    ik2, vals = two_hop(n, a, b, sc)
    code_d = a * n + b
    code_2 = ik2[:, 0] * n + ik2[:, 1]
    codes = np.union1d(code_d, code_2)
    m = len(codes)
    X = np.full((m, len(PAIR_FEATURES)), np.nan)
    col = {f: k for k, f in enumerate(PAIR_FEATURES)}
    i, k = codes // n, codes % n
    pos_d = np.searchsorted(codes, code_d)
    X[pos_d, col["direct_logit"]] = _logit(sc)
    X[:, col["n_paths"]] = 0.0
    x, y = r * np.cos(phi), r * np.sin(phi)
    xyz = np.stack([x, y, z], 1)
    vol = _volume_from_radius(r, z)
    sub = np.where(np.isin(vol, (7, 8, 9)), 0, np.where(np.isin(vol, (12, 13, 14)), 1, 2)).astype(np.float64)
    if len(ik2):
        pos_2 = np.searchsorted(codes, code_2)
        j = vals[:, 2].astype(np.int64)
        i2, k2 = ik2[:, 0], ik2[:, 1]
        av = xyz[j] - xyz[i2]
        bv = xyz[k2] - xyz[j]
        cv = xyz[k2] - xyz[i2]
        la = np.maximum(np.hypot(av[:, 0], av[:, 1]), 1e-6)
        lb = np.maximum(np.hypot(bv[:, 0], bv[:, 1]), 1e-6)
        lc = np.maximum(np.hypot(cv[:, 0], cv[:, 1]), 1e-6)
        cross = av[:, 0] * bv[:, 1] - av[:, 1] * bv[:, 0]
        dot = av[:, 0] * bv[:, 0] + av[:, 1] * bv[:, 1]
        cos3 = (av * bv).sum(1) / np.maximum(np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1), 1e-6)
        X[pos_2, col["path_min"]] = _logit(vals[:, 0])
        X[pos_2, col["path_max"]] = _logit(vals[:, 1])
        X[pos_2, col["n_paths"]] = vals[:, 3]
        X[pos_2, col["kappa"]] = 2e3 * np.abs(cross) / (la * lb * lc)
        X[pos_2, col["turn"]] = np.abs(np.arctan2(cross, dot))
        X[pos_2, col["dcot"]] = np.abs(bv[:, 2] / lb - av[:, 2] / la)
        X[pos_2, col["angle3d"]] = np.arccos(np.clip(cos3, -1, 1))
        X[pos_2, col["len_ratio"]] = np.abs(np.log(lb / la))
        X[pos_2, col["sub_j"]] = sub[j]
    d = xyz[k] - xyz[i]
    dt = np.hypot(d[:, 0], d[:, 1])
    X[:, col["d_t"]] = dt
    X[:, col["d_z"]] = np.abs(d[:, 2])
    X[:, col["seg_cot"]] = np.abs(d[:, 2]) / np.maximum(dt, 1e-3)
    strong = sc > 0.5
    deg = np.bincount(a[strong], minlength=n) + np.bincount(b[strong], minlength=n)
    X[:, col["deg_i"]] = deg[i]
    X[:, col["deg_k"]] = deg[k]
    X[:, col["sub_i"]] = sub[i]
    X[:, col["sub_k"]] = sub[k]
    if L is not None:                        # detector layer ids (volume * 1000 + layer), hnl.trackbuild.mutex_ws.hit_layers
        X[:, col["same_layer"]] = L[i] == L[k]
        X[:, col["same_volume"]] = L[i] // 1000 == L[k] // 1000
    return np.stack([i, k], 1), X


def mws_two_pass(n, a, b, sc, L, s1: float, s2: float, min_keep: int = 4) -> np.ndarray:
    """Two passes of the same-layer Mutex Watershed: tracks of >= min_keep hits from s0 = s1 are kept; the other hits
    (unassigned or on shorter tracks) are rebuilt from the graph edges among them alone at the lower s0 = s2, which
    collects short (displaced) tracks whose edges score between s2 and s1 without letting them merge into the kept
    tracks."""
    a, b, sc, L = np.asarray(a), np.asarray(b), np.asarray(sc), np.asarray(L)
    lab = mws_layer(n, a, b, sc, L, s1)
    trk, cnt = np.unique(lab[lab >= 0], return_counts=True)
    keep = np.isin(lab, trk[cnt >= min_keep])
    out = np.where(keep, lab, -1)
    e = ~keep[a] & ~keep[b]
    if e.any():
        lab2 = mws_layer(n, a[e], b[e], sc[e], L, s2)
        m = ~keep & (lab2 >= 0)
        out[m] = lab2[m] + (out.max() + 1)
    _, inv = np.unique(out, return_inverse=True)
    return np.where(out >= 0, inv - (1 if (out < 0).any() else 0), -1).astype(np.int64)


class MWS:
    """Signed weights from the learned pair head (or the raw GNN logits of the graph edges); one partition per beta."""

    def __init__(self, path: str | None = None):
        self.model = None
        if path:
            import joblib
            self.model = joblib.load(path)

    def prepare(self, n, a, b, sc, r, phi, z, L=None):
        if self.model is None:
            return n, a, b, _logit(sc)
        ik, X = pair_candidates(n, a, b, sc, r, phi, z, L)
        p = self.model.predict_proba(X)[:, 1] if len(ik) else np.zeros(0)
        return n, ik[:, 0], ik[:, 1], _logit(p)

    @staticmethod
    def labels(state, beta: float) -> np.ndarray:
        n, u, v, w = state
        return mutex_watershed(n, u, v, w + float(beta))

    @staticmethod
    def labels_rep(state, a, b, sc, s0: float, p_rep: float) -> np.ndarray:
        """Attraction from the GNN (graph edges above s0, w = logit(s) - logit(s0)); repulsion only from the pair
        head's confident 'different particle' pairs (p < p_rep, w = logit(p)); nothing else."""
        n, u, v, w = state
        wa = _logit(sc) - _logit(s0)
        att = wa > 0
        rep = w < _logit(p_rep)
        return mutex_watershed(n, np.r_[a[att], u[rep]], np.r_[b[att], v[rep]], np.r_[wa[att], w[rep]])


def _event_rows(path, row_frac):
    from hnl.eval.trackbuild_val import load_scored
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    L = hit_layers([int(eid)])[int(eid)]
    assert len(L) == len(pid)
    ik, X = pair_candidates(len(pid), a, b, sc, r, phi, z, L)
    y = (pid[ik[:, 0]] == pid[ik[:, 1]]) & (pid[ik[:, 0]] != 0)
    if row_frac < 1:                         # uniform row sampling (keeps the class prior: calibrated p)
        sel = np.random.default_rng(int(eid)).random(len(y)) < row_frac
        X, y = X[sel], y[sel]
    return X.astype(np.float32), y, np.full(len(y), int(eid), dtype=np.int64)


def _chunk(args):
    paths, row_frac = args
    res = [_event_rows(p, row_frac) for p in paths]
    return [np.concatenate([x[i] for x in res]) for i in range(3)]


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--graphs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-events", type=int, default=None)
    d.add_argument("--row-frac", type=float, default=0.25)
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-rows", type=int, default=12_000_000)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "data":
        paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
        if a.max_events is not None and a.max_events < len(paths):    # a seeded random subset (all samples)
            pick = np.random.default_rng(0).choice(len(paths), a.max_events, replace=False)
            paths = [paths[k] for k in np.sort(pick)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            parts = list(ex.map(_chunk, [(paths[k:k + 20], a.row_frac) for k in range(0, len(paths), 20)]))
        X, y, ev = (np.concatenate([p[i] for p in parts]) for i in range(3))
        np.savez(a.out, X=X, y=y, event=ev, features=np.array(PAIR_FEATURES))
        print(json.dumps({"events": len(paths), "pairs": int(len(y)), "positive_frac": float(y.mean()),
                          "row_frac": a.row_frac, "elapsed_s": round(time.time() - t0, 1)}))
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
        tr = np.flatnonzero(~hold)
        if len(tr) > a.max_rows:
            tr = np.sort(rng.choice(tr, a.max_rows, replace=False))
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=127,
                                             l2_regularization=1.0, random_state=a.seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[hold])[:, 1]
        has_d = ~np.isnan(X[hold][:, PAIR_FEATURES.index("direct_logit")])
        info = {"pairs_train": int(len(tr)), "pairs_holdout": int(hold.sum()),
                "holdout_auc": float(roc_auc_score(y[hold], p)),
                "holdout_auc_direct": float(roc_auc_score(y[hold][has_d], p[has_d])),
                "holdout_auc_lifted_only": float(roc_auc_score(y[hold][~has_d], p[~has_d])),
                "positive_frac": float(y.mean()), "features": PAIR_FEATURES, "elapsed_s": round(time.time() - t0, 1)}
        joblib.dump(clf, a.out)
        with open(a.out + ".json", "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
