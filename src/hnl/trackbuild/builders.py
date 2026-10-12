"""Track builders on a scored graph (brief §6C): connected components (the baseline), acorn-style junction removal,
and a walkthrough on the graph directed outward. All return a track label per hit (-1 = no track; tracks have >= 2
hits), the convention of hnl.graphs.ceiling.cc_labels and hnl.eval.matching.

Edges are given once per unordered pair (a, b, score); scores and thresholds are compared in float64 (as in
hnl.eval.evaluate_gnn). The walkthrough orients every edge from the hit with the smaller
R = sqrt(r^2 + z^2) to the larger (ties by hit index), which makes the graph a DAG whose topological order is the R
order. walkthrough_triplet adds the triplet (helix-consistency) condition of brief §6C to the walkthrough.
"""

from __future__ import annotations

import numpy as np

from hnl.graphs.ceiling import cc_labels


def cc(n: int, a: np.ndarray, b: np.ndarray, score: np.ndarray, cut: float) -> np.ndarray:
    keep = np.asarray(score, dtype=np.float64) > float(cut)
    return cc_labels(n, np.stack([a[keep], b[keep]]))


def orient(a: np.ndarray, b: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(src, dst) with R[src] < R[dst] (ties: smaller index first)."""
    fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    return np.where(fwd, a, b), np.where(fwd, b, a)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Labels of components with < 2 hits -> -1; others renumbered 0..K-1."""
    out = np.full(len(labels), -1, dtype=np.int64)
    ok = labels >= 0
    if not ok.any():
        return out
    u, inv, cnt = np.unique(labels[ok], return_inverse=True, return_counts=True)
    good = cnt[inv] >= 2
    idx = np.flatnonzero(ok)[good]
    _, out[idx] = np.unique(labels[idx], return_inverse=True)
    return out


def junction_removal(n: int, a: np.ndarray, b: np.ndarray, score: np.ndarray, R: np.ndarray, cut: float,
                     junction_cut: float, min_chain: int = 3) -> np.ndarray:
    """acorn CCandJunctionRemoval: components of the cut graph that are simple chains (every hit has at most one
    inward and one outward kept edge) and have >= min_chain hits are tracks; on the remaining hits, keep only edges
    above junction_cut that are both the best outward edge of their source and the best inward edge of their
    target, and run CC on those."""
    score = np.asarray(score, dtype=np.float64)
    cut, junction_cut = float(cut), float(junction_cut)
    src, dst = orient(a, b, R)
    keep = score > cut
    s, d = src[keep], dst[keep]
    lab = cc_labels(n, np.stack([s, d]))
    out_deg = np.bincount(s, minlength=n)
    in_deg = np.bincount(d, minlength=n)
    n_comp = lab.max() + 1 if (lab >= 0).any() else 0
    bad = np.zeros(max(n_comp, 1), dtype=bool)
    on = lab >= 0
    np.logical_or.at(bad, lab[on & (in_deg > 1)], True)
    np.logical_or.at(bad, lab[on & (out_deg > 1)], True)
    size = np.bincount(lab[on], minlength=max(n_comp, 1))
    bad |= size < min_chain
    chain = on & ~bad[np.maximum(lab, 0)]
    labels = np.where(chain, lab, -1)
    rest = ~chain
    j = (score > junction_cut) & rest[src] & rest[dst]
    js, jd, jsc = src[j], dst[j], score[j]
    if len(jsc):
        # best outward edge per source and best inward edge per target (ties: first in score order)
        order = np.argsort(-jsc, kind="stable")
        best_out = np.full(n, -1)
        best_in = np.full(n, -1)
        for e in order[::-1]:                 # descending order applied last wins -> highest score kept
            best_out[js[e]] = e
            best_in[jd[e]] = e
        mutual = np.array([best_out[js[e]] == e and best_in[jd[e]] == e for e in range(len(jsc))], dtype=bool)
        jl = cc_labels(n, np.stack([js[mutual], jd[mutual]]))
        add = rest & (jl >= 0)
        labels[add] = jl[add] + (labels.max() + 1 if (labels >= 0).any() else 0)
    return _relabel(labels)


def walkthrough(n: int, a: np.ndarray, b: np.ndarray, score: np.ndarray, R: np.ndarray, th_min: float,
                th_add: float) -> np.ndarray:
    """acorn-style walkthrough: hits in R order; from each unused hit, follow the best unused outward neighbour with
    score > th_min and also every other unused outward neighbour with score > th_add; the longest such path (ties:
    the larger summed score) becomes a track and its hits are used."""
    score = np.asarray(score, dtype=np.float64)
    th_min, th_add = float(th_min), float(th_add)
    src, dst = orient(a, b, R)
    m = score > th_min
    src, dst, sc = src[m], dst[m], score[m]
    order = np.lexsort((-sc, src))            # by source, then descending score
    src, dst, sc = src[order], dst[order], sc[order]
    start = np.searchsorted(src, np.arange(n + 1))
    used = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    topo = np.lexsort((np.arange(n), R))
    k = 0
    for v0 in topo:
        if used[v0] or start[v0] == start[v0 + 1]:
            continue
        # longest path from v0 over allowed edges among unused hits (DAG: memoised DFS in reverse topological order)
        best_len, best_sc, nxt = {}, {}, {}
        stack = [(v0, False)]
        while stack:
            v, done = stack.pop()
            if v in best_len:
                continue
            lo, hi = start[v], start[v + 1]
            cand = [i for i in range(lo, hi) if not used[dst[i]]]
            allowed = cand[:1] + [i for i in cand[1:] if sc[i] > th_add]
            if not done:
                stack.append((v, True))
                for i in allowed:
                    if dst[i] not in best_len:
                        stack.append((dst[i], False))
                continue
            bl, bs, bn = 1, 0.0, -1
            for i in allowed:
                u = dst[i]
                L, S = best_len[u] + 1, best_sc[u] + sc[i]
                if L > bl or (L == bl and S > bs):
                    bl, bs, bn = L, S, u
            best_len[v], best_sc[v], nxt[v] = bl, bs, bn
        if best_len[v0] < 2:
            continue
        v = v0
        while v != -1:
            used[v] = True
            labels[v] = k
            v = nxt[v]
        k += 1
    return labels


def cc_and_walk(n: int, a: np.ndarray, b: np.ndarray, score: np.ndarray, R: np.ndarray, cut: float, th_min: float,
                th_add: float) -> np.ndarray:
    """CC at `cut`; components that are simple chains are tracks; the others are rebuilt with the walkthrough on their
    own kept edges (score > cut, so th_min / th_add act only above the cut), as acorn CCandWalk."""
    score = np.asarray(score, dtype=np.float64)
    cut = float(cut)
    src, dst = orient(a, b, R)
    keep = score > cut
    s, d = src[keep], dst[keep]
    lab = cc_labels(n, np.stack([s, d]))
    out_deg = np.bincount(s, minlength=n)
    in_deg = np.bincount(d, minlength=n)
    on = lab >= 0
    n_comp = lab.max() + 1 if on.any() else 1
    branchy = np.zeros(n_comp, dtype=bool)
    np.logical_or.at(branchy, lab[on & ((in_deg > 1) | (out_deg > 1))], True)
    chain = on & ~branchy[np.maximum(lab, 0)]
    labels = np.where(chain, lab, -1)
    rest = on & ~chain
    e = keep & rest[a] & rest[b]              # kept edges of the branching components only
    w = walkthrough(n, a[e], b[e], score[e], R, th_min, th_add)
    add = rest & (w >= 0)
    labels[add] = w[add] + (labels.max() + 1 if (labels >= 0).any() else 0)
    return _relabel(labels)


def walkthrough_triplet(n: int, a: np.ndarray, b: np.ndarray, score: np.ndarray, R: np.ndarray, xyz: np.ndarray,
                        th_min: float, th_add: float, max_angle: float = 0.3, max_dcot: float = 0.3,
                        min_seg: float = 5.0) -> np.ndarray:
    """Triplet-consistent walkthrough (brief §6C): the walkthrough above, except that a path may continue with an edge
    only if it is helix-consistent with the path's last long edge (the reference): their 3D angle is below max_angle
    [rad] and their s-z slopes (dz / transverse length) differ by less than max_dcot. An edge shorter than min_seg
    [mm] (e.g. between the two hits of a particle on overlapping modules of one layer) is not checked itself and does
    not replace the reference, so the edges on both sides of a short hop are still compared with each other (a path
    cannot switch particles through a short fake edge). Among the consistent continuations the best-scored one is
    always allowed and the others need score > th_add. Longest path (ties: larger summed score) by dynamic
    programming over states (last edge, reference edge), from each unused hit in R order.
    xyz: (n, 3) hit positions [mm]. Only geometry and scores are used."""
    score = np.asarray(score, dtype=np.float64)
    th_min, th_add = float(th_min), float(th_add)
    cos_max = np.cos(float(max_angle))
    src, dst = orient(a, b, R)
    m = score > th_min
    src, dst, sc = src[m], dst[m], score[m]
    order = np.lexsort((-sc, src))            # by source, then descending score
    src, dst, sc = src[order], dst[order], sc[order]
    start = np.searchsorted(src, np.arange(n + 1))
    seg = np.asarray(xyz, dtype=np.float64)[dst] - np.asarray(xyz, dtype=np.float64)[src]
    seg_len = np.linalg.norm(seg, axis=1)
    seg_lt = np.hypot(seg[:, 0], seg[:, 1])
    unit = seg / np.maximum(seg_len, 1e-9)[:, None]
    cot = seg[:, 2] / np.maximum(seg_lt, 1e-9)
    short = seg_len < float(min_seg)
    flat = seg_lt < float(min_seg)            # (nearly) along z: slope undefined

    def consistent(ref: int, f: int) -> bool:
        if ref < 0 or short[f]:
            return True
        if float(unit[ref] @ unit[f]) < cos_max:
            return False
        return bool(flat[ref] or flat[f] or abs(cot[f] - cot[ref]) <= max_dcot)

    def next_ref(ref: int, f: int) -> int:
        return ref if short[f] else f

    used = np.zeros(n, dtype=bool)

    def allowed(v: int, ref: int) -> list[int]:
        cand = [i for i in range(start[v], start[v + 1]) if not used[dst[i]] and consistent(ref, i)]
        return cand[:1] + [i for i in cand[1:] if sc[i] > th_add]

    labels = np.full(n, -1, dtype=np.int64)
    topo = np.lexsort((np.arange(n), R))
    k = 0
    for v0 in topo:
        if used[v0] or start[v0] == start[v0 + 1]:
            continue
        first = [(e, next_ref(-1, e)) for e in allowed(v0, -1)]
        if not first:
            continue
        # state (edge, reference) -> (hits on the best path from it, score sum, next state or None)
        best: dict[tuple[int, int], tuple[int, float, tuple[int, int] | None]] = {}
        stack = [(st, False) for st in first]
        while stack:
            st, done = stack.pop()
            if st in best:
                continue
            e, ref = st
            cont = [(f, next_ref(ref, f)) for f in allowed(dst[e], ref)]
            if not done:
                stack.append((st, True))
                stack.extend((c, False) for c in cont if c not in best)
                continue
            bh, bs, bn = 2, sc[e], None
            for c in cont:
                h, s_, _ = best[c]
                if h + 1 > bh or (h + 1 == bh and sc[e] + s_ > bs):
                    bh, bs, bn = h + 1, sc[e] + s_, c
            best[st] = (bh, bs, bn)
        st = max(first, key=lambda c: (best[c][0], best[c][1]))
        used[v0] = True
        labels[v0] = k
        while st is not None:
            used[dst[st[0]]] = True
            labels[dst[st[0]]] = k
            st = best[st][2]
        k += 1
    return labels
