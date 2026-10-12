"""Helix fragment merging without refitting every candidate (docs/trackml_plan.md section 6.7, gate G3), same labels.

hnl.trackbuild.merge_fragments.merge_fragments hands every candidate pair to _merge, which refits all their unions in
one batch: about 3e6 refits per TrackML event in the sequential pass (> 30 min, ~24 GB). merge_sequential (and, through
merge_exact, hnl.trackml.trackbuild.merge_fragments_fast's any-order pass) returns the same labels and calls the
unchanged _merge (imported read-only), handing it only the merges and a few pinning pairs.

1. Candidates C: the original loops (for a in trk: for b in trk) keep (a, b), a != b, when b's innermost hit (in
   R = hypot(r, z)) lies in [-20, max_gap_r] of a's outermost hit and their phi differ by <= max_dphi. Here the b of each
   a come from a window of the R-sorted inner ends wider than the test by 1e-6 mm, then the original's floating-point
   tests are applied unchanged; endpoints by the same np.argsort; pairs in the loops' order (tested: identical).

2. Necessary conditions (pre-filter of the sequential pass). _merge accepts a pair (finite q) only if the robust helix
   fit of the union has rms <= max_rms, zrms <= max_zrms, n_out <= max_out and span >= min_span_mm. The fit's
   transverse residual e_i is the distance of hit i from the fitted circle (or line) (hnl.truth.helix.perigee_residuals),
   rms^2 = sum_i e_i^2 / n, n_out = #{i : |e_i| > 3 st_i}, st the per-volume Huber scale of fit_tracks. For any 4 hits
   Q let lam(Q) = min over circles and lines of max_{i in Q} dist(p_i, .) (half the width of the minimum-width annulus
   or slab containing Q). Whatever circle the fit ends on:
     (S) span is computed exactly as _merge computes it: span < min_span_mm  =>  rejected.
     (O) if lam(Q) > max_{i in Q} 3 st_i, every circle leaves at least one hit of Q outside its tolerance, so the hits
         beyond tolerance hit every such "bad" quadruple: n_out >= tau, a lower bound on the smallest hitting set (1 if
         one is bad; 2 if the bad ones share no hit; the number of pairwise disjoint bad ones). tau > max_out => rejected.
     (R) for 4 hits and any circle, sum_{i in Q} e_i^2 >= (max e - min e)^2 / 2 >= 2 lam(Q)^2; over pairwise disjoint
         quadruples sum_k 2 lam(Q_k)^2 <= n rms^2. lb = that sum > n max_rms^2  =>  rejected.
   No test uses max_zrms, so the filter is exact for every value of max_rms, max_zrms, max_out, min_span_mm, max_dphi
   and max_gap_r; with a cut set (BinnedCuts, per |eta| bin of the union) a pair is dropped only if the cuts of every
   bin reject it (its bin is known only after the fit). Quadruples per pair (bounds): a hit of one fragment with a
   spread triple of the other, both ways, also on the 6 hits next to the junction of a long fragment; two hits of each
   next to the junction; every quadruple of a union of <= 7 hits (each family pairwise disjoint). lam of 4 points is
   min(delta_12|34, delta_13|24, delta_14|23, W/2): each 2-2 pattern delta is the half-width of the annulus centred on
   the intersection of the two pairs' perpendicular bisectors, W the minimum width of a slab containing the points.
   Every term is the half-width of an annulus (slab) containing Q, so lam <= each term; at a finite minimiser c of
   w(c) = max_i |p_i - c| - min_i |p_i - c| the gradients u_i = (c - p_i)/|c - p_i| of the farthest and of the nearest
   points must have intersecting convex hulls (else a separating direction lowers w), which for distinct unit vectors
   happens only for two farthest and two nearest points whose chords cross (degenerate cases by continuity); minimisers
   at infinity are slabs. Floating point: coordinate differences only; pattern terms with chords within MIN_SIN of
   parallel and quadruples with points closer than MIN_CHORD_MM give 0; MARGIN_MM is subtracted. Both only weaken it.

3. The refit is a batch, and a pair's fit values depend on the other pairs (only in the last bits):
     - hnl.truth.helix.fit_perigee: per track, a converged track is frozen, so the result does not depend on when the
       loop stops. Its lstsq fallback (on every track of the batch) needs np.linalg.solve to raise LinAlgError, i.e. an
       exactly zero LU pivot; the matrices are J^T W J + (lam diag + 1e-300) with lam >= 1e-9, positive definite, and
       with finite hits everything stays finite (a step is accepted only if the Huber objective does not grow), so the
       pivots are of order lam * diag, never zero (0 calls in every reference run).
     - hnl.truth.helix.fit_line_sz iterates all tracks until EVERY track passes its stopping test (or 30 iterations), and
       a track keeps moving after its own test passed (Huber weights of z outliers): zrms depends on the batch's
       iteration count K.
     - every row sum (np.sum over the hits axis) runs over the batch's padded width H, and numpy's pairwise summation
       groups the terms by blocks of 8 (recursively above 128) that depend on H: the last bits depend on H.
   Given H's summation signature for its row (_sum_sig, same_sums) and K, a pair's fit values are a function of its own
   hits. prefit therefore fits any set of candidates bit for bit as _merge's batch of all of C would: in batches of one
   size class whose padded width sums like C's width (a padding pair of C; h, C's longest union, where needed), each
   with the witnesses W: pairs of C that have not yet passed fit_line_sz's stopping test at iterations 0..28 (then every
   batch runs 30 iterations, as C, which contains them) found in a sample of the rejected pairs, or, when no sample
   shows K(C) = 30, from replicating all of C (_line_sz_done: fit_tracks and fit_line_sz's loop operation for
   operation, recording the stopping test). Narrow and small batches are cheap (fit_perigee stops when a batch's own
   slowest track has converged) and independent (n_proc processes).

4. merge_exact: q and acceptance exactly as _merge computes them from those values (per-candidate cut arrays for a cut
   set: the union's |eta| = asinh(|cot theta|) of this fit), _merge's acceptance loop (greedy, statement for statement)
   gives the merges S, and the unchanged _merge on S + h + W refits them with the same values, takes the same decisions
   and returns the labels (info["consistent"] compares them with greedy's; never different; else the unfiltered _merge
   is run). Exact ties of finite q would leave the order to np.argsort (not stable): structurally they need (a, b) and
   (b, a), both in C only if both fragments lie within 40 mm in R (span < min_span_mm = 50); info["finite_q_ties"]
   counts them (0 in every run). merge_sequential's method "pinned" (an earlier, slower variant) hands _merge every
   kept pair + h + W instead.
Verified (hnl.trackml.g3check, tests/test_seqmerge.py): candidate lists, the bounds on every candidate of full events,
the pre-fit values bit for bit, and the labels against the unfiltered passes on full TrackML events and HNL val events.
"""

from __future__ import annotations

import functools
import time

import numpy as np
from numba import njit

from hnl.trackbuild.merge_fragments import _merge, _volume_from_radius
from hnl.truth import helix as hx
from hnl.vertex import pairing as PA

MARGIN_MM = 1e-4         # subtracted from every computed annulus half-width (rounding allowance, >> the errors)
MIN_SIN = 1e-6           # 2-2 patterns whose chords are this close to parallel contribute 0 (no safe value)
MIN_CHORD_MM = 1e-6      # quadruples with two points this close contribute 0
LINE_SZ_ITER = 30        # hnl.truth.helix.fit_line_sz default n_iter (as called by hnl.vertex.pairing.fit_tracks)


# ------------------------------------------------------------------------------------------------ 0. cut sets
class BinnedCuts:
    """Merge cuts per |eta| bin of the union (the cut set C of the TrackML grid; plan section 6.6, calibration_v1.md).
    The union's |eta| is asinh(|cot theta|), cot theta being the s-z slope of the union's helix fit ('cot' of
    hnl.vertex.pairing.fit_tracks: the very fit _merge makes, z = z0 + cot * s with s the transverse arc length, so
    cot = pz / pT = sinh(eta)). Bin k holds edges[k] <= |eta| < edges[k + 1]; |eta| >= edges[-1] and a NaN slope fall
    in the last bin. _merge receives the cuts as arrays aligned with its candidates; with one bin (or equal bins) they
    are the scalars of the H set and _merge's comparisons are unchanged."""

    def __init__(self, edges, max_rms, max_zrms, max_out, name: str = ""):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.max_rms = np.asarray(max_rms, dtype=np.float64)
        self.max_zrms = np.asarray(max_zrms, dtype=np.float64)
        self.max_out = np.asarray(max_out, dtype=np.float64)
        self.name = name
        nb = len(self.max_rms)
        if not (len(self.edges) == nb + 1 and len(self.max_zrms) == nb == len(self.max_out) and self.edges[0] == 0
                and (np.diff(self.edges) > 0).all()):
            raise ValueError("BinnedCuts: edges must start at 0, increase, and have one more entry than the cuts")

    @classmethod
    def uniform(cls, max_rms: float = 1.0, max_zrms: float = 5.0, max_out: float = 1, name: str = "H"):
        return cls([0.0, np.inf], [max_rms], [max_zrms], [max_out], name)

    @classmethod
    def from_tsv(cls, path: str, rms: str = "max_rms_widen_only", zrms: str = "max_zrms_widen_only",
                 out: str = "diag_max_out_99", name: str = "C"):
        """The columns of configs/trackml/merge_cuts_v1.tsv (abs_eta = 'lo-hi', contiguous bins)."""
        import pandas as pd
        t = pd.read_csv(path, sep="\t", dtype={"abs_eta": str})
        lo_hi = np.array([[float(x) for x in s.split("-")] for s in t.abs_eta])
        if not np.allclose(lo_hi[1:, 0], lo_hi[:-1, 1]):
            raise ValueError(f"{path}: |eta| bins are not contiguous")
        return cls(np.r_[lo_hi[:, 0], lo_hi[-1, 1]], t[rms], t[zrms], t[out], name)

    @property
    def scalar(self) -> bool:
        return bool((self.max_rms == self.max_rms[0]).all() and (self.max_zrms == self.max_zrms[0]).all()
                    and (self.max_out == self.max_out[0]).all())

    def bin(self, cot) -> np.ndarray:
        a = np.arcsinh(np.abs(np.asarray(cot, dtype=np.float64)))
        k = np.searchsorted(self.edges, a, side="right") - 1
        return np.where(np.isnan(a), len(self.max_rms) - 1, np.clip(k, 0, len(self.max_rms) - 1))

    def for_cot(self, cot):
        """(max_rms, max_zrms, max_out) per candidate from its fitted slope; scalars for a scalar set."""
        if self.scalar:
            return float(self.max_rms[0]), float(self.max_zrms[0]), float(self.max_out[0])
        k = self.bin(cot)
        return self.max_rms[k], self.max_zrms[k], self.max_out[k]

    def describe(self) -> dict:
        return {"name": self.name, "edges": self.edges.tolist(), "max_rms": self.max_rms.tolist(),
                "max_zrms": self.max_zrms.tolist(), "max_out": self.max_out.tolist()}


# ------------------------------------------------------------------------------------------------ 1. candidates
def endpoints(labels: np.ndarray, trk: np.ndarray, R: np.ndarray):
    """Per fragment of trk: its hits ordered as merge_fragments orders them (o = idx[np.argsort(R[idx])] with
    idx = np.flatnonzero(labels == t)), as one flat array plus start offsets and lengths."""
    ok = np.flatnonzero(labels >= 0)
    o = ok[np.argsort(labels[ok], kind="stable")]
    lab = labels[o]
    starts = np.searchsorted(lab, trk, side="left")
    ends = np.searchsorted(lab, trk, side="right")
    flat = np.empty(int((ends - starts).sum()), dtype=np.int64)
    fs = np.empty(len(trk), dtype=np.int64)
    pos = 0
    for k, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        idx = o[s:e]                                  # = np.flatnonzero(labels == trk[k]) (increasing)
        flat[pos:pos + e - s] = idx[np.argsort(R[idx])]
        fs[k] = pos
        pos += e - s
    return flat, fs, (ends - starts).astype(np.int64)


def sequential_candidates(R_in, R_out, phi_in, phi_out, max_dphi: float, max_gap_r: float,
                          max_pairs: int = 4_000_000):
    """(ia, ib): indices into trk of merge_fragments' sequential candidates, in its loops' order."""
    T = len(R_in)
    order = np.argsort(R_in, kind="stable")
    Rs = R_in[order]
    slack = 1e-6
    lo = np.searchsorted(Rs, R_out - 20.0 - slack, side="left")
    if np.isnan(max_gap_r) or max_gap_r == np.inf:
        hi = np.full(T, T)
    else:
        hi = np.searchsorted(Rs, R_out + max_gap_r + slack, side="right")
    hi = np.maximum(hi, lo)
    cnt = hi - lo
    A, B = [], []
    i0 = 0
    csum = np.cumsum(cnt)
    while i0 < T:
        base = csum[i0 - 1] if i0 else 0
        i1 = int(np.searchsorted(csum, base + max_pairs, side="right"))
        i1 = min(max(i1, i0 + 1), T)
        ca = cnt[i0:i1]
        ia = np.repeat(np.arange(i0, i1), ca)
        off = np.arange(int(ca.sum())) - np.repeat(np.cumsum(ca) - ca, ca)
        ib = order[np.repeat(lo[i0:i1], ca) + off]
        # the original's tests, same floating-point operations
        dR = R_in[ib] - R_out[ia]
        ok = ~((dR < -20.0) | (dR > max_gap_r))
        dphi = np.abs(np.remainder(phi_in[ib] - phi_out[ia] + np.pi, 2 * np.pi) - np.pi)
        ok &= ~(dphi > max_dphi)
        ok &= ia != ib
        A.append(ia[ok])
        B.append(ib[ok])
        i0 = i1
    ia = np.concatenate(A) if A else np.zeros(0, dtype=np.int64)
    ib = np.concatenate(B) if B else np.zeros(0, dtype=np.int64)
    o = np.lexsort((ib, ia))
    return ia[o], ib[o]


# ------------------------------------------------------------------------------------------------ 2. pre-filter
@njit(cache=True)
def _pattern(ax, ay, bx, by, cx, cy, dx, dy):
    """Half-width of the annulus centred where the perpendicular bisectors of (a, b) and (c, d) meet, so that a, b lie
    on one circle and c, d on a concentric one. 0 when the chords are within MIN_SIN of parallel (no safe value)."""
    e1x = bx - ax
    e1y = by - ay
    e2x = dx - cx
    e2y = dy - cy
    L1 = np.sqrt(e1x * e1x + e1y * e1y)
    L2 = np.sqrt(e2x * e2x + e2y * e2y)
    u1x = e1x / L1
    u1y = e1y / L1
    u2x = e2x / L2
    u2y = e2y / L2
    if u1x * u2x + u1y * u2y < 0.0:
        u2x = -u2x
        u2y = -u2y
    D = u1x * u2y - u1y * u2x                  # sin of the angle between the chords
    if abs(D) < MIN_SIN:
        return 0.0
    # centre c = m1 + t n1 = m2 + s n2 (n = u rotated by +90 deg); d = m2 - m1
    ddx = 0.5 * ((cx - ax) + (dx - bx))
    ddy = 0.5 * ((cy - ay) + (dy - by))
    t = (ddx * u2x + ddy * u2y) / D
    s = (ddx * u1x + ddy * u1y) / D
    tms = (ddx * (u2x - u1x) + ddy * (u2y - u1y)) / D     # t - s without cancellation
    h1 = 0.5 * L1
    h2 = 0.5 * L2
    rho1 = np.sqrt(h1 * h1 + t * t)
    rho2 = np.sqrt(h2 * h2 + s * s)
    return 0.5 * abs(((h1 - h2) * (h1 + h2) + tms * (t + s)) / (rho1 + rho2))


@njit(cache=True)
def lam4_raw(x0, y0, x1, y1, x2, y2, x3, y3):
    """min over circles and lines of the largest distance of the 4 points (no margin; 0 for near-coincident points)."""
    xs = (x0, x1, x2, x3)
    ys = (y0, y1, y2, y3)
    W = np.inf
    for i in range(3):
        for j in range(i + 1, 4):
            ex = xs[j] - xs[i]
            ey = ys[j] - ys[i]
            L = np.sqrt(ex * ex + ey * ey)
            if not L >= MIN_CHORD_MM:
                return 0.0
            nx = -ey / L
            ny = ex / L
            lo = 0.0
            hi = 0.0
            for k in range(4):
                if k != i and k != j:
                    pr = (xs[k] - xs[i]) * nx + (ys[k] - ys[i]) * ny
                    lo = min(lo, pr)
                    hi = max(hi, pr)
            W = min(W, hi - lo)
    lam = 0.5 * W
    lam = min(lam, _pattern(x0, y0, x1, y1, x2, y2, x3, y3))
    lam = min(lam, _pattern(x0, y0, x2, y2, x1, y1, x3, y3))
    lam = min(lam, _pattern(x0, y0, x3, y3, x1, y1, x2, y2))
    return lam


@njit(cache=True)
def lam4(x0, y0, x1, y1, x2, y2, x3, y3):
    """lam4_raw minus MARGIN_MM (>= 0): a value that never exceeds the true minimax distance."""
    v = lam4_raw(x0, y0, x1, y1, x2, y2, x3, y3) - MARGIN_MM
    return v if v > 0.0 else 0.0


@njit(cache=True)
def _g(l, na, sa, sb):
    return sa + l if l < na else sb + l - na


@njit(cache=True)
def _quad(l0, l1, l2, l3, na, sa, sb, X, Y, TL):
    g0 = _g(l0, na, sa, sb)
    g1 = _g(l1, na, sa, sb)
    g2 = _g(l2, na, sa, sb)
    g3 = _g(l3, na, sa, sb)
    lam = lam4(X[g0], Y[g0], X[g1], Y[g1], X[g2], Y[g2], X[g3], Y[g3])
    tmax = max(max(TL[g0], TL[g1]), max(TL[g2], TL[g3]))
    return lam, lam > tmax


@njit(cache=True)
def _bit(l, use_mask):
    return (np.int64(1) << np.int64(l)) if use_mask else np.int64(0)


@njit(cache=True)
def _tau(nbad, inter, use_mask, pack):
    t = pack
    if nbad > 0 and t < 1:
        t = 1
    if nbad > 0 and use_mask and inter == 0 and t < 2:
        t = 2
    return t


@njit(cache=True)
def _fam_len(fam, na, nb):
    """Number of quadruples of family fam for a pair with na + nb hits (each family pairwise disjoint)."""
    if fam == 0:                       # F1: hit a[-1-k] + b's triple spread over all of b
        return min(na, nb // 3)
    if fam == 1:                       # F2: hit b[k] + a's triple spread over all of a
        return min(nb, na // 3)
    if fam == 2:                       # F3: two hits on each side of the junction
        return min(na // 2, nb // 2)
    if fam == 3:                       # F1 on b's first 6 hits only (tight inner tolerances), long b
        return min(na, 2) if nb >= 9 else 0
    return min(nb, 2) if na >= 9 else 0    # F2 on a's last 6 hits only, long a


@njit(cache=True)
def _fam_quad(fam, k, na, nb):
    """Local hit indices (a: 0..na-1 by R, b: na..na+nb-1 by R) of quadruple k of family fam."""
    if fam == 0:
        m = nb // 3
        return na - 1 - k, na + k, na + k + m, na + k + 2 * m
    if fam == 1:
        m = na // 3
        return na + k, na - 1 - k, na - 1 - k - m, na - 1 - k - 2 * m
    if fam == 2:
        return na - 1 - 2 * k, na - 2 - 2 * k, na + 2 * k, na + 2 * k + 1
    if fam == 3:
        return na - 1 - k, na + k, na + k + 2, na + k + 4
    return na + k, na - 1 - k, na - 3 - k, na - 5 - k


@njit(cache=True)
def bounds(ia, ib, fs, fn, X, Y, TL, active, max_rms2, max_out, lb, tau):
    """For every active pair (a, b): lb, a lower bound on the union fit's sum of squared transverse residuals, and tau, a
    lower bound on its n_out, from quadruples of its hits: families _fam_len / _fam_quad (pairwise disjoint within a
    family: lb = max over families of sum 2 lam^2, tau >= max over families of the number of bad quadruples) and, for
    unions of <= 7 hits, every quadruple; tau >= 1 if any quadruple is bad, >= 2 if the bad ones share no hit. Stops at
    the first quadruple that proves lb > n max_rms2 or tau > max_out (lb and tau are then partial, still bounds); with
    max_rms2 = max_out = inf the bounds are complete. Returns the number of pairs that stay active (active is cleared
    for the others)."""
    n_keep = 0
    for p in range(len(ia)):
        if not active[p]:
            continue
        a = ia[p]
        b = ib[p]
        na = fn[a]
        nb = fn[b]
        n = na + nb
        sa = fs[a]
        sb = fs[b]
        thr = n * max_rms2 * (1.0 + 1e-9)
        use_mask = n <= 62
        inter = np.int64(-1)
        nbad = 0
        maxlam = 0.0
        best_ssq = 0.0
        best_pack = 0
        rej = False
        for fam in range(5):
            if rej:
                break
            ssq = 0.0
            pack = 0
            for k in range(_fam_len(fam, na, nb)):
                l0, l1, l2, l3 = _fam_quad(fam, k, na, nb)
                lam, bad = _quad(l0, l1, l2, l3, na, sa, sb, X, Y, TL)
                ssq += 2.0 * lam * lam
                maxlam = max(maxlam, lam)
                if bad:
                    pack += 1
                    nbad += 1
                    inter &= _bit(l0, use_mask) | _bit(l1, use_mask) | _bit(l2, use_mask) | _bit(l3, use_mask)
                if ssq > thr or _tau(nbad, inter, use_mask, max(best_pack, pack)) > max_out:
                    rej = True
                    break
            best_ssq = max(best_ssq, ssq)
            best_pack = max(best_pack, pack)
        # every quadruple of a small union (one of them alone bounds the sum of squares)
        if not rej and n <= 7:
            for l0 in range(n - 3):
                for l1 in range(l0 + 1, n - 2):
                    for l2 in range(l1 + 1, n - 1):
                        for l3 in range(l2 + 1, n):
                            lam, bad = _quad(l0, l1, l2, l3, na, sa, sb, X, Y, TL)
                            maxlam = max(maxlam, lam)
                            if bad:
                                nbad += 1
                                inter &= (_bit(l0, use_mask) | _bit(l1, use_mask) | _bit(l2, use_mask)
                                          | _bit(l3, use_mask))
                            if 2.0 * maxlam * maxlam > thr or _tau(nbad, inter, use_mask, best_pack) > max_out:
                                rej = True
                                break
                        if rej:
                            break
                    if rej:
                        break
                if rej:
                    break
        lb[p] = max(best_ssq, 2.0 * maxlam * maxlam)
        tau[p] = _tau(nbad, inter, use_mask, best_pack)
        if rej or lb[p] > thr or tau[p] > max_out:
            active[p] = False
        else:
            n_keep += 1
    return n_keep


class Table:
    """Per-event arrays of the pre-filter: fragments of trk with their hits in R order, hit x, y, tolerance 3 st,
    R = hypot(r, z); the endpoints of merge_fragments."""

    def __init__(self, r, phi, z, labels, trk, volume):
        self.R = np.hypot(r, z)
        self.flat, self.fs, self.fn = endpoints(labels, trk, self.R)
        f = self.flat
        self.X = r[f] * np.cos(phi[f])
        self.Y = r[f] * np.sin(phi[f])
        st, _ = PA._scales(np.asarray(volume)[f])
        self.TL = 3.0 * st
        first, last = f[self.fs], f[self.fs + self.fn - 1]
        self.R_in, self.R_out = self.R[first], self.R[last]
        self.phi_in, self.phi_out = phi[first], phi[last]
        Rf = self.R[f]
        self.Rmin = np.minimum.reduceat(Rf, self.fs)
        self.Rmax = np.maximum.reduceat(Rf, self.fs)



def prefilter(tab: Table, ia, ib, max_rms: float, max_out: float, min_span_mm: float, max_zrms: float = 0.0,
              full: bool = False):
    """keep (bool per candidate), lb, tau, span. full=True computes the complete bounds (no early stop; for checks)."""
    span = np.maximum(tab.Rmax[ia], tab.Rmax[ib]) - np.minimum(tab.Rmin[ia], tab.Rmin[ib])
    n = tab.fn[ia] + tab.fn[ib]
    keep = (span >= min_span_mm) & (n >= 3)              # _merge: span test; fit_tracks: >= 3 hits
    lb = np.zeros(len(ia))
    tau = np.zeros(len(ia), dtype=np.int64)
    if not (max_rms >= 0 and max_out >= 0 and max_zrms >= 0):   # rms, zrms, n_out >= 0: nothing passes (NaN too)
        keep[:] = False
    if full:
        act = keep.copy()
        bounds(ia, ib, tab.fs, tab.fn, tab.X, tab.Y, tab.TL, act, np.inf, np.inf, lb, tau)
        keep &= ~(lb > n * float(max_rms) ** 2 * (1.0 + 1e-9)) & ~(tau > max_out)
    else:
        bounds(ia, ib, tab.fs, tab.fn, tab.X, tab.Y, tab.TL, keep, float(max_rms) ** 2, float(max_out), lb, tau)
    return keep, lb, tau, span


def prefilter_cuts(tab: Table, ia, ib, cuts: BinnedCuts, min_span_mm: float, full: bool = False):
    """prefilter for a cut set: a pair is kept if it could pass the cuts of at least one |eta| bin (its bin is only
    known after the fit). The bounds are computed with the loosest values (early stop only where every bin rejects)."""
    keep, lb, tau, span = prefilter(tab, ia, ib, float(np.max(cuts.max_rms)), float(np.max(cuts.max_out)),
                                    min_span_mm, float(np.max(cuts.max_zrms)), full=full)
    if not cuts.scalar:
        n = (tab.fn[ia] + tab.fn[ib]).astype(np.float64)
        any_bin = np.zeros(len(ia), dtype=bool)
        for rms_k, zrms_k, out_k in zip(cuts.max_rms, cuts.max_zrms, cuts.max_out):
            if rms_k >= 0 and zrms_k >= 0 and out_k >= 0:
                any_bin |= ~(tau > out_k) & ~(lb > n * rms_k ** 2 * (1.0 + 1e-9))
        keep &= any_bin
    return keep, lb, tau, span


# ------------------------------------------------------------------------------------------------ 3. pinning
def union_rows(tab: Table, ia, ib):
    """rows and pseudo-labels exactly as _merge builds them (np.flatnonzero((labels == a) | (labels == b)))."""
    rows, plab = [], []
    for k, (a, b) in enumerate(zip(ia.tolist(), ib.tolist())):
        idx = np.sort(np.concatenate([tab.flat[tab.fs[a]:tab.fs[a] + tab.fn[a]],
                                      tab.flat[tab.fs[b]:tab.fs[b] + tab.fn[b]]]))
        rows.append(idx)
        plab.append(np.full(len(idx), k))
    return np.concatenate(rows), np.concatenate(plab)


def _line_sz_done(r, phi, z, volume, labels, min_hits: int = 3, n_iter: int = LINE_SZ_ITER) -> dict:
    """hnl.vertex.pairing.fit_tracks, operation for operation, up to the s-z fit, then the loop of
    hnl.truth.helix.fit_line_sz with its stopping test recorded per track for all n_iter iterations ('done', (T, n_iter))
    and the values it returns when it stops ('z0', 'cot', 'zrms'; with 'rms', 'n_out' of the transverse fit and 'K', the
    iterations it runs). Used to find pairs that keep fit_line_sz iterating; tested bit for bit against fit_tracks."""
    labels = np.asarray(labels)
    ok = labels >= 0
    lab, cnt = np.unique(labels[ok], return_counts=True)
    keep = lab[cnt >= min_hits]
    sel = np.flatnonzero(np.isin(labels, keep))
    R3 = np.hypot(r, z)
    order = sel[np.lexsort((R3[sel], labels[sel]))]
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    p = hx.pad_tracks(labels[order], xyz[order], rows=order)
    st, sz = PA._scales(np.asarray(volume))
    pad = lambda v: np.where(p.rows >= 0, v[np.maximum(p.rows, 0)], 1.0)
    p0, p1, p2 = hx._first_last(p)
    c, Rc = hx.circle_from_3(p0, p1, p2)
    Rc = np.clip(Rc, 10.0, 1e7)
    sigma = hx.rotation_sense(p, c[:, 0], c[:, 1])
    pg = hx.perigee(c[:, 0], c[:, 1], Rc, sigma)
    f = hx.fit_perigee(p, pg.d0, pg.phi0, sigma / Rc, free_kappa=True, scale=pad(st))
    s = hx.arc_length_perigee(p, f.d0, f.phi0, f.kappa)
    # hx.fit_line_sz(s, p.z, p.mask, scale=pad(sz)), recording `done`
    zz, mask, scale = p.z, p.mask, pad(sz)
    m = mask.astype(np.float64)
    w = m.copy()
    z0 = np.zeros(len(s))
    ct = np.zeros(len(s))
    done_hist = np.zeros((len(s), n_iter), dtype=bool)
    stop = None
    zrms_hist = np.empty((len(s), n_iter))
    for it in range(n_iter):
        S0 = np.sum(w, 1)
        S1 = np.sum(w * s, 1)
        S2 = np.sum(w * s * s, 1)
        Z0 = np.sum(w * zz, 1)
        Z1 = np.sum(w * s * zz, 1)
        det = S0 * S2 - S1**2
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        ct_n = (S0 * Z1 - S1 * Z0) / det
        z0_n = (S2 * Z0 - S1 * Z1) / det
        done = (np.abs(ct_n - ct) < 1e-10) & (np.abs(z0_n - z0) < 1e-7)
        z0, ct = z0_n, ct_n
        res = zz - z0[:, None] - ct[:, None] * s
        w = hx.huber_weights(res, scale) * m
        done_hist[:, it] = done
        # the zrms fit_line_sz returns if it stops after this iteration (same residuals, same expression)
        zrms_hist[:, it] = np.sqrt(np.sum(m * res**2, 1) / np.maximum(m.sum(1), 1))
        if stop is None and (done.all() or it == n_iter - 1):
            stop = (it + 1, z0.copy(), ct.copy(), zrms_hist[:, it].copy())
    return {"track": p.track_ids, "done": done_hist, "K": stop[0], "z0": stop[1], "cot": stop[2], "zrms": stop[3],
            "zrms_hist": zrms_hist, "rms": f.rms, "n_out": f.n_out, "H": p.mask.shape[1]}


def certificate(d: dict, span_sel, K_lo: int, max_rms, max_zrms, max_out, min_span_mm) -> tuple[bool, str, int]:
    """Does _merge's outcome on this batch depend on the s-z iteration count K, for K_lo <= K <= n_iter? d is the
    replica of the batch (_line_sz_done); rms, n_out and span do not depend on K, zrms does. True if every pair is
    accepted (finite q) for all those K or for none, and the accepted pairs' q are strictly increasing in one and the
    same order for every K (same greedy sequence, no exact ties). Returns (ok, reason, number of exactly tied q)."""
    zr = d["zrms_hist"][:, K_lo - 1:]
    base = (d["rms"] <= max_rms) & (d["n_out"] <= max_out) & (span_sel[d["track"]] >= min_span_mm)
    acc = base[:, None] & (zr <= max_zrms)
    if not (acc == acc[:, :1]).all():
        return False, "acceptance depends on K", 0
    A = acc[:, 0]
    q = d["rms"][A, None] + 0.1 * zr[A]                # _merge: q = rms + 0.1 zrms
    qs = q[np.argsort(q[:, 0], kind="stable")]
    if len(qs) > 1 and not (np.diff(qs, axis=0) > 0).all():
        ties = int((np.diff(qs, axis=0) == 0).any(1).sum())
        return False, ("exactly equal q" if ties else "order depends on K"), ties
    return True, "", 0


def pin(tab: Table, ia, ib, keep, span, r, phi, z, volume, max_rms, max_zrms, max_out, min_span_mm,
        sample: int = 4096, seed: int = 0, n_iter: int = LINE_SZ_ITER) -> tuple[np.ndarray, dict]:
    """Indices into C of the batch F' = F + h + W handed to _merge (sorted; section 3). h pins the padded width H. The
    s-z iteration count K: (1) a sample of the rejected pairs is replicated; if, at every iteration 0..n_iter-2, one of
    them has not yet passed fit_line_sz's stopping test, they (W) make K(F') = K(C) = n_iter. (2) Otherwise the batch
    F + h + those found is replicated: _merge(F') runs K' iterations and K' <= K(C) <= n_iter (C contains F'); if the
    outcome is the same for every K in that range (certificate), F' is exact whatever K(C) is. (3) Otherwise more
    witnesses are searched, and finally the whole of C is replicated to reproduce K(C) exactly."""
    sizes = tab.fn[ia] + tab.fn[ib]
    fitted = sizes >= 3
    h = int(np.flatnonzero(fitted & (sizes == sizes[fitted].max()))[0])
    need = n_iter - 1                                  # iterations 0..n_iter-2 must each see a track not yet done
    covered = np.zeros(need, dtype=bool)
    chosen = [h]
    rng = np.random.default_rng(seed)
    F = np.flatnonzero(keep)
    pool = rng.permutation(np.flatnonzero(~keep & fitted))
    info = {"H": int(sizes[h])}

    def replica(batch):
        rows, plab = union_rows(tab, ia[batch], ib[batch])
        d = _line_sz_done(r[rows], phi[rows], z[rows], np.asarray(volume)[rows], plab, n_iter=n_iter)
        info["H_replica"] = d["H"]
        return d

    def cover(batch):
        nonlocal covered
        nd = ~replica(batch)["done"][:, :need]
        while not covered.all():
            gain = (nd & ~covered).sum(1)
            g = int(np.argmax(gain))
            if gain[g] == 0:
                break
            chosen.append(int(batch[g]))
            covered |= nd[g]

    pos = min(sample, len(pool))
    if pos:
        cover(np.r_[h, pool[:pos]])
    how = "witness"
    if not covered.all():
        sel = np.union1d(F, np.asarray(chosen, dtype=np.int64))
        d = replica(sel)
        ok, why, ties = certificate(d, span[sel], d["K"], max_rms, max_zrms, max_out, min_span_mm)
        info.update({"K_replica": d["K"], "certificate": ok, "certificate_fail": why, "certificate_ties": ties,
                     "accepted": int(((d["rms"] <= max_rms) & (d["n_out"] <= max_out)
                                      & (d["zrms"] <= max_zrms) & (span[sel] >= min_span_mm)).sum())})
        if ok:
            info.update({"K": d["K"], "K_proof": "certificate", "K_range": [d["K"], n_iter],
                         "examined_for_K": int(pos), "pinning_pairs_added": int(len(sel) - len(F))})
            return sel, info
        how = "witness_escalated"
        size = 4 * sample
        while not covered.all() and pos < len(pool):
            S = pool[pos:pos + size]
            pos += len(S)
            cover(np.r_[h, S])
            size *= 4
        if not covered.all():                          # the whole of C: K(C) exactly
            how = "exact"
            for k in range(0, len(F), 20000):
                cover(np.r_[h, F[k:k + 20000]])
    K = int(np.argmin(covered)) + 1 if not covered.all() else n_iter
    sel = np.union1d(F, np.asarray(chosen, dtype=np.int64))
    info.update({"K": K, "K_proof": how, "pinning_pairs_added": int(len(sel) - len(F)),
                 "examined_for_K": int(pos + (len(F) if how == "exact" else 0))})
    return sel, info


# ------------------------------------------------------------------------------------------------ 4. exact pre-fit
@functools.lru_cache(maxsize=None)
def _sum_sig(W: int, k: int) -> tuple:
    """The order in which numpy's pairwise summation adds a row of W entries whose first k are values and the rest
    padding zeros (np.sum along the hits axis; numpy/_core/src/umath/loops_utils.h.src): sequential below 8 entries;
    8 accumulators over blocks of 8 up to 128 entries, the entries past the last whole block added one by one; above
    128 the row is split at n2 = W / 2 rounded down to a multiple of 8 and both halves summed recursively. A half that
    holds only padding adds a zero, which changes no value. Rows of the same signature give the same bits."""
    if k <= 0:
        return ("z",)
    if W < 8:
        return ("s", k)
    if W <= 128:
        B = 8 * (W // 8)
        return ("b", k) if B >= k else ("t", B, k)
    n2 = W // 2
    n2 -= n2 % 8
    left, right = _sum_sig(n2, min(k, n2)), _sum_sig(W - n2, k - n2)
    return left if right == ("z",) else ("p", left, right)


def same_sums(W: int, m: int, H: int) -> bool:
    """Is a union of m hits summed in a batch padded to W as in one padded to H? The fit sums rows of W entries with m
    values, rotation_sense rows of W - 1 with m - 1; one entry fewer is also required (in case a reduction starts
    from its first entry)."""
    return all(_sum_sig(W - j, m - j) == _sum_sig(H - j, m - j) for j in (0, 1, 2))


_PREFIT = {}


def _prefit_batch(bt):
    """fit_tracks of one pre-fit batch (in a worker process: the event's arrays are inherited by fork)."""
    tab, ia, ib, r, phi, z, volume = _PREFIT["args"]
    rows, plab = union_rows(tab, ia[bt], ib[bt])
    f = PA.fit_tracks(r[rows], phi[rows], z[rows], np.asarray(volume)[rows], plab, min_hits=3)
    return {k: f[k] for k in ("track", "n_hits", "rms", "zrms", "n_out", "cot", "z0", "d0", "R")}


def prefit(tab: Table, ia, ib, members, h: int, W, r, phi, z, volume, batch: int = 4096, n_proc: int = 1) -> dict:
    """Fit values (rms, zrms, n_out, cot, z0, d0, R) of the candidates `members` (indices into C) bit for bit as in the
    batch of all candidates C: they are fitted by hnl.vertex.pairing.fit_tracks in batches of at most `batch` pairs of
    one size class, each with the witnesses W (so fit_line_sz runs as many iterations as in C) and a padding pair of C
    that sets the padded width to one that sums every member's rows as C's width does (same_sums; else h, the pair
    with C's width). Smaller and narrower batches cost less: fit_perigee freezes converged tracks, so a batch stops
    when its own slowest track has converged, without changing any track's result."""
    sizes = tab.fn[ia] + tab.fn[ib]
    H = int(sizes[h])
    W = np.asarray(W, dtype=np.int64)
    w_size = int(sizes[W].max()) if len(W) else 0
    fitted = np.flatnonzero(sizes >= 3)
    avail, first = np.unique(sizes[fitted], return_index=True)
    rep = fitted[first]                                 # one pair of C per union size
    wmin = {}
    for m in np.unique(sizes[members]).tolist():
        wmin[m] = next(w for w in range(m, H + 1) if same_sums(w, m, H))
    key = np.array([wmin[m] for m in sizes[members].tolist()], dtype=np.int64)
    keys = ("rms", "zrms", "n_out", "cot", "z0", "d0", "R")
    out = {k: np.empty(len(members), dtype=np.int64 if k == "n_out" else np.float64) for k in keys}
    jobs, widths = [], {}
    w_sizes = np.unique(sizes[W]).tolist() if len(W) else []
    for w in np.unique(key).tolist():
        pos = np.flatnonzero(key == w)
        # the witnesses must keep their own trajectories too (else the batch could stop before C's iteration count)
        ms = sorted(set(np.unique(sizes[members[pos]]).tolist()) | set(w_sizes))
        pad = h
        for s, p in zip(avail.tolist(), rep.tolist()):
            if s >= w and all(same_sums(max(s, w_size), m, H) for m in ms):
                pad = p
                break
        Hb = max(int(sizes[pad]), w_size)
        if not all(same_sums(Hb, m, H) for m in ms):
            pad, Hb = h, H
        widths[w] = Hb
        for k0 in range(0, len(pos), batch):
            ch = pos[k0:k0 + batch]
            jobs.append((ch, np.r_[pad, W, members[ch]], Hb))
    _PREFIT["args"] = (tab, ia, ib, r, phi, z, volume)
    try:
        if n_proc > 1 and len(jobs) > 1:                # independent batches: same values in any process
            import concurrent.futures as cf
            import multiprocessing as mp
            with cf.ProcessPoolExecutor(max_workers=n_proc, mp_context=mp.get_context("fork")) as ex:
                fits = list(ex.map(_prefit_batch, [bt for _, bt, _ in jobs]))
        else:
            fits = [_prefit_batch(bt) for _, bt, _ in jobs]
    finally:
        _PREFIT.clear()
    off = 1 + len(W)
    for (ch, bt, Hb), f in zip(jobs, fits):
        if not (np.array_equal(f["track"], np.arange(len(bt))) and int(f["n_hits"].max()) == Hb):
            raise RuntimeError("prefit: unexpected batch layout")
        for k in keys:
            out[k][ch] = f[k][off:]
    out["batches"], out["class_widths"] = len(jobs), widths
    return out


def greedy(cand_a, cand_b, q, trk, one_per_end: bool = True):
    """The acceptance loop of hnl.trackbuild.merge_fragments._merge, statement for statement: the candidates (a, b)
    in np.argsort(q) order until the first inf; skip if a and b are already joined; one_per_end (sequential pass):
    skip if a's outer or b's inner end is used; else (any-order pass): skip if a or b already has two merges.
    Returns the accepted positions in order and the labels map {fragment: root}."""
    order = np.argsort(q)
    used_out, used_in = set(), set()
    parent = {t: t for t in trk.tolist()}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    n_merges = {t: 0 for t in trk.tolist()}
    acc = []
    for k in order.tolist():
        if not np.isfinite(q[k]):
            break
        a, b = cand_a[k], cand_b[k]
        if find(a) == find(b):
            continue
        if one_per_end:
            if a in used_out or b in used_in:
                continue
            used_out.add(a)
            used_in.add(b)
        else:
            if n_merges[a] >= 2 or n_merges[b] >= 2:
                continue
            n_merges[a] += 1
            n_merges[b] += 1
        parent[find(b)] = find(a)
        acc.append(k)
    return acc, {t: find(t) for t in trk.tolist()}


def merge_exact(r, phi, z, labels, trk, volume, ia, ib, members, cuts, min_span_mm: float, one_per_end: bool,
                tab=None, keep=None, sample: int = 4096, seed: int = 0, batch: int = 4096, n_proc: int = 1,
                info: dict | None = None):
    """_merge(C, ...) for the candidates C = (trk[ia], trk[ib]) without refitting all of them in _merge: the pairs
    `members` (those that could pass the cuts; all of C if no filter) get their fit values bit for bit as in _merge's
    batch of all of C (prefit; h pins the padded width, W the s-z iteration count), q and acceptance exactly as
    _merge computes them (per candidate cuts for a cut set), _merge's acceptance loop (greedy) gives the merges S;
    then the unchanged _merge on S + h + W returns the labels (it refits them with the same values and takes the same
    decisions). If its labels differ from greedy's (never seen), the unchanged _merge is run on all of C instead.
    keep: the pre-filter's mask over C (rejected pairs are the witness pool); None = every pair is a member."""
    info = {} if info is None else info
    tab = Table(r, phi, z, labels, trk, volume) if tab is None else tab
    sizes = tab.fn[ia] + tab.fn[ib]
    fitted = sizes >= 3
    h = int(np.flatnonzero(fitted & (sizes == sizes[fitted].max()))[0])
    keep = np.zeros(len(ia), dtype=bool) if keep is None else keep
    t0 = time.time()
    W, K, how = _witnesses(tab, ia, ib, keep, h, r, phi, z, volume, sample, seed)
    pins = np.r_[h, W]
    Fp = np.union1d(np.asarray(members, dtype=np.int64), pins)
    vals = prefit(tab, ia, ib, Fp, h, W, r, phi, z, volume, batch=batch, n_proc=n_proc)
    t1 = time.time()
    span = np.maximum(tab.Rmax[ia[Fp]], tab.Rmax[ib[Fp]]) - np.minimum(tab.Rmin[ia[Fp]], tab.Rmin[ib[Fp]])
    cr, cz, co = cuts.for_cot(vals["cot"])
    acc = (vals["rms"] <= cr) & (vals["zrms"] <= cz) & (vals["n_out"] <= co) & (span >= min_span_mm)
    q = np.full(len(Fp), np.inf)
    q[acc] = vals["rms"][acc] + 0.1 * vals["zrms"][acc]
    acc_k, roots = greedy(trk[ia[Fp]].tolist(), trk[ib[Fp]].tolist(), q, trk, one_per_end=one_per_end)
    sel = np.union1d(Fp[acc_k], pins)                  # the merges and the pinning pairs, in C's order
    pos = np.searchsorted(Fp, sel)
    qf = q[acc]
    info.update({"H": int(sizes[h]), "K": K, "K_from": how, "witnesses": int(len(W)), "h": h, "W": W.tolist(),
                 "prefit_s": round(t1 - t0, 2), "prefit_batches": vals["batches"],
                 "class_widths": vals["class_widths"], "accepted": int(acc.sum()), "merges": int(len(acc_k)),
                 "pairs_to_merge": int(len(sel)), "finite_q_ties": int(len(qf) - len(np.unique(qf)))})
    cand = list(zip(trk[ia[sel]], trk[ib[sel]]))
    before = labels.copy()
    out = _merge(cand, trk, labels, r, phi, z, volume, *cuts.for_cot(vals["cot"][pos]), one_per_end=one_per_end,
                 min_span_mm=min_span_mm)
    pred = before.copy()
    for t, rt in roots.items():
        if rt != t:
            pred[before == t] = rt
    info["consistent"] = bool(np.array_equal(pred, out))
    info["merge_s"] = round(time.time() - t1, 2)
    if not info["consistent"]:                          # a bug somewhere: fall back to the definition itself
        import warnings
        warnings.warn("merge_exact: _merge's labels differ from its acceptance loop's; running the unfiltered _merge")
        allc = list(zip(trk[ia], trk[ib]))
        if cuts.scalar:
            cuts_all = cuts.for_cot(None)
        else:
            vals_all = prefit(tab, ia, ib, np.flatnonzero(fitted), h, W, r, phi, z, volume, batch=batch, n_proc=n_proc)
            cuts_all = cuts.for_cot(vals_all["cot"])
        out = _merge(allc, trk, before.copy(), r, phi, z, volume, *cuts_all, one_per_end=one_per_end,
                     min_span_mm=min_span_mm)
    return out, cand


def _witnesses(tab, ia, ib, keep, h, r, phi, z, volume, sample, seed, n_iter=LINE_SZ_ITER, small: int = 8,
               chunk: int = 20000):
    """Pairs W of C that make every batch containing them run as many s-z iterations K as the batch of all of C:
    for each iteration j < K - 1, one pair of W has not yet passed fit_line_sz's stopping test at j (a batch stops
    at the first iteration where all its tracks have; it cannot run longer than C, which contains it). First a sample
    of the rejected pairs (unions of <= `small` hits first: they never widen a batch): if it covers iterations
    0..n_iter-2, K = n_iter. Otherwise every fitted pair of C is replicated and K(C) found exactly. Returns (W, K,
    how) with how 'witness' or 'exact'."""
    sizes = tab.fn[ia] + tab.fn[ib]
    need = n_iter - 1
    covered = np.zeros(need, dtype=bool)
    chosen = []
    rng = np.random.default_rng(seed)
    fitted = np.flatnonzero(sizes >= 3)
    rej = fitted[~keep[fitted]]

    def cover(idx):
        nonlocal covered
        bt = np.r_[h, idx]
        rows, plab = union_rows(tab, ia[bt], ib[bt])
        nd = ~_line_sz_done(r[rows], phi[rows], z[rows], np.asarray(volume)[rows], plab, n_iter=n_iter)["done"][:, :need]
        nd[0] = False                                   # h only pins the width
        while not covered.all():
            gain = (nd & ~covered).sum(1)
            g = int(np.argmax(gain))
            if gain[g] == 0:
                break
            chosen.append(int(bt[g]))
            covered |= nd[g]

    for pool in (rej[sizes[rej] <= small], rej[sizes[rej] > small]):
        if not covered.all() and len(pool):
            cover(rng.permutation(pool)[:sample])
    how = "witness"
    if not covered.all():                               # the whole of C (small events): K(C) exactly
        how = "exact"
        order = np.r_[fitted[sizes[fitted] <= small], fitted[sizes[fitted] > small]]
        for k0 in range(0, len(order), chunk):
            cover(order[k0:k0 + chunk])
    K = n_iter if covered.all() else int(np.argmin(covered)) + 1
    return np.asarray(sorted(set(chosen)), dtype=np.int64), K, how


def merge_sequential_reference(r, phi, z, labels, volume=None, min_frag: int = 2, max_dphi: float = 0.3,
                               max_gap_r: float = 400.0, max_rms: float = 1.0, max_zrms: float = 5.0,
                               max_out: int = 1, min_span_mm: float = 50.0, cuts: BinnedCuts | None = None):
    """The definition without any filter (checks only: about 3e6 refits on a TrackML event): every candidate of the
    sequential pass to the unchanged _merge. With a cut set, each candidate's cuts come from its |eta| bin, computed
    from a fit of the very batch _merge fits (same rows, same pseudo-labels: the same values); with scalar cuts this
    is merge_fragments(mode="sequential") itself."""
    cuts = BinnedCuts.uniform(max_rms, max_zrms, max_out) if cuts is None else cuts
    labels = np.asarray(labels).copy()
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    if volume is None:
        volume = _volume_from_radius(r, z)
    ok = labels >= 0
    if not ok.any():
        return labels
    trk, cnt = np.unique(labels[ok], return_counts=True)
    trk = trk[cnt >= min_frag]
    if len(trk) < 2:
        return labels
    tab = Table(r, phi, z, labels, trk, volume)
    ia, ib = sequential_candidates(tab.R_in, tab.R_out, tab.phi_in, tab.phi_out, max_dphi, max_gap_r)
    cand = list(zip(trk[ia], trk[ib]))
    if not cand:
        return labels
    if cuts.scalar:
        cr, cz, co = cuts.for_cot(None)
    else:
        rows, plab = union_rows(tab, ia, ib)
        f = PA.fit_tracks(r[rows], phi[rows], z[rows], np.asarray(volume)[rows], plab, min_hits=3)
        if not np.array_equal(f["track"], np.arange(len(cand))):
            raise RuntimeError("merge_sequential_reference: candidates with fewer than 3 hits")
        cr, cz, co = cuts.for_cot(f["cot"])
    return _merge(cand, trk, labels, r, phi, z, volume, cr, cz, co, one_per_end=True, min_span_mm=min_span_mm)


# ------------------------------------------------------------------------------------------------ the pass
def merge_sequential(r, phi, z, labels, volume=None, min_frag: int = 2, max_dphi: float = 0.3,
                     max_gap_r: float = 400.0, max_rms: float = 1.0, max_zrms: float = 5.0, max_out: int = 1,
                     min_span_mm: float = 50.0, cuts: BinnedCuts | None = None, method: str = "prefit",
                     sample: int = 4096, seed: int = 0, batch: int = 4096, n_proc: int = 1,
                     info: dict | None = None, return_batch: bool = False, check_ties: bool = False):
    """= hnl.trackbuild.merge_fragments.merge_fragments(r, phi, z, labels, volume, min_frag, max_dphi, max_gap_r, max_rms,
    max_zrms, max_out, mode="sequential", min_span_mm=min_span_mm); with cuts (a BinnedCuts) each candidate is held to
    the cuts of its union's |eta| bin instead (the unchanged _merge receives them as arrays).
    method "prefit" (default): the kept pairs' fit values bit for bit as in the batch of all candidates (prefit, size
    classes + witnesses), q and acceptance as _merge computes them, _merge's acceptance loop (greedy) for the merges,
    then the unchanged _merge on those merges + the pinning pairs (h, W), which reproduces them; info["consistent"]
    checks _merge's labels against greedy's; n_proc > 1 spreads the pre-fit batches over processes (same values).
    method "pinned": _merge on every kept pair + h + W (scalar cuts only).
    info receives counts and timings; return_batch=True also returns the candidate list handed to _merge."""
    t0 = time.time()
    info = {} if info is None else info
    cuts = BinnedCuts.uniform(max_rms, max_zrms, max_out) if cuts is None else cuts
    labels = np.asarray(labels).copy()
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    if volume is None:
        volume = _volume_from_radius(r, z)
    out = (labels, []) if return_batch else labels
    ok = labels >= 0
    if not ok.any():
        return out
    trk, cnt = np.unique(labels[ok], return_counts=True)
    trk = trk[cnt >= min_frag]
    if len(trk) < 2:
        return out
    tab = Table(r, phi, z, labels, trk, volume)
    ia, ib = sequential_candidates(tab.R_in, tab.R_out, tab.phi_in, tab.phi_out, max_dphi, max_gap_r)
    t1 = time.time()
    info.update({"fragments": int(len(trk)), "candidates": int(len(ia)), "candidates_s": round(t1 - t0, 2),
                 "cuts": cuts.name, "method": method})
    if len(ia) == 0:
        return out
    keep, _, _, span = prefilter_cuts(tab, ia, ib, cuts, min_span_mm)
    t2 = time.time()
    info.update({"kept": int(keep.sum()), "prefilter_s": round(t2 - t1, 2)})
    if not keep.any():                                 # every q is inf: _merge would merge nothing
        info.update({"merge_s": 0.0, "pairs_to_merge": 0, "total_s": round(t2 - t0, 2)})
        return out
    if method == "pinned":
        if not cuts.scalar:
            raise ValueError("method 'pinned' takes scalar cuts only")
        mr, mz, mo = cuts.for_cot(None)
        sel, pinfo = pin(tab, ia, ib, keep, span, r, phi, z, volume, mr, mz, mo, min_span_mm, sample=sample,
                         seed=seed)
        t3 = time.time()
        info.update(pinfo)
        info.update({"pin_s": round(t3 - t2, 2), "pairs_to_merge": int(len(sel))})
        cand = list(zip(trk[ia[sel]], trk[ib[sel]]))
        labels = _merge(cand, trk, labels, r, phi, z, volume, mr, mz, mo, one_per_end=True, min_span_mm=min_span_mm)
        info.update({"merge_s": round(time.time() - t3, 2), "total_s": round(time.time() - t0, 2)})
        return (labels, cand) if return_batch else labels
    labels, cand = merge_exact(r, phi, z, labels, trk, volume, ia, ib, np.flatnonzero(keep), cuts, min_span_mm, True,
                               tab=tab, keep=keep, sample=sample, seed=seed, batch=batch, n_proc=n_proc, info=info)
    info["total_s"] = round(time.time() - t0, 2)
    return (labels, cand) if return_batch else labels
