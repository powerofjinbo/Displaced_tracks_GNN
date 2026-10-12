"""hnl.trackml.seqmerge: the exact pre-filter of the sequential helix fragment merging (docs/trackml_plan.md 6.7, G3).

    pytest src/hnl/trackml/tests/test_seqmerge.py
"""

import numpy as np
import pytest

import hnl.trackbuild.merge_fragments as MF
from hnl.trackbuild.merge_fragments import _volume_from_radius, merge_fragments
from hnl.trackml import seqmerge as SM
from hnl.vertex import pairing as PA

BARREL = np.array([32.0, 72.0, 116.0, 172.0, 260.0, 360.0, 500.0, 660.0, 820.0, 1020.0])
DISKS = np.array([600.0, 700.0, 820.0, 960.0, 1100.0, 1300.0, 1500.0, 1800.0, 2200.0, 2600.0])


def _event(seed=0, n_tracks=150, n_junk=150, z_max=1100.0):
    """Helices (0.3-10 GeV, some displaced) through barrel layers and endcap disks, cut into 1-3 fragments, a few
    with a hit of another track; junk fragments of 2-4 random hits. Dense on purpose: many sequential candidates."""
    rng = np.random.default_rng(seed)
    r, phi, z, lab = [], [], [], []
    nxt = 0
    pool = []
    for _ in range(n_tracks):
        pt, q = rng.uniform(0.3, 10.0), rng.choice([-1, 1])
        p0, cot = rng.uniform(-np.pi, np.pi), rng.uniform(-2.5, 2.5)
        d0 = rng.normal(0, 0.05) if rng.random() < 0.8 else rng.uniform(-40, 40)
        Rc = 1000 / 0.6 * pt
        # circle through the PCA (d0) with radius Rc; points at the barrel radii / disk z
        cx, cy = (d0 + q * Rc) * -np.sin(p0), (d0 + q * Rc) * np.cos(p0)
        pts = []
        for rr in BARREL:
            a = (rr**2 + cx**2 + cy**2 - Rc**2) / (2 * rr * np.hypot(cx, cy))
            if abs(a) > 1:
                continue
            ph = np.arctan2(cy, cx) - q * np.arccos(a)
            x, y = rr * np.cos(ph), rr * np.sin(ph)
            ang = np.arctan2(y - cy, x - cx) - np.arctan2(-cy, -cx)
            s = abs(np.remainder(ang + np.pi, 2 * np.pi) - np.pi) * Rc
            zz = cot * s
            if abs(zz) < z_max:
                pts.append((rr, ph, zz))
        for zd in DISKS * np.sign(cot or 1.0):
            s = zd / cot if cot else np.inf
            if not np.isfinite(s) or s <= 0 or s > np.pi * Rc:
                continue
            ang0 = np.arctan2(-cy, -cx)
            ang = ang0 + q * s / Rc
            x, y = cx + Rc * np.cos(ang), cy + Rc * np.sin(ang)
            rr = np.hypot(x, y)
            if 32 < rr < 1020 and abs(zd) >= z_max:
                pts.append((rr, np.arctan2(y, x), zd))
        if len(pts) < 4:
            continue
        pts.sort(key=lambda t: np.hypot(t[0], t[2]))
        pts = [(rr + rng.normal(0, 0.03), ph + rng.normal(0, 5e-5), zz + rng.normal(0, 0.3)) for rr, ph, zz in pts]
        n = len(pts)
        cuts = sorted(rng.choice(np.arange(2, n - 1), size=min(rng.integers(0, 3), max(n - 3, 0)), replace=False)) \
            if n >= 5 else []
        piece = np.searchsorted(np.asarray(cuts), np.arange(n), side="right")
        for k, (rr, ph, zz) in enumerate(pts):
            r.append(rr)
            phi.append(ph)
            z.append(zz)
            lab.append(nxt + piece[k])
            pool.append(len(r) - 1)
        nxt += piece[-1] + 1
    lab = np.asarray(lab, dtype=np.int64)
    # a few fragments get one hit of another fragment (n_out = 1 unions)
    for _ in range(n_tracks // 10):
        i, j = rng.choice(len(lab), 2, replace=False)
        lab[i] = lab[j]
    lab = list(lab)
    for _ in range(n_junk):
        k = rng.integers(2, 5)
        rr = np.sort(rng.choice(BARREL, k, replace=False)) + rng.normal(0, 0.03, k)
        ph0 = rng.uniform(-np.pi, np.pi)
        for x in rr:
            r.append(x)
            phi.append(ph0 + rng.normal(0, 0.05))
            z.append(rng.uniform(-z_max, z_max))
            lab.append(nxt)
        nxt += 1
    for _ in range(100):                                           # unlabelled noise
        r.append(rng.uniform(30, 1000))
        phi.append(rng.uniform(-np.pi, np.pi))
        z.append(rng.uniform(-1000, 1000))
        lab.append(-1)
    return np.asarray(r), np.asarray(phi), np.asarray(z), np.asarray(lab, dtype=np.int64)


def _setup(r, phi, z, lab, min_frag=2, max_dphi=0.3, max_gap_r=400.0):
    vol = _volume_from_radius(r, z)
    ok = lab >= 0
    trk, cnt = np.unique(lab[ok], return_counts=True)
    trk = trk[cnt >= min_frag]
    tab = SM.Table(r, phi, z, lab, trk, vol)
    ia, ib = SM.sequential_candidates(tab.R_in, tab.R_out, tab.phi_in, tab.phi_out, max_dphi, max_gap_r)
    return vol, trk, tab, ia, ib


def _fit_all(tab, ia, ib, r, phi, z, vol):
    rows, plab = SM.union_rows(tab, ia, ib)
    f = PA.fit_tracks(r[rows], phi[rows], z[rows], vol[rows], plab, min_hits=3)
    assert np.array_equal(f["track"], np.arange(len(ia)))
    return f


# ------------------------------------------------------------------------------------------------ the annulus bound
def _lam_brute(P, rng):
    """A feasible minimax circle (or slab) found by search: never below the true minimum."""
    from scipy.optimize import minimize
    P = np.asarray(P, dtype=np.float64)
    best = np.inf
    for i in range(4):                                       # slabs flush with every pair
        for j in range(i + 1, 4):
            e = P[j] - P[i]
            nrm = np.array([-e[1], e[0]]) / np.hypot(*e)
            pr = (P - P[i]) @ nrm
            best = min(best, 0.5 * (pr.max() - pr.min()))
    w = lambda c: np.ptp(np.hypot(*(P - c).T))
    ctr = P.mean(0)
    span = np.ptp(P, 0).max()
    starts = [ctr + span * rng.normal(size=2) * s for s in (0.3, 1, 3, 10, 100) for _ in range(5)]
    for s in starts:
        res = minimize(w, s, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
        best = min(best, 0.5 * res.fun)
    return best


@pytest.mark.parametrize("kind", ["box", "arc", "line", "junction"])
def test_lam4_never_above_a_feasible_circle(kind):
    """lam4_raw <= the max distance of any circle found by search (else it would overstate the true minimum)."""
    rng = np.random.default_rng(7)
    worst = -np.inf
    for _ in range(80):
        if kind == "box":
            P = rng.uniform(-500, 500, (4, 2))
        elif kind == "arc":                                  # 4 hits of one helix, mm-level noise
            Rc, c = rng.uniform(300, 20000), rng.uniform(-1, 1)
            ang = c + np.sort(rng.uniform(0, 1000 / Rc, 4))
            P = Rc * np.stack([np.cos(ang), np.sin(ang)], 1) + rng.normal(0, rng.choice([0.01, 0.3, 3.0]), (4, 2))
        elif kind == "line":
            t = np.sort(rng.uniform(0, 1000, 4))
            P = np.stack([t, 1e-3 * t + rng.normal(0, 0.05, 4)], 1)
        else:                                                # two hits each of two different helices
            a = rng.uniform(30, 120, 2)
            b = rng.uniform(150, 500, 2)
            ph = rng.uniform(-0.3, 0.3)
            P = np.stack([np.r_[a, b] * np.cos(np.r_[0, 0.002, ph, ph + 0.004]),
                          np.r_[a, b] * np.sin(np.r_[0, 0.002, ph, ph + 0.004])], 1)
        lam = SM.lam4_raw(*P.ravel())
        worst = max(worst, lam - _lam_brute(P, rng))
    assert worst <= 1e-6, worst


def test_lam4_matches_high_precision():
    """The float64 formula against the same formula in 60-digit arithmetic, near-collinear helix hits included."""
    import mpmath as mp
    mp.mp.dps = 60

    def pattern(a, b, c, d):
        e1, e2 = b - a, d - c
        L1, L2 = mp.sqrt(e1[0]**2 + e1[1]**2), mp.sqrt(e2[0]**2 + e2[1]**2)
        u1, u2 = e1 / L1, e2 / L2
        if u1[0] * u2[0] + u1[1] * u2[1] < 0:
            u2 = u2 * -1
        D = u1[0] * u2[1] - u1[1] * u2[0]
        if abs(D) < SM.MIN_SIN:
            return mp.mpf(0)
        dd = ((c - a) + (d - b)) / 2
        t = (dd[0] * u2[0] + dd[1] * u2[1]) / D
        s = (dd[0] * u1[0] + dd[1] * u1[1]) / D
        return abs(mp.sqrt(L1**2 / 4 + t**2) - mp.sqrt(L2**2 / 4 + s**2)) / 2

    def lam_mp(P):
        P = [mp.matrix([mp.mpf(float(x)) for x in p]) for p in P]
        W = mp.inf
        for i in range(4):
            for j in range(i + 1, 4):
                e = P[j] - P[i]
                L = mp.sqrt(e[0]**2 + e[1]**2)
                nrm = mp.matrix([-e[1] / L, e[0] / L])
                pr = [((P[k] - P[i]).T * nrm)[0] for k in range(4)]
                W = min(W, max(pr) - min(pr))
        return min(W / 2, pattern(P[0], P[1], P[2], P[3]), pattern(P[0], P[2], P[1], P[3]),
                   pattern(P[0], P[3], P[1], P[2]))

    def sin_between(P, i, j, k, l):
        e1, e2 = P[j] - P[i], P[l] - P[k]
        return abs(e1[0] * e2[1] - e1[1] * e2[0]) / (np.hypot(*e1) * np.hypot(*e2))

    rng = np.random.default_rng(3)
    worst, n = 0.0, 0
    for k in range(300):
        Rc = 10 ** rng.uniform(2.5, 6)
        ang = rng.uniform(-3, 3) + np.sort(rng.uniform(0, 1100 / Rc, 4))
        P = Rc * np.stack([np.cos(ang), np.sin(ang)], 1) + rng.normal(0, 10 ** rng.uniform(-3, 1), (4, 2))
        P += rng.uniform(-1000, 1000, 2) - P.mean(0)
        D = [sin_between(P, *q) for q in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))]
        if any(0.5 * SM.MIN_SIN < d < 2 * SM.MIN_SIN for d in D):       # either side of the guard: not comparable
            continue
        worst = max(worst, abs(SM.lam4_raw(*P.ravel()) - float(lam_mp(P))))
        n += 1
    assert n > 250 and worst < 1e-6 < SM.MARGIN_MM, (n, worst)


# ------------------------------------------------------------------------------------------------ candidates, bounds
@pytest.mark.parametrize("seed", [0, 1])
def test_candidates_are_the_original_ones(seed, monkeypatch):
    r, phi, z, lab = _event(seed)
    for kw in ({}, {"max_dphi": 0.1, "max_gap_r": 150.0}, {"max_dphi": 0.8, "max_gap_r": 900.0},
               {"max_dphi": np.inf, "max_gap_r": np.nan}, {"max_gap_r": -30.0}):
        got = {"cand": []}

        def capture(cand, trk, labels, *a, **k):
            got["cand"] = cand
            return labels
        monkeypatch.setattr(MF, "_merge", capture)
        merge_fragments(r, phi, z, lab, mode="sequential", **kw)
        monkeypatch.undo()
        vol, trk, tab, ia, ib = _setup(r, phi, z, lab, max_dphi=kw.get("max_dphi", 0.3),
                                       max_gap_r=kw.get("max_gap_r", 400.0))
        ref = np.array([(int(a), int(b)) for a, b in got["cand"]], dtype=np.int64).reshape(-1, 2)
        assert len(ref) > 100 or kw.get("max_gap_r") == -30.0
        assert np.array_equal(ref, np.stack([trk[ia], trk[ib]], 1))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bounds_hold_on_every_candidate(seed):
    """lb <= n rms^2 and tau <= n_out for the actual fit of every union (any cut values depend only on these)."""
    r, phi, z, lab = _event(seed)
    vol, trk, tab, ia, ib = _setup(r, phi, z, lab)
    keep, lb, tau, span = SM.prefilter(tab, ia, ib, np.inf, np.inf, -np.inf, np.inf, full=True)
    f = _fit_all(tab, ia, ib, r, phi, z, vol)
    n = (tab.fn[ia] + tab.fn[ib]).astype(np.float64)
    assert (lb <= n * f["rms"] ** 2 * (1 + 1e-9) + 1e-12).all()
    assert (tau <= f["n_out"]).all()
    # the bounds are informative here (else the test proves little)
    assert (tau >= 2).mean() > 0.05 and (lb > n).mean() > 0.05, ((tau >= 2).mean(), (lb > n).mean())
    # early stopping gives the same decision as the complete bounds
    for max_rms, max_out in ((1.0, 1), (0.5, 0), (3.0, 2), (np.inf, 1), (1.0, np.inf)):
        k_full, *_ = SM.prefilter(tab, ia, ib, max_rms, max_out, 50.0, 5.0, full=True)
        k_fast, *_ = SM.prefilter(tab, ia, ib, max_rms, max_out, 50.0, 5.0)
        assert np.array_equal(k_full, k_fast)


# ------------------------------------------------------------------------------------------------ batch effects
def test_replica_is_fit_tracks_bit_for_bit():
    r, phi, z, lab = _event(4)
    vol, trk, tab, ia, ib = _setup(r, phi, z, lab)
    rng = np.random.default_rng(0)
    for size in (50, 400):
        pick = np.sort(rng.choice(len(ia), size, replace=False))
        rows, plab = SM.union_rows(tab, ia[pick], ib[pick])
        f = PA.fit_tracks(r[rows], phi[rows], z[rows], vol[rows], plab, min_hits=3)
        d = SM._line_sz_done(r[rows], phi[rows], z[rows], vol[rows], plab)
        for k_f, k_d in (("rms", "rms"), ("n_out", "n_out"), ("zrms", "zrms"), ("z0", "z0"), ("cot", "cot")):
            assert np.array_equal(f[k_f], d[k_d]), k_f


@pytest.mark.parametrize("seed,sample", [(0, 64), (5, 64), (1, 0), (2, 100000)])
def test_pinned_batch_gives_the_full_batch_fits(seed, sample):
    """Every pair of F' = F + h + W is fit as in the batch of all candidates: the transverse fit bit for bit always;
    the s-z fit bit for bit when K is pinned, else (certificate) bit for bit at the full batch's own K, which lies in the
    certified range."""
    r, phi, z, lab = _event(seed)
    vol, trk, tab, ia, ib = _setup(r, phi, z, lab)
    cuts = (1.0, 5.0, 1, 50.0)
    keep, _, _, span = SM.prefilter(tab, ia, ib, cuts[0], cuts[2], cuts[3], cuts[1])
    sel, info = SM.pin(tab, ia, ib, keep, span, r, phi, z, vol, *cuts, sample=sample)
    full = _fit_all(tab, ia, ib, r, phi, z, vol)
    part = _fit_all(tab, ia[sel], ib[sel], r, phi, z, vol)
    for key in ("rms", "n_out", "d0", "R", "xc", "yc", "sigma"):
        assert np.array_equal(full[key][sel], part[key], equal_nan=True), key
    assert info["H"] == info["H_replica"] == (tab.fn[ia] + tab.fn[ib]).max()
    rows, plab = SM.union_rows(tab, ia, ib)
    K_C = SM._line_sz_done(r[rows], phi[rows], z[rows], vol[rows], plab)["K"]
    if info["K_proof"] == "certificate":
        assert info["K"] <= K_C
        rows, plab = SM.union_rows(tab, ia[sel], ib[sel])
        d = SM._line_sz_done(r[rows], phi[rows], z[rows], vol[rows], plab)
        assert np.array_equal(d["zrms_hist"][:, K_C - 1], full["zrms"][sel], equal_nan=True)
    else:
        assert info["K"] == K_C
        for key in ("zrms", "z0", "cot"):
            assert np.array_equal(full[key][sel], part[key], equal_nan=True), key


# ------------------------------------------------------------------------------------------------ end to end
CUTS = [{}, {"max_rms": 0.3}, {"max_rms": 3.0, "max_out": 2}, {"max_rms": 1.0, "max_out": 0, "max_zrms": 1.0},
        {"max_rms": np.inf, "max_out": 1}, {"max_rms": 2.0, "max_out": 5, "max_zrms": 50.0},
        {"max_dphi": 0.1, "max_gap_r": 150.0}, {"max_dphi": 0.6, "max_gap_r": 800.0, "max_rms": 1.5},
        {"min_span_mm": 45.0}]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_merge_sequential_is_merge_fragments(seed):
    r, phi, z, lab = _event(seed)
    merged_any = False
    for kw in CUTS:
        ref = merge_fragments(r, phi, z, lab, mode="sequential", **kw)
        info = {}
        got = SM.merge_sequential(r, phi, z, lab, info=info, check_ties=True, **kw)
        assert np.array_equal(ref, got), (kw, info)
        assert got.dtype == ref.dtype
        assert info.get("finite_q_ties", 0) == 0 and info.get("mutual_pairs_kept", 0) == 0
        merged_any |= bool((ref != lab).any())
        assert info.get("kept", 0) < info["candidates"]
    assert merged_any


def test_merge_sequential_after_any_pass_and_trivial_inputs():
    r, phi, z, lab = _event(6)
    lab2 = merge_fragments(r, phi, z, lab, mode="any")
    assert np.array_equal(merge_fragments(r, phi, z, lab2), SM.merge_sequential(r, phi, z, lab2))
    none = np.full(len(r), -1)
    assert np.array_equal(SM.merge_sequential(r, phi, z, none), none)
    one = none.copy()
    one[:5] = 3
    assert np.array_equal(SM.merge_sequential(r, phi, z, one), one)


# ------------------------------------------------------------------------------------------------ cut sets, pre-fit
TABLE = "configs/trackml/merge_cuts_v1.tsv"


def _table():
    from hnl.trackml.config import repo_path
    return SM.BinnedCuts.from_tsv(repo_path(TABLE))


def test_cut_table_and_bins():
    c = _table()
    assert len(c.max_rms) == 10 and np.allclose(c.edges, np.arange(11) * 0.4)
    assert c.max_out.tolist() == [5, 5, 8, 10, 11, 11, 10, 10, 1, 1] and not c.scalar
    eta = np.array([0.0, 0.39, 0.41, 1.99, 2.01, 3.99, 4.0, 7.0, np.nan])
    assert c.bin(np.sinh(eta)).tolist() == [0, 0, 1, 4, 5, 9, 9, 9, 9]
    assert c.bin(-np.sinh(eta[:8])).tolist() == [0, 0, 1, 4, 5, 9, 9, 9]
    h = SM.BinnedCuts.uniform()
    assert h.scalar and h.for_cot(np.zeros(3)) == (1.0, 5.0, 1.0)


def _mixed_table():
    """A table whose bins differ a lot (tight, loose, very loose) over the |eta| range of _event."""
    return SM.BinnedCuts([0, 0.3, 0.7, 1.1, 1.5, np.inf], [1.0, 0.4, 3.0, 1.0, 8.0], [5.0, 2.0, 20.0, 5.0, 50.0],
                         [1, 0, 4, 2, 9], name="mixed")


@pytest.mark.parametrize("seed", [0, 3])
def test_methods_are_merge_fragments(seed):
    r, phi, z, lab = _event(seed)
    for kw in CUTS[:6]:
        ref = merge_fragments(r, phi, z, lab, mode="sequential", **kw)
        for method in ("prefit", "pinned"):
            info = {}
            got = SM.merge_sequential(r, phi, z, lab, method=method, info=info, **kw)
            assert np.array_equal(ref, got), (method, kw, info)
            if info.get("method") == "prefit" and "consistent" in info:
                assert info["consistent"] and info["finite_q_ties"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cut_sets_match_the_unfiltered_definition(seed):
    r, phi, z, lab = _event(seed)
    lab = merge_fragments(r, phi, z, lab, mode="any")
    hnl_as_table = SM.BinnedCuts(np.arange(11) * 0.4, [1.0] * 10, [5.0] * 10, [1] * 10, name="H-table")
    for cuts in (_table(), _mixed_table(), hnl_as_table):
        ref = SM.merge_sequential_reference(r, phi, z, lab, cuts=cuts)
        info = {}
        got = SM.merge_sequential(r, phi, z, lab, cuts=cuts, info=info)
        assert np.array_equal(ref, got), (cuts.name, info)
        assert info.get("consistent", True), info
    assert np.array_equal(SM.merge_sequential_reference(r, phi, z, lab, cuts=hnl_as_table),
                          merge_fragments(r, phi, z, lab, mode="sequential"))
    info = {}                                           # too small a witness sample: K(C) found exactly
    got = SM.merge_sequential(r, phi, z, lab, cuts=_table(), sample=2, info=info)
    assert np.array_equal(SM.merge_sequential_reference(r, phi, z, lab, cuts=_table()), got) and info["consistent"]


@pytest.mark.parametrize("seed", [0, 5])
def test_prefit_is_the_full_batch_bit_for_bit(seed):
    r, phi, z, lab = _event(seed)
    vol, trk, tab, ia, ib = _setup(r, phi, z, lab)
    keep, *_ = SM.prefilter(tab, ia, ib, 1.0, 1, 50.0, 5.0)
    sizes = tab.fn[ia] + tab.fn[ib]
    h = int(np.flatnonzero(sizes == sizes.max())[0])
    W, K, how = SM._witnesses(tab, ia, ib, keep, h, r, phi, z, vol, 4096 if seed else 8, 0)
    rows, plab = SM.union_rows(tab, ia, ib)
    assert K == SM._line_sz_done(r[rows], phi[rows], z[rows], vol[rows], plab)["K"], (K, how)
    full = _fit_all(tab, ia, ib, r, phi, z, vol)
    members = np.arange(len(ia))
    vals = SM.prefit(tab, ia, ib, members, h, W, r, phi, z, vol, batch=257)
    for key in ("rms", "zrms", "n_out", "cot", "z0", "d0", "R"):
        assert np.array_equal(vals[key], full[key], equal_nan=True), key
    assert vals["batches"] > len(vals["class_widths"])          # several classes and batches were used


def test_same_sums_classes():
    """numpy's row sums are bit-identical for padded widths with the same summation signature, up to and beyond the
    128-entry block of the pairwise summation (where it splits recursively)."""
    rng = np.random.default_rng(2)
    for m in (3, 5, 8, 9, 12, 17, 24, 30, 61, 100, 140):
        x = rng.normal(size=(300, m)) * np.exp(rng.normal(0, 3, size=(300, 1)))
        for H in (30, 33, 44, 64, 129, 150, 203, 260, 300):
            if H < m:
                continue
            ref = np.zeros((300, H))
            ref[:, :m] = x
            want = np.sum(ref, 1)
            n_same = 0
            for Wd in range(m, H + 1):
                if SM._sum_sig(Wd, m) == SM._sum_sig(H, m):
                    p = np.zeros((300, Wd))
                    p[:, :m] = x
                    assert np.array_equal(np.sum(p, 1), want), (m, H, Wd)
                    n_same += 1
            assert n_same >= 1
    # and the rule prefit uses picks narrow widths for short unions even when the longest union exceeds 128 hits
    assert SM.same_sums(17, 10, 150) and SM.same_sums(10, 4, 150) and not SM.same_sums(100, 100, 150)


def test_any_pass_with_cut_sets():
    from hnl.trackml.trackbuild import merge_fragments_fast
    r, phi, z, lab = _event(4)
    hnl_as_table = SM.BinnedCuts(np.arange(11) * 0.4, [1.0] * 10, [5.0] * 10, [1] * 10, name="H-table")
    ref = merge_fragments(r, phi, z, lab, mode="any")
    assert np.array_equal(merge_fragments_fast(r, phi, z, lab, mode="any", cuts=hnl_as_table), ref)
    loose = merge_fragments_fast(r, phi, z, lab, mode="any", cuts=_mixed_table())
    assert len(np.unique(loose[loose >= 0])) != len(np.unique(ref[ref >= 0]))


def test_grid_configs_carry_the_cut_set():
    from hnl.trackml import trackbuild as TBM
    plain = TBM.grid_configs(["cc", "walkmergeX", "mwslayermergeX", "mws2passmergeX"])
    both = TBM.grid_configs(["cc", "walkmergeX", "mwslayermergeX", "mws2passmergeX"], ("H", "C"))
    n_merge = sum(c[0] in TBM.MERGE_FAMILIES for c in plain)
    assert n_merge == 28 and len(both) == len(plain) + n_merge
    assert {TBM.cut_set_of(c) for c in both if c[0] in TBM.MERGE_FAMILIES} == {"H", "C"}
    assert all(TBM.cut_set_of(c) == "H" for c in plain)
    spec = [{"name": "x", "family": c[0], "params": ",".join(f"{v:g}" for v in c[1:])} for c in both]
    cfgs, _ = TBM.explicit_configs(__import__("json").dumps(spec))
    assert [TBM.cut_set_of(c) for c in cfgs] == [TBM.cut_set_of(c) for c in both]


def test_prefit_with_unions_longer_than_128_hits():
    """A 150-hit fragment makes the batch of all candidates wider than numpy's 128-entry summation block (recursive
    pairwise summation): the pre-fit must still reproduce it bit for bit, and the pass the unfiltered definition."""
    r, phi, z, lab = _event(7)
    rng = np.random.default_rng(11)
    rr = np.linspace(60.0, 1000.0, 150)
    ph = 0.02 + 1e-4 * rr + rng.normal(0, 1e-4, 150)            # a long, slightly curved fragment near phi = 0
    zz = 0.3 * rr + rng.normal(0, 0.2, 150)
    r, phi, z = np.r_[r, rr], np.r_[phi, ph], np.r_[z, zz]
    lab = np.r_[lab, np.full(150, lab.max() + 1)]
    vol, trk, tab, ia, ib = _setup(r, phi, z, lab)
    sizes = tab.fn[ia] + tab.fn[ib]
    assert sizes.max() > 128
    keep, *_ = SM.prefilter(tab, ia, ib, 1.0, 1, 50.0, 5.0)
    h = int(np.flatnonzero(sizes == sizes.max())[0])
    W, K, how = SM._witnesses(tab, ia, ib, keep, h, r, phi, z, vol, 4096, 0)
    full = _fit_all(tab, ia, ib, r, phi, z, vol)
    vals = SM.prefit(tab, ia, ib, np.arange(len(ia)), h, W, r, phi, z, vol, batch=500)
    for key in ("rms", "zrms", "n_out", "cot", "z0", "d0", "R"):
        assert np.array_equal(vals[key], full[key], equal_nan=True), key
    assert min(vals["class_widths"].values()) < 128                # short unions still go into narrow batches
    for cuts in (None, _table()):
        ref = SM.merge_sequential_reference(r, phi, z, lab, cuts=cuts)
        assert np.array_equal(ref, SM.merge_sequential(r, phi, z, lab, cuts=cuts))


def test_any_pass_parallel_and_prefit_binning(monkeypatch):
    """The any-order pass with a cut set: candidates searched in parallel blocks, |eta| bins from the exact pre-fit;
    the same as the definition (candidates of the original, bins from a fit of _merge's very batch)."""
    from hnl.trackml import trackbuild as TBM
    r, phi, z, lab = _event(8)
    vol = _volume_from_radius(r, z)
    R = np.hypot(r, z)
    ok = lab >= 0
    trk, cnt = np.unique(lab[ok], return_counts=True)
    trk = trk[cnt >= 2]
    hits = TBM._tracks_of(lab, trk)
    c1 = TBM._candidates_any_fast(trk, hits, r, phi, z, R, 0.2, 0.2, block=37)
    c3 = TBM._candidates_any_fast(trk, hits, r, phi, z, R, 0.2, 0.2, block=37, n_proc=3)
    ref_c = MF._candidates_any(list(trk), lab, r, phi, z, R, 0.2, 0.2)
    assert c1 == c3 and [(int(a), int(b)) for a, b in c1] == [(int(a), int(b)) for a, b in ref_c] and len(c1) > 50
    cuts = _mixed_table()
    rows = [np.sort(np.concatenate([hits[int(a)], hits[int(b)]])) for a, b in c1]
    plab = np.concatenate([np.full(len(x), k) for k, x in enumerate(rows)])
    f = PA.fit_tracks(r[np.concatenate(rows)], phi[np.concatenate(rows)], z[np.concatenate(rows)],
                      vol[np.concatenate(rows)], plab, min_hits=3)
    assert np.array_equal(TBM.any_cot(r, phi, z, lab, trk, vol, c1), f["cot"])
    assert np.array_equal(TBM.any_cot(r, phi, z, lab, trk, vol, c1, n_proc=3), f["cot"])
    want = MF._merge(c1, trk, lab.copy(), r, phi, z, vol, *cuts.for_cot(f["cot"]), one_per_end=False)
    for procs in (1, 3):
        monkeypatch.setattr(TBM, "PREFIT_PROCS", procs)
        got = TBM.merge_fragments_fast(r, phi, z, lab, mode="any", max_dphi_dir=0.2, max_dcot=0.2, cuts=cuts)
        assert np.array_equal(got, want)
        seq = TBM.merge_fragments_fast(r, phi, z, got, cuts=cuts)
        assert np.array_equal(seq, SM.merge_sequential_reference(r, phi, z, got, cuts=cuts))
