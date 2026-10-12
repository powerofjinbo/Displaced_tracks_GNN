"""Toy-helix tests of hnl.truth.helix: perigee convention, fixed-/free-R fits, s-z fit, vertex, chain order."""

import numpy as np
import pytest

from hnl.truth import helix as hx


def toy_helix(d0, phi0, R, sigma, z0, cot, s, noise_xy=0.0, noise_z=0.0, rng=None):
    """Points of a helix with perigee (d0, phi0) under the module's sign convention, at arc lengths s."""
    n = np.array([-np.sin(phi0), np.cos(phi0)])          # z x t
    p = -d0 * n                                          # PCA: p x t = d0
    c = p + sigma * R * n
    a0 = np.arctan2(p[1] - c[1], p[0] - c[0])
    a = a0 + sigma * s / R
    x, y = c[0] + R * np.cos(a), c[1] + R * np.sin(a)
    z = z0 + cot * s
    if rng is not None:
        x = x + rng.normal(0, noise_xy, len(s))
        y = y + rng.normal(0, noise_xy, len(s))
        z = z + rng.normal(0, noise_z, len(s))
    return np.stack([x, y, z], 1), c


def pad_one(tracks):
    tid = np.concatenate([np.full(len(t), i) for i, t in enumerate(tracks)])
    return hx.pad_tracks(tid, np.concatenate(tracks))


def test_perigee_convention_exact():
    for d0 in (-50.0, -0.5, 0.0, 3.0, 80.0):
        for sigma in (1, -1):
            for phi0 in (-2.0, 0.3, 2.9):
                R = 2000.0
                pts, c = toy_helix(d0, phi0, R, sigma, 0.0, 0.0, np.array([10.0, 100.0]))
                pg = hx.perigee(np.array([c[0]]), np.array([c[1]]), np.array([R]), np.array([sigma]))
                assert pg.d0[0] == pytest.approx(d0, abs=1e-9)
                dphi = np.angle(np.exp(1j * (pg.phi0[0] - phi0)))
                assert abs(dphi) < 1e-9


@pytest.mark.parametrize("pt", [1.0, 5.0, 40.0])
@pytest.mark.parametrize("d0", [0.0, -2.0, 15.0, -90.0])
def test_fixed_r_fit_recovers_d0(pt, d0):
    rng = np.random.default_rng(0)
    R = hx.MM_PER_GEV * pt
    tracks, truth = [], []
    for k in range(40):
        sigma = 1 if k % 2 else -1
        phi0 = rng.uniform(-np.pi, np.pi)
        s = np.sort(rng.uniform(40, 900, 10))
        pts, _ = toy_helix(d0, phi0, R, sigma, 5.0, 0.7, s, 0.02, 0.1, rng)
        tracks.append(pts)
        truth.append(sigma)
    p = pad_one(tracks)
    sig = np.array(truth)
    f = hx.fit_circle(p, np.full(len(tracks), R), sig, scale=0.05)
    assert f.converged.all()
    pg = hx.perigee(f.xc, f.yc, f.R, f.sigma)
    err = pg.d0 - d0
    assert np.median(np.abs(err)) < 0.05, err
    # the free-R fit finds the same rotation sense and pT
    g = hx.fit_circle(p, free_r=True, scale=0.05)
    assert np.array_equal(g.sigma, sig)
    assert np.median(np.abs(g.R / R - 1)) < (0.01 if pt < 10 else 0.1)


def test_wrong_rotation_sense_gives_bad_fit():
    rng = np.random.default_rng(1)
    R = hx.MM_PER_GEV * 5.0
    s = np.linspace(40, 900, 10)
    pts, _ = toy_helix(0.0, 0.4, R, 1, 0.0, 0.0, s, 0.02, 0.0, rng)
    p = pad_one([pts])
    good = hx.fit_circle(p, np.array([R]), np.array([1]), scale=0.05)
    bad = hx.fit_circle(p, np.array([R]), np.array([-1]), scale=0.05)
    assert good.rms[0] < 0.05 < bad.rms[0]


def test_sz_fit_and_arc_length():
    rng = np.random.default_rng(2)
    R = hx.MM_PER_GEV * 3.0
    s = np.sort(rng.uniform(30, 1500, 14))                # more than a quarter turn at 3 GeV
    pts, c = toy_helix(4.0, -1.1, R, -1, -37.0, 1.3, s, 0.01, 0.05, rng)
    p = pad_one([pts])
    f = hx.fit_circle(p, np.array([R]), np.array([-1]), scale=0.05)
    pg = hx.perigee(f.xc, f.yc, f.R, f.sigma)
    sfit = hx.arc_length(p, f.xc, f.yc, f.R, f.sigma, pg.x_pca, pg.y_pca)
    assert np.max(np.abs(sfit[0] - s)) < 0.5
    z0, ct, rms = hx.fit_line_sz(sfit, p.z, p.mask, scale=0.2)
    assert z0[0] == pytest.approx(-37.0, abs=0.2)
    assert ct[0] == pytest.approx(1.3, abs=1e-3)


def test_two_track_vertex():
    rng = np.random.default_rng(3)
    v = np.array([60.0, -35.0, 12.0])                      # true decay point
    trks, r_first = [], []
    for k, (pt, sigma, phi) in enumerate(((8.0, 1, 0.1), (5.0, -1, -0.9))):
        R = hx.MM_PER_GEV * pt
        t = np.array([np.cos(phi), np.sin(phi)])
        n = np.array([-t[1], t[0]])
        c = v[:2] + sigma * R * n
        a0 = np.arctan2(v[1] - c[1], v[0] - c[0])
        s = np.sort(rng.uniform(60, 800, 9))               # hits after the decay point
        a = a0 + sigma * s / R
        x, y = c[0] + R * np.cos(a), c[1] + R * np.sin(a)
        z = v[2] + 0.4 * (k + 1) * s
        pts = np.stack([x, y, z], 1) + rng.normal(0, [0.02, 0.02, 0.1], (len(s), 3))
        trks.append(pts)
        r_first.append(np.hypot(x[0], y[0]))
    p = pad_one(trks)
    R = np.array([hx.MM_PER_GEV * 8.0, hx.MM_PER_GEV * 5.0])
    f = hx.fit_circle(p, R, np.array([1, -1]), scale=0.05)
    pg = hx.perigee(f.xc, f.yc, f.R, f.sigma)
    s = hx.arc_length(p, f.xc, f.yc, f.R, f.sigma, pg.x_pca, pg.y_pca)
    z0, ct, _ = hx.fit_line_sz(s, p.z, p.mask, scale=0.3)
    h = [dict(xc=f.xc[[i]], yc=f.yc[[i]], R=f.R[[i]], sigma=f.sigma[[i]], x_pca=pg.x_pca[[i]],
              y_pca=pg.y_pca[[i]], z0=z0[[i]], cot=ct[[i]]) for i in (0, 1)]
    vx = hx.two_track_vertex(h[0], h[1], np.array([min(r_first)]))
    assert vx.ok[0]
    assert np.hypot(vx.x[0] - v[0], vx.y[0] - v[1]) < 1.0
    assert abs(vx.z[0] - v[2]) < 2.0


def test_chain_order():
    # two particles, rows shuffled; chains 3->0->5 (pid 7) and 1->4->2->6 (pid 9)
    pid = np.array([7, 9, 9, 7, 9, 7, 9])
    edges = np.array([[3, 0, 1, 4, 2], [0, 5, 4, 2, 6]])
    order = hx.order_hits(pid, edges)
    assert list(order) == [3, 0, 5, 1, 4, 2, 6]
    assert list(hx.chain_rank(7, edges)) == [1, 3, 1, 2, 2, 0, 0]


@pytest.mark.parametrize("pt", [0.8, 5.0, 60.0, 500.0])
@pytest.mark.parametrize("d0", [0.0, 7.0, -120.0])
def test_perigee_fit_free_and_fixed(pt, d0):
    rng = np.random.default_rng(4)
    R = hx.MM_PER_GEV * pt
    tracks, sig, phis = [], [], []
    for k in range(30):
        sigma = 1 if k % 2 else -1
        phi0 = rng.uniform(-np.pi, np.pi)
        s = np.sort(rng.uniform(40, 900, 9))
        pts, _ = toy_helix(d0, phi0, R, sigma, 0.0, 0.0, s, 0.02, 0.0, rng)
        tracks.append(pts)
        sig.append(sigma)
        phis.append(phi0)
    p = pad_one(tracks)
    sig = np.array(sig)
    f = hx.fit_circle(p, np.full(len(tracks), R), sig, scale=0.05)
    pg = hx.perigee(f.xc, f.yc, f.R, f.sigma)
    k0 = sig / R
    for free in (False, True):
        g = hx.fit_perigee(p, pg.d0, pg.phi0, k0, free_kappa=free, scale=0.05)
        assert g.converged.all()
        assert np.median(np.abs(g.d0 - d0)) < 0.05
        dphi = np.angle(np.exp(1j * (g.phi0 - np.array(phis))))
        assert np.max(np.abs(dphi)) < 1e-3
        if free:
            # curvature resolution: 9 hits, 0.02 mm, ~0.86 m lever arm -> sigma(kappa) ~ 1e-7 / mm
            assert np.median(np.abs(g.kappa - k0)) < 3e-7
            if pt <= 60:
                assert np.array_equal(g.sigma, sig)
    # the residual is smooth through kappa = 0: a straight line fit with kappa free stays finite
    e = hx.perigee_residuals(p, pg.d0, pg.phi0, np.zeros(len(tracks)))
    assert np.isfinite(e).all()


def test_arc_length_perigee_matches_centre_form():
    rng = np.random.default_rng(5)
    for pt, d0, sigma in ((0.9, 3.0, 1), (3.0, -40.0, -1), (300.0, 0.5, 1)):
        R = hx.MM_PER_GEV * pt
        s = np.sort(rng.uniform(30, 2500 if pt < 1 else 1000, 12))
        pts, c = toy_helix(d0, 0.7, R, sigma, 0.0, 0.0, s)
        p = pad_one([pts])
        sp = hx.arc_length_perigee(p, np.array([d0]), np.array([0.7]), np.array([sigma / R]))
        assert np.max(np.abs(sp[0] - s)) < 1e-6 * max(1.0, s.max())


def test_toy_propagation_consistent():
    from hnl.truth import toys
    rng = np.random.default_rng(6)
    n = 50
    d0 = rng.uniform(-100, 100, n)
    phi0 = rng.uniform(-np.pi, np.pi, n)
    kappa = rng.choice([-1, 1], n) / (hx.MM_PER_GEV * rng.uniform(1, 50, n))
    z0 = rng.normal(0, 50, n)
    cot = rng.uniform(-2, 2, n)
    for R in (120.0, 500.0, 1000.0):
        s = toys.s_at_radius(d0, kappa, R)
        x, y, _ = toys.position(d0, phi0, kappa, z0, cot, np.nan_to_num(s))
        ok = np.isfinite(s)
        assert ok.any()
        assert np.allclose(np.hypot(x[ok], y[ok]), R, atol=1e-6)
    # position agrees with the circle of the module's perigee convention, and s=0 is the PCA
    x0, y0, _ = toys.position(d0, phi0, kappa, z0, cot, np.zeros(n))
    assert np.allclose(np.hypot(x0, y0), np.abs(d0), atol=1e-9)
    pts = []
    for i in range(3):
        ss = np.linspace(50, 600, 8)
        xx, yy, zz = toys.position(np.full(8, d0[i]), np.full(8, phi0[i]), np.full(8, kappa[i]), np.full(8, z0[i]),
                                   np.full(8, cot[i]), ss)
        pts.append(np.stack([xx, yy, zz], 1))
    p = pad_one(pts)
    e = hx.perigee_residuals(p, d0[:3], phi0[:3], kappa[:3])
    assert np.max(np.abs(e[p.mask])) < 1e-6
    sp = hx.arc_length_perigee(p, d0[:3], phi0[:3], kappa[:3])
    assert np.allclose(sp[p.mask].reshape(3, 8), np.linspace(50, 600, 8)[None], atol=1e-6)
