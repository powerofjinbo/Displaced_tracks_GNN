"""Weighted robust helix fit in the hits' module frames."""

import numpy as np

from hnl.geometry import resolution as RS
from hnl.trackbuild import helix_fit as HF

RADII = np.array([32.0, 72, 116, 172, 260, 360, 500, 660, 820, 1020])
VOLS = np.array([8, 8, 8, 8, 13, 13, 13, 13, 17, 17])


def toy(R0=3000.0, phi0=0.4, cot=0.3, x0=5.0, y0=-3.0, z0=20.0, sign=1, rng=None):
    """Hits of a helix on the barrel layers (module axes at the hit azimuth), smeared along the module axes."""
    c = np.array([x0 - sign * R0 * np.sin(phi0), y0 + sign * R0 * np.cos(phi0)])
    t = np.linspace(0, 1200 / R0, 400001)
    tl = np.interp(RADII, np.hypot(c[0] + sign * R0 * np.sin(phi0 + sign * t),
                                   c[1] - sign * R0 * np.cos(phi0 + sign * t)), t)
    x = c[0] + sign * R0 * np.sin(phi0 + sign * tl)
    y = c[1] - sign * R0 * np.cos(phi0 + sign * tl)
    z = z0 + cot * R0 * tl
    pc = np.arctan2(y, x)
    if rng is not None:
        u, v = RS.axes(VOLS, pc)
        su, sv = RS.sigmas(VOLS)
        du, dv = rng.normal(size=len(x)) * su, rng.normal(size=len(x)) * sv
        x, y, z = x + du * u[:, 0] + dv * v[:, 0], y + du * u[:, 1] + dv * v[:, 1], z + dv * v[:, 2]
    return x, y, z, pc, c


def test_exact_helix_is_recovered():
    x, y, z, pc, c = toy()
    f = HF.fit(x, y, z, np.zeros(len(x), int), VOLS, pc)
    assert abs(f["R"][0] - 3000.0) < 1e-3 and np.allclose([f["xc"][0], f["yc"][0]], c, atol=1e-3)
    assert abs(f["cot"][0] - 0.3) < 1e-6 and f["rms_n"][0] < 1e-3 and f["n_out"][0] == 0


def test_fit_commutes_with_translations_and_rotations():
    rng = np.random.default_rng(1)
    x, y, z, pc, _ = toy(rng=rng)
    lab = np.zeros(len(x), int)
    f = HF.fit(x, y, z, lab, VOLS, pc)
    ang = 0.9
    ca, sa = np.cos(ang), np.sin(ang)
    g = HF.fit(ca * x - sa * y + 40.0, sa * x + ca * y - 25.0, z + 13.0, lab, VOLS, pc + ang)
    for k in ("R", "cot", "rms_n", "zrms_n", "dtheta_min", "dtheta_max"):
        assert np.allclose(f[k], g[k], rtol=1e-8, atol=1e-10), k
    assert np.allclose(f["pull_t"], g["pull_t"], atol=1e-8) and np.allclose(f["pull_z"], g["pull_z"], atol=1e-8)


def test_outlier_hit_does_not_pull_the_fit():
    rng = np.random.default_rng(2)
    x, y, z, pc, _ = toy(rng=rng)
    lab = np.zeros(len(x), int)
    ref = HF.fit(x, y, z, lab, VOLS, pc)
    x2 = x.copy()
    x2[5] += 5.0                                               # one hit 5 mm off the helix
    f = HF.fit(x2, y, z, lab, VOLS, pc)
    assert abs(f["R"][0] / ref["R"][0] - 1) < 0.01 and abs(f["pull_t"][0][5]) > 50
    assert np.abs(np.delete(f["pull_t"][0], 5)).max() < 5 and f["n_out"][0] >= 1


def test_leverage_corrected_pulls_have_unit_width():
    rng = np.random.default_rng(3)
    pt, pz = [], []
    for _ in range(300):
        x, y, z, pc, _ = toy(R0=rng.uniform(2000, 20000), phi0=rng.uniform(-3, 3), cot=rng.uniform(-1, 1),
                             sign=rng.choice([-1, 1]), rng=rng)
        f = HF.fit(x, y, z, np.zeros(len(x), int), VOLS, pc, leverage=True)
        pt.append(f["pull_t"][0])
        pz.append(f["pull_z"][0])
    assert 0.93 < np.concatenate(pt).std() < 1.07 and 0.93 < np.concatenate(pz).std() < 1.07
