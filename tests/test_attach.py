"""Graph-free hit attachment: candidates in the track's helix frame, labels, decoding."""

import numpy as np
import pytest

from hnl.trackbuild import attach as AT

LAYERS = np.array([32.0, 72, 116, 172, 260, 360, 500, 660, 820])
VOLS = np.array([8, 8, 8, 8, 13, 13, 13, 13, 17])


def _helix(n=9, R=4000.0, phi0=0.3, cot=0.4, sigma=1, x0=0.0, y0=0.0):
    layers = LAYERS[:n]
    c = np.array([x0 - sigma * R * np.sin(phi0), y0 + sigma * R * np.cos(phi0)])
    t = np.linspace(0, 1000.0 / R, 200001)
    xs = c[0] + sigma * R * np.sin(phi0 + sigma * t)
    ys = c[1] - sigma * R * np.cos(phi0 + sigma * t)
    tl = np.interp(layers, np.hypot(xs, ys), t)
    x = c[0] + sigma * R * np.sin(phi0 + sigma * tl)
    y = c[1] - sigma * R * np.cos(phi0 + sigma * tl)
    return np.hypot(x, y), np.arctan2(y, x), cot * R * tl


def _event():
    r, phi, z = _helix()
    r2, phi2, z2 = _helix(phi0=2.0, x0=40.0, y0=-30.0, sigma=-1)       # a displaced second track far away in phi
    r = np.r_[r, r2, 400.0]
    phi = np.r_[phi, phi2, -1.0]
    z = np.r_[z, z2, 0.0]
    vol = np.r_[VOLS, VOLS, 13]
    geo = {"volume": vol, "phi_c": phi.copy(), "layer_id": vol * 1000 + np.r_[np.arange(9), np.arange(9), 5]}
    return r, phi, z, geo


def test_candidates_find_the_missing_hit_of_a_track():
    r, phi, z, geo = _event()
    pid = np.r_[np.full(9, 7), np.full(9, 8), 99]
    labels = np.r_[np.zeros(9, int), np.ones(9, int), -1]
    labels[4] = -1                                        # hit 4 of track 0 was lost by the first pass
    labels[9 + 8] = -1                                    # last hit of track 1 too
    for g in (geo, None):                                 # module geometry, or the fallback from the radius
        rows, X = AT.candidates(labels, r, phi, z, geo=g)
        pairs = {tuple(x) for x in rows.tolist()}
        assert (0, 4) in pairs and (1, 17) in pairs and not any(h == 18 for _, h in pairs)
        k = [i for i, x in enumerate(rows.tolist()) if tuple(x) == (0, 4)][0]
        f = dict(zip(AT.FEATURES, X[k]))
        assert f["pull_t"] < 1e-3 and f["pull_z"] < 1e-3 and f["inside"] == 1.0
        assert f["same_layer"] == 0.0 if g is not None else np.isnan(f["same_layer"])     # no layer ids: missing
    y = AT.labels_for(rows, labels, pid)
    assert y[k] and y.sum() == 2
    new = AT.apply(labels, rows, y.astype(float), 0.5)
    assert new[4] == 0 and new[17] == 1 and new[18] == -1


def test_features_invariant_under_translation_and_rotation():
    rng = np.random.default_rng(0)
    r, phi, z, geo = _event()
    x, y = r * np.cos(phi), r * np.sin(phi)
    x, y, z = x + rng.normal(0, 0.02, len(x)), y + rng.normal(0, 0.02, len(x)), z + rng.normal(0, 0.3, len(x))
    labels = np.r_[np.zeros(9, int), np.ones(9, int), -1]
    labels[[3, 6, 9 + 2]] = -1
    rows, X = AT.candidates(labels, np.hypot(x, y), np.arctan2(y, x), z, geo=geo)
    assert len(rows) >= 3
    for k in range(5):
        ang = rng.uniform(-np.pi, np.pi)
        dx, dy, dz = rng.normal(0, 60, 3)
        xr, yr = np.cos(ang) * x - np.sin(ang) * y + dx, np.sin(ang) * x + np.cos(ang) * y + dy
        geo2 = dict(geo, phi_c=geo["phi_c"] + ang)          # the module frames turn with the hits
        rows2, X2 = AT.candidates(labels, np.hypot(xr, yr), np.arctan2(yr, xr), z + dz, geo=geo2)
        assert np.array_equal(rows, rows2)
        assert np.allclose(X, X2, rtol=1e-8, atol=1e-9), k


def test_score_ablation_and_first_pass():
    r, phi, z, geo = _event()
    labels = np.r_[np.zeros(9, int), np.ones(9, int), -1]
    labels[4] = -1
    a, b, s = np.array([3, 4]), np.array([4, 5]), np.array([0.05, 0.7])
    rows, X = AT.candidates(labels, r, phi, z, a, b, s, geo=geo, with_score=True)
    assert X.shape[1] == len(AT.FEATURES) + 1
    k = [i for i, x in enumerate(rows.tolist()) if tuple(x) == (0, 4)][0]
    assert X[k, -1] == 0.7
    with pytest.raises(ValueError):
        AT.first_pass(19, a, b, s, np.hypot(r, z), (r, phi, z), "mwslayer", None)
    lab = AT.first_pass(19, a, b, s, np.hypot(r, z), (r, phi, z), "mwslayer", geo["layer_id"])
    assert lab[4] == lab[5] >= 0 and lab[3] == -1                # only the 0.7 edge is above s0 = 0.1
