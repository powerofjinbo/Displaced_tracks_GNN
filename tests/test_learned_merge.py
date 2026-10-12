"""Learned fragment merging (hnl.trackbuild.learned_merge)."""

import numpy as np

from hnl.trackbuild import learned_merge as LM


def _helix(R0, phi0, cot, n, r0=30.0, r1=900.0, charge=1, x0=0.0, y0=0.0, z0=0.0):
    """Hits of a helix through (x0, y0, z0) at radii r0..r1 from that point (transverse radius R0 mm)."""
    t = np.linspace(r0, r1, n) / R0
    cx, cy = x0 - charge * R0 * np.sin(phi0), y0 + charge * R0 * np.cos(phi0)
    ang = phi0 - charge * np.pi / 2 + charge * t
    x, y = cx + R0 * np.cos(ang), cy + R0 * np.sin(ang)
    z = z0 + cot * R0 * t
    return x, y, z


def _event():
    xa, ya, za = _helix(3000.0, 0.3, 0.5, 10)
    xb, yb, zb = _helix(2000.0, 2.0, -0.3, 10)
    x, y, z = np.r_[xa, xb], np.r_[ya, yb], np.r_[za, zb]
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    lab = np.array([0] * 5 + [1] * 5 + [2] * 10)           # particle A split 5 + 5, particle B whole
    pid = np.array([7] * 10 + [9] * 10)
    vol = np.where(r < 200, 8, np.where(r < 720, 13, 17))
    geo = {"volume": vol, "phi_c": phi.copy(), "layer_id": vol * 1000 + np.arange(20) % 10}
    return x, y, z, r, phi, lab, pid, geo


def test_candidates_and_labels():
    x, y, z, r, phi, lab, pid, geo = _event()
    pairs = LM.candidates(lab, r, phi, z)
    s = {tuple(p) for p in pairs.tolist()}
    assert (0, 1) in s                      # end-to-end candidates may add (0, 2) / (1, 2): the model decides
    y_ = LM.labels_for(lab, pairs, pid)
    lst = [tuple(p) for p in pairs.tolist()]
    assert y_[lst.index((0, 1))] and not any(y_[lst.index(p)] for p in ((0, 2), (1, 2)) if p in lst)


def test_features_invariant_under_translation_and_rotation():
    x, y, z, r, phi, lab, pid, geo = _event()
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    X0 = LM.features(lab, pairs, r, phi, z, geo)
    a = 0.8
    xr, yr = np.cos(a) * x - np.sin(a) * y + 40.0, np.sin(a) * x + np.cos(a) * y - 25.0
    zr = z + 60.0
    geo2 = dict(geo, phi_c=geo["phi_c"] + a)
    X1 = LM.features(lab, pairs, np.hypot(xr, yr), np.arctan2(yr, xr), zr, geo2)
    keep = [k for k, f in enumerate(LM.FEATURES) if not f.startswith("sub_")]
    assert np.allclose(np.nan_to_num(X0[:, keep], nan=-9), np.nan_to_num(X1[:, keep], nan=-9), rtol=1e-6, atol=1e-6)
    f = dict(zip(LM.FEATURES, X0[0]))
    assert f["u_rms_n"] < 0.5 and f["dkappa"] < 0.05 and f["gap_s"] > 0 and f["shared_layers"] == 0


def test_apply_respects_tau_and_shared_layers():
    lab = np.array([0, 0, 1, 1, 2, 2])
    lay = np.array([1, 2, 3, 4, 1, 5])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    p = np.array([0.9, 0.95, 0.2])
    out = LM.apply(lab, pairs, p, 0.5, lay)
    # (0, 2) comes first but shares layer 1 once (allowed with max_shared 1): merged; then (0, 1): merged
    assert len(set(out.tolist())) == 1
    out = LM.apply(lab, pairs, p, 0.5, lay, max_shared=0)
    assert out[0] == out[2] and out[0] != out[4]
    assert np.array_equal(LM.apply(lab, pairs, p, 0.99, lay), lab)


def test_end_candidates_find_a_sequential_split():
    # one helix split 4 + 4 with a gap; the line window may miss short fragments, the end search must not
    x, y, z = _helix(2500.0, 0.7, 0.2, 8, r0=40.0, r1=700.0)
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    lab = np.array([0] * 4 + [1] * 4)
    assert (0, 1) in {tuple(p) for p in LM.end_candidates(lab, r, phi, z).tolist()}
    assert (0, 1) in {tuple(p) for p in LM.candidates(lab, r, phi, z).tolist()}
