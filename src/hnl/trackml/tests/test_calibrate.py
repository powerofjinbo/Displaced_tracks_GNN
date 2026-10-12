"""hnl.trackml.calibrate: the field estimator on exact uniform-field helices, the slope-window bookkeeping against
hnl.graphs.helix_union.candidate_triplets, the leg errors against helix_union.legs, the window constant, the field-map
filling and the union-fit acceptance on clean helices.

    pytest src/hnl/trackml/tests
"""

import numpy as np
import pandas as pd

from hnl.graphs import helix_union as HU
from hnl.trackml import calibrate as K


def _active_res():
    """The resolution table helix_union uses in this process (TrackML stages: configs/trackml/res_v1.json)."""
    return {v: (HU.FINE[v], HU.COARSE[v]) for v in HU.FINE}

RADII = [32.0, 72.0, 116.0, 172.0, 260.0, 360.0, 500.0, 660.0, 820.0, 1020.0]
VOLS = [8, 8, 8, 8, 13, 13, 13, 13, 17, 17]


def helix_event(n=30, B=2.0, seed=0, noise=0.0):
    """Hits of positive and negative helices from the origin in a uniform field B (Bz > 0: positive charges turn
    clockwise), with truth (exact points and momenta) and particles."""
    rng = np.random.default_rng(seed)
    rows, truth, parts = [], [], []
    hid = 0
    for j in range(n):
        pt, q = rng.uniform(2.5, 10.0), int(rng.choice([-1, 1]))
        phi0, cot = rng.uniform(-np.pi, np.pi), rng.uniform(-0.8, 0.8)
        Rc = pt / (0.3 * B) * 1000.0
        pid = j + 1
        for k, (r, v) in enumerate(zip(RADII, VOLS)):
            a = np.arcsin(r / (2 * Rc))
            ph = phi0 - q * a                        # clockwise for q > 0
            x, y, z = r * np.cos(ph), r * np.sin(ph), cot * 2 * Rc * a
            pdir = phi0 - q * 2 * a
            rows.append({"row": hid, "x": x, "y": y, "z": z + rng.normal(0, noise), "volume": v, "layer": 2 * k + 2,
                         "module": 1, "pid": pid, "rank": k})
            truth.append({"row": hid, "tx": x, "ty": y, "tz": z, "tpx": pt * np.cos(pdir), "tpy": pt * np.sin(pdir),
                          "tpz": pt * cot})
            hid += 1
        eta = np.arcsinh(cot)
        parts.append({"particle_id": pid, "q": q, "pt": pt, "eta": eta, "n_hits": len(RADII), "is_target": True,
                      "category": "primary", "rv": 0.0})
    h = pd.DataFrame(rows)
    h["r"] = np.hypot(h.x, h.y).astype(np.float32)
    h["phi"] = np.arctan2(h.y, h.x).astype(np.float32)
    for c in ("x", "y", "z"):
        h[c] = h[c].astype(np.float32)
    t = pd.DataFrame(truth)
    for c in ("tx", "ty", "tz", "tpx", "tpy", "tpz"):
        t[c] = t[c].astype(np.float32)
    return h, pd.DataFrame(parts), t


def test_field_triplets_uniform_field():
    h, p, t = helix_event(B=2.0)
    f = K.field_triplets(h, p, t)
    assert len(f) > 100
    assert abs(np.median(f.B) - 2.0) < 2e-3
    assert (f.sense < 0).all()                     # positive charges clockwise
    h1, p1, t1 = helix_event(B=1.3, seed=1)
    assert abs(np.median(K.field_triplets(h1, p1, t1).B) - 1.3) < 2e-3


def test_legs_res_equals_helix_union():
    rng = np.random.default_rng(3)
    xyz = rng.normal(0, 300, size=(200, 3))
    vol = rng.choice([7, 8, 9, 12, 13, 14, 16, 17, 18], 200)
    src, dst = rng.integers(0, 200, 500), rng.integers(0, 200, 500)
    a = HU.legs(xyz, src, dst, vol)
    b = K.legs_res(xyz, src, dst, vol, _active_res())
    for k in ("L", "cot", "s_cot"):
        assert np.array_equal(a[k], b[k])


def test_slope_window_agrees_with_candidate_triplets():
    """A triplet is inside the code's window exactly when its needed constant term is <= 0.05."""
    rng = np.random.default_rng(5)
    n = 400
    b = np.stack([rng.uniform(100, 300, n), np.zeros(n), np.zeros(n)], 1)
    a = b - np.stack([rng.uniform(30, 80, n), rng.normal(0, 1, n), np.zeros(n)], 1)
    c = b + np.stack([rng.uniform(30, 80, n), rng.normal(0, 1, n), np.zeros(n)], 1)
    cot = rng.uniform(-2, 2, n)
    a[:, 2] = -cot * (b[:, 0] - a[:, 0]) * (1 + rng.normal(0, 0.5, n))
    c[:, 2] = cot * (c[:, 0] - b[:, 0]) * (1 + rng.normal(0, 0.5, n))
    xyz = np.concatenate([a, b, c])
    ia, ib, ic = np.arange(n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n)
    vol = np.full(3 * n, 13)
    src, dst = np.concatenate([ia, ib]), np.concatenate([ib, ic])
    lg = HU.legs(xyz, src, dst, vol)
    e_in, e_out = HU.candidate_triplets(src, dst, lg, np.ones(len(src), dtype=bool))
    found = np.zeros(n, dtype=bool)
    found[e_in[(e_out - e_in) == n]] = True          # the true pair: in-leg k, out-leg n + k
    tr = {"a": ia, "b": ib, "c": ic, "mono": np.ones(n, bool), "same_layer": np.zeros(n, bool),
          "eta": np.zeros(n), "xyz": xyz, "vol": vol}
    rows = K.slope_rows(tr, _active_res())
    assert np.array_equal(~rows.outside.to_numpy(), found)
    assert np.array_equal(rows.c_need.to_numpy() <= K.C0, found)
    assert 0 < found.sum() < n


def test_true_triplets_orientation():
    h, p, _ = helix_event(n=5)
    tr = K.true_triplets(h, p)
    assert tr["mono"].all() and not tr["same_layer"].any()
    R = np.hypot(h.r.to_numpy().astype(np.float64), h.z.to_numpy().astype(np.float64))
    assert (R[tr["a"]] < R[tr["b"]]).all() and (R[tr["b"]] < R[tr["c"]]).all()
    assert len(tr["a"]) == 5 * (len(RADII) - 2)


def test_needed_constant():
    x = np.linspace(-1, 1, 1001)
    c = K.needed_constant(x, 0.005)
    assert np.sum(x > c) <= 5 and np.sum(x >= c) > 5


def test_field_map_fill():
    trip = pd.DataFrame({"r": [25.0, 25.0, 525.0], "absz": [50.0, 50.0, 1050.0], "B": [2.0, 2.2, 1.0]})
    fm = K.field_map(trip)
    assert fm["counts"][0, 0] == 2 and fm["B"][0, 0] == 2.1 and fm["B"][10, 10] == 1.0
    assert fm["B"][1, 0] == 2.1 and fm["B"][9, 10] == 1.0 and not np.isnan(fm["B"]).any()


def test_merge_rows_accept_clean_helices():
    h, p, _ = helix_event(n=20, seed=2)
    m = K.merge_rows(h, p)
    assert len(m) == 20 and (m.n_unions == len(RADII) - 3).all()
    tab = K.merge_table(m)
    assert tab.frac_fail.max() == 0.0
