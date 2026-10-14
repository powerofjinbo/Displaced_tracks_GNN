import numpy as np
import pandas as pd

from hnl.truth import nav_losses as N


def _layers():
    return pd.DataFrame([
        dict(key=802, volume=8, layer=2, barrel=True, n_modules=1, R=32.0, Z=np.nan, rlo=31.0, rhi=34.0,
             zlo=-491.0, zhi=491.0, gaps=[]),
        dict(key=1302, volume=13, layer=2, barrel=True, n_modules=1, R=261.5, Z=np.nan, rlo=256.0, rhi=266.0,
             zlo=-1084.0, zhi=1084.0, gaps=[]),
        dict(key=1702, volume=17, layer=2, barrel=True, n_modules=1, R=821.5, Z=np.nan, rlo=815.0, rhi=825.0,
             zlo=-1084.0, zhi=1084.0, gaps=[]),
        dict(key=1402, volume=14, layer=2, barrel=False, n_modules=1, R=np.nan, Z=1218.5, rlo=240.0, rhi=701.0,
             zlo=1215.0, zhi=1225.0, gaps=[]),
    ])


def _h(x, y, z, px, py, pz, q):
    a = lambda v: np.atleast_1d(np.asarray(v, float))
    return N.helix(a(x), a(y), a(z), a(px), a(py), a(pz), a(q))


def test_stiff_track_crosses_barrels_at_their_radius():
    H = _h(0, 0, 0, 100.0, 0, 0, 1)
    s_hi, by_loop = N.path_end(H)
    assert not by_loop[0] and abs(s_hi[0] - N.R_WORLD) < 1.0
    C = N.crossings(H, _layers(), s_hi, 20.0, np.zeros(1), 0.0)
    for i, R in enumerate((32.0, 261.5, 821.5)):
        s = C["S"][0, i]
        s = s[~np.isnan(s)]
        assert len(s) == 1 and abs(s[0] - R) < 0.5
        assert abs(np.hypot(C["X"][0, i, 0], C["Y"][0, i, 0]) - R) < 1e-6
    assert C["robust"][0, :3, 0].all()


def test_positive_charge_bends_clockwise_in_positive_bz():
    H = _h(0, 0, 0, 1.0, 0, 0, 1)
    x, y, _ = N.at(H, np.array([[100.0]]))
    assert y[0, 0] < 0                     # F = q v x B: v = +x, B = +z -> force along -y


def test_loop_protection_ends_a_curler_after_half_a_turn():
    H = _h(0, 0, 0, 0.2, 0, 0.0, 1)        # R = 333.6 mm, never reaches the world radius
    s_hi, by_loop = N.path_end(H)
    assert by_loop[0] and abs(s_hi[0] - np.pi * H["R"][0]) < 1e-9
    x, y, _ = N.at(H, s_hi[:, None])
    assert abs(np.hypot(x, y)[0, 0] - 2 * H["R"][0]) < 1e-6


def test_restart_of_end_state_helix_returns_to_the_vertex():
    H = _h(10.0, -5.0, 3.0, 0.8, 0.3, 1.1, -1)
    s = np.array([[700.0]])
    xe, ye, ze = N.at(H, s)
    psi = H["psi0"] - H["h"] * 700.0 / H["R"]
    phi = psi - H["h"] * np.pi / 2
    He = _h(xe[0, 0], ye[0, 0], ze[0, 0], H["pt"][0] * np.cos(phi[0]), H["pt"][0] * np.sin(phi[0]), 1.1, -1)
    Hb, s_e = N.restart(He, np.array([10.0]), np.array([-5.0]))
    assert abs(s_e[0] - 700.0) < 1e-6
    assert abs(Hb["x0"][0] - 10.0) < 1e-6 and abs(Hb["y0"][0] + 5.0) < 1e-6 and abs(Hb["z0"][0] - 3.0) < 1e-6


def test_analyse_counts_crossings_after_the_last_hit():
    L = _layers()
    H = _h(0, 0, 0, 100.0, 0, 0, 1)
    s_hi, _ = N.path_end(H)
    hits = np.array([[32.0, 0.0, 0.0], [261.5, -0.1, 0.0]])
    r = N.analyse(H, L, s_hi, 20.0, np.zeros(1), np.array([0, 0]), np.array([0, 1]), hits)
    assert r["n_rob"][0] == 3 and r["n_after"][0] == 1 and r["first_missed_key"][0] == 1702
    assert r["n_holes"][0] == 0 and r["n_unm"][0] == 0
    r = N.analyse(H, L, s_hi, 20.0, np.zeros(1), np.array([0]), np.array([2]), hits[:1] * [821.5 / 32, 1, 1])
    assert r["n_after"][0] == 0 and r["n_holes"][0] == 2


def test_end_category():
    cat = N.end_category(np.array([2000.0, 1028.538, 300.0, 300.0, 300.0]), np.zeros(5),
                         np.array([0.0, 0.0, 100.0, 100.0, 100.0]), np.array([5.0, 5.0, 0.3, 0.8, 0.8]),
                         np.array([0.1, 0.1, 0.1, np.pi, 1.0]))
    assert list(cat) == ["outside", "world", "below_pmin", "loop", "inside"]


def test_module_depth_inside_and_outside():
    M = N.modules()
    for key in (802, 1204, 1410, 1702):
        tree, c, R, bt, p = M.layers[key]
        n = R[:5, :, 2]
        dep = M.depth(key, c[:5], n)
        assert (dep > 0).all()
        far = c[:5] * 1.0
        far[:, 2] += 400.0 if key in (802, 1702) else 0.0
        if key in (802, 1702):                     # beyond the barrel's z extent
            far[:, 2] = 2000.0
            assert (M.depth(key, far, n) < 0).all()
