"""hnl.overlay.modules: corners, nearest-centre matching, gate G0 and the ACTS -> TrackML module-id table.

    pytest tests/test_overlay_modules.py
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import modules as M


def _disc(volume, layer, z, side, n=12, r=300.0, minhu=16.4, maxhu=24.2, hv=38.0, stagger=3.0, mirror=True):
    """A ring of trapezoids built as ProtoLayerCreatorT does: v radial, w = side * z, u = v x w. With mirror=False the
    -z centres are the point reflection of the +z ones (unpatched v15)."""
    rows = []
    for i in range(n):
        phi = -np.pi + (i + 0.5) * 2 * np.pi / n
        zz = z + (0.5 * stagger if i % 2 == 0 else -0.5 * stagger)
        p = np.array([r * np.cos(phi), r * np.sin(phi), zz])
        c = np.array([p[0], p[1], side * p[2]]) if mirror else side * p
        v = np.array([np.cos(phi), np.sin(phi), 0.0])
        w = np.array([0.0, 0.0, float(side)])
        u = np.cross(v, w)
        R = np.stack([u, v, w], axis=1)
        rows.append({"volume_id": volume, "layer_id": layer, "module_id": i + 1, "cx": c[0], "cy": c[1], "cz": c[2],
                     **{k: R.ravel()[j] for j, k in enumerate(M.ROT)},
                     "module_minhu": minhu, "module_maxhu": maxhu, "module_hv": hv, "module_t": 0.25,
                     "pitch_u": 0.08, "pitch_v": 1.2})
    return pd.DataFrame(rows)


def _barrel(volume=13, layer=2, n=10, r=260.0):
    rows = []
    for i in range(n):
        phi = 2 * np.pi * i / n
        c = np.array([r * np.cos(phi), r * np.sin(phi), -400.0 + 80.0 * i])
        w = np.array([np.cos(phi), np.sin(phi), 0.0])
        v = np.array([0.0, 0.0, 1.0])
        u = np.array([-np.sin(phi), np.cos(phi), 0.0])
        R = np.stack([u, v, w], axis=1)
        rows.append({"volume_id": volume, "layer_id": layer, "module_id": i + 1, "cx": c[0], "cy": c[1], "cz": c[2],
                     **{k: R.ravel()[j] for j, k in enumerate(M.ROT)},
                     "module_minhu": 24.0, "module_maxhu": 24.0, "module_hv": 54.0, "module_t": 0.25,
                     "pitch_u": 0.08, "pitch_v": 1.2})
    return pd.DataFrame(rows)


def _detector(mirror=True):
    return pd.concat([_barrel(), _disc(12, 2, 1300.0, -1, mirror=mirror), _disc(14, 2, 1300.0, +1, mirror=mirror)],
                     ignore_index=True)


def _permute_ids(d, seed=0):
    """Renumber the endcap modules of every layer by a fixed permutation (the ACTS sensitive-id order)."""
    rng = np.random.default_rng(seed)
    d = d.copy()
    for (v, l), g in d.groupby(["volume_id", "layer_id"]):
        if v in M.BARREL_VOLUMES:
            continue
        d.loc[g.index, "module_id"] = rng.permutation(g.module_id.to_numpy())
    return d


def _g0(tml, acts, tol=M.G0_TOL_MM):
    m = M.match_modules(tml, acts)
    bij = M.bijection_report(m, tml, acts)
    cmp = M.compare(tml, acts, m)
    s = M.g0_summary(cmp, bij, tol)
    return m, bij, cmp, s


def test_module_key_matches_acts_geometry_id():
    # 504403295704449025 is ACTS v15 (volume 7, layer 2, sensitive 1)
    assert int(M.module_key(7, 2, 1)) == 504403295704449025
    with pytest.raises(ValueError):
        M.module_key(7, 2, 1 << 28)


def test_corners_follow_minhu_at_negative_v():
    d = _disc(14, 2, 1300.0, +1, n=4)
    K = M.corners(d)
    R = M.rotations(d)
    loc = np.einsum("nij,nki->nkj", R, K - M.centres(d)[:, None, :])     # back to local u, v, w
    np.testing.assert_allclose(loc[:, :, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(loc[0, :, 0], [-16.4, 16.4, 24.2, -24.2], atol=1e-12)
    np.testing.assert_allclose(loc[0, :, 1], [-38.0, -38.0, 38.0, 38.0], atol=1e-12)
    # v points outward: the short edge (minhu) is at the smaller radius
    r = np.hypot(K[:, :, 0], K[:, :, 1])
    assert (r[:, :2].mean(axis=1) < r[:, 2:].mean(axis=1)).all()


def test_identical_geometry_with_permuted_ids_passes_and_recovers_permutation():
    tml = _detector()
    acts = _permute_ids(tml)
    m, bij, cmp, s = _g0(tml, acts)
    assert bij["bijective"] and bij["barrel_identity"]
    assert s["n_fail"]["any"] == 0
    assert s["max_um"]["d_corner"] < 1e-6
    # the table undoes the permutation: the ACTS module sits where the TrackML module of the mapped id is
    j = acts.merge(m, left_on=["volume_id", "layer_id", "module_id"],
                   right_on=["volume_id", "layer_id", "acts_module_id"])
    back = tml.set_index(["volume_id", "layer_id", "module_id"]).loc[
        list(zip(j.volume_id, j.layer_id, j.trackml_module_id))]
    np.testing.assert_allclose(j[["cx", "cy", "cz"]].to_numpy(), back[["cx", "cy", "cz"]].to_numpy())
    # pass requires the full module count
    assert not s["pass"]
    assert s["n_compared"] == len(tml)


def test_point_reflection_fails_like_unpatched_v15():
    tml = _detector(mirror=True)
    acts = _detector(mirror=False)
    m, bij, cmp, s = _g0(tml, acts)
    neg = cmp.volume_id == 12
    # ring of 12: 6 per half-turn is even, so the stagger survives; the trapezoids are inverted
    assert (cmp.loc[neg, "d_corner_mm"] > 1.0).all()
    assert (cmp.loc[neg, "dot_v"] < -0.999).all()
    assert (cmp.loc[~neg, "d_corner_mm"] < 1e-9).all()
    assert s["n_fail"]["corner"] == int(neg.sum())


def test_odd_half_turn_ring_moves_off_plane_without_mirror():
    tml = pd.concat([_disc(12, 2, 1300.0, -1, n=14, mirror=True)], ignore_index=True)
    acts = pd.concat([_disc(12, 2, 1300.0, -1, n=14, mirror=False)], ignore_index=True)
    m, bij, cmp, s = _g0(tml, acts)
    # 7 modules per half-turn: the point reflection swaps the stagger parity, 3 mm off the TrackML plane
    np.testing.assert_allclose(cmp.d_plane_mm, 3.0, atol=1e-9)
    assert bij["bijective"]


def test_small_offsets_inside_and_outside_tolerance():
    tml = _detector()
    acts = _permute_ids(tml)
    acts.loc[0, "cx"] += 0.003                               # 3 um: inside 4 um
    acts.loc[1, "cz"] += 0.005                               # 5 um along the barrel v axis: centre and corners fail
    k = acts.index[acts.volume_id == 14][0]
    acts.loc[k, "cz"] += 0.0045                              # 4.5 um along w of an endcap: plane fails
    m, bij, cmp, s = _g0(tml, acts)
    f = cmp.set_index(["volume_id", "acts_module_id"])
    assert f.loc[(13, 1), "d_centre_mm"] == pytest.approx(0.003)
    assert s["n_fail"]["centre"] == 2
    assert f.loc[(14, int(acts.loc[k, "module_id"])), "d_plane_mm"] == pytest.approx(0.0045)
    assert s["n_fail"]["plane"] == 1


def test_tilt_moves_corners_off_plane():
    tml = _barrel(n=3)
    acts = tml.copy()
    a = 1e-4                                                 # rad about u: corners at v = 54 mm move 5.4 um along w
    R = M.rotations(acts)[0]
    c, s_ = np.cos(a), np.sin(a)
    Ru = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    R2 = R @ Ru
    for j, col in enumerate(M.ROT):
        acts.loc[0, col] = R2.ravel()[j]
    m, bij, cmp, s = _g0(tml, acts)
    assert cmp.loc[0, "d_centre_mm"] < 1e-12
    assert cmp.loc[0, "d_plane_mm"] == pytest.approx(54.0 * np.sin(a), rel=1e-3)
    assert s["n_fail"]["plane"] == 1


def test_frame_turned_180_about_w_keeps_corner_set():
    tml = _barrel(n=2)
    acts = tml.copy()
    for col in ("rot_xu", "rot_yu", "rot_zu", "rot_xv", "rot_yv", "rot_zv"):
        acts[col] = -acts[col]
    m, bij, cmp, s = _g0(tml, acts)
    assert (cmp.d_corner_mm > 50).all()
    assert (cmp.d_corner_set_mm < 1e-9).all()
    assert (cmp.d_plane_mm < 1e-9).all()


def test_non_bijective_nearest_falls_back_to_assignment():
    tml = _barrel(n=4)
    acts = tml.copy()
    # module 2 moved next to module 1: nearest neighbour picks module 1 twice; the assignment keeps a bijection
    acts.loc[1, ["cx", "cy", "cz"]] = tml.loc[0, ["cx", "cy", "cz"]].to_numpy() + [0.5, 0.0, 0.0]
    m, bij, cmp, s = _g0(tml, acts)
    assert bij["layers_needing_hungarian"] == ["13_2"]
    assert bij["bijective"]
    assert s["n_fail"]["centre"] >= 1


def test_missing_module_breaks_bijection():
    tml = _detector()
    acts = tml.drop(index=3).reset_index(drop=True)
    m, bij, cmp, s = _g0(tml, acts)
    assert not bij["bijective"]
    assert bij["trackml_modules_not_reached"] == 1


def test_writer_layout_is_read_like_the_dump():
    d = _detector()
    w = d.drop(columns=["module_minhu", "module_maxhu", "module_hv", "module_t", "pitch_u", "pitch_v"]).copy()
    rect = d.module_minhu == d.module_maxhu
    w["bounds_type"] = np.where(rect, M.BOUNDS_RECTANGLE, M.BOUNDS_TRAPEZOID)
    w["bound_param0"] = np.where(rect, -d.module_minhu, d.module_minhu)
    w["bound_param1"] = np.where(rect, -d.module_hv, d.module_maxhu)
    w["bound_param2"] = np.where(rect, d.module_minhu, d.module_hv)
    w["bound_param3"] = np.where(rect, d.module_hv, -1.0)
    w = pd.concat([w, w.iloc[:1].assign(module_id=0, bounds_type=3)], ignore_index=True)   # a non-sensitive row
    a = M.acts_table(w)
    assert len(a) == len(d)
    np.testing.assert_allclose(a[["module_minhu", "module_maxhu", "module_hv"]].to_numpy(),
                               d[["module_minhu", "module_maxhu", "module_hv"]].to_numpy())
    dump = M.acts_table(d.assign(bounds=np.where(rect, "rectangle", "trapezoid")))
    assert M.writer_cross_check(dump, a) == {"same_modules": True, "n_dump": len(d), "n_writer": len(d),
                                             "max_abs_diff": 0.0}


def test_module_map_roundtrip_and_lookup(tmp_path):
    tml = _detector()
    acts = _permute_ids(tml, seed=3)
    m = M.match_modules(tml, acts)
    p = str(tmp_path / "map.csv")
    sha = M.write_module_map(m, p)
    with open(str(tmp_path / "map.json"), "w") as f:
        json.dump({"sha256": sha}, f)
    mm = M.ModuleMap.from_csv(p)
    assert len(mm) == len(tml)
    out = mm.trackml_module(m.volume_id, m.layer_id, m.acts_module_id)
    np.testing.assert_array_equal(out, m.trackml_module_id.to_numpy())
    gid = M.module_key(m.volume_id, m.layer_id, m.acts_module_id).astype(np.uint64)
    v, l, t = mm.from_geometry_id(gid)
    np.testing.assert_array_equal(t, m.trackml_module_id.to_numpy())
    np.testing.assert_array_equal(v, m.volume_id.to_numpy())
    with pytest.raises(KeyError):
        mm.trackml_module([12], [2], [99])
    with pytest.raises(ValueError):
        mm.from_geometry_id([int(M.module_key(12, 2, 0))])
    with open(p, "a") as f:
        f.write("12,2,99,99\n")
    with pytest.raises(ValueError, match="sha256"):
        M.ModuleMap.from_csv(p)


def test_module_map_rejects_non_bijection():
    t = pd.DataFrame({"volume_id": [12, 12], "layer_id": [2, 2], "acts_module_id": [1, 2], "trackml_module_id": [5, 5]})
    with pytest.raises(ValueError, match="bijection"):
        M.ModuleMap(t)


@pytest.mark.skipif(not os.path.exists(M.MODULE_MAP), reason="module map not committed yet")
def test_committed_module_map():
    with open(os.path.splitext(M.MODULE_MAP)[0] + ".json") as f:
        side = json.load(f)
    mm = M.ModuleMap.from_csv(M.MODULE_MAP)            # checks the sha256 of the sidecar
    t = mm.table
    assert len(t) == M.N_MODULES == side["rows"]
    barrel = t.volume_id.isin(M.BARREL_VOLUMES)
    assert (t.loc[barrel, "acts_module_id"] == t.loc[barrel, "trackml_module_id"]).all()
    for _, g in t.groupby(["volume_id", "layer_id"]):
        assert sorted(g.acts_module_id) == sorted(g.trackml_module_id) == list(range(1, len(g) + 1))
