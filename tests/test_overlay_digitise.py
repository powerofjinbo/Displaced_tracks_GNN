"""hnl.overlay.digitise: the TrackML-style digitiser of the pileup overlay (docs/pileup_overlay_plan.md section 5.2) and
its C1 comparison (hnl.overlay.c1).

Synthetic modules test the geometry (slab thickness, Lorentz drift, grid, strip values, lookup). Tests marked
`trackml` read TrackML training data on /ceph (detectors.csv, tp_v1 calibration event 21000) and compare the digitiser,
replayed on TrackML truth, with TrackML's own clusters under the C1 thresholds of section 10.

    pytest tests/test_overlay_digitise.py
"""

import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import c1 as C
from hnl.overlay import digitise as D

HAS_DETECTORS = os.path.exists(D.DETECTORS)
HAS_TP = os.path.isdir(os.path.join(C.TP, "truth"))
trackml = pytest.mark.skipif(not (HAS_DETECTORS and HAS_TP), reason="needs TrackML data on /ceph")

# TrackML module types (detectors.csv): (volume, module_t, minhu, maxhu, hv, pitch_u, pitch_v)
PIX_EC = (7, 0.15, 8.4, 8.4, 36.0, 0.05, 0.05625)
PIX_BA = (8, 0.15, 8.4, 8.4, 36.0, 0.05, 0.05625)
SS_BA = (13, 0.25, 24.0, 24.0, 54.0, 0.08, 1.2)
SS_EC = (12, 0.25, 16.4, 24.2, 78.0, 0.08, 1.2)


def make_modules(types, layer=2):
    """Modules with u = x, v = y, w = z, centred at z = 100 * (i + 1); module ids 1, 2, ..."""
    n = len(types)
    vol = np.array([t[0] for t in types], np.int64)
    mod = np.arange(1, n + 1)
    key = D.module_key(vol, np.full(n, layer), mod)
    o = np.argsort(key)
    arr = lambda k: np.array([t[k] for t in types], np.float64)[o]  # noqa: E731
    c = np.zeros((n, 3))
    c[:, 2] = 100.0 * (np.arange(n) + 1)
    return D.Modules(key=key[o], c=c[o], R=np.repeat(np.eye(3)[None], n, axis=0), t=arr(1), minhu=arr(2),
                     maxhu=arr(3), hv=arr(4), pu=arr(5), pv=arr(6), volume=vol[o])


def local_one(t, lu, lv, tu, tv, rng=None):
    m = make_modules([t])
    p = D.module_params(m, np.array([0]))
    return D.digitise_local(np.array([lu]), np.array([lv]), np.array([tu]), np.array([tv]), p, rng)


def cell_centre(ch, pitch, half):
    return (ch + 0.5) * pitch - half


# ------------------------------------------------------------------------------------------------ geometry
def test_endcap_slab_is_twice_module_thickness():
    u0 = cell_centre(100, 0.05, 8.4)
    v0 = cell_centre(300, 0.05625, 36.0)
    cells, h = local_one(PIX_EC, u0, v0, 0.0, 0.0)
    assert h["ncells"][0] == 1 and cells["ch0"][0] == 100 and cells["ch1"][0] == 300
    assert h["charge"][0] == pytest.approx(2 * 0.15)
    assert h["u"][0] == pytest.approx(u0) and h["v"][0] == pytest.approx(v0)


def test_barrel_slab_and_lorentz_shift():
    u0 = cell_centre(100, 0.05, 8.4)
    cells, h = local_one(PIX_BA, u0, 0.01, 0.0, 0.0)
    assert h["charge"][0] == pytest.approx(0.15)                     # slab = module thickness
    assert h["ncells"][0] == 1                                       # 18 um of drift stays inside one 50 um cell
    # readout side -1: the reported position is the cell centre plus module_t tan(0.12)
    assert h["u"][0] == pytest.approx(u0 + 0.15 * np.tan(0.12))
    # the drift runs towards -u for the pixels: a point 5 um above a cell edge spreads into the cell below
    edge = cell_centre(100, 0.05, 8.4) - 0.025
    cells, h = local_one(PIX_BA, edge + 0.005, 0.01, 0.0, 0.0)
    assert sorted(cells["ch0"]) == [99, 100]
    assert cells["path"][cells["ch0"] == 100][0] == pytest.approx(0.15 * 0.005 / (0.15 * np.tan(0.12)))


def test_inclined_path_and_mean_span():
    rng = np.random.default_rng(3)
    n = 4000
    m = make_modules([PIX_EC])
    p = D.module_params(m, np.zeros(n, np.int64))
    tv = 0.3
    lu = rng.uniform(-8, 8, n)
    lv = rng.uniform(-35, 35, n)
    cells, h = D.digitise_local(lu, lv, np.zeros(n), np.full(n, tv), p)
    assert np.allclose(h["path"], 2 * 0.15 * np.sqrt(1 + tv * tv))
    # mean span = 1 + slab width / pitch for uniformly placed tracks
    assert h["span1"].mean() == pytest.approx(1 + 2 * 0.15 * tv / 0.05625, rel=0.03)
    assert (h["span0"] == 1).all()


def test_strips_value_one_and_plain_mean():
    # a short-strip barrel track crossing three strips with unequal path in each
    tu = 0.6
    u0 = cell_centre(300, 0.08, 24.0) + 0.02
    cells, h = local_one(SS_BA, u0, 0.3, tu, 0.0)
    assert (cells["value"] == 1).all() and h["charge"][0] == h["ncells"][0]
    centres = cell_centre(cells["ch0"], 0.08, 24.0)
    assert len(np.unique(np.round(cells["path"], 6))) > 1             # unequal paths, yet the plain mean is reported
    assert h["u"][0] == pytest.approx(centres.mean() - 0.25 * np.tan(0.12))
    assert h["v"][0] == pytest.approx(cell_centre(cells["ch1"][0], 1.2, 54.0))


def test_trapezoid_uses_bounding_box_grid():
    cells, h = local_one(SS_EC, 0.0, 0.0, 0.0, 0.0)
    assert cells["ch0"][0] == int(np.floor(24.2 / 0.08)) and cells["ch1"][0] == int(np.floor(78 / 1.2))
    m = make_modules([SS_EC])
    p = D.module_params(m, np.array([0]))
    assert p["n0"][0] == 605 and p["n1"][0] == 130 and p["h"][0] == pytest.approx(0.25)
    assert p["tan_la"][0] == 0 and p["side"][0] == 1


def test_path_outside_grid_makes_no_cell():
    # the track leaves the grid at u = +8.4: only the inside part makes cells
    cells, h = local_one(PIX_EC, 8.4 - 0.01, 0.0, 1.0, 0.0)
    assert h["ncells"][0] >= 1 and (cells["ch0"] < 336).all()
    assert h["path"][0] < 2 * 0.15 * np.sqrt(2)
    cells, h = local_one(PIX_EC, 9.0, 0.0, 0.0, 0.0)
    assert h["ncells"][0] == 0 and np.isnan(h["u"][0])


def test_cell_factor():
    n = 20000
    m = make_modules([PIX_BA, SS_BA])
    p = D.module_params(m, np.repeat([0, 1], n // 2))
    lu = np.full(n, cell_centre(100, 0.05, 8.4))
    lv = np.where(np.arange(n) < n // 2, 0.01, 0.6)
    c0, h0 = D.digitise_local(lu, lv, np.zeros(n), np.zeros(n), p)
    c1, h1 = D.digitise_local(lu, lv, np.zeros(n), np.zeros(n), p, rng=np.random.default_rng(5))
    pix = p["pixel"][c1["hit"]]
    r = c1["value"][pix] / c0["value"][pix]
    assert r.mean() == pytest.approx(1.0, abs=0.005) and r.std() == pytest.approx(D.CELL_SIGMA, rel=0.03)
    assert (c1["value"][~pix] == 1).all()
    np.testing.assert_array_equal(c0["ch0"], c1["ch0"])
    c2, _ = D.digitise_local(lu, lv, np.zeros(n), np.zeros(n), p, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(c1["value"], c2["value"])


def test_per_hit_factor_compatibility():
    """The per-hit factor of plan section 4.4 still multiplies the pixel values, warns, and refuses to combine with the
    per-cell factor."""
    m = make_modules([PIX_BA])
    lk = D.Lookup([8], [2], [3], [1])
    sim = pd.DataFrame({"particle_id": [1, 2], "geometry_id": [gid(8, 2, 3)] * 2, "tx": [0.01, 0.2], "ty": [0.0, 1.0],
                        "tz": [100.0, 100.0], "tpx": [0.0, 0.0], "tpy": [0.0, 0.0], "tpz": [1.0, 1.0]})
    with pytest.warns(UserWarning, match="per-cell factor"):
        dig = D.digitise(sim, lk, m, charge_factor=np.array([2.0, 0.5]))
    np.testing.assert_allclose(dig.hits.charge, [0.30, 0.075])
    with pytest.raises(ValueError, match="double"):
        D.digitise(sim, lk, m, rng=np.random.default_rng(0), charge_factor=np.array([2.0, 0.5]))


def test_round_sig():
    x = np.array([-65.79654321, 1502.5, 0.000123456789, 0.0, -1234567.0])
    np.testing.assert_allclose(D.round_sig(x), [-65.7965, 1502.5, 0.000123457, 0.0, -1234570.0], rtol=1e-12)


# ------------------------------------------------------------------------------------------------ lookup and frames
def gid(volume, layer, sensitive):
    return (np.uint64(volume) << np.uint64(56)) | (np.uint64(layer) << np.uint64(36)) | np.uint64(sensitive)


@pytest.mark.parametrize("layout", ["modules", "hit_format", "geometry"])
def test_lookup_layouts(tmp_path, layout):
    v, l, a, t = [7, 7, 8], [2, 2, 2], [1, 2, 1], [2, 1, 1]
    if layout == "modules":
        d = pd.DataFrame({"volume_id": v, "layer_id": l, "acts_module_id": a, "trackml_module_id": t})
    elif layout == "hit_format":
        d = pd.DataFrame({"volume": v, "layer": l, "acts_module": a,
                          "acts_geometry_id": [gid(*x) for x in zip(v, l, a)], "trackml_module": t})
    else:
        d = pd.DataFrame({"volume": v, "layer": l, "tml_module": t, "acts_module": a,
                          "acts_geoid": [gid(*x) for x in zip(v, l, a)]})
    path = tmp_path / "map.csv"
    d.to_csv(path, index=False)
    lk = D.load_lookup(str(path))
    vol, lay, mod = lk.from_geometry_id(np.array([gid(7, 2, 1), gid(7, 2, 2), gid(8, 2, 1), gid(9, 2, 1)]))
    assert vol.tolist() == [7, 7, 8, -1] and mod.tolist() == [2, 1, 1, -1]


def test_lookup_rejects_bad_tables(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"volume_id": [7, 7], "layer_id": [2, 2], "acts_module_id": [1, 2],
                  "trackml_module_id": [1, 1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="TrackML module appears twice"):
        D.load_lookup(str(p))
    pd.DataFrame({"volume": [7], "layer": [2], "acts_module": [1], "acts_geometry_id": [gid(7, 2, 3)],
                  "trackml_module": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="disagrees"):
        D.load_lookup(str(p))


def test_digitise_frames_and_stats():
    m = make_modules([PIX_EC, PIX_BA])            # volume 7 at z = 100, volume 8 at z = 200 (rows sorted by key)
    lk = D.Lookup([7, 8], [2, 2], [5, 6], [1, 2])
    zc = {7: m.c[m.index([7], [2], [1])[0], 2], 8: m.c[m.index([8], [2], [2])[0], 2]}
    u0 = cell_centre(10, 0.05, 8.4)
    sim = pd.DataFrame({
        "particle_id": [11, 12, 13, 14],
        "geometry_id": [gid(7, 2, 5), gid(7, 2, 5), gid(8, 2, 6), gid(9, 2, 1)],
        "tx": [u0, u0 + 0.01, 0.0, 0.0], "ty": [1.0, 1.0, 0.0, 0.0],
        "tz": [zc[7], zc[7] + 0.002, zc[8], 0.0],             # the second hit sits 2 um off the plane
        "tpx": [0.0, 1.0, 0.0, 0.0], "tpy": [0.0, 0.0, 0.0, 0.0], "tpz": [1.0, 2.0, 1.0, 1.0]})
    dig = D.digitise(sim, lk, m)
    assert dig.stats["n_sim"] == 4 and dig.stats["n_unknown_module"] == 1 and dig.stats["n_hits"] == 3
    h = dig.hits
    assert h.sim_row.tolist() == [0, 1, 2] and h.module_id.tolist() == [1, 1, 2]
    assert h.x.iloc[0] == pytest.approx(u0) and h.z.iloc[0] == pytest.approx(zc[7])
    assert h.u_true.iloc[1] == pytest.approx(u0 + 0.01 - 0.002 * 0.5)    # moved along the direction onto w = 0
    assert h.tz.iloc[1] == pytest.approx(zc[7]) and dig.stats["max_abs_w_moved_mm"] == pytest.approx(0.002)
    assert (dig.cells.hit.to_numpy() >= 0).all() and dig.cells.hit.max() == 2
    mem = D.trackml_members(dig, first_hit_id=7)
    assert mem["hits"].hit_id.tolist() == [7, 8, 9] and (mem["truth"].weight == 0).all()
    assert set(mem["cells"].hit_id) == {7, 8, 9}


def test_match_weights_reproduce_reference_mix():
    """Reweighting the test hits of a bin to the reference's (x, y) mix: a quantity that depends only on y gets the
    reference mean, while the plain mean of the test hits does not."""
    rng = np.random.default_rng(0)
    xr, yr = rng.uniform(0.8, 2.4, 20000), rng.exponential(0.8, 20000)
    xt, yt = rng.uniform(0.8, 2.4, 20000), rng.exponential(4.0, 20000)
    f = lambda y: 1.0 / (1.0 + y)  # noqa: E731
    wr, wt = C.match_weights(xr, yr, xt, yt, C.EDGES["abs"])
    assert wr.mean() > 0.98
    assert np.average(f(yt), weights=wt) == pytest.approx(np.average(f(yr), weights=wr), rel=0.02)
    assert abs(f(yt).mean() / f(yr).mean() - 1) > 0.2
    x = rng.normal(0, 1, 100001)
    assert C.wstat(x, np.ones_like(x), "mad") == pytest.approx(C.mad(x), rel=1e-3)


# ------------------------------------------------------------------------------------------------ TrackML data
@pytest.fixture(scope="module")
def tml21000():
    mods = D.Modules.load()
    return mods, C.trackml_hits([21000], mods)


@trackml
def test_members_pass_frozen_converter():
    """A digitised event written as TrackML members passes hnl.trackml.convert.event_tables."""
    from hnl.trackml.convert import load_detectors, event_tables
    mods = D.Modules.load()
    rng = np.random.default_rng(1)
    rows = rng.choice(len(mods.key), 300, replace=False)
    loc = np.stack([rng.uniform(-0.5, 0.5, 300) * mods.minhu[rows], rng.uniform(-0.9, 0.9, 300) * mods.hv[rows],
                    np.zeros(300)], axis=1)
    pos = np.einsum("nij,nj->ni", mods.R[rows], loc) + mods.c[rows]
    mom = pos / np.linalg.norm(pos, axis=1, keepdims=True) * 2.0
    key = mods.key[rows]
    sim = pd.DataFrame({"particle_id": np.repeat(np.arange(1, 31), 10).astype(np.int64),
                        "tx": pos[:, 0], "ty": pos[:, 1], "tz": pos[:, 2],
                        "tpx": mom[:, 0], "tpy": mom[:, 1], "tpz": mom[:, 2]})
    dig = D.digitise_on_modules(sim, key // 1_000_000, key // 10_000 % 100, key % 10_000, mods,
                                rng=np.random.default_rng(2))
    mem = D.trackml_members(dig)
    nh = mem["truth"].groupby("particle_id").size()
    parts = pd.DataFrame({"particle_id": nh.index.to_numpy(np.int64), "vx": 0.0, "vy": 0.0, "vz": 0.0, "px": 1.0,
                          "py": 0.0, "pz": 0.5, "q": 1, "nhits": nh.to_numpy(np.int64)})
    tables, dq = event_tables({**mem, "particles": parts}, 50_000_001, load_detectors(), keep_weight=True,
                              keep_truth=True)
    assert tables is not None and not dq["problems"] and dq["max_outside_mm"] < 1e-3
    hits = tables["hits"].to_pandas()
    assert len(hits) == dig.stats["n_hits"] and (hits.ncells >= 1).all() and (hits.span0 >= 1).all()


@trackml
def test_replay_matches_trackml_clusters_c1(tml21000):
    """TrackML truth of training event 21000 digitised again: span, pixel charge x cos and residual MAD per volume in
    bins of local incidence within the C1 thresholds of every bin with >= 200 hits, mean u bias within 1 um and
    effective thickness within 0.04 mm."""
    mods, ref = tml21000
    rep = C.replay(ref, mods, seed=7)
    res = C.compare(ref, rep)
    assert res["n_bins_counted"] > 300
    bad = [(v, q, b["lo"], b["hi"], b["rel_diff"]) for v, r in res["volumes"].items() for q, bins in r["bins"].items()
           for b in bins if b["counted"] and not b["pass"]]
    assert not bad
    assert res["pass"]


@trackml
def test_resolution_needs_cell_factor(tml21000):
    """Without the per-cell charge factor the pixel-barrel v residual is far narrower than TrackML's (the cause of the
    narrow prototype residuals in the barrel); the pixel endcaps are near binary resolution either way."""
    mods, ref = tml21000
    rep = C.replay(ref, mods, seed=None)
    m = lambda f, v, c: C.mad(f[f.volume == v][c])  # noqa: E731
    assert m(rep, 8, "dv") < 0.4 * m(ref, 8, "dv")
    for v in (7, 9):
        assert m(rep, v, "du") == pytest.approx(m(ref, v, "du"), rel=0.05)
        assert m(ref, v, "du") > 0.015
