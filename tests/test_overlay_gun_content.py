"""hnl.overlay.gun_content and the final states of hnl.overlay.fatras_io: navigator losses in bins, silent world exits
from the Fatras final states, and premature ends and holes of material-free trajectories through a field against the
layers of a module table (coverage, grazing, loop protection). Synthetic modules, fields and particles only."""

import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun_content as GC

N_PHI = 12
BARREL_R = (100.0, 200.0, 300.0)


def uniform(b=2.0):
    return lambda r, z: (np.zeros_like(np.asarray(r, float)), np.full_like(np.asarray(r, float), b))


def _modules(drop=()) -> D.Modules:
    """Three barrel layers (volume 8, layers 2, 4, 6) of N_PHI flat modules tangent at radius R (u along phi, v along z,
    w radial; half widths with 2 % overlap, |z| <= 400 mm) and one disk (volume 9, layer 2, z = 600 mm) of four
    300 x 300 mm squares. drop: (volume, layer, module) left out (a gap)."""
    key, c, R, hu, hv = [], [], [], [], []
    for li, rl in enumerate(BARREL_R):
        for i in range(N_PHI):
            f = 2 * np.pi * i / N_PHI
            key.append((8, 2 * (li + 1), i + 1))
            c.append([rl * np.cos(f), rl * np.sin(f), 0.0])
            R.append(np.column_stack([[-np.sin(f), np.cos(f), 0], [0, 0, 1], [np.cos(f), np.sin(f), 0]]))
            hu.append(rl * np.tan(np.pi / N_PHI) * 1.02)
            hv.append(400.0)
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        key.append((9, 2, i + 1))
        c.append([150.0 * sx, 150.0 * sy, 600.0])
        R.append(np.eye(3))
        hu.append(150.0)
        hv.append(150.0)
    keep = [k not in drop for k in key]
    k = D.module_key(*np.array(key).T)
    o = np.argsort(k)
    o = o[np.asarray(keep)[o]]
    n = len(o)
    return D.Modules(key=k[o], c=np.array(c)[o], R=np.array(R)[o], t=np.full(n, 0.3), minhu=np.array(hu)[o],
                     maxhu=np.array(hu)[o], hv=np.array(hv)[o], pu=np.full(n, 0.05), pv=np.full(n, 0.05),
                     volume=np.array([x[0] for x in key])[o])


def _track(pt, eta, phi, q=1, x0=(0.0, 0.0, 0.0)):
    return np.array([x0]), np.array([[pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)]]), np.array([q])


@pytest.fixture(scope="module")
def layers():
    return GC.Layers.from_modules(_modules())


def test_layer_table(layers):
    L = layers.table
    assert L[["volume", "layer"]].values.tolist() == [[8, 2], [8, 4], [8, 6], [9, 2]]
    assert L.R.iloc[:3].tolist() == pytest.approx(list(BARREL_R)) and L.Z.iloc[3] == pytest.approx(600.0)
    assert L.zlo.iloc[0] == pytest.approx(-400) and L.rhi.iloc[3] == pytest.approx(300 * np.sqrt(2))
    assert layers.index([8, 9, 8], [4, 2, 8]).tolist() == [1, 3, -1]
    # depth: the centre of a barrel module, and a point beyond the disk's outer corner
    d = layers.depth(0, np.array([[100.0, 0, 0]]), np.array([[1.0, 0, 0]]))
    assert d[0] == pytest.approx(100 * np.tan(np.pi / N_PHI) * 1.02, rel=1e-6)
    assert layers.depth(3, np.array([[310.0, 310.0, 600.0]]), np.array([[0, 0, 1.0]]))[0] < 0


def _hits_on(cr, which):
    s = cr[cr.acc].reset_index(drop=True)
    s = s.iloc[list(which)] if which is not None else s.iloc[:0]
    return s[["track", "li", "x", "y", "z"]]


def test_barrel_track_holes_and_premature_ends(layers):
    x0, p0, q = _track(2.0, 0.2, 2 * np.pi / N_PHI + 0.03)
    cr, tr = GC.crossings(x0, p0, q, uniform(), layers)
    assert cr.li.tolist() == [0, 1, 2] and cr.robust.all() and not tr.by_loop.iloc[0]
    assert cr.s.is_monotonic_increasing
    full = GC.match_hits(cr, _hits_on(cr, [0, 1, 2]), 1).iloc[0]
    assert (full.n_matched, full.n_robust, full.n_after, full.n_holes, full.first_missed_li) == (3, 3, 0, 0, -1)
    assert full.max_match_mm < 1e-6
    hole = GC.match_hits(cr, _hits_on(cr, [0, 2]), 1).iloc[0]
    assert (hole.n_after, hole.n_holes, hole.n_before) == (0, 1, 3)
    early = GC.match_hits(cr, _hits_on(cr, [0]), 1).iloc[0]
    assert (early.n_after, early.n_holes, early.first_missed_li) == (2, 0, 1)
    none = GC.match_hits(cr, _hits_on(cr, None), 1).iloc[0]
    assert (none.n_hits, none.n_after) == (0, 3)


def test_disk_crossing_and_a_gap_is_no_hole():
    lay = GC.Layers.from_modules(_modules())
    phi = 2 * np.pi / N_PHI
    x0, p0, q = _track(2.0, 2.0, phi + 0.03)
    cr, _ = GC.crossings(x0, p0, q, uniform(), lay)
    # barrel layer 0 at z = 363 mm, the disk at z = 600 mm; barrel layers 1 and 2 are crossed beyond their |z| extent
    assert cr.li.tolist() == [0, 3, 1, 2] and cr.acc.tolist() == [True, True, False, False]
    assert cr.robust.tolist() == [True, True, False, False]
    # a track through barrel layer 1 where its module is removed: no acceptance there, so no hole
    gap = GC.Layers.from_modules(_modules(drop=[(8, 4, 2)]))
    x0, p0, q = _track(2.0, 0.2, phi + 0.03)
    cr, _ = GC.crossings(x0, p0, q, uniform(), gap)
    assert cr.acc.tolist() == [True, False, True]
    r = GC.match_hits(cr, _hits_on(cr, [0, 1]), 1).iloc[0]               # hits on layers 0 and 2
    assert (r.n_holes, r.n_after, r.n_robust) == (0, 0, 2)


def test_loop_protection_and_grazing(layers):
    # pT 0.2 GeV in 2 T: R_T = 333.6 mm; the loop protection ends it after half a transverse turn (r = 667 mm)
    x0, p0, q = _track(0.2, 0.0, 2 * np.pi / N_PHI)
    s_loop = GC.loop_limit(x0, p0, uniform())[0]
    assert s_loop == pytest.approx(np.pi * 0.2 / (GC.K_GEV_PER_T_M * 2.0) * 1000)
    cr, tr = GC.crossings(x0, p0, q, uniform(), layers)
    assert tr.by_loop.iloc[0] and tr.s_hi.iloc[0] == pytest.approx(s_loop)
    assert cr.li.tolist() == [0, 1, 2]                                   # outward only: no crossing on the way back
    # turning point 2 R_T = 310 mm: the crossing of the 300 mm layer grazes (10 mm < MARGIN_MM) and is not robust
    pt = 155.0 * GC.K_GEV_PER_T_M * 2.0 / 1000
    x0, p0, q = _track(pt, 0.0, 2 * np.pi / N_PHI)
    cr, _ = GC.crossings(x0, p0, q, uniform(), layers)
    last = cr[cr.li == 2]
    assert len(last) and not last.robust.any() and cr[cr.li < 2].robust.all()


def test_trajectory_misses_by_particle(layers):
    phi = 2 * np.pi / N_PHI + 0.03
    x0, p0, q = _track(2.0, 0.2, phi)
    init = pd.DataFrame({"key": [11, 12], "x": 0.0, "y": 0.0, "z": 0.0, "px": p0[0, 0], "py": p0[0, 1],
                         "pz": p0[0, 2], "q": 1.0})
    cr, _ = GC.crossings(x0, p0, q, uniform(), layers)
    pts = cr[["x", "y", "z"]].to_numpy()
    hits = pd.DataFrame({"key": [11, 11, 11, 12], "volume": 8, "layer": [2, 4, 6, 2], "x": pts[[0, 1, 2, 0], 0],
                         "y": pts[[0, 1, 2, 0], 1], "z": pts[[0, 1, 2, 0], 2]})
    r = GC.trajectory_misses(init, hits, uniform(), layers, chunk=1).set_index("key")
    assert not r.loc[11, "premature"] and not r.loc[11, "hole"]
    assert r.loc[12, "premature"] and r.loc[12, "n_after"] == 2 and r.loc[12, "first_missed_layer"] == 804
    with pytest.raises(ValueError, match="layers missing"):
        GC.trajectory_misses(init, hits.assign(layer=8), uniform(), layers)
    with pytest.raises(KeyError):
        GC.trajectory_misses(init, hits.assign(key=99), uniform(), layers)
    info = pd.DataFrame({"key": [11, 12], "component": "iso", "pt": 2.0, "eta": 0.2, "pp_like": [True, False]})
    s = {x["group"]: x for x in GC.summarise_misses(r.reset_index(), info)}
    assert s["all"]["premature"] == 1 and s["P_P-like"]["particles"] == 1 and s["dense"]["particles"] == 0


def _gun_truth():
    pid = [(1001 << 52) | (1 << 36), (1001 << 52) | (2 << 36), (1003 << 52) | (1 << 36), (1009 << 52) | (1 << 36)]
    return pd.DataFrame({"event": 50_021_117, "particle_id": pid, "pdg": 211, "q": [1, -1, 1, 1],
                         "vx": [150.0, 150.0, 5.0, 300.0], "vy": [0.0, 0.0, 0.0, 0.0], "vz": 10.0,
                         "px": [2.0, -2.0, 0.0, 3.0], "py": [0.0, 0.5, 2.0, 0.0], "pz": [0.5, 0.5, 0.5, 0.5],
                         "sim_status": ["failed", "simulated", "simulated", "failed"]})


def test_loss_table_and_bins():
    t = GC.loss_table(_gun_truth())
    assert t.key.tolist() == [0, 1, 2, 3] and t.component.tolist() == ["iso", "iso", "iso", "dense"]
    assert t.inward.tolist() == [False, True, False, False]
    # particle 3 starts 5 mm from the axis and moves perpendicular to it: |d0| = 5 mm, r_v 5 mm, not P_P-like
    assert t.abs_d0.iloc[2] == pytest.approx(5.0, abs=1e-3) and not t.pp_like.iloc[2]
    assert t.pp_like.tolist() == [True, True, False, False]
    b = GC.nav_loss_bins(t)
    assert (b["lost"], b["pp_like"], b["pp_like_lost"], b["pp_like_lost_share"]) == (2, 2, 1, 0.5)
    assert b["lost_by_component"] == {"dense": 1, "iso": 1}
    d = {x["bin"]: x for x in b["bins"]["direction at the vertex"]}
    assert (d["inward"]["lost"], d["outward"]["lost"]) == (0, 1)
    rv = [x for x in b["bins"]["r_v [mm]"] if x["pp_like"]]
    assert len(rv) == 1 and rv[0]["bin"] == "[100.0, 200.0)" and rv[0]["share"] == 0.5


def test_silent_exits_from_final_states():
    bc = FIO.barcode(1, 1001, [1, 2, 3])
    status = pd.DataFrame({"event": 4, "particle_id": bc, "status": ["simulated", "simulated", "failed"]})
    simp = pd.DataFrame({"event": [4, 4], "particle_id": bc[:2], "initial": True, "final": True,
                         "final_vx": [1100.0, 200.0], "final_vy": [0.0, 0.0], "final_vz": [10.0, 3001.5],
                         "final_px": 1.0, "final_py": 0.0, "final_pz": 0.0})
    r = GC.silent_exits(status, simp)
    assert (r["final_states"], r["simulated"], r["outside"]) == (True, 2, 1)
    assert r["particles"][0]["particle_id"] == (1001 << 52) | (1 << 36) and r["particles"][0]["r_end"] == 1100.0
    old = simp[["event", "particle_id", "initial", "final"]]
    assert GC.silent_exits(status, old)["outside"] is None


def test_pack_keeps_the_final_states(tmp_path):
    bc = FIO.barcode(1, 1001, [1, 2])
    gid = (np.uint64(8) << np.uint64(56)) | (np.uint64(2) << np.uint64(36)) | np.uint64(17)
    pd.DataFrame({"particle_id": bc[:1], "geometry_id": [gid], "tx": [30.0], "ty": [1.0], "tz": [2.0], "tt": [0.1],
                  "tpx": [1.0], "tpy": [0.0], "tpz": [0.0], "te": [1.0], "deltapx": [0.0], "deltapy": [0.0],
                  "deltapz": [0.0], "deltae": [0.0], "index": [0]}).to_csv(tmp_path / "event000000003-hits.csv",
                                                                         index=False)
    cols = dict(particle_type=211, process=0, vt=0.0, m=0.1396, q=1)
    pd.DataFrame({"particle_id": bc, "vx": 0.0, "vy": 0.0, "vz": 0.0, "px": 1.0, "py": 0.0, "pz": 0.0, **cols}).to_csv(
        tmp_path / "event000000003-particles_initial.csv", index=False)
    pd.DataFrame({"particle_id": bc[1:], "vx": [1500.25], "vy": [-3.5], "vz": [2.0], "px": [0.9], "py": [0.1],
                  "pz": [0.0], **cols}).to_csv(tmp_path / "event000000003-particles_final.csv", index=False)
    out = str(tmp_path / "simhits.parquet")
    FIO.pack_sim_hits(str(tmp_path), out)
    p = pd.read_parquet(FIO.sim_particles_path(out)).set_index("particle_id")
    assert list(p.columns) == ["event", "initial", "final"] + [f"final_{k}" for k in FIO.FINAL_STATE]
    assert np.isnan(p.loc[bc[0], "final_vx"]) and not p.loc[bc[0], "final"]
    assert p.loc[bc[1], ["final_vx", "final_vy", "final_px"]].tolist() == [1500.25, -3.5, 0.9]
    status = pd.DataFrame({"event": 3, "particle_id": bc, "status": "simulated"})
    r = GC.silent_exits(status, p.reset_index())
    assert (r["outside"], r["without_final_state"]) == (1, 1)


# ------------------------------------------------------------------------------------------------ the driver
from test_overlay_version import det, hval_set, modules  # noqa: E402,F401  (fixtures: an assembled H-val-like set)


@pytest.mark.skipif(not os.path.exists(D.DETECTORS), reason="needs external/trackml/detectors.csv")
def test_run_on_an_assembled_set(hval_set):
    """The driver on two events whose gun particles carry the same TrackML ids (ids repeat across events): every
    simulated gun particle once per source, keyed by its truth row; synthetic Fatras hits sit at module centres, so
    only the bookkeeping is checked here."""
    work = os.path.dirname(hval_set["hroot"])
    truth = pd.read_parquet(os.path.join(hval_set["oroot"], "gun_truth_hval.parquet"))
    assert truth.duplicated("particle_id").any()                          # the same ids in both events
    r = GC.run("hval", work, hval_set["oroot"], sources=("fatras", "stored"))
    n_sim = int((truth.sim_status == "simulated").sum())
    for src in ("fatras", "stored"):
        s = {x["group"]: x for x in r["misses"][src]}
        assert s["all"]["particles"] == n_sim and s["iso"]["particles"] + s["dense"]["particles"] == n_sim
    assert r["nav_losses"]["gun_particles"] == len(truth) and r["events"] == 2
    assert r["silent_exits"]["final_states"] is False                    # the synthetic list has no final states
