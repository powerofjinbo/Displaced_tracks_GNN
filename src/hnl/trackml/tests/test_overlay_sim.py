"""hnl.trackml.overlay_sim and scripts/acts/overlay_sim.py: the RZ map order, the Fatras command, the log parser,
the bending residual and B_eff on exact helices, the loop field, the F1 tables, the module map and containment, and
the probe gun layout.

    pytest src/hnl/trackml/tests/test_overlay_sim.py
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from hnl.trackml import overlay_sim as OS
from hnl.trackml.split import REPO

K = OS.K_GEV_PER_T_M


def _write_map(path, r, z, r_fastest=True):
    with open(path, "w") as fh:
        fh.write("# r z Br Bz\n")
        pairs = [(a, b) for b in z for a in r] if r_fastest else [(a, b) for a in r for b in z]
        for a, b in pairs:
            fh.write(f"{a} {b} {0.001 * a + 0.0001 * b} {2.0 + 0.0002 * b - 0.0003 * a}\n")


def test_read_rz_map_order(tmp_path):
    r, z = [0.0, 20.0, 40.0], [-40.0, -20.0, 0.0, 20.0]
    p = tmp_path / "m.txt"
    _write_map(p, r, z)
    rn, zn, br, bz = OS.read_rz_map(str(p))
    assert br.shape == (3, 4) and np.allclose(rn, r) and np.allclose(zn, z)
    assert np.isclose(br[2, 1], 0.001 * 40 + 0.0001 * -20)
    assert np.isclose(bz[1, 3], 2.0 + 0.0002 * 20 - 0.0003 * 20)
    q = tmp_path / "zfast.txt"
    _write_map(q, r, z, r_fastest=False)
    with pytest.raises(ValueError):
        OS.read_rz_map(str(q))


def test_node_field_bilinear_and_clamp(tmp_path):
    p = tmp_path / "m.txt"
    _write_map(p, [0.0, 20.0, 40.0], [-40.0, -20.0, 0.0, 20.0])
    f = OS.NodeField.from_rz_map(str(p))
    br, bz = f(np.array([10.0, 33.0, 100.0]), np.array([-5.0, 12.0, 500.0]))
    assert np.allclose(br[:2], 0.001 * np.array([10.0, 33.0]) + 0.0001 * np.array([-5.0, 12.0]))
    assert np.allclose(bz[:2], 2.0 + 0.0002 * np.array([-5.0, 12.0]) - 0.0003 * np.array([10.0, 33.0]))
    assert np.isclose(br[2], 0.001 * 40 + 0.0001 * 20)          # clamped to the last node


def test_fatras_args_and_container():
    st = OS.load_settings()
    args = OS.fatras_args(st, "/work/in", "/work/out", seed=650014, threads=4, skip=3, events=10)
    s = " ".join(args)
    assert args[0] == "ActsExampleFatrasGeneric"
    assert "--fatras-pmin-gev 0.1" in s and "--fatras-em-radiation false" in s
    assert "--bf-map-type rz" in s and f"--bf-map-file {st['field_map']['path']}" in s
    assert "--rnd-seed 650014" in s and "--skip 3" in s and "-n 10" in s and "--output-csv" in s
    assert "--fatras-hits sensitive" in s
    cmd = OS.container_command(st, "/work/b/build", args, ["/work/in", "/work/out"])
    assert cmd[:3] == ["apptainer", "exec", "--cleanenv"] and st["container"] in cmd
    assert "/work/b/build/bin/ActsExampleFatrasGeneric" in cmd[-1] and "/work/b/build/lib64" in cmd[-1]
    with pytest.raises(ValueError):
        OS.container_command(st, "/work/b/build", args, ["/ceph/x"])


def test_event_range(tmp_path):
    for e in range(5, 10):
        (tmp_path / f"event{e:09d}-particles.csv").write_text("x\n")
    assert OS.event_range(str(tmp_path)) == (0, 5, 5)
    assert OS.event_range(str(tmp_path), 7, 2) == (2, 7, 2)
    with pytest.raises(FileNotFoundError):
        OS.event_range(str(tmp_path), 8, 5)
    (tmp_path / f"event{3:09d}-particles.csv").write_text("x\n")
    with pytest.raises(FileNotFoundError):
        OS.event_range(str(tmp_path), 5, 2)          # event 4 missing below the requested range


def test_parse_fatras_log():
    text = "\n".join([
        "12:23:02    Simulation     ERROR     Step failed with MagneticFieldError:1: Interpolation out of bounds",
        "12:23:02    FatrasSimula   ERROR     event 7 particle 4503599644147712 failed to simulate with error 1: x",
        "12:25:31    Sequencer      INFO      Processed 200 events in 1.749547 s (wall clock)"])
    r = OS.parse_fatras_log(text)
    assert r["particle_failures"] == 1 and r["failed"][0] == {"event": 7, "particle": "4503599644147712"}
    assert r["error_lines"] == 2 and r["fatal_lines"] == 0 and r["events_processed"] == 200
    assert OS.parse_fatras_log("")["events_processed"] is None


def _helix(pt, q, phi0, cot, b, z0=0.0, ds=0.05, smax=4.0):
    """Exact helix in a uniform Bz = b [T] (Bz > 0 turns q > 0 clockwise): points every ds metres of transverse
    path; positions in mm, momenta in GeV."""
    kappa = K * q * b / pt
    s = np.arange(0.0, smax, ds)
    phi = phi0 - kappa * s
    x = 1000 * (np.sin(phi0) - np.sin(phi)) / kappa
    y = 1000 * (np.cos(phi) - np.cos(phi0)) / kappa
    z = z0 + 1000 * cot * s
    p = np.stack([pt * np.cos(phi), pt * np.sin(phi), np.full_like(phi, pt * cot)], 1)
    keep = (np.hypot(x, y) < 1000) & (np.abs(z) < 3100)
    return np.stack([x, y, z], 1)[keep], p[keep]


def test_bending_residual_uniform_field_exact():
    x, p = _helix(1.5, 1, 0.3, 0.8, 2.0)
    q = np.ones(len(x) - 1)
    res, bend = OS.bending_residual(x[:-1], x[1:], p[:-1], p[1:], q, OS.NodeField([0, 2000], [-4000, 4000],
                                                                                    np.zeros((2, 2)),
                                                                                    np.full((2, 2), 2.0)))
    assert np.abs(res).max() < 1e-9
    _, bend_w = OS.bending_residual(x[:-1], x[1:], p[:-1], p[1:], q, OS.NodeField([0, 2000], [-4000, 4000],
                                                                                    np.zeros((2, 2)),
                                                                                    np.full((2, 2), 2.1)))
    # a model field 0.1 T too strong over-predicts the bend by K * 0.1 T * L_T; the residual has one sign
    lt = np.hypot(*(x[1:, :2] - x[:-1, :2]).T) / 1000
    assert np.allclose(np.abs(bend_w), K * 0.1 * lt, rtol=0.02)
    assert np.all(np.sign(bend_w) == np.sign(bend_w[0]))


def test_beff_uniform_field():
    x, p = _helix(3.0, -1, -1.0, 0.5, 2.0, ds=0.1)
    b = OS.beff(x[:-1], x[1:], p[:-1], p[1:], -np.ones(len(x) - 1))
    assert np.allclose(b, 2.0, rtol=1e-3)


def test_loop_field_axis_and_symmetry():
    R = 1250.0
    z = np.array([-900.0, 0.0, 400.0])
    br, bz = OS.loop_field(np.zeros(3), z, R)
    assert np.allclose(bz, 0.5 * R * R / (R * R + z * z) ** 1.5) and np.allclose(br, 0)
    br1, bz1 = OS.loop_field(np.full(3, 1e-3), z, R)
    assert np.allclose(bz1, bz, rtol=1e-6)
    a, _ = OS.loop_field(np.array([300.0]), np.array([500.0]), R)
    b, _ = OS.loop_field(np.array([300.0]), np.array([-500.0]), R)
    assert np.isclose(a[0], -b[0]) and a[0] != 0
    tbr, tbz = OS.loop_model(np.array([0.0]), np.array([0.0]), [2.0, 3.0], [(R, 100.0), (R, -100.0)])
    assert np.isclose(tbz[0], 5.0 * 0.5 * R * R / (R * R + 100.0 ** 2) ** 1.5)


def _segments_from_helices(b_true, n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n):
        q = rng.choice([-1, 1])
        cot = rng.choice([-1, 1]) * np.sinh(rng.uniform(0, 3))
        x, p = _helix(rng.uniform(1.5, 5), q, rng.uniform(-np.pi, np.pi), cot, b_true, ds=0.04, smax=3.0)
        k = len(x) - 1
        if k < 1:
            continue
        ends = (("0", slice(0, k)), ("1", slice(1, k + 1)))
        d = {f"t{c}{s}": x[sl, i] for i, c in enumerate("xyz") for s, sl in ends}
        d.update({f"tp{c}{s}": p[sl, i] for i, c in enumerate("xyz") for s, sl in ends})
        d.update({f"tp{c}0a": p[:k, i] for i, c in enumerate("xyz")})
        rows.append(pd.DataFrame({**d, "volume0": 8, "volume1": 8, "layer0": np.arange(k) * 2,
                                  "layer1": np.arange(k) * 2 + 2, "q": q, "pdg": 211 * q, "rv": 0.0}))
    return pd.concat(rows, ignore_index=True)


def test_f1_tables_pass_and_fail():
    seg = _segments_from_helices(2.0)
    good = OS.NodeField([0, 2000], [-4000, 4000], np.zeros((2, 2)), np.full((2, 2), 2.0))
    r = OS.f1_tables(seg, good)
    assert r["all_bands_populated"] and r["pass"] and r["max_abs_mean_MeV"] < 1e-6
    assert {b["side"] for b in r["bands"]} == {"+z", "-z"} and len(r["bands"]) == 2 * len(OS.Z_BANDS)
    # a field off by 0.01 T on the -z side only fails there: L_T = 40 mm gives K * 0.01 T * 0.04 m = 0.12 MeV
    zn = np.array([-4000.0, -1.0, 0.0, 4000.0])
    skew = OS.NodeField([0, 2000], zn, np.zeros((2, 4)), np.array([[2.01, 2.01, 2.0, 2.0]] * 2))
    r2 = OS.f1_tables(seg, skew)
    minus = [b for b in r2["bands"] if b["side"] == "-z"]
    plus = [b for b in r2["bands"] if b["side"] == "+z"]
    assert not r2["pass"] and all(not b["pass"] for b in minus) and all(b["pass"] for b in plus)


def test_reference_closure():
    """Uniform field: the scoring is exact for any number of samples. A field falling along z: 96 samples close
    at least ten times better than 6."""
    uni = OS.NodeField([0, 2000], [-4000, 4000], np.zeros((2, 2)), np.full((2, 2), 2.0))
    c = OS.reference_closure(uni, n_tracks=60, nsubs=(6,))
    rows = [r for v in c["nsub6"]["by_segment_length"].values() for r in v]
    assert sum(r["n"] for r in rows) > 1000 and c["nsub6"]["max_abs_mean_MeV"] < 1e-6
    z = np.linspace(-4000, 4000, 81)
    bz = np.tile(2.0 / (1 + (z / 2500.0) ** 4), (2, 1))
    br = np.tile(0.3 * np.tanh(z / 1500.0), (2, 1)) * np.array([[0.0], [1.0]])
    fall = OS.NodeField([0, 2000], z, br, bz)
    c = OS.reference_closure(fall, n_tracks=60, nsubs=(6, 96))
    assert c["nsub96"]["max_abs_mean_MeV"] < min(0.01, c["nsub6"]["max_abs_mean_MeV"] / 10)


def test_read_module_map(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("volume_id,layer_id,acts_module_id,trackml_module_id\n7,2,1,4\n8,2,5,5\n")
    m = OS.read_module_map(str(p))
    assert m.columns.tolist() == ["volume", "layer", "acts_module", "trackml_module"]
    assert m.trackml_module.tolist() == [4, 5]


def _det(rows, acts=False):
    cols = ["volume_id", "layer_id", "module_id", "cx", "cy", "cz"] + OS.ROT
    d = pd.DataFrame(rows, columns=cols)
    d["module_t"], d["module_minhu"], d["module_maxhu"], d["module_hv"] = 0.15, 8.0, 12.0, 30.0
    return d


def test_module_map_and_containment():
    eye = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    tml = _det([[8, 2, 1, 32.0, 0.0, 0.0, *eye], [8, 2, 2, 0.0, 32.0, 0.0, *eye],
                [7, 2, 1, 60.0, 0.0, -1500.0, *eye], [7, 2, 2, -60.0, 0.0, -1500.0, *eye]])
    acts = _det([[8, 2, 1, 32.0, 0.0, 0.0, *eye], [8, 2, 2, 0.0, 32.0, 0.0, *eye],
                 [7, 2, 1, -60.0, 0.0, -1497.0, *eye], [7, 2, 2, 60.0, 0.0, -1500.0, *eye]])
    m = OS.build_module_map(acts, tml)
    m7 = m[m.volume == 7].set_index("acts_module").trackml_module.to_dict()
    assert m7 == {1: 2, 2: 1} and (m[m.volume == 8].acts_module == m[m.volume == 8].trackml_module).all()
    hits = pd.DataFrame({
        "volume": [8, 8, 7, 7, 7, 9],
        "layer": [2, 2, 2, 2, 2, 2],
        "module": [1, 2, 2, 2, 1, 1],
        # inside; outside in u at v = -30 (half width 8); inside the endcap module; outside in v; 3 mm off plane;
        # a volume without map entries
        "tx": [32.0 + 5.0, 9.0, 60.0 + 7.0, 60.0, -60.0, 0.0],
        "ty": [0.0 - 29.0, 32.0 - 29.0, 0.0, 31.0, 0.0, 0.0],
        "tz": [0.0, 0.0, -1500.0, -1500.0, -1497.0, 1500.0]})
    f = OS.containment(hits, m, tml)
    assert f.ok.tolist() == [True, False, True, False, False, False]
    assert f.on_plane.tolist()[:5] == [True, True, True, True, False] and not f.mapped.iloc[5]
    s = OS.containment_summary(f, [np.zeros(6), np.array([1, 1, 2, 3, 3, 4])])
    assert s["failed"] == 4 and s["off_plane"] == 1 and s["unmapped"] == 1 and not s["pass"]
    assert s["particles"] == 4 and s["particles_with_failed_hit"] == 3
    assert np.isclose(s["per_volume"]["7"]["max_abs_w_mm"], 3.0)


def _script():
    path = os.path.join(REPO, "scripts", "acts", "overlay_sim.py")
    spec = importlib.util.spec_from_file_location("overlay_sim_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_gun_layout():
    S = _script()
    pv = (0.01, -0.02, 37.0, 0.0)
    df, truth, _ = S.probe_gun_event(pv, np.random.default_rng([990001, 3]))
    t = pd.DataFrame(truth)
    bc = df.particle_id.to_numpy().astype(np.uint64)
    vsec = ((bc >> np.uint64(40)) & np.uint64(0xFFF)).astype(int)
    assert set(vsec) == set(range(1001, 1011)) and ((bc >> np.uint64(52)) == 1).all()
    assert not df.particle_id.duplicated().any() and set(df.q) <= {-1, 1}
    n = t.groupby("llp").n_prongs.first()
    assert (n[n.index <= 8] == 2).all() and n[n.index > 8].between(3, 8).all()
    assert (t.groupby("llp").component.first() == ["iso"] * 8 + ["dense"] * 2).all()
    llp = t.groupby("llp")[["llp_eta", "llp_phi", "r_v", "vx", "vy", "vz"]].first()
    e, f = llp.llp_eta.to_numpy(), llp.llp_phi.to_numpy()
    dphi = (f[:, None] - f[None, :] + np.pi) % (2 * np.pi) - np.pi
    dr = np.hypot(e[:, None] - e[None, :], dphi)[np.triu_indices(10, 1)]
    assert dr.min() >= 0.8
    assert np.allclose(np.hypot(llp.vx - pv[0], llp.vy - pv[1]), llp.r_v)
    assert np.allclose(llp.vz - pv[2], llp.r_v * np.sinh(llp.llp_eta)) and (llp.vz.abs() <= 2900).all()
    df2, _, _ = S.probe_gun_event(pv, np.random.default_rng([990001, 3]))
    pd.testing.assert_frame_equal(df, df2)


def test_probe_gun_separation_always_holds():
    """The rule holds for every LLP pair, also in a crowded layout (24 isolated LLPs) where the azimuths of some eta
    run out and eta is drawn again."""
    S = _script()
    eta_redraws = 0
    for ev in range(20):
        _, truth, n = S.probe_gun_event((0.0, 0.0, 0.0, 0.0), np.random.default_rng([12345, ev]), n_iso=24, n_dense=0)
        llp = pd.DataFrame(truth).groupby("llp")[["llp_eta", "llp_phi"]].first()
        e, f = llp.llp_eta.to_numpy(), llp.llp_phi.to_numpy()
        dphi = (f[:, None] - f[None, :] + np.pi) % (2 * np.pi) - np.pi
        assert len(e) == 24 and np.hypot(e[:, None] - e[None, :], dphi)[np.triu_indices(24, 1)].min() >= 0.8
        eta_redraws += n[1]
    assert eta_redraws > 0


def test_free_azimuths():
    S = _script()
    assert S.free_azimuths(0.0, [], 0.8) == [(0.0, 2 * np.pi)]
    h = np.sqrt(0.8 ** 2 - 0.6 ** 2)
    free = S.free_azimuths(0.0, [(0.0, 0.1), (0.6, 3.0)], 0.8)      # arcs 0.1 +- 0.8 (wraps) and 3.0 +- h
    assert np.allclose(free, [(0.9, 3.0 - h), (3.0 + h, 2 * np.pi - 0.7)])
    ring = [(0.0, f) for f in np.arange(0, 2 * np.pi, 1.0)]
    assert S.free_azimuths(0.0, ring, 0.8) == []
