"""hnl.overlay.gun: hard-scatter vertex of a host, the 8 + 2 LLP layout with Delta R >= 0.8, the spec-v1 content of
each LLP, barcodes and the TrackML particle-id encoding (vertex field 1001-1010, positive int64), reproducibility."""

import io
import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun as GUN
from hnl.trackml import throughput as T

HS = {"vx": 0.01, "vy": -0.02, "vz": 35.0}


def _event(seed=1, event=59021117, host=21117):
    return GUN.make_gun_event(HS, np.random.default_rng([seed, event]), event, host)


def test_hard_scatter_vertex_is_the_most_common_prompt_vertex_of_vertex_1():
    pid = lambda v, i: (v << 52) | (i << 36)
    p = pd.DataFrame({"particle_id": [pid(1, 1), pid(1, 2), pid(1, 3), pid(1, 4), pid(2, 1), pid(2, 2), pid(2, 3)],
                      "vx": [0.1, 0.1, 0.1, 30.0, 0.2, 0.2, 0.2], "vy": [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                      "vz": [12.5, 12.5, 12.5, 40.0, -3.0, -3.0, -3.0]})
    hs = GUN.hard_scatter_vertex(p)
    assert (hs["vx"], hs["vy"], hs["vz"]) == pytest.approx((0.1, 0.0, 12.5)) and hs["n_at_vertex"] == 3
    with pytest.raises(ValueError):
        GUN.hard_scatter_vertex(p[(p.particle_id.to_numpy() >> 52) == 2])


def test_layout_components_and_separation():
    parts, tr = _event()
    llp = tr.drop_duplicates("llp").set_index("llp")
    assert llp.index.tolist() == list(range(1, 11))
    assert (llp.component == ["iso"] * 8 + ["dense"] * 2).all()
    eta, phi = llp.llp_eta.to_numpy(), llp.llp_phi.to_numpy()
    for i in range(10):
        for j in range(i):
            assert GUN.delta_r(eta[i], phi[i], eta[j], phi[j]) >= GUN.DR_MIN
    assert (llp.r_v.between(1, 800)).all() and (llp.vz.abs() <= 2900).all()
    assert (np.abs(llp.llp_eta) <= 3).all()
    # iso: two opposite charges of one species; dense: 3-8 prongs, total charge 0 or +-1
    for k, g in tr.groupby("llp"):
        if k <= 8:
            assert len(g) == 2 and g.q.sum() == 0 and g.pdg.abs().nunique() == 1
        else:
            assert 3 <= len(g) <= 8 and abs(g.q.sum()) <= 1
    # daughters start at their LLP's decay point, which lies on the line from the hard-scatter vertex
    m = parts.merge(tr[["acts_barcode", "llp", "r_v", "llp_phi"]], left_on="particle_id", right_on="acts_barcode")
    assert np.allclose(np.hypot(m.vx - HS["vx"], m.vy - HS["vy"]), m.r_v)
    assert np.allclose(np.arctan2(m.vy - HS["vy"], m.vx - HS["vx"]) % (2 * np.pi), m.llp_phi % (2 * np.pi))
    assert list(parts.columns) == FIO.PARTICLE_COLUMNS and parts.particle_id.dtype == np.uint64


def test_barcodes_and_trackml_ids():
    parts, tr = _event()
    d = GUN.decode_barcode(parts.particle_id.to_numpy(np.uint64).view(np.int64))
    assert (d["vprim"] == 1).all() and d["vsec"].min() >= 1001 and d["vsec"].max() <= 1010
    f = GUN.trackml_fields(tr.particle_id.to_numpy())
    assert (tr.particle_id > 0).all() and tr.particle_id.dtype == np.int64
    assert np.array_equal(f["vertex"], 1000 + tr.llp.to_numpy())
    assert np.array_equal(f["primary"], GUN.decode_barcode(tr.acts_barcode.to_numpy(np.uint64).view(np.int64))[
        "particle"])
    assert (f["generation"] == 0).all() and (f["secondary"] == 0).all() and (f["process"] == 0).all()
    assert not pd.Series(tr.particle_id).duplicated().any()
    # a Fatras secondary of a gun particle keeps its LLP vertex and carries generation / sub-particle
    b = GUN.G.barcode(1, 1003, 2, gen=1, sub=5)
    assert GUN.trackml_fields(GUN.trackml_particle_id(b)) == {"vertex": 1003, "primary": 2, "generation": 1,
                                                              "secondary": 5, "process": 0}


@pytest.mark.parametrize("vsec", [4001, 2048, 1011, 1000, 7])
def test_vertex_fields_outside_the_gun_range_are_refused(vsec):
    with pytest.raises(ValueError):
        GUN.trackml_particle_id(GUN.G.barcode(1, vsec, 1))


def test_gun_ids_read_back_as_positive_int64_by_the_converter_csv_reader():
    _, tr = _event()
    csv = pd.DataFrame({"particle_id": tr.particle_id, "particle_type": tr.pdg, "vx": tr.vx, "vy": tr.vy,
                        "vz": tr.vz, "px": tr.px, "py": tr.py, "pz": tr.pz, "q": tr.q, "nhits": 0}).to_csv(index=False)
    back = pd.read_csv(io.StringIO(csv), dtype=T.CSV_DTYPES["particles"])
    assert back.particle_id.dtype == np.int64 and (back.particle_id > 0).all()
    assert np.array_equal(back.particle_id.to_numpy(), tr.particle_id.to_numpy())


def test_first_llp_matches_the_spec_v1_gun():
    """With no LLP placed yet, the first (isolated) LLP draws exactly as scripts/acts/gun.py make_event."""
    gen = pd.DataFrame({"particle_id": np.array([GUN.G.barcode(1, 0, 1)], np.uint64), "vx": [HS["vx"]],
                        "vy": [HS["vy"]], "vz": [HS["vz"]], "vt": [0.0]})
    ref, _ = GUN.G.make_event(gen, "iso", 1, np.random.default_rng([3, 9]), 9)
    parts, _ = GUN.make_gun_event(HS, np.random.default_rng([3, 9]), 9, 0)
    first = parts[GUN.decode_barcode(parts.particle_id.to_numpy(np.uint64).view(np.int64))["vsec"] == 1001]
    assert len(first) == len(ref) == 2
    for c in ("particle_type", "vx", "vy", "vz", "px", "py", "pz", "q"):
        assert np.allclose(first[c].to_numpy(np.float64), ref[c].to_numpy(np.float64)), c


def test_reproducible():
    a, ta = _event(seed=5)
    b, tb = _event(seed=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


def test_placement_redraw_rate_and_separation():
    """Azimuth redraws per LLP for 8 + 2 LLPs at Delta R >= 0.8. Analytically about 0.4-0.5 for eta ~ U(-3, 3). The
    study's 0.88 (study_design overlap_mc) is a mean that includes pseudo-events where its MC kept eta fixed and gave up
    after 1000 azimuths; here eta is redrawn instead, which must stay rare."""
    rng = np.random.default_rng(12)
    per, n_eta = [], 0
    for _ in range(3000):
        placed = []
        for _k in GUN.LLP_INDICES:
            eta, phi, rv, a, b = GUN.place_llp(rng, 0.0, placed)
            assert all(GUN.delta_r(eta, phi, e, f) >= GUN.DR_MIN for e, f in placed)
            placed.append((eta, phi))
            per.append(a)
            n_eta += b
    per = np.array(per)
    assert 0.3 < per[per < GUN.MAX_PHI_TRIES].mean() < 0.8
    assert n_eta / len(per) < 0.005


def test_make_gun_set_writes_fatras_inputs(tmp_path):
    hv = {21117: HS, 21124: {"vx": 0.0, "vy": 0.0, "vz": -80.0}}
    rec = GUN.make_gun_set(hv, {21117: 59021117, 21124: 59021124}, 990001, str(tmp_path),
                           fatras_events={21117: 0, 21124: 1})
    assert [e["overlay_event"] for e in rec["events"]] == [59021117, 59021124]
    p0 = pd.read_csv(FIO.particles_path(str(tmp_path / "fatras_in"), 0), dtype={"particle_id": np.uint64})
    ref, _ = GUN.make_gun_event(HS, np.random.default_rng([990001 + 7, 59021117]), 59021117, 21117)
    assert np.array_equal(p0.particle_id.to_numpy(), ref.particle_id.to_numpy())
    assert np.allclose(p0.px.to_numpy(), ref.px.to_numpy())
    tr = pd.read_parquet(tmp_path / "gun_truth.parquet")
    assert set(tr.event) == {59021117, 59021124} and set(tr.fatras_event) == {0, 1}


def test_make_gun_set_uses_the_given_fatras_event_numbers(tmp_path):
    """A host's Fatras input file carries its own number (gaps allowed), and its gun particles depend on its overlay id
    only: the same host made alone, in another job, gives the same file under the same number."""
    hv = {21117: HS, 21124: {"vx": 0.0, "vy": 0.0, "vz": -80.0}, 21289: {"vx": 0.02, "vy": 0.0, "vz": 5.0}}
    oid = {h: 50_000_000 + h for h in hv}
    fe = {21117: 0, 21124: 1, 21289: 36}
    rec = GUN.make_gun_set(hv, oid, 600_000_001, str(tmp_path / "all"), fatras_events=fe)
    assert {e["host"]: e["fatras_event"] for e in rec["events"]} == fe
    assert rec["fatras_seed"] == 600_000_014
    fin = tmp_path / "all" / "fatras_in"
    assert sorted(os.listdir(fin)) == [os.path.basename(FIO.particles_path(str(fin), n)) for n in (0, 1, 36)]
    GUN.make_gun_set({21289: hv[21289]}, {21289: oid[21289]}, 600_000_001, str(tmp_path / "one"),
                     fatras_events={21289: 36})
    a = (fin / os.path.basename(FIO.particles_path(str(fin), 36))).read_bytes()
    b = (tmp_path / "one" / "fatras_in" / os.path.basename(FIO.particles_path(str(fin), 36))).read_bytes()
    assert a == b
    tr = pd.read_parquet(tmp_path / "all" / "gun_truth.parquet")
    assert dict(zip(tr.host, tr.fatras_event)) == fe


@pytest.mark.parametrize("fe", [{21117: 0}, {21117: 0, 21124: 0}, {21117: -1, 21124: 2},
                                {21117: 0, 21124: 1, 21127: 2}])
def test_fatras_event_numbers_must_be_one_distinct_number_per_host(tmp_path, fe):
    hv = {21117: HS, 21124: HS}
    with pytest.raises(ValueError):
        GUN.make_gun_set(hv, {21117: 50_021_117, 21124: 50_021_124}, 600_000_001, str(tmp_path), fatras_events=fe)


def test_resolve_seeds_reads_the_split_file(capsys):
    from hnl.overlay import split_h as SH
    split = SH.load_split()
    val = split["hval"]
    hosts = [val[5], val[0], val[149]]
    base, fe, rec = GUN.resolve_seeds("hval", hosts, None, None)
    assert base == split["sets"]["hval"]["base_seed"] == 600_000_001
    assert fe == {val[5]: 5, val[0]: 0, val[149]: 149}
    assert rec["set"] == "hval" and rec["split_h"]["base_seed_from_split"] and rec["split_h"]["fatras_events_from_split"]
    assert rec["split_h"]["sha256"] == SH.split_sha256() and rec["split_h"]["path"] == "configs/trackml/split_h_v1.json"
    assert capsys.readouterr().out == ""
    base, fe, rec = GUN.resolve_seeds("closure", f"{val[1]}:{val[2]}", 990101, "7:9")
    assert base == 990101 and fe == {val[1]: 7, val[2]: 9}
    assert not rec["split_h"]["base_seed_from_split"] and not rec["split_h"]["fatras_events_from_split"]
    assert "Version check" in capsys.readouterr().out
    with pytest.raises(KeyError):
        GUN.resolve_seeds("pilot", [21117], None, None)              # not a pilot host
    with pytest.raises(LookupError):
        GUN.resolve_seeds("htest", [21117], None, None)              # no H-test list before H-T1
    with pytest.raises(ValueError):
        GUN.resolve_seeds("hval", [val[0], val[1]], None, "3")
