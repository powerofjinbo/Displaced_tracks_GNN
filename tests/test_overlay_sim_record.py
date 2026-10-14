"""The Fatras record of the pileup overlay: failed particles parsed from the v15 log, the simulated-particle list
written by the sim-hit packer, the status of every gun particle (simulated, failed, below pMin), and the provenance check of
the overlay writer (seed rule, module table that belongs to the build)."""

import json

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun as GUN
from hnl.overlay import writer as W

LOG = """14:57:30    FatrasSimula   INFO      event 375 started
14:57:30    FatrasSimula   ERROR     event 375 particle id=(1|1008|1|0|0)|pdg=K+|q=1|m=0.493677|p=2.20003 failed to simulate with error MagneticFieldError:1: Interpolation out of bounds was requested
14:57:30    FatrasSimula   ERROR     event 450 particle id=(1|1001|12|0|0)|pdg=mu+|q=1|m=0.105658|p=7.7549 failed to simulate with error MagneticFieldError:1: Interpolation out of bounds was requested
14:57:30    FatrasSimula   ERROR     some other error line
"""


def test_failed_particles_parses_the_v15_log():
    f = FIO.failed_particles(LOG)
    assert f.event.tolist() == [375, 450]
    assert f.particle_id.dtype == np.uint64
    assert f.particle_id.tolist() == [int(FIO.barcode(1, 1008, 1)), int(FIO.barcode(1, 1001, 12))]
    assert (GUN.trackml_particle_id(f.particle_id.to_numpy()) >> 52).tolist() == [1008, 1001]
    assert f.error.str.startswith("MagneticFieldError").all()
    assert len(FIO.failed_particles("no failures here\n")) == 0


def test_barcode_matches_the_spec_v1_gun():
    assert int(FIO.barcode(1, 1003, 7)) == GUN.G.barcode(1, 1003, 7)
    assert int(FIO.barcode(1, 1003, 7, 2, 5)) == GUN.G.barcode(1, 1003, 7, 2, 5)


def _inputs():
    bc = FIO.barcode(1, 1001, np.arange(1, 5))
    return pd.DataFrame({"event": [0, 0, 0, 0], "particle_id": bc, "q": [1, -1, 1, 0],
                         "px": [2.0, 1.0, 0.05, 1.0], "py": [0.0, 0.0, 0.0, 0.0], "pz": [0.0, 0.0, 0.0, 0.0]})


def test_simulation_status_classifies_every_input():
    inp = _inputs()
    bc = inp.particle_id.to_numpy(np.uint64)
    sim = pd.DataFrame({"event": [0], "particle_id": bc[:1], "initial": [True], "final": [True]})
    fail = pd.DataFrame({"event": [0], "particle_id": bc[1:2], "error": ["x"]})
    st = FIO.simulation_status(inp, sim, fail, 0.1)
    assert dict(zip(st.particle_id.tolist(), st.status)) == {int(bc[0]): "simulated", int(bc[1]): "failed",
                                                             int(bc[2]): "below_pmin", int(bc[3]): "below_pmin"}


def test_simulation_status_refuses_inconsistent_records():
    inp = _inputs()
    bc = inp.particle_id.to_numpy(np.uint64)
    sim = pd.DataFrame({"event": [0], "particle_id": bc[:1], "initial": [True], "final": [True]})
    none = pd.DataFrame({"event": [], "particle_id": np.zeros(0, np.uint64), "error": []})
    with pytest.raises(ValueError, match="neither simulated nor logged"):
        FIO.simulation_status(inp, sim, none, 0.1)                 # particle 2 vanished without a log line
    with pytest.raises(ValueError, match="both simulated and logged"):
        FIO.simulation_status(inp, sim, pd.DataFrame({"event": [0, 0], "particle_id": bc[:2], "error": ["", ""]}),
                              0.1)
    extra = pd.concat([sim, pd.DataFrame({"event": [0], "particle_id": FIO.barcode(1, 1001, [9]), "initial": [True],
                                          "final": [True]})])
    with pytest.raises(ValueError, match="not inputs"):
        FIO.simulation_status(inp, extra, pd.DataFrame({"event": [0], "particle_id": bc[1:2], "error": [""]}), 0.1)


def test_pack_writes_the_simulated_particle_list(tmp_path):
    gid = (np.uint64(8) << np.uint64(56)) | (np.uint64(2) << np.uint64(36)) | np.uint64(17)
    bc = FIO.barcode(1, 1001, [1, 2])
    for e in (0, 1):
        pd.DataFrame({"particle_id": bc[:1], "geometry_id": [gid], "tx": [30.0], "ty": [1.0], "tz": [2.0],
                      "tt": [0.1], "tpx": [1.0], "tpy": [0.0], "tpz": [0.0], "te": [1.0], "deltapx": [0.0],
                      "deltapy": [0.0], "deltapz": [0.0], "deltae": [0.0], "index": [0]}).to_csv(
            tmp_path / f"event{e:09d}-hits.csv", index=False)
        pd.DataFrame({"particle_id": bc[:1]}).to_csv(tmp_path / f"event{e:09d}-particles_initial.csv", index=False)
        pd.DataFrame({"particle_id": bc[:1]}).to_csv(tmp_path / f"event{e:09d}-particles_final.csv", index=False)
    out = str(tmp_path / "simhits.parquet")
    r = FIO.pack_sim_hits(str(tmp_path), out, remove_csv=True)
    assert r["events"] == 2 and r["hits"] == 2 and r["csv_removed"] == 6
    p = pd.read_parquet(FIO.sim_particles_path(out))
    assert p.event.tolist() == [0, 1] and (p.particle_id.to_numpy(np.uint64) == bc[0]).all()
    assert p.initial.all() and p["final"].all()


def _summary(path, build, seed):
    rec = {"build": build, "seed": seed, "field_map": "map.txt", "field_map_md5": "x", "fatras": {"pmin_gev": 0.1},
           "events": 3, "particle_failures": 0}
    with open(path / "sim_summary.json", "w") as fh:
        json.dump(rec, fh)


def test_sim_provenance_checks_seed_and_build(tmp_path):
    gev = {"fatras_seed": 990014}
    _summary(tmp_path, "/work/x/acts/build-v15-zmirror", 990014)
    rec = W.sim_provenance(str(tmp_path), gev, D.PATCHED_LOOKUP)
    assert rec["pmin_gev"] == 0.1 and rec["seed"] == 990014
    with pytest.raises(ValueError, match="needs hits of"):
        W.sim_provenance(str(tmp_path), gev, D.UNPATCHED_LOOKUP)
    _summary(tmp_path, "/work/x/acts/build-v15", 990014)
    with pytest.raises(ValueError, match="z-mirror build"):
        W.sim_provenance(str(tmp_path), gev, D.PATCHED_LOOKUP)
    _summary(tmp_path, "/work/x/acts/build-v15-zmirror", 990015)
    with pytest.raises(ValueError, match="base seed"):
        W.sim_provenance(str(tmp_path), gev, D.PATCHED_LOOKUP)


def test_sim_provenance_reads_the_fatras_sbatch_record(tmp_path):
    with open(tmp_path / "fatras.json", "w") as fh:
        json.dump({"build": "/work/x/acts/build-v15-zmirror", "map": "m", "map_sha256": "s", "pmin_gev": 0.1,
                   "rnd_seed": 990014, "events": 3}, fh)
    assert W.sim_provenance(str(tmp_path), {"fatras_seed": 990014}, D.PATCHED_LOOKUP)["record"] == "fatras.json"
    with pytest.raises(FileNotFoundError):
        W.sim_provenance(str(tmp_path / "missing"), {"fatras_seed": 1}, D.PATCHED_LOOKUP)
