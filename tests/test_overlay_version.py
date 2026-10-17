"""Seeds, Fatras event numbers and version of an overlay set (addendum H section 2; hnl.overlay.writer): the seed
record and provenance manifest that `assemble` writes into the set's meta and sidecar, and the per-set Version check
(base seed and Fatras event numbers against configs/trackml/split_h_v1.json, overlay_hash and manifest against a
committed version file). An H-val-like set of two synthetic hosts is assembled end to end from synthetic Fatras output
with the split file's Fatras event numbers, which are not contiguous (0 and 36). Truth only; no model."""

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun as GUN
from hnl.overlay import split_h as SH
from hnl.overlay import writer as W
from hnl.trackml import convert as C
from hnl.trackml import overlay_sim as OS

from test_overlay_closure import _sim_hits
from test_overlay_writer import _store_host, host_tables_of

pytestmark = pytest.mark.skipif(not os.path.exists(C.DETECTORS), reason="needs external/trackml/detectors.csv")
HOSTS = (21117, 21289)                                  # tp_v1 val ids: Fatras events 0 and 36 of H-val
HS = {"vx": 0.01, "vy": -0.02, "vz": 35.0}
RESOLVED = "/cvmfs/unpacked.cern.ch/.flat/ab/ab242818-test"


@pytest.fixture(scope="module")
def det():
    return C.load_detectors()


@pytest.fixture(scope="module")
def modules():
    return pd.read_csv(C.DETECTORS)


def fake_fatras(gun_dir: str, sim_dir: str, seed: int) -> str:
    """What scripts/acts/overlay_sim.py sim and hnl.overlay.fatras_io pack leave for a gun set: packed sim hits of
    every gun particle above pMin (synthetic crossings at module centres), the simulated-particle list, fatras.log
    and sim_summary.json. Returns the packed sim-hit path."""
    with open(os.path.join(gun_dir, "gun_events.json")) as fh:
        gev = json.load(fh)
    mods, lookup = D.Modules.load(), D.load_lookup(D.PATCHED_LOOKUP)
    hits, simp = [], []
    for ev in gev["events"]:
        fe = int(ev["fatras_event"])
        parts = pd.read_csv(W.FIO_path(gun_dir, fe), dtype={"particle_id": np.uint64})
        sim = parts[np.sqrt(parts.px ** 2 + parts.py ** 2 + parts.pz ** 2) > 0.1]
        hits.append(_sim_hits(lookup, mods, sim, fe).assign(event=np.int32(fe)))
        simp.append(pd.DataFrame({"event": np.int32(fe), "particle_id": sim.particle_id.to_numpy(np.uint64),
                                  "initial": True, "final": True}))
    os.makedirs(sim_dir)
    path = os.path.join(sim_dir, "simhits.parquet")
    pd.concat(hits, ignore_index=True).to_parquet(path, index=False)
    pd.concat(simp, ignore_index=True).to_parquet(FIO.sim_particles_path(path), index=False)
    open(os.path.join(sim_dir, "fatras.log"), "w").close()
    st = OS.load_settings()
    nums = sorted(int(e["fatras_event"]) for e in gev["events"])
    with open(os.path.join(sim_dir, "sim_summary.json"), "w") as fh:
        json.dump({"build": "/work/x/acts/build-v15-zmirror", "seed": seed, "field_map": st["field_map"]["path"],
                   "field_map_md5": st["field_map"]["md5"], "fatras": st["fatras"], "executable_md5": "e" * 32,
                   "events": len(nums), "fatras_events": nums, "container": st["container"],
                   "container_resolved": RESOLVED, "lcg_view": st["lcg_view"], "particle_failures": 0}, fh)
    return path


@pytest.fixture(scope="module")
def hval_set(modules, det, tmp_path_factory):
    """An H-val-like set of two synthetic hosts, made by the production path with the split file's seeds."""
    from hnl.overlay import emulate as EM
    tmp = tmp_path_factory.mktemp("hval")
    hroot, groot, oroot = str(tmp / "host"), str(tmp / "gun"), str(tmp / "overlay")
    _store_host(hroot, "training_part01", {h: host_tables_of(modules, det, h)[0] for h in HOSTS})
    base, fe, rec0 = GUN.resolve_seeds("hval", list(HOSTS), None, None)
    GUN.make_gun_set({h: HS for h in HOSTS}, {h: W.overlay_id("hval", h) for h in HOSTS}, base, groot,
                     fatras_events=fe, record=rec0)
    sim = fake_fatras(groot, os.path.join(groot, "fatras_out"), base + 13)
    os.makedirs(oroot)
    r = W.assemble("hval", groot, sim, oroot, EM.TABLES, D.PATCHED_LOOKUP, host_root=hroot)
    return {"r": r, "fe": fe, "base": base, "oroot": oroot, "hroot": hroot, "tables": EM.TABLES}


def test_assembled_set_records_the_split_seeds_and_fatras_numbers(hval_set):
    meta, side, fe, base = hval_set["r"]["meta"], hval_set["r"]["sidecar"], hval_set["fe"], hval_set["base"]
    assert fe == {21117: 0, 21289: 36} and base == 600_000_001
    assert meta["base_seed"] == side["base_seed"] == base and meta["fatras_seed"] == base + 13
    assert meta["fatras_events"] == {str(W.overlay_id("hval", h)): n for h, n in fe.items()}
    assert {s["host"]: s["fatras_event"] for s in side["events"]} == fe
    assert meta["split_h"]["base_seed_from_split"] and meta["split_h"]["fatras_events_from_split"]
    assert meta["split_h"]["sha256"] == SH.split_sha256() and "emulation_seed_override" not in meta
    assert meta["sim"]["fatras_events"] == [0, 36] and meta["sim"]["container_resolved"] == RESOLVED
    with open(os.path.join(hval_set["oroot"], "meta_hval.json")) as fh:
        assert json.load(fh)["manifest"] == meta["manifest"]
    assert W.verify_set("hval", hval_set["oroot"], host_root=hval_set["hroot"])["problems"] == []


def test_manifest_lists_what_overlay_hash_does_not_cover(hval_set):
    m = hval_set["r"]["meta"]["manifest"]
    assert list(m["code_sha256"]) == list(W.MANIFEST_CODE)
    for f in ("src/hnl/overlay/gun.py", "src/hnl/overlay/writer.py", "src/hnl/overlay/fatras_io.py",
              "src/hnl/overlay/closure.py", "src/hnl/trackml/convert.py", "scripts/acts/overlay_sim.py"):
        assert m["code_sha256"][f] == C.sha256_file(os.path.join(W.REPO, f))
    rel = {os.path.relpath(p, W.REPO) for p in (W.SIM_SETTINGS, hval_set["tables"], D.PATCHED_LOOKUP)}
    assert set(m["config_sha256"]) == rel == {"configs/trackml/overlay_sim_v1.json",
                                              "configs/trackml/overlay_emulation_v1.json",
                                              "configs/trackml/module_map_acts_zmirror_v1.csv"}
    st = OS.load_settings()
    assert m["field_map_md5"] == st["field_map"]["md5"] and m["executable_md5"] == "e" * 32
    assert m["container"] == {"tag": st["container"], "resolved_at_fatras": RESOLVED, "lcg_view": st["lcg_view"]}
    # a Fatras record without these values (fatras.sbatch, or a run before H-T0) leaves them empty
    m2 = W.provenance_manifest({"build": "b"}, hval_set["tables"], D.PATCHED_LOOKUP)
    assert m2["executable_md5"] is None and m2["container"]["resolved_at_fatras"] is None
    assert m2["code_sha256"] == m["code_sha256"]


def test_version_check_passes_the_production_set(hval_set):
    meta, side = hval_set["r"]["meta"], hval_set["r"]["sidecar"]
    v = W.version_check("hval", meta, side)
    assert v["pass"], v["problems"]
    assert v["base_seed"] == v["split_base_seed"] == 600_000_001 and v["fatras_events_checked"] == 2
    committed = {"overlay_hash": meta["overlay_hash"], "manifest": meta["manifest"]}
    v = W.version_check("hval", meta, side, committed=copy.deepcopy(committed))
    assert v["pass"] and v["committed_version"] and v["manifest_differences"] == {}


def _fails(set_name, meta, side, split=None, committed=None, match=""):
    v = W.version_check(set_name, meta, side, split=split, committed=committed)
    assert not v["pass"] and any(match in p for p in v["problems"]), v["problems"]
    return v


def test_version_check_fails_other_seeds_numbers_and_versions(hval_set):
    meta, side = hval_set["r"]["meta"], hval_set["r"]["sidecar"]
    split = SH.load_split()
    _fails("hval", {**meta, "base_seed": 650001}, {**side, "base_seed": 650001}, match="split_h_v1.json")
    _fails("hval", meta, {**side, "base_seed": 600_000_002}, match="sidecar")
    _fails("hval", {k: v for k, v in meta.items() if k != "base_seed"}, side, match="no base seed")
    _fails("hval", {**meta, "sim": {**meta["sim"], "seed": 990114}}, side, match="Fatras seed")
    _fails("hval", {**meta, "emulation_seed_override": 1}, side, match="emulation")
    moved = {**side, "events": [{**side["events"][0], "fatras_event": 1}, *side["events"][1:]]}
    _fails("hval", meta, moved, match="Fatras event number")
    other = copy.deepcopy(split)
    other["sets"]["hval"]["base_seed"] = 1
    _fails("hval", meta, side, split=other, match="split_h_v1.json")
    # the same stored set read as another set: the closure base seed and list
    _fails("closure", meta, side, match="split_h_v1.json")
    _fails("hval", {k: v for k, v in meta.items() if k != "manifest"}, side, match="manifest")
    _fails("hval", meta, {**side, "overlay_hash": "0" * 64}, match="overlay_hash")
    committed = {"overlay_hash": meta["overlay_hash"], "manifest": copy.deepcopy(meta["manifest"])}
    _fails("hval", meta, side, committed={**committed, "overlay_hash": "1" * 64}, match="committed")
    changed = copy.deepcopy(committed)
    changed["manifest"]["code_sha256"]["src/hnl/overlay/gun.py"] = "0" * 64
    changed["manifest"]["container"]["resolved_at_fatras"] = "/cvmfs/elsewhere"
    v = _fails("hval", meta, side, committed=changed, match="manifest values")
    assert set(v["manifest_differences"]) == {"code_sha256.src/hnl/overlay/gun.py", "container.resolved_at_fatras"}


def test_version_check_of_an_h_test_set_needs_the_h_t1_lists(hval_set):
    meta, side = hval_set["r"]["meta"], hval_set["r"]["sidecar"]
    split = SH.load_split()
    m = {**meta, "base_seed": SH.base_seed("htest", split), "sim": {**meta["sim"], "seed": 620_000_014}}
    s = {**side, "base_seed": SH.base_seed("htest", split)}
    _fails("htest", m, s, split=split, match="cannot be checked")


def test_load_version_before_the_version_file_exists(tmp_path):
    assert W.load_version(str(tmp_path / "overlay_version_h_v1.json")) is None
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"overlay_hash": "a", "manifest": {}}))
    assert W.load_version(str(p)) == {"overlay_hash": "a", "manifest": {}}
