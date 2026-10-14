"""hnl.overlay.truth_checks: local-frame residuals and incidence slopes, the C1 comparison at the test sample's
incidence, the CSV tar reader, and the same-module near-coincidence counts, on synthetic inputs."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from hnl.overlay import c1 as C1
from hnl.overlay import digitise as D
from hnl.overlay import truth_checks as TC
from hnl.overlay import writer as W
from hnl.trackml import convert as C


def _modules():
    """Two modules with identity rotation (u = x, v = y, w = z) centred at the origin and at z = 100 mm."""
    n = 2
    return D.Modules(key=D.module_key([8, 8], [2, 2], [1, 2]), c=np.array([[0.0, 0, 0], [0, 0, 100.0]]),
                     R=np.repeat(np.eye(3)[None], n, axis=0), t=np.full(n, 0.15), minhu=np.full(n, 8.0),
                     maxhu=np.full(n, 8.0), hv=np.full(n, 30.0), pu=np.full(n, 0.05), pv=np.full(n, 0.05625),
                     volume=np.array([8, 8]))


def test_local_features_residuals_and_slopes():
    mods = _modules()
    # true point (1, 2, 0.05) with direction (0.5, -0.25, 1): on w = 0 it sits at (0.975, 2.0125)
    txyz = np.array([[1.0, 2.0, 0.05], [0.0, 0.0, 100.0]])
    tp = np.array([[0.5, -0.25, 1.0], [1.0, 0.0, 0.0]])            # the second runs in the module plane
    xyz = np.array([[0.975 + 0.003, 2.0125 - 0.004, 0.0], [0.0, 0.0, 100.0]])
    f = TC.local_features(xyz, [8, 8], [2, 2], [1, 2], txyz, tp, mods, ncells=[2, 1], charge=[0.3, 0.1],
                          span0=[2, 1], span1=[1, 1])
    assert len(f) == 1 and f.keep_index.tolist() == [0]
    assert f.tan_u.iloc[0] == pytest.approx(0.5) and f.abs_tan_v.iloc[0] == pytest.approx(0.25)
    assert f.du.iloc[0] == pytest.approx(0.003) and f.dv.iloc[0] == pytest.approx(-0.004)
    assert f.charge_cos.iloc[0] == pytest.approx(0.3 / np.sqrt(1 + 0.25 + 0.0625))
    with pytest.raises(KeyError):
        TC.local_features(xyz[:1], [8], [2], [3], txyz[:1], tp[:1], mods)


def _feat(volume, tan_u, tan_v, du, rng):
    n = len(tan_u)
    return C1.features(np.full(n, volume), tan_u, tan_v, np.ones(n), np.ones(n), np.ones(n), du, rng.normal(0, 1e-3, n))


def test_at_test_incidence_reweights_the_reference():
    rng = np.random.default_rng(3)
    # reference: two incidence cells with different mean residuals; the test sample populates only the second one
    tu_r = np.r_[np.full(4000, 0.01), np.full(4000, 0.5)]
    ref = _feat(13, tu_r, np.zeros(8000), np.where(tu_r < 0.1, 0.0, 0.002) + rng.normal(0, 1e-4, 8000), rng)
    test = _feat(13, np.full(300, 0.5), np.zeros(300), 0.002 + rng.normal(0, 1e-4, 300), rng)
    rows = {r["quantity"]: r for r in TC.at_test_incidence(ref, test, n_boot=50)}
    b = rows["bias_u"]
    assert b["ref"] == pytest.approx(2.0, abs=0.05) and b["test"] == pytest.approx(2.0, abs=0.05)   # um
    assert abs(b["diff"]) < 3 * b["sigma_test"] + 0.02 and b["coverage"] == 1.0
    assert rows["mad_u"]["ref"] == pytest.approx(0.1, rel=0.15)


def test_csv_members_reads_the_writer_tar(tmp_path):
    mem = {"hits": pd.DataFrame({"hit_id": [1], "x": [1.5], "y": [2.0], "z": [3.0], "volume_id": [8], "layer_id": [2],
                                 "module_id": [1]}),
           "cells": pd.DataFrame({"hit_id": [1], "ch0": [3], "ch1": [4], "value": [0.2]}),
           "particles": pd.DataFrame({"particle_id": [(1001 << 52) | (1 << 36)], "particle_type": [211], "vx": [0.0],
                                      "vy": [0.0], "vz": [0.0], "px": [1.0], "py": [0.0], "pz": [0.0], "q": [1],
                                      "nhits": [1]}),
           "truth": pd.DataFrame({"hit_id": [1], "particle_id": [(1001 << 52) | (1 << 36)], "tx": [1.5], "ty": [2.0],
                                  "tz": [3.0], "tpx": [1.0], "tpy": [0.0], "tpz": [0.0], "weight": [0.0]})}
    gz = W.members_gz({k: m[W.CSV_COLUMNS[k]] for k, m in mem.items()})
    W.write_csv_tar(str(tmp_path / "s.tar"), {59021117: gz})
    r = TC.csv_members(str(tmp_path / "s.tar"))
    assert list(r) == [59021117]
    assert r[59021117]["hits_gz"] == gz["hits"]
    assert r[59021117]["particles"].particle_id.dtype == np.int64
    assert r[59021117]["particles"].particle_id.iloc[0] == (1001 << 52) | (1 << 36)


def _hits_table(rows):
    cols = list(zip(*rows))
    n = len(rows)
    return pa.Table.from_pydict({
        "event": pa.array(np.full(n, 1, np.int32)), "x": pa.array(np.array(cols[0], np.float32)),
        "y": pa.array(np.array(cols[1], np.float32)), "z": pa.array(np.zeros(n, np.float32)),
        "volume": pa.array(np.full(n, 8, np.uint8)), "layer": pa.array(np.full(n, 2, np.uint8)),
        "module": pa.array(np.array(cols[2], np.uint16)), "pid": pa.array(np.array(cols[3], np.int64))})


def test_near_coincidences_counts_other_particles_on_the_module():
    g1, g2, h1, h2 = (1001 << 52) | (1 << 36), (1002 << 52) | (1 << 36), 1 << 52 | 5 << 36, 1 << 52 | 6 << 36
    # module 1: a gun hit 0.05 mm from a host target hit; module 2: a gun hit 0.5 mm from noise and 3 mm from a
    # second gun particle; module 3: a lone host target hit, and a second hit of the same target 0.01 mm away
    hits = _hits_table([(0.0, 0.0, 1, g1), (0.05, 0.0, 1, h1), (10.0, 0.0, 2, g1), (10.5, 0.0, 2, 0),
                        (13.0, 0.0, 2, g2), (20.0, 0.0, 3, h2), (20.01, 0.0, 3, h2)])
    parts = pd.DataFrame({"particle_id": [h1, h2, g1, g2], "is_target": [True, True, True, True]})
    ctx = {"events": {1: {"n_gun_particles": 2}},
           "stored": {1: {"hits": hits, "particles": pa.Table.from_pandas(parts, preserve_index=False)}}}
    r = TC.near_coincidences(ctx)
    g = r["gun"]
    assert g["hits"] == 3 and g["module_shared"] == 3
    assert g["within_1mm"] == 2 and g["within_0.1mm"] == 1
    gh = r["gun_host_only"]
    assert gh["module_shared"] == 3 and gh["within_1mm"] == 2 and gh["within_0.1mm"] == 1   # noise counts as host
    n = r["native_targets"]
    assert n["hits"] == 3 and n["module_shared"] == 1 and n["within_0.1mm"] == 1 and n["within_1mm"] == 1


def test_region_names_cover_the_volumes():
    assert set(TC.region(list(TC.REGION_OF_VOLUME))) == {
        "pixel barrel", "pixel endcap", "short-strip barrel", "short-strip endcap", "long-strip barrel",
        "long-strip endcap"}
    assert C.module_key is not None
