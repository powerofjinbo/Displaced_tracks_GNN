"""hnl.overlay.readouts, hnl.overlay.signatures and hnl.overlay.longstrip on synthetic inputs (addendum H section 7 and
section 3): the gun-hit fake count against the frozen event_counts, the pooled read-outs and their twin split, the
signature definition of gnn-displaced (quantisation, order independence, vertex grouping, twins), and the long-strip
crossing count and table copy. Truth or toy inputs only; no model output and no grant."""

import json

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import blind as B
from hnl.overlay import longstrip as LS
from hnl.overlay import readouts as R
from hnl.overlay import signatures as S
from hnl.trackml.evaluate import event_counts

GUN = (1001 << 52) | (1 << 36)                     # a gun particle id (vertex field 1001)
GUN2 = (1002 << 52) | (1 << 36)
HOST = (1 << 52) | (5 << 36)                       # a hard-scatter particle (vertex field 1)
HOST2 = (7 << 52) | (3 << 36)


def _parts(ids, n_hits):
    return pd.DataFrame({"particle_id": np.asarray(ids, np.int64), "n_hits": np.asarray(n_hits, np.int16),
                         "pt": 2.0, "eta": 0.5, "d0": 0.0, "rv": 0.0, "vz": 0.0, "is_target": True,
                         "is_prompt_target": True, "is_displaced_target": False, "is_far_displaced_target": False,
                         "is_hard_scatter": True, "is_target_pix3": True, "is_soft_displaced": False,
                         "category": "primary", "vprim": 1, "n_pixel_hits": 3, "r_first": 30.0, "q": 1})


def test_gun_hit_fakes_match_event_counts():
    # track 0: 6 hits of HOST (matched); track 1: 3 gun hits + 3 noise (fake, gun hit); track 2: 3 HOST2 + 2 noise +
    # 1 GUN2 (fake, gun hit: HOST2 holds 3 of 6, not more than half); track 3: 5 noise (fake, no gun hit)
    pid = np.array([HOST] * 6 + [GUN] * 3 + [0] * 3 + [HOST2] * 3 + [0] * 2 + [GUN2] + [0] * 5, np.int64)
    lab = np.array([0] * 6 + [1] * 6 + [2] * 6 + [3] * 5, np.int64)
    parts = _parts([HOST, HOST2, GUN, GUN2], [6, 3, 3, 1])
    c, _ = event_counts(pid, lab, parts)
    nf, nfg = R.gun_hit_fakes(pid, lab)
    assert nf == c["n_fake_ge5"] == 3
    assert nfg == 2


def test_gun_hit_fakes_unassigned_hits_and_empty():
    pid = np.array([GUN] * 5 + [HOST] * 5, np.int64)
    lab = np.array([-1] * 5 + [0] * 5, np.int64)
    assert R.gun_hit_fakes(pid, lab) == (0, 0)
    assert R.gun_hit_fakes(np.array([GUN], np.int64), np.array([-1], np.int64)) == (0, 0)


def test_n_llp_counts_vertex_fields():
    p = _parts([HOST, GUN, GUN | 7, GUN2], [5, 5, 5, 5])
    assert R.n_llp(p) == 2


def _pe(rows):
    return pd.DataFrame(rows)


def test_summarize_pools_and_splits_by_twin():
    rows = []
    for e, tw in ((50021117, True), (50021395, False), (50021734, True)):
        rows.append({"event": e, "config": 0, "name": "WP-A", "n_tracks": 10, "n_tracks_ge5": 10, "n_fake_ge5": 1,
                     "dup_sum_targets": 0, "n_target": 20, "m_target": 18, "n_prompt_target": 15,
                     "m_prompt_target": 14, "n_fake_ge5_gun_hit": 1 if tw else 0, "n_llp_rows": 10, "n_llp": 10,
                     "host_n_tracks_ge5": 9, "host_n_fake_ge5": 1, "host_n_prompt_target": 15,
                     "host_m_prompt_target": 14, "host_trackml_score": 0.9})
    pe = _pe(rows)
    twins = {21117: True, 21395: False, 21734: True}
    s = R.summarize(pe, pd.DataFrame(), [{"name": "WP-A"}], twins, "full")["WP-A"]
    assert s["fake_rate"]["value"] == pytest.approx(3 / 30)
    assert s["fake_ge5_with_gun_hit_per_llp"]["value"] == pytest.approx(2 / 30)
    assert s["host_prompt_efficiency"]["value"] == pytest.approx(42 / 45)
    assert s["by_hs_twin"]["twin"]["events"] == 2 and s["by_hs_twin"]["no_twin"]["events"] == 1
    assert s["host_trackml_score"]["value"] == pytest.approx(0.9)
    with pytest.raises(KeyError):
        R.summarize(pe, pd.DataFrame(), [{"name": "WP-A"}], {21117: True}, "full")


def test_bootstrap_index_is_evaluate_table():
    from hnl.trackml.evaluate import bootstrap  # noqa: F401  (the same generator and size)
    a = R._boot_index(7)
    b = np.random.default_rng(12345).integers(0, 7, size=(500, 7))
    assert np.array_equal(a, b)


def test_production_inputs_refused_for_explicit_paths():
    with pytest.raises(PermissionError):
        R.run("hval", "1gev", "/tmp/x", scores="/tmp/s")
    with pytest.raises(PermissionError):
        R.run("closure", "full", "/tmp/x")
    with pytest.raises(PermissionError, match="--twins"):
        R.run("htest", "full", "/tmp/x")


# ------------------------------------------------------------------------------------------------ signatures
def test_quantize_matches_gnn_displaced_definition():
    x = np.array([0.0, 1.234567, -1.234549, 9.999996, 123456.7, 1e-7, -0.000123456])
    q = S.quantize(x, 5)
    # same decimal at 5 significant digits -> same code; different -> different. Zero has the code (0, e 0) -> 500,
    # as in gd.prep.signatures (whose docstring says 0 but whose code gives 500).
    assert q[0] == 500 == S.quantize([0.0], 5)[0] == S.quantize([-0.0], 5)[0]
    assert S.quantize([1.23456], 5)[0] == S.quantize([1.234564], 5)[0]
    assert S.quantize([1.23456], 5)[0] != S.quantize([1.23446], 5)[0]
    assert S.quantize([9.999996], 5)[0] == S.quantize([10.0], 5)[0]
    assert len(set(q.tolist())) == len(q)


def test_signature_order_independent_and_grouped():
    rows = pd.DataFrame({"event": [1, 1, 1, 1, 2, 2], "particle_id": [HOST, HOST | 1, HOST2, (1 << 52) | (1 << 24),
                                                                      HOST | 1, HOST],
                         "particle_type": [211, -211, 13, 22, -211, 211], "q": [1, -1, -1, 0, -1, 1],
                         "px": [1.0, 2.0, 3.0, 4.0, 2.0, 1.0], "py": [0.5, 0.25, 0.1, 0.0, 0.25, 0.5],
                         "pz": [10.0, -3.0, 2.0, 1.0, -3.0, 10.0]})
    s = S.event_signatures(rows)
    # event 1: vertex 1 (two gen-0 particles; the generation-1 row is left out) and vertex 7; event 2: vertex 1
    assert s[["event", "vtx", "n_gen0"]].values.tolist() == [[1, 1, 2], [1, 7, 1], [2, 1, 2]]
    assert s.signature.iloc[0] == s.signature.iloc[2]
    t = S.twins(s[s.event == 2], s[s.event == 1])
    assert t["hs_twin"] == {"2": True} and t["n_collisions_twin"] == 1
    with pytest.raises(ValueError):
        S.twins(s, s)


# ------------------------------------------------------------------------------------------------ long strips
def test_crossing_misses_counts_interior_crossings():
    # particle 5: 1602 (2 hits, both dropped), 1804 (kept), 1302 (kept): 1602 and 1804 are interior (a later kept hit),
    # 1602 is missed. particle 6: 1602 (one of two hits kept), 1804 (dropped, last crossing: not interior)
    hits = pd.DataFrame({"particle_id": np.array([5, 5, 5, 5, 6, 6, 6], np.uint64),
                         "volume": [16, 16, 18, 13, 16, 16, 18], "layer": [2, 2, 4, 2, 2, 2, 4],
                         "order": [0, 1, 2, 3, 0, 1, 2]})
    keep = np.array([0, 0, 1, 1, 0, 1, 0], bool)
    em = {"keep": keep, "stopped": np.zeros(7, bool), "dropped": ~keep}
    assert LS.crossing_misses(hits, em) == (2, 1)
    # a stop after the first crossing of particle 5: nothing kept later, so no interior crossing on it
    keep2 = np.array([0, 0, 0, 0, 0, 1, 0], bool)
    em2 = {"keep": keep2, "stopped": np.array([0, 0, 1, 1, 0, 0, 0], bool), "dropped": ~keep2}
    assert LS.crossing_misses(hits, em2) == (0, 0)


def test_with_eps_changes_only_long_strip_disks():
    class T:
        pass
    t = T()
    t.eps = np.zeros(2000)
    t.eps[[702, 1302, 1602, 1612, 1802]] = [0.01, 0.02, 0.0098, 0.0098, 0.0098]
    u = LS._with_eps(t, 0.03)
    assert u.eps[1602] == u.eps[1612] == u.eps[1802] == 0.03
    assert u.eps[702] == 0.01 and u.eps[1302] == 0.02 and u.eps[1604] == 0.0
    assert t.eps[1602] == 0.0098
    assert LS._with_eps(t, None) is t
