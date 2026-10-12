"""hnl.trackml.trackbuild: the vectorised candidate searches reproduce hnl.trackbuild.merge_fragments and
hnl.trackbuild.learned_merge.candidates exactly; hnl.trackml.evaluate: metric definitions on hand-made events.

    pytest src/hnl/trackml/tests
"""

import numpy as np
import pandas as pd
import pytest

from hnl.trackbuild import learned_merge as LM
from hnl.trackbuild.merge_fragments import merge_fragments
from hnl.trackml import trackbuild as TBM
from hnl.trackml.evaluate import event_counts, pooled


def _fragments(seed=0, n_tracks=60, noise=150):
    """Helix-like tracks (2 T) cut into fragments at random gaps, plus noise hits on single-hit labels."""
    rng = np.random.default_rng(seed)
    radii = np.array([32.0, 72.0, 116.0, 172.0, 260.0, 360.0, 500.0, 660.0, 820.0, 1020.0])
    r, phi, z, lab = [], [], [], []
    nxt = 0
    for t in range(n_tracks):
        pt, q = rng.uniform(0.5, 5), rng.choice([-1, 1])
        p0, cot = rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)
        R = 1000 / 0.6 * pt
        rr = radii[radii < 2 * R]
        a = np.arcsin(rr / (2 * R))
        cut = rng.integers(2, max(3, len(rr) - 1)) if len(rr) >= 5 and rng.random() < 0.7 else len(rr)
        for k in range(len(rr)):
            r.append(rr[k] + rng.normal(0, 0.02))
            phi.append(p0 - q * a[k] + rng.normal(0, 1e-4))
            z.append(cot * 2 * R * a[k] + rng.normal(0, 0.1))
            lab.append(nxt if k < cut else nxt + 1)
        nxt += 2
    for _ in range(noise):
        r.append(rng.uniform(30, 1000))
        phi.append(rng.uniform(-np.pi, np.pi))
        z.append(rng.uniform(-1000, 1000))
        lab.append(-1)
    return (np.asarray(r), np.asarray(phi), np.asarray(z), np.asarray(lab, dtype=np.int64))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode", ["any", "sequential"])
def test_merge_fragments_fast_is_exact(seed, mode):
    r, phi, z, lab = _fragments(seed)
    for kw in ({}, {"max_rms": 0.5}, {"max_dphi_dir": 0.2, "max_dcot": 0.2, "max_gap_r": 800.0, "max_dphi": 0.6}):
        ref = merge_fragments(r, phi, z, lab, mode=mode, **kw)
        got = TBM.merge_fragments_fast(r, phi, z, lab, mode=mode, **kw)
        assert np.array_equal(ref, got)
        assert (ref != lab).any() or mode == "sequential"


@pytest.mark.parametrize("seed", [0, 3])
def test_lm_candidates_is_exact(seed):
    r, phi, z, lab = _fragments(seed, n_tracks=80)
    ref = LM.candidates(lab, r, phi, z)
    for block in (7, 64, 4096):
        got = TBM.lm_candidates(lab, r, phi, z, block=block)
        assert np.array_equal(ref, got)


def _parts(ids, pt, eta, nh, target=None):
    t = (np.asarray(pt) > 1) & (np.abs(eta) < 4) & (np.asarray(nh) >= 5) if target is None else target
    return pd.DataFrame({"particle_id": ids, "pt": pt, "eta": eta, "n_hits": nh, "is_target": t,
                         "is_prompt_target": t, "is_displaced_target": np.zeros(len(ids), bool),
                         "is_far_displaced_target": np.zeros(len(ids), bool), "is_hard_scatter": np.zeros(len(ids), bool),
                         "is_target_pix3": t, "category": ["primary"] * len(ids)})


def test_event_counts_definitions():
    # particles 1 (6 hits, target), 2 (6 hits, target), 3 (5 hits, pT 0.7), 4 (no hits, pT 0.6), noise 0
    pid = np.array([1] * 6 + [2] * 6 + [3] * 5 + [0] * 4)
    parts = _parts([1, 2, 3, 4], [2.0, 1.5, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4], [6, 6, 5, 0])
    lab = np.full(len(pid), -1)
    lab[0:6] = 0                 # track 0 = particle 1 exactly: DM
    lab[6:9] = 1                 # track 1: half of particle 2 (3 of 6): SM, not DM
    lab[9:12] = 2                # track 2: other half: SM, not DM
    lab[12:17] = 3               # track 3 = particle 3: DM
    lab[17:21] = 4               # track 4: four noise hits: matches nothing
    c, rows = event_counts(pid, lab, parts)
    assert c["n_target"] == 2 and c["m_target"] == 1
    assert c["n_tracks"] == 5 and c["n_tracks_ge5"] == 2 and c["n_fake_ge5"] == 0
    assert c["n_P1_selected_pt05"] == 4 and c["m_P1_selected_pt05"] == 2          # particle 4 (no hits) counts
    assert c["n_P1_reconstructable_pt05"] == 3 and c["m_P1_reconstructable_pt05"] == 2
    assert c["tracks_dm_P1_selected_pt05"] == 2 and c["n_tracks_dm_any"] == 2
    assert c["n_cand_ge3"] == 5 and c["n_cand_ge3_unmatched"] == 1                  # the noise track
    assert c["dup_sum_ge3_all"] == 1                                                 # particle 2 matched twice
    assert c["m_target_sm3"] == 2
    p = pooled(pd.DataFrame([c]))
    assert p["ours_eff_target"] == 0.5 and p["P1_eps_phys_pt05"] == 0.5 and np.isclose(p["P2_fr_liu"], 1 - 2 / 5)
    assert np.isclose(p["P3_fake"], 1 / 5) and np.isclose(p["P3_duplication"], 1 / 5)
    assert rows.set_index("particle_id").loc[2, "sm_any"] and not rows.set_index("particle_id").loc[2, "dm"]
    # J': the noise track disappears, |T| without noise
    assert c["n_tracks_jprime"] == 4


def test_trackml_score_perfect_and_empty():
    from hnl.trackml.evaluate import trackml_score
    pid = np.array([1] * 5 + [2] * 4 + [0] * 3)
    w = np.r_[np.full(9, 1 / 9.0), np.zeros(3)]
    hid = np.arange(1, 13)
    perfect = np.where(pid > 0, pid, -1)
    assert np.isclose(trackml_score(hid, pid, w, perfect), 1.0)
    assert np.isclose(trackml_score(hid, pid, w, np.full(12, -1)), 0.0)


def test_builder_rule_one_standard_error_tie_and_fallback():
    s = pd.DataFrame({"config": [0, 1, 2, 3], "family": ["a"] * 4, "params": ["1", "2", "3", "4"],
                      "ours_eff_target": [0.9960, 0.9955, 0.9900, 0.9990], "ours_fake_rate": [0.028, 0.011, 0.005, 0.05],
                      "n_targets": [1331] * 4})
    row, reached = TBM.builder_rule(s, fr=0.029368)
    # config 3 is above FR*; 0 is the best within FR*; 1 is within one SE of it (~0.17 points) with a lower fake rate
    assert reached and int(row.config) == 1
    row, reached = TBM.builder_rule(s, fr=0.001)
    assert not reached and int(row.config) == 2          # fallback: the lowest fake rate


def test_select_working_points():
    full = pd.DataFrame({"config": [0, 1], "family": ["walkmerge", "mwslayer"], "params": ["0.2,0.7,1", "0.2"],
                         "ours_eff_target": [0.99, 0.98], "ours_fake_rate": [0.01, 0.02], "n_targets": [10000, 10000],
                         "P0_trackml_score": [0.90, 0.92], "P2_eff": [0.97, 0.98], "P2_fr_liu": [0.50, 0.60],
                         "P3_efficiency": [0.95, 0.96], "P3_fake": [0.001, 0.003]})
    sel = TBM.select(full, 0.029368)
    assert sel["WP-A"]["family"] == "walkmerge" and sel["WP-S"]["family"] == "mwslayer"
    assert sel["WP-R"]["P2"]["family"] == "walkmerge"     # mwslayer's FR_Liu 0.60 is above the published 0.5558
    assert sel["WP-R"]["P3"]["family"] == "walkmerge" and sel["WP-R"]["P1"] is None


def test_select_fr_cc_sensitivity():
    full = pd.DataFrame({"config": [0, 1, 2, 3], "family": ["cc", "cc", "walkmerge", "walkmerge"],
                         "params": ["0.5", "0.7", "0.1,1", "0.2,1"],
                         "ours_eff_target": [0.995, 0.990, 0.997, 0.993], "ours_fake_rate": [0.04, 0.02, 0.035, 0.01],
                         "n_targets": [100000] * 4, "P2_eff": [0.99] * 4, "P2_fr_liu": [0.1] * 4,
                         "P3_efficiency": [0.99] * 4, "P3_fake": [0.001] * 4})
    sel = TBM.select(full, 0.029368)
    # FR_T,CC = CC's fake rate at its highest-efficiency cut (config 0: 0.04); under it walkmerge config 2 passes
    assert sel["WP-A"]["config"] == 3 and sel["WP-A-FRcc"]["fr_target_cc"] == 0.04
    assert sel["WP-A-FRcc"]["config"] == 2 and not sel["WP-A-FRcc"]["same_as_WP-A"]


def test_frozen_configs_per_condition(tmp_path):
    from hnl.trackml import freeze
    sel = {"selection": {
        "WP-A": {"config": 90, "family": "mwslmerge", "params": "0.3"},
        "WP-A-FRcc": {"config": 90, "family": "mwslmerge", "params": "0.3", "same_as_WP-A": True},
        "WP-S": {"config": 88, "family": "mwslmerge", "params": "0.1"},
        "WP-R": {"P1": {"reached": True, "config": 91, "family": "mwslmerge", "params": "0.5"},
                 "P2": {"reached": True, "config": 91, "family": "mwslmerge", "params": "0.5"},
                 "P3": {"reached": False}, "P4": {"reached": True, "config": 19, "family": "walkmerge",
                                                  "params": "0.1,0.5,1,0"}}}}
    p = tmp_path / "sel.json"
    p.write_text(__import__("json").dumps(sel))
    names = {c: [d["name"] for d in freeze.frozen_configs(str(p), c)] for c in ("full", "nonoise", "1gev")}
    assert names == {"full": ["WP-A", "WP-S", "WP-R-P2"], "nonoise": ["WP-A", "WP-S", "WP-R-P1"],
                     "1gev": ["WP-A", "WP-R-P4"]}
