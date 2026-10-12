"""Working-point logic of hnl.eval.baseline_val (preregistration §4, §6; DECISIONS.md 2026-10-03)."""

import numpy as np
import pandas as pd

from hnl.eval import baseline_val as BV

CUTS = np.round(np.arange(0.01, 0.991, 0.01), 4)


def _curve(peak, fr_scale, prim_shift):
    # all-target efficiency peaks at `peak`; fake rate falls with the cut; primary efficiency = all - shift
    eff = 0.95 - 0.5 * (CUTS - peak) ** 2
    fr = fr_scale * (1.0 - CUTS)
    return pd.DataFrame({"eff_all_targets": eff, "fake_rate_hnl": fr, "eff_primary": eff - prim_shift,
                         "edge_fpr_hnl": 0.1 * (1.0 - CUTS) ** 2}, index=CUTS)


def test_s_star_reference_and_cuts():
    cv = {("i", 0): _curve(0.50, 0.020, 0.20), ("i", 1): _curve(0.50, 0.022, 0.20),
          ("ii", 0): _curve(0.30, 0.010, 0.05), ("ii", 1): _curve(0.30, 0.012, 0.05)}
    wp = BV.working_points(cv)
    assert wp["s_star"] == {"i": 0.5, "ii": 0.3}
    # FR at s*: arm ii 0.011 * 0.7 = 0.0077; arm ii has the higher primary efficiency everywhere -> reference ii
    assert wp["reference_arm"] == "ii" and wp["reference_rule"] == "consistent fixed point"
    assert abs(wp["FR_star"] - 0.011 * 0.7) < 1e-12
    # per-seed cut: largest fake rate <= FR*: seed 0 of ii: 0.010 (1 - c) <= 0.0077 -> c >= 0.23
    assert wp["cuts_at_FR_star"]["ii"][0] == 0.23 and wp["cuts_at_FR_star"]["ii"][1] == 0.36
    assert wp["cuts_at_FR_star"]["i"][0] == 0.62
    for arm in ("i", "ii"):
        for s in (0, 1):
            c = cv[(arm, s)]
            assert c.loc[wp["cuts_at_FR_star"][arm][s], "fake_rate_hnl"] <= wp["FR_star"] + 1e-12
    e = wp["e_star"]
    assert abs(e - 0.1 * 0.7 ** 2) < 1e-12 and wp["cuts_at_e_star"]["ii"][0] == 0.3


def test_cut_rule_ties_take_lowest():
    c = pd.DataFrame({"fake_rate_hnl": [0.03, 0.01, 0.01, 0.005]}, index=[0.1, 0.2, 0.3, 0.4])
    assert BV.cut_at_fake_rate(c, 0.02) == (0.2, True)
    assert BV.cut_at_fake_rate(c, 0.001) == (0.4, False)      # unreachable: the minimum-rate cut, flagged


def test_unreachable_target_on_u_shaped_curve():
    # real fake-rate curves are U-shaped in the cut: never fall back to the highest (nearly worst) cut
    c = pd.DataFrame({"fake_rate_hnl": [0.03, 0.01, 0.008, 0.02]}, index=[0.1, 0.2, 0.3, 0.4])
    assert BV.cut_at_fake_rate(c, 0.001) == (0.3, False)
    assert BV.cut_at_fake_rate(c, 0.015) == (0.2, True)
    # working_points flags a seed that cannot reach FR*
    cv = {("i", 0): _curve(0.50, 0.020, 0.20), ("i", 1): _curve(0.50, 0.022, 0.20),
          ("ii", 0): _curve(0.30, 0.010, 0.05), ("ii", 1): _curve(0.30, 0.012, 0.05)}
    cv[("i", 1)] = cv[("i", 1)].assign(fake_rate_hnl=np.maximum(cv[("i", 1)].fake_rate_hnl, 0.009))
    wp = BV.working_points(cv)
    assert ["i", 1] in wp["unreachable_at_FR_star"] and wp["cuts_at_FR_star"]["i"][1] < 0.99


def test_curves_on_synthetic_tables():
    # 2 HNL events (code 2) with 2 daughters + 1 W muon each, 1 ttbar event (code 1) with 2 prompt targets
    ev = [200000001, 200000002, 100000005]
    info = pd.DataFrame({
        "event_id": [ev[0]] * 3 + [ev[1]] * 3 + [ev[2]] * 2,
        "particle_id": [1, 2, 3, 1, 2, 3, 7, 8],
        "is_target": True, "category": ["hnl", "hnl", "w"] * 2 + ["other", "other"],
        "d0": [20.0, 50.0, 0.01, 2.0, 0.5, 0.02, 0.1, 0.2], "rv": [150.0, 150.0, np.nan, 5.0, 5.0, np.nan, np.nan, np.nan],
        "n_pixel_hits": [0, 0, 4, 4, 4, 4, 4, 4]})
    rows = []
    for cut in (0.5, 0.9):
        for e, pid, dm in zip(info.event_id, info.particle_id, [True, cut < 0.6, True, True, True, True, True, False]):
            rows.append({"event_id": e, "particle_id": pid, "cut": cut, "dm_matched": dm, "dm_nofp": True, "n_sm_tracks": 1})
    pp = pd.DataFrame(rows)
    pp["cut"] = pp.cut.astype(np.float32)          # as written by evaluate_gnn (per-event cuts stay float64)
    pe = pd.DataFrame({"event_id": ev * 2, "cut": [0.5] * 3 + [0.9] * 3, "n_fake": [1, 0, 2, 0, 0, 1],
                       "n_tracks_ge5": [10, 10, 20, 10, 10, 20], "tp": 9, "fp": 1, "n_true": 10, "n_edges": 20,
                       "n_true_in_graph": 10})
    c = BV.curves(pp, pe, info)
    assert c.loc[0.5, "eff_primary"] == 1.0 and c.loc[0.9, "eff_primary"] == 0.5     # event 1: both primary
    assert c.loc[0.5, "eff_w_muons"] == 1.0 and c.loc[0.5, "n_all_targets"] == 6
    assert abs(c.loc[0.5, "fake_rate_hnl"] - 0.05) < 1e-12 and abs(c.loc[0.5, "fake_rate_ttbar"] - 0.1) < 1e-12
    assert c.loc[0.9, "eff_ttbar_prompt"] == 0.5 and abs(c.loc[0.5, "edge_fpr_hnl"] - 0.1) < 1e-12
