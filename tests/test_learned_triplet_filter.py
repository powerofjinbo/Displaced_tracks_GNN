"""Learned triplet support: the selection reproduces the hand-cut filter for the hand-cut ranking; invariance."""

import numpy as np

from hnl.graphs import helix_union as HU
from hnl.graphs import learned_triplet_filter as LT
from test_helix_union import _cyl, _toy


def _hand_like_p(lg, e_in, e_out):
    """A 'probability' that ranks the triplets as the hand-cut score and is <= 0 for the triplets it rejects."""
    good, score, _, _ = HU.triplet_features(lg, e_in, e_out)
    return np.where(good, 1.0 / (1.0 + score), -1.0)


def test_select_with_the_hand_cut_ranking_reproduces_the_support_filter():
    for seed in range(4):
        xyz, vol, lid, cand, chain = _toy(seed)
        r, phi, z = _cyl(xyz)
        s, d, lg, sl, ei, eo, X = LT.legs_and_triplets(r, phi, z, vol, lid, cand)
        assert X.shape == (len(ei), len(LT.FEATURES)) and np.isfinite(X).all()
        kept = LT.select(s, d, lg, sl, ei, eo, _hand_like_p(lg, ei, eo), 0.0, 2, len(r))
        hand, _ = HU.support_filter(r, phi, z, vol, lid, cand, k=2)
        assert set(map(tuple, kept.T.tolist())) == set(map(tuple, hand.T.tolist()))


def test_triplet_features_invariant_under_translation_and_rotation_about_z():
    xyz, vol, lid, cand, _ = _toy(1)
    a = LT.legs_and_triplets(*_cyl(xyz), vol, lid, cand)
    c, s_ = np.cos(0.7), np.sin(0.7)
    moved = xyz @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]).T + np.array([0.0, 0.0, 25.0])
    b = LT.legs_and_triplets(*_cyl(moved), vol, lid, cand)
    # same legs and triplets (the hits keep their outward order here), same features up to rounding
    assert np.array_equal(a[4], b[4]) and np.array_equal(a[5], b[5])
    assert np.allclose(a[6], b[6], rtol=1e-4, atol=1e-4)


def test_operating_point_uses_ttbar_events_and_the_hand_cut_size():
    import pandas as pd
    rows = []
    for g, pairs, true_in in (("base", 100, 90), ("hand", 130, 97), ("k2_tau0.1", 120, 98), ("k3_tau0.05", 125, 99),
                              ("k4_tau0.01", 140, 100)):
        rows.append({"event_id": 100000001, "graph": g, "n_pairs": pairs, "n_true_in_graph": true_in, "n_true": 100})
        # an HNL event that would favour another configuration must not matter
        rows.append({"event_id": 200000001, "graph": g, "n_pairs": 10, "n_true_in_graph": 100 if g == "k2_tau0.1" else 50,
                     "n_true": 100})
    summ = pd.DataFrame({"graph": ["base", "hand", "k2_tau0.1", "k3_tau0.05", "k4_tau0.01"],
                         "ceil_primary": [0.98, 0.99, 0.99, 0.991, 0.995], "ceil_d0_ge300": [0.7] * 5,
                         "ceil_w_muons": [1.0] * 5, "ceil_ttbar_targets": [0.998] * 5})
    best, t = LT.operating_point(pd.DataFrame(rows), summ)
    assert best == "k3_tau0.05"                         # k4 is above the hand-cut size, k2 has a lower ttbar efficiency
    assert t.loc["hand", "pairs_ttbar"] == 130 and abs(t.loc["k3_tau0.05", "ttbar_edge_eff"] - 0.99) < 1e-12
