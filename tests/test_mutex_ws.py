"""Mutex Watershed track builder."""

import numpy as np

from hnl.trackbuild import builders as TB
from hnl.trackbuild import mutex_ws as MW


def _same_partition(x, y):
    if not np.array_equal(x < 0, y < 0):
        return False
    on = x >= 0
    pairs = set(zip(x[on].tolist(), y[on].tolist()))
    return len(pairs) == len(set(x[on].tolist())) == len(set(y[on].tolist()))


def test_only_attractive_edges_equal_cc():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 40
        a = rng.integers(0, n, 80)
        b = rng.integers(0, n, 80)
        ok = a != b
        a, b = a[ok], b[ok]
        s = rng.random(len(a))
        cut = 0.5
        keep = s > cut
        lab = MW.mutex_watershed(n, a[keep], b[keep], MW._logit(s[keep]) - MW._logit(cut))
        assert _same_partition(lab, TB.cc(n, a, b, s, cut))


def test_repulsion_blocks_a_weak_bridge():
    # chains 0-1-2-3 and 4-5-6-7, a weak attractive bridge 1-5 (0.6) and strong repulsions 0-4, 2-6 (0.001)
    a = np.array([0, 1, 2, 4, 5, 6, 1, 0, 2])
    b = np.array([1, 2, 3, 5, 6, 7, 5, 4, 6])
    s = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.6, 0.001, 0.001])
    w = MW._logit(s)
    lab = MW.mutex_watershed(8, a, b, w)
    assert len(set(lab[:4])) == 1 and len(set(lab[4:])) == 1 and lab[0] != lab[4]
    assert TB.cc(8, a, b, s, 0.5).tolist().count(TB.cc(8, a, b, s, 0.5)[0]) == 8      # CC merges them


def test_mutex_survives_unions():
    # 0-1 repulsive strongest; then 1-2 and 0-3 attractive; then 2-3 attractive must be blocked (2 ~ 1, 3 ~ 0)
    a = np.array([0, 1, 0, 2])
    b = np.array([1, 2, 3, 3])
    w = np.array([-9.0, 8.0, 7.0, 6.0])
    lab = MW.mutex_watershed(4, a, b, w)
    assert lab[1] == lab[2] and lab[0] == lab[3] and lab[0] != lab[1]


def test_pair_candidates_features():
    # a straight chain 0-1-2 and a branch 1-3 at a large angle; direct edges 0-1, 1-2, 1-3
    r = np.array([30.0, 70.0, 110.0, 110.0])
    phi = np.array([0.1, 0.1, 0.1, 0.6])
    z = np.zeros(4)
    a, b = np.array([0, 1, 1]), np.array([1, 2, 3])
    s = np.array([0.9, 0.9, 0.5])
    ik, X = MW.pair_candidates(4, a, b, s, r, phi, z)
    pairs = [tuple(x) for x in ik.tolist()]
    assert set(pairs) == {(0, 1), (1, 2), (1, 3), (0, 2), (0, 3), (2, 3)} and len(pairs) == 6
    f = {p: dict(zip(MW.PAIR_FEATURES, x)) for p, x in zip(pairs, X)}
    assert f[(0, 2)]["angle3d"] < 1e-6 and f[(0, 3)]["angle3d"] > 0.3 and f[(2, 3)]["angle3d"] > 2.0
    assert np.isnan(f[(0, 2)]["direct_logit"]) and np.isnan(f[(0, 1)]["path_min"]) and f[(0, 1)]["n_paths"] == 0
    assert abs(f[(1, 3)]["direct_logit"]) < 1e-9 and f[(0, 2)]["n_paths"] == 1
    assert abs(f[(0, 2)]["d_t"] - 80.0) < 1e-9 and f[(1, 2)]["deg_i"] == 2       # hit 1: two edges > 0.5


def test_pair_features_invariant_under_translation_and_rotation():
    rng = np.random.default_rng(1)
    n = 30
    x, y, z = rng.normal(0, 200, n), rng.normal(0, 200, n), rng.normal(0, 300, n)
    a = rng.integers(0, n, 60)
    b = rng.integers(0, n, 60)
    ok = a < b
    a, b = a[ok], b[ok]
    code = np.unique(a * n + b)
    a, b = code // n, code % n
    s = rng.random(len(a))

    def feats(x, y, z):
        ik, X = MW.pair_candidates(n, a, b, s, np.hypot(x, y), np.arctan2(y, x), z)
        cols = [MW.PAIR_FEATURES.index(c) for c in MW.PAIR_FEATURES if not c.startswith("sub")]
        return ik, X[:, cols]
    ik0, X0 = feats(x, y, z)
    c, s_ = np.cos(0.7), np.sin(0.7)
    ik1, X1 = feats(c * x - s_ * y + 15.0, s_ * x + c * y - 40.0, z + 25.0)
    assert np.array_equal(ik0, ik1)
    assert np.allclose(np.nan_to_num(X0, nan=-99), np.nan_to_num(X1, nan=-99), atol=1e-6)


def test_loss_classes_toy():
    from hnl.eval.loss_classes import particle_classes
    # particle 1: 6 hits on one track (found); 2: 3 + 3 on two tracks (fragmented); 3: 4 of 6 hits on a track that
    # also holds 5 hits of particle 4 (impure for 3? no: 4/9 < 50 % purity -> impure needs > 50 % share: 4/6 yes);
    # 5: 3 of 8 hits on a track, rest unassigned, one short of nothing (short); 6: not in the truth ceiling (graph)
    pid = np.array([1] * 6 + [2] * 6 + [3] * 6 + [4] * 5 + [5] * 8 + [6] * 4)
    lab = np.array([0] * 6 + [1] * 3 + [2] * 3 + [3] * 4 + [-1] * 2 + [3] * 5 + [4] * 3 + [-1] * 5 + [-1] * 4)
    truth = np.repeat(np.arange(6), [6, 6, 6, 5, 8, 4])
    truth[-4:] = -1
    has_edge = np.ones(len(pid), dtype=bool)
    has_edge[-2:] = False
    d = particle_classes(pid, lab, truth, has_edge).set_index("particle_id")
    assert d.loc[1, "cls"] == "found" and d.loc[2, "cls"] == "fragmented" and d.loc[3, "cls"] == "impure"
    assert d.loc[5, "cls"] == "short" and d.loc[6, "cls"] == "graph" and d.loc[6, "n_hits_no_edge"] == 2
    assert d.loc[4, "cls"] == "found"          # 5 of the 9 hits of track 3: > 50 % purity and all its hits
    assert bool(d.loc[2, "one_short"])          # 3 of 6: one more hit would make it DM


def test_two_pass_rebuilds_leftovers_without_touching_kept_tracks():
    # track A: 5 hits with strong edges; track B: 4 hits with weak edges (0.05); a weak bridge A-B (0.06)
    a = np.array([0, 1, 2, 3, 5, 6, 7, 4])
    b = np.array([1, 2, 3, 4, 6, 7, 8, 5])
    s = np.array([0.9, 0.9, 0.9, 0.9, 0.05, 0.05, 0.05, 0.06])
    L = np.arange(9)                                     # all on different layers
    one = MW.mws_layer(9, a, b, s, L, 0.2)
    assert len(set(one[:5])) == 1 and (one[5:] < 0).all()
    two = MW.mws_two_pass(9, a, b, s, L, 0.2, 0.02)
    assert len(set(two[:5])) == 1 and len(set(two[5:])) == 1 and two[0] != two[5]     # B rebuilt, bridge unused
