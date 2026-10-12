"""Track builders on toy scored graphs."""

import numpy as np

from hnl.trackbuild import builders as TB


def _two_chains_with_bridge():
    # chain A: hits 0-1-2-3-4 (R = 1..5), chain B: hits 5-6-7-8-9 (R = 1..5); a fake bridge 2 -> 8 (R 3 -> 4)
    R = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
    a = np.array([0, 1, 2, 3, 5, 6, 7, 8, 2])
    b = np.array([1, 2, 3, 4, 6, 7, 8, 9, 8])
    s = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6])
    return R, a, b, s


def test_cc_merges_and_alternatives_split():
    R, a, b, s = _two_chains_with_bridge()
    lab = TB.cc(10, a, b, s, 0.5)
    assert len(set(lab)) == 1                                        # one merged component
    jr = TB.junction_removal(10, a, b, s, R, cut=0.5, junction_cut=0.5)
    assert len(set(jr[:5])) == 1 and len(set(jr[5:])) == 1 and jr[0] != jr[5]   # bridge is not mutual-best
    wk = TB.walkthrough(10, a, b, s, R, th_min=0.5, th_add=0.95)
    assert len(set(wk[:5])) == 1 and len(set(wk[5:])) == 1 and wk[0] != wk[5]
    cw = TB.cc_and_walk(10, a, b, s, R, cut=0.5, th_min=0.5, th_add=0.95)
    assert len(set(cw[:5])) == 1 and len(set(cw[5:])) == 1 and cw[0] != cw[5]


def test_walkthrough_longest_path_and_singletons():
    # 0 -> 1 -> 2 -> 3 and a short branch 0 -> 4 with a higher score; hit 5 isolated
    R = np.array([1, 2, 3, 4, 2, 9], dtype=float)
    a = np.array([0, 1, 2, 0])
    b = np.array([1, 2, 3, 4])
    s = np.array([0.8, 0.8, 0.8, 0.95])
    # th_add below 0.8: both branches allowed -> the longest (0-1-2-3) wins; 4 left alone (no track)
    w = TB.walkthrough(6, a, b, s, R, th_min=0.5, th_add=0.7)
    assert (w[[0, 1, 2, 3]] == w[0]).all() and w[0] >= 0 and w[4] == -1 and w[5] == -1
    # th_add above 0.8: only the best edge 0 -> 4 is followed from 0; then 1-2-3 is a separate track
    w = TB.walkthrough(6, a, b, s, R, th_min=0.5, th_add=0.99)
    assert w[0] == w[4] and w[1] == w[2] == w[3] and w[0] != w[1]


def test_relabel_and_chains():
    R = np.arange(6, dtype=float)
    a, b, s = np.array([0, 1, 3]), np.array([1, 2, 4]), np.array([0.9, 0.9, 0.9])
    jr = TB.junction_removal(6, a, b, s, R, cut=0.5, junction_cut=0.5, min_chain=3)
    # 0-1-2 is a chain of 3 -> track; 3-4 has 2 hits < min_chain -> rebuilt by the mutual-best step -> track
    assert jr[0] == jr[1] == jr[2] >= 0 and jr[3] == jr[4] >= 0 and jr[5] == -1 and jr[0] != jr[3]


def _line(r, angle, z0=0.0, cot=0.2):
    return np.stack([r * np.cos(angle), r * np.sin(angle), z0 + cot * r], 1)


def test_triplet_walk_does_not_switch_particles():
    # A: 5 hits along phi = 0 (r 30-260); B: 8 hits along phi = 0.5 (r 70-820); a fake edge A(110) -> B(170) with a
    # high score makes the plain walkthrough's longest path switch from A to B; the 0.5 rad kink is not helix-like
    ra = np.array([30.0, 70, 110, 170, 260])
    rb = np.array([70.0, 110, 170, 260, 360, 500, 660, 820])
    xyz = np.concatenate([_line(ra, 0.0), _line(rb, 0.5)])
    R = np.linalg.norm(xyz, axis=1)
    a = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 2])
    b = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 7])
    s = np.array([0.9] * 11 + [0.95])
    w = TB.walkthrough(13, a, b, s, R, th_min=0.5, th_add=0.5)
    assert w[0] == w[7] == w[12]                                     # plain walk: A's start continues on B
    t = TB.walkthrough_triplet(13, a, b, s, R, xyz, th_min=0.5, th_add=0.5, max_angle=0.3)
    assert len(set(t[:5])) == 1 and len(set(t[5:])) == 1 and t[0] != t[5] and (t >= 0).all()


def test_triplet_walk_keeps_double_hits():
    # one particle with two hits 2 mm apart on one layer (overlapping modules): the short segment is not checked
    r = np.array([30.0, 70, 110, 111.0, 170, 260])
    xyz = _line(r, 0.2)
    xyz[3, 1] += 1.5                                                 # second hit of the double, slightly off
    R = np.linalg.norm(xyz, axis=1)
    a, b = np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])
    t = TB.walkthrough_triplet(6, a, b, np.full(5, 0.9), R, xyz, th_min=0.5, th_add=0.5, max_angle=0.1)
    assert len(set(t)) == 1 and t[0] >= 0


def test_cc_and_walk_stays_on_kept_edges():
    # review toy: two branchy components at cut 0.5 joined only by a 0.2 edge; th_min 0.1 < cut must not use it
    R = np.array([1, 2, 2.5, 2.7, 3, 3.5])
    a = np.array([0, 0, 3, 3, 2])
    b = np.array([1, 2, 4, 5, 4])
    s = np.array([0.9, 0.8, 0.9, 0.8, 0.2])
    lab = TB.cc_and_walk(6, a, b, s, R, cut=0.5, th_min=0.1, th_add=0.5)
    comp = {0: {0, 1, 2}, 1: {3, 4, 5}}
    for t in set(lab[lab >= 0]):
        hits = set(np.flatnonzero(lab == t))
        assert hits <= comp[0] or hits <= comp[1]              # no track across the two CC components


def test_triplet_walk_no_switch_through_short_fake_edge():
    # review toy: B starts 2.2 mm from A's third hit; the short fake hop A2 -> B0 is not checked itself, but the next
    # long edge B0 -> B1 (0.5 rad off A's direction) must be compared with A1 -> A2 and rejected
    ra = np.array([30.0, 70, 110, 170, 260])
    A = _line(ra, 0.0)
    b0 = A[2] + np.array([0.0, 2.2, 0.0])
    d = np.array([0.0, 60, 150, 250, 390, 550])
    u = np.array([np.cos(0.5), np.sin(0.5), 0.2])
    xyz = np.concatenate([A, b0 + d[:, None] * u])
    R = np.linalg.norm(xyz, axis=1)
    a = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 2])
    b = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 5])
    s = np.array([0.9] * 9 + [0.6])
    t = TB.walkthrough_triplet(11, a, b, s, R, xyz, th_min=0.5, th_add=0.5, max_angle=0.2)
    assert len(set(t[:5])) == 1 and len(set(t[5:])) == 1 and t[0] != t[5] and (t >= 0).all()


def test_triplet_walk_double_hit_then_fake_kink():
    # A has two hits 1.5 mm apart on one layer (true short edge), then a fake long edge with a 0.49 rad kink to B: the
    # reference direction passes through the double hit, so the kink is still rejected
    r = np.array([30.0, 70, 110, 111.0, 170, 260])
    A = _line(r, 0.0)
    A[3, 1] += 1.0
    u = np.array([np.cos(0.49), np.sin(0.49), 0.2])
    B = A[3] + np.array([60.0, 160, 300, 460])[:, None] * u + np.array([0.0, 8.0, 0.0])
    xyz = np.concatenate([A, B])
    R = np.linalg.norm(xyz, axis=1)
    a = np.array([0, 1, 2, 3, 4, 6, 7, 8, 3])
    b = np.array([1, 2, 3, 4, 5, 7, 8, 9, 6])
    s = np.array([0.9] * 8 + [0.95])
    t = TB.walkthrough_triplet(10, a, b, s, R, xyz, th_min=0.5, th_add=0.5, max_angle=0.2)
    assert len(set(t[:6])) == 1 and t[0] >= 0 and t[6] != t[0]
