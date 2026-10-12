"""Graph-ceiling machinery on toy events."""

import numpy as np

from hnl.graphs import ceiling as C


def test_unordered_unique():
    e = np.array([[0, 1, 2, 2, 3], [1, 0, 2, 4, 1]])
    u = C.unordered_unique(e)
    assert u.tolist() == [[0, 1, 2], [1, 3, 4]]


def test_full_truth_graph_is_fully_efficient_and_extra_edges_do_not_hurt():
    # two particles of 6 hits each, chains 0-1-2-3-4-5 and 6-...-11
    pid = np.array([1] * 6 + [2] * 6)
    te = np.array([[0, 1, 2, 3, 4, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]])
    fake = np.array([[0, 5], [6, 11]])                    # cross-particle edges: removed by the perfect classifier
    stats, part, _ = C.evaluate_graph(12, pid, te, np.concatenate([te, fake], 1))
    assert stats["n_true_in_graph"] == 10 and stats["n_graph"] == 12
    assert part.dm_matched.all()


def test_missing_edge_splits_track():
    pid = np.array([1] * 6)
    te = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    g = te[:, [0, 1, 3, 4]]                               # drop 2-3: components {0,1,2} and {3,4,5}: 3/6 -> no DM
    _, part, _ = C.evaluate_graph(6, pid, te, g)
    assert not part.dm_matched.any()
    g2 = te[:, [0, 1, 2, 4]]                              # drop 4-5: {0..4} keeps 5/6 -> DM
    _, part2, _ = C.evaluate_graph(6, pid, te, g2)
    assert part2.dm_matched.all()


def test_layer_pair_table():
    pid = np.array([1] * 4)
    te = np.array([[0, 1, 2], [1, 2, 3]])
    lay = np.array([10, 20, 30, 40])
    _, _, lp = C.evaluate_graph(4, pid, te, te[:, :2], layer_id=lay)
    assert lp.loc[(30, 40), "n_kept"] == 0 and lp.loc[(10, 20), "n_kept"] == 1


def test_isolated_hits_are_not_tracks():
    lab = C.cc_labels(5, np.array([[0], [1]]))
    assert lab[0] == lab[1] and lab[0] >= 0 and (lab[2:] == -1).all()


def test_same_pid_ceiling_bridges_an_outlier():
    # chain 0-1-2-3-4-5 where hit 2 is an outlier: the graph has the skip edge 1-3 but no edges to hit 2
    pid = np.array([1] * 6)
    te = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    g = np.array([[0, 1, 3, 4], [1, 3, 4, 5]])
    _, part, _ = C.evaluate_graph(6, pid, te, g)
    assert not part.loc[1, "dm_matched"]               # chain truth: {0,1} and {3,4,5} -> 3/6, no majority
    assert part.loc[1, "dm_matched_samepid"]           # same-particle truth keeps 1-3: 5/6
