"""Single-linkage dendrogram and the exact dynamic-programming cut (learned dendrogram cut, LDC)."""

import itertools

import numpy as np

from hnl.trackbuild import builders as TB
from hnl.trackbuild import dendrogram as D


def _same_partition(x, y):
    """Equal track partitions up to relabelling (-1 = no track)."""
    if not np.array_equal(x < 0, y < 0):
        return False
    on = x >= 0
    pairs = set(zip(x[on].tolist(), y[on].tolist()))
    return len(pairs) == len(set(x[on].tolist())) == len(set(y[on].tolist()))


def _random_graph(rng, n=40, e=90):
    a = rng.integers(0, n, e)
    b = rng.integers(0, n, e)
    keep = a != b
    a, b = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    code = np.unique(a * n + b)
    a, b = code // n, code % n
    return a, b, np.round(rng.random(len(a)), 3)


def test_tree_structure_and_slices():
    rng = np.random.default_rng(0)
    a, b, s = _random_graph(rng)
    t = D.single_linkage(40, a, b, s)
    assert np.all(np.diff(t.level) <= 0)                       # creation order = descending birth level
    for k in range(t.m):
        v = t.n + k
        hits = set(t.hits(v).tolist())
        assert hits == set(t.hits(t.left[k]).tolist()) | set(t.hits(t.right[k]).tolist())
        assert len(hits) == t.size[v]
    assert sorted(t.order.tolist()) == list(range(40))


def test_dp_with_cc_weights_equals_connected_components():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b, s = _random_graph(rng)
        t = D.single_linkage(40, a, b, s)
        for cut in (0.0, 0.2, 0.5, 0.8, 0.95):
            assert _same_partition(D.dp_cut(t, D.cc_weights(t, cut)), TB.cc(40, a, b, s, cut)), cut


def test_dp_is_optimal_on_small_trees():
    rng = np.random.default_rng(2)
    for _ in range(30):
        a, b, s = _random_graph(rng, n=9, e=20)
        t = D.single_linkage(9, a, b, s)
        w = rng.normal(size=t.m)
        lab = D.dp_cut(t, w)
        got = sum(w[k] for k in range(t.m) if len(set(lab[t.hits(t.n + k)])) == 1 and lab[t.hits(t.n + k)][0] >= 0
                  and (t.parent[t.n + k] < 0 or len(set(lab[t.hits(t.parent[t.n + k])])) > 1
                       or lab[t.hits(t.parent[t.n + k])][0] < 0))
        # brute force over all antichains of internal nodes
        best = 0.0
        nodes = list(range(t.m))
        for r in range(1, min(t.m, 5) + 1):
            for comb in itertools.combinations(nodes, r):
                hs = [set(t.hits(t.n + k).tolist()) for k in comb]
                if all(not (x & y) for x, y in itertools.combinations(hs, 2)):
                    best = max(best, sum(w[k] for k in comb))
        if t.m <= 5:
            assert abs(got - best) < 1e-12
        else:
            assert got >= best - 1e-12              # the DP is at least as good as every antichain of <= 5 nodes


def _toy_tracks():
    """Two clean helix-like tracks (8 hits each) plus a fake edge joining them at low score."""
    layers = np.array([32.0, 72, 116, 172, 260, 360, 500, 660])
    r = np.r_[layers, layers]
    phi = np.r_[0.10 + 1e-4 * layers, 1.20 - 2e-4 * layers]
    z = np.r_[0.3 * layers, -0.2 * layers]
    pid = np.repeat([11, 22], 8)
    a = np.r_[np.arange(7), 8 + np.arange(7), 3]
    b = np.r_[np.arange(1, 8), 9 + np.arange(7), 12]
    s = np.r_[np.full(14, 0.9), 0.3]
    return r, phi, z, pid, a, b, s


def test_ldc_features_and_labels():
    from hnl.trackbuild import ldc as L
    r, phi, z, pid, a, b, s = _toy_tracks()
    t = D.single_linkage(16, a, b, s)
    ks = L.candidates(t)
    X = L.node_features(t, ks, r, phi, z)
    assert X.shape == (len(ks), len(L.FEATURES)) and np.isfinite(X[:, L.FEATURES.index("size")]).all()
    y, w = L.node_labels(t, ks, pid)
    sizes = t.size[t.n + ks]
    # the two full tracks are DM-matched nodes; the root (both tracks, joined by the 0.3 edge) is not
    assert y[sizes == 8].all() and not y[sizes == 16].any()
    root = ks[sizes == 16][0]
    fit = dict(zip(L.FEATURES, X[list(ks).index(root)]))
    single = dict(zip(L.FEATURES, X[np.flatnonzero(sizes == 8)[0]]))
    assert fit["merge_chi2"] > single["merge_chi2"]                 # the union of two helices fits badly
    # perfect node scores reproduce the two tracks with the DP
    q = np.zeros(t.m)
    q[ks] = y.astype(float)
    mask = np.isin(np.arange(t.m), ks)
    lab = D.dp_cut(t, L.weights(t, q, mask, 0.5))
    assert len(set(lab[:8])) == 1 and len(set(lab[8:])) == 1 and lab[0] != lab[8] and (lab >= 0).all()


def test_ldc_dp_splits_a_dm_positive_union():
    # particle 11 (10 hits) and particle 22 (6 hits) joined at a low score: the union is DM-positive for particle 11
    # (10 / 16 > 50 %), as are both children; a perfect q must still give two tracks (the size-weighted objective tied)
    from hnl.trackbuild import ldc as L
    n = 16
    a = np.r_[np.arange(9), 10 + np.arange(5), 4]
    b = np.r_[np.arange(1, 10), 11 + np.arange(5), 12]
    s = np.r_[np.full(14, 0.9), 0.2]
    t = D.single_linkage(n, a, b, s)
    sizes = t.size[t.n:]
    q = np.where(sizes == 16, 1.0, np.where((sizes == 10) | (sizes == 6), 1.0, 0.0))
    lab = D.dp_cut(t, L.weights(t, q, np.ones(t.m, dtype=bool), 0.5))
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[10] and (lab >= 0).all()


def test_truncation_keeps_graph_consistent():
    torch = __import__("pytest").importorskip("torch")
    from torch_geometric.data import Data
    from hnl.eval.truncation_probe import truncate
    pid = torch.tensor([1, 1, 1, 1, 2, 2, 2])
    g = Data(r=torch.arange(7.0), z=torch.zeros(7), phi=torch.zeros(7), hit_particle_id=pid, hit_id=torch.arange(7),
             edge_index=torch.tensor([[0, 1, 2, 4, 5, 2], [1, 2, 3, 5, 6, 5]]), y=torch.tensor([1, 1, 1, 1, 1, 0]).bool(),
             track_edges=torch.tensor([[0, 1, 2, 4, 5], [1, 2, 3, 5, 6]]), truth_map=torch.tensor([0, 1, 2, 3, 4]),
             pt=torch.ones(5), num_nodes=7)
    h = truncate(g, np.array([True, False, False, False, False, False, False]))     # drop hit 0
    assert h.num_nodes == 6 and h.hit_id.tolist() == list(range(6))
    assert h.edge_index.tolist() == [[0, 1, 3, 4, 1], [1, 2, 4, 5, 4]]
    assert h.track_edges.tolist() == [[0, 1, 3, 4], [1, 2, 4, 5]] and h.truth_map.tolist() == [0, 1, 2, 3]
    assert h.y.tolist() == [True, True, True, True, False] and h.pt.shape == (4,)
