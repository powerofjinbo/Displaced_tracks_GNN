"""HelixFilter union graph (hand-cut control): triplet support, invariance, radius search, acorn truth."""

import itertools

import numpy as np
import pytest

from hnl.graphs import helix_union as HU


def _helix(x0, y0, z0, phi, cot, pt, s, q=1):
    """Hits at arc lengths s of a helix starting at (x0, y0, z0) with transverse direction phi."""
    Rc = pt * HU.MM_PER_GEV
    th0 = phi - q * np.pi / 2
    xc, yc = x0 - Rc * np.cos(th0), y0 - Rc * np.sin(th0)
    th = th0 + q * s / Rc
    return np.stack([xc + Rc * np.cos(th), yc + Rc * np.sin(th), z0 + cot * s], 1)


def _toy(seed=0):
    """Two displaced helices (12 hits each) and 10 noise hits; candidates = chain edges + random noise pairs."""
    rng = np.random.default_rng(seed)
    s = np.linspace(40.0, 640.0, 12)
    h1 = _helix(30.0, -20.0, 15.0, 0.4, 0.6, 2.0, s)
    h2 = _helix(-60.0, 80.0, -40.0, 2.3, -1.2, 1.2, s, q=-1)
    noise = rng.uniform(-500, 500, (10, 3))
    xyz = np.concatenate([h1, h2, noise])
    n = len(xyz)
    chain = np.array([(i, i + 1) for i in range(11)] + [(12 + i, 13 + i) for i in range(11)]).T
    nz = np.array([(24 + j, int(rng.integers(0, 24))) for j in range(10)]).T
    cand = np.concatenate([chain, nz], 1)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    vol = np.where(r < 200, 8, np.where(r < 700, 13, 17))
    lid = np.arange(n)                        # every hit on its own layer
    return xyz, vol, lid, cand, chain


def _cyl(xyz):
    return np.hypot(xyz[:, 0], xyz[:, 1]), np.arctan2(xyz[:, 1], xyz[:, 0]), xyz[:, 2]


def test_support_filter_keeps_clean_displaced_helices_and_rejects_noise():
    xyz, vol, lid, cand, chain = _toy()
    kept, st = HU.support_filter(*_cyl(xyz), vol, lid, cand, k=2)
    kc = set(map(tuple, kept.T.tolist()))
    assert set(map(tuple, np.sort(chain, 0).T.tolist())) <= kc
    assert not any(a >= 24 or b >= 24 for a, b in kc)                     # no noise leg has helix support
    assert st["n_kept"] == chain.shape[1]


def test_triplet_features_invariant_under_translation_and_rotation():
    xyz, vol, lid, cand, _ = _toy(1)
    r, phi, z = _cyl(xyz)
    R = np.hypot(r, z)
    a, b = cand
    fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    src, dst = np.where(fwd, a, b), np.where(fwd, b, a)
    lg = HU.legs(xyz, src, dst, vol)
    e_in, e_out = HU.candidate_triplets(src, dst, lg, lg["L"] > 1e-3)
    assert len(e_in) > 10
    ref = HU.triplet_features(lg, e_in, e_out)
    c, s_ = np.cos(1.1), np.sin(1.1)
    moved = np.stack([c * xyz[:, 0] - s_ * xyz[:, 1] + 140.0, s_ * xyz[:, 0] + c * xyz[:, 1] - 75.0, xyz[:, 2] + 300.0], 1)
    out = HU.triplet_features(HU.legs(moved, src, dst, vol), e_in, e_out)   # same legs, same detector volumes
    assert np.array_equal(ref[0], out[0])
    for x, y in zip(ref[1:], out[1:]):
        assert np.allclose(x, y, rtol=1e-9, atol=1e-9)


def test_support_filter_invariant_under_rotation_about_z():
    xyz, vol, lid, cand, _ = _toy(2)
    k0, _ = HU.support_filter(*_cyl(xyz), vol, lid, cand)
    for ang in (0.3, 2.0, -2.9):
        c, s_ = np.cos(ang), np.sin(ang)
        rot = np.stack([c * xyz[:, 0] - s_ * xyz[:, 1], s_ * xyz[:, 0] + c * xyz[:, 1], xyz[:, 2]], 1)
        k1, _ = HU.support_filter(*_cyl(rot), vol, lid, cand)
        assert np.array_equal(k0, k1)


def test_candidate_triplets_stay_at_their_middle_hit():
    # steep and short legs (|cot| at the clip, huge slope errors): windows that would cross into the neighbouring
    # middle hit's key range with the prototype's fixed stride
    rng = np.random.default_rng(3)
    n, E = 40, 400
    src = rng.integers(0, n, E)
    dst = rng.integers(0, n, E)
    lg = {"cot": rng.choice([-300.0, -199.0, -5.0, 0.0, 3.0, 199.0, 250.0], E),
          "s_cot": rng.choice([0.01, 1.0, 50.0], E)}
    use = src != dst
    e_in, e_out = HU.candidate_triplets(src, dst, lg, use)
    got = set(zip(e_in.tolist(), e_out.tolist()))
    c = np.clip(lg["cot"], -HU.COT_CLIP, HU.COT_CLIP)
    w = np.minimum(5.0 * (np.sqrt(lg["s_cot"] ** 2 + HU.MS0 ** 2 * (1 + lg["cot"] ** 2)) + 0.05 * (1 + np.abs(lg["cot"]))),
                   2 * HU.COT_CLIP + 1)
    want = {(i, o) for i, o in itertools.product(np.flatnonzero(use), repeat=2)
            if dst[i] == src[o] and abs(c[i] - c[o]) <= w[o]}
    assert got == want


@pytest.mark.parametrize("knn", [None, 4])
def test_radius_pairs_matches_brute_force(knn):
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(60, 3)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    r = 0.6
    d = np.linalg.norm(emb[:, None].astype(np.float64) - emb[None].astype(np.float64), axis=2)
    within = (d <= r) & ~np.eye(60, dtype=bool)
    if knn:                                    # FRNN: K nearest per query point, the point itself included
        rank = np.argsort(np.argsort(np.where(within, d, np.inf), axis=1), axis=1)
        mine = within & (rank < knn - 1)
        within = mine | mine.T
    i, j = np.nonzero(np.triu(within, 1))
    assert np.array_equal(HU.radius_pairs(emb, r, knn), np.stack([i, j]))


def test_best_configs_follows_the_builder_rule():
    import pandas as pd
    cfgs = [["cc", 0.3], ["cc", 0.4], ["cc", 0.5], ["mwslayer", 0.1], ["mwslayer", 0.2], ["walkmerge", 0.1, 0.5, 1.0]]
    c = pd.DataFrame({"eff_primary": [0.95, 0.96, 0.96, 0.97, 0.98, 0.99],
                      "fake_rate_hnl": [0.02, 0.030, 0.025, 0.03, 0.04, 0.05]})
    assert HU.best_configs(c, cfgs, 0.031) == {"cc": 2, "mwslayer": 3}      # ties -> lower fake rate; none for walk


def test_union_graph_contains_base_and_recomputes_acorn_truth():
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data
    rng = np.random.default_rng(5)
    n = 30
    r = rng.uniform(30, 1000, n)
    z = rng.uniform(-1000, 1000, n)
    hit_R = torch.tensor((r ** 2 + z ** 2) / 1e6, dtype=torch.float32)
    te = np.array([[0, 1], [1, 2], [2, 3], [5, 6], [6, 7], [10, 11]]).T
    R = hit_R.numpy()
    te = np.where(R[te[0]] <= R[te[1]], te, te[::-1])          # chain truth oriented outward, as the feature store
    base_pairs = np.array([[0, 1], [2, 3], [5, 6], [4, 9], [12, 13], [6, 7]]).T
    g = Data(r=torch.tensor(r, dtype=torch.float32), z=torch.tensor(z, dtype=torch.float32), hit_R=hit_R,
             track_edges=torch.tensor(te), num_nodes=n, config=[[], {"stage": "graph_construction"}])
    g.edge_index, g.y, g.truth_map = HU.acorn_truth(base_pairs, g.track_edges, hit_R)
    # recomputing the stored truth reproduces it exactly
    e0, y0, tm0 = HU.acorn_truth(g.edge_index, g.track_edges, hit_R)
    assert torch.equal(e0, g.edge_index) and torch.equal(y0, g.y) and torch.equal(tm0, g.truth_map)
    h, union = HU.union_graph(g, np.array([[1, 2], [10, 11], [3, 0]]).T, {"stage": "helix_union"})
    ei = h.edge_index.numpy()
    assert (R[ei[0]] <= R[ei[1]]).all()
    pairs = set(map(tuple, np.sort(ei, 0).T.tolist()))
    assert set(map(tuple, np.sort(base_pairs, 0).T.tolist())) <= pairs and len(pairs) == ei.shape[1] == 9
    truth = set(map(tuple, np.sort(te, 0).T.tolist()))
    assert h.y.numpy().tolist() == [tuple(p) in truth for p in np.sort(ei, 0).T.tolist()]
    tm = h.truth_map.numpy()
    assert (tm >= 0).all()                                   # every chain edge is in the union
    assert all(set(ei[:, tm[t]].tolist()) == set(te[:, t].tolist()) for t in range(te.shape[1]))
    assert len(h.config) == 3 and len(g.config) == 2 and g.edge_index.shape[1] == 6


def test_ml_form_strips_the_edge_classifier_output_only():
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data
    ei = torch.tensor([[0, 1], [1, 2]])
    g = Data(r=torch.ones(3), edge_index=ei, y=torch.tensor([True, False]), scores=torch.tensor([0.9, 0.1]),
             num_nodes=3, config=[[], {"stage": "graph_construction"}, {"stage": "edge_classifier"}])
    h = HU.ml_form(g)
    assert "scores" not in h and torch.equal(h.edge_index, ei) and h.y.tolist() == [True, False]
    assert h.config == [[], {"stage": "graph_construction"}]
    assert HU.ml_form(h).config == [[], {"stage": "graph_construction"}]      # idempotent on ml_graphs/ graphs


def test_build_refuses_test_split_with_metrics(tmp_path):
    with pytest.raises(SystemExit, match="no-metrics"):
        HU.main(["build", "--ml-run", str(tmp_path), "--split", "test", "--out", str(tmp_path / "o")])


def test_ml_form_keeps_exactly_the_reference_attributes(tmp_path):
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data
    from acorn.utils.loading_utils import save_pyg
    ref = Data(r=torch.ones(3), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([True]), num_nodes=3,
               config=[{"stage": "graph_construction"}])
    keep = sorted(ref.keys())
    g = ref.clone()
    g.scores, g.extra = torch.tensor([0.5]), torch.zeros(3)
    g.config = list(ref.config) + [{"stage": "edge_classifier"}]
    save_pyg(g, str(tmp_path / "event1.pyg"))
    save_pyg(ref.clone(), str(tmp_path / "event2.pyg"))
    assert HU.mlform_dir(str(tmp_path), keep) == 1                   # only the scored graph is rewritten
    h = HU._load_bytes(open(tmp_path / "event1.pyg.gz", "rb").read())
    assert sorted(h.keys()) == keep and h.config == [{"stage": "graph_construction"}]
    with pytest.raises(ValueError, match="lacks"):
        HU.ml_form(Data(r=torch.ones(3), config=[]), keep)
