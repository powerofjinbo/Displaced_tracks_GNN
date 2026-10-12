"""Inner-hit truncation augmentation (hnl.models.augment)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")


def _toy_event(eid=100000007, n_part=6, hits=8, seed=0):
    """A raw (unprefixed) metric-learning graph: particles with `hits` hits on increasing radii, chain truth edges,
    plus random fake edges; keys as in the stored ml_graphs."""
    from torch_geometric.data import Data
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(1, n_part + 1), hits)
    n = len(pid)
    r = np.tile(np.linspace(30, 900, hits), n_part) + rng.normal(0, 1, n)
    z = rng.normal(0, 100, n)
    phi = rng.uniform(-3, 3, n)
    perm = rng.permutation(n)                              # stored hit order is not R order
    pid, r, z, phi = pid[perm], r[perm], z[perm], phi[perm]
    R = np.hypot(r, z)
    chain = []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        idx = idx[np.argsort(R[idx])]
        chain += list(zip(idx[:-1], idx[1:]))
    chain = np.array(chain).T
    fake = rng.integers(0, n, (2, 3 * n))
    fake = fake[:, pid[fake[0]] != pid[fake[1]]]
    ei = np.concatenate([chain, fake], axis=1)
    y = np.r_[np.ones(chain.shape[1], bool), np.zeros(fake.shape[1], bool)]
    sh = rng.permutation(ei.shape[1])
    ei, y = ei[:, sh], y[sh]
    tm = np.argsort(sh)[:chain.shape[1]]                  # truth_map: position of each chain edge in edge_index
    T = chain.shape[1]
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
    return Data(hit_id=t(np.arange(n), torch.long), track_edges=t(chain, torch.long), hit_y=t(r * np.sin(phi)),
                hit_particle_id=t(pid, torch.long), event_id=str(eid), config=[], num_nodes=n,
                batch=torch.zeros(n, dtype=torch.long), ptr=torch.tensor([0, n]), edge_index=t(ei, torch.long),
                hit_R=t(R), eta=t(np.zeros(n)), nhits=t(np.full(T, hits), torch.long), pt=t(np.ones(T)),
                eta_particle=t(np.zeros(T)), y=t(y, torch.bool), z=t(z), particle_id=t(pid[chain[0]], torch.long),
                truth_map=t(tm, torch.long), phi=t(phi), x=t(r * np.cos(phi)), r=t(r))


def test_truncation_mask_rules():
    from hnl.models import augment as A
    rng = np.random.default_rng(0)
    pid = np.repeat([0, 1, 2, 3], [9, 12, 5, 6])
    R = rng.uniform(0, 1000, len(pid))
    assert not A.truncation_mask(pid, R, np.random.default_rng(1), p=0.0).any()
    for s in range(30):
        m = A.truncation_mask(pid, R, np.random.default_rng(s), p=1.0)
        assert not m[pid == 0].any() and not m[pid == 2].any()          # noise and 5-hit particles never cut
        for p in (1, 3):
            idx = np.flatnonzero(pid == p)
            k = int(m[idx].sum())
            assert 1 <= k <= len(idx) - A.TRUNC_KEEP
            # the removed hits are the k innermost
            assert set(idx[m[idx]]) == set(idx[np.argsort(R[idx])[:k]])
    a = A.truncation_mask(pid, R, np.random.default_rng([4, 2, 7]), p=0.5)
    b = A.truncation_mask(pid, R, np.random.default_rng([4, 2, 7]), p=0.5)
    assert np.array_equal(a, b)


def test_truncate_event_consistency_and_determinism():
    from hnl.models import augment as A
    g = _toy_event()
    h = A.truncate_event(g, 3, 0, p=1.0)
    n0, n1 = g.num_nodes, h.num_nodes
    assert n1 < n0 and h.edge_index.max() < n1 and h.hit_particle_id.shape[0] == n1
    # truth map points at an edge joining the same two hits as each kept chain edge
    tm = h.truth_map
    assert (tm >= 0).all()
    e = h.edge_index[:, tm]
    te = h.track_edges
    assert bool((((e[0] == te[0]) & (e[1] == te[1])) | ((e[0] == te[1]) & (e[1] == te[0]))).all())
    assert bool(h.y[tm].all()) and int(h.y.sum()) == te.shape[1]
    # every particle keeps >= 5 hits and loses only its innermost hits
    pid0, pid1 = g.hit_particle_id.numpy(), h.hit_particle_id.numpy()
    for p in np.unique(pid0):
        assert (pid1 == p).sum() >= A.TRUNC_KEEP
        R0 = np.sort(g.hit_R.numpy()[pid0 == p])
        R1 = np.sort(h.hit_R.numpy()[pid1 == p])
        assert np.allclose(R0[-len(R1):], R1)
    # deterministic per (seed, epoch, event), different across epochs
    h2 = A.truncate_event(_toy_event(), 3, 0, p=1.0)
    assert torch.equal(h.edge_index, h2.edge_index)
    masks = {A.truncate_event(_toy_event(), 3, ep, p=0.5).num_nodes for ep in range(8)}
    assert len(masks) > 1
    # p = 0: the very same event object
    assert A.truncate_event(g, 3, 0, p=0.0) is g


def test_dataset_trainset_only(tmp_path):
    from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
    from hnl.models import augment as A
    for name in ("trainset", "valset"):
        os.makedirs(tmp_path / name)
        for k in range(3):
            torch.save(_toy_event(eid=100000000 + k, seed=k), tmp_path / name / f"event{100000000 + k}.pyg")
    hp = {"node_features": ["hit_r", "hit_phi", "hit_z"], "node_scales": [1000, 3.14, 1000], "trunc_p": 1.0,
          "trunc_seed": 5}                     # names as after add_variable_name_prefix_in_config
    ref_val = GraphDataset(str(tmp_path), "valset", 3, "fit", hparams=dict(hp))
    aug_val = A.TruncationGraphDataset(str(tmp_path), "valset", 3, "fit", hparams=dict(hp))
    for k in range(3):
        a, b = ref_val.get(k), aug_val.get(k)
        assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.hit_r, b.hit_r)
    aug_tr = A.TruncationGraphDataset(str(tmp_path), "trainset", 3, "fit", hparams=dict(hp))
    x0 = aug_tr.get(1)                       # epoch 0, event 1
    x1 = aug_tr.get(1 + 3)                   # epoch 1, event 1
    ref = GraphDataset(str(tmp_path), "trainset", 3, "fit", hparams=dict(hp)).get(1)
    assert x0.num_nodes < ref.num_nodes and "edge_y" in x0 and x0.edge_y.shape[0] == x0.edge_index.shape[1]
    assert torch.equal(aug_tr.get(1).edge_index, x0.edge_index)
    hp0 = dict(hp, trunc_p=0.0)
    same = A.TruncationGraphDataset(str(tmp_path), "trainset", 3, "fit", hparams=hp0).get(1)
    assert torch.equal(same.edge_index, ref.edge_index) and torch.equal(same.hit_r, ref.hit_r)
    assert x1.num_nodes != x0.num_nodes or not torch.equal(x1.edge_index, x0.edge_index)


def test_epoch_tagged_sampler():
    from hnl.models.augment import _EpochTaggedSampler

    class M:
        current_epoch = 0
    m = M()
    s = _EpochTaggedSampler(4, m)
    assert list(s) == [0, 1, 2, 3]
    m.current_epoch = 2
    assert list(s) == [8, 9, 10, 11] and len(s) == 4


def test_train_dataloader_collates_and_tags_epochs(tmp_path):
    from hnl.models import augment as A
    os.makedirs(tmp_path / "trainset")
    for k in range(2):
        torch.save(_toy_event(eid=100000000 + k, seed=k), tmp_path / "trainset" / f"event{100000000 + k}.pyg")
    hp = {"node_features": ["hit_r", "hit_phi", "hit_z"], "node_scales": [1000, 3.14, 1000], "trunc_p": 1.0,
          "trunc_seed": 5, "num_workers": [0, 0, 0]}
    ds = A.TruncationGraphDataset(str(tmp_path), "trainset", 2, "fit", hparams=dict(hp))

    class M:
        current_epoch = 0
        trainset = ds
        hparams = hp
    m = M()
    dl = A._trunc_class(object, "X").train_dataloader(m)
    batches = list(dl)
    assert len(batches) == 2 and batches[0].edge_index.shape[0] == 2 and "edge_y" in batches[0]
    m.current_epoch = 3
    again = list(dl)
    assert any(x.num_nodes != y.num_nodes or not torch.equal(x.edge_index, y.edge_index) for x, y in zip(batches, again))
