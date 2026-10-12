"""Stage-1 training variants: same-particle relabelling and focal loss."""

import copy

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")
from torch_geometric.data import Data  # noqa: E402

from acorn.utils.loading_utils import add_variable_name_prefix_in_config  # noqa: E402
from hnl.models import variants as V  # noqa: E402


def _hparams(model, config="configs/baseline/gnn_train.yaml"):
    cfg = yaml.safe_load(open(config))
    cfg.update({"model": model, "input_dir": "/nonexistent", "stage_dir": "/tmp/x", "data_split": [0, 0, 0]})
    return add_variable_name_prefix_in_config(copy.deepcopy(cfg))


def _toy_graph(n=60, E=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Data(hit_r=torch.rand(n, generator=g), hit_phi=torch.rand(n, generator=g) * 2 - 1,
                hit_z=torch.rand(n, generator=g) * 2 - 1, edge_index=torch.randint(0, n, (2, E), generator=g),
                edge_y=torch.rand(E, generator=g) > 0.8, edge_weights=torch.ones(E),
                edge_xd=torch.rand(E, generator=g) > 0.95)


def test_same_pid_relabel():
    ds = V.SamePidGraphDataset.__new__(V.SamePidGraphDataset)
    ds.hparams = _hparams("hnl.models.variants.IGNNSamePid")
    ev = Data(hit_particle_id=torch.tensor([5, 5, 5, 7, 7]), edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 4]]),
              edge_y=torch.tensor([True, False, False, False, True]), hit_r=torch.rand(5), hit_phi=torch.rand(5),
              hit_z=torch.rand(5), hit_id=torch.arange(5))
    out = ds.preprocess_event(ev)
    assert out.edge_y.tolist() == [True, True, False, False, True]      # 0-2 is a same-particle skip edge
    m = V.IGNNSamePid(_hparams("hnl.models.variants.IGNNSamePid"))
    assert m.hparams["dataset_class"] == "SamePidGraphDataset"
    assert m.dataset_resolver.lookup("SamePidGraphDataset") is V.SamePidGraphDataset


def test_focal_reduces_to_bce_at_gamma0():
    m = V.IGNNFocal({**_hparams("hnl.models.variants.IGNNFocal"), "focal_gamma": 0.0})
    out = torch.randn(50)
    batch = Data(edge_y=torch.rand(50) > 0.7, edge_weights=torch.ones(50))
    from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
    ref = InteractionGNN.loss_function(m, out, batch)
    got = m.loss_function(out, batch)
    assert torch.allclose(ref[0], got[0], atol=1e-6)
    m.hparams["focal_gamma"] = 2.0
    assert m.loss_function(out, batch)[0] < ref[0]


def test_vertex_model_forward_and_loss():
    from hnl.models import vertex_gnn as VG
    m = VG.VertexTripletIGNN(_hparams("hnl.models.vertex_gnn.VertexTripletIGNN", "configs/stage1/vertex.yaml"))
    assert m.hparams["dataset_class"] == "VertexGraphDataset" and m.hparams["triplet_topk"] == 4
    E = 300
    g = _toy_graph(E=E)
    out = m(g)
    assert out.shape == (E,) and m._aux.shape == (E,)
    loss, _, _ = m.loss_function(out, g)
    base, _, _ = super(VG.VertexTripletIGNN, m).loss_function(out, g)
    assert loss > base
    loss.backward()
    assert m.vertex_head[0].weight.grad is not None


def test_displaced_weighting(monkeypatch):
    from hnl.models import vertex_gnn as VG

    class FakeLookup:
        def decorate(self, ev):
            ev.hit_absd0 = torch.tensor([0.5, 0.5, 50.0, 50.0])
            return ev
    monkeypatch.setattr(VG, "_LOOKUP", FakeLookup())
    ds = V.DisplacedWeightGraphDataset.__new__(V.DisplacedWeightGraphDataset)
    ds.hparams = _hparams("hnl.models.variants.IGNNDisplacedWeight")
    ev = Data(edge_index=torch.tensor([[0, 2, 0], [1, 3, 2]]), edge_y=torch.tensor([True, True, False]),
              hit_id=torch.arange(4))
    out = ds.construct_weighting(ev)
    assert out.edge_weights.tolist() == [1.0, 3.0, 1.0]


def test_triplet_model_needs_topk_and_checkpointing_is_exact(monkeypatch):
    from hnl.models import triplet_gnn as TG
    with pytest.raises(ValueError):
        TG.TripletIGNN(_hparams("hnl.models.triplet_gnn.TripletIGNN"))          # use_triplets without top-k
    hp = _hparams("hnl.models.triplet_gnn.TripletIGNN", "configs/stage1/triplet.yaml")
    torch.manual_seed(0)
    m = TG.TripletIGNN(hp)
    g = _toy_graph()
    grads = []
    for ck in (True, False):
        if not ck:
            monkeypatch.setattr(TG, "checkpoint", lambda f, *a, **k: f(*a))
        m.zero_grad()
        out = m(g)
        loss, _, _ = m.loss_function(out, g)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-5, atol=1e-7)
    with torch.no_grad():                                                         # inference path: no checkpoint
        assert m(g).shape == (g.edge_index.shape[1],)


def _helix_event(n_tracks=6, seed=1):
    """Hits of a few straight-ish tracks from displaced origins (physical mm), all edges between consecutive layers."""
    g = torch.Generator().manual_seed(seed)
    layers = torch.tensor([60.0, 110, 170, 260, 360, 500])
    xs, ys, zs, src, dst = [], [], [], [], []
    for t in range(n_tracks):
        o = (torch.rand(2, generator=g) - 0.5) * 40
        ang = torch.rand(1, generator=g).item() * 6.28
        cot = (torch.rand(1, generator=g).item() - 0.5)
        for k, rl in enumerate(layers):
            xs.append(o[0] + rl * torch.cos(torch.tensor(ang)))
            ys.append(o[1] + rl * torch.sin(torch.tensor(ang)))
            zs.append(cot * rl)
    x, y, z = torch.stack(xs), torch.stack(ys), torch.tensor(zs)
    n = len(x)
    for i in range(n):                                    # every pair of hits on consecutive layers
        for j in range(n):
            if j % 6 == i % 6 + 1:
                src.append(i)
                dst.append(j)
    return x, y, z, torch.tensor([src, dst])


def _graph_from(x, y, z, ei):
    r = torch.hypot(x, y)
    return Data(hit_r=r / 1000, hit_phi=torch.atan2(y, x) / 3.14, hit_z=z / 1000, edge_index=ei,
                edge_y=torch.zeros(ei.shape[1], dtype=torch.bool), edge_weights=torch.ones(ei.shape[1]))


def test_symmetry_ladder_invariance():
    from hnl.models import triplet_gnn as TG
    x, y, z, ei = _helix_event()
    base = _graph_from(x, y, z, ei)
    c, s_ = torch.cos(torch.tensor(0.7)), torch.sin(torch.tensor(0.7))
    rot = _graph_from(c * x - s_ * y, s_ * x + c * y, z, ei)               # rotation about the beam axis
    shift = _graph_from(x + 3.0, y - 2.0, z, ei)                           # small transverse translation
    R0 = torch.hypot(torch.hypot(x, y), z)
    R1 = torch.hypot(torch.hypot(x + 3.0, y - 2.0), z)
    assert torch.equal(R0[ei[0]] < R0[ei[1]], R1[ei[0]] < R1[ei[1]])       # same outward orientation of every edge
    outs = {}
    for mode in ("absolute", "none"):
        hp = _hparams("hnl.models.triplet_gnn.TripletIGNN", "configs/stage1/triplet.yaml")
        hp.update({"node_input": mode, "edge_geometry": "pointing" if mode == "absolute" else "invariant"})
        torch.manual_seed(0)
        m = TG.TripletIGNN(hp).eval()
        with torch.no_grad():
            outs[mode] = [m(g) for g in (base, rot, shift)]
    inv = outs["none"]
    assert torch.allclose(inv[0], inv[1], atol=1e-4) and torch.allclose(inv[0], inv[2], atol=1e-4)
    absl = outs["absolute"]
    assert not torch.allclose(absl[0], absl[2], atol=1e-4)                 # absolute coordinates see the shift


def test_symmetric_triplet_ignn_exact_invariance():
    from hnl.models import invariant_gnn as IG
    x, y, z, ei = _helix_event()
    hp = _hparams("hnl.models.invariant_gnn.SymmetricTripletIGNN", "configs/stage1/triplet_sym.yaml")
    torch.manual_seed(0)
    m = IG.SymmetricTripletIGNN(hp).eval()
    c, s_ = torch.cos(torch.tensor(1.3)), torch.sin(torch.tensor(1.3))
    variants = {
        "base": _graph_from(x, y, z, ei),
        "big shift": _graph_from(x + 150.0, y - 120.0, z + 300.0, ei),          # flips the R order of many edges
        "rotation": _graph_from(c * x - s_ * y, s_ * x + c * y, z, ei),
        "reversed edges": _graph_from(x, y, z, ei.flip(0)),
    }
    with torch.no_grad():
        outs = {k: m(g) for k, g in variants.items()}
    for k in ("big shift", "rotation", "reversed edges"):
        assert torch.allclose(outs["base"], outs[k], atol=1e-4), k
    # triplet bound and gradient flow
    a, b = ei
    xyz = torch.stack([x, y, z], 1).double()
    e1, e2, i, j, k = IG.undirected_triplets(a, b, len(x))
    assert bool((e1 < e2).all()) and bool((i != k).all())
    tf = IG.symmetric_triplet_features(xyz, i, j, k)
    keep = IG.topk_per_edge_end(e1, e2, j, a, tf[:, 3], 4)
    assert int(keep.sum()) <= 2 * 4 * ei.shape[1]
    m.train()
    g = variants["base"]
    g.edge_y = torch.rand(ei.shape[1]) > 0.7
    loss, _, _ = m.loss_function(m(g), g)
    loss.backward()
    assert all(p.grad is not None for p in m.trip_network.parameters())


def test_symmetric_topk_per_edge_end_is_per_end_and_numbering_free():
    from hnl.models import invariant_gnn as IG
    torch.manual_seed(3)
    # a star: hub 0 with 12 spokes, plus a second star at hub 13 sharing two spokes' outer hits
    a = torch.tensor([0] * 12 + [13] * 6)
    b = torch.tensor(list(range(1, 13)) + [1, 2, 14, 15, 16, 17])
    n = 18
    xyz = torch.randn(n, 3, dtype=torch.float64) * 50
    k = 2

    def kept(perm):
        ap, bp = a[perm], b[perm]
        e1, e2, i, j, kk = IG.undirected_triplets(ap, bp, n)
        tf = IG.symmetric_triplet_features(xyz, i, j, kk)
        m = IG.topk_per_edge_end(e1, e2, j, ap, tf[:, 3], k)
        # per (edge, end) at most k kept triplets
        cnt = {}
        for x1, x2, jj in zip(e1[m].tolist(), e2[m].tolist(), j[m].tolist()):
            for e in (x1, x2):
                cnt[(e, jj)] = cnt.get((e, jj), 0) + 1
        orig = perm.tolist()
        return {(min(orig[x1], orig[x2]), max(orig[x1], orig[x2]), jj)
                for x1, x2, jj in zip(e1[m].tolist(), e2[m].tolist(), j[m].tolist())}, int(m.sum())
    base, nb = kept(torch.arange(len(a)))
    assert nb <= 2 * k * len(a)
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(a), generator=g)
        other, _ = kept(perm)
        assert other == base, seed
    # the hub's 12 spokes: each spoke's hub end keeps its k straightest partners, so the hub has <= 12 * k / 1 kept
    hub = [t for t in base if t[2] == 0]
    assert len(hub) <= 12 * k


def test_symmetric_triplet_ignn_edge_permutation():
    from hnl.models import invariant_gnn as IG
    x, y, z, ei = _helix_event()
    hp = _hparams("hnl.models.invariant_gnn.SymmetricTripletIGNN", "configs/stage1/triplet_sym.yaml")
    torch.manual_seed(0)
    m = IG.SymmetricTripletIGNN(hp).eval()
    perm = torch.randperm(ei.shape[1], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        o0 = m(_graph_from(x, y, z, ei))
        o1 = m(_graph_from(x, y, z, ei[:, perm]))
    assert torch.allclose(o0[perm], o1, atol=1e-4)
