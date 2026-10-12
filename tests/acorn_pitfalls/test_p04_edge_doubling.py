"""p04 "Edge doubling" (brief §4.5): characterisation of how acorn's edge classifiers treat edge direction.

Findings at the pin 99d15341 (no patch touches these files; line numbers are the same in the patched tree unless
noted):

* InteractionGNN.forward (acorn/stages/edge_classifier/models/interaction_gnn.py L200-202) appends the reversed edges
  when hparams.get("undirected_message_passing", True), i.e. by default: directed copy i (src->dst) is at position i and
  its reverse (dst->src) at position E+i. Message passing runs on these 2E directed edges.
* The edge encoder input is cat(x[start], x[end]) (L207-209), built from node embeddings; stored edge features are never
  read by InteractionGNN, so the reversed copy is consistent by construction (its "features" are the swapped node
  embeddings).
* output_step (L184-194) maps back to the E stored edges with torch.mean(scores.view(2, -1), dim=0): the returned
  logit of edge i is the mean of the logits of copies i and E+i, so the score does not depend on the stored direction.
* The flag is not part of the weights: a checkpoint runs with either setting, so a train/infer mismatch is silent.
* InteractionGNN2 (+ the jittable Recurrent/Chained variants) do NOT double edges ("Operates on directed graphs",
  L386-391); they read hparams["edge_features"] directly and are direction-sensitive. The *WithPyG variants (which also
  double, L331-333 / L363-367) cannot be constructed with the pinned PyG 2.7.0.
* The dataset-level GraphDataset.to_undirected (edge_classifier_stage.py L740-776 patched, L742-778 at the pin) copies
  every edge-like tensor verbatim (no sign flip). It is only called by HeteroGraphDataset, which the default
  EdgeClassifierStage resolver cannot even select; the call in GraphDataset.handle_edge_list is commented out, so
  `undirected: True` does not double edges in the default GraphDataset.
* acorn's own difference features (handle_edge_features, loading_utils.py L376-417 patched / L361-402 at the pin) are
  computed as dst - src: dr, dphi, dz, deta are odd under edge reversal, phislope and rphislope are even. Features that
  are already present are never recomputed.
* acorn's metric-learning graphs hold one column per unordered hit pair (rearrange_by_distance + torch.unique), so
  their edge_index width is the unique-pair count while the IGNN processes twice as many directed edges.
"""

import itertools

import pytest
import torch
from torch_geometric.data import Data

from acorn.stages.edge_classifier import InteractionGNN, InteractionGNN2, InteractionGNN2WithPyG, InteractionGNNWithPyG
from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.stages.edge_classifier.models.jittable_gnn import ChainedInteractionGNN2, RecurrentInteractionGNN2
from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.utils.loading_utils import handle_edge_features

H = 16  # hidden size of the toy models
N_NODES = 6
EDGES = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 5]])  # one stored column per unordered pair
E = EDGES.shape[1]
FLIP = [1, 3]  # columns whose stored direction is reversed in the direction tests


def ignn_hparams(**overrides):
    """Minimal InteractionGNN hparams, shaped like examples/Example_1/gnn_train.yaml (smaller, 8 iterations)."""
    hp = dict(
        node_features=["hit_r", "hit_phi", "hit_z"],
        hidden=H,
        n_graph_iters=8,
        nb_node_layer=2,
        nb_edge_layer=2,
        layernorm=True,
        batchnorm=False,
        hidden_activation="SiLU",
        output_activation="Tanh",
        aggregation=["sum", "mean", "max", "std"],
    )
    hp.update(overrides)
    return hp


def ignn2_hparams(**overrides):
    """Minimal InteractionGNN2 hparams (keys read in interaction_gnn.py L393-550)."""
    hp = dict(
        node_features=["hit_r", "hit_phi", "hit_z"],
        hidden=H,
        n_graph_iters=4,
        n_node_net_layers=2,
        n_edge_net_layers=2,
        n_edge_decoder_layers=2,
        layernorm=True,
        batchnorm=False,
        hidden_activation="SiLU",
        output_activation="Tanh",
        edge_output_transform_final_activation=None,
        concat=True,
        in_out_diff_agg=True,
        edge_net_recurrent=True,
        node_net_recurrent=True,
        checkpointing=False,
    )
    hp.update(overrides)
    return hp


def toy_event(edge_index=EDGES):
    g = torch.Generator().manual_seed(0)
    return Data(
        hit_r=torch.rand(N_NODES, generator=g),
        hit_phi=torch.rand(N_NODES, generator=g),
        hit_z=torch.rand(N_NODES, generator=g),
        edge_index=edge_index.clone(),
        num_nodes=N_NODES,
    )


def build(cls, hp, seed=1):
    torch.manual_seed(seed)
    return cls(hp).eval()


def capture_first_call(module, store, key):
    """Forward hook that records (input, output) of the first call. It must return None (a returned value would
    replace the module output)."""

    def hook(mod, args, out):
        if key not in store:
            store[key] = (args[0].detach().clone(), out.detach().clone())

    return module.register_forward_hook(hook)


def run(model, event):
    with torch.no_grad():
        return model(event)


def unique_unordered_pairs(edge_index):
    return torch.unique(torch.sort(edge_index, dim=0).values, dim=1).shape[1]


# --------------------------------------------------------------------------------------------- InteractionGNN


def test_ignn_doubles_edges_by_default():
    hp = ignn_hparams()
    assert "undirected_message_passing" not in hp  # the default path: hparams.get(..., True)
    model = build(InteractionGNN, hp)
    store = {}
    capture_first_call(model.edge_encoder, store, "enc")
    out = run(model, toy_event())

    enc_in, _ = store["enc"]
    assert out.shape == (E,)
    assert enc_in.shape == (2 * E, 2 * H)  # 2E directed edges enter the edge encoder
    # copy E+i is the reverse of copy i, in the same order: its encoder input is cat(x[dst], x[src])
    assert torch.equal(enc_in[E:, :H], enc_in[:E, H:])
    assert torch.equal(enc_in[E:, H:], enc_in[:E, :H])


def test_ignn_output_is_mean_of_the_two_directed_logits():
    model = build(InteractionGNN, ignn_hparams())
    store = {}
    capture_first_call(model.output_edge_classifier, store, "cls")
    out = run(model, toy_event())

    raw = store["cls"][1].squeeze(-1)
    assert raw.shape == (2 * E,)
    torch.testing.assert_close(out, (raw[:E] + raw[E:]) / 2, rtol=0, atol=1e-7)
    # the two directed logits are not equal in general: the mean really combines two different numbers
    assert (raw[:E] - raw[E:]).abs().max() > 1e-3


def test_ignn_scores_do_not_depend_on_stored_direction():
    model = build(InteractionGNN, ignn_hparams())
    flipped = EDGES.clone()
    flipped[:, FLIP] = flipped[:, FLIP].flip(0)
    out = run(model, toy_event())
    out_flipped = run(model, toy_event(flipped))
    torch.testing.assert_close(out_flipped, out, rtol=0, atol=1e-6)


def test_ignn_flag_false_disables_doubling_and_is_not_stored_in_the_weights():
    model = build(InteractionGNN, ignn_hparams())
    directed = build(InteractionGNN, ignn_hparams(undirected_message_passing=False), seed=2)
    # identical parameter shapes: a checkpoint trained with one setting loads strictly into the other
    directed.load_state_dict(model.state_dict(), strict=True)
    store = {}
    capture_first_call(directed.edge_encoder, store, "enc")
    out_directed = run(directed, toy_event())
    out = run(model, toy_event())

    assert store["enc"][0].shape == (E, 2 * H)
    assert out_directed.shape == (E,)
    assert (out_directed - out).abs().max() > 1e-3  # same weights, different scores: a silent change


def test_ignn_input_with_both_directions_double_counts_messages():
    """Storing both (i, j) and (j, i) gives 4 directed copies per pair inside the IGNN: the sum channels of the
    aggregated messages double (mean / max / std do not), so the scores change, and every pair is scored twice."""
    model = build(InteractionGNN, ignn_hparams())
    store = {}
    capture_first_call(model.node_network, store, "node")
    out_unique = run(model, toy_event())
    msg_unique = store.pop("node")[0]

    both = torch.cat([EDGES, EDGES.flip(0)], dim=1)
    assert unique_unordered_pairs(both) == E
    capture_first_call(model.node_network, store, "node")
    out_both = run(model, toy_event(both))
    msg_both = store.pop("node")[0]

    assert out_both.shape == (2 * E,)
    torch.testing.assert_close(out_both[:E], out_both[E:], rtol=0, atol=1e-6)  # (i,j) and (j,i) get one score
    # node_network input = [x | sum mean max std (to end) | sum mean max std (to start)], H columns each
    blocks = msg_unique.split(H, dim=1)
    blocks_both = msg_both.split(H, dim=1)
    assert len(blocks) == 9
    for k, (b, b2) in enumerate(zip(blocks, blocks_both)):
        if k in (1, 5):  # sum aggregation
            torch.testing.assert_close(b2, 2 * b, rtol=1e-5, atol=1e-5)
        else:  # x, mean, max, std
            torch.testing.assert_close(b2, b, rtol=1e-4, atol=1e-4)
    assert (out_both[:E] - out_unique).abs().max() > 1e-3


def test_ignn_ignores_edge_features():
    """InteractionGNN never reads hparams["edge_features"]: changing a stored edge feature changes nothing."""
    model = build(InteractionGNN, ignn_hparams(edge_features=["edge_dr"]))
    assert model.edge_encoder[0].in_features == 2 * H  # input is cat(x[start], x[end]), not the edge features
    event = toy_event()
    handle_edge_features(event, ["edge_dr"])
    out = run(model, event)
    event.edge_dr = -event.edge_dr + 123.0
    torch.testing.assert_close(run(model, event), out, rtol=0, atol=0)


def test_with_pyg_variants_cannot_be_built_with_pinned_pyg():
    """The *WithPyG variants (which double edges too, L331-333) pass all hparams to torch_geometric's
    MessagePassing.__init__, which rejects them in PyG 2.7.0: not usable in this environment."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'node_features'"):
        InteractionGNNWithPyG(ignn_hparams())
    with pytest.raises(TypeError, match="unexpected keyword argument 'node_features'"):
        InteractionGNN2WithPyG(ignn2_hparams())


# ---------------------------------------------------------------------- InteractionGNN2 and jittable variants


def test_ignn2_does_not_double_and_ignores_the_flag():
    model = build(InteractionGNN2, ignn2_hparams())
    model_flag = build(InteractionGNN2, ignn2_hparams(undirected_message_passing=True))
    store = {}
    capture_first_call(model.edge_encoder, store, "enc")
    out = run(model, toy_event())
    assert store["enc"][0].shape == (E, 2 * H)
    assert out.shape == (E,)
    torch.testing.assert_close(run(model_flag, toy_event()), out, rtol=0, atol=0)


def test_ignn2_is_direction_sensitive():
    model = build(InteractionGNN2, ignn2_hparams())
    flipped = EDGES.clone()
    flipped[:, FLIP] = flipped[:, FLIP].flip(0)
    out = run(model, toy_event())
    out_flipped = run(model, toy_event(flipped))
    assert (out_flipped - out).abs().max() > 1e-3


def test_ignn2_reads_edge_features_without_doubling():
    feats = ["edge_dr", "edge_dz"]
    model = build(InteractionGNN2, ignn2_hparams(edge_features=feats))
    event = toy_event()
    handle_edge_features(event, feats)
    store = {}
    capture_first_call(model.edge_encoder, store, "enc")
    run(model, event)
    enc_in = store["enc"][0]
    assert enc_in.shape == (E, len(feats))
    torch.testing.assert_close(enc_in, torch.stack([event.edge_dr, event.edge_dz], dim=-1))


def test_ignn2_empty_edge_feature_list_crashes():
    """forward checks len(self.hparams) instead of len(self.hparams["edge_features"]) (L557): an empty list crashes,
    so omit the key instead of writing edge_features: []."""
    model = build(InteractionGNN2, ignn2_hparams(edge_features=[]))
    with pytest.raises(RuntimeError, match="non-empty TensorList"):
        run(model, toy_event())


@pytest.mark.parametrize("cls", [RecurrentInteractionGNN2, ChainedInteractionGNN2])
def test_jittable_ignn2_variants_do_not_double(cls):
    model = build(cls, ignn2_hparams())
    event = toy_event()
    x = torch.stack([event.hit_r, event.hit_phi, event.hit_z], dim=-1)
    store = {}
    capture_first_call(model.edge_encoder, store, "enc")
    with torch.no_grad():
        out = model(x, EDGES)
    assert store["enc"][0].shape == (E, 2 * H)
    assert out.shape == (E,)


# ---------------------------------------------------------------------------- dataset-level doubling (dead code)


def _dataset_event():
    hit_r = torch.tensor([30.0, 60.0, 90.0, 120.0, 150.0, 40.0])
    ev = Data(
        hit_id=torch.arange(N_NODES),
        hit_r=hit_r,
        hit_phi=torch.tensor([0.10, 0.12, 0.15, 0.19, 0.24, -0.90]),
        hit_z=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, -3.0]),
        hit_eta=torch.tensor([0.10, 0.20, 0.30, 0.35, 0.40, -0.50]),
        edge_index=EDGES.clone(),
        edge_y=torch.tensor([True, True, True, True, False]),
        track_edges=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        track_to_edge_map=torch.tensor([0, 1, 2, 3]),
        track_particle_id=torch.tensor([7, 7, 7, 7]),
    )
    ev.num_nodes = N_NODES
    return ev


def _empty_dataset(tmp_path, hparams):
    with pytest.warns(UserWarning, match="No data files found"):
        return GraphDataset(str(tmp_path), data_name="trainset", hparams=hparams)


def test_graph_dataset_undirected_flag_does_not_double_edges(tmp_path):
    ds = _empty_dataset(tmp_path, {"undirected": True})
    event = ds.preprocess_event(_dataset_event())
    torch.testing.assert_close(event.edge_index, EDGES)
    assert event.edge_y.shape == (E,) and event.edge_weights.shape == (E,)


def test_to_undirected_copies_edge_features_without_sign_flip(tmp_path):
    ds = _empty_dataset(tmp_path, {})
    event = _dataset_event()
    src, dst = event.edge_index
    event.edge_dr = event.hit_r[dst] - event.hit_r[src]  # an antisymmetric (odd) edge feature, stored
    dr = event.edge_dr.clone()
    out = ds.to_undirected(event)

    torch.testing.assert_close(out.edge_index, torch.cat([EDGES, EDGES.flip(0)], dim=1))
    # current behaviour: the reversed copy carries the SAME dr (a correct reversal would carry -dr)
    torch.testing.assert_close(out.edge_dr, torch.cat([dr, dr]))
    src2, dst2 = out.edge_index
    assert not torch.allclose(out.edge_dr[E:], (out.hit_r[dst2] - out.hit_r[src2])[E:])
    torch.testing.assert_close(out.edge_y, torch.cat([event.edge_y[:E], event.edge_y[:E]]))
    torch.testing.assert_close(out.track_to_edge_map, torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]))


def test_hetero_dataset_cannot_be_selected_by_the_default_resolver():
    """HeteroGraphDataset (the only caller of to_undirected) is not registered in EdgeClassifierStage's resolver."""
    model = build(InteractionGNN, ignn_hparams())
    assert model.dataset_resolver.lookup(None) is GraphDataset
    with pytest.raises(KeyError, match="HeteroGraphDataset"):
        model.dataset_resolver.lookup("HeteroGraphDataset")


# ------------------------------------------------------------------------------- edge features and reversal


ACORN_EDGE_FEATURES = ["edge_dr", "edge_dphi", "edge_dz", "edge_deta", "edge_phislope", "edge_rphislope"]


def test_acorn_edge_feature_parity_under_reversal():
    """What a reversed copy must carry: odd features flip sign, even ones do not."""
    fwd = _dataset_event()
    rev = _dataset_event()
    rev.edge_index = rev.edge_index.flip(0)
    handle_edge_features(fwd, ACORN_EDGE_FEATURES)
    handle_edge_features(rev, ACORN_EDGE_FEATURES)
    for name in ["edge_dr", "edge_dphi", "edge_dz", "edge_deta"]:
        torch.testing.assert_close(rev[name], -fwd[name], msg=name)
    for name in ["edge_phislope", "edge_rphislope"]:
        torch.testing.assert_close(rev[name], fwd[name], msg=name)
        assert fwd[name].abs().max() > 0


def test_handle_edge_features_keeps_stale_precomputed_features():
    """handle_edge_features skips a feature that already exists: after re-orienting edges it is not recomputed."""
    event = _dataset_event()
    handle_edge_features(event, ["edge_dr"])
    dr = event.edge_dr.clone()
    event.edge_index = event.edge_index.flip(0)
    handle_edge_features(event, ["edge_dr"])
    torch.testing.assert_close(event.edge_dr, dr)  # stale: should be -dr for the flipped edges


# --------------------------------------------------------------------------------------------- graph size


def test_metric_learning_graph_has_one_column_per_unordered_pair():
    """acorn's metric-learning truth step orients every edge inner -> outer in R = r^2 + z^2 and drops duplicates, so
    its edge_index width equals the number of unique unordered hit pairs; the IGNN then runs on twice that many."""
    ml = MetricLearning(dict(node_features=["hit_r", "hit_phi", "hit_z"], emb_hidden=8, nb_layer=2, emb_dim=4,
                             activation="Tanh"))
    hit_r = torch.tensor([30.0, 60.0, 90.0, 120.0, 150.0])
    hit_z = torch.tensor([5.0, -10.0, 15.0, -20.0, 25.0])
    batch = Data(hit_r=hit_r, hit_z=hit_z, track_edges=torch.tensor([[0, 1], [1, 2]]), num_nodes=5)
    all_ordered = torch.tensor([p for p in itertools.permutations(range(5), 2)]).T  # 20 directed edges
    with pytest.warns(UserWarning, match="hit_R not found"):
        edges, truth, _, _ = ml.get_truth(batch, all_ordered.clone(), undirected=False)

    assert edges.shape[1] == 10 == unique_unordered_pairs(all_ordered)
    assert unique_unordered_pairs(edges) == edges.shape[1]  # no pair twice
    R = hit_r**2 + hit_z**2
    assert bool((R[edges[0]] <= R[edges[1]]).all())
    assert int(truth.sum()) == 2

    model = build(InteractionGNN, ignn_hparams())
    store = {}
    capture_first_call(model.edge_encoder, store, "enc")
    event = Data(hit_r=hit_r, hit_phi=torch.zeros(5), hit_z=hit_z, edge_index=edges, num_nodes=5)
    out = run(model, event)
    assert out.shape == (10,) and store["enc"][0].shape[0] == 20


def test_truth_matching_is_orientation_sensitive_unless_undirected():
    """The stored edges are oriented inner -> outer in R, but graph_intersection (used by MetricLearning.get_truth
    and EdgeClassifierStage.save_edge_scores) compares directed pairs unless undirected=True: a true edge that points
    inward in R (about 1 % of our true edges, RESULTS.md) is labelled false. MetricLearning inference reads
    hparams.get("undirected", False) (metric_learning.py L389 patched, L385 at the pin)."""
    ml = MetricLearning(dict(node_features=["hit_r", "hit_phi", "hit_z"], emb_hidden=8, nb_layer=2, emb_dim=4,
                             activation="Tanh"))
    hit_r = torch.tensor([30.0, 60.0, 90.0, 120.0, 150.0])
    hit_z = torch.zeros(5)
    track_edges = torch.tensor([[0, 3], [1, 2]])  # 0 -> 1 points outward, 3 -> 2 points inward
    all_ordered = torch.tensor([p for p in itertools.permutations(range(5), 2)]).T
    n_true = {}
    for undirected in (False, True):
        batch = Data(hit_r=hit_r, hit_z=hit_z, track_edges=track_edges, num_nodes=5)
        with pytest.warns(UserWarning, match="hit_R not found"):
            edges, truth, truth_map, _ = ml.get_truth(batch, all_ordered.clone(), undirected=undirected)
        assert edges.shape[1] == 10
        n_true[undirected] = int(truth.sum())
        if not undirected:
            assert truth_map.tolist()[1] == -1  # the inward true edge has no partner in the graph
    assert n_true == {False: 1, True: 2}
