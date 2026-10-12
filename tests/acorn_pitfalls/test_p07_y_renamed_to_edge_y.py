"""p07 "y is renamed to edge_y" (brief §4.5): characterisation of acorn's variable-name prefixing.

Findings at the pin 99d15341 (unpatched; patch 0005 only shifts acorn/utils/loading_utils.py by +15 lines):

* loading_utils.variable_name_prefix_map (L441-527 patched, L426-512 at the pin) has 84 entries: 66 hit keys
  (x -> hit_x, z -> hit_z, r, phi, eta, module_id, cluster_*, ...), 9 edge keys (y -> edge_y, dr, dphi, dz, deta,
  phislope, rphislope, scores -> edge_scores, labels -> edge_track_labels) and 9 track keys (particle_id, pt, radius,
  primary, nhits, pdgId -> track_particle_*, eta_particle -> track_particle_eta, redundant_split_edges,
  truth_map -> track_to_edge_map). There is no entry producing hit_y: a bare `y` always means the EDGE truth label.
* add_variable_name_prefix_in_pyg (L530-534 / L515-519) renames on every load unless variable_with_prefix is set
  (GraphDataset.get edge_classifier_stage.py L658-661, EventDataset.preprocess_graph graph_construction_stage.py
  L305-308); remove_variable_name_prefix_in_pyg (L537-544 / L522-529) undoes it on every save. hit_y passes both
  unchanged. add_variable_name_prefix_in_config (L547-602 / L532-587) applies the same map to node_features,
  edge_features, weighting conditions, target_tracks and hard_cuts.
* get_variable_type (mapping_utils.py L61-69) types a tensor only by its prefix (hit_/edge_/track_); anything else is
  OTHER and is not subset by node cuts or edge score cuts.
"""

import os
import types

import pytest
import torch
from torch_geometric.data import Data

from acorn.stages.edge_classifier import InteractionGNN
from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.stages.graph_construction.graph_construction_stage import EventDataset
from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.utils.loading_utils import (
    add_variable_name_prefix_in_config,
    add_variable_name_prefix_in_pyg,
    handle_hard_node_cuts_pyg,
    load_pyg,
    remove_variable_name_prefix_in_pyg,
    save_pyg,
    variable_name_prefix_map,
)
from acorn.utils.mapping_utils import VariableType, get_variable_type

HIT_X = torch.tensor([30.0, 60.0, 90.0, 35.0, 65.0, 95.0])
HIT_Y = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
HIT_Z = torch.tensor([5.0, 10.0, 15.0, -5.0, -10.0, -15.0])
N = HIT_X.shape[0]
TRACK_EDGES = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])

EDGE_KEYS = {
    "y": "edge_y", "dr": "edge_dr", "dphi": "edge_dphi", "dz": "edge_dz", "deta": "edge_deta",
    "phislope": "edge_phislope", "rphislope": "edge_rphislope", "scores": "edge_scores",
    "labels": "edge_track_labels",
}
TRACK_KEYS = {
    "particle_id": "track_particle_id", "pt": "track_particle_pt", "radius": "track_particle_radius",
    "primary": "track_particle_primary", "nhits": "track_particle_nhits", "pdgId": "track_particle_pdgId",
    "eta_particle": "track_particle_eta", "redundant_split_edges": "track_redundant_split_edges",
    "truth_map": "track_to_edge_map",
}


def raw_event(hit_y_name):
    """A converter's output in acorn's unprefixed on-disk convention, with the hit y coordinate named hit_y_name."""
    event = Data(
        hit_id=torch.arange(N),
        x=HIT_X.clone(),
        z=HIT_Z.clone(),
        r=torch.sqrt(HIT_X**2 + HIT_Y**2),
        phi=torch.atan2(HIT_Y, HIT_X),
        track_edges=TRACK_EDGES.clone(),
        particle_id=torch.tensor([11, 11, 22, 22]),
        pt=torch.tensor([2.0, 2.0, 0.5, 0.5]),
    )
    event[hit_y_name] = HIT_Y.clone()
    event.config = [{}]
    event.event_id = "000000001"
    return event


# ------------------------------------------------------------------------------------------- the map itself


def test_prefix_map_contents():
    m = variable_name_prefix_map
    assert len(m) == 84 and len(set(m.values())) == 84  # one-to-one, so remove_... inverts add_... exactly
    assert m["y"] == "edge_y" and m["x"] == "hit_x" and m["z"] == "hit_z"
    assert "hit_y" not in m.values()  # nothing is ever renamed to hit_y
    assert {k: v for k, v in m.items() if v.startswith("edge_")} == EDGE_KEYS
    assert {k: v for k, v in m.items() if v.startswith("track_")} == TRACK_KEYS
    hit = {k: v for k, v in m.items() if v.startswith("hit_")}
    assert len(hit) == 66 and all(v == f"hit_{k}" for k, v in hit.items())
    # every other y-like name is a hit feature; only the bare "y" is not
    assert {k for k in hit if "y" in k} == {"cluster_y_1", "cluster_y_2", "norm_y_1", "norm_y_2", "ly", "layer_id"}


def test_variable_type_is_decided_by_prefix_only():
    assert get_variable_type("hit_y") == VariableType.NODE_LIKE
    assert get_variable_type("edge_y") == VariableType.EDGE_LIKE
    assert get_variable_type("track_particle_pt") == VariableType.TRACK_LIKE
    for name in ["y", "x", "pt", "d0", "pos"]:
        assert get_variable_type(name) == VariableType.OTHER


# ------------------------------------------------------------------------------------- toy Data objects


def test_hit_coordinate_named_y_becomes_edge_y():
    event = add_variable_name_prefix_in_pyg(raw_event("y"))
    keys = set(event.keys())
    assert {"hit_x", "hit_z", "hit_r", "hit_phi", "track_particle_id", "track_particle_pt"} <= keys
    assert "hit_y" not in keys and "y" not in keys
    torch.testing.assert_close(event.edge_y, HIT_Y)  # the hit coordinates now pose as the edge truth label
    assert get_variable_type("edge_y") == VariableType.EDGE_LIKE


def test_hit_coordinate_named_y_overwrites_existing_edge_truth():
    event = raw_event("y")
    event.edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    event.edge_y = torch.tensor([True, True, False])
    event = add_variable_name_prefix_in_pyg(event)
    torch.testing.assert_close(event.edge_y, HIT_Y)  # the truth labels are gone, no error
    assert event.edge_y.shape[0] != event.edge_index.shape[1]


def test_hit_y_survives_save_and_load_naming():
    event = add_variable_name_prefix_in_pyg(raw_event("hit_y"))
    torch.testing.assert_close(event.hit_y, HIT_Y)
    torch.testing.assert_close(event.hit_x, HIT_X)
    on_disk = remove_variable_name_prefix_in_pyg(event.clone())
    assert {"x", "z", "hit_y", "particle_id", "pt"} <= set(on_disk.keys())
    assert "hit_x" not in on_disk.keys() and "y" not in on_disk.keys()
    again = add_variable_name_prefix_in_pyg(on_disk)
    assert set(again.keys()) == set(event.keys())
    torch.testing.assert_close(again.hit_y, HIT_Y)


def test_config_names_are_mapped_the_same_way():
    config = add_variable_name_prefix_in_config(
        dict(
            node_features=["x", "y", "z"],
            edge_features=["dr", "dphi"],
            weighting=[{"weight": 1.0, "conditions": {"y": True, "pt": [1.0, float("inf")]}}],
            hard_cuts={"pt": [1.0, float("inf")]},
            target_tracks={"pt": [1.0, float("inf")], "nhits": [5, float("inf")]},
        )
    )
    assert config["node_features"] == ["hit_x", "edge_y", "hit_z"]  # asks for the truth label as a node feature
    assert config["edge_features"] == ["edge_dr", "edge_dphi"]
    assert set(config["weighting"][0]["conditions"]) == {"edge_y", "track_particle_pt"}
    assert set(config["hard_cuts"]) == {"track_particle_pt"}
    assert set(config["target_tracks"]) == {"track_particle_pt", "track_particle_nhits"}
    safe = add_variable_name_prefix_in_config(dict(node_features=["hit_x", "hit_y", "hit_z"]))
    assert safe["node_features"] == ["hit_x", "hit_y", "hit_z"]


# ------------------------------------------------------------------------------------------ real loaders


def _write_event(root, event, name="event000000001-graph.pyg"):
    os.makedirs(os.path.join(root, "trainset"), exist_ok=True)
    save_pyg(event, os.path.join(root, "trainset", name))


@pytest.mark.parametrize("hit_y_name", ["y", "hit_y"])
def test_graph_construction_loader(tmp_path, hit_y_name):
    _write_event(str(tmp_path), raw_event(hit_y_name))
    event = EventDataset(str(tmp_path), "trainset", 1, hparams={})[0]
    if hit_y_name == "y":
        assert "hit_y" not in event.keys()
        torch.testing.assert_close(event.edge_y, HIT_Y)
    else:
        assert "edge_y" not in event.keys()
        torch.testing.assert_close(event.hit_y, HIT_Y)


def _gnn_stage_event(n_edges):
    event = raw_event("y")
    event.edge_index = torch.stack([torch.arange(n_edges) % N, (torch.arange(n_edges) + 1) % N])
    event.edge_y = torch.zeros(n_edges, dtype=torch.bool)
    event.edge_y[:2] = True
    event.truth_map = torch.tensor([0, 1, -1, -1])
    return event


def test_gnn_stage_loader_fails_only_by_accident(tmp_path):
    """In the edge-classifier GraphDataset the misnamed hit y replaces edge_y; the only guard is the length assert
    in construct_weighting (edge_classifier_stage.py L711-714), which passes when N == E."""
    _write_event(str(tmp_path / "a"), _gnn_stage_event(n_edges=5), "event000000001.pyg")
    with pytest.raises(AssertionError, match="Input graph has 5 edges, but 6 truth labels"):
        GraphDataset(str(tmp_path / "a"), data_name="trainset", hparams={})[0]

    _write_event(str(tmp_path / "b"), _gnn_stage_event(n_edges=N), "event000000001.pyg")
    event = GraphDataset(str(tmp_path / "b"), data_name="trainset", hparams={})[0]
    torch.testing.assert_close(event.edge_y, HIT_Y)  # silently used as edge truth


# --------------------------------------------------------------- metric-learning inference (silent hit-y loss)


def _metric_learning_predict(tmp_path, hit_y_name, node_features):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_event(str(in_dir), raw_event(hit_y_name))
    config = add_variable_name_prefix_in_config(
        dict(
            input_dir=str(in_dir), stage_dir=str(out_dir), data_split=[1, 0, 0], num_workers=[0, 0, 0],
            node_features=node_features, node_scales=[100.0, 10.0, 100.0],
            emb_hidden=8, nb_layer=2, emb_dim=4, activation="Tanh", margin=0.1,
            r_infer=2.1, knn_infer=10,  # embeddings lie on the unit sphere: r = 2.1 connects every pair
            weighting=[{"weight": 1.0, "conditions": {"y": True}}],
        )
    )
    torch.manual_seed(0)
    model = MetricLearning(config).eval()
    model.trainset = EventDataset(str(in_dir), "trainset", 1, hparams=model.hparams)
    model.valset = model.testset = None
    model._trainer = types.SimpleNamespace(state=types.SimpleNamespace(stage="predict"))  # skips metric logging
    batch = next(iter(model.train_dataloader()))
    with torch.no_grad(), pytest.warns(UserWarning, match="hit_R not found"):
        model.predict_step(batch, 0, 0)
    return load_pyg(str(out_dir / "trainset" / "event000000001.pyg"), weights_only=False)


def _truth_of(saved):
    """Expected edge truth of the saved graph: the complete graph on 6 hits (15 unordered pairs), 4 of them true."""
    pairs = saved.edge_index.T.tolist()
    assert len({tuple(p) for p in pairs}) == 15
    true_pairs = {tuple(t) for t in TRACK_EDGES.T.tolist()}
    return torch.tensor([tuple(p) in true_pairs for p in pairs])


def test_metric_learning_recipe_features_drop_a_hit_y_named_y(tmp_path):
    """node_features [r, phi, z] (the brief's recipe): the converter's `y` is loaded as edge_y and overwritten by the
    truth labels in shared_evaluation (metric_learning.py L383-390, L379-386 at the pin); the saved graph has no
    hit y coordinate left."""
    saved = _metric_learning_predict(tmp_path, "y", ["r", "phi", "z"])
    assert "hit_y" not in saved.keys()
    assert not any(v.shape == HIT_Y.shape and torch.allclose(v.float(), HIT_Y) for _, v in saved.items()
                   if isinstance(v, torch.Tensor))
    assert saved.y.dtype == torch.bool and torch.equal(saved.y, _truth_of(saved))


def test_metric_learning_xyz_features_drop_hit_y_and_corrupt_truth(tmp_path):
    """node_features [x, y, z]: the config maps y -> edge_y, so the embedding uses the hit y coordinate, then the
    truth overwrites edge_y and predict_step's unscale_features (metric_learning.py L486, L482 at the pin) multiplies
    the truth labels by y's node scale."""
    saved = _metric_learning_predict(tmp_path, "y", ["x", "y", "z"])
    assert "hit_y" not in saved.keys()
    truth = _truth_of(saved)
    assert saved.y.dtype == torch.float32
    torch.testing.assert_close(saved.y, truth.float() * 10.0)  # labels are {0, 10}, not {False, True}
    assert int((saved.y == 1).sum()) == 0  # the GNN loss's positive mask (edge_y == 1) is empty


def test_metric_learning_with_hit_y_keeps_everything(tmp_path):
    saved = _metric_learning_predict(tmp_path, "hit_y", ["x", "hit_y", "z"])
    torch.testing.assert_close(saved.hit_y, HIT_Y)
    assert saved.y.dtype == torch.bool and torch.equal(saved.y, _truth_of(saved))
    assert int(saved.y.sum()) == 4


def test_corrupted_truth_gives_no_positive_loss_in_the_gnn_stage(tmp_path):
    """The {0, 10} labels written above reach the GNN stage as edge_y; EdgeClassifierStage.loss_function
    (edge_classifier_stage.py L318) then finds no positive edge: silent garbage training."""
    bad = _metric_learning_predict(tmp_path / "bad", "y", ["x", "y", "z"])
    good = _metric_learning_predict(tmp_path / "good", "hit_y", ["x", "hit_y", "z"])
    gnn = InteractionGNN(dict(node_features=["hit_r", "hit_phi", "hit_z"], hidden=8, n_graph_iters=1,
                              nb_node_layer=1, nb_edge_layer=1, layernorm=False, hidden_activation="SiLU"))
    pos = {}
    for tag, saved in [("bad", bad), ("good", good)]:
        root = tmp_path / f"gnn_{tag}"
        _write_event(str(root), saved, "event000000001.pyg")
        event = GraphDataset(str(root), data_name="trainset", hparams={})[0]
        _, pos[tag], _ = gnn.loss_function(torch.zeros(event.edge_index.shape[1]), event)
    assert pos["bad"] == 0 and pos["good"] > 0


# ------------------------------------------------------------------------- unprefixed names are not cut


def test_unprefixed_tensors_are_not_cut_with_the_graph(tmp_path):
    """Only hit_/edge_/track_ tensors follow node cuts and edge score cuts; an unprefixed name silently misaligns."""
    event = add_variable_name_prefix_in_pyg(raw_event("hit_y"))
    event.num_nodes = N
    event.d0 = torch.arange(N, dtype=torch.float)  # per-hit, unprefixed
    event = handle_hard_node_cuts_pyg(event, {"track_particle_pt": [1.0, float("inf")]})
    assert event.hit_y.shape == (3,) and event.hit_x.shape == (3,)
    assert event.d0.shape == (N,)  # not cut

    with pytest.warns(UserWarning, match="No data files found"):
        ds = GraphDataset(str(tmp_path), data_name="trainset", hparams={})
    edge_event = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_y=torch.tensor([True, False, True, False]),
        edge_scores=torch.tensor([0.9, 0.1, 0.8, 0.2]),
        edge_mine=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        mine=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        track_to_edge_map=torch.tensor([0, 2]),
        num_nodes=5,
    )
    ds.apply_score_cut(edge_event, 0.5)
    torch.testing.assert_close(edge_event.edge_mine, torch.tensor([1.0, 3.0]))
    assert edge_event.mine.shape == (4,)  # not cut
