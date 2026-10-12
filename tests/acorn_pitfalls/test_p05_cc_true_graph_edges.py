"""p05 (brief §4.5): ConnectedComponents(on_true_graph_edges) reads graph.y.

acorn's track-building GraphDataset renames y -> edge_y (add_variable_name_prefix_in_pyg) before
ConnectedComponents._build_event runs. At the pin, _build_event selects ``graph.edge_index[:, graph.y]``; graph.y is
then None, ``edge_index[:, None]`` has shape (2, 1, E) and remove_isolated_nodes flattens it back to all E edges, so
the "perfect classifier" CC keeps every edge and returns one giant component. Patch 0002 selects the edges with
``edge_y`` (falling back to ``y``) cast to bool.

Toy event: particle 11 (7 hits), particle 22 (5 hits), one noise hit, four fake edges that join everything into a
single component if they are kept. All tests assert the CORRECT (patched) behaviour, so they fail at the pin
(ACORN_PIN_SRC); hit_id is arange(N) here so that p06 (load_reconstruction_df) cannot interfere.
"""

import glob
import os

import pytest
import torch
from torch_geometric.data import Data

from acorn.stages.graph_construction.models.utils import graph_intersection
from acorn.stages.track_building.models.connected_components import ConnectedComponents
from acorn.stages.track_building.track_building_stage import GraphDataset
from acorn.utils.loading_utils import add_variable_name_prefix_in_pyg, load_pyg, save_pyg

P1, P2 = 11, 22
P1_NODES = [0, 2, 4, 6, 8, 10, 12]  # 7 hits, ordered outward
P2_NODES = [1, 3, 5, 7, 9]  # 5 hits, ordered outward
NOISE_NODES = [11]
N_NODES = 13
FAKE_EDGES = [(2, 3), (7, 8), (9, 11), (10, 11)]  # P1-P2, P2-P1, P2-noise, P1-noise
EVENT_ID = "000000001"


def _chain(nodes):
    return list(zip(nodes[:-1], nodes[1:]))


def gnn_stage_event(truth_dtype=torch.bool):
    """The toy event as acorn's edge-classifier stage writes it: un-prefixed keys (y, scores, truth_map, ...)."""
    r = torch.zeros(N_NODES)
    phi = torch.zeros(N_NODES)
    for nodes, radii, ph in (
        (P1_NODES, [32.0, 72.0, 116.0, 172.0, 260.0, 360.0, 500.0], 0.3),
        (P2_NODES, [33.0, 73.0, 117.0, 173.0, 261.0], -1.2),
        (NOISE_NODES, [300.0], 2.0),
    ):
        r[nodes] = torch.tensor(radii)
        phi[nodes] = ph
    z = 0.5 * r
    track_pairs = _chain(P1_NODES) + _chain(P2_NODES)
    track_edges = torch.tensor(track_pairs, dtype=torch.long).T
    track_pid = torch.tensor([P1] * (len(P1_NODES) - 1) + [P2] * (len(P2_NODES) - 1))
    edge_index = torch.tensor(track_pairs + FAKE_EDGES, dtype=torch.long).T
    # edge truth exactly as acorn's graph builders compute it (py_module_map / pymmg_graph_builder)
    y, truth_map = graph_intersection(edge_index, track_edges, return_y_pred=True, return_truth_to_pred=True)
    assert int(y.sum()) == track_edges.shape[1] and bool((truth_map >= 0).all())
    return Data(
        x=r * torch.cos(phi),
        r=r,
        phi=phi,
        z=z,
        hit_id=torch.arange(N_NODES),
        edge_index=edge_index,
        y=y.to(truth_dtype),
        # every edge scores 0.9: a score-based CC would also give one component, so only the truth path can pass
        scores=torch.full((edge_index.shape[1],), 0.9),
        truth_map=truth_map,
        track_edges=track_edges,
        particle_id=track_pid,
        pt=torch.full((track_edges.shape[1],), 2000.0),
        event_id=[EVENT_ID],  # collated form; read identically at the pin and with patch 0005
        config=[],
    )


def write_gnn_output(root, event):
    os.makedirs(os.path.join(root, "testset"), exist_ok=True)
    save_pyg(event, os.path.join(root, "testset", f"event{EVENT_ID}.pyg"))


def cc_config(tmp_path, **extra):
    cfg = {
        "input_dir": str(tmp_path / "gnn"),
        "stage_dir": str(tmp_path / "cc"),
        "data_split": [0, 0, 1],
        "on_true_graph_edges": True,
        "score_cut": 0.5,
        "max_workers": 1,
    }
    cfg.update(extra)
    return cfg


def assert_true_components(labels):
    labels = [int(v) for v in labels]
    p1 = {labels[i] for i in P1_NODES}
    p2 = {labels[i] for i in P2_NODES}
    assert len(p1) == 1 and len(p2) == 1, f"a particle is split: labels={labels}"
    assert p1 != p2, f"particles {P1} and {P2} merged into one component: labels={labels}"
    assert min(p1 | p2) >= 0, labels
    assert all(labels[i] == -1 for i in NOISE_NODES), f"noise hit kept in a track: labels={labels}"
    assert len({v for v in labels if v >= 0}) == 2, labels


@pytest.mark.parametrize("truth_dtype", [torch.bool, torch.int64, torch.float32], ids=["bool", "int64", "float32"])
def test_build_event_selects_edge_y_after_dataset_rename(tmp_path, truth_dtype):
    """_build_event on an event read through acorn's own track-building GraphDataset (which renames y -> edge_y)."""
    write_gnn_output(str(tmp_path / "gnn"), gnn_stage_event(truth_dtype))
    cfg = cc_config(tmp_path)
    dataset = GraphDataset(cfg["input_dir"], "testset", 1, "predict", cfg)
    graph = dataset[0]
    assert "edge_y" in graph and graph.y is None  # the rename that makes the pin read None
    graph = ConnectedComponents(cfg)._build_event(graph, on_true_graph_edges=True)
    assert_true_components(graph.hit_track_labels)


def test_stage_infer_writes_true_tracks(tmp_path):
    """Full ConnectedComponents.infer (GraphDataset -> _build_and_save -> save_tracks/save_graph)."""
    write_gnn_output(str(tmp_path / "gnn"), gnn_stage_event())
    cfg = cc_config(tmp_path)
    ConnectedComponents.infer(cfg)

    track_files = glob.glob(os.path.join(cfg["stage_dir"], "testset_tracks", "*.txt"))
    assert len(track_files) == 1
    with open(track_files[0]) as f:
        tracks = [frozenset(int(h) for h in line.split()) for line in f if line.strip()]
    assert sorted(tracks, key=min) == [frozenset(P1_NODES), frozenset(P2_NODES)], tracks

    graph_files = glob.glob(os.path.join(cfg["stage_dir"], "testset", "*.pyg.gz"))
    assert len(graph_files) == 1
    saved = load_pyg(graph_files[0][: -len(".gz")], weights_only=False)
    assert_true_components(saved.hit_track_labels)


def test_stage_perfect_classifier_efficiency_is_one(tmp_path):
    """The 'perfect classifier' ceiling through acorn's own evaluation (setup('test') + tracking_efficiency)."""
    write_gnn_output(str(tmp_path / "gnn"), gnn_stage_event())
    cfg = cc_config(
        tmp_path,
        dataset="testset",
        target_tracks={"pt": [1000, 1e12]},
        matching_style="ATLAS",
        matching_fraction=0.5,
        min_track_length=5,
    )
    ConnectedComponents.infer(cfg)
    builder = ConnectedComponents(cfg)
    builder.setup(stage="test")
    result = builder.tracking_efficiency({"variables": {}}, cfg)
    # pin: one 13-hit component; particle 11 matched at purity 7/13, so efficiency 0.5 and 1 track
    assert result["num_particles"] == 2, result
    assert result["efficiency"] == pytest.approx(1.0), result
    assert result["num_tracks"] == 2, result
    assert result["num_matched_tracks"] == 2, result
    assert result["fake_rate"] == pytest.approx(0.0), result


def test_missing_edge_truth_raises_instead_of_keeping_all_edges():
    """Without any edge truth the patched code fails loudly; the pin silently returned one giant component."""
    graph = add_variable_name_prefix_in_pyg(gnn_stage_event())
    del graph["edge_y"]
    assert "edge_y" not in graph and graph.y is None
    with pytest.raises(Exception):
        ConnectedComponents({"on_true_graph_edges": True})._build_event(graph, on_true_graph_edges=True)
