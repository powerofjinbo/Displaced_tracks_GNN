"""p06 (brief §4.5): load_reconstruction_df treats track_edges node indices as hit ids.

acorn/stages/track_building/utils.py::load_reconstruction_df builds the (hit_id, track_id) table from graph.hit_id but,
at the pin, the (hit_id, particle_id) table from ``graph.track_edges.reshape(-1)``, i.e. from NODE INDICES. The merge
on "hit_id" is right only if hit_id == arange(num_nodes). A node-level hard cut (handle_hard_node_cuts_pyg) keeps the
original hit ids but remaps track_edges to the new node indices, and our npz hid is a permutation in geometry order,
not row order; in both cases every particle is attached to the wrong hits. Patch 0001 maps through
``hit_id[graph.track_edges.reshape(-1)]``.

Tests of the correct behaviour fail at the pin (ACORN_PIN_SRC), except the arange control and the numpy
characterisation test, which pass in both trees.
"""

import glob
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from acorn.stages.graph_construction.graph_construction_stage import EventDataset
from acorn.stages.graph_construction.models.utils import graph_intersection
from acorn.stages.track_building import utils as tb_utils
from acorn.stages.track_building.models.connected_components import ConnectedComponents
from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg, save_pyg

EVENT_ID = "000000001"

# ---------------------------------------------------------------- unit level: load_reconstruction_df

UNIT_PARTICLES = {11: [0, 2, 4, 6], 22: [1, 3, 5, 7, 9]}  # node indices, chain order; node 8 is noise
UNIT_LABELS = [0, 1, 0, 1, 0, 1, 0, 1, -1, 1]  # hit_track_labels per node (a perfect reconstruction)
HIT_ID_CONVENTIONS = {
    "after_node_hard_cut": [0, 2, 4, 6, 8, 9, 11, 13, 15, 16],  # surviving ids keep their gaps
    "npz_hid_permutation": [5, 0, 7, 2, 9, 4, 1, 8, 3, 6],  # hid order != row order (no fixed point)
    "offset_ids": [1000 + i for i in range(10)],
    "arange_control": list(range(10)),  # the only case the pin gets right
}


def unit_graph(hit_id, pid_container=torch.tensor):
    pairs, pids = [], []
    for pid, nodes in UNIT_PARTICLES.items():
        pairs += list(zip(nodes[:-1], nodes[1:]))
        pids += [pid] * (len(nodes) - 1)
    graph = Data(
        hit_id=torch.tensor(hit_id),
        track_edges=torch.tensor(pairs).T,
        track_particle_id=pid_container(pids),
        hit_track_labels=torch.tensor(UNIT_LABELS),
    )
    graph.num_nodes = len(hit_id)
    return graph


def expected_rows(hit_id):
    """{hit_id: (track_id, particle_id)} for the unit graph."""
    pid_of_node = {n: pid for pid, nodes in UNIT_PARTICLES.items() for n in nodes}
    return {hit_id[n]: (UNIT_LABELS[n], pid_of_node.get(n, 0)) for n in range(len(hit_id))}


def as_rows(df):
    return {int(h): (int(t), int(p)) for h, t, p in zip(df.hit_id, df.track_id, df.particle_id)}


@pytest.mark.parametrize("convention", list(HIT_ID_CONVENTIONS))
def test_load_reconstruction_df_attaches_particles_by_hit_id(convention):
    hit_id = HIT_ID_CONVENTIONS[convention]
    df = tb_utils.load_reconstruction_df(unit_graph(hit_id))
    # one row per hit (no shared hits in the toy), no phantom hit ids taken from node indices
    assert sorted(df.hit_id.astype(int)) == sorted(hit_id), df.sort_values("hit_id").to_string()
    assert len(df) == len(hit_id), df.sort_values("hit_id").to_string()
    assert as_rows(df) == expected_rows(hit_id), df.sort_values("hit_id").to_string()


def test_numpy_track_particle_id_is_mispaired_characterisation():
    """Characterisation (both trees): pids are paired with track_edges.reshape(-1) = [src..., dst...] via
    ``track_particle_id.repeat(2)``. That tiles a torch tensor but interleaves a numpy array, so a converter that
    stores numpy arrays in the PyG event silently attaches hits to the wrong particle."""
    hit_id = HIT_ID_CONVENTIONS["arange_control"]
    df = tb_utils.load_reconstruction_df(unit_graph(hit_id, pid_container=np.array))
    rows = df.groupby("hit_id").particle_id.apply(lambda s: sorted(int(p) for p in s)).to_dict()
    # truth: nodes 0,2,4,6 -> 11 and 1,3,5,7,9 -> 22. numpy interleaving gives node 1 only pid 11, node 6 only
    # pid 22, and nodes 2-5 both pids
    assert rows[1] == [11] and rows[6] == [22], rows
    assert all(rows[n] == [11, 22] for n in (2, 3, 4, 5)), rows
    assert rows[0] == [11] and rows[7] == [22] and rows[9] == [22] and rows[8] == [0], rows
    # the same graph with a torch tensor is paired correctly
    assert as_rows(tb_utils.load_reconstruction_df(unit_graph(hit_id))) == expected_rows(hit_id)


# ------------------------------------------------- stage level: real node-level hard cut -> CC -> evaluation

FULL_PARTICLES = {  # pid: (node indices in the full event, pt [MeV], radii [mm], phi)
    101: ([0, 2, 4, 6, 8], 2000.0, [32.0, 72.0, 116.0, 172.0, 260.0], 0.3),
    202: ([1, 3, 5, 7], 500.0, [33.0, 73.0, 117.0, 173.0], -1.2),  # removed by the pt hard cut
    303: ([9, 11, 13, 15, 16], 3000.0, [34.0, 74.0, 118.0, 174.0, 262.0], 2.0),
}
FULL_NOISE = {10: 50.0, 12: 90.0, 14: 140.0}  # node: r; noise is removed by any node-level hard cut
N_FULL = 17


def feature_store_event():
    """The full event as acorn's data reader writes it (un-prefixed keys, hit_id = arange like TrackML)."""
    r = torch.zeros(N_FULL)
    phi = torch.zeros(N_FULL)
    pairs, pids, pts = [], [], []
    for pid, (nodes, pt, radii, ph) in FULL_PARTICLES.items():
        r[nodes] = torch.tensor(radii)
        phi[nodes] = ph
        pairs += list(zip(nodes[:-1], nodes[1:]))
        pids += [pid] * (len(nodes) - 1)
        pts += [pt] * (len(nodes) - 1)
    for node, rad in FULL_NOISE.items():
        r[node], phi[node] = rad, 1.0
    return Data(
        x=r * torch.cos(phi),
        r=r,
        phi=phi,
        z=0.5 * r,
        hit_id=torch.arange(N_FULL),
        track_edges=torch.tensor(pairs).T,
        particle_id=torch.tensor(pids),
        pt=torch.tensor(pts),
        event_id=[EVENT_ID],
        config=[],
    )


def hard_cut_event(tmp_path):
    """Read the event through acorn's graph-construction EventDataset with a node-level hard cut."""
    fs = tmp_path / "feature_store"
    os.makedirs(fs / "testset")
    save_pyg(feature_store_event(), str(fs / "testset" / f"event{EVENT_ID}.pyg"))
    dataset = EventDataset(
        str(fs), "testset", 1, use_csv=False, hparams={"hard_cuts": {"track_particle_pt": [1000.0, float("inf")]}}
    )
    return dataset[0]


def write_gnn_stage_output(event, root):
    """Add a candidate graph with acorn's edge truth and perfect scores, then save as the GNN stage does."""
    fakes = torch.tensor([[1, 3], [6, 8]])  # particle 101 -> particle 303, in node indices after the cut
    edge_index = torch.cat([event.track_edges, fakes], dim=1)
    y, truth_map = graph_intersection(edge_index, event.track_edges, return_y_pred=True, return_truth_to_pred=True)
    event.edge_index = edge_index
    event.edge_y = y
    event.track_to_edge_map = truth_map
    event.edge_scores = torch.where(y, 0.9, 0.1)
    os.makedirs(os.path.join(root, "testset"))
    save_pyg(remove_variable_name_prefix_in_pyg(event), os.path.join(root, "testset", f"event{EVENT_ID}.pyg"))


def test_hard_cut_leaves_hit_ids_that_are_not_node_indices(tmp_path):
    """Precondition of the pitfall, identical at the pin: hit_id keeps gaps, track_edges are new node indices."""
    event = hard_cut_event(tmp_path)
    surviving = sorted(FULL_PARTICLES[101][0] + FULL_PARTICLES[303][0])
    assert event.hit_id.tolist() == surviving
    assert int(event.num_nodes) == len(surviving)
    assert int(event.track_edges.max()) == len(surviving) - 1
    assert not torch.equal(event.hit_id, torch.arange(len(surviving)))


@pytest.mark.parametrize("mode", ["on_truth_particle_edges", "perfect_scores"])
def test_cc_efficiency_after_node_hard_cut_is_one(tmp_path, mode):
    event = hard_cut_event(tmp_path)
    write_gnn_stage_output(event, str(tmp_path / "gnn"))
    cfg = {
        "input_dir": str(tmp_path / "gnn"),
        "stage_dir": str(tmp_path / "cc"),
        "data_split": [0, 0, 1],
        "on_truth_particle_edges": mode == "on_truth_particle_edges",
        "score_cut": 0.5,
        "max_workers": 1,
        "dataset": "testset",
        "target_tracks": {"pt": [1000, 1e12]},
        "matching_style": "ATLAS",
        "matching_fraction": 0.5,
        "min_track_length": 5,
    }
    ConnectedComponents.infer(cfg)

    # the CC output itself is right in both trees (it maps labels through hit_id): two true tracks
    track_files = glob.glob(os.path.join(cfg["stage_dir"], "testset_tracks", "*.txt"))
    with open(track_files[0]) as f:
        tracks = sorted((sorted(int(h) for h in line.split()) for line in f if line.strip()), key=min)
    assert tracks == [FULL_PARTICLES[101][0], FULL_PARTICLES[303][0]]

    builder = ConnectedComponents(cfg)
    builder.setup(stage="test")
    result = builder.tracking_efficiency({"variables": {}}, cfg)
    # main project, TrackML: CC on the truth graph scored 0.000 instead of 1.000; this toy gives 0.5 at the pin
    assert result["num_particles"] == 2, result
    assert result["efficiency"] == pytest.approx(1.0), result
    assert result["num_matched_tracks"] == result["num_tracks"] == 2, result
    assert result["fake_rate"] == pytest.approx(0.0), result
