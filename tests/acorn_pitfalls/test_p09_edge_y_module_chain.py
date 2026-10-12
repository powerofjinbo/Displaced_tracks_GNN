"""p09 (brief §4.5): acorn's edge_y is module-wise consecutive truth. Not patched: characterisation tests.

How the truth is built (no patch touches any of these functions):
1. Truth graph ``track_edges``: EventReader._build_true_tracks (stages/data_reading/data_reading_stage.py) sorts each
   particle's hits by 3D distance from its production vertex, groups them by (particle, module) and links every hit
   of one module group to every hit of the NEXT module group. It is a chain over the modules the particle actually
   hit, not over layers. (Our npz ``layerless_true_edges``, mapped to track_edges by row index, are the same kind of
   truth: per RESULTS.md M1 each particle is one chain, in/out degree <= 1, consecutive hits on different modules.)
2. ``edge_y`` / ``track_to_edge_map`` (truth_map): MetricLearning.get_truth orients every candidate edge outward in
   hit_R = r^2 + z^2 (track_building/utils.py::rearrange_by_distance) and intersects it with track_edges
   (graph_construction/models/utils.py::graph_intersection). edge_y[e] is True iff e equals a track edge (in either
   direction only when undirected=True); truth_map[t] is the index of the candidate edge equal to track edge t, else
   -1. The module-map builders call graph_intersection directly (directed, no re-orientation); the GNN stage keeps
   edge_y and only recomputes truth_map (save_edge_scores).
Consequences shown here: a same-particle pair that skips a hit is a negative; a pair spanning a layer where the
particle left no hit is a positive; an inward-pointing track edge never matches when undirected=False. The last two
tests show the "perfect classifier" CC split at a skip edge and a skip-truth relabelling recipe for our own code
(augment the truth graph and every track_* tensor; never just overwrite edge_y).
"""

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from acorn.stages.data_reading.data_reading_stage import EventReader
from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.stages.track_building import utils as tb_utils
from acorn.stages.track_building.models.connected_components import ConnectedComponents

INF = float("inf")
WEIGHTING = [  # natural weighting with a track-like pt condition (TRACK_LIKE conditions go through truth_map)
    {"weight": 1.0, "conditions": {"edge_y": False}},
    {"weight": 1.0, "conditions": {"edge_y": True, "track_particle_pt": [1000.0, INF]}},
]


# ------------------------------------------------------------------- 1. the truth graph is a module chain

def reader_hits():
    """Particle 7 from the origin along +x: layers 2, 4, 4 (overlap module), 6, 6 (same module), 8.
    Particle 9: layers 2 and 8 only (no hit in layers 4 and 6). Hit 6 is noise."""
    rows = [  # hit_id, particle_id, x [mm], volume, layer, module
        (0, 7, 30.0, 8, 2, 100),
        (1, 7, 70.0, 8, 4, 200),
        (2, 7, 72.0, 8, 4, 201),
        (3, 7, 115.0, 8, 6, 300),
        (4, 7, 116.0, 8, 6, 300),
        (5, 7, 170.0, 8, 8, 400),
        (6, 0, 50.0, 8, 2, 101),
        (7, 9, 31.0, 8, 2, 110),
        (8, 9, 171.0, 8, 8, 410),
    ]
    hits = pd.DataFrame(
        rows, columns=["hit_id", "hit_particle_id", "hit_x", "hit_volume_id", "hit_layer_id", "hit_module_id"]
    )
    hits["hit_y"] = 0.0
    hits["hit_z"] = 0.0
    for c in ("particle_vx", "particle_vy", "particle_vz"):
        hits[c] = np.where(hits.hit_particle_id == 0, np.nan, 0.0)
    return hits


def test_reader_truth_is_a_chain_over_modules_not_layers():
    reader = EventReader(
        {
            "module_columns": ["volume_id", "layer_id", "module_id"],
            "feature_sets": {"track_features": ["track_particle_id"]},
        }
    )
    track_edges, _, hits = reader._build_true_tracks(reader_hits(), "1")
    assert hits.hit_id.tolist() == list(range(9))  # node index == hit id in this toy
    edges = {tuple(e) for e in np.asarray(track_edges).T.tolist()}
    assert edges == {(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (7, 8)}
    assert (1, 2) in edges  # same layer, overlapping modules: TRUE
    assert (0, 2) not in edges  # adjacent layers 2 -> 4, but hit 1 lies between: negative
    assert (1, 3) not in edges and (1, 4) not in edges  # layers 4 -> 6 with hit 2 between: negative
    assert (0, 3) not in edges  # skips layer 4: negative
    assert (3, 4) not in edges and (4, 3) not in edges  # two hits on one module are never linked
    assert (7, 8) in edges  # spans layers 4 and 6 where particle 9 has no hit: TRUE


# ------------------------------------------------------------------- 2. edge_y from MetricLearning.get_truth

def metric_learning(weighting=None):
    """Minimal MetricLearning (its embedding is never used: get_truth and get_weights only read the event)."""
    hparams = {"node_features": ["hit_r"], "emb_hidden": 8, "nb_layer": 1, "emb_dim": 4, "activation": "Tanh"}
    if weighting is not None:
        hparams["weighting"] = weighting
    torch.manual_seed(0)
    return MetricLearning(hparams)


# particle 7: nodes 0 -> 1 -> 2 -> 3 (consecutive hits); particle 9: nodes 4 -> 5
R = [30.0, 70.0, 115.0, 170.0, 31.0, 171.0]
CHAIN = [(0, 1), (1, 2), (2, 3), (4, 5)]
PIDS = [7, 7, 7, 9]


def event(track_pairs=CHAIN, pids=PIDS, pts=None):
    r = torch.tensor(R)
    graph = Data(
        hit_r=r,
        hit_z=0.4 * r,
        hit_x=r.clone(),
        hit_id=torch.arange(len(R)),
        track_edges=torch.tensor(track_pairs, dtype=torch.long).T,
        track_particle_id=torch.tensor(pids),
        track_particle_pt=torch.tensor(pts if pts is not None else [2000.0] * len(pids)),
    )
    graph.num_nodes = len(R)
    return graph


def get_truth(graph, pairs, undirected, model=None):
    model = model or metric_learning()
    pred = torch.tensor(pairs, dtype=torch.long).T
    pred_edges, edge_y, truth_map, _ = model.get_truth(graph, pred, undirected)
    labels = {tuple(e): bool(t) for e, t in zip(pred_edges.T.tolist(), edge_y.tolist())}
    return pred_edges, edge_y, truth_map, labels


@pytest.mark.parametrize("undirected", [False, True], ids=["directed", "undirected"])
def test_edge_y_is_true_only_for_consecutive_same_particle_pairs(undirected):
    candidates = [(1, 0), (1, 2), (2, 3), (4, 5), (0, 2), (3, 1), (0, 3), (0, 4)]  # (1, 0), (3, 1) point inward
    graph = event()
    pred_edges, _, truth_map, labels = get_truth(graph, candidates, undirected)
    assert labels == {
        (0, 1): True,  # consecutive (candidate given inward, re-oriented outward by rearrange_by_distance)
        (1, 2): True,
        (2, 3): True,
        (4, 5): True,
        (0, 2): False,  # same particle, skips hit 1
        (1, 3): False,  # same particle, skips hit 2
        (0, 3): False,  # same particle, skips two hits
        (0, 4): False,  # different particles
    }
    # truth_map[t] points at the candidate edge equal to track edge t
    assert (truth_map >= 0).all()
    assert torch.equal(pred_edges[:, truth_map], graph.track_edges)


def test_directed_matching_misses_inward_track_edges():
    """Characterisation: with undirected=False (default of shared_evaluation, i.e. val/test/inference graphs, and of
    the GNN stage's truth_map) a track edge pointing inward in r^2+z^2 never matches, because only the candidate
    edges are re-oriented. With undirected=True (default only for training_step) it does."""
    inward = [(1, 0), (1, 2), (3, 2), (4, 5)]  # same pairs as CHAIN, two of them pointing inward
    candidates = [(0, 1), (1, 2), (2, 3), (4, 5)]
    _, _, truth_map, labels = get_truth(event(track_pairs=inward), candidates, undirected=False)
    assert labels == {(0, 1): False, (1, 2): True, (2, 3): False, (4, 5): True}
    assert truth_map.tolist()[0] == -1 and truth_map.tolist()[2] == -1
    _, _, truth_map, labels = get_truth(event(track_pairs=inward), candidates, undirected=True)
    assert all(labels.values()) and (truth_map >= 0).all()


# ------------------------------------------------------------- 3. consequence for the perfect-classifier ceiling

SKIP_GRAPH = [(0, 2), (2, 3), (4, 5), (3, 5)]  # hit 1 has no candidate edge; 0 -> 2 skips it; (3, 5) is fake


def perfect_cc(graph, pred_edges, edge_y):
    """CC with perfect scores (score path, unaffected by p05) on acorn's edge truth."""
    graph = graph.clone()
    graph.edge_index = pred_edges
    graph.edge_scores = edge_y.float()
    graph = ConnectedComponents({})._build_event(graph, score_cut=0.5)
    return graph.hit_track_labels.tolist()


def test_perfect_classifier_cc_splits_a_particle_at_a_skip_edge():
    graph = event()
    pred_edges, edge_y, _, labels = get_truth(graph, SKIP_GRAPH, undirected=True)
    assert labels[(0, 2)] is False and labels[(2, 3)] is True
    hit_labels = perfect_cc(graph, pred_edges, edge_y)
    assert hit_labels[0] == -1 and hit_labels[1] == -1  # hit 0 is lost although the graph links it to hit 2
    assert hit_labels[2] == hit_labels[3] >= 0 and hit_labels[4] == hit_labels[5] >= 0
    assert hit_labels[2] != hit_labels[4]


# ------------------------------------------------------- 4. skip-truth relabelling recipe for our own code

def add_skip_truth(graph, max_skip):
    """Reference recipe (not acorn code): extend the chain truth graph with same-particle pairs that skip up to
    `max_skip` hits along the particle's chain, and extend every track_* tensor to match, so that acorn's
    graph_intersection, truth_map and TRACK_LIKE weighting treat skip pairs as ordinary true edges.
    Chain rank (k-th next hit of the particle), not layer number, defines a skip."""
    src, dst = graph.track_edges.tolist()
    assert len(set(src)) == len(src) and len(set(dst)) == len(dst), "expects chains (in/out degree <= 1)"
    nxt = dict(zip(src, dst))
    edge_of = {s: t for t, s in enumerate(src)}
    starts = [s for s in src if s not in set(dst)]
    pairs, from_edge, skip = list(zip(src, dst)), list(range(len(src))), [0] * len(src)
    for start in starts:
        chain = [start]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        for k in range(2, max_skip + 2):
            for i in range(len(chain) - k):
                pairs.append((chain[i], chain[i + k]))
                from_edge.append(edge_of[chain[i]])  # same particle: copy that particle's track features
                skip.append(k - 1)
    out = graph.clone()
    index = torch.tensor(from_edge)
    for key in list(out.keys()):
        if key.startswith("track_") and key != "track_edges" and torch.is_tensor(out[key]):
            out[key] = out[key][..., index]
    out.track_edges = torch.tensor(pairs, dtype=torch.long).T
    out.track_skip = torch.tensor(skip)
    return out


def weights_for(model, graph, pred_edges, edge_y, truth_map):
    graph = graph.clone()
    graph.edge_index, graph.edge_y, graph.track_to_edge_map = pred_edges, edge_y, truth_map
    return model.get_weights(graph)


def test_skip_truth_must_augment_track_edges_not_just_edge_y():
    model = metric_learning(WEIGHTING)
    chain = event()
    pred_edges, edge_y, truth_map, _ = get_truth(chain, SKIP_GRAPH, undirected=True, model=model)
    skip_e = pred_edges.T.tolist().index([0, 2])

    # acorn as is: the skip pair is a weighted NEGATIVE (pushed apart in metric learning)
    w = weights_for(model, chain, pred_edges, edge_y, truth_map)
    assert not edge_y[skip_e] and w[skip_e] == 1.0

    # naive relabel (edge_y only): the TRACK_LIKE pt condition maps through truth_map, which has no entry for the
    # skip pair, so the new positive gets the default positive weight 0, i.e. it is masked from the loss
    naive_y = edge_y.clone()
    naive_y[skip_e] = True
    w = weights_for(model, chain, pred_edges, naive_y, truth_map)
    assert w[skip_e] == 0.0

    # recipe: augmented truth graph + extended track_* tensors -> positive with weight 1 and a truth_map entry
    aug = add_skip_truth(chain, max_skip=1)
    assert {tuple(e) for e in aug.track_edges.T.tolist()} == set(CHAIN) | {(0, 2), (1, 3)}
    assert aug.track_particle_id.tolist() == PIDS + [7, 7] and aug.track_skip.tolist() == [0, 0, 0, 0, 1, 1]
    a_pred, a_y, a_map, a_labels = get_truth(aug, SKIP_GRAPH, undirected=True, model=model)
    assert a_labels == {(0, 2): True, (2, 3): True, (4, 5): True, (3, 5): False}
    assert torch.equal(a_pred, pred_edges)
    w = weights_for(model, aug, a_pred, a_y, a_map)
    assert w.tolist() == [1.0, 1.0, 1.0, 1.0]
    mapped = {tuple(e): int(m) for e, m in zip(aug.track_edges.T.tolist(), a_map.tolist())}
    assert mapped[(0, 2)] == skip_e and mapped[(1, 3)] == -1 and mapped[(0, 1)] == -1

    # the perfect-classifier CC now keeps hit 0 with its particle
    hit_labels = perfect_cc(aug, a_pred, a_y)
    assert hit_labels[0] == hit_labels[2] == hit_labels[3] >= 0 and hit_labels[1] == -1

    # acorn's track evaluation reads particle hit lists from track_edges: the augmentation does not change them
    def hit_particle_pairs(graph):
        graph.hit_track_labels = torch.tensor(hit_labels)
        df = tb_utils.load_reconstruction_df(graph)
        return {(int(h), int(p)) for h, p in zip(df.hit_id, df.particle_id)}

    assert hit_particle_pairs(aug) == hit_particle_pairs(chain) == {(0, 7), (1, 7), (2, 7), (3, 7), (4, 9), (5, 9)}
