"""p01 (brief §4.5): weighting conditions can be dropped silently.

acorn/stages/graph_construction/utils.py::map_value_to_edges dispatches a condition mask on its LENGTH only:
E or E/2 -> edge-like, T or T/2 -> track-like, N -> node-like. At the pin 99d15341 every other length fell
through to (lines 172-175)

    else:
        # Unsure what shape value mask is, simply "ignore" it by returning a mask of all True
        ...
        return torch.ones_like(edges[0], dtype=torch.bool)

so a wrong-length condition (e.g. a per-particle prompt flag) was silently dropped and a prompt-only (PMASK)
weighting acted as natural training. Patch 0004 (patched lines 172-181) raises ValueError instead.

Live callers of that function (there is no other copy of it in acorn): get_weight_mask (lines 35-48) <-
build_signal_edges (20-32) <- MetricLearning / DoubleMetricLearning .append_signal_edges and .log_metrics, and the
module's own handle_weighting (51-115), which no stock stage imports. The loss weights of MetricLearning, the
edge classifier and track building come from acorn/utils/loading_utils.py::handle_weighting / get_weight_mask,
which map conditions by NAME prefix through mapping_utils.map_tensor_handler and have no all-True fallback (a
wrong length raises there), but can still drop or misapply a condition silently in other ways (test_char_*).

Tests without the `char` prefix assert the correct (patched) behaviour; the ones marked "fixed by 0004" FAIL
against ACORN_PIN_SRC, which demonstrates the bug. `test_char_*` tests pin down behaviour that no patch changes
(they pass against both copies): the remaining ways a condition can still be dropped or misapplied silently.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage, GraphDataset
from acorn.stages.graph_construction import utils as gcu
from acorn.stages.graph_construction.graph_construction_stage import EventDataset, GraphConstructionStage
from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.stages.track_building import utils as tb_utils
from acorn.stages.track_building.models import hgnn_matching_utils
from acorn.utils import loading_utils

# ---------------------------------------------------------------------------------------------------------------
# Toy event. Particle 1 (prompt): hits 0-3; particle 2 (displaced): hits 4-7; particle 3 (displaced): hits 8-10;
# hit 11 is noise. N = 12 hits, P = 3 particles, T = 8 module-wise consecutive true edges.
# Predicted graph: E = 7 edges, two of them fake ((3,7) and (0,11)). P = 3 coincides with none of E, E/2, T, T/2, N.
# ---------------------------------------------------------------------------------------------------------------
N_HITS = 12
TRACK_EDGES = torch.tensor([[0, 1, 2, 4, 5, 6, 8, 9], [1, 2, 3, 5, 6, 7, 9, 10]])
PROMPT_PER_TRACK_EDGE = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)  # length T: correct shape
PROMPT_PER_PARTICLE = torch.tensor([True, False, False])  # length P = 3: matches none of E, T, N
PRED_EDGES = torch.tensor([[0, 1, 4, 8, 3, 0, 2], [1, 2, 5, 9, 7, 11, 3]])
EDGE_Y = torch.tensor([1, 1, 1, 1, 0, 0, 1], dtype=torch.bool)
TRACK_TO_EDGE_MAP = torch.tensor([0, 1, 6, 2, -1, -1, 3, -1])  # true edge -> predicted edge (-1: not built)

# weight 3.0 on prompt true edges; true edges matching no rule keep weight 0 (masked); fakes default to 1.0
PMASK_WRONG = [{"weight": 3.0, "conditions": {"edge_y": True, "particle_prompt": True}}]
PMASK_RIGHT = [{"weight": 3.0, "conditions": {"edge_y": True, "track_particle_prompt": True}}]
PMASK_WEIGHTS = [3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 3.0]  # per PRED_EDGES column
NATURAL_WEIGHTS = [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0]  # what the pin returns for PMASK_WRONG
PROMPT_TRUE_EDGES = [[0, 1, 2], [1, 2, 3]]


def toy_event():
    r = torch.linspace(30.0, 1000.0, N_HITS)
    return Data(
        hit_id=torch.arange(N_HITS),
        hit_r=r,
        hit_z=torch.zeros(N_HITS),
        hit_R=r**2,
        track_edges=TRACK_EDGES.clone(),
        track_particle_id=torch.tensor([1, 1, 1, 2, 2, 2, 3, 3]),
        track_particle_pt=torch.tensor([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0, 1500.0, 1500.0]),
        track_particle_prompt=PROMPT_PER_TRACK_EDGE.clone(),
        particle_prompt=PROMPT_PER_PARTICLE.clone(),
        edge_index=PRED_EDGES.clone(),
        edge_y=EDGE_Y.clone(),
        track_to_edge_map=TRACK_TO_EDGE_MAP.clone(),
        num_nodes=N_HITS,
    )


def make_metric_learning(weighting):
    torch.manual_seed(0)
    hparams = {
        "node_features": ["hit_r", "hit_z"],
        "emb_hidden": 8,
        "nb_layer": 1,
        "emb_dim": 4,
        "activation": "Tanh",
        "margin": 0.1,
        "weighting": weighting,
    }
    return MetricLearning(hparams)


def edge_set(edges):
    return {tuple(e) for e in edges.t().tolist()}


def test_toy_event_is_consistent():
    ev = toy_event()
    true_pairs = edge_set(ev.track_edges)
    assert ev.edge_y.tolist() == [tuple(p) in true_pairs for p in ev.edge_index.t().tolist()]
    for t, e in enumerate(ev.track_to_edge_map.tolist()):
        if e >= 0:
            assert ev.edge_index[:, e].tolist() == ev.track_edges[:, t].tolist()
        else:
            assert tuple(ev.track_edges[:, t].tolist()) not in edge_set(ev.edge_index)


# ------------------------------------------- fixed by patch 0004 ------------------------------------------------


def test_map_value_to_edges_raises_on_unmatched_length():
    """Fixed by 0004. Pin: returns an all-True mask of length E (the condition is dropped)."""
    ev = toy_event()
    with pytest.raises(ValueError, match=r"length 3 matches none of E=7, T=8 \(or E/2, T/2\), N=12"):
        out = gcu.map_value_to_edges(ev, PROMPT_PER_PARTICLE, ev.edge_index, ev.track_edges, ev.track_to_edge_map)
        pytest.fail(f"condition silently dropped, mask = {out.tolist()}")


def test_get_weight_mask_raises_on_unmatched_length():
    """Fixed by 0004. Pin: the per-particle condition contributes an all-True factor."""
    ev = toy_event()
    with pytest.raises(ValueError, match="matches none of"):
        out = gcu.get_weight_mask(ev, ev.edge_index, {"particle_prompt": True}, ev.track_edges, ev.track_to_edge_map)
        pytest.fail(f"condition silently dropped, mask = {out.tolist()}")


def test_build_signal_edges_wrong_length_prompt_condition_raises():
    """Fixed by 0004. Pin: every true edge becomes a signal edge, i.e. PMASK silently becomes natural."""
    ev = toy_event()
    with pytest.raises(ValueError, match=r"length 3 matches none of"):
        signal = gcu.build_signal_edges(ev, PMASK_WRONG, ev.track_edges)
        pytest.fail(
            f"PMASK silently became natural: {signal.shape[1]} of {ev.track_edges.shape[1]} true edges are"
            " signal, 5 of them non-prompt"
        )


def test_graph_construction_handle_weighting_wrong_length_raises():
    """Fixed by 0004. Pin: weights == NATURAL_WEIGHTS (non-prompt true edges 2 and 3 get the PMASK weight)."""
    ev = toy_event()
    with pytest.raises(ValueError, match="matches none of"):
        w = gcu.handle_weighting(ev, PMASK_WRONG)
        pytest.fail(
            f"condition silently dropped, weights = {w.tolist()}"
            f" ({'natural' if w.tolist() == NATURAL_WEIGHTS else 'unexpected'}; PMASK is {PMASK_WEIGHTS})"
        )


def test_metric_learning_append_signal_edges_wrong_length_raises():
    """Fixed by 0004, through the real MetricLearning method used in training_step (get_training_edges)."""
    ml = make_metric_learning(PMASK_WRONG)
    ev = toy_event()
    with pytest.raises(ValueError, match="matches none of"):
        edges = ml.append_signal_edges(ev, torch.empty((2, 0), dtype=torch.long))
        pytest.fail(f"all {edges.shape[1]} true edges appended as signal training pairs")


# ------------------------------- correct-length conditions (unchanged by the patch) -----------------------------


def test_map_value_to_edges_valid_lengths():
    ev = toy_event()
    e, t, tm = ev.edge_index, ev.track_edges, ev.track_to_edge_map
    # E: identity
    m_e = torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    assert torch.equal(gcu.map_value_to_edges(ev, m_e, e, t, tm), m_e)
    # E/2 on a doubled (undirected) graph: the mask is repeated for the flipped half
    e2 = torch.cat([e, e.flip(0)], dim=1)
    assert gcu.map_value_to_edges(ev, m_e, e2, t, tm).tolist() == m_e.tolist() * 2
    # T: through the truth map; true edges that were not built are lost, fakes are False
    assert gcu.map_value_to_edges(ev, PROMPT_PER_TRACK_EDGE, e, t, tm).tolist() == [
        True, True, False, False, False, False, True,
    ]
    # T/2: track mask per directed true edge, truth graph and map doubled
    t2 = torch.cat([t, t.flip(0)], dim=1)
    tm2 = torch.cat([tm, torch.full_like(tm, -1)])
    assert gcu.map_value_to_edges(ev, PROMPT_PER_TRACK_EDGE, e, t2, tm2).tolist() == [
        True, True, False, False, False, False, True,
    ]
    # N: an edge passes if EITHER endpoint passes (any over the two nodes)
    node_mask = torch.zeros(N_HITS, dtype=torch.bool)
    node_mask[[3, 11]] = True
    assert gcu.map_value_to_edges(ev, node_mask, e, t, tm).tolist() == [
        False, False, False, False, True, True, True,
    ]


def test_pmask_with_track_length_condition_is_prompt_only():
    """The intended PMASK behaviour with a length-T condition, in all graph_construction helpers."""
    ev = toy_event()
    assert gcu.build_signal_edges(ev, PMASK_RIGHT, ev.track_edges).tolist() == PROMPT_TRUE_EDGES
    assert gcu.handle_weighting(ev, PMASK_RIGHT).tolist() == PMASK_WEIGHTS
    # the loading_utils copy (loss weights of every stock stage) gives the same weights
    assert loading_utils.handle_weighting(ev, PMASK_RIGHT).tolist() == PMASK_WEIGHTS


def test_metric_learning_pmask_signal_edges_and_loss_weights():
    """Real MetricLearning methods: append_signal_edges (graph_construction copy) and get_truth + get_weights
    (loading_utils copy) agree on the prompt-only selection."""
    ml = make_metric_learning(PMASK_RIGHT)
    ev = toy_event()
    assert ml.append_signal_edges(ev, torch.empty((2, 0), dtype=torch.long)).tolist() == PROMPT_TRUE_EDGES
    pred = torch.tensor([[0, 1, 4, 8, 3, 0, 2, 6], [1, 2, 5, 9, 7, 11, 3, 7]])
    ev.edge_index, ev.edge_y, ev.track_to_edge_map, _ = ml.get_truth(ev, pred, False)
    assert ev.edge_index.tolist() == [[0, 0, 1, 2, 3, 4, 6, 8], [1, 11, 2, 3, 7, 5, 7, 9]]
    w = ml.get_weights(ev)
    assert w.tolist() == [3.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    positive_true = edge_set(ev.edge_index[:, (w > 0) & ev.edge_y])
    assert positive_true == edge_set(torch.tensor(PROMPT_TRUE_EDGES))


# --------------------------------- characterisation: behaviour no patch changes --------------------------------


@pytest.mark.parametrize(
    "key, value, exc, match",
    [
        ("particle_prompt", PROMPT_PER_PARTICLE, ValueError, "OTHER to VariableType.EDGE_LIKE not supported"),
        ("track_particle_prompt_pp", PROMPT_PER_PARTICLE, IndexError, "shape of the mask"),
        ("hit_prompt", torch.zeros(N_HITS, dtype=torch.bool), IndexError, "too many indices"),
    ],
    ids=["other_named_per_particle", "track_named_per_particle", "hit_named_per_hit"],
)
def test_char_loss_weight_path_raises_on_wrong_length_or_node_condition(key, value, exc, match):
    """MetricLearning.get_weights -> loading_utils.handle_weighting dispatches by NAME (map_tensor_handler):
    OTHER-named keys are unsupported, a track_-named key of length != T fails the boolean index, and ANY hit_
    (node-like) condition crashes (map_nodes_to_edges with aggr=None returns a (2, E) mask). So in stock acorn
    the pin's all-True fallback in build_signal_edges was followed by a loud failure one call later; it is
    silent end-to-end only where graph_construction.utils.handle_weighting/get_weight_mask are called directly."""
    ev = toy_event()
    ev[key] = value
    ml = make_metric_learning([{"weight": 3.0, "conditions": {"edge_y": True, key: True}}])
    with pytest.raises(exc, match=match):
        ml.get_weights(ev)


def test_char_length_coincidence_is_silently_misapplied():
    """Residual risk after 0004: dispatch is by length only. 3 particles x 3 hits: T = 6 = 2P, so a
    per-particle mask is taken as an (E/2) mask of the true-edge list, doubled, and applied to the wrong edges."""
    te = torch.tensor([[0, 1, 3, 4, 6, 7], [1, 2, 4, 5, 7, 8]])
    ev = Data(hit_id=torch.arange(9), track_edges=te, particle_prompt=torch.tensor([True, False, False]), num_nodes=9)
    signal = gcu.build_signal_edges(ev, [{"weight": 1.0, "conditions": {"edge_y": True, "particle_prompt": True}}], te)
    assert signal.tolist() == [[0, 4], [1, 5]]  # correct would be [[0, 1], [1, 2]] (particle A only)


def test_char_node_condition_any_endpoint_in_signal_edges():
    """A hit_ condition in build_signal_edges selects a true edge if EITHER endpoint passes."""
    ev = toy_event()  # hit_r = 30, 118.2, 206.4, ...
    cfg = [{"weight": 1.0, "conditions": {"edge_y": True, "hit_r": [0.0, 120.0]}}]
    assert gcu.build_signal_edges(ev, cfg, ev.track_edges).tolist() == [[0, 1], [1, 2]]


def test_char_integer_edge_y_silently_corrupts_weights():
    """loading_utils.get_weight_mask starts from torch.ones_like(event.edge_y): with an int64 edge_y the mask
    is int64 and `weights[mask] = w` becomes integer (fancy) indexing of positions 0 and 1. Through the real
    GraphDataset.construct_weighting (edge classifier; track building is identical). float edge_y raises."""
    stub = SimpleNamespace(hparams={"weighting": PMASK_RIGHT})
    ev_bool = toy_event()
    assert GraphDataset.construct_weighting(stub, ev_bool).edge_weights.tolist() == PMASK_WEIGHTS
    ev_long = toy_event()
    ev_long.edge_y = EDGE_Y.long()
    assert GraphDataset.construct_weighting(stub, ev_long).edge_weights.tolist() == [
        3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 0.0,  # prompt true edge 6 lost its weight, silently
    ]
    ev_float = toy_event()
    ev_float.edge_y = EDGE_Y.float()
    with pytest.raises(IndexError, match="tensors used as indices"):
        GraphDataset.construct_weighting(stub, ev_float)


def test_char_same_type_condition_skips_length_check():
    """mapping_utils.map_tensor_handler returns the input unchecked when input and output types agree, so an
    edge_ condition of length 1 broadcasts over all edges (here: the rule matches no edge at all)."""
    ev = toy_event()
    ev.edge_flag = torch.tensor([False])
    w = loading_utils.handle_weighting(ev, [{"weight": 3.0, "conditions": {"edge_y": True, "edge_flag": True}}])
    assert w.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_char_is_not_conditions_are_silently_ignored():
    """["is_not", v] evaluates `tensor != ["is_not", v]` -> Python True in both get_condition_lambda copies
    (p02), and several callers multiply it in without complaint: the condition is silently dropped."""
    # edge classifier / track building hard cuts (loading_utils.handle_hard_cuts): no track is removed
    ev = toy_event()
    stub = SimpleNamespace(hparams={"hard_cuts": {"track_particle_pt": ["is_not", 500.0]}})
    GraphDataset.apply_hard_cuts(stub, ev)
    assert ev.track_edges.tolist() == TRACK_EDGES.tolist()  # pt == 500 tracks should have gone
    # graph-construction (metric learning) node hard cuts on a hit_ key: no hit is removed
    ev = toy_event()
    stub = SimpleNamespace(hparams={"hard_cuts": {"hit_r": ["is_not", 30.0]}})
    ev = EventDataset.apply_hard_cuts(stub, ev)
    assert ev.hit_id.tolist() == list(range(N_HITS))  # hit 0 has r == 30.0 and should have gone
    # loss weights with an edge_ key: the rule matches every edge
    ev = toy_event()
    w = loading_utils.handle_weighting(ev, [{"weight": 3.0, "conditions": {"edge_y": ["is_not", 1]}}])
    assert w.tolist() == [3.0] * 7
    # evaluation targets of the edge classifier: every track stays a target
    ev = toy_event()
    ev.track_to_passing_edge_map = ev.track_to_edge_map.clone()
    EdgeClassifierStage.apply_target_conditions(
        SimpleNamespace(device="cpu"), ev, {"track_particle_pt": ["is_not", 500.0]}
    )
    assert ev.track_target_mask.tolist() == [True] * 8
    # ... while ["is", v] silently selects nothing
    EdgeClassifierStage.apply_target_conditions(SimpleNamespace(device="cpu"), ev, {"track_particle_pt": ["is", 500.0]})
    assert ev.track_target_mask.tolist() == [False] * 8
    # HGNN truth matching target selection: every particle kept
    ev = toy_event()
    ev.particle_id = ev.track_particle_id
    for name in ["eta", "radius", "nhits", "pdgId", "primary"]:
        ev[f"track_particle_{name}"] = torch.zeros(8)
    _, info = hgnn_matching_utils.build_truth_bgraph(ev, {}, {"track_particle_pt": ["is_not", 500.0]})
    assert info["original_pid"].tolist() == [1, 2, 3]  # particle 2 (pt == 500) should have gone


def test_char_is_not_conditions_fail_loudly_elsewhere():
    ev = toy_event()
    cfg = [{"weight": 1.0, "conditions": {"edge_y": True, "track_particle_pt": ["is_not", 500.0]}}]
    with pytest.raises(AttributeError, match="'bool' object has no attribute 'shape'"):
        gcu.build_signal_edges(ev, cfg, ev.track_edges)
    with pytest.raises(AttributeError, match="'bool' object has no attribute 'to'"):
        GraphConstructionStage.apply_target_conditions(
            SimpleNamespace(device="cpu"), ev, {"track_particle_pt": ["is_not", 500.0]}
        )


def test_char_track_building_fiducial_selection_ignores_false():
    """track_building/utils.py::apply_fiducial_sel (TrackBuildingStage.tracking_efficiency target_tracks) applies
    a scalar as `df[key] >= value`: `primary: False` keeps everything, and `nhits: 5` means >= 5, not == 5."""
    df = pd.DataFrame({"track_particle_primary": [True, False, True, False], "nhits": [3, 5, 7, 5]})
    sel = tb_utils.apply_fiducial_sel(df.copy(), {"track_particle_primary": False})
    assert sel["is_reconstructable"].tolist() == [True, True, True, True]
    sel = tb_utils.apply_fiducial_sel(df.copy(), {"track_particle_primary": True})
    assert sel["is_reconstructable"].tolist() == [True, False, True, False]
    sel = tb_utils.apply_fiducial_sel(df.copy(), {"nhits": 5})
    assert sel["is_reconstructable"].tolist() == [False, True, True, True]


@pytest.mark.gpu
def test_gpu_wrong_length_raises_and_pmask_is_prompt_only_on_cuda():
    dev = torch.device("cuda")
    ev = toy_event().to(dev)
    with pytest.raises(ValueError, match="matches none of"):
        gcu.build_signal_edges(ev, PMASK_WRONG, ev.track_edges)
    signal = gcu.build_signal_edges(ev, PMASK_RIGHT, ev.track_edges)
    assert signal.device.type == "cuda" and signal.cpu().tolist() == PROMPT_TRUE_EDGES
    assert gcu.handle_weighting(ev, PMASK_RIGHT).cpu().tolist() == PMASK_WEIGHTS
    assert loading_utils.handle_weighting(ev, PMASK_RIGHT).cpu().tolist() == PMASK_WEIGHTS
