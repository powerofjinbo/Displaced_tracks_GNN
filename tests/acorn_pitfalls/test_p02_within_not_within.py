"""p02 (brief §4.5): the within / not_within conditions of get_condition_lambda.

acorn has two copies of get_condition_lambda. At the pin 99d15341 the copy in
acorn/stages/graph_construction/utils.py (lines 213-218) read

    "within": lambda event: (condition_val[0] <= event[condition_key].float())
    & (event[condition_key].float() <= condition_val[1]),
    "not_within": lambda event: not (
        (condition_val[0] <= event[condition_key].float())
        & (event[condition_key].float() <= condition_val[1])
    ),

with condition_val == ["within", [lo, hi]]: `"within" <= tensor` raises TypeError (and `not tensor` would raise
for more than one element). Patch 0003 (patched lines 219-226) reads the bounds from condition_val[1] and negates
with `~`, exactly as acorn/utils/mapping_utils.py lines 34-39 (identical at pin and patched) always did.

Which copy each path uses (asserted below):
- graph_construction/utils copy: get_weight_mask <- build_signal_edges <- MetricLearning / DoubleMetricLearning
  .append_signal_edges (true pairs added to the training pairs) and .log_metrics (signal_eff denominator). Its
  handle_weighting / handle_hard_cuts are imported by no stage.
- mapping_utils copy (re-exported by acorn.utils): loading_utils.handle_weighting (loss weights of MetricLearning,
  edge classifier and track building), loading_utils.handle_hard_cuts (edge classifier and track building
  hard_cuts), loading_utils.handle_hard_node_cuts (graph-construction hard_cuts), the target_tracks evaluation of
  the graph-construction and edge-classifier stages, and the HGNN truth matching.
- Track building has neither: TrackBuildingStage.apply_target_conditions and track_building/utils.py
  apply_fiducial_sel are inline implementations without within / not_within.

Tests without the `char` prefix that touch the graph_construction copy's within / not_within FAIL against
ACORN_PIN_SRC (that demonstrates the bug); `test_char_*` tests pin down behaviour no patch changes.
"""

import pathlib
import re
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import acorn
import acorn.utils as acorn_utils
from acorn.stages.edge_classifier import edge_classifier_stage as ec_mod
from acorn.stages.graph_construction import graph_construction_stage as gc_mod
from acorn.stages.graph_construction import utils as gcu
from acorn.stages.graph_construction.models import double_metric_learning as dml_mod
from acorn.stages.graph_construction.models import metric_learning as ml_mod
from acorn.stages.track_building import track_building_stage as tb_mod
from acorn.stages.track_building import utils as tb_utils
from acorn.stages.track_building.models import hgnn_matching_utils
from acorn.utils import loading_utils, mapping_utils

COPIES = {"graph_construction": gcu, "mapping_utils": mapping_utils}
LO_HI = [1000.0, 2500.0]
WITHIN = ["within", LO_HI]
NOT_WITHIN = ["not_within", LO_HI]

# lambda-level feature with both bounds, values just outside them, +-inf and NaN
X = torch.tensor([float("-inf"), 0.0, 999.9, 1000.0, 1500.0, 2500.0, 2500.1, float("inf"), float("nan")])
X_WITHIN = [False, False, False, True, True, True, False, False, False]
X_NOT_WITHIN = [True, True, True, False, False, False, True, True, True]  # note: NaN passes not_within

# toy event: particle 1 hits 0-3 (pt 1000 = lower bound), particle 2 hits 4-7 (pt 999),
# particle 3 hits 8-10 (pt 2500 = upper bound), hit 11 noise; T = 8 true edges
N_HITS = 12
TRACK_EDGES = torch.tensor([[0, 1, 2, 4, 5, 6, 8, 9], [1, 2, 3, 5, 6, 7, 9, 10]])
TRACK_PT = torch.tensor([1000.0, 1000.0, 1000.0, 999.0, 999.0, 999.0, 2500.0, 2500.0])
TRACK_WITHIN = [True, True, True, False, False, False, True, True]
SIGNAL_WITHIN = [[0, 1, 2, 8, 9], [1, 2, 3, 9, 10]]
SIGNAL_NOT_WITHIN = [[4, 5, 6], [5, 6, 7]]
# predicted graph before / after MetricLearning.get_truth (sorted, with truth labels and the truth map)
PRED = torch.tensor([[0, 1, 4, 8, 3, 0, 2, 6], [1, 2, 5, 9, 7, 11, 3, 7]])
GRAPH_EDGES = [[0, 0, 1, 2, 3, 4, 6, 8], [1, 11, 2, 3, 7, 5, 7, 9]]
GRAPH_EDGE_Y = [True, False, True, True, False, True, True, True]
GRAPH_TMAP = [0, 2, 3, 5, -1, 6, 7, -1]
WEIGHTS_WITHIN = [3.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 3.0]
WEIGHTS_NOT_WITHIN = [0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 0.0]


def weighting(cond):
    return [
        {"weight": 1.0, "conditions": {"edge_y": False}},
        {"weight": 3.0, "conditions": {"edge_y": True, "track_particle_pt": cond}},
    ]


def toy_event(with_graph=False):
    r = torch.linspace(30.0, 1000.0, N_HITS)
    ev = Data(
        hit_id=torch.arange(N_HITS),
        hit_r=r,
        hit_z=torch.zeros(N_HITS),
        hit_R=r**2,
        track_edges=TRACK_EDGES.clone(),
        track_particle_pt=TRACK_PT.clone(),
        num_nodes=N_HITS,
    )
    if with_graph:
        ev.edge_index = torch.tensor(GRAPH_EDGES)
        ev.edge_y = torch.tensor(GRAPH_EDGE_Y)
        ev.track_to_edge_map = torch.tensor(GRAPH_TMAP)
    return ev


def make_metric_learning(weighting_config):
    torch.manual_seed(0)
    return ml_mod.MetricLearning(
        {
            "node_features": ["hit_r", "hit_z"],
            "emb_hidden": 8,
            "nb_layer": 1,
            "emb_dim": 4,
            "activation": "Tanh",
            "margin": 0.1,
            "weighting": weighting_config,
        }
    )


def edge_set(edges):
    return {tuple(e) for e in torch.as_tensor(edges).t().tolist()}


# ------------------------------------------- fixed by patch 0003 ------------------------------------------------


def test_graph_construction_within_not_within_are_elementwise_inclusive_complements():
    """Fixed by 0003. Pin: TypeError ('<=' not supported between instances of 'str' and 'Tensor')."""
    ev = Data(track_particle_pt=X.clone())
    w = gcu.get_condition_lambda("track_particle_pt", WITHIN)(ev)
    nw = gcu.get_condition_lambda("track_particle_pt", NOT_WITHIN)(ev)
    assert isinstance(w, torch.Tensor) and w.dtype == torch.bool and w.tolist() == X_WITHIN
    assert isinstance(nw, torch.Tensor) and nw.dtype == torch.bool and nw.tolist() == X_NOT_WITHIN
    assert torch.equal(nw, ~w)


@pytest.mark.parametrize(
    "key, cond",
    [
        ("track_flag", True),
        ("track_flag", False),
        ("track_particle_pt", LO_HI),
        ("track_particle_pt", WITHIN),
        ("track_particle_pt", NOT_WITHIN),
        ("track_particle_pt", ["in", [999.0, 2500.0]]),
        ("track_particle_pt", ["not_in", [999.0, 2500.0]]),
    ],
    ids=["true", "false", "bare_list", "within", "not_within", "in", "not_in"],
)
def test_both_copies_agree(key, cond):
    """within / not_within fixed by 0003 (fail at the pin); the other forms always agreed."""
    ev = Data(track_particle_pt=X.clone(), track_flag=torch.tensor([True, False] * 4 + [True]))
    a = gcu.get_condition_lambda(key, cond)(ev)
    b = mapping_utils.get_condition_lambda(key, cond)(ev)
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert a.dtype == b.dtype == torch.bool and torch.equal(a, b)


@pytest.mark.parametrize(
    "cond, expected", [(WITHIN, SIGNAL_WITHIN), (NOT_WITHIN, SIGNAL_NOT_WITHIN)], ids=["within", "not_within"]
)
def test_metric_learning_signal_edges_within_not_within(cond, expected):
    """Fixed by 0003: the real MetricLearning.append_signal_edges (build_signal_edges, graph_construction copy).
    Pin: TypeError, i.e. any metric-learning weighting with within / not_within crashed in training_step."""
    ml = make_metric_learning(weighting(cond))
    assert ml.append_signal_edges(toy_event(), torch.empty((2, 0), dtype=torch.long)).tolist() == expected


@pytest.mark.parametrize(
    "cond, weights", [(WITHIN, WEIGHTS_WITHIN), (NOT_WITHIN, WEIGHTS_NOT_WITHIN)], ids=["within", "not_within"]
)
def test_metric_learning_signal_edges_agree_with_loss_weights(cond, weights):
    """Fixed by 0003: in MetricLearning the signal pairs (graph_construction copy) and the loss weights
    (loading_utils -> mapping_utils copy) select the same true edges."""
    ml = make_metric_learning(weighting(cond))
    ev = toy_event()
    signal = ml.append_signal_edges(ev, torch.empty((2, 0), dtype=torch.long))
    ev.edge_index, ev.edge_y, ev.track_to_edge_map, _ = ml.get_truth(ev, PRED.clone(), False)
    w = ml.get_weights(ev)
    assert w.tolist() == weights
    positive_true = edge_set(ev.edge_index[:, (w > 0) & ev.edge_y])
    assert positive_true == edge_set(signal) & edge_set(ev.edge_index)


# --------------------------- mapping_utils copy and its users: correct at pin and patched ----------------------


def test_mapping_utils_within_not_within_correct():
    ev = Data(track_particle_pt=X.clone())
    assert mapping_utils.get_condition_lambda("track_particle_pt", WITHIN)(ev).tolist() == X_WITHIN
    assert mapping_utils.get_condition_lambda("track_particle_pt", NOT_WITHIN)(ev).tolist() == X_NOT_WITHIN
    for mod in COPIES.values():  # the bare [lo, hi] form never had the bug, in either copy
        assert mod.get_condition_lambda("track_particle_pt", LO_HI)(ev).tolist() == X_WITHIN


def test_routing_which_copy_each_stage_uses():
    assert gcu.get_condition_lambda.__module__ == "acorn.stages.graph_construction.utils"
    assert gcu.get_condition_lambda is not mapping_utils.get_condition_lambda
    assert acorn_utils.get_condition_lambda is mapping_utils.get_condition_lambda
    assert loading_utils.get_condition_lambda is mapping_utils.get_condition_lambda
    assert acorn_utils.handle_weighting is loading_utils.handle_weighting
    assert acorn_utils.handle_hard_cuts is loading_utils.handle_hard_cuts
    # metric learning: signal pairs via graph_construction/utils, loss weights via loading_utils
    for mod in (ml_mod, dml_mod):
        assert mod.build_signal_edges is gcu.build_signal_edges
        assert mod.handle_weighting is loading_utils.handle_weighting
    # edge classifier: weighting, hard_cuts and target_tracks all through mapping_utils
    assert ec_mod.handle_weighting is loading_utils.handle_weighting
    assert ec_mod.handle_hard_cuts is loading_utils.handle_hard_cuts
    assert ec_mod.get_condition_lambda is mapping_utils.get_condition_lambda
    # graph construction stage (metric learning / module map datasets and evaluation)
    assert gc_mod.get_condition_lambda is mapping_utils.get_condition_lambda
    assert gc_mod.handle_hard_node_cuts is loading_utils.handle_hard_node_cuts
    # track building: loading_utils for cuts / weights, no get_condition_lambda for its target_tracks
    assert tb_mod.handle_weighting is loading_utils.handle_weighting
    assert tb_mod.handle_hard_cuts is loading_utils.handle_hard_cuts
    assert not hasattr(tb_mod, "get_condition_lambda")
    assert hgnn_matching_utils.get_condition_lambda is mapping_utils.get_condition_lambda


def _imports_from(path, package_parts):
    """(module, name) pairs of every `from X import a, b` in a source file, relative imports resolved."""
    text = path.read_text()
    pairs = []
    for m in re.finditer(r"^\s*from\s+(\S+)\s+import\s+(\([^)]*\)|[^\n]+)", text, flags=re.M):
        mod, names = m.group(1), m.group(2)
        if mod.startswith("."):
            level = len(mod) - len(mod.lstrip("."))
            base = package_parts[: len(package_parts) - (level - 1)]
            mod = ".".join(base + ([mod.lstrip(".")] if mod.lstrip(".") else []))
        for line in names.strip("()").splitlines():
            for name in line.split("#")[0].split(","):
                if name.strip():
                    pairs.append((mod, name.split(" as ")[0].strip()))
    return pairs


def test_routing_only_build_signal_edges_is_imported_from_graph_construction_utils():
    """Static scan of the imported acorn package: graph_construction/utils.py reaches the stages only through
    build_signal_edges; its handle_weighting / get_weight_mask / handle_hard_cuts are dead code in stock acorn."""
    root = pathlib.Path(acorn.__file__).parent
    used = []
    for path in sorted(root.rglob("*.py")):
        package_parts = ["acorn", *path.relative_to(root).parent.parts]
        text = path.read_text()
        assert "import acorn.stages.graph_construction.utils" not in text
        for mod, name in _imports_from(path, package_parts):
            if mod == "acorn.stages.graph_construction.utils":
                used.append((path.relative_to(root).as_posix(), name))
            assert not (mod == "acorn.stages.graph_construction" and name == "utils"), path
    assert sorted(used) == [
        ("stages/graph_construction/models/double_metric_learning.py", "build_signal_edges"),
        ("stages/graph_construction/models/metric_learning.py", "build_signal_edges"),
    ]


@pytest.mark.parametrize(
    "cond, weights", [(WITHIN, WEIGHTS_WITHIN), (NOT_WITHIN, WEIGHTS_NOT_WITHIN)], ids=["within", "not_within"]
)
def test_loss_weights_within_not_within_correct(cond, weights):
    """MetricLearning.get_weights and the edge-classifier GraphDataset.construct_weighting (loading_utils ->
    mapping_utils copy) were never affected."""
    ml = make_metric_learning(weighting(cond))
    ev = toy_event()
    ev.edge_index, ev.edge_y, ev.track_to_edge_map, _ = ml.get_truth(ev, PRED.clone(), False)
    assert ev.edge_index.tolist() == GRAPH_EDGES and ev.edge_y.tolist() == GRAPH_EDGE_Y
    assert ev.track_to_edge_map.tolist() == GRAPH_TMAP
    assert ml.get_weights(ev).tolist() == weights
    ev = ec_mod.GraphDataset.construct_weighting(SimpleNamespace(hparams={"weighting": weighting(cond)}), toy_event(True))
    assert ev.edge_weights.tolist() == weights


def test_edge_classifier_hard_cuts_and_targets_within_not_within_correct():
    stub = SimpleNamespace(hparams={"hard_cuts": {"track_particle_pt": WITHIN}})
    ev = ec_mod.GraphDataset.apply_hard_cuts(stub, toy_event(True))
    assert ev.track_edges.tolist() == SIGNAL_WITHIN
    assert ev.edge_index.tolist() == [[0, 1, 2, 8], [1, 2, 3, 9]]
    assert ev.track_to_edge_map.tolist() == [0, 1, 2, 3, -1]
    stub = SimpleNamespace(hparams={"hard_cuts": {"track_particle_pt": NOT_WITHIN}})
    ev = ec_mod.GraphDataset.apply_hard_cuts(stub, toy_event(True))
    assert ev.track_edges.tolist() == SIGNAL_NOT_WITHIN
    assert ev.edge_index.tolist() == [[4, 6], [5, 7]]
    assert ev.track_to_edge_map.tolist() == [0, -1, 1]
    for cond, expected in [(WITHIN, TRACK_WITHIN), (NOT_WITHIN, [not v for v in TRACK_WITHIN])]:
        ev = toy_event(True)
        ev.track_to_passing_edge_map = ev.track_to_edge_map.clone()
        ec_mod.EdgeClassifierStage.apply_target_conditions(
            SimpleNamespace(device="cpu"), ev, {"track_particle_pt": cond}
        )
        assert ev.track_target_mask.tolist() == expected


def test_graph_construction_hard_cuts_and_targets_within_not_within_correct():
    cases = [
        ({"hit_r": ["within", [0.0, 500.0]]}, list(range(6)), [[0, 1, 2, 4], [1, 2, 3, 5]]),
        ({"hit_r": ["not_within", [0.0, 500.0]]}, list(range(6, 12)), [[0, 2, 3], [1, 3, 4]]),
        ({"track_particle_pt": WITHIN}, [0, 1, 2, 3, 8, 9, 10], [[0, 1, 2, 4, 5], [1, 2, 3, 5, 6]]),
    ]
    for hard_cuts, hits, track_edges in cases:
        ev = gc_mod.EventDataset.apply_hard_cuts(SimpleNamespace(hparams={"hard_cuts": hard_cuts}), toy_event())
        assert ev.hit_id.tolist() == hits, hard_cuts
        assert ev.track_edges.tolist() == track_edges, hard_cuts
    for cond, expected in [(WITHIN, TRACK_WITHIN), (NOT_WITHIN, [not v for v in TRACK_WITHIN])]:
        ev = toy_event(True)
        gc_mod.GraphConstructionStage.apply_target_conditions(
            SimpleNamespace(device="cpu"), ev, {"track_particle_pt": cond}
        )
        assert ev.track_target_mask.tolist() == expected


# --------------------------------- characterisation: behaviour no patch changes --------------------------------


@pytest.mark.parametrize("copy", list(COPIES))
def test_char_is_and_is_not_return_a_python_bool(copy):
    """Both copies, pin and patched: "is" / "is_not" compare the tensor with the whole list ["is", v], which
    torch cannot convert, so Python falls back to identity: the result is the scalar False / True, not a mask."""
    ev = Data(track_particle_pt=X.clone())
    lam = COPIES[copy].get_condition_lambda
    assert lam("track_particle_pt", ["is", 1000.0])(ev) is False
    assert lam("track_particle_pt", ["is_not", 1000.0])(ev) is True


def test_char_track_building_conditions_have_no_within():
    """TrackBuildingStage.apply_target_conditions and track_building/utils.py::apply_fiducial_sel are separate,
    unpatched implementations: a string-led list is compared as a number (TypeError); only [lo, hi] works,
    and the stage writes `target_mask`, not the `track_target_mask` the edge-level evaluators read."""
    stage = SimpleNamespace()
    for cond in (WITHIN, NOT_WITHIN):
        with pytest.raises(TypeError, match="not supported between instances of 'str' and 'Tensor'"):
            tb_mod.TrackBuildingStage.apply_target_conditions(stage, toy_event(True), {"track_particle_pt": cond})
    ev = toy_event(True)
    tb_mod.TrackBuildingStage.apply_target_conditions(stage, ev, {"track_particle_pt": LO_HI})
    assert ev.target_mask.tolist() == TRACK_WITHIN and "track_target_mask" not in ev
    df = pd.DataFrame({"pt": [999.0, 1000.0, 2500.0, 2500.5]})
    for cond in (WITHIN, NOT_WITHIN):
        with pytest.raises(TypeError, match="Invalid comparison"):
            tb_utils.apply_fiducial_sel(df.copy(), {"pt": cond})
    assert tb_utils.apply_fiducial_sel(df.copy(), {"pt": LO_HI})["is_reconstructable"].tolist() == [
        False, True, True, False,
    ]


def test_char_within_compares_in_float32():
    """within and [lo, hi] compare event[key].float() (float32) with the Python bounds, which torch also casts
    to float32: int64 ids above 2**24 collide (16777217 -> 16777216.0), so selecting one id selects its
    neighbour too. "in" compares in the original dtype and is exact."""
    ids = torch.tensor([16777216, 16777217, 16777218])
    ev = Data(track_particle_id=ids)
    assert mapping_utils.get_condition_lambda("track_particle_id", ["within", [16777217, 16777217]])(ev).tolist() == [
        True, True, False,
    ]
    for mod in COPIES.values():
        for wanted in (16777216, 16777217):
            assert mod.get_condition_lambda("track_particle_id", [wanted, wanted])(ev).tolist() == [True, True, False]
        assert mod.get_condition_lambda("track_particle_id", ["in", [16777217]])(ev).tolist() == [False, True, False]


def test_char_not_within_selects_nan():
    ev = Data(track_particle_pt=torch.tensor([float("nan"), 1500.0]))
    assert mapping_utils.get_condition_lambda("track_particle_pt", WITHIN)(ev).tolist() == [False, True]
    assert mapping_utils.get_condition_lambda("track_particle_pt", NOT_WITHIN)(ev).tolist() == [True, False]


@pytest.mark.gpu
def test_gpu_within_not_within_both_copies_on_cuda():
    ev = Data(track_particle_pt=X.cuda())
    for mod in COPIES.values():
        w = mod.get_condition_lambda("track_particle_pt", WITHIN)(ev)
        nw = mod.get_condition_lambda("track_particle_pt", NOT_WITHIN)(ev)
        assert w.is_cuda and w.cpu().tolist() == X_WITHIN
        assert nw.is_cuda and nw.cpu().tolist() == X_NOT_WITHIN
    ml = make_metric_learning(weighting(WITHIN)).cuda()
    signal = ml.append_signal_edges(toy_event().cuda(), torch.empty((2, 0), dtype=torch.long, device="cuda"))
    assert signal.is_cuda and signal.cpu().tolist() == SIGNAL_WITHIN
