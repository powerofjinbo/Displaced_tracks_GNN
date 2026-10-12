"""p10: event ids nested in lists end up in output filenames (patch 0005).

PyG's DataLoader wraps every non-tensor attribute in a list, once per stage that batches events, so after
metric learning -> GNN the id is [['000021009']]. At the pin the savers unwrapped one level
(`event_id[0] if isinstance(event_id, list)`) or indexed `graph.event_id[0]` unconditionally:
  edge_classifier_stage.py:517-519, 551-553   graph_construction/models/metric_learning.py:492-494
  track_building/track_building_stage.py:332, 334, 337, 354   track_building/models/walkthrough.py:130
Patched: acorn/utils/loading_utils.py:37-49 flatten_event_id, used at edge_classifier_stage.py:518, 550-551;
metric_learning.py:496-497; track_building_stage.py:322, 352, 356; walkthrough.py:132, 135.
Not patched (one-level unwrap, characterised below): graph_construction/models/double_metric_learning.py:521-523,
544-546 and track_building/models/hgnn.py:274-278.
"""

import glob
import os
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

EID = "000021009"
COMMON = dict(accelerator="cpu", devices=1, nodes=1, num_workers=[0, 0, 0], data_split=[0, 0, 1])


def toy_event(event_id, scores=False):
    """4 tracks x 4 hits (outward in r) + 3 noise hits, acorn's prefixed names."""
    n_tracks, hpt = 4, 4
    n = n_tracks * hpt + 3
    r = torch.cat([torch.tensor([30.0, 70.0, 120.0, 170.0]).repeat(n_tracks), torch.tensor([50.0, 90.0, 140.0])])
    phi = torch.cat([torch.full((hpt,), 0.5 * t) for t in range(n_tracks)] + [torch.tensor([2.5, 2.7, 2.9])])
    src = [t * hpt + i for t in range(n_tracks) for i in range(hpt - 1)]
    track_edges = torch.tensor([src, [s + 1 for s in src]])
    fakes = torch.tensor([src, [((t + 1) % n_tracks) * hpt + i + 1 for t in range(n_tracks) for i in range(hpt - 1)]])
    edge_index = torch.cat([track_edges, fakes, torch.tensor([[16, 17, 0], [17, 18, 16]])], dim=1)
    edge_y = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    edge_y[: track_edges.shape[1]] = True
    event = Data(
        hit_id=torch.arange(n),
        hit_x=r * torch.cos(phi),
        hit_r=r,
        hit_phi=phi,
        hit_z=torch.linspace(-100.0, 100.0, n),
        track_edges=track_edges,
        track_particle_pt=torch.tensor([0.5, 0.8, 2.0, 5.0]).repeat_interleave(hpt - 1),
        track_particle_id=torch.arange(1, n_tracks + 1).repeat_interleave(hpt - 1),
        edge_index=edge_index,
        edge_y=edge_y,
        track_to_edge_map=torch.arange(track_edges.shape[1]),
        event_id=event_id,
        config=[{"stage": "toy"}],
    )
    if scores:
        event.edge_scores = edge_y.float() * 0.9 + 0.05  # true edges 0.95, fakes 0.05
    return event


def files_under(d):
    return sorted(os.path.relpath(p, d) for p in glob.glob(os.path.join(str(d), "**", "*"), recursive=True)
                  if os.path.isfile(p))


def stored_ids(d):
    from acorn.utils.loading_utils import load_pyg

    return [load_pyg(p[: -len(".gz")], weights_only=False).event_id
            for p in sorted(glob.glob(os.path.join(str(d), "**", "*.pyg.gz"), recursive=True))]


def predict(module, tmp_path):
    from pytorch_lightning import Trainer

    Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
            default_root_dir=str(tmp_path)).predict(module)


# --------------------------------------------------------------------------------------------------------------
# flatten_event_id (patched only)
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [EID, [EID], [[EID]], ((EID,),), [[[EID]]]],
                         ids=["str", "list", "list2", "tuple2", "list3"])
def test_flatten_event_id_unwraps_any_nesting(raw):
    from acorn.utils.loading_utils import flatten_event_id

    assert flatten_event_id(raw) == EID


def test_flatten_event_id_stringifies_and_rejects_ambiguous_containers():
    from acorn.utils.loading_utils import flatten_event_id

    assert flatten_event_id(21009) == "21009"
    for bad in ([], [[]], [EID, "000021010"], [[EID, "000021010"]]):
        with pytest.raises(ValueError):
            flatten_event_id(bad)


def test_pyg_dataloader_nests_string_event_ids_once_per_collation():
    """Why the ids nest: characterises torch_geometric (independent of acorn)."""
    from torch_geometric.loader import DataLoader

    def collate_once(event_id):
        return next(iter(DataLoader([Data(x=torch.zeros(2, 1), event_id=event_id)], batch_size=1))).event_id

    assert collate_once(EID) == [EID]
    assert collate_once([EID]) == [[EID]]


# --------------------------------------------------------------------------------------------------------------
# real three-stage chain on CPU: MetricLearning -> Filter -> ConnectedComponents
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("subdir_size", [None, 1000])
def test_three_stage_chain_writes_and_stores_flat_event_ids(tmp_path, monkeypatch, subdir_size):
    import acorn.stages.graph_construction.models.utils as gc_utils
    from acorn.stages.edge_classifier.models.filter import Filter
    from acorn.stages.graph_construction.models.metric_learning import MetricLearning
    from acorn.stages.track_building.models.connected_components import ConnectedComponents
    from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg, save_pyg

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)  # CPU run: torch_cluster radius (FRNN is CUDA-only)
    common = dict(COMMON, **({"subdir_size": subdir_size} if subdir_size else {}))
    raw = tmp_path / "raw"
    os.makedirs(raw / "testset")
    save_pyg(remove_variable_name_prefix_in_pyg(toy_event(EID)), str(raw / "testset" / f"event{EID}-graph.pyg"))

    feats = dict(node_features=["hit_r", "hit_phi", "hit_z"], node_scales=[1000.0, 3.14, 1000.0])
    ml_dir, gnn_dir, cc_dir = tmp_path / "ml", tmp_path / "gnn", tmp_path / "cc"
    torch.manual_seed(0)
    predict(MetricLearning(dict(stage="graph_construction", model="MetricLearning", input_dir=str(raw),
                                stage_dir=str(ml_dir), emb_hidden=8, nb_layer=2, emb_dim=4, activation="Tanh",
                                r_infer=2.1, knn_infer=50, margin=0.1,
                                weighting=[{"weight": 1.0, "conditions": {"edge_y": True}}], **feats, **common)),
            tmp_path)
    predict(Filter(dict(stage="edge_classifier", model="Filter", input_dir=str(ml_dir), stage_dir=str(gnn_dir),
                        hidden=8, nb_layer=2, layernorm=False, batchnorm=False, hidden_activation="ReLU",
                        edge_cut=0.5, ratio=0, **feats, **common)),
            tmp_path)
    ConnectedComponents.infer(dict(stage="track_building", model="ConnectedComponents", input_dir=str(gnn_dir),
                                   stage_dir=str(cc_dir), score_cut=0.5, **common))

    shard = f"{int(EID) // subdir_size}/" if subdir_size else ""
    assert files_under(ml_dir) == [f"testset/{shard}event{EID}.pyg.gz"]
    assert files_under(gnn_dir) == [f"testset/{shard}event{EID}.pyg.gz"]
    assert files_under(cc_dir) == [f"testset/{shard}event{EID}.pyg.gz", f"testset_tracks/event{EID}.txt"]
    for d in (ml_dir, gnn_dir, cc_dir):
        assert stored_ids(d) == [EID]  # a plain string, not ['...'] / [['...']]


def test_track_building_does_not_truncate_plain_string_ids(tmp_path):
    """At the pin graph.event_id[0] took the first character: both events below became event0.* (one overwrote
    the other)."""
    from acorn.stages.track_building.models.connected_components import ConnectedComponents
    from acorn.utils.loading_utils import save_pyg

    scored = tmp_path / "scored"
    os.makedirs(scored / "testset")
    for eid in ("000021009", "000021010"):
        save_pyg(toy_event(eid, scores=True), str(scored / "testset" / f"event{eid}.pyg"))
    cc_dir = tmp_path / "cc"
    ConnectedComponents.infer(dict(stage="track_building", model="ConnectedComponents", input_dir=str(scored),
                                   stage_dir=str(cc_dir), score_cut=0.5, **dict(COMMON, data_split=[0, 0, 2])))
    assert files_under(cc_dir) == [
        "testset/event000021009.pyg.gz",
        "testset/event000021010.pyg.gz",
        "testset_tracks/event000021009.txt",
        "testset_tracks/event000021010.txt",
    ]
    # 4 true tracks per event, each a CC component of 4 hits
    with open(cc_dir / "testset_tracks" / "event000021009.txt") as f:
        assert len(f.read().split("\n")) - 1 == 4


def test_athena_csv_track_names_use_the_whole_id_except_event_zero(tmp_path):
    """save_tracks with athena_csv_format names files <prefix>_<id without leading zeros>.csv
    (track_building_stage.py:327-334). Pin: graph.event_id[0] is the first character "0", so every zero-padded id
    became <prefix>_.csv. Patched: the whole id, except that event 0 still loses its number."""
    from acorn.stages.track_building.models.connected_components import ConnectedComponents

    cc = ConnectedComponents(dict(stage_dir=str(tmp_path), save_tracks_as_csv=True, athena_csv_format=True,
                                  event_prefix="track_RUN"))
    for eid in ("000021009", "000000000"):
        cc.save_tracks(Data(event_id=eid), [[1, 2, 3]], str(tmp_path / "testset"))
    assert sorted(os.listdir(tmp_path / "testset_tracks")) == ["track_RUN_.csv", "track_RUN_21009.csv"]


# --------------------------------------------------------------------------------------------------------------
# savers that still unwrap one level only (not touched by patch 0005): characterisation, same at pin and patched
# --------------------------------------------------------------------------------------------------------------


def test_double_metric_learning_saver_still_unwraps_one_level(tmp_path):
    from acorn.stages.graph_construction.models.double_metric_learning import DoubleMetricLearning

    stub = SimpleNamespace(hparams={"stage_dir": str(tmp_path)})
    # normal case (flat id on disk, one DataLoader collation): right file name, nested id stored in the graph
    DoubleMetricLearning.save_graph(stub, Data(hit_x=torch.zeros(3), event_id=[EID], config=[]), "testset")
    assert files_under(tmp_path) == [f"testset/event{EID}.pyg.gz"]
    assert stored_ids(tmp_path) == [[EID]]
    # input already nested once (e.g. written by an unpatched stage): the list ends up in the file name
    DoubleMetricLearning.save_graph(stub, Data(hit_x=torch.zeros(3), event_id=[[EID]], config=[]), "valset")
    assert files_under(tmp_path / "valset") == [f"event['{EID}'].pyg.gz"]
    # ... and sharding cannot parse it
    stub.hparams["subdir_size"] = 1000
    with pytest.raises(ValueError, match="Cannot derive event id"):
        DoubleMetricLearning.save_graph(stub, Data(hit_x=torch.zeros(3), event_id=[[EID]], config=[]), "testset")


def test_hgnn_predict_path_still_unwraps_one_level(tmp_path):
    from acorn.stages.track_building.models.hgnn import HierarchicalGNN

    paths = []
    stub = SimpleNamespace(
        predict_dataloader=lambda: [SimpleNamespace(dataset=SimpleNamespace(data_name="testset"))],
        hparams={"stage_dir": str(tmp_path)},
        build_track=lambda batch, output_dir, track_dir: paths.append((output_dir, track_dir)),
    )
    # PartialGraphDataset uses collate_fn=lambda lst: lst[0], so a flat stored id stays flat: correct names
    HierarchicalGNN.predict_step(stub, SimpleNamespace(full_event=SimpleNamespace(event_id=EID)), 0)
    assert [os.path.basename(p) for p in paths[-1]] == [f"event{EID}.pyg", f"event{EID}.txt"]
    # an id nested twice (written by an unpatched stage) still reaches the file name
    HierarchicalGNN.predict_step(stub, SimpleNamespace(full_event=SimpleNamespace(event_id=[[EID]])), 0)
    assert [os.path.basename(p) for p in paths[-1]] == [f"event['{EID}'].pyg", f"event['{EID}'].txt"]


def test_sharded_data_reading_output_is_invisible_to_graph_construction(tmp_path):
    """subdir_size shards by the digits after "event" (loading_utils.py:52-65, pin 37-50; needs a flat id), but
    the graph construction EventDataset lists input_dir/<split> non-recursively (graph_construction_stage.py:319-342)
    and opens unsharded paths (:278-291): a sharded data-reading output yields an empty dataset."""
    from acorn.stages.graph_construction.graph_construction_stage import EventDataset
    from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg, save_pyg

    save_pyg(remove_variable_name_prefix_in_pyg(toy_event(EID)), str(tmp_path / "testset" / f"event{EID}-graph.pyg"),
             subdir=1000)
    assert files_under(tmp_path) == [f"testset/21/event{EID}-graph.pyg.gz"]
    with pytest.warns(UserWarning, match="No events found"):
        assert len(EventDataset(str(tmp_path), "testset", None, hparams={})) == 0


def test_source_audit_only_known_one_level_unwraps_remain():
    """Static audit of the imported acorn tree: `<obj>.event_id[0]` outside comments. Patched: only the two
    unpatched stages characterised above. At the pin also edge_classifier_stage (2), metric_learning (1),
    track_building_stage (4) and walkthrough (1)."""
    import re
    from collections import Counter

    import acorn

    base = os.path.dirname(os.path.dirname(os.path.abspath(acorn.__file__)))
    hits = Counter()
    for path in glob.glob(os.path.join(base, "acorn", "**", "*.py"), recursive=True):
        with open(path) as f:
            for line in f:
                if re.search(r"\.event_id\[0\]", line.split("#", 1)[0]):
                    hits[os.path.relpath(path, base)] += 1
    assert dict(hits) == {
        "acorn/stages/graph_construction/models/double_metric_learning.py": 2,
        "acorn/stages/track_building/models/hgnn.py": 1,
    }
