"""p08: inference loads the newest best*.ckpt and reuses the training hparams (hard_cuts, input_dir, data_split).

No patch touches this: acorn/core/core_utils.py, infer_stage.py and eval_stage.py are identical at the pin
(99d15341) and in the patched tree, so every test here is a characterisation test and passes in both.

Source (same line numbers at the pin and patched):
  acorn/core/core_utils.py:78-88   find_latest_checkpoint: Path(base).rglob(template) for "best*.ckpt", only if none
                                   is found anywhere "*.ckpt"; max(..., key=os.path.getctime)
  acorn/core/infer_stage.py:77-93  lightning_infer: checkpoint or find_latest_checkpoint(config["stage_dir"],
                                   ["best*.ckpt", "*.ckpt"]); load_from_checkpoint;
                                   _hparams = {**ckpt_hparams, **config}
  acorn/core/eval_stage.py:75-97   evaluate: same discovery, config = {**checkpoint_config, **config}, then
                                   stage_module.evaluate(config) WITHOUT the checkpoint
  acorn/stages/edge_classifier/edge_classifier_stage.py:561-573 (pin 563-575) EdgeClassifierStage.evaluate builds
                                   cls(config) and loads weights only if it is handed `checkpoint`
  acorn's own examples/Example_3/{filter,gnn}_infer.yaml:1-8 and metric_learning_infer.yaml:1-16 omit hard_cuts,
  while filter_train.yaml:33-34, gnn_train.yaml:28-29, metric_learning_train.yaml:34-35 set pt >= 1000 MeV cuts
"""

import glob
import os
import time

import torch
import yaml
from torch_geometric.data import Data

TRAIN_HARD_CUTS = {"track_particle_pt": [1.0, 1.0e9]}
N_TRACKS, HITS_PER_TRACK = 4, 4
TRACK_PT = [0.5, 0.8, 2.0, 5.0]  # GeV; tracks 2 and 3 pass TRAIN_HARD_CUTS
N_EDGES_FULL = 27  # 12 true + 12 cross-track fakes + 3 noise edges
N_EDGES_HARD_CUT = 9  # 6 true edges of tracks 2,3 + 3 fakes between them
N_TRACK_EDGES_FULL, N_TRACK_EDGES_HARD_CUT = 12, 6


def toy_event(event_id):
    """4 tracks x 4 hits (outward in r) + 3 noise hits; per-track-edge pT; acorn's prefixed variable names."""
    n = N_TRACKS * HITS_PER_TRACK + 3
    r = torch.cat([torch.tensor([30.0, 70.0, 120.0, 170.0]).repeat(N_TRACKS), torch.tensor([50.0, 90.0, 140.0])])
    phi = torch.cat([torch.full((HITS_PER_TRACK,), 0.5 * t) for t in range(N_TRACKS)] + [torch.tensor([2.5, 2.7, 2.9])])
    z = torch.linspace(-100.0, 100.0, n)
    src = [t * HITS_PER_TRACK + i for t in range(N_TRACKS) for i in range(HITS_PER_TRACK - 1)]
    track_edges = torch.tensor([src, [s + 1 for s in src]])
    # fake edges: hit i of track t -> hit i+1 of track t+1
    dst_fake = [((t + 1) % N_TRACKS) * HITS_PER_TRACK + i + 1
                for t in range(N_TRACKS) for i in range(HITS_PER_TRACK - 1)]
    fakes = torch.tensor([src, dst_fake])
    noise = torch.tensor([[16, 17, 0], [17, 18, 16]])
    edge_index = torch.cat([track_edges, fakes, noise], dim=1)
    edge_y = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    edge_y[: track_edges.shape[1]] = True
    return Data(
        hit_id=torch.arange(n),
        hit_x=r * torch.cos(phi),
        hit_r=r,
        hit_phi=phi,
        hit_z=z,
        track_edges=track_edges,
        track_particle_pt=torch.tensor(TRACK_PT).repeat_interleave(HITS_PER_TRACK - 1),
        track_particle_id=torch.arange(1, N_TRACKS + 1).repeat_interleave(HITS_PER_TRACK - 1),
        edge_index=edge_index,
        edge_y=edge_y,
        track_to_edge_map=torch.arange(track_edges.shape[1]),
        event_id=event_id,
        config=[{"stage": "toy"}],
    )


def write_inputs(root, events_per_split):
    from acorn.utils.loading_utils import save_pyg

    for split, ids in events_per_split.items():
        os.makedirs(os.path.join(root, split), exist_ok=True)
        for eid in ids:
            save_pyg(toy_event(eid), os.path.join(root, split, f"event{eid}.pyg"))
    return str(root)


def train_hparams(input_dir, stage_dir, tag):
    return dict(
        stage="edge_classifier",
        model="Filter",
        input_dir=str(input_dir),
        stage_dir=str(stage_dir),
        data_split=[1, 1, 1],
        node_features=["hit_r", "hit_phi", "hit_z"],
        node_scales=[1000.0, 3.14, 1000.0],
        hidden=8,
        nb_layer=2,
        layernorm=False,
        batchnorm=False,
        hidden_activation="ReLU",
        edge_cut=0.5,
        ratio=0,
        hard_cuts=dict(TRAIN_HARD_CUTS),
        ckpt_tag=tag,
        num_workers=[0, 0, 0],
    )


def write_ckpt(path, hparams, fill=None, cls=None):
    """A Lightning checkpoint as ModelCheckpoint writes it (state_dict + hyper_parameters + hparams_name)."""
    import pytorch_lightning as pl
    from acorn.stages.edge_classifier.models.filter import Filter

    torch.manual_seed(0)
    model = (cls or Filter)(dict(hparams))
    if fill is not None:
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(fill)
    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "hparams_name": "hparams",
        "pytorch-lightning_version": pl.__version__,
        "epoch": 0,
        "global_step": 0,
    }
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    torch.save(ckpt, str(path))
    return str(path)


def touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    return str(path)


def bump_ctime_after(path, ref):
    """Metadata-only change (chmod to the same mode) until ctime(path) > ctime(ref); content and mtime untouched."""
    for _ in range(200):
        if os.stat(path).st_ctime_ns > os.stat(ref).st_ctime_ns:
            return
        time.sleep(0.01)
        os.chmod(path, os.stat(path).st_mode)
    raise RuntimeError("filesystem ctime did not advance")


def load_outputs(stage_dir, split):
    from acorn.utils.loading_utils import load_pyg

    files = sorted(glob.glob(os.path.join(stage_dir, split, "**", "*.pyg.gz"), recursive=True))
    return [load_pyg(f[: -len(".gz")], weights_only=False) for f in files]


# --------------------------------------------------------------------------------------------------------------
# 1. how the checkpoint is chosen when none is given
# --------------------------------------------------------------------------------------------------------------


def test_newest_ctime_best_wins_over_better_metric_and_over_last(tmp_path):
    from acorn.core.core_utils import find_latest_checkpoint

    art = tmp_path / "artifacts"
    good = touch(art / "best-runA-val_loss=0.100000-epoch=9.ckpt")
    worse = touch(art / "best-runB-val_loss=0.900000-epoch=0.ckpt")
    bump_ctime_after(worse, good)
    last = touch(art / "last-runB.ckpt")
    bump_ctime_after(last, worse)

    # templates used by infer_stage.lightning_infer and eval_stage.evaluate
    chosen = find_latest_checkpoint(str(tmp_path), templates=["best*.ckpt", "*.ckpt"])
    assert chosen == worse  # newest best*.ckpt: the val_loss in the name is ignored, last-*.ckpt is never considered


def test_selection_follows_ctime_so_chmod_or_touch_changes_the_checkpoint(tmp_path):
    from acorn.core.core_utils import find_latest_checkpoint

    art = tmp_path / "artifacts"
    a = touch(art / "best-a.ckpt")
    b = touch(art / "best-b.ckpt")
    bump_ctime_after(b, a)
    assert find_latest_checkpoint(str(tmp_path), ["best*.ckpt", "*.ckpt"]) == b

    # `chmod a-w` (as done for frozen data) on the OLDER checkpoint makes it "the newest"
    time.sleep(0.01)
    os.chmod(a, 0o444)
    bump_ctime_after(a, b)
    assert find_latest_checkpoint(str(tmp_path), ["best*.ckpt", "*.ckpt"]) == a

    # setting b's mtime one year into the past is itself a metadata change: ctime(b) = now, so b wins again
    old = time.time() - 365 * 86400
    time.sleep(0.01)
    os.utime(b, (old, old))
    bump_ctime_after(b, a)
    assert os.path.getmtime(b) < os.path.getmtime(a)
    assert find_latest_checkpoint(str(tmp_path), ["best*.ckpt", "*.ckpt"]) == b


def test_search_is_recursive_and_falls_back_to_any_ckpt(tmp_path):
    from acorn.core.core_utils import find_latest_checkpoint

    stage = tmp_path / "stage"
    last = touch(stage / "artifacts" / "last-run1.ckpt")
    # no best*.ckpt anywhere: falls back to "*.ckpt", i.e. a last-epoch checkpoint
    assert find_latest_checkpoint(str(stage), ["best*.ckpt", "*.ckpt"]) == last

    best = touch(stage / "artifacts" / "best-run1-val_loss=0.100000-epoch=3.ckpt")
    bump_ctime_after(best, last)
    assert find_latest_checkpoint(str(stage), ["best*.ckpt", "*.ckpt"]) == best

    # a best*.ckpt in ANY subdirectory of stage_dir (another run, an old copy, wandb files) is a candidate
    stale = touch(stage / "old_runs" / "2025" / "artifacts" / "best-old-val_loss=0.500000-epoch=1.ckpt")
    bump_ctime_after(stale, best)
    assert find_latest_checkpoint(str(stage), ["best*.ckpt", "*.ckpt"]) == stale

    # the default template (training resume, core_utils.get_stage_module) takes any newest *.ckpt
    newest_last = touch(stage / "artifacts" / "last-run2.ckpt")
    bump_ctime_after(newest_last, stale)
    assert find_latest_checkpoint(str(stage)) == newest_last
    assert find_latest_checkpoint(str(tmp_path / "empty")) is None


# --------------------------------------------------------------------------------------------------------------
# 2. what inference takes from the checkpoint (real acorn entry point, CPU, toy events)
# --------------------------------------------------------------------------------------------------------------


def test_lightning_infer_reuses_training_hard_cuts_input_dir_and_data_split(tmp_path, monkeypatch):
    from acorn.core.infer_stage import lightning_infer
    from acorn.stages.edge_classifier.models.filter import Filter

    monkeypatch.chdir(tmp_path)  # Trainer() writes lightning_logs/ into the cwd
    train_in = write_inputs(
        tmp_path / "train_input", {"trainset": ["000000001"], "valset": ["000000002"], "testset": ["000000003"]}
    )
    write_inputs(tmp_path / "infer_input", {"testset": ["000000007"]})  # what the user meant to run on
    stage = tmp_path / "stage"
    art = stage / "artifacts"
    ck_a = write_ckpt(art / "best-runA-val_loss=0.100000-epoch=9.ckpt", train_hparams(train_in, stage, "A"))
    ck_b = write_ckpt(art / "best-runB-val_loss=0.900000-epoch=0.ckpt", train_hparams(train_in, stage, "B"))
    bump_ctime_after(ck_b, ck_a)

    # an inference config that does not repeat input_dir, data_split, hard_cuts (and passes no checkpoint)
    config = dict(stage="edge_classifier", model="Filter", stage_dir=str(stage), accelerator="cpu", devices=1,
                  nodes=1, num_workers=[0, 0, 0])
    lightning_infer(dict(config), Filter)

    outputs = {split: load_outputs(str(stage), split) for split in ("trainset", "valset", "testset")}
    assert {s: len(v) for s, v in outputs.items()} == {"trainset": 1, "valset": 1, "testset": 1}  # data_split [1,1,1]
    for split, events in outputs.items():
        out = events[0]
        used = out.config[-1]  # hparams appended by save_edge_scores
        assert used["ckpt_tag"] == "B"  # newest best*.ckpt, not the better val_loss
        assert used["input_dir"] == train_in  # training input_dir, not infer_input
        assert used["data_split"] == [1, 1, 1]
        assert used["hard_cuts"] == TRAIN_HARD_CUTS
        # the training hard cut was applied at inference: only edges between hits of pT >= 1 GeV tracks survive
        assert out.edge_index.shape[1] == N_EDGES_HARD_CUT
        assert out.track_edges.shape[1] == N_TRACK_EDGES_HARD_CUT
    assert not glob.glob(os.path.join(str(stage), "*", "*000000007*"))  # the intended event was never processed


def test_lightning_infer_explicit_checkpoint_and_keys_override_training(tmp_path, monkeypatch):
    """The safe usage: explicit checkpoint, input_dir, data_split and hard_cuts ({} = no cut)."""
    from acorn.core.infer_stage import lightning_infer
    from acorn.stages.edge_classifier.models.filter import Filter

    monkeypatch.chdir(tmp_path)
    train_in = write_inputs(
        tmp_path / "train_input", {"trainset": ["000000001"], "valset": ["000000002"], "testset": ["000000003"]}
    )
    infer_in = write_inputs(tmp_path / "infer_input", {"testset": ["000000007"]})
    ckdir = tmp_path / "ckpts"
    ck_a = write_ckpt(ckdir / "best-runA-val_loss=0.100000-epoch=9.ckpt", train_hparams(train_in, tmp_path, "A"))
    ck_b = write_ckpt(ckdir / "best-runB-val_loss=0.900000-epoch=0.ckpt", train_hparams(train_in, tmp_path, "B"))
    bump_ctime_after(ck_b, ck_a)
    stage = tmp_path / "infer_stage"
    config = dict(stage="edge_classifier", model="Filter", stage_dir=str(stage), input_dir=infer_in,
                  data_split=[0, 0, 1], hard_cuts={}, accelerator="cpu", devices=1, nodes=1, num_workers=[0, 0, 0])
    lightning_infer(dict(config), Filter, checkpoint=ck_a)

    assert not os.path.exists(stage / "trainset") and not os.path.exists(stage / "valset")
    (out,) = load_outputs(str(stage), "testset")
    used = out.config[-1]
    assert used["ckpt_tag"] == "A" and used["input_dir"] == infer_in and used["data_split"] == [0, 0, 1]
    assert used["hard_cuts"] == {}
    assert out.edge_index.shape[1] == N_EDGES_FULL and out.track_edges.shape[1] == N_TRACK_EDGES_FULL
    assert out.scores.shape[0] == N_EDGES_FULL


def test_lightning_infer_without_stage_dir_writes_into_the_training_stage_dir(tmp_path, monkeypatch):
    from acorn.core.infer_stage import lightning_infer
    from acorn.stages.edge_classifier.models.filter import Filter

    monkeypatch.chdir(tmp_path)
    infer_in = write_inputs(tmp_path / "infer_input", {"testset": ["000000007"]})
    train_stage = tmp_path / "train_stage"
    ck = write_ckpt(tmp_path / "ckpts" / "best-A.ckpt", train_hparams(tmp_path / "train_input", train_stage, "A"))
    config = dict(stage="edge_classifier", model="Filter", input_dir=infer_in, data_split=[0, 0, 1], hard_cuts={},
                  accelerator="cpu", devices=1, nodes=1, num_workers=[0, 0, 0])
    lightning_infer(dict(config), Filter, checkpoint=ck)  # with -c, stage_dir is not needed to find a checkpoint
    (out,) = load_outputs(str(train_stage), "testset")  # ... and the outputs land in the TRAINING stage_dir
    assert out.config[-1]["stage_dir"] == str(train_stage)


def test_metric_learning_inference_inherits_node_level_hard_cuts(tmp_path, monkeypatch):
    """Graph construction applies hard_cuts to NODES (graph_construction_stage.py:375-394): with the training cut
    inherited, hits of particles failing it (and all noise) are removed before the graph is built.
    acorn's own examples/Example_3/metric_learning_infer.yaml omits hard_cuts while metric_learning_train.yaml:34-35
    sets pt >= 1 GeV."""
    import acorn.stages.graph_construction.models.utils as gc_utils
    from acorn.core.infer_stage import lightning_infer
    from acorn.stages.graph_construction.models.metric_learning import MetricLearning
    from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg, save_pyg

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)  # CPU run: torch_cluster radius (FRNN is CUDA-only)
    raw = tmp_path / "feature_store"
    os.makedirs(raw / "testset")
    save_pyg(remove_variable_name_prefix_in_pyg(toy_event("000000003")),
             str(raw / "testset" / "event000000003-graph.pyg"))  # data-reading layout, unprefixed names
    hp = dict(stage="graph_construction", model="MetricLearning", input_dir=str(raw), stage_dir=str(tmp_path / "ml"),
              data_split=[0, 0, 1], node_features=["hit_r", "hit_phi", "hit_z"], node_scales=[1000.0, 3.14, 1000.0],
              emb_hidden=8, nb_layer=2, emb_dim=4, activation="Tanh", r_infer=2.1, knn_infer=50, margin=0.1,
              weighting=[{"weight": 1.0, "conditions": {"edge_y": True}}], hard_cuts=dict(TRAIN_HARD_CUTS),
              num_workers=[0, 0, 0])
    ck = write_ckpt(tmp_path / "ckpts" / "best-ml.ckpt", hp, cls=MetricLearning)
    n_hits = {}
    for name, extra in [("inherited", {}), ("explicit", {"hard_cuts": {}})]:
        stage = tmp_path / f"ml_{name}"
        lightning_infer(dict(stage="graph_construction", model="MetricLearning", stage_dir=str(stage),
                             accelerator="cpu", devices=1, nodes=1, **extra), MetricLearning, checkpoint=ck)
        (out,) = load_outputs(str(stage), "testset")
        n_hits[name] = out.hit_id.shape[0]
        assert out.config[-1]["hard_cuts"] == extra.get("hard_cuts", TRAIN_HARD_CUTS)
    assert n_hits == {"inherited": 2 * HITS_PER_TRACK, "explicit": N_TRACKS * HITS_PER_TRACK + 3}


# --------------------------------------------------------------------------------------------------------------
# 3. the eval entry point: hparams from the checkpoint, weights never loaded, default dataset = valset
# --------------------------------------------------------------------------------------------------------------


def test_eval_entry_point_merges_ckpt_hparams_but_does_not_load_weights(tmp_path, monkeypatch):
    from acorn.core import eval_stage
    from acorn.stages.edge_classifier.models.filter import Filter
    from acorn.utils import eval_utils
    from acorn.utils.loading_utils import save_pyg

    monkeypatch.chdir(tmp_path)
    fill = 0.123
    ckpt_hp = train_hparams(tmp_path / "train_input", tmp_path / "train_stage", "A")
    ck = write_ckpt(tmp_path / "ckpts" / "best-A.ckpt", ckpt_hp, fill=fill)
    stage = tmp_path / "stage"
    os.makedirs(stage / "testset")
    save_pyg(toy_event("000000003"), str(stage / "testset" / "event000000003.pyg"))  # an already-inferred event

    seen = []
    monkeypatch.setattr(
        eval_utils, "p08_spy_plot", lambda module, plot_config, config: seen.append((module, dict(config))),
        raising=False,
    )
    cfg = dict(stage="edge_classifier", model="Filter", stage_dir=str(stage), data_split=[0, 0, 1],
               plots={"p08_spy_plot": {}})
    cfg_file = tmp_path / "eval.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    eval_stage.evaluate(str(cfg_file), checkpoint=ck)  # what `acorn eval eval.yaml -c best-A.ckpt` runs
    (module, used), = seen
    assert used["dataset"] == "valset"  # CLI default, not testset
    assert used["hard_cuts"] == TRAIN_HARD_CUTS and used["ckpt_tag"] == "A"  # merged from the checkpoint
    weights = [p.detach() for p in module.net.parameters()]
    assert not all(torch.all(w == fill) for w in weights)  # freshly initialised network: weights NOT loaded

    # contrast: the stage's own evaluate() loads the weights when it is handed the checkpoint
    seen.clear()
    Filter.evaluate({**ckpt_hp, **cfg, "dataset": "testset"}, checkpoint=ck)
    (module, _), = seen
    assert all(torch.all(p.detach() == fill) for p in module.net.parameters())
