"""Run the acorn baseline stages (brief §5f) with explicit data, seeds, checkpoints and run metadata.

Rules (docs/acorn_pitfalls.md, docs/acorn_map.md): per-event files live only on node-local disk; checkpoints and
packed outputs on /ceph; every inference gets an explicit checkpoint and explicit input_dir / stage_dir / data_split /
hard_cuts / undirected; training uses acorn's trainer setup (single-device DDPStrategy, so training events are
reshuffled every epoch; edge classifiers continue that order after a mid-epoch resume, hnl.train.resume_sampler), a
periodic checkpoint every 30 min, and a pre-timeout checkpoint on SIGUSR1.

    python -m hnl.train.acorn_runner stage --arm ii --source fs --what train,valmon --dest $LOCAL/fs
    python -m hnl.train.acorn_runner train --stage ml --arm ii --seed 0 --run-dir $RUN --input-dir $LOCAL/fs
    python -m hnl.train.acorn_runner infer --stage ml --ckpt $CKPT --input-dir $LOCAL/fs --out-dir $LOCAL/ml \\
        [--r-infer 0.1 --knn-infer 1000]
    python -m hnl.train.acorn_runner pack --src $LOCAL/ml --dest $RUN/ml_graphs
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time

import numpy as np
import yaml

from hnl.data.acorn_store import SAMPLE_CODE, event_id, parse_event_id
from hnl.data.splits import read_split

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
FS_DIR = os.path.join(DERIVED, "acorn_fs", "v1")
TEMPLATES = {"ml": os.path.join(REPO, "configs", "baseline", "ml_train.yaml"),
             "gnn": os.path.join(REPO, "configs", "baseline", "gnn_train.yaml")}
N_VAL_MONITOR = 500
ACORN_SPLIT = {"train": "trainset", "valmon": "valset", "val": "valset", "val_all": "valset", "test": "testset",
               "test_all": "testset", "external": "testset"}
LABELED = ("ttbar", "hnl10_100", "hnl15_200")


# --------------------------------------------------------------------------------------------- arms and event lists
def arm_events(arm: str, what: str) -> list[tuple[str, int]]:
    """(sample, index) of one arm's events. what: train | valmon (checkpoint monitoring subset) | val (the arm's own
    val mix) | val_all / test_all (the val / test splits of all three labeled samples, whatever the arm) | test |
    external. Arm (iii) train: all ttbar train + HNL train events (hnl10_100 index < 5000, hnl15_200 index >= 5000),
    trimmed to the same count as ttbar (DECISIONS.md 2026-10-03)."""
    comps = {"i": ["ttbar"], "ii": ["hnl10_100", "hnl15_200"], "iii": ["ttbar", "hnl10_100", "hnl15_200"]}[arm]
    if what == "external":
        return [("hnl15_100", int(i)) for i in read_split("hnl15_100", "external")]
    if what in ("val_all", "test_all"):
        return [(s, int(i)) for s in LABELED for i in read_split(s, what[:-4])]
    split = {"train": "train", "valmon": "val", "val": "val", "test": "test"}[what]
    out: list[tuple[str, int]] = []
    if arm == "iii" and what == "train":
        tt = [("ttbar", int(i)) for i in read_split("ttbar", "train")]
        hnl = [("hnl10_100", int(i)) for i in read_split("hnl10_100", "train") if i < 5000] + \
              [("hnl15_200", int(i)) for i in read_split("hnl15_200", "train") if i >= 5000]
        return tt + hnl[:len(tt)]
    per = []
    for s in comps:
        per.append([(s, int(i)) for i in read_split(s, split)])
    if what == "valmon":
        k = N_VAL_MONITOR // len(per)
        per = [p[:k] for p in per]
    for p in per:
        out += p
    return out


def stage_events(events: list[tuple[str, int]], source: str, dest_split_dir: str, run_dir: str | None = None,
                 tar_name: str | None = None) -> int:
    """Extract the given events from packed tars into dest_split_dir (flat).
    source 'fs': the feature store tars <FS_DIR>/<sample>_<split>.tar (members event<ID>-graph.pyg.gz);
    source 'tar': one packed stage output <tar_name> (members event<ID>.pyg.gz)."""
    os.makedirs(dest_split_dir, exist_ok=True)
    want = {event_id(s, i) for s, i in events}
    tars = []
    if source == "fs":
        by_sample: dict[str, set[str]] = {}
        for s, i in events:
            by_sample.setdefault(s, set())
        for s in by_sample:
            for sp in ("train", "val", "test", "external"):
                p = os.path.join(FS_DIR, f"{s}_{sp}.tar")
                if os.path.exists(p):
                    tars.append(p)
    else:
        tars = [tar_name]
    n = 0
    for p in tars:
        with tarfile.open(p, "r") as tar:
            for m in tar:
                eid = m.name.split("event")[-1].split("-")[0].split(".")[0]
                if eid in want:
                    m.name = os.path.basename(m.name)
                    tar.extract(m, dest_split_dir, filter="data")
                    n += 1
    if n != len(want):
        raise RuntimeError(f"staged {n} of {len(want)} events into {dest_split_dir}")
    return n


def count_split_files(input_dir: str) -> list[int]:
    out = []
    for sp in ("trainset", "valset", "testset"):
        d = os.path.join(input_dir, sp)
        out.append(len([f for f in os.listdir(d) if f.endswith(".pyg.gz")]) if os.path.isdir(d) else 0)
    return out


# --------------------------------------------------------------------------------------------- metadata
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(extra: dict) -> dict:
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    meta = {"git": git("rev-parse", "HEAD"), "git_dirty_code": git("status", "--porcelain", "--", "src", "env",
                                                                    "configs", "slurm"),
            "env_lock_sha256": _sha256(os.path.join(REPO, "env", "lock.txt")),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "start_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover
        meta["gpu"] = None
    meta.update(extra)
    return meta


# --------------------------------------------------------------------------------------------- model classes
def model_class(stage: str, model: str):
    """acorn class by (stage, name), or one of our own classes by dotted path (e.g. hnl.models.triplet_gnn.TripletIGNN)."""
    if "." in model:
        import importlib
        mod, name = model.rsplit(".", 1)
        return getattr(importlib.import_module(mod), name)
    from acorn.core.core_utils import str_to_class
    return str_to_class(stage, model)


def checkpoint_class(ckpt: str):
    import torch
    hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    return model_class(hp["stage"], hp["model"])


# --------------------------------------------------------------------------------------------- training
class _Preempt:
    flag = False


def _on_usr1(signum, frame):
    _Preempt.flag = True


def train(stage: str, arm: str, seed: int, run_dir: str, input_dir: str, max_epochs: int | None,
          overrides: dict | None = None, resume: str | None = None, limit_train_batches=None,
          config: str | None = None) -> dict:
    import torch
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from acorn.core.throughput import ThroughputCallback
    from acorn.stages.graph_construction.models import utils as gc_utils
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config

    if stage == "ml" and not gc_utils.FRNN_AVAILABLE:
        raise RuntimeError("FRNN not available: refusing to train metric learning without it (acorn_pitfalls p03)")
    stage_dir = os.path.join(run_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    cfg = yaml.safe_load(open(config or TEMPLATES[stage]))
    cfg.update({"input_dir": input_dir, "stage_dir": stage_dir, "data_split": count_split_files(input_dir)})
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs
    cfg.update(overrides or {})
    cfg_path = os.path.join(stage_dir, "train_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = run_metadata({"stage": stage, "arm": arm, "seed": seed, "config": cfg_path,
                         "config_sha256": _sha256(cfg_path), "data_split": cfg["data_split"], "resume": resume})

    t0 = time.time()
    seed_everything(seed, workers=True)
    pcfg = add_variable_name_prefix_in_config(copy.deepcopy(cfg))
    cls = model_class(pcfg["stage"], pcfg["model"])
    model = cls(pcfg)
    resume_cbs = []
    if stage == "gnn":
        # edge classifiers: acorn's own training loader -> the same epoch order, resumable mid-epoch (the Trunc*
        # classes of hnl.models.augment have it already), and Lightning's epoch counters repaired after a resume from
        # an epoch's last batch (hnl.train.resume_sampler)
        from hnl.train.resume_sampler import EpochCountersFix, use_resumable_train_loader
        if use_resumable_train_loader(model):
            print("[train] acorn training loader with hnl.train.resume_sampler (same order, resumable)", flush=True)
        resume_cbs.append(EpochCountersFix())
    meta["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    metric = cfg.get("metric_to_monitor", "val_loss")
    mode = cfg.get("metric_mode", "min")
    art = os.path.join(stage_dir, "artifacts")
    # save_top_k=-1: no best file is ever deleted. With 1, the validation that Lightning runs when a chunk stops at the
    # pre-timeout signal (after preempt.ckpt is written) could delete the epoch-end best that the next chunk restores as
    # best_model_path (arm (iii) of the training-sample study lost its epoch-28 checkpoint that way; DECISIONS
    # 2026-10-07 ~21:15 UTC). best_model_path / best_model_score are unchanged by this.
    best_cb = ModelCheckpoint(dirpath=art, filename="best-{epoch}-{" + metric + ":.6f}", monitor=metric, mode=mode,
                              save_top_k=-1, save_last=True, auto_insert_metric_name=True)
    # HNL_PERIODIC_MIN (default 30): minutes of training between periodic.ckpt writes (TrackML S4: 10, DECISIONS 2026-10-09)
    periodic_cb = ModelCheckpoint(dirpath=os.path.join(art, "periodic"), filename="periodic",
                                  train_time_interval=dt.timedelta(minutes=float(os.environ.get("HNL_PERIODIC_MIN", 30))),
                                  save_top_k=1, enable_version_counter=False)
    # every 5th epoch kept (weights of the run at equal training length, e.g. for shorter Stage-1 screening runs); only
    # after completed epochs, not when a chunk's pre-timeout stop ends one (hnl.train.resume_sampler)
    from hnl.train.resume_sampler import CompletedEpochCheckpoint
    epochs_cb = CompletedEpochCheckpoint(dirpath=os.path.join(art, "epochs"), filename="epoch{epoch}", every_n_epochs=5,
                                         save_top_k=-1, save_on_train_epoch_end=True, auto_insert_metric_name=False)

    class EpochLog(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            m = {k: round(float(v), 6) for k, v in trainer.callback_metrics.items() if "/" not in k}
            print(f"[epoch] {trainer.current_epoch} step {trainer.global_step} t {time.time() - t0:.0f}s {m}",
                  flush=True)

    # pre-timeout stop: preempt.ckpt at the next batch end, then stop without validating an unfinished epoch
    # (hnl.train.resume_sampler.PreemptStop; until 2026-10-08 Lightning validated at every stop)
    from hnl.train.resume_sampler import PreemptStop
    preempt_cb = PreemptStop(os.path.join(art, "preempt.ckpt"), lambda: _Preempt.flag)

    signal.signal(signal.SIGUSR1, _on_usr1)
    # a plain single-process environment: no dependence on SLURM_NTASKS / srun, no Lightning auto-requeue; the
    # pre-timeout signal is handled by preempt_cb above
    plugins = [LightningEnvironment()]
    trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, max_epochs=cfg["max_epochs"],
                      callbacks=[best_cb, periodic_cb, epochs_cb, ThroughputCallback(), LearningRateMonitor("epoch"),
                                 preempt_cb, EpochLog(), *resume_cbs], enable_progress_bar=False,
                      logger=CSVLogger(save_dir=stage_dir), precision=cfg.get("precision", 32),
                      strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
                      default_root_dir=stage_dir, plugins=plugins, limit_train_batches=limit_train_batches)
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, ckpt_path=resume)
    meta.update({"end_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "fit_seconds": round(time.time() - t0, 1),
                 "epochs_completed": int(trainer.current_epoch), "global_step": int(trainer.global_step),
                 "state": "preempted" if _Preempt.flag else "done",
                 "best_ckpt": best_cb.best_model_path, "best_score": float(best_cb.best_model_score)
                 if best_cb.best_model_score is not None else None,
                 "last_ckpt": best_cb.last_model_path,
                 "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
    with open(os.path.join(stage_dir, f"run_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------------------------------- inference
def infer(stage: str, ckpt: str, input_dir: str, out_dir: str, overrides: dict | None = None,
          accelerator: str = "gpu") -> dict:
    """accelerator "cpu" only for GNN scores that feed learned track building when no GPU slot is free (the same
    model and inputs; float32 numerics may differ from the GPU at the 1e-6 level)."""
    import torch
    from pytorch_lightning import Trainer
    from acorn.stages.graph_construction.models import utils as gc_utils
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config

    if stage == "ml" and not gc_utils.FRNN_AVAILABLE:
        raise RuntimeError("FRNN not available: refusing to build graphs without it (acorn_pitfalls p03)")
    cls = checkpoint_class(ckpt)
    model = cls.load_from_checkpoint(ckpt, map_location=torch.device("cpu"))
    over = {"input_dir": input_dir, "stage_dir": out_dir, "data_split": count_split_files(input_dir),
            "hard_cuts": {}, "undirected": True, "skip_existing": False, "accelerator": accelerator, "devices": 1,
            "nodes": 1, "num_workers": [4, 4, 4]}
    if stage == "gnn":
        over["undirected_message_passing"] = True
    over.update(overrides or {})
    ck = model._hparams
    for k in ("node_features", "node_scales"):
        if k in over and list(over[k]) != list(ck.get(k, over[k])):
            raise ValueError(f"{k} differs from the checkpoint")
    model._hparams = add_variable_name_prefix_in_config({**ck, **over})
    model.setup(stage="predict")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    trainer = Trainer(accelerator=accelerator, devices=1, num_nodes=1, logger=False, default_root_dir=out_dir,
                      enable_progress_bar=False)
    with torch.inference_mode():
        trainer.predict(model)
    meta = run_metadata({"stage": stage, "ckpt": ckpt, "ckpt_sha256": _sha256(ckpt), "overrides": over,
                         "predict_seconds": round(time.time() - t0, 1)})
    with open(os.path.join(out_dir, f"infer_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def pack(src: str, dest: str) -> dict:
    """Pack every split directory of an acorn stage output into <dest>/<split>.tar (flat members)."""
    os.makedirs(dest, exist_ok=True)
    out = {}
    for sp in ("trainset", "valset", "testset"):
        d = os.path.join(src, sp)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".pyg.gz"))
        if not files:
            continue
        path = os.path.join(dest, f"{sp}.tar")
        with tarfile.open(path + ".tmp", "w") as tar:
            for f in files:
                tar.add(os.path.join(d, f), arcname=f)
        os.replace(path + ".tmp", path)
        out[sp] = len(files)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--arm", required=True)
    s.add_argument("--what", required=True, help="comma list of train,valmon,val,test,external")
    s.add_argument("--source", default="fs", choices=["fs", "tar"])
    s.add_argument("--tar-dir", default=None, help="for --source tar: directory holding <split>.tar files")
    s.add_argument("--dest", required=True)
    t = sub.add_parser("train")
    t.add_argument("--stage", required=True, choices=["ml", "gnn"])
    t.add_argument("--arm", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--input-dir", required=True)
    t.add_argument("--max-epochs", type=int, default=None)
    t.add_argument("--resume", default=None)
    t.add_argument("--limit-train-batches", type=float, default=None)
    t.add_argument("--set", action="append", default=[], help="key=yaml-value overrides")
    t.add_argument("--config", default=None, help="config template (default: configs/baseline/<stage>_train.yaml)")
    i = sub.add_parser("infer")
    i.add_argument("--stage", required=True, choices=["ml", "gnn"])
    i.add_argument("--ckpt", required=True)
    i.add_argument("--input-dir", required=True)
    i.add_argument("--out-dir", required=True)
    i.add_argument("--set", action="append", default=[])
    i.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    p = sub.add_parser("pack")
    p.add_argument("--src", required=True)
    p.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    over = lambda kv: {x.split("=", 1)[0]: yaml.safe_load(x.split("=", 1)[1]) for x in kv}
    if a.cmd == "stage":
        for w in a.what.split(","):
            ev = arm_events(a.arm, w)
            if a.source == "fs":
                n = stage_events(ev, "fs", os.path.join(a.dest, ACORN_SPLIT[w]))
            else:
                n = stage_events(ev, "tar", os.path.join(a.dest, ACORN_SPLIT[w]),
                                 tar_name=os.path.join(a.tar_dir, f"{ACORN_SPLIT[w]}.tar"))
            print(f"[stage] arm {a.arm} {w}: {n} events -> {os.path.join(a.dest, ACORN_SPLIT[w])}", flush=True)
    elif a.cmd == "train":
        lim = a.limit_train_batches
        if lim is not None and lim >= 1:
            lim = int(lim)
        train(a.stage, a.arm, a.seed, a.run_dir, a.input_dir, a.max_epochs, over(a.set), a.resume, lim, a.config)
    elif a.cmd == "infer":
        print(json.dumps(infer(a.stage, a.ckpt, a.input_dir, a.out_dir, over(a.set), a.accelerator), indent=1,
                         default=str))
    elif a.cmd == "pack":
        print(json.dumps(pack(a.src, a.dest)))


if __name__ == "__main__":
    main()
