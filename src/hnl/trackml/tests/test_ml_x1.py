"""X1 (plan D3): the cluster features as metric-learning inputs, and the resume behaviour of the TrackML metric-learning
sampler.

- hnl.trackml.events.cluster_feature: the logs of ncells, charge (floored), span0, span1; kept by `subset`.
- hnl.trackml.ml.fs_features / fs_event_bytes: the feature store carries exactly the config's cluster features, and
  acorn's input matrix built from it (prefix map, node_scales, NaN -> 0) equals the Embedder's input matrix bit for bit,
  so training (feature store) and inference (Embedder) see the same inputs. S1's config adds nothing.
- hnl.trackml.ml._VirtualEpochSampler: under the runner's single-device DDPStrategy its order equals S1's (the S1
  sampler inside Lightning's shuffling DistributedSamplerWrapper); a run stopped mid-epoch (the PreemptCallback
  pattern) and resumed from its checkpoint trains on the same event sequence as an uninterrupted run; resuming from the
  end-of-epoch last.ckpt written after such a stop would skip the rest of the epoch (why slurm/trackml/_common.sh
  resume_ckpt prefers preempt.ckpt).

    pytest src/hnl/trackml/tests/test_ml_x1.py
"""

import gzip
import io
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hnl.trackml import events as E
from hnl.trackml.split import REPO
from hnl.trackml.tests.test_events import _toy_event

X1_CONFIG = os.path.join(REPO, "configs", "trackml", "ml_train_x1.yaml")
S1_CONFIG = os.path.join(REPO, "configs", "trackml", "ml_train.yaml")


def _x1_event(seed=0) -> E.Event:
    """The toy event of test_events with cluster values (pixel-like fractional charge on half the hits, an anomalous
    charge, a zero charge) and a full particles table."""
    ev = _toy_event(seed)
    rng = np.random.default_rng(seed + 100)
    n = ev.n
    ncells = rng.integers(1, 40, n).astype(np.uint16)
    span0 = np.minimum(ncells, rng.integers(1, 20, n)).astype(np.uint16)
    span1 = np.minimum(ncells, rng.integers(1, 30, n)).astype(np.uint16)
    charge = np.where(np.arange(n) % 2 == 0, ncells.astype(np.float32), rng.uniform(0.02, 3.0, n)).astype(np.float32)
    charge[3] = 716689.0
    charge[5] = 0.0
    p = ev.particles.copy()
    counts = pd.Series(ev.pid[ev.pid > 0]).value_counts()
    p["eta"] = np.linspace(-2, 2, len(p)).astype(np.float32)
    p["n_hits"] = counts.reindex(p.particle_id).fillna(0).to_numpy().astype(np.int16)
    from dataclasses import replace
    return replace(ev, ncells=ncells, charge=charge, span0=span0, span1=span1, particles=p)


def test_cluster_feature_values():
    ev = _x1_event()
    np.testing.assert_array_equal(E.cluster_feature(ev, "log_ncells"),
                                  np.log(ev.ncells.astype(np.float64)).astype(np.float32))
    np.testing.assert_array_equal(E.cluster_feature(ev, "log_span1"),
                                  np.log(ev.span1.astype(np.float64)).astype(np.float32))
    lc = E.cluster_feature(ev, "log_charge")
    assert lc.dtype == np.float32
    assert lc[5] == np.float32(np.log(E.CHARGE_FLOOR))                   # zero charge -> floor
    assert lc[3] == np.float32(np.log(716689.0)) and lc[3] < 14          # anomaly compressed
    assert np.all(np.isfinite(lc))
    for f in E.CLUSTER_FEATURES:                                         # node_input dispatches
        np.testing.assert_array_equal(E.node_input(ev, f), E.cluster_feature(ev, f))
    np.testing.assert_array_equal(E.node_input(ev, "r"), ev.r)
    with pytest.raises(KeyError):
        E.cluster_feature(ev, "ncells")
    from dataclasses import replace
    with pytest.raises(ValueError):
        E.cluster_feature(replace(ev, span0=None), "log_span0")
    bad = ev.ncells.copy()
    bad[0] = 0
    with pytest.raises(ValueError):
        E.cluster_feature(replace(ev, ncells=bad), "log_ncells")


def test_subset_keeps_spans():
    ev = _x1_event(1)
    sub = E.with_condition(ev, "nonoise")
    assert sub.n < ev.n
    np.testing.assert_array_equal(sub.span0, ev.span0[sub.full_rows])
    np.testing.assert_array_equal(sub.span1, ev.span1[sub.full_rows])
    np.testing.assert_array_equal(E.cluster_feature(sub, "log_charge"), E.cluster_feature(ev, "log_charge")[sub.rows])
    from dataclasses import replace
    assert E.with_condition(replace(ev, span0=None, span1=None), "nonoise").span0 is None


def test_fs_features_from_configs():
    from hnl.trackml.ml import fs_features
    s1 = yaml.safe_load(open(S1_CONFIG))
    x1 = yaml.safe_load(open(X1_CONFIG))
    assert fs_features(s1["node_features"]) == []
    assert fs_features(x1["node_features"]) == list(E.CLUSTER_FEATURES)
    assert len(x1["node_scales"]) == len(x1["node_features"])
    # X1 changes nothing but the inputs
    for k in set(s1) | set(x1):
        if k not in ("node_features", "node_scales"):
            assert s1.get(k) == x1.get(k), k
    assert x1["node_features"][:3] == s1["node_features"] and x1["node_scales"][:3] == s1["node_scales"]
    with pytest.raises(ValueError):
        fs_features(["r", "phi", "z", "hit_ncells"])


def _load_fs(data: bytes):
    return torch.load(io.BytesIO(gzip.decompress(data)), weights_only=False)


def _acorn_inputs(g, cfg: dict) -> torch.Tensor:
    """What acorn's GraphConstructionStage (preprocess_graph: prefix map, scale_features) and MetricLearning
    (get_input_data) feed the network for a feature-store event."""
    import copy
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config, add_variable_name_prefix_in_pyg
    hp = add_variable_name_prefix_in_config(copy.deepcopy(cfg))
    g = add_variable_name_prefix_in_pyg(g)
    for f, s in zip(hp["node_features"], hp["node_scales"]):
        g[f] = g[f] / s
    x = torch.stack([g[f] for f in hp["node_features"]], dim=-1).float()
    x[x != x] = 0
    return x


def test_feature_store_equals_embedder_inputs():
    from hnl.trackml.ml import fs_event_bytes, fs_features
    ev = _x1_event(2)
    for path in (X1_CONFIG, S1_CONFIG):
        cfg = yaml.safe_load(open(path))
        g = _load_fs(fs_event_bytes(ev, fs_features(cfg["node_features"])))
        keys = set(g.keys())
        assert {f"hit_{f}" in keys for f in E.CLUSTER_FEATURES} == {path == X1_CONFIG}
        x_train = _acorn_inputs(g, cfg)
        x_inf = E.Embedder(None, hparams=cfg, seed=0).inputs(ev)
        assert x_train.shape == (ev.n, len(cfg["node_features"]))
        assert torch.equal(x_train, x_inf), path


def test_stage_fs_parallel_same_content(tmp_path, monkeypatch):
    """stage_fs with nproc > 1 writes the same events (decompressed content) as nproc = 1 (fork here so that the
    monkeypatched reader reaches the workers; production spawns them, checked on real events by the smoke job)."""
    from hnl.trackml import ml
    evs = {100 + i: _x1_event(i) for i in range(3)}
    from dataclasses import replace
    evs = {e: replace(v, event=e) for e, v in evs.items()}

    def fake_iter(events, *a, **k):
        for e in events:
            yield evs[int(e)]
    monkeypatch.setattr(ml.E, "iter_events", fake_iter)
    monkeypatch.setattr(ml, "_PART", 1)
    out = {}
    for nproc in (1, 2):
        d = tmp_path / f"n{nproc}"
        info = ml.stage_fs(list(evs), str(d), features=list(E.CLUSTER_FEATURES), nproc=nproc, mp_context="fork")
        assert info["events"] == 3
        out[nproc] = {f: gzip.decompress(open(d / f, "rb").read()) for f in sorted(os.listdir(d))}
    assert list(out[1]) == list(out[2]) and len(out[1]) == 3
    for f in out[1]:
        a, b = (torch.load(io.BytesIO(out[k][f]), weights_only=False) for k in (1, 2))
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            va, vb = a[k], b[k]
            assert (torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb), k


# ------------------------------------------------------------------------------------------------ sampler and resume
N_EV, PER, EPOCHS = 17, 6, 4


class _S1Sampler(torch.utils.data.Sampler):
    """The S1 sampler as it ran in job 6688289 (before X1): the epoch's draws in cycle order; Lightning wrapped it."""

    def __init__(self, n, per_epoch, seed, module):
        self.n, self.per_epoch, self.seed, self.module = n, per_epoch, seed, module

    def __len__(self):
        return self.per_epoch

    def __iter__(self):
        from hnl.trackml.classifier import cycle_event
        g0 = int(self.module.current_epoch) * self.per_epoch
        return iter([cycle_event(g, self.n, self.seed) for g in range(g0, g0 + self.per_epoch)])


def test_sampler_order_equals_s1_wrapped_order():
    """The new sampler's order equals S1's: the old sampler inside Lightning's DistributedSamplerWrapper(shuffle=True,
    seed=PL_GLOBAL_SEED) with set_epoch(e)."""
    from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
    from hnl.trackml.ml import _VirtualEpochSampler

    class M:
        current_epoch = 0
    m = M()
    for seed, gseed in ((0, 0), (3, 5)):
        old = DistributedSamplerWrapper(_S1Sampler(N_EV, PER, seed, m), num_replicas=1, rank=0, shuffle=True,
                                        seed=gseed)
        new = _VirtualEpochSampler(N_EV, PER, seed, m, shuffle_seed=gseed)
        assert isinstance(new, torch.utils.data.DistributedSampler)       # Lightning leaves it unwrapped
        for e in range(6):
            m.current_epoch = e
            old.set_epoch(e)
            assert list(iter(new)) == list(iter(old)), (seed, e)


def _toy_fit(art, sampler_cls, stop_at=None, resume=None):
    """A toy LightningModule trained with sampler_cls under hnl.train.acorn_runner's Trainer setup (single-device
    DDPStrategy, LightningEnvironment; gloo on the CPU): (epoch, event) per training step. stop_at: save preempt.ckpt
    at that global step and stop (acorn_runner's PreemptCallback)."""
    import torch.distributed as dist
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return N_EV

        def __getitem__(self, i):
            return torch.tensor([float(i)])

    seen = []

    class Toy(LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def train_dataloader(self):
            return DataLoader(DS(), batch_size=1, num_workers=0, sampler=sampler_cls(N_EV, PER, 7, self))

        def val_dataloader(self):
            return DataLoader(DS(), batch_size=1)

        def training_step(self, b, i):
            seen.append((int(self.current_epoch), int(b.item())))
            return self.lin(b).sum()

        def validation_step(self, b, i):
            self.log("f1", float(self.current_epoch))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    class Stop(Callback):
        def on_train_batch_end(self, trainer, m, o, b, i):
            if stop_at is not None and trainer.global_step == stop_at and not trainer.should_stop:
                trainer.save_checkpoint(os.path.join(art, "preempt.ckpt"))
                trainer.should_stop = True

    seed_everything(0, workers=True)
    t = Trainer(accelerator="cpu", devices=1, num_nodes=1, max_epochs=EPOCHS, enable_progress_bar=False, logger=False,
                enable_model_summary=False, num_sanity_val_steps=0, plugins=[LightningEnvironment()],
                strategy=DDPStrategy(find_unused_parameters=False, static_graph=True, process_group_backend="gloo"),
                callbacks=[ModelCheckpoint(dirpath=art, monitor="f1", mode="max", save_last=True), Stop()])
    try:
        t.fit(Toy(), ckpt_path=resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return seen, int(t.global_step)


@pytest.mark.filterwarnings("ignore")
def test_sampler_resume_reproduces_s1_sequence(tmp_path):
    from hnl.trackml.ml import _VirtualEpochSampler, epoch_events
    ref, steps = _toy_fit(str(tmp_path / "s1"), _S1Sampler)              # S1's sampler, uninterrupted
    assert steps == EPOCHS * PER
    assert ref == [(e, x) for e in range(EPOCHS) for x in epoch_events(e, N_EV, PER, 7, 0)]
    new, _ = _toy_fit(str(tmp_path / "ref"), _VirtualEpochSampler)       # the new sampler, uninterrupted
    assert new == ref
    for at in (3, 6, 8, 11):                             # mid-epoch, at an epoch end, mid-epoch later
        art = str(tmp_path / f"s{at}")
        a, s1 = _toy_fit(art, _VirtualEpochSampler, stop_at=at)
        assert s1 == at
        ck = torch.load(os.path.join(art, "preempt.ckpt"), weights_only=False)
        assert ck["global_step"] == at
        b, s2 = _toy_fit(art, _VirtualEpochSampler, resume=os.path.join(art, "preempt.ckpt"))
        assert s2 == EPOCHS * PER and a + b == ref, at
        if at % PER:
            # last.ckpt was written after the stop at the same step, epoch marked as ended: resuming from it would lose
            # the rest of the epoch (why slurm/trackml/_common.sh resume_ckpt prefers preempt.ckpt on a tie)
            lk = torch.load(os.path.join(art, "last.ckpt"), weights_only=False)
            assert lk["global_step"] >= at
            art2 = str(tmp_path / f"t{at}")
            _toy_fit(art2, _VirtualEpochSampler, stop_at=at)
            c, s3 = _toy_fit(art2 + "_r", _VirtualEpochSampler, resume=os.path.join(art2, "last.ckpt"))
            assert s3 < EPOCHS * PER


def test_batches_done_without_trainer():
    from hnl.trackml.ml import _VirtualEpochSampler, batches_done, epoch_events

    class M:
        current_epoch = 2
    assert batches_done(M(), 10) == 0
    s = _VirtualEpochSampler(5, 4, 3, M(), shuffle_seed=1)
    assert list(iter(s)) == epoch_events(2, 5, 4, 3, 1)
    assert sorted(epoch_events(2, 5, 4, 3, 1)) == sorted(__import__("hnl.trackml.classifier", fromlist=["x"])
                                                         .cycle_event(g, 5, 3) for g in range(8, 12))
