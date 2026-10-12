"""Resumable training order of the Stage-1 edge classifiers (hnl.train.resume_sampler; docs/feedback/
sampler_resume_decisions.md).

- Every epoch's order equals the order the runs trained with so far: Lightning's DistributedSampler around acorn's own
  loader (acorn's InteractionGNN, e.g. the capacity-matched IGNN) and DistributedSamplerWrapper around
  hnl.models.augment._EpochTaggedSampler (the truncation-augmented classifiers, e.g. l1tl), both shuffled with
  PL_GLOBAL_SEED + epoch; sampler by sampler, and in full fits under hnl.train.acorn_runner's trainer setup
  (single-device DDPStrategy, LightningEnvironment; gloo on the CPU) with num_workers 0 and 2.
- A fit stopped at a batch end with preempt.ckpt (acorn_runner's PreemptCallback) and resumed from it trains the same
  sequence as the uninterrupted fit: stops in the middle of an epoch, at an epoch's first and last batch, several stops
  in a row, limit_train_batches; with the truncation augmentation (the real TruncationGraphDataset) the augmented events
  are identical too.
- What the old code did after such a resume (described in the decisions file), pinned.
- Old checkpoints: one written by the old code resumes exactly with the new code; the real preempt.ckpt files of the
  training-sample study (read-only, copied) load into the real model class and continue at their batch; a checkpoint
  without loop state starts from the first batch without error.

    pytest tests/test_resume_sampler.py
"""

import hashlib
import os
import shutil
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")

from torch.utils.data import Dataset  # noqa: E402

# imported before any fit, as in the runs: Lightning records the arguments of DataLoader subclasses that exist when it
# requests a loader (to re-instantiate it with its sampler)
import torch_geometric.loader  # noqa: E402,F401
import hnl.models.augment  # noqa: E402,F401
import hnl.train.resume_sampler  # noqa: E402,F401

N, EPOCHS = 7, 4
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
TS_RUNS = {a: os.path.join(DATA, "runs", "stage1_final_learned_ts", "l1tl", f"arm{a}_s0") for a in ("ii", "iii")}


class _IdxDS(Dataset):
    """Item i -> tensor([i]); any index (epoch-tagged indices exceed len)."""

    def __init__(self, n=N):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([float(i)])


# ------------------------------------------------------------------------------------------------ loaders, old and new
def _old_acorn(m):
    """acorn's own training loader (EdgeClassifierStage.train_dataloader, unchanged; Lightning gives it its
    DistributedSampler)."""
    from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
    return EdgeClassifierStage.train_dataloader(m)


def _new_acorn(m):
    """What use_resumable_train_loader installs in its place."""
    from hnl.train.resume_sampler import _acorn_train_dataloader
    return _acorn_train_dataloader(m)


def _old_tagged(m):
    """hnl.models.augment's training loader as it was before 2026-10-06."""
    from torch_geometric.loader import DataLoader
    from hnl.models.augment import _EpochTaggedSampler
    return DataLoader(m.trainset, batch_size=1, num_workers=m.hparams.get("num_workers", [1, 1, 1])[0],
                      sampler=_EpochTaggedSampler(len(m.trainset), m))


def _new_tagged(m):
    """hnl.models.augment's training loader now (the method of every Trunc* class)."""
    from hnl.models import augment as A
    return A._trunc_class(object, "X").train_dataloader(m)


LOADERS = {"acorn": (_old_acorn, _new_acorn), "tagged": (_old_tagged, _new_tagged)}


def _expected(kind, n=N, epochs=EPOCHS, seed=0, per=None):
    """(epoch, global_step, index) of an uninterrupted fit: epoch_order of every epoch, its first `per` batches."""
    from hnl.train.resume_sampler import epoch_order
    per = n if per is None else per
    out = []
    for e in range(epochs):
        for j, x in enumerate(epoch_order(e, n, seed, tagged=kind == "tagged")[:per]):
            out.append((e, e * per + j, x))
    return out


# ------------------------------------------------------------------------------------------------ sampler level
@pytest.mark.parametrize("kind", ["acorn", "tagged"])
def test_order_equals_lightning_sampler(kind):
    """The new sampler's order equals what Lightning made of the old loader (its own _get_distributed_sampler, the
    function the trainer calls under the runner's single-device DDPStrategy), every epoch, several seeds and sizes."""
    from pytorch_lightning.trainer.connectors.data_connector import _get_distributed_sampler
    from pytorch_lightning.trainer.states import RunningStage
    from torch.utils.data import DistributedSampler
    from hnl.train.resume_sampler import EpochShuffleSampler, ResumeDataLoader

    class M:
        current_epoch = 0
        hparams = {"num_workers": [0, 0, 0]}

    old_f, new_f = LOADERS[kind]
    for n, gseed in ((1, 0), (7, 0), (16, 0), (101, 3), (15999, 0)):
        m = M()
        m.trainset = _IdxDS(n)
        old_dl, new_dl = old_f(m), new_f(m)
        assert not isinstance(old_dl.sampler, DistributedSampler)                 # Lightning replaces it
        assert isinstance(new_dl, ResumeDataLoader) and isinstance(new_dl.sampler, EpochShuffleSampler)
        assert isinstance(new_dl.sampler, DistributedSampler)                     # Lightning leaves it as it is
        new_dl.sampler.shuffle_seed = gseed
        old = _get_distributed_sampler(old_dl, shuffle=True, overfit_batches=0, mode=RunningStage.TRAINING,
                                       num_replicas=1, rank=0, seed=gseed)
        assert len(new_dl.sampler) == len(old) == n
        for e in (range(8) if n < 15999 else (0, 6, 29)):
            m.current_epoch = e
            old.set_epoch(e)
            new_dl.sampler.set_epoch(12345)                                           # ignored
            assert list(iter(new_dl.sampler)) == list(iter(old)), (kind, n, gseed, e)


def test_shuffle_seed_from_environment(monkeypatch):
    from hnl.train.resume_sampler import EpochShuffleSampler, epoch_order

    class M:
        current_epoch = 2
    monkeypatch.setenv("PL_GLOBAL_SEED", "5")
    s = EpochShuffleSampler(9, M(), tagged=True)
    assert s.shuffle_seed == 5 and list(s) == epoch_order(2, 9, 5, tagged=True)
    assert sorted(epoch_order(2, 9, 5, tagged=True)) == list(range(18, 27))


def test_resume_point_without_trainer_state():
    """No trainer, or a trainer object without Lightning's loop state: the epoch from the module, batch 0."""
    from hnl.train.resume_sampler import EpochShuffleSampler, epoch_order, fit_resuming, resume_point

    class M:
        current_epoch = 3
    m = M()
    assert resume_point(m, 10) == (3, 0) and not fit_resuming(m) and not fit_resuming(None)
    m._trainer = object()
    assert resume_point(m, 10) == (3, 0) and not fit_resuming(m)
    assert list(EpochShuffleSampler(10, m, shuffle_seed=1)) == epoch_order(3, 10, 1)


def test_use_resumable_train_loader_classes():
    """Installed for acorn's InteractionGNN (the capacity-matched IGNN's class), not for classes with their own
    loader (the Trunc* classes, which use it already)."""
    import yaml
    from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config
    from hnl.models.augment import TruncSymmetricTripletIGNN
    from hnl.train.acorn_runner import REPO
    from hnl.train.resume_sampler import EpochShuffleSampler, ResumeDataLoader, use_resumable_train_loader
    for name, cls, want in (("ignn_capacity.yaml", InteractionGNN, True),
                            ("triplet_symregion_trunc.yaml", TruncSymmetricTripletIGNN, False)):
        cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "stage1", name)))
        cfg.update({"input_dir": "/nonexistent", "stage_dir": "/nonexistent", "data_split": [5, 0, 0]})
        model = cls(add_variable_name_prefix_in_config(cfg))
        assert use_resumable_train_loader(model) is want, name
        model.trainset = _IdxDS(5)
        dl = model.train_dataloader()
        assert isinstance(dl, ResumeDataLoader) and isinstance(dl.sampler, EpochShuffleSampler)
        assert dl.sampler.tagged is (not want) and dl.num_workers == cfg["num_workers"][0] and dl.batch_size == 1
        assert dl.sampler.module is model
    assert use_resumable_train_loader(torch.nn.Linear(1, 1)) is False


# ------------------------------------------------------------------------------------------------ full fits
def _toy_module(loader, nw, n, seen):
    from pytorch_lightning import LightningModule

    class Toy(LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters({"num_workers": [nw, nw, nw]})
            self.lin = torch.nn.Linear(1, 1)
            self.trainset = _IdxDS(n)

        def train_dataloader(self):
            return loader(self)

        def val_dataloader(self):
            from torch_geometric.loader import DataLoader
            return DataLoader(_IdxDS(2), batch_size=1)

        def training_step(self, b, i):
            seen.append((int(self.current_epoch), int(self.global_step), int(b.item())))
            return self.lin(b).sum()

        def validation_step(self, b, i):
            self.log("f1", float(self.current_epoch))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)
    return Toy()


def _trainer(art, extra, max_epochs=EPOCHS, limit=None, fix=True, **kw):
    """hnl.train.acorn_runner's trainer setup for an edge classifier on the CPU (gloo); fix: its EpochCountersFix."""
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from hnl.train.resume_sampler import EpochCountersFix
    kw.setdefault("callbacks", [ModelCheckpoint(dirpath=art, monitor="f1", mode="max", save_last=True), *extra])
    if fix:
        kw["callbacks"] = [*kw["callbacks"], EpochCountersFix()]
    return Trainer(accelerator="cpu", devices=1, num_nodes=1, max_epochs=max_epochs, enable_progress_bar=False,
                   logger=False, enable_model_summary=False, num_sanity_val_steps=0, plugins=[LightningEnvironment()],
                   strategy=DDPStrategy(find_unused_parameters=False, static_graph=True, process_group_backend="gloo"),
                   limit_train_batches=limit, **kw)


class _StopAt:
    """acorn_runner's PreemptCallback, triggered at the end of the batch that makes global_step == step."""

    def __new__(cls, art, step):
        from pytorch_lightning import Callback

        class Stop(Callback):
            def on_train_batch_end(self, trainer, m, o, b, i):
                if step is not None and trainer.global_step == step and not trainer.should_stop:
                    trainer.save_checkpoint(os.path.join(art, "preempt.ckpt"))
                    trainer.should_stop = True
        return Stop()


def _fit(model_fn, art, stop=None, resume=None, limit=None, max_epochs=EPOCHS, fix=True):
    import torch.distributed as dist
    from pytorch_lightning import seed_everything
    seen = []
    seed_everything(0, workers=True)
    t = _trainer(art, [_StopAt(art, stop)], max_epochs=max_epochs, limit=limit, fix=fix)
    try:
        t.fit(model_fn(seen), ckpt_path=resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return seen, int(t.global_step)


def _chain(model_fns, art, stops, limit=None, fix=True):
    """Chunks: stop at each of `stops`, resume from preempt.ckpt (resume_ckpt's choice: largest step, preempt.ckpt on
    the tie with last.ckpt); model_fns[i] builds the module of chunk i (the last one for any further chunk)."""
    seq, resume = [], None
    for i, s in enumerate(list(stops) + [None]):
        a, gs = _fit(model_fns[min(i, len(model_fns) - 1)], art, stop=s, resume=resume, limit=limit, fix=fix)
        if s is not None:
            assert gs == s
            assert torch.load(os.path.join(art, "preempt.ckpt"), weights_only=False)["global_step"] == s
        seq += a
        resume = os.path.join(art, "preempt.ckpt")
    return seq


STOPS = [[3], [6], [7], [8], [10], [13], [6, 7, 8], [7, 10], [2, 3, 7, 12, 22]]   # N = 7: 7, 14, 21 end epochs


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("nw", [0, 2])
@pytest.mark.parametrize("kind", ["acorn", "tagged"])
def test_fit_order_and_resume(tmp_path, kind, nw):
    old_f, new_f = LOADERS[kind]
    old = lambda seen: _toy_module(old_f, nw, N, seen)
    new = lambda seen: _toy_module(new_f, nw, N, seen)
    ref, steps = _fit(old, str(tmp_path / "old"))                       # the old code, uninterrupted
    assert steps == EPOCHS * N and ref == _expected(kind)
    unint, _ = _fit(new, str(tmp_path / "new"))                         # the new code, uninterrupted: unchanged
    assert unint == ref
    for stops in STOPS:
        art = str(tmp_path / ("s" + "_".join(map(str, stops))))
        assert _chain([new], art, stops) == ref, stops


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("kind,nw", [("acorn", 2), ("tagged", 0)])
def test_resume_with_limit_train_batches(tmp_path, kind, nw):
    """limit_train_batches (the screening runs' LIMIT_TRAIN): an epoch is the first `limit` positions of its order."""
    old_f, new_f = LOADERS[kind]
    lim = 5
    ref, steps = _fit(lambda s: _toy_module(old_f, nw, N, s), str(tmp_path / "old"), limit=lim)
    assert steps == EPOCHS * lim and ref == _expected(kind, per=lim)
    new = lambda s: _toy_module(new_f, nw, N, s)
    assert _fit(new, str(tmp_path / "new"), limit=lim)[0] == ref
    for stops in ([2], [4], [5], [6], [3, 9, 15]):
        art = str(tmp_path / ("s" + "_".join(map(str, stops))))
        assert _chain([new], art, stops, limit=lim) == ref, stops


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("nw", [0, 2])
def test_old_code_after_a_resume(tmp_path, nw):
    """What the old code did (docs/feedback/sampler_resume_decisions.md): after a resume in the middle of epoch
    e, the remaining step count of e ran from the start of a permutation: with num_workers 0 the epoch's own
    (PL_GLOBAL_SEED + e), with num_workers > 0 (the runs: 4) that of epoch 0 (Lightning made the iterator before it set
    the sampler's epoch); the augmentation draws were those of epoch e. After a stop at an epoch's last batch, with
    num_workers > 0 the next epoch e + 1 ran with the tags (augmentation draws) of epoch e in epoch 0's order; so did
    the first epoch of a run continued from last.ckpt written at an epoch end."""
    from hnl.train.resume_sampler import epoch_order
    old = lambda s: _toy_module(_old_tagged, nw, N, s)
    p = lambda e: epoch_order(e, N, 0)
    seq = _chain([old], str(tmp_path / "mid"), [10])                   # epoch 1, 3 batches done
    resumed = [x for e, g, x in seq if g >= 10 and e == 1]
    assert resumed == [7 + i for i in (p(1) if nw == 0 else p(0))[:4]]
    seq = _chain([old], str(tmp_path / "end"), [7])                    # last batch of epoch 0
    ep1 = [x for e, g, x in seq if e == 1]
    assert ep1 == ([7 + i for i in p(1)] if nw == 0 else [0 + i for i in p(0)])
    # a run continued from last.ckpt written at the end of epoch 1 (e.g. the 30 -> 60 epoch continuations): with
    # num_workers > 0 its first epoch (2) ran in epoch 0's order with the tags of epoch 1
    art = str(tmp_path / "last")
    _fit(old, art, max_epochs=2, fix=False)
    seq, _ = _fit(old, art, resume=os.path.join(art, "last.ckpt"), fix=False)
    assert [x for e, g, x in seq if e == 2] == ([14 + i for i in p(2)] if nw == 0 else [7 + i for i in p(0)])
    assert [x for e, g, x in seq if e == 3] == [21 + i for i in p(3)]


@pytest.mark.filterwarnings("ignore")
def test_epoch_counters_after_a_stop_at_an_epoch_end(tmp_path):
    """Why EpochCountersFix: without it, after a resume from a stop at an epoch's last batch, a stop in the middle of
    the next epoch is resumed as the start of that epoch (Lightning reruns it with its full step count)."""
    new = lambda s: _toy_module(_new_acorn, 0, N, s)
    ref = _expected("acorn")
    seq = _chain([new], str(tmp_path / "nofix"), [7, 10], fix=False)
    assert seq != ref and len(seq) == len(ref) + 3 and seq[10][1:] == (10, ref[7][2])     # epoch 1 again from batch 0
    assert _chain([new], str(tmp_path / "fix"), [7, 10]) == ref


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("limit,stops", [(None, [3, 7, 8, 17]), (5, [2, 5, 6, 12])])
def test_epoch_end_saves_after_completed_epochs_only(tmp_path, limit, stops):
    """A stop in the middle of an epoch runs Lightning's epoch-end hooks too: epoch_completed is False there and
    CompletedEpochCheckpoint (acorn_runner's every-5th-epoch files) saves nothing, so its file for epoch e holds the
    state after the epoch's last batch; stops in the middle of an epoch, at its last and at its first batch."""
    import torch.distributed as dist
    from pytorch_lightning import Callback, seed_everything
    from hnl.train.resume_sampler import CompletedEpochCheckpoint, epoch_completed
    per = N if limit is None else limit
    calls = []

    class Rec(Callback):
        def on_train_epoch_end(self, trainer, m):
            calls.append((int(trainer.current_epoch), int(trainer.global_step), epoch_completed(trainer)))

    art, ep, resume = str(tmp_path / "art"), str(tmp_path / "epochs"), None
    for s in [*stops, None]:
        seed_everything(0, workers=True)
        cb = CompletedEpochCheckpoint(dirpath=ep, filename="epoch{epoch}", every_n_epochs=1, save_top_k=-1,
                                      save_on_train_epoch_end=True, auto_insert_metric_name=False)
        t = _trainer(art, [_StopAt(art, s), Rec(), cb], limit=limit)
        try:
            t.fit(_toy_module(_new_acorn, 0, N, []), ckpt_path=resume)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        resume = os.path.join(art, "preempt.ckpt")
    assert [(e, g) for e, g, c in calls if c] == [(e, (e + 1) * per) for e in range(EPOCHS)]
    assert [g for e, g, c in calls if not c] == [g for g in stops if g % per]
    assert sorted(os.listdir(ep)) == [f"epoch{e}.ckpt" for e in range(EPOCHS)]
    for e in range(EPOCHS):
        assert torch.load(os.path.join(ep, f"epoch{e}.ckpt"), weights_only=False)["global_step"] == (e + 1) * per


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("stops", [[3, 7, 10], [6, 7, 8], [2, 3, 7, 12, 22]])
def test_preempt_stop_validates_completed_epochs_only(tmp_path, stops):
    """acorn_runner's PreemptStop: no validation after a stop in the middle of an epoch (Lightning would run one), the
    epoch-end validation when the stop falls on an epoch's last batch, and the same training sequence as an
    uninterrupted fit."""
    import torch.distributed as dist
    from pytorch_lightning import Callback, seed_everything
    from hnl.train.resume_sampler import PreemptStop
    ref = _expected("acorn")
    val_steps, seq, resume = [], [], None
    art = str(tmp_path / "art")
    for s in [*stops, None]:
        flag = [False]

        class Trigger(Callback):                 # sets the flag at the end of the batch that makes global_step == s
            def on_train_batch_end(self, trainer, m, o, b, i):
                if s is not None and trainer.global_step == s:
                    flag[0] = True

        class Rec(Callback):
            def on_validation_epoch_end(self, trainer, m):
                if not trainer.sanity_checking:
                    val_steps.append(int(trainer.global_step))

        seed_everything(0, workers=True)
        t = _trainer(art, [Trigger(), PreemptStop(os.path.join(art, "preempt.ckpt"), lambda: flag[0]), Rec()])
        seen = []
        try:
            t.fit(_toy_module(_new_acorn, 0, N, seen), ckpt_path=resume)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        if s is not None:
            assert int(t.global_step) == s
            assert torch.load(os.path.join(art, "preempt.ckpt"), weights_only=False)["global_step"] == s
        seq += seen
        resume = os.path.join(art, "preempt.ckpt")
    assert seq == ref
    ends = [(e + 1) * N for e in range(EPOCHS)]
    assert all(g in ends for g in val_steps), val_steps
    assert sorted(set(val_steps)) == ends, val_steps


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("nw", [0, 2])
def test_resume_from_last_ckpt_at_an_epoch_end(tmp_path, nw):
    """A run that ended at an epoch end (last.ckpt written there; e.g. a shorter max_epochs) and is continued from
    last.ckpt, then stopped in the middle of an epoch and resumed: the uninterrupted sequence."""
    for kind, (old_f, new_f) in LOADERS.items():
        new = lambda s: _toy_module(new_f, nw, N, s)
        art = str(tmp_path / kind)
        a, gs = _fit(new, art, max_epochs=2)
        assert gs == 2 * N
        b, gs = _fit(new, art, stop=17, resume=os.path.join(art, "last.ckpt"))
        assert gs == 17
        c, gs = _fit(new, art, resume=os.path.join(art, "preempt.ckpt"))
        assert a + b + c == _expected(kind), kind


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("nw", [0, 2])
def test_old_checkpoint_resumes_with_new_code(tmp_path, nw):
    """preempt.ckpt written by the old code, the next chunks with the new code: the uninterrupted sequence (the old
    chunk started its epochs itself, so its batches were the epoch's first ones). Without loop state (not written by
    any of our runs) Lightning starts at epoch 0 and the order starts at its first batch: no error."""
    for kind, (old_f, new_f) in LOADERS.items():
        old = lambda s: _toy_module(old_f, nw, N, s)
        new = lambda s: _toy_module(new_f, nw, N, s)
        ref = _expected(kind)
        for stops in ([3], [10], [10, 12]):
            art = str(tmp_path / f"{kind}_{'_'.join(map(str, stops))}")
            assert _chain([old, new], art, stops) == ref, (kind, stops)
    art = str(tmp_path / "noloops")
    _fit(lambda s: _toy_module(_old_tagged, nw, N, s), art, stop=10)
    ck = torch.load(os.path.join(art, "preempt.ckpt"), weights_only=False)
    del ck["loops"]
    torch.save(ck, os.path.join(art, "noloops.ckpt"))
    seq, steps = _fit(lambda s: _toy_module(_new_tagged, nw, N, s), art + "_r",
                      resume=os.path.join(art, "noloops.ckpt"))
    assert seq == _expected("tagged") and steps == EPOCHS * N


# ------------------------------------------------------------------------------------------------ augmentation
def _graph_module(root, nw, seen, old=False):
    """A toy model with the real Trunc* training path: _trunc_class's train_dataloader over the real
    TruncationGraphDataset (truncation p = 0.5, trunc_seed unset as in the runs: torch.initial_seed()); old: the
    training loader as it was before 2026-10-06."""
    from pytorch_lightning import LightningModule
    from hnl.models import augment as A

    class Base(LightningModule):
        def __init__(self, hparams):
            super().__init__()
            self.save_hyperparameters(hparams)
            self.lin = torch.nn.Linear(1, 1)
            self.trainset = None

        def setup(self, stage="fit"):
            self.trainset = A.TruncationGraphDataset(root, "trainset", N, "fit", hparams=dict(self.hparams))

        def val_dataloader(self):
            from torch_geometric.loader import DataLoader
            return DataLoader(_IdxDS(2), batch_size=1)

        def training_step(self, b, i):
            h = hashlib.sha1()
            for k in ("edge_index", "hit_r", "edge_y"):
                h.update(b[k].numpy().tobytes())
            ev = b.event_id[0] if isinstance(b.event_id, (list, tuple)) else b.event_id
            seen.append((int(self.current_epoch), int(self.global_step), str(ev), int(b.num_nodes), h.hexdigest()))
            return self.lin(b.hit_r[:1].view(1, 1)).sum()

        def validation_step(self, b, i):
            self.log("f1", float(self.current_epoch))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    Toy = A._trunc_class(Base, "ToyTrunc")
    if old:
        Toy = type("ToyTruncOld", (Toy,), {"train_dataloader": lambda self: _old_tagged(self)})
    return Toy({"node_features": ["hit_r", "hit_phi", "hit_z"], "node_scales": [1000, 3.14, 1000], "trunc_p": 0.5,
                "num_workers": [nw, nw, nw]})


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("nw", [0, 2])
def test_truncation_draws_reproduce(tmp_path, nw):
    """With the real dataset and augmentation: the uninterrupted fit gives the same augmented events as the old code
    path, and chunked fits give the same augmented events as the uninterrupted one."""
    from test_augment import _toy_event
    root = tmp_path / "graphs"
    os.makedirs(root / "trainset")
    for k in range(N):
        torch.save(_toy_event(eid=100000000 + k, seed=k), root / "trainset" / f"event{100000000 + k}.pyg")
    model = lambda s: _graph_module(str(root), nw, s)
    ref, steps = _fit(model, str(tmp_path / "ref"))
    assert steps == EPOCHS * N
    # the events in the expected order, and some epochs truncate an event differently
    order = [(e, g, str(100000000 + x % N)) for e, g, x in _expected("tagged")]
    assert [r[:3] for r in ref] == order
    per_event = {}
    for e, g, ev, nn, h in ref:
        per_event.setdefault(ev, set()).add(h)
    assert any(len(v) > 1 for v in per_event.values())
    # the old code path (same dataset, _EpochTaggedSampler wrapped by Lightning) gives the same augmented events
    old_ref, _ = _fit(lambda s: _graph_module(str(root), nw, s, old=True), str(tmp_path / "old"))
    assert old_ref == ref
    for stops in ([4], [7], [11], [3, 10, 20]):
        art = str(tmp_path / ("s" + "_".join(map(str, stops))))
        assert _chain([model], art, stops) == ref, stops


# ------------------------------------------------------------------------------------------------ real checkpoints
def _copy_cpu(src, dst):
    """A CPU copy of a checkpoint that a running chunk may rewrite (read-only on the original; retried)."""
    for attempt in range(5):
        try:
            shutil.copyfile(src, dst + ".raw")
            ck = torch.load(dst + ".raw", map_location="cpu", weights_only=False)
            torch.save(ck, dst)
            os.remove(dst + ".raw")
            return ck
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("arm", ["ii", "iii"])
def test_real_preempt_checkpoint(tmp_path, arm):
    """A real preempt.ckpt of the training-sample study (written by the old code) under the new code: the real model
    class (TruncSymmetricTripletIGNN, hyper-parameters from the checkpoint) and acorn_runner's trainer setup resume at
    the checkpoint's epoch and batch and train epoch_order(e)[k], [k + 1], ... (a stand-in training set of the run's
    size gives the epoch-tagged indices; the network is not run). The old code path from the same checkpoint (pinned):
    epoch 0's order from its first position, as in the running chunks."""
    src = os.path.join(TS_RUNS[arm], "gnn", "artifacts", "preempt.ckpt")
    if not os.path.exists(src):
        pytest.skip(f"no {src}")
    from pytorch_lightning import Callback
    from hnl.models import augment as A
    from hnl.train.resume_sampler import epoch_order
    ck = _copy_cpu(src, str(tmp_path / "preempt.ckpt"))
    hp = dict(ck["hyper_parameters"])
    n = int(hp["data_split"][0])
    loops = ck["loops"]["fit_loop"]
    e = int(loops["epoch_progress"]["current"]["processed"])
    k = int(loops["epoch_loop.batch_progress"]["current"]["ready"])
    assert hp["model"].endswith("TruncSymmetricTripletIGNN") and 0 < k < n and ck["global_step"] >= k
    n_rec = 3

    def run(loader):
        seen = []

        class Probe(A.TruncSymmetricTripletIGNN):
            def setup(self, stage="fit"):
                self.trainset = _IdxDS(n)

            def train_dataloader(self):
                return loader(self)

            def training_step(self, batch, batch_idx):
                seen.append((int(self.current_epoch), int(self.global_step), int(batch.item())))
                return sum((p * 0).sum() for p in self.parameters())

        class Enough(Callback):
            def on_train_batch_end(self, trainer, m, o, b, i):
                if len(seen) >= n_rec:
                    trainer.should_stop = True

        import torch.distributed as dist
        from pytorch_lightning import seed_everything
        seed_everything(0, workers=True)                                   # the run's seed (PL_GLOBAL_SEED 0)
        t = _trainer(str(tmp_path), [], max_epochs=int(hp["max_epochs"]), callbacks=[Enough()],
                     enable_checkpointing=False, limit_val_batches=0)
        model = Probe(hp)
        assert model.hparams["num_workers"][0] == 4                         # the runs' loader workers
        try:
            t.fit(model, ckpt_path=str(tmp_path / "preempt.ckpt"))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return seen

    new = run(lambda self: A.TruncSymmetricTripletIGNN.train_dataloader(self))
    gs = int(ck["global_step"])
    assert new == [(e, gs + j, x) for j, x in enumerate(epoch_order(e, n, 0, tagged=True)[k:k + n_rec])]
    old = run(_old_tagged)
    assert old == [(e, gs + j, e * n + i) for j, i in enumerate(epoch_order(0, n, 0)[:n_rec])]
