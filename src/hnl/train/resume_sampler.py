"""Training order of the Stage-1 edge classifiers, resumable in the middle of an epoch (docs/feedback/
sampler_resume_decisions.md).

Under hnl.train.acorn_runner's single-device DDPStrategy, Lightning replaces the sampler of a training loader that is
not a DistributedSampler: acorn's own loader (EdgeClassifierStage.train_dataloader, the training set in file order) gets
DistributedSampler(dataset, shuffle=True, seed=PL_GLOBAL_SEED), and the epoch-tagged loader of the truncation-augmented
classifiers (hnl.models.augment) gets DistributedSamplerWrapper(_EpochTaggedSampler, shuffle=True, seed=PL_GLOBAL_SEED).
In both, epoch e trains position j of the order torch.randperm(n) drawn with a generator seeded PL_GLOBAL_SEED + e;
the tagged loader passes the index e * n + i (the dataset decodes the epoch for the augmentation draws).

EpochShuffleSampler gives exactly this order itself (`epoch_order`), as a one-replica DistributedSampler so that
Lightning does not wrap it, and starts at the first batch of the epoch not yet trained when Lightning resumes a
checkpoint written in the middle of an epoch (preempt.ckpt or periodic.ckpt): Lightning itself runs only the epoch's
remaining step count from a new iterator. Lightning makes the iterator of the first epoch of a resumed fit before it
applies its restart logic (FitLoop.setup_data) and, with num_workers > 0, the DataLoader takes its first indices at that
moment (so the old loaders resumed with the epoch counters of the checkpoint and the DistributedSampler's epoch still
0); ResumeDataLoader therefore makes that one iterator at the first batch, when the trainer's epoch and batch counters
are those of the batch Lightning is about to train. Every other iterator (every epoch of a run that is not resumed,
every later epoch of a resumed one) is made as before.

The resume point comes from Lightning's loop state (epoch and batch progress), which every checkpoint carries, so
checkpoints written before this module existed resume the same way. Without a trainer (or without that state) the
epoch's order starts at its first batch.

EpochCountersFix (a callback of hnl.train.acorn_runner) repairs Lightning's epoch counters after a resume from a
checkpoint written at the end of an epoch's last batch, so that a later mid-epoch checkpoint of the next epoch is
resumed as one (without it, Lightning reruns that epoch from its first batch).

A stop in the middle of an epoch ends the epoch for Lightning as well: it validates and runs the epoch-end hooks with
the weights of an unfinished epoch. epoch_completed tells the two apart, and CompletedEpochCheckpoint (acorn_runner's
every-5th-epoch files) saves only after completed epochs.
"""

from __future__ import annotations

import os
import types

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader


def epoch_order(e: int, n: int, shuffle_seed: int, tagged: bool = False) -> list[int]:
    """Indices of epoch e in training order: torch.randperm(n) with a generator seeded shuffle_seed + e, the order of
    Lightning's one-replica DistributedSampler(shuffle=True, seed=shuffle_seed) after set_epoch(e); tagged: index i
    as e * n + i (the order of DistributedSamplerWrapper around hnl.models.augment._EpochTaggedSampler)."""
    g = torch.Generator()
    g.manual_seed(int(shuffle_seed) + int(e))
    perm = torch.randperm(int(n), generator=g).tolist()
    return [int(e) * int(n) + i for i in perm] if tagged else perm


def _trainer(module):
    return getattr(module, "_trainer", None) if module is not None else None


def fit_resuming(module) -> bool:
    """True from the start of a fit resumed from a checkpoint until the end of its first epoch (Lightning's
    is_resuming; also restarting, which a checkpoint sets unless its epoch had ended)."""
    fl = getattr(_trainer(module), "fit_loop", None)
    return bool(getattr(fl, "is_resuming", False) or getattr(fl, "restarting", False))


def resume_point(module, n: int) -> tuple[int, int]:
    """(epoch, batches of it already trained) for the training iterator being made: the trainer's current epoch and,
    when Lightning resumes in the middle of an epoch, its batch count (0 < ready < batches per epoch); else 0. Without
    a trainer: (module.current_epoch, 0)."""
    e = int(getattr(module, "current_epoch", 0))
    tr = _trainer(module)
    if tr is None:
        return e, 0
    try:
        k = int(tr.fit_loop.epoch_loop.batch_progress.current.ready)
        nb = tr.num_training_batches
    except (AttributeError, TypeError, ValueError):
        return e, 0
    limit = int(n) if nb is None or nb == float("inf") else min(int(nb), int(n))
    return (e, k) if 0 < k < limit else (e, 0)


class EpochShuffleSampler(DistributedSampler):
    """Epoch e, batch b -> epoch_order(e, ...)[b] (see the module docstring). A DistributedSampler with one replica so
    that Lightning leaves it unwrapped; set_epoch is ignored (the epoch is the trainer's). shuffle_seed: PL_GLOBAL_SEED
    when the sampler is made, as Lightning reads it when it makes its own. Always yields n indices, the resumed epoch's
    remaining ones first (Lightning stops after the epoch's remaining count; the rest is never trained)."""

    def __init__(self, n: int, module, tagged: bool = False, shuffle_seed: int | None = None):
        super().__init__(range(int(n)), num_replicas=1, rank=0, shuffle=False, drop_last=False)
        self.n, self.module, self.tagged = int(n), module, bool(tagged)
        self.shuffle_seed = int(os.getenv("PL_GLOBAL_SEED", 0)) if shuffle_seed is None else int(shuffle_seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        e, k = resume_point(self.module, self.n)
        order = epoch_order(e, self.n, self.shuffle_seed, self.tagged)
        return iter(order[k:] + order[:k])


class _FirstBatchIterator:
    """Makes the DataLoader iterator when the first batch is requested."""

    def __init__(self, make):
        self._make, self._it = make, None

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = self._make()
        return next(self._it)


class ResumeDataLoader(DataLoader):
    """PyG DataLoader with an EpochShuffleSampler. In the first epoch of a fit resumed from a checkpoint the iterator is
    made at the first batch (after Lightning's restart logic), otherwise at once as in DataLoader."""

    def __iter__(self):
        if fit_resuming(getattr(self.sampler, "module", None)):
            return _FirstBatchIterator(super().__iter__)
        return super().__iter__()


def resumable_train_loader(dataset, module, num_workers: int, tagged: bool = False) -> ResumeDataLoader:
    """The training loader (batch size 1) with the resumable epoch order."""
    return ResumeDataLoader(dataset, batch_size=1, num_workers=num_workers,
                            sampler=EpochShuffleSampler(len(dataset), module, tagged=tagged))


def _acorn_train_dataloader(self):
    """acorn's EdgeClassifierStage.train_dataloader with the resumable order (same dataset, batch size, workers)."""
    if self.trainset is None:
        return None
    num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
    return resumable_train_loader(self.trainset, self, num_workers, tagged=False)


def use_resumable_train_loader(model) -> bool:
    """For an edge classifier whose training loader is acorn's own (EdgeClassifierStage.train_dataloader, e.g. acorn's
    InteractionGNN), replace it on this instance by the same loader with EpochShuffleSampler (tagged=False); the order
    of every epoch that is not resumed mid-epoch is the one Lightning's DistributedSampler gave. Returns whether it
    did; classes with their own train_dataloader (hnl.models.augment) are left as they are."""
    from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
    if not isinstance(model, EdgeClassifierStage):
        return False
    if type(model).train_dataloader is not EdgeClassifierStage.train_dataloader:
        return False
    model.train_dataloader = types.MethodType(_acorn_train_dataloader, model)
    return True


class EpochCountersFix(Callback):
    """Lightning 2.6 resuming a checkpoint written at the end of an epoch's last batch (preempt.ckpt, periodic.ckpt)
    counts that epoch as processed and completed (FitLoop.reset) but never as ready and started the epoch that follows
    (on_advance_start skips both on a mid-epoch restart). Every checkpoint written in the middle of that next epoch
    then reads as written at its start, and resuming it reruns the epoch from its first batch with its full step count.
    At the start of such a resume, count the next epoch as ready and started, as Lightning does at every other epoch
    start (its on_train_epoch_start hooks stay skipped, as on every mid-epoch restart). Any other start: no change."""

    def on_train_start(self, trainer, pl_module):
        fl = trainer.fit_loop
        try:
            ep = fl.epoch_progress
            hit = (fl.restarted_mid_epoch and fl.epoch_loop.restarted_on_train_batch_end
                   and fl.epoch_loop.batch_progress.is_last_batch and ep.total.ready == ep.total.processed)
        except AttributeError:
            return
        if hit:
            ep.increment_ready()
            ep.increment_started()


def epoch_completed(trainer) -> bool:
    """In an epoch-end hook: whether the epoch trained its last batch (False when a stop request, acorn_runner's
    PreemptCallback at a chunk's pre-timeout signal, ended it in its middle). Without loop state: True."""
    try:
        return bool(trainer.fit_loop.epoch_loop.batch_progress.is_last_batch)
    except AttributeError:
        return True


class CompletedEpochCheckpoint(ModelCheckpoint):
    """ModelCheckpoint whose epoch-end save (save_on_train_epoch_end=True) happens only after a completed epoch: a
    chunk stopped in the middle of epoch e would otherwise keep that state under epoch e's name, and the real end of
    the epoch would go to a versioned copy."""

    def on_train_epoch_end(self, trainer, pl_module):
        if epoch_completed(trainer):
            super().on_train_epoch_end(trainer, pl_module)


class PreemptStop(Callback):
    """acorn_runner's pre-timeout stop: at the end of the first training batch after flag() turns true, write the
    checkpoint `path` and stop the fit. Lightning validates on every stop request; in the middle of an epoch that
    validation is of no use (epoch-end checkpoints and the best checkpoint come from completed epochs) and, for the
    TrackML classifier (about 6.5 min per validation), it does not fit the 5 min before the time limit. So validation is
    switched off for the rest of the fit, unless the stop falls on the epoch's last batch (then the epoch-end validation
    and checkpoints run as without the stop)."""

    def __init__(self, path: str, flag):
        self.path, self.flag = path, flag

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.flag() and not trainer.should_stop:
            trainer.save_checkpoint(self.path)
            trainer.should_stop = True
            if not epoch_completed(trainer):
                trainer.check_val_every_n_epoch = 10 ** 9
