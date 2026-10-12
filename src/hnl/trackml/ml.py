"""Canto-L metric learning on TrackML (docs/trackml_plan.md section 6.1).

TrackMLMetricLearning: acorn's MetricLearning unchanged (configs/trackml/ml_train.yaml: the baseline recipe) except the
training sampler: each virtual epoch draws events_per_epoch training events (default 1,000) from seeded permutations
that cycle through every training event (cycle c = global step // N, default_rng([ml_seed, c]).permutation; the same
rule as the classifier windows, hnl.trackml.classifier.cycle_event), so all events are used although acorn's own loader
does not shuffle. Within an epoch the draws are permuted by torch.randperm seeded PL_GLOBAL_SEED + epoch (`epoch_events`):
in S1 Lightning's DistributedSamplerWrapper (single-device DDPStrategy) applied this permutation; the sampler is now a
DistributedSampler that applies it itself, so the order is S1's and Lightning does not wrap it. The epoch is read from the
trainer when the epoch's iterator is created, and when Lightning resumes a mid-epoch checkpoint (preempt.ckpt /
periodic.ckpt) the iterator starts at the first batch of the epoch not yet run (Lightning itself would restart the
epoch's order and run only the remaining step count), so a resumed run trains on the same event sequence as an
uninterrupted one. Validation (valmon) is acorn's: every valmon event, r_train, knn_val.

Feature store: `stage_fs` writes acorn feature-store events (hnl.data.acorn_store.acorn_event: r, phi, z, x, y, eta,
hit_particle_id, true edges with their particle id, pT, eta and nhits) from the compact TrackML tables to node-local
disk, <dest>/<trainset|valset|testset>/event<eid>-graph.pyg.gz, eid = 9 * 10^8 + event. X1 (plan D3; configs/trackml/
ml_train_x1.yaml): the cluster features a config lists among its node_features (`fs_features`; hnl.trackml.events.
CLUSTER_FEATURES, the logs of ncells, charge, span0, span1) are added as float32 attributes hit_<name>, computed by
hnl.trackml.events.cluster_feature exactly as the Embedder computes them at inference. A config with r, phi, z only
(S1) gives the same files as before.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader

from acorn.stages.graph_construction.models.metric_learning import MetricLearning

from hnl.trackml import events as E
from hnl.trackml.classifier import cycle_event

EVENTS_PER_EPOCH = 1000
_PART = 25                                # events per parallel feature-store part (one table read each)


def batches_done(module, per_epoch: int) -> int:
    """Training batches of the current epoch already run when its iterator is made: > 0 only when Lightning resumes a
    mid-epoch checkpoint (the restored batch progress); 0 at the start of every epoch and without a trainer."""
    tr = getattr(module, "_trainer", None)
    try:
        k = int(tr.fit_loop.epoch_loop.batch_progress.current.ready)
    except AttributeError:
        return 0
    return k if 0 < k < per_epoch else 0


def epoch_events(e: int, n: int, per_epoch: int, seed: int, shuffle_seed: int) -> list[int]:
    """Training-event positions of virtual epoch e in training order: the epoch's draws cycle_event(e * per_epoch + j)
    (j < per_epoch) permuted by torch.randperm(per_epoch) with a generator seeded shuffle_seed + e. This is the order S1
    trained with: under hnl.train.acorn_runner's single-device DDPStrategy, Lightning wrapped the sampler in a
    DistributedSamplerWrapper(shuffle=True, seed=PL_GLOBAL_SEED), which permutes the sampler's list exactly so."""
    import torch
    draws = [cycle_event(g, n, seed) for g in range(e * per_epoch, (e + 1) * per_epoch)]
    gen = torch.Generator()
    gen.manual_seed(int(shuffle_seed) + int(e))
    return [draws[i] for i in torch.randperm(per_epoch, generator=gen).tolist()]


class _VirtualEpochSampler(DistributedSampler):
    """Epoch e, batch b -> training event epoch_events(e, ...)[b]. A DistributedSampler (one replica) so that Lightning
    does not wrap it again; it applies the permutation the wrapper applied in S1 itself (shuffle_seed: PL_GLOBAL_SEED,
    read when the iterator is made, as Lightning reads it). The iterator starts at b = batches_done, so a run resumed
    from a mid-epoch checkpoint trains on the same events as an uninterrupted one; it always yields per_epoch items
    (Lightning stops after the epoch's remaining count; the filler is never trained on)."""

    def __init__(self, n: int, per_epoch: int, seed: int, module, shuffle_seed: int | None = None):
        super().__init__(range(int(per_epoch)), num_replicas=1, rank=0, shuffle=False, drop_last=False)
        self.n, self.per_epoch, self.seed, self.module = int(n), int(per_epoch), int(seed), module
        self.shuffle_seed = shuffle_seed

    def __len__(self):
        return self.per_epoch

    def __iter__(self):
        e = int(self.module.current_epoch)
        k = batches_done(self.module, self.per_epoch)
        ss = int(os.getenv("PL_GLOBAL_SEED", 0)) if self.shuffle_seed is None else int(self.shuffle_seed)
        order = epoch_events(e, self.n, self.per_epoch, self.seed, ss)
        return iter(order[k:] + order[:k])


class TrackMLMetricLearning(MetricLearning):
    def __init__(self, hparams):
        super().__init__(hparams)
        self._sampler_seed = int(self.hparams.get("ml_seed", torch.initial_seed()))

    def training_step(self, batch, batch_idx):
        # HNL_TML_EVENT_LOG (smoke tests only): append "global_step epoch batch_idx event_id" per training step
        log = os.environ.get("HNL_TML_EVENT_LOG")
        if log:
            from acorn.utils.loading_utils import flatten_event_id
            with open(log, "a") as fh:
                fh.write(f"{self.global_step} {self.current_epoch} {batch_idx} {flatten_event_id(batch.event_id)}\n")
        return super().training_step(batch, batch_idx)

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        per = int(self.hparams.get("events_per_epoch", EVENTS_PER_EPOCH))
        return DataLoader(self.trainset, batch_size=1, num_workers=num_workers,
                          sampler=_VirtualEpochSampler(len(self.trainset), per, self._sampler_seed, self))


def fs_features(node_features) -> list[str]:
    """The X1 cluster features among a metric-learning config's node_features (acorn names, hit_ prefix optional), in
    config order; every other node feature must be one the base feature store carries (r, phi, z)."""
    out = []
    for f in node_features:
        name = str(f).removeprefix("hit_")
        if name in E.CLUSTER_FEATURES:
            out.append(name)
        elif name not in ("r", "phi", "z"):
            raise ValueError(f"node feature {f!r}: neither r, phi, z nor a cluster feature {E.CLUSTER_FEATURES}")
    return out


def fs_event_bytes(ev: E.Event, features=()) -> bytes:
    """One acorn feature-store event (hnl.data.acorn_store format) of a TrackML event, plus the cluster features
    `features` (hnl.trackml.events.CLUSTER_FEATURES) as float32 node attributes hit_<name>."""
    from hnl.data.acorn_store import _serialise, acorn_event
    hits = pd.DataFrame({"row": np.arange(ev.n), "x": ev.x, "y": ev.y, "z": ev.z, "r": ev.r, "phi": ev.phi,
                         "pid": ev.pid})
    edges = pd.DataFrame({"src": ev.true_edges[0], "dst": ev.true_edges[1]})
    parts = ev.particles[["particle_id", "pt", "eta", "n_hits"]]
    g = acorn_event(hits, edges, parts, ev.eid)
    for name in features:
        g[f"hit_{name}"] = torch.from_numpy(E.cluster_feature(ev, name))
    return _serialise(g)


def _stage_part(args) -> tuple[int, int]:
    events, dest_split_dir, condition, features = args
    n, nbytes = 0, 0
    for ev in E.iter_events(events):
        ev = E.with_condition(ev, condition)
        data = fs_event_bytes(ev, features)
        path = os.path.join(dest_split_dir, f"event{ev.eid}-graph.pyg.gz")
        with open(path + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(path + ".tmp", path)
        n += 1
        nbytes += len(data)
    return n, nbytes


def stage_fs(events, dest_split_dir: str, condition: str = "full", features=(), nproc: int = 1,
             mp_context: str = "spawn") -> dict:
    """Feature-store files of the given events (condition: plan section 3.5; cluster features: fs_event_bytes) into
    dest_split_dir; nproc > 1 writes parts of _PART events in parallel processes (the same files; spawned, since the
    parent already runs Arrow / torch threads)."""
    os.makedirs(dest_split_dir, exist_ok=True)
    t0 = time.time()
    events = [int(e) for e in events]
    features = list(features)
    parts = [(events[k:k + _PART], dest_split_dir, condition, features) for k in range(0, len(events), _PART)]
    if nproc > 1 and len(parts) > 1:
        with mp.get_context(mp_context).Pool(min(int(nproc), len(parts))) as pool:
            res = pool.map(_stage_part, parts, chunksize=1)
    else:
        res = [_stage_part(p) for p in parts]
    return {"events": sum(r[0] for r in res), "bytes": sum(r[1] for r in res), "seconds": round(time.time() - t0, 1),
            "dest": dest_split_dir, "condition": condition, "features": features, "nproc": int(nproc)}
