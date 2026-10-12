"""Inner-hit truncation augmentation for edge classifiers trained on ttbar only (Stage 1, design synthesis step 1).

In a uniform field along z, a prompt track whose first m hits are removed is exactly a track produced between hit m
and hit m + 1: to a production-point-invariant classifier (hnl.models.invariant_gnn) it is a displaced track. The
truncation probe (hnl.eval.truncation_probe) shows that a ttbar-trained GNN rejects such late-starting tracks (it
learned that tracks start in the pixels), which invariance of the inputs alone cannot remove; this augmentation removes
that pattern from the training distribution.

Training events only (data_name "trainset"; val and test untouched): every particle with >= TRUNC_MIN_HITS hits is
truncated with probability trunc_p (default 0.3): its first m hits in R = sqrt(r^2 + z^2) order, m uniform in
{1, ..., n - 5}, are removed together with every graph edge touching them (hnl.eval.truncation_probe.truncate: node,
edge and track tensors filtered, nodes renumbered, truth_map re-pointed); all other hits, edges and labels are
unchanged (the radius graph is per hit pair and its knn cap never binds, so the remaining graph is what the graph
stage would have built). Truth chooses which hits to drop; it never enters as an input. p and the m distribution are
fixed here and never tuned on HNL data.

Randomness is deterministic per (seed, epoch, event): the training loader passes each index tagged with its epoch,
e * n + i, and the dataset decodes it. The order of epoch e is Lightning's shuffled order of that epoch (until
2026-10-06: _EpochTaggedSampler inside Lightning's DistributedSamplerWrapper; since then hnl.train.resume_sampler,
the same order, which also continues an epoch interrupted by a mid-epoch checkpoint at its next batch). Model classes:
TruncIGNN, TruncTripletIGNN, TruncSymmetricTripletIGNN (configs/stage1/*_trunc.yaml).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data

from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
from acorn.utils.loading_utils import add_variable_name_prefix_in_pyg, load_pyg

from hnl.models.invariant_gnn import SymmetricTripletIGNN
from hnl.models.triplet_gnn import TripletIGNN
from hnl.models.variants import _register
from hnl.train.resume_sampler import resumable_train_loader

TRUNC_P = 0.3
TRUNC_MIN_HITS = 6
TRUNC_KEEP = 5                          # hits always left on a truncated particle


def truncation_mask(pid: np.ndarray, R: np.ndarray, rng: np.random.Generator, p: float = TRUNC_P) -> np.ndarray:
    """Hits to remove (bool, per hit): for each particle (pid != 0) with >= TRUNC_MIN_HITS hits, with probability p,
    its m innermost hits in R, m uniform in {1, ..., n - TRUNC_KEEP}. Particles are visited in increasing pid."""
    remove = np.zeros(len(pid), dtype=bool)
    if p <= 0:
        return remove
    parts, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
    order = np.lexsort((R, inv))                         # hits grouped by particle, innermost first
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    for k in range(len(parts)):
        if parts[k] == 0 or cnt[k] < TRUNC_MIN_HITS:
            continue
        if rng.random() >= p:
            continue
        m = int(rng.integers(1, cnt[k] - TRUNC_KEEP + 1))
        remove[order[start[k]:start[k] + m]] = True
    return remove


class _EpochTaggedSampler(Sampler):
    """Sequential indices (acorn's order) tagged with the trainer's current epoch: epoch * n + i. The training sampler
    of the Trunc* classes until 2026-10-06 (Lightning wrapped it in DistributedSamplerWrapper(shuffle=True)); kept as
    the reference of that order (tests/test_resume_sampler.py) and used by hnl.trackml.classifier."""

    def __init__(self, n: int, module):
        self.n = n
        self.module = module

    def __len__(self):
        return self.n

    def __iter__(self):
        e = int(self.module.current_epoch)              # 0 without a trainer
        return iter(range(e * self.n, (e + 1) * self.n))


class TruncationGraphDataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_seed = int(self.hparams.get("trunc_seed", torch.initial_seed()))

    def __getitem__(self, idx):
        # PyG's Dataset maps an int index through range(len): epoch-tagged training indices bypass it
        if self.data_name == "trainset" and isinstance(idx, (int, np.integer)):
            data = self.get(int(idx))
            return data if self.transform is None else self.transform(data)
        return super().__getitem__(idx)

    def get(self, idx):
        if self.data_name != "trainset":
            return super().get(idx)
        epoch, i = divmod(int(idx), len(self.input_paths))
        event = load_pyg(self.input_paths[i], map_location=torch.device("cpu"), weights_only=False)
        event = truncate_event(event, self.base_seed, epoch, float(self.hparams.get("trunc_p", TRUNC_P)))
        # from here on as GraphDataset.get
        if (self.transform is not None) and (self.stage != "test"):
            event = Data(**event.to_dict())
        if (not self.hparams.get("variable_with_prefix")) or self.hparams.get("add_variable_name_prefix_in_pyg"):
            event = add_variable_name_prefix_in_pyg(event)
        if not self.preprocess:
            return event
        event = self.preprocess_event(event)
        if (self.transform is not None) and (self.stage != "test"):
            event = self.transform(event)
        return event


def truncate_event(event, base_seed: int, epoch: int, p: float = TRUNC_P):
    """The augmented copy of one stored (unprefixed) metric-learning graph for (seed, epoch, event)."""
    from hnl.eval.evaluate_gnn import _event_id
    from hnl.eval.truncation_probe import truncate
    rng = np.random.default_rng([base_seed, epoch, int(_event_id(event))])
    pid = event.hit_particle_id.numpy()
    R = torch.sqrt(event.r.double() ** 2 + event.z.double() ** 2).numpy()
    remove = truncation_mask(pid, R, rng, p)
    if not remove.any():
        return event
    return truncate(event, remove)


def _trunc_class(base, name: str):
    def __init__(self, hparams):
        hparams = dict(hparams)
        hparams["dataset_class"] = "TruncationGraphDataset"
        base.__init__(self, hparams)
        _register(self, TruncationGraphDataset)

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        return resumable_train_loader(self.trainset, self, num_workers, tagged=True)
    return type(name, (base,), {"__init__": __init__, "train_dataloader": train_dataloader,
                                "__doc__": f"{base.__name__} trained with inner-hit truncation augmentation."})


TruncIGNN = _trunc_class(InteractionGNN, "TruncIGNN")
TruncTripletIGNN = _trunc_class(TripletIGNN, "TruncTripletIGNN")
TruncSymmetricTripletIGNN = _trunc_class(SymmetricTripletIGNN, "TruncSymmetricTripletIGNN")
