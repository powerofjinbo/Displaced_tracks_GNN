"""Stage-1 training variants of acorn's InteractionGNN that change only the truth labels or the loss (brief §6A, §6E).
acorn is not modified: these are subclasses with their own dataset classes.

  * IGNNSamePid: same-particle relabelling (brief §6A, "relabel truth so that same-particle skip edges count as
    positives"): every graph edge joining two hits of one particle is a positive in training and validation, not only
    the npz chain edges. Evaluation is unchanged (DM after CC against the particles; chain truth for edge metrics).
  * IGNNFocal: focal loss (gamma from the config key focal_gamma, default 2) in place of the weighted BCE; same
    proportional normalisation by the sum of weights.
  * IGNNDisplacedWeight (brief §6E, static group weighting): true edges of particles with truth |d0| >=
    displaced_d0_mm (default 10) weighted displaced_weight (default 3) in the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from class_resolver import ClassResolver
from torch_geometric.data import Dataset

from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN


class SamePidGraphDataset(GraphDataset):
    def preprocess_event(self, event):
        a, b = event.edge_index
        pid = event.hit_particle_id
        event.edge_y = (pid[a] == pid[b]) & (pid[a] != 0)
        return super().preprocess_event(event)


def _register(model, *datasets):
    model.dataset_resolver = ClassResolver([GraphDataset, *datasets], base=Dataset, default=GraphDataset)


class IGNNSamePid(InteractionGNN):
    def __init__(self, hparams):
        hparams = dict(hparams)
        hparams["dataset_class"] = "SamePidGraphDataset"
        super().__init__(hparams)
        _register(self, SamePidGraphDataset)


class IGNNFocal(InteractionGNN):
    def loss_function(self, output, batch, balance="proportional"):
        gamma = float(self.hparams.get("focal_gamma", 2.0))
        y = batch.edge_y.float()
        w = batch.edge_weights.abs()
        neg = ((batch.edge_y == 0) & (batch.edge_weights != 0)) | (batch.edge_weights < 0)
        pos = (batch.edge_y == 1) & (batch.edge_weights > 0)
        use = pos | neg
        target = torch.where(pos, torch.ones_like(y), torch.zeros_like(y))
        bce = F.binary_cross_entropy_with_logits(output, target, reduction="none")
        p = torch.sigmoid(output)
        pt = torch.where(pos, p, 1 - p)
        fl = (1 - pt).pow(gamma) * bce * w
        sow = w[use].sum()
        pl, nl = fl[pos].sum(), fl[neg].sum()
        return (pl + nl) / sow, pl.detach() / sow, nl.detach() / sow


class DisplacedWeightGraphDataset(GraphDataset):
    """Natural weighting, except that true edges of particles with truth |d0| >= displaced_d0_mm get weight
    displaced_weight (truth from hnl.models.truth_lookup; a training weight only, never an input)."""

    def construct_weighting(self, event):
        event = super().construct_weighting(event)
        from hnl.models.vertex_gnn import _lookup
        event = _lookup().decorate(event)
        a = event.edge_index[0]
        d0 = event.hit_absd0[a]
        disp = event.edge_y.bool() & torch.nan_to_num(d0, nan=0.0).ge(float(self.hparams.get("displaced_d0_mm", 10.0)))
        event.edge_weights = torch.where(disp, event.edge_weights * float(self.hparams.get("displaced_weight", 3.0)),
                                         event.edge_weights)
        return event


class IGNNDisplacedWeight(InteractionGNN):
    def __init__(self, hparams):
        hparams = dict(hparams)
        hparams["dataset_class"] = "DisplacedWeightGraphDataset"
        super().__init__(hparams)
        _register(self, DisplacedWeightGraphDataset)
