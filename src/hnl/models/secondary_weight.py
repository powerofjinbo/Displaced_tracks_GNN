"""The Canto classifier (L1 + inner-hit truncation, hnl.models.augment.TruncSymmetricTripletIGNN) with the true edges of
displaced ttbar particles upweighted in the training loss: the secondary-particle upweighting test of
docs/secondary_upweight_plan.md (supervisor feedback, 2026-10-14, item 4).

The ttbar events hold no K_S or Lambda decays; their displaced particles are B- and D-hadron decay products and
particles made in the detector simulation (photon conversions, hadronic interactions). The table has no production
point, so the class is defined by the truth-fit |d0| (hnl.models.truth_lookup), the quantity that defines the
displaced populations of the paper: in training events only (data_name "trainset"), every true edge whose first hit
belongs to a particle with |d0| >= secw_d0_mm gets weight secw_weight instead of 1, after the truncation of
hnl.models.augment (which keeps |d0|). Truth is a training weight only, never an input. Val and test graphs keep
acorn's natural weighting, so the checkpoint-selection loss is the reference's.

Config keys: secw_d0_mm, secw_weight (both required; configs/stage1/triplet_symregion_trunc_secw.yaml).
"""

from __future__ import annotations

import torch

from hnl.models.augment import TruncationGraphDataset, _trunc_class
from hnl.models.invariant_gnn import SymmetricTripletIGNN
from hnl.models.variants import _register


class TruncSecondaryWeightGraphDataset(TruncationGraphDataset):
    def construct_weighting(self, event):
        event = super().construct_weighting(event)
        if self.data_name != "trainset":
            return event
        from hnl.models.vertex_gnn import _lookup
        event = _lookup().decorate(event)
        a = event.edge_index[0]
        d0 = torch.nan_to_num(event.hit_absd0[a], nan=0.0)
        disp = event.edge_y.bool() & d0.ge(float(self.hparams["secw_d0_mm"]))
        event.edge_weights = torch.where(disp, event.edge_weights * float(self.hparams["secw_weight"]),
                                         event.edge_weights)
        for k in ("hit_cat", "hit_absd0", "hit_rv", "hit_vtx"):
            if k in event:
                del event[k]
        return event


_Base = _trunc_class(SymmetricTripletIGNN, "_TruncSymmetricTripletIGNN")


class TruncSecWeightSymmetricTripletIGNN(_Base):
    """TruncSymmetricTripletIGNN with the displaced true edges of training events upweighted (module docstring)."""

    def __init__(self, hparams):
        hparams = dict(hparams)
        for k in ("secw_d0_mm", "secw_weight"):
            if k not in hparams:
                raise ValueError(f"config key {k} is required")
        SymmetricTripletIGNN.__init__(self, {**hparams, "dataset_class": "TruncSecondaryWeightGraphDataset"})
        _register(self, TruncationGraphDataset, TruncSecondaryWeightGraphDataset)
