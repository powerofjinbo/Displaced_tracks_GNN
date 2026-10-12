"""Stage-1 D: vertex-aware multi-task edge classifier (brief §6D).

VertexTripletIGNN = TripletIGNN plus a second edge head trained to recognise "cross-daughter" edges: edges joining
hits of the two different HNL daughters of one decay (the same barcode vertex, different particles; truth from
hnl.models.truth_lookup, used only as a training target). These edges are the ones that merge the daughter pair in
connected components. The head shares the whole message-passing trunk, so the vertex topology shapes the edge
representation (multi-task), and its score is saved with the edge scores (`edge_vtx_scores`) so that track building
can veto edges that look like cross-daughter links (vertex consistency fed back into the track decisions).

Loss = main weighted BCE (acorn's, unchanged) + vtx_weight * class-balanced BCE of the auxiliary head
(mean over positives + mean over negatives, / 2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from class_resolver import ClassResolver
from torch_geometric.data import Dataset

from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.utils import make_mlp

from hnl.models.triplet_gnn import TripletIGNN

_LOOKUP = None


def _lookup():
    global _LOOKUP
    if _LOOKUP is None:
        from hnl.models.truth_lookup import TruthLookup
        _LOOKUP = TruthLookup()
    return _LOOKUP


class VertexGraphDataset(GraphDataset):
    """Adds hit truth (hit_cat, hit_absd0, hit_rv, hit_vtx) and the cross-daughter edge label edge_xd."""

    def preprocess_event(self, event):
        event = _lookup().decorate(event)
        a, b = event.edge_index
        v = event.hit_vtx
        pid = event.hit_particle_id
        event.edge_xd = (v[a] >= 0) & (v[a] == v[b]) & (pid[a] != pid[b])
        return super().preprocess_event(event)


class VertexTripletIGNN(TripletIGNN):
    def __init__(self, hparams):
        hparams = dict(hparams)
        hparams["dataset_class"] = "VertexGraphDataset"
        super().__init__(hparams)
        self.dataset_resolver = ClassResolver([GraphDataset, VertexGraphDataset], base=Dataset, default=GraphDataset)
        H = self.hparams["hidden"]
        self.vertex_head = make_mlp(3 * H, [H] * self.hparams["nb_edge_layer"] + [1],
                                    hidden_activation=self.hparams.get("hidden_activation", "SiLU"),
                                    output_activation=None, layer_norm=self.hparams.get("layernorm", True))
        self._aux = None

    def edge_representation_hook(self, x, src, dst, e):
        self._aux = self.vertex_head(torch.cat([x[src], x[dst], e], dim=1)).squeeze(-1)

    def loss_function(self, output, batch, balance="proportional"):
        loss, pos, neg = super().loss_function(output, batch, balance)
        if self._aux is None or not hasattr(batch, "edge_xd"):
            return loss, pos, neg
        y = batch.edge_xd.float()
        l_all = F.binary_cross_entropy_with_logits(self._aux, y, reduction="none")
        p, n = y > 0.5, y < 0.5
        aux = 0.5 * ((l_all[p].mean() if p.any() else l_all.new_zeros(())) + (l_all[n].mean() if n.any() else 0.0))
        return loss + float(self.hparams.get("vtx_weight", 1.0)) * aux, pos, neg

    def shared_evaluation(self, batch, batch_idx):
        out = super().shared_evaluation(batch, batch_idx)
        if self._aux is not None:
            out["batch"].edge_vtx_scores = torch.sigmoid(self._aux).detach()
        return out
