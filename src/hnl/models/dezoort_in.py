"""The DeZoort interaction network (IN, DeZoort et al., Comput. Softw. Big Sci. 5 (2021) 26) as an acorn edge
classifier, so that it runs on the same HNL graphs, splits, working points and track builders as every other model
here (the comparison the user asked for; the main project's IN numbers are on TrackML pixel graphs).

Ported from the main project's re-implementation (gnn-displaced repo, src/gd/models/dz_in.py, itself checked
against the original gnn_scout code with shared weights):

    m   = R1([x_dst, x_src, e])             relational model, (2 n_in + e_in) -> 4
    agg = sum of m over incoming edges
    x'  = O([x, agg])                       object model, (n_in + 4) -> 3
    s   = R2([x'_dst, x'_src, m])           relational model, 10 -> 1

each model Linear-ReLU-Linear-ReLU-Linear with `hidden` units (6,448 parameters at hidden 40, 1 round = the IN; the
main project's IN-L is hidden 128, 3 rounds, 127k parameters). Node inputs (r/1000, phi/pi, z/1000), edge inputs
(dr/1000, dphi/pi, dz/1000, dR in eta-phi), every edge doubled with negated edge features as in the original.
Differences forced by acorn's interface: the network returns one logit per stored edge, the minimum of its two
directions' logits (the original scored an edge by the minimum of the two directions' probabilities; sigmoid is
monotonic), and the loss is acorn's weighted BCE on that logit (the original trained both directions separately).
The graphs are acorn's metric-learning graphs of the full detector (not the original's truth-filtered pixel graphs).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage

NODE_LATENT, EDGE_LATENT = 3, 4


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


class DeZoortIN(EdgeClassifierStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        h = self.hparams
        hidden, rounds = int(h.get("dz_hidden", 40)), int(h.get("dz_rounds", 1))
        self.R1, self.O = nn.ModuleList(), nn.ModuleList()
        n, e = 3, 4
        for _ in range(rounds):
            self.R1.append(_mlp(2 * n + e, EDGE_LATENT, hidden))
            self.O.append(_mlp(n + EDGE_LATENT, NODE_LATENT, hidden))
            n, e = NODE_LATENT, EDGE_LATENT
        self.R2 = _mlp(2 * NODE_LATENT + EDGE_LATENT, 1, hidden)

    def _inputs(self, batch):
        feats = self.hparams["node_features"]
        scales = self.hparams.get("node_scales", [1.0] * len(feats))
        get = {f: batch[f] * s for f, s in zip(feats, scales)}
        r = get[[f for f in feats if f.endswith("r")][0]].float()
        phi = get[[f for f in feats if f.endswith("phi")][0]].float()
        z = get[[f for f in feats if f.endswith("z")][0]].float()
        return r, phi, z

    def forward(self, batch, **kwargs):
        r, phi, z = self._inputs(batch)
        x = torch.stack([r / 1000.0, phi / math.pi, z / 1000.0], dim=1)
        src, dst = batch.edge_index
        dphi = torch.remainder(phi[dst] - phi[src] + math.pi, 2 * math.pi) - math.pi
        eta = -torch.log(torch.tan(torch.atan2(r, z) / 2.0).clamp_min(1e-12))
        dR = torch.sqrt((eta[dst] - eta[src]) ** 2 + dphi ** 2)
        ea = torch.stack([(r[dst] - r[src]) / 1000.0, dphi / math.pi, (z[dst] - z[src]) / 1000.0, dR], dim=1)
        E = src.shape[0]
        s2, d2 = torch.cat([src, dst]), torch.cat([dst, src])          # doubled edges, negated features
        e = torch.cat([ea, -ea])
        h = x
        for R1, O in zip(self.R1, self.O):
            m = R1(torch.cat([h[d2], h[s2], e], dim=1))
            agg = torch.zeros(h.shape[0], m.shape[1], dtype=m.dtype, device=m.device)
            agg.index_add_(0, d2, m)
            h = O(torch.cat([h, agg], dim=1))
            e = m
        out = self.R2(torch.cat([h[d2], h[s2], e], dim=1)).squeeze(1)
        return torch.minimum(out[:E], out[E:])
