"""Exactly production-point-invariant edge classifier (the "L0" rung of the beam-line-blind ladder; the user's
ttbar-only direction, DECISIONS.md).

SymmetricTripletIGNN: the TripletIGNN trunk with every input invariant under the symmetry group of a helix in a
uniform field along z (translations in x, y, z; rotations about z) and under path reversal, so the network cannot
compute "pointing to the beam line" from anything:
  * no edge orientation: edges are unordered pairs; node aggregation sums over incident edges, and the two end nodes
    of an edge enter symmetrically (x_a + x_b, x_a * x_b);
  * edge inputs: log transverse length, asinh(|dz| / transverse length), log 3D length (unsigned);
  * orientation-free triplets: every unordered pair of distinct edges sharing a hit j (other ends i, k), with the
    symmetric helix-consistency features |kappa| (circle through i, j, k), |turning angle|, |dcot|, 3D angle and
    |log(l_T(jk) / l_T(ij))|; kept by top-k straightness per edge END (each edge keeps its k straightest
    continuations at each of its two hits), so at most 4 k E triplets;
  * node input per detector_input: "none" (a constant: exactly invariant) or "region" (coarse pixel / short / long
    strip x barrel / endcap one-hot: the L1 rung, which tells the network which subsystem a hit is in, nothing finer).
Same EdgeClassifierStage plumbing, loss, checkpoints and activation checkpointing as TripletIGNN.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add

from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
from acorn.utils import make_mlp

from hnl.models import geometry_features as GF
from hnl.models.triplet_gnn import _region

N_EDGE_SYM = 3
N_TRIP_SYM = 5


def undirected_triplets(a: torch.Tensor, b: torch.Tensor, n: int):
    """All unordered pairs of distinct edges sharing a hit. Returns (e1, e2, i, j, k): e1 < e2 edge ids, j the shared
    hit, i / k the other ends of e1 / e2."""
    E = a.shape[0]
    eid = torch.arange(E, device=a.device)
    src2, dst2, eid2 = torch.cat([a, b]), torch.cat([b, a]), torch.cat([eid, eid])
    e_in, e_out = GF.build_triplets(src2, dst2, n)
    e1, e2 = eid2[e_in], eid2[e_out]
    keep = e1 < e2                            # distinct edges, one copy of each unordered pair
    e_in, e_out = e_in[keep], e_out[keep]
    return eid2[e_in], eid2[e_out], src2[e_in], dst2[e_in], dst2[e_out]


def symmetric_triplet_features(xyz: torch.Tensor, i, j, k) -> torch.Tensor:
    """(T, 5): |kappa| [1/m], |turning angle|, |dcot|, 3D angle, |log(l_T(jk) / l_T(ij))|; symmetric under i <-> k."""
    a = xyz[j] - xyz[i]
    b = xyz[k] - xyz[j]
    c = xyz[k] - xyz[i]
    la = torch.linalg.vector_norm(a[:, :2], dim=1).clamp_min(1e-6)
    lb = torch.linalg.vector_norm(b[:, :2], dim=1).clamp_min(1e-6)
    lc = torch.linalg.vector_norm(c[:, :2], dim=1).clamp_min(1e-6)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    kappa = 2.0 * cross.abs() / (la * lb * lc)
    turn = torch.atan2(cross, dot).abs()
    dcot = (b[:, 2] / lb - a[:, 2] / la).abs()
    cos3 = (a * b).sum(1) / (torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)).clamp_min(1e-6)
    ang3 = torch.acos(cos3.clamp(-1 + 1e-7, 1 - 1e-7))
    return torch.stack([1e3 * kappa, turn, dcot, ang3, torch.log(lb / la).abs()], dim=1)


def topk_per_edge_end(e1, e2, j, a, score, k: int) -> torch.Tensor:
    """Keep a triplet if it is among the k lowest-score (straightest) triplets at either of its two edge ends: each
    (edge, end at hit j) ranks ALL triplets it belongs to, whichever of e1 / e2 the edge is, so at most k per edge end
    are chosen there and the choice does not depend on the edge numbering (ties aside)."""
    if len(score) == 0:
        return torch.zeros(0, dtype=torch.bool, device=score.device)
    end1 = (j != a[e1]).long()                # 0: hit j is end a of the edge, 1: end b
    end2 = (j != a[e2]).long()
    T = score.shape[0]
    group = torch.cat([e1 * 2 + end1, e2 * 2 + end2])
    sc2 = torch.cat([score, score])
    tid = torch.cat([torch.arange(T, device=score.device), torch.arange(T, device=score.device)])
    order = torch.argsort(sc2, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    g = group[order]
    start = torch.ones_like(g, dtype=torch.bool)
    start[1:] = g[1:] != g[:-1]
    first = torch.cummax(torch.where(start, torch.arange(len(g), device=g.device), torch.zeros_like(g)), 0).values
    rank = torch.arange(len(g), device=g.device) - first
    keep = torch.zeros(T, dtype=torch.bool, device=score.device)
    keep[tid[order[rank < k]]] = True
    return keep


def _scale_trip_sym(f: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.asinh(f[:, 0] / 0.5), f[:, 1] * 10.0, torch.asinh(f[:, 2] * 5.0), f[:, 3] * 5.0,
                        f[:, 4]], dim=1)


class SymmetricTripletIGNN(EdgeClassifierStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        h = self.hparams
        H = h["hidden"]
        act = h.get("hidden_activation", "SiLU")
        ln = h.get("layernorm", True)
        mlp = lambda i, n, out=None: make_mlp(i, [H] * n + ([out] if out else []), hidden_activation=act,
                                               output_activation=None if out else h.get("output_activation"),
                                               layer_norm=ln, batch_norm=False)
        self.detector_input = h.get("detector_input", "none")
        if self.detector_input not in ("none", "region"):
            raise ValueError(f"detector_input {self.detector_input!r}")
        if not h.get("triplet_topk"):
            raise ValueError("SymmetricTripletIGNN needs triplet_topk")
        n_node = 1 if self.detector_input == "none" else 6
        self.node_encoder = mlp(n_node, h["nb_node_layer"])
        self.edge_encoder = mlp(2 * H + N_EDGE_SYM, h["nb_edge_layer"])
        self.trip_encoder = mlp(N_TRIP_SYM, h["nb_edge_layer"])
        self.trip_network = mlp(3 * H, h["nb_edge_layer"])
        self.node_network = mlp(2 * H, h["nb_node_layer"])
        self.edge_network = mlp(4 * H, h["nb_edge_layer"])
        self.output_edge_classifier = mlp(3 * H, h["nb_edge_layer"], out=1)

    def _xyz(self, batch):
        feats = self.hparams["node_features"]
        scales = self.hparams.get("node_scales", [1.0] * len(feats))
        get = {f: batch[f] * s for f, s in zip(feats, scales)}
        r = get[[f for f in feats if f.endswith("r")][0]].double()
        phi = get[[f for f in feats if f.endswith("phi")][0]].double()
        z = get[[f for f in feats if f.endswith("z")][0]].double()
        return r, z, GF.to_xyz(r, phi, z)

    def forward(self, batch, **kwargs):
        a, b = batch.edge_index
        r, z, xyz = self._xyz(batch)
        n = xyz.shape[0]
        ck = torch.is_grad_enabled()
        run = lambda f, *args: checkpoint(f, *args, use_reentrant=False) if ck else f(*args)
        x0 = (torch.ones(n, 1, dtype=torch.float32, device=xyz.device) if self.detector_input == "none"
              else _region(r, z))
        x = run(self.node_encoder, x0)
        d = xyz[b] - xyz[a]
        lt = torch.linalg.vector_norm(d[:, :2], dim=1).clamp_min(1e-6)
        l3 = torch.linalg.vector_norm(d, dim=1).clamp_min(1e-6)
        geo = torch.stack([torch.log(lt / 10.0), torch.asinh(d[:, 2].abs() / lt), torch.log(l3 / 10.0)], dim=1).float()
        e = run(self.edge_encoder, torch.cat([x[a] + x[b], x[a] * x[b], geo], dim=1))
        e1, e2, i, j, k = undirected_triplets(a, b, n)
        tf = symmetric_triplet_features(xyz, i, j, k)
        keep = topk_per_edge_end(e1, e2, j, a, tf[:, 3] + 0.5 * tf[:, 2].clamp(max=10.0), int(self.hparams["triplet_topk"]))
        e1, e2, tf = e1[keep], e2[keep], tf[keep]
        t = run(self.trip_encoder, _scale_trip_sym(tf).float())
        E = e.shape[0]

        def step(x, e, t):
            agg = scatter_add(e, a, dim=0, dim_size=n) + scatter_add(e, b, dim=0, dim_size=n)
            x = x + self.node_network(torch.cat([x, agg], dim=1))
            m = self.trip_network(torch.cat([e[e1] + e[e2], e[e1] * e[e2], t], dim=1))
            tm = scatter_add(m, e1, dim=0, dim_size=E) + scatter_add(m, e2, dim=0, dim_size=E)
            return x, e + self.edge_network(torch.cat([x[a] + x[b], x[a] * x[b], e, tm], dim=1))

        for _ in range(self.hparams["n_graph_iters"]):
            x, e = run(step, x, e, t)
        return self.output_edge_classifier(torch.cat([x[a] + x[b], x[a] * x[b], e], dim=1)).squeeze(-1)
