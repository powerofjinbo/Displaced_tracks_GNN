"""Stage-1 B edge classifiers built on acorn's EdgeClassifierStage (same datasets, weighting, loss, metrics, precision
policy and checkpoints as the baseline; only the network differs). acorn itself is not modified.

TripletIGNN: an interaction network on the outward-directed graph with
  * node features: the baseline's scaled (r, phi, z) (absolute coordinates are kept, brief §6B);
  * edge input features: origin-free geometry (hnl.models.geometry_features.edge_features);
  * triplet (line-graph) messages between consecutive edges i -> j -> k sharing hit j, carrying the origin-free
    helix-consistency features (curvature, turning angle, s-z slope difference, 3D angle, length ratio);
  * residual node and edge updates, separate inward / outward aggregation, n_graph_iters recurrent iterations
    (shared weights, as acorn's recurrent default), with activation checkpointing of the encoders and of every
    iteration while gradients are recorded (as acorn's InteractionGNN; numerically the same, less memory).
With use_triplets: false the same network without triplet messages (the "residual IGNN" control).

Symmetry ladder (production-point invariance; the user's ttbar-only direction, DECISIONS.md):
  node_input: "absolute" (default: the baseline's scaled r, phi, z), "rz" (r, z: drops the azimuth, so the network is
  rotation invariant about the beam axis), "region" (a coarse detector-region one-hot: pixel / short strips / long
  strips x barrel / endcap, nothing finer), "none" (a constant);
  edge_geometry: "pointing" (default: dr, dphi, dz, transverse length, log 3D length; dr and dphi say how radial a
  segment is, i.e. whether it points to the beam line) or "invariant" (transverse length, dz, log 3D length:
  invariant under rotations about z and translations of the production point).
  With node_input "none" or "region", edge_geometry "invariant" and the (already invariant) triplet features, the
  only remaining dependence on the origin is the outward edge orientation (by R = sqrt(r^2 + z^2)).

Config keys (besides acorn's): hidden, n_graph_iters, nb_node_layer, nb_edge_layer, layernorm, hidden_activation,
use_triplets (default true), use_edge_geometry (default true), triplet_topk (per edge, keep the k straightest
continuations and the k straightest predecessors, so at most 2 k E triplets), max_triplets_per_node; with
use_triplets one of the two is required (all triplets of a ttbar graph, up to ~3 M, do not fit on a 24 GB GPU).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add

from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
from acorn.utils import make_mlp

from hnl.models import geometry_features as GF

N_EDGE_GEO = {"pointing": 5, "invariant": 3}
N_NODE_IN = {"rz": 2, "region": 6, "none": 1}
N_TRIP_GEO = 5


def _scale_edge_geo(f: torch.Tensor) -> torch.Tensor:
    # dr, dphi, dz, transverse length [mm] -> O(1); log 3D length already O(1)
    return torch.stack([f[:, 0] / 100.0, f[:, 1] * 10.0, torch.asinh(f[:, 2] / 100.0), f[:, 3] / 100.0,
                        f[:, 4] / 5.0], dim=1)


def _scale_edge_geo_invariant(f: torch.Tensor) -> torch.Tensor:
    # transverse length, dz, log 3D length of edge_features (columns 3, 2, 4)
    return torch.stack([f[:, 3] / 100.0, torch.asinh(f[:, 2] / 100.0), f[:, 4] / 5.0], dim=1)


def _region(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Coarse detector region one-hot (6): pixel / short strips / long strips x barrel / endcap (generic-detector
    extents, as hnl.trackbuild.merge_fragments._volume_from_radius)."""
    sub = torch.where(r < 200, 0, torch.where(r < 720, 1, 2))
    endcap = (torch.abs(z) > torch.where(r < 200, 600.0, 1200.0)).long()
    return torch.nn.functional.one_hot(sub * 2 + endcap, 6).float()


def _scale_trip_geo(f: torch.Tensor) -> torch.Tensor:
    # kappa [1/m] ~ 0.6/pT for true triplets; asinh keeps the fake tail bounded
    return torch.stack([torch.asinh(f[:, 0] / 0.5), f[:, 1] * 10.0, torch.asinh(f[:, 2] * 5.0), f[:, 3] * 5.0,
                        f[:, 4]], dim=1)


class TripletIGNN(EdgeClassifierStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        h = self.hparams
        H = h["hidden"]
        act = h.get("hidden_activation", "SiLU")
        ln = h.get("layernorm", True)
        mlp = lambda i, n, out=None: make_mlp(i, [H] * n + ([out] if out else []), hidden_activation=act,
                                               output_activation=None if out else h.get("output_activation"),
                                               layer_norm=ln, batch_norm=False)
        self.node_input = h.get("node_input", "absolute")
        self.edge_geometry = h.get("edge_geometry", "pointing")
        if self.node_input not in ("absolute", *N_NODE_IN) or self.edge_geometry not in N_EDGE_GEO:
            raise ValueError(f"node_input {self.node_input!r} / edge_geometry {self.edge_geometry!r}")
        n_node = len(h["node_features"]) if self.node_input == "absolute" else N_NODE_IN[self.node_input]
        self.use_triplets = h.get("use_triplets", True)
        if self.use_triplets and not (h.get("triplet_topk") or h.get("max_triplets_per_node")):
            raise ValueError("use_triplets needs triplet_topk or max_triplets_per_node (all triplets do not fit in memory)")
        self.use_edge_geo = h.get("use_edge_geometry", True)
        self.node_encoder = mlp(n_node, h["nb_node_layer"])
        self.edge_encoder = mlp(2 * H + (N_EDGE_GEO[self.edge_geometry] if self.use_edge_geo else 0), h["nb_edge_layer"])
        if self.use_triplets:
            self.trip_encoder = mlp(N_TRIP_GEO, h["nb_edge_layer"])
            self.trip_network = mlp(3 * H, h["nb_edge_layer"])
        self.node_network = mlp(3 * H, h["nb_node_layer"])
        self.edge_network = mlp((4 if self.use_triplets else 3) * H, h["nb_edge_layer"])
        self.output_edge_classifier = mlp(3 * H, h["nb_edge_layer"], out=1)

    def _physical(self, batch):
        feats = self.hparams["node_features"]
        scales = self.hparams.get("node_scales", [1.0] * len(feats))
        get = {f: batch[f] * s for f, s in zip(feats, scales)}
        r = get[[f for f in feats if f.endswith("r")][0]]
        phi = get[[f for f in feats if f.endswith("phi")][0]]
        z = get[[f for f in feats if f.endswith("z")][0]]
        return r.double(), phi.double(), z.double()

    def forward(self, batch, **kwargs):
        feats = self.hparams["node_features"]
        a, b = batch.edge_index
        r, phi, z = self._physical(batch)
        if self.node_input == "absolute":
            x = torch.stack([batch[f] for f in feats], dim=-1).float()
        elif self.node_input == "rz":
            x = torch.stack([r / 1000.0, z / 1000.0], dim=-1).float()
        elif self.node_input == "region":
            x = _region(r, z)
        else:
            x = torch.ones(r.shape[0], 1, dtype=torch.float32, device=r.device)
        R = torch.sqrt(r * r + z * z)
        fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
        src, dst = torch.where(fwd, a, b), torch.where(fwd, b, a)
        n = x.shape[0]
        ck = torch.is_grad_enabled()
        run = lambda f, *a: checkpoint(f, *a, use_reentrant=False) if ck else f(*a)
        x = run(self.node_encoder, x)
        enc_in = [x[src], x[dst]]
        if self.use_edge_geo:
            ef = GF.edge_features(r, phi, z, src, dst)
            enc_in.append((_scale_edge_geo(ef) if self.edge_geometry == "pointing" else _scale_edge_geo_invariant(ef)).float())
        e = run(self.edge_encoder, torch.cat(enc_in, dim=1))
        if self.use_triplets:
            e_in, e_out = GF.build_triplets(src, dst, n, self.hparams.get("max_triplets_per_node"))
            tf = GF.triplet_features(r, phi, z, src, dst, e_in, e_out)
            k = self.hparams.get("triplet_topk")
            if k:
                keep = GF.select_triplets(e_in, e_out, GF.straightness(tf), int(k))
                e_in, e_out, tf = e_in[keep], e_out[keep], tf[keep]
            t = run(self.trip_encoder, _scale_trip_geo(tf).float())
        else:
            e_in = e_out = t = None
        E = e.shape[0]

        def step(x, e, t):
            agg_in = scatter_add(e, dst, dim=0, dim_size=n)
            agg_out = scatter_add(e, src, dim=0, dim_size=n)
            x = x + self.node_network(torch.cat([x, agg_in, agg_out], dim=1))
            parts = [x[src], x[dst], e]
            if self.use_triplets:
                m = self.trip_network(torch.cat([e[e_in], e[e_out], t], dim=1))
                parts.append(scatter_add(m, e_in, dim=0, dim_size=E) + scatter_add(m, e_out, dim=0, dim_size=E))
            return x, e + self.edge_network(torch.cat(parts, dim=1))

        for _ in range(self.hparams["n_graph_iters"]):
            x, e = run(step, x, e, t)
        self.edge_representation_hook(x, src, dst, e)
        return self.output_edge_classifier(torch.cat([x[src], x[dst], e], dim=1)).squeeze(-1)

    def edge_representation_hook(self, x, src, dst, e):
        """Subclasses (multi-task heads) read the final node / edge states here."""
        return None
