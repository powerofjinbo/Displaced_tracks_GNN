"""Origin-free geometric features of edges and triplets (brief §6B), in torch.

Hits are given in cylindrical coordinates (r, phi, z) [mm, rad, mm]. Edges are oriented outward (inner hit first,
see hnl.trackbuild.builders.orient). A triplet is a pair of consecutive edges (i -> j, j -> k) sharing the middle hit
j. Triplet features are invariant under rotations about the beam axis and translations in the transverse plane and
along z, so they do not prefer tracks from the origin:
  kappa      signed curvature [1/mm] of the circle through the three hits in the transverse plane
             (|kappa| ~ 0.6e-3 / pT[GeV] for a helix in 2 T, whatever its d0);
  turn       signed turning angle between the transverse segments i->j and j->k [rad];
  dcot       difference of the s-z slopes (dz/ds_T) of the two segments (~0 for a helix);
  angle3d    3D angle between the two segments [rad];
  len_ratio  log of the ratio of the transverse segment lengths.
"""

from __future__ import annotations

import torch


def to_xyz(r: torch.Tensor, phi: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def wrap(dphi: torch.Tensor) -> torch.Tensor:
    return torch.remainder(dphi + torch.pi, 2 * torch.pi) - torch.pi


def edge_features(r, phi, z, src, dst) -> torch.Tensor:
    """(E, 5): dr, dphi, dz, transverse length, log 3D length; rotation invariant (dphi is a difference)."""
    xyz = to_xyz(r, phi, z)
    d = xyz[dst] - xyz[src]
    lt = torch.linalg.vector_norm(d[:, :2], dim=1)
    l3 = torch.linalg.vector_norm(d, dim=1)
    return torch.stack([r[dst] - r[src], wrap(phi[dst] - phi[src]), z[dst] - z[src], lt, torch.log(l3 + 1e-3)], dim=1)


def build_triplets(src: torch.Tensor, dst: torch.Tensor, n_nodes: int, max_per_node: int | None = None):
    """All pairs (e_in, e_out) with dst[e_in] == src[e_out]. Returns (e_in, e_out) index tensors.
    max_per_node: keep at most this many in-edges and out-edges per middle node (by edge order), to bound memory."""
    dev = src.device
    E = src.shape[0]
    eid = torch.arange(E, device=dev)
    in_order = torch.argsort(dst, stable=True)
    out_order = torch.argsort(src, stable=True)
    in_cnt = torch.bincount(dst, minlength=n_nodes)
    out_cnt = torch.bincount(src, minlength=n_nodes)
    if max_per_node is not None:
        in_cnt_eff = in_cnt.clamp(max=max_per_node)
        out_cnt_eff = out_cnt.clamp(max=max_per_node)
    else:
        in_cnt_eff, out_cnt_eff = in_cnt, out_cnt
    in_start = torch.cumsum(in_cnt, 0) - in_cnt
    out_start = torch.cumsum(out_cnt, 0) - out_cnt
    n_trip = in_cnt_eff * out_cnt_eff
    total = int(n_trip.sum())
    if total == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z
    node = torch.repeat_interleave(torch.arange(n_nodes, device=dev), n_trip)
    offset = torch.arange(total, device=dev) - torch.repeat_interleave(torch.cumsum(n_trip, 0) - n_trip, n_trip)
    oc = out_cnt_eff[node]
    ii = offset // oc
    jj = offset % oc
    e_in = eid[in_order[in_start[node] + ii]]
    e_out = eid[out_order[out_start[node] + jj]]
    return e_in, e_out


def triplet_features(r, phi, z, src, dst, e_in, e_out) -> torch.Tensor:
    """(T, 5): kappa [1/mm] (scaled by 1e3 -> 1/m), turn, dcot, angle3d, len_ratio for triplets i -> j -> k."""
    xyz = to_xyz(r, phi, z)
    i, j, k = src[e_in], dst[e_in], dst[e_out]
    a = xyz[j] - xyz[i]
    b = xyz[k] - xyz[j]
    c = xyz[k] - xyz[i]
    la = torch.linalg.vector_norm(a[:, :2], dim=1).clamp_min(1e-6)
    lb = torch.linalg.vector_norm(b[:, :2], dim=1).clamp_min(1e-6)
    lc = torch.linalg.vector_norm(c[:, :2], dim=1).clamp_min(1e-6)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    kappa = 2.0 * cross / (la * lb * lc)
    turn = torch.atan2(cross, dot)
    dcot = b[:, 2] / lb - a[:, 2] / la
    cos3 = (a * b).sum(1) / (torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)).clamp_min(1e-6)
    ang3 = torch.acos(cos3.clamp(-1 + 1e-7, 1 - 1e-7))
    return torch.stack([1e3 * kappa, turn, dcot, ang3, torch.log(lb / la)], dim=1)


def straightness(f: torch.Tensor) -> torch.Tensor:
    """Cheap compatibility score of a triplet (smaller = straighter): 3D angle plus the s-z slope mismatch."""
    return f[:, 3] + 0.5 * f[:, 2].abs().clamp(max=10.0)


def select_triplets(e_in: torch.Tensor, e_out: torch.Tensor, score: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask keeping, for every edge, its k best (lowest score) continuations and its k best predecessors
    (union of the two), so that the triplet count is at most 2 k E whatever the node degrees."""
    def topk_mask(group):
        order = torch.argsort(score, stable=True)
        order = order[torch.argsort(group[order], stable=True)]
        g = group[order]
        start = torch.ones_like(g, dtype=torch.bool)
        start[1:] = g[1:] != g[:-1]
        first = torch.cummax(torch.where(start, torch.arange(len(g), device=g.device), torch.zeros_like(g)), 0).values
        rank = torch.arange(len(g), device=g.device) - first
        m = torch.zeros_like(score, dtype=torch.bool)
        m[order[rank < k]] = True
        return m
    if len(score) == 0:
        return torch.zeros(0, dtype=torch.bool, device=score.device)
    return topk_mask(e_in) | topk_mask(e_out)
