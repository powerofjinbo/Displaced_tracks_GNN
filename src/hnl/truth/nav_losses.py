"""Truth-only check of Fatras navigation losses in the no-pileup samples: do particles stop producing hits before they
physically leave the tracker?

Samples: generic_iso_v1 and generic_dense_v1 (train and val splits only, splits/generic_*_{train,val}.txt; no test
event, not the extension generic_iso_x1) and the Zenodo ttbar (train split). No model runs and no model output is read.
The truth sidecars hold whole 500-event chunks; rows of events outside the train and val lists are dropped on load and
enter no computation.

Method (per charged particle; constant 2 T field along +z, as in production: scripts/acts/acts_sim.py):
  * Sensitive layers from acts/geometry/detectors.csv (the v15 GenericDetector, the Zenodo geometry): 10 barrel
    layers (cylinders at the median module radius, z extent of the module edges) and 38 disks (median module z, r
    extent of the module edges), and every module surface for the coverage test.
  * Method A, the initial-state helix (generated samples only): from the stored initial state (gun particles: the
    gun_truth sidecar, every gun particle; ttbar particles: the truth sidecar, particles with hits). Its expected path
    ends where Fatras ends a particle that is not lost: at the world boundary (r = 1028.538 mm or |z| = 3001 mm,
    acts/geometry/layer-volumes.csv) or at the loop protection of the v15 propagator (Acts::detail::LoopProtection,
    on by default: path limit 0.5 x 2 pi p / B, i.e. half a turn in the transverse plane), whichever comes first.
  * Method B, the end-state helix (every sample; the only one for the Zenodo files, which store the END state of each
    particle, hnl.data.npz): the helix through the stored end position with the end momentum, started at the point
    of its circle nearest to the production vertex (generated samples) or to the z axis (Zenodo prompt particles). Its
    path ends at the end position, or at the world boundary if the particle ended outside it. After the last hit the
    end-state helix follows the particle's own final trajectory; for a particle that flew on without navigation it is
    exact, since no material acts on it then.
  * Expected crossings: intersections of a helix with the layer surfaces along its path that lie on a module: the
    straight line through the crossing along the track direction meets the plane of one of the K_MODULES nearest
    modules inside its bounds ("depth" = distance inside the bounds; the -z short-strip disks of the GenericDetector
    cover 96.6 % of their area, the pixel disks have a 1.7 mm gap between rings). A crossing is "robust" if its depth
    is at least COV_MARGIN_A_MM (A) or COV_MARGIN_B_MM (B), it lies at least MARGIN_MM (A) or MARGIN_B_MM (B) inside
    the layer's outer edges, the layer radius is that far from the helix's largest and smallest radius (no grazing
    crossing), and, for a path ended by the loop protection in A, END_MARGIN_MM of arc before its end.
  * Each hit is matched to the crossing of its own layer nearest to it (a barrel layer can be crossed twice). The last
    hit is the matched hit furthest along the helix. "After": robust crossings beyond the last hit; a particle has a
    premature end if at least one robust crossing follows its last hit (for a particle without hits: if its path has
    a robust crossing); ">= 2" is also reported. "Holes": robust crossings before the last hit without a hit.
  * End state (particle table): "outside" = more than 1 mm beyond the world boundary (the particle flew on without
    navigation), "world" = on the world boundary, "below_pmin" = p_end < 0.5 GeV inside the world (ionisation or
    bremsstrahlung took it below pMin and Fatras killed it), "loop" = inside the world with the transverse direction
    turned by about pi from the start (the loop protection), "inside" = anything else.
  * Fatras of these samples has multiple scattering, ionisation, bremsstrahlung and photon conversion and no nuclear
    interactions (docs/new_samples_spec.md section 2), so the only physical early ends are energy loss below pMin and
    the loop protection; there is no hadronic expectation in the simulation.

    python -m hnl.truth.nav_losses scan --sample generic_iso|generic_dense|ttbar [--workers 8] [--limit N]
    python -m hnl.truth.nav_losses summary
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd

from hnl.data import npz
from hnl.data.npz import COL, particle_ids
from hnl.data.splits import read_split
from hnl.geometry.ids import BARREL_VOLUMES, decode_barcode, decode_geometry_id

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DETECTORS = "/work/submit/jinboz1/hnl/acts/geometry/detectors.csv"
GENERATED = os.path.join(npz.DATA_ROOT, "generated")
OUT_DATA = os.path.join(npz.DATA_ROOT, "derived", "nav_losses_v1")
OUT_MD = os.path.join(REPO, "results", "generated_nav_losses_v1.md")
OUT_JSON = os.path.join(REPO, "results", "generated_nav_losses_v1.json")

R_WORLD = 1028.538
Z_WORLD = 3001.0
B_TESLA = 2.0
MM_PER_GEV = 1000.0 / (0.299792458 * B_TESLA)
PMIN = 0.5
EPS = 1e-3
MARGIN_MM = 20.0
END_MARGIN_MM = 10.0
MARGIN_B_MM = 10.0
NEAR_ARC_MM = 400.0
PT_MIN_B = 0.01
COV_MARGIN_A_MM = 3.0
COV_MARGIN_B_MM = 1.0
K_MODULES = 8
WORLD_TOL_MM = 1.0
MIN_AFTER = 2

GEN_DIRS = {"generic_iso": "generic_iso_v1", "generic_dense": "generic_dense_v1"}
SPLITS = ("train", "val")


# --------------------------------------------------------------------------------------------- geometry
def _module_edges(d: pd.DataFrame, n: int = 33) -> tuple[np.ndarray, np.ndarray]:
    """Global r and z of n points along each edge of every module (rectangle: bounds type 6, [minX, minY, maxX, maxY];
    trapezoid: bounds type 7, [halfX at -halfY, halfX at +halfY, halfY])."""
    p = d[[f"bound_param{i}" for i in range(4)]].to_numpy(float)
    rect = d.bounds_type.to_numpy() == 6
    corners = np.zeros((len(d), 4, 2))
    corners[rect] = np.stack([p[rect][:, [0, 1]], p[rect][:, [2, 1]], p[rect][:, [2, 3]], p[rect][:, [0, 3]]], 1)
    tr = ~rect
    hxn, hxp, hy = p[tr, 0], p[tr, 1], p[tr, 2]
    corners[tr] = np.stack([np.stack([-hxn, -hy], 1), np.stack([hxn, -hy], 1), np.stack([hxp, hy], 1),
                            np.stack([-hxp, hy], 1)], 1)
    t = np.linspace(0.0, 1.0, n)
    a, b = corners, np.roll(corners, -1, axis=1)
    loc = (a[:, :, None, :] * (1 - t)[None, None, :, None] + b[:, :, None, :] * t[None, None, :, None]).reshape(
        len(d), -1, 2)
    rot = d[["rot_xu", "rot_xv", "rot_yu", "rot_yv", "rot_zu", "rot_zv"]].to_numpy(float)
    c = d[["cx", "cy", "cz"]].to_numpy(float)
    x = c[:, 0, None] + rot[:, 0, None] * loc[..., 0] + rot[:, 1, None] * loc[..., 1]
    y = c[:, 1, None] + rot[:, 2, None] * loc[..., 0] + rot[:, 3, None] * loc[..., 1]
    z = c[:, 2, None] + rot[:, 4, None] * loc[..., 0] + rot[:, 5, None] * loc[..., 1]
    return np.hypot(x, y), z


def _union_gaps(lo: np.ndarray, hi: np.ndarray) -> list[list[float]]:
    order = np.argsort(lo)
    gaps, reach = [], hi[order[0]]
    for i in order[1:]:
        if lo[i] > reach:
            gaps.append([float(reach), float(lo[i])])
        reach = max(reach, hi[i])
    return gaps


def layer_table(path: str = DETECTORS) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.module_id > 0].reset_index(drop=True)
    r, z = _module_edges(d)
    d["r_lo"], d["r_hi"], d["z_lo"], d["z_hi"] = r.min(1), r.max(1), z.min(1), z.max(1)
    d["rc"] = np.hypot(d.cx, d.cy)
    rows = []
    for (v, lay), g in d.groupby(["volume_id", "layer_id"]):
        barrel = v in BARREL_VOLUMES
        rows.append(dict(
            key=int(v) * 100 + int(lay), volume=int(v), layer=int(lay), barrel=barrel, n_modules=len(g),
            R=float(np.median(g.rc)) if barrel else np.nan, Z=np.nan if barrel else float(np.median(g.cz)),
            rlo=float(g.r_lo.min()), rhi=float(g.r_hi.max()), zlo=float(g.z_lo.min()), zhi=float(g.z_hi.max()),
            gaps=_union_gaps(g.z_lo.to_numpy(), g.z_hi.to_numpy()) if barrel
            else _union_gaps(g.r_lo.to_numpy(), g.r_hi.to_numpy())))
    return pd.DataFrame(rows).sort_values("key").reset_index(drop=True)


class Modules:
    """Module surfaces per layer for the coverage test: depth of a crossing inside the nearest modules."""

    def __init__(self, path: str = DETECTORS):
        from scipy.spatial import cKDTree
        d = pd.read_csv(path)
        d = d[d.module_id > 0]
        self.layers = {}
        for key, g in d.groupby(d.volume_id * 100 + d.layer_id):
            c = g[["cx", "cy", "cz"]].to_numpy(float)
            R = g[["rot_xu", "rot_xv", "rot_xw", "rot_yu", "rot_yv", "rot_yw", "rot_zu", "rot_zv", "rot_zw"]] \
                .to_numpy(float).reshape(-1, 3, 3)
            self.layers[int(key)] = (cKDTree(c), c, R, g.bounds_type.to_numpy(),
                                     g[[f"bound_param{i}" for i in range(4)]].to_numpy(float))

    def depth(self, key: int, P: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Largest distance inside a module boundary (negative outside every module) of the straight line through P
        with direction D where it meets each of the nearest module planes."""
        if len(P) == 0:
            return np.zeros(0)
        tree, c, R, bt, p = self.layers[key]
        k = min(K_MODULES, len(c))
        _, idx = tree.query(P, k=k)
        idx = idx.reshape(len(P), k)
        Rm = R[idx]                                                     # (m, k, 3, 3); columns: local u, v, w
        loc = np.einsum("mkij,mki->mkj", Rm, P[:, None, :] - c[idx])
        Dl = np.einsum("mkij,mi->mkj", Rm, D)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(Dl[..., 2]) > 1e-3, -loc[..., 2] / Dl[..., 2], np.nan)
        t = np.where(np.abs(t) < 50.0, t, np.nan)
        u, v = loc[..., 0] + t * Dl[..., 0], loc[..., 1] + t * Dl[..., 1]
        q = p[idx]
        rect = bt[idx] == 6
        d_rect = np.minimum(np.minimum(u - q[..., 0], q[..., 2] - u), np.minimum(v - q[..., 1], q[..., 3] - v))
        hxn, hxp, hy = q[..., 0], q[..., 1], q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            hx = hxn + (hxp - hxn) * (v + hy) / (2 * hy)
            cosa = np.cos(np.arctan((hxp - hxn) / (2 * hy)))
        d_trap = np.minimum(hy - np.abs(v), (hx - np.abs(u)) * cosa)
        dep = np.where(rect, d_rect, d_trap)
        return np.max(np.where(np.isnan(dep), -np.inf, dep), axis=1)


_MODULES: Modules | None = None


def modules() -> Modules:
    global _MODULES
    if _MODULES is None:
        _MODULES = Modules()
    return _MODULES


# --------------------------------------------------------------------------------------------- helix
def helix(x0, y0, z0, px, py, pz, q) -> dict:
    """Helix in Bz = +2 T. Transverse arc s >= 0 along the motion; psi = angle of (P - C) seen from the centre C;
    psi(s) = psi0 - h s / R with h = sign(q)."""
    pt = np.hypot(px, py)
    R = pt * MM_PER_GEV
    h = np.sign(q).astype(float)
    phi0 = np.arctan2(py, px)
    xc, yc = x0 + h * R * np.sin(phi0), y0 - h * R * np.cos(phi0)
    return dict(x0=x0, y0=y0, z0=z0, R=R, h=h, xc=xc, yc=yc, psi0=np.arctan2(y0 - yc, x0 - xc), cot=pz / pt,
                phi0=phi0, p=np.sqrt(pt ** 2 + pz ** 2), pt=pt)


def at(H: dict, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sh = (-1,) + (1,) * (np.ndim(s) - 1)
    g = {k: np.reshape(H[k], sh) for k in ("psi0", "h", "R", "xc", "yc", "z0", "cot")}
    psi = g["psi0"] - g["h"] * s / g["R"]
    return g["xc"] + g["R"] * np.cos(psi), g["yc"] + g["R"] * np.sin(psi), g["z0"] + g["cot"] * s


def _s_of_psi(H: dict, psi: np.ndarray) -> np.ndarray:
    sh = (-1,) + (1,) * (np.ndim(psi) - 1)
    h, psi0, R = (np.reshape(H[k], sh) for k in ("h", "psi0", "R"))
    return np.mod(h * (psi0 - psi), 2 * np.pi) * R


def cylinder_roots(H: dict, radius) -> np.ndarray:
    """Arc lengths (N, ..., 2) where the helix circle meets the circle of the given radius (NaN if it does not)."""
    d = np.hypot(H["xc"], H["yc"])
    psic = np.arctan2(H["yc"], H["xc"])
    radius = np.asarray(radius, float)
    sh = (-1,) + (1,) * radius.ndim
    d_, R_, psic_ = (np.reshape(a, sh) for a in (d, H["R"], psic))
    with np.errstate(invalid="ignore", divide="ignore"):
        cosA = (radius ** 2 - d_ ** 2 - R_ ** 2) / (2 * R_ * d_)
    ok = np.abs(cosA) <= 1.0
    A = np.arccos(np.clip(cosA, -1.0, 1.0))
    s = np.stack([_s_of_psi(H, psic_ + A), _s_of_psi(H, psic_ - A)], -1)
    return np.where(ok[..., None], s, np.nan)


def path_world(H: dict) -> np.ndarray:
    """Transverse arc of the first exit through the world boundary."""
    sr = cylinder_roots(H, R_WORLD)
    sr = np.where(sr > EPS, sr, np.inf)
    s_r = np.min(np.where(np.isnan(sr), np.inf, sr), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_z = np.where(H["cot"] != 0, (np.sign(H["cot"]) * Z_WORLD - H["z0"]) / H["cot"], np.inf)
    s_z = np.where(s_z > EPS, s_z, np.inf)
    return np.minimum(s_r, s_z)


def path_end(H: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expected transverse arc where Fatras ends the particle without losses (world boundary or loop protection: half
    a turn), and whether the loop protection ends it."""
    s_world = path_world(H)
    s_loop = np.pi * H["R"]
    return np.minimum(s_world, s_loop), s_loop < s_world


def crossings(H: dict, L: pd.DataFrame, s_hi: np.ndarray, margin: float, end_margin: np.ndarray,
              cov_margin: float) -> dict:
    """Crossings (N, n_layers, 2) of the helix with the layer surfaces for arcs in (EPS, s_hi]: arc s, position,
    in-acceptance and robust flags."""
    barrel = L.barrel.to_numpy()
    n, nl = len(H["R"]), len(L)
    S = np.full((n, nl, 2), np.nan)
    S[:, barrel, :] = cylinder_roots(H, L.R.to_numpy()[barrel])
    Zd = L.Z.to_numpy()[~barrel]
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = (Zd[None, :] - H["z0"][:, None]) / H["cot"][:, None]
    S[:, ~barrel, 0] = np.where(np.isfinite(sd), sd, np.nan)
    with np.errstate(invalid="ignore"):
        S = np.where((S > EPS) & (S <= s_hi[:, None, None]), S, np.nan)
    X, Y, Zp = at(H, np.nan_to_num(S, nan=0.0))
    Rp = np.hypot(X, Y)
    valid = ~np.isnan(S)
    zlo, zhi = L.zlo.to_numpy()[None, :, None], L.zhi.to_numpy()[None, :, None]
    rlo, rhi = L.rlo.to_numpy()[None, :, None], L.rhi.to_numpy()[None, :, None]
    bmask = barrel[None, :, None]
    acc = valid & np.where(bmask, (Zp >= zlo) & (Zp <= zhi), (Rp >= rlo) & (Rp <= rhi))
    d = np.hypot(H["xc"], H["yc"])
    rmax, rmin = d + H["R"], np.abs(d - H["R"])
    Rl = np.nan_to_num(L.R.to_numpy())[None, :, None]
    graze_ok = ((rmax[:, None, None] - Rl) >= margin) & ((Rl - rmin[:, None, None]) >= margin)
    inner = np.where(bmask, (Zp >= zlo + margin) & (Zp <= zhi - margin) & graze_ok,
                     (Rp >= rlo + margin) & (Rp <= rhi - margin))
    # module coverage: depth of the crossing inside the nearest module, along the track direction
    psi = np.reshape(H["psi0"], (-1, 1, 1)) - np.reshape(H["h"], (-1, 1, 1)) * np.nan_to_num(S) / np.reshape(
        H["R"], (-1, 1, 1))
    phi = psi - np.reshape(H["h"], (-1, 1, 1)) * np.pi / 2
    cot = np.broadcast_to(np.reshape(H["cot"], (-1, 1, 1)), S.shape)
    nrm = np.sqrt(1 + cot ** 2)
    depth = np.full(S.shape, -np.inf)
    M = modules()
    for i, key in enumerate(L.key.to_numpy()):
        sel = acc[:, i, :]
        if sel.any():
            P = np.stack([X[:, i, :][sel], Y[:, i, :][sel], Zp[:, i, :][sel]], 1)
            D = np.stack([np.cos(phi[:, i, :][sel]), np.sin(phi[:, i, :][sel]), cot[:, i, :][sel]], 1) / \
                nrm[:, i, :][sel][:, None]
            depth[:, i, :][sel] = M.depth(int(key), P, D)
    acc = acc & (depth >= 0)
    with np.errstate(invalid="ignore"):
        robust = acc & inner & (depth >= cov_margin) & (S <= (s_hi - end_margin)[:, None, None])
    return dict(S=S, X=X, Y=Y, Z=Zp, acc=acc, robust=robust, rmax=rmax, depth=depth)


def restart(H: dict, xv, yv) -> tuple[dict, np.ndarray]:
    """The same helix restarted at the point of its circle nearest to (xv, yv); returns it and the transverse arc
    from that point to the original start (the end point, for an end-state helix)."""
    xc, yc, R, h = H["xc"], H["yc"], H["R"], H["h"]
    psi_v = np.arctan2(yv - yc, xv - xc)
    s_e = np.mod(h * (psi_v - H["psi0"]), 2 * np.pi) * R
    x0, y0 = xc + R * np.cos(psi_v), yc + R * np.sin(psi_v)
    z0 = H["z0"] - H["cot"] * s_e
    phi = psi_v - h * np.pi / 2
    pt = H["pt"]
    pz = H["cot"] * pt
    return helix(x0, y0, z0, pt * np.cos(phi), pt * np.sin(phi), pz, h), s_e


# --------------------------------------------------------------------------------------------- per event
def analyse(H: dict, L: pd.DataFrame, s_hi: np.ndarray, margin: float, end_margin: np.ndarray,
            hit_part: np.ndarray, hit_layer: np.ndarray, hit_xyz: np.ndarray, cov_margin: float = 0.0) -> dict:
    """Match hits to the crossings of one helix per particle and count (arrays of length N)."""
    C = crossings(H, L, s_hi, margin, end_margin, cov_margin)
    S, n = C["S"], len(H["R"])
    nk = S.shape[1] * S.shape[2]
    ar = np.arange(len(hit_part))
    cand = S[hit_part, hit_layer]                                     # (n_hits, 2)
    dx = C["X"][hit_part, hit_layer] - hit_xyz[:, 0, None]
    dy = C["Y"][hit_part, hit_layer] - hit_xyz[:, 1, None]
    dz = C["Z"][hit_part, hit_layer] - hit_xyz[:, 2, None]
    d2 = np.where(np.isnan(cand), np.inf, dx * dx + dy * dy + dz * dz)
    k = np.argmin(d2, 1) if len(ar) else np.zeros(0, int)
    matched = np.isfinite(d2[ar, k])
    dist = np.sqrt(np.where(matched, d2[ar, k], np.nan))
    s_hit = np.where(matched, cand[ar, k], np.nan)
    hits_on = np.zeros(S.shape, np.int32)
    np.add.at(hits_on, (hit_part[matched], hit_layer[matched], k[matched]), 1)
    s_last = np.full(n, -np.inf)
    np.maximum.at(s_last, hit_part[matched], s_hit[matched])
    s_last = np.where(np.isfinite(s_last), s_last, 0.0)              # no matched hit: every crossing is "after"
    Sx = np.nan_to_num(S, nan=-1.0)
    after = Sx > s_last[:, None, None] + 1e-6
    before = (Sx >= 0) & ~after
    near = before & (Sx >= (s_last - NEAR_ARC_MM)[:, None, None])
    rob, acc = C["robust"], C["acc"]
    keys = L.key.to_numpy()
    Sr = np.where(rob, S, -np.inf).reshape(n, nk)
    j = np.argmax(Sr, 1)
    has_rob = np.isfinite(Sr[np.arange(n), j])
    Sa = np.where(rob & after, S, np.inf).reshape(n, nk)              # first robust crossing after the last hit
    j1 = np.argmin(Sa, 1)
    has_a = np.isfinite(Sa[np.arange(n), j1])
    # first hole (robust crossing before the last hit without a hit): its rank among the robust crossings and layer
    Sf = np.where(rob, S, np.inf).reshape(n, nk)
    rank = np.argsort(np.argsort(Sf, 1), 1)
    hole = (rob & before & (hits_on == 0)).reshape(n, nk)
    rk = np.where(hole, rank, 10 ** 6)
    jh = np.argmin(rk, 1)
    has_h = hole[np.arange(n), jh]
    return dict(
        first_hole_rank=np.where(has_h, rank[np.arange(n), jh], -1), first_hole_key=np.where(has_h, keys[jh // 2], -1),
        n_unm=np.bincount(hit_part[~matched], minlength=n),
        match_max_mm=_group_max(hit_part[matched], dist[matched], n),
        n_rob=rob.sum((1, 2)), n_acc=acc.sum((1, 2)),
        n_after=(rob & after).sum((1, 2)), n_acc_after=(acc & after).sum((1, 2)),
        n_before=(rob & before).sum((1, 2)), n_holes=(rob & before & (hits_on == 0)).sum((1, 2)),
        n_near=(rob & near).sum((1, 2)), n_holes_near=(rob & near & (hits_on == 0)).sum((1, 2)),
        exit_key=np.where(has_rob, keys[j // 2], -1),
        exit_r=np.where(has_rob, np.hypot(C["X"], C["Y"]).reshape(n, nk)[np.arange(n), j], np.nan),
        exit_z=np.where(has_rob, C["Z"].reshape(n, nk)[np.arange(n), j], np.nan),
        first_missed_key=np.where(has_a, keys[j1 // 2], -1),
        s_last=s_last, s_hi=s_hi, rmax_helix=C["rmax"],
        final_hit=np.where(has_rob, hits_on.reshape(n, nk)[np.arange(n), j] > 0, False),
        _layers=dict(before=(rob & before).sum((0, 2)), holes=(rob & before & (hits_on == 0)).sum((0, 2)),
                     after=(rob & after).sum((0, 2))),
    )


def _group_max(idx: np.ndarray, val: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    if len(idx):
        tmp = np.full(n, -np.inf)
        np.maximum.at(tmp, idx, val)
        out = np.where(np.isfinite(tmp), tmp, np.nan)
    return out


def end_category(xe, ye, ze, pe, turn) -> np.ndarray:
    re = np.hypot(xe, ye)
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(re), "not_in_table",
               np.where((re > R_WORLD + WORLD_TOL_MM) | (np.abs(ze) > Z_WORLD + WORLD_TOL_MM), "outside",
               np.where((np.abs(re - R_WORLD) <= WORLD_TOL_MM) | (np.abs(np.abs(ze) - Z_WORLD) <= WORLD_TOL_MM),
                        "world",
               np.where(pe < PMIN, "below_pmin",
               np.where((turn > np.pi - 0.3) & (turn < np.pi + 0.6), "loop", "inside")))))


def layer_index(L: pd.DataFrame, gid: np.ndarray) -> np.ndarray:
    g = decode_geometry_id(gid)
    key = g["volume"] * 100 + g["layer"]
    lut = {k: i for i, k in enumerate(L.key.to_numpy())}
    u, inv = np.unique(key, return_inverse=True)
    idx = np.array([lut.get(int(k), -1) for k in u], dtype=np.int64)
    if (idx < 0).any():
        raise ValueError(f"hits on layers outside the sensitive-layer table: {sorted(set(u[idx < 0]))}")
    return idx[inv]


def event_rows(ev: dict, L: pd.DataFrame, sel_ids: np.ndarray, init: dict | None, vertex_xy: tuple | None,
               extra: dict) -> pd.DataFrame:
    """Rows for the particles sel_ids of one event.
    init: initial states (x, y, z, px, py, pz, q aligned with sel_ids) for method A, or None (Zenodo).
    Method B (end-state helix) runs for every particle in the table with pT_end >= PT_MIN_B; its start is the point
    of the end-state circle nearest to the production vertex (init) or to the z axis (vertex_xy = (0, 0))."""
    P = ev["particles"]
    ids = particle_ids(P)
    order = np.argsort(ids)
    row = order[np.clip(np.searchsorted(ids, sel_ids, sorter=order), 0, len(ids) - 1)]
    in_table = ids[row] == sel_ids
    T = np.where(in_table[:, None], P[row], np.nan)
    xe, ye, ze = T[:, COL["vx"]], T[:, COL["vy"]], T[:, COL["vz"]]
    pxe, pye, pze, qe = T[:, COL["px"]], T[:, COL["py"]], T[:, COL["pz"]], T[:, COL["q"]]
    pe = np.sqrt(pxe ** 2 + pye ** 2 + pze ** 2)
    pid = ev["pid"]
    m = np.isin(pid, sel_ids)
    srt = np.argsort(sel_ids)
    hp = srt[np.searchsorted(sel_ids, pid[m], sorter=srt)]
    hl = layer_index(L, ev["layers"][m])
    xyz = ev["x"][m].astype(np.float64)
    hxyz = np.stack([xyz[:, 0] * np.cos(xyz[:, 1]), xyz[:, 0] * np.sin(xyz[:, 1]), xyz[:, 2]], 1)
    n = len(sel_ids)
    layers = {}
    out = {"particle_id": sel_ids, "in_table": in_table, "n_hits": np.bincount(hp, minlength=n),
           "pdg_table": T[:, COL["particle_type"]], "pt_table": T[:, COL["pt"]], "eta_table": T[:, COL["eta"]],
           "r_end": np.hypot(xe, ye), "z_end": ze, "p_end": pe}
    # method A: the initial-state helix
    if init is not None:
        HA = helix(init["x"], init["y"], init["z"], init["px"], init["py"], init["pz"], init["q"])
        s_hi, by_loop = path_end(HA)
        ra = analyse(HA, L, s_hi, MARGIN_MM, np.where(by_loop, END_MARGIN_MM, 0.0), hp, hl, hxyz, COV_MARGIN_A_MM)
        layers["a"] = ra.pop("_layers")
        out.update({k + "_a": v for k, v in ra.items()})
        out.update(by_loop_a=by_loop, p0=HA["p"], pt0=HA["pt"], eta0=np.arcsinh(HA["cot"]),
                   r0=np.hypot(HA["x0"], HA["y0"]), z0=HA["z0"])
        xv, yv = init["x"], init["y"]
        phi_start = HA["phi0"]
    else:
        xv, yv = np.full(n, vertex_xy[0], float), np.full(n, vertex_xy[1], float)
        phi_start = None
    # method B: the end-state helix, from the vertex projection to the end point (or the world exit)
    okb = in_table & (np.hypot(pxe, pye) >= PT_MIN_B) & (qe != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        HE = helix(*(np.where(okb, a, b) for a, b in ((xe, 500.0), (ye, 0.0), (ze, 0.0), (pxe, 1.0), (pye, 0.0),
                                                      (pze, 0.0), (qe, 1.0))))
        HB, s_e = restart(HE, xv, yv)
    s_world = path_world(HB)
    s_hi_b = np.minimum(s_e, s_world)
    sub = np.flatnonzero(okb)
    HBs = {k: (v[sub] if np.ndim(v) else v) for k, v in HB.items()}
    remap = np.full(n, -1)
    remap[sub] = np.arange(len(sub))
    hm = okb[hp]
    rbs = analyse(HBs, L, s_hi_b[sub], MARGIN_B_MM, np.zeros(len(sub)), remap[hp[hm]], hl[hm], hxyz[hm],
                  COV_MARGIN_B_MM)
    layers["b"] = rbs.pop("_layers")
    rb = {}
    for k, v in rbs.items():
        v = np.asarray(v)
        full = np.full(n, -1 if v.dtype.kind in "iu" else (False if v.dtype.kind == "b" else np.nan), dtype=v.dtype)
        full[sub] = v
        rb[k] = full
    for k, v in rb.items():
        v = np.asarray(v)
        out[k + "_b"] = v
    out["ok_b"] = okb
    out["z0_b"] = np.where(okb, HB["z0"], np.nan)
    out["vtx_miss_b"] = np.where(okb, np.hypot(HB["x0"] - xv, HB["y0"] - yv), np.nan)
    if phi_start is None:
        phi_start = HB["phi0"]
    hq = np.sign(np.where(np.isnan(qe), 1.0, qe))
    turn = np.mod(hq * (phi_start - np.arctan2(pye, pxe)), 2 * np.pi)
    out["turn"] = turn
    out["end_cat"] = end_category(xe, ye, ze, pe, turn)
    df = pd.DataFrame(out)
    for k, v in extra.items():
        df[k] = v
    return df, layers


# --------------------------------------------------------------------------------------------- samples
def allowed_events(sample: str) -> dict[int, str]:
    out = {}
    for sp in SPLITS:
        for e in read_split(sample, sp):
            out[int(e)] = sp
    return out


class _LayerTally:
    """Per (kind, method) and layer: robust crossings before the last hit, holes among them, crossings after it."""

    def __init__(self, n_layers: int):
        self.n = n_layers
        self.t: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def add(self, kind: str, layers: dict) -> None:
        for m, c in layers.items():
            cur = self.t.setdefault((kind, m), {k: np.zeros(self.n, np.int64) for k in ("before", "holes", "after")})
            for k in cur:
                cur[k] += c[k]

    def frame(self, keys: np.ndarray) -> pd.DataFrame:
        rows = []
        for (kind, m), c in self.t.items():
            for i, key in enumerate(keys):
                rows.append(dict(kind=kind, method=m, key=int(key), before=int(c["before"][i]),
                                 holes=int(c["holes"][i]), after=int(c["after"][i])))
        return pd.DataFrame(rows)


def _generated_chunk(args):
    sample, first, allowed, limit = args
    L = layer_table()
    tally = _LayerTally(len(L))
    d = GEN_DIRS[sample]
    with np.load(os.path.join(GENERATED, d + "_truth", f"truth_{first}.npz")) as f:
        tr = {k: f[k] for k in f.files if f[k].ndim}
    with np.load(os.path.join(GENERATED, d + "_truth", f"gun_truth_{first}.npz")) as f:
        gt = {k: f[k] for k in f.files if f[k].ndim}
    # the sidecars hold the whole 500-event chunk: keep only train and val events, before anything else
    keep = np.isin(tr["p_event"], list(allowed))
    tr = {k: (v[keep] if k.startswith("p_") else v) for k, v in tr.items()}
    keep = np.isin(gt["event"], list(allowed))
    gt = {k: v[keep] for k, v in gt.items()}
    tev, gev = tr["p_event"], gt["event"]
    events = sorted(allowed)
    if limit:
        events = events[:limit]
    frames = []
    for e in events:
        with np.load(os.path.join(GENERATED, d, f"{e}.npz")) as f:
            ev = {k: f[k] for k in ("x", "pid", "layers", "particles")}
        # gun particles: every one, initial state from gun_truth
        g = gev == e
        gids = gt["particle_id"][g].astype(np.int64)
        init = dict(x=gt["vx"][g], y=gt["vy"][g], z=gt["vz"][g], px=gt["px"][g], py=gt["py"][g], pz=gt["pz"][g],
                    q=gt["q"][g].astype(float))
        df, lay = event_rows(ev, L, gids, init, None,
                             dict(event=e, split=allowed[e], kind="gun", pdg=gt["pdg"][g], llp=gt["llp"][g],
                                  n_prongs=gt["n_prongs"][g], r_v=gt["r_v"][g]))
        frames.append(df)
        tally.add("gun", lay)
        # ttbar prompt particles with hits: barcode secondary vertex 0, generation 0, sub-particle 0, charged
        t = tev == e
        pids = tr["p_particle_id"][t].astype(np.int64)
        bc = decode_barcode(pids)
        sel = (bc["vsec"] == 0) & (bc["gen"] == 0) & (bc["sub"] == 0) & (tr["p_q"][t] != 0) & ~np.isin(pids, gids)
        if sel.any():
            init = dict(x=tr["p_vx"][t][sel], y=tr["p_vy"][t][sel], z=tr["p_vz"][t][sel], px=tr["p_px"][t][sel],
                        py=tr["p_py"][t][sel], pz=tr["p_pz"][t][sel], q=tr["p_q"][t][sel].astype(float))
            df, lay = event_rows(ev, L, pids[sel], init, None,
                                 dict(event=e, split=allowed[e], kind="prompt", pdg=tr["p_particle_type"][t][sel],
                                      llp=0, n_prongs=0, r_v=np.nan))
            frames.append(df)
            tally.add("prompt", lay)
        # census of every charged prompt particle in the table (with or without hits): end category only
        frames.append(_census(ev, e, allowed[e], exclude=gids))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), tally.frame(L.key.to_numpy())


def _census(ev: dict, e: int, split: str, exclude: np.ndarray) -> pd.DataFrame:
    """End category of every charged prompt particle in the table (barcode as above), with its hit count."""
    P = ev["particles"]
    ids = particle_ids(P)
    bc = decode_barcode(ids)
    sel = (bc["vsec"] == 0) & (bc["gen"] == 0) & (bc["sub"] == 0) & (P[:, COL["q"]] != 0) & ~np.isin(ids, exclude)
    T = P[sel]
    up, cnt = np.unique(ev["pid"], return_counts=True)
    nh = np.zeros(sel.sum(), np.int64)
    pos = np.searchsorted(up, ids[sel])
    ok = (pos < len(up)) & (up[np.clip(pos, 0, len(up) - 1)] == ids[sel])
    nh[ok] = cnt[pos[ok]]
    pe = np.sqrt(T[:, COL["px"]] ** 2 + T[:, COL["py"]] ** 2 + T[:, COL["pz"]] ** 2)
    cat = end_category(T[:, COL["vx"]], T[:, COL["vy"]], T[:, COL["vz"]], pe, np.zeros(len(T)))
    return pd.DataFrame(dict(particle_id=ids[sel], event=e, split=split, kind="census", n_hits=nh,
                             pdg_table=T[:, COL["particle_type"]], pt_table=T[:, COL["pt"]],
                             eta_table=T[:, COL["eta"]], p_end=pe, r_end=np.hypot(T[:, COL["vx"]], T[:, COL["vy"]]),
                             z_end=T[:, COL["vz"]], end_cat=np.where(cat == "loop", "inside", cat), in_table=True))


def _zenodo_chunk(args):
    events, limit = args
    L = layer_table()
    tally = _LayerTally(len(L))
    frames = []
    for e in events[:limit] if limit else events:
        ev = npz.load_event("ttbar", int(e))
        P = ev["particles"]
        ids = particle_ids(P)
        bc = decode_barcode(ids)
        sel = (bc["vsec"] == 0) & (bc["gen"] == 0) & (bc["sub"] == 0) & (P[:, COL["q"]] != 0)
        sel &= np.isin(ids, ev["pid"])
        if sel.any():
            df, lay = event_rows(ev, L, ids[sel], None, (0.0, 0.0),
                                 dict(event=int(e), split="train", kind="prompt", pdg=P[sel, COL["particle_type"]],
                                      llp=0, n_prongs=0, r_v=np.nan))
            frames.append(df)
            tally.add("prompt", lay)
        frames.append(_census(ev, int(e), "train", exclude=np.zeros(0, np.int64)))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), tally.frame(L.key.to_numpy())


def cmd_scan(a) -> None:
    os.makedirs(OUT_DATA, exist_ok=True)
    if a.sample in GEN_DIRS:
        allowed = allowed_events(a.sample)
        firsts = sorted(int(os.path.basename(f)[len("truth_"):-4])
                        for f in glob.glob(os.path.join(GENERATED, GEN_DIRS[a.sample] + "_truth", "truth_*.npz")))
        jobs = [(a.sample, f, {e: s for e, s in allowed.items() if f <= e < f + 500}, a.limit) for f in firsts]
        fn = _generated_chunk
    elif a.sample == "ttbar":
        ev = read_split("ttbar", "train")
        jobs = [(ev[i:i + 250], a.limit) for i in range(0, len(ev), 250)]
        fn = _zenodo_chunk
    else:
        raise SystemExit(f"unknown sample {a.sample}")
    if a.limit:
        jobs = jobs[:1]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(fn, jobs))
    df = pd.concat([r[0] for r in res], ignore_index=True)
    lay = pd.concat([r[1] for r in res], ignore_index=True).groupby(["kind", "method", "key"], as_index=False).sum()
    df["sample"] = a.sample
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    tag = f"{a.sample}{'_limit' if a.limit else ''}"
    df.to_parquet(os.path.join(OUT_DATA, f"{tag}.parquet"), index=False)
    lay.to_csv(os.path.join(OUT_DATA, f"{tag}_layers.csv"), index=False)
    meta = dict(sample=a.sample, events=int(df.event.nunique()), rows=len(df), git=_git(),
                time=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                splits=sorted(df.split.unique().tolist()))
    with open(os.path.join(OUT_DATA, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", tag, meta)


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--sample", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("summary")
    p.set_defaults(fn=lambda a: __import__("hnl.truth.nav_losses_report", fromlist=["main"]).main())
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
