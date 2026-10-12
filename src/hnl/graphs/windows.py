"""Geometric "physics window" graphs (brief §5e, §6A) for the full generic detector.

A doublet (a, b) on an allowed layer pair is kept if
  transverse:   |dphi| <= (kappa_max / 2 + d0_max / (r_a r_b)) |dr| + slack_phi(pair)
  longitudinal: distance of the r-z line through a and b from the origin <= D_max + slack_rz(pair)
with kappa_max = 1 / R(pT_min) (pT_min = 1 GeV). For a helix of curvature kappa and impact parameter d0,
phi(r) ~ phi0 - kappa r / 2 - d0 / r, so dphi ~ -(kappa / 2 + d0 / (r_a r_b)) dr: d0_max = 0 is the prompt
window and the allowance grows only where r_a r_b is small (inner pairs). Pairs with dr ~ 0 (same-layer overlaps,
endcap disks) get a window of order slack_phi. The r-z distance plays the role of z0 (barrel) and r0 (endcaps):
about |z0| sin(theta) for prompt tracks, up to the decay radius for displaced ones.
Allowed layer pairs and the slacks are calibrated on true edges of the TRAIN splits (`calibrate`); the slacks come
from prompt targets (|d0| < 1 mm), so (d0_max = 0, D_max ~ beam spot) is a tight prompt window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hnl.truth.helix import MM_PER_GEV

KAPPA_MAX = 1.0 / (MM_PER_GEV * 1.0)        # 1 / R at pT = 1 GeV [1/mm]


def layer_id(volume: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return np.asarray(volume, dtype=np.int64) * 100 + np.asarray(layer, dtype=np.int64)


def wrap(dphi):
    return np.remainder(dphi + np.pi, 2 * np.pi) - np.pi


def rz_distance(r_a, z_a, r_b, z_b):
    """Distance of the straight r-z line through a and b from the origin of the (z, r) plane."""
    dr, dz = r_b - r_a, z_b - z_a
    return np.abs(r_a * z_b - r_b * z_a) / np.maximum(np.hypot(dr, dz), 1e-9)


def pair_quantities(r_a, phi_a, z_a, r_b, phi_b, z_b):
    dr = r_b - r_a
    dphi = np.abs(wrap(phi_b - phi_a))
    prompt_phi = 0.5 * KAPPA_MAX * np.abs(dr)
    excess = dphi - prompt_phi                         # what the prompt window does not cover
    d0_coef = np.abs(dr) / np.maximum(r_a * r_b, 1e-9)  # dphi allowance per mm of d0
    return dphi, excess, d0_coef, rz_distance(r_a, z_a, r_b, z_b)


@dataclass
class WindowCalibration:
    pairs: pd.DataFrame          # index (lid_lo, lid_hi): n_true, slack_phi, slack_rz
    layers: pd.DataFrame         # index lid: r_min, r_max, z_min, z_max (from hits)


def _robust_upper(x: pd.Series, n_sigma: float) -> float:
    """median + n_sigma * 1.4826 * MAD: insensitive to the ~8 % of true edges that touch outlier hits."""
    x = x.to_numpy()
    med = np.median(x)
    return float(med + n_sigma * 1.4826 * np.median(np.abs(x - med)))


def calibrate(edge_frames: list[pd.DataFrame], hit_frames: list[pd.DataFrame], min_count: int = 5,
              n_sigma: float = 5.0) -> WindowCalibration:
    """edge_frames: per event, true edges with columns lid_a, lid_b, r_a, phi_a, z_a, r_b, phi_b, z_b, prompt (bool:
    the particle is a prompt target). hit_frames: per event hits with lid, r, z (for layer extents).
    Per layer pair: slack_phi = robust upper edge (median + n_sigma robust widths) of the phi excess of prompt-target
    edges; rz_prompt_q = the same for their r-z line distance. Robust estimates are used because outlier hits (DECISIONS
    2026-10-03) put long tails into every quantile above ~0.9."""
    e = pd.concat(edge_frames, ignore_index=True)
    lo = np.minimum(e.lid_a, e.lid_b)
    hi = np.maximum(e.lid_a, e.lid_b)
    _, excess, _, rz = pair_quantities(e.r_a.to_numpy(), e.phi_a.to_numpy(), e.z_a.to_numpy(),
                                       e.r_b.to_numpy(), e.phi_b.to_numpy(), e.z_b.to_numpy())
    df = pd.DataFrame({"lid_lo": lo, "lid_hi": hi, "excess": excess, "rz": rz, "prompt": e.prompt.to_numpy()})
    n_true = df.groupby(["lid_lo", "lid_hi"]).size().rename("n_true")
    pr = df[df.prompt]
    g = pr.groupby(["lid_lo", "lid_hi"])
    slack_phi = g.excess.apply(lambda x: _robust_upper(x, n_sigma)).clip(lower=0).rename("slack_phi")
    rz_q = g.rz.apply(lambda x: _robust_upper(x, n_sigma)).rename("rz_prompt_q")
    n_prompt = g.size().rename("n_prompt")
    pairs = pd.concat([n_true, n_prompt, slack_phi, rz_q], axis=1)
    pairs = pairs[pairs.n_true >= min_count]
    # pairs with too few prompt edges: the median over pairs
    few = pairs.n_prompt.fillna(0) < 20
    pairs.loc[few, "slack_phi"] = pairs.loc[~few, "slack_phi"].median()
    pairs.loc[few, "rz_prompt_q"] = pairs.loc[~few, "rz_prompt_q"].median()
    h = pd.concat(hit_frames, ignore_index=True)
    layers = h.groupby("lid").agg(r_min=("r", "min"), r_max=("r", "max"), z_min=("z", "min"), z_max=("z", "max"))
    return WindowCalibration(pairs, layers)


def build_graph(r: np.ndarray, phi: np.ndarray, z: np.ndarray, lid: np.ndarray, cal: WindowCalibration,
                d0_max: float, D_max: float | str, slack_scale: float = 1.0) -> np.ndarray:
    """All doublets of one event inside the window; returns (2, E) unique unordered hit-row pairs (a < b).
    D_max: a distance in mm, np.inf (no longitudinal cut), or "prompt" (per-pair 99.5 % quantile of prompt targets,
    i.e. a beam-spot-pointing window)."""
    r = np.asarray(r, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    order = np.lexsort((phi, lid))
    lid_s, phi_s = lid[order], phi[order]
    ulid, start = np.unique(lid_s, return_index=True)
    stop = np.r_[start[1:], len(lid_s)]
    pos = {int(l): (int(a), int(b)) for l, a, b in zip(ulid, start, stop)}
    out = []
    for (la, lb), row in cal.pairs.iterrows():
        if la not in pos or lb not in pos:
            continue
        a0, a1 = pos[la]
        b0, b1 = pos[lb]
        ia = order[a0:a1]
        ib_sorted = order[b0:b1]
        pb = phi_s[b0:b1]
        # loose bound on dphi for every hit of A against the whole extent of layer B
        Lb = cal.layers.loc[lb]
        dr_max = np.maximum(np.abs(Lb.r_max - r[ia]), np.abs(Lb.r_min - r[ia]))
        bound = (0.5 * KAPPA_MAX + d0_max / np.maximum(r[ia] * max(Lb.r_min, 1.0), 1e-9)) * dr_max \
            + slack_scale * row.slack_phi + 1e-9
        bound = np.minimum(bound, np.pi)
        # phi windows with wrap-around: search [phi - bound, phi + bound] on the sorted phi of B (tripled)
        pb3 = np.concatenate([pb - 2 * np.pi, pb, pb + 2 * np.pi])
        lo_i = np.searchsorted(pb3, phi[ia] - bound, side="left")
        hi_i = np.searchsorted(pb3, phi[ia] + bound, side="right")
        cnt = hi_i - lo_i
        if cnt.sum() == 0:
            continue
        a_rep = np.repeat(ia, cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        k = np.repeat(lo_i, cnt) + offs
        b_rep = ib_sorted[k % len(pb)]
        if la == lb:
            keep = a_rep < b_rep
            a_rep, b_rep = a_rep[keep], b_rep[keep]
        dphi, excess, d0c, rz = pair_quantities(r[a_rep], phi[a_rep], z[a_rep], r[b_rep], phi[b_rep], z[b_rep])
        ok = excess <= d0_max * d0c + slack_scale * row.slack_phi
        if isinstance(D_max, str):
            if D_max != "prompt":
                raise ValueError(f"D_max must be a number or 'prompt', got {D_max!r}")
            ok &= rz <= row.rz_prompt_q
        elif np.isfinite(D_max):
            ok &= rz <= D_max
        out.append(np.stack([a_rep[ok], b_rep[ok]]))
    if not out:
        return np.empty((2, 0), dtype=np.int64)
    e = np.concatenate(out, axis=1)
    return np.unique(np.stack([np.minimum(e[0], e[1]), np.maximum(e[0], e[1])], 1), axis=0).T
