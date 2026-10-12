"""Track building step after any builder (brief §6C): merge track fragments of one helix.

A missed true edge splits a particle's hits into fragments; with 6 hits a 3 + 3 split loses the particle under double
majority. Candidate pairs (A, B) of tracks with >= min_frag hits: B starts outward of where A ends (R order, small
overlap allowed), the gap between A's last and B's first hit is small in phi and in the s-z slope. The union is refit
with the robust helix fit (hnl.vertex.pairing.fit_tracks); pairs whose union fits as well as one particle (transverse
rms <= max_rms, longitudinal rms <= max_zrms, no more than max_out outlier hits) and has a lever arm (its hits span
at least min_span_mm in R = sqrt(r^2 + z^2); a few hits on one module or disk fit any tiny circle) are merged greedily
by fit quality, each fragment joined at most once at each end. Only geometry is used (no truth).
"""

from __future__ import annotations

import numpy as np

from hnl.vertex.pairing import fit_tracks


def _volume_from_radius(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Coarse subsystem (for the robust-fit scales) when the volume id is not at hand: pixel / short / long strips,
    barrel vs endcap by |z| (generic-detector extents)."""
    v = np.where(r < 200, 8, np.where(r < 720, 13, 17))
    endcap = np.abs(z) > np.where(r < 200, 600, np.where(r < 720, 1200, 1200))
    return np.where(endcap, v - 1, v)


def _candidates_any(trk, labels, r, phi, z, R, max_dphi_dir: float, max_dcot: float):
    """Pairs of fragments that may be one helix in any R order (interleaved or overlapping fragments, e.g. the two
    halves of a particle with double hits in overlapping modules that a walkthrough splits): both fragments' local
    phi(R) and z(r) lines agree at the pair's median radius."""
    par = {}
    for t in trk:
        idx = np.flatnonzero(labels == t)
        Rt, ph, rt, zt = R[idx], np.unwrap(phi[idx][np.argsort(R[idx])]), r[idx], z[idx]
        Rs = np.sort(Rt)
        if len(idx) >= 2 and Rs[-1] - Rs[0] > 1e-6:
            k_phi = np.polyfit(Rs, ph, 1)
            k_z = np.polyfit(rt, zt, 1)
        else:
            k_phi = np.array([0.0, ph[0]])
            k_z = np.array([zt[0] / max(rt[0], 1e-6), 0.0])
        par[t] = (k_phi, k_z, np.median(Rt), np.median(rt))
    out = []
    for i, a in enumerate(trk):
        pa = par[a]
        for b in trk[i + 1:]:
            pb = par[b]
            Rm, rm = 0.5 * (pa[2] + pb[2]), 0.5 * (pa[3] + pb[3])
            dphi = abs((np.polyval(pa[0], Rm) - np.polyval(pb[0], Rm) + np.pi) % (2 * np.pi) - np.pi)
            if dphi > max_dphi_dir:
                continue
            if abs(np.polyval(pa[1], rm) - np.polyval(pb[1], rm)) / max(rm, 1.0) > max_dcot:
                continue
            out.append((a, b))
    return out


def merge_fragments(r, phi, z, labels, volume=None, min_frag: int = 2, max_dphi: float = 0.3, max_gap_r: float = 400.0,
                    max_rms: float = 1.0, max_zrms: float = 5.0, max_out: int = 1, mode: str = "sequential",
                    max_dphi_dir: float = 0.05, max_dcot: float = 0.05, min_span_mm: float = 50.0) -> np.ndarray:
    """mode "sequential": B starts outward of where A ends (the default, for CC); mode "any": also interleaved or
    overlapping fragments with matching local directions (for walkthrough output)."""
    labels = np.asarray(labels).copy()
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    if volume is None:
        volume = _volume_from_radius(r, z)
    R = np.hypot(r, z)
    ok = labels >= 0
    if not ok.any():
        return labels
    trk, cnt = np.unique(labels[ok], return_counts=True)
    trk = trk[cnt >= min_frag]
    if len(trk) < 2:
        return labels
    if mode == "any":
        cand = _candidates_any(list(trk), labels, r, phi, z, R, max_dphi_dir, max_dcot)
        return _merge(cand, trk, labels, r, phi, z, volume, max_rms, max_zrms, max_out, one_per_end=False,
                      min_span_mm=min_span_mm)
    # endpoints of every fragment (innermost / outermost hit in R)
    inner, outer = {}, {}
    for t in trk:
        idx = np.flatnonzero(labels == t)
        o = idx[np.argsort(R[idx])]
        inner[t], outer[t] = o[0], o[-1]
    cand = []
    for a in trk:
        ia_out = outer[a]
        for b in trk:
            if a == b:
                continue
            ib_in = inner[b]
            dR = R[ib_in] - R[ia_out]
            if dR < -20.0 or dR > max_gap_r:
                continue
            dphi = abs((phi[ib_in] - phi[ia_out] + np.pi) % (2 * np.pi) - np.pi)
            if dphi > max_dphi:
                continue
            cand.append((a, b))
    return _merge(cand, trk, labels, r, phi, z, volume, max_rms, max_zrms, max_out, one_per_end=True,
                  min_span_mm=min_span_mm)


def _merge(cand, trk, labels, r, phi, z, volume, max_rms, max_zrms, max_out, one_per_end: bool,
           min_span_mm: float = 50.0):
    if not cand:
        return labels
    # refit every candidate union as one track (vectorised: one pseudo-label per candidate)
    rows, plab = [], []
    for k, (a, b) in enumerate(cand):
        idx = np.flatnonzero((labels == a) | (labels == b))
        rows.append(idx)
        plab.append(np.full(len(idx), k))
    rows = np.concatenate(rows)
    plab = np.concatenate(plab)
    fit = fit_tracks(r[rows], phi[rows], z[rows], np.asarray(volume)[rows], plab, min_hits=3)
    q = np.full(len(cand), np.inf)
    R3 = np.hypot(r[rows], z[rows])
    lo = np.full(len(cand), np.inf)
    hi = np.full(len(cand), -np.inf)
    np.minimum.at(lo, plab, R3)
    np.maximum.at(hi, plab, R3)
    span = hi - lo
    good = (fit["rms"] <= max_rms) & (fit["zrms"] <= max_zrms) & (fit["n_out"] <= max_out) \
        & (span[fit["track"]] >= min_span_mm)
    q[fit["track"][good]] = fit["rms"][good] + 0.1 * fit["zrms"][good]
    order = np.argsort(q)
    used_out, used_in = set(), set()
    parent = {t: t for t in trk}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    n_merges = {t: 0 for t in trk}
    for k in order:
        if not np.isfinite(q[k]):
            break
        a, b = cand[k]
        if find(a) == find(b):
            continue
        if one_per_end:
            if a in used_out or b in used_in:
                continue
            used_out.add(a)
            used_in.add(b)
        else:
            if n_merges[a] >= 2 or n_merges[b] >= 2:
                continue
            n_merges[a] += 1
            n_merges[b] += 1
        parent[find(b)] = find(a)
    for t in trk:
        root = find(t)
        if root != t:
            labels[labels == t] = root
    return labels
