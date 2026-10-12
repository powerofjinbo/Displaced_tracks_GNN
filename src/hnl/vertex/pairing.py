"""Two-track displaced vertices from reconstructed tracks (brief §6D: vertex efficiency vs r_v).

fit_tracks: robust helix fit of every reconstructed track (hits ordered by R = sqrt(r^2 + z^2); start from the circle
through the first, middle and last hit; Huber perigee fit with free curvature, as the truth fit of hnl.truth.helix;
s-z line fit). find_vertices: every pair of displaced fitted tracks (|d0| > min_abs_d0_mm) with >= min_hits hits
whose helices meet (circle
intersection, or the closest points of non-intersecting circles if their gap is < max_gap_mm) inside both first hits
(+ tol) with |dz| < max_dz_mm and r_v > min_rv_mm; pairs are accepted greedily by increasing |dz|, each track in at
most one vertex. All parameters are fixed on val (DECISIONS.md) before any test use.

Vertex efficiency of an event whose two HNL daughters are targets: both daughters DM-matched, and their two matched
tracks form one accepted vertex.
"""

from __future__ import annotations

import numpy as np

from hnl.truth import helix as hx

# Huber scales per volume (transverse, z) as in hnl.truth.fit_events.SCALES
SCALES = {7: (0.05, 0.10), 8: (0.05, 0.10), 9: (0.05, 0.10), 12: (0.30, 0.20), 13: (0.10, 0.60), 14: (0.30, 0.20),
          16: (1.00, 0.30), 17: (0.15, 3.00), 18: (1.00, 0.30)}


def _scales(volume: np.ndarray):
    t = np.full(len(volume), 0.3)
    z = np.full(len(volume), 1.0)
    for v, (st, sz) in SCALES.items():
        m = volume == v
        t[m], z[m] = st, sz
    return t, z


def fit_tracks(r, phi, z, volume, labels, min_hits: int = 4) -> dict:
    """Helix parameters per reconstructed track label with >= min_hits hits (dict of arrays, key 'track')."""
    labels = np.asarray(labels)
    ok = labels >= 0
    lab, cnt = np.unique(labels[ok], return_counts=True)
    keep = lab[cnt >= min_hits]
    sel = np.flatnonzero(np.isin(labels, keep))
    if len(sel) == 0:
        return {"track": np.zeros(0, dtype=np.int64)}
    R3 = np.hypot(r, z)
    order = sel[np.lexsort((R3[sel], labels[sel]))]
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    p = hx.pad_tracks(labels[order], xyz[order], rows=order)
    st, sz = _scales(np.asarray(volume))
    pad = lambda v: np.where(p.rows >= 0, v[np.maximum(p.rows, 0)], 1.0)
    p0, p1, p2 = hx._first_last(p)
    c, Rc = hx.circle_from_3(p0, p1, p2)
    Rc = np.clip(Rc, 10.0, 1e7)
    sigma = hx.rotation_sense(p, c[:, 0], c[:, 1])
    pg = hx.perigee(c[:, 0], c[:, 1], Rc, sigma)
    f = hx.fit_perigee(p, pg.d0, pg.phi0, sigma / Rc, free_kappa=True, scale=pad(st))
    s = hx.arc_length_perigee(p, f.d0, f.phi0, f.kappa)
    z0, ct, zrms = hx.fit_line_sz(s, p.z, p.mask, scale=pad(sz))
    xc, yc = f.centre()
    return {"track": p.track_ids, "n_hits": p.n_hits, "xc": xc, "yc": yc, "R": f.R, "sigma": f.sigma,
            "x_pca": f.d0 * np.sin(f.phi0), "y_pca": -f.d0 * np.cos(f.phi0), "z0": z0, "cot": ct, "d0": f.d0,
            "r_first": np.hypot(p.x[:, 0], p.y[:, 0]), "rms": f.rms, "zrms": zrms, "n_out": f.n_out}


def find_vertices(h: dict, min_hits: int = 5, max_dz_mm: float = 10.0, max_gap_mm: float = 2.0,
                  min_rv_mm: float = 2.0, tol_mm: float = 5.0, min_abs_d0_mm: float = 0.5,
                  min_pt_gev: float = 1.0) -> np.ndarray:
    """(V, 2) array of track-label pairs forming accepted vertices, and their (V,) radii as a second return.
    Only displaced tracks (fitted |d0| > min_abs_d0_mm, fitted pT > min_pt_gev) enter the pairing."""
    T = len(h["track"])
    if T < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    good = np.flatnonzero((h["n_hits"] >= min_hits) & (np.abs(h["d0"]) > min_abs_d0_mm)
                          & (h["R"] > min_pt_gev * hx.MM_PER_GEV))
    if len(good) < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    ia, ib = np.triu_indices(len(good), 1)
    a, b = good[ia], good[ib]
    take = lambda idx: {k: h[k][idx] for k in ("xc", "yc", "R", "sigma", "x_pca", "y_pca", "z0", "cot")}
    rf = np.minimum(h["r_first"][a], h["r_first"][b])
    v = hx.two_track_vertex(take(a), take(b), rf, tol_mm=tol_mm)
    d = np.hypot(h["xc"][a] - h["xc"][b], h["yc"][a] - h["yc"][b])
    gap = np.where(v.n_intersections > 0, 0.0,
                   np.minimum(np.abs(d - (h["R"][a] + h["R"][b])), np.abs(np.abs(h["R"][a] - h["R"][b]) - d)))
    rv = np.hypot(v.x, v.y)
    acc = v.ok & (v.dz < max_dz_mm) & (gap < max_gap_mm) & (rv > min_rv_mm)
    cand = np.flatnonzero(acc)
    cand = cand[np.argsort(v.dz[cand], kind="stable")]
    used = np.zeros(T, dtype=bool)
    pairs, radii = [], []
    for k in cand:
        if used[a[k]] or used[b[k]]:
            continue
        used[a[k]] = used[b[k]] = True
        pairs.append((h["track"][a[k]], h["track"][b[k]]))
        radii.append(rv[k])
    return np.array(pairs, dtype=np.int64).reshape(-1, 2), np.array(radii)
