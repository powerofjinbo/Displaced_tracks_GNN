"""Helix fits of each particle's own true hits (brief §5b): d0, z0, free-radius pT, and two-track vertices.

Conventions (recorded in docs/definitions.md once fixed on data):
  * Transverse circle with centre c and radius R; the particle moves along it with rotation sense sigma
    (+1 counter-clockwise seen from +z, -1 clockwise), from its first to its last hit along the true-edge chain.
  * Fixed-R fit: R [mm] = 1000 pT [GeV] / (0.3 B [T]) with B = 2 T, i.e. R = 1666.67 mm per GeV; sigma is given
    (from the charge and the empirically fixed field sign) and the initial centre is placed on that side of the
    first-last chord. Free-R fit: (xc, yc, R) all free; sigma follows from the fitted circle and the hit order.
  * Point of closest approach to the z axis: p = c (1 - R/|c|). d0 = (p x t)_z with t the unit direction of motion
    at p, so |d0| = | |c| - R |; its sign follows the usual "angular momentum about the origin" convention.
  * s: signed transverse arc length from p along the direction of motion; z = z0 + s cot(theta) (robust line fit).
All arithmetic is float64; the robust loss is Huber, solved by iteratively reweighted Gauss-Newton with
Levenberg-Marquardt damping, vectorised over tracks (padded to the longest track in the batch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

B_TESLA = 2.0
MM_PER_GEV = 1000.0 / (0.3 * B_TESLA)          # R [mm] per pT [GeV]


# --------------------------------------------------------------------------------------------- hit ordering
def chain_rank(n_hits: int, edges: np.ndarray) -> np.ndarray:
    """Distance of every hit to the END of its chain (0 for the last hit), by pointer doubling.

    `edges` are true edges (2, E) as row indices with out-degree and in-degree <= 1 (each particle one chain)."""
    nxt = np.full(n_hits, -1, dtype=np.int64)
    nxt[edges[0]] = edges[1]
    dist = (nxt >= 0).astype(np.int64)
    ptr = nxt.copy()
    while True:
        live = ptr >= 0
        if not live.any():
            break
        idx = np.flatnonzero(live)
        tgt = ptr[idx]
        new_dist = dist[idx] + dist[tgt]
        new_ptr = ptr[tgt]
        dist[idx] = new_dist
        ptr[idx] = new_ptr
    return dist


def order_hits(pid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Row permutation grouping hits by particle, each particle ordered along its chain (first hit first)."""
    rank = chain_rank(len(pid), edges)
    return np.lexsort((-rank, pid))


@dataclass
class Padded:
    """Hits of T tracks padded to H hits: arrays (T, H); mask marks real hits; track_ids (T,)."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    mask: np.ndarray
    track_ids: np.ndarray
    n_hits: np.ndarray
    rows: np.ndarray        # (T, H) original row index of every hit, -1 for padding


def pad_tracks(track_of_hit: np.ndarray, xyz: np.ndarray, rows: np.ndarray | None = None) -> Padded:
    """`track_of_hit` must already be grouped (all hits of a track contiguous, in trajectory order)."""
    if rows is None:
        rows = np.arange(len(track_of_hit))
    starts = np.flatnonzero(np.r_[True, track_of_hit[1:] != track_of_hit[:-1]])
    n = np.diff(np.r_[starts, len(track_of_hit)])
    T, H = len(starts), int(n.max()) if len(n) else 0
    pos = np.arange(len(track_of_hit)) - np.repeat(starts, n)
    t = np.repeat(np.arange(T), n)
    out = {k: np.zeros((T, H)) for k in ("x", "y", "z")}
    for j, k in enumerate(("x", "y", "z")):
        out[k][t, pos] = xyz[:, j]
    mask = np.zeros((T, H), dtype=bool)
    mask[t, pos] = True
    r = np.full((T, H), -1, dtype=np.int64)
    r[t, pos] = rows
    return Padded(out["x"], out["y"], out["z"], mask, track_of_hit[starts], n, r)


# --------------------------------------------------------------------------------------------- robust weights
def huber_weights(res: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    a = np.abs(res) / scale
    return np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-300))


def _first_last(p: Padded):
    T = len(p.n_hits)
    last = p.n_hits - 1
    mid = last // 2
    ar = np.arange(T)
    return (np.stack([p.x[:, 0], p.y[:, 0]], 1), np.stack([p.x[ar, mid], p.y[ar, mid]], 1),
            np.stack([p.x[ar, last], p.y[ar, last]], 1))


def circle_from_3(p0, p1, p2):
    """Centre and radius of the circle through three points (rows); collinear -> huge radius."""
    ax, ay = p0[:, 0], p0[:, 1]
    bx, by = p1[:, 0], p1[:, 1]
    cx, cy = p2[:, 0], p2[:, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    c = np.stack([ux, uy], 1)
    return c, np.hypot(ax - ux, ay - uy)


def initial_centre_fixed_r(p: Padded, R: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Centre at distance R from the first and last hit, on the side given by the rotation sense."""
    p0, _, p1 = _first_last(p)
    m = 0.5 * (p0 + p1)
    ch = p1 - p0
    L = np.hypot(ch[:, 0], ch[:, 1])
    L = np.where(L < 1e-9, 1e-9, L)
    left = np.stack([-ch[:, 1], ch[:, 0]], 1) / L[:, None]       # chord direction rotated by +90 degrees
    h = np.sqrt(np.clip(R**2 - (0.5 * L) ** 2, 0.0, None))
    return m + (sigma * h)[:, None] * left                        # counter-clockwise motion: centre on the left


# --------------------------------------------------------------------------------------------- circle fits
@dataclass
class CircleFit:
    xc: np.ndarray
    yc: np.ndarray
    R: np.ndarray
    sigma: np.ndarray          # rotation sense used / found (+1 ccw, -1 cw)
    rms: np.ndarray            # rms of transverse residuals over real hits [mm]
    n_out: np.ndarray          # hits with |residual| > 3 Huber scales (outlier candidates)
    converged: np.ndarray
    iterations: int


def fit_circle(p: Padded, R0: np.ndarray | None = None, sigma: np.ndarray | None = None,
               free_r: bool = False, scale: np.ndarray | float = 0.1, n_iter: int = 50, tol: float = 1e-7,
               lam0: float = 1e-3) -> CircleFit:
    """Robust (Huber) circle fit, vectorised over tracks.

    Fixed R (free_r=False): parameters (xc, yc), R = R0, start on the side given by sigma.
    Free R (free_r=True): parameters (xc, yc, R), start from the circle through the first, middle and last hit
    (or from the fixed-R start if R0 and sigma are given)."""
    T = len(p.n_hits)
    m = p.mask.astype(np.float64)
    if free_r:
        if R0 is not None and sigma is not None:
            c = initial_centre_fixed_r(p, R0, sigma)
            R = R0.astype(np.float64).copy()
        else:
            c, R = circle_from_3(*_first_last(p))
    else:
        if R0 is None or sigma is None:
            raise ValueError("fixed-R fit needs R0 and sigma")
        c = initial_centre_fixed_r(p, R0, sigma)
        R = R0.astype(np.float64).copy()
    xc, yc = c[:, 0].copy(), c[:, 1].copy()
    lam = np.full(T, lam0)
    npar = 3 if free_r else 2
    converged = np.zeros(T, dtype=bool)

    def residuals(xc, yc, R):
        dx, dy = p.x - xc[:, None], p.y - yc[:, None]
        rho = np.hypot(dx, dy)
        return dx, dy, np.maximum(rho, 1e-12), rho - R[:, None]

    def huber_obj(res):
        a = np.abs(res)
        return np.sum(m * np.where(a <= scale, 0.5 * res**2, scale * a - 0.5 * scale**2), axis=1)

    dx, dy, rho, res = residuals(xc, yc, R)
    obj = huber_obj(res)
    it = 0
    for it in range(1, n_iter + 1):
        # IRLS: weights from the current residuals, then one damped Gauss-Newton step; a step is accepted only if
        # it lowers the Huber objective itself (not the reweighted sum of squares), so the iteration is monotone
        w = huber_weights(res, scale) * m
        J = [-dx / rho, -dy / rho] + ([-np.ones_like(rho)] if free_r else [])
        A = np.empty((T, npar, npar))
        g = np.empty((T, npar))
        for i in range(npar):
            g[:, i] = -np.sum(w * J[i] * res, axis=1)
            for j in range(i, npar):
                A[:, i, j] = A[:, j, i] = np.sum(w * J[i] * J[j], axis=1)
        eye = np.eye(npar)[None]
        Ad = A + lam[:, None, None] * np.einsum("tii->ti", A)[:, :, None] * eye + 1e-12 * eye
        step = np.linalg.solve(Ad, g[:, :, None])[:, :, 0]
        xn, yn = xc + step[:, 0], yc + step[:, 1]
        Rn = R + step[:, 2] if free_r else R
        dxn, dyn, rhon, resn = residuals(xn, yn, Rn)
        obj_n = huber_obj(resn)
        ok = (obj_n <= obj) & ~converged
        small_step = np.max(np.abs(step), axis=1) < tol
        flat = (obj - obj_n) <= 1e-13 * np.maximum(obj, 1e-30)
        converged |= ok & (small_step | flat)
        converged |= ~ok & (lam >= 1e8)                     # no descent direction left: at the minimum
        xc = np.where(ok, xn, xc)
        yc = np.where(ok, yn, yc)
        R = np.where(ok, Rn, R)
        obj = np.where(ok, obj_n, obj)
        lam = np.where(ok, np.maximum(lam / 3.0, 1e-9), np.minimum(lam * 10.0, 1e9))
        dx, dy, rho, res = residuals(xc, yc, R)
        if converged.all():
            break
    n = np.maximum(m.sum(1), 1)
    rms = np.sqrt(np.sum(m * res**2, axis=1) / n)
    n_out = np.sum(m * (np.abs(res) > 3.0 * np.asarray(scale)), axis=1).astype(np.int64)
    if free_r or sigma is None:
        sigma = rotation_sense(p, xc, yc)
    return CircleFit(xc, yc, R, np.asarray(sigma, dtype=np.int64), rms, n_out, converged, it)


def rotation_sense(p: Padded, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """+1 if the hits advance counter-clockwise around the centre (first -> second ... -> last), else -1.
    Uses the sum of successive turning angles, robust for arcs of any length below one turn."""
    ang = np.arctan2(p.y - yc[:, None], p.x - xc[:, None])
    d = np.diff(ang, axis=1)
    d = np.remainder(d + np.pi, 2 * np.pi) - np.pi
    valid = p.mask[:, 1:] & p.mask[:, :-1]
    tot = np.sum(np.where(valid, d, 0.0), axis=1)
    return np.where(tot >= 0, 1, -1)


# --------------------------------------------------------------------------------------------- perigee
@dataclass
class Perigee:
    d0: np.ndarray            # signed [mm]
    phi0: np.ndarray          # direction of motion at the PCA [rad]
    x_pca: np.ndarray
    y_pca: np.ndarray


def perigee(xc, yc, R, sigma) -> Perigee:
    rc = np.hypot(xc, yc)
    rc = np.where(rc < 1e-12, 1e-12, rc)
    px, py = xc * (1.0 - R / rc), yc * (1.0 - R / rc)
    ux, uy = (px - xc) / R, (py - yc) / R                      # unit radial vector at the PCA
    tx, ty = -sigma * uy, sigma * ux                            # direction of motion: sigma * z x u
    d0 = px * ty - py * tx
    return Perigee(d0, np.arctan2(ty, tx), px, py)


def arc_length(p: Padded, xc, yc, R, sigma, x0, y0) -> np.ndarray:
    """Signed transverse arc length from (x0, y0) (on the circle) to every hit, along the direction of motion.
    Angles are unwrapped along the hit order, so arcs longer than half a turn are handled."""
    a0 = np.arctan2(y0 - yc, x0 - xc)
    ah = np.arctan2(p.y - yc[:, None], p.x - xc[:, None])
    d = sigma[:, None] * (ah - a0[:, None])
    d = np.remainder(d + np.pi, 2 * np.pi) - np.pi              # first hit: nearest turn
    # unwrap along the hit order so that later hits continue past +-pi
    step = np.diff(d, axis=1)
    step = np.remainder(step + np.pi, 2 * np.pi) - np.pi
    d = np.concatenate([d[:, :1], d[:, :1] + np.cumsum(step, axis=1)], axis=1)
    return d * R[:, None]


def fit_line_sz(s: np.ndarray, z: np.ndarray, mask: np.ndarray, scale: np.ndarray | float = 0.5,
                n_iter: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust (Huber IRLS) fit z = z0 + s * cot(theta) per track. Returns (z0, cot_theta, rms)."""
    m = mask.astype(np.float64)
    w = m.copy()
    z0 = np.zeros(len(s))
    ct = np.zeros(len(s))
    for _ in range(n_iter):
        S0 = np.sum(w, 1)
        S1 = np.sum(w * s, 1)
        S2 = np.sum(w * s * s, 1)
        Z0 = np.sum(w * z, 1)
        Z1 = np.sum(w * s * z, 1)
        det = S0 * S2 - S1**2
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        ct_n = (S0 * Z1 - S1 * Z0) / det
        z0_n = (S2 * Z0 - S1 * Z1) / det
        done = (np.abs(ct_n - ct) < 1e-10) & (np.abs(z0_n - z0) < 1e-7)
        z0, ct = z0_n, ct_n
        res = z - z0[:, None] - ct[:, None] * s
        w = huber_weights(res, scale) * m
        if done.all():
            break
    res = z - z0[:, None] - ct[:, None] * s
    rms = np.sqrt(np.sum(m * res**2, 1) / np.maximum(m.sum(1), 1))
    return z0, ct, rms


# --------------------------------------------------------------------------------------------- two-track vertex
@dataclass
class Vertex:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dz: np.ndarray             # z difference of the two helices at the chosen transverse point
    n_intersections: np.ndarray  # 2, 1 (tangent) or 0 (closest approach of the circles used)
    inside_first_hits: np.ndarray  # chosen point at r <= min(first-hit radii) + tolerance
    ok: np.ndarray


def _z_at(xq, yq, xc, yc, R, sigma, x0, y0, z0, ct):
    """z of a helix at a transverse point on (or near) its circle: z0 + s * cot(theta), s measured from the PCA
    (x0, y0) along the direction of motion, taking the branch nearest the PCA."""
    a0 = np.arctan2(y0 - yc, x0 - xc)
    aq = np.arctan2(yq - yc, xq - xc)
    d = np.remainder(sigma * (aq - a0) + np.pi, 2 * np.pi) - np.pi
    return z0 + d * R * ct


def two_track_vertex(h1: dict, h2: dict, r_first: np.ndarray, tol_mm: float = 5.0) -> Vertex:
    """Vertex of two helices (dicts with xc, yc, R, sigma, x_pca, y_pca, z0, cot), vectorised over pairs.

    Transverse candidates: the two circle intersections, or the midpoint of the circles' closest points if they do
    not intersect. Choice: candidates with r <= min(first-hit radii) + tol are preferred; among those the smallest
    |z1 - z2|."""
    c1 = np.stack([h1["xc"], h1["yc"]], 1)
    c2 = np.stack([h2["xc"], h2["yc"]], 1)
    R1, R2 = h1["R"], h2["R"]
    dvec = c2 - c1
    d = np.hypot(dvec[:, 0], dvec[:, 1])
    d = np.where(d < 1e-12, 1e-12, d)
    e = dvec / d[:, None]
    a = (R1**2 - R2**2 + d**2) / (2 * d)
    h2_ = R1**2 - a**2
    n_int = np.where(h2_ > 0, 2, np.where(h2_ == 0, 1, 0))
    h = np.sqrt(np.clip(h2_, 0, None))
    base = c1 + a[:, None] * e
    perp = np.stack([-e[:, 1], e[:, 0]], 1)
    cand_a = base + h[:, None] * perp
    cand_b = base - h[:, None] * perp
    # no intersection: closest points of the two circles along the centre line
    sep = d > R1 + R2                                       # circles apart
    inner = d < np.abs(R1 - R2)                             # one inside the other
    q1 = c1 + R1[:, None] * e
    q2 = c2 - R2[:, None] * e
    q1i = c1 + np.where(R1 >= R2, 1, -1)[:, None] * R1[:, None] * e
    q2i = c2 + np.where(R1 >= R2, 1, -1)[:, None] * R2[:, None] * e
    mid = np.where(sep[:, None], 0.5 * (q1 + q2), 0.5 * (q1i + q2i))
    cand_a = np.where((n_int == 0)[:, None], mid, cand_a)
    cand_b = np.where((n_int == 0)[:, None], mid, cand_b)

    def zs(cand):
        z1 = _z_at(cand[:, 0], cand[:, 1], h1["xc"], h1["yc"], R1, h1["sigma"], h1["x_pca"], h1["y_pca"],
                   h1["z0"], h1["cot"])
        z2 = _z_at(cand[:, 0], cand[:, 1], h2["xc"], h2["yc"], R2, h2["sigma"], h2["x_pca"], h2["y_pca"],
                   h2["z0"], h2["cot"])
        return z1, z2

    za1, za2 = zs(cand_a)
    zb1, zb2 = zs(cand_b)
    ra = np.hypot(cand_a[:, 0], cand_a[:, 1])
    rb = np.hypot(cand_b[:, 0], cand_b[:, 1])
    ina = ra <= r_first + tol_mm
    inb = rb <= r_first + tol_mm
    dza, dzb = np.abs(za1 - za2), np.abs(zb1 - zb2)
    # prefer the candidate inside the first-hit radius; if both or neither, the one with smaller |dz|
    pick_b = np.where(ina != inb, inb, dzb < dza)
    cand = np.where(pick_b[:, None], cand_b, cand_a)
    z = np.where(pick_b, 0.5 * (zb1 + zb2), 0.5 * (za1 + za2))
    dz = np.where(pick_b, dzb, dza)
    inside = np.where(pick_b, inb, ina)
    return Vertex(cand[:, 0], cand[:, 1], z, dz, n_int, inside, inside & np.isfinite(z))


# --------------------------------------------------------------------------------------------- perigee-parameter fit
def perigee_residuals(p: Padded, d0: np.ndarray, phi0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Distance of every hit to the circle with perigee (d0, phi0) and signed curvature kappa = sigma / R, in a form
    that is smooth through kappa = 0 (straight line): with (a, b) the hit's coordinates along / left of the
    direction of motion at the PCA, e = (kappa (a^2 + b^2) - 2 b) / (1 + sqrt(kappa^2 a^2 + (1 - kappa b)^2)).
    |e| is the distance to the circle; e < 0 means the hit lies on the centre side."""
    s, c = np.sin(phi0)[:, None], np.cos(phi0)[:, None]
    px, py = d0[:, None] * s, -d0[:, None] * c
    wx, wy = p.x - px, p.y - py
    a = wx * c + wy * s
    b = -wx * s + wy * c
    k = kappa[:, None]
    return (k * (a * a + b * b) - 2.0 * b) / (1.0 + np.sqrt(k * k * a * a + (1.0 - k * b) ** 2))


@dataclass
class PerigeeFit:
    d0: np.ndarray
    phi0: np.ndarray
    kappa: np.ndarray          # signed curvature [1/mm]: sigma / R
    rms: np.ndarray
    n_out: np.ndarray
    converged: np.ndarray
    iterations: int

    @property
    def R(self):
        return 1.0 / np.maximum(np.abs(self.kappa), 1e-300)

    @property
    def sigma(self):
        return np.where(self.kappa >= 0, 1, -1)

    def centre(self):
        n = np.stack([-np.sin(self.phi0), np.cos(self.phi0)], 1)
        p = self.d0[:, None] * np.stack([np.sin(self.phi0), -np.cos(self.phi0)], 1)
        c = p + n / np.where(np.abs(self.kappa) < 1e-300, 1e-300, self.kappa)[:, None]
        return c[:, 0], c[:, 1]


def fit_perigee(p: Padded, d0: np.ndarray, phi0: np.ndarray, kappa: np.ndarray, free_kappa: bool,
                scale: np.ndarray | float = 0.1, n_iter: int = 100, tol=(1e-7, 1e-10, 1e-13),
                lam0: float = 1e-3) -> PerigeeFit:
    """Robust (Huber) circle fit in perigee parameters (d0, phi0[, kappa]); monotone in the Huber objective
    (Levenberg-Marquardt on IRLS weights, forward-difference Jacobian). Start from a converged centre-based fit."""
    m = p.mask.astype(np.float64)
    T = len(d0)
    par = np.stack([d0, phi0, kappa], 1).astype(np.float64)
    free = [0, 1, 2] if free_kappa else [0, 1]
    tol = np.asarray(tol)[free]
    h = np.array([1e-6, 1e-9, 1e-12])[free]                   # finite-difference steps (mm, rad, 1/mm)

    def resid(q):
        return perigee_residuals(p, q[:, 0], q[:, 1], q[:, 2])

    def huber_obj(e):
        a = np.abs(e)
        return np.sum(m * np.where(a <= scale, 0.5 * e * e, scale * a - 0.5 * scale**2), axis=1)

    e = resid(par)
    obj = huber_obj(e)
    lam = np.full(T, lam0)
    converged = np.zeros(T, dtype=bool)
    it = 0
    for it in range(1, n_iter + 1):
        w = huber_weights(e, scale) * m
        J = []
        for j, k in enumerate(free):
            q = par.copy()
            q[:, k] += h[j]
            J.append((resid(q) - e) / h[j])
        npar = len(free)
        A = np.empty((T, npar, npar))
        g = np.empty((T, npar))
        for i in range(npar):
            g[:, i] = -np.sum(w * J[i] * e, axis=1)
            for j in range(i, npar):
                A[:, i, j] = A[:, j, i] = np.sum(w * J[i] * J[j], axis=1)
        diag = np.einsum("tii->ti", A)
        Ad = A + (lam[:, None] * diag + 1e-300)[:, :, None] * np.eye(npar)[None]
        try:
            step = np.linalg.solve(Ad, g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros((T, npar))
            for t in range(T):
                step[t] = np.linalg.lstsq(Ad[t], g[t], rcond=None)[0]
        q = par.copy()
        q[:, free] += step
        en = resid(q)
        obj_n = huber_obj(en)
        ok = (obj_n <= obj) & ~converged
        small = np.all(np.abs(step) < tol[None], axis=1)
        flat = (obj - obj_n) <= 1e-13 * np.maximum(obj, 1e-30)
        converged |= ok & (small | flat)
        converged |= ~ok & (lam >= 1e8)
        par = np.where(ok[:, None], q, par)
        e = np.where(ok[:, None], en, e)
        obj = np.where(ok, obj_n, obj)
        lam = np.where(ok, np.maximum(lam / 3.0, 1e-9), np.minimum(lam * 10.0, 1e9))
        if converged.all():
            break
    par[:, 1] = np.remainder(par[:, 1] + np.pi, 2 * np.pi) - np.pi
    n = np.maximum(m.sum(1), 1)
    rms = np.sqrt(np.sum(m * e * e, axis=1) / n)
    n_out = np.sum(m * (np.abs(e) > 3.0 * np.asarray(scale)), axis=1).astype(np.int64)
    return PerigeeFit(par[:, 0], par[:, 1], par[:, 2], rms, n_out, converged, it)


def arc_length_perigee(p: Padded, d0: np.ndarray, phi0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Signed transverse arc length from the PCA to every hit's projection on the circle, smooth through
    kappa = 0: s = atan2(kappa a, 1 - kappa b) / kappa (-> a for kappa -> 0), unwrapped along the hit order so
    loopers beyond half a turn keep increasing."""
    s_, c_ = np.sin(phi0)[:, None], np.cos(phi0)[:, None]
    wx, wy = p.x - d0[:, None] * s_, p.y + d0[:, None] * c_
    a = wx * c_ + wy * s_
    b = -wx * s_ + wy * c_
    k = kappa[:, None]
    ang = np.arctan2(k * a, 1.0 - k * b)                       # turning angle from the PCA, in (-pi, pi]
    small = np.abs(k) < 1e-12
    step = np.diff(ang, axis=1)
    step = np.remainder(step + np.pi, 2 * np.pi) - np.pi
    ang = np.concatenate([ang[:, :1], ang[:, :1] + np.cumsum(step, axis=1)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(small, a, ang / np.where(small, 1.0, k))
    return s
