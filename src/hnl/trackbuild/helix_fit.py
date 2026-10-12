"""Weighted, robust helix fit of labelled hits in each hit's module frame (attach v2, resolution calibration).

Per track (label >= 0 with >= min_hits hits), vectorised over tracks:
  transverse: a robust (Huber) parabola along the hits' principal axis, in coordinates centred on their weighted
  mean, gives a start circle through three of its points; then Gauss-Newton steps with backtracking on the Huber
  objective of the geometric distance |p - c| - R over s_n (s_n: the module-frame sigma along the circle normal at
  the hit, hnl.geometry.resolution.sigma_transverse);
  longitudinal: z = z_ref + cot * s with s = R * (theta - theta_ref) the transverse arc length from the hits' circular
  mean angle theta_ref around the centre, weights 1 / s_z^2 (resolution.sigma_z, which depends on cot: iterated).
Every step is a function of hit differences and of each hit's module axes, with fixed iteration counts, so the fit
commutes with translations of the hits and with rotations about z (module axes rotated alike) up to rounding.
No multiple-scattering term (the residual widths of low-momentum tracks include it).
With leverage=True the per-hit leverages h of both weighted least-squares problems are returned too: e^2 / (1 - h)
then has the expectation s^2 for a correctly weighted fit (used by the resolution calibration).
"""

from __future__ import annotations

import numpy as np

from hnl.geometry import resolution as RS

HUBER = 3.0                     # pulls beyond this get Huber weights (outlier hits)


def _wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def _pad(labels, min_hits):
    labels = np.asarray(labels)
    ok = np.flatnonzero(labels >= 0)
    trk, inv, cnt = np.unique(labels[ok], return_inverse=True, return_counts=True)
    good = cnt >= min_hits
    sel = ok[good[inv]]
    tid = np.searchsorted(trk[good], labels[sel])
    order = np.argsort(tid, kind="stable")
    sel, tid = sel[order], tid[order]
    T = int(good.sum())
    n = np.bincount(tid, minlength=T)
    H = int(n.max()) if T else 0
    pos = np.arange(len(sel)) - np.repeat(np.cumsum(n) - n, n)
    rows = np.full((T, H), -1, dtype=np.int64)
    rows[tid, pos] = sel
    return trk[good], n, rows


def fit(x, y, z, labels, volume, phi_c, min_hits: int = 4, res: dict | None = None, n_iter: int = 10,
        nz_iter: int = 4, leverage: bool = False) -> dict:
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    volume = np.asarray(volume)
    phi_c = np.asarray(phi_c, dtype=np.float64)
    trk, n_hits, rows = _pad(labels, min_hits)
    T = len(trk)
    if T == 0:
        return {"track": trk}
    m = rows >= 0
    rr = np.maximum(rows, 0)
    mf = m.astype(np.float64)
    X, Y, Z = x[rr], y[rr], z[rr]
    vol = np.where(m, volume[rr], volume[rr[:, :1]])           # padding: any valid volume (masked out)
    pc = phi_c[rr]
    fr = RS.Frames(vol, pc, res)
    su = fr.su
    # centred coordinates and the hits' principal axis (translation / rotation equivariant start)
    w0 = mf / su**2
    xm = (w0 * X).sum(1) / w0.sum(1)
    ym = (w0 * Y).sum(1) / w0.sum(1)
    Xc, Yc = (X - xm[:, None]) * mf, (Y - ym[:, None]) * mf
    alpha = 0.5 * np.arctan2(2 * (w0 * Xc * Yc).sum(1), (w0 * (Xc * Xc - Yc * Yc)).sum(1))
    ca, sa = np.cos(alpha)[:, None], np.sin(alpha)[:, None]
    xp, yp = ca * Xc + sa * Yc, -sa * Xc + ca * Yc
    L = np.maximum(np.abs(xp).max(1), 1e-6)[:, None]
    tt = xp / L
    # robust (Huber IRLS) parabola y' = a + b t + c t^2 along the principal axis
    w = w0.copy()
    cols = [mf, tt * mf, tt * tt * mf]
    for _ in range(15):
        A = np.empty((T, 3, 3))
        b = np.empty((T, 3))
        for i in range(3):
            b[:, i] = (w * cols[i] * yp).sum(1)
            for j in range(i, 3):
                A[:, i, j] = A[:, j, i] = (w * cols[i] * cols[j]).sum(1)
        A += 1e-12 * np.trace(A, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 1e-300 * np.eye(3)
        q = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        rp = (yp - q[:, :1] - q[:, 1:2] * tt - q[:, 2:3] * tt * tt) * mf
        a_ = np.abs(rp) / su
        w = w0 * np.where(a_ <= HUBER, 1.0, HUBER / np.maximum(a_, 1e-300))
    # circle through three points of the parabola (ends and middle of the hits' span)
    t3 = np.stack([np.where(m, tt, np.inf).min(1), np.zeros(T), np.where(m, tt, -np.inf).max(1)], 1)
    t3[:, 1] = 0.5 * (t3[:, 0] + t3[:, 2])
    P = np.stack([t3 * L, q[:, :1] + q[:, 1:2] * t3 + q[:, 2:3] * t3 * t3], -1)          # (T, 3, 2)
    (ax_, ay_), (bx_, by_), (cx_, cy_) = (P[:, k].T for k in range(3))
    dd = 2 * (ax_ * (by_ - cy_) + bx_ * (cy_ - ay_) + cx_ * (ay_ - by_))
    dd = np.where(np.abs(dd) < 1e-300, 1e-300, dd)
    a2, b2, c2 = ax_**2 + ay_**2, bx_**2 + by_**2, cx_**2 + cy_**2
    ux = (a2 * (by_ - cy_) + b2 * (cy_ - ay_) + c2 * (ay_ - by_)) / dd
    uy = (a2 * (cx_ - bx_) + b2 * (ax_ - cx_) + c2 * (bx_ - ax_)) / dd
    R = np.hypot(ux - ax_, uy - ay_)
    big = R > 1e7                                             # numerically straight: cap the radius
    k = np.where(big, 1e7 / np.maximum(R, 1e-300), 1.0)
    ux, uy, R = (ux - bx_) * k + bx_, (uy - by_) * k + by_, np.minimum(R, 1e7)
    cx, cy = ca[:, 0] * ux - sa[:, 0] * uy, sa[:, 0] * ux + ca[:, 0] * uy

    def geo(cx, cy, R):
        dx, dy = Xc - cx[:, None], Yc - cy[:, None]
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        nvec = np.stack([dx / d, dy / d], -1)
        e = (d - R[:, None]) * mf
        st = fr.transverse(nvec)
        return nvec, e, st

    def weights(e, s):
        a = np.abs(e) / s
        return mf * np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-300)) / s**2

    def objective(cx, cy, R):
        _, e, st = geo(cx, cy, R)
        a = np.abs(e) / st
        return (mf * np.where(a <= HUBER, 0.5 * a * a, HUBER * a - 0.5 * HUBER**2)).sum(1)
    obj = objective(cx, cy, R)
    for _ in range(n_iter):                  # Gauss-Newton (IRLS) steps with backtracking on the Huber objective
        nvec, e, st = geo(cx, cy, R)
        W = weights(e, st)
        J = np.stack([-nvec[..., 0], -nvec[..., 1], -mf], -1) * mf[..., None]
        JW = J * W[..., None]
        N = np.einsum("thi,thj->tij", JW, J)
        g = -np.einsum("thi,th->ti", JW, e)
        N += 1e-12 * np.trace(N, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 1e-300 * np.eye(3)
        step = np.linalg.solve(N, g[:, :, None])[:, :, 0]
        lam = np.ones(T)
        todo = np.ones(T, dtype=bool)
        for _ in range(12):
            ncx, ncy, nR = cx + lam * step[:, 0], cy + lam * step[:, 1], np.abs(R + lam * step[:, 2])
            nobj = objective(ncx, ncy, nR)
            ok = todo & (nobj <= obj)
            cx, cy, R, obj = (np.where(ok, nv, ov) for nv, ov in ((ncx, cx), (ncy, cy), (nR, R), (nobj, obj)))
            todo &= ~ok
            if not todo.any():
                break
            lam = np.where(todo, 0.5 * lam, lam)
    nvec, e, st = geo(cx, cy, R)
    W = weights(e, st)
    # arc length from the hits' circular mean angle around the centre
    th = np.arctan2(nvec[..., 1], nvec[..., 0])
    th_ref = np.arctan2((mf * np.sin(th)).sum(1), (mf * np.cos(th)).sum(1))
    dth = _wrap(th - th_ref[:, None]) * mf
    s = R[:, None] * dth
    tvec = np.stack([-nvec[..., 1], nvec[..., 0]], -1)
    # s-z line, weights from the slope-dependent z sigma
    cot = np.zeros(T)
    z_ref = np.zeros(T)
    wz = mf.copy()
    for it in range(nz_iter + 1):
        S0, S1, S2 = wz.sum(1), (wz * s).sum(1), (wz * s * s).sum(1)
        Z0, Z1 = (wz * Z).sum(1), (wz * s * Z).sum(1)
        det = S0 * S2 - S1 * S1
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        cot = (S0 * Z1 - S1 * Z0) / det
        z_ref = (S2 * Z0 - S1 * Z1) / det
        ez = (Z - z_ref[:, None] - cot[:, None] * s) * mf
        sz = fr.z(tvec, cot[:, None])
        wz = weights(ez, sz)
    pull_t, pull_z = e / st, ez / sz
    out = {"track": trk, "n_hits": n_hits, "xc": cx + xm, "yc": cy + ym, "R": R, "cot": cot, "z_ref": z_ref,
           "theta_ref": th_ref, "dtheta_min": np.where(m, dth, np.inf).min(1),
           "dtheta_max": np.where(m, dth, -np.inf).max(1),
           "rms_n": np.sqrt((mf * pull_t**2).sum(1) / n_hits), "zrms_n": np.sqrt((mf * pull_z**2).sum(1) / n_hits),
           "n_out": (m & ((np.abs(pull_t) > HUBER) | (np.abs(pull_z) > HUBER))).sum(1),
           "rows": rows, "dtheta": dth, "volume": vol, "pull_t": pull_t, "pull_z": pull_z}
    if leverage:
        u, v = fr.u, fr.v
        nu = nvec[..., 0] * u[..., 0] + nvec[..., 1] * u[..., 1]
        nv = nvec[..., 0] * v[..., 0] + nvec[..., 1] * v[..., 1]
        tu = tvec[..., 0] * u[..., 0] + tvec[..., 1] * u[..., 1]
        tv = tvec[..., 0] * v[..., 0] + tvec[..., 1] * v[..., 1]
        J = np.stack([-nvec[..., 0], -nvec[..., 1], -mf], -1) * mf[..., None]
        N = np.einsum("thi,thj->tij", J * W[..., None], J)
        Ni = np.linalg.pinv(N)
        h_t = W * np.einsum("thi,tij,thj->th", J, Ni, J)
        Jz = np.stack([mf, s], -1)
        Nz = np.einsum("thi,thj->tij", Jz * wz[..., None], Jz)
        h_z = wz * np.einsum("thi,tij,thj->th", Jz, np.linalg.pinv(Nz), Jz)
        barrel = fr.barrel
        c2 = cot[:, None] ** 2
        out.update({"h_t": h_t, "h_z": h_z, "at_u": nu * nu, "at_v": nv * nv, "az_u": c2 * tu * tu,
                    "az_v": np.where(barrel, 1.0, c2 * tv * tv),
                    "et2": e * e / np.clip(1 - h_t, 0.05, None), "ez2": ez * ez / np.clip(1 - h_z, 0.05, None),
                    "pull_t": e / (st * np.sqrt(np.clip(1 - h_t, 0.05, None))),
                    "pull_z": ez / (sz * np.sqrt(np.clip(1 - h_z, 0.05, None)))})
    return out


def hit_table(f: dict) -> dict:
    """Per-hit rows of a fit (only hits of fitted tracks): track label, input row, volume and the residual columns."""
    m = f["rows"] >= 0
    out = {"track": np.repeat(f["track"], m.sum(1)), "row": f["rows"][m], "volume": f["volume"][m]}
    for k in ("pull_t", "pull_z", "at_u", "at_v", "az_u", "az_v", "et2", "ez2"):
        if k in f:
            out[k] = f[k][m]
    return out
