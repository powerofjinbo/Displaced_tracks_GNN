"""d0 resolution from toy helices through the generic-detector layers, smeared with residuals measured on W muons
(brief §5b, first of three resolution estimates).

Geometry: per (volume, layer) from the module envelopes of the §2.6 scan: barrel layers as cylinders (median module
radius, z extent), endcap layers as disks (median module z, r extent). One hit per crossed layer.
Residuals: measured on W muons with the truth fit of `hnl.truth.fit_events` (robust sigma = 1.4826 MAD per volume):
the fine transverse residual (distance to the fitted circle) and the coarse one (barrel: z - z_helix at the hit
radius; endcap: r - r_helix at the hit z). Toys are smeared in r-phi by the fine sigma and along z (barrel) or r
(endcap) by the coarse sigma, then fitted exactly like the truth (fixed R from the true pT, or free curvature).

    python -m hnl.truth.toys --envelopes <scan>/module_envelopes.parquet --out <dir> [--calib-events 1000]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.data.npz import load_event
from hnl.geometry.ids import BARREL_VOLUMES, cylindrical_to_cartesian, decode_geometry_id
from hnl.truth import helix as hx
from hnl.truth.fit_events import SCALES, fit_event, hit_scales


# --------------------------------------------------------------------------------------------- geometry
def layer_geometry(envelopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, lay), g in envelopes.groupby(["volume", "layer"]):
        if v in BARREL_VOLUMES:
            rows.append(dict(volume=v, layer=lay, kind="barrel", r=float(np.median(0.5 * (g.r_lo + g.r_hi))),
                             zmin=float(g.z_lo.min()), zmax=float(g.z_hi.max()), z=np.nan, rmin=np.nan, rmax=np.nan))
        else:
            for side, gs in ((-1, g[g.z_hi < 0]), (1, g[g.z_lo > 0])):
                if len(gs):
                    rows.append(dict(volume=v, layer=lay, kind="endcap", z=float(np.median(0.5 * (gs.z_lo + gs.z_hi))),
                                     rmin=float(gs.r_lo.min()), rmax=float(gs.r_hi.max()), r=np.nan, zmin=np.nan,
                                     zmax=np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- helix propagation
def _ab(kappa, s):
    """Local coordinates along (a) and left of (b) the direction at the PCA after arc length s (stable at kappa=0)."""
    a = s * np.sinc(kappa * s / np.pi)
    b = 0.5 * kappa * s * s * np.sinc(kappa * s / (2 * np.pi)) ** 2
    return a, b


def position(d0, phi0, kappa, z0, cot, s):
    a, b = _ab(kappa, s)
    t = np.stack([np.cos(phi0), np.sin(phi0)], -1)
    n = np.stack([-np.sin(phi0), np.cos(phi0)], -1)
    p = -d0[..., None] * n
    xy = p + a[..., None] * t + b[..., None] * n
    return xy[..., 0], xy[..., 1], z0 + cot * s


def s_at_radius(d0, kappa, R):
    """Smallest positive arc length at which the helix reaches transverse radius R (first, outgoing crossing),
    NaN if never: |P(s)|^2 = d0^2 + 4 u (1 - kappa d0) / kappa^2 with u = sin^2(kappa s / 2)."""
    k = np.where(np.abs(kappa) < 1e-15, 1e-15, kappa)
    num = (R**2 - d0**2) * k * k
    den = 4.0 * (1.0 - k * d0)
    u = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    ok = (u >= 0) & (u <= 1) & (R >= np.abs(d0))
    s = 2.0 * np.arcsin(np.sqrt(np.clip(u, 0, 1))) / np.abs(k)
    return np.where(ok, s, np.nan)


def toy_tracks(n, pt, d0_abs, rng, geo: pd.DataFrame, res: dict, eta_max=2.5, z0_sigma=55.0):
    """n toy muons with |d0| = d0_abs (random sign), true pT, uniform phi0 and eta. Returns (Padded, truth dict)."""
    q = rng.choice([-1, 1], n)
    sigma = -q
    R = hx.MM_PER_GEV * pt
    kappa = sigma / R * np.ones(n)
    d0 = d0_abs * rng.choice([-1, 1], n)
    phi0 = rng.uniform(-np.pi, np.pi, n)
    eta = rng.uniform(-eta_max, eta_max, n)
    cot = np.sinh(eta)
    z0 = rng.normal(0, z0_sigma, n)
    s_max = 0.9 * np.pi / np.abs(kappa)                   # first half turn only
    S, VOL = [], []
    for _, L in geo.iterrows():
        if L.kind == "barrel":
            s = s_at_radius(d0, kappa, L.r)
            z = z0 + cot * s
            ok = np.isfinite(s) & (z >= L.zmin) & (z <= L.zmax)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (L.z - z0) / cot
            x, y, _ = position(d0, phi0, kappa, z0, cot, np.where(np.isfinite(s), s, 0.0))
            r = np.hypot(x, y)
            ok = np.isfinite(s) & (s > 0) & (r >= L.rmin) & (r <= L.rmax)
        ok &= (s > 0) & (s < s_max)
        S.append(np.where(ok, s, np.nan))
        VOL.append(np.full(n, L.volume))
    S = np.stack(S, 1)
    VOL = np.stack(VOL, 1)
    order = np.argsort(np.where(np.isfinite(S), S, np.inf), axis=1)
    S = np.take_along_axis(S, order, 1)
    VOL = np.take_along_axis(VOL, order, 1)
    mask = np.isfinite(S)
    x, y, z = position(d0[:, None] * np.ones_like(S), phi0[:, None] * np.ones_like(S),
                       kappa[:, None] * np.ones_like(S), z0[:, None] * np.ones_like(S),
                       cot[:, None] * np.ones_like(S), np.where(mask, S, 0.0))
    # smearing: r-phi fine everywhere; z (barrel) or r (endcap) coarse
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    fine = np.vectorize(lambda v: res["fine"].get(int(v), 0.03))(VOL)
    coarse = np.vectorize(lambda v: res["coarse"].get(int(v), 0.03))(VOL)
    barrel = np.isin(VOL, BARREL_VOLUMES)
    phi = phi + rng.normal(0, 1, S.shape) * fine / np.maximum(r, 1.0)
    r = np.where(barrel, r, r + rng.normal(0, 1, S.shape) * coarse)
    z = np.where(barrel, z + rng.normal(0, 1, S.shape) * coarse, z)
    x, y = r * np.cos(phi), r * np.sin(phi)
    keep = mask.sum(1) >= 5
    # compact each track's hits to the front (already sorted: NaNs last)
    p = hx.Padded(np.where(mask, x, 0)[keep], np.where(mask, y, 0)[keep], np.where(mask, z, 0)[keep], mask[keep],
                  np.flatnonzero(keep), mask[keep].sum(1), np.where(mask, 0, -1)[keep])
    truth = dict(d0=d0[keep], phi0=phi0[keep], kappa=kappa[keep], z0=z0[keep], cot=cot[keep], vol=VOL[keep])
    return p, truth


def fit_toys(p: hx.Padded, truth: dict, free: bool):
    vol = truth["vol"]
    st = np.vectorize(lambda v: SCALES[int(v)][0])(vol)
    sz = np.vectorize(lambda v: SCALES[int(v)][1])(vol)
    R0 = 1.0 / np.abs(truth["kappa"])
    sigma = np.sign(truth["kappa"]).astype(np.int64)
    f0 = hx.fit_circle(p, R0, sigma, scale=st)
    pg = hx.perigee(f0.xc, f0.yc, f0.R, f0.sigma)
    f = hx.fit_perigee(p, pg.d0, pg.phi0, sigma / R0, free_kappa=False, scale=st)
    if free:
        f = hx.fit_perigee(p, f.d0, f.phi0, sigma / R0, free_kappa=True, scale=st)
    s = hx.arc_length_perigee(p, f.d0, f.phi0, f.kappa)
    z0, _, _ = hx.fit_line_sz(s, p.z, p.mask, scale=sz)
    return f.d0 - truth["d0"], z0 - truth["z0"], f.converged


# --------------------------------------------------------------------------------------------- residual calibration
def measure_residuals(n_events: int, sample: str = "hnl10_100") -> dict:
    """Fine (transverse) and coarse (z in barrel, r in endcaps) residual sigmas per volume, from W muons."""
    from hnl.data.splits import dropped_events
    drop = dropped_events()[sample]
    fine, coarse = {}, {}
    for i in range(n_events):
        if i in drop:
            continue
        ev = load_event(sample, i)
        df, _ = fit_event(ev, field_sign=-1)
        w = df[df.label_w & (df.n_hits >= 8)]
        if not len(w):
            continue
        pid = ev["pid"]
        xyz = cylindrical_to_cartesian(ev["x"])
        vol = decode_geometry_id(ev["layers"])["volume"]
        order = hx.order_hits(pid, ev["layerless_true_edges"])
        for _, t in w.iterrows():
            rows = order[pid[order] == t.particle_id]
            p = hx.pad_tracks(np.zeros(len(rows), int), xyz[rows])
            d0, phi0, k = np.array([t.fix_d0]), np.array([t.fix_phi0]), np.array([t.fix_kappa])
            e = hx.perigee_residuals(p, d0, phi0, k)[0]
            s = hx.arc_length_perigee(p, d0, phi0, k)[0]
            zres = xyz[rows, 2] - (t.fix_z0 + t.fix_cot * s)
            # endcap coarse residual: r of the hit minus r of the helix at the hit's z
            s_at_z = (xyz[rows, 2] - t.fix_z0) / t.fix_cot if t.fix_cot != 0 else np.full(len(rows), np.nan)
            hxp, hyp, _ = position(np.full(len(rows), t.fix_d0), np.full(len(rows), t.fix_phi0),
                                   np.full(len(rows), t.fix_kappa), np.full(len(rows), t.fix_z0),
                                   np.full(len(rows), t.fix_cot), s_at_z)
            rres = np.hypot(xyz[rows, 0], xyz[rows, 1]) - np.hypot(hxp, hyp)
            for v, a, zr, rr in zip(vol[rows], e, zres, rres):
                fine.setdefault(int(v), []).append(a)
                coarse.setdefault(int(v), []).append(zr if v in BARREL_VOLUMES else rr)
    mad = lambda x: float(1.4826 * np.median(np.abs(np.asarray(x) - np.median(x))))
    return {"fine": {v: mad(x) for v, x in fine.items()}, "coarse": {v: mad(x) for v, x in coarse.items()},
            "n": {v: len(x) for v, x in fine.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelopes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-events", type=int, default=1000)
    ap.add_argument("--n", type=int, default=2000, help="toys per (pT, |d0|) point")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = measure_residuals(a.calib_events)
    with open(os.path.join(a.out, "residuals.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("residual sigmas [mm]", json.dumps(res))
    geo = layer_geometry(pd.read_parquet(a.envelopes))
    geo.to_csv(os.path.join(a.out, "layers.csv"), index=False)
    rng = np.random.default_rng(0)
    rows = []
    for pt in (1.5, 3.0, 7.0, 20.0, 50.0):
        for d0a in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0):
            p, truth = toy_tracks(a.n, pt, d0a, rng, geo, res)
            for free in (False, True):
                dd0, dz0, conv = fit_toys(p, truth, free)
                q16, q50, q84 = np.percentile(dd0, [16, 50, 84])
                rows.append(dict(pt=pt, d0_abs=d0a, fit="free" if free else "fixed", n=len(dd0),
                                 conv=float(conv.mean()), d0_bias=float(q50), d0_sigma68=float(0.5 * (q84 - q16)),
                                 d0_sigma_mad=float(1.4826 * np.median(np.abs(dd0 - q50))),
                                 d0_abs_err_p90=float(np.percentile(np.abs(dd0), 90)),
                                 z0_sigma68=float(0.5 * np.diff(np.percentile(dz0, [16, 84]))[0]),
                                 n_hits_median=float(np.median(p.n_hits))))
            print(rows[-2], rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "toy_resolution.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
