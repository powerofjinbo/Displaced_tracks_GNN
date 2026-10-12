"""Module-frame hit resolution of the generic detector (Stage 1: attach v2, strict ttbar-only constants).

Every hit lies on a planar module and is smeared along the module's two measurement axes: u (fine) and v (coarse).
Barrel modules (volumes 8, 13, 17): u is the tangent of the module's centre azimuth phi_c, v = z. Endcap modules
(7, 9, 12, 14, 16, 18): u is the same tangent, v the radial direction at phi_c (endcap strips are radial, so the
coarse coordinate is r, not z). Pixels have two fine axes. A residual along a transverse unit vector n then has
    s_n^2 = s_u^2 (n.u)^2 + s_v^2 (n.v)^2,
and the z residual of a hit against a track's s-z line (z = z_ref + cot * s, s the transverse arc length)
    barrel: s_z^2 = s_v^2 + cot^2 (t.u)^2 s_u^2,     endcap: s_z^2 = cot^2 ((t.u)^2 s_u^2 + (t.v)^2 s_v^2)
(t the transverse direction of the track at the hit: an endcap hit's z is the disk's, its r error moves s). The
isotropic per-volume tables used before (hnl.trackbuild.ldc until 2026-10-03: HNL W-muon residuals, the endcaps'
coarse r sigma used as a z sigma) are off by large factors for tracks that do not cross an endcap strip radially.

RES holds (s_u, s_v) per volume in mm, measured on ttbar TRAIN events only (`python -m hnl.geometry.resolution
measure`). Strips: overlap double hits of prompt charged particles without electrons (|d0| < 1 mm, pT > 2 GeV, >= 8
hits): two hits of one particle on overlapping modules of one layer, a few mm apart, whose separation across and
along the track's local direction (from the helix fit of all its hits) is free of multiple scattering and of the
fit's own errors. Pixels: leverage-corrected residuals of the weighted helix fits (hnl.trackbuild.helix_fit) of ttbar
W muons with pT > 20 GeV, iterated to self-consistency (overlap pairs of all prompt tracks give 20-50 % larger pixel
sigmas, which leave the W-muon pulls at 0.6-0.8, while muon-only pairs agree with the W-muon fits; for the strips all
three agree within about 15 %). In both, each sigma is rescaled until the median squared pull of the rows its term
dominates matches chi2_1 (robust against the few % of hits that lie mm off their particle's helix). Global-fit
residuals of tracks below ~20 GeV are dominated by multiple scattering (a self-consistent calibration on prompt tracks
above 10 GeV diverges). Multiple scattering (Highland) is a separate term the caller adds:
theta0 = 0.0136 / p * sqrt(x) * (1 + 0.038 ln x), x = x/X0 of one module crossing (X_X0 at normal incidence /
|cos alpha|); X_X0 are nominal values (not measured).

    python -m hnl.geometry.resolution measure [--events 800] [--wmu-events 3000]     (ttbar train)
Data check on val (pull widths per volume and population): python -m hnl.eval.attach_pulls.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

PIXEL = (7, 8, 9)
BARREL = (8, 13, 17)
VOLUMES = (7, 8, 9, 12, 13, 14, 16, 17, 18)
# (s_u, s_v) [mm] per volume, measured by `measure` (see RES_PROVENANCE). The endcap long strips' fine axis is the
# barrel long strips' (endcap overlap pairs are dominated by the coarse axis: 5-6 usable pairs).
RES = {
    7: (0.01826, 0.02000), 8: (0.02034, 0.02467), 9: (0.01594, 0.01599),
    12: (0.02661, 0.39927), 13: (0.02608, 0.39965), 14: (0.02722, 0.38988),
    16: (0.0347, 3.19191), 17: (0.0346, 3.12244), 18: (0.0347, 3.21425),
}
RES_PROVENANCE = ("ttbar train only (splits/ttbar_train.txt), `measure --events 800 --wmu-events 3000 --pt-min 2` "
                  "(2026-10-03): strips from 50,254 overlap double hits of prompt non-electron tracks in events 0-799 "
                  "(rows u / v: 12: 73 / 1866, 13: 12815 / 12815, 14: 78 / 1338, 16: 5 / 3646, 17: 3512 / 3512, "
                  "18: 6 / 3616); pixels from the weighted helix fits of 539 W muons with pT > 20 GeV in events 0-2999 "
                  "(rows 7: 259 / 259, 8: 2397 / 2315, 9: 235 / 235); overlap pixel values 7: (0.0227, 0.0239), "
                  "8: (0.0302, 0.0349), 9: (0.0239, 0.0253). Cross-check, overlap pairs of prompt muons only (4,237 "
                  "pairs, events 0-2999): pixels 7: (0.0178, 0.0199), 8: (0.0235, 0.0246), 9: (0.0169, 0.0179), "
                  "strips 13: (0.0237, 0.368), 17: (0.0407, 3.13), endcap coarse 12 / 14 / 16 / 18: 0.38 / 0.34 / "
                  "3.20 / 2.93")
# x/X0 of one module crossing at normal incidence, nominal (not measured): pixel / short strips / long strips
X_X0 = {7: 0.015, 8: 0.015, 9: 0.015, 12: 0.025, 13: 0.025, 14: 0.025, 16: 0.030, 17: 0.030, 18: 0.030}
MIN_PROJ = 0.1                     # floor of a projected sigma, in units of s_u (grazing crossings of barrel modules)
_ENVELOPES = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived", "scans",
                          "v1", "module_envelopes.parquet")
_PHI_C = None


def _table(volume, table: dict, k: int) -> np.ndarray:
    volume = np.asarray(volume)
    out = np.full(volume.shape, np.nan)
    for v, val in table.items():
        out[volume == v] = val[k] if isinstance(val, tuple) else val
    return out


def is_barrel(volume) -> np.ndarray:
    return np.isin(np.asarray(volume), BARREL)


def axes(volume, phi_c):
    """Unit vectors (u, v) of each hit's module, (N, 3) each."""
    phi_c = np.asarray(phi_c, dtype=np.float64)
    c, s = np.cos(phi_c), np.sin(phi_c)
    zero = np.zeros_like(c)
    u = np.stack([-s, c, zero], -1)
    barrel = is_barrel(volume)[..., None]
    v = np.where(barrel, np.stack([zero, zero, np.ones_like(c)], -1), np.stack([c, s, zero], -1))
    return u, v


def sigmas(volume, res: dict | None = None):
    res = RES if res is None else res
    return _table(volume, res, 0), _table(volume, res, 1)


class Frames:
    """Per-hit module frames and sigmas, computed once (for repeated projections, e.g. inside a fit)."""

    def __init__(self, volume, phi_c, res: dict | None = None):
        self.su, self.sv = sigmas(volume, res)
        self.u, self.v = axes(volume, phi_c)
        self.barrel = is_barrel(volume)

    def transverse(self, n) -> np.ndarray:
        """Sigma of a residual along the transverse unit vector n (..., 2): s_u^2 (n.u)^2 + s_v^2 (n.v)^2, floored."""
        nu = n[..., 0] * self.u[..., 0] + n[..., 1] * self.u[..., 1]
        nv = n[..., 0] * self.v[..., 0] + n[..., 1] * self.v[..., 1]
        su, sv = self.su, self.sv
        return np.sqrt(su * su * nu * nu + sv * sv * nv * nv + (MIN_PROJ * su) ** 2)

    def z(self, t, cot) -> np.ndarray:
        """Sigma of a hit's z residual against an s-z line with slope cot, t (..., 2) the track's transverse
        direction."""
        tu = t[..., 0] * self.u[..., 0] + t[..., 1] * self.u[..., 1]
        tv = t[..., 0] * self.v[..., 0] + t[..., 1] * self.v[..., 1]
        cot = np.asarray(cot, dtype=np.float64)
        su, sv = self.su, self.sv
        s2 = np.where(self.barrel, sv * sv + cot * cot * tu * tu * su * su,
                      cot * cot * (tu * tu * su * su + tv * tv * sv * sv))
        return np.sqrt(s2 + (MIN_PROJ * su) ** 2)


def sigma_transverse(volume, phi_c, n, res: dict | None = None) -> np.ndarray:
    return Frames(volume, phi_c, res).transverse(n)


def sigma_z(volume, phi_c, t, cot, res: dict | None = None) -> np.ndarray:
    return Frames(volume, phi_c, res).z(t, cot)


def momentum_gev(R_mm, cot) -> np.ndarray:
    from hnl.truth.helix import MM_PER_GEV
    return np.asarray(R_mm, dtype=np.float64) / MM_PER_GEV * np.sqrt(1.0 + np.asarray(cot, dtype=np.float64) ** 2)


def crossing_x_x0(volume, phi_c, t, cot) -> np.ndarray:
    """x/X0 of one module crossing of a track with transverse direction t (..., 2) and slope cot."""
    x0 = _table(volume, X_X0, 0)
    cot = np.asarray(cot, dtype=np.float64)
    norm = np.sqrt(1.0 + cot * cot)
    phi_c = np.asarray(phi_c, dtype=np.float64)
    cos_barrel = np.abs(t[..., 0] * np.cos(phi_c) + t[..., 1] * np.sin(phi_c)) / norm
    cos_endcap = np.abs(cot) / norm
    cosa = np.where(is_barrel(volume), cos_barrel, cos_endcap)
    return x0 / np.maximum(cosa, 0.05)


def highland(p_gev, x_x0) -> np.ndarray:
    """RMS plane scattering angle [rad] of a unit-charge particle (beta = 1) through x/X0 of material."""
    x = np.maximum(np.asarray(x_x0, dtype=np.float64), 1e-6)
    return 0.0136 / np.maximum(np.asarray(p_gev, dtype=np.float64), 1e-3) * np.sqrt(x) * (1.0 + 0.038 * np.log(x))


def ms_sigmas(theta0, L_t, cot):
    """Transverse and z displacement sigmas after one scattering theta0 a transverse distance L_t before the hit."""
    k = np.sqrt(1.0 + np.asarray(cot, dtype=np.float64) ** 2)
    return theta0 * L_t * k, theta0 * L_t * k * k


# --------------------------------------------------------------------------------------------- per-hit geometry
def module_phi_c(geoid) -> np.ndarray:
    """Centre azimuth of each hit's module (from the section-2.6 scan's module envelopes)."""
    global _PHI_C
    if _PHI_C is None:
        import pandas as pd
        m = pd.read_parquet(_ENVELOPES, columns=["geoid", "phi_c"])
        _PHI_C = (m.geoid.to_numpy().astype(np.int64), m.phi_c.to_numpy().astype(np.float64))
        o = np.argsort(_PHI_C[0])
        _PHI_C = (_PHI_C[0][o], _PHI_C[1][o])
    ids, phis = _PHI_C
    geoid = np.asarray(geoid, dtype=np.int64)
    k = np.clip(np.searchsorted(ids, geoid), 0, len(ids) - 1)
    if not (ids[k] == geoid).all():
        raise KeyError(f"{int((ids[k] != geoid).sum())} hits on modules without an envelope")
    return phis[k]


def hit_geometry(eids) -> dict:
    """Per event id: dict(volume, layer_id = volume * 1000 + layer, phi_c) per hit, in graph node (= row) order."""
    from hnl.data.convert import read_table
    from hnl.eval.particles import CODE_SAMPLE, DERIVED
    eids = np.asarray(eids, dtype=np.int64)
    codes = eids // 100_000_000
    out = {}
    for c in np.unique(codes):
        ev = (eids[codes == c] % 100_000_000).tolist()
        h = read_table(os.path.join(DERIVED, "events", "v1"), "hits", CODE_SAMPLE[int(c)], events=ev,
                       columns=["event", "row", "volume", "layer", "geoid"])
        for e, g in h.groupby("event"):
            g = g.sort_values("row")
            assert (g.row.to_numpy() == np.arange(len(g))).all(), e
            vol = g.volume.to_numpy().astype(np.int64)
            out[int(c) * 100_000_000 + int(e)] = {"volume": vol, "layer_id": vol * 1000 + g.layer.to_numpy(),
                                                  "phi_c": module_phi_c(g.geoid.to_numpy())}
    return out


# --------------------------------------------------------------------------------------------- calibration
def _tracks(sample: str, events):
    """Per event: (event, hits sorted by row, truth-fit particle table)."""
    import pandas as pd
    from hnl.data.convert import read_table
    from hnl.eval.particles import DERIVED
    tf = pd.read_parquet(os.path.join(DERIVED, "truth_fits", "v1", f"particles_{sample}.parquet"),
                         columns=["event", "particle_id", "pdg", "pt", "vsec", "n_hits", "free_d0", "label_w",
                                  "label_hnl"],
                         filters=[("event", "in", [int(e) for e in events])])
    h = read_table(os.path.join(DERIVED, "events", "v1"), "hits", sample, events=events,
                   columns=["event", "row", "x", "y", "z", "volume", "layer", "geoid", "pid"])
    for e, g in h.groupby("event"):
        yield e, g.sort_values("row"), tf[tf.event == e]


def _selected(t, pt_min, prompt_d0, min_hits, kind: str):
    # prompt = small truth-fit |d0| (the barcode's vsec field is > 0 for most ttbar particles, W muons included)
    if kind == "prompt":
        m = (t.pdg.abs() != 11) & (t.pt > pt_min) & (t.n_hits >= min_hits) & (t.free_d0.abs() < prompt_d0)
    elif kind == "muon":
        m = (t.pdg.abs() == 13) & (t.pt > pt_min) & (t.n_hits >= min_hits) & (t.free_d0.abs() < prompt_d0)
    elif kind == "w":
        m = t.label_w & (t.n_hits >= min_hits)
    elif kind == "hnl":
        m = t.label_hnl & (t.n_hits >= min_hits)
    else:
        raise ValueError(kind)
    return t[m]


def _robust_update(A, B, y, su: float, sv: float, n_min: int = 30, dominance: float = 0.8):
    """Fixed point of (s_u, s_v) for squared residuals y with expectation A s_u^2 + B s_v^2: each sigma is rescaled by
    the median squared pull of the rows its term dominates (> 80 % of the variance) over the chi2_1 median 0.4549, so
    outlier hits (a few % of the hits are mm off their particle's helix) do not enter."""
    for _ in range(30):
        Vu, Vv = A * su * su, B * sv * sv
        V = Vu + Vv + 1e-30
        q = y / V
        du, dv = Vu / V > dominance, Vv / V > dominance
        old = (su, sv)
        if du.sum() >= n_min:
            su *= float(np.sqrt(np.median(q[du]) / 0.45494))
        if dv.sum() >= n_min:
            sv *= float(np.sqrt(np.median(q[dv]) / 0.45494))
        if abs(su / old[0] - 1) < 1e-4 and abs(sv / old[1] - 1) < 1e-4:
            break
    return su, sv, int(du.sum()), int(dv.sum())


def overlap_rows(sample, events, res=None, kind="prompt", pt_min=2.0, prompt_d0=1.0, min_hits=8, max_dist=30.0):
    """Double hits of one particle on overlapping modules of one layer (different modules, < max_dist mm apart): their
    separation against the track's local direction (from the robust helix fit of all its hits) is free of multiple
    scattering and of the fit's own errors. Per pair two rows, each linear in (s_u^2, s_v^2) of the volume: the
    transverse offset across the track (d . n) and the offset along it (barrel: dz - cot (d . t); endcap: the disks
    fix z, so d . t - dz / cot)."""
    from hnl.trackbuild import helix_fit as HF
    out = []
    for e, g, t in _tracks(sample, events):
        sel = _selected(t, pt_min, prompt_d0, min_hits, kind)
        if not len(sel):
            continue
        g = g[g.pid.isin(sel.particle_id)]
        x, y, z = g.x.to_numpy(), g.y.to_numpy(), g.z.to_numpy()
        vol = g.volume.to_numpy().astype(np.int64)
        lay = vol * 1000 + g.layer.to_numpy()
        gid = g.geoid.to_numpy()
        phi_c = module_phi_c(gid)
        f = HF.fit(x, y, z, g.pid.to_numpy(), vol, phi_c, min_hits=min_hits, res=res)
        for k in range(len(f["track"])):
            idx = f["rows"][k][f["rows"][k] >= 0]
            i, j = np.triu_indices(len(idx), 1)
            i, j = idx[i], idx[j]
            d = np.stack([x[j] - x[i], y[j] - y[i], z[j] - z[i]], 1)
            ok = (lay[i] == lay[j]) & (gid[i] != gid[j]) & (np.linalg.norm(d, axis=1) < max_dist)
            if not ok.any():
                continue
            i, j, d = i[ok], j[ok], d[ok]
            mx, my = 0.5 * (x[i] + x[j]) - f["xc"][k], 0.5 * (y[i] + y[j]) - f["yc"][k]
            nn = np.stack([mx, my], 1) / np.maximum(np.hypot(mx, my), 1e-12)[:, None]
            tt = np.stack([-nn[:, 1], nn[:, 0]], 1)
            cot = f["cot"][k]
            ui, vi = axes(vol[i], phi_c[i])
            uj, vj = axes(vol[j], phi_c[j])
            dot = lambda a2, b3: a2[:, 0] * b3[:, 0] + a2[:, 1] * b3[:, 1]
            dn, dt = dot(nn, d), dot(tt, d)
            barrel = is_barrel(vol[i])
            r_t = dn
            A_t = dot(nn, ui) ** 2 + dot(nn, uj) ** 2
            B_t = dot(nn, vi) ** 2 + dot(nn, vj) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                r_l = np.where(barrel, d[:, 2] - cot * dt, dt - d[:, 2] / cot)
            A_l = np.where(barrel, cot * cot, 1.0) * (dot(tt, ui) ** 2 + dot(tt, uj) ** 2)
            B_l = np.where(barrel, 2.0, dot(tt, vi) ** 2 + dot(tt, vj) ** 2)
            out.append({"event": np.full(2 * len(i), int(e)), "volume": np.r_[vol[i], vol[i]],
                        "kind": np.r_[np.zeros(len(i), int), np.ones(len(i), int)], "y": np.r_[r_t, r_l] ** 2,
                        "A": np.r_[A_t, A_l], "B": np.r_[B_t, B_l], "pt": np.full(2 * len(i), np.nan)})
    import pandas as pd
    return pd.concat([pd.DataFrame(o) for o in out], ignore_index=True)


def wmuon_pixel_fit(events, res: dict, pt_min: float = 20.0, iterations: int = 5) -> tuple[dict, dict]:
    """Pixel (s_u, s_v) from the leverage-corrected residuals of weighted helix fits of ttbar W muons with pT > pt_min
    (multiple scattering negligible), iterated to self-consistency; strips keep their sigmas."""
    from hnl.trackbuild import helix_fit as HF
    import pandas as pd
    data = []
    for e, g, t in _tracks("ttbar", events):
        sel = t[t.label_w & (t.n_hits >= 8) & (t.pt > pt_min)]
        if len(sel):
            data.append(g[g.pid.isin(sel.particle_id)])
    res = dict(res)
    n_used = {}
    for _ in range(iterations):
        rows = []
        for g in data:
            vol = g.volume.to_numpy().astype(np.int64)
            f = HF.fit(g.x.to_numpy(), g.y.to_numpy(), g.z.to_numpy(), g.pid.to_numpy(), vol,
                       module_phi_c(g.geoid.to_numpy()), min_hits=8, res=res, leverage=True)
            rows.append(pd.DataFrame(HF.hit_table(f)))
        d = pd.concat(rows, ignore_index=True)
        for v in PIXEL:
            dv = d[d.volume == v]
            A = np.r_[dv.at_u.to_numpy(), dv.az_u.to_numpy()]
            B = np.r_[dv.at_v.to_numpy(), dv.az_v.to_numpy()]
            y = np.r_[dv.et2.to_numpy(), dv.ez2.to_numpy()]
            ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(y)
            su, sv, nu, nv = _robust_update(A[ok], B[ok], y[ok], *res[v], n_min=20)
            res[v] = (su, sv)
            n_used[v] = (nu, nv)
    return res, {"muons": int(sum(g.pid.nunique() for g in data)), "rows_u_v": n_used}


def measure(events, wmu_events, pt_min: float = 2.0, res: dict | None = None) -> dict:
    """(s_u, s_v) per volume: strips from overlap double hits of ttbar prompt tracks (no electrons); pixels from the
    weighted helix fits of high-pT ttbar W muons (the overlap pairs give 20-50 % larger pixel sigmas, which leave the
    W-muon pulls at 0.6-0.8)."""
    res = dict(RES if res is None else res)
    d = overlap_rows("ttbar", events, res, kind="prompt", pt_min=pt_min)
    new, n_used = {}, {}
    for v in VOLUMES:
        dv = d[(d.volume == v) & np.isfinite(d.y) & np.isfinite(d.A) & np.isfinite(d.B)]
        su, sv, nu, nv = _robust_update(dv.A.to_numpy(), dv.B.to_numpy(), dv.y.to_numpy(), *res[v], n_min=20)
        new[v] = (su, sv)
        n_used[v] = (nu, nv)
    overlap = {str(v): [round(new[v][0], 5), round(new[v][1], 5)] for v in VOLUMES}
    pix, info = wmuon_pixel_fit(wmu_events, new)
    return {"res": {str(v): [round(pix[v][0], 5), round(pix[v][1], 5)] for v in VOLUMES}, "overlap": overlap,
            "rows_u_v": {str(v): n_used[v] for v in VOLUMES}, "pairs": int((d.kind == 0).sum()), "wmuon_pixel": info}


def main(argv=None):
    from hnl.data.splits import read_split
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--events", type=int, default=800, help="ttbar train events for the overlap pairs")
    m.add_argument("--wmu-events", type=int, default=3000, help="ttbar train events for the W-muon pixel fits")
    m.add_argument("--pt-min", type=float, default=2.0)
    m.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    train = read_split("ttbar", "train")
    r = measure(train[:a.events], train[:a.wmu_events], a.pt_min)
    r.update({"events": a.events, "wmu_events": a.wmu_events, "sample": "ttbar train", "pt_min": a.pt_min,
              "elapsed_s": time.time() - t0})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    print(json.dumps(r["res"]))


if __name__ == "__main__":
    main()
