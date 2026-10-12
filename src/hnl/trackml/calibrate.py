"""Calibrations of Canto-L's uniform-field (2 T) pieces on TrackML (docs/trackml_plan.md sections 6.5 and 6.6), on the
calibration events of the throughput TRAINING set only (the 100 dev ids + the next 100 training ids: "calib" of
configs/trackml/split_tp_v1.json), before any val use. No val or test event is read (`calib_events` refuses them).

  field   the measured field map, with the feasibility study's estimator (work/content.py): consecutive true
          intersection points A, B, C of a particle (rank order), pT at B > 2 GeV (truth momentum at B), transverse
          |AB| and |BC| > 5 mm; B_eff = pT(B) |kappa_ABC| / 0.3 with kappa the curvature of the transverse circle
          through A, B, C in 1/m; the median per 50 mm x 100 mm bin of (r_B, |z_B|); an empty bin takes the value of
          the nearest filled bin (bin-centre distance). hnl.trackml.events.FieldMap reads it (bilinear interpolation
          between bin centres). -> configs/trackml/field_map_v1.json
  res     the module-frame resolution table: hit minus true intersection, projected on u = the tangent at the module
          centre azimuth phi_c and v = z (barrel volumes 8, 13, 17) or the radial direction at phi_c (endcaps), the
          frame of hnl.geometry.resolution (phi_c from the detectors.csv module centre); sigma = 1.4826 MAD per
          volume. Cross-check: the detectors.csv local axes. -> configs/trackml/res_v1.json
  slope   the candidate slope window of hnl.graphs.helix_union.candidate_triplets, |cot_in - cot_out| <=
          5 (sqrt(s_cot_out^2 + MS0^2 (1 + cot_out^2)) + 0.05 (1 + |cot_out|)) (the out-leg's window, cot clipped to
          +-200), on the true target triplets: three consecutive hits of a target particle whose two legs, oriented
          outward in R = hypot(r, z) as the support orients them, meet at the middle hit, neither leg same-layer or
          shorter than 1e-3 mm transversally (the support's usable legs). Hit positions as the pipeline builds them
          (x, y from the stored float32 r and phi). Fraction outside per |eta| bin (0.4 steps, particle eta) with the
          ttbar resolutions the code uses (hnl.geometry.resolution.RES) and with res_v1; where it exceeds 0.5 %, the
          constant term 0.05 is widened to the smallest value that leaves <= 0.5 % outside.
          -> configs/trackml/slope_window_v1.tsv
  merge   the union fit of the hand-cut fragment merging (hnl.trackbuild.merge_fragments._merge: hnl.vertex.pairing.
          fit_tracks, min_hits 3, the subsystem from hnl.trackbuild.merge_fragments._volume_from_radius as in
          hnl.trackml.trackbuild; accepted if rms <= 1.0 mm, zrms <= 5 mm, n_out <= 1 and R span >= 50 mm) on true
          target tracks split into two fragments at every gap. The union of the two fragments is the whole track, so
          each track is fitted once and counts once per split with >= 2 hits on each side (min_frag = 2). Fail
          fraction per |eta| bin; where it exceeds 1 %, max_rms / max_zrms become the bin's 99th percentiles (the
          plan's rule; also shown: a widen-only variant). Diagnostic beyond the plan: the max_out that 99 % of the
          unions satisfy, the acceptance with it, and the fail fraction by pT. -> configs/trackml/merge_cuts_v1.tsv

    python -m hnl.trackml.calibrate [--root DIR] [--split FILE] [--outdir configs/trackml] [--note FILE.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

from hnl.graphs import helix_union as HU
from hnl.trackml import convert as C
from hnl.trackml import throughput as T
from hnl.trackml.split import REPO

OUTDIR = os.path.join(REPO, "configs", "trackml")
CHUNK = 20
ETA_EDGES = np.round(np.arange(0.0, 4.0001, 0.4), 1)
# field estimator (feasibility study, plan section 6.5)
FIELD_PT_MIN = 2.0
FIELD_MIN_LEG_MM = 5.0
R_EDGES = np.arange(0.0, 1100.0 + 1, 50.0)
ABSZ_EDGES = np.arange(0.0, 3100.0 + 1, 100.0)
# slope window (hnl.graphs.helix_union.candidate_triplets defaults)
NSIG = 5.0
C0 = 0.05
SLOPE_MAX_OUT = 0.005
# merge cuts (hnl.trackbuild.merge_fragments defaults)
MAX_RMS, MAX_ZRMS, MAX_OUT, MIN_SPAN, MIN_FRAG = 1.0, 5.0, 1, 50.0, 2
MERGE_MAX_FAIL = 0.01
# feasibility-study field table (accuracy phase, docs/trackml_feasibility.md section 1): |z| band -> Bz [T]
FEAS_FIELD = {"0-250": "2.00", "250-500": "1.98", "500-750": "1.94", "750-1000": "1.88", "1000-1250": "1.78",
              "1250-1500": "1.6", "1500-2000": "1.3", "2000-2500": "0.7-0.8", "2500-3000": "0.24-0.50"}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def calib_events(split: dict) -> list[int]:
    ev = sorted(split["calib"])
    if set(ev) & (split["val"] | split["test"]) or not set(ev) <= split["train"]:
        raise SystemExit("calibration events must be training events, never val or test")
    return ev


def iter_calib(events, root: str):
    """(event, hits, particles, truth) per calibration event, read CHUNK events at a time."""
    hcols = ["event", "row", "x", "y", "z", "r", "phi", "volume", "layer", "module", "pid", "rank"]
    pcols = ["event", "particle_id", "q", "pt", "eta", "n_hits", "is_target", "category", "rv"]
    for k in range(0, len(events), CHUNK):
        part = events[k:k + CHUNK]
        h = dict(tuple(C.read_table("hits", part, hcols, root=root).groupby("event")))
        p = dict(tuple(C.read_table("particles", part, pcols, root=root).groupby("event")))
        t = dict(tuple(C.read_table("truth", part, root=root).groupby("event")))
        for e in part:
            yield e, h[e].reset_index(drop=True), p[e].reset_index(drop=True), t[e].reset_index(drop=True)


def eta_bin(eta) -> np.ndarray:
    return np.clip(np.digitize(np.abs(np.asarray(eta, dtype=np.float64)), ETA_EDGES) - 1, 0, len(ETA_EDGES) - 2)


def eta_label(i: int) -> str:
    return f"{ETA_EDGES[i]:.1f}-{ETA_EDGES[i + 1]:.1f}"


# ------------------------------------------------------------------------------------------------ field map
def field_triplets(h: pd.DataFrame, p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Truth triplets of one event (module docstring, `field`): r_B, |z_B|, B_eff and the rotation sense."""
    row = t.row.to_numpy()
    pid = h.pid.to_numpy()[row]
    rank = h["rank"].to_numpy()[row]
    o = np.lexsort((rank, pid))
    pid, rank = pid[o], rank[o]
    tx, ty, tz, tpx, tpy = (t[c].to_numpy().astype(np.float64)[o] for c in ("tx", "ty", "tz", "tpx", "tpy"))
    tri = np.flatnonzero((pid[2:] == pid[1:-1]) & (pid[1:-1] == pid[:-2]))
    A = np.stack([tx[tri], ty[tri]], 1)
    B = np.stack([tx[tri + 1], ty[tri + 1]], 1)
    Cc = np.stack([tx[tri + 2], ty[tri + 2]], 1)
    ab, bc, ca = B - A, Cc - B, A - Cc
    nab, nbc, nca = (np.linalg.norm(v, axis=1) for v in (ab, bc, ca))
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = 2 * cross / (nab * nbc * nca)
    ptb = np.hypot(tpx[tri + 1], tpy[tri + 1])
    ok = (ptb > FIELD_PT_MIN) & (nab > FIELD_MIN_LEG_MM) & (nbc > FIELD_MIN_LEG_MM) & np.isfinite(kappa)
    q = p.set_index("particle_id").q.reindex(pid[tri + 1]).to_numpy()
    return pd.DataFrame({"r": np.hypot(B[:, 0], B[:, 1])[ok], "absz": np.abs(tz[tri + 1])[ok],
                         "B": (ptb * np.abs(kappa) * 1000.0 / 0.3)[ok],
                         "sense": (np.sign(kappa) * np.sign(q))[ok].astype(np.int8)})


def field_map(trip: pd.DataFrame) -> dict:
    """Median per (r, |z|) bin, empty bins from the nearest filled bin centre."""
    nr, nz = len(R_EDGES) - 1, len(ABSZ_EDGES) - 1
    ir = np.digitize(trip.r.to_numpy(), R_EDGES) - 1
    iz = np.digitize(trip.absz.to_numpy(), ABSZ_EDGES) - 1
    inside = (ir >= 0) & (ir < nr) & (iz >= 0) & (iz < nz)
    med = np.full((nr, nz), np.nan)
    cnt = np.zeros((nr, nz), dtype=np.int64)
    g = pd.DataFrame({"k": ir[inside] * nz + iz[inside], "B": trip.B.to_numpy()[inside]}).groupby("k").B
    m, n = g.median(), g.size()
    med.flat[m.index.to_numpy()] = m.to_numpy()
    cnt.flat[n.index.to_numpy()] = n.to_numpy()
    rc = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    zc = 0.5 * (ABSZ_EDGES[1:] + ABSZ_EDGES[:-1])
    RR, ZZ = np.meshgrid(rc, zc, indexing="ij")
    filled = ~np.isnan(med)
    out = med.copy()
    fr, fz, fv = RR[filled], ZZ[filled], med[filled]
    for i, j in zip(*np.nonzero(~filled)):
        k = int(np.argmin((fr - RR[i, j]) ** 2 + (fz - ZZ[i, j]) ** 2))
        out[i, j] = fv[k]
    return {"B": out, "median": med, "counts": cnt, "filled_from_neighbour": ~filled,
            "outside_grid": int((~inside).sum())}


def field_band_table(trip: pd.DataFrame) -> pd.DataFrame:
    """Median B_eff per |z| band (the feasibility study's bands), all r, and its 16-84 % spread."""
    rows = []
    for band in FEAS_FIELD:
        lo, hi = (float(x) for x in band.split("-"))
        s = trip.B[(trip.absz >= lo) & (trip.absz < hi)]
        rows.append({"absz_mm": band, "n": int(len(s)), "B_median_T": round(float(s.median()), 3) if len(s) else None,
                     "B_p16_T": round(float(s.quantile(0.16)), 3) if len(s) else None,
                     "B_p84_T": round(float(s.quantile(0.84)), 3) if len(s) else None,
                     "feasibility_accuracy_phase_T": FEAS_FIELD[band]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ RES
def module_lookup(det: dict, volume, layer, module) -> np.ndarray:
    key = C.module_key(volume, layer, module)
    pos = np.minimum(np.searchsorted(det["key"], key), len(det["key"]) - 1)
    if not np.array_equal(det["key"][pos], key):
        raise RuntimeError("hit on a module missing from detectors.csv")
    return pos


def residuals(h: pd.DataFrame, t: pd.DataFrame, det: dict) -> pd.DataFrame:
    """Hit minus true intersection of every non-noise hit, in the phi_c frame and in the detectors.csv local frame."""
    row = t.row.to_numpy()
    hh = h.iloc[row]
    vol = hh.volume.to_numpy().astype(np.int64)
    d = np.stack([hh[c].to_numpy().astype(np.float64) - t[tc].to_numpy().astype(np.float64)
                  for c, tc in (("x", "tx"), ("y", "ty"), ("z", "tz"))], 1)
    pos = module_lookup(det, vol, hh.layer.to_numpy(), hh.module.to_numpy())
    cx, cy = det["c"][pos, 0], det["c"][pos, 1]
    phic = np.arctan2(cy, cx)
    cs, sn = np.cos(phic), np.sin(phic)
    barrel = np.isin(vol, (8, 13, 17))
    ru = -sn * d[:, 0] + cs * d[:, 1]
    rv = np.where(barrel, d[:, 2], cs * d[:, 0] + sn * d[:, 1])
    loc = np.einsum("nji,nj->ni", det["R"][pos], d)
    return pd.DataFrame({"volume": vol.astype(np.int8), "u": ru.astype(np.float32), "v": rv.astype(np.float32),
                         "lu": loc[:, 0].astype(np.float32), "lv": loc[:, 1].astype(np.float32),
                         "lw": loc[:, 2].astype(np.float32)})


def mad_sigma(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def res_table(res: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    from hnl.geometry.resolution import RES as RES_TTBAR
    rows, table = [], {}
    for v, g in res.groupby("volume"):
        mu, su = mad_sigma(g.u.to_numpy())
        mv, sv = mad_sigma(g.v.to_numpy())
        _, slu = mad_sigma(g.lu.to_numpy())
        _, slv = mad_sigma(g.lv.to_numpy())
        table[int(v)] = (round(su, 5), round(sv, 5))
        rows.append({"volume": int(v), "n_hits": int(len(g)), "s_u_mm": round(su, 5), "s_v_mm": round(sv, 5),
                     "median_u_mm": round(mu, 5), "median_v_mm": round(mv, 5),
                     "std_u_mm": round(float(g.u.std()), 5), "std_v_mm": round(float(g.v.std()), 5),
                     "local_s_u_mm": round(slu, 5), "local_s_v_mm": round(slv, 5),
                     "local_w_absmax_mm": round(float(g.lw.abs().max()), 4),
                     "ttbar_s_u_mm": RES_TTBAR[int(v)][0], "ttbar_s_v_mm": RES_TTBAR[int(v)][1]})
    return table, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ slope window
def hit_sigmas_res(vol: np.ndarray, res: dict):
    """hnl.graphs.helix_union.hit_sigmas with a resolution table {volume: (fine, coarse)}."""
    vol = np.asarray(vol, dtype=np.int64)
    fine = np.zeros(vol.max(initial=0) + 1)
    coarse = np.zeros_like(fine)
    for v, (f, c) in res.items():
        if int(v) < len(fine):
            fine[int(v)], coarse[int(v)] = f, c
    if not np.isin(vol, [int(v) for v in res]).all():
        raise ValueError("unknown volumes")
    barrel = np.isin(vol, HU.BARREL)
    return fine[vol], np.where(barrel, coarse[vol], 0.0), np.where(barrel, 0.0, coarse[vol])


def legs_res(xyz: np.ndarray, src: np.ndarray, dst: np.ndarray, vol: np.ndarray, res: dict) -> dict:
    """hnl.graphs.helix_union.legs with a resolution table (the same expressions)."""
    fine, sz, sr = hit_sigmas_res(vol, res)
    d = xyz[dst] - xyz[src]
    L = np.hypot(d[:, 0], d[:, 1])
    Lc = np.maximum(L, 1e-3)
    cot = d[:, 2] / Lc
    s_cot = np.sqrt(sz[src] ** 2 + sz[dst] ** 2 + cot ** 2 * (sr[src] ** 2 + sr[dst] ** 2)) / Lc
    return {"L": L, "cot": cot, "s_cot": s_cot}


def true_triplets(h: pd.DataFrame, p: pd.DataFrame) -> dict:
    """True target triplets of one event as (in-leg, out-leg) hit pairs, oriented as the support orients legs."""
    tg = p.loc[p.is_target.to_numpy(), ["particle_id", "eta"]]
    pid = h.pid.to_numpy()
    sel = np.flatnonzero(np.isin(pid, tg.particle_id.to_numpy()))
    o = sel[np.lexsort((h["rank"].to_numpy()[sel], pid[sel]))]
    po = pid[o]
    i3 = np.flatnonzero((po[2:] == po[1:-1]) & (po[1:-1] == po[:-2]))
    h0, h1, h2 = o[i3], o[i3 + 1], o[i3 + 2]
    r = h.r.to_numpy().astype(np.float64)
    phi = h.phi.to_numpy().astype(np.float64)
    z = h.z.to_numpy().astype(np.float64)
    R = np.hypot(r, z)
    fwd = lambda a, b: (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    f1, f2 = fwd(h0, h1), fwd(h1, h2)
    # legs (src -> dst): L1 = h0 -> h1 if f1 else h1 -> h0; L2 = h1 -> h2 if f2 else h2 -> h1
    a_in = np.where(f1 & f2, h0, h2)          # f1 & f2: in = h0 -> h1, out = h1 -> h2
    a_out = np.where(f1 & f2, h2, h0)         # ~f1 & ~f2: in = h2 -> h1, out = h1 -> h0
    mono = f1 == f2                            # else the middle hit is an R extremum: no triplet
    lid = h.volume.to_numpy().astype(np.int64) * 1000 + h.layer.to_numpy().astype(np.int64)
    same = (lid[a_in] == lid[h1]) | (lid[h1] == lid[a_out])
    eta = tg.set_index("particle_id").eta.reindex(pid[h1]).to_numpy()
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    return {"a": a_in, "b": h1, "c": a_out, "mono": mono, "same_layer": same, "eta": eta, "xyz": xyz,
            "vol": h.volume.to_numpy().astype(np.int64)}


def slope_rows(tr: dict, res: dict) -> pd.DataFrame:
    """Per true target triplet with usable legs: |eta| bin, |dcot| against the window and the constant term needed."""
    keep = tr["mono"] & ~tr["same_layer"]
    a, b, c = tr["a"][keep], tr["b"][keep], tr["c"][keep]
    src = np.concatenate([a, b])
    dst = np.concatenate([b, c])
    lg = legs_res(tr["xyz"], src, dst, tr["vol"], res)
    n = len(a)
    use = lg["L"] > 1e-3
    ok = use[:n] & use[n:]
    cot_in, cot_out = lg["cot"][:n], lg["cot"][n:]
    s_out = lg["s_cot"][n:]
    dc = np.abs(np.clip(cot_in, -HU.COT_CLIP, HU.COT_CLIP) - np.clip(cot_out, -HU.COT_CLIP, HU.COT_CLIP))
    S = np.sqrt(s_out ** 2 + HU.MS0 ** 2 * (1 + cot_out ** 2))
    w = np.minimum(NSIG * (S + C0 * (1 + np.abs(cot_out))), 2 * HU.COT_CLIP + 1.0)
    need = (dc / NSIG - S) / (1 + np.abs(cot_out))
    return pd.DataFrame({"eta_bin": eta_bin(tr["eta"][keep])[ok], "outside": (dc > w)[ok], "c_need": need[ok]})


def needed_constant(c_need: np.ndarray, max_out: float = SLOPE_MAX_OUT) -> float:
    """Smallest constant term c with at most a fraction max_out of the triplets outside (c_need > c); it exceeds C0
    exactly when more than max_out fall outside the code's window."""
    s = np.sort(np.asarray(c_need, dtype=np.float64))
    k = int(np.floor(max_out * len(s)))
    return float(s[len(s) - k - 1]) if len(s) else float("nan")


def slope_table(rows_ttbar: pd.DataFrame, rows_tml: pd.DataFrame, n_excl: dict) -> pd.DataFrame:
    out = []
    up = lambda c: float(np.ceil(c * 1e4) / 1e4)
    for i in range(len(ETA_EDGES) - 1):
        r0 = rows_ttbar[rows_ttbar.eta_bin == i]
        r1 = rows_tml[rows_tml.eta_bin == i]
        f0 = float(r0.outside.mean()) if len(r0) else np.nan
        f1 = float(r1.outside.mean()) if len(r1) else np.nan
        n0, n1 = needed_constant(r0.c_need.to_numpy()), needed_constant(r1.c_need.to_numpy())
        out.append({"abs_eta": eta_label(i), "n_triplets": int(len(r0)),
                    "n_excluded_same_layer": int(n_excl["same_layer"].get(i, 0)),
                    "n_excluded_R_extremum": int(n_excl["extremum"].get(i, 0)),
                    "frac_outside_ttbar_res": round(f0, 5), "frac_outside_trackml_res": round(f1, 5),
                    "c_995_ttbar_res": round(n0, 5), "c_995_trackml_res": round(n1, 5),
                    "c0_ttbar_res": up(max(C0, n0)) if f0 > SLOPE_MAX_OUT else C0,
                    "c0_trackml_res": up(max(C0, n1)) if f1 > SLOPE_MAX_OUT else C0})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ merge cuts
def merge_rows(h: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Union fit of every true target track of one event (module docstring, `merge`)."""
    from hnl.trackbuild.merge_fragments import _volume_from_radius
    from hnl.vertex.pairing import fit_tracks
    tg = p.loc[p.is_target.to_numpy(), ["particle_id", "eta", "pt"]].reset_index(drop=True)
    pid = h.pid.to_numpy()
    lab = pd.Series(np.arange(len(tg)), index=tg.particle_id.to_numpy()).reindex(pid).fillna(-1).to_numpy()
    lab = lab.astype(np.int64)
    sel = np.flatnonzero(lab >= 0)
    r = h.r.to_numpy().astype(np.float64)[sel]
    phi = h.phi.to_numpy().astype(np.float64)[sel]
    z = h.z.to_numpy().astype(np.float64)[sel]
    lb = lab[sel]
    fit = fit_tracks(r, phi, z, _volume_from_radius(r, z), lb, min_hits=3)
    R3 = np.hypot(r, z)
    n = np.bincount(lb, minlength=len(tg))
    lo = np.full(len(tg), np.inf)
    hi = np.full(len(tg), -np.inf)
    np.minimum.at(lo, lb, R3)
    np.maximum.at(hi, lb, R3)
    k = fit["track"]
    return pd.DataFrame({"eta_bin": eta_bin(tg.eta.to_numpy()[k]), "pt": tg.pt.to_numpy()[k], "n_hits": n[k],
                         "rms": fit["rms"], "zrms": fit["zrms"], "n_out": fit["n_out"], "span": (hi - lo)[k],
                         "n_unions": np.maximum(n[k] - 2 * MIN_FRAG + 1, 0)})


def max_out_needed(n_out: np.ndarray, w: np.ndarray, max_fail: float = MERGE_MAX_FAIL) -> int:
    """Smallest integer m with at most max_fail of the (weighted) unions at n_out > m (a diagnostic, not a plan
    calibration: the plan calibrates max_rms / max_zrms only)."""
    n_out, w = np.asarray(n_out), np.asarray(w, dtype=np.float64)
    W = w.sum()
    for m in range(int(n_out.max(initial=0)) + 1):
        if w[n_out > m].sum() <= max_fail * W:
            return m
    return int(n_out.max(initial=0))


def merge_by_pt(m: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: union-weighted fail fraction and its n_out part by pT and |eta| region."""
    pt_edges = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
    regions = {"0.0-1.2": (0, 3), "1.2-2.8": (3, 7), "2.8-4.0": (7, 10)}
    out = []
    for name, (b0, b1) in regions.items():
        g0 = m[(m.eta_bin >= b0) & (m.eta_bin < b1)]
        for lo, hi in zip(pt_edges[:-1], pt_edges[1:]):
            g = g0[(g0.pt > lo) & (g0.pt <= hi)]
            w = g.n_unions.to_numpy()
            W = max(int(w.sum()), 1)
            good = ((g.rms <= MAX_RMS) & (g.zrms <= MAX_ZRMS) & (g.n_out <= MAX_OUT) & (g.span >= MIN_SPAN)).to_numpy()
            out.append({"abs_eta": name, "pt_gev": f"{lo:g}-{hi:g}", "n_tracks": int(len(g)),
                        "frac_fail": round(float((w * ~good).sum() / W), 4),
                        "fail_n_out": round(float((w * (g.n_out > MAX_OUT).to_numpy()).sum() / W), 4),
                        "median_n_out": float(np.median(g.n_out)) if len(g) else np.nan})
    return pd.DataFrame(out)


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Quantile of x with integer weights w (each value repeated w times; linear interpolation as numpy)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.int64)
    return float(np.percentile(np.repeat(x, w), q)) if w.sum() else float("nan")


def merge_table(m: pd.DataFrame) -> pd.DataFrame:
    out = []
    for i in range(len(ETA_EDGES) - 1):
        g = m[m.eta_bin == i]
        w = g.n_unions.to_numpy()
        W = max(int(w.sum()), 1)
        # the acceptance of hnl.trackbuild.merge_fragments._merge (a NaN fit value fails, as there)
        ok_rms, ok_z = (g.rms <= MAX_RMS).to_numpy(), (g.zrms <= MAX_ZRMS).to_numpy()
        ok_out, ok_span = (g.n_out <= MAX_OUT).to_numpy(), (g.span >= MIN_SPAN).to_numpy()
        f_rms, f_z, f_out, f_span = ~ok_rms, ~ok_z, ~ok_out, ~ok_span
        fail = ~(ok_rms & ok_z & ok_out & ok_span)
        frac = float((w * fail).sum() / W)
        p99r, p99z = wquantile(g.rms, w, 99), wquantile(g.zrms, w, 99)
        if frac > MERGE_MAX_FAIL:
            lit = (p99r, p99z)
            wid = (max(MAX_RMS, p99r), max(MAX_ZRMS, p99z))
        else:
            lit = wid = (MAX_RMS, MAX_ZRMS)
        acc = lambda cr, cz: float((w * ((g.rms <= cr).to_numpy() & (g.zrms <= cz).to_numpy() & ok_out
                                         & ok_span)).sum() / W)
        mo = max(MAX_OUT, max_out_needed(g.n_out.to_numpy(), w)) if len(g) else MAX_OUT
        acc_diag = float((w * ((g.rms <= wid[0]).to_numpy() & (g.zrms <= wid[1]).to_numpy()
                               & (g.n_out <= mo).to_numpy() & ok_span)).sum() / W)
        out.append({"abs_eta": eta_label(i), "n_tracks": int(len(g)), "n_unions": int(w.sum()),
                    "frac_fail": round(frac, 5), "frac_fail_tracks": round(float(fail.mean()), 5) if len(g) else None,
                    "fail_rms": round(float((w * f_rms).sum() / W), 5), "fail_zrms": round(float((w * f_z).sum() / W), 5),
                    "fail_n_out": round(float((w * f_out).sum() / W), 5),
                    "fail_span": round(float((w * f_span).sum() / W), 5),
                    "p99_rms_mm": round(p99r, 4), "p99_zrms_mm": round(p99z, 4),
                    "max_rms_plan": round(lit[0], 4), "max_zrms_plan": round(lit[1], 4),
                    "accept_plan": round(acc(*lit), 5),
                    "max_rms_widen_only": round(wid[0], 4), "max_zrms_widen_only": round(wid[1], 4),
                    "accept_widen_only": round(acc(*wid), 5),
                    "p99_n_out": wquantile(g.n_out, w, 99) if len(g) else np.nan,
                    "diag_max_out_99": int(mo), "diag_accept_widen_and_max_out": round(acc_diag, 5)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ main
def run(root: str, split_path: str, outdir: str, note: str | None) -> dict:
    split = T.load_tp_split(split_path)
    events = calib_events(split)
    det = C.load_detectors()
    trips, res_parts, slope_t, slope_x, merges = [], [], [], [], []
    n_excl = {"same_layer": {}, "extremum": {}}
    from hnl.geometry.resolution import RES as RES_TTBAR
    tr_all = []
    for k, (e, h, p, t) in enumerate(iter_calib(events, root)):
        trips.append(field_triplets(h, p, t).assign(event=e))
        res_parts.append(residuals(h, t, det))
        tr = true_triplets(h, p)
        tr_all.append(tr)
        bins = eta_bin(tr["eta"])
        for key, mask in (("same_layer", tr["mono"] & tr["same_layer"]), ("extremum", ~tr["mono"])):
            for i, c in zip(*np.unique(bins[mask], return_counts=True)):
                n_excl[key][int(i)] = n_excl[key].get(int(i), 0) + int(c)
        slope_t.append(slope_rows(tr, RES_TTBAR))
        merges.append(merge_rows(h, p))
        if (k + 1) % 20 == 0:
            print(f"[calibrate] {k + 1}/{len(events)} events", flush=True)
    trip = pd.concat(trips, ignore_index=True)
    res = pd.concat(res_parts, ignore_index=True)
    res_tml, res_tab = res_table(res)
    slope_x = [slope_rows(tr, res_tml) for tr in tr_all]
    st_ttbar = pd.concat(slope_t, ignore_index=True)
    st_tml = pd.concat(slope_x, ignore_index=True)
    mg = pd.concat(merges, ignore_index=True)
    fm = field_map(trip)
    ev_even = events[0::2]
    half = [field_map(trip[trip.event.isin(s)]) for s in (ev_even, events[1::2])]
    both = (half[0]["counts"] >= 50) & (half[1]["counts"] >= 50)
    dB = np.abs(half[0]["median"] - half[1]["median"])[both]
    prov = {"events": "calib of " + os.path.relpath(split_path, REPO) + f" ({len(events)} throughput training events: "
                      "the 100 dev ids + the next 100 training ids)",
            "event_ids_sha256": hashlib.sha256(",".join(str(e) for e in events).encode()).hexdigest(),
            "first_last_event": [events[0], events[-1]], "split_sha256": split["sha256"],
            "tables": root, "code_sha256": C.sha256_file(os.path.abspath(__file__)),
            "made_utc": utc(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "plan": "docs/trackml_plan.md sections 6.5, 6.6"}
    flat = lambda s: " ".join(s.split())
    fmap = {"version": "v1", "units": {"r": "mm", "absz": "mm", "B": "T"},
            "estimator": flat(__doc__.split("  field   ")[1].split("  res     ")[0].split(" -> ")[0]),
            "r_edges": R_EDGES.tolist(), "absz_edges": ABSZ_EDGES.tolist(),
            "B": np.round(fm["B"], 5).tolist(), "counts": fm["counts"].tolist(),
            "filled_from_neighbour": fm["filled_from_neighbour"].astype(int).tolist(),
            "n_triplets": int(len(trip)), "triplets_outside_grid": fm["outside_grid"],
            "sense_clockwise_positive_fraction": round(float((trip.sense < 0).mean()), 4),
            "split_half_stability": {"bins_with_50_per_half": int(both.sum()),
                                     "max_abs_dB_T": round(float(dB.max()), 4) if both.any() else None,
                                     "p95_abs_dB_T": round(float(np.percentile(dB, 95)), 4) if both.any() else None},
            "provenance": prov}
    os.makedirs(outdir, exist_ok=True)
    paths = {"field": os.path.join(outdir, "field_map_v1.json"), "res": os.path.join(outdir, "res_v1.json"),
             "slope": os.path.join(outdir, "slope_window_v1.tsv"), "merge": os.path.join(outdir, "merge_cuts_v1.tsv")}
    with open(paths["field"], "w") as fh:
        json.dump(fmap, fh, indent=1)
    with open(paths["res"], "w") as fh:
        json.dump({"version": "v1", "units": "mm", "RES": {str(k): list(v) for k, v in sorted(res_tml.items())},
                   "frame": "u = tangent at the module-centre azimuth phi_c, v = z (barrel 8, 13, 17) or radial at "
                            "phi_c (endcaps), as hnl.geometry.resolution; phi_c from detectors.csv",
                   "estimator": "hit minus true intersection (truth tx, ty, tz) of every non-noise hit; sigma = "
                                "1.4826 x median absolute deviation from the median, per volume",
                   "table": json.loads(res_tab.to_json(orient="records")), "provenance": prov}, fh, indent=1)
    sl = slope_table(st_ttbar, st_tml, n_excl)
    sl.to_csv(paths["slope"], sep="\t", index=False)
    mt = merge_table(mg)
    mt.to_csv(paths["merge"], sep="\t", index=False)
    mpt = merge_by_pt(mg)
    bands = field_band_table(trip)
    summary = {"paths": paths, "n_events": len(events), "field_bands": bands.to_dict(orient="records"),
               "field": {k: fmap[k] for k in ("n_triplets", "triplets_outside_grid",
                                              "sense_clockwise_positive_fraction", "split_half_stability")},
               "res": res_tab.to_dict(orient="records"), "slope": sl.to_dict(orient="records"),
               "merge": mt.to_dict(orient="records"), "merge_by_pt": mpt.to_dict(orient="records"),
               "merge_totals": {"tracks": int(len(mg)), "unions": int(mg.n_unions.sum())},
               "slope_totals": {"triplets_ttbar_res": int(len(st_ttbar)), "triplets_trackml_res": int(len(st_tml))}}
    if note:
        write_note(note, summary, prov, fmap, res_tab, sl, mt, mpt, bands)
    return summary


def _cell(x) -> str:
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        if x == 0:
            return "0"
        return np.format_float_positional(float(x), precision=5, unique=True, trim="-")
    return str(x)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(x) for x in r) + " |")
    return "\n".join(lines)


def write_note(path, summary, prov, fmap, res_tab, sl, mt, mpt, bands):
    sh = fmap["split_half_stability"]
    rt = res_tab[["volume", "n_hits", "s_u_mm", "s_v_mm", "ttbar_s_u_mm", "ttbar_s_v_mm", "median_u_mm",
                  "median_v_mm", "std_u_mm", "std_v_mm"]]
    sl_n = sl[["abs_eta", "n_triplets", "n_excluded_same_layer", "n_excluded_R_extremum", "frac_outside_ttbar_res",
               "frac_outside_trackml_res", "c_995_ttbar_res", "c0_ttbar_res", "c0_trackml_res"]]
    mt_n = mt[["abs_eta", "n_tracks", "n_unions", "frac_fail", "fail_rms", "fail_zrms", "fail_n_out", "fail_span",
               "p99_rms_mm", "p99_zrms_mm", "accept_plan", "accept_widen_only", "p99_n_out", "diag_max_out_99",
               "diag_accept_widen_and_max_out"]]
    widened = sl[(sl.c0_ttbar_res > C0) | (sl.c0_trackml_res > C0)]
    over = mt[mt.frac_fail > MERGE_MAX_FAIL]
    ent = [(int(r.volume), ax, getattr(r, f"s_{ax}_mm"), getattr(r, f"ttbar_s_{ax}_mm"))
           for r in res_tab.itertuples() for ax in ("u", "v")]
    res_below = sum(1 for e in ent if e[2] < e[3])
    res_above = ", ".join(f"{v} {ax}" for v, ax, a, b in ent if a >= b) or "none"
    pix = res_tab[res_tab.volume == 8].iloc[0]
    pix_u, pix_u_tt = pix.s_u_mm, pix.ttbar_s_u_mm
    offs = 1000 * res_tab[res_tab.volume.isin([8, 13, 17])].median_u_mm.abs()
    off_lo, off_hi = float(offs.min()), float(offs.max())
    share = float((over.fail_n_out / over.frac_fail).min()) if len(over) else float("nan")
    dloc = 1000 * float(max((res_tab.s_u_mm - res_tab.local_s_u_mm).abs().max(),
                            (res_tab.s_v_mm - res_tab.local_s_v_mm).abs().max()))
    text = f"""# TrackML calibrations v1 (plan sections 6.5 and 6.6)

- Made by `python -m hnl.trackml.calibrate` (`src/hnl/trackml/calibrate.py`; its docstring defines every
  estimator), Slurm job {prov['slurm_job']}, {prov['made_utc']}.
- Events: the 200 calibration events of `configs/trackml/split_tp_v1.json` (the 100 throughput dev ids + the next 100
  training ids), ids {prov['first_last_event'][0]} ... {prov['first_last_event'][1]}, sha256 of the comma-joined ids
  `{prov['event_ids_sha256'][:12]}`. Tables: `derived/tp_v1`. Only these training events were read; no val or test
  event.

Summary:
- **Field map.** The median B_eff per |z| band agrees with the accuracy-phase feasibility values (table below):
  {bands.B_median_T.iloc[0]:.2f} T at |z| < 250 mm, falling to {bands.B_median_T.iloc[-1]:.2f} T at 2.5-3 m.
- **RES.** The TrackML sigma is below the ttbar table's in {res_below} of 18 entries; above in {res_above}. Pixel
  barrel u: {pix_u} against {pix_u_tt} mm. The barrel volumes show u offsets (medians) of {off_lo:.0f}-{off_hi:.0f} um.
- **Slope window.** At most {max(sl.frac_outside_ttbar_res.max(), sl.frac_outside_trackml_res.max()):.3%} of the true
  target triplets fall outside it in any |eta| bin (limit 0.5 %). No bin is widened.
- **Merge cuts.** The union fit rejects {mt.frac_fail.min():.1%}-{mt.frac_fail.max():.1%} of true unions per |eta| bin
  (more than 1 % in {len(over)} of 10 bins). In every bin above 1 %, the outlier cut (n_out <= 1) accounts for at least
  {share:.0%} of the rejections. They fall steeply with pT (multiple scattering; table at the end), and in
  1.2 < |eta| < 2.4 the rms cut adds to them (the field). The plan's rule changes only max_rms / max_zrms, so it does
  not restore the acceptance (accept_plan). This needs a decision.

## Field map (`field_map_v1.json`)

- {fmap['n_triplets']:,} truth triplets (pT at the middle point > 2 GeV, transverse legs > 5 mm).
- 50 mm x 100 mm bins in (r, |z|), r 0-1100 mm, |z| 0-3100 mm; median per bin; empty bins take the nearest filled
  bin. `hnl.trackml.events.FieldMap` reads the file.
- Positive charges turn clockwise in {fmap['sense_clockwise_positive_fraction']:.1%} of the triplets (Bz > 0, as the
  converter assumes).
- Split-half stability (even against odd calibration events, {sh['bins_with_50_per_half']} bins with >= 50 triplets in
  each half): max |dB| {sh['max_abs_dB_T']} T, 95th percentile {sh['p95_abs_dB_T']} T.

Median B_eff per |z| band (all r), against the accuracy-phase feasibility values:

{_md(bands)}

## Resolution table RES (`res_v1.json`)

Hit minus true intersection in the module frame of `hnl.geometry.resolution` (u: the tangent at the module-centre
azimuth; v: z in the barrel, radial in the endcaps). sigma = 1.4826 MAD per volume [mm]. The ttbar values
(`hnl.geometry.resolution.RES`, what the code uses now) are shown for comparison, with the median (bias) and the
standard deviation. The detectors.csv local axes give the same sigmas within {dloc:.2f} um (`local_*` in
`res_v1.json`).

{_md(rt)}

## Candidate slope window (`slope_window_v1.tsv`)

- True target triplets with usable legs: {summary['slope_totals']['triplets_ttbar_res']:,}. The table gives the
  fraction outside the window of `hnl.graphs.helix_union.candidate_triplets` per |eta| bin, with the ttbar resolutions
  the code uses and with RES v1.
- c_995: the constant term that would leave exactly 0.5 % outside. It is negative where the window holds 99.5 % even
  without the constant term.
- c0: the constant term to use (0.05 in the code), widened only where more than 0.5 % fall outside.
- `n_excluded_*`: true triplets the support never forms (a same-layer leg; the middle hit at an extremum of R).

{_md(sl_n)}

Bins widened: {', '.join(widened.abs_eta) if len(widened) else 'none'}.

## Fragment-merging union fit (`merge_cuts_v1.tsv`)

- True target tracks: {summary['merge_totals']['tracks']:,}, split at every gap with >= 2 hits on each side:
  {summary['merge_totals']['unions']:,} unions. The union of the two fragments is the whole track, so each track is
  fitted once (`hnl.vertex.pairing.fit_tracks`, as `hnl.trackbuild.merge_fragments._merge` does).
- frac_fail counts unions; fail_* splits it by condition: rms > 1 mm, zrms > 5 mm, n_out > 1 (hits beyond 3 Huber
  scales), R span < 50 mm.
- Where frac_fail > 1 %:
  - accept_plan: both cuts set to the bin's 99th percentiles (the plan's rule);
  - accept_widen_only: the cuts only raised (max of the default and the 99th percentile).
- diag_*: a diagnostic beyond the plan. diag_max_out_99 is the smallest max_out that keeps 99 % of the unions under
  it; diag_accept_* is the acceptance with that max_out and the widened cuts.

{_md(mt_n)}

Bins above 1 %: {', '.join(over.abs_eta) if len(over) else 'none'}.

The failures come from the outlier count, not from the rms cuts that the plan calibrates. Fail fraction by pT and
|eta| region (unions; diagnostic):

{_md(mpt)}
"""
    with open(path, "w") as fh:
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=T.DERIVED_TP)
    ap.add_argument("--split", default=T.SPLIT_TP)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--note", default=os.path.join(OUTDIR, "calibration_v1.md"))
    ap.add_argument("--summary", default=None, help="JSON with every table (e.g. in the job's log directory)")
    a = ap.parse_args(argv)
    s = run(a.root, a.split, a.outdir, a.note)
    print(json.dumps({k: v for k, v in s.items() if k not in ("res", "slope", "merge", "merge_by_pt")},
                     indent=1, default=str))
    for k in ("res", "slope", "merge", "merge_by_pt"):
        print(pd.DataFrame(s[k]).to_string(index=False), flush=True)
    if a.summary:
        with open(a.summary, "w") as fh:
            json.dump(s, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
