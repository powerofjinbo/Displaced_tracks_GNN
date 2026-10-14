"""TrackML-style digitiser for the pileup overlay (docs/pileup_overlay_plan.md section 5.2; gate C1 of section 10).

Input. Simulated hits of the ACTS v15 GenericDetector in the CsvSimHitWriter layout (particle_id, geometry_id, tx, ty,
tz, tpx, tpy, tpz, ...; one row per sensitive-surface crossing) and a lookup from the ACTS geometry id to the TrackML
(volume, layer, module) id. Barrel ids are the same in both; endcap module numbers are permuted (plan section 3). The
lookup is a parameter (`load_lookup`) and must belong to the build that simulated the hits: PATCHED_LOOKUP for the
z-mirror build of the production route, UNPATCHED_LOOKUP for build-v15. Every module frame, bound, thickness and pitch
comes from TrackML's detectors.csv.

Model. Measured on the cells of TrackML throughput training events (results/pileup_hp/c1_v1.md):
- The sim hit is moved along its true direction onto the mid-plane w = 0 of its TrackML module (a no-op when the
  planes agree, as in the patched build; the distance moved is kept in `w_moved`).
- The track is a straight segment through a slab |w| <= h, with h = module_t / 2 in the barrel volumes (8, 13, 17)
  and h = module_t in the endcaps. These are the effective thicknesses of TrackML's clusters: the module thickness in
  the barrel and twice it in the endcaps.
- Charge from depth w drifts to the readout side w = s h (s = -1 for the pixels, +1 for the strips) and moves by
  (s h - w) tan(LA) in u, as in ACTS CartesianSegmentation::digitizationStep. The Lorentz angles are those of the
  GenericDetector: 0.12 rad in the pixel and short-strip barrels, 0.08 rad in the long-strip barrel, 0 in the endcaps.
- Cells lie on the grid of the module's bounding box with TrackML's pitch: ch0 = floor((u + maxhu) / pitch_u) and
  ch1 = floor((v + hv) / pitch_v). Path outside the grid makes no cell.
- Cell value: in the pixels the path length [mm] in the cell times a factor drawn per cell from a normal distribution
  of mean 1 and width CELL_SIGMA = 0.100; in the strips 1. TrackML's pixel cell values scatter in this way around the
  path length: the ratio has a width of 0.100 for paths above 0.04 mm, the same in barrel and endcaps, and the
  factors of two cells of one hit are uncorrelated (results/pileup_hp/c1_cell_factor_v1.json, 0.74 M cells). The
  per-cell factor produces TrackML's per-hit charge spread and the pixel-barrel resolution in v, so it replaces the
  per-hit factor of plan section 4.4; no per-hit factor goes on top. Without a random generator (`rng=None`) the
  values are the bare path lengths.
- Hit position: the value-weighted mean of the cell centres ((ch + 0.5) pitch - half length), plus the Lorentz shift
  -s module_t tan(LA) in u (ACTS PlanarSteppingAlgorithm), on the plane w = 0. Strip values are 1, so a strip hit sits
  at the plain mean of its strip centres, as in TrackML. TrackML's CSV keeps 6 significant digits (`round_sig`).
- Every crossing is its own hit, including a second crossing of one module by the same particle. A crossing whose
  path misses the grid has no cell and is dropped (counted in `stats`).

    python -m hnl.overlay.digitise --sim-dir DIR --lookup LOOKUP.csv --out-dir OUT --seed S [--events N]
writes the TrackML members hits, cells and truth of each event of DIR (particles come from the gun writer). The
charge factors of event N use the generator `event_rng(S, N)`.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EXT = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml"
DETECTORS = os.path.join(EXT, "detectors.csv")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Module-id tables; every caller passes one explicitly, and it must belong to the build that simulated the hits. The
# patched build's table is written by hnl.overlay.modules (plan section 3, step 3). The unpatched v15 build's table
# (hit_format study, job 6812130; identical to geometry/out/module_map.csv) holds for hits of build-v15 only: the
# patch moves every -z endcap module, so the ids of volumes 7, 12 and 16 differ between the builds.
PATCHED_LOOKUP = os.path.join(REPO, "configs", "trackml", "module_map_acts_zmirror_v1.csv")
UNPATCHED_LOOKUP = "/work/submit/jinboz1/hnl/pileup_design/hit_format/out/module_map_acts_v15_to_trackml.csv"

PIXEL_VOLUMES = (7, 8, 9)
BARREL_VOLUMES = (8, 13, 17)
LORENTZ_ANGLE = {8: 0.12, 13: 0.12, 17: 0.08}        # rad; GenericDetector barrel, endcaps 0
READOUT_SIDE = {7: -1, 8: -1, 9: -1}                 # pixels -1, strips +1
ROT = ("rot_xu", "rot_xv", "rot_xw", "rot_yu", "rot_yv", "rot_yw", "rot_zu", "rot_zv", "rot_zw")
MIN_COS = 1e-6                                       # |cos(incidence)| below this: the track runs in the plane
CELL_SIGMA = 0.100                                   # width of the per-cell pixel charge factor (TrackML cells)
CELL_FACTOR_MIN = 1e-3
_LOOKUP_ALIASES = {"volume": ("volume_id", "volume"), "layer": ("layer_id", "layer"),
                   "acts_module": ("acts_module_id", "acts_module"),
                   "trackml_module": ("trackml_module_id", "trackml_module", "tml_module")}


def module_key(volume, layer, module) -> np.ndarray:
    """The key of hnl.trackml.convert.module_key."""
    return (np.asarray(volume, np.int64) * 1_000_000 + np.asarray(layer, np.int64) * 10_000
            + np.asarray(module, np.int64))


def round_sig(x, digits: int = 6) -> np.ndarray:
    """Round to `digits` significant digits, as TrackML's CSV writes positions."""
    x = np.asarray(x, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(x != 0, np.floor(np.log10(np.abs(x))), 0.0)
    scale = 10.0 ** (digits - 1 - e)
    return np.round(x * scale) / scale


# ------------------------------------------------------------------------------------------------ geometry
@dataclass
class Modules:
    """TrackML detectors.csv, sorted by module key (global = R local + c; local u, v, w)."""
    key: np.ndarray
    c: np.ndarray
    R: np.ndarray
    t: np.ndarray
    minhu: np.ndarray
    maxhu: np.ndarray
    hv: np.ndarray
    pu: np.ndarray
    pv: np.ndarray
    volume: np.ndarray

    @classmethod
    def load(cls, path: str = DETECTORS) -> "Modules":
        d = pd.read_csv(path)
        key = module_key(d.volume_id, d.layer_id, d.module_id)
        o = np.argsort(key, kind="stable")
        if np.any(np.diff(key[o]) == 0):
            raise ValueError(f"duplicate module in {path}")
        return cls(key=key[o], c=d[["cx", "cy", "cz"]].to_numpy(np.float64)[o],
                   R=d[list(ROT)].to_numpy(np.float64).reshape(-1, 3, 3)[o],
                   t=d.module_t.to_numpy(np.float64)[o], minhu=d.module_minhu.to_numpy(np.float64)[o],
                   maxhu=d.module_maxhu.to_numpy(np.float64)[o], hv=d.module_hv.to_numpy(np.float64)[o],
                   pu=d.pitch_u.to_numpy(np.float64)[o], pv=d.pitch_v.to_numpy(np.float64)[o],
                   volume=d.volume_id.to_numpy(np.int64)[o])

    def index(self, volume, layer, module) -> np.ndarray:
        """Row of each (volume, layer, module); -1 if detectors.csv lacks it."""
        k = module_key(volume, layer, module)
        pos = np.minimum(np.searchsorted(self.key, k), len(self.key) - 1)
        return np.where(self.key[pos] == k, pos, -1)


def decode_geometry_id(gid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ACTS v15 geometry id -> (volume, layer, sensitive, boundary | approach): the bit layout of hnl.geometry.ids."""
    g = np.asarray(gid).astype(np.uint64)
    return ((g >> np.uint64(56)).astype(np.int64), ((g >> np.uint64(36)) & np.uint64(0xFFF)).astype(np.int64),
            (g & np.uint64(0x0FFFFFFF)).astype(np.int64),
            (((g >> np.uint64(48)) & np.uint64(0xFF)) | ((g >> np.uint64(28)) & np.uint64(0xFF))).astype(np.int64))


class Lookup:
    """ACTS sensitive module (volume, layer, sensitive id) -> TrackML module id, a bijection within every
    (volume, layer). `from_geometry_id` has the signature of hnl.overlay.modules.ModuleMap, so either can be passed to
    `digitise`; this one returns -1 for an unknown module instead of raising."""

    def __init__(self, volume, layer, acts_module, trackml_module, source: str = ""):
        v, l, a, t = (np.asarray(x, np.int64) for x in (volume, layer, acts_module, trackml_module))
        k = module_key(v, l, a)
        o = np.argsort(k, kind="stable")
        if np.any(np.diff(k[o]) == 0):
            raise ValueError(f"lookup {source}: an ACTS module appears twice")
        if len(np.unique(module_key(v, l, t))) != len(t):
            raise ValueError(f"lookup {source}: a TrackML module appears twice")
        self.key, self.volume, self.layer, self.trackml = k[o], v[o], l[o], t[o]
        self.source = source

    def __len__(self) -> int:
        return len(self.key)

    def from_geometry_id(self, geometry_id) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vol, lay, sen, other = decode_geometry_id(geometry_id)
        k = module_key(vol, lay, sen)
        pos = np.minimum(np.searchsorted(self.key, k), len(self.key) - 1)
        ok = (self.key[pos] == k) & (other == 0) & (sen > 0)
        return np.where(ok, vol, -1), np.where(ok, lay, -1), np.where(ok, self.trackml[pos], -1)


def load_lookup(path: str) -> Lookup:
    """Read an ACTS -> TrackML module table. Accepts the layout of hnl.overlay.modules.write_module_map (volume_id,
    layer_id, acts_module_id, trackml_module_id), of the hit_format study (volume, layer, acts_module,
    acts_geometry_id, trackml_module) and of the geometry study (tml_module, acts_module, acts_geoid). Where the table
    also lists the full ACTS geometry id, its fields must agree with the (volume, layer, module) columns."""
    d = pd.read_csv(path)
    cols = {}
    for name, aliases in _LOOKUP_ALIASES.items():
        hit = [a for a in aliases if a in d.columns]
        if not hit:
            raise ValueError(f"{path}: no column for {name} (tried {aliases})")
        cols[name] = d[hit[0]].to_numpy(np.int64)
    for gcol in ("acts_geometry_id", "acts_geoid"):
        if gcol in d.columns:
            vol, lay, sen, other = decode_geometry_id(d[gcol].to_numpy(np.uint64))
            if not (np.array_equal(vol, cols["volume"]) and np.array_equal(lay, cols["layer"])
                    and np.array_equal(sen, cols["acts_module"]) and not other.any()):
                raise ValueError(f"{path}: {gcol} disagrees with the volume, layer and module columns")
    return Lookup(cols["volume"], cols["layer"], cols["acts_module"], cols["trackml_module"], source=path)


def module_params(modules: Modules, row: np.ndarray) -> dict:
    """Per-hit digitisation parameters of the module rows: half thickness h of the slab, Lorentz tan, readout side,
    pixel flag, grid sizes."""
    vol = modules.volume[row]
    barrel = np.isin(vol, BARREL_VOLUMES)
    t = modules.t[row]
    return {"h": np.where(barrel, 0.5 * t, t), "t": t,
            "tan_la": np.tan(np.array([LORENTZ_ANGLE.get(int(v), 0.0) for v in range(32)]))[vol],
            "side": np.array([READOUT_SIDE.get(int(v), 1) for v in range(32)], np.int64)[vol],
            "pixel": np.isin(vol, PIXEL_VOLUMES), "maxhu": modules.maxhu[row], "hv": modules.hv[row],
            "pu": modules.pu[row], "pv": modules.pv[row], "minhu": modules.minhu[row],
            "n0": np.rint(2 * modules.maxhu[row] / modules.pu[row]).astype(np.int64),
            "n1": np.rint(2 * modules.hv[row] / modules.pv[row]).astype(np.int64), "volume": vol}


# ------------------------------------------------------------------------------------------------ core
def _crossings(a0, a1, half, pitch, n, hit):
    """Parameters t in (0, 1) where the segment a(t) = a0 + t (a1 - a0) crosses a grid boundary k pitch - half,
    0 <= k <= n; returned flat with their hit index."""
    lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
    kmin = np.maximum(np.floor((lo + half) / pitch).astype(np.int64) + 1, 0)
    kmax = np.minimum(np.floor((hi + half) / pitch).astype(np.int64), n)
    cnt = np.maximum(kmax - kmin + 1, 0)
    cnt = np.where(a1 != a0, cnt, 0)
    idx = np.repeat(np.arange(len(a0)), cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    k = kmin[idx] + (np.arange(len(idx)) - start)
    t = (k * pitch[idx] - half[idx] - a0[idx]) / (a1[idx] - a0[idx])
    return hit[idx], t


def digitise_local(lu, lv, tu, tv, p: dict, rng: np.random.Generator | None = None, cell_sigma: float = CELL_SIGMA,
                   charge_factor=None):
    """Cells and positions of straight segments through the slab |w| <= h, given the point (lu, lv) on w = 0 and the
    slopes tu = du/dw, tv = dv/dw (arrays of n hits; module parameters `p` from `module_params`). With `rng`, every
    pixel cell value is the path times a factor N(1, cell_sigma) (clipped at CELL_FACTOR_MIN), drawn in the order of
    the returned cells. charge_factor: an optional factor per hit on the pixel values (see `digitise`).

    Returns (cells, hits): cells = dict(hit, ch0, ch1, value, path) sorted by (hit, ch0, ch1); hits = dict(ncells,
    charge, span0, span1, u, v, path) of length n (ncells 0 and NaN positions for hits without cells)."""
    lu, lv, tu, tv = (np.asarray(a, np.float64) for a in (lu, lv, tu, tv))
    n = len(lu)
    h, tan_la, side = p["h"], p["tan_la"], p["side"]
    x0 = lu - h * tu + (side * h + h) * tan_la
    x1 = lu + h * tu + (side * h - h) * tan_la
    y0, y1 = lv - h * tv, lv + h * tv
    L = 2 * h * np.sqrt(1 + tu * tu + tv * tv)
    hit = np.arange(n)
    hu_, tu_ = _crossings(x0, x1, p["maxhu"], p["pu"], p["n0"], hit)
    hv_, tv_ = _crossings(y0, y1, p["hv"], p["pv"], p["n1"], hit)
    H = np.concatenate([hit, hit, hu_, hv_])
    T = np.clip(np.concatenate([np.zeros(n), np.ones(n), tu_, tv_]), 0.0, 1.0)
    o = np.lexsort((T, H))
    H, T = H[o], T[o]
    same = H[1:] == H[:-1]
    hs, ta, tb = H[:-1][same], T[:-1][same], T[1:][same]
    dt = tb - ta
    keep = dt > 1e-12
    hs, ta, dt = hs[keep], ta[keep], dt[keep]
    tm = ta + 0.5 * dt
    xm = x0[hs] + tm * (x1 - x0)[hs]
    ym = y0[hs] + tm * (y1 - y0)[hs]
    c0 = np.floor((xm + p["maxhu"][hs]) / p["pu"][hs]).astype(np.int64)
    c1 = np.floor((ym + p["hv"][hs]) / p["pv"][hs]).astype(np.int64)
    ok = (c0 >= 0) & (c0 < p["n0"][hs]) & (c1 >= 0) & (c1 < p["n1"][hs])
    hs, c0, c1, path = hs[ok], c0[ok], c1[ok], dt[ok] * L[hs[ok]]
    key = (hs << 32) | (c0 << 16) | c1
    uk, inv = np.unique(key, return_inverse=True)
    cpath = np.bincount(inv, weights=path, minlength=len(uk))
    ch = (uk >> 32).astype(np.int64)
    cc0 = ((uk >> 16) & 0xFFFF).astype(np.int64)
    cc1 = (uk & 0xFFFF).astype(np.int64)
    pix = p["pixel"][ch]
    fac = np.ones(len(ch))
    if rng is not None and cell_sigma > 0:
        fac[pix] = np.maximum(rng.normal(1.0, cell_sigma, int(pix.sum())), CELL_FACTOR_MIN)
    if charge_factor is not None:
        fac = fac * np.asarray(charge_factor, np.float64)[ch]
    val = np.where(pix, cpath * fac, 1.0)
    ncells = np.bincount(ch, minlength=n)
    charge = np.bincount(ch, weights=val, minlength=n)
    tot_path = np.bincount(ch, weights=cpath, minlength=n)
    uc = (cc0 + 0.5) * p["pu"][ch] - p["maxhu"][ch]
    vc = (cc1 + 0.5) * p["pv"][ch] - p["hv"][ch]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.bincount(ch, weights=val * uc, minlength=n) / charge
        v = np.bincount(ch, weights=val * vc, minlength=n) / charge
    u = u - side * p["t"] * tan_la
    span0 = np.zeros(n, np.int64)
    span1 = np.zeros(n, np.int64)
    if len(ch):
        st = np.flatnonzero(np.r_[True, ch[1:] != ch[:-1]])
        span0[ch[st]] = np.maximum.reduceat(cc0, st) - np.minimum.reduceat(cc0, st) + 1
        span1[ch[st]] = np.maximum.reduceat(cc1, st) - np.minimum.reduceat(cc1, st) + 1
    none = ncells == 0
    u[none], v[none] = np.nan, np.nan
    cells = {"hit": ch, "ch0": cc0, "ch1": cc1, "value": val, "path": cpath}
    hits = {"ncells": ncells, "charge": charge, "span0": span0, "span1": span1, "u": u, "v": v, "path": tot_path}
    return cells, hits


# ------------------------------------------------------------------------------------------------ sim hits
@dataclass
class Digitised:
    """Output of `digitise`. hits: one row per digitised crossing, in input order (column sim_row = row of the input);
    cells: hit (row of `hits`), ch0, ch1, value; stats: counts of the dropped crossings and the containment checks."""
    hits: pd.DataFrame
    cells: pd.DataFrame
    stats: dict = field(default_factory=dict)


def digitise(simhits: pd.DataFrame, lookup, modules: Modules, rng: np.random.Generator | None = None,
             cell_sigma: float = CELL_SIGMA, charge_factor=None) -> Digitised:
    """Digitise the sim hits of one event (columns geometry_id, particle_id, tx, ty, tz, tpx, tpy, tpz).
    lookup: a `Lookup` or an hnl.overlay.modules.ModuleMap (anything with from_geometry_id), of the build that
    simulated the hits. rng: the generator of the per-cell pixel charge factor (None: bare path lengths); production
    passes `event_rng(seed, event)`.
    charge_factor: one factor per input row on the pixel values, the per-hit charge emulation of plan section 4.4.
    Gate C1 replaced it by the per-cell factor (results/pileup_hp/c1_v1.md): the per-cell factor alone gives TrackML's
    per-hit charge spread and pixel resolution, while a per-hit factor leaves the pixel-barrel v resolution at 0.7 um
    against TrackML's 3.9 um. Both together would widen the charge spread twice, so that combination raises."""
    vol, lay, mod = lookup.from_geometry_id(simhits.geometry_id.to_numpy(np.uint64))
    return digitise_on_modules(simhits, vol, lay, mod, modules, rng, cell_sigma, charge_factor)


def digitise_on_modules(simhits: pd.DataFrame, vol, lay, mod, modules: Modules,
                        rng: np.random.Generator | None = None, cell_sigma: float = CELL_SIGMA,
                        charge_factor=None) -> Digitised:
    """`digitise` with the TrackML (volume, layer, module) of every row given (-1: unknown). Also replays TrackML truth
    hits (tx, ..., tpz of truth.csv on their own modules) for the C1 comparison."""
    n_in = len(simhits)
    if charge_factor is not None:
        if rng is not None and cell_sigma > 0:
            raise ValueError("a per-hit charge factor and the per-cell factor together double the charge spread; "
                             "pass rng only (results/pileup_hp/c1_v1.md)")
        warnings.warn("per-hit charge factor without the per-cell factor: the pixel resolution fails gate C1 "
                      "(results/pileup_hp/c1_v1.md); pass rng=event_rng(seed, event) instead", stacklevel=3)
        if len(charge_factor) != n_in:
            raise ValueError("charge_factor needs one value per input row")
    vol, lay, mod = (np.asarray(x, np.int64) for x in (vol, lay, mod))
    row = modules.index(vol, lay, mod)
    known = row >= 0
    rr = np.where(known, row, 0)
    pos = simhits[["tx", "ty", "tz"]].to_numpy(np.float64)
    mom = simhits[["tpx", "tpy", "tpz"]].to_numpy(np.float64)
    R, c = modules.R[rr], modules.c[rr]
    lp = np.einsum("nij,ni->nj", R, pos - c)
    ld = np.einsum("nij,ni->nj", R, mom)
    pn = np.linalg.norm(ld, axis=1)
    cosw = np.abs(ld[:, 2]) / np.where(pn > 0, pn, 1.0)
    inplane = cosw < MIN_COS
    use = known & ~inplane
    dw = np.where(use, ld[:, 2], 1.0)
    tu, tv = ld[:, 0] / dw, ld[:, 1] / dw
    lu, lv = lp[:, 0] - lp[:, 2] * tu, lp[:, 1] - lp[:, 2] * tv
    idx = np.flatnonzero(use)
    p = module_params(modules, rr[idx])
    fac = None if charge_factor is None else np.asarray(charge_factor, np.float64)[idx]
    cells, hh = digitise_local(lu[idx], lv[idx], tu[idx], tv[idx], p, rng, cell_sigma, fac)
    good = hh["ncells"] > 0
    gi = idx[good]
    loc = np.stack([hh["u"][good], hh["v"][good], np.zeros(good.sum())], axis=1)
    xyz = np.einsum("nij,nj->ni", R[gi], loc) + c[gi]
    tloc = np.stack([lu[gi], lv[gi], np.zeros(len(gi))], axis=1)
    txyz = np.einsum("nij,nj->ni", R[gi], tloc) + c[gi]
    # containment of the true point on the TrackML plane (trapezoid: half width minhu at v = -hv to maxhu at +hv)
    hv_, mn, mx = p["hv"][good], p["minhu"][good], p["maxhu"][good]
    hu_v = mn + (mx - mn) * (np.clip(lv[gi], -hv_, hv_) + hv_) / (2 * hv_)
    outside = np.hypot(np.maximum(np.abs(lu[gi]) - hu_v, 0), np.maximum(np.abs(lv[gi]) - hv_, 0))
    hits = pd.DataFrame({
        "sim_row": gi, "particle_id": simhits.particle_id.to_numpy()[gi],
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "volume_id": vol[gi], "layer_id": lay[gi], "module_id": mod[gi],
        "u": hh["u"][good], "v": hh["v"][good], "u_true": lu[gi], "v_true": lv[gi],
        "tan_u": tu[gi], "tan_v": tv[gi], "cos_inc": cosw[gi],
        "tx": txyz[:, 0], "ty": txyz[:, 1], "tz": txyz[:, 2],
        "tpx": mom[gi, 0], "tpy": mom[gi, 1], "tpz": mom[gi, 2],
        "ncells": hh["ncells"][good], "charge": hh["charge"][good], "span0": hh["span0"][good],
        "span1": hh["span1"][good], "w_moved": lp[gi, 2], "outside_mm": outside})
    newrow = np.full(len(idx), -1, np.int64)
    newrow[good] = np.arange(good.sum())
    cl = pd.DataFrame({"hit": newrow[cells["hit"]], "ch0": cells["ch0"], "ch1": cells["ch1"],
                       "value": cells["value"]})
    stats = {"n_sim": int(n_in), "n_unknown_module": int((~known).sum()), "n_in_plane": int((known & inplane).sum()),
             "n_no_cells": int((~good).sum()), "n_hits": int(good.sum()), "n_cells": int(len(cl)),
             "max_abs_w_moved_mm": float(np.abs(lp[gi, 2]).max()) if len(gi) else 0.0,
             "n_outside_bounds": int((outside > 1e-6).sum()),
             "max_outside_mm": float(outside.max()) if len(gi) else 0.0}
    return Digitised(hits=hits, cells=cl, stats=stats)


def trackml_members(dig: Digitised, first_hit_id: int = 1, digits: int = 6) -> dict[str, pd.DataFrame]:
    """The hits, cells and truth members of TrackML's CSV layout (hit ids from first_hit_id; weight 0 for every hit,
    plan section 5.3). Positions, cell values and truth rounded to `digits` significant digits."""
    h = dig.hits
    hid = np.arange(first_hit_id, first_hit_id + len(h), dtype=np.int64)
    hits = pd.DataFrame({"hit_id": hid, "x": round_sig(h.x, digits), "y": round_sig(h.y, digits),
                         "z": round_sig(h.z, digits), "volume_id": h.volume_id.to_numpy(np.int64),
                         "layer_id": h.layer_id.to_numpy(np.int64), "module_id": h.module_id.to_numpy(np.int64)})
    c = dig.cells
    cells = pd.DataFrame({"hit_id": hid[c.hit.to_numpy()], "ch0": c.ch0.to_numpy(np.int64),
                          "ch1": c.ch1.to_numpy(np.int64), "value": round_sig(c.value, digits)})
    truth = pd.DataFrame({"hit_id": hid, "particle_id": h.particle_id.to_numpy(np.int64),
                          **{k: round_sig(h[k], digits) for k in ("tx", "ty", "tz", "tpx", "tpy", "tpz")},
                          "weight": np.zeros(len(h))})
    return {"hits": hits, "cells": cells, "truth": truth}


def event_rng(seed: int, event: int) -> np.random.Generator:
    """The generator of the charge factors of one event: independent streams per (seed, event)."""
    return np.random.default_rng([int(seed), int(event)])


def event_number(stem: str) -> int:
    """Event number of an ACTS CSV stem 'event000000042' (or a path ending in it)."""
    return int(os.path.basename(stem).split("-")[0][len("event"):])


def read_sim_event(path: str) -> pd.DataFrame:
    """One CsvSimHitWriter file (geometry and particle ids read as uint64)."""
    return pd.read_csv(path, dtype={"geometry_id": np.uint64, "particle_id": np.uint64})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--detectors", default=DETECTORS)
    ap.add_argument("--events", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    mods = Modules.load(a.detectors)
    lk = load_lookup(a.lookup)
    files = sorted(f for f in os.listdir(a.sim_dir) if f.endswith("-hits.csv"))[: a.events]
    os.makedirs(a.out_dir, exist_ok=True)
    for f in files:
        stem = f[: -len("-hits.csv")]
        dig = digitise(read_sim_event(os.path.join(a.sim_dir, f)), lk, mods, rng=event_rng(a.seed, event_number(stem)))
        for k, df in trackml_members(dig).items():
            df.to_csv(os.path.join(a.out_dir, f"{stem}-{k}.csv"), index=False, float_format="%.6g")
        print(stem, dig.stats, flush=True)


if __name__ == "__main__":
    main()
