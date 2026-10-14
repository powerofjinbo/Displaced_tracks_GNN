"""Emulation of TrackML detector effects on the simulated gun hits of the pileup overlay (docs/pileup_overlay_plan.md
section 4.4, decision D8): hit inefficiency beyond the geometric holes and hadronic stops, which Fatras of ACTS v15
lacks. The third effect of section 4.4, the pixel charge spread, is applied by the digitiser as a per-cell factor
(hnl.overlay.digitise, gate C1 in results/pileup_hp/c1_v1.md), which replaced the per-hit factor of the plan; the
per-hit factor measured here is kept as a diagnostic only.

Measurement (truth only; training_part02 of the throughput phase, every stored event, all of them training events of
configs/trackml/split_tp_v1.json; no calibration, val, test or part03 event; no model output).
1. `select`: prompt charged particles (production radius < 1 mm, pT > 1 GeV, |eta| < 3.5, PDG e, mu, pi, K, p, at
   least one hit) and their TrackML hits. Their initial kinematics are written as Fatras input (several TrackML events
   per Fatras event; particles do not interact with each other in Fatras), so that the same particles are simulated
   in the TrackML-equivalent setup (slurm/overlay/fatras.sbatch: field map, pMin 0.1 GeV, no bremsstrahlung).
2. `tables`: with the same code on both samples (TrackML hits and the Fatras hits of the same particles):
   - hole rates per layer. Barrel: the 10 barrel layers, interior layers between the first and the last hit layer of
     a track. Disks: per endcap volume, interior disks between the first and the last hit disk. A hole is an interior
     layer without a hit. The TrackML rate is geometry plus inefficiency; the Fatras rate is geometry only.
   - the per-hit drop rate eps of each layer solves (1 - g) [eps (1 - f2) + eps^2 f2] = h - g, with h the TrackML hole
     rate, g the Fatras (geometric) rate and f2 the Fatras share of layer crossings with two hits (a layer with two
     hits becomes a hole only if both are dropped). Layers that are never interior (first and last barrel layer,
     outermost disks) take the pooled rate of their volume.
   - hadronic stops: the Fatras hits of a particle give the layers it would cross without interactions, in order.
     The TrackML particle stops after crossing k if its last TrackML hit is at crossing k and Fatras continues. The
     hazard of a layer is stops / exposures (crossings before the last TrackML hit or at it, Fatras continuing). A
     particle that reaches the last-but-one crossing and loses its last hit to inefficiency looks like a stop; the
     expected number of such cases (miss rate of the last layer relative to Fatras) is subtracted. Bins: species
     (pi, K, p by |PDG|), momentum P_EDGES, layer; cells with fewer than MIN_EXPOSURE exposures fall back to coarser
     cells (HAZARD_LEVELS). e and mu are measured as a check and not stopped.
   - modules without hits: modules whose miss rate relative to Fatras exceeds their layer's (MODULE_MIN_*), e.g.
     the |z| ends of the first pixel layer, get their own drop rate (eps_module).
   - pixel charge per hit (diagnostic): charge x cos(incidence) / t_eff of TrackML pixel hits (t_eff = module_t in
     the barrel and 2 module_t in the endcaps), the incidence from the matched Fatras hit and the module normal.
3. `gate` (plan section 10, "Emulation"): the emulation applied to the Fatras hits of the selected particles
   (REPLICAS draws), then the total hole rate per barrel layer (|eta| < 0.8, pT > 2 GeV, hit on the first layer) and
   per disk (pT > 2 GeV, at least two disks of the volume hit) within 0.2 points of TrackML, and the share of pi, K and
   p reaching the last barrel layer within 1 point. With --half the tables come from the even events and the gate runs
   on the odd ones.

Application (`emulate`): on the sim hits of the gun, in this order: hadronic stops (pi, K, p: at each crossing but the
last, stop with probability 1 - exp(-lambda s), lambda of its species, momentum bin and layer, s its path factor;
later hits are removed), then each remaining hit dropped with eps of its layer (or of its module). `emulation_hash`
fixes the version (tables plus this file).

    python -m hnl.overlay.emulate select --work DIR [--events-per-pack 20]
    sbatch --export=ALL,IN=DIR/fatras_in,OUT=DIR/fatras_out,NEV=<packs>,SEED=<s> slurm/overlay/fatras.sbatch
    python -m hnl.overlay.emulate tables --work DIR [--half even] [--out FILE]
    python -m hnl.overlay.emulate gate --work DIR [--tables FILE] [--half odd] --out FILE
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hnl.overlay import fatras_io as FIO
from hnl.trackml import convert as C
from hnl.trackml.split import REPO
from hnl.trackml.throughput import DERIVED_TP, SPLIT_TP, load_tp_split

TABLES = os.path.join(REPO, "configs", "trackml", "overlay_emulation_v1.json")
PART = "training_part02"
BARREL_LAYERS = ((8, 2), (8, 4), (8, 6), (8, 8), (13, 2), (13, 4), (13, 6), (13, 8), (17, 2), (17, 4))
DISK_VOLUMES = (7, 9, 12, 14, 16, 18)
N_DISKS = {7: 7, 9: 7, 12: 6, 14: 6, 16: 6, 18: 6}           # layers 2, 4, ..., 2 n (detectors.csv)
PIXEL_VOLUMES = (7, 8, 9)
BARREL_VOLUMES = (8, 13, 17)
SPECIES = ("e", "mu", "pi", "K", "p")
PDG_SPECIES = {11: 0, 13: 1, 211: 2, 321: 3, 2212: 4}
HADRONS = (2, 3, 4)
P_EDGES = (0.0, 2.0, 4.0, 8.0, np.inf)                      # GeV, initial momentum
SEL_PT_MIN, SEL_RV_MAX, SEL_ABS_ETA_MAX = 1.0, 1.0, 3.5
GATE_PT_MIN, CENTRAL_ABS_ETA = 2.0, 0.8
HOLE_TOL, REACH_TOL = 0.002, 0.01                           # plan section 10: 0.2 and 1 percentage points
MIN_EXPOSURE = 2000
S_MAX = 20.0                                                # cap of the path factor 1 / |cos(incidence)|
MIN_INTERIOR_LAYER, MIN_INTERIOR_VOLUME = 500, 2000         # hole counts below these: pooled rates (see measure)
MODULE_MIN_N, MODULE_MIN_EXCESS, MODULE_MIN_Z = 100, 0.05, 5.0   # module overrides of the drop rate (see measure)
PATCHED_MAP = os.path.join(REPO, "configs", "trackml", "module_map_acts_zmirror_v1.csv")
UNPATCHED_MAP = "/work/submit/jinboz1/hnl/pileup_design/hit_format/out/module_map_acts_v15_to_trackml.csv"
HAZARD_LEVELS = ("species, p bin, layer", "species, layer", "species, p bin, volume", "species, volume",
                 "hadrons, volume")
REPLICAS = 10
NCODE = 2000                                                # layer code = volume * 100 + layer < 2000


def code_of(volume, layer) -> np.ndarray:
    return np.asarray(volume, np.int64) * 100 + np.asarray(layer, np.int64)


def p_bin(p) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(P_EDGES[1:-1]), np.asarray(p, np.float64), side="right"), 0,
                   len(P_EDGES) - 2)


def species_index(pdg) -> np.ndarray:
    a = np.abs(np.asarray(pdg, np.int64))
    out = np.full(len(a), -1, np.int64)
    for k, i in PDG_SPECIES.items():
        out[a == k] = i
    return out


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------ hole counting
_BARREL_IDX = np.full(NCODE, -1, np.int64)
for _i, (_v, _l) in enumerate(BARREL_LAYERS):
    _BARREL_IDX[_v * 100 + _l] = _i


def barrel_presence(track, volume, layer, n_tracks: int) -> np.ndarray:
    """(n_tracks, 10) bool: track has a hit on barrel layer i."""
    idx = _BARREL_IDX[code_of(volume, layer)]
    m = idx >= 0
    pres = np.zeros((n_tracks, len(BARREL_LAYERS)), bool)
    pres[np.asarray(track)[m], idx[m]] = True
    return pres


def disk_presence(track, volume, layer, n_tracks: int, vol: int) -> np.ndarray:
    v = np.asarray(volume)
    m = v == vol
    d = np.asarray(layer)[m] // 2 - 1
    ok = (d >= 0) & (d < N_DISKS[vol])
    pres = np.zeros((n_tracks, N_DISKS[vol]), bool)
    pres[np.asarray(track)[m][ok], d[ok]] = True
    return pres


def interior_holes(pres: np.ndarray, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per column: number of interior positions (strictly between the first and the last True of a row) and of holes
    among them (False), over the rows given (all rows with at least two True by default)."""
    p = pres if rows is None else pres[rows]
    p = p[p.sum(1) >= 2]
    n = p.shape[1]
    first = np.argmax(p, axis=1)
    last = n - 1 - np.argmax(p[:, ::-1], axis=1)
    j = np.arange(n)[None, :]
    inter = (j > first[:, None]) & (j < last[:, None])
    return inter.sum(0), (inter & ~p).sum(0)


def last_barrel_index(pres: np.ndarray) -> np.ndarray:
    """Index of the outermost barrel layer with a hit, -1 for none."""
    n = pres.shape[1]
    any_ = pres.any(1)
    return np.where(any_, n - 1 - np.argmax(pres[:, ::-1], axis=1), -1)


def hole_rates(track, volume, layer, n_tracks: int, central: np.ndarray, forward: np.ndarray) -> dict:
    """Hole counts per barrel layer (tracks in `central` with a hit on the first barrel layer) and per disk (tracks in
    `forward` with at least two hit disks in the volume). Returns {code: [interior, holes]} and the presence matrix
    of the barrel (for the reach fractions)."""
    out = {}
    bp = barrel_presence(track, volume, layer, n_tracks)
    rows = np.flatnonzero(central & bp[:, 0])
    ni, nh = interior_holes(bp, rows)
    for i, (v, l) in enumerate(BARREL_LAYERS):
        out[v * 100 + l] = [int(ni[i]), int(nh[i])]
    for vol in DISK_VOLUMES:
        dp = disk_presence(track, volume, layer, n_tracks, vol)
        ni, nh = interior_holes(dp, np.flatnonzero(forward))
        for d in range(N_DISKS[vol]):
            out[vol * 100 + 2 * (d + 1)] = [int(ni[d]), int(nh[d])]
    return {"holes": out, "barrel_presence": bp}


def double_hit_counts(track, code) -> pd.DataFrame:
    """Per layer code: crossings (track-layer pairs with a hit) and those with two or more hits."""
    s = pd.DataFrame({"t": np.asarray(track), "c": np.asarray(code)}).groupby(["t", "c"]).size()
    g = pd.DataFrame({"n": s.to_numpy(), "c": s.index.get_level_values(1)}).groupby("c").n
    return pd.DataFrame({"crossings": g.size(), "doubles": g.apply(lambda x: int((x >= 2).sum()))})


def per_hit_rate(h: float, g: float, f2: float) -> float:
    """Per-hit drop rate eps with (1 - g) [eps (1 - f2) + eps^2 f2] = h - g (0 if h <= g)."""
    x = max(h - g, 0.0) / max(1.0 - g, 1e-12)
    if x <= 0:
        return 0.0
    if f2 <= 1e-12:
        return min(x, 1.0)
    return float(min((-(1 - f2) + np.sqrt((1 - f2) ** 2 + 4 * f2 * x)) / (2 * f2), 1.0))


# ------------------------------------------------------------------------------------------------ hadronic stops
def path_factor(volume, x, y, px, py, pz) -> np.ndarray:
    """Path length through a layer relative to normal incidence, 1 / |cos(incidence)|: against the radial direction
    in the barrel volumes, against the z axis in the disks; capped at S_MAX."""
    v = np.asarray(volume)
    x, y, px, py, pz = (np.asarray(a, np.float64) for a in (x, y, px, py, pz))
    p = np.sqrt(px * px + py * py + pz * pz)
    r = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(np.isin(v, BARREL_VOLUMES), np.abs(px * x + py * y) / (p * r), np.abs(pz) / p)
    cos = np.nan_to_num(cos, nan=1.0)
    return 1.0 / np.maximum(cos, 1.0 / S_MAX)


def crossing_sequence(track, code, order, s=None, module=None) -> pd.DataFrame:
    """Distinct layers of each track in the order of first crossing: columns track, code, pos, last_pos, s (the
    path factor of the first hit on the layer; 1 if not given) and module (of the first hit; -1 if not given)."""
    n = len(np.asarray(track))
    d = pd.DataFrame({"track": np.asarray(track), "code": np.asarray(code), "order": np.asarray(order),
                      "s": np.ones(n) if s is None else np.asarray(s, np.float64),
                      "module": np.full(n, -1, np.int64) if module is None else np.asarray(module, np.int64)})
    d = d.sort_values(["track", "order"], kind="stable").drop_duplicates(["track", "code"], keep="first")
    d["pos"] = d.groupby("track").cumcount()
    d["last_pos"] = d.groupby("track").pos.transform("max")
    return d[["track", "code", "pos", "last_pos", "s", "module"]].reset_index(drop=True)


def stop_table(seq: pd.DataFrame, obs_track, obs_code) -> pd.DataFrame:
    """Per expected crossing (rows of `crossing_sequence`) whether the observed sample has a hit there, the observed
    last position k of the track and the derived flags: exposure (pos <= k, pos < last_pos), stop (pos == k <
    last_pos), interior (pos < k, for the miss rate) and pen (pos == last_pos - 1 with a hit: the population of
    spurious stops), plus the code of the track's last expected crossing."""
    obs = pd.DataFrame({"track": np.asarray(obs_track), "code": np.asarray(obs_code)}).drop_duplicates()
    obs["obs"] = True
    s = seq.merge(obs, on=["track", "code"], how="left")
    s["obs"] = s.obs.fillna(False).astype(bool)
    s["k"] = s.pos.where(s.obs, -1).groupby(s.track).transform("max")
    s = s[s.k >= 0].copy()
    s["exposure"] = (s.pos <= s.k) & (s.pos < s.last_pos)
    s["stop"] = (s.pos == s.k) & (s.k < s.last_pos)
    s["interior"] = s.pos < s.k
    s["pen"] = s.obs & (s.pos == s.last_pos - 1)
    last_code = s[s.pos == s.last_pos].set_index("track").code
    s["next_code"] = s.track.map(last_code).fillna(-1).astype(np.int64)
    return s


def miss_rates(st: pd.DataFrame) -> pd.Series:
    """Per code: share of interior expected crossings (a later observed hit exists) without an observed hit."""
    i = st[st.interior]
    g = i.groupby("code").obs
    return 1.0 - g.mean()


def module_overrides(st: pd.DataFrame, miss: pd.Series, min_n: int = MODULE_MIN_N,
                     min_excess: float = MODULE_MIN_EXCESS, min_z: float = MODULE_MIN_Z) -> pd.DataFrame:
    """Modules whose miss rate relative to the expectation (interior crossings, module of the expected hit) exceeds
    their layer's by at least min_excess with at least min_n crossings and a binomial z of at least min_z: TrackML
    regions without hits (e.g. the |z| ends of the first pixel layer). Columns code, module, n, miss, layer_miss, z."""
    i = st[st.interior & (st.module >= 0)]
    g = i.groupby(["code", "module"]).obs.agg(["size", "mean"]).reset_index()
    g["miss"] = 1.0 - g["mean"]
    g["layer_miss"] = g.code.map(miss).fillna(0.0)
    sig = np.sqrt(np.maximum(g.layer_miss * (1 - g.layer_miss), 1e-6) / g["size"])
    g["z"] = (g.miss - g.layer_miss) / sig
    f = g[(g["size"] >= min_n) & (g.miss - g.layer_miss >= min_excess) & (g.z >= min_z)]
    return f.rename(columns={"size": "n"})[["code", "module", "n", "miss", "layer_miss", "z"]].reset_index(drop=True)


def hazard_counts(st: pd.DataFrame, miss: pd.Series, species: np.ndarray, pbin: np.ndarray) -> pd.DataFrame:
    """Exposures, the sums S1, S2, S3 of s, s^2, s^3 over the exposures (s = path factor), stops and expected
    spurious stops per (species, p bin, code); species and pbin per track index."""
    s = st.assign(sp=species[st.track.to_numpy()], pb=pbin[st.track.to_numpy()])
    m = miss.reindex(np.arange(NCODE)).fillna(0.0).to_numpy()
    s["spur"] = np.where(s.pen, m[np.clip(s.next_code.to_numpy(), 0, NCODE - 1)], 0.0)
    e = s.exposure.to_numpy()
    s["S1"], s["S2"], s["S3"] = np.where(e, s.s, 0.0), np.where(e, s.s ** 2, 0.0), np.where(e, s.s ** 3, 0.0)
    g = s.groupby(["sp", "pb", "code"])
    return pd.DataFrame({"exposure": g.exposure.sum(), "S1": g.S1.sum(), "S2": g.S2.sum(), "S3": g.S3.sum(),
                         "stop": g.stop.sum(), "spurious": g.spur.sum()}).reset_index()


def solve_lambda(K, S1, S2, S3) -> np.ndarray:
    """Hazard per unit path lambda with sum_i (1 - exp(-lambda s_i)) = K over the exposures, the left side expanded to
    third order (lambda S1 - lambda^2 S2 / 2 + lambda^3 S3 / 6, increasing in lambda since S2^2 <= S1 S3); Newton
    from K / S1."""
    K, S1, S2, S3 = (np.asarray(a, np.float64) for a in (K, S1, S2, S3))
    lam = np.where(S1 > 0, np.maximum(K, 0) / np.where(S1 > 0, S1, 1), 0.0)
    for _ in range(30):
        f = lam * S1 - lam ** 2 * S2 / 2 + lam ** 3 * S3 / 6 - np.maximum(K, 0)
        d = S1 - lam * S2 + lam ** 2 * S3 / 2
        lam = np.where(d > 0, np.maximum(lam - f / np.where(d > 0, d, 1), 0.0), lam)
    return lam


def resolve_hazards(cnt: pd.DataFrame, min_exposure: int = MIN_EXPOSURE) -> dict:
    """Hazard per unit path lambda per (species, p bin, code) with the fallback levels of HAZARD_LEVELS (K = stops -
    expected spurious stops, `solve_lambda`). The hazard of a crossing with path factor s is 1 - exp(-lambda s).
    Returns {species: {code: {"lam": [per p bin], "level": [per p bin], "exposure": [...]}}}."""
    c = cnt.copy()
    c["vol"] = c.code // 100
    hadron = c.sp.isin(HADRONS)

    def rate(df, keys):
        g = df.groupby(keys)[["exposure", "S1", "S2", "S3", "stop", "spurious"]].sum()
        g["h"] = solve_lambda(g.stop - g.spurious, g.S1, g.S2, g.S3)
        return g

    levels = [rate(c, ["sp", "pb", "code"]), rate(c, ["sp", "code"]), rate(c, ["sp", "pb", "vol"]),
              rate(c, ["sp", "vol"]), rate(c[hadron], ["vol"])]
    out = {}
    nb = len(P_EDGES) - 1
    for sp in sorted(c.sp.unique()):
        out[SPECIES[sp]] = {}
        for code in sorted(c[c.sp == sp].code.unique()):
            vol = code // 100
            hs, lv, ex = [], [], []
            for b in range(nb):
                keys = [(sp, b, code), (sp, code), (sp, b, vol), (sp, vol), vol]
                chosen = (0.0, -1, 0)
                for li, (tab, key) in enumerate(zip(levels, keys)):
                    if key in tab.index and tab.loc[key, "exposure"] >= min_exposure:
                        chosen = (float(tab.loc[key, "h"]), li, int(tab.loc[key, "exposure"]))
                        break
                hs.append(round(chosen[0], 7))
                lv.append(chosen[1])
                ex.append(chosen[2])
            out[SPECIES[sp]][str(int(code))] = {"lam": hs, "level": lv, "exposure": ex}
    return out


# ------------------------------------------------------------------------------------------------ application
class Tables:
    """The emulation tables (configs/trackml/overlay_emulation_v1.json) as lookup arrays."""

    def __init__(self, d: dict):
        self.d = d
        m = d["model"]
        self.p_edges = np.array(m["p_edges"], np.float64)
        self.eps = np.zeros(NCODE)
        for k, v in m["eps_per_hit"].items():
            self.eps[int(k)] = float(v)
        mo = m.get("eps_module", {})
        self.eps_module = pd.Series({int(c) * 100000 + int(mm): float(v) for c, mm, v in
                                     (k.split(":") + [v] for k, v in mo.items())}, dtype=np.float64)
        nb = len(self.p_edges) - 1
        self.lam = np.zeros((len(SPECIES), nb, NCODE))
        for sp, tab in m["hazard"].items():
            i = SPECIES.index(sp)
            if i not in HADRONS:
                continue
            for k, v in tab.items():
                self.lam[i, :, int(k)] = v["lam"]

    @classmethod
    def load(cls, path: str = TABLES) -> "Tables":
        with open(path) as fh:
            return cls(json.load(fh))

    def p_bin(self, p) -> np.ndarray:
        return np.clip(np.searchsorted(self.p_edges[1:-1], np.asarray(p, np.float64), side="right"), 0,
                       len(self.p_edges) - 2)



def emulate(hits: pd.DataFrame, particles: pd.DataFrame, tables: Tables, rng: np.random.Generator) -> dict:
    """Hadronic stops and inefficiency for one event's sim hits.

    hits: particle_id, volume, layer, order (position along the particle, e.g. the Fatras hit index), optionally
    module (TrackML module id; modules listed in the tables' eps_module take that drop rate), and either
    path_factor or the true position and momentum at the surface (tx, ty, tpx, tpy, tpz; `path_factor`); particles:
    particle_id, pdg, p (initial momentum, GeV). The stop hazard of a crossing is 1 - exp(-lambda s) with s the path
    factor of its first hit. Random numbers are drawn in a fixed order (one per crossing, then one per hit, in
    (particle_id, order) order), so the result depends only on the inputs and the generator state.
    Returns keep (bool per input row), stopped (removed by a stop) and dropped (removed by inefficiency)."""
    n = len(hits)
    pid = hits.particle_id.to_numpy()
    o = np.lexsort((hits.order.to_numpy(), pid))
    vol = hits.volume.to_numpy(np.int64)[o]
    code = code_of(vol, hits.layer.to_numpy()[o])
    if "path_factor" in hits.columns:
        s_hit = hits.path_factor.to_numpy(np.float64)[o]
    else:
        s_hit = path_factor(vol, *(hits[c].to_numpy()[o] for c in ("tx", "ty", "tpx", "tpy", "tpz")))
    sp_pid = particles.particle_id.to_numpy()
    sidx = pd.Index(sp_pid)
    if not sidx.is_unique:
        raise ValueError("duplicate particle_id in particles")
    prow = sidx.get_indexer(pid[o])
    if (prow < 0).any():
        raise KeyError("sim hits of particles missing from the particle table")
    spc = species_index(particles.pdg.to_numpy())[prow]
    pb = tables.p_bin(particles.p.to_numpy())[prow]
    ps = pid[o]
    newp = np.r_[True, ps[1:] != ps[:-1]] if n else np.zeros(0, bool)
    newc = newp | np.r_[True, code[1:] != code[:-1]] if n else np.zeros(0, bool)
    cross = np.cumsum(newc) - 1                               # crossing index per sorted hit
    nc = int(cross[-1]) + 1 if n else 0
    c_first = np.flatnonzero(newc)
    c_part = np.cumsum(newp)[c_first] - 1                      # particle ordinal per crossing
    c_last = np.r_[c_part[1:] != c_part[:-1], True] if nc else np.zeros(0, bool)
    u_stop = rng.random(nc)
    u_drop = rng.random(n)
    sp_c, pb_c, code_c = spc[c_first], pb[c_first], code[c_first]
    had = np.isin(sp_c, HADRONS)
    lam = np.where(had, tables.lam[np.clip(sp_c, 0, None), pb_c, code_c], 0.0)
    h = 1.0 - np.exp(-lam * s_hit[c_first])
    stop_here = (u_stop < h) & ~c_last
    # first stop crossing of each particle; crossings after it are removed
    first_stop = np.full(int(c_part.max()) + 1 if nc else 0, np.iinfo(np.int64).max)
    sc = np.flatnonzero(stop_here)
    np.minimum.at(first_stop, c_part[sc], sc)
    removed_c = np.arange(nc) > first_stop[c_part] if nc else np.zeros(0, bool)
    stopped = removed_c[cross] if n else np.zeros(0, bool)
    eps = tables.eps[code]
    if "module" in hits.columns and len(tables.eps_module):
        mk = code * 100000 + hits.module.to_numpy(np.int64)[o]
        ov = tables.eps_module.reindex(mk).to_numpy()
        eps = np.where(np.isfinite(ov), ov, eps)
    dropped = ~stopped & (u_drop < eps)
    keep = np.empty(n, bool)
    st = np.empty(n, bool)
    dr = np.empty(n, bool)
    keep[o], st[o], dr[o] = ~stopped & ~dropped, stopped, dropped
    return {"keep": keep, "stopped": st, "dropped": dr}


def emulation_hash(path: str = TABLES) -> str:
    """sha256 of the emulation model (the 'model' block of the tables, canonical JSON) and of this file."""
    with open(path) as fh:
        model = json.load(fh)["model"]
    h = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode())
    with open(os.path.abspath(__file__), "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ select
def part_events(root: str = DERIVED_TP, part: str = PART) -> list[tuple[str, int, int]]:
    """(shard, row group, event) of every stored event of the part."""
    out = []
    with open(os.path.join(root, f"meta_{part}.json")) as fh:
        meta = json.load(fh)
    for first, sh in sorted(meta["shards"].items()):
        f = pq.ParquetFile(os.path.join(root, sh["hits"]["file"]))
        col = f.schema_arrow.get_field_index("event")
        for i in range(f.metadata.num_row_groups):
            out.append((sh["hits"]["file"], i, int(f.metadata.row_group(i).column(col).statistics.min)))
    return out


def select(work: str, events_per_pack: int = 20, root: str = DERIVED_TP, part: str = PART,
           max_events: int | None = None) -> dict:
    """Step 1 of the measurement (module docstring)."""
    t0 = time.time()
    split = load_tp_split(SPLIT_TP)
    forbid = split["val"] | split["test"] | split["calib"] | split["dev"]
    evs = part_events(root, part)[:max_events]
    if any(e in forbid or e not in split["train"] for _, _, e in evs):
        raise RuntimeError("a non-training or calibration event in the measurement set")
    fin = os.path.join(work, "fatras_in")
    os.makedirs(fin, exist_ok=True)
    P, H = [], []
    pack_rows: dict[int, list] = {}
    track0 = 0
    files: dict[str, tuple] = {}
    for j, (fname, rg, e) in enumerate(evs):
        if fname not in files:
            files[fname] = tuple(pq.ParquetFile(os.path.join(root, fname.replace("hits/", f"{k}/", 1)))
                                 for k in ("hits", "particles", "ptype"))
        fh, fp, fy = files[fname]
        p = fp.read_row_group(rg, columns=["event", "particle_id", "vx", "vy", "vz", "px", "py", "pz", "q", "pt", "eta",
                                           "rv", "n_hits"]).to_pandas()
        y = fy.read_row_group(rg, columns=["event", "particle_id", "particle_type"]).to_pandas()
        if not (p.event == e).all() or not np.array_equal(y.particle_id.to_numpy(), p.particle_id.to_numpy()):
            raise RuntimeError(f"event {e}: particles and ptype rows differ")
        p["pdg"] = y.particle_type.to_numpy()
        sp = species_index(p.pdg.to_numpy())
        sel = p[(p.rv < SEL_RV_MAX) & (p.pt > SEL_PT_MIN) & (p.eta.abs() < SEL_ABS_ETA_MAX) & (sp >= 0)
                & (p.n_hits > 0)].copy()
        if len(sel) >= 0xFFFF:
            raise RuntimeError(f"event {e}: too many selected particles for the barcode field")
        pack, slot = j // events_per_pack, j % events_per_pack
        sel["track"] = np.arange(track0, track0 + len(sel), dtype=np.int64)
        track0 += len(sel)
        sel["pack"] = pack
        sel["barcode"] = ((np.uint64(slot + 1) << np.uint64(52))
                          | (np.arange(1, len(sel) + 1, dtype=np.uint64) << np.uint64(24)))
        sel["p"] = np.sqrt(sel.px.astype(np.float64) ** 2 + sel.py ** 2 + sel.pz ** 2)
        h = fh.read_row_group(rg, columns=["event", "x", "y", "z", "volume", "layer", "module", "pid", "ncells",
                                           "charge"]).to_pandas()
        h = h[h.pid.isin(sel.particle_id)]
        h["track"] = h.pid.map(pd.Series(sel.track.to_numpy(), index=sel.particle_id.to_numpy())).astype(np.int64)
        H.append(h.drop(columns=["pid"]))
        P.append(sel[["track", "event", "particle_id", "pdg", "q", "vx", "vy", "vz", "px", "py", "pz", "pt", "eta", "p",
                      "n_hits", "pack", "barcode"]])
        pdg_abs = np.abs(sel.pdg.to_numpy())
        m = np.array([FIO.MASS[int(a)] for a in pdg_abs])
        pack_rows.setdefault(pack, []).append(pd.DataFrame({
            "particle_id": sel.barcode.to_numpy(np.uint64), "particle_type": sel.pdg.to_numpy(np.int64), "process": 0,
            "vx": sel.vx.to_numpy(np.float64), "vy": sel.vy.to_numpy(np.float64), "vz": sel.vz.to_numpy(np.float64),
            "vt": 0.0, "px": sel.px.to_numpy(np.float64), "py": sel.py.to_numpy(np.float64),
            "pz": sel.pz.to_numpy(np.float64), "m": m, "q": sel.q.to_numpy(np.int64)}))
    for pack, rows in sorted(pack_rows.items()):
        FIO.write_particles(pd.concat(rows, ignore_index=True)[FIO.PARTICLE_COLUMNS], fin, pack)
    P = pd.concat(P, ignore_index=True)
    H = pd.concat(H, ignore_index=True)
    P.to_parquet(os.path.join(work, "sel_particles.parquet"), index=False)
    H.to_parquet(os.path.join(work, "sel_hits_trackml.parquet"), index=False)
    rec = {"part": part, "root": root, "events": [int(e) for _, _, e in evs], "n_events": len(evs),
           "n_packs": len(pack_rows), "events_per_pack": events_per_pack, "particles": int(len(P)),
           "trackml_hits": int(len(H)), "split": {"path": os.path.relpath(SPLIT_TP, REPO), "sha256": split["sha256"]},
           "selection": {"rv_max_mm": SEL_RV_MAX, "pt_min_gev": SEL_PT_MIN, "abs_eta_max": SEL_ABS_ETA_MAX,
                         "species": SPECIES, "n_hits_min": 1},
           "utc": utc(), "seconds": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    with open(os.path.join(work, "select.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ------------------------------------------------------------------------------------------------ measurement inputs
def load_work(work: str, half: str | None = None, fatras: str = "fatras_out") -> dict:
    """Selected particles, TrackML hits and Fatras hits (track index attached; Fatras output directory `fatras` of
    the work directory), optionally one half of the events (even / odd event ids)."""
    P = pd.read_parquet(os.path.join(work, "sel_particles.parquet"))
    H = pd.read_parquet(os.path.join(work, "sel_hits_trackml.parquet"))
    F = FIO.read_packed(os.path.join(work, fatras, "simhits.parquet"),
                        columns=["event", "particle_id", "geometry_id", "volume", "layer", "tx", "ty", "tz", "tpx",
                                 "tpy", "tpz", "index"])
    from hnl.overlay import digitise as D
    lk = D.load_lookup(PATCHED_MAP if os.path.exists(os.path.join(work, fatras, "PATCHED")) else UNPATCHED_MAP)
    F["module"] = lk.from_geometry_id(F.geometry_id.to_numpy(np.uint64))[2]
    F = F.drop(columns=["geometry_id"])
    key = pd.DataFrame({"event": P.pack.to_numpy(np.int64), "particle_id": P.barcode.to_numpy(np.uint64),
                        "track": P.track.to_numpy(np.int64)})
    if key.duplicated(["event", "particle_id"]).any():
        raise RuntimeError("duplicate (pack, barcode) in sel_particles")
    F["event"] = F.event.astype(np.int64)
    F = F.merge(key, on=["event", "particle_id"], how="left")
    unknown = int(F.track.isna().sum())
    F = F[F.track.notna()].copy()
    F["track"] = F.track.astype(np.int64)
    if half in ("even", "odd"):
        keep_ev = P.event % 2 == (0 if half == "even" else 1)
        tracks = set(P.track[keep_ev])
        P = P[keep_ev]
        H = H[H.track.isin(tracks)]
        F = F[F.track.isin(tracks)]
    return {"P": P.reset_index(drop=True), "H": H.reset_index(drop=True), "F": F.reset_index(drop=True),
            "unknown_fatras_hits": unknown}


def selections(P: pd.DataFrame, n_tracks: int) -> tuple[np.ndarray, np.ndarray]:
    central = np.zeros(n_tracks, bool)
    forward = np.zeros(n_tracks, bool)
    t = P.track.to_numpy()
    central[t] = (P.pt.to_numpy() > GATE_PT_MIN) & (np.abs(P.eta.to_numpy()) < CENTRAL_ABS_ETA)
    forward[t] = P.pt.to_numpy() > GATE_PT_MIN
    return central, forward


def reach_fractions(bp: np.ndarray, central: np.ndarray, species: np.ndarray) -> dict:
    rows = central & bp[:, 0]
    last = last_barrel_index(bp)
    out = {}
    for i in (1, 2, 3, 4):
        m = rows & (species == i)
        if m.sum():
            out[SPECIES[i]] = {"tracks": int(m.sum()), "reach_last": float(np.mean(last[m] == len(BARREL_LAYERS) - 1)),
                               "last_layer_hist": np.bincount(last[m], minlength=len(BARREL_LAYERS)).tolist()}
    return out


def charge_factor_sample(H: pd.DataFrame, F: pd.DataFrame, max_dist_mm: float = 2.0) -> pd.DataFrame:
    """TrackML pixel hits matched to the nearest Fatras hit of the same particle on the same layer (within max_dist):
    region, ncells, |cos| of the incidence on the TrackML module and the factor charge x |cos| / t_eff."""
    h = H[H.volume.isin(PIXEL_VOLUMES)].reset_index(drop=True)
    h["hrow"] = np.arange(len(h))
    f = F[F.volume.isin(PIXEL_VOLUMES)][["track", "volume", "layer", "tx", "ty", "tz", "tpx", "tpy", "tpz"]]
    m = h.merge(f, on=["track", "volume", "layer"], how="inner")
    d = np.sqrt((m.x - m.tx) ** 2 + (m.y - m.ty) ** 2 + (m.z - m.tz) ** 2)
    m = m.assign(d=d.to_numpy()).sort_values(["hrow", "d"]).drop_duplicates("hrow", keep="first")
    m = m[m.d < max_dist_mm]
    det = C.load_detectors()
    key = C.module_key(m.volume, m.layer, m.module)
    pos = np.searchsorted(det["key"], key)
    if not np.array_equal(det["key"][pos], key):
        raise KeyError("TrackML hit on a module missing from detectors.csv")
    w = det["R"][pos][:, :, 2]
    pv = m[["tpx", "tpy", "tpz"]].to_numpy(np.float64)
    cos = np.abs((w * pv).sum(1)) / np.linalg.norm(pv, axis=1)
    barrel = m.volume.to_numpy() == 8
    t_eff = np.where(barrel, det["t"][pos], 2 * det["t"][pos])
    return pd.DataFrame({"region": np.where(barrel, "barrel", "endcap"), "ncells": m.ncells.to_numpy(),
                         "cos": cos, "factor": m.charge.to_numpy(np.float64) * cos / t_eff,
                         "dist_mm": m.d.to_numpy()})


def quantile_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 5),
            **{f"q{int(round(q * 100))}": round(float(np.quantile(x, q)), 5) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}


# ------------------------------------------------------------------------------------------------ tables
def measure(work: str, half: str | None = None, fatras: str = "fatras_out") -> dict:
    """Step 2 of the measurement (module docstring): the tables dict (with the diagnostics)."""
    t0 = time.time()
    W = load_work(work, half, fatras)
    P, H, F = W["P"], W["H"], W["F"]
    n_tracks = int(max(P.track.max(), H.track.max() if len(H) else 0, F.track.max() if len(F) else 0)) + 1
    central, forward = selections(P, n_tracks)
    species = np.full(n_tracks, -1, np.int64)
    species[P.track.to_numpy()] = species_index(P.pdg.to_numpy())
    pbin = np.zeros(n_tracks, np.int64)
    pbin[P.track.to_numpy()] = p_bin(P.p.to_numpy())

    tml = hole_rates(H.track.to_numpy(), H.volume.to_numpy(), H.layer.to_numpy(), n_tracks, central, forward)
    v15 = hole_rates(F.track.to_numpy(), F.volume.to_numpy(), F.layer.to_numpy(), n_tracks, central, forward)
    dbl = double_hit_counts(F.track.to_numpy(), code_of(F.volume, F.layer))
    dbl_t = double_hit_counts(H.track.to_numpy(), code_of(H.volume, H.layer))

    # per-hit inefficiency per layer, pooled per volume where a layer is never interior
    eps, rows = {}, []
    for c in sorted(tml["holes"]):
        ni, nh = tml["holes"][c]
        gi, gh = v15["holes"][c]
        f2 = float(dbl.doubles.get(c, 0) / dbl.crossings.get(c, 1)) if c in dbl.index else 0.0
        h = nh / ni if ni else np.nan
        g = gh / gi if gi else np.nan
        rows.append({"code": c, "volume": c // 100, "tml_interior": ni, "tml_holes": nh, "tml_rate": h,
                     "v15_interior": gi, "v15_holes": gh, "v15_rate": g, "v15_double_frac": f2,
                     "tml_double_frac": float(dbl_t.doubles.get(c, 0) / dbl_t.crossings.get(c, 1))
                     if c in dbl_t.index else 0.0})
    R = pd.DataFrame(rows)
    # -z strip disks of the unpatched build differ from TrackML (plan section 3): their geometric rate comes from the
    # z-mirrored +z disk (the patched geometry is the z-mirror of +z) unless the Fatras sample comes from it
    patched = os.path.exists(os.path.join(work, fatras, "PATCHED"))
    mirror = {12: 14, 16: 18}
    R["g_used"] = R.v15_rate
    R["g_source"] = "same layer"
    if not patched:
        for vneg, vpos in mirror.items():
            for i in R.index[R.volume == vneg]:
                twin = R[(R.volume == vpos) & (R.code % 100 == R.at[i, "code"] % 100)].iloc[0]
                R.at[i, "g_used"] = twin.v15_rate
                R.at[i, "g_source"] = f"mirror {int(twin.code)} (unpatched -z strip disk)"
    R["eps"] = [per_hit_rate(r.tml_rate, r.g_used, r.v15_double_frac) if r.tml_interior > 0 and np.isfinite(r.g_used)
                else np.nan for r in R.itertuples()]
    # layers with fewer than MIN_INTERIOR_LAYER interior crossings take their volume's pooled rate; volumes with
    # fewer than MIN_INTERIOR_VOLUME (the long-strip disks: prompt tracks cross at most two of them) take the pooled
    # rate of the measured strip layers
    meas = R[(R.tml_interior >= MIN_INTERIOR_LAYER) & np.isfinite(R.eps)]
    strips = meas[~meas.volume.isin(PIXEL_VOLUMES)]
    strip_pool = float(np.average(strips.eps, weights=strips.tml_interior)) if len(strips) else 0.0
    pooled, src = {}, {}
    for vol in sorted(set(R.volume)):
        g = meas[meas.volume == vol]
        if g.tml_interior.sum() >= MIN_INTERIOR_VOLUME:
            pooled[int(vol)] = float(np.average(g.eps, weights=g.tml_interior))
            src[int(vol)] = f"volume {vol} pooled"
        else:
            pooled[int(vol)] = strip_pool
            src[int(vol)] = "strip layers pooled (too few interior crossings in the volume)"
    R["eps_used"] = np.nan
    R["eps_source"] = ""
    for i, r in R.iterrows():
        if r.tml_interior >= MIN_INTERIOR_LAYER and np.isfinite(r.eps) and \
                R[R.volume == r.volume].tml_interior.sum() >= MIN_INTERIOR_VOLUME:
            R.at[i, "eps_used"], R.at[i, "eps_source"] = r.eps, "layer"
        else:
            R.at[i, "eps_used"], R.at[i, "eps_source"] = pooled[int(r.volume)], src[int(r.volume)]
    for r in R.itertuples():
        eps[str(int(r.code))] = round(float(r.eps_used), 6)

    # hadronic stops relative to the Fatras expectation
    seq = crossing_sequence(F.track.to_numpy(), code_of(F.volume, F.layer), F["index"].to_numpy(),
                            path_factor(F.volume, F.tx, F.ty, F.tpx, F.tpy, F.tpz), F.module.to_numpy())
    st = stop_table(seq, H.track.to_numpy(), code_of(H.volume, H.layer))
    miss = miss_rates(st)
    mo = module_overrides(st, miss)
    # spurious stops need the miss rate of the last expected layer; layers that are rarely interior (the outermost
    # barrel layer, outer disks) take the miss probability of their drop rate, eps (1 - f2) + eps^2 f2
    n_int = st[st.interior].groupby("code").size()
    miss_spur = miss.copy()
    for r in R.itertuples():
        if n_int.get(r.code, 0) < MIN_INTERIOR_LAYER:
            e, f2 = float(r.eps_used), float(r.v15_double_frac)
            miss_spur[r.code] = e * (1 - f2) + e * e * f2
    eps_module = {f"{int(r.code)}:{int(r.module)}": round(float(min(max(r.miss, 0.0), 1.0)), 5)
                  for r in mo.itertuples()}
    cnt = hazard_counts(st, miss_spur, species, pbin)
    hazard = resolve_hazards(cnt)
    n_matched = int(st.track.nunique())

    # pixel charge factor
    q = charge_factor_sample(H, F)
    diag_q = {}
    for reg in ("barrel", "endcap"):
        x = q[q.region == reg]
        diag_q[reg] = {"all": quantile_summary(x.factor),
                       "near_normal_cos_gt_0.9": quantile_summary(x.factor[x.cos > 0.9]),
                       "by_ncells": {k: quantile_summary(x.factor[sel]) for k, sel in
                                     (("1", x.ncells == 1), ("2", x.ncells == 2),
                                      ("3-4", x.ncells.between(3, 4)), (">=5", x.ncells >= 5))},
                       "by_cos": {f"{lo}-{hi}": quantile_summary(x.factor[(x.cos >= lo) & (x.cos < hi)])
                                  for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01))},
                       "match_dist_mm": quantile_summary(x.dist_mm)}

    # e / mu hazards (method check: no hadronic interactions)
    lep = cnt[cnt.sp.isin((0, 1))].groupby("sp")[["exposure", "stop", "spurious"]].sum()
    lep_check = {SPECIES[int(s)]: {"exposure": int(r.exposure), "stops": int(r.stop),
                                   "spurious_expected": round(float(r.spurious), 1),
                                   "hazard_per_crossing": round(float(max(r.stop - r.spurious, 0) / max(r.exposure, 1)), 6)}
                 for s, r in lep.iterrows()}
    had = cnt[cnt.sp.isin(HADRONS)].groupby("sp")[["exposure", "S1", "stop", "spurious"]].sum()
    had_sum = {SPECIES[int(s)]: {"exposure": int(r.exposure), "stops": int(r.stop),
                                 "spurious_expected": round(float(r.spurious), 1),
                                 "hazard_per_crossing": round(float(max(r.stop - r.spurious, 0) / max(r.exposure, 1)), 6),
                                 "mean_path_factor": round(float(r.S1 / max(r.exposure, 1)), 4),
                                 "lambda_first_order": round(float(max(r.stop - r.spurious, 0) / max(r.S1, 1)), 6)}
               for s, r in had.iterrows()}
    # lambda per p bin in the barrel and the disks, all hadrons (is the path-factor model flat in p?)
    hc = cnt[cnt.sp.isin(HADRONS)].assign(region=lambda d: np.where(np.isin(d.code // 100, BARREL_VOLUMES),
                                                                   "barrel", "disks"))
    gp = hc.groupby(["region", "pb"])[["exposure", "S1", "S2", "S3", "stop", "spurious"]].sum()
    gp["lam"] = solve_lambda(gp.stop - gp.spurious, gp.S1, gp.S2, gp.S3)
    lam_by_p = {f"{reg} p bin {int(pb)}": {"exposure": int(r.exposure), "lambda": round(float(r.lam), 6),
                                           "mean_path_factor": round(float(r.S1 / max(r.exposure, 1)), 3)}
                for (reg, pb), r in gp.iterrows()}

    sel = json.load(open(os.path.join(work, "select.json")))
    fat = json.load(open(os.path.join(work, fatras, "fatras.json"))) if os.path.exists(
        os.path.join(work, fatras, "fatras.json")) else {}
    model = {"p_edges": [float(x) if np.isfinite(x) else 1e9 for x in P_EDGES], "eps_per_hit": eps,
             "eps_module": eps_module, "hazard": hazard}
    out = {
        "version": "overlay_emulation_v1", "made_utc": utc(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "plan": "docs/pileup_overlay_plan.md section 4.4 (decision D8)",
        "source": {"part": sel["part"], "n_events": sel["n_events"], "first_last": [min(sel["events"]),
                                                                                   max(sel["events"])],
                   "half": half or "all", "split": sel["split"], "selection": sel["selection"],
                   "fatras": fat, "fatras_patched_build": patched,
                   "code_sha256": C.sha256_file(os.path.abspath(__file__))},
        "model": model,
        "diagnostics": {
            "holes_per_layer": R.round(6).to_dict(orient="records"),
            "eps_pooled_per_volume": {str(k): round(v, 6) for k, v in pooled.items()},
            "miss_rate_vs_fatras": {str(int(k)): round(float(v), 6) for k, v in miss.items()},
            "miss_rate_for_spurious_stops": {str(int(k)): round(float(v), 6) for k, v in miss_spur.items()},
            "hazard_counts": cnt.assign(sp=cnt.sp.map(lambda s: SPECIES[int(s)])).to_dict(orient="records"),
            "hazard_summary_hadrons": had_sum, "hazard_check_leptons": lep_check, "lambda_by_p_bin": lam_by_p,
            "eps_strip_pool": round(strip_pool, 6),
            "module_overrides": {"rule": f"n >= {MODULE_MIN_N}, miss - layer miss >= {MODULE_MIN_EXCESS}, "
                                         f"z >= {MODULE_MIN_Z}", "n_modules": int(len(mo)),
                                 "per_layer": {str(int(c)): int(n) for c, n in mo.groupby("code").size().items()},
                                 "rows": mo.round(5).to_dict(orient="records")},
            "tracks_matched_for_stops": n_matched, "tracks": int(len(P)),
            "unknown_fatras_hits": W["unknown_fatras_hits"],
            "reach_trackml": reach_fractions(tml["barrel_presence"], central, species),
            "reach_v15_no_emulation": reach_fractions(v15["barrel_presence"], central, species),
            "charge_per_hit_over_path": diag_q,
            "plan_comparison": plan_comparison(H, P, n_tracks, species),
        },
        "seconds": round(time.time() - t0, 1),
    }
    return out


def plan_comparison(H: pd.DataFrame, P: pd.DataFrame, n_tracks: int, species: np.ndarray) -> dict:
    """The pooled hole rates of field_material/track_stats.py on the same events (barrel |eta| < 0.8 and pixel disks
    1.5 < |eta| < 2.5, pT > 2 GeV), per species, for comparison with plan section 4.4."""
    t = P.track.to_numpy()
    pt, eta = np.zeros(n_tracks), np.zeros(n_tracks)
    pt[t], eta[t] = P.pt.to_numpy(), P.eta.to_numpy()
    out = {}
    bp = barrel_presence(H.track.to_numpy(), H.volume.to_numpy(), H.layer.to_numpy(), n_tracks)
    for i in range(len(SPECIES)):
        rows = np.flatnonzero((species == i) & (pt > 2) & (np.abs(eta) < 0.8) & bp[:, 0])
        ni, nh = interior_holes(bp, rows)
        r = {"barrel_tracks": int(len(rows)), "barrel_hole_rate": round(float(nh.sum() / max(ni.sum(), 1)), 5)}
        fw = (species == i) & (pt > 2) & (np.abs(eta) > 1.5) & (np.abs(eta) < 2.5)
        ni = nh = 0
        for vol in (7, 9):
            dp = disk_presence(H.track.to_numpy(), H.volume.to_numpy(), H.layer.to_numpy(), n_tracks, vol)
            a, b = interior_holes(dp, np.flatnonzero(fw))
            ni, nh = ni + a.sum(), nh + b.sum()
        r["pixel_disk_hole_rate"] = round(float(nh / max(ni, 1)), 5)
        out[SPECIES[i]] = r
    return out


# ------------------------------------------------------------------------------------------------ gate
def gate(work: str, tables_path: str, half: str | None = None, replicas: int = REPLICAS, seed: int = 20261014,
         fatras: str = "fatras_out") -> dict:
    """Step 3 (module docstring): the emulation on the Fatras hits of the selected particles against TrackML."""
    t0 = time.time()
    tab = Tables.load(tables_path)
    W = load_work(work, half, fatras)
    P, H, F = W["P"], W["H"], W["F"]
    n_tracks = int(max(P.track.max(), H.track.max(), F.track.max())) + 1
    central, forward = selections(P, n_tracks)
    species = np.full(n_tracks, -1, np.int64)
    species[P.track.to_numpy()] = species_index(P.pdg.to_numpy())
    tml = hole_rates(H.track.to_numpy(), H.volume.to_numpy(), H.layer.to_numpy(), n_tracks, central, forward)
    reach_t = reach_fractions(tml["barrel_presence"], central, species)
    v15 = hole_rates(F.track.to_numpy(), F.volume.to_numpy(), F.layer.to_numpy(), n_tracks, central, forward)
    parts = pd.DataFrame({"particle_id": P.track.to_numpy(), "pdg": P.pdg.to_numpy(), "p": P.p.to_numpy()})
    hits = pd.DataFrame({"particle_id": F.track.to_numpy(), "volume": F.volume.to_numpy(np.int64),
                         "layer": F.layer.to_numpy(np.int64), "order": F["index"].to_numpy(),
                         "module": F.module.to_numpy(np.int64),
                         "path_factor": path_factor(F.volume, F.tx, F.ty, F.tpx, F.tpy, F.tpz)})
    rng = np.random.default_rng(seed)
    acc = {}
    reach_acc = {}
    seq = crossing_sequence(F.track.to_numpy(), code_of(F.volume, F.layer), F["index"].to_numpy())
    miss_t = miss_rates(stop_table(seq, H.track.to_numpy(), code_of(H.volume, H.layer)))
    miss_e = None
    for r in range(replicas):
        em = emulate(hits, parts, tab, rng)
        k = em["keep"]
        if r == 0:
            miss_e = miss_rates(stop_table(seq, F.track.to_numpy()[k], code_of(F.volume, F.layer)[k]))
        e = hole_rates(F.track.to_numpy()[k], F.volume.to_numpy()[k], F.layer.to_numpy()[k], n_tracks, central,
                       forward)
        for c, (ni, nh) in e["holes"].items():
            a = acc.setdefault(c, [0, 0])
            a[0] += ni
            a[1] += nh
        for sp, v in reach_fractions(e["barrel_presence"], central, species).items():
            reach_acc.setdefault(sp, []).append(v["reach_last"])
    rows, fails = [], []
    for c in sorted(tml["holes"]):
        ni, nh = tml["holes"][c]
        ei, eh = acc[c]
        gi, gh = v15["holes"][c]
        vol = c // 100
        pending = vol in (12, 16) and not os.path.exists(os.path.join(work, fatras, "PATCHED"))
        rt = nh / ni if ni else np.nan
        re_ = eh / ei if ei else np.nan
        d = re_ - rt if ni and ei else np.nan
        sig = np.sqrt(max(rt * (1 - rt), 1e-12) / ni) if ni else np.nan
        ok = bool(np.isfinite(d) and abs(d) <= HOLE_TOL) if ni else None
        row = {"code": c, "tml_interior": ni, "tml_rate": rt, "v15_rate": gh / gi if gi else np.nan,
               "emulated_interior": ei / replicas, "emulated_rate": re_, "diff_points": d * 100 if np.isfinite(d) else
               None, "tml_stat_sigma_points": sig * 100 if np.isfinite(sig) else None, "pass": ok,
               "note": "unpatched -z strip disk (geometry differs from TrackML); decided on the patched build"
               if pending else ""}
        rows.append(row)
        if ni and not ok and not pending:
            fails.append(f"holes {c}")
    reach_rows = {}
    for sp in ("pi", "K", "p"):
        rt = reach_t[sp]["reach_last"]
        re_ = float(np.mean(reach_acc[sp]))
        ok = abs(re_ - rt) <= REACH_TOL
        reach_rows[sp] = {"tracks": reach_t[sp]["tracks"], "tml": rt, "emulated": re_,
                          "diff_points": (re_ - rt) * 100, "pass": ok,
                          "tml_stat_sigma_points": 100 * np.sqrt(rt * (1 - rt) / reach_t[sp]["tracks"])}
        if not ok:
            fails.append(f"reach {sp}")
    return {"gate": "emulation (plan section 10)", "made_utc": utc(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "tables": os.path.relpath(tables_path, REPO) if tables_path.startswith(REPO) else tables_path,
            "emulation_hash": emulation_hash(tables_path), "half": half or "all", "replicas": replicas, "seed": seed,
            "thresholds": {"hole_points": HOLE_TOL * 100, "reach_points": REACH_TOL * 100},
            "holes": rows, "reach_last_barrel_layer": reach_rows,
            "miss_rate_vs_fatras_descriptive": {str(int(c)): {"trackml": round(float(miss_t[c]), 6),
                                                              "emulated": round(float(miss_e.get(c, np.nan)), 6)}
                                                for c in miss_t.index},
            "reach_v15_no_emulation": reach_fractions(v15["barrel_presence"], central, species),
            "result": "pass" if not fails else "fail", "failures": fails, "seconds": round(time.time() - t0, 1)}


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def dump(obj: dict, path: str):
    with open(path, "w") as fh:
        json.dump(_clean(json.loads(json.dumps(obj, default=_json_default))), fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--work", required=True)
    s.add_argument("--events-per-pack", type=int, default=20)
    s.add_argument("--max-events", type=int, default=None)
    t = sub.add_parser("tables")
    t.add_argument("--work", required=True)
    t.add_argument("--half", choices=["even", "odd"], default=None)
    t.add_argument("--out", default=None)
    t.add_argument("--fatras", default="fatras_out", help="Fatras output directory inside --work")
    g = sub.add_parser("gate")
    g.add_argument("--work", required=True)
    g.add_argument("--tables", default=TABLES)
    g.add_argument("--half", choices=["even", "odd"], default=None)
    g.add_argument("--replicas", type=int, default=REPLICAS)
    g.add_argument("--out", required=True)
    g.add_argument("--fatras", default="fatras_out", help="Fatras output directory inside --work")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        r = select(a.work, a.events_per_pack, max_events=a.max_events)
        print(json.dumps({k: v for k, v in r.items() if k != "events"}, indent=1))
    elif a.cmd == "tables":
        out = a.out or (TABLES if a.half is None else os.path.join(a.work, f"tables_{a.half}.json"))
        r = measure(a.work, a.half, a.fatras)
        dump(r, out)
        print(json.dumps({"out": out, "eps": r["model"]["eps_per_hit"],
                          "hazard_summary": r["diagnostics"]["hazard_summary_hadrons"],
                          "lepton_check": r["diagnostics"]["hazard_check_leptons"],
                          "reach_trackml": {k: v["reach_last"] for k, v in r["diagnostics"]["reach_trackml"].items()},
                          "plan_comparison": r["diagnostics"]["plan_comparison"]}, indent=1, default=_json_default))
    else:
        r = gate(a.work, a.tables, a.half, a.replicas, fatras=a.fatras)
        dump(r, a.out)
        print(json.dumps({"result": r["result"], "failures": r["failures"],
                          "reach": r["reach_last_barrel_layer"]}, indent=1, default=_json_default))


if __name__ == "__main__":
    main()
