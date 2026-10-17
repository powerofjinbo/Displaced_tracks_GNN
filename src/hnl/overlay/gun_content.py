"""Content checks of a gun overlay set that addendum H adds to hnl.overlay.truth_checks.content
(docs/preregistration_pileup.md sections 4, 7 and 9.2, "Content (descriptive)"). Truth only: no model is run and no
model output is read. Inputs: the stored set (gun truth with sim_status, the gun rows of the stored tables), the work
directory of slurm/overlay/overlay_set.sbatch (gun events, packed Fatras hits and the simulated-particle list with the
final states), the module table and the field map of the chain.

1. Navigator losses (`loss_table`, `nav_loss_bins`). A gun particle that Fatras failed to simulate (sim_status
   "failed"; the v15 navigator's "Interpolation out of bounds") keeps its particle row without hits. Per set: their
   number, by component, and their share among the generator-level P_P-like particles (isolated, pT > 1 GeV,
   |eta| < 4, |d0| >= 10 mm or r_v >= 100 mm, from the initial state, d0 as the converter computes it), binned in r_v,
   signed eta, |d0|, |d0|/r_v, the direction at the vertex (inward: the transverse momentum points towards the z axis)
   and the bins of addendum G section 7 that exist without hits (|d0|, r_v, pT). The bound on Delta of section 7 is
   hnl.overlay.h1.nav_loss_bound.
2. Silent world exits (`silent_exits`). A simulated gun particle (no log line) whose final state lies outside the
   detector world, r > 1030 mm or |z| > 3002 mm: the boundary of scripts/acts/overlay_sim.py gates, which counted 36
   such particles in the gun probe of the patched build (results/pileup_hp/f1_containment_build-v15-zmirror.md). The
   final states are those hnl.overlay.fatras_io pack keeps beside the packed hits.
3. Premature ends and holes (`trajectory_misses`), by the method of ad94518 (hnl.truth.nav_losses, method A), with the
   field map in place of the constant field. Each simulated gun particle is propagated without material from its
   initial state through the RZ map (hnl.trackml.overlay_sim.rk4_tracks) until it leaves the world (r = 1028.538 mm,
   |z| = 3001 mm, hnl.truth.nav_losses) or reaches the loop protection of the v15 propagator: a 3D path of
   pi p / (0.2998 |B(vertex)|), half a transverse turn (results/generated_nav_losses_v1.md section 4: lost particles end
   with a transverse turn within 0.05 rad of pi). Its crossings of the sensitive layers of TrackML's detectors.csv
   (barrel: the cylinder at the median module radius; disk: the plane at the median module z) count where a module can
   record them: the straight line through the crossing along the track meets one of the K_MODULES nearest module planes
   of the layer inside that module's bounds, and the distance inside is the depth. A crossing is robust if its depth is
   at least COV_MARGIN_MM, it lies MARGIN_MM inside the layer's outer edges, a barrel crossing is not grazing (the local
   transverse circle reaches MARGIN_MM beyond the layer radius on both sides) and, on a path that the loop protection
   ends, it lies END_MARGIN_MM before the end. Each hit is matched to the crossing of its own layer nearest to it; the
   last hit is the matched hit furthest along the path. A premature end is at least one robust crossing after the last
   hit (for a particle without a matched hit: any robust crossing); a hole is a robust crossing before the last hit
   without a hit. Source "fatras": the Fatras hits before the emulation (navigation alone). Source "stored": the hits
   written in the overlay event, where the emulation's dropped hits and hadronic stops add to the misses.

    python -m hnl.overlay.gun_content run --set hval --work DIR --root ROOT --out-json J [--sources fatras,stored]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun as GUN

R_WORLD, Z_WORLD = 1028.538, 3001.0          # = hnl.truth.nav_losses.R_WORLD, Z_WORLD (path end of the trajectory)
EXIT_R_MM, EXIT_ABSZ_MM = 1030.0, 3002.0     # = hnl.trackml.overlay_sim.WORLD_R_MM, WORLD_ABSZ_MM (silent exits)
K_GEV_PER_T_M = 0.299792458
MARGIN_MM = 20.0                             # = hnl.truth.nav_losses.MARGIN_MM (method A)
END_MARGIN_MM = 10.0                         # = hnl.truth.nav_losses.END_MARGIN_MM
COV_MARGIN_MM = 3.0                          # = hnl.truth.nav_losses.COV_MARGIN_A_MM
K_MODULES = 8
STEP_MM, EVERY, PATH_MAX_MM, CHUNK = 1.0, 2, 6200.0, 1000
PP_D0_MM, PP_RV_MM = 10.0, 100.0
BARREL_VOLUMES = (8, 13, 17)
FIELD_MAP = "/work/submit/jinboz1/hnl/pileup_design/field_material/trackml_like_rz_map_v1.txt"
# bins (addendum G section 7: hnl.eval.addendum_g_summary D0_BINS, RV_BINS, PT_BINS, RATIO_BINS)
D0_BINS = [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]
RV_BINS = [0, 10, 30, 100, 200, 300, 450, 650, np.inf]
PT_BINS = [1, 2, 5, 10, 20, np.inf]
RATIO_BINS = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]
ETA_BINS = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]


# ------------------------------------------------------------------------------------------------ 1. navigator losses
def loss_table(gun_truth: pd.DataFrame) -> pd.DataFrame:
    """One row per gun particle of the gun truth table (hnl.overlay.writer.assemble: every gun particle with its
    sim_status) with its initial-state kinematics as the converter computes them (float32 values, d0 of the helix
    through the production vertex in 2 T about the z axis, r_v from the z axis). key = the row of the truth table:
    TrackML particle ids repeat across events, so (event, particle_id) or key name a particle."""
    from hnl.trackml.convert import d0_from_vertex
    t = gun_truth
    f = {k: t[k].to_numpy(np.float64).astype(np.float32).astype(np.float64) for k in ("vx", "vy", "vz", "px", "py",
                                                                                       "pz")}
    q = t.q.to_numpy(np.int64)
    pt = np.hypot(f["px"], f["py"])
    llp = (t.particle_id.to_numpy(np.int64) >> 52) - GUN.VERTEX0
    out = pd.DataFrame({
        "key": np.arange(len(t)), "event": t.event.to_numpy(np.int64), "particle_id": t.particle_id.to_numpy(np.int64),
        "llp": llp,
        "component": np.where(llp <= GUN.N_ISO, "iso", "dense"), "pdg": t.pdg.to_numpy(np.int64), "q": q,
        "pt": pt, "eta": np.arcsinh(f["pz"] / np.maximum(pt, 1e-12)), "p": np.sqrt(pt ** 2 + f["pz"] ** 2),
        "rv": np.hypot(f["vx"], f["vy"]), "d0": d0_from_vertex(f["vx"], f["vy"], f["px"], f["py"], q),
        "inward": f["vx"] * f["px"] + f["vy"] * f["py"] < 0, "sim_status": t.sim_status.to_numpy()})
    out["abs_d0"] = np.abs(out.d0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["d0_over_rv"] = np.where(out.rv > 0, out.abs_d0 / out.rv, np.nan)
    out["pp_like"] = ((out.llp <= GUN.N_ISO) & (out.pt > 1.0) & (out.eta.abs() < 4.0)
                      & ((out.abs_d0 >= PP_D0_MM) | (out.rv >= PP_RV_MM)))
    out["lost"] = out.sim_status == "failed"
    return out


def _binned(t: pd.DataFrame, col: str, edges) -> list[dict]:
    b = pd.cut(t[col], edges, right=False)
    g = t.groupby(b, observed=False)
    return [{"bin": str(k), "pp_like": int(len(x)), "lost": int(x.lost.sum()),
             "share": float(x.lost.mean()) if len(x) else None} for k, x in g]


def nav_loss_bins(t: pd.DataFrame) -> dict:
    """Section 7: the lost gun particles of a set and their share among the generator-level P_P-like particles, in
    bins. t: `loss_table`."""
    pp = t[t.pp_like]
    out = {"gun_particles": int(len(t)), "lost": int(t.lost.sum()),
           "lost_by_component": {c: int(g.lost.sum()) for c, g in t.groupby("component")},
           "below_pmin": int((t.sim_status == "below_pmin").sum()),
           "pp_like": int(len(pp)), "pp_like_lost": int(pp.lost.sum()),
           "pp_like_lost_share": float(pp.lost.mean()) if len(pp) else None,
           "bins": {"r_v [mm]": _binned(pp, "rv", RV_BINS), "eta": _binned(pp, "eta", ETA_BINS),
                    "|d0| [mm]": _binned(pp, "abs_d0", D0_BINS), "|d0| / r_v": _binned(pp, "d0_over_rv", RATIO_BINS),
                    "pT [GeV]": _binned(pp, "pt", PT_BINS),
                    "direction at the vertex": [{"bin": k, "pp_like": int(len(g)), "lost": int(g.lost.sum()),
                                                 "share": float(g.lost.mean()) if len(g) else None}
                                                for k, g in (("inward", pp[pp.inward]), ("outward", pp[~pp.inward]))]}}
    return out


# ------------------------------------------------------------------------------------------------ 2. silent exits
def outside_world(x, y, z) -> np.ndarray:
    return (np.hypot(x, y) > EXIT_R_MM) | (np.abs(z) > EXIT_ABSZ_MM)


def silent_exits(status: pd.DataFrame, sim_particles: pd.DataFrame) -> dict:
    """Simulated gun particles whose final state lies outside the world. status: hnl.overlay.fatras_io.
    simulation_status rows (event = Fatras event, particle_id = barcode, status); sim_particles: the packed list
    (`FIO.sim_particles_path`) with the final-state columns. Without final states (a list packed before they were
    kept) the count is None."""
    cols = [f"final_{k}" for k in FIO.FINAL_STATE]
    sim = status[status.status == "simulated"][["event", "particle_id"]]
    if not set(cols) <= set(sim_particles.columns) or sim_particles[cols[:3]].isna().all().all():
        return {"final_states": False, "simulated": int(len(sim)), "outside": None}
    f = sim_particles.astype({"event": np.int64, "particle_id": np.uint64})
    j = sim.astype({"event": np.int64, "particle_id": np.uint64}).merge(f, on=["event", "particle_id"], how="left",
                                                                       validate="one_to_one")
    no_final = j.final_vx.isna().to_numpy()
    out = outside_world(j.final_vx.to_numpy(), j.final_vy.to_numpy(), j.final_vz.to_numpy()) & ~no_final
    rows = j[out]
    return {"final_states": True, "simulated": int(len(j)), "without_final_state": int(no_final.sum()),
            "outside": int(out.sum()), "share": float(out.mean()) if len(j) else None,
            "particles": [{"fatras_event": int(r.event), "particle_id": int(GUN.trackml_particle_id(np.uint64(
                r.particle_id))), "r_end": float(np.hypot(r.final_vx, r.final_vy)), "z_end": float(r.final_vz)}
                for r in rows.itertuples()]}


# ------------------------------------------------------------------------------------------------ 3. geometry
@dataclass
class Layers:
    """Sensitive layers of a module table (hnl.overlay.digitise.Modules: TrackML's detectors.csv) and the coverage
    test (module docstring)."""
    table: pd.DataFrame
    per_layer: list

    @classmethod
    def from_modules(cls, m: D.Modules, n_edge: int = 33) -> "Layers":
        from scipy.spatial import cKDTree
        vol = m.key // 1_000_000
        lay = (m.key // 10_000) % 100
        # points along the four edges: (-minhu, -hv), (minhu, -hv), (maxhu, hv), (-maxhu, hv)
        cu = np.stack([-m.minhu, m.minhu, m.maxhu, -m.maxhu], 1)
        cv = np.stack([-m.hv, -m.hv, m.hv, m.hv], 1)
        t = np.linspace(0.0, 1.0, n_edge)
        eu = (cu[:, :, None] * (1 - t) + np.roll(cu, -1, 1)[:, :, None] * t).reshape(len(m.key), -1)
        ev = (cv[:, :, None] * (1 - t) + np.roll(cv, -1, 1)[:, :, None] * t).reshape(len(m.key), -1)
        g = m.c[:, None, :] + eu[..., None] * m.R[:, None, :, 0] + ev[..., None] * m.R[:, None, :, 1]
        gr, gz = np.hypot(g[..., 0], g[..., 1]), g[..., 2]
        rows, per = [], []
        for (v, ly) in sorted(set(zip(vol.tolist(), lay.tolist()))):
            s = np.flatnonzero((vol == v) & (lay == ly))
            barrel = v in BARREL_VOLUMES
            rows.append({"volume": int(v), "layer": int(ly), "barrel": barrel, "n_modules": int(len(s)),
                         "R": float(np.median(np.hypot(m.c[s, 0], m.c[s, 1]))) if barrel else np.nan,
                         "Z": np.nan if barrel else float(np.median(m.c[s, 2])),
                         "rlo": float(gr[s].min()), "rhi": float(gr[s].max()), "zlo": float(gz[s].min()),
                         "zhi": float(gz[s].max())})
            per.append((cKDTree(m.c[s]), m.c[s], m.R[s], m.minhu[s], m.maxhu[s], m.hv[s]))
        return cls(table=pd.DataFrame(rows), per_layer=per)

    def index(self, volume, layer) -> np.ndarray:
        """Layer index of (volume, layer); -1 if the table lacks it."""
        key = self.table.volume.to_numpy() * 100 + self.table.layer.to_numpy()
        lut = pd.Series(np.arange(len(key)), index=key)
        return lut.reindex(np.asarray(volume, np.int64) * 100 + np.asarray(layer, np.int64)).fillna(-1).to_numpy(
            np.int64)

    def depth(self, li: int, P: np.ndarray, Dir: np.ndarray) -> np.ndarray:
        """Largest distance inside the bounds (negative outside every module) where the straight line through P along
        Dir meets the planes of the K_MODULES nearest modules of layer li."""
        if not len(P):
            return np.zeros(0)
        tree, c, R, a, b, hv = self.per_layer[li]
        k = min(K_MODULES, len(c))
        _, idx = tree.query(P, k=k)
        idx = np.asarray(idx).reshape(len(P), k)
        Rm = R[idx]
        loc = np.einsum("mkij,mki->mkj", Rm, P[:, None, :] - c[idx])
        Dl = np.einsum("mkij,mi->mkj", Rm, Dir)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(Dl[..., 2]) > 1e-3, -loc[..., 2] / Dl[..., 2], np.nan)
        t = np.where(np.abs(t) < 50.0, t, np.nan)
        u, v = loc[..., 0] + t * Dl[..., 0], loc[..., 1] + t * Dl[..., 1]
        A, B, H = a[idx], b[idx], hv[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            hx = A + (B - A) * (v + H) / (2 * H)
            cosa = np.cos(np.arctan((B - A) / (2 * H)))
        dep = np.minimum(H - np.abs(v), (hx - np.abs(u)) * cosa)
        return np.max(np.where(np.isnan(dep), -np.inf, dep), axis=1)


# ------------------------------------------------------------------------------------------------ 3. trajectories
def _bvec(field, x: np.ndarray) -> np.ndarray:
    r = np.hypot(x[:, 0], x[:, 1])
    br, bz = field(r, x[:, 2])
    rs = np.maximum(r, 1e-9)
    return np.stack([br * x[:, 0] / rs, br * x[:, 1] / rs, bz], 1)


def loop_limit(x0: np.ndarray, p0: np.ndarray, field) -> np.ndarray:
    """3D path [mm] at which the v15 loop protection ends a particle: pi p / (0.2998 |B(start)|)."""
    b = np.linalg.norm(_bvec(field, np.asarray(x0, np.float64)), axis=1)
    p = np.linalg.norm(p0, axis=1)
    with np.errstate(divide="ignore"):
        return np.where(b > 0, np.pi * p / (K_GEV_PER_T_M * b) * 1000.0, np.inf)


def crossings(x0, p0, q, field, layers: Layers, step_mm: float = STEP_MM, every: int = EVERY,
              path_mm: float = PATH_MAX_MM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer crossings of material-free trajectories (module docstring): one row per crossing (track, li, s [mm of 3D
    path], x, y, z, acc, robust) and one row per track (s_hi, by_loop)."""
    from hnl.trackml.overlay_sim import rk4_tracks
    x0, p0, q = np.asarray(x0, np.float64), np.asarray(p0, np.float64), np.asarray(q, np.float64)
    n = len(x0)
    X, Pm, _ = rk4_tracks(x0, p0, q, field, step_mm=step_mm, nsteps=int(np.ceil(path_mm / step_mm)), every=every)
    s = np.arange(X.shape[1]) * step_mm * every
    r, z = np.hypot(X[..., 0], X[..., 1]), X[..., 2]
    inside = np.cumprod((r < R_WORLD) & (np.abs(z) < Z_WORLD), axis=1).astype(bool)
    n_in = inside.sum(1)
    s_world = np.where(n_in < X.shape[1], s[np.minimum(n_in, X.shape[1] - 1)], s[-1])
    s_loop = loop_limit(x0, p0, field)
    by_loop = s_loop < s_world
    s_hi = np.minimum(s_world, s_loop)
    seg = inside[:, :-1]
    L = layers.table
    rows = []
    for li in range(len(L)):
        barrel = bool(L.barrel.iloc[li])
        f = (r - L.R.iloc[li]) if barrel else (z - L.Z.iloc[li])
        sc = ((f[:, :-1] < 0) != (f[:, 1:] < 0)) & seg
        ti, k = np.nonzero(sc)
        if not len(ti):
            continue
        a = f[ti, k] / (f[ti, k] - f[ti, k + 1])
        P = X[ti, k] + a[:, None] * (X[ti, k + 1] - X[ti, k])
        Dm = Pm[ti, k] + a[:, None] * (Pm[ti, k + 1] - Pm[ti, k])
        sc_s = s[k] + a * (s[k + 1] - s[k])
        keep = sc_s <= s_hi[ti]
        ti, P, Dm, sc_s = ti[keep], P[keep], Dm[keep], sc_s[keep]
        Pr = np.hypot(P[:, 0], P[:, 1])
        if barrel:
            edges = (P[:, 2] >= L.zlo.iloc[li]) & (P[:, 2] <= L.zhi.iloc[li])
            inner = (P[:, 2] >= L.zlo.iloc[li] + MARGIN_MM) & (P[:, 2] <= L.zhi.iloc[li] - MARGIN_MM)
            # grazing: the local transverse circle must reach MARGIN_MM beyond the layer radius on both sides
            bz = field(Pr, P[:, 2])[1]
            pt = np.hypot(Dm[:, 0], Dm[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                Rt = np.where(np.abs(bz) > 0, pt / (K_GEV_PER_T_M * np.abs(bz)) * 1000.0, np.inf)
                h = np.sign(q[ti] * bz)
                phi = np.arctan2(Dm[:, 1], Dm[:, 0])
                xc, yc = P[:, 0] + h * Rt * np.sin(phi), P[:, 1] - h * Rt * np.cos(phi)
                d = np.hypot(xc, yc)
                graze_ok = np.where(np.isfinite(Rt), ((d + Rt) - L.R.iloc[li] >= MARGIN_MM)
                                    & (L.R.iloc[li] - np.abs(d - Rt) >= MARGIN_MM), True)
            inner &= graze_ok
        else:
            edges = (Pr >= L.rlo.iloc[li]) & (Pr <= L.rhi.iloc[li])
            inner = (Pr >= L.rlo.iloc[li] + MARGIN_MM) & (Pr <= L.rhi.iloc[li] - MARGIN_MM)
        dep = np.full(len(ti), -np.inf)
        if edges.any():
            Du = Dm[edges] / np.linalg.norm(Dm[edges], axis=1, keepdims=True)
            dep[edges] = layers.depth(li, P[edges], Du)
        acc = edges & (dep >= 0)
        end_ok = sc_s <= s_hi[ti] - np.where(by_loop[ti], END_MARGIN_MM, 0.0)
        rows.append(pd.DataFrame({"track": ti, "li": li, "s": sc_s, "x": P[:, 0], "y": P[:, 1], "z": P[:, 2],
                                  "depth": dep, "acc": acc, "robust": acc & inner & (dep >= COV_MARGIN_MM) & end_ok}))
    cr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        {"track": np.zeros(0, np.int64), "li": np.zeros(0, np.int64), "s": [], "x": [], "y": [], "z": [], "depth": [],
         "acc": np.zeros(0, bool), "robust": np.zeros(0, bool)})
    cr = cr.sort_values(["track", "s"], kind="stable").reset_index(drop=True)
    return cr, pd.DataFrame({"track": np.arange(n), "s_hi": s_hi, "by_loop": by_loop})


def match_hits(cr: pd.DataFrame, hits: pd.DataFrame, n_tracks: int) -> pd.DataFrame:
    """Per track: hits matched to the nearest crossing of their own layer, the last matched hit, and the robust
    crossings after it (premature end) and before it without a hit (holes). hits: track, li, x, y, z."""
    cr = cr.assign(cid=np.arange(len(cr)))
    h = hits.reset_index(drop=True).assign(hid=np.arange(len(hits)))
    j = h.merge(cr[["track", "li", "cid", "s", "x", "y", "z"]], on=["track", "li"], suffixes=("", "_c"))
    j["d"] = np.sqrt((j.x - j.x_c) ** 2 + (j.y - j.y_c) ** 2 + (j.z - j.z_c) ** 2)
    best = j.loc[j.groupby("hid").d.idxmin()] if len(j) else j
    on = np.bincount(best.cid.to_numpy(np.int64), minlength=len(cr))
    s_last = best.groupby("track").s.max().reindex(np.arange(n_tracks)).fillna(0.0).to_numpy()
    after = cr.s.to_numpy() > s_last[cr.track.to_numpy()] + 1e-6
    rob = cr.robust.to_numpy(bool)
    first_missed = cr[rob & after].groupby("track").li.first()
    t = np.arange(n_tracks)
    cnt = lambda m: np.bincount(cr.track.to_numpy()[m], minlength=n_tracks)      # noqa: E731
    return pd.DataFrame({
        "track": t, "n_hits": np.bincount(h.track.to_numpy(np.int64), minlength=n_tracks),
        "n_matched": np.bincount(best.track.to_numpy(np.int64), minlength=n_tracks),
        "max_match_mm": best.groupby("track").d.max().reindex(t).to_numpy(),
        "n_robust": cnt(rob), "n_acc": cnt(cr.acc.to_numpy(bool)), "n_after": cnt(rob & after),
        "n_before": cnt(rob & ~after), "n_holes": cnt(rob & ~after & (on == 0)),
        "first_missed_li": first_missed.reindex(t).fillna(-1).to_numpy(np.int64)})


def trajectory_misses(init: pd.DataFrame, hits: pd.DataFrame, field, layers: Layers,
                      chunk: int = CHUNK) -> pd.DataFrame:
    """Premature ends and holes of every particle of `init` (key, x, y, z, px, py, pz, q) against its hits (key,
    volume, layer, x, y, z). Returns one row per particle (`match_hits` columns, by_loop, the flags and the layer of the
    first missed crossing)."""
    init = init.reset_index(drop=True)
    li = layers.index(hits.volume.to_numpy(), hits.layer.to_numpy())
    if (li < 0).any():
        raise ValueError(f"{int((li < 0).sum())} hits on layers missing from the module table")
    pos = pd.Series(np.arange(len(init)), index=init.key.to_numpy())
    if pos.index.duplicated().any():
        raise ValueError("duplicate particle key")
    ht = pos.reindex(hits.key.to_numpy()).to_numpy()
    if np.isnan(ht).any():
        raise KeyError(f"{int(np.isnan(ht).sum())} hits of particles without an initial state")
    hits = pd.DataFrame({"track": ht.astype(np.int64), "li": li, "x": hits.x.to_numpy(np.float64),
                         "y": hits.y.to_numpy(np.float64), "z": hits.z.to_numpy(np.float64)})
    out = []
    for a in range(0, len(init), chunk):
        sl = init.iloc[a:a + chunk]
        cr, tr = crossings(sl[["x", "y", "z"]].to_numpy(), sl[["px", "py", "pz"]].to_numpy(), sl.q.to_numpy(), field,
                           layers)
        hh = hits[(hits.track >= a) & (hits.track < a + len(sl))].assign(track=lambda d: d.track - a)
        m = match_hits(cr, hh, len(sl)).merge(tr, on="track")
        m["track"] += a
        out.append(m)
    r = pd.concat(out, ignore_index=True)
    r.insert(0, "key", init.key.to_numpy())
    r["premature"] = r.n_after >= 1
    r["premature_2"] = r.n_after >= 2
    r["hole"] = r.n_holes >= 1
    L = layers.table
    r["first_missed_layer"] = np.where(r.first_missed_li >= 0, (L.volume.to_numpy() * 100 + L.layer.to_numpy())[
        np.maximum(r.first_missed_li.to_numpy(), 0)], -1)
    return r


def summarise_misses(r: pd.DataFrame, info: pd.DataFrame) -> list[dict]:
    """Shares per component, for all simulated gun particles, those with pT > 1 GeV and the P_P-like ones; info: the
    `loss_table` rows of the same keys."""
    j = r.merge(info, on="key", how="left", validate="one_to_one")
    out = []
    groups = [("all", j), ("iso", j[j.component == "iso"]), ("dense", j[j.component == "dense"]),
              ("pT > 1 GeV", j[j.pt > 1.0]), ("P_P-like", j[j.pp_like])]
    for name, g in groups:
        wh = g[g.n_hits > 0]
        would = (g.pt > 1.0) & (g.eta.abs() < 4.0) & (g.n_hits < 5) & (g.n_hits + g.n_after + g.n_holes >= 5)
        out.append({"group": name, "particles": int(len(g)), "with_hits": int(len(wh)),
                    "premature": int(g.premature.sum()), "premature_share": float(g.premature.mean()) if len(g) else
                    None, "premature_2": int(g.premature_2.sum()), "with_hole": int(g.hole.sum()),
                    "hole_share": float(g.hole.mean()) if len(g) else None,
                    "holes_per_crossing_before_last_hit": float(g.n_holes.sum() / max(g.n_before.sum(), 1)),
                    "by_loop": int(g.by_loop.sum()), "unmatched_hits": int((g.n_hits - g.n_matched).sum()),
                    "median_max_match_mm": float(np.nanmedian(wh.max_match_mm)) if len(wh) else None,
                    "below_5_hits_by_misses": int(would.sum())})
    return out


# ------------------------------------------------------------------------------------------------ driver
def _gun_hits_stored(root: str, set_name: str, events) -> pd.DataFrame:
    from hnl.trackml.convert import read_table
    out = []
    ev = sorted(int(e) for e in events)
    for a in range(0, len(ev), 20):
        h = read_table("hits", ev[a:a + 20], ["event", "x", "y", "z", "volume", "layer", "pid"], root=root)
        v = h.pid.to_numpy(np.int64) >> 52
        out.append(h[(v >= GUN.VERTEX0 + 1) & (v <= GUN.VERTEX0 + GUN.N_ISO + GUN.N_DENSE)])
    return pd.concat(out, ignore_index=True)


def run(set_name: str, work: str, root: str, sources=("fatras", "stored"), field_map: str = FIELD_MAP,
        lookup_path: str | None = None) -> dict:
    from hnl.trackml.overlay_sim import NodeField
    t0 = time.time()
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    gdir = os.path.join(work, "gun")
    with open(os.path.join(gdir, "gun_events.json")) as fh:
        gev = json.load(fh)
    truth = pd.read_parquet(os.path.join(root, f"gun_truth_{set_name}.parquet")).reset_index(drop=True)
    sim_path = os.path.join(gdir, "fatras_out", "simhits.parquet")
    lt = loss_table(truth)
    out = {"set": set_name, "root": root, "work": work, "events": int(truth.event.nunique()),
           "nav_losses": nav_loss_bins(lt)}
    from hnl.overlay import writer as W
    status = W.sim_status(sim_path, gdir, gev, float(meta["sim"]["pmin_gev"]))
    simp = pd.read_parquet(FIO.sim_particles_path(sim_path))
    out["silent_exits"] = silent_exits(status, simp)
    field = NodeField.from_rz_map(field_map)
    layers = Layers.from_modules(D.Modules.load())
    sim_t = truth[truth.sim_status == "simulated"]
    init = pd.DataFrame({"key": sim_t.index.to_numpy(np.int64), **{k: sim_t[k].to_numpy(np.float64) for k in (
        "vx", "vy", "vz", "px", "py", "pz")}, "q": sim_t.q.to_numpy(np.float64)}).rename(
        columns={"vx": "x", "vy": "y", "vz": "z"})
    out["misses"], out["field_map"] = {}, field_map
    for src in sources:
        if src == "fatras":
            lk = D.load_lookup(lookup_path or meta["lookup"])
            sim = pd.read_parquet(sim_path, columns=["event", "particle_id", "geometry_id", "tx", "ty", "tz"])
            vol, lay, mod = lk.from_geometry_id(sim.geometry_id.to_numpy(np.uint64))
            key = pd.Series(np.arange(len(truth)), index=pd.MultiIndex.from_arrays(
                [truth.fatras_event.to_numpy(np.int64), truth.acts_barcode.to_numpy(np.uint64)]))
            pid = key.reindex(pd.MultiIndex.from_arrays([sim.event.to_numpy(np.int64),
                                                         sim.particle_id.to_numpy(np.uint64)])).to_numpy()
            if np.isnan(pid.astype(np.float64)).any() or (mod < 0).any():
                raise RuntimeError("Fatras hits of unknown particles or modules")
            hits = pd.DataFrame({"key": pid.astype(np.int64), "volume": vol, "layer": lay,
                                 "x": sim.tx.to_numpy(np.float64), "y": sim.ty.to_numpy(np.float64),
                                 "z": sim.tz.to_numpy(np.float64)})
        elif src == "stored":
            h = _gun_hits_stored(root, set_name, truth.event.unique())
            kmap = pd.Series(np.arange(len(truth)), index=pd.MultiIndex.from_arrays(
                [truth.event.to_numpy(np.int64), truth.particle_id.to_numpy(np.int64)]))
            k = kmap.reindex(pd.MultiIndex.from_arrays([h.event.to_numpy(np.int64), h.pid.to_numpy(np.int64)]))
            if k.isna().any():
                raise RuntimeError("stored gun hits of particles missing from the gun truth table")
            hits = pd.DataFrame({"key": k.to_numpy(np.int64), "volume": h.volume.to_numpy(np.int64),
                                 "layer": h.layer.to_numpy(np.int64), "x": h.x.to_numpy(np.float64),
                                 "y": h.y.to_numpy(np.float64), "z": h.z.to_numpy(np.float64)})
        else:
            raise ValueError(f"unknown source {src!r}")
        lost_hits = hits.key.isin(np.flatnonzero(truth.sim_status.to_numpy() != "simulated"))
        if lost_hits.any():
            raise RuntimeError(f"{src}: hits of gun particles that Fatras did not simulate")
        r = trajectory_misses(init, hits, field, layers)
        out["misses"][src] = summarise_misses(r, lt)
        print(f"[gun content] {src}: {len(r)} particles ({time.time() - t0:.0f} s)", flush=True)
    out["settings"] = {"margin_mm": MARGIN_MM, "end_margin_mm": END_MARGIN_MM, "cov_margin_mm": COV_MARGIN_MM,
                       "k_modules": K_MODULES, "step_mm": STEP_MM, "sample_every": EVERY, "path_max_mm": PATH_MAX_MM,
                       "world_path_end": [R_WORLD, Z_WORLD], "world_silent_exit": [EXIT_R_MM, EXIT_ABSZ_MM]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--root", required=True)
    r.add_argument("--out-json", required=True)
    r.add_argument("--sources", default="fatras,stored")
    r.add_argument("--field-map", default=FIELD_MAP)
    r.add_argument("--lookup", default=None)
    a = ap.parse_args(argv)
    out = run(a.set, a.work, a.root, tuple(a.sources.split(",")), a.field_map, a.lookup)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({k: out[k] for k in ("set", "events", "silent_exits")}, default=str)[:2000])


if __name__ == "__main__":
    main()
