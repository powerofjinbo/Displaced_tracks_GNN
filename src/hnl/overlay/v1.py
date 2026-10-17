"""Gates V1 and V1-Delta of addendum H (docs/preregistration_pileup.md section 9.2): the frozen Canto-L (TrackML) on the
closure set (k = 1) against the stored val outputs of the same natives. The closure set holds no gun particle, so V1
and V1-Delta read it in full. This module evaluates outputs it is given; it runs no model. Its tests use truth-made
placeholder outputs only.

Pairs (`truth_pairs`). One row per replaced native of the closure set (hnl.overlay.closure: closure id = native id +
2^62), with the native's population, region, kinematics and r_first from the closure truth table, the target flag and
hits of its re-simulated version from the stored closure rows, the native's first hit (rank 0 in tp_v1) and the
regions of its hits, the closure track's first hit, and:
- produced at a sensor (`at_sensor`): the native's first hit lies within SENSOR_TOL_MM = 1 mm of its production
  vertex measured exactly along the normal of the hit's module, |(x_hit - v) . w|, w the module's local w axis in
  TrackML's detectors.csv (R = [u v w]). Fatras starts a re-simulated particle at the vertex and records no hit there.
  The radial distance |r_hit - r_v| is reported beside it.
A pair enters V1 when both versions are targets. Outputs (`attach_outputs`): (event, particle_id, dm) tables at WP-A of
the closure run (overlay ids, closure ids) and of the stored val run (host ids, native ids;
runs/canto_l_v1/seed0_x1/grid/{full,1gev}/per_target.parquet), in C-full and C-1GeV.

Populations and rules (`v1`):
  a  gen-0, |d0| >= 10 mm or r_v >= 100 mm             |Delta| <= 1.0 point in all, <= 1.5 in each region      decisive
  b  gen >= 1, |d0| >= 10 mm, without the natives       |Delta| <= max(3 points, 2 sigma_paired) in all         flag
     produced at a sensor (b_all and b_at_sensor reported)
  c  the prompt subset; the rule uses |eta| < 3         |Delta| <= max(0.2 points, 2 sigma_paired)              decisive
     (all members reported)
  d  all generations, |d0| >= 10 mm and                 |Delta| <= max(3 points, 2 sigma_paired) in all         flag
     |d0| / r_first >= 0.2 (members of a or b)
Delta = eps(closure) - eps(native) [points] on the pairs, DM at WP-A in C-full. sigma_paired: the paired event
bootstrap over the closure events (hnl.overlay.hstats, B = 500, default_rng(12345), ddof = 1). Reported with it: the
unconditional V1 (every native; a re-simulated version that is no target counts as not found), the share of natives
whose re-simulated version has fewer than 5 hits, the first-hit layer agreement, and without a rule V1 in |eta| bins,
by detector region (a pair counts in every region holding one of its native hits), in r_first and |d0|/r_first bins
(a, b, d), and gun-weighted (a, b: reweighted to the H-val gun's P_P distribution in the cells of section 6).

V1-Delta (`v1_delta`): on the pairs of V1, D = Delta_closure - Delta_native with Delta = eps(C-full) - eps(C-1GeV);
rule on (a): |D| <= max(0.5 points, 2 sigma_paired), decisive; b, c, d reported.

Grids read (`closure_grid`; refused otherwise, before any pair is formed). `run`: first a grant (hnl.overlay.grant,
kind eval) for the closure events in each condition given, whose out holds the output JSON (slurm/overlay/v1.sbatch
asks slurm/overlay/guard_h.sh for them); then every closure grid passes hnl.overlay.tracks.grid_provenance, the check
that hnl.overlay.h1 also makes on H-val and H-test: hnl.overlay.tracks.grid_record for the set closure in the
condition it is given as (every closure event, no placeholder), its scores carry the frozen classifier's sha256 of
selection_v1.json (every block) and were made in that condition, its learned merge model is the one selection_v1.json
names, its events are those of the closure tables read and it was built on those tables (the production overlay root;
k = 1 is never read from elsewhere). `dry-run`: the pilot closure dry run only
(`pilot_closure_dry`: a scratch root, never the production overlay root; every event a closure id of a pilot host; the
set made with the pilot base seed). It writes the truth-made placeholder outputs of the closure side as grids in the
layout of hnl.overlay.tracks, marked placeholder (`write_placeholder_grid`, never under the production runs root), and
reads them back through the same check, which then takes placeholder grids only; the native side stays truth-made in
memory, so no stored model output is read.

    python -m hnl.overlay.v1 run --root ROOT --closure-full DIR --closure-1gev DIR --out-json J [--gun-cells PQ]
    python -m hnl.overlay.v1 dry-run --root ROOT --out-json J [--grids DIR]     (truth-made placeholders; code test only)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import closure as CL
from hnl.overlay import digitise as D
from hnl.overlay import hstats as HS
from hnl.overlay import truth_checks as TC
from hnl.overlay import writer as W
from hnl.trackml import throughput as T

TP = T.DERIVED_TP
SET = "closure"
SENSOR_TOL_MM = 1.0
D0_CUT_MM, RATIO_CUT = 10.0, 0.2
RULES = {"a": {"kind": "decisive", "all": 1.0, "region": 1.5},
         "b": {"kind": "flag", "floor": 3.0},
         "c": {"kind": "decisive", "floor": 0.2, "abs_eta_max": 3.0},
         "d": {"kind": "flag", "floor": 3.0}}
V1D_FLOOR = 0.5
REGIONS = CL.REGIONS                                   # central, fwd+, fwd-
DETECTOR_REGIONS = ("pixel barrel", "pixel endcap", "short-strip barrel", "short-strip endcap", "long-strip barrel",
                    "long-strip endcap")
ABS_ETA_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
RFIRST_BINS = [0, 200, 700, np.inf]                    # pixel, short strip, long strip (section 6 cells)
RATIO_CELLS = [0, 0.05, 0.2, np.inf]
VAL_RUN = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1"
# stored val outputs named by the addendum (sha256 prefixes of section 9.2; the diagnostic tables are listed in full
# in results/pileup_hp/addendum_h_inputs/val_diag_sha256.txt)
NATIVE_OUTPUTS = {"full": ("grid/full/per_target.parquet", "622e0d7f"), "1gev": ("grid/1gev/per_target.parquet",
                                                                                  "df64d65b")}
VAL_DIAG_SHA = os.path.join(W.REPO, "results", "pileup_hp", "addendum_h_inputs", "val_diag_sha256.txt")
SELECTION = os.path.join(W.REPO, "configs", "trackml", "selection_v1.json")   # = hnl.overlay.blind.SELECTION
CLOSURE_CONDITIONS = ("full", "1gev")
DRY_GRIDS = "v1_dry_grids"


# ------------------------------------------------------------------------------------------------ truth
def at_sensor(xyz, vertex, volume, layer, module, modules: D.Modules, tol: float = SENSOR_TOL_MM) -> tuple[
        np.ndarray, np.ndarray]:
    """(produced at a sensor, distance along the module normal [mm]) of first hits xyz (n, 3) and production vertices
    (n, 3): |(x_hit - v) . w| <= tol with w the normal of the hit's module."""
    row = modules.index(volume, layer, module)
    if (row < 0).any():
        raise KeyError(f"{int((row < 0).sum())} first hits on modules missing from detectors.csv")
    w = modules.R[row][:, :, 2]
    d = np.abs(np.einsum("ni,ni->n", np.asarray(xyz, np.float64) - np.asarray(vertex, np.float64), w))
    return d <= tol, d


def pair_rows(truth: pd.DataFrame, closure_particles: pd.DataFrame, native_hits: pd.DataFrame,
              closure_hits: pd.DataFrame, modules: D.Modules) -> pd.DataFrame:
    """The pair table of one or more closure events from their tables (module docstring). truth: closure truth rows;
    closure_particles: stored closure rows (event, particle_id, is_target, n_hits); native_hits: the host hits of the
    natives (event = overlay id, pid = native id, rank, x, y, z, volume, layer, module); closure_hits: the stored closure
    hits (event, pid = closure id, rank, volume, layer)."""
    t = truth.reset_index(drop=True)
    nid = t.native_particle_id.to_numpy(np.int64)
    out = pd.DataFrame({
        "event": t.event.to_numpy(np.int64), "host": t.host.to_numpy(np.int64), "native_id": nid,
        "closure_id": t.particle_id.to_numpy(np.int64), "population": t.population.to_numpy(),
        "region": t.region.to_numpy(), "gen": CL.generation(nid), "eta": t.eta.to_numpy(np.float64),
        "abs_eta": np.abs(t.eta.to_numpy(np.float64)), "abs_d0": np.abs(t.d0.to_numpy(np.float64)),
        "rv": t.rv.to_numpy(np.float64), "r_first": t.native_r_first.to_numpy(np.float64),
        "native_n_hits": t.native_n_hits.to_numpy(np.int64), "sim_status": t.get("sim_status", pd.Series(
            ["simulated"] * len(t))).to_numpy()})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ratio"] = out.abs_d0 / out.r_first
    if not np.array_equal(out.closure_id.to_numpy(), CL.closure_particle_id(nid)):
        raise ValueError("closure truth: closure id != native id + 2^62")
    cp = closure_particles.set_index(["event", "particle_id"])
    k = pd.MultiIndex.from_arrays([out.event, out.closure_id])
    if not k.isin(cp.index).all():
        raise LookupError("closure truth rows without a stored closure row")
    out["closure_target"] = cp.is_target.reindex(k).to_numpy(bool)
    out["closure_n_hits"] = cp.n_hits.reindex(k).to_numpy(np.int64)
    # native first hit, its distance from the vertex along the module normal, and the regions of the native hits
    nh = native_hits
    f = nh[nh["rank"] == 0].drop_duplicates(["event", "pid"]).set_index(["event", "pid"])
    kn = pd.MultiIndex.from_arrays([out.event, out.native_id])
    if not kn.isin(f.index).all():
        raise LookupError("natives without a rank-0 hit")
    f = f.reindex(kn)
    xyz = f[["x", "y", "z"]].to_numpy(np.float64)
    vtx = t[["vx", "vy", "vz"]].to_numpy(np.float32).astype(np.float64)
    out["at_sensor"], out["d_normal_mm"] = at_sensor(xyz, vtx, f.volume.to_numpy(), f.layer.to_numpy(),
                                                     f.module.to_numpy(), modules)
    out["d_radial_mm"] = np.abs(np.hypot(xyz[:, 0], xyz[:, 1]) - np.hypot(vtx[:, 0], vtx[:, 1]))
    out["first_layer_native"] = f.volume.to_numpy(np.int64) * 1000 + f.layer.to_numpy(np.int64)
    reg = nh.assign(reg=TC.region(nh.volume)).groupby(["event", "pid"]).reg.agg(lambda s: set(s))
    regs = reg.reindex(kn)
    for r in DETECTOR_REGIONS:
        out[f"in_{r}"] = [isinstance(s, set) and r in s for s in regs]
    ch = closure_hits[closure_hits["rank"] == 0].drop_duplicates(["event", "pid"]).set_index(["event", "pid"])
    kc = pd.MultiIndex.from_arrays([out.event, out.closure_id])
    fc = ch.reindex(kc)
    out["first_layer_closure"] = np.where(fc.volume.isna(), -1, fc.volume.fillna(0).to_numpy(np.int64) * 1000
                                          + fc.layer.fillna(0).to_numpy(np.int64))
    out["first_layer_same"] = out.first_layer_closure == out.first_layer_native
    # population memberships
    out["in_a"] = out.population == "a"
    out["in_b_all"] = out.population == "b"
    out["in_b"] = out.in_b_all & ~out.at_sensor
    out["in_b_at_sensor"] = out.in_b_all & out.at_sensor
    out["in_c_all"] = out.population == "c"
    out["in_c"] = out.in_c_all & (out.abs_eta < RULES["c"]["abs_eta_max"])
    out["in_d"] = (out.in_a | out.in_b_all) & (out.abs_d0 >= D0_CUT_MM) & (out.ratio >= RATIO_CUT)
    return out


def truth_pairs(root: str, set_name: str = SET, host_root: str = TP, events=None) -> pd.DataFrame:
    """`pair_rows` of every event of a stored closure set (module docstring), read from the set's closure truth and
    sidecar, its stored closure rows and the hosts' stored rows."""
    truth = pd.read_parquet(os.path.join(root, f"closure_truth_{set_name}.parquet"))
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        side = {int(r["overlay_event"]): r for r in json.load(fh)["events"]}
    ids = sorted(side) if events is None else sorted(int(e) for e in events)
    mods = D.Modules.load()
    out = []
    for e in ids:
        s = side[e]
        st = W.host_tables(e, root)
        nc, nh = int(s["n_closure_particles"]), int(s["n_closure_hits"])
        cp = st["particles"].slice(st["particles"].num_rows - nc).to_pandas()[["particle_id", "is_target", "n_hits"]]
        chits = st["hits"].slice(st["hits"].num_rows - nh).to_pandas()[["pid", "rank", "volume", "layer"]]
        host = W.host_tables(s["host"], host_root)["hits"].to_pandas()
        tr = truth[truth.event == e]
        nat = host[host.pid.isin(tr.native_particle_id)][["pid", "rank", "x", "y", "z", "volume", "layer", "module"]]
        out.append(pair_rows(tr, cp.assign(event=e), nat.assign(event=e), chits.assign(event=e), mods))
    return pd.concat(out, ignore_index=True)


def attach_outputs(pairs: pd.DataFrame, outputs: dict) -> pd.DataFrame:
    """dm of both versions in each condition given: outputs {(side, condition): (event, particle_id, dm)} with side
    'closure' (overlay ids, closure ids) or 'native' (host ids, native ids). Every native and every closure version
    that is a target must have a row."""
    p = pairs.copy()
    for (side, cond), tab in outputs.items():
        if side == "closure":
            p[f"dm_closure_{cond}"] = HS.attach_dm(p, tab, "event", "closure_id", f"closure {cond}", p.closure_target)
        elif side == "native":
            p[f"dm_native_{cond}"] = HS.attach_dm(p, tab, "host", "native_id", f"native {cond}", np.ones(len(p), bool))
        else:
            raise ValueError(f"unknown side {side!r}")
    return p


# ------------------------------------------------------------------------------------------------ statistics
def _counts(p: pd.DataFrame, events, cols) -> np.ndarray:
    """Per event (in the given order) the number of rows and the sums of the given bool columns."""
    ev = pd.Index(np.asarray(events, np.int64))
    pos = ev.get_indexer(p.event.to_numpy(np.int64))
    if (pos < 0).any():
        raise ValueError("pairs of events outside the event list")
    out = [np.bincount(pos, minlength=len(ev))]
    out += [np.bincount(pos, weights=p[c].to_numpy(bool).astype(np.float64), minlength=len(ev)) for c in cols]
    return np.stack(out, 1)


def delta_pairs(p: pd.DataFrame, events, cond: str = "full") -> dict:
    """Delta = eps(closure) - eps(native) on the pairs p (both targets), with its paired bootstrap sigma."""
    c = _counts(p, events, [f"dm_closure_{cond}", f"dm_native_{cond}"])
    r = HS.paired_delta(c[:, 0], c[:, 1], c[:, 2])
    return {"pairs": r["n"], "eps_closure": r["eps_a"], "eps_native": r["eps_b"], "delta": r["delta"],
            "sigma_paired": r["sigma"]}


def unconditional(m: pd.DataFrame, events, cond: str = "full") -> dict:
    """Every native of the population; a re-simulated version that is no target counts as not found."""
    q = m.assign(found=m.closure_target & m[f"dm_closure_{cond}"])
    c = _counts(q, events, ["found", f"dm_native_{cond}"])
    r = HS.paired_delta(c[:, 0], c[:, 1], c[:, 2])
    return {"natives": r["n"], "eps_closure": r["eps_a"], "eps_native": r["eps_b"], "delta": r["delta"],
            "sigma_paired": r["sigma"]}


def _rule(pop: str, d: dict, regions: dict | None) -> dict:
    rule = RULES[pop]
    if not d["pairs"] or not np.isfinite(d["delta"]):
        return {"limit": None, "pass": False, "reason": "no pairs"}
    if pop == "a":
        reg_ok = {r: bool(x["pairs"] and abs(x["delta"]) <= rule["region"]) for r, x in regions.items()}
        ok = abs(d["delta"]) <= rule["all"] and all(reg_ok.values())
        return {"limit": rule["all"], "limit_region": rule["region"], "regions_pass": reg_ok, "pass": bool(ok)}
    lim = max(rule["floor"], 2 * d["sigma_paired"]) if np.isfinite(d["sigma_paired"]) else rule["floor"]
    return {"limit": lim, "pass": bool(abs(d["delta"]) <= lim)}


def _binned(p: pd.DataFrame, events, col: str, edges) -> list[dict]:
    out = []
    b = pd.cut(p[col], edges, right=False)
    for k, g in p.groupby(b, observed=False):
        out.append({"bin": str(k), **(delta_pairs(g, events) if len(g) else {"pairs": 0})})
    return out


def gun_weights(gun_pp: pd.DataFrame) -> pd.Series:
    """Shares of the H-val gun's P_P targets in the cells of section 6 (r_first below 200, 200-700, above 700 mm times
    |d0|/r_first below 0.05, 0.05-0.2, 0.2 and above). gun_pp: r_first, d0 (or abs_d0) of the P_P targets."""
    ad0 = np.abs(gun_pp.d0.to_numpy(np.float64)) if "d0" in gun_pp else gun_pp.abs_d0.to_numpy(np.float64)
    rf = gun_pp.r_first.to_numpy(np.float64)
    cell = _cell(rf, ad0 / rf)
    s = pd.Series(cell).value_counts(normalize=True).sort_index()
    return s


def _cell(r_first, ratio) -> np.ndarray:
    i = np.clip(np.searchsorted(RFIRST_BINS, r_first, side="right") - 1, 0, 2)
    j = np.clip(np.searchsorted(RATIO_CELLS, ratio, side="right") - 1, 0, 2)
    return i * 3 + j


def gun_weighted(p: pd.DataFrame, events, weights: pd.Series, cond: str = "full") -> dict:
    """Delta with the pairs of each section-6 cell weighted to the gun's share of that cell (renormalised over the
    cells that hold pairs; coverage = the gun weight of those cells); paired bootstrap sigma."""
    cell = _cell(p.r_first.to_numpy(np.float64), p.ratio.to_numpy(np.float64))
    cols = []
    for c in range(9):
        q = p.assign(**{f"n_{c}": cell == c, f"mc_{c}": (cell == c) & p[f"dm_closure_{cond}"],
                        f"mn_{c}": (cell == c) & p[f"dm_native_{cond}"]})
        cols.append(_counts(q, events, [f"n_{c}", f"mc_{c}", f"mn_{c}"])[:, 1:])
    cnt = np.concatenate(cols, 1)                                   # (events, 27): n, mc, mn per cell
    w = np.array([float(weights.get(c, 0.0)) for c in range(9)])

    def stat(s):
        s = s.reshape(*s.shape[:-1], 9, 3)
        n, mc, mn = s[..., 0], s[..., 1], s[..., 2]
        d = HS.pct(mc, n) - HS.pct(mn, n)
        have = n > 0
        ww = np.where(have, w, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ww.sum(-1) > 0, np.nansum(ww * np.where(have, d, 0.0), -1) / ww.sum(-1), np.nan)

    d, sd, _ = HS.bootstrap_stat(cnt, stat)
    n_cell = cnt.reshape(len(cnt), 9, 3)[:, :, 0].sum(0)
    return {"delta": d, "sigma_paired": sd, "coverage": float(w[n_cell > 0].sum()),
            "cells": [{"cell": c, "gun_weight": float(w[c]), "pairs": int(n_cell[c])} for c in range(9)]}


def first_layer(p: pd.DataFrame) -> dict:
    q = p[p.closure_n_hits > 0]
    return {"natives": int(len(p)), "closure_with_hits": int(len(q)),
            "first_hit_on_native_first_layer": float(q.first_layer_same.mean()) if len(q) else None}


def v1(p: pd.DataFrame, events, gun_cells: pd.Series | None = None) -> dict:
    """V1 per population with its rule (module docstring); p: `attach_outputs` with C-full."""
    events = sorted(int(e) for e in events)
    out = {}
    pops = {"a": "in_a", "b": "in_b", "b_all": "in_b_all", "b_at_sensor": "in_b_at_sensor", "c": "in_c",
            "c_all": "in_c_all", "d": "in_d"}
    for name, col in pops.items():
        m = p[p[col]]
        both = m[m.closure_target]
        d = delta_pairs(both, events)
        rec = {"natives": int(len(m)), **d,
               "closure_below_5_hits": int((m.closure_n_hits < 5).sum()),
               "closure_below_5_hits_share": float((m.closure_n_hits < 5).mean()) if len(m) else None,
               "unconditional": unconditional(m, events), "first_layer": first_layer(m),
               "abs_eta_bins": _binned(both, events, "abs_eta", ABS_ETA_BINS),
               "detector_regions": {r: (delta_pairs(both[both[f"in_{r}"]], events) if both[f"in_{r}"].any()
                                        else {"pairs": 0}) for r in DETECTOR_REGIONS}}
        if name in ("a", "b", "b_all", "d"):
            rec["r_first_bins"] = _binned(both, events, "r_first", RFIRST_BINS)
            rec["ratio_bins"] = _binned(both, events, "ratio", RATIO_CELLS)
        if name == "a":
            rec["regions"] = {r: delta_pairs(both[both.region == r], events) for r in REGIONS}
        if name in ("a", "b") and gun_cells is not None:
            rec["gun_weighted"] = gun_weighted(both, events, gun_cells)
        if name in RULES:
            rec["rule"] = {"kind": RULES[name]["kind"], **_rule(name, d, rec.get("regions"))}
        out[name] = rec
    out["at_sensor"] = {"tolerance_mm": SENSOR_TOL_MM,
                        "b_natives": int(p.in_b_all.sum()), "b_at_sensor": int(p.in_b_at_sensor.sum()),
                        "b_radial_within_1mm": int((p.in_b_all & (p.d_radial_mm <= SENSOR_TOL_MM)).sum()),
                        "a_at_sensor": int((p.in_a & p.at_sensor).sum()),
                        "c_at_sensor": int((p.in_c_all & p.at_sensor).sum())}
    out["decisive_pass"] = bool(out["a"]["rule"]["pass"] and out["c"]["rule"]["pass"])
    out["flags"] = [k for k in ("b", "d") if not out[k]["rule"]["pass"]]
    return out


def v1_delta(p: pd.DataFrame, events) -> dict:
    """V1-Delta (module docstring); p: `attach_outputs` with C-full and C-1GeV of both sides."""
    events = sorted(int(e) for e in events)
    out = {}
    for name, col in (("a", "in_a"), ("b", "in_b"), ("c", "in_c"), ("c_all", "in_c_all"), ("d", "in_d")):
        m = p[p[col] & p.closure_target]
        c = _counts(m, events, ["dm_closure_full", "dm_closure_1gev", "dm_native_full", "dm_native_1gev"])

        def stat(s):
            return ((HS.pct(s[..., 1], s[..., 0]) - HS.pct(s[..., 2], s[..., 0]))
                    - (HS.pct(s[..., 3], s[..., 0]) - HS.pct(s[..., 4], s[..., 0])))

        d, sd, _ = HS.bootstrap_stat(c, stat)
        n = c[:, 0].sum()
        rec = {"pairs": int(n), "delta_closure": float(HS.pct(c[:, 1].sum(), n) - HS.pct(c[:, 2].sum(), n)),
               "delta_native": float(HS.pct(c[:, 3].sum(), n) - HS.pct(c[:, 4].sum(), n)), "difference": d,
               "sigma_paired": sd}
        if name == "a":
            lim = max(V1D_FLOOR, 2 * sd) if np.isfinite(sd) else V1D_FLOOR
            rec["rule"] = {"kind": "decisive", "limit": lim,
                           "pass": bool(n > 0 and np.isfinite(d) and abs(d) <= lim)}
        out[name] = rec
    out["decisive_pass"] = out["a"]["rule"]["pass"]
    return out


# ------------------------------------------------------------------------------------------------ outputs
def native_outputs(cond: str, run: str = VAL_RUN, check: bool = True) -> pd.DataFrame:
    """The stored val outputs of the natives at WP-A (C-full or C-1GeV), after their sha256 check (section 9.2)."""
    from hnl.trackml.convert import sha256_file
    rel, prefix = NATIVE_OUTPUTS[cond]
    path = os.path.join(run, rel)
    if check and not sha256_file(path).startswith(prefix):
        raise ValueError(f"{path}: sha256 differs from the addendum's {prefix}...")
    return HS.read_per_target(os.path.dirname(path), "WP-A")


def diag_outputs(cond: str, run: str = VAL_RUN, sha_list: str = VAL_DIAG_SHA) -> pd.DataFrame:
    """(event, particle_id, dm) of the stored WP-A diagnostic tables of one condition (diag/displaced/diag_<cond>_b*,
    hnl.trackml.displaced_diag), each file checked against its sha256 in the addendum's list first."""
    from hnl.trackml.convert import sha256_file
    with open(sha_list) as fh:
        want = dict(reversed(line.split()) for line in fh if line.strip())
    files = sorted(f for f in want if os.path.basename(f).startswith(f"diag_{cond}_b"))
    if not files:
        raise LookupError(f"{sha_list}: no diagnostic table of condition {cond}")
    out = []
    for f in files:
        path = os.path.join(run, f)
        if sha256_file(path) != want[f]:
            raise ValueError(f"{path}: sha256 differs from {sha_list}")
        out.append(pd.read_parquet(path, columns=["event", "particle_id", "dm"]))
    t = pd.concat(out, ignore_index=True).astype({"event": np.int64, "particle_id": np.int64, "dm": bool})
    if t.duplicated(["event", "particle_id"]).any():
        raise ValueError(f"diagnostic tables of {cond}: a target appears twice")
    return t


def compare_dm(p: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """The dm of the natives of p in two output tables (per_target and the diagnostic tables): they must agree."""
    da = HS.attach_dm(p, a, "host", "native_id", "per_target", np.ones(len(p), bool))
    db = HS.attach_dm(p, b, "host", "native_id", "diagnostic tables", np.ones(len(p), bool))
    return {"natives": int(len(p)), "differ": int((da != db).sum()), "equal": bool((da == db).all())}


def placeholder_outputs(p: pd.DataFrame, seed: int = 20261014) -> dict:
    """Code test only: truth-made (event, particle_id, dm) tables of both sides in C-full and C-1GeV for the targets of
    p, with dm drawn from fixed probabilities per population and side (numpy default_rng([seed, side, condition])).
    No model is run and no model output is read."""
    prob = {"a": 0.93, "b": 0.6, "c": 0.995}
    base = p.population.map(prob).to_numpy(np.float64)
    out = {}
    for si, side in enumerate(("closure", "native")):
        for ci, cond in enumerate(("full", "1gev")):
            rng = np.random.default_rng([seed, si, ci])
            dm = rng.random(len(p)) < base - 0.01 * ci
            if side == "closure":
                keep = p.closure_target.to_numpy(bool)
                tab = pd.DataFrame({"event": p.event, "particle_id": p.closure_id, "dm": dm})[keep]
            else:
                tab = pd.DataFrame({"event": p.host, "particle_id": p.native_id, "dm": dm})
            out[(side, cond)] = tab.reset_index(drop=True)
    return out


# ------------------------------------------------------------------------------------------------ grids
def _under(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def pilot_closure_dry(root: str, events, split: dict | None = None) -> None:
    """Raises unless the closure tables at root are the pilot closure dry run (module docstring): a scratch root, never
    the production overlay root; every event a closure id (k = 1) of a pilot host; the set's meta made with the pilot
    base seed of split_h_v1.json (the pilot hosts are closure hosts too, so the ids alone do not tell)."""
    from hnl.overlay import blind as B
    split = split or B.load_split()
    if os.path.realpath(root) == os.path.realpath(B.DERIVED_OVERLAY):
        raise PermissionError(f"{root}: the production closure set takes grids of the frozen classifier only; a dry "
                              "run reads the pilot closure set in a scratch root")
    pilot = {int(h) for h in split["sets"]["pilot"]["hosts"]}
    events = [int(e) for e in events]
    bad = [e for e in events if e < B.OVERLAY_FIRST_ID or B.split_overlay_id(e)[0] != 1
           or B.split_overlay_id(e)[1] not in pilot]
    if not events or bad:
        raise PermissionError(f"{root}: not the pilot closure dry run (events outside the closure ids of the pilot "
                              f"hosts: {bad[:5]})")
    with open(os.path.join(root, f"meta_{SET}.json")) as fh:
        meta = json.load(fh)
    if int(meta.get("base_seed", -1)) != int(split["sets"]["pilot"]["base_seed"]):
        raise PermissionError(f"{root}: closure set made with base seed {meta.get('base_seed')}, not the pilot's "
                              f"{split['sets']['pilot']['base_seed']}")


def closure_grid(d: str, cond: str, root: str, events, dry: bool = False, split: dict | None = None,
                 selection: str = SELECTION) -> dict:
    """The grid.json record of a closure grid that V1 may read, after the checks of the module docstring (raises
    otherwise). root: the closure tables V1 reads; events: their events (the sidecar's)."""
    from hnl.overlay import blind as B
    split = split or B.load_split()
    events = sorted(int(e) for e in events)
    if cond not in CLOSURE_CONDITIONS:
        raise ValueError(f"the closure set runs in {CLOSURE_CONDITIONS}, not {cond!r}")
    if not dry:                                         # the check shared with hnl.overlay.h1 (H-val, H-test)
        from hnl.overlay import tracks as TR
        return TR.grid_provenance(d, SET, cond, root, events, split, selection)
    pilot_closure_dry(root, events, split)
    with open(os.path.join(d, "grid.json")) as fh:
        rec = json.load(fh)
    if rec.get("set") != SET or rec.get("condition") != cond or int(rec.get("k", -1)) != 1:
        raise ValueError(f"{d}: a grid of {rec.get('set')} (k = {rec.get('k')}) in {rec.get('condition')}, not "
                         f"{SET} in {cond}")
    if not rec["scores_provenance"].get("placeholder"):
        raise PermissionError(f"{d}: the pilot closure dry run reads truth-made placeholder grids only")
    if sorted(int(e) for e in rec["events"]) != events:
        raise ValueError(f"{d}: the grid's events are not those of the closure tables at {root}")
    if os.path.realpath(rec.get("root") or "") != os.path.realpath(root):
        raise ValueError(f"{d}: built on the tables at {rec.get('root')}, not {root}")
    return rec


def closure_outputs(d: str, cond: str, root: str, events, dry: bool = False, split: dict | None = None,
                    selection: str = SELECTION) -> tuple[dict, pd.DataFrame]:
    """(grid record, (event, particle_id, dm) at WP-A) of a closure grid, read only after `closure_grid`."""
    rec = closure_grid(d, cond, root, events, dry, split, selection)
    return rec, HS.read_per_target(d, "WP-A", selection)


def write_placeholder_grid(d: str, tab: pd.DataFrame, cond: str, events, root: str,
                           selection: str = SELECTION) -> dict:
    """Code test only: the truth-made placeholder dm of the closure targets (`placeholder_outputs`) as a grid at WP-A in
    the layout of hnl.overlay.tracks (per_target.parquet, configs.json, config_names.json, grid.json), marked
    placeholder. Refused under the production runs root and over an existing grid. No track is built."""
    from hnl.overlay import blind as B
    if _under(d, B.RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    if os.path.exists(os.path.join(d, "grid.json")):
        raise FileExistsError(f"{d} holds a grid already: refusing to overwrite it")
    fam, *par = HS.wp_params("WP-A", selection)
    os.makedirs(d, exist_ok=True)
    t = tab.astype({"event": np.int64, "particle_id": np.int64, "dm": bool}).assign(config=0)
    t[["event", "config", "particle_id", "dm"]].to_parquet(os.path.join(d, "per_target.parquet"), index=False)
    with open(os.path.join(d, "configs.json"), "w") as fh:
        json.dump([[fam, *par]], fh)
    with open(os.path.join(d, "config_names.json"), "w") as fh:
        json.dump(["WP-A"], fh)
    rec = {"set": SET, "k": 1, "condition": cond, "events": sorted(int(e) for e in events),
           "n_events": len(events), "block": 0, "n_blocks": 1, "root": os.path.abspath(root), "scores": None,
           "scores_provenance": {"placeholder": True, "ckpt_sha256": None, "condition": cond,
                                 "kind": "truth-made dm (hnl.overlay.v1.placeholder_outputs); no model, no scores"},
           "configs": [{"name": "WP-A", "family": fam, "params": ",".join(f"{x:g}" for x in par)}],
           "merge_model": None, "builders": "none (V1 dry run of the pilot closure set)",
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename}
    with open(os.path.join(d, "grid.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def evaluate(p: pd.DataFrame, events, gun_cells=None, with_1gev: bool = True) -> dict:
    out = {"events": len(events), "pairs_table": {"natives": int(len(p)), "closure_targets": int(p.closure_target.sum())},
           "v1": v1(p, events, gun_cells)}
    if with_1gev:
        out["v1_delta"] = v1_delta(p, events)
    out["decisive_pass"] = bool(out["v1"]["decisive_pass"] and (out.get("v1_delta", {}).get("decisive_pass", True)))
    return out


def _dump(out: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--closure-full", required=True, help="grid directory of the closure run in C-full")
    r.add_argument("--closure-1gev", default=None, help="grid directory of the closure run in C-1GeV (V1-Delta)")
    r.add_argument("--gun-cells", default=None, help="parquet of the H-val gun's P_P targets (r_first, d0)")
    r.add_argument("--out-json", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--root", required=True)
    d.add_argument("--out-json", required=True)
    d.add_argument("--grids", default=None, help=f"directory of the placeholder grids (default: <out-json dir>/"
                                                 f"{DRY_GRIDS}); refused if it holds grids already")
    a = ap.parse_args(argv)
    t0 = time.time()
    with open(os.path.join(a.root, f"sidecar_{SET}.json")) as fh:
        events = sorted(int(x["overlay_event"]) for x in json.load(fh)["events"])
    grids = {}
    if a.cmd == "dry-run":
        from hnl.overlay import blind as B
        pilot_closure_dry(a.root, events)
        gd = a.grids or os.path.join(os.path.dirname(os.path.abspath(a.out_json)), DRY_GRIDS)
        if _under(a.out_json, B.RUNS_H) or _under(gd, B.RUNS_H):
            raise PermissionError("dry-run outputs never go under the production runs root")
        pairs = truth_pairs(a.root)
        ph = placeholder_outputs(pairs)
        outs = {}
        for c in CLOSURE_CONDITIONS:
            write_placeholder_grid(os.path.join(gd, c), ph[("closure", c)], c, events, a.root)
            grids[c], outs[("closure", c)] = closure_outputs(os.path.join(gd, c), c, a.root, events, dry=True)
            outs[("native", c)] = ph[("native", c)]
        gun = None
        label = ("DRY RUN: truth-made placeholder outputs (closure side read back from placeholder grids); no model "
                 "output was read")
    else:
        from hnl.overlay import grant as G
        given = {"full": a.closure_full, "1gev": a.closure_1gev}
        for c, gdir in given.items():         # a grant for every condition read, before anything is read
            if gdir:
                G.check(events, "eval", c, out=a.out_json)
        for c, gdir in given.items():         # every grid checked before anything is read
            if gdir:
                grids[c] = closure_grid(gdir, c, a.root, events)
        pairs = truth_pairs(a.root)
        outs = {}
        for c, gdir in given.items():
            if gdir:
                grids[c], outs[("closure", c)] = closure_outputs(gdir, c, a.root, events)
                outs[("native", c)] = native_outputs(c)
        for c in grids:                       # the stored per_target and the WP-A diagnostic tables must agree
            chk = compare_dm(pairs, outs[("native", c)], diag_outputs(c))
            if not chk["equal"]:
                raise RuntimeError(f"stored val outputs in {c}: per_target and diagnostic tables differ ({chk})")
        gun = gun_weights(pd.read_parquet(a.gun_cells)) if a.gun_cells else None
        label = "closure set, frozen Canto-L (TrackML) at WP-A"
    p = attach_outputs(pairs, outs)
    out = {"label": label, "root": a.root,
           "grids": {c: {k: r.get(k) for k in ("set", "k", "condition", "n_events", "root", "scores",
                                               "scores_provenance", "merge_model", "git", "slurm_job")}
                     for c, r in grids.items()},
           **evaluate(p, events, gun, with_1gev=("closure", "1gev") in outs)}
    out["seconds"] = round(time.time() - t0, 1)
    _dump(out, a.out_json)
    print(json.dumps({"label": label, "decisive_pass": out["decisive_pass"], "flags": out["v1"]["flags"]}))


if __name__ == "__main__":
    main()
