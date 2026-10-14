"""Truth-only checks of a stored overlay set before any model run (docs/pileup_overlay_plan.md section 10). No model is
run and no model output is read. Inputs: the stored set (ROOT/{hits,particles,ptype}/<set>, meta, sidecar and gun truth
of hnl.overlay.writer), the work directory of slurm/overlay/overlay_set.sbatch (gun events, packed sim hits, the
gun-only CSV tar), the hosts' tp_v1 tables (hits and particles; development hosts are tp_v1 val events) and, for
C1 and V3, the hits and truth of TrackML calibration training events.

Checks.
- format: hnl.overlay.writer.verify_set; the frozen read path (hnl.trackml.events.load_events with the table_root of
  the overlay ids, config.event_set on an explicit list); per event: rows = host + gun, hit ids 1..N, gun hits last,
  host rows and host true edges identical to tp_v1, gun true edges = sum over gun particles of (hits - 1), no mixed
  true edge, every X1 cluster feature computable (hnl.trackml.events.cluster_feature), every (volume, layer, module)
  in TrackML's detectors.csv (track building raises otherwise), particle ids unique and positive as int64, gun vertex
  fields 1001-1010, host vertex fields below 1001, gun particles = the gun truth table, n_hits = hit counts, gun
  particles that Fatras did not simulate without hits, the stored gun positions = the CSV members, hosts in tp_v1 val.
- weights: gun weights 0 and the host weights summing to one within 1e-6 (H-val rule); the closure rule exercised
  on the stored events: host particles of the V1 (a) definition dropped in memory (hnl.overlay.writer.merge_event),
  the merged event passes hnl.overlay.writer.check_event with the expected sum 1 - (weights of the removed hits) and
  fails with 1.
- production: every gun event recomputed from the packed sim hits (hnl.overlay.writer.gun_event: emulation and
  digitiser with the set's seeds) gives byte-identical CSV members, and their frozen conversion equals the stored
  gun rows; the digitiser's cluster values equal the stored ones.
- C1 on the overlaid hits: the stored cluster features and the residuals of the gun hits (true point and direction
  from the CSV truth member) against TrackML calibration hits with hnl.overlay.c1.compare (matched incidence; bins
  need 200 hits on each side). A few hosts give a few hundred gun hits, so `at_test_incidence` adds a descriptive
  comparison per volume: TrackML reweighted to the gun's joint incidence distribution (both angles), with a bootstrap
  uncertainty of the gun value.
- V3 (descriptive): cluster features and residuals of gun hits against native hits (targets, and displaced targets
  with |d0| >= 10 mm or r_v >= 100 mm) per region, in bins of |tan(incidence)| and of signed eta. tp_v1 stores true
  hit kinematics for the calibration training events only, so the native side with residuals and incidence is the
  C1 reference events; the hosts' own native target hits enter with their cluster features per region and signed eta
  (`same_event_clusters`). Same-module near-coincidences (another particle's hit on the module, within 1 mm and
  0.1 mm) for gun and native target hits of the hosts.
- content (descriptive): gun particles, hits, targets, P_P and P_D yields per LLP, species, Delta R between targets of
  different LLPs, modules shared with host hits, emulation and digitiser counts, Fatras status.

    python -m hnl.overlay.truth_checks run --set pilot --work DIR --out-json J [--root ROOT] [--ref-events 40]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa

from hnl.overlay import c1 as C1
from hnl.overlay import digitise as D
from hnl.overlay import gun as GUN
from hnl.overlay import writer as W
from hnl.trackml import convert as C
from hnl.trackml import events as EV
from hnl.trackml import throughput as T

TP = T.DERIVED_TP
REGION_OF_VOLUME = {7: "pixel endcap", 8: "pixel barrel", 9: "pixel endcap", 12: "short-strip endcap",
                    13: "short-strip barrel", 14: "short-strip endcap", 16: "long-strip endcap",
                    17: "long-strip barrel", 18: "long-strip endcap"}
INC_EDGES = (0.0, 0.2, 0.6, 1.6, np.inf)               # |tan(incidence)|
ETA_EDGES = (-np.inf, -1.0, 1.0, np.inf)               # signed eta of the particle: forward -z, central, forward +z
NEAR_MM = (1.0, 0.1)
N_BOOT = 200
PP_D0_MM, PP_RV_MM = 10.0, 100.0                       # P_P / V1 (a): |d0| >= 10 mm or r_v >= 100 mm
EXPECTED_YIELD_PER_LLP = {"P_P": 0.603, "P_D": 1.687}  # plan section 7 (2 T HNL setup, no emulation)


# ------------------------------------------------------------------------------------------------ inputs
def csv_members(tar_path: str) -> dict[int, dict[str, pd.DataFrame]]:
    """The gun-only CSV members of a set tar (hnl.overlay.writer.write_csv_tar): {event: {kind: table}} and the raw
    gzip bytes under kind + '_gz'."""
    out: dict[int, dict] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            ev, kind = name[len("event"):].split(".")[0].split("-")
            data = tf.extractfile(m).read()
            d = out.setdefault(int(ev), {})
            d[kind + "_gz"] = data
            d[kind] = pd.read_csv(io.BytesIO(gzip.decompress(data)), dtype=C.CSV_DTYPES[kind])
    return out


def load_context(set_name: str, work: str, root: str = W.DERIVED_OVERLAY) -> dict:
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        side = json.load(fh)
    with open(os.path.join(work, "gun", "gun_events.json")) as fh:
        gev = json.load(fh)
    events = {int(s["overlay_event"]): s for s in side["events"]}
    stored = {e: W.host_tables(e, root) for e in events}
    hosts = {e: W.host_tables(s["host"], TP) for e, s in events.items()}
    return {"set": set_name, "root": root, "work": work, "meta": meta, "side": side, "gev": gev, "events": events,
            "stored": stored, "hosts": hosts,
            "gun_truth": pd.read_parquet(os.path.join(root, f"gun_truth_{set_name}.parquet")),
            "csv": csv_members(os.path.join(work, f"gun_csv_{set_name}.tar")),
            "sim_path": os.path.join(work, "gun", "fatras_out", "simhits.parquet")}


def gun_slices(ctx: dict, e: int) -> dict[str, pa.Table]:
    """The gun rows of a stored overlay event (the last rows of each table)."""
    t, s = ctx["stored"][e], ctx["events"][e]
    nh, ng = s["n_gun_hits"], s["n_gun_particles"]
    return {"hits": t["hits"].slice(t["hits"].num_rows - nh), "particles": t["particles"].slice(
        t["particles"].num_rows - ng), "ptype": t["ptype"].slice(t["ptype"].num_rows - ng)}


# ------------------------------------------------------------------------------------------------ local frame
def local_features(xyz, volume, layer, module, txyz, tp, modules: D.Modules, ncells=None, charge=None, span0=None,
                   span1=None) -> pd.DataFrame:
    """Per hit in its TrackML module frame: tan_u = du/dw, tan_v = dv/dw of the true direction, and the residual of the
    hit against the true point moved along the true direction onto the hit's plane (the definitions of
    hnl.overlay.c1.trackml_hits); with the cluster values, the c1.features table. Hits whose true direction lies in
    the module plane are dropped (column keep of the returned frame marks the rest)."""
    row = modules.index(volume, layer, module)
    if (row < 0).any():
        raise KeyError("hit on a module missing from detectors.csv")
    R, c = modules.R[row], modules.c[row]
    lh = np.einsum("nij,ni->nj", R, np.asarray(xyz, np.float64) - c)
    lt = np.einsum("nij,ni->nj", R, np.asarray(txyz, np.float64) - c)
    ld = np.einsum("nij,ni->nj", R, np.asarray(tp, np.float64))
    keep = np.abs(ld[:, 2]) > D.MIN_COS * np.linalg.norm(ld, axis=1)
    lh, lt, ld = lh[keep], lt[keep], ld[keep]
    tu, tv = ld[:, 0] / ld[:, 2], ld[:, 1] / ld[:, 2]
    s = lh[:, 2] - lt[:, 2]
    du = lh[:, 0] - (lt[:, 0] + s * tu)
    dv = lh[:, 1] - (lt[:, 1] + s * tv)
    sel = lambda a: np.asarray(a)[keep] if a is not None else np.full(int(keep.sum()), np.nan)   # noqa: E731
    f = C1.features(np.asarray(volume)[keep], tu, tv, sel(span0), sel(span1), sel(charge), du, dv)
    f["ncells"] = sel(ncells)
    f["keep_index"] = np.flatnonzero(keep)
    return f


def region(volume) -> np.ndarray:
    return np.array([REGION_OF_VOLUME.get(int(v), "other") for v in np.asarray(volume)], dtype=object)


# ------------------------------------------------------------------------------------------------ format and ids
def format_checks(ctx: dict) -> dict:
    from hnl.trackml.config import event_set
    det = C.load_detectors()
    ids = sorted(ctx["events"])
    v = W.verify_set(ctx["set"], ctx["root"])
    default_root = all(os.path.abspath(EV.table_root(e)) == os.path.abspath(ctx["root"]) for e in ids)
    evs = EV.load_events(ids, root=None if default_root else ctx["root"])
    val = set(event_set("val"))
    out = {"verify_problems": v["problems"], "events": len(ids), "read_path_default_table_root": default_root,
           "event_set_explicit_list": event_set(",".join(str(e) for e in ids)) == ids,
           "hosts_in_tp_v1_val": all(int(s["host"]) in val for s in ctx["events"].values()), "per_event": {}}
    problems = []
    for e in ids:
        s, st, ht = ctx["events"][e], ctx["stored"][e], ctx["hosts"][e]
        ev = evs[e]
        hev = EV.load_events([s["host"]], root=TP)[s["host"]]
        nh, ng, nhost = s["n_gun_hits"], s["n_gun_particles"], ht["hits"].num_rows
        p = st["particles"].to_pandas()
        pid = p.particle_id.to_numpy()
        gpid = pid[len(pid) - ng:]
        vert = pid >> 52
        h = st["hits"].to_pandas()
        gun_rows = np.arange(nhost, nhost + nh)
        gun_hit = np.isin(h.pid.to_numpy(), gpid)
        te = ev.true_edges
        gte = gun_hit[te[0]] & gun_hit[te[1]]
        mixed = gun_hit[te[0]] != gun_hit[te[1]]
        gp = p.iloc[len(p) - ng:]
        exp_gun_edges = int(np.maximum(gp.n_hits.to_numpy(np.int64) - 1, 0).sum())
        feat_ok = {}
        for f in EV.CLUSTER_FEATURES:
            try:
                x = EV.cluster_feature(ev, f)
                feat_ok[f] = bool(np.isfinite(x).all())
            except (ValueError, KeyError) as exc:
                feat_ok[f] = f"raises: {exc}"
        key = C.module_key(h.volume.to_numpy(), h.layer.to_numpy(), h.module.to_numpy())
        known = np.isin(key, det["key"])
        truth = ctx["gun_truth"][ctx["gun_truth"].event == e]
        lost = truth[truth.sim_status != "simulated"]
        lost_hits = gp.set_index("particle_id").n_hits.reindex(lost.particle_id.to_numpy(np.int64)).to_numpy()
        cm = ctx["csv"][e]["hits"]
        xyz_csv = cm[["x", "y", "z"]].to_numpy(np.float64).astype(np.float32)
        xyz_st = h[["x", "y", "z"]].to_numpy()[gun_rows]
        w = h.weight.to_numpy(np.float64)
        r = {"rows": int(len(h)), "host_rows": nhost, "gun_hits": nh, "gun_particles": ng,
             "rows_equal_host_plus_gun": len(h) == nhost + nh and ev.n == len(h),
             "gun_csv_hit_ids_1_to_m": bool(np.array_equal(cm.hit_id.to_numpy(), np.arange(1, len(cm) + 1))),
             "gun_hits_last_rows": bool(gun_hit[nhost:].all() and not gun_hit[:nhost].any()),
             "host_rows_identical": all(W.columns_equal(ht["hits"].column(c), st["hits"].slice(0, nhost).column(c))
                                        for c in ht["hits"].schema.names if c != "event"),
             "host_true_edges_identical": bool(np.array_equal(te[:, ~gte & ~mixed], hev.true_edges)),
             "gun_true_edges": int(gte.sum()), "gun_true_edges_expected": exp_gun_edges,
             "mixed_true_edges": int(mixed.sum()),
             "cluster_features_finite": feat_ok,
             "hits_on_unknown_modules": int((~known).sum()),
             "particle_ids_unique_positive_int64": bool(p.particle_id.dtype == np.int64 and p.particle_id.is_unique
                                                        and (pid > 0).all()),
             "gun_vertex_range": [int(vert[len(pid) - ng:].min()), int(vert[len(pid) - ng:].max())],
             "host_vertex_max": int(vert[:len(pid) - ng].max()),
             "gun_particles_equal_truth": bool(np.array_equal(np.sort(gpid),
                                                              np.sort(truth.particle_id.to_numpy(np.int64)))),
             "n_hits_equal_hit_counts": bool(np.array_equal(
                 pd.Series(h.pid.to_numpy()[h.pid.to_numpy() != 0]).value_counts().reindex(pid).fillna(0)
                 .to_numpy(np.int64), p.n_hits.to_numpy(np.int64))),
             "ptype_rows_equal_particles": bool(np.array_equal(st["ptype"].column("particle_id").to_numpy(), pid)),
             "gun_weight_max_abs": float(np.abs(w[gun_hit]).max()) if gun_hit.any() else 0.0,
             "host_weight_sum": float(np.nansum(w[~gun_hit])),
             "not_simulated_gun_particles": int(len(lost)), "not_simulated_with_hits": int(np.nansum(lost_hits)),
             "csv_hits_equal_stored": bool(len(cm) == nh and np.array_equal(xyz_csv, xyz_st)),
             "gun_categories": gp.category.value_counts().to_dict()}
        out["per_event"][e] = r
        bad = [k for k in ("rows_equal_host_plus_gun", "gun_csv_hit_ids_1_to_m", "gun_hits_last_rows",
                           "host_rows_identical", "host_true_edges_identical", "particle_ids_unique_positive_int64",
                           "gun_particles_equal_truth", "n_hits_equal_hit_counts", "ptype_rows_equal_particles",
                           "csv_hits_equal_stored") if not r[k]]
        bad += [f"cluster feature {k}" for k, x in feat_ok.items() if x is not True]
        if r["gun_true_edges"] != exp_gun_edges or r["mixed_true_edges"]:
            bad.append("true edges")
        if r["hits_on_unknown_modules"]:
            bad.append("unknown modules")
        if not (W.GUN_VERTEX[0] <= r["gun_vertex_range"][0] and r["gun_vertex_range"][1] <= W.GUN_VERTEX[1]):
            bad.append("gun vertex field")
        if r["host_vertex_max"] >= W.GUN_VERTEX[0]:
            bad.append("host vertex field")
        if r["gun_weight_max_abs"] != 0 or abs(r["host_weight_sum"] - 1.0) > W.WEIGHT_TOL:
            bad.append("weights")
        if r["not_simulated_with_hits"]:
            bad.append("hits of a particle that Fatras did not simulate")
        problems += [f"event {e}: {b}" for b in bad]
    for k in ("read_path_default_table_root", "event_set_explicit_list", "hosts_in_tp_v1_val"):
        if not out[k]:
            problems.append(k)
    out["problems"] = v["problems"] + problems
    out["pass"] = not out["problems"]
    out["max_abs_host_weight_sum_minus_1"] = float(max(abs(r["host_weight_sum"] - 1)
                                                       for r in out["per_event"].values()))
    return out


def weight_rule_checks(ctx: dict) -> dict:
    """The closure weight-sum rule of plan section 10 on the stored events (module docstring)."""
    out = {"per_event": {}, "problems": []}
    for e, s in sorted(ctx["events"].items()):
        host, gun = ctx["hosts"][e], gun_slices(ctx, e)
        p = host["particles"].to_pandas()
        pid = p.particle_id.to_numpy(np.int64)
        gen = (pid >> 24) & 0xFFF
        drop = pid[p.is_target.to_numpy(bool) & (gen == 0)
                   & ((np.abs(p.d0.to_numpy()) >= PP_D0_MM) | (p.rv.to_numpy() >= PP_RV_MM))]
        h = host["hits"].to_pandas()
        removed = float(np.nansum(h.weight.to_numpy(np.float64)[np.isin(h.pid.to_numpy(), drop)]))
        merged, rm = W.merge_event(host, gun, e, drop_pids=drop)
        ok = W.check_event(merged, host, s["n_gun_particles"], 1.0 - removed, drop)
        wrong = W.check_event(merged, host, s["n_gun_particles"], 1.0, drop)
        stored_ok = W.check_event(ctx["stored"][e], host, s["n_gun_particles"], 1.0)
        r = {"dropped_particles": int(len(drop)), "dropped_hits": rm["info"]["dropped_host_hits"],
             "removed_weight": removed, "problems_with_expected_sum": ok,
             "flags_unadjusted_sum": any("weight sum" in x for x in wrong), "stored_event_problems": stored_ok}
        out["per_event"][e] = r
        if ok or stored_ok or (removed > W.WEIGHT_TOL and not r["flags_unadjusted_sum"]):
            out["problems"].append(f"event {e}: {r}")
    out["pass"] = not out["problems"]
    return out


# ------------------------------------------------------------------------------------------------ production path
def production_closure(ctx: dict) -> dict:
    """Recompute every gun event from the packed sim hits and compare with the tar and the stored rows."""
    from hnl.overlay import emulate as EM
    meta, gev = ctx["meta"], ctx["gev"]
    tab = EM.Tables.load(meta["emulation_tables"])
    out = {"emulation_hash_now": EM.emulation_hash(meta["emulation_tables"]), "emulation_hash_stored":
           meta["emulation_hash"], "per_event": {}, "problems": []}
    mods = D.Modules.load()
    lookup = D.load_lookup(meta["lookup"])
    det = C.load_detectors()
    sim = pd.read_parquet(ctx["sim_path"])
    for ev in gev["events"]:
        e, fe = int(ev["overlay_event"]), int(ev["fatras_event"])
        parts = pd.read_csv(W.FIO_path(os.path.join(ctx["work"], "gun"), fe), dtype={"particle_id": np.uint64})
        g = W.gun_event(sim[sim.event == fe], parts, lookup, mods, tab, gev["base_seed"], e)
        same_bytes = all(g["gz"][k] == ctx["csv"][e][k + "_gz"] for k in ("hits", "cells", "particles", "truth"))
        gtab, _ = W.convert_gun(g["gz"], e, det)
        stored = gun_slices(ctx, e)
        equal = {k: W.tables_equal(gtab[k].replace_schema_metadata(None), stored[k]) for k in W.TABLES}
        dig = g["digitised"].hits
        sh = stored["hits"].to_pandas()
        o = np.lexsort((np.arange(len(dig)), dig.module_id, dig.layer_id, dig.volume_id))   # gun_members' hit order
        d = dig.iloc[o].reset_index(drop=True)
        r = {"csv_bytes_identical": same_bytes, "stored_rows_equal_conversion": equal,
             "ncells_equal": bool(np.array_equal(d.ncells.to_numpy(), sh.ncells.to_numpy())),
             "spans_equal": bool(np.array_equal(d.span0.to_numpy(), sh.span0.to_numpy())
                                 and np.array_equal(d.span1.to_numpy(), sh.span1.to_numpy())),
             "max_rel_charge_diff": float(np.max(np.abs(sh.charge.to_numpy(np.float64) / d.charge.to_numpy() - 1)))
             if len(d) else 0.0,
             "max_abs_xyz_diff_mm": float(np.max(np.abs(sh[["x", "y", "z"]].to_numpy(np.float64)
                                                       - d[["x", "y", "z"]].to_numpy()))) if len(d) else 0.0}
        out["per_event"][e] = r
        if not (same_bytes and all(equal.values()) and r["ncells_equal"] and r["spans_equal"]):
            out["problems"].append(f"event {e}: {r}")
    if out["emulation_hash_now"] != out["emulation_hash_stored"]:
        out["problems"].append("emulation hash changed since the set was written")
    if "overlay_hash" in meta:
        out["overlay_hash_now"] = W.overlay_hash(meta["emulation_tables"])
        out["overlay_hash_stored"] = meta["overlay_hash"]
        if out["overlay_hash_now"] != meta["overlay_hash"]:
            out["problems"].append("overlay hash (emulation, digitiser, seed rules) changed since the set was written")
    out["pass"] = not out["problems"]
    return out


# ------------------------------------------------------------------------------------------------ hit features
def gun_hit_features(ctx: dict, modules: D.Modules) -> pd.DataFrame:
    """Stored cluster features and local residuals of every gun hit, with its particle's truth."""
    out = []
    for e in sorted(ctx["events"]):
        g = gun_slices(ctx, e)
        h = g["hits"].to_pandas()
        p = g["particles"].to_pandas().set_index("particle_id")
        tr = ctx["csv"][e]["truth"]
        if len(tr) != len(h) or not np.array_equal(tr.particle_id.to_numpy(np.int64), h.pid.to_numpy(np.int64)):
            raise RuntimeError(f"event {e}: CSV truth rows differ from the stored gun rows")
        f = local_features(h[["x", "y", "z"]].to_numpy(np.float64), h.volume, h.layer, h.module,
                           tr[["tx", "ty", "tz"]].to_numpy(), tr[["tpx", "tpy", "tpz"]].to_numpy(), modules,
                           h.ncells.to_numpy(), h.charge.to_numpy(np.float64), h.span0.to_numpy(), h.span1.to_numpy())
        pid = h.pid.to_numpy(np.int64)[f.keep_index.to_numpy()]
        pp = p.reindex(pid)
        llp = (pid >> 52) - GUN.VERTEX0
        f = f.assign(event=e, pid=pid, llp=llp, eta=pp.eta.to_numpy(), pt=pp.pt.to_numpy(), d0=pp.d0.to_numpy(),
                     rv=pp.rv.to_numpy(), is_target=pp.is_target.to_numpy(bool))
        f["is_pp_target"] = (f.llp <= GUN.N_ISO) & f.is_target & ((f.d0.abs() >= PP_D0_MM) | (f.rv >= PP_RV_MM))
        out.append(f)
    return pd.concat(out, ignore_index=True)


def native_hit_features(events, modules: D.Modules, root: str = TP) -> pd.DataFrame:
    """The same features for the particle hits of TrackML events with stored truth kinematics (in tp_v1 only the
    calibration training events have them; val events store hits and particles only)."""
    from hnl.trackml.convert import read_table
    out = []
    for k in range(0, len(events), 10):
        part = [int(e) for e in events[k:k + 10]]
        h = read_table("hits", part, ["event", "row", "x", "y", "z", "volume", "layer", "module", "pid", "ncells",
                                      "charge", "span0", "span1"], root=root)
        t = read_table("truth", part, root=root)
        h = h[h.pid != 0].merge(t, on=["event", "row"], how="inner", validate="one_to_one")
        p = read_table("particles", part, ["event", "particle_id", "eta", "pt", "d0", "rv", "is_target"], root=root)
        f = local_features(h[["x", "y", "z"]].to_numpy(np.float64), h.volume, h.layer, h.module,
                           h[["tx", "ty", "tz"]].to_numpy(np.float64), h[["tpx", "tpy", "tpz"]].to_numpy(np.float64),
                           modules, h.ncells.to_numpy(), h.charge.to_numpy(np.float64), h.span0.to_numpy(),
                           h.span1.to_numpy())
        ki = f.keep_index.to_numpy()
        f = f.assign(event=h.event.to_numpy()[ki], pid=h.pid.to_numpy(np.int64)[ki])
        f = f.merge(p.rename(columns={"particle_id": "pid"}), on=["event", "pid"], how="left", validate="many_to_one")
        out.append(f)
    f = pd.concat(out, ignore_index=True)
    f["is_target"] = f.is_target.fillna(False).astype(bool)
    return f


def same_event_clusters(ctx: dict) -> list[dict]:
    """Cluster features (no truth kinematics needed) of gun hits and of the native target hits of the same hosts, per
    region and signed-eta bin of the particle: the same-event part of V3."""
    rows = []
    for e in sorted(ctx["events"]):
        st = ctx["stored"][e]
        h = st["hits"].to_pandas()
        p = st["particles"].to_pandas()
        ng = ctx["events"][e]["n_gun_particles"]
        p["gun"] = np.arange(len(p)) >= len(p) - ng
        h = h[h.pid != 0].merge(p[["particle_id", "eta", "is_target", "d0", "rv", "gun"]].rename(
            columns={"particle_id": "pid"}), on="pid", how="left", validate="many_to_one")
        rows.append(h)
    h = pd.concat(rows, ignore_index=True)
    h["region"] = region(h.volume)
    h["etab"] = np.digitize(h.eta, ETA_EDGES) - 1
    disp = (h.d0.abs() >= PP_D0_MM) | (h.rv >= PP_RV_MM)
    groups = {"gun": h[h.gun], "gun_targets": h[h.gun & h.is_target], "native_targets": h[~h.gun & h.is_target],
              "native_displaced_targets": h[~h.gun & h.is_target & disp]}
    out = []
    for name, g in groups.items():
        for (reg, b), gb in g.groupby(["region", "etab"]):
            out.append({"group": name, "region": reg, "eta": ["< -1", "-1 to 1", "> 1"][int(b)], "n": int(len(gb)),
                        "ncells": float(gb.ncells.mean()), "charge": float(gb.charge.mean()),
                        "span0": float(gb.span0.mean()), "span1": float(gb.span1.mean())})
    return out


# ------------------------------------------------------------------------------------------------ C1 on the overlay
def _cells(f: pd.DataFrame, q: str, volume: int) -> np.ndarray:
    _, _, axis, _ = C1.QUANTITIES[q]
    x, xe = C1.axis_values(f, axis, volume)
    y, ye = C1.axis_values(f, C1.OTHER_AXIS[axis], volume)
    return (np.digitize(x, xe) * (len(ye) + 1) + np.digitize(y, ye)).astype(np.int64)


def at_test_incidence(ref: pd.DataFrame, test: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 1) -> list[dict]:
    """Per volume and C1 quantity: the reference reweighted to the test sample's joint distribution of both incidence
    angles (cells of the C1 bins of the binning variable times the bins of the other angle), the test statistic on
    the common cells and its bootstrap standard deviation (test hits resampled). Descriptive."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in C1.VOLUMES:
        rv, tv = ref[ref.volume == v], test[test.volume == v]
        if not len(rv) or not len(tv):
            continue
        for q, (col, stat, _, pix_only) in C1.QUANTITIES.items():
            if pix_only and v not in D.PIXEL_VOLUMES:
                continue
            cr, ct = _cells(rv, q, v), _cells(tv, q, v)
            m = int(max(cr.max(), ct.max())) + 1
            nr, nt = np.bincount(cr, minlength=m), np.bincount(ct, minlength=m)
            common = (nr > 0) & (nt > 0)
            if not common[ct].any():
                continue
            wr = np.where(common[cr], nt[cr] / np.maximum(nr[cr], 1), 0.0)
            xt = tv[col].to_numpy(np.float64)[common[ct]]
            ref_v = C1.wstat(rv[col].to_numpy(np.float64), wr, stat)
            test_v = C1.wstat(xt, np.ones(len(xt)), stat)
            boot = [C1.wstat(xt[i], np.ones(len(i)), stat) for i in rng.integers(0, len(xt), (n_boot, len(xt)))]
            scale = 1e3 if col in ("du", "dv") else 1.0
            rows.append({"volume": int(v), "quantity": q, "n_test": int(len(xt)), "coverage": float(common[ct].mean()),
                         "ref": ref_v * scale, "test": test_v * scale, "diff": (test_v - ref_v) * scale,
                         "sigma_test": float(np.std(boot)) * scale,
                         "rel_diff": (test_v - ref_v) / ref_v if q != "bias_u" and ref_v else float("nan"),
                         "unit": "um" if col in ("du", "dv") else ""})
    return rows


def c1_overlay(ref: pd.DataFrame, gun: pd.DataFrame) -> dict:
    cmp = C1.compare(ref, gun, matched=True)
    return {"n_gun_hits": int(len(gun)), "n_bins_counted": cmp["n_bins_counted"], "n_bins_failed": cmp["n_bins_failed"],
            "pass_on_counted_bins": cmp["pass"] if cmp["n_bins_counted"] else None,
            "t_eff": {int(v): r["t_eff"] for v, r in cmp["volumes"].items()},
            "bias_u": {int(v): r.get("bias_u") for v, r in cmp["volumes"].items()},
            "overall_gun": C1.overall(gun), "at_gun_incidence": at_test_incidence(ref, gun)}


# ------------------------------------------------------------------------------------------------ V3
def _stats(g: pd.DataFrame) -> dict:
    pix = np.isin(g.volume, D.PIXEL_VOLUMES)
    return {"n": int(len(g)), "ncells": float(g.ncells.mean()), "span0": float(g.span0.mean()),
            "span1": float(g.span1.mean()),
            "charge_cos_pixel": float(g.charge_cos[pix].mean()) if pix.any() else None,
            "mad_u_um": C1.mad(g.du) * 1e3, "mad_v_um_pixel": C1.mad(g.dv[pix]) * 1e3 if pix.any() else None}


def v3_tables(gun: pd.DataFrame, native: pd.DataFrame, min_hits: int = 5) -> dict:
    """Descriptive V3 tables (module docstring): per group, region and |tan| bin, and per group, region and signed-eta
    bin; bins with fewer than min_hits hits are left out."""
    groups = {"gun": gun, "gun_pp_targets": gun[gun.is_pp_target],
              "native_targets": native[native.is_target],
              "native_displaced_targets": native[native.is_target & ((native.d0.abs() >= PP_D0_MM)
                                                                     | (native.rv >= PP_RV_MM))]}
    out = {"incidence": [], "signed_eta": [], "by_region": []}
    for name, g in groups.items():
        g = g.assign(region=region(g.volume), inc=np.digitize(g.abs_tan, INC_EDGES) - 1,
                     etab=np.digitize(g.eta, ETA_EDGES) - 1)
        for reg, gr in g.groupby("region"):
            out["by_region"].append({"group": name, "region": reg, **_stats(gr)})
            for b, gb in gr.groupby("inc"):
                if len(gb) >= min_hits:
                    out["incidence"].append({"group": name, "region": reg,
                                             "abs_tan": f"{INC_EDGES[b]:g}-{INC_EDGES[b + 1]:g}", **_stats(gb)})
            for b, gb in gr.groupby("etab"):
                if len(gb) >= min_hits:
                    out["signed_eta"].append({"group": name, "region": reg,
                                              "eta": ["< -1", "-1 to 1", "> 1"][int(b)], **_stats(gb)})
    return out


def near_coincidences(ctx: dict) -> dict:
    """Hits with another particle's hit (any other pid, noise included) on the same module, and within NEAR_MM in 3D:
    gun hits against all other hits, and native target hits of the same events for comparison."""
    from scipy.spatial import cKDTree
    acc = {k: {"hits": 0, "module_shared": 0, **{f"within_{d:g}mm": 0 for d in NEAR_MM}}
           for k in ("gun", "gun_host_only", "native_targets")}
    for e in sorted(ctx["events"]):
        st = ctx["stored"][e]
        h = st["hits"].to_pandas()
        p = st["particles"].to_pandas()
        ng = ctx["events"][e]["n_gun_particles"]
        gpid = p.particle_id.to_numpy()[len(p) - ng:]
        tgt = p.particle_id.to_numpy()[: len(p) - ng][p.is_target.to_numpy(bool)[: len(p) - ng]]
        pid = h.pid.to_numpy(np.int64)
        mk = C.module_key(h.volume.to_numpy(), h.layer.to_numpy(), h.module.to_numpy())
        xyz = h[["x", "y", "z"]].to_numpy(np.float64)
        is_gun = np.isin(pid, gpid)
        groups = {"gun": np.flatnonzero(is_gun), "native_targets": np.flatnonzero(np.isin(pid, tgt))}
        # hits per module and per (module, particle): another particle's hit on the module
        ukey, inv = np.unique(mk, return_inverse=True)
        n_mod = np.bincount(inv)
        mp = pd.Series(1, index=pd.MultiIndex.from_arrays([mk, pid])).groupby(level=[0, 1]).size()
        own = mp.reindex(pd.MultiIndex.from_arrays([mk, pid])).to_numpy()
        noise_own = np.where(pid == 0, 1, own)            # noise hits are separate particles
        other_on_module = n_mod[inv] - noise_own > 0
        n_host_mod = np.bincount(inv, weights=(~is_gun).astype(np.float64))
        tree = cKDTree(xyz)
        pairs = tree.query_pairs(max(NEAR_MM), output_type="ndarray")
        a, b = pairs[:, 0], pairs[:, 1]
        ok = (mk[a] == mk[b]) & ((pid[a] != pid[b]) | (pid[a] == 0))
        a, b = a[ok], b[ok]
        dist = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        for name, rows in groups.items():
            acc[name]["hits"] += len(rows)
            acc[name]["module_shared"] += int(other_on_module[rows].sum())
            for d in NEAR_MM:
                near = np.zeros(len(h), bool)
                sel = dist <= d
                near[a[sel]] = True
                near[b[sel]] = True
                acc[name][f"within_{d:g}mm"] += int(near[rows].sum())
        rows = groups["gun"]
        acc["gun_host_only"]["hits"] += len(rows)
        acc["gun_host_only"]["module_shared"] += int((n_host_mod[inv[rows]] > 0).sum())
        for d in NEAR_MM:
            near = np.zeros(len(h), bool)
            sel = (dist <= d) & (is_gun[a] != is_gun[b])
            near[a[sel]] = True
            near[b[sel]] = True
            acc["gun_host_only"][f"within_{d:g}mm"] += int(near[rows].sum())
    for v in acc.values():
        for k in list(v):
            if k != "hits":
                v[k + "_frac"] = v[k] / v["hits"] if v["hits"] else float("nan")
    return acc


# ------------------------------------------------------------------------------------------------ content
def content(ctx: dict) -> dict:
    rows, tgt_dr = [], []
    for e in sorted(ctx["events"]):
        s = ctx["events"][e]
        g = gun_slices(ctx, e)
        p = g["particles"].to_pandas()
        y = g["ptype"].to_pandas()
        llp = (p.particle_id.to_numpy(np.int64) >> 52) - GUN.VERTEX0
        disp = (p.d0.abs() >= PP_D0_MM) | (p.rv >= PP_RV_MM)
        pp = p.is_target & disp & (llp <= GUN.N_ISO)
        pdd = p.is_target & disp & (llp > GUN.N_ISO)
        tr = ctx["gun_truth"][ctx["gun_truth"].event == e].set_index("particle_id").reindex(p.particle_id)
        t = p[p.is_target.to_numpy(bool)]
        if len(t):
            phi = np.arctan2(t.py, t.px).to_numpy(np.float64)
            eta, lt = t.eta.to_numpy(np.float64), llp[p.is_target.to_numpy(bool)]
            for i in range(len(t)):
                o = lt != lt[i]
                if o.any():
                    tgt_dr.append(float(GUN.delta_r(eta[i], phi[i], eta[o], phi[o]).min()))
        h = ctx["stored"][e]["hits"].to_pandas()
        mk = C.module_key(h.volume.to_numpy(), h.layer.to_numpy(), h.module.to_numpy())
        gh = np.isin(h.pid.to_numpy(), p.particle_id.to_numpy())
        rows.append({"event": e, "host": int(s["host"]), "gun_particles": int(len(p)),
                     "with_hits": int((p.n_hits > 0).sum()), "gun_hits": int(s["n_gun_hits"]),
                     "targets": int(p.is_target.sum()), "P_P": int(pp.sum()), "P_D": int(pdd.sum()),
                     "sim_failed": int((tr.sim_status == "failed").sum()),
                     "sim_below_pmin": int((tr.sim_status == "below_pmin").sum()),
                     "sim_hits": s["sim_hits"], "stopped_hits": s["stopped_hits"], "dropped_hits": s["dropped_hits"],
                     "no_cell_crossings": s["digitiser"]["n_no_cells"],
                     "modules_shared_with_host": int(len(np.intersect1d(mk[gh], mk[~gh]))),
                     "species_with_hits": pd.Series(np.abs(y.particle_type.to_numpy()[p.n_hits.to_numpy() > 0]))
                     .value_counts().to_dict()})
    df = pd.DataFrame(rows)
    gt = ctx["gun_truth"]
    failed = gt[gt.sim_status == "failed"].assign(pt=lambda d: np.hypot(d.px, d.py))
    n_llp = len(df) * (GUN.N_ISO + GUN.N_DENSE)
    tot = {k: int(df[k].sum()) for k in ("gun_particles", "with_hits", "gun_hits", "targets", "P_P", "P_D",
                                         "sim_failed", "sim_below_pmin", "sim_hits", "stopped_hits", "dropped_hits",
                                         "no_cell_crossings", "modules_shared_with_host")}
    return {"per_event": rows, "total": tot,
            "hits_per_llp": tot["gun_hits"] / n_llp,
            "P_P_per_isolated_llp": tot["P_P"] / (len(df) * GUN.N_ISO),
            "P_D_per_dense_llp": tot["P_D"] / (len(df) * GUN.N_DENSE),
            "expected_per_llp": EXPECTED_YIELD_PER_LLP,
            "sim_failed_fraction": tot["sim_failed"] / tot["gun_particles"],
            "sim_failed_particles": failed[["event", "particle_id", "llp", "component", "pdg", "r_v", "pt"]]
            .to_dict("records"),
            "min_dr_targets_of_different_llps": min(tgt_dr) if tgt_dr else None,
            "targets_with_other_llp_target_within_0.05": int(np.sum(np.array(tgt_dr) < 0.05))}


# ------------------------------------------------------------------------------------------------ driver
def cmd_run(a) -> dict:
    t0 = time.time()
    ctx = load_context(a.set, a.work, a.root)
    out = {"plan": "docs/pileup_overlay_plan.md section 10 (truth-only checks before any model run)",
           "set": a.set, "root": a.root, "work": a.work, "events": sorted(ctx["events"]),
           "hosts": sorted(int(s["host"]) for s in ctx["events"].values()),
           "emulation_hash": ctx["meta"]["emulation_hash"], "sim": ctx["meta"].get("sim")}
    out["format"] = format_checks(ctx)
    print(f"[checks] format pass={out['format']['pass']} ({time.time() - t0:.0f} s)", flush=True)
    out["weights"] = weight_rule_checks(ctx)
    print(f"[checks] weights pass={out['weights']['pass']} ({time.time() - t0:.0f} s)", flush=True)
    out["production"] = production_closure(ctx)
    print(f"[checks] production pass={out['production']['pass']} ({time.time() - t0:.0f} s)", flush=True)
    out["content"] = content(ctx)
    mods = D.Modules.load()
    gun = gun_hit_features(ctx, mods)
    ev = C1.calib_events(a.ref_events)
    ref = C1.trackml_hits(ev, mods)                                   # the C1 reference, by C1's own code
    native = native_hit_features(ev, mods)                            # the same hits with particle info (V3)
    mine, theirs = C1.overall(native), C1.overall(ref)
    out["native_features_equal_c1"] = all(
        abs(mine[v]["mad_u_um"] - theirs[v]["mad_u_um"]) < 1e-6 and abs(mine[v]["mad_v_um"] - theirs[v]["mad_v_um"])
        < 1e-6 and mine[v]["n"] == theirs[v]["n"] for v in theirs)
    out["v3"] = {"native_reference": {"events": ev, "note": "calibration training events (true hit kinematics are "
                                      "stored for these only)"}, **v3_tables(gun, native),
                 "same_event_clusters": same_event_clusters(ctx)}
    out["near_coincidences"] = near_coincidences(ctx)
    print(f"[checks] V3 done ({time.time() - t0:.0f} s)", flush=True)
    out["c1_overlay"] = {"reference": {"events": ev, "n_hits": int(len(ref))}, **c1_overlay(ref, gun)}
    print(f"[checks] C1 on the overlay: {out['c1_overlay']['n_bins_counted']} counted bins "
          f"({time.time() - t0:.0f} s)", flush=True)
    out["pass"] = {k: out[k]["pass"] for k in ("format", "weights", "production")}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"[checks] wrote {a.out_json} ({out['seconds']} s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=sorted(W.SETS))
    r.add_argument("--work", required=True)
    r.add_argument("--root", default=W.DERIVED_OVERLAY)
    r.add_argument("--out-json", required=True)
    r.add_argument("--ref-events", type=int, default=40)
    a = ap.parse_args(argv)
    cmd_run(a)


if __name__ == "__main__":
    main()
