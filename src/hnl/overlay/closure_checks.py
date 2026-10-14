"""Truth-only checks of a stored V1 closure set (hnl.overlay.closure; docs/pileup_overlay_plan.md section 10, gate V1
and "Converter") before any model run. No model is run and no model output is read. Inputs: the stored set
(ROOT/{hits,particles,ptype}/closure, meta, sidecar and closure truth), the work directory of
slurm/overlay/closure_set.sbatch (population table and its record, closure events, packed sim hits, the closure-only
CSV tar) and the hosts' tp_v1 tables.

Checks (pass or fail):
- population: the counts of the three populations per region over all 150 val events equal the plan's
  plan_revision/val_regions.json; the prompt subset holds 3,000 targets with no tie at the cut; the closure truth of
  every host equals the population table's rows of that host.
- ids: closure id = native id + 2^62 for every replaced native, positive and unique as int64, vertex field 1024-2024,
  disjoint from the host's remaining ids and from the gun range 1001-1010, exact inverse; the CSV particle member
  read back by pandas without a dtype is int64; barcodes (2, 0, 1..n, 0, 0), unique per event.
- format: hnl.overlay.closure.verify_closure_set; the frozen read path (hnl.trackml.events.load_events on the stored
  set, config.event_set on the explicit id list); rows = host - removed + closure; closure hits last; CSV hit ids
  1..M; stored closure positions = the CSV members; the host's true edges without those of the removed natives,
  renumbered, equal the stored host edges; closure true edges = sum over closure particles of (hits - 1); no mixed
  edge; X1 cluster features finite; every module in TrackML's detectors.csv; hosts are tp_v1 val events.
- weights: closure hits 0; host weights sum to 1 - W_removed within 1e-6, and the gate flags the sum 1.
- converter view: for every closure row, q, vx, vy, vz, px, py, pz, pt, eta, r_v, d0 and particle_type equal the
  native's exactly; vprim = native vprim + 1024; is_hard_scatter is False.
- production: every closure event recomputed from the packed sim hits with the set's seeds gives byte-identical CSV
  members, and their frozen conversion equals the stored closure rows.
Descriptive: Fatras status per population; category of the closure row against the native's; and the truth-level
closure of the re-simulated tracks against their native tracks (hit counts, target flag, pixel hits, r_first, the
set of crossed layers, the distance of the first hits, and cluster features of hits on the same layer).

    python -m hnl.overlay.closure_checks run --work DIR --root ROOT --out-json J
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import closure as CL
from hnl.overlay import digitise as D
from hnl.overlay import truth_checks as TC
from hnl.overlay import writer as W
from hnl.trackml import convert as C
from hnl.trackml import events as EV
from hnl.trackml import throughput as T

TP = T.DERIVED_TP
SET = "closure"


# ------------------------------------------------------------------------------------------------ inputs
def load_context(work: str, root: str, set_name: str = SET) -> dict:
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        side = json.load(fh)
    cdir = os.path.join(work, "closure")
    with open(os.path.join(cdir, "closure_events.json")) as fh:
        cev = json.load(fh)
    with open(os.path.join(work, "population_v1.json")) as fh:
        poprec = json.load(fh)
    events = {int(s["overlay_event"]): s for s in side["events"]}
    return {"set": set_name, "root": root, "work": work, "dir": cdir, "meta": meta, "side": side, "cev": cev,
            "population_record": poprec, "population": pd.read_parquet(os.path.join(work, "population_v1.parquet")),
            "events": events, "stored": {e: W.host_tables(e, root) for e in events},
            "hosts": {e: W.host_tables(s["host"], TP) for e, s in events.items()},
            "truth": pd.read_parquet(os.path.join(root, f"closure_truth_{set_name}.parquet")),
            "csv": TC.csv_members(os.path.join(work, f"closure_csv_{set_name}.tar")),
            "sim_path": os.path.join(cdir, "fatras_out", "simhits.parquet")}


def closure_slices(ctx: dict, e: int) -> dict:
    """The closure rows of a stored closure event (the last rows of each table)."""
    t, s = ctx["stored"][e], ctx["events"][e]
    nh, nc = s["n_closure_hits"], s["n_closure_particles"]
    return {"hits": t["hits"].slice(t["hits"].num_rows - nh),
            "particles": t["particles"].slice(t["particles"].num_rows - nc),
            "ptype": t["ptype"].slice(t["ptype"].num_rows - nc)}


# ------------------------------------------------------------------------------------------------ population
def population_checks(ctx: dict) -> dict:
    rec = ctx["population_record"]
    pop, truth = ctx["population"], ctx["truth"]
    out = {"val_regions": rec["val_regions"], "prompt_subset": rec["prompt_subset"], "selected_all_val": rec["selected"],
           "max_native_vertex_field": rec["max_native_vertex_field"],
           "particles_with_nonzero_process_field": rec["particles_with_nonzero_process_field"],
           "particles_read": rec["particles_read"], "population_table_sha256": rec["table_sha256"],
           "population_job": rec.get("slurm_job"), "population_sha256_matches": rec["table_sha256"]
           == ctx["cev"]["population_sha256"] == C.sha256_file(os.path.join(ctx["work"], "population_v1.parquet")),
           "per_host": {}, "problems": []}
    for e, s in sorted(ctx["events"].items()):
        p = pop[pop.event == s["host"]].sort_values("particle_id")
        t = truth[truth.event == e].sort_values("native_particle_id")
        same = (np.array_equal(p.particle_id.to_numpy(np.int64), t.native_particle_id.to_numpy(np.int64))
                and np.array_equal(p.population.to_numpy(), t.population.to_numpy())
                and np.array_equal(p.region.to_numpy(), t.region.to_numpy()))
        out["per_host"][int(s["host"])] = {"truth_equals_population_table": bool(same),
                                           **{k: int((t.population == k).sum()) for k in CL.POPULATIONS}}
        if not same:
            out["problems"].append(f"host {s['host']}: closure truth differs from the population table")
    if not rec["val_regions"]["equal"]:
        out["problems"].append("population counts differ from val_regions.json")
    ps = rec["prompt_subset"]
    if ps["n"] != CL.PROMPT_N or ps["tie_at_cut"]:
        out["problems"].append("prompt subset size or tie at the cut")
    if rec["max_native_vertex_field"] > CL.HOST_VERTEX_MAX:
        out["problems"].append("a val vertex field above the host range")
    if not out["population_sha256_matches"]:
        out["problems"].append("population table sha256")
    out["pass"] = not out["problems"]
    return out


# ------------------------------------------------------------------------------------------------ ids
def id_checks(ctx: dict) -> dict:
    from hnl.geometry.ids import decode_barcode
    out = {"per_event": {}, "problems": []}
    for e, s in sorted(ctx["events"].items()):
        t = ctx["truth"][ctx["truth"].event == e]
        st = ctx["stored"][e]["particles"].column("particle_id").to_numpy()
        nc = s["n_closure_particles"]
        host_ids, cl_ids = st[:len(st) - nc], st[len(st) - nc:]
        nat = t.native_particle_id.to_numpy(np.int64)
        cid = t.particle_id.to_numpy(np.int64)
        bc = decode_barcode(t.acts_barcode.to_numpy(np.uint64).view(np.int64))
        csv_p = pd.read_csv(io.BytesIO(gzip.decompress(ctx["csv"][e]["particles_gz"])))
        r = {"closure_particles": int(nc),
             "id_rule": bool(np.array_equal(cid, nat + (1 << CL.CLOSURE_BIT))),
             "inverse": bool(np.array_equal(CL.native_particle_id(cid), nat)),
             "stored_equal_truth": bool(np.array_equal(np.sort(cl_ids), np.sort(cid))),
             "positive_unique_int64": bool(st.dtype == np.int64 and (st > 0).all() and len(np.unique(st)) == len(st)),
             "closure_vertex_range": [int((cid >> 52).min()), int((cid >> 52).max())],
             "host_vertex_max": int((host_ids >> 52).max()),
             "collides_with_host": int(np.isin(cid, host_ids).sum()),
             "in_gun_range": int((((cid >> 52) >= W.GUN_VERTEX[0]) & ((cid >> 52) <= W.GUN_VERTEX[1])).sum()),
             "csv_particle_id_dtype_default_read": str(csv_p.particle_id.dtype),
             "barcodes": bool((bc["vprim"] == CL.BARCODE_VPRIM).all() and (bc["vsec"] == 0).all()
                              and np.array_equal(np.sort(bc["particle"]), np.arange(1, nc + 1))
                              and (bc["gen"] == 0).all() and (bc["sub"] == 0).all())}
        out["per_event"][e] = r
        bad = [k for k in ("id_rule", "inverse", "stored_equal_truth", "positive_unique_int64", "barcodes") if not r[k]]
        if not (CL.CLOSURE_VERTEX[0] <= r["closure_vertex_range"][0] and r["closure_vertex_range"][1]
                <= CL.CLOSURE_VERTEX[1]):
            bad.append("closure vertex range")
        if r["host_vertex_max"] > CL.HOST_VERTEX_MAX or r["collides_with_host"] or r["in_gun_range"]:
            bad.append("collision")
        if r["csv_particle_id_dtype_default_read"] != "int64":
            bad.append("CSV ids not read as int64")
        out["problems"] += [f"event {e}: {b}" for b in bad]
    out["pass"] = not out["problems"]
    return out


# ------------------------------------------------------------------------------------------------ format and weights
def format_checks(ctx: dict) -> dict:
    from hnl.trackml.config import event_set
    det = C.load_detectors()
    ids = sorted(ctx["events"])
    v = CL.verify_closure_set(ctx["set"], ctx["root"])
    evs = EV.load_events(ids, root=ctx["root"])
    val = set(event_set("val"))
    out = {"verify_problems": v["problems"], "events": len(ids),
           "event_set_explicit_list": event_set(",".join(str(e) for e in ids)) == ids,
           "hosts_in_tp_v1_val": all(int(s["host"]) in val for s in ctx["events"].values()), "per_event": {}}
    problems = []
    for e in ids:
        s, st, ht = ctx["events"][e], ctx["stored"][e], ctx["hosts"][e]
        ev = evs[e]
        hev = EV.load_events([s["host"]], root=TP)[s["host"]]
        natives = np.asarray(s["native_particle_ids"], np.int64)
        hp = ht["hits"].column("pid").to_numpy()
        keep = ~np.isin(hp, natives)
        row_map = np.where(keep, np.cumsum(keep) - 1, -1)
        nkeep, nc_h = int(keep.sum()), s["n_closure_hits"]
        p = st["particles"].to_pandas()
        cl = p.iloc[len(p) - s["n_closure_particles"]:]
        cl_hit = np.isin(ev.pid, cl.particle_id.to_numpy())
        te = ev.true_edges
        cte = cl_hit[te[0]] & cl_hit[te[1]]
        mixed = cl_hit[te[0]] != cl_hit[te[1]]
        hte = hev.true_edges
        hte_kept = hte[:, keep[hte[0]] & keep[hte[1]]]
        partial = int((keep[hte[0]] != keep[hte[1]]).sum())
        feat_ok = {}
        for f in EV.CLUSTER_FEATURES:
            try:
                feat_ok[f] = bool(np.isfinite(EV.cluster_feature(ev, f)).all())
            except (ValueError, KeyError) as exc:
                feat_ok[f] = f"raises: {exc}"
        known = np.isin(C.module_key(ev.volume, ev.layer, ev.module), det["key"])
        cm = ctx["csv"][e]["hits"]
        xyz_st = np.stack([ev.x, ev.y, ev.z], axis=1)[nkeep:]
        w = ev.weight.astype(np.float64)
        sum_host = float(np.nansum(w[~cl_hit]))
        r = {"rows": int(ev.n), "host_rows_kept": nkeep, "removed_host_hits": int((~keep).sum()),
             "closure_hits": int(nc_h), "rows_equal_kept_plus_closure": ev.n == nkeep + nc_h,
             "closure_hits_last_rows": bool(cl_hit[nkeep:].all() and not cl_hit[:nkeep].any()),
             "csv_hit_ids_1_to_m": bool(np.array_equal(cm.hit_id.to_numpy(), np.arange(1, len(cm) + 1))),
             "csv_hits_equal_stored": bool(len(cm) == nc_h and np.array_equal(
                 cm[["x", "y", "z"]].to_numpy(np.float64).astype(np.float32), xyz_st)),
             "host_edges_of_removed_natives": int(hte.shape[1] - hte_kept.shape[1]),
             "host_edges_crossing_the_removal": partial,
             "host_true_edges_equal_after_removal": bool(np.array_equal(row_map[hte_kept], te[:, ~cte & ~mixed])),
             "closure_true_edges": int(cte.sum()),
             "closure_true_edges_expected": int(np.maximum(cl.n_hits.to_numpy(np.int64) - 1, 0).sum()),
             "mixed_true_edges": int(mixed.sum()), "cluster_features_finite": feat_ok,
             "hits_on_unknown_modules": int((~known).sum()),
             "closure_weight_max_abs": float(np.abs(w[cl_hit]).max()) if cl_hit.any() else 0.0,
             "host_weight_sum": sum_host, "removed_weight": s["removed_weight"],
             "host_weight_sum_minus_expected": sum_host - (1.0 - s["removed_weight"]),
             "gate_flags_unadjusted_sum": any("weight sum" in x for x in CL.check_closure_event(
                 st, ht, natives, 1.0))}
        out["per_event"][e] = r
        bad = [k for k in ("rows_equal_kept_plus_closure", "closure_hits_last_rows", "csv_hit_ids_1_to_m",
                           "csv_hits_equal_stored", "host_true_edges_equal_after_removal") if not r[k]]
        bad += [f"cluster feature {k}" for k, x in feat_ok.items() if x is not True]
        if r["closure_true_edges"] != r["closure_true_edges_expected"] or r["mixed_true_edges"] or partial:
            bad.append("true edges")
        if r["hits_on_unknown_modules"]:
            bad.append("unknown modules")
        if r["closure_weight_max_abs"] != 0 or abs(r["host_weight_sum_minus_expected"]) > W.WEIGHT_TOL:
            bad.append("weights")
        if s["removed_weight"] > W.WEIGHT_TOL and not r["gate_flags_unadjusted_sum"]:
            bad.append("the gate does not flag the unadjusted weight sum")
        problems += [f"event {e}: {b}" for b in bad]
    for k in ("event_set_explicit_list", "hosts_in_tp_v1_val"):
        if not out[k]:
            problems.append(k)
    out["problems"] = v["problems"] + problems
    out["pass"] = not out["problems"]
    out["max_abs_weight_sum_minus_expected"] = float(max(abs(r["host_weight_sum_minus_expected"])
                                                         for r in out["per_event"].values()))
    return out


# ------------------------------------------------------------------------------------------------ converter view
def pairs(ctx: dict) -> pd.DataFrame:
    """One row per replaced native: its stored host row and its stored closure row (suffixes _n, _c), with the
    closure truth (population, region, Fatras status)."""
    rows = []
    for e, s in sorted(ctx["events"].items()):
        hp = ctx["hosts"][e]["particles"].to_pandas()
        hp["particle_type"] = ctx["hosts"][e]["ptype"].column("particle_type").to_numpy()
        cs = closure_slices(ctx, e)
        cp = cs["particles"].to_pandas()
        cp["particle_type"] = cs["ptype"].column("particle_type").to_numpy()
        cp["native_particle_id"] = CL.native_particle_id(cp.particle_id.to_numpy(np.int64))
        t = ctx["truth"][ctx["truth"].event == e][["native_particle_id", "population", "region", "sim_status"]]
        j = t.merge(hp.rename(columns={"particle_id": "native_particle_id"}), on="native_particle_id", how="left",
                    validate="one_to_one").merge(cp, on="native_particle_id", how="left", suffixes=("_n", "_c"),
                                                 validate="one_to_one")
        rows.append(j.assign(overlay_event=e))
    return pd.concat(rows, ignore_index=True)


def converter_view(ctx: dict, pr: pd.DataFrame) -> dict:
    out = {"particles": int(len(pr)), "problems": []}
    eq = {}
    for k in (*CL.KINEMATICS, "particle_type"):
        a, b = pr[f"{k}_c"].to_numpy(), pr[f"{k}_n"].to_numpy()
        eq[k] = bool(a.dtype == b.dtype and np.array_equal(a, b))
    out["exactly_equal"] = eq
    out["vprim_offset_1024"] = bool(np.array_equal(pr.vprim_c.to_numpy(np.int64), pr.vprim_n.to_numpy(np.int64) + 1024))
    out["is_hard_scatter_closure"] = int(pr.is_hard_scatter_c.sum())
    out["is_hard_scatter_native"] = int(pr.is_hard_scatter_n.sum())
    out["category"] = {pop: {f"{a} -> {b}": int(n) for (a, b), n in g.groupby(["category_n", "category_c"]).size()
                             .items()} for pop, g in pr.groupby("population")}
    out["problems"] = [f"{k} differs" for k, v in eq.items() if not v]
    if not out["vprim_offset_1024"]:
        out["problems"].append("vprim offset")
    if out["is_hard_scatter_closure"]:
        out["problems"].append("closure rows flagged hard scatter")
    out["pass"] = not out["problems"]
    return out


# ------------------------------------------------------------------------------------------------ production
def production_replay(ctx: dict) -> dict:
    from hnl.overlay import emulate as EM
    meta, cev = ctx["meta"], ctx["cev"]
    tab = EM.Tables.load(meta["emulation_tables"])
    out = {"emulation_hash_now": EM.emulation_hash(meta["emulation_tables"]),
           "emulation_hash_stored": meta["emulation_hash"],
           "overlay_hash_now": W.overlay_hash(meta["emulation_tables"]), "overlay_hash_stored": meta["overlay_hash"],
           "closure_code_sha256_now": C.sha256_file(os.path.abspath(CL.__file__)),
           "closure_code_sha256_stored": meta.get("closure_code_sha256"), "per_event": {}, "problems": []}
    mods, lookup, det = D.Modules.load(), D.load_lookup(meta["lookup"]), C.load_detectors()
    sim = pd.read_parquet(ctx["sim_path"])
    for ev in cev["events"]:
        e, fe = int(ev["overlay_event"]), int(ev["fatras_event"])
        fp = pd.read_csv(W.FIO_path(ctx["dir"], fe), dtype={"particle_id": np.uint64})
        rows = CL.closure_rows(ctx["truth"][ctx["truth"].event == e], fp)
        g = CL.closure_event(sim[sim.event == fe], rows, lookup, mods, tab, int(cev["base_seed"]), e)
        same = all(g["gz"][k] == ctx["csv"][e][k + "_gz"] for k in ("hits", "cells", "particles", "truth"))
        ctab, _ = W.convert_gun(g["gz"], e, det)
        stored = closure_slices(ctx, e)
        equal = {k: W.tables_equal(ctab[k].replace_schema_metadata(None), stored[k]) for k in W.TABLES}
        out["per_event"][e] = {"csv_bytes_identical": same, "stored_rows_equal_conversion": equal}
        if not (same and all(equal.values())):
            out["problems"].append(f"event {e}")
    for k in ("emulation_hash", "overlay_hash", "closure_code_sha256"):
        if out[f"{k}_now"] != out[f"{k}_stored"]:
            out["problems"].append(f"{k} changed since the set was written")
    out["pass"] = not out["problems"]
    return out


# ------------------------------------------------------------------------------------------------ descriptive
def sim_status(pr: pd.DataFrame) -> dict:
    return {pop: {str(k): int(v) for k, v in g.sim_status.value_counts().items()} for pop, g in pr.groupby(
        "population")}


def track_pairs(ctx: dict, pr: pd.DataFrame) -> pd.DataFrame:
    """Per replaced native: hit counts, layers crossed, first-hit distance, against its re-simulated track."""
    rows = []
    for e in sorted(ctx["events"]):
        hh = ctx["hosts"][e]["hits"].to_pandas()
        sh = closure_slices(ctx, e)["hits"].to_pandas()
        nh = {k: g for k, g in hh[hh.pid.isin(pr.native_particle_id)].groupby("pid")}
        ch = {k: g for k, g in sh.groupby("pid")}
        for r in pr[pr.overlay_event == e].itertuples():
            a = nh.get(r.native_particle_id, hh.iloc[:0])
            b = ch.get(CL.closure_particle_id(r.native_particle_id), sh.iloc[:0])
            la, lb = set(zip(a.volume, a.layer)), set(zip(b.volume, b.layer))
            d_first = np.nan
            if len(a) and len(b):
                fa, fb = a[a["rank"] == 0].iloc[0], b[b["rank"] == 0].iloc[0]
                if (fa.volume, fa.layer) == (fb.volume, fb.layer):
                    d_first = float(np.linalg.norm(np.array([fa.x - fb.x, fa.y - fb.y, fa.z - fb.z], np.float64)))
            rows.append({"population": r.population, "region": r.region, "sim_status": r.sim_status,
                         "n_hits_n": int(r.n_hits_n), "n_hits_c": int(r.n_hits_c),
                         "n_pixel_n": int(r.n_pixel_hits_n), "n_pixel_c": int(r.n_pixel_hits_c),
                         "r_first_n": float(r.r_first_n), "r_first_c": float(r.r_first_c),
                         "target_c": bool(r.is_target_c), "layers_n": len(la), "layers_c": len(lb),
                         "layers_shared": len(la & lb), "same_layers": la == lb, "first_hit_same_layer": not np.isnan(
                             d_first), "d_first_mm": d_first})
    return pd.DataFrame(rows)


def hit_pairs(ctx: dict, pr: pd.DataFrame) -> pd.DataFrame:
    """Hits of a native and of its re-simulated track on the same (volume, layer), where each crosses that layer
    once: position distance and cluster features."""
    rows = []
    for e in sorted(ctx["events"]):
        hh = ctx["hosts"][e]["hits"].to_pandas()
        sh = closure_slices(ctx, e)["hits"].to_pandas()
        sh["native"] = CL.native_particle_id(sh.pid.to_numpy(np.int64))
        a = hh[hh.pid.isin(pr[pr.overlay_event == e].native_particle_id)]
        a = a[~a.duplicated(["pid", "volume", "layer"], keep=False)]
        b = sh[~sh.duplicated(["pid", "volume", "layer"], keep=False)]
        j = a.merge(b, left_on=["pid", "volume", "layer"], right_on=["native", "volume", "layer"],
                    suffixes=("_n", "_c"))
        if len(j):
            rows.append(pd.DataFrame({
                "region": TC.region(j.volume), "volume": j.volume.to_numpy(),
                "d_mm": np.linalg.norm(j[["x_n", "y_n", "z_n"]].to_numpy(np.float64)
                                       - j[["x_c", "y_c", "z_c"]].to_numpy(np.float64), axis=1),
                **{f"{k}_{s}": j[f"{k}_{s}"].to_numpy(np.float64) for k in ("ncells", "span0", "span1", "charge")
                   for s in ("n", "c")}}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise_tracks(tp: pd.DataFrame) -> list[dict]:
    out = []
    for key, g in [(("all", "all"), tp), *[((p, "all"), g) for p, g in tp.groupby("population")],
                   *[((p, r), g) for (p, r), g in tp.groupby(["population", "region"])]]:
        sim = g[g.sim_status == "simulated"]
        out.append({"population": key[0], "region": key[1], "natives": int(len(g)),
                    "fatras_failed": int((g.sim_status == "failed").sum()),
                    "mean_hits_native": float(g.n_hits_n.mean()), "mean_hits_closure": float(sim.n_hits_c.mean()),
                    "mean_hits_diff": float((sim.n_hits_c - sim.n_hits_n).mean()),
                    "sem_hits_diff": float((sim.n_hits_c - sim.n_hits_n).std(ddof=1) / np.sqrt(len(sim)))
                    if len(sim) > 1 else None,
                    "closure_target_frac": float(sim.target_c.mean()) if len(sim) else None,
                    "mean_pixel_hits_native": float(g.n_pixel_n.mean()),
                    "mean_pixel_hits_closure": float(sim.n_pixel_c.mean()),
                    "same_layer_set_frac": float(sim.same_layers.mean()) if len(sim) else None,
                    "layers_shared_frac": float(sim.layers_shared.sum() / max(sim.layers_n.sum(), 1)),
                    "first_hit_same_layer_frac": float(sim.first_hit_same_layer.mean()) if len(sim) else None,
                    "median_first_hit_distance_mm": float(np.nanmedian(sim.d_first_mm))
                    if sim.first_hit_same_layer.any() else None,
                    "median_abs_r_first_diff_mm": float(np.nanmedian(np.abs(sim.r_first_c - sim.r_first_n)))
                    if len(sim) else None})
    return out


ABS_ETA_BANDS = (0.0, 1.0, 2.0, 3.0, 4.0)


def hits_per_layer(ctx: dict) -> list[dict]:
    """Per detector region and |eta| band of the native: hits and crossed layers of the native track, of its Fatras
    sim hits before the emulation (mapped to TrackML modules with the set's lookup) and of the stored closure track.
    Separates what Fatras records from what the emulation removes."""
    lookup = D.load_lookup(ctx["meta"]["lookup"])
    sim = pd.read_parquet(ctx["sim_path"], columns=["event", "particle_id", "geometry_id"])
    vol, lay, _ = lookup.from_geometry_id(sim.geometry_id.to_numpy(np.uint64))
    sim = sim.assign(volume=vol, layer=lay)
    rows = []
    for e in sorted(ctx["events"]):
        hh = ctx["hosts"][e]["hits"].to_pandas()
        sh = closure_slices(ctx, e)["hits"].to_pandas()
        t = ctx["truth"][ctx["truth"].event == e]
        se = sim[sim.event == int(t.fatras_event.iloc[0])]
        src = {"native": hh[hh.pid.isin(t.native_particle_id)].assign(key=lambda d: d.pid),
               "sim": se.assign(key=se.particle_id.map(dict(zip(t.acts_barcode.to_numpy(np.uint64).tolist(),
                                                                t.native_particle_id.tolist())))),
               "closure": sh.assign(key=CL.native_particle_id(sh.pid.to_numpy(np.int64)))}
        eta = dict(zip(t.native_particle_id, t.eta))
        for name, h in src.items():
            g = h.assign(region=TC.region(h.volume), vl=h.volume.to_numpy(np.int64) * 1000
                         + h.layer.to_numpy(np.int64)).groupby(["key", "region"]).agg(
                hits=("vl", "size"), layers=("vl", "nunique"))
            for (k, reg), r in g.iterrows():
                rows.append({"source": name, "native": int(k), "region": reg, "abs_eta": abs(float(eta[k])),
                             "hits": int(r.hits), "layers": int(r.layers)})
    d = pd.DataFrame(rows)
    d["band"] = pd.cut(d.abs_eta, ABS_ETA_BANDS, right=False).astype(str)
    out = []
    for (reg, band), g in d.groupby(["region", "band"]):
        rec = {"region": reg, "abs_eta": band}
        for name in ("native", "sim", "closure"):
            x = g[g.source == name]
            rec.update({f"tracks_{name}": int(len(x)), f"hits_{name}": int(x.hits.sum()),
                        f"layers_{name}": int(x.layers.sum()),
                        f"hits_per_layer_{name}": float(x.hits.sum() / x.layers.sum()) if len(x) else None})
        out.append(rec)
    return out


def summarise_hits(hp: pd.DataFrame) -> list[dict]:
    out = []
    if not len(hp):
        return out
    for reg, g in [("all", hp), *hp.groupby("region")]:
        pix = np.isin(g.volume, D.PIXEL_VOLUMES)
        out.append({"region": reg, "hit_pairs": int(len(g)), "median_distance_mm": float(np.median(g.d_mm)),
                    **{f"mean_{k}_{s}": float(g[f"{k}_{s}"].mean()) for k in ("ncells", "span0", "span1")
                       for s in ("n", "c")},
                    "mean_charge_ratio_pixel": float((g.charge_c[pix] / g.charge_n[pix]).mean()) if pix.any()
                    else None})
    return out


# ------------------------------------------------------------------------------------------------ driver
def cmd_run(a) -> dict:
    t0 = time.time()
    ctx = load_context(a.work, a.root)
    out = {"plan": "docs/pileup_overlay_plan.md section 10, gate V1 (truth-only checks of the closure machinery; no "
                   "model run, no model output read)", "set": ctx["set"], "root": a.root, "work": a.work,
           "events": sorted(ctx["events"]), "hosts": sorted(int(s["host"]) for s in ctx["events"].values()),
           "base_seed": ctx["cev"]["base_seed"], "emulation_hash": ctx["meta"]["emulation_hash"],
           "overlay_hash": ctx["meta"]["overlay_hash"], "sim": ctx["meta"]["sim"],
           "id_rule": ctx["meta"]["id_rule"], "barcode_rule": ctx["meta"]["barcode_rule"]}
    out["population"] = population_checks(ctx)
    out["ids"] = id_checks(ctx)
    out["format"] = format_checks(ctx)
    print(f"[closure checks] population {out['population']['pass']} ids {out['ids']['pass']} format "
          f"{out['format']['pass']} ({time.time() - t0:.0f} s)", flush=True)
    pr = pairs(ctx)
    out["converter_view"] = converter_view(ctx, pr)
    out["production"] = production_replay(ctx)
    print(f"[closure checks] converter view {out['converter_view']['pass']} production {out['production']['pass']} "
          f"({time.time() - t0:.0f} s)", flush=True)
    out["sim_status"] = sim_status(pr)
    out["per_event"] = [{"overlay_event": int(e), "host": int(s["host"]), **s["populations"],
                         "removed_host_hits": s["removed_host_hits"], "removed_weight": s["removed_weight"],
                         "closure_hits": s["n_closure_hits"], "sim_hits": s["sim_hits"], "stopped_hits": s["stopped_hits"],
                         "dropped_hits": s["dropped_hits"], "sim_failed": len(s["sim_failed_particles"])}
                        for e, s in sorted(ctx["events"].items())]
    tp = track_pairs(ctx, pr)
    out["tracks"] = summarise_tracks(tp)
    out["hits"] = summarise_hits(hit_pairs(ctx, pr))
    out["hits_per_layer"] = hits_per_layer(ctx)
    out["pass"] = {k: out[k]["pass"] for k in ("population", "ids", "format", "converter_view", "production")}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"[closure checks] pass {out['pass']}; wrote {a.out_json} ({out['seconds']} s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--root", required=True)
    r.add_argument("--out-json", required=True)
    cmd_run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
