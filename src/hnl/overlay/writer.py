"""Overlay writer of the TrackML pileup version (docs/pileup_overlay_plan.md sections 5.3, 5.4 and 6; decisions D9,
D10): the gun-only event as the four TrackML CSV members, converted by the frozen converter, its rows appended to the
host's rows.

Per overlay event:
1. `gun_members`: the digitised gun hits (hnl.overlay.digitise.Digitised; particle ids are ACTS barcodes) and the gun
   particles become TrackML's hits, cells, particles and truth members. Particle ids are re-encoded
   (hnl.overlay.gun.trackml_particle_id), hits are sorted by module (volume, layer, module) as TrackML sorts them
   (stable within a module; TrackML has no order there), hit ids run 1..M, every truth weight is 0 and nhits counts the
   hits written. Particles without hits keep their row (nhits 0), as in TrackML.
2. `convert_gun`: the gzip CSV members through the frozen hnl.trackml.throughput.event_tables_tp (= parse_members,
   hnl.trackml.convert.event_tables and throughput.ptype_table) with score weights kept and no truth table. An event
   that fails the converter's data-quality rule raises.
3. `merge_event`: the host's stored rows (hits, particles, ptype of derived/tp_v1, read unchanged and in stored order)
   followed by the gun rows. Only the event column changes, to the overlay id. Gun hit rows (graph nodes) therefore
   continue after the host's: hit id = n_host + gun hit id. Closure events may drop host particles (hits and rows);
   the row map is returned.
4. `check_event`: the converter gate of plan section 10 on the merged tables, read back as int64 (unique positive
   particle ids, gun vertex field 1001-1010, gun weights 0, host weights summing to the expected value within 1e-6,
   n_hits equal to the hit count, host rows unchanged).
Before step 1, `sim_provenance` checks the Fatras run (seed = base seed + 13; a module table that belongs to the
build) and `sim_status` finds the gun particles Fatras did not simulate (failed propagation or p <= pMin): they keep
their particle row with no hits and are listed in the sidecar (sim_failed_particles, sim_below_pmin_particles).
5. `write_set`: <root>/{hits,particles,ptype}/<set>/shard_<first>.parquet (first = event // 500 * 500, one row group
   per event, increasing ids; the schemas and encodings of the converter) and meta_<set>.json in the layout of the
   converter's meta files, so hnl.trackml.convert.read_table(table, events, root=<root>) reads them unchanged. The
   default root is derived/overlay_v1 (D9); hnl.trackml.events.table_root sends ids >= OVERLAY_FIRST_ID there.
   `verify_set` re-reads the stored shards and repeats the checks.

Event ids: overlay id = 50,000,000 + 1,000,000 k + host (plan section 5.4) with the set index k of SETS.

    python -m hnl.overlay.writer assemble --set pilot --gun-dir DIR --sim DIR/fatras_out/simhits.parquet \
        --root ROOT [--tables configs/trackml/overlay_emulation_v1.json] [--lookup LOOKUP.csv]
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import os
import tarfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hnl.data.convert import SHARD
from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun as GUN
from hnl.trackml import convert as C
from hnl.trackml import throughput as T
from hnl.trackml.split import EXT, REPO

OVERLAY_FIRST_ID = 50_000_000                       # = hnl.trackml.events.OVERLAY_FIRST_ID
DERIVED_OVERLAY = os.path.join(EXT, "derived", "overlay_v1")
TABLES = ("hits", "particles", "ptype")
SETS = {"hval": (0, 650001), "closure": (1, 950001), "htest": (2, 750001), "htrain": (3, 850001), "pilot": (9, None)}
CSV_COLUMNS = {"hits": ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"],
               "cells": ["hit_id", "ch0", "ch1", "value"],
               "particles": ["particle_id", "particle_type", "vx", "vy", "vz", "px", "py", "pz", "q", "nhits"],
               "truth": ["hit_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"]}
CSV_FLOAT = {"hits": "%.6g", "cells": "%.6g", "particles": "%.9g", "truth": "%.6g"}
WEIGHT_TOL = 1e-6
EMULATION_STREAM = 1                                # third seed word of the emulation's generator (see assemble)
GUN_VERTEX = (GUN.VERTEX0 + GUN.LLP_INDICES[0], GUN.VERTEX0 + GUN.LLP_INDICES[-1])


def overlay_id(set_name: str, host: int) -> int:
    k = SETS[set_name][0]
    host = int(host)
    if not 0 <= host < 1_000_000:
        raise ValueError(f"host id {host} outside 0..999999")
    return OVERLAY_FIRST_ID + 1_000_000 * k + host


def split_overlay_id(oid: int) -> tuple[int, int]:
    d = int(oid) - OVERLAY_FIRST_ID
    if d < 0:
        raise ValueError(f"{oid} is not an overlay id")
    return d // 1_000_000, d % 1_000_000


# ------------------------------------------------------------------------------------------------ gun members
def gun_members(dig: D.Digitised, particles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """TrackML CSV members of a gun-only event.

    dig: the digitiser output (hits.particle_id = ACTS barcode of the sim hit). particles: the gun's Fatras input
    (FIO.PARTICLE_COLUMNS, ACTS barcodes) or any table with particle_id (barcode), particle_type, vx, vy, vz, px, py,
    pz, q. Charged particles only (the converter needs q = +-1)."""
    h = dig.hits.reset_index(drop=True)
    n = len(h)
    bc_p = particles.particle_id.to_numpy().astype(np.uint64)
    if pd.Series(bc_p).duplicated().any():
        raise ValueError("duplicate gun barcode")
    tml_p = GUN.trackml_particle_id(bc_p)
    bc_h = h.particle_id.to_numpy().astype(np.uint64)
    known = pd.Index(bc_p).get_indexer(bc_h)
    if (known < 0).any():
        raise KeyError(f"{int((known < 0).sum())} gun hits of particles not in the particle table")
    o = np.lexsort((np.arange(n), h.module_id.to_numpy(np.int64), h.layer_id.to_numpy(np.int64),
                    h.volume_id.to_numpy(np.int64)))
    hs = h.iloc[o].reset_index(drop=True)
    hs["particle_id"] = tml_p[known[o]]
    inv = np.empty(n, np.int64)
    inv[o] = np.arange(n)
    c = dig.cells
    c = c[c.hit.to_numpy() >= 0]
    cs = pd.DataFrame({"hit": inv[c.hit.to_numpy(np.int64)], "ch0": c.ch0.to_numpy(), "ch1": c.ch1.to_numpy(),
                       "value": c.value.to_numpy()}).sort_values(["hit", "ch0", "ch1"], kind="stable")
    m = D.trackml_members(D.Digitised(hits=hs, cells=cs.reset_index(drop=True)), first_hit_id=1)
    m["truth"]["particle_id"] = hs.particle_id.to_numpy(np.int64)
    nh = pd.Series(hs.particle_id.to_numpy()).value_counts()
    q = particles.q.to_numpy(np.int64)
    if not np.isin(q, (-1, 1)).all():
        raise ValueError("gun particles must have charge +-1")
    m["particles"] = pd.DataFrame({
        "particle_id": tml_p.astype(np.int64), "particle_type": particles.particle_type.to_numpy(np.int64),
        **{k: particles[k].to_numpy(np.float64) for k in ("vx", "vy", "vz", "px", "py", "pz")},
        "q": q, "nhits": nh.reindex(tml_p).fillna(0).to_numpy(np.int64)})
    return {k: m[k][CSV_COLUMNS[k]] for k in ("hits", "cells", "particles", "truth")}


def members_gz(members: dict[str, pd.DataFrame]) -> dict[str, bytes]:
    """gzip CSV bytes of the four members (the throughput archives' member format; mtime 0 for reproducibility)."""
    out = {}
    for k in ("hits", "cells", "particles", "truth"):
        if list(members[k].columns) != CSV_COLUMNS[k]:
            raise ValueError(f"{k}: columns {list(members[k].columns)}")
        buf = io.StringIO()
        members[k].to_csv(buf, index=False, float_format=CSV_FLOAT[k])
        out[k] = gzip.compress(buf.getvalue().encode(), mtime=0)
    return out


def convert_gun(gz: dict[str, bytes], event: int, det: dict) -> tuple[dict[str, pa.Table], dict]:
    """The frozen conversion of a gun-only event (module docstring, step 2)."""
    tables, dq = T.event_tables_tp(gz, int(event), det, keep_weight=True, keep_truth=False)
    if tables is None:
        raise RuntimeError(f"gun-only event {event} failed the converter: {dq['problems']}")
    return tables, dq


# ------------------------------------------------------------------------------------------------ host rows
def host_tables(host: int, root: str = T.DERIVED_TP) -> dict[str, pa.Table]:
    """The stored rows of one host event (hits, particles, ptype), unchanged and in stored order."""
    out = {}
    for k in TABLES:
        arch = C.event_index(root, k).get(int(host))
        if arch is None:
            raise KeyError(f"host {host} not stored in {root} ({k})")
        path = C.shard_path(root, k, arch, int(host) // SHARD * SHARD)
        f = pq.ParquetFile(path)
        col = f.schema_arrow.get_field_index("event")
        rgs = [i for i in range(f.metadata.num_row_groups)
               if f.metadata.row_group(i).column(col).statistics.min == int(host)]
        if len(rgs) != 1:
            raise RuntimeError(f"host {host}: {len(rgs)} row groups in {path}")
        t = f.read_row_group(rgs[0])
        if not t.schema.equals(T.SCHEMAS[k], check_metadata=False):
            raise RuntimeError(f"{path}: schema differs from the converter's")
        out[k] = t.replace_schema_metadata(None)
    return out


def _with_event(t: pa.Table, event: int) -> pa.Table:
    i = t.schema.get_field_index("event")
    return t.set_column(i, "event", pa.array(np.full(t.num_rows, event, np.int32)))


def merge_event(host: dict[str, pa.Table], gun: dict[str, pa.Table], event: int,
                drop_pids=None) -> tuple[dict[str, pa.Table], dict]:
    """Host rows (event set to the overlay id; optionally without the hits and rows of drop_pids) followed by the gun
    rows. Returns the merged tables and the row map (host hit row -> overlay hit row, -1 for a dropped hit)."""
    drop = np.unique(np.asarray([] if drop_pids is None else drop_pids, np.int64))
    hp = host["hits"].column("pid").to_numpy()
    keep_h = ~np.isin(hp, drop) if len(drop) else np.ones(len(hp), bool)
    row_map = np.where(keep_h, np.cumsum(keep_h) - 1, -1)
    out = {}
    for k in TABLES:
        ht = host[k]
        if len(drop):
            col = "pid" if k == "hits" else "particle_id"
            mask = ~np.isin(ht.column(col).to_numpy(), drop)
            ht = ht.filter(pa.array(mask))
        gt = gun[k]
        if not gt.schema.equals(ht.schema, check_metadata=False):
            raise RuntimeError(f"{k}: gun and host schemas differ")
        if not (pc.all(pc.equal(gt.column("event"), int(event))).as_py() if gt.num_rows else True):
            raise RuntimeError(f"{k}: gun rows carry another event id")
        out[k] = pa.concat_tables([_with_event(ht, event), gt.replace_schema_metadata(None)]).combine_chunks()
    info = {"event": int(event), "n_host_hits": int(keep_h.sum()), "n_host_hits_stored": int(len(hp)),
            "n_gun_hits": int(gun["hits"].num_rows), "n_host_particles": int(out["particles"].num_rows
                                                                             - gun["particles"].num_rows),
            "n_gun_particles": int(gun["particles"].num_rows), "dropped_pids": drop.tolist(),
            "dropped_host_hits": int((~keep_h).sum())}
    return out, {"info": info, "row_map": row_map}


# ------------------------------------------------------------------------------------------------ checks
def columns_equal(a: pa.ChunkedArray | pa.Array, b: pa.ChunkedArray | pa.Array) -> bool:
    """Equal type, length and values; NaN equals NaN in floating columns (r_first is NaN without hits)."""
    if a.type != b.type or len(a) != len(b):
        return False
    if pa.types.is_floating(a.type):
        x, y = a.to_numpy(), b.to_numpy()
        return bool(np.array_equal(x, y, equal_nan=True) and np.array_equal(np.signbit(x), np.signbit(y)))
    return a.equals(b)


def tables_equal(a: pa.Table, b: pa.Table) -> bool:
    return a.schema.equals(b.schema, check_metadata=False) and all(columns_equal(a.column(n), b.column(n))
                                                                   for n in a.schema.names)


def check_event(merged: dict[str, pa.Table], host: dict[str, pa.Table], n_gun_particles: int,
                expect_weight_sum: float | None = 1.0, drop_pids=None) -> list[str]:
    """The converter gate of plan section 10 on one merged event; returns the problems (empty: pass)."""
    probs = []
    h = merged["hits"].to_pandas()
    p = merged["particles"].to_pandas()
    y = merged["ptype"].to_pandas()
    pid = p.particle_id.to_numpy().astype(np.int64)
    if pd.Series(pid).duplicated().any():
        probs.append("duplicate particle_id")
    if (pid <= 0).any():
        probs.append("particle_id <= 0 (as int64)")
    ng = int(n_gun_particles)
    gp = pid[len(pid) - ng:] if ng else pid[:0]
    vert = gp >> 52
    if ng and ((vert < GUN_VERTEX[0]) | (vert > GUN_VERTEX[1])).any():
        probs.append("gun vertex field outside 1001-1010")
    if (pid[:len(pid) - ng] >> 52 >= GUN_VERTEX[0]).any():
        probs.append("host vertex field in the gun range")
    if not np.array_equal(y.particle_id.to_numpy(), p.particle_id.to_numpy()):
        probs.append("ptype rows differ from particles rows")
    gun_hit = np.isin(h.pid.to_numpy(), gp)
    w = h.weight.to_numpy(np.float64)
    if gun_hit.any() and not np.all(w[gun_hit] == 0):
        probs.append("gun hit weight != 0")
    if expect_weight_sum is not None:
        s = float(np.nansum(w[~gun_hit]))
        if abs(s - expect_weight_sum) > WEIGHT_TOL:
            probs.append(f"host weight sum {s:.9f} != {expect_weight_sum:.9f}")
    cnt = pd.Series(h.pid.to_numpy()[h.pid.to_numpy() != 0]).value_counts()
    if not np.array_equal(cnt.reindex(pid).fillna(0).to_numpy(np.int64), p.n_hits.to_numpy(np.int64)):
        probs.append("n_hits differs from the hit count")
    if not np.isin(h.pid.to_numpy()[h.pid.to_numpy() != 0], pid).all():
        probs.append("hits of particles without a row")
    nh = merged["hits"].num_rows
    if not gun_hit[nh - int(gun_hit.sum()):].all():
        probs.append("gun hits are not the last rows")
    # host rows unchanged (apart from the event column and dropped particles)
    drop = np.asarray([] if drop_pids is None else drop_pids, np.int64)
    for k in TABLES:
        ht = host[k]
        if len(drop):
            col = "pid" if k == "hits" else "particle_id"
            ht = ht.filter(pa.array(~np.isin(ht.column(col).to_numpy(), drop)))
        mt = merged[k].slice(0, ht.num_rows)
        for name in ht.schema.names:
            if name == "event":
                continue
            if not columns_equal(ht.column(name), mt.column(name)):
                probs.append(f"host {k}.{name} altered")
    return probs


# ------------------------------------------------------------------------------------------------ storage
def write_set(set_name: str, events: dict[int, dict[str, pa.Table]], root: str, extra_meta: dict | None = None,
              level: int = 9, overwrite: bool = False) -> dict:
    """Write the merged events of one set (module docstring, step 5)."""
    ids = sorted(int(e) for e in events)
    if any(split_overlay_id(e)[0] != SETS[set_name][0] for e in ids):
        raise ValueError(f"an event id outside set {set_name}")
    if root.startswith("/ceph/"):
        C.storage_check(2.5e6 * len(ids))
    shards: dict[int, list[int]] = {}
    for e in ids:
        shards.setdefault(e // SHARD * SHARD, []).append(e)
    meta_shards = {}
    for first, evs in sorted(shards.items()):
        meta_shards[str(first)] = {}
        for k in TABLES:
            final = C.shard_path(root, k, set_name, first)
            if os.path.exists(final) and not overwrite:
                raise FileExistsError(final)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            tmp = final + ".tmp"
            with pq.ParquetWriter(tmp, T.SCHEMAS[k], compression="zstd", compression_level=level,
                                  write_statistics=True, **T.WRITE_OPTS[k]) as w:
                for e in evs:
                    t = events[e][k]
                    w.write_table(t.cast(T.SCHEMAS[k]), row_group_size=max(t.num_rows, 1))
            os.replace(tmp, final)
            meta_shards[str(first)][k] = {"file": os.path.relpath(final, root), "events": evs,
                                          "bytes": os.path.getsize(final), "sha256": C.sha256_file(final)}
    meta = {"archive": set_name, "phase": "overlay", "plan": "docs/pileup_overlay_plan.md section 5.4",
            "set_index": SETS[set_name][0], "n_events_converted": len(ids),
            "converted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "code_sha256": {f: C.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                            for f in ("writer.py", "gun.py", "emulate.py", "digitise.py")},
            "converter_sha256": {"convert.py": C.sha256_file(os.path.abspath(C.__file__)),
                                 "throughput.py": C.sha256_file(os.path.abspath(T.__file__))},
            "shard_rule": f"first = event // {SHARD} * {SHARD}", "zstd_level": level,
            "shards": meta_shards, **(extra_meta or {})}
    with open(os.path.join(root, f"meta_{set_name}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def verify_set(set_name: str, root: str, host_root: str = T.DERIVED_TP, expect_weight_sum: dict | None = None) -> dict:
    """Re-read the stored set: sha256 and schema of every shard, one row group per listed event in increasing order,
    and `check_event` against the host's stored rows (sidecar_<set>.json gives the host, the gun particle count and
    the dropped particles)."""
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        side = {int(r["overlay_event"]): r for r in json.load(fh)["events"]}
    problems = []
    for first, sh in meta["shards"].items():
        for k, info in sh.items():
            path = os.path.join(root, info["file"])
            if C.sha256_file(path) != info["sha256"]:
                problems.append(f"{info['file']}: sha256")
                continue
            f = pq.ParquetFile(path)
            if not f.schema_arrow.equals(T.SCHEMAS[k], check_metadata=False):
                problems.append(f"{info['file']}: schema")
            col = f.schema_arrow.get_field_index("event")
            evs = []
            for i in range(f.metadata.num_row_groups):
                s = f.metadata.row_group(i).column(col).statistics
                if s.min != s.max:
                    problems.append(f"{info['file']}: row group {i} holds several events")
                evs.append(s.min)
            if evs != info["events"] or evs != sorted(evs):
                problems.append(f"{info['file']}: row groups differ from the listed events")
    ids = sorted({e for sh in meta["shards"].values() for e in sh["hits"]["events"]})
    per_event = {}
    if problems:                                   # unreadable or changed shards: the event checks would not run
        return {"set": set_name, "events": len(ids), "problems": problems, "per_event": per_event}
    for e in ids:
        merged = {k: pa.Table.from_pandas(C.read_table(k, [e], root=root) if k != "hits" else
                                          C.read_table("hits", [e], T.SCHEMAS["hits"].names, root=root),
                                          schema=T.SCHEMAS[k], preserve_index=False) for k in TABLES}
        s = side[e]
        host = host_tables(s["host"], host_root)
        exp = (expect_weight_sum or {}).get(e, s.get("expected_host_weight_sum", 1.0))
        pr = check_event(merged, host, s["n_gun_particles"], exp, s.get("dropped_pids"))
        per_event[e] = pr
        problems += [f"event {e}: {x}" for x in pr]
    return {"set": set_name, "events": len(ids), "problems": problems, "per_event": per_event}


def write_csv_tar(path: str, members: dict[int, dict[str, bytes]]) -> dict:
    """The gun-only CSV members of a set, one tar (event<id>-<kind>.csv.gz, plan section 5.4)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with tarfile.open(tmp, "w") as tf:
        for e in sorted(members):
            for k in ("cells", "hits", "particles", "truth"):
                data = members[e][k]
                ti = tarfile.TarInfo(f"event{int(e):09d}-{k}.csv.gz")
                ti.size = len(data)
                ti.mtime = 0
                tf.addfile(ti, io.BytesIO(data))
    os.replace(tmp, path)
    return {"tar": path, "events": len(members), "bytes": os.path.getsize(path), "sha256": C.sha256_file(path)}


# ------------------------------------------------------------------------------------------------ simulation record
PATCHED_BUILD_PREFIX = "build-v15-zmirror"        # z-mirror builds of plan section 3 (G0: docs/pileup_overlay_hp_g0.md)
UNPATCHED_BUILD = "build-v15"


def sim_provenance(sim_dir: str, gev: dict, lookup_path: str) -> dict:
    """Build, seed and field map of the Fatras run in sim_dir (sim_summary.json of scripts/acts/overlay_sim.py sim, or
    fatras.json of slurm/overlay/fatras.sbatch). Raises if the Fatras seed is not the set's base seed + 13 (plan
    section 7) or if the module-id lookup does not belong to the build: the patched table only with a z-mirror build,
    the unpatched one only with build-v15 (the patch moves every -z endcap module, plan section 3)."""
    p_sum, p_json = os.path.join(sim_dir, "sim_summary.json"), os.path.join(sim_dir, "fatras.json")
    if os.path.exists(p_sum):
        with open(p_sum) as fh:
            s = json.load(fh)
        rec = {"record": "sim_summary.json", "build": s["build"], "seed": int(s["seed"]),
               "field_map": s["field_map"], "field_map_md5": s["field_map_md5"], "pmin_gev": s["fatras"]["pmin_gev"],
               "executable_md5": s.get("executable_md5"), "events": s.get("events"),
               "logged_particle_failures": s.get("particle_failures")}
    elif os.path.exists(p_json):
        with open(p_json) as fh:
            s = json.load(fh)
        rec = {"record": "fatras.json", "build": s["build"], "seed": int(s["rnd_seed"]), "field_map": s["map"],
               "field_map_sha256": s["map_sha256"], "pmin_gev": float(s["pmin_gev"]), "events": s.get("events")}
    else:
        raise FileNotFoundError(f"{sim_dir}: neither sim_summary.json nor fatras.json (unknown Fatras settings)")
    if rec["seed"] != int(gev["fatras_seed"]):
        raise ValueError(f"Fatras seed {rec['seed']} differs from the set's base seed + 13 = {gev['fatras_seed']}")
    build = os.path.basename(os.path.normpath(rec["build"]))
    lk = os.path.abspath(lookup_path)
    if lk == os.path.abspath(D.PATCHED_LOOKUP) and not build.startswith(PATCHED_BUILD_PREFIX):
        raise ValueError(f"the patched module table needs hits of a z-mirror build, not {build}")
    if lk == os.path.abspath(D.UNPATCHED_LOOKUP) and build != UNPATCHED_BUILD:
        raise ValueError(f"the unpatched module table needs hits of {UNPATCHED_BUILD}, not {build}")
    return rec


def sim_status(sim_path: str, gun_dir: str, gev: dict, pmin_gev: float) -> pd.DataFrame:
    """Fatras status of every gun particle of the set (hnl.overlay.fatras_io.simulation_status): from the simulated-
    particle list beside the packed hits and the failures in fatras.log of the same directory."""
    sim_dir = os.path.dirname(os.path.abspath(sim_path))
    p_sim, p_log = FIO.sim_particles_path(sim_path), os.path.join(sim_dir, "fatras.log")
    for f in (p_sim, p_log):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} is missing: without it the gun particles Fatras dropped are unknown")
    with open(p_log) as fh:
        failed = FIO.failed_particles(fh.read())
    simulated = pd.read_parquet(p_sim)
    inputs = pd.concat([pd.read_csv(FIO_path(gun_dir, int(ev["fatras_event"])), dtype={"particle_id": np.uint64})
                        .assign(event=int(ev["fatras_event"])) for ev in gev["events"]], ignore_index=True)
    events = inputs.event.unique()
    return FIO.simulation_status(inputs[["event", "particle_id", "q", "px", "py", "pz"]],
                                 simulated[simulated.event.isin(events)], failed[failed.event.isin(events)], pmin_gev)


# ------------------------------------------------------------------------------------------------ assembly
SEED_RULES = ("emulation: default_rng([base + 13, overlay id, 1]); digitiser cell factors: "
              "default_rng([base + 13, overlay id]); gun: default_rng([base + 7, overlay id]); Fatras: base + 13")


def overlay_hash(tables_path: str) -> str:
    """The version of everything applied after Fatras: the emulation hash (its tables and emulate.py), the digitiser
    (digitise.py, which carries the pixel charge spread as the per-cell factor CELL_SIGMA since gate C1) and the seed
    rules. The plan's emulation hash (section 4.4, fixed at the end of H-T0.5) predates the move of the charge spread
    into the digitiser; this hash covers both."""
    import hashlib
    from hnl.overlay import emulate as EM
    h = hashlib.sha256(EM.emulation_hash(tables_path).encode())
    with open(os.path.abspath(D.__file__), "rb") as fh:
        h.update(fh.read())
    h.update(f"CELL_SIGMA={D.CELL_SIGMA!r};{SEED_RULES}".encode())
    return h.hexdigest()


def gun_event(sim_hits: pd.DataFrame, particles: pd.DataFrame, lookup, modules: D.Modules, tables, base_seed: int,
              event: int) -> dict:
    """Emulation and digitisation of one gun event: sim_hits = the packed Fatras hits of the event
    (hnl.overlay.fatras_io), particles = its Fatras input (ACTS barcodes), tables = hnl.overlay.emulate.Tables. Returns
    the emulation flags, the digitised hits, the TrackML members and their gzip CSV bytes (deterministic in the
    inputs, the seed rule of `assemble` and the overlay event id)."""
    from hnl.overlay import emulate as EM
    s = sim_hits.reset_index(drop=True)
    pinfo = pd.DataFrame({"particle_id": particles.particle_id.to_numpy(np.uint64),
                          "pdg": particles.particle_type.to_numpy(np.int64),
                          "p": np.sqrt(particles.px ** 2 + particles.py ** 2 + particles.pz ** 2).to_numpy()})
    rng = np.random.default_rng([base_seed + 13, int(event), EMULATION_STREAM])
    vol, lay, mod = lookup.from_geometry_id(s.geometry_id.to_numpy(np.uint64))
    if (mod < 0).any():
        raise RuntimeError(f"overlay event {event}: {int((mod < 0).sum())} sim hits on modules missing from the lookup")
    em = EM.emulate(pd.DataFrame({"particle_id": s.particle_id.to_numpy(np.uint64), "volume": vol, "layer": lay,
                                  "module": mod, "order": s["index"].to_numpy(),
                                  **{c: s[c].to_numpy(np.float64) for c in ("tx", "ty", "tpx", "tpy", "tpz")}}),
                    pinfo, tables, rng)
    k = em["keep"]
    dig = D.digitise_on_modules(s[k].reset_index(drop=True), vol[k], lay[k], mod[k], modules,
                                rng=D.event_rng(base_seed + 13, int(event)))
    mem = gun_members(dig, particles)
    return {"emulation": em, "digitised": dig, "members": mem, "gz": members_gz(mem)}


def assemble(set_name: str, gun_dir: str, sim_path: str, root: str, tables_path: str, lookup_path: str,
             host_root: str = T.DERIVED_TP, emulation_seed: int | None = None, csv_tar: str | None = None,
             overwrite: bool = False) -> dict:
    """Emulation, digitisation, conversion, merge and storage of a gun set made by hnl.overlay.gun (gun_dir) and
    simulated by scripts/acts/overlay_sim.py sim or slurm/overlay/fatras.sbatch, then packed by hnl.overlay.fatras_io
    (sim_path = packed sim hits; the simulated-particle list, fatras.log and the run record lie beside it). Seeds from
    base_seed + 13 (the Fatras and digitisation seed of spec v1 section 4): the digitiser's per-cell charge factor
    draws from hnl.overlay.digitise.event_rng(base + 13, overlay id) = default_rng([base + 13, id]), the emulation
    from the independent stream default_rng([base + 13, id, EMULATION_STREAM]). Gun particles that Fatras dropped
    (failed propagation) or never simulated (p <= pMin) keep their particle row with no hits, as in TrackML; the
    sidecar lists them per event and gun_truth_<set>.parquet carries the status (column sim_status)."""
    from hnl.overlay import emulate as EM
    with open(os.path.join(gun_dir, "gun_events.json")) as fh:
        gev = json.load(fh)
    truth = pd.read_parquet(os.path.join(gun_dir, "gun_truth.parquet"))
    sim = pd.read_parquet(sim_path)
    prov = sim_provenance(os.path.dirname(os.path.abspath(sim_path)), gev, lookup_path)
    status = sim_status(sim_path, gun_dir, gev, prov["pmin_gev"])
    st_key = pd.Series(status.status.to_numpy(), index=pd.MultiIndex.from_arrays(
        [status.event.to_numpy(np.int64), status.particle_id.to_numpy(np.uint64)]))
    truth["sim_status"] = st_key.reindex(pd.MultiIndex.from_arrays(
        [truth.fatras_event.to_numpy(np.int64), truth.acts_barcode.to_numpy(np.uint64)])).to_numpy()
    if truth.sim_status.isna().any():
        raise RuntimeError("gun truth rows without a Fatras status")
    tab = EM.Tables.load(tables_path)
    ehash = EM.emulation_hash(tables_path)
    ohash = overlay_hash(tables_path)
    mods = D.Modules.load()
    lookup = D.load_lookup(lookup_path)
    det = C.load_detectors()
    base = gev["base_seed"] if emulation_seed is None else emulation_seed
    merged_all, side, gz_all, stats = {}, [], {}, []
    for ev in gev["events"]:
        oid, host, fe = int(ev["overlay_event"]), int(ev["host"]), int(ev["fatras_event"])
        parts = pd.read_csv(FIO_path(gun_dir, fe), dtype={"particle_id": np.uint64})
        s = sim[sim.event == fe].reset_index(drop=True)
        st = status[status.event == fe]
        lost = st[st.status != "simulated"]
        if np.isin(s.particle_id.to_numpy(np.uint64), lost.particle_id.to_numpy(np.uint64)).any():
            raise RuntimeError(f"overlay event {oid}: sim hits of a particle that Fatras did not simulate")
        g = gun_event(s, parts, lookup, mods, tab, base, oid)
        em, dig, gz = g["emulation"], g["digitised"], g["gz"]
        gtab, dq = convert_gun(gz, oid, det)
        host_t = host_tables(host, host_root)
        merged, rm = merge_event(host_t, gtab, oid)
        exp_w = 1.0
        probs = check_event(merged, host_t, gtab["particles"].num_rows, exp_w)
        if probs:
            raise RuntimeError(f"overlay event {oid}: {probs}")
        merged_all[oid] = merged
        gz_all[oid] = gz
        cat = gtab["particles"].column("category").to_numpy(zero_copy_only=False)
        side.append({**rm["info"], "overlay_event": oid, "host": host, "fatras_event": fe,
                     "n_gun_particles": int(gtab["particles"].num_rows), "expected_host_weight_sum": exp_w,
                     "gun_hit_rows": [int(rm["info"]["n_host_hits"]),
                                      int(rm["info"]["n_host_hits"] + rm["info"]["n_gun_hits"])],
                     "sim_hits": int(len(s)), "stopped_hits": int(em["stopped"].sum()),
                     "dropped_hits": int(em["dropped"].sum()), "digitiser": dig.stats,
                     "gun_dq_max_outside_mm": dq["max_outside_mm"],
                     "gun_categories": {str(c): int((cat == c).sum()) for c in np.unique(cat)},
                     "sim_failed_particles": [int(x) for x in GUN.trackml_particle_id(
                         lost[lost.status == "failed"].particle_id.to_numpy(np.uint64))],
                     "sim_below_pmin_particles": [int(x) for x in GUN.trackml_particle_id(
                         lost[lost.status == "below_pmin"].particle_id.to_numpy(np.uint64))],
                     "hs_vertex": ev["hs_vertex"]})
        stats.append(dig.stats)
    meta_extra = {"emulation_hash": ehash, "overlay_hash": ohash, "seed_rules": SEED_RULES,
                  "emulation_tables": tables_path, "lookup": lookup_path,
                  "lookup_sha256": C.sha256_file(lookup_path), "sim": prov,
                  "sim_status_counts": {str(k): int(v) for k, v in status.status.value_counts().items()},
                  "gun": {k: v for k, v in gev.items() if k != "events"}, "host_root": host_root}
    meta = write_set(set_name, merged_all, root, meta_extra, overwrite=overwrite)
    side_rec = {"set": set_name, "emulation_hash": ehash, "overlay_hash": ohash,
                "gun_particles": "gun_truth_" + set_name + ".parquet",
                "sim": prov, "sim_status_counts": meta_extra["sim_status_counts"], "events": side}
    with open(os.path.join(root, f"sidecar_{set_name}.json"), "w") as fh:
        json.dump(side_rec, fh, indent=1)
    truth.to_parquet(os.path.join(root, f"gun_truth_{set_name}.parquet"), index=False)
    tar = write_csv_tar(csv_tar, gz_all) if csv_tar else None
    return {"meta": meta, "sidecar": side_rec, "csv_tar": tar}


def FIO_path(gun_dir: str, fatras_event: int) -> str:
    return FIO.particles_path(os.path.join(gun_dir, "fatras_in"), fatras_event)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("assemble")
    a_.add_argument("--set", required=True, choices=sorted(SETS))
    a_.add_argument("--gun-dir", required=True)
    a_.add_argument("--sim", required=True)
    a_.add_argument("--root", required=True)
    a_.add_argument("--tables", default=os.path.join(REPO, "configs", "trackml", "overlay_emulation_v1.json"))
    a_.add_argument("--lookup", default=D.PATCHED_LOOKUP)
    a_.add_argument("--csv-tar", default=None)
    a_.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--set", required=True, choices=sorted(SETS))
    v.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        r = assemble(a.set, a.gun_dir, a.sim, a.root, a.tables, a.lookup, csv_tar=a.csv_tar, overwrite=a.overwrite)
        print(json.dumps({"events": r["meta"]["n_events_converted"], "csv_tar": r["csv_tar"],
                          "sidecar": r["sidecar"]["events"]}, indent=1, default=str))
    else:
        r = verify_set(a.set, a.root)
        print(json.dumps({k: r[k] for k in ("set", "events", "problems")}, indent=1))
        if r["problems"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
