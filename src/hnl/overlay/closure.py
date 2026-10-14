"""V1 closure set of the TrackML pileup overlay (docs/pileup_overlay_plan.md sections 5.4, 9 and 10, gate V1). Native
targets of tp_v1 val events are removed from their host, simulated again through the overlay chain (Fatras of the
patched build with the TrackML-like field map, emulation, digitiser, frozen converter) and merged back. Truth only: no
model is run and no model output is read.

Which particles are replaced (`population_table`, `closure_population`). Native targets (the stored is_target flag:
pT > 1 GeV, |eta| < 4, >= 5 hits) of the 150 tp_v1 val events, with generation = (particle_id >> 24) & 0xFFF:
  a  generation 0 with |d0| >= 10 mm or r_v >= 100 mm (the P_P definition; plan V1 (a))
  b  generation >= 1 with |d0| >= 10 mm (V1 (b))
  c  generation 0 with |d0| < 1 mm and r_v < 1 mm (V1 (c)), restricted to the first PROMPT_N = 3,000 of all val
     events in the order of int(sha256("canto-trackml-overlay-v1-prompt:<event>:<particle_id>")[:8], 16), ties broken
     by event and then particle_id (event = the tp_v1 host id, particle_id = the native id in decimal)
The three are disjoint. The plan has one closure set (set index 1, one overlay event per host), so each closure event
replaces the union of the three in its host. A target's region is 'central' without a hit at |z| > 1000 mm, else
'fwd+' or 'fwd-' by the sign of eta (plan section 10; the definitions of pileup_design/plan_revision/val_regions.py).

Removal and the weight rule. A replaced native loses its particles row, its ptype row and all its hits
(hnl.overlay.writer.merge_event with drop_pids). Nothing else changes: its secondaries and every other particle stay,
and the host rows keep their order. Each replaced native gets exactly one closure row, also when Fatras dropped it
(then without hits). Closure hits carry weight 0, so the host weights of a closure event sum to 1 - W_removed, the
summed weights of the removed hits, within 1e-6 (plan section 10, "Converter").

Ids. closure particle_id = native particle_id + 2^62, i.e. bit 62 set. The TrackML vertex field [52, 64) becomes
1024 + the native vertex field; primary, generation, secondary and process keep the native values. Native vertex
fields are at most 1000 (the writer's host rule; the largest in tp_v1 val is 251), so closure vertex fields lie in
1024-2024. That range is disjoint from host fields (0-1000) and gun fields (1001-1010), stays below 2048 (bit 63 clear:
every id is positive as int64 and pandas reads it as int64), and inverts exactly: native = closure - 2^62. The process
field cannot carry a marker, since TrackML uses it. In Fatras a closure particle has the ACTS barcode vertex primary 2,
secondary vertex 0, particle i = 1..n (the rank of its native id within the host), generation 0, sub-particle 0. The
closure truth table maps barcodes to ids; barcodes never reach the stored tables.

What the frozen converter sees. The closure-only event is written as the four TrackML CSV members in the gun-only
layout of hnl.overlay.writer (hits sorted by module, hit ids 1..M, weight 0, nhits counted after digitisation), with
the closure ids, the native particle_type and charge, and the native's stored float32 production vertex and momentum
(%.9g in the CSV reproduces float32 exactly). The frozen event_tables_tp therefore recomputes pt, eta, r_v and d0
bit-identical to the native row. n_hits, n_pixel_hits, r_first, the hit ranks and the target flags come from the
re-simulated hits. vprim is 1024 + the native vertex, so is_hard_scatter is False. category comes from the pair
tagging inside the closure-only event: a K_S or Lambda daughter keeps its label only if its partner is replaced too.
The model reads hits only. An evaluation pairs a closure particle with its native through native = closure - 2^62
and takes the population, region, hard-scatter flag and category from the native (closure truth table).

Seeds and versions: those of the gun sets (hnl.overlay.writer.SEED_RULES): Fatras base + 13, emulation
default_rng([base + 13, overlay id, 1]), digitiser default_rng([base + 13, overlay id]); the gun's base + 7 stream is
not used. The production base seed is the closure entry of configs/trackml/split_h_v1.json (`default_base_seed`; the
older value in hnl.overlay.writer.SETS is not used). The closure set holds no gun LLPs, so the replacement is the only
change against the stored val event.

    python -m hnl.overlay.closure select --out POP.parquet
    python -m hnl.overlay.closure make --hosts 21289:21332 --set closure --population POP.parquet --out DIR \
        [--base-seed S]
    python -m hnl.overlay.closure assemble --set closure --dir DIR --sim DIR/fatras_out/simhits.parquet --root ROOT \
        [--csv-tar TAR]
    python -m hnl.overlay.closure verify --set closure --root ROOT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import writer as W
from hnl.trackml import convert as C
from hnl.trackml import throughput as T

TP = T.DERIVED_TP
CLOSURE_BIT = 62
CLOSURE_FLAG = 1 << CLOSURE_BIT                       # vertex field + 1024
HOST_VERTEX_MAX = W.GUN_VERTEX[0] - 1                 # 1000: the writer's host rule
CLOSURE_VERTEX = (1024, 1024 + HOST_VERTEX_MAX)       # 1024-2024
BARCODE_VPRIM = 2                                     # ACTS vertex primary of a closure particle (the gun uses 1)
POPULATIONS = {"a": "gen-0, |d0| >= 10 mm or r_v >= 100 mm (P_P definition)",
               "b": "gen >= 1, |d0| >= 10 mm",
               "c": "gen-0, |d0| < 1 mm and r_v < 1 mm (first 3,000 of val by hash)"}
VAL_REGIONS_NAME = {"a": "pp_gen0", "b": "d10_gen1", "c": "prompt"}   # keys of plan_revision/val_regions.json
REGIONS = ("central", "fwd+", "fwd-")
PP_D0_MM, PP_RV_MM, PROMPT_MM = 10.0, 100.0, 1.0
FORWARD_Z_MM = 1000.0
PROMPT_N = 3000
PROMPT_SALT = "canto-trackml-overlay-v1-prompt"
VAL_REGIONS = "/work/submit/jinboz1/hnl/pileup_design/plan_revision/val_regions.json"
SPLIT_H = os.path.join(W.REPO, "configs", "trackml", "split_h_v1.json")
# GeV (PDG); the Fatras input needs a mass, TrackML stores none. Unknown species raise.
MASS = {11: 0.000510999, 13: 0.105658, 211: 0.139570, 321: 0.493677, 2212: 0.938272, 3112: 1.197449,
        3222: 1.18937, 3312: 1.32171, 3334: 1.67245}
KINEMATICS = ("q", "vx", "vy", "vz", "px", "py", "pz", "pt", "eta", "rv", "d0")
TRUTH_COLUMNS = ["event", "host", "fatras_event", "native_particle_id", "particle_id", "acts_barcode", "population",
                 "region", "prompt_rank", "particle_type", "q", "vx", "vy", "vz", "px", "py", "pz", "pt", "eta", "rv",
                 "d0", "mass", "native_n_hits", "native_n_pixel_hits", "native_r_first", "native_category",
                 "native_vprim", "native_is_hard_scatter"]
P_COLUMNS = ["event", "particle_id", "vx", "vy", "vz", "px", "py", "pz", "q", "n_hits", "pt", "eta", "rv", "d0",
             "category", "vprim", "n_pixel_hits", "r_first", "is_target"]


# ------------------------------------------------------------------------------------------------ ids
def closure_particle_id(native) -> np.ndarray | int:
    """Native TrackML particle_id -> closure particle_id (module docstring). Refuses ids that are not positive or whose
    vertex field exceeds the host range."""
    scalar = np.ndim(native) == 0
    n = np.atleast_1d(np.asarray(native, np.int64))
    if (n <= 0).any():
        raise ValueError("native particle_id <= 0")
    if ((n >> 52) > HOST_VERTEX_MAX).any():
        raise ValueError(f"native vertex field above {HOST_VERTEX_MAX}")
    out = n | np.int64(CLOSURE_FLAG)
    return int(out[0]) if scalar else out


def native_particle_id(closure) -> np.ndarray | int:
    """Closure particle_id -> the native particle_id it replaces."""
    scalar = np.ndim(closure) == 0
    c = np.atleast_1d(np.asarray(closure, np.int64))
    if not is_closure_id(c).all():
        raise ValueError("not a closure particle_id")
    out = c & ~np.int64(CLOSURE_FLAG)
    return int(out[0]) if scalar else out


def is_closure_id(pid) -> np.ndarray:
    v = np.asarray(pid, np.int64) >> 52
    return (v >= CLOSURE_VERTEX[0]) & (v <= CLOSURE_VERTEX[1])


def closure_barcode(index) -> np.ndarray:
    """ACTS barcode of the closure particle with rank `index` (1..65535) in its event."""
    i = np.asarray(index, np.int64)
    if (i < 1).any() or (i >= 1 << 16).any():
        raise ValueError("closure particle index outside 1..65535")
    return FIO.barcode(BARCODE_VPRIM, 0, i)


def default_base_seed(set_name: str = "closure", path: str = SPLIT_H) -> int:
    """The production base seed of a set from split_h_v1.json (fixed at H-T0)."""
    with open(path) as fh:
        return int(json.load(fh)["sets"][set_name]["base_seed"])


def generation(pid) -> np.ndarray:
    return (np.asarray(pid, np.int64) >> 24) & 0xFFF


def mass_of(pdg) -> np.ndarray:
    a = np.abs(np.asarray(pdg, np.int64))
    unknown = sorted(set(a.tolist()) - set(MASS))
    if unknown:
        raise KeyError(f"no mass for |PDG| {unknown}")
    return np.array([MASS[int(x)] for x in a], np.float64)


# ------------------------------------------------------------------------------------------------ populations
def prompt_key(event: int, particle_id: int) -> int:
    return int(hashlib.sha256(f"{PROMPT_SALT}:{int(event)}:{int(particle_id)}".encode()).hexdigest()[:8], 16)


def classify(p: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """Population ('a', 'b', 'c' or '') and region of every particle of one event. p: particle_id, d0, rv, eta,
    is_target (stored float32 values); h: z, pid of the event's hits."""
    pid = p.particle_id.to_numpy(np.int64)
    gen = generation(pid)
    tgt = p.is_target.to_numpy(bool)
    ad0, rv = np.abs(p.d0.to_numpy()), p.rv.to_numpy()
    pop = np.where(tgt & (gen == 0) & ((ad0 >= PP_D0_MM) | (rv >= PP_RV_MM)), "a",
                   np.where(tgt & (gen >= 1) & (ad0 >= PP_D0_MM), "b",
                            np.where(tgt & (gen == 0) & (ad0 < PROMPT_MM) & (rv < PROMPT_MM), "c", "")))
    hp, hz = h.pid.to_numpy(np.int64), h.z.to_numpy()
    fwd = np.isin(pid, np.unique(hp[(hp != 0) & (np.abs(hz) > FORWARD_Z_MM)]))
    region = np.where(~fwd, "central", np.where(p.eta.to_numpy() > 0, "fwd+", "fwd-"))
    return pd.DataFrame({"population": pop, "region": region, "gen": gen}, index=p.index)


def prompt_subset(c: pd.DataFrame, n: int = PROMPT_N) -> tuple[pd.DataFrame, dict]:
    """Rank the prompt candidates (event, particle_id) of all val events by the hash order and mark the first n.
    Returns the frame with prompt_key, prompt_rank, in_subset, and a record (ties, the key at the cut)."""
    c = c.copy()
    c["prompt_key"] = [prompt_key(e, q) for e, q in zip(c.event.to_numpy(), c.particle_id.to_numpy())]
    c = c.sort_values(["prompt_key", "event", "particle_id"], kind="stable").reset_index(drop=True)
    c["prompt_rank"] = np.arange(len(c), dtype=np.int64)
    c["in_subset"] = c.prompt_rank < n
    k = c.prompt_key.to_numpy(np.int64)
    rec = {"candidates": int(len(c)), "n": int(min(n, len(c))),
           "duplicate_keys": int(len(k) - len(np.unique(k))),
           "tie_at_cut": bool(len(c) > n and k[n - 1] == k[n]),
           "last_key_in": int(k[min(n, len(c)) - 1]) if len(c) else None}
    return c, rec


def population_table(events=None, root: str = TP, chunk: int = 10, require_all_val: bool = True) -> tuple[
        pd.DataFrame, dict]:
    """The closure populations of tp_v1 val events (module docstring): one row per selected native (a, b, and c in the
    prompt subset) with its stored kinematics and flags, and a record with the counts of every population and region
    over the events read (c counted before the subset), for comparison with plan_revision/val_regions.json. Reads the
    particles, ptype and hits (z, pid) tables of val events only."""
    split = T.load_tp_split(T.SPLIT_TP)
    val = sorted(split["val"])
    events = val if events is None else sorted(int(e) for e in events)
    if not set(events) <= split["val"]:
        raise PermissionError("closure hosts must be tp_v1 val events")
    if require_all_val and events != val:
        raise ValueError("the prompt subset is defined over all val events")
    rows, n_particles, vmax, nproc = [], 0, 0, 0
    for k in range(0, len(events), chunk):
        part = events[k:k + chunk]
        P = C.read_table("particles", part, P_COLUMNS, root=root)
        Y = C.read_table("ptype", part, ["event", "particle_id", "particle_type"], root=root)
        H = C.read_table("hits", part, ["event", "z", "pid"], root=root)
        for e in part:
            p = P[P.event == e].reset_index(drop=True)
            y = Y[Y.event == e]
            if not np.array_equal(y.particle_id.to_numpy(), p.particle_id.to_numpy()):
                raise RuntimeError(f"event {e}: ptype rows differ from particles rows")
            p["particle_type"] = y.particle_type.to_numpy()
            pid = p.particle_id.to_numpy(np.int64)
            n_particles += len(p)
            vmax = max(vmax, int((pid >> 52).max()))
            nproc += int(((pid & 0xFFF) != 0).sum())
            cl = classify(p, H[H.event == e])
            p = pd.concat([p, cl], axis=1)
            rows.append(p[p.population != ""])
    t = pd.concat(rows, ignore_index=True)
    counts = {VAL_REGIONS_NAME[k]: {**{r: int(n) for r, n in g.region.value_counts().items()}, "all": int(len(g))}
              for k, g in t.groupby("population")}
    c, sub = prompt_subset(t[t.population == "c"][["event", "particle_id"]]) if require_all_val else (None, None)
    t = t.merge(c[["event", "particle_id", "prompt_rank", "in_subset"]], on=["event", "particle_id"], how="left") \
        if c is not None else t.assign(prompt_rank=np.nan, in_subset=False)
    t["prompt_rank"] = t.prompt_rank.fillna(-1).astype(np.int64)
    t = t[(t.population != "c") | t.in_subset.fillna(False).astype(bool)].drop(columns="in_subset")
    t = t.sort_values(["event", "particle_id"], kind="stable").reset_index(drop=True)
    rec = {"events": len(events), "all_val": events == val, "split_sha256": split["sha256"], "counts": counts,
           "prompt_subset": sub, "selected": {k: int((t.population == k).sum()) for k in POPULATIONS},
           "particles_read": n_particles, "max_native_vertex_field": vmax,
           "particles_with_nonzero_process_field": nproc}
    return t, rec


def compare_val_regions(counts: dict, path: str = VAL_REGIONS) -> dict:
    """The population counts against the plan's val_regions.json (job 6812736)."""
    with open(path) as fh:
        ref = json.load(fh)["counts"]
    diff = {k: {r: [counts.get(k, {}).get(r, 0), ref[k].get(r, 0)] for r in (*REGIONS, "all")
                if counts.get(k, {}).get(r, 0) != ref[k].get(r, 0)} for k in ref}
    return {"reference": path, "equal": not any(diff.values()), "differences": diff}


def closure_population(table: pd.DataFrame, hosts) -> pd.DataFrame:
    """The replaced natives of the given hosts (all three populations; c already restricted to the subset)."""
    hosts = [int(h) for h in hosts]
    missing = sorted(set(hosts) - set(table.event.unique()))
    if missing:
        raise KeyError(f"hosts without closure particles in the population table: {missing}")
    return table[table.event.isin(hosts)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ Fatras input
def make_closure_set(pop: pd.DataFrame, overlay_ids: dict[int, int], base_seed: int, out_dir: str,
                     population_sha256: str | None = None) -> dict:
    """Fatras input and closure truth of a set of closure events. pop: `closure_population` rows of the hosts;
    overlay_ids: {host: overlay id}. Fatras event i = position of the host in sorted overlay ids (as the gun sets).
    Writes out_dir/fatras_in/event<i>-particles.csv, closure_truth.parquet and closure_events.json (the layout of
    hnl.overlay.gun's gun_events.json, read by hnl.overlay.writer.sim_provenance and sim_status)."""
    fin = os.path.join(out_dir, "fatras_in")
    os.makedirs(fin, exist_ok=True)
    order = sorted(overlay_ids, key=lambda h: overlay_ids[h])
    truth, events = [], []
    for i, host in enumerate(order):
        oid = overlay_ids[host]
        r = pop[pop.event == host].sort_values("particle_id", kind="stable").reset_index(drop=True)
        if not len(r):
            raise ValueError(f"host {host}: no closure particle")
        n = len(r)
        bc = closure_barcode(np.arange(1, n + 1))
        m = mass_of(r.particle_type.to_numpy())
        f64 = lambda c: r[c].to_numpy(np.float32).astype(np.float64)       # noqa: E731  (exact float32 values)
        parts = pd.DataFrame({"particle_id": bc, "particle_type": r.particle_type.to_numpy(np.int64), "process": 0,
                              "vx": f64("vx"), "vy": f64("vy"), "vz": f64("vz"), "vt": 0.0, "px": f64("px"),
                              "py": f64("py"), "pz": f64("pz"), "m": m, "q": r.q.to_numpy(np.int64)})
        FIO.write_particles(parts[FIO.PARTICLE_COLUMNS], fin, i)
        tr = pd.DataFrame({
            "event": oid, "host": int(host), "fatras_event": i, "native_particle_id": r.particle_id.to_numpy(np.int64),
            "particle_id": closure_particle_id(r.particle_id.to_numpy(np.int64)), "acts_barcode": bc,
            "population": r.population.to_numpy(), "region": r.region.to_numpy(),
            "prompt_rank": r.prompt_rank.to_numpy(np.int64), "particle_type": r.particle_type.to_numpy(np.int64),
            **{k: r[k].to_numpy(np.float32) for k in ("vx", "vy", "vz", "px", "py", "pz", "pt", "eta", "rv", "d0")},
            "q": r.q.to_numpy(np.int8), "mass": m, "native_n_hits": r.n_hits.to_numpy(np.int16),
            "native_n_pixel_hits": r.n_pixel_hits.to_numpy(np.int16), "native_r_first": r.r_first.to_numpy(np.float32),
            "native_category": r.category.to_numpy(), "native_vprim": r.vprim.to_numpy(np.int16),
            "native_is_hard_scatter": r.vprim.to_numpy(np.int64) == 1})
        truth.append(tr[TRUTH_COLUMNS])
        events.append({"fatras_event": i, "overlay_event": oid, "host": int(host), "n_particles": n,
                       "populations": {k: int((r.population == k).sum()) for k in POPULATIONS}})
    t = pd.concat(truth, ignore_index=True)
    t["acts_barcode"] = t.acts_barcode.astype(np.uint64)
    t.to_parquet(os.path.join(out_dir, "closure_truth.parquet"), index=False)
    rec = {"kind": "closure", "base_seed": int(base_seed), "fatras_seed": int(base_seed) + 13,
           "seed_rules": W.SEED_RULES, "id_rule": "closure particle_id = native particle_id + 2^62",
           "barcode_rule": f"ACTS barcode ({BARCODE_VPRIM}, 0, rank of the native id in its host, 0, 0)",
           "population_sha256": population_sha256, "events": events,
           "populations": {k: int((t.population == k).sum()) for k in POPULATIONS}}
    with open(os.path.join(out_dir, "closure_events.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ------------------------------------------------------------------------------------------------ members
def members(dig: D.Digitised, particles: pd.DataFrame, tml_ids) -> dict[str, pd.DataFrame]:
    """TrackML CSV members of an event whose particles carry the given TrackML ids (one per particles row, in its
    order): hnl.overlay.writer.gun_members with the id rule as an argument. particles: particle_id (ACTS barcode of
    the hits), particle_type, vx, vy, vz, px, py, pz, q."""
    h = dig.hits.reset_index(drop=True)
    n = len(h)
    bc_p = particles.particle_id.to_numpy().astype(np.uint64)
    if pd.Series(bc_p).duplicated().any():
        raise ValueError("duplicate barcode")
    tml_p = np.asarray(tml_ids, np.int64)
    if len(tml_p) != len(bc_p) or pd.Series(tml_p).duplicated().any() or (tml_p <= 0).any():
        raise ValueError("TrackML ids must be one positive, unique id per particle")
    bc_h = h.particle_id.to_numpy().astype(np.uint64)
    known = pd.Index(bc_p).get_indexer(bc_h)
    if (known < 0).any():
        raise KeyError(f"{int((known < 0).sum())} hits of particles not in the particle table")
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
        raise ValueError("particles must have charge +-1")
    m["particles"] = pd.DataFrame({
        "particle_id": tml_p, "particle_type": particles.particle_type.to_numpy(np.int64),
        **{k: particles[k].to_numpy(np.float64) for k in ("vx", "vy", "vz", "px", "py", "pz")},
        "q": q, "nhits": nh.reindex(tml_p).fillna(0).to_numpy(np.int64)})
    return {k: m[k][W.CSV_COLUMNS[k]] for k in ("hits", "cells", "particles", "truth")}


def closure_rows(truth: pd.DataFrame, fatras_particles: pd.DataFrame) -> pd.DataFrame:
    """The particle rows of one closure event in Fatras input order: barcode, the native particle_type and charge, and
    the native's stored float32 vertex and momentum (exact); checks that the Fatras input carries the same values."""
    t = truth.set_index(truth.acts_barcode.to_numpy(np.uint64))
    bc = fatras_particles.particle_id.to_numpy().astype(np.uint64)
    if set(bc.tolist()) != set(t.index.tolist()) or len(bc) != len(t):
        raise ValueError("Fatras input and closure truth list different particles")
    t = t.loc[bc]
    out = pd.DataFrame({"particle_id": bc, "particle_type": t.particle_type.to_numpy(np.int64),
                        **{k: t[k].to_numpy(np.float32).astype(np.float64) for k in ("vx", "vy", "vz", "px", "py",
                                                                                     "pz")},
                        "q": t.q.to_numpy(np.int64), "tml_id": t.particle_id.to_numpy(np.int64)})
    for k in ("vx", "vy", "vz", "px", "py", "pz"):
        if not np.array_equal(fatras_particles[k].to_numpy(np.float64).astype(np.float32), out[k].to_numpy(np.float32)):
            raise ValueError(f"Fatras input {k} differs from the native float32 value")
    if not (np.array_equal(fatras_particles.particle_type.to_numpy(np.int64), out.particle_type.to_numpy())
            and np.array_equal(fatras_particles.q.to_numpy(np.int64), out.q.to_numpy())):
        raise ValueError("Fatras input type or charge differs from the native")
    return out


def closure_event(sim_hits: pd.DataFrame, rows: pd.DataFrame, lookup, modules: D.Modules, tables, base_seed: int,
                  event: int) -> dict:
    """Emulation and digitisation of one closure event: hnl.overlay.writer.gun_event with the closure ids. rows:
    `closure_rows` (barcode, type, kinematics, q, tml_id). Same seeds and order of random draws as the gun sets."""
    from hnl.overlay import emulate as EM
    s = sim_hits.reset_index(drop=True)
    pinfo = pd.DataFrame({"particle_id": rows.particle_id.to_numpy(np.uint64),
                          "pdg": rows.particle_type.to_numpy(np.int64),
                          "p": np.sqrt(rows.px ** 2 + rows.py ** 2 + rows.pz ** 2).to_numpy()})
    rng = np.random.default_rng([base_seed + 13, int(event), W.EMULATION_STREAM])
    vol, lay, mod = lookup.from_geometry_id(s.geometry_id.to_numpy(np.uint64))
    if (mod < 0).any():
        raise RuntimeError(f"closure event {event}: {int((mod < 0).sum())} sim hits on modules missing from the lookup")
    em = EM.emulate(pd.DataFrame({"particle_id": s.particle_id.to_numpy(np.uint64), "volume": vol, "layer": lay,
                                  "module": mod, "order": s["index"].to_numpy(),
                                  **{c: s[c].to_numpy(np.float64) for c in ("tx", "ty", "tpx", "tpy", "tpz")}}),
                    pinfo, tables, rng)
    k = em["keep"]
    dig = D.digitise_on_modules(s[k].reset_index(drop=True), vol[k], lay[k], mod[k], modules,
                                rng=D.event_rng(base_seed + 13, int(event)))
    mem = members(dig, rows, rows.tml_id.to_numpy(np.int64))
    return {"emulation": em, "digitised": dig, "members": mem, "gz": W.members_gz(mem)}


# ------------------------------------------------------------------------------------------------ converter gate
def check_closure_event(merged: dict[str, pa.Table], host: dict[str, pa.Table], natives,
                        expect_weight_sum: float | None) -> list[str]:
    """The converter gate of plan section 10 for a closure event (hnl.overlay.writer.check_event with the closure
    rules); returns the problems (empty: pass). natives: the replaced native ids; host: the host's stored rows."""
    probs = []
    natives = np.unique(np.asarray(natives, np.int64))
    nc = len(natives)
    h = merged["hits"].to_pandas()
    p = merged["particles"].to_pandas()
    y = merged["ptype"].to_pandas()
    pid = p.particle_id.to_numpy().astype(np.int64)
    if pd.Series(pid).duplicated().any():
        probs.append("duplicate particle_id")
    if (pid <= 0).any():
        probs.append("particle_id <= 0 (as int64)")
    host_pid, cl_pid = pid[:len(pid) - nc], pid[len(pid) - nc:]
    if not np.array_equal(np.sort(cl_pid), np.sort(closure_particle_id(natives)) if nc else cl_pid):
        probs.append("closure rows differ from the replaced natives")
    if nc and not is_closure_id(cl_pid).all():
        probs.append(f"closure vertex field outside {CLOSURE_VERTEX[0]}-{CLOSURE_VERTEX[1]}")
    if ((host_pid >> 52) > HOST_VERTEX_MAX).any():
        probs.append(f"host vertex field above {HOST_VERTEX_MAX}")
    if np.isin(host_pid, natives).any():
        probs.append("a replaced native keeps its particle row")
    hp = h.pid.to_numpy(np.int64)
    if np.isin(hp, natives).any():
        probs.append("a replaced native keeps hits")
    if not np.array_equal(y.particle_id.to_numpy(), p.particle_id.to_numpy()):
        probs.append("ptype rows differ from particles rows")
    cl_hit = np.isin(hp, cl_pid) if nc else np.zeros(len(hp), bool)
    w = h.weight.to_numpy(np.float64)
    if cl_hit.any() and not np.all(w[cl_hit] == 0):
        probs.append("closure hit weight != 0")
    if expect_weight_sum is not None:
        s = float(np.nansum(w[~cl_hit]))
        if abs(s - expect_weight_sum) > W.WEIGHT_TOL:
            probs.append(f"host weight sum {s:.9f} != {expect_weight_sum:.9f}")
    cnt = pd.Series(hp[hp != 0]).value_counts()
    if not np.array_equal(cnt.reindex(pid).fillna(0).to_numpy(np.int64), p.n_hits.to_numpy(np.int64)):
        probs.append("n_hits differs from the hit count")
    if not np.isin(hp[hp != 0], pid).all():
        probs.append("hits of particles without a row")
    nh = len(hp)
    if not cl_hit[nh - int(cl_hit.sum()):].all():
        probs.append("closure hits are not the last rows")
    for k in W.TABLES:
        col = "pid" if k == "hits" else "particle_id"
        ht = host[k].filter(pa.array(~np.isin(host[k].column(col).to_numpy(), natives)))
        mt = merged[k].slice(0, ht.num_rows)
        for name in ht.schema.names:
            if name != "event" and not W.columns_equal(ht.column(name), mt.column(name)):
                probs.append(f"host {k}.{name} altered")
    # the converter view: kinematics, type and vertex field of each closure row against its native
    hpart = host["particles"].to_pandas().set_index("particle_id")
    if nc and is_closure_id(cl_pid).all() and np.isin(native_particle_id(cl_pid), hpart.index).all():
        cl = p.iloc[len(p) - nc:].reset_index(drop=True)
        nid = native_particle_id(cl_pid)
        nat = hpart.loc[nid]
        for k in KINEMATICS:
            if not W.columns_equal(pa.array(cl[k].to_numpy()), pa.array(nat[k].to_numpy())):
                probs.append(f"closure {k} differs from the native")
        if not np.array_equal(cl.vprim.to_numpy(np.int64), nat.vprim.to_numpy(np.int64) + CLOSURE_VERTEX[0]):
            probs.append("closure vprim != native vprim + 1024")
        hy = host["ptype"].to_pandas().set_index("particle_id").particle_type
        if not np.array_equal(y.particle_type.to_numpy()[len(y) - nc:], hy.loc[nid].to_numpy()):
            probs.append("closure particle_type differs from the native")
    elif nc:
        probs.append("closure rows do not map to native rows of the host")
    return probs


def removed_weight(host: dict[str, pa.Table], natives) -> tuple[float, int]:
    """Summed score weight and count of the host hits of the replaced natives."""
    hp = host["hits"].column("pid").to_numpy()
    m = np.isin(hp, np.asarray(natives, np.int64))
    return float(np.nansum(host["hits"].column("weight").to_numpy().astype(np.float64)[m])), int(m.sum())


# ------------------------------------------------------------------------------------------------ assembly
def _sha256_self() -> str:
    return C.sha256_file(os.path.abspath(__file__))


def assemble(set_name: str, closure_dir: str, sim_path: str, root: str, tables_path: str, lookup_path: str,
             host_root: str = TP, csv_tar: str | None = None, overwrite: bool = False) -> dict:
    """Emulation, digitisation, conversion, merge and storage of a closure set made by `make_closure_set`
    (closure_dir) and simulated by scripts/acts/overlay_sim.py sim, packed by hnl.overlay.fatras_io (sim_path)."""
    from hnl.overlay import emulate as EM
    if set_name != "closure":
        raise ValueError("closure events belong to the closure set (set index 1)")
    with open(os.path.join(closure_dir, "closure_events.json")) as fh:
        cev = json.load(fh)
    truth = pd.read_parquet(os.path.join(closure_dir, "closure_truth.parquet"))
    sim = pd.read_parquet(sim_path)
    prov = W.sim_provenance(os.path.dirname(os.path.abspath(sim_path)), cev, lookup_path)
    status = W.sim_status(sim_path, closure_dir, cev, prov["pmin_gev"])
    st_key = pd.Series(status.status.to_numpy(), index=pd.MultiIndex.from_arrays(
        [status.event.to_numpy(np.int64), status.particle_id.to_numpy(np.uint64)]))
    truth["sim_status"] = st_key.reindex(pd.MultiIndex.from_arrays(
        [truth.fatras_event.to_numpy(np.int64), truth.acts_barcode.to_numpy(np.uint64)])).to_numpy()
    if truth.sim_status.isna().any():
        raise RuntimeError("closure truth rows without a Fatras status")
    tab = EM.Tables.load(tables_path)
    ehash, ohash = EM.emulation_hash(tables_path), W.overlay_hash(tables_path)
    mods, lookup, det = D.Modules.load(), D.load_lookup(lookup_path), C.load_detectors()
    base = int(cev["base_seed"])
    merged_all, side, gz_all = {}, [], {}
    for ev in cev["events"]:
        oid, host, fe = int(ev["overlay_event"]), int(ev["host"]), int(ev["fatras_event"])
        if W.split_overlay_id(oid) != (W.SETS[set_name][0], host):
            raise ValueError(f"overlay id {oid} is not the closure id of host {host}")
        fp = pd.read_csv(W.FIO_path(closure_dir, fe), dtype={"particle_id": np.uint64})
        tr = truth[truth.event == oid]
        rows = closure_rows(tr, fp)
        s = sim[sim.event == fe].reset_index(drop=True)
        st = status[status.event == fe]
        lost = st[st.status != "simulated"]
        if np.isin(s.particle_id.to_numpy(np.uint64), lost.particle_id.to_numpy(np.uint64)).any():
            raise RuntimeError(f"closure event {oid}: sim hits of a particle that Fatras did not simulate")
        g = closure_event(s, rows, lookup, mods, tab, base, oid)
        em, dig, gz = g["emulation"], g["digitised"], g["gz"]
        ctab, dq = W.convert_gun(gz, oid, det)
        host_t = W.host_tables(host, host_root)
        natives = tr.native_particle_id.to_numpy(np.int64)
        w_rm, n_rm = removed_weight(host_t, natives)
        exp_w = 1.0 - w_rm
        merged, rm = W.merge_event(host_t, ctab, oid, drop_pids=natives)
        probs = check_closure_event(merged, host_t, natives, exp_w)
        if probs:
            raise RuntimeError(f"closure event {oid}: {probs}")
        merged_all[oid], gz_all[oid] = merged, gz
        cat = ctab["particles"].column("category").to_numpy(zero_copy_only=False)
        bc_to_tml = dict(zip(rows.particle_id.to_numpy(np.uint64).tolist(), rows.tml_id.tolist()))
        info = {k: v for k, v in rm["info"].items() if k not in ("n_gun_hits", "n_gun_particles")}
        side.append({**info, "overlay_event": oid, "host": host, "fatras_event": fe, "n_gun_particles": 0,
                     "n_closure_particles": int(len(natives)), "n_closure_hits": int(rm["info"]["n_gun_hits"]),
                     "native_particle_ids": [int(x) for x in natives],
                     "populations": {k: int((tr.population == k).sum()) for k in POPULATIONS},
                     "removed_host_hits": n_rm, "removed_weight": w_rm, "expected_host_weight_sum": exp_w,
                     "closure_hit_rows": [int(rm["info"]["n_host_hits"]),
                                          int(rm["info"]["n_host_hits"] + rm["info"]["n_gun_hits"])],
                     "sim_hits": int(len(s)), "stopped_hits": int(em["stopped"].sum()),
                     "dropped_hits": int(em["dropped"].sum()), "digitiser": dig.stats,
                     "closure_dq_max_outside_mm": dq["max_outside_mm"],
                     "closure_categories": {str(c): int((cat == c).sum()) for c in np.unique(cat)},
                     "sim_failed_particles": [int(bc_to_tml[int(b)]) for b in
                                              lost[lost.status == "failed"].particle_id.to_numpy(np.uint64)],
                     "sim_below_pmin_particles": [int(bc_to_tml[int(b)]) for b in
                                                  lost[lost.status == "below_pmin"].particle_id.to_numpy(np.uint64)]})
    meta_extra = {"kind": "closure", "emulation_hash": ehash, "overlay_hash": ohash, "seed_rules": W.SEED_RULES,
                  "id_rule": cev["id_rule"], "barcode_rule": cev["barcode_rule"], "emulation_tables": tables_path,
                  "lookup": lookup_path, "lookup_sha256": C.sha256_file(lookup_path), "sim": prov,
                  "sim_status_counts": {str(k): int(v) for k, v in status.status.value_counts().items()},
                  "closure": {k: v for k, v in cev.items() if k != "events"}, "host_root": host_root,
                  "closure_code_sha256": _sha256_self()}
    meta = W.write_set(set_name, merged_all, root, meta_extra, overwrite=overwrite)
    side_rec = {"set": set_name, "kind": "closure", "emulation_hash": ehash, "overlay_hash": ohash,
                "closure_particles": f"closure_truth_{set_name}.parquet", "sim": prov,
                "sim_status_counts": meta_extra["sim_status_counts"], "events": side}
    with open(os.path.join(root, f"sidecar_{set_name}.json"), "w") as fh:
        json.dump(side_rec, fh, indent=1)
    truth.to_parquet(os.path.join(root, f"closure_truth_{set_name}.parquet"), index=False)
    tar = W.write_csv_tar(csv_tar, gz_all) if csv_tar else None
    return {"meta": meta, "sidecar": side_rec, "csv_tar": tar}


def verify_closure_set(set_name: str, root: str, host_root: str = TP) -> dict:
    """Re-read a stored closure set: sha256, schema and row groups of every shard (as hnl.overlay.writer.verify_set),
    then `check_closure_event` of every event against the host's stored rows (sidecar: host, replaced natives,
    expected weight sum)."""
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        side = {int(r["overlay_event"]): r for r in json.load(fh)["events"]}
    problems = []
    for sh in meta["shards"].values():
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
    if problems:
        return {"set": set_name, "events": len(ids), "problems": problems, "per_event": per_event}
    if sorted(side) != ids:
        problems.append("sidecar events differ from the stored events")
    for e in ids:
        merged = W.host_tables(e, root)
        s = side[e]
        pr = check_closure_event(merged, W.host_tables(s["host"], host_root), s["native_particle_ids"],
                                 s["expected_host_weight_sum"])
        per_event[e] = pr
        problems += [f"event {e}: {x}" for x in pr]
    return {"set": set_name, "events": len(ids), "problems": problems, "per_event": per_event}


# ------------------------------------------------------------------------------------------------ CLI
def _write_population(out: str) -> dict:
    t0 = time.time()
    t, rec = population_table()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    t.to_parquet(out, index=False)
    rec.update({"table": out, "table_sha256": C.sha256_file(out), "rows": int(len(t)),
                "val_regions": compare_val_regions(rec["counts"]), "seconds": round(time.time() - t0, 1),
                "slurm_job": os.environ.get("SLURM_JOB_ID")})
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    m = sub.add_parser("make")
    m.add_argument("--hosts", required=True, help="host ids separated by commas or colons")
    m.add_argument("--set", default="closure", choices=["closure"])
    m.add_argument("--population", required=True)
    m.add_argument("--base-seed", type=int, default=None, help="default: the closure base seed of split_h_v1.json")
    m.add_argument("--out", required=True)
    a_ = sub.add_parser("assemble")
    a_.add_argument("--set", default="closure", choices=["closure"])
    a_.add_argument("--dir", required=True)
    a_.add_argument("--sim", required=True)
    a_.add_argument("--root", required=True)
    a_.add_argument("--tables", default=os.path.join(W.REPO, "configs", "trackml", "overlay_emulation_v1.json"))
    a_.add_argument("--lookup", default=D.PATCHED_LOOKUP)
    a_.add_argument("--csv-tar", default=None)
    a_.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--set", default="closure", choices=["closure"])
    v.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        rec = _write_population(a.out)
        print(json.dumps({k: rec[k] for k in ("events", "counts", "selected", "prompt_subset", "val_regions")},
                         indent=1))
    elif a.cmd == "make":
        hosts = [int(x) for x in a.hosts.replace(":", ",").split(",") if x]
        base = a.base_seed if a.base_seed is not None else default_base_seed(a.set)
        pop = closure_population(pd.read_parquet(a.population), hosts)
        rec = make_closure_set(pop, {h: W.overlay_id(a.set, h) for h in hosts}, base, a.out,
                               C.sha256_file(a.population))
        print(json.dumps({k: v_ for k, v_ in rec.items() if k != "events"}, indent=1))
    elif a.cmd == "assemble":
        r = assemble(a.set, a.dir, a.sim, a.root, a.tables, a.lookup, csv_tar=a.csv_tar, overwrite=a.overwrite)
        print(json.dumps({"events": r["meta"]["n_events_converted"], "csv_tar": r["csv_tar"],
                          "sim_status_counts": r["sidecar"]["sim_status_counts"]}, indent=1, default=str))
    else:
        r = verify_closure_set(a.set, a.root)
        print(json.dumps({k: r[k] for k in ("set", "events", "problems")}, indent=1))
        if r["problems"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
