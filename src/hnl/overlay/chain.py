"""Model stages of the frozen Canto-L (TrackML) chain on overlay events (addendum H, docs/preregistration_pileup.md
sections 5 and 10, item 5 before the first model run on an overlay id).

For one overlay set (pilot, hval, closure, htest), one condition (full, 1gev, hs) and one block of the set's ids,
three stages call the frozen functions of the native chain, unchanged and in the order of
slurm/trackml/test_chain.sh, with the inputs of configs/trackml/selection_v1.json:
  additions  hnl.trackml.graphs.cmd_additions on CUDA (FRNN base and wide graphs, torch learned support: the backends of
             the native val and test additions)               -> OUT/additions/additions_<first>.parquet (+ .json)
  store      hnl.trackml.graphs.cmd_stage on CUDA (base graph rebuilt from the embedding plus the stored additions,
             --allow-backend-change and --skip-existing as slurm/trackml/stage_store.sbatch)
                                                              -> OUT/store/<block>/event<eid>.pyg.gz
  score      hnl.trackml.infer.cmd_score on the CPU, the whole event in one sector with the memory-bounded forward
             hnl.trackml.chunked (addendum section 5)         -> OUT/scores/scores_<first>.parquet (+ .json)
`preflight` runs hnl.trackml.freeze.verify first and refuses HNL_TML_CONFIG / HNL_TML_SPLIT (the stages read
configs/trackml/pipeline_v1.yaml, the file selection_v1.json pins).

Conditions. C-full and C-1GeV are the frozen conditions of hnl.trackml.events. C-hs is the truth filter
hnl.overlay.conditions.with_condition, applied to the input tables before graph building: inside these calls only,
hnl.trackml.events.with_condition passes "hs" to hnl.overlay.conditions.with_condition, and the event reader keeps the
particles table that the filter needs. The frozen CLIs and their --condition choices are unchanged, and C-full and
C-1GeV run through exactly the frozen code. No model code changes.

Requests (`check_request`; slurm/overlay/guard_h.sh checks them first, and every stage calls hnl.overlay.grant.check
with the job's grant: kind plumb on the pilot, model otherwise, out = OUT, under which every stage writes; the frozen
CLIs check it again with their own output directory):
  pilot    (k = 9)  kind plumb only (decision D6): the plumbing test of the model stages, C-hs and the sealed write. One
                    block; outputs in a scratch directory outside the TrackML data root, kept, sealed right after the
                    scores are written (`finish`), and checked for structure only (`check_outputs`). No track building
                    and no evaluation reads them (hnl.overlay.tracks and hnl.overlay.blind take placeholder scores only
                    on k = 9).
  hval     (k = 0)  before configs/trackml/overlay_version_h_v1.json (overlay_hash) is committed: C-full only, into the
                    sealed directory RUNS_H/sealed/hval_full (scores/ as hnl.overlay.blind reads them), refused once it
                    is sealed or its blinded evaluation has begun; after that commit: C-1GeV and C-hs only.
  closure  (k = 1)  C-full and C-1GeV.
  htest    (k = 2)  C-full, C-1GeV and C-hs (the guard requires R15 and the launch token).
Production outputs (`paths`): RUNS_H/model/<set>_<cond> (H-val C-full: the sealed directory) and the grids of
hnl.overlay.tracks under RUNS_H/grid/<set>_<cond>; nothing else is accepted for a production set. A block holds at most
50 events (one scored-edge file).

Records. OUT/chain/<stage>_<block>.json per stage: the grant, the code state it binds (hnl.overlay.grant.code_state:
the blob hashes of the code an overlay job runs and the addendum blob, decision E1), the frozen inputs used and their
sha256, the node, GPU and CPU count, and per event the wall time (from the read of its input to the read of the next
one) and the peak resident and CUDA memory. OUT/chain/<block>.done once the block's scores are written and their
sidecars checked (classifier sha256 = selection_v1.json, condition, forward); the block's store is then removed unless
kept (always kept on the pilot).

Plumbing checks (`check_outputs`, decision D6): files, schemas, row counts per event (additions, store hits and edges,
score rows), sidecar provenance, and the hits each condition keeps (store rows and particle ids against the truth
tables: C-full every hit; C-1GeV hits of particles with pT > 1 GeV, no noise; C-hs vertex fields 1 and 1001-1010 only).
No score value and no edge label is read, and no efficiency, fake rate, matched count or per-target number is computed.
`plumb_record` collects the checks, the timing and memory, and extrapolates the cost of H-val, closure and H-test under
both score placements: scores in CPU jobs on the express CPU nodes (slurm/overlay/chain_score.sbatch), at a CPU-node
score time (of the pilot CPU job that scores a copy of the GPU job's additions and store again,
DIR/pilot_<cond>_cpuscore, sealed and checked the same way, or of the chunked forward on the pilot hosts' native val
events from a chunked-vs-plain check record, --cpu-timing; the one measured at SCORE_CPUS threads first), or every
stage inside the GPU job.

Score placement (decision E2, `PLACEMENT`): production sets score on the CPU nodes. `run` refuses, for a production
set, a stage list that holds the score stage together with additions or store (slurm/overlay/chain.sbatch runs
additions,store; slurm/overlay/chain_score.sbatch runs score); the pilot plumbing run keeps all three in one job.
Job layout (`layout`): the GPU array makes the additions and the store of N blocks of at most 50 events; each block is
then scored by `score_parts` CPU jobs of SCORE_CPUS threads and at most SCORE_EVENTS_PER_JOB[cond] events each
(`score_ids`: 5 in C-full, 50 in C-1GeV and C-hs), so that every score job ends within the 1 h limit; `run` refuses a
production score job without its part when a block has several, and one with fewer threads than SCORE_CPUS. The score
jobs of a block write side by side into scores/; the last of them to see every event of the block scored removes the
block's store. `extrapolate` and `cost` compute the cost of exactly this layout.

    python -m hnl.overlay.chain preflight
    python -m hnl.overlay.chain paths --set S --cond C [--key out|scores|grid]
    python -m hnl.overlay.chain block-ids --set S --block K --n-blocks N
    python -m hnl.overlay.chain layout --set S --cond C                    (N, S and the two sbatch lines)
    python -m hnl.overlay.chain score-parts --set S --cond C --n-blocks N
    python -m hnl.overlay.chain score-ids --set S --cond C --block K --n-blocks N --part J
    python -m hnl.overlay.chain run --set S --cond C --block K --n-blocks N [--out DIR] [--stages additions,store,score]
        [--keep-store] [--part J]
    python -m hnl.overlay.chain finish --set pilot --cond C --out DIR --json CHECK_JSON     (seal, verify, check)
    python -m hnl.overlay.chain dry --set pilot --cond C --out DIR [--root R]     (placeholder scores, kind dry)
    python -m hnl.overlay.chain plumb-record --dir DIR --out-json J [--cpu-timing CHUNKED_CHECK_JSON]
    python -m hnl.overlay.chain cost --record PLUMB_JSON [--out-json J]     (the cost of this layout, from a record)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import time

import numpy as np

from hnl.overlay import blind as B

CONDITIONS = B.CONDITIONS_H
SETS = {"pilot": 9, "hval": 0, "closure": 1, "htest": 2}
STAGES = ("additions", "store", "score")
MAX_BLOCK = 50                    # events per block: hnl.trackml.infer writes 50 events per scored-edge file
FORWARD = "chunked"               # addendum section 5: the memory-bounded forward (hnl.trackml.chunked)
DEVICE_GPU = "cuda"               # additions and store: the backends of the native val / test additions (frnn_cuda)
# event counts for the cost extrapolation (split_h_v1.json; addendum section 10)
PLAN = {"hval": (150, ("full", "1gev", "hs")), "closure": (150, ("full", "1gev")),
        "htest": (600, ("full", "1gev", "hs"))}
HOUR_S = 3600.0
PLACEMENT = "scores_on_cpu_nodes"  # decision E2 (2026-10-15): production edge scores in CPU jobs (chain_score.sbatch)
SCORE_CPUS = 6                     # chain_score.sbatch -c 6: the thread count at which the CPU score time was measured
# Events per score job (decision E2; `score_parts`), so that a score job of SCORE_CPUS threads ends within 75 % of the
# 1 h express limit at the per-event times of the 6-thread forward check on the pilot hosts' native val events
# (pileup_design/h_fix/chunked_check_v1.json): C-full 472 s per event (slowest of five 632 s), so 5 events, about 40 min
# with the job's overhead and under 1 h even at the slowest time; C-1GeV 23.2 s, a whole block of 50 (about 20 min).
# C-hs was not timed at 6 threads; its events hold fewer hits than C-1GeV events and scored faster in the GPU job
# (13.4 s against 17.6 s per event at 16 CPUs), so C-hs takes the C-1GeV value.
SCORE_EVENTS_PER_JOB = {"full": 5, "1gev": 50, "hs": 50}
STORE_SCHEMA = {"r": "float32", "phi": "float32", "z": "float32", "pid": "int64", "pairs": "int64", "y": "bool",
                "rows": "int64"}
ADDITION_COLUMNS = [("event", "int32"), ("a", "int32"), ("b", "int32")]
SCORE_COLUMNS = [("event", "int32"), ("a", "int32"), ("b", "int32"), ("score", "float"), ("y", "bool")]


def kind_of(set_name: str) -> str:
    """The guard / grant kind of a model-stage job: plumb on the pilot (decision D6), model otherwise."""
    return "plumb" if set_name == "pilot" else "model"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ layout and requests
def paths(set_name: str, cond: str, runs: str = B.RUNS_H) -> dict:
    """Output directories of a production set: model outputs RUNS_H/model/<set>_<cond> (H-val C-full: the sealed
    directory RUNS_H/sealed/hval_full), scores in its scores/, grids of hnl.overlay.tracks RUNS_H/grid/<set>_<cond>
    (none for H-val C-full, which hnl.overlay.blind alone evaluates)."""
    if set_name == "pilot":
        raise ValueError("pilot outputs go to a scratch directory named explicitly (OUT)")
    if set_name not in SETS:
        raise KeyError(f"no overlay set {set_name!r} for the model stages")
    if cond not in CONDITIONS:
        raise ValueError(f"condition {cond!r} not in {CONDITIONS}")
    if set_name == "closure" and cond == "hs":
        raise PermissionError("the closure set (k = 1) runs in C-full and C-1GeV only")
    sealed = set_name == "hval" and cond == "full"
    out = B.sealed_dir("hval", "full", runs) if sealed else os.path.join(runs, "model", f"{set_name}_{cond}")
    return {"out": out, "scores": os.path.join(out, "scores"),
            "grid": None if sealed else os.path.join(runs, "grid", f"{set_name}_{cond}")}


def block_name(block: int, n_blocks: int) -> str:
    return B.block_name(block, n_blocks)


def block_ids(set_name: str, block: int, n_blocks: int, split: dict | None = None) -> list[int]:
    """The ids of block K of N of a set (the slicing of hnl.overlay.blind.evaluate and hnl.overlay.tracks.grid)."""
    if not 0 <= int(block) < int(n_blocks):
        raise ValueError(f"block {block} not in 0..{n_blocks - 1}")
    events = B.set_ids(set_name, split)
    per = -(-len(events) // int(n_blocks))
    if per > MAX_BLOCK:
        raise ValueError(f"{len(events)} events in {n_blocks} blocks: {per} per block, at most {MAX_BLOCK}")
    ids = events[int(block) * per:(int(block) + 1) * per]
    if not ids:
        raise ValueError(f"block {block} of {n_blocks} is empty")
    return ids


def score_parts(set_name: str, cond: str, n_blocks: int, split: dict | None = None) -> int:
    """Score jobs per block of the GPU array (decision E2): the GPU jobs (slurm/overlay/chain.sbatch) make the additions
    and the event store of blocks of up to MAX_BLOCK events; each block is then scored by this many CPU jobs
    (slurm/overlay/chain_score.sbatch), each of at most SCORE_EVENTS_PER_JOB[cond] of the block's events, so that every
    score job ends within the 1 h limit. Block 0 is the largest block."""
    if cond not in SCORE_EVENTS_PER_JOB:
        raise ValueError(f"condition {cond!r} not in {CONDITIONS}")
    per = len(block_ids(set_name, 0, n_blocks, split))
    return -(-per // SCORE_EVENTS_PER_JOB[cond])


def score_ids(set_name: str, cond: str, block: int, n_blocks: int, part: int, split: dict | None = None) -> list[int]:
    """The ids that score job `part` (0 .. score_parts - 1) of block `block` of N scores: the block's ids cut into
    score_parts equal slices, each of at most SCORE_EVENTS_PER_JOB[cond] events (empty for a part beyond the events
    of a short last block)."""
    ids = block_ids(set_name, block, n_blocks, split)
    n_parts = score_parts(set_name, cond, n_blocks, split)
    if not 0 <= int(part) < n_parts:
        raise ValueError(f"part {part} not in 0..{n_parts - 1} (block {block} of {n_blocks}, condition {cond})")
    per = -(-len(ids) // n_parts)
    sel = ids[int(part) * per:(int(part) + 1) * per]
    if len(sel) > SCORE_EVENTS_PER_JOB[cond]:              # cannot happen: ceil(a / ceil(a / m)) <= m
        raise ValueError(f"{len(sel)} events in one score job, at most {SCORE_EVENTS_PER_JOB[cond]} in {cond}")
    return sel


def part_name(block: int, n_blocks: int, part: int, n_parts: int) -> str:
    """Name of a score job's records: the block's name, with the part when the block has several score jobs."""
    b = block_name(block, n_blocks)
    return b if int(n_parts) == 1 else f"{b}p{int(part):02d}of{int(n_parts):02d}"


def layout(set_name: str, cond: str, split: dict | None = None) -> dict:
    """The production job arrays of a set in a condition (decision E2): N GPU blocks of at most MAX_BLOCK events
    (slurm/overlay/chain.sbatch) and N x S score jobs (slurm/overlay/chain_score.sbatch), with the sbatch lines."""
    paths(set_name, cond)                                  # production sets and their conditions only
    n = len(B.set_ids(set_name, split))
    nb = -(-n // MAX_BLOCK)
    s = score_parts(set_name, cond, nb, split)
    return {"set": set_name, "condition": cond, "events": n, "n_blocks": nb, "events_per_block": -(-n // nb),
            "score_parts": s, "score_jobs": nb * s, "score_events_per_job_max": SCORE_EVENTS_PER_JOB[cond],
            "gpu": f"sbatch --array=0-{nb - 1} slurm/overlay/chain.sbatch {set_name} {cond} {nb}",
            "score": f"sbatch --dependency=afterok:<GPU array job> --array=0-{nb * s - 1} "
                     f"slurm/overlay/chain_score.sbatch {set_name} {cond} {nb}"}


def evaluation_started(sealed: str) -> bool:
    """True once hnl.overlay.blind has written into a sealed directory (blocks of counts or raw track labels)."""
    return any(os.path.exists(os.path.join(sealed, d)) for d in ("counts", "tracks"))


def check_request(set_name: str, cond: str, out: str, ids, n_blocks: int = 1, split: dict | None = None,
                  runs: str = B.RUNS_H, repo: str = B.REPO, root: str | None = None) -> tuple[int, str | None]:
    """(set index, table root for hnl.trackml.events.iter_events) of an allowed request; raises PermissionError
    otherwise (module docstring)."""
    split = split or B.load_split()
    if set_name not in SETS:
        raise PermissionError(f"no model stage for set {set_name!r}")
    k = B.set_index(set_name, split)
    if k != SETS[set_name]:
        raise ValueError(f"set {set_name}: k = {k} in split_h_v1.json, expected {SETS[set_name]}")
    if cond not in CONDITIONS:
        raise ValueError(f"condition {cond!r} not in {CONDITIONS}")
    bad = B.check_ids(set_name, ids, split)
    if bad:
        raise PermissionError(f"ids of {set_name}: {bad}")
    if len(list(ids)) > MAX_BLOCK:
        raise ValueError(f"{len(list(ids))} ids: a block holds at most {MAX_BLOCK} events")
    root = B.overlay_root(k, root)
    if B.is_sealed(out):
        raise PermissionError(f"{out} is sealed: nothing is written into it again")
    if k == 9:
        if B._under(out, B.EXT) or B._under(out, runs):
            raise PermissionError("plumbing outputs of the pilot set go to a scratch directory outside the TrackML "
                                  "data root (decision D6)")
        if int(n_blocks) != 1:
            raise PermissionError("the pilot plumbing run is one block (its seal and checks cover the whole set)")
        return k, root
    if k == 1 and cond == "hs":
        raise PermissionError("the closure set (k = 1) runs in C-full and C-1GeV only")
    if k == 0:
        if B.unblinded(repo):
            if cond == "full":
                raise PermissionError("the C-full model run on H-val is not repeated after the overlay_hash commit")
        elif cond != "full":
            raise PermissionError("until configs/trackml/overlay_version_h_v1.json (overlay_hash) is committed, the "
                                  "model runs on H-val in C-full only, into the sealed directory")
        if cond == "full":
            sealed = B.sealed_dir("hval", "full", runs)
            if os.path.realpath(out) != os.path.realpath(sealed):
                raise PermissionError(f"H-val C-full model outputs go to the sealed directory {sealed} only")
            if B.is_sealed(sealed):
                raise PermissionError(f"{sealed} is sealed: no further model output")
            if evaluation_started(sealed):
                raise PermissionError(f"the blinded evaluation of {sealed} has begun: no further model output")
            return k, root
    want = paths(set_name, cond, runs)["out"]
    if os.path.realpath(out) != os.path.realpath(want):
        raise PermissionError(f"{set_name} in {cond}: model outputs go to {want} only")
    return k, root


# ------------------------------------------------------------------------------------------------ frozen inputs
def frozen_inputs(selection: str = B.SELECTION) -> dict:
    """Run directory, graph choice, classifier checkpoint and pipeline config of selection_v1.json, with sha256."""
    with open(selection) as fh:
        d = json.load(fh)
    i = d["inputs"]
    return {"selection": os.path.abspath(selection), "selection_sha256": B.sha256_file(selection), "run": d["run"],
            "graph_choice": i["graph_choice"]["path"], "graph_choice_sha256": i["graph_choice"]["sha256"],
            "classifier": i["classifier_checkpoint"]["path"],
            "classifier_sha256": i["classifier_checkpoint"]["sha256"],
            "pipeline_config": i["pipeline_config"]["path"], "pipeline_config_sha256": i["pipeline_config"]["sha256"]}


def preflight(selection: str = B.SELECTION) -> dict:
    """hnl.trackml.freeze.verify on selection_v1.json and the environment of the stages; SystemExit on a problem.
    Returns the frozen inputs."""
    from hnl.trackml import config as CFG
    from hnl.trackml import freeze
    problems = [f"{v} is set: the stages read configs/trackml/pipeline_v1.yaml and its split only"
                for v in ("HNL_TML_CONFIG", "HNL_TML_SPLIT") if os.environ.get(v)]
    fi = frozen_inputs(selection)
    if os.path.realpath(fi["pipeline_config"]) != os.path.realpath(CFG.DEFAULT):
        problems.append(f"selection pins {fi['pipeline_config']}, the stages read {CFG.DEFAULT}")
    problems += [f"freeze verify: {p}" for p in freeze.verify(selection, fi["run"])]
    if problems:
        raise SystemExit("refusing: " + "; ".join(problems))
    return fi


def _grant(ids, kind: str, cond: str, out: str | None = None) -> dict:
    """hnl.overlay.grant.check for these ids, writing under `out` (it must lie under the grant's out); returns the
    grant (path, sha256 and content) for the records."""
    from hnl.overlay import grant as G
    G.check([int(i) for i in ids], kind, cond, out=out)
    path = os.environ.get("HNL_H_GRANT")
    rec = {"path": path, "kind": kind, "cond": cond, "out": os.path.realpath(out) if out else None}
    if path and os.path.exists(path):
        rec["sha256"] = B.sha256_file(path)
        with open(path) as fh:
            rec["content"] = json.load(fh)
    return rec


def _code() -> dict | None:
    """The code state a grant binds (hnl.overlay.grant.code_state: the blob hashes at HEAD of the code an overlay job
    runs and of the addendum, decision E1), for the records."""
    from hnl.overlay import grant as G
    try:
        return G.code_state()
    except RuntimeError:
        return None


# ------------------------------------------------------------------------------------------------ per-event meter
def _vmhwm_gb() -> float | None:
    try:
        with open("/proc/self/status") as fh:
            m = re.search(r"VmHWM:\s+(\d+)\s+kB", fh.read())
        return round(int(m.group(1)) / 1e6, 3) if m else None
    except OSError:
        return None


def _reset_vmhwm() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


class Meter:
    """Per-event wall time and peak memory of one stage. An event's interval runs from the read of its input to the
    read of the next event's input (or the end of the stage). Peak resident memory of this process (VmHWM, reset at
    every event through /proc/self/clear_refs; cumulative if the reset is refused) and peak CUDA memory allocated
    (reset at every event)."""

    def __init__(self):
        self.rows: list[dict] = []
        self.cur = None
        self.t0 = None
        self.start = time.time()
        self.rss_reset = None

    def _cuda(self):
        """torch.cuda once CUDA is in use in this process, else None."""
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda
        return None

    def mark(self, event) -> None:
        now = time.time()
        if self.cur is not None:
            t = self._cuda()
            self.rows.append({"event": int(self.cur), "seconds": round(now - self.t0, 2), "peak_rss_gb": _vmhwm_gb(),
                              "peak_gpu_gb": round(t.max_memory_allocated() / 1e9, 3) if t else None})
        self.cur, self.t0 = event, now
        if event is not None:
            self.rss_reset = _reset_vmhwm()
            t = self._cuda()
            if t:
                t.reset_peak_memory_stats()

    def close(self) -> None:
        if self.cur is not None:
            self.mark(None)

    def summary(self) -> dict:
        return {"seconds_total": round(time.time() - self.start, 2), "per_event": self.rows,
                "seconds_events": round(sum(r["seconds"] for r in self.rows), 2), "rss_reset_per_event": self.rss_reset}


@contextlib.contextmanager
def overlay_condition(cond: str, table_root: str | None = None, meter: Meter | None = None):
    """Inside: hnl.trackml.events.with_condition passes "hs" to hnl.overlay.conditions.with_condition (C-full and
    C-1GeV stay the frozen function, which is not replaced), and hnl.trackml.events.iter_events reads the particles
    table for C-hs, reads from `table_root` (a pilot scratch root; None: the production tables, as
    hnl.trackml.events.table_root) and marks each event for the meter."""
    from hnl.overlay import conditions as OC
    from hnl.trackml import events as E
    if cond not in CONDITIONS:
        raise ValueError(f"condition {cond!r} not in {CONDITIONS}")
    orig_wc, orig_it = E.with_condition, E.iter_events

    def with_condition(ev, c):
        return OC.with_condition(ev, c) if c == "hs" else orig_wc(ev, c)

    def iter_events(events, chunk: int = 25, root: str | None = None, particles: bool = True):
        for ev in orig_it(events, chunk=chunk, root=root or table_root, particles=particles or cond == "hs"):
            if meter is not None:
                meter.mark(ev.event)
            yield ev

    if cond == "hs":
        E.with_condition = with_condition
    E.iter_events = iter_events
    try:
        yield
    finally:
        E.with_condition, E.iter_events = orig_wc, orig_it
        if meter is not None:
            meter.close()


@contextlib.contextmanager
def metered_store_reads(meter: Meter):
    """Inside: hnl.trackml.classifier.load_event_graph marks each stored event for the meter (score stage)."""
    from hnl.trackml import classifier as CL
    from hnl.trackml.events import event_from_eid
    orig = CL.load_event_graph

    def load_event_graph(path: str):
        meter.mark(event_from_eid(int(os.path.basename(path)[5:].split(".")[0])))
        return orig(path)

    CL.load_event_graph = load_event_graph
    try:
        yield
    finally:
        CL.load_event_graph = orig
        meter.close()


# ------------------------------------------------------------------------------------------------ stages
def _cfg():
    from hnl.trackml import config as CFG
    return CFG.load(CFG.DEFAULT)


def _events_arg(ids) -> str:
    return ",".join(str(int(i)) for i in ids)


def stage_additions(ids, cond: str, out: str, fi: dict, device: str = DEVICE_GPU, root: str | None = None,
                    meter: Meter | None = None) -> None:
    """hnl.trackml.graphs additions on the block (events already stored in OUT/additions are skipped)."""
    from hnl.trackml import graphs as GR
    a = argparse.Namespace(events=_events_arg(ids), graph_choice=fi["graph_choice"],
                           out=os.path.join(out, "additions"), per_file=MAX_BLOCK, condition=cond,
                           support_backend=None, device=device, allow_test=False)
    with overlay_condition(cond, root, meter):
        GR.cmd_additions(a, _cfg())


def store_dir(out: str, block: str) -> str:
    return os.path.join(out, "store", block)


def stage_store(ids, cond: str, out: str, block: str, fi: dict, device: str = DEVICE_GPU, root: str | None = None,
                meter: Meter | None = None) -> None:
    """hnl.trackml.graphs stage of the block into OUT/store/<block> (events already there are kept)."""
    from hnl.trackml import graphs as GR
    a = argparse.Namespace(events=_events_arg(ids), graph_choice=fi["graph_choice"],
                           additions=os.path.join(out, "additions"), dest=store_dir(out, block), condition=cond,
                           device=device, allow_test=False, allow_backend_change=True, skip_existing=True)
    with overlay_condition(cond, root, meter):
        GR.cmd_stage(a, _cfg())


def stored_events(directory: str) -> dict[int, str]:
    """{event: path} of the event files below a store directory (recursive, as acorn's load_datafiles_in_dir)."""
    from hnl.trackml.events import event_from_eid
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, "**", "event*.pyg*"), recursive=True)):
        name = os.path.basename(p)
        if not (name.endswith(".pyg") or name.endswith(".pyg.gz")):
            continue
        e = event_from_eid(int(name[5:].split(".")[0]))
        if e in out:
            raise ValueError(f"{directory}: event {e} stored twice")
        out[e] = p
    return out


def score_sidecars(scores: str, strict: bool = True) -> dict[str, dict]:
    """{file: sidecar} of the scored-edge files of a scores directory. strict=False skips a file whose sidecar is not
    (yet) complete: the score jobs of one block write into the same directory side by side, and hnl.trackml.infer
    writes a file's sidecar just after the file."""
    sides = {}
    for p in sorted(glob.glob(os.path.join(scores, "*.parquet"))):
        try:
            with open(p + ".json") as fh:
                sides[os.path.basename(p)] = json.load(fh)
        except (OSError, ValueError):
            if strict:
                raise
    return sides


def scored_events(scores: str, strict: bool = True) -> dict[int, str]:
    out = {}
    for f, s in score_sidecars(scores, strict).items():
        for e in s["events"]:
            if int(e) in out:
                raise ValueError(f"{scores}: event {e} scored in {out[int(e)]} and {f}")
            out[int(e)] = f
    return out


def check_score_sidecars(scores: str, ids, cond: str, fi: dict) -> list[str]:
    """Problems (empty if none) of the scored-edge files holding these events: each event once, the frozen classifier
    (path and sha256 of selection_v1.json), the condition, the whole-event layout and the memory-bounded forward."""
    have = scored_events(scores)
    bad = [f"no scores for event {e}" for e in ids if int(e) not in have]
    for f in sorted({have[int(e)] for e in ids if int(e) in have}):
        s = score_sidecars(scores)[f]
        if s.get("ckpt_sha256") != fi["classifier_sha256"] or s.get("ckpt") != fi["classifier"]:
            bad.append(f"{f}: not scored by the frozen classifier of selection_v1.json")
        if s.get("placeholder") or s.get("oracle_sigma") is not None:
            bad.append(f"{f}: placeholder or oracle scores")
        if s.get("condition") != cond:
            bad.append(f"{f}: condition {s.get('condition')}, not {cond}")
        if s.get("forward") != FORWARD or (s.get("layout") or {}).get("n_sectors") != 1 or s.get("device") != "cpu":
            bad.append(f"{f}: not the whole-event CPU forward {FORWARD}")
    return bad


def stage_score(ids, cond: str, out: str, block: str, fi: dict, meter: Meter | None = None,
                block_ids=None) -> None:
    """hnl.trackml.infer score of the block's store (or of `ids`, one score job's part of the block `block_ids`):
    whole event, CPU, memory-bounded forward, frozen classifier."""
    from hnl.trackml import infer as INF
    d = store_dir(out, block)
    whole = [int(e) for e in (ids if block_ids is None else block_ids)]
    part = [int(e) for e in ids]
    if not set(part) <= set(whole):
        raise ValueError(f"the ids to score are not events of block {block}")
    have = stored_events(d)
    if sorted(have) != sorted(whole):
        raise RuntimeError(f"{d}: the store holds {len(have)} events, not exactly the {len(whole)} of block {block}")
    if os.path.isdir(os.path.join(d, ".tmp")) and os.listdir(os.path.join(d, ".tmp")):
        raise RuntimeError(f"{d}/.tmp is not empty (an interrupted store): run the store stage again")
    scores = os.path.join(out, "scores")
    already = set(scored_events(scores, strict=False)) if os.path.isdir(scores) else set()
    if already & set(part):
        raise RuntimeError(f"events of block {block} are scored already: refusing to score them twice")
    a = argparse.Namespace(store=d, ckpt=fi["classifier"], out=scores, sectors="full",
                           device="cpu", oracle=None, oracle_true=2.2, oracle_fake=-4.6, condition=cond,
                           events=None if sorted(part) == sorted(whole) else _events_arg(part),
                           block=None, chunked=FORWARD == "chunked")
    with contextlib.ExitStack() as st:
        if meter is not None:
            st.enter_context(metered_store_reads(meter))
        INF.cmd_score(a, _cfg())


def drop_store(out: str, block: str, ids) -> bool:
    """Removes the block's store once every event of the block is scored (the last of the block's score jobs to see
    it; the directory is first renamed, so that two jobs never remove it together). True if this call removed it."""
    d = store_dir(out, block)
    scores = os.path.join(out, "scores")
    if not os.path.isdir(d) or not os.path.isdir(scores):
        return False
    if not {int(e) for e in ids} <= set(scored_events(scores, strict=False)):
        return False
    gone = f"{d}.removing.{os.getpid()}"
    try:
        os.rename(d, gone)
    except FileNotFoundError:
        return False
    shutil.rmtree(gone)
    return True


def _score_threads() -> int:
    """The threads of the score forward (hnl.trackml.config.threads: OMP_NUM_THREADS, else the CPUs of the process)."""
    return int(os.environ.get("OMP_NUM_THREADS") or len(os.sched_getaffinity(0)))


def _node() -> dict:
    import sys
    info = {"host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "cpus": len(os.sched_getaffinity(0)),
            "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        with open("/proc/cpuinfo") as fh:
            m = re.search(r"model name\s*:\s*(.+)", fh.read())
        info["cpu_model"] = m.group(1).strip() if m else None
    except OSError:
        pass
    return info


def run(set_name: str, cond: str, block: int = 0, n_blocks: int = 1, out: str | None = None,
        stages=STAGES, keep_store: bool = False, device: str = DEVICE_GPU, selection: str = B.SELECTION,
        split: dict | None = None, runs: str = B.RUNS_H, repo: str = B.REPO, root: str | None = None,
        part: int | None = None) -> dict:
    """The model stages of one block (module docstring). A production score job (stages = score alone) scores part
    `part` of the block (`score_ids`; it may be omitted when the block has one score job). Returns the block's
    record."""
    stages = [s for s in STAGES if s in set(stages)]
    if not stages or set(stages) - set(STAGES):
        raise ValueError(f"stages must be a subset of {STAGES}")
    if set_name != "pilot" and "score" in stages and len(stages) > 1:
        raise PermissionError(f"decision E2: the production edge scores run in a CPU job of their own "
                              f"(slurm/overlay/chain_score.sbatch, --stages score), not with {', '.join(stages[:-1])}")
    production_score = set_name != "pilot" and stages == ["score"]
    if part is not None and not production_score:
        raise ValueError("--part is for the score jobs of production sets (slurm/overlay/chain_score.sbatch)")
    fi = preflight(selection)
    split = split or B.load_split()
    if out is None:
        out = paths(set_name, cond, runs)["out"]
    ids = block_ids(set_name, block, n_blocks, split)
    n_parts, todo = 1, ids
    if production_score:
        n_parts = score_parts(set_name, cond, n_blocks, split)
        if part is None:
            if n_parts > 1:
                raise PermissionError(f"block {block} of {n_blocks} holds {len(ids)} events and a score job in {cond} "
                                      f"at most {SCORE_EVENTS_PER_JOB[cond]} (decision E2, the 1 h limit): give "
                                      f"--part 0..{n_parts - 1} (slurm/overlay/chain_score.sbatch)")
            part = 0
        todo = score_ids(set_name, cond, block, n_blocks, part, split)
        if not todo:
            return {"block": block_name(block, n_blocks), "part": part, "n_parts": n_parts, "empty": True}
        threads = _score_threads()
        if threads < SCORE_CPUS:
            raise PermissionError(f"a score job runs {SCORE_CPUS} threads or more (decision E2: the score times of "
                                  f"the cost estimate and of SCORE_EVENTS_PER_JOB), not {threads}")
    k, root = check_request(set_name, cond, out, todo, n_blocks, split, runs, repo, root)
    kind = kind_of(set_name)
    grant = _grant(todo, kind, cond, out)
    bname = block_name(block, n_blocks)
    name = part_name(block, n_blocks, part or 0, n_parts)
    done = os.path.join(out, "chain", name + ".done")
    if os.path.exists(done):
        return {"block": bname, "part": name, "done": True, "skipped": True}
    if device.startswith("cuda") and {"additions", "store"} & set(stages):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device: additions and store run on the GPU (FRNN), as for the native events")
    keep_store = keep_store or k == 9
    os.makedirs(os.path.join(out, "chain"), exist_ok=True)
    base = {"set": set_name, "k": k, "condition": cond, "block": bname, "ids": [int(e) for e in todo], "kind": kind,
            "part": name, "n_parts": n_parts, "block_ids": [int(e) for e in ids],
            "out": os.path.realpath(out), "frozen_inputs": fi, "git": B._git_head(), "code": _code(), "node": _node(),
            "forward": FORWARD, "device_gpu_stages": device}
    recs = {}
    for stage in stages:
        _grant(todo, kind, cond, out)                            # every stage checks the grant again
        m = Meter()
        t0 = _now()
        if stage == "additions":
            stage_additions(ids, cond, out, fi, device, root, m)
        elif stage == "store":
            stage_store(ids, cond, out, bname, fi, device, root, m)
        else:
            scores = os.path.join(out, "scores")
            scored = set(scored_events(scores, strict=False)) if os.path.isdir(scores) else set()
            if not {int(e) for e in todo} <= scored:          # a part whose scores file was written is not scored again
                stage_score(todo, cond, out, bname, fi, m, block_ids=ids)
            bad = check_score_sidecars(scores, todo, cond, fi)
            if bad:
                raise RuntimeError(f"block {name}: {bad[:5]}")
        recs[stage] = {**base, "stage": stage, "grant": grant, "started_utc": t0, "ended_utc": _now(), **m.summary(),
                       "node": _node()}
        _write_json(os.path.join(out, "chain", f"{stage}_{name}.json"), recs[stage])
    dropped = False
    if "score" in stages:
        if not keep_store:
            dropped = drop_store(out, bname, ids)
        _write_json(done, {**base, "stages": stages, "store_kept": keep_store, "store_removed_here": dropped,
                           "grant": grant, "written_utc": _now()})
        if n_parts > 1 and {int(e) for e in ids} <= set(scored_events(os.path.join(out, "scores"), strict=False)):
            _write_json(os.path.join(out, "chain", bname + ".done"),
                        {**base, "part": None, "ids": [int(e) for e in ids], "stages": stages,
                         "written_utc": _now(), "parts": n_parts})
    return {"block": bname, "part": name, "done": "score" in stages, "stages": stages,
            "seconds": {s: r["seconds_total"] for s, r in recs.items()}}


# ------------------------------------------------------------------------------------------------ pilot: seal, checks
def blocks_complete(out: str, set_name: str, n_blocks: int = 1, split: dict | None = None) -> bool:
    return all(os.path.exists(os.path.join(out, "chain", block_name(b, n_blocks) + ".done")) for b in range(n_blocks)) \
        and set(scored_events(os.path.join(out, "scores"))) == set(B.set_ids(set_name, split))


def _parquet_counts(path: str, columns) -> tuple[list[str], dict[int, int]]:
    """Schema problems against the expected (name, type) columns and the row count per event (the event column only
    is read)."""
    import pyarrow.parquet as pq
    sch = pq.read_schema(path)
    bad = []
    names = [f.name for f in sch]
    if names != [c for c, _ in columns]:
        bad.append(f"{os.path.basename(path)}: columns {names}, expected {[c for c, _ in columns]}")
    else:
        for f, (c, t) in zip(sch, columns):
            if str(f.type) != t:
                bad.append(f"{os.path.basename(path)}: {c} is {f.type}, expected {t}")
    ev = pq.read_table(path, columns=["event"]).column("event").to_numpy()
    u, n = np.unique(ev, return_counts=True)
    return bad, {int(e): int(c) for e, c in zip(u, n)}


def expected_rows(ev, cond: str) -> np.ndarray:
    """Rows of the full event that a condition keeps (truth only)."""
    from hnl.overlay import conditions as OC
    from hnl.trackml import events as E
    if cond == "full":
        return np.arange(ev.n)
    if cond == "1gev":
        return np.flatnonzero(E.condition_mask(ev, "1gev"))
    return np.flatnonzero(OC.hs_mask(ev))


def check_outputs(set_name: str, cond: str, out: str, fi: dict | None = None, split: dict | None = None,
                  root: str | None = None, ids=None) -> dict:
    """Structure of the model-stage outputs of a set in one condition (decision D6): files, schemas, row counts per
    event, sidecar provenance, and the hits each condition keeps against the truth tables. Reads no score value and no
    edge label."""
    from hnl.overlay import conditions as OC
    from hnl.trackml import classifier as CL
    from hnl.trackml import events as E
    split = split or B.load_split()
    fi = fi or frozen_inputs()
    k = B.set_index(set_name, split)
    root = B.overlay_root(k, root)
    ids = [int(e) for e in (ids if ids is not None else B.set_ids(set_name, split))]
    bad: list[str] = []
    # additions
    add_rows: dict[int, int] = {}
    prov = {"additions": [], "scores": []}
    for p in sorted(glob.glob(os.path.join(out, "additions", "*.parquet"))):
        with open(p + ".json") as fh:
            s = json.load(fh)
        b, counts = _parquet_counts(p, ADDITION_COLUMNS)
        bad += b
        for e, c in s["events"].items():
            if counts.get(int(e), 0) != int(c):
                bad.append(f"{os.path.basename(p)}: event {e} has {counts.get(int(e), 0)} rows, sidecar {c}")
            add_rows[int(e)] = int(c)
        prov["additions"].append({"file": os.path.basename(p), "condition": s.get("condition"),
                                  "backend": s.get("backend"), "support_backend": s.get("support_backend"),
                                  "graph_choice": s.get("graph_choice")})
        if s.get("condition") != cond:
            bad.append(f"{os.path.basename(p)}: condition {s.get('condition')}, not {cond}")
        if s.get("graph_choice") != fi["graph_choice"]:
            bad.append(f"{os.path.basename(p)}: graph choice {s.get('graph_choice')} is not the frozen one")
        if s.get("backend") != "frnn_cuda" or s.get("support_backend") != "torch":
            bad.append(f"{os.path.basename(p)}: backends {s.get('backend')} / {s.get('support_backend')}, not the "
                       "native val / test additions' frnn_cuda / torch")
    # scores
    score_rows: dict[int, int] = {}
    for p in sorted(glob.glob(os.path.join(out, "scores", "*.parquet"))):
        with open(p + ".json") as fh:
            s = json.load(fh)
        b, counts = _parquet_counts(p, SCORE_COLUMNS)
        bad += b
        for e, c in s["events"].items():
            if counts.get(int(e), 0) != int(c):
                bad.append(f"{os.path.basename(p)}: event {e} has {counts.get(int(e), 0)} rows, sidecar {c}")
            score_rows[int(e)] = int(c)
        prov["scores"].append({"file": os.path.basename(p), "ckpt": s.get("ckpt"), "ckpt_sha256": s.get("ckpt_sha256"),
                               "ckpt_sha256_is_selection": s.get("ckpt_sha256") == fi["classifier_sha256"],
                               "condition": s.get("condition"), "forward": s.get("forward"), "device": s.get("device"),
                               "n_sectors": (s.get("layout") or {}).get("n_sectors"), "slurm_job": s.get("slurm_job"),
                               "host": s.get("host")})
    bad += check_score_sidecars(os.path.join(out, "scores"), ids, cond, fi)
    # store and the hits of the condition
    store = stored_events(os.path.join(out, "store"))
    truth = E.load_events(ids, root=root, particles=True)
    per = {}
    for e in ids:
        row = {"additions_rows": add_rows.get(e), "score_rows": score_rows.get(e)}
        if e not in add_rows:
            bad.append(f"event {e}: no additions")
        if e not in store:
            bad.append(f"event {e}: not in the store")
            per[str(e)] = row
            continue
        g = CL.load_event_graph(store[e])
        n = len(g["r"])
        for key, t in STORE_SCHEMA.items():
            arr = g[key]
            if str(arr.dtype) != t:
                bad.append(f"event {e}: store {key} is {arr.dtype}, expected {t}")
        if not (len(g["phi"]) == len(g["z"]) == len(g["pid"]) == len(g["rows"]) == n):
            bad.append(f"event {e}: store hit arrays differ in length")
        if g["pairs"].ndim != 2 or g["pairs"].shape[0] != 2 or g["y"].shape != (g["pairs"].shape[1],):
            bad.append(f"event {e}: store edge arrays have shapes {g['pairs'].shape}, {g['y'].shape}")
        elif g["pairs"].size and (g["pairs"].min() < 0 or g["pairs"].max() >= n):
            bad.append(f"event {e}: store edges index beyond its {n} hits")
        n_edges = int(g["pairs"].shape[1])
        if score_rows.get(e) is not None and score_rows[e] != n_edges:
            bad.append(f"event {e}: {score_rows[e]} score rows, {n_edges} store edges")
        ev = truth[e]
        want = expected_rows(ev, cond)
        pid = np.asarray(g["pid"], np.int64)
        v = OC.vertex_field(pid)
        filt = {"rows_equal_condition": bool(np.array_equal(g["rows"], want)),
                "pid_equal_truth": bool(len(want) == n and np.array_equal(pid, ev.pid[want])),
                "noise_hits": int((pid == 0).sum()), "full_event_hits": int(ev.n)}
        if cond == "hs":
            filt["vertex_fields"] = sorted(int(x) for x in np.unique(v))
            filt["hits_outside_vertex_1_or_1001_1010"] = int((~OC.hs_particle(pid)).sum())
            ok = filt["hits_outside_vertex_1_or_1001_1010"] == 0
        elif cond == "1gev":
            p = ev.particles
            fast = p.particle_id.to_numpy()[p.pt.to_numpy() > 1.0]
            filt["hits_of_particles_at_or_below_1gev_or_noise"] = int((~np.isin(pid, fast) | (pid == 0)).sum())
            ok = filt["hits_of_particles_at_or_below_1gev_or_noise"] == 0
        else:
            ok = n == ev.n
        if not (ok and filt["rows_equal_condition"] and filt["pid_equal_truth"]):
            bad.append(f"event {e}: the store's hits are not those of condition {cond}: {filt}")
        row.update({"hits": n, "edges": n_edges, "filter": filt, "store_bytes": os.path.getsize(store[e] if store[e]
                    .endswith(".gz") else store[e] + ".gz")})
        per[str(e)] = row
    return {"set": set_name, "condition": cond, "out": os.path.realpath(out), "events": per, "provenance": prov,
            "selection_classifier_sha256": fi["classifier_sha256"], "problems": bad, "pass": not bad,
            "read": "files, schemas, row counts, sidecars, store rows and particle ids; no score value, no edge label"}


def finish(set_name: str, cond: str, out: str, check_json: str, split: dict | None = None,
           root: str | None = None) -> dict:
    """Pilot plumbing run (decision D6): once every block is done, seal OUT (hnl.overlay.blind.seal: manifest,
    read-only) right after the writes, verify it, and write the structure checks to check_json (outside OUT)."""
    split = split or B.load_split()
    k = B.set_index(set_name, split)
    if k != 9:
        raise PermissionError("finish seals and checks the pilot plumbing outputs only")
    if B._under(check_json, out):
        raise PermissionError("the check record is written outside the sealed directory")
    ids = B.set_ids(set_name, split)
    _grant(ids, "plumb", cond, out)                       # the seal writes into OUT; the check record lies outside it
    if not blocks_complete(out, set_name, 1, split):
        raise SystemExit(f"{out}: not every event of {set_name} is scored yet")
    seal = B.seal(out, required=(), rule=B.SEAL_RULES["plumb"]) if not B.is_sealed(out) else \
        json.load(open(os.path.join(out, B.SEALED_JSON)))
    problems = B.verify(out, seal["manifest_sha256"])
    if problems:
        raise RuntimeError(f"{out} does not verify after sealing: {problems[:5]}")
    res = check_outputs(set_name, cond, out, split=split, root=root, ids=ids)
    res.update({"sealed": {"manifest_sha256": seal["manifest_sha256"], "n_files": seal["n_files"],
                           "bytes": seal["bytes"], "sealed_utc": seal["sealed_utc"], "verify": "ok"},
                "written_utc": _now(), "git": B._git_head(), "code": _code(),
                "slurm_job": os.environ.get("SLURM_JOB_ID")})
    _write_json(check_json, res)
    return res


# ------------------------------------------------------------------------------------------------ placeholders (dry)
def dry(set_name: str, cond: str, out: str, root: str | None = None, split: dict | None = None) -> dict:
    """Truth-made placeholder scores of the pilot set (hnl.overlay.blind.placeholder_scores) into OUT/scores, the
    layout of the model stages (kind dry; code tests of the grid jobs and slurm/overlay/h1.sbatch). No model."""
    split = split or B.load_split()
    if B.set_index(set_name, split) != 9:
        raise PermissionError("placeholder scores are made for the pilot set (k = 9) only")
    if B._under(out, B.RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    _grant(B.set_ids(set_name, split), "dry", cond, os.path.join(out, "scores"))
    return B.placeholder_scores(set_name, os.path.join(out, "scores"), cond, split=split, root=root)


# ------------------------------------------------------------------------------------------------ plumbing record
def _stage_events(rec: dict) -> dict[int, dict]:
    return {int(r["event"]): r for r in rec["per_event"]}


def grid_timing(selection: str = B.SELECTION) -> dict:
    """Seconds per event of the frozen configurations' track building in the native TrackML test grids (timing.parquet
    of the test chain: the seconds column only), for the grid estimate."""
    import pandas as pd
    with open(selection) as fh:
        run = json.load(fh)["run"]
    out = {}
    for cond, d in (("full", "test/test"), ("1gev", "test/test_1gev")):
        p = os.path.join(run, d, "timing.parquet")
        if os.path.exists(p):
            t = pd.read_parquet(p, columns=["event", "seconds"])
            s = t.groupby("event").seconds.sum()
            out[cond] = {"mean_s": round(float(s.mean()), 1), "max_s": round(float(s.max()), 1), "events": int(len(s)),
                         "source": p}
    return out


def _jobs(n: int, per_event_s: float, overhead_s: float, hour_limit_s: float = HOUR_S) -> tuple[int, int, float]:
    """(events per job, jobs, hours) of n events at per_event_s each, jobs of at most 75 % of the time limit and at most
    one block (MAX_BLOCK events), plus the per-job overhead once per job."""
    per_job = max(1, min(MAX_BLOCK, int(0.75 * hour_limit_s // max(per_event_s, 1e-9))))
    jobs = math.ceil(n / per_job)
    return per_job, jobs, (n * per_event_s + jobs * overhead_s) / HOUR_S


def score_parts_n(n: int, gpu_per_job: int, cond: str) -> int:
    """`score_parts` for n events in GPU blocks of at most gpu_per_job events (no split file needed)."""
    per = -(-n // math.ceil(n / gpu_per_job))
    return -(-per // SCORE_EVENTS_PER_JOB[cond])


def _score_layout(n: int, gpu_per_job: int, cond: str) -> tuple[int, int]:
    """(score jobs, events in the largest score job) of n events in GPU blocks of at most gpu_per_job events, each
    block scored by `score_parts` jobs of at most SCORE_EVENTS_PER_JOB[cond] events (the slicing of `block_ids` and
    `score_ids`, which `run` and chain_score.sbatch use)."""
    nb = math.ceil(n / gpu_per_job)
    per = -(-n // nb)
    s = score_parts_n(n, gpu_per_job, cond)
    jobs, largest = 0, 0
    for b in range(nb):
        size = len(range(n)[b * per:(b + 1) * per])
        q = -(-size // s)
        parts = [min(q, size - j * q) for j in range(s) if size - j * q > 0]
        jobs += len(parts)
        largest = max([largest, *parts])
    return jobs, largest


def extrapolate(per_cond: dict, plan: dict = PLAN, grid: dict | None = None, hour_limit_s: float = HOUR_S) -> dict:
    """Cost of the model stages of H-val, closure and H-test from the pilot's per-event times (pilot events are tp_v1
    val hosts; H-test hosts come from training_part03, of the same size), under both score placements (decision E2
    chose scores_on_cpu_nodes, `PLACEMENT`; both are given):
      scores_on_cpu_nodes  the layout the code runs: GPU jobs (chain.sbatch STAGES=additions,store) of blocks of at most
                           MAX_BLOCK events and at most 75 % of the time limit; each block scored by `score_parts` CPU
                           jobs (chain_score.sbatch) of SCORE_CPUS threads and at most SCORE_EVENTS_PER_JOB[c] events,
                           at the score time measured on a CPU node (else the GPU job's, flagged); core-hours at
                           SCORE_CPUS per job. The longest score job (its events at the mean time, plus the overhead)
                           is given against the limit.
      scores_in_gpu_job    every stage in the GPU job (chain.sbatch), at the score time measured inside the pilot's GPU
                           job; its GPU hours include the scores
    per_cond[c]: gpu_s_per_event (additions + store), gpu_overhead_s_per_job, score_s_per_event_gpu_job,
    score_overhead_s_per_job_gpu_job, score_cpus_gpu_job, and, when measured, score_s_per_event_cpu_node,
    score_overhead_s_per_job_cpu_node, score_cpus_cpu_node."""
    rows = []
    tot = {"scores_on_cpu_nodes": {"gpu_hours": 0.0, "score_wall_hours": 0.0, "score_cpu_core_hours": 0.0},
           "scores_in_gpu_job": {"gpu_hours": 0.0}, "grid_cpu_hours": 0.0}
    longest = 0.0
    for s, (n, conds) in plan.items():
        for c in conds:
            p = per_cond.get(c)
            if not p:
                continue
            g, go = p["gpu_s_per_event"], p["gpu_overhead_s_per_job"]
            measured = p.get("score_s_per_event_cpu_node") is not None
            source = (p.get("score_cpu_node_source") or {}).get("what") if measured else \
                "not measured on a CPU node: the GPU job's score time stands in"
            sc_cpu = p["score_s_per_event_cpu_node"] if measured else p["score_s_per_event_gpu_job"]
            so_cpu = p["score_overhead_s_per_job_cpu_node"] if measured else p["score_overhead_s_per_job_gpu_job"]
            cpus_t = p["score_cpus_cpu_node"] if measured else p["score_cpus_gpu_job"]
            gj, gn, gh = _jobs(n, g, go, hour_limit_s)
            cn, cl = _score_layout(n, gj, c)
            cw = (n * sc_cpu + cn * so_cpu) / HOUR_S
            job_s = cl * sc_cpu + so_cpu
            longest = max(longest, job_s / hour_limit_s)
            aj, an, ah = _jobs(n, g + p["score_s_per_event_gpu_job"], go + p["score_overhead_s_per_job_gpu_job"],
                               hour_limit_s)
            row = {"set": s, "condition": c, "events": n,
                   "scores_on_cpu_nodes": {"gpu_events_per_job": gj, "gpu_jobs": gn, "gpu_hours": round(gh, 2),
                                           "score_parts_per_block": score_parts_n(n, gj, c),
                                           "score_events_per_job_max": cl, "score_jobs": cn,
                                           "score_wall_hours": round(cw, 2),
                                           "score_cpus_per_job": SCORE_CPUS,
                                           "score_cpu_core_hours": round(cw * SCORE_CPUS, 1),
                                           "score_s_per_event": sc_cpu, "score_time_threads": cpus_t,
                                           "score_time_at_job_threads": measured and cpus_t == SCORE_CPUS,
                                           "longest_score_job_s": round(job_s, 1),
                                           "longest_score_job_share_of_limit": round(job_s / hour_limit_s, 3),
                                           "score_time_measured_on_cpu_node": measured,
                                           "score_time_source": source},
                   "scores_in_gpu_job": {"events_per_job": aj, "jobs": an, "gpu_hours": round(ah, 2)}}
            if grid and c in grid:
                row["grid_cpu_hours"] = round(n * grid[c]["mean_s"] / HOUR_S, 1)
                tot["grid_cpu_hours"] += row["grid_cpu_hours"]
            rows.append(row)
            for pl, keys in (("scores_on_cpu_nodes", ("gpu_hours", "score_wall_hours", "score_cpu_core_hours")),
                             ("scores_in_gpu_job", ("gpu_hours",))):
                for key in keys:
                    tot[pl][key] += row[pl][key]
    total = {k: ({kk: round(vv, 1) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 1))
             for k, v in tot.items()}
    total["scores_on_cpu_nodes"]["longest_score_job_share_of_limit"] = round(longest, 3)
    return {"rows": rows, "total": total, "placement_chosen": PLACEMENT,
            "layout": {"max_block": MAX_BLOCK, "score_cpus": SCORE_CPUS, "score_events_per_job": SCORE_EVENTS_PER_JOB,
                       "hour_limit_s": hour_limit_s},
            "placement_decision": "decision E2 (2026-10-15): scores on the CPU nodes, under the 1.5x rule; the "
                                  "scores-in-GPU-job placement is not used"}


def cost(record: str, out_json: str | None = None) -> dict:
    """The cost of the production model stages recomputed by `extrapolate` (the layout this code runs) from the cost
    inputs and the native grid times of a plumbing record (plumb_check_v*.json). Reads that record's JSON only."""
    with open(record) as fh:
        r = json.load(fh)
    res = {"title": "Cost of the model stages of H-val, the closure set and H-test under decision E2, in the job "
                    "layout of hnl.overlay.chain (GPU blocks of at most 50 events, each scored by CPU jobs of at most "
                    "SCORE_EVENTS_PER_JOB events at SCORE_CPUS threads)",
           "source_record": os.path.realpath(record), "source_sha256": B.sha256_file(record),
           "source_git": r.get("git"), "cost_inputs": r["cost_inputs"],
           "extrapolation": extrapolate(r["cost_inputs"], grid=r.get("grid_timing_native_test")),
           "plan": {k: [v[0], list(v[1])] for k, v in PLAN.items()},
           "budget": {"gpu_hours_estimate": 10.4, "rule": "a stage above 1.5 times its estimate, or 75 GPU-h, goes to "
                                                           "the user first (addendum section 10)"},
           "written_utc": _now(), "git": B._git_head(), "code": _code(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    gh = res["extrapolation"]["total"]["scores_on_cpu_nodes"]["gpu_hours"]
    res["budget"]["ratio"] = round(gh / 10.4, 2)
    if out_json:
        _write_json(out_json, res)
    return res


def _records(out: str) -> dict:
    """{stage: record} of the stage records of one block directory (OUT/chain/<stage>_<block>.json)."""
    recs = {}
    for st in STAGES:
        f = sorted(glob.glob(os.path.join(out, "chain", f"{st}_*.json")))
        if f:
            with open(f[0]) as fh:
                recs[st] = json.load(fh)
    return recs


def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _sacct(job) -> dict | None:
    """Slurm accounting of a finished job (best effort; None without sacct): state, node, start, end, elapsed seconds
    and the batch step's peak resident memory."""
    import subprocess
    if not job:
        return None
    try:
        p = subprocess.run(["sacct", "-j", str(job), "-n", "-P", "-o", "JobID,State,NodeList,Start,End,Elapsed,MaxRSS"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = {r.split("|")[0]: r.split("|") for r in p.stdout.splitlines() if r.count("|") == 6}
    if str(job) not in rows:
        return None

    def secs(t):
        d, _, hms = t.rpartition("-")
        h, m, s_ = (int(x) for x in hms.split(":"))
        return (int(d) if d else 0) * 86400 + h * 3600 + m * 60 + s_

    def gb(v):
        m = re.match(r"^([\d.]+)([KMG]?)$", v or "")
        return round(float(m.group(1)) * {"K": 1e3, "M": 1e6, "G": 1e9, "": 1.0}[m.group(2)] / 1e9, 3) if m else None

    _, state, node, start, end, el, _ = rows[str(job)]
    batch = rows.get(f"{job}.batch")
    return {"job": str(job), "state": state, "node": node, "start": start, "end": end, "elapsed_s": secs(el),
            "batch_max_rss_gb": gb(batch[6]) if batch else None}


def _seal_state(d: str, chk: dict | None) -> dict:
    """The seal of a plumbing directory verified again (every file against its manifest, the manifest against the
    sha256 the check record holds) and its permissions (no writable file or directory)."""
    want = ((chk or {}).get("sealed") or {}).get("manifest_sha256")
    bad = B.verify(d, want) if want else ["no manifest sha256 in the check record"]
    writable = [os.path.relpath(os.path.join(r, f), d) for r, ds, fs in os.walk(d) for f in fs + ds
                if os.stat(os.path.join(r, f)).st_mode & 0o222]
    return {"sealed": B.is_sealed(d), "verified_again": not bad, "problems": bad[:5], "writable_paths": writable[:5],
            "manifest_sha256": want, "checked_utc": _now()}


def cpu_timing_from_check(path: str) -> dict:
    """{condition: per-event seconds, threads, host, events, jobs} of the whole-event chunked forward (hnl.trackml.infer
    score_event, chunked) on native val events, from a chunked-vs-plain check record (seconds_chunked per event): the
    score time on a CPU node when no pilot CPU score job has run."""
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for c, v in d.get("conditions", {}).items():
        rows = v.get("events") or []
        if not rows:
            continue
        out[c] = {"seconds_per_event": round(float(np.mean([r["seconds_chunked"] for r in rows])), 2),
                  "threads": sorted({r.get("threads") for r in rows}), "hosts": sorted({r.get("host") for r in rows}),
                  "events": [r["event"] for r in rows], "jobs": sorted({r.get("slurm_job") for r in rows} - {None}),
                  "source": os.path.realpath(path)}
    return out


def _pick_cpu_timing(cands: list) -> tuple[dict | None, dict | None]:
    """(the CPU-node score timing the estimate uses, the other one): the first measured at SCORE_CPUS threads (the
    threads of chain_score.sbatch), else the first given (the pilot CPU score job comes before the check record)."""
    if not cands:
        return None, None
    order = sorted(cands, key=lambda x: x["score_cpus_cpu_node"] != SCORE_CPUS)       # stable
    return order[0], (order[1] if len(order) > 1 else None)


def plumb_record(directory: str, out_json: str, set_name: str = "pilot", selection: str = B.SELECTION,
                 cpu_timing: str | None = None) -> dict:
    """plumb_check_v1.json: per condition the checks, the sealed manifests, wall time and peak memory per event and
    stage (GPU job: additions, store and score; CPU job, when run: the score stage again on a copy of the GPU job's
    additions and store, DIR/<set>_<cond>_cpuscore), and the cost extrapolation under both score placements (decision
    E2). Reads only the JSON records, the structure checks and the native test grid's seconds; no model output value
    of any kind."""
    per_cond, conds = {}, {}
    for c in CONDITIONS:
        out = os.path.join(directory, f"{set_name}_{c}")
        chk = _load(os.path.join(directory, "checks", f"check_{c}.json"))
        if chk is None:
            conds[c] = {"missing": os.path.join(directory, "checks", f"check_{c}.json")}
            continue
        recs = _records(out)
        cpu_out = os.path.join(directory, f"{set_name}_{c}_cpuscore")
        cpu_chk = _load(os.path.join(directory, "checks", f"check_{c}_cpuscore.json"))
        cpu_rec = _records(cpu_out).get("score") if cpu_chk is not None else None
        ev = {}
        for e, row in chk["events"].items():
            r = {k: row.get(k) for k in ("hits", "edges", "additions_rows", "score_rows", "store_bytes")}
            for st, rec in recs.items():
                m = _stage_events(rec).get(int(e), {})
                r[f"{st}_s"] = m.get("seconds")
                r[f"{st}_peak_rss_gb"] = m.get("peak_rss_gb")
                r[f"{st}_peak_gpu_gb"] = m.get("peak_gpu_gb")
            if cpu_rec is not None:
                m = _stage_events(cpu_rec).get(int(e), {})
                r["score_cpu_node_s"] = m.get("seconds")
                r["score_cpu_node_peak_rss_gb"] = m.get("peak_rss_gb")
            ev[e] = r

        def mean(key):
            v = [r[key] for r in ev.values() if r.get(key) is not None]
            return round(float(np.mean(v)), 2) if v else None

        def peak(key):
            v = [r[key] for r in ev.values() if r.get(key) is not None]
            return max(v) if v else None

        over = {st: round(rec["seconds_total"] - rec["seconds_events"], 2) for st, rec in recs.items()}
        summ = {"events": len(ev), "mean_hits": mean("hits"), "mean_edges": mean("edges"),
                "seconds_per_event": {st: mean(f"{st}_s") for st in recs},
                "peak_rss_gb": {st: peak(f"{st}_peak_rss_gb") for st in recs},
                "peak_gpu_gb": {st: peak(f"{st}_peak_gpu_gb") for st in recs},
                "stage_overhead_s": over, "stage_total_s": {st: rec["seconds_total"] for st, rec in recs.items()},
                "node": {st: rec["node"] for st, rec in recs.items()},
                "rss_reset_per_event": {st: rec.get("rss_reset_per_event") for st, rec in recs.items()}}
        job = _sacct((recs.get("score") or {}).get("node", {}).get("slurm_job"))
        summ["job"] = job
        if job and recs:
            summ["job_outside_stages_s"] = round(job["elapsed_s"] - sum(r["seconds_total"] for r in recs.values()), 1)
        conds[c] = {"checks_pass": chk["pass"], "problems": chk["problems"], "sealed": chk.get("sealed"),
                    "seal_now": _seal_state(out, chk), "provenance": chk["provenance"], "per_event": ev,
                    "summary": summ,
                    "grant": (recs.get("score") or recs.get("additions") or {}).get("grant")}
        if cpu_chk is not None:
            conds[c]["cpu_score_job"] = {
                "directory": os.path.realpath(cpu_out), "checks_pass": cpu_chk["pass"],
                "problems": cpu_chk["problems"], "sealed": cpu_chk.get("sealed"),
                "seal_now": _seal_state(cpu_out, cpu_chk), "provenance_scores": cpu_chk["provenance"]["scores"],
                "seconds_per_event": mean("score_cpu_node_s"), "peak_rss_gb": peak("score_cpu_node_peak_rss_gb"),
                "stage_overhead_s": round(cpu_rec["seconds_total"] - cpu_rec["seconds_events"], 2) if cpu_rec else None,
                "stage_total_s": cpu_rec["seconds_total"] if cpu_rec else None,
                "node": cpu_rec["node"] if cpu_rec else None, "grant": cpu_rec["grant"] if cpu_rec else None,
                "job": _sacct(cpu_rec["node"].get("slurm_job")) if cpu_rec else None}
        if all(st in recs for st in STAGES):
            outside = summ.get("job_outside_stages_s") or 0.0      # freeze verify, guard, start-up, pilot seal
            pc = {"gpu_s_per_event": round((mean("additions_s") or 0) + (mean("store_s") or 0), 2),
                  "gpu_overhead_s_per_job": round(over["additions"] + over["store"] + outside, 2),
                  "score_s_per_event_gpu_job": mean("score_s") or 0,
                  "score_overhead_s_per_job_gpu_job": over["score"],
                  "score_cpus_gpu_job": recs["score"]["node"]["cpus"],
                  "gpu_job_node": recs["score"]["node"].get("host"), "gpu": recs["additions"]["node"].get("gpu"),
                  "score_s_per_event_cpu_node": None}
            # CPU-node score times: the pilot CPU score job, and the chunked-vs-plain check (--cpu-timing); the
            # estimate takes the one measured at SCORE_CPUS threads, the threads of chain_score.sbatch (the pilot job
            # first), else the pilot job, else the check; the other is kept beside it
            cands = []
            if cpu_rec is not None and cpu_chk["pass"]:
                cj = conds[c]["cpu_score_job"]["job"]
                cpu_outside = cj["elapsed_s"] - cpu_rec["seconds_total"] if cj else 0.0
                conds[c]["cpu_score_job"]["job_outside_stage_s"] = round(cpu_outside, 1) if cj else None
                cands.append({"score_s_per_event_cpu_node": mean("score_cpu_node_s"),
                              "score_overhead_s_per_job_cpu_node": round(cpu_rec["seconds_total"]
                                                                          - cpu_rec["seconds_events"] + cpu_outside, 2),
                              "score_cpus_cpu_node": cpu_rec["node"]["cpus"], "cpu_node": cpu_rec["node"].get("host"),
                              "score_cpu_node_source": {"what": "pilot CPU score job (score stage again on a copy of "
                                                                "the GPU job's store)"}})
            ct = cpu_timing_from_check(cpu_timing).get(c) if cpu_timing else None
            if ct and len(ct["threads"]) == 1:
                cands.append({"score_s_per_event_cpu_node": ct["seconds_per_event"],
                              "score_overhead_s_per_job_cpu_node": round(over["score"] + outside, 2),
                              "score_cpus_cpu_node": ct["threads"][0], "cpu_node": ",".join(ct["hosts"]),
                              "score_cpu_node_source": {
                                  "what": "whole-event chunked forward on the native val events of the pilot hosts "
                                          "(no gun hits), chunked-vs-plain check jobs; per-job overhead as the GPU "
                                          "job's", **ct}})
            use, other = _pick_cpu_timing(cands)
            if use:
                pc.update(use)
            if other:
                pc["score_cpu_node_other"] = other
            per_cond[c] = pc
    gt = grid_timing(selection)
    res = {"title": "Pilot plumbing run of the frozen model stages on the five pilot overlay events (decision D6): "
                    "structure, provenance, timing and memory only",
           "rule": "No efficiency, fake rate, matched count, V2 value, TrackML score or per-target number was "
                   "computed, printed, logged or written from these outputs; no track building and no "
                   "hnl.overlay.blind evaluation ran on them. Outputs sealed right after writing.",
           "set": set_name, "directory": os.path.realpath(directory), "conditions": conds,
           "cost_inputs": per_cond, "grid_timing_native_test": gt,
           "extrapolation": extrapolate(per_cond, grid=gt),
           "extrapolation_notes": [
               "Two score placements; decision E2 chose scores_on_cpu_nodes. scores_on_cpu_nodes: GPU jobs run the "
               "union additions and the event store only, CPU jobs (slurm/overlay/chain_score.sbatch, "
               "submit-gpu-express without the A30 and GTX 1080 nodes: submit77-1, and submit37 when it returns from "
               "drain) run the "
               "whole-event CPU forward (hnl.trackml.chunked) at the time measured in the pilot's CPU score job. "
               "scores_in_gpu_job: every stage in the GPU job (slurm/overlay/chain.sbatch) at the score time measured "
               "inside the pilot's GPU job, so the GPU is held while the CPUs score",
               "per-job overhead from the pilot jobs: the stage totals minus the per-event intervals (model "
               "loading, reads, the scores file) plus the job's time outside the stages (sacct elapsed: freeze "
               "verify, guard, start-up, and the pilot's seal and checks, so slightly high), once per job; GPU jobs of "
               "at most 75 % of the 1 h express limit and at most one block (50 events); score jobs of at most "
               "SCORE_EVENTS_PER_JOB events of one block at SCORE_CPUS threads (core-hours at SCORE_CPUS)",
               "the addendum's budget (section 10): 10.4 GPU-h for one H-T0.5 pass and H-T2; a stage above 1.5 times "
               "its estimate, or 75 GPU-h, goes to the user first",
               "grid_cpu_hours: track building at WP-A, WP-S and WP-R-P3 (plus WP-R-P2 in the native test grid) "
               "from the native test grid's per-event seconds, one worker per event; C-hs grids are small and not "
               "estimated"],
           "written_utc": _now(), "git": B._git_head(), "code": _code(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    _write_json(out_json, res)
    return res


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    p = sub.add_parser("paths")
    p.add_argument("--set", required=True)
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p.add_argument("--key", default=None, choices=["out", "scores", "grid"])
    p = sub.add_parser("block-ids")
    p.add_argument("--set", required=True)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=1)
    p = sub.add_parser("score-parts")
    p.add_argument("--set", required=True)
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p.add_argument("--n-blocks", type=int, required=True)
    p = sub.add_parser("score-ids")
    p.add_argument("--set", required=True)
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--n-blocks", type=int, required=True)
    p.add_argument("--part", type=int, required=True)
    p = sub.add_parser("layout")
    p.add_argument("--set", required=True)
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p = sub.add_parser("run")
    p.add_argument("--set", required=True, choices=sorted(SETS))
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=1)
    p.add_argument("--out", default=None, help="pilot: the scratch directory (required); production sets: the default "
                   "(`paths`) only")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--keep-store", action="store_true")
    p.add_argument("--part", type=int, default=None, help="production score jobs: the part of the block (score-ids)")
    p = sub.add_parser("finish")
    p.add_argument("--set", default="pilot", choices=["pilot"])
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p.add_argument("--out", required=True)
    p.add_argument("--json", required=True)
    p = sub.add_parser("dry")
    p.add_argument("--set", default="pilot", choices=["pilot"])
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=None)
    p = sub.add_parser("plumb-record")
    p.add_argument("--dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--cpu-timing", default=None, help="a chunked-vs-plain check record: CPU-node score times on native "
                   "val events (used where they were measured at SCORE_CPUS threads, or where no pilot CPU score job "
                   "ran)")
    p = sub.add_parser("cost")
    p.add_argument("--record", required=True, help="a plumbing record (plumb_check_v*.json)")
    p.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        fi = preflight()
        print(f"chain preflight: every frozen input, the run and the classifier checkpoint verified "
              f"(selection sha256 {fi['selection_sha256'][:12]}...)")
    elif a.cmd == "paths":
        try:
            d = paths(a.set, a.cond)
        except (PermissionError, ValueError, KeyError) as e:
            raise SystemExit(f"chain paths: {e}")
        if a.key:
            print(d[a.key] or "")
        else:
            print("\n".join(f"{k.upper()}={v or ''}" for k, v in d.items()))
    elif a.cmd == "block-ids":
        try:
            print(",".join(str(e) for e in block_ids(a.set, a.block, a.n_blocks)))
        except (ValueError, KeyError) as e:
            raise SystemExit(f"chain block-ids: {e}")
    elif a.cmd in ("score-parts", "score-ids", "layout"):
        try:
            if a.cmd == "score-parts":
                print(score_parts(a.set, a.cond, a.n_blocks))
            elif a.cmd == "score-ids":
                print(",".join(str(e) for e in score_ids(a.set, a.cond, a.block, a.n_blocks, a.part)))
            else:
                print(json.dumps(layout(a.set, a.cond), indent=1))
        except (ValueError, KeyError, PermissionError) as e:
            raise SystemExit(f"chain {a.cmd}: {e}")
    elif a.cmd == "run":
        if a.set == "pilot" and not a.out:
            raise SystemExit("pilot runs need --out (a scratch directory)")
        print(json.dumps(run(a.set, a.cond, a.block, a.n_blocks, a.out, a.stages.split(","), a.keep_store,
                             part=a.part)))
    elif a.cmd == "cost":
        r = cost(a.record, a.out_json)
        print(json.dumps({"total": r["extrapolation"]["total"], "budget": r["budget"], "out": a.out_json}))
    elif a.cmd == "finish":
        r = finish(a.set, a.cond, a.out, a.json)
        print(json.dumps({"condition": r["condition"], "pass": r["pass"], "problems": r["problems"][:5],
                          "manifest_sha256": r["sealed"]["manifest_sha256"], "check": a.json}))
        if not r["pass"]:
            raise SystemExit(1)
    elif a.cmd == "dry":
        print(json.dumps(dry(a.set, a.cond, a.out, a.root)))
    else:
        r = plumb_record(a.dir, a.out_json, cpu_timing=a.cpu_timing)
        print(json.dumps({"conditions": {c: v.get("checks_pass") for c, v in r["conditions"].items()},
                          "extrapolation": r["extrapolation"]["total"], "out": a.out_json}))


if __name__ == "__main__":
    main()
