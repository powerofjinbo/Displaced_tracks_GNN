"""Preregistration addendum G, step T4 (docs/preregistration_new_sample.md section 9): the retrained models of T3,
acorn (ttbar+G) and Canto-L (ttbar+G), seeds 0-2 (docs/addendum_g_t3_plan.md), on the generic sets of T2, and claims
G2 and G3 (section 6).

Chain per seed (the T2 chain of hnl.eval.addendum_g with the retrained models; outputs under DATA/runs/addendum_g_t4_v1):
  pre    acorn (ttbar+G)    feature store -> metric learning (the run's checkpoint, r_infer, knn_infer of its val
                            inference) -> edge classifier (the checkpoint of its val scores) -> acorn_g_s<S>/<set>/testset.tar
         Canto-L (ttbar+G)  hnl.graphs.helix_union build --no-metrics on those graphs with the run's final learned
                            triplet support (k 5, tau 0.005) -> canto_g_s<S>/<set>/union/blocks/<k>_of_<n>/
  post   Canto-L (ttbar+G)  its edge classifier (checkpoint from the selection file) on the union graphs, in parts
                            -> canto_g_s<S>/<set>/scored/testset.tar (gather-scored, then the union tars are removed)
         evaluation         the selected configuration of each model and seed (results/addendum_g_selection_v1.json)
                            and, where the file has one, the guarded alternative: builder, double-majority matching,
                            graph limits, stored tracks (hnl.eval.addendum_g evaluate) -> eval/<run>/<set>/
         summary            claims G2 and G3 (test), the alternatives as sensitivity rows, the populations and bins of
                            addendum G section 7 -> results/addendum_g_t4_<set>_v1.{md,csv,json}
Stages "pre" produce no metric: no track is built and nothing is matched. Their outputs get a sha256 manifest when
written, and every post stage checks the manifest before it reads them. slurm/addendum_g_t4/guard.sh gates the test
set: pre stages need ledger entry R16 committed; post stages need R16 and the selection file committed (scheduling
deviation of R16: the pre stages may run on test before the selection file exists, since they pass no test
information to any selection).

Pre-stage settings come from the run directories (independent of the selection): ML checkpoint and graph radius from
the run's ml_graphs/infer_*.json, acorn's classifier from gnn_scores/infer_*.json, the triplet support from
ltf_union_graphs/report_val/build.json. The post stages check that the selection file names the same files (sha256).
Runs: acorn_g (acorn (ttbar+G)), canto_g (Canto-L (ttbar+G)); with the guarded alternative: acorn_g_alt, canto_g_alt.

    python -m hnl.eval.addendum_g_t4 acorn-args --seed S        (ML ckpt, r, knn, GNN ckpt; pre)
    python -m hnl.eval.addendum_g_t4 union-args --seed S        (graphs run, triplet model, tau, k; pre)
    python -m hnl.eval.addendum_g_t4 manifest --what acorn|union --seed S --set SET [--block k --n-blocks n]
    python -m hnl.eval.addendum_g_t4 canto-args --seed S        (Canto-L classifier checkpoint; post)
    python -m hnl.eval.addendum_g_t4 gather-union --seed S --set SET --dest DIR [--part k --n-parts n]
    python -m hnl.eval.addendum_g_t4 gather-scored --seed S --set SET [--prune-union]
    python -m hnl.eval.addendum_g_t4 evaluate --run acorn_g --seed S --set SET [--block k --n-blocks n]
    python -m hnl.eval.addendum_g_t4 merge --run acorn_g --seed S --set SET
    python -m hnl.eval.addendum_g_t4 runs                       (the evaluation runs of the selection file)
    python -m hnl.eval.addendum_g_t4 summary --set val|test [--seeds 0,1,2]
Code tests on val may pass --selection (another selection file) and --canto-ckpt (another classifier checkpoint).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import tarfile
import time

import numpy as np
import pandas as pd

from hnl.eval import addendum_g as T2

REPO = T2.REPO
DATA = T2.DATA
RUNS = T2.RUNS
OUT = os.path.join(RUNS, "addendum_g_t4_v1")
ACORN = os.path.join(RUNS, "baseline_g1")
CANTO_GRAPHS = os.path.join(RUNS, "baseline_g1_ts")
CANTO = os.path.join(RUNS, "stage1_g1", "l1tl")
SELECTION = os.path.join(REPO, "results", "addendum_g_selection_v1.json")
SEEDS = (0, 1, 2)
RUN_MODEL = {"acorn_g": "acorn", "canto_g": "canto", "acorn_g_alt": "acorn", "canto_g_alt": "canto"}
LABELS = {"acorn_g": "acorn (ttbar+G)", "canto_g": "Canto-L (ttbar+G)",
          "acorn_g_alt": "acorn (ttbar+G), guarded alternative", "canto_g_alt": "Canto-L (ttbar+G), guarded alternative"}
MARGIN, FR_GUARD, G3_Z = 1.0, 0.5, 3.0


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: str) -> str:
    return T2._sha256(path)


def _one(pattern: str) -> dict:
    fs = sorted(glob.glob(pattern))
    if len(fs) != 1:
        raise SystemExit(f"expected one file {pattern}, found {len(fs)}")
    return json.load(open(fs[0]))


# ------------------------------------------------------------------------------------------------- settings
def pre_settings(seed: int) -> dict:
    """The pre-stage settings of seed S, from the run directories (no selection involved)."""
    a = os.path.join(ACORN, f"armg_s{seed}")
    ml = _one(os.path.join(a, "ml_graphs", "infer_*.json"))
    gi = _one(os.path.join(a, "gnn_scores", "infer_*.json"))
    ub = json.load(open(os.path.join(CANTO_GRAPHS, f"armg_s{seed}", "ltf_union_graphs", "report_val", "build.json")))
    return {"acorn_run": a, "ml_ckpt": ml["ckpt"], "ml_sha256": ml["ckpt_sha256"],
            "r_infer": ml["overrides"]["r_infer"], "knn_infer": ml["overrides"]["knn_infer"],
            "gnn_ckpt": gi["ckpt"], "gnn_sha256": gi["ckpt_sha256"],
            "graphs_run": os.path.join(CANTO_GRAPHS, f"armg_s{seed}"), "triplet_model": ub["triplet_model"],
            "triplet_sha256": _sha256(ub["triplet_model"]), "tau": ub["tau"], "k": ub["k"], "r_wide": ub["r_wide"],
            "nsig": ub["nsig"], "pt_min": ub["pt_min"], "union_ml_ckpt": ub["ml_ckpt"]}


def selection(path: str | None = None) -> dict:
    return json.load(open(path or SELECTION))


def entry(run: str, seed: int, path: str | None = None) -> dict | None:
    """The selection-file entry of an evaluation run and seed (None: no guarded alternative for this model)."""
    sel = selection(path)
    m = sel["models"][RUN_MODEL[run]]
    if run.endswith("_alt"):
        alt = m.get("guarded_alternative")
        if not alt or not alt.get("family"):
            return None
        seeds = alt["seeds"]
    else:
        seeds = m["seeds"]
    e = [x for x in seeds if int(x["seed"]) == seed]
    if len(e) != 1:
        raise SystemExit(f"{run} seed {seed}: {len(e)} entries in the selection file")
    return e[0]


def runs(path: str | None = None) -> list[str]:
    return [r for r in RUN_MODEL if entry(r, 0, path) is not None]


def check_entry(run: str, seed: int, path: str | None = None) -> dict:
    """The entry, with its checkpoints checked against the pre-stage settings (same files, same sha256)."""
    e = entry(run, seed, path)
    p = pre_settings(seed)
    pairs = [("ml_ckpt", "ml_sha256", p["ml_sha256"])]
    if RUN_MODEL[run] == "acorn":
        pairs.append(("gnn_ckpt", "gnn_sha256", p["gnn_sha256"]))
    else:
        if os.path.realpath(e["triplet_model"]) != os.path.realpath(p["triplet_model"]):
            raise SystemExit(f"{run} s{seed}: triplet model {e['triplet_model']} != {p['triplet_model']}")
        pairs.append(("triplet_model", "triplet_model_sha256", p["triplet_sha256"]))
    for f, h, want in pairs:
        if e[h] != want or _sha256(e[f]) != want:
            raise SystemExit(f"{run} s{seed}: {f} of the selection file differs from the pre-stage file")
    if _sha256(e["merge_model"]) != e["merge_sha256"]:
        raise SystemExit(f"{run} s{seed}: merge model sha256 differs from the selection file")
    return e


def canto_ckpt(seed: int, path: str | None = None, override: str | None = None) -> str:
    if override:
        return override
    e = entry("canto_g", seed, path)
    if _sha256(e["gnn_ckpt"]) != e["gnn_sha256"]:
        raise SystemExit(f"Canto-L (ttbar+G) s{seed}: classifier sha256 differs from the selection file")
    return e["gnn_ckpt"]


def cmd_acorn_args(a) -> None:
    p = pre_settings(a.seed)
    for k in ("ml", "gnn"):
        if _sha256(p[f"{k}_ckpt"]) != p[f"{k}_sha256"]:
            raise SystemExit(f"{p[f'{k}_ckpt']}: sha256 differs from the run's inference record")
    print(p["ml_ckpt"], p["r_infer"], p["knn_infer"], p["gnn_ckpt"])


def cmd_union_args(a) -> None:
    p = pre_settings(a.seed)
    if os.path.realpath(p["union_ml_ckpt"]) != os.path.realpath(p["ml_ckpt"]):
        raise SystemExit(f"union ML checkpoint {p['union_ml_ckpt']} != acorn ML checkpoint {p['ml_ckpt']}")
    print(p["graphs_run"], p["triplet_model"], p["tau"], p["k"])


def cmd_canto_args(a) -> None:
    print(canto_ckpt(a.seed, a.selection, a.canto_ckpt))


# ------------------------------------------------------------------------------------------------- pre-stage outputs
def acorn_dir(seed: int, set_: str) -> str:
    return os.path.join(OUT, f"acorn_g_s{seed}", set_)


def canto_dir(seed: int, set_: str) -> str:
    return os.path.join(OUT, f"canto_g_s{seed}", set_)


def union_block_dir(seed: int, set_: str, k: int, n: int) -> str:
    return os.path.join(canto_dir(seed, set_), "union", "blocks", f"{k:03d}_of_{n:03d}")


def cmd_manifest(a) -> None:
    """sha256 manifest of a pre-stage output, written next to it (manifest.json)."""
    d = acorn_dir(a.seed, a.set) if a.what == "acorn" else union_block_dir(a.seed, a.set, a.block, a.n_blocks)
    files = sorted(f for f in os.listdir(d) if f.endswith(".tar"))
    if not files:
        raise SystemExit(f"{d}: nothing to record")
    man = {"what": a.what, "seed": a.seed, "set": a.set, "dir": d, "written_utc": _now(), "git": T2._git(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "pre_settings": pre_settings(a.seed),
           "files": {f: {"bytes": os.path.getsize(os.path.join(d, f)), "sha256": _sha256(os.path.join(d, f))}
                     for f in files}}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"[manifest] {d}: {files}")


def check_manifest(d: str) -> None:
    man = json.load(open(os.path.join(d, "manifest.json")))
    for f, rec in man["files"].items():
        p = os.path.join(d, f)
        if os.path.getsize(p) != rec["bytes"] or _sha256(p) != rec["sha256"]:
            raise SystemExit(f"{p} differs from its manifest")


def _complete(d: str) -> list[str]:
    return T2._complete(d)


def cmd_gather_union(a) -> None:
    """As hnl.eval.addendum_g gather-union for Canto-L (ttbar+G): the union blocks (checked against their manifests)
    of seed S into DEST, all or part k of n contiguous parts; together exactly the events of the acorn graphs."""
    blocks = _complete(os.path.join(canto_dir(a.seed, a.set), "union", "blocks"))
    if a.n_parts > 1 or a.part:
        blocks = [blocks[i] for i in np.array_split(np.arange(len(blocks)), a.n_parts)[a.part]]
    src = T2._members(os.path.join(acorn_dir(a.seed, a.set), "testset.tar"))
    got = []
    os.makedirs(a.dest, exist_ok=True)
    for b in blocks:
        check_manifest(b)
        with tarfile.open(os.path.join(b, "testset.tar")) as t:
            got += [m for m in t.getnames() if m.endswith(".pyg.gz")]
            t.extractall(a.dest, filter="data")
    if len(set(got)) != len(got) or (sorted(got) != src if a.n_parts == 1 else not set(got) <= set(src)):
        raise SystemExit(f"union blocks hold {len(got)} graphs, the acorn source {len(src)}")
    print(f"[gather-union] {len(got)} union graphs from {len(blocks)} blocks (part {a.part} of {a.n_parts}) -> {a.dest}")


def cmd_gather_scored(a) -> None:
    """As hnl.eval.addendum_g gather-scored for Canto-L (ttbar+G)."""
    d = os.path.join(canto_dir(a.seed, a.set), "scored")
    if os.path.exists(os.path.join(d, "done")):
        raise SystemExit(f"{d} is done: refusing to repeat it")
    parts = _complete(os.path.join(d, "parts"))
    src = T2._members(os.path.join(acorn_dir(a.seed, a.set), "testset.tar"))
    where = {}
    for p in parts:
        for m in T2._members(os.path.join(p, "testset.tar")):
            if m in where:
                raise SystemExit(f"{m} in two parts")
            where[m] = p
    if sorted(where) != src:
        raise SystemExit(f"scored parts hold {len(where)} graphs, the acorn source {len(src)}")
    tmp = os.path.join(d, "testset.tar.tmp")
    handles = {p: tarfile.open(os.path.join(p, "testset.tar")) for p in parts}
    try:
        with tarfile.open(tmp, "w") as o:
            for m in src:
                t = handles[where[m]]
                ti = t.getmember(m)
                o.addfile(ti, t.extractfile(ti))
    finally:
        for t in handles.values():
            t.close()
    os.replace(tmp, os.path.join(d, "testset.tar"))
    for p in parts:
        for f in glob.glob(os.path.join(p, "infer_*.json")):
            shutil.copy(f, os.path.join(d, os.path.basename(f)))
    with open(os.path.join(d, "parts.json"), "w") as f:
        json.dump({"parts": [os.path.relpath(p, d) for p in parts], "n_graphs": len(src), "git": T2._git(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "finished_utc": _now()}, f, indent=1)
    open(os.path.join(d, "done"), "w").close()
    for p in parts:
        os.remove(os.path.join(p, "testset.tar"))
    print(f"[gather-scored] {len(src)} scored graphs from {len(parts)} parts -> {d}/testset.tar")
    if a.prune_union:
        blocks = _complete(os.path.join(canto_dir(a.seed, a.set), "union", "blocks"))
        for b in blocks:
            if os.path.exists(os.path.join(b, "testset.tar")):
                os.remove(os.path.join(b, "testset.tar"))
                open(os.path.join(b, "pruned"), "w").close()
        print(f"[gather-scored] removed the graph tars of {len(blocks)} union blocks")


# ------------------------------------------------------------------------------------------------- evaluation
def scored_tar(run: str, seed: int, set_: str) -> str:
    if RUN_MODEL[run] == "acorn":
        return os.path.join(acorn_dir(seed, set_), "testset.tar")
    return os.path.join(canto_dir(seed, set_), "scored", "testset.tar")


def _eval_chunk(args):
    """hnl.eval.addendum_g._eval_chunk with acorn's CC through evaluate_gnn (as its grid) and every other configuration,
    Canto-L's CC included, through trackbuild_val (as its grid)."""
    paths, model, cfg, keep = args
    from hnl.eval import evaluate_gnn as EG
    labels = {}
    if model == "acorn" and cfg[0] == "cc":
        res = [EG.evaluate_event(p, [cfg[1]], {}, keep, labels_out=labels) for p in paths]
        part = pd.concat([r[0][["event_id", "particle_id", "dm_matched", "n_sm_tracks"]] for r in res], ignore_index=True)
        ev = pd.concat([r[1][["event_id", "n_tracks_ge5", "n_fake"]] for r in res], ignore_index=True)
        ceil = [r[2] for r in res]
    else:
        ceil = [EG.evaluate_event(p, [], {}, keep)[2] for p in paths]
        part, ev = T2._tv_chunk(paths, cfg, keep, labels)
        part = part[["event_id", "particle_id", "dm_matched", "n_sm_tracks"]]
        ev = ev[["event_id", "n_tracks_ge5", "n_fake"]]
    if len(labels) != len(paths):
        raise RuntimeError(f"track labels of {len(labels)} of {len(paths)} events")
    ceil = pd.concat(ceil, ignore_index=True)
    part = part.merge(ceil, on=["event_id", "particle_id"], how="left", validate="one_to_one")
    return part, ev, T2.tracks_frame(labels)


def cmd_evaluate(a) -> None:
    T2._require_generated()
    e = check_entry(a.run, a.seed, a.selection)
    tar = scored_tar(a.run, a.seed, a.set)
    if RUN_MODEL[a.run] == "acorn":
        check_manifest(os.path.dirname(tar))
    elif not os.path.exists(os.path.join(os.path.dirname(tar), "done")):
        raise SystemExit(f"no finished scored graphs at {tar}")
    out = os.path.join(a.eval_root, f"{a.run}_s{a.seed}", a.set)
    blk = os.path.join(out, "blocks", f"{a.block:03d}_of_{a.n_blocks:03d}")
    if os.path.exists(os.path.join(blk, "done")):
        raise SystemExit(f"{blk} is done: refusing to repeat it")
    os.environ["HNL_MERGE_MODEL"] = os.path.abspath(e["merge_model"])
    cfg = tuple([e["config"].split(",")[0], *map(float, e["config"].split(",")[1:])])
    local = os.path.join("/tmp", os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", "local"), "g4")
    os.makedirs(local, exist_ok=True)
    t0 = time.time()
    with tarfile.open(tar) as t:
        names = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))
        sel = [names[i] for i in np.array_split(np.arange(len(names)), a.n_blocks)[a.block]]
        for nm in sel:
            t.extract(nm, local, filter="data")
    paths = [os.path.join(local, nm) for nm in sel]
    keep_all = T2.keep_ids(a.set)
    chunks = []
    for k in range(0, len(paths), 10):
        ch = paths[k:k + 10]
        keep = {x: keep_all[x] for x in (os.path.basename(p).split("event")[-1].split(".")[0] for p in ch)
                if x in keep_all}
        chunks.append((ch, RUN_MODEL[a.run], cfg, keep))
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_eval_chunk, chunks))
    os.makedirs(blk, exist_ok=True)
    for i, nm in enumerate(("per_particle", "per_event", "tracks")):
        pd.concat([r[i] for r in res], ignore_index=True).to_parquet(os.path.join(blk, f"{nm}.parquet"), index=False)
    prov = {"run": a.run, "label": LABELS[a.run], "seed": a.seed, "set": a.set, "entry": e,
            "builder": list(map(lambda x: x if isinstance(x, str) else float(x), cfg)), "scored_graphs": tar,
            "selection_file": a.selection or SELECTION, "block": a.block, "n_blocks": a.n_blocks, "n_events": len(sel),
            "git": T2._git(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "elapsed_s": round(time.time() - t0, 1), "finished_utc": _now()}
    with open(os.path.join(blk, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    open(os.path.join(blk, "done"), "w").close()
    shutil.rmtree(local, ignore_errors=True)
    print(json.dumps({k: prov[k] for k in ("run", "seed", "set", "block", "n_events", "elapsed_s")}))


def cmd_merge(a) -> None:
    out = os.path.join(a.eval_root, f"{a.run}_s{a.seed}", a.set)
    blocks = _complete(os.path.join(out, "blocks"))
    pp = pd.concat([pd.read_parquet(os.path.join(b, "per_particle.parquet")) for b in blocks], ignore_index=True)
    pe = pd.concat([pd.read_parquet(os.path.join(b, "per_event.parquet")) for b in blocks], ignore_index=True)
    n_graphs = len(T2._members(scored_tar(a.run, a.seed, a.set)))
    if pe.event_id.nunique() != n_graphs or len(pe) != n_graphs:
        raise SystemExit(f"{out}: {pe.event_id.nunique()} events evaluated, {n_graphs} scored graphs")
    pp.to_parquet(os.path.join(out, "per_particle.parquet"), index=False)
    pe.to_parquet(os.path.join(out, "per_event.parquet"), index=False)
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump({"blocks": [json.load(open(os.path.join(b, "provenance.json"))) for b in blocks],
                   "n_events": n_graphs, "merged_utc": _now()}, f, indent=1)
    print(f"[merge] {out}: {n_graphs} events, {len(pp)} particle rows")


def cmd_runs(a) -> None:
    print(" ".join(runs(a.selection)))


# ------------------------------------------------------------------------------------------------- summary
def _diff(D, A: tuple, B: tuple) -> dict:
    from hnl.eval.bootstrap import linear_combination
    d = linear_combination([(1.0, A[0], A[1]), (-1.0, B[0], B[1])])
    return {"delta_points": 100 * d["value"], "sigma_points": 100 * d["sigma"],
            "sigma_boot_points": 100 * d["sigma_boot"], "sigma_seeds_points": 100 * d["sigma_seeds"]}


def claim(D, name: str, a: str, b: str, labels: dict, sensitivity: bool = False) -> dict:
    """G2 (non-inferiority, margin 1 point, fake-rate guard +0.5 points) or G3 (Delta >= 3 sigma, fake rate not
    higher), section 6; A and B are evaluation runs (frozen T2 runs or T4 runs)."""
    s = "generic_iso"
    mask = D.truth[s].in_pop.to_numpy(bool)
    A, B = D.eff(a, s, mask), D.eff(b, s, mask)
    d = _diff(D, A, B)
    fa, fb = D.fake(a, s), D.fake(b, s)
    fra, frb = float(np.mean(fa[0])), float(np.mean(fb[0]))
    dfr = 100 * (fra - frb)
    out = {"claim": name, "sensitivity": sensitivity, "A": labels[a], "B": labels[b], "population": "P_G",
           "split": "generic_iso union test (T1 deviation 13a)" if D.parts[0][1] == "union_test" else D.parts[0][1],
           "n_targets": int(mask.sum()), "eff_A": float(np.mean(A[0])), "eff_B": float(np.mean(B[0])),
           "eff_A_seeds": [float(x) for x in A[0]], "eff_B_seeds": [float(x) for x in B[0]], **d,
           "fake_rate_A": fra, "fake_rate_B": frb, "fake_rate_diff_points": dfr}
    z = d["delta_points"] / d["sigma_points"] if d["sigma_points"] > 0 else np.nan
    out["z"] = z
    if name.startswith("G2"):
        guard = dfr <= FR_GUARD
        lo, hi = d["delta_points"] - 2 * d["sigma_points"], d["delta_points"] + 2 * d["sigma_points"]
        if lo > -MARGIN and guard:
            outcome = "holds"
        elif hi < -MARGIN:
            outcome = "does not hold, resolved below the margin"
        else:
            outcome = "does not hold, inconclusive" + ("" if guard else " (fake-rate guard fails)")
        out.update(delta_minus_2sigma=lo, delta_plus_2sigma=hi, margin=MARGIN, fake_rate_guard_ok=bool(guard))
    else:
        guard = fra <= frb
        outcome = "holds" if (z >= G3_Z and guard) else "does not hold"
        out.update(threshold_sigma=G3_Z, fake_rate_not_higher=bool(guard))
    out["outcome"] = outcome
    out["underpowered"] = out["n_targets"] < 1000
    return out


class Data4:
    """The T2 Data of a set (frozen runs: truth, bootstrap tables, Canto-L (ttbar) and the acorn runs) with the T4
    evaluation runs added; seeds restricted for code tests."""

    def __init__(self, set_: str, seeds=SEEDS, eval_root: str | None = None, selection_path: str | None = None):
        from hnl.eval import addendum_g_summary as S
        self.seeds = tuple(seeds)
        self.D = S.Data(set_, eval_root=os.path.join(T2.OUT, "eval"))
        self.root = eval_root or os.path.join(OUT, "eval")
        self.runs = [r for r in runs(selection_path)]
        for r in self.runs:
            for s in self.seeds:
                d = os.path.join(self.root, f"{r}_s{s}", set_)
                if not os.path.exists(os.path.join(d, "per_particle.parquet")):
                    raise SystemExit(f"no merged evaluation {d}")
                self.D.res[(r, s)] = (pd.read_parquet(os.path.join(d, "per_particle.parquet")),
                                      pd.read_parquet(os.path.join(d, "per_event.parquet")),
                                      json.load(open(os.path.join(d, "provenance.json"))))

    def _seeds(self, model):
        return self.seeds if model in RUN_MODEL else T2.model_seeds(model)

    def eff(self, model, sample, mask, what="dm"):
        import hnl.eval.addendum_g_summary as S
        orig = S.model_seeds
        S.model_seeds = lambda m: self._seeds(m)
        try:
            return self.D.eff(model, sample, mask, what)
        finally:
            S.model_seeds = orig

    def fake(self, model, sample):
        import hnl.eval.addendum_g_summary as S
        orig = S.model_seeds
        S.model_seeds = lambda m: self._seeds(m)
        try:
            return self.D.fake(model, sample)
        finally:
            S.model_seeds = orig

    @property
    def truth(self):
        return self.D.truth

    @property
    def parts(self):
        return self.D.parts


def summary(set_: str, seeds=SEEDS, selection_path: str | None = None, out_dir: str | None = None,
            eval_root: str | None = None) -> dict:
    from hnl.eval import addendum_g_summary as S
    from hnl.eval.bootstrap import seed_mean
    X = Data4(set_, seeds, eval_root, selection_path)
    labels = {**T2.LABELS, **LABELS}
    frozen = [m for m in ("canto_l_ttbar", "acorn_hnl_lm") if all((m, s) in X.D.res for s in T2.model_seeds(m))]
    models = frozen + X.runs
    claims = []
    if "canto_l_ttbar" in frozen:
        claims.append(claim(X, "G2", "canto_l_ttbar", "acorn_g", labels))
        if "acorn_g_alt" in X.runs:
            claims.append(claim(X, "G2 (guarded alternative of acorn)", "canto_l_ttbar", "acorn_g_alt", labels, True))
    claims.append(claim(X, "G3", "canto_g", "acorn_g", labels))
    if "canto_g_alt" in X.runs or "acorn_g_alt" in X.runs:
        a = "canto_g_alt" if "canto_g_alt" in X.runs else "canto_g"
        b = "acorn_g_alt" if "acorn_g_alt" in X.runs else "acorn_g"
        claims.append(claim(X, "G3 (guarded alternatives)", a, b, labels, True))
    rows = []
    for s, _ in X.parts:
        t = X.truth[s]
        for pop, mask in S.populations(s, t).items():
            for m in models:
                pts, reps, n = X.eff(m, s, mask)
                cp, cr, _ = X.eff(m, s, mask, "ceil")
                sm, sc = seed_mean(pts, reps), seed_mean(cp, cr)
                rows.append({"sample": s, "population": pop, "run": m, "label": labels[m], "n": n,
                             **{f"eff_s{i}": p for i, p in enumerate(pts)}, "mean": sm["mean"], "sigma": sm["sigma"],
                             "ceiling": sc["mean"]})
        for m in models:
            pts, reps = X.fake(m, s)
            sm = seed_mean(pts, reps)
            rows.append({"sample": s, "population": "track fake rate", "run": m, "label": labels[m], "n": 0,
                         **{f"eff_s{i}": p for i, p in enumerate(pts)}, "mean": sm["mean"], "sigma": sm["sigma"]})
    main = pd.DataFrame(rows)
    tag = "" if tuple(seeds) == SEEDS and selection_path is None else "_codetest"
    od = out_dir or os.path.join(REPO, "results")
    stem = os.path.join(od, f"addendum_g_t4_{set_}{tag}_v1")
    main.to_csv(stem + ".csv", index=False)
    res = {"set": set_, "seeds": list(seeds), "selection_file": selection_path or SELECTION,
           "selection_sha256": _sha256(selection_path or SELECTION), "written_utc": _now(), "git": T2._git(),
           "claims": claims, "code_test": bool(tag)}
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    L = [f"# Addendum G, step T4 ({set_}){': CODE TEST (not a claim read-out)' if tag else ''}", "",
         f"Written {_now()} by `hnl.eval.addendum_g_t4` (git {T2._git()[:7]}); selection file "
         f"`{os.path.relpath(selection_path or SELECTION, REPO)}` (sha256 {res['selection_sha256'][:12]}); seeds "
         f"{', '.join(map(str, seeds))}. Statistics of addendum G section 6 (hnl.eval.addendum_g_summary: B = 500, "
         "the claims' bootstrap table of T2). Efficiencies and fake rates in percent.", ""]
    for c in claims:
        L += [f"## {c['claim']}{' (sensitivity, decides no claim)' if c['sensitivity'] else ''}", "",
              f"- {c['A']} against {c['B']} on P_G of the {c['split']} ({c['n_targets']} targets"
              f"{', underpowered' if c['underpowered'] else ''}).",
              f"- Efficiency {100 * c['eff_A']:.2f} against {100 * c['eff_B']:.2f}; Delta = {c['delta_points']:+.2f} +- "
              f"{c['sigma_points']:.2f} points (bootstrap {c['sigma_boot_points']:.2f}, seeds "
              f"{c['sigma_seeds_points']:.2f}; Delta / sigma = {c['z']:+.2f}).",
              f"- Track fake rate {100 * c['fake_rate_A']:.2f} against {100 * c['fake_rate_B']:.2f} "
              f"({c['fake_rate_diff_points']:+.2f} points).", f"- **Outcome: {c['outcome']}.**", ""]
    for s, _ in X.parts:
        sub = main[main["sample"] == s]
        pops = list(dict.fromkeys(sub.population))
        L += [f"## {s}", "", "| run | " + " | ".join(pops) + " |", "|---|" + "---|" * len(pops)]
        for m in models:
            cells = []
            for p in pops:
                r = sub[(sub.run == m) & (sub.population == p)].iloc[0]
                cells.append(f"{100 * r['mean']:.2f} +- {100 * r['sigma']:.2f}")
            L.append(f"| {labels[m]} | " + " | ".join(cells) + " |")
        L.append("")
    with open(stem + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sets = list(T2.SETS)
    for name in ("acorn-args", "union-args", "canto-args"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True)
        if name == "canto-args":
            p.add_argument("--selection", default=None)
            p.add_argument("--canto-ckpt", default=None, help="code tests on val only")
    p = sub.add_parser("manifest")
    p.add_argument("--what", required=True, choices=["acorn", "union"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", required=True, choices=sets)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=1)
    p = sub.add_parser("gather-union")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", required=True, choices=sets)
    p.add_argument("--dest", required=True)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--n-parts", type=int, default=1)
    p = sub.add_parser("gather-scored")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", required=True, choices=sets)
    p.add_argument("--prune-union", action="store_true")
    for name in ("evaluate", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, choices=list(RUN_MODEL))
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--set", required=True, choices=sets)
        p.add_argument("--eval-root", default=os.path.join(OUT, "eval"))
        p.add_argument("--selection", default=None)
        if name == "evaluate":
            p.add_argument("--block", type=int, default=0)
            p.add_argument("--n-blocks", type=int, default=1)
            p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p = sub.add_parser("runs")
    p.add_argument("--selection", default=None)
    p = sub.add_parser("summary")
    p.add_argument("--set", required=True, choices=sets)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--selection", default=None)
    p.add_argument("--eval-root", default=None)
    p.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        summary(a.set, tuple(int(x) for x in a.seeds.split(",")), a.selection, a.out_dir, a.eval_root)
        return
    {"acorn-args": cmd_acorn_args, "union-args": cmd_union_args, "canto-args": cmd_canto_args,
     "manifest": cmd_manifest, "gather-union": cmd_gather_union, "gather-scored": cmd_gather_scored,
     "evaluate": cmd_evaluate, "merge": cmd_merge, "runs": cmd_runs}[a.cmd](a)


if __name__ == "__main__":
    main()
