"""Preregistration addendum G (docs/preregistration_new_sample.md), step T2: the frozen models on the generated samples
(docs/new_samples_spec.md), after the T1 commit (hnl.eval.addendum_g_t1, results/addendum_g_t1_v1.md).

Nothing here is trained or chosen. Every network, builder and working point is the frozen one, read from the committed
records: results/final_selection_v1.json (Canto-L (ttbar)), results/strong_baselines_v3_spec_test.json (acorn + LM,
acorn (HNL) + CC), results/training_sample_test_selection_v1.json (Canto-L (HNL), Canto-L (ttbar+HNL), seed 0,
section 7). Checkpoints and (r_infer, knn_infer) are those of each run's own one-time test inference (the infer_*.json
records next to its stored test graphs), checked by sha256 before use. The chain per run is the one of the frozen test
(slurm/baseline/test_eval.sbatch, slurm/stage1/final_test_chain.sh, final_test.sbatch), with new input and output
directories:
  acorn arm A seed S   feature store -> metric learning (ML ckpt, r_infer, knn_infer) -> edge classifier
                       -> DATA/runs/addendum_g_v1/acorn_<A>_s<S>/<set>/testset.tar
  Canto-L              hnl.graphs.helix_union build on those scored graphs (--no-metrics; the run's learned triplet
                       filter, tau, k) -> its edge classifier -> DATA/runs/addendum_g_v1/<model>_s<S>/<set>/scored/
  evaluation           the frozen builder (hnl.eval.trackbuild_val, learned merging with the run's merge model; CC
                       through hnl.eval.evaluate_gnn at the preregistered cut) and double-majority matching on every
                       hit and particle; graph ceilings (perfect classifier + CC, hnl.eval.evaluate_gnn); per-particle
                       rows for targets and gun particles only
Sets: val = generic_iso val + generic_dense val (allowed by addendum G T2; nothing frozen can change); test =
generic_iso test + generic_dense test; hss = hss15_100 + hss40_300, every event (evaluation only).

    python -m hnl.eval.addendum_g prepare                      (event tables + feature-store tars; truth only)
    python -m hnl.eval.addendum_g acorn-args --arm i --seed 0  (shell words: ML ckpt, r, knn, GNN ckpt)
    python -m hnl.eval.addendum_g canto-args --model canto_l_ttbar --seed 0
    python -m hnl.eval.addendum_g stage --set val --dest DIR
    python -m hnl.eval.addendum_g evaluate --model canto_l_ttbar --seed 0 --set val [--block k --n-blocks n]
    python -m hnl.eval.addendum_g merge --model canto_l_ttbar --seed 0 --set val
    python -m hnl.eval.addendum_g summary --set val|test|hss
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
import subprocess
import tarfile
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
RUNS = os.path.join(DATA, "runs")
B = os.path.join(RUNS, "baseline_v1")
OUT = os.path.join(RUNS, "addendum_g_v1")
FS = os.path.join(DATA, "derived", "acorn_fs", "generated_v1")
T1 = os.path.join(DATA, "derived", "generated_v1", "t1")
SETS = {"val": (("generic_iso", "val"), ("generic_dense", "val")),
        "test": (("generic_iso", "test"), ("generic_dense", "test")),
        "hss": (("hss15_100", "all"), ("hss40_300", "all"))}
LABELS = {
    "canto_l_ttbar": "Canto-L (ttbar)",
    "acorn_hnl_lm": "acorn (HNL) + LM",
    "acorn_ttbar_lm": "acorn (ttbar) + LM",
    "acorn_hnl_cc": "acorn (HNL) + CC",
    "canto_l_hnl": "Canto-L (HNL)",
    "canto_l_tthnl": "Canto-L (ttbar+HNL)",
}
SPEC_LABEL = {"acorn_hnl_lm": "acorn (HNL) + MWS-layer + learned merging (Canto-L's builder)",
              "acorn_ttbar_lm": "acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder)",
              "acorn_hnl_cc": "acorn (HNL) + CC, preregistered cut (claims' reference)"}


def _require_generated():
    from hnl.data import npz
    if "generic_iso" not in npz.SAMPLES:
        raise SystemExit("the generated samples are not registered: run with HNL_GENERATED=1")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _repo(p: str) -> str:
    return os.path.join(REPO, p)


# ------------------------------------------------------------------------------------------------- frozen models
def _infer_record(d: str, prefix: str = "") -> dict:
    fs = sorted(glob.glob(os.path.join(d, f"{prefix}infer_*.json")))
    if len(fs) != 1:
        raise SystemExit(f"expected one {prefix}infer_*.json in {d}, found {len(fs)}")
    return json.load(open(fs[0]))


def acorn_settings(arm: str, seed: int) -> dict:
    """The checkpoints and graph radius of acorn run arm/seed as used in its one-time test inference."""
    d = os.path.join(B, f"arm{arm}_s{seed}", "gnn_scores_test")
    ml, gnn = _infer_record(d, "ml_"), _infer_record(d)
    return {"run": os.path.join(B, f"arm{arm}_s{seed}"), "ml_ckpt": ml["ckpt"], "ml_sha256": ml["ckpt_sha256"],
            "r_infer": ml["overrides"]["r_infer"], "knn_infer": ml["overrides"]["knn_infer"],
            "gnn_ckpt": gnn["ckpt"], "gnn_sha256": gnn["ckpt_sha256"]}


def canto_settings(model: str, seed: int) -> dict:
    if model == "canto_l_ttbar":
        sel = json.load(open(_repo("results/final_selection_v1.json")))
        e = sel["pipelines"]["l1tl"]["seeds"][str(seed)]
        graphs_run, arm, source = os.path.join(B, f"armi_s{seed}"), "i", "results/final_selection_v1.json"
    else:
        sel = json.load(open(_repo("results/training_sample_test_selection_v1.json")))
        p = sel["pipelines"][{"canto_l_hnl": "l1tl_ii", "canto_l_tthnl": "l1tl_iii"}[model]]
        e = p["seeds"][str(seed)]
        graphs_run, arm, source = p["graphs_run"], {"canto_l_hnl": "ii", "canto_l_tthnl": "iii"}[model], \
            "results/training_sample_test_selection_v1.json"
    ub = json.load(open(os.path.join(graphs_run, "ltf_union_graphs", "report_test", "build.json")))
    gi = _infer_record(os.path.join(e["run"], "gnn_scores", "test_infer"))
    return {"run": e["run"], "graphs_run": graphs_run, "acorn_arm": arm, "acorn_seed": seed, "config": e["config"],
            "merge_model": e["merge_model"], "cut_e": e["cut_e"], "source": source,
            "triplet_model": ub["triplet_model"], "tau": ub["tau"], "k": ub["k"], "r_wide": ub["r_wide"],
            "nsig": ub["nsig"], "pt_min": ub["pt_min"], "union_ml_ckpt": ub["ml_ckpt"],
            "gnn_ckpt": gi["ckpt"], "gnn_sha256": gi["ckpt_sha256"]}


def baseline_settings(model: str, seed: int) -> dict:
    spec = json.load(open(_repo("results/strong_baselines_v3_spec_test.json")))
    m = [x for x in spec["models"] if x["label"] == SPEC_LABEL[model]]
    if len(m) != 1:
        raise SystemExit(f"{SPEC_LABEL[model]}: {len(m)} entries in the spec")
    e = m[0]["entries"][seed]
    arm = os.path.basename(e["run"]).split("_")[0][3:]
    if not e["run"].endswith(f"_s{seed}"):
        raise SystemExit(f"entry {seed} of {model} is {e['run']}")
    out = {"run": e["run"], "acorn_arm": arm, "acorn_seed": seed, "source": "results/strong_baselines_v3_spec_test.json"}
    if m[0]["kind"] == "cc":
        out["cut"] = float(e["cut"])
    else:
        prov = json.load(open(os.path.join(e["sub"], "provenance.json")))
        if prov["config"] != e["config"]:
            raise SystemExit(f"{model} seed {seed}: provenance config {prov['config']} != spec {e['config']}")
        out.update(config=e["config"], merge_model=prov["merge_model"], merge_sha256=prov["merge_model_sha256"])
    return out


def model_settings(model: str, seed: int) -> dict:
    s = canto_settings(model, seed) if model.startswith("canto") else baseline_settings(model, seed)
    s["model"], s["seed"], s["label"] = model, seed, LABELS[model]
    return s


def model_seeds(model: str) -> list[int]:
    return [0] if model in ("canto_l_hnl", "canto_l_tthnl") else [0, 1, 2]


def scored_tar(model: str, seed: int, set_: str) -> str:
    s = model_settings(model, seed)
    if model.startswith("canto"):
        return os.path.join(OUT, f"{model}_s{seed}", set_, "scored", "testset.tar")
    return os.path.join(OUT, f"acorn_{s['acorn_arm']}_s{seed}", set_, "testset.tar")


def cmd_acorn_args(a) -> None:
    s = acorn_settings(a.arm, a.seed)
    for k in ("ml_ckpt", "gnn_ckpt"):
        h = _sha256(s[k])
        if h != s[k.replace("ckpt", "sha256")]:
            raise SystemExit(f"{s[k]}: sha256 {h} differs from the test record")
    print(s["ml_ckpt"], s["r_infer"], s["knn_infer"], s["gnn_ckpt"])


def cmd_canto_args(a) -> None:
    s = canto_settings(a.model, a.seed)
    if _sha256(s["gnn_ckpt"]) != s["gnn_sha256"]:
        raise SystemExit(f"{s['gnn_ckpt']}: sha256 differs from the test record")
    acorn = acorn_settings(s["acorn_arm"], s["acorn_seed"])
    if os.path.realpath(s["union_ml_ckpt"]) != os.path.realpath(acorn["ml_ckpt"]):
        raise SystemExit(f"union ML checkpoint {s['union_ml_ckpt']} != acorn ML checkpoint {acorn['ml_ckpt']}")
    print(s["graphs_run"], s["triplet_model"], s["tau"], s["k"], s["gnn_ckpt"], s["acorn_arm"])


def cmd_models(a) -> None:
    """The frozen settings of every model and seed, written once for the record (results/addendum_g_models_v1.json)."""
    out = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": _git(),
           "note": "frozen settings read from the committed records; nothing chosen on the generated samples",
           "models": {m: [model_settings(m, s) for s in model_seeds(m)] for m in LABELS},
           "acorn": {f"{arm}_s{s}": acorn_settings(arm, s) for arm, seeds in (("i", (0, 1, 2)), ("ii", (0, 1, 2)),
                                                                             ("iii", (0,))) for s in seeds}}
    path = _repo("results/addendum_g_models_v1.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path)


# ------------------------------------------------------------------------------------------------- data
def _convert_shard(args):
    """hnl.data.convert.convert_sample for one shard of 500 events (the same tables and file names)."""
    s, first, root = args
    from hnl.data import convert as CV
    from hnl.data.npz import SAMPLES, load_event
    n = SAMPLES[s].n_events
    frames = {"hits": [], "particles": [], "edges": []}
    for i in range(first, min(first + CV.SHARD, n)):
        for k, v in CV.event_tables(load_event(s, i), s, i).items():
            frames[k].append(v)
    out = {}
    for k, fr in frames.items():
        df = pd.concat(fr, ignore_index=True)
        d = os.path.join(root, k, s)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"shard_{first:05d}.parquet")
        if os.path.exists(path):
            raise FileExistsError(path)
        df.to_parquet(path + ".tmp", index=False, row_group_size=200_000)
        os.replace(path + ".tmp", path)
        out[k] = len(df)
    return s, first, out


def _tar(args):
    from hnl.data import acorn_store as AS
    s, sp = args
    return s, sp, AS.build_tar(s, sp, FS)


def cmd_prepare(a) -> None:
    """Event tables (hnl.data.convert.event_tables, one shard per 500 events) and acorn feature-store tars
    (hnl.data.acorn_store.build_tar) of the generated samples, from the truth files only; split lists from T1
    (splits/<sample>_<split>.txt). No event is dropped (T1)."""
    _require_generated()
    from hnl.data import acorn_store as AS
    from hnl.data import convert as CV
    from hnl.data.npz import SAMPLES
    root = os.path.join(DATA, "derived", "events", "v1")
    meta = {"convert": [], "tars": {}, "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    jobs = [(s, first, root) for s in AS.GENERATED_CODE for first in range(0, SAMPLES[s].n_events, CV.SHARD)
            if not os.path.exists(os.path.join(root, "edges", s, f"shard_{first:05d}.parquet"))]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        for s, first, out in ex.map(_convert_shard, jobs):
            meta["convert"].append({"sample": s, "first": first, **out})
            print(f"[prepare] {s} shard {first}: {out}", flush=True)
    os.makedirs(FS, exist_ok=True)
    tars = [(s, sp) for s, splits in AS.GENERATED_SPLITS.items() for sp in splits
            if not os.path.exists(os.path.join(FS, f"{s}_{sp}.tar"))]
    with cf.ProcessPoolExecutor(max_workers=max(min(a.workers, len(tars)), 1)) as ex:
        for s, sp, out in ex.map(_tar, tars):
            meta["tars"][f"{s}_{sp}"] = out
            print(f"[prepare] {s} {sp}: {out}", flush=True)
    with open(os.path.join(FS, f"prepare_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as f:
        json.dump(meta, f, indent=1)


def cmd_stage(a) -> None:
    """Extract the feature-store events of a set into DEST (flat), as acorn's input split directory."""
    os.makedirs(a.dest, exist_ok=True)
    n = 0
    for s, sp in SETS[a.set]:
        with tarfile.open(os.path.join(FS, f"{s}_{sp}.tar")) as t:
            t.extractall(a.dest, filter="data")
            n += len(t.getnames())
    print(f"[stage] {a.set}: {n} events -> {a.dest}")


def _members(tar: str) -> list[str]:
    with tarfile.open(tar) as t:
        return sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))


def cmd_block_tar(a) -> None:
    """Block k of n of a packed graph tar (sorted members, numpy.array_split), written as a tar of the same members."""
    names = _members(a.tar)
    sel = set(names[i] for i in np.array_split(np.arange(len(names)), a.n_blocks)[a.block])
    os.makedirs(os.path.dirname(a.dest), exist_ok=True)
    with tarfile.open(a.tar) as t, tarfile.open(a.dest, "w") as o:
        for m in t.getmembers():
            if m.name in sel:
                o.addfile(m, t.extractfile(m))
    print(f"[block-tar] block {a.block} of {a.n_blocks}: {len(sel)} of {len(names)} graphs -> {a.dest}")


def cmd_gather_union(a) -> None:
    """Extract every union block of (model, seed, set) into DEST; check the blocks are complete and that together they
    hold exactly the events of the acorn graphs they were built from."""
    s = canto_settings(a.model, a.seed)
    d = os.path.join(OUT, f"{a.model}_s{a.seed}", a.set, "union", "blocks")
    blocks = sorted(glob.glob(os.path.join(d, "*")))
    n = {int(os.path.basename(b).split("_of_")[1]) for b in blocks}
    if len(n) != 1 or len(blocks) != n.pop() or not all(os.path.exists(os.path.join(b, "done")) for b in blocks):
        raise SystemExit(f"{d}: incomplete union blocks {blocks}")
    src = _members(os.path.join(OUT, f"acorn_{s['acorn_arm']}_s{a.seed}", a.set, "testset.tar"))
    got = []
    os.makedirs(a.dest, exist_ok=True)
    for b in blocks:
        with tarfile.open(os.path.join(b, "testset.tar")) as t:
            got += [m for m in t.getnames() if m.endswith(".pyg.gz")]
            t.extractall(a.dest, filter="data")
    if sorted(got) != src or len(set(got)) != len(got):
        raise SystemExit(f"union blocks hold {len(got)} graphs, the acorn source {len(src)}")
    print(f"[gather-union] {len(got)} union graphs from {len(blocks)} blocks -> {a.dest}")


# ------------------------------------------------------------------------------------------------- evaluation
def _truth(samples) -> pd.DataFrame:
    from hnl.data.acorn_store import SAMPLE_CODE
    out = []
    for s in samples:
        t = pd.read_parquet(os.path.join(T1, f"truth_{s}.parquet"))
        t["sample"] = s
        t["event_id"] = np.int64(SAMPLE_CODE[s]) * 100_000_000 + t.event.to_numpy(np.int64)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def keep_ids(set_: str) -> dict:
    """event_id (str) -> sorted particle ids written to the per-particle tables: targets and gun particles."""
    t = _truth([s for s, _ in SETS[set_]])
    k = t.is_target.to_numpy(bool).copy()
    if "gun" in t:
        k |= t.gun.fillna(False).to_numpy(bool)
    t = t[k]
    return {str(e): np.sort(g.particle_id.to_numpy(np.int64)) for e, g in t.groupby("event_id")}


def _eval_chunk(args):
    paths, model, cfg, keep = args
    from hnl.eval import evaluate_gnn as EG
    from hnl.eval import trackbuild_val as TV
    if cfg[0] == "cc":                     # hnl.eval.evaluate_gnn at the cut, as the frozen eval_test
        res = [EG.evaluate_event(p, [cfg[1]], {}, keep) for p in paths]
        part = pd.concat([r[0][["event_id", "particle_id", "dm_matched", "n_sm_tracks"]] for r in res], ignore_index=True)
        ev = pd.concat([r[1][["event_id", "n_tracks_ge5", "n_fake"]] for r in res], ignore_index=True)
        ceil = [r[2] for r in res]
    else:
        ceil = [EG.evaluate_event(p, [], {}, keep)[2] for p in paths]
        part, ev = TV._chunk((paths, [cfg], {}, keep))
        part = part[["event_id", "particle_id", "dm_matched", "n_sm_tracks"]]
        ev = ev[["event_id", "n_tracks_ge5", "n_fake"]]
    ceil = pd.concat(ceil, ignore_index=True)
    part = part.merge(ceil, on=["event_id", "particle_id"], how="left", validate="one_to_one")
    return part, ev


def cmd_evaluate(a) -> None:
    _require_generated()
    s = model_settings(a.model, a.seed)
    tar = scored_tar(a.model, a.seed, a.set)
    out = os.path.join(OUT, "eval", f"{a.model}_s{a.seed}", a.set)
    blk = os.path.join(out, "blocks", f"{a.block:03d}_of_{a.n_blocks:03d}")
    if os.path.exists(os.path.join(blk, "done")):
        raise SystemExit(f"{blk} is done: refusing to repeat it")
    if a.set in ("test", "hss") and not os.path.exists(os.path.join(os.path.dirname(tar), "done")):
        raise SystemExit(f"no finished scored graphs at {tar}")
    if "merge_model" in s:
        if a.model.startswith("acorn") and _sha256(s["merge_model"]) != s["merge_sha256"]:
            raise SystemExit(f"{s['merge_model']}: sha256 differs from the record")
        os.environ["HNL_MERGE_MODEL"] = os.path.abspath(s["merge_model"])
    cfg = ("cc", s["cut"]) if "cut" in s else tuple([s["config"].split(",")[0], *map(float, s["config"].split(",")[1:])])
    local = os.path.join("/tmp", os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", "local"), "g")
    os.makedirs(local, exist_ok=True)
    t0 = time.time()
    with tarfile.open(tar) as t:
        names = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))
        mine = np.array_split(np.arange(len(names)), a.n_blocks)[a.block]
        sel = [names[i] for i in mine]
        for nm in sel:
            t.extract(nm, local, filter="data")
    paths = [os.path.join(local, nm) for nm in sel]
    keep_all = keep_ids(a.set)
    chunks = []
    for k in range(0, len(paths), 10):
        ch = paths[k:k + 10]
        keep = {e: keep_all[e] for e in (os.path.basename(p).split("event")[-1].split(".")[0] for p in ch) if e in keep_all}
        chunks.append((ch, a.model, cfg, keep))
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_eval_chunk, chunks))
    os.makedirs(blk, exist_ok=True)
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(os.path.join(blk, "per_particle.parquet"), index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(os.path.join(blk, "per_event.parquet"), index=False)
    prov = {"model": a.model, "seed": a.seed, "set": a.set, "settings": s, "builder": list(map(
        lambda x: x if isinstance(x, str) else float(x), cfg)), "scored_graphs": tar, "block": a.block,
        "n_blocks": a.n_blocks, "n_events": len(sel), "first": sel[0] if sel else None, "last": sel[-1] if sel else None,
        "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
        "elapsed_s": round(time.time() - t0, 1),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(blk, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    open(os.path.join(blk, "done"), "w").close()
    shutil.rmtree(local, ignore_errors=True)
    print(json.dumps({k: prov[k] for k in ("model", "seed", "set", "block", "n_events", "elapsed_s")}))


def cmd_merge(a) -> None:
    """Concatenate the blocks of one (model, seed, set) in block order; check they cover every scored graph once."""
    out = os.path.join(OUT, "eval", f"{a.model}_s{a.seed}", a.set)
    blocks = sorted(glob.glob(os.path.join(out, "blocks", "*")))
    n = {int(os.path.basename(b).split("_of_")[1]) for b in blocks}
    if len(n) != 1 or len(blocks) != n.pop():
        raise SystemExit(f"{out}: blocks {blocks} do not form one complete set")
    if not all(os.path.exists(os.path.join(b, "done")) for b in blocks):
        raise SystemExit(f"{out}: unfinished blocks")
    pp = pd.concat([pd.read_parquet(os.path.join(b, "per_particle.parquet")) for b in blocks], ignore_index=True)
    pe = pd.concat([pd.read_parquet(os.path.join(b, "per_event.parquet")) for b in blocks], ignore_index=True)
    with tarfile.open(scored_tar(a.model, a.seed, a.set)) as t:
        n_graphs = sum(1 for m in t.getnames() if m.endswith(".pyg.gz"))
    if pe.event_id.nunique() != n_graphs or len(pe) != n_graphs:
        raise SystemExit(f"{out}: {pe.event_id.nunique()} events evaluated, {n_graphs} scored graphs")
    pp.to_parquet(os.path.join(out, "per_particle.parquet"), index=False)
    pe.to_parquet(os.path.join(out, "per_event.parquet"), index=False)
    provs = [json.load(open(os.path.join(b, "provenance.json"))) for b in blocks]
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump({"blocks": provs, "n_events": n_graphs, "merged_utc":
                   dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, f, indent=1)
    print(f"[merge] {out}: {n_graphs} events, {len(pp)} particle rows")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    sub.add_parser("models")
    p = sub.add_parser("acorn-args")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("canto-args")
    p.add_argument("--model", required=True, choices=[m for m in LABELS if m.startswith("canto")])
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("block-tar")
    p.add_argument("--tar", required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--n-blocks", type=int, required=True)
    p.add_argument("--dest", required=True)
    p = sub.add_parser("gather-union")
    p.add_argument("--model", required=True, choices=[m for m in LABELS if m.startswith("canto")])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--dest", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--dest", required=True)
    for name in ("evaluate", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=list(LABELS))
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--set", required=True, choices=list(SETS))
        if name == "evaluate":
            p.add_argument("--block", type=int, default=0)
            p.add_argument("--n-blocks", type=int, default=1)
            p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p = sub.add_parser("summary")
    p.add_argument("--set", required=True, choices=list(SETS))
    sub.add_parser("val-sanity")
    a = ap.parse_args(argv)
    if a.cmd == "val-sanity":
        from hnl.eval import addendum_g_summary as S
        S.val_sanity()
        return
    if a.cmd == "summary":
        from hnl.eval import addendum_g_summary as S
        S.summary(a.set)
        return
    {"prepare": cmd_prepare, "models": cmd_models, "acorn-args": cmd_acorn_args, "canto-args": cmd_canto_args,
     "stage": cmd_stage, "evaluate": cmd_evaluate, "merge": cmd_merge, "block-tar": cmd_block_tar,
     "gather-union": cmd_gather_union}[a.cmd](a)


if __name__ == "__main__":
    main()
