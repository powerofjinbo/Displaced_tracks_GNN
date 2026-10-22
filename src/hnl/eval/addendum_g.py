"""Preregistration addendum G (docs/preregistration_new_sample.md), step T2: the frozen models on the generated samples
(docs/new_samples_spec.md), after the T1 commit (hnl.eval.addendum_g_t1, results/addendum_g_t1_v1.md) and, for the
extension, its T1 (hnl.eval.addendum_g_t1_x1, results/addendum_g_t1_x1_v1.md).

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
                       rows for targets and gun particles (HNL points: targets, N daughters and W muons) only
Sets: val = generic_iso val + generic_dense val (allowed by addendum G T2; nothing frozen can change); test = the
generic_iso union test split (RESULTS T1 deviation 13a: splits/generic_iso_union_test.txt = the T1 test events of
generic_iso_v1, numbers 0-7999, + the test events of the extension generic_iso_x1, numbers 8000-23999, ordered by event
number; evaluated once, no claim on either part alone) + generic_dense test; hss = hss15_100 + hss40_300, every event
(evaluation only); hnl = the five HNL points of docs/hnl_points_spec.md (hnl5_50, hnl20_200, hnl15_500 and the in-stack
references hnl10_100_ref, hnl15_200_ref), every event (evaluation only, no claim).

Inputs. generic_iso_v1, generic_dense_v1 and the h -> s s samples are per-event npz files (hnl.data.npz). The extension
and the HNL points are uncompressed tars of the same npz members (hnl.data.shards); `prepare-inputs` reads them only
through the shard reader and only for the events of their lists (for the extension its test list: no train or val
event of the extension is read, RESULTS T1 deviation 13c), and converts them with the same function
(hnl.data.convert.event_tables) into the same tables and feature-store graphs. Extension events keep the sample key
generic_iso and its code 5 (event ids 5 * 1e8 + event number), so the union is one sample with one truth table, one
bootstrap table and one table directory; the HNL points have codes 10-14 (hnl.data.acorn_store.POINT_CODE).
`shard-identity` packs 500 generic_iso_v1 val events into a scratch tar and shows that the shard path gives the npz
path's tables and graphs (results/addendum_g_t2_inputs_v1.json).

Tracks. Every evaluation block also stores the builder's track label of each hit on a track (tracks.parquet: event_id,
row, track), from which hnl.eval.addendum_g_readouts computes the read-outs of RESULTS T1 deviation 15 that need the
tracks (the label-exchange emulation); the stored tracks reproduce the block's matching exactly (checked there).

Canto-L scoring runs in parts (GPU jobs of at most 1 h, each on a contiguous range of union blocks); `gather-scored`
packs the parts into the one scored tar that the evaluation reads, checks that it holds exactly the events of the acorn
graphs, and removes the union-block tars (the scored graphs contain them; storage). Sets other than val are refused by
slurm/addendum_g/guard.sh in every job that runs a model, until the ledger entry is committed.

    python -m hnl.eval.addendum_g prepare                      (event tables + feature-store tars; truth only)
    python -m hnl.eval.addendum_g prepare-inputs --inputs x1|points|all     (tar-sharded inputs; truth only)
    python -m hnl.eval.addendum_g shard-identity --scratch DIR  (the reader identity proof on 500 val events)
    python -m hnl.eval.addendum_g acorn-args --arm i --seed 0  (shell words: ML ckpt, r, knn, GNN ckpt)
    python -m hnl.eval.addendum_g canto-args --model canto_l_ttbar --seed 0
    python -m hnl.eval.addendum_g stage --set val --dest DIR
    python -m hnl.eval.addendum_g gather-union --model M --seed S --set SET --dest DIR [--part k --n-parts n]
    python -m hnl.eval.addendum_g gather-scored --model M --seed S --set SET [--prune-union]
    python -m hnl.eval.addendum_g evaluate --model canto_l_ttbar --seed 0 --set val [--block k --n-blocks n]
    python -m hnl.eval.addendum_g merge --model canto_l_ttbar --seed 0 --set val
    python -m hnl.eval.addendum_g summary --set val|test|hss|hnl
    python -m hnl.eval.addendum_g val-check --eval-root ROOT --scratch DIR   (the code on the existing val outputs)
    python -m hnl.eval.addendum_g dry-outputs --set hnl --eval-root DIR      (truth-made placeholders; code test only)
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
OUT = os.environ.get("ADDG_OUT", os.path.join(RUNS, "addendum_g_v1"))   # ADDG_OUT: seeds 1-2 of the training-
#   sample runs (ledger R20; slurm/ts_seeds12), an output root on /work whose acorn_ii_s<S> and eval/<model>_s<S> of
#   the R14 runs are links to this default root; default unchanged
TS_SEEDS = tuple(int(x) for x in os.environ.get("ADDG_TS_SEEDS", "0").split(","))   # seeds of canto_l_hnl / _tthnl
SEL_TS_SEEDS12 = "results/training_sample_test_selection_seeds12_v1.json"      # their seeds 1 and 2 (ledger R19)
FS = os.path.join(DATA, "derived", "acorn_fs", "generated_v1")
T1 = os.path.join(DATA, "derived", "generated_v1", "t1")
T2 = os.path.join(DATA, "derived", "generated_v1", "t2")        # truth tables of hnl.eval.addendum_g_truth
GENERATED_ROOT = os.path.join(DATA, "generated")
X1_FIRST = 8000                                                  # extension generic_iso_x1: event numbers 8000-23999
X1_DIR = os.path.join(GENERATED_ROOT, "generic_iso_x1")
X1_TRUTH = os.path.join(T1, "truth_generic_iso_x1.parquet")       # hnl.eval.addendum_g_t1_x1 scan
POINTS_ORDER = ("hnl5_50", "hnl20_200", "hnl15_500", "hnl10_100_ref", "hnl15_200_ref")
REFERENCE = {"hnl5_50": "hnl10_100_ref", "hnl20_200": "hnl10_100_ref", "hnl15_500": "hnl15_200_ref"}
SPLIT_STEM = {"hnl5_50": "hnl_5_50", "hnl20_200": "hnl_20_200", "hnl15_500": "hnl_15_500",
              "hnl10_100_ref": "hnl_10_100", "hnl15_200_ref": "hnl_15_200"}  # splits/<stem>_all.txt
SETS = {"val": (("generic_iso", "val"), ("generic_dense", "val")),
        "test": (("generic_iso", "union_test"), ("generic_dense", "test")),
        "hss": (("hss15_100", "all"), ("hss40_300", "all")),
        "hnl": tuple((p, "all") for p in POINTS_ORDER)}
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
    from hnl.data.acorn_store import SAMPLE_CODE
    if "generic_iso" not in npz.SAMPLES or "hnl5_50" not in SAMPLE_CODE:
        raise SystemExit("the generated samples are not registered: run with HNL_GENERATED=1")


# ------------------------------------------------------------------------------------------------- sets and inputs
def check_union(union: np.ndarray) -> None:
    """The union test list of the claims (T1 deviation 13a): exactly the T1 test list (numbers < 8000) plus the
    extension's test list (numbers >= 8000, every one a test event under the hash rule of addendum G section 3),
    disjoint, sorted by event number."""
    from hnl.data.splits import read_split
    from hnl.eval.addendum_g_t1 import split_of
    v1 = np.sort(read_split("generic_iso", "test"))
    x1 = np.sort(read_split("generic_iso", "x1_test"))
    if len(v1) and v1.max() >= X1_FIRST or len(x1) and x1.min() < X1_FIRST:
        raise SystemExit("generic_iso test lists: T1 events must be < 8000 and extension events >= 8000")
    if not np.array_equal(union, np.union1d(v1, x1)) or len(union) != len(v1) + len(x1):
        raise SystemExit("splits/generic_iso_union_test.txt is not the union of generic_iso_test and generic_iso_x1_test")
    bad = [int(e) for e in x1 if split_of("iso", int(e)) != "test"]
    if bad:
        raise SystemExit(f"extension events that the hash rule does not put in test: {bad[:10]}")


def split_events(sample: str, split: str) -> np.ndarray:
    """Sorted event numbers of one part of a set (the union test list is checked; HNL points: splits/<stem>_all.txt)."""
    from hnl.data.splits import read_split
    if sample in SPLIT_STEM:
        return np.sort(read_split(SPLIT_STEM[sample], split))
    ev = np.sort(read_split(sample, split))
    if (sample, split) == ("generic_iso", "union_test"):
        check_union(ev)
    if (sample, split) == ("generic_iso", "x1_test"):
        check_union(np.union1d(read_split("generic_iso", "test"), ev))
    return ev


def fs_tars(sample: str, split: str) -> list[str]:
    """Feature-store tars (under FS) that hold one part of a set."""
    if (sample, split) == ("generic_iso", "union_test"):
        return ["generic_iso_test.tar", "generic_iso_x1_test.tar"]
    return [f"{sample}_{split}.tar"]


def event_ids(sample: str, events) -> np.ndarray:
    from hnl.data.acorn_store import SAMPLE_CODE
    return np.int64(SAMPLE_CODE[sample]) * 100_000_000 + np.asarray(events, dtype=np.int64)


def shard_inputs() -> dict:
    """(table sample, list) -> (shard directory, sample record for event_tables, feature-store tar) of the inputs stored
    as tar shards: the extension's test events (table sample generic_iso) and every event of each HNL point."""
    from hnl.data import npz, shards
    out = {("generic_iso", "x1_test"): (X1_DIR, npz.SAMPLES["generic_iso"], "generic_iso_x1_test.tar")}
    for p in POINTS_ORDER:
        out[(p, "all")] = (shards.sample_dir(p), shards.POINTS[p], f"{p}_all.tar")
    return out


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
        source = "results/training_sample_test_selection_v1.json"
        pname = {"canto_l_hnl": "l1tl_ii", "canto_l_tthnl": "l1tl_iii"}[model]
        sel = json.load(open(_repo(source)))
        if str(seed) not in sel["pipelines"][pname]["seeds"]:      # seeds 1 and 2 (ledger R19): their own selection
            source = SEL_TS_SEEDS12
            sel = json.load(open(_repo(source)))
        p = sel["pipelines"][pname]
        e = p["seeds"][str(seed)]
        graphs_run, arm = e.get("graphs_run", p.get("graphs_run")), {"canto_l_hnl": "ii", "canto_l_tthnl": "iii"}[model]
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
    return list(TS_SEEDS) if model in ("canto_l_hnl", "canto_l_tthnl") else [0, 1, 2]


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
                                                                             ("iii", TS_SEEDS)) for s in seeds}}
    path = _repo(a.out)
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
    return s, first, _write_frames(frames, root, s, first)


def _write_frames(frames: dict, root: str, s: str, first: int) -> dict:
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
    return out


def _convert_from_shards(args):
    """The tables of _convert_shard for the listed events of one 500-number shard, read through hnl.data.shards (the
    events' tar members only)."""
    s, directory, info, first, events, root = args
    from hnl.data import convert as CV
    from hnl.data import shards
    frames = {"hits": [], "particles": [], "edges": []}
    got = []
    for i, ev in shards.iter_events_from(directory, events, check=True):
        for k, v in CV.event_tables(ev, s, i, info=info).items():
            frames[k].append(v)
        got.append(i)
    if got != sorted(int(e) for e in events):
        raise RuntimeError(f"{directory}: read {len(got)} of {len(events)} events of shard {first}")
    return s, first, _write_frames(frames, root, s, first)


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


def cmd_prepare_inputs(a) -> None:
    """Event tables and feature-store tars of the tar-sharded inputs of T2 (truth only; no model): the extension's test
    events (splits/generic_iso_x1_test.txt, checked against the hash rule and the union list) and every event of the HNL
    points (splits/hnl_<m>_<ctau>_all.txt). Only the listed events' members are read. Tables go to
    derived/events/v1/<table>/<sample>/shard_<first>.parquet (extension: sample generic_iso, shards 08000-23500, next
    to the T1 shards 00000-07500, which are not touched); tars to FS/<name>."""
    _require_generated()
    from hnl.data import acorn_store as AS
    from hnl.data import convert as CV
    from hnl.data import shards
    root = os.path.join(DATA, "derived", "events", "v1")
    want = {"x1": [("generic_iso", "x1_test")], "points": [(p, "all") for p in POINTS_ORDER]}
    keys = want["x1"] + want["points"] if a.inputs == "all" else want[a.inputs]
    specs = shard_inputs()
    meta = {"convert": [], "tars": {}, "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    jobs, lists = [], {}
    for key in keys:
        directory, info, _ = specs[key]
        events = split_events(*key)
        have = shards.event_indices_from(directory)
        missing = np.setdiff1d(events, have)
        if len(missing):
            raise SystemExit(f"{directory}: {len(missing)} listed events are not in the shards ({missing[:5]})")
        lists[key] = events
        for first in np.unique(events // CV.SHARD * CV.SHARD):
            if os.path.exists(os.path.join(root, "edges", key[0], f"shard_{first:05d}.parquet")):
                continue
            ev = events[(events >= first) & (events < first + CV.SHARD)]
            jobs.append((key[0], directory, info, int(first), [int(e) for e in ev], root))
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        for s, first, out in ex.map(_convert_from_shards, jobs):
            meta["convert"].append({"sample": s, "first": first, **out})
            print(f"[prepare-inputs] {s} shard {first}: {out}", flush=True)
    os.makedirs(FS, exist_ok=True)
    for key in keys:
        name = specs[key][2]
        if os.path.exists(os.path.join(FS, name)):
            continue
        meta["tars"][name] = AS.build_tar(key[0], key[1], FS, events=lists[key], name=name)
        print(f"[prepare-inputs] {name}: {meta['tars'][name]}", flush=True)
    with open(os.path.join(FS, f"prepare_inputs_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as f:
        json.dump(meta, f, indent=1)


def _frame_diff(a: pd.DataFrame, b: pd.DataFrame) -> str:
    """'' if the two frames are identical (columns, dtypes, values, NaN in the same places), else a description."""
    if list(a.columns) != list(b.columns):
        return f"columns {list(a.columns)} != {list(b.columns)}"
    if not a.dtypes.equals(b.dtypes):
        return "dtypes differ"
    if len(a) != len(b):
        return f"{len(a)} != {len(b)} rows"
    if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
        cols = [c for c in a.columns if not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True))]
        return f"values differ in {cols}"
    return ""


def cmd_shard_identity(a) -> None:
    """Reader identity (results/new_samples_sufficiency_v1.md 6.1, at the level of the prepared tables): the first N
    events of the generic_iso val list are packed, as the exact bytes of their npz files, into one scratch tar
    (events_<first>_<last>.tar, members <event>.npz); they are then converted (a) through hnl.data.npz from the files and
    (b) through hnl.data.shards from the tar, each into its own scratch table root, by the production functions
    (_convert_shard's writer, hnl.data.convert.event_tables), and turned into feature-store graphs
    (hnl.data.acorn_store.build_tar). Compared: member bytes against file bytes, the arrays, the three event tables
    (exact frame equality) of (a), (b) and the production tables that the val chain read (derived/events/v1), every
    graph field of (b) against (a) (same host: the reader identity) and against the production val tar
    (FS/generic_iso_val.tar; the hit eta feature, a float32 arcsinh of the table values, can differ in the last bit
    between hosts, so the maximal difference per field is recorded). Writes results/addendum_g_t2_inputs_v1.json. No
    model runs."""
    _require_generated()
    import io
    import torch
    from hnl.data import acorn_store as AS
    from hnl.data import convert as CV
    from hnl.data import npz, shards
    from hnl.data.convert import read_table
    ev_all = split_events("generic_iso", "val")[:a.n]
    first, last = int(ev_all.min()), int(ev_all.max())
    pack = os.path.join(a.scratch, "shards")
    os.makedirs(pack, exist_ok=True)
    tar_path = os.path.join(pack, shards.shard_name(first, last))
    res = {"events": len(ev_all), "first": first, "last": last, "git": _git(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "scratch": a.scratch}
    bytes_same = 0
    with tarfile.open(tar_path + ".tmp", "w", format=tarfile.USTAR_FORMAT) as tf:
        for e in ev_all:
            data = open(npz.event_path("generic_iso", int(e)), "rb").read()
            ti = tarfile.TarInfo(f"{int(e)}.npz")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    os.replace(tar_path + ".tmp", tar_path)
    shards.list_shards.cache_clear()
    arrays_same = 0
    for e, ev in shards.iter_events_from(pack, ev_all):
        path = npz.event_path("generic_iso", int(e))
        member = shards._members(tar_path)[f"{int(e)}.npz"]
        with open(tar_path, "rb") as f:
            f.seek(member.offset_data)
            mb = f.read(member.size)
        bytes_same += int(hashlib.sha256(mb).digest() == hashlib.sha256(open(path, "rb").read()).digest())
        ref = npz.load_event("generic_iso", int(e))
        arrays_same += int(ref.keys() == ev.keys() and all(
            ref[k].dtype == ev[k].dtype and np.array_equal(ref[k], ev[k], equal_nan=ref[k].dtype.kind == "f")
            for k in ref))
    res["member_bytes_identical"], res["arrays_identical"] = bytes_same, arrays_same
    roots = {"npz": os.path.join(a.scratch, "tables_npz"), "shards": os.path.join(a.scratch, "tables_shards")}
    info = npz.SAMPLES["generic_iso"]
    for first_s in np.unique(ev_all // CV.SHARD * CV.SHARD):
        ev = ev_all[(ev_all >= first_s) & (ev_all < first_s + CV.SHARD)]
        frames = {"hits": [], "particles": [], "edges": []}
        for e in ev:
            for k, v in CV.event_tables(npz.load_event("generic_iso", int(e)), "generic_iso", int(e)).items():
                frames[k].append(v)
        _write_frames(frames, roots["npz"], "generic_iso", int(first_s))
        _convert_from_shards(("generic_iso", pack, info, int(first_s), [int(x) for x in ev], roots["shards"]))
    prod = os.path.join(DATA, "derived", "events", "v1")
    res["tables"] = {}
    for k in ("hits", "particles", "edges"):
        A = read_table(roots["npz"], k, "generic_iso")
        Bt = read_table(roots["shards"], k, "generic_iso")
        P = read_table(prod, k, "generic_iso", events=ev_all)
        res["tables"][k] = {"rows": len(Bt), "npz_vs_shards": _frame_diff(A, Bt) or "identical",
                            "production_vs_shards": _frame_diff(P, Bt) or "identical"}
    out = os.path.join(a.scratch, "fs")
    os.makedirs(out, exist_ok=True)
    for k in roots:
        if os.path.exists(os.path.join(out, f"generic_iso_val_{k}.tar")):
            os.remove(os.path.join(out, f"generic_iso_val_{k}.tar"))
        AS.build_tar("generic_iso", "val", out, events=ev_all, name=f"generic_iso_val_{k}.tar", root=roots[k])

    def graphs(path, names=None):
        import gzip
        out_ = {}
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if names is None or m.name in names:
                    out_[m.name] = torch.load(io.BytesIO(gzip.decompress(t.extractfile(m).read())), weights_only=False)
        return out_

    def compare(A, B):
        same, maxdiff = 0, {}
        for nm, g in A.items():
            h = B.get(nm)
            ok = h is not None and sorted(g.keys()) == sorted(h.keys())
            for key in (g.keys() if ok else []):
                x, y = g[key], h[key]
                if torch.is_tensor(x):
                    eq = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
                    if not eq and x.shape == y.shape and x.dtype.is_floating_point:
                        maxdiff[key] = max(maxdiff.get(key, 0.0), float((x.double() - y.double()).abs().max()))
                else:
                    eq = x == y
                ok &= bool(eq)
            same += int(ok)
        return same, maxdiff
    g_sh = graphs(os.path.join(out, "generic_iso_val_shards.tar"))
    g_np = graphs(os.path.join(out, "generic_iso_val_npz.tar"))
    g_pr = graphs(os.path.join(FS, "generic_iso_val.tar"), set(g_sh))
    s1, d1 = compare(g_sh, g_np)
    s2, d2 = compare(g_sh, g_pr)
    res["graphs"] = {"built": len(g_sh), "host": os.uname().nodename, "shards_vs_npz_identical": s1,
                     "shards_vs_npz_max_abs_diff": d1, "shards_vs_production_val_tar_identical": s2,
                     "shards_vs_production_max_abs_diff": d2,
                     "production_val_tar_host": "the prepare job 6810986 (submit37)"}
    res["verdict"] = ("identical" if bytes_same == arrays_same == s1 == len(ev_all) and all(
        v["npz_vs_shards"] == v["production_vs_shards"] == "identical" for v in res["tables"].values()) else "DIFFERENT")
    path = os.path.join(REPO, "results", "addendum_g_t2_inputs_v1.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.setdefault("shard_identity", {})[os.uname().nodename.split(".")[0]] = res
    with open(path, "w") as f:
        json.dump(old, f, indent=1)
    print(json.dumps(res, indent=1))
    if res["verdict"] != "identical":
        raise SystemExit("shard path differs from the npz path")


def cmd_stage(a) -> None:
    """Extract the feature-store events of a set into DEST (flat), as acorn's input split directory; check that exactly
    the events of the set's lists were staged."""
    os.makedirs(a.dest, exist_ok=True)
    want = set()
    for s, sp in SETS[a.set]:
        want |= {f"event{e}-graph.pyg.gz" for e in event_ids(s, split_events(s, sp))}
    got = []
    for s, sp in SETS[a.set]:
        for name in fs_tars(s, sp):
            with tarfile.open(os.path.join(FS, name)) as t:
                t.extractall(a.dest, filter="data")
                got += t.getnames()
    if sorted(got) != sorted(want):
        raise SystemExit(f"[stage] {a.set}: staged {len(got)} graphs, the set's lists hold {len(want)}")
    print(f"[stage] {a.set}: {len(got)} events -> {a.dest}")


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


def _complete(d: str) -> list[str]:
    """The block (or part) directories <k>_of_<n> under d, if they form one complete, finished set."""
    blocks = sorted(glob.glob(os.path.join(d, "*_of_*")))
    n = {int(os.path.basename(b).split("_of_")[1]) for b in blocks}
    if len(n) != 1 or len(blocks) != n.pop() or not all(os.path.exists(os.path.join(b, "done")) for b in blocks):
        raise SystemExit(f"{d}: incomplete blocks {blocks}")
    return blocks


def cmd_gather_union(a) -> None:
    """Extract the union blocks of (model, seed, set) into DEST: all of them, or, with --part k --n-parts n, part k of
    the blocks split into n contiguous parts (numpy.array_split, one Canto-L scoring job each). The blocks must be
    complete; together they must hold exactly the events of the acorn graphs they were built from (a part: a subset)."""
    s = canto_settings(a.model, a.seed)
    d = os.path.join(OUT, f"{a.model}_s{a.seed}", a.set, "union", "blocks")
    blocks = _complete(d)
    if a.n_parts > 1 or a.part:
        blocks = [blocks[i] for i in np.array_split(np.arange(len(blocks)), a.n_parts)[a.part]]
    src = _members(os.path.join(OUT, f"acorn_{s['acorn_arm']}_s{a.seed}", a.set, "testset.tar"))
    got = []
    os.makedirs(a.dest, exist_ok=True)
    for b in blocks:
        with tarfile.open(os.path.join(b, "testset.tar")) as t:
            got += [m for m in t.getnames() if m.endswith(".pyg.gz")]
            t.extractall(a.dest, filter="data")
    if len(set(got)) != len(got) or (sorted(got) != src if a.n_parts == 1 else not set(got) <= set(src)):
        raise SystemExit(f"union blocks hold {len(got)} graphs, the acorn source {len(src)}")
    print(f"[gather-union] {len(got)} union graphs from {len(blocks)} blocks (part {a.part} of {a.n_parts}) -> {a.dest}")


def cmd_gather_scored(a) -> None:
    """Pack the scored parts of (model, seed, set) (canto_score.sbatch array tasks, scored/parts/<k>_of_<n>) into
    scored/testset.tar, members in sorted order, after checking that the parts are complete and together hold every
    event of the acorn source graphs exactly once; copy the parts' inference records; write scored/done. With
    --prune-union, then remove the union blocks' graph tars (their reports and done markers stay): every union graph is
    in the scored tar with its scores."""
    s = canto_settings(a.model, a.seed)
    d = os.path.join(OUT, f"{a.model}_s{a.seed}", a.set, "scored")
    if os.path.exists(os.path.join(d, "done")):
        raise SystemExit(f"{d} is done: refusing to repeat it")
    parts = _complete(os.path.join(d, "parts"))
    src = _members(os.path.join(OUT, f"acorn_{s['acorn_arm']}_s{a.seed}", a.set, "testset.tar"))
    where = {}
    for p in parts:
        for m in _members(os.path.join(p, "testset.tar")):
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
        json.dump({"parts": [os.path.relpath(p, d) for p in parts], "n_graphs": len(src), "git": _git(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID"),
                   "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, f, indent=1)
    open(os.path.join(d, "done"), "w").close()
    for p in parts:                                   # the part tars are now in testset.tar
        os.remove(os.path.join(p, "testset.tar"))
    print(f"[gather-scored] {len(src)} scored graphs from {len(parts)} parts -> {d}/testset.tar")
    if a.prune_union:
        blocks = _complete(os.path.join(OUT, f"{a.model}_s{a.seed}", a.set, "union", "blocks"))
        for b in blocks:
            if os.path.exists(os.path.join(b, "testset.tar")):
                os.remove(os.path.join(b, "testset.tar"))
                open(os.path.join(b, "pruned"), "w").close()
        print(f"[gather-scored] removed the graph tars of {len(blocks)} union blocks")


# ------------------------------------------------------------------------------------------------- evaluation
def _truth(samples) -> pd.DataFrame:
    """Per-particle truth of the samples: T1 tables (generic_iso with the extension's T1 table appended when it exists:
    events 8000-23999), the HNL points' tables of hnl.eval.addendum_g_truth."""
    from hnl.data.acorn_store import SAMPLE_CODE
    out = []
    for s in samples:
        if s in SPLIT_STEM:
            t = pd.read_parquet(os.path.join(T2, f"truth_{s}.parquet"))
        else:
            t = pd.read_parquet(os.path.join(T1, f"truth_{s}.parquet"))
            if s == "generic_iso" and os.path.exists(X1_TRUTH):
                x = pd.read_parquet(X1_TRUTH)
                if x.event.min() < X1_FIRST or list(x.columns) != list(t.columns):
                    raise SystemExit(f"{X1_TRUTH}: events below {X1_FIRST} or columns unlike the T1 table")
                t = pd.concat([t, x], ignore_index=True)
        t["sample"] = s
        t["event_id"] = np.int64(SAMPLE_CODE[s]) * 100_000_000 + t.event.to_numpy(np.int64)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def keep_ids(set_: str) -> dict:
    """event_id (str) -> sorted particle ids written to the per-particle tables: targets and gun particles (HNL points:
    targets, N daughters and W muons)."""
    t = _truth([s for s, _ in SETS[set_]])
    k = t.is_target.to_numpy(bool).copy()
    if "gun" in t:
        k |= t.gun.fillna(False).to_numpy(bool)
    if "category" in t:
        k |= t.category.isin(["hnl", "w"]).to_numpy(bool)
    t = t[k]
    return {str(e): np.sort(g.particle_id.to_numpy(np.int64)) for e, g in t.groupby("event_id")}


def _tv_chunk(paths: list[str], cfg: tuple, keep: dict, labels: dict):
    """hnl.eval.trackbuild_val._chunk for one configuration, keeping each event's track labels in `labels`."""
    from hnl.eval import trackbuild_val as TV
    layers = geometry = None
    eids = [int(TV._eid_of(p)) for p in paths]
    if cfg[0] in TV.GEO_KINDS:
        from hnl.geometry.resolution import hit_geometry
        geometry = hit_geometry(eids)
        layers = {e: g["layer_id"] for e, g in geometry.items()}
    elif cfg[0] in TV.LAYER_KINDS:
        from hnl.trackbuild.mutex_ws import hit_layers
        layers = hit_layers(eids)
    res = [TV.evaluate_event(p, [cfg], {}, keep, layers, geometry, labels_out=labels) for p in paths]
    return pd.concat([r[0] for r in res], ignore_index=True), pd.concat([r[1] for r in res], ignore_index=True)


def tracks_frame(labels: dict) -> pd.DataFrame:
    """The stored tracks: one row per hit on a track (label >= 0): event_id, row (graph node = hits-table row), track."""
    out = []
    for (eid, _), lab in sorted(labels.items(), key=lambda kv: int(kv[0][0])):
        lab = np.asarray(lab)
        if len(lab) and lab.max() >= 2 ** 31:
            raise ValueError(f"event {eid}: track label beyond int32")
        rows = np.flatnonzero(lab >= 0)
        out.append(pd.DataFrame({"event_id": np.int64(eid), "row": rows.astype(np.int32),
                                 "track": lab[rows].astype(np.int32)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        {"event_id": np.zeros(0, np.int64), "row": np.zeros(0, np.int32), "track": np.zeros(0, np.int32)})


def _eval_chunk(args):
    paths, model, cfg, keep = args
    from hnl.eval import evaluate_gnn as EG
    labels = {}
    if cfg[0] == "cc":                     # hnl.eval.evaluate_gnn at the cut, as the frozen eval_test
        res = [EG.evaluate_event(p, [cfg[1]], {}, keep, labels_out=labels) for p in paths]
        part = pd.concat([r[0][["event_id", "particle_id", "dm_matched", "n_sm_tracks"]] for r in res], ignore_index=True)
        ev = pd.concat([r[1][["event_id", "n_tracks_ge5", "n_fake"]] for r in res], ignore_index=True)
        ceil = [r[2] for r in res]
    else:
        ceil = [EG.evaluate_event(p, [], {}, keep)[2] for p in paths]
        part, ev = _tv_chunk(paths, cfg, keep, labels)
        part = part[["event_id", "particle_id", "dm_matched", "n_sm_tracks"]]
        ev = ev[["event_id", "n_tracks_ge5", "n_fake"]]
    if len(labels) != len(paths):
        raise RuntimeError(f"track labels of {len(labels)} of {len(paths)} events")
    ceil = pd.concat(ceil, ignore_index=True)
    part = part.merge(ceil, on=["event_id", "particle_id"], how="left", validate="one_to_one")
    return part, ev, tracks_frame(labels)


def cmd_evaluate(a) -> None:
    _require_generated()
    s = model_settings(a.model, a.seed)
    tar = scored_tar(a.model, a.seed, a.set)
    out = os.path.join(a.eval_root, f"{a.model}_s{a.seed}", a.set)
    blk = os.path.join(out, "blocks", f"{a.block:03d}_of_{a.n_blocks:03d}")
    if os.path.exists(os.path.join(blk, "done")):
        raise SystemExit(f"{blk} is done: refusing to repeat it")
    if a.set != "val" and not os.path.exists(os.path.join(os.path.dirname(tar), "done")):
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
    pd.concat([r[2] for r in res], ignore_index=True).to_parquet(os.path.join(blk, "tracks.parquet"), index=False)
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
    """Concatenate the blocks of one (model, seed, set) in block order; check they cover every scored graph once. The
    stored tracks stay in the blocks (blocks/*/tracks.parquet)."""
    out = os.path.join(a.eval_root, f"{a.model}_s{a.seed}", a.set)
    blocks = sorted(glob.glob(os.path.join(out, "blocks", "*")))
    n = {int(os.path.basename(b).split("_of_")[1]) for b in blocks}
    if len(n) != 1 or len(blocks) != n.pop():
        raise SystemExit(f"{out}: blocks {blocks} do not form one complete set")
    if not all(os.path.exists(os.path.join(b, "done")) for b in blocks):
        raise SystemExit(f"{out}: unfinished blocks")
    if a.set != "val" and not all(os.path.exists(os.path.join(b, "tracks.parquet")) for b in blocks):
        raise SystemExit(f"{out}: blocks without stored tracks")
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


def cmd_dry_outputs(a) -> None:
    """Code test only: placeholder evaluation outputs of every model and seed of a set, made from truth alone, under
    --eval-root (refused for the production evaluation root), so that the summary can run on a set whose real outputs
    must not exist yet. Each kept particle is matched (and inside the graph limit) with a fixed model-dependent
    probability drawn from numpy default_rng([seed, model index, sample code]); one track with >= 5 hits per event, a
    fake one with probability 0.01. No model is run and no model output is read."""
    if os.path.realpath(a.eval_root) == os.path.realpath(os.path.join(OUT, "eval")):
        raise SystemExit("dry outputs never go to the production evaluation root")
    from hnl.data.acorn_store import SAMPLE_CODE
    keep = keep_ids(a.set)
    eids = np.concatenate([event_ids(s, split_events(s, sp)) for s, sp in SETS[a.set]])
    for mi, m in enumerate(LABELS):
        for seed in model_seeds(m):
            d = os.path.join(a.eval_root, f"{m}_s{seed}", a.set)
            os.makedirs(d, exist_ok=True)
            pp = []
            for s, _ in SETS[a.set]:
                code = SAMPLE_CODE[s]
                ev = [e for e in eids if e // 100_000_000 == code]
                ids = [keep.get(str(e), np.zeros(0, np.int64)) for e in ev]
                n = np.array([len(x) for x in ids])
                rng = np.random.default_rng([seed, mi, code])
                ok = rng.random(int(n.sum())) < 0.90 + 0.01 * mi
                pp.append(pd.DataFrame({"event_id": np.repeat(np.asarray(ev, np.int64), n),
                                        "particle_id": np.concatenate(ids) if len(ids) else np.zeros(0, np.int64),
                                        "dm_matched": ok, "n_sm_tracks": ok.astype(np.int16), "dm_ceiling": True,
                                        "dm_ceiling_samepid": True}))
            pd.concat(pp, ignore_index=True).to_parquet(os.path.join(d, "per_particle.parquet"), index=False)
            rng = np.random.default_rng([seed, mi, 0])
            pe = pd.DataFrame({"event_id": eids.astype(np.int64), "n_tracks_ge5": 1,
                               "n_fake": (rng.random(len(eids)) < 0.01).astype(int)})
            pe.to_parquet(os.path.join(d, "per_event.parquet"), index=False)
            pp = pd.concat(pp, ignore_index=True)       # placeholder label-exchange read-out: 2 % fewer matches
            lx = pp.dm_matched.to_numpy() & (np.random.default_rng([seed, mi, 1]).random(len(pp)) > 0.02)
            pd.DataFrame({"event_id": pp.event_id, "particle_id": pp.particle_id, "dm_lx": lx}).to_parquet(
                os.path.join(d, "per_particle_lx.parquet"), index=False)
            pd.DataFrame({"event_id": pe.event_id, "n_tracks_ge5_lx": 1, "n_fake_lx": pe.n_fake}).to_parquet(
                os.path.join(d, "per_event_lx.parquet"), index=False)
            for name in ("provenance.json", "lx.json"):
                with open(os.path.join(d, name), "w") as f:
                    json.dump({"dry_outputs": True, "git": _git(), "hits": 1, "hits_exchanged": 0}, f)
    print(f"[dry-outputs] {a.set}: {len(eids)} events, {sum(len(v) for v in keep.values())} kept particles -> "
          f"{a.eval_root}")


def cmd_val_check(a) -> None:
    """The T2 code on the val outputs that already exist: (1) the val summary written by this code (default evaluation
    outputs) must reproduce the committed results/addendum_g_val_v1.{csv,_bins.csv,_differences.csv} byte for byte and
    its .md except the "Written ..." line; (2) the val evaluation re-run with stored tracks (--eval-root ROOT) must
    reproduce the stored per-particle and per-event tables of every model and seed exactly, and so its summary too.
    Writes the val_check section of results/addendum_g_t2_inputs_v1.json."""
    from hnl.eval import addendum_g_summary as S
    res = os.path.join(REPO, "results")
    out = {"git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "eval_root_tracks": a.eval_root}

    def compare(d: str) -> dict:
        r = {}
        for suf in (".csv", "_bins.csv", "_differences.csv", ".md"):
            new_, old_ = (open(os.path.join(x, f"addendum_g_val_v1{suf}")).read() for x in (d, res))
            if suf == ".md":
                new_, old_ = ("\n".join(ln for ln in t.split("\n") if not ln.startswith("Written "))
                              for t in (new_, old_))
            r[suf] = "identical" if new_ == old_ else "DIFFERENT"
        return r
    for name, root in (("summary_default_outputs", None), ("summary_tracks_outputs", a.eval_root)):
        d = os.path.join(a.scratch, name)
        S.summary("val", out_dir=d, eval_root=root)
        out[name] = compare(d)
    tabs = {}
    for m in LABELS:
        for seed in model_seeds(m):
            k = f"{m}_s{seed}"
            r = {}
            for t in ("per_particle", "per_event"):
                A = pd.read_parquet(os.path.join(OUT, "eval", k, "val", f"{t}.parquet"))
                Bq = pd.read_parquet(os.path.join(a.eval_root, k, "val", f"{t}.parquet"))
                key = ["event_id", "particle_id"] if t == "per_particle" else ["event_id"]
                A, Bq = (x.sort_values(key).reset_index(drop=True) for x in (A, Bq))
                r[t] = _frame_diff(A, Bq) or "identical"
            r["tracks_rows"] = int(sum(len(pd.read_parquet(f, columns=["event_id"])) for f in glob.glob(
                os.path.join(a.eval_root, k, "val", "blocks", "*", "tracks.parquet"))))
            tabs[k] = r
    out["re_evaluation"] = tabs
    ok = all(v == "identical" for n in ("summary_default_outputs", "summary_tracks_outputs") for v in out[n].values()) \
        and all(r[t] == "identical" for r in tabs.values() for t in ("per_particle", "per_event"))
    out["verdict"] = "identical" if ok else "DIFFERENT"
    path = os.path.join(res, "addendum_g_t2_inputs_v1.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old["val_check"] = out
    with open(path, "w") as f:
        json.dump(old, f, indent=1)
    print(json.dumps(out, indent=1))
    if not ok:
        raise SystemExit("the T2 code does not reproduce the val outputs")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p = sub.add_parser("prepare-inputs")
    p.add_argument("--inputs", required=True, choices=["x1", "points", "all"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p = sub.add_parser("shard-identity")
    p.add_argument("--scratch", required=True)
    p.add_argument("--n", type=int, default=500)
    p = sub.add_parser("dry-outputs")
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--eval-root", required=True)
    p = sub.add_parser("val-check")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--scratch", required=True)
    p = sub.add_parser("models")
    p.add_argument("--out", default="results/addendum_g_models_v1.json")
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
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--n-parts", type=int, default=1)
    p = sub.add_parser("gather-scored")
    p.add_argument("--model", required=True, choices=[m for m in LABELS if m.startswith("canto")])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--prune-union", action="store_true")
    p = sub.add_parser("stage")
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--dest", required=True)
    for name in ("evaluate", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=list(LABELS))
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--set", required=True, choices=list(SETS))
        p.add_argument("--eval-root", default=os.path.join(OUT, "eval"))
        if name == "evaluate":
            p.add_argument("--block", type=int, default=0)
            p.add_argument("--n-blocks", type=int, default=1)
            p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p = sub.add_parser("summary")
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--eval-root", default=None)
    p.add_argument("--out-dir", default=None)
    sub.add_parser("val-sanity")
    a = ap.parse_args(argv)
    if a.cmd == "val-sanity":
        from hnl.eval import addendum_g_summary as S
        S.val_sanity()
        return
    if a.cmd == "summary":
        from hnl.eval import addendum_g_summary as S
        S.summary(a.set, out_dir=a.out_dir, eval_root=a.eval_root)
        return
    {"prepare": cmd_prepare, "prepare-inputs": cmd_prepare_inputs, "shard-identity": cmd_shard_identity,
     "models": cmd_models, "acorn-args": cmd_acorn_args, "canto-args": cmd_canto_args, "stage": cmd_stage,
     "evaluate": cmd_evaluate, "merge": cmd_merge, "block-tar": cmd_block_tar, "gather-union": cmd_gather_union,
     "gather-scored": cmd_gather_scored, "val-check": cmd_val_check, "dry-outputs": cmd_dry_outputs}[a.cmd](a)


if __name__ == "__main__":
    main()
