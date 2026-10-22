"""Canto-L trained on HNL (arm ii) and on ttbar + HNL (arm iii), seeds 1 and 2 of the training-sample study
(docs/training_sample_study.md section 9), on the test sets of seed 0's post-freeze read-out (docs/feedback/
test_use_ledger.md R11), so that both models have three seeds on test like Canto-L (ttbar). Post-freeze, descriptive,
no claim (ledger R19); nothing is chosen on test.

  select   validation only. Per new seed the inputs of slurm/stage1/final_test_shard.sbatch and final_test_merge.sbatch,
           by seed 0's rules (DECISIONS.md 2026-10-12 22:24 and 22:38 UTC): the FR* configuration (builder rule at FR*
           with the one-standard-error tie rule; hnl.eval.final_select.run_rows over trackbuild_val_ext,
           trackbuild_val_lmerge, trackbuild_val, family mwslmerge, with the guard baseline of
           hnl.eval.training_sample_board and hnl.eval.training_sample_3seed), the LLP-blind configuration of the same
           family (section-6c rule on ttbar val, the same call), the classifier's e* cut (hnl.eval.final_select.cut_e)
           and the run's merge model. The same calls must reproduce seed 0's entries of
           results/training_sample_test_selection_v1.json, and the FR* configurations of seeds 1 and 2 must be those of
           results/training_sample_val_3seed_v1.csv; the classifier checkpoint (best checkpoint of the run) must be the
           one that scored the run's val graphs. Every test output of seeds 1 and 2 goes under OUT_ROOT (the hnl_lrt file
           budget): the selection's "run" and "graphs_run" are mirror directories there whose inputs are symbolic links
           to the trained runs ("source_run", "graphs_source_run"; read only), made by slurm/ts_seeds12/chain.sh.
  readout  hnl.eval.training_sample_test's rows (primary efficiency per test sample and pooled; Wang et al.'s
           definitions on hnl15_100_external; V0 daughters and photon-conversion e+-) for Canto-L (HNL) and Canto-L
           (ttbar+HNL) with three seeds each (results/training_sample_spec_test_3seed_v1.json: seed 0 the R11 outputs,
           seeds 1 and 2 the outputs of R19), per seed with that module's own row functions (event bootstrap alone) and
           as the three-seed mean with the sigma of paper Eq. (2), sigma^2 = sigma_boot^2 + s_seeds^2 / 3
           (hnl.eval.bootstrap.seed_mean; hnl.eval.second_population.summary); the preregistered test tables (B = 500,
           seed 12345). Checks, all before anything is written: spec entries against the two selection files and the
           runs' provenance (final_test.done, final_selection_used.json, the single configuration of
           RUN/trackbuild_test); the frozen Canto-L identity of hnl.eval.training_sample_test (before any output of a
           seed-1 or seed-2 run is read); seed 0's per-seed rows reproduce every row of
           results/training_sample_test_v1.csv; each three-seed value is the mean of its per-seed values. Reads only
           RUN/trackbuild_test of the spec runs (and the frozen Canto-L's), never eval_test, trackbuild_test_llpblind
           or edge_tpr_test*. Writes <out>/<name>.csv, .md, _meta.json (refuses to overwrite).

    python -m hnl.eval.training_sample_seeds12 select --out results/training_sample_test_selection_seeds12_v1.json
    python -m hnl.eval.training_sample_seeds12 spec --out results/training_sample_spec_test_3seed_v1.json
    python -m hnl.eval.training_sample_seeds12 readout --out results --name training_sample_test_seeds12_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import per_sample as PSM
from hnl.eval import published_defs as PD
from hnl.eval import second_population as SP
from hnl.eval import training_sample_test as TST
from hnl.eval.particles import particle_table

REPO = BT.REPO
RUNS = os.path.join(BT.DATA, "runs")
OUT_ROOT = "/work/submit/jinboz1/hnl/runs_extra/ts_seeds12"
SEL_V1 = "results/training_sample_test_selection_v1.json"
SEL_NEW = "results/training_sample_test_selection_seeds12_v1.json"
SPEC_3 = "results/training_sample_spec_test_3seed_v1.json"
REF_V1 = "results/training_sample_test_v1.csv"
VAL_3SEED = "results/training_sample_val_3seed_v1.csv"
WP = "results/working_points_v1.json"
FAMILY = "mwslmerge"
ARMS = {"ii": ("l1tl_ii", "Canto-L (HNL)", "arm (ii), HNL"),
        "iii": ("l1tl_iii", "Canto-L (ttbar+HNL)", "arm (iii), ttbar+HNL")}
NEW_SEEDS = ("1", "2")
SEEDS = ("0",) + NEW_SEEDS
SUB = TST.SUB
UNCERTAINTY = ("three seeds per model: value = mean of the seeds, sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (paper "
               f"Eq. (2); sigma_boot = event-bootstrap sd of the seed mean, {TST.TABLES}); per-seed rows: the event "
               "bootstrap alone (hnl.eval.training_sample_test)")


def _repo(p: str) -> str:
    return BT._repo(p)


def _sha(p: str) -> str:
    return PSM.sha256(p)


# --------------------------------------------------------------------------------------------------------- select
def _val_record_ckpt(run: str) -> dict:
    fs = sorted(glob.glob(os.path.join(run, "gnn_scores", "infer_*.json")))
    if len(fs) != 1:
        raise SystemExit(f"{run}/gnn_scores: {len(fs)} val inference records")
    return json.load(open(fs[0]))


def cmd_select(a) -> None:
    from hnl.eval import final_select as FS
    from hnl.eval import training_sample_board as TSB
    from hnl.graphs.helix_union import best_ckpt
    if os.path.exists(a.out) and not a.check_only:
        raise SystemExit(f"{a.out} exists: refusing to overwrite")
    wpj = json.load(open(_repo(WP)))
    FS.matching_baseline = lambda run: os.path.join(TSB.R, "baseline_v1", "armi_s0")   # the board's guard baseline
    v1 = json.load(open(_repo(SEL_V1)))
    val3 = pd.read_csv(_repo(VAL_3SEED))
    checks: list = []
    pipelines = {}
    for arm, (pname, label, sample) in ARMS.items():
        seeds_out = {}
        for s in SEEDS:
            src = os.path.join(RUNS, "stage1_final_learned_ts", "l1tl", f"arm{arm}_s{s}")
            t, _ = FS.run_rows(src, TSB.GRIDS, wpj, wpj["reference_arm"])
            r = t[t.family == FAMILY].iloc[0]
            cfg, llp, ce = str(r.config), str(r.llp_blind_config), FS.cut_e(src, wpj)
            wp_txt = f"{FAMILY} {cfg.split(',', 1)[1]} ({r.grid})"
            print(f"{label} s{s}: FR* {cfg} ({r.grid}); LLP-blind {llp}; e* cut {ce}", flush=True)
            if s == "0":
                ref = v1["pipelines"][pname]["seeds"]["0"]
                TST.check(checks, f"{label} s0: run = {SEL_V1}", src, ref["run"], how="str")
                TST.check(checks, f"{label} s0: FR* configuration = {SEL_V1}", cfg, ref["config"], how="str")
                TST.check(checks, f"{label} s0: LLP-blind configuration = {SEL_V1}", llp, ref["llp_blind_config"],
                          how="str")
                TST.check(checks, f"{label} s0: e* cut = {SEL_V1}", ce, float(ref["cut_e"]), tol=0.0)
                TST.check(checks, f"{label} s0: merge model = {SEL_V1}", os.path.join(src, "lmerge", "model.joblib"),
                          ref["merge_model"], how="str")
                continue
            v = val3[(val3.family == FAMILY) & (val3.model == label) & (val3.seed == int(s))]
            TST.check(checks, f"{label} s{s}: one row in {VAL_3SEED}", len(v), 1, tol=0.0)
            TST.check(checks, f"{label} s{s}: FR* working point = {VAL_3SEED}", wp_txt, v.working_point.iloc[0],
                      how="str")
            TST.check(checks, f"{label} s{s}: run = {VAL_3SEED}", src, v.run.iloc[0], how="str")
            gnn = best_ckpt(src, "gnn")
            rec = _val_record_ckpt(src)
            gsha = _sha(gnn)
            TST.check(checks, f"{label} s{s}: best checkpoint = val scores' checkpoint", gnn, rec["ckpt"], how="str")
            TST.check(checks, f"{label} s{s}: checkpoint sha256 = val record", gsha, rec["ckpt_sha256"], how="str")
            gsrc = os.path.join(RUNS, "baseline_v1_ts", f"arm{arm}_s{s}")
            ml_run = os.path.join(RUNS, "baseline_v1", f"arm{arm}_s{s}")
            ub = json.load(open(os.path.join(gsrc, "ltf_union_graphs", "report_val", "build.json")))
            TST.check(checks, f"{label} s{s}: union ML checkpoint = baseline best ML checkpoint",
                      os.path.realpath(ub["ml_ckpt"]), os.path.realpath(best_ckpt(ml_run, "ml")), how="str")
            TST.check(checks, f"{label} s{s}: union triplet model is the run's", ub["triplet_model"],
                      os.path.join(gsrc, "learned_triplet_filter_v2", "model.joblib"), how="str")
            TST.check(checks, f"{label} s{s}: acorn one-time test done", int(os.path.exists(
                os.path.join(ml_run, "test_eval.done"))), 1, tol=0.0)
            merge = os.path.join(src, "lmerge", "model.joblib")
            seeds_out[s] = {
                "run": os.path.join(OUT_ROOT, "l1tl", f"arm{arm}_s{s}"), "source_run": src,
                "graphs_run": os.path.join(OUT_ROOT, "baseline_v1_ts", f"arm{arm}_s{s}"), "graphs_source_run": gsrc,
                "ml_run": ml_run, "test_graphs_source": os.path.join(ml_run, "gnn_scores_test", "testset.tar"),
                "cut_e": ce, "merge_model": merge, "merge_model_sha256": _sha(merge), "config": cfg,
                "llp_blind_config": llp, "val_working_point": wp_txt, "gnn_ckpt": gnn, "gnn_ckpt_sha256": gsha,
                "triplet_model": ub["triplet_model"], "triplet_model_sha256": _sha(ub["triplet_model"]),
                "tau": ub["tau"], "k": ub["k"], "r_wide": ub["r_wide"], "nsig": ub["nsig"], "pt_min": ub["pt_min"],
                "union_ml_ckpt": ub["ml_ckpt"]}
        pipelines[pname] = {"dir": os.path.join(OUT_ROOT, "l1tl"), "graphs": "ltf_union_graphs",
                            "training_sample": sample, "family": FAMILY, "seeds": seeds_out}
    TST.stop_if_failed(checks, "selection checks")
    out = {"created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git": PSM.git_state(),
           "purpose": "Post-freeze, descriptive test of seeds 1 and 2 of the training-sample study runs Canto-L (HNL) "
                      "and Canto-L (ttbar+HNL) (the first author's request, 2026-10-17; docs/feedback/test_use_ledger.md "
                      "R19). Not a selection: results/final_selection_v1.json and the claims are unchanged. Validation "
                      "settings only; input of slurm/stage1/final_test_shard.sbatch and final_test_merge.sbatch "
                      "(pipelines l1tl_ii / l1tl_iii, seeds 1 and 2); seed 0 stays in "
                      "results/training_sample_test_selection_v1.json.",
           "working_points": WP, "working_points_sha256": _sha(_repo(WP)), "FR_star": wpj["FR_star"],
           "e_star": wpj["e_star"], "reference_arm": wpj["reference_arm"],
           "grid": "trackbuild_val_ext,trackbuild_val_lmerge,trackbuild_val",
           "rules": "family mwslmerge; per run on val: FR* configuration by the builder rule with the one-standard-error "
                    "tie rule and LLP-blind configuration by the section-6c rule (hnl.eval.final_select.run_rows, as "
                    "hnl.eval.training_sample_board and training_sample_3seed), e* cut from RUN/edge_tpr_estar_efpr.json "
                    "(final_select.cut_e); seed 0 reproduces results/training_sample_test_selection_v1.json",
           "out_root": OUT_ROOT,
           "outputs": "run and graphs_run are mirror directories under out_root (symbolic links to source_run's gnn and "
                      "lmerge, and to graphs_source_run / ml_run inputs); every test output is written there",
           "candidate": None, "pipelines": pipelines,
           "checks": f"{len(checks)} of {len(checks)} ok", "check_list": [[c[0], str(c[1]), str(c[2]), c[3]] for c in checks]}
    if a.check_only:
        print(json.dumps({k: v for k, v in out.items() if k != "check_list"}, indent=1))
        return
    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}")


def cmd_spec(a) -> None:
    """The read-out spec of the two three-seed models (format of results/training_sample_spec_test_v1.json) from the
    selection files: seed 0 from SEL_V1, seeds 1 and 2 from SEL_NEW; RUN/trackbuild_test, configuration 0."""
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: refusing to overwrite")
    sels = {"0": json.load(open(_repo(SEL_V1))), "new": json.load(open(_repo(SEL_NEW)))}
    spec = []
    for pname, label, _ in ARMS.values():
        ents = []
        for s in SEEDS:
            d = sels["0" if s == "0" else "new"]["pipelines"][pname]["seeds"][s]
            fam, params = d["config"].split(",", 1)
            ents.append({"run": d["run"], "sub": SUB, "cut": 0, "wp": f"{fam} {params} ({SUB}, configuration 0)"})
        spec.append({"label": f"{label}: learned triplet union, learned merging (training-sample study, three seeds, "
                              "post-freeze)", "short": label, "group": "Canto (training sample)", "entries": ents})
    with open(a.out, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    print(json.dumps(spec, indent=1))


# -------------------------------------------------------------------------------------------------------- readout
def load_models(checks: list) -> list[dict]:
    """The three-seed models of the spec, every entry checked against its selection file (seed 0: SEL_V1, seeds 1
    and 2: SEL_NEW): run, sub trackbuild_test, configuration index 0, working-point text of the FR* configuration."""
    spec = json.load(open(_repo(SPEC_3)))
    sels = {SEL_V1: json.load(open(_repo(SEL_V1))), SEL_NEW: json.load(open(_repo(SEL_NEW)))}
    by_short = {lab: arm for arm, (_, lab, _) in ARMS.items()}
    if sorted(m["short"] for m in spec) != sorted(by_short):
        raise SystemExit(f"{SPEC_3}: models {[m['short'] for m in spec]}")
    models = []
    for m in spec:
        pname = ARMS[by_short[m["short"]]][0]
        if len(m["entries"]) != len(SEEDS):
            raise SystemExit(f"{m['label']}: {len(m['entries'])} entries, expected {len(SEEDS)}")
        ents = []
        for s, e in zip(SEEDS, m["entries"]):
            sp = SEL_V1 if s == "0" else SEL_NEW
            d = sels[sp]["pipelines"][pname]["seeds"][s]
            fam, params = d["config"].split(",", 1)
            w = f"spec {m['short']} s{s}"
            TST.check(checks, f"{w}: run = {sp}", os.path.normpath(e["run"]), os.path.normpath(d["run"]), how="str")
            TST.check(checks, f"{w}: sub", e.get("sub"), SUB, how="str")
            TST.check(checks, f"{w}: configuration index", float(e.get("cut", np.nan)), 0.0, tol=0.0)
            TST.check(checks, f"{w}: working point = selection FR* configuration", e.get("wp"),
                      f"{fam} {params} ({SUB}, configuration 0)", how="str")
            ents.append({"run": os.path.normpath(e["run"]), "sub": SUB, "cut": 0.0, "wp": e["wp"], "config": d["config"],
                         "seed": s, "selection": sp})
        models.append({"label": m["label"], "short": m["short"], "group": m.get("group", ""), "entries": ents})
    return models


def provenance(models: list[dict], checks: list) -> None:
    """Each run was evaluated once by final_test.sbatch / final_test_merge.sbatch with its committed selection file."""
    sels = {SEL_V1: json.load(open(_repo(SEL_V1))), SEL_NEW: json.load(open(_repo(SEL_NEW)))}
    for m in models:
        for e in m["entries"]:
            w = f"{m['short']} s{e['seed']}"
            run = e["run"]
            TST.check(checks, f"{w}: final_test.done", int(os.path.exists(os.path.join(run, "final_test.done"))), 1,
                      tol=0.0)
            used = os.path.join(run, "final_selection_used.json")
            same = os.path.exists(used) and json.load(open(used)) == sels[e["selection"]]
            TST.check(checks, f"{w}: final_selection_used.json = {e['selection']}", int(same), 1, tol=0.0)
            cp = os.path.join(run, SUB, "configs.json")
            cfgs = json.load(open(cp)) if os.path.exists(cp) else []
            TST.check(checks, f"{w}: one configuration in {SUB}", len(cfgs), 1, tol=0.0)
            TST.check(checks, f"{w}: configuration 0 = selection", ",".join(str(x) for x in cfgs[0]) if cfgs else "",
                      e["config"], how="str")


def one_seed_models(models: list[dict]) -> list[dict]:
    """One one-seed model per (model, seed), for hnl.eval.training_sample_test's row functions."""
    return [{"label": f"{m['label']} [seed {e['seed']}]", "short": f"{m['short']} s{e['seed']}", "group": m["group"],
             "entries": [e], "parent": m["short"], "seed": e["seed"]} for m in models for e in m["entries"]]


def check_seed0(rows: pd.DataFrame, checks: list) -> None:
    """Seed 0's one-seed rows = every row of results/training_sample_test_v1.csv of the same model (R11)."""
    ref = pd.read_csv(_repo(REF_V1), float_precision="round_trip")
    key = ["block", "set", "population", "quantity"]
    ref[key] = ref[key].fillna("")                 # empty cells (population of the published-definition rows)
    rows = rows.copy()
    rows[key] = rows[key].fillna("")
    for short in (lab for _, lab, _ in ARMS.values()):
        r = ref[ref.model == short].set_index(key).sort_index()
        mine = rows[rows.model == f"{short} s0"].set_index(key).sort_index()
        TST.check(checks, f"{short} s0: rows = {REF_V1}", list(mine.index), list(r.index), how="str")
        for k in r.index:
            if k not in mine.index:
                continue
            for col in ("value", "sigma", "sigma_boot", "boot68_lo", "boot68_hi", "cp68_lo", "cp68_hi"):
                TST.check(checks, f"{short} s0 {'/'.join(map(str, k))}: {col}", mine.loc[k, col], r.loc[k, col])
            for col in ("k", "n", "B", "events", "units"):
                TST.check(checks, f"{short} s0 {'/'.join(map(str, k))}: {col}", mine.loc[k, col], r.loc[k, col],
                          tol=0.0)
            TST.check(checks, f"{short} s0 {'/'.join(map(str, k))}: working point", mine.loc[k, "working_point"],
                      r.loc[k, "working_point"], how="str")


def _base3(block: str, m: dict, st: str, population: str, quantity: str) -> dict:
    return {"block": block, "seed": "mean", "set": st, "population": population, "quantity": quantity,
            "model": m["short"], "label": m["label"], "group": m["group"],
            "run": ";".join(e["run"] for e in m["entries"]), "working_point": "; ".join(e["wp"] for e in m["entries"]),
            "n_seeds": len(m["entries"])}


def _sm3(sm: dict, per_seed) -> dict:
    if sm["n_seeds"] != len(SEEDS):
        raise SystemExit(f"not a three-seed estimate: {sm}")
    return {"value": sm["mean"], "sigma": sm["sigma"], "sigma_boot": sm["sigma_boot"], "seed_sd": sm["s_seeds"],
            "per_seed": ";".join(f"{x:.10g}" for x in per_seed)}


def three_seed_rows(models: list[dict], res: dict, units: dict, pdo: dict, sps: dict) -> list[dict]:
    rows = []
    for m in models:
        for st in PSM.TEST_SETS:
            x = res[(m["label"], st)]
            sm = BS.seed_mean(x["pts"], x["reps"])
            ks, ns = [int(round(float(k))) for k in x["k"]], [int(n) for n in x["n"]]
            joint = "; joint units of hnl10_100_test and hnl15_200_test" if st in ("hnl10_100_test",
                                                                                   "hnl15_200_test") else ""
            rows.append({**_base3("per_sample", m, st, "primary", "eff_primary"), "events": PSM.EVENTS[st],
                         "units": units[st], "k": sum(ks), "n": sum(ns), "n_per_seed": ns[0],
                         **_sm3(sm, x["pts"]), "B": len(x["reps"][0]), "definition": TST.PRIMARY_DEF,
                         "tables": TST.TABLES + joint})
        for q in TST.PD_QS:
            x = pdo[(m["label"], q)]
            c = x["counts"]
            rows.append({**_base3("published_definitions", m, TST.PD_SET, "", q), "events": x["events"],
                         "k": int(c["k_pooled"]), "n": int(c["n_pooled"]), "n_per_seed": int(c["per_seed_n"][0]),
                         **_sm3(x["sig"], c["per_seed_eff"]), "B": BS.B,
                         "definition": PD.POPS[q][1] if q in PD.POPS else TST.PD_FAKE_DEF, "tables": TST.TABLES})
        sm_ = sps[m["label"]]
        for k in TST.sp_keys(sm_):
            s = SP.summary(sm_, k)
            if not TST.sp_kept(k, s["n_per_seed"]):
                continue
            if s["n_seeds"] != len(SEEDS):
                raise SystemExit(f"{m['label']} {k}: not a three-seed estimate")
            ev_set, popn, q = k
            rows.append({**_base3("second_population", m, ev_set, popn, q), "k": s["k_pooled"], "n": s["n_pooled"],
                         "n_per_seed": s["n_per_seed"], "value": s["value"], "sigma": s["combined"],
                         "sigma_boot": s["boot_sd"], "seed_sd": s["seed_sd"], "per_seed": s["per_seed"],
                         "B": sm_.seeds[0][k][1].shape[0], "boot68_lo": s["boot68_lo"], "boot68_hi": s["boot68_hi"],
                         "cp68_lo": s["cp68_lo"], "cp68_hi": s["cp68_hi"],
                         "definition": f"{SP.POP_LABEL[popn]}; {SP.Q_LABEL[q]}", "tables": TST.TABLES})
    return rows


def check_means(df: pd.DataFrame, checks: list) -> None:
    """Each three-seed value is the mean of the seeds' values (per-seed rows of the same block, set, quantity)."""
    df = df.copy()
    df["population"] = df.population.fillna("")
    for r in df[df.seed == "mean"].itertuples():
        g = df[(df.seed != "mean") & (df.parent == r.model) & (df.block == r.block) & (df.set == r.set)
               & (df.population == r.population) & (df.quantity == r.quantity)]
        w = f"{r.model} mean {r.block}/{r.set}/{r.population}/{r.quantity}"
        TST.check(checks, f"{w}: three per-seed rows", len(g), len(SEEDS), tol=0.0)
        if len(g):
            TST.check(checks, f"{w}: value = mean of the seeds", r.value, float(g.value.mean()), tol=1e-12)
            TST.check(checks, f"{w}: k = sum of the seeds", r.k, float(g.k.sum()), tol=0.0)


COLUMNS = ["block", "seed", "parent", "set", "population", "quantity", "model", "label", "group", "run", "working_point",
           "n_seeds", "events", "units", "k", "n", "n_per_seed", "value", "sigma", "sigma_boot", "seed_sd", "per_seed",
           "B", "boot68_lo", "boot68_hi", "cp68_lo", "cp68_hi", "definition", "tables"]


def _cell1(r) -> str:
    return f"{100 * r.value:.2f} ± {100 * r.sigma:.2f} ({int(r.k)}/{int(r.n)})"


def _cell3(r) -> str:
    return f"{100 * r.value:.2f} ± {100 * r.sigma:.2f} (sd {100 * r.seed_sd:.2f})"


def markdown(df: pd.DataFrame, meta: dict) -> str:
    shorts = [lab for _, lab, _ in ARMS.values()]
    L = ["# Test read-out of Canto-L trained on HNL and on ttbar+HNL with three seeds (seeds 1 and 2 added; per "
         "sample, Wang et al.'s definitions, second population)", "",
         f"Generated by `python -m hnl.eval.training_sample_seeds12 readout` (git {meta['git']}; Slurm job "
         f"{meta['slurm_job']}; {meta['finished_utc']}). Post-freeze, descriptive (docs/feedback/test_use_ledger.md "
         f"R19; seed 0 is the R11 read-out): the configurations of seeds 1 and 2 were fixed on validation in "
         f"{SEL_NEW} before their test jobs, seed 0's in {SEL_V1}; nothing chosen on test; no claim; "
         "results/claims_test_v1.json and results/final_selection_v1.json unchanged. Canto-L (HNL) and Canto-L "
         "(ttbar+HNL) are the paper's Canto (HNL) and Canto (ttbar+HNL).", "",
         "- Models and per-seed working points: " + "; ".join(
             f"{s} ({w})" for s, w in dict(zip(df[df.seed == 'mean'].model, df[df.seed == 'mean'].working_point)).items())
         + ".",
         f"- Uncertainty: {UNCERTAINTY}. Three-seed cells: value ± sigma (seed sd); per-seed cells: value ± sigma_boot "
         "(k/n). Percent.",
         "- Pooled HNL test rows with the |d0| and r_v bins, W muons, ttbar prompt and fake rate: "
         "results/bootstrap_ts_test_3seed_v1.* (hnl.eval.bootstrap_tables --extra-test-spec); paired differences: "
         "results/training_sample_paired_test_3seed_v1.*.",
         f"- Checks: {meta['checks']['spec_and_provenance']} (spec, selections, provenance); "
         f"{meta['checks']['identity']} (frozen Canto-L identity of hnl.eval.training_sample_test); "
         f"{meta['checks']['seed0']} (seed 0 against {REF_V1}); {meta['checks']['means']} (three-seed means).", ""]
    ps = df[df.block == "per_sample"]
    L += ["## Primary efficiency per test sample", "",
          "| model | seed | " + " | ".join(PSM.SET_TITLE[s] for s in PSM.TEST_SETS) + " |",
          "|---|---|" + "---|" * len(PSM.TEST_SETS)]
    for short in shorts:
        for s in SEEDS + ("mean",):
            g = ps[(ps.model == (short if s == "mean" else f"{short} s{s}"))].set_index("set")
            cell = _cell3 if s == "mean" else _cell1
            L.append(f"| {short} | {'3 seeds' if s == 'mean' else s} | "
                     + " | ".join(cell(g.loc[st]) for st in PSM.TEST_SETS) + " |")
    pdf = df[(df.block == "published_definitions") & (df.seed == "mean")]
    L += ["", f"## Wang et al.'s definitions on hnl15_100_external ({TST.PD_SET}), three seeds", "",
          "| model | " + " | ".join(TST.PD_QS) + " |", "|---|" + "---|" * len(TST.PD_QS)]
    for short in shorts:
        g = pdf[pdf.model == short].set_index("quantity")
        L.append(f"| {short} | " + " | ".join(_cell3(g.loc[q]) for q in TST.PD_QS) + " |")
    spr = df[(df.block == "second_population") & (df.seed == "mean") & (df.quantity.isin(["all", "r_ge100"]))]
    combos = list(dict.fromkeys(zip(spr.set, spr.population, spr.quantity)))
    L += ["", "## Second population (V0 daughters, photon-conversion e±), three seeds: all and r_start >= 100 mm", "",
          "| set | population | quantity | " + " | ".join(shorts) + " |", "|---|---|---|" + "---|" * len(shorts)]
    for s, p, q in combos:
        cells = []
        for short in shorts:
            r = spr[(spr.model == short) & (spr.set == s) & (spr.population == p) & (spr.quantity == q)]
            cells.append(_cell3(r.iloc[0]) if len(r) else "—")
        L.append(f"| {SP.SET_LABEL[s]} | {SP.POP_LABEL[p]} | {SP.Q_LABEL[q]} | " + " | ".join(cells) + " |")
    L += ["", f"Every row (per seed and three-seed; all start-radius and |d0| bins of the second population): "
          f"{meta['name']}.csv; inputs with sha256 in {meta['name']}_meta.json.", ""]
    return "\n".join(L)


def cmd_readout(a) -> None:
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    TST.refuse_existing(a.out, a.name)
    for p in (SPEC_3, SEL_V1, SEL_NEW):
        if not TST.git_committed(p):
            raise SystemExit(f"{p} is not committed as is")
    spec_checks: list = []
    models = load_models(spec_checks)
    frozen = TST.frozen_canto(spec_checks)
    TST.stop_if_failed(spec_checks, "spec and frozen entries")

    id_checks: list = []                       # identity of the code path, before any seed-1 / seed-2 output is read
    events = SP.expected_events("test")
    info = particle_table(events)
    bs = BS.Bootstrap()
    pop = SP.population_table(events)
    res_f, units = TST.read_per_sample([frozen])
    TST.check_per_sample(res_f, units, id_checks)
    pd_f = TST.read_published([frozen], info, bs)
    TST.check_published(pd_f, id_checks)
    sp_f = TST.read_second_population([frozen], events, pop)
    TST.check_second_population(sp_f[TST.CANTO], id_checks)
    TST.stop_if_failed(id_checks, "identity (frozen Canto-L)")

    prov: list = []
    provenance(models, prov)
    TST.stop_if_failed(prov, "provenance of the spec runs")

    ones = one_seed_models(models)
    res1, u1 = TST.read_per_sample(ones)
    pd1 = TST.read_published(ones, info, bs)
    sp1 = TST.read_second_population(ones, events, pop)
    res3, u3 = TST.read_per_sample(models)
    pd3 = TST.read_published(models, info, bs)
    sp3 = TST.read_second_population(models, events, pop)
    if u1 != units or u3 != units:
        raise SystemExit(f"resampling units differ: {u1}, {u3} vs {units}")
    parent = {o["short"]: (o["parent"], o["seed"]) for o in ones}
    r1 = pd.DataFrame(TST.per_sample_rows(ones, res1, u1) + TST.published_rows(ones, pd1)
                      + TST.second_population_rows(ones, sp1))
    r1["parent"] = [parent[x][0] for x in r1.model]
    r1["seed"] = [parent[x][1] for x in r1.model]
    s0_checks: list = []
    check_seed0(r1, s0_checks)
    TST.stop_if_failed(s0_checks, f"seed 0 against {REF_V1}")
    r3 = pd.DataFrame(three_seed_rows(models, res3, u3, pd3, sp3))
    r3["parent"] = r3.model
    df = pd.concat([r1, r3], ignore_index=True).reindex(columns=COLUMNS)
    mean_checks: list = []
    check_means(df, mean_checks)
    TST.stop_if_failed(mean_checks, "three-seed means")

    meta = {"name": a.name, "git": PSM.git_state(), "slurm_job": os.environ.get("SLURM_JOB_ID", "local"),
            "host": os.uname().nodename, "started_utc": started,
            "spec": {"path": SPEC_3, "sha256": _sha(_repo(SPEC_3))},
            "selections": {p: _sha(_repo(p)) for p in (SEL_V1, SEL_NEW)},
            "status": "post-freeze, descriptive test read-out (docs/feedback/test_use_ledger.md R19; seed 0 = R11); "
                      "configurations fixed on validation; nothing chosen on test; no claim",
            "uncertainty": UNCERTAINTY,
            "bootstrap_tables": {"dir": BS.DEFAULT_DIR, "B": BS.B, "seed": BS.SEED},
            "models": [{"label": m["label"], "short": m["short"], "entries": m["entries"]} for m in models],
            "frozen_reference": frozen,
            "checks": {"spec_and_provenance": f"{len(spec_checks) + len(prov)} of {len(spec_checks) + len(prov)} ok",
                       "identity": f"{len(id_checks)} of {len(id_checks)} ok",
                       "seed0": f"{len(s0_checks)} of {len(s0_checks)} ok",
                       "means": f"{len(mean_checks)} of {len(mean_checks)} ok"},
            "check_list": [[c[0], str(c[1]), str(c[2]), c[3]] for c in spec_checks + id_checks + prov + s0_checks
                           + mean_checks],
            "rows": {f"{b} {s}": int(((df.block == b) & (df.seed == s)).sum()) for b in dict.fromkeys(df.block)
                     for s in SEEDS + ("mean",)}}
    files = [_repo(p) for p in (SPEC_3, SEL_V1, SEL_NEW, REF_V1)] + TST.input_files(models, frozen, SPEC_3, SEL_NEW)
    meta["inputs"] = {p: _sha(p) for p in dict.fromkeys(files)}
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = markdown(df, meta)
    TST.refuse_existing(a.out, a.name)
    os.makedirs(a.out, exist_ok=True)
    csv_p, md_p, meta_p = TST.output_paths(a.out, a.name)
    for path, write in ((csv_p, lambda f: df.to_csv(f, index=False)), (md_p, lambda f: f.write(md)),
                        (meta_p, lambda f: json.dump(meta, f, indent=1, default=str))):
        with open(path + ".tmp", "w") as f:
            write(f)
        os.replace(path + ".tmp", path)
    print(md)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("inputs", "check_list")}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", default=_repo(SEL_NEW))
    p.add_argument("--check-only", action="store_true", help="checks and the would-be file on stdout; nothing written")
    p = sub.add_parser("spec")
    p.add_argument("--out", default=_repo(SPEC_3))
    p = sub.add_parser("readout")
    p.add_argument("--out", default=_repo("results"))
    p.add_argument("--name", default="training_sample_test_seeds12_v1")
    a = ap.parse_args(argv)
    {"select": cmd_select, "spec": cmd_spec, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    main()
