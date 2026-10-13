"""One-seed test read-out of Canto-L trained on HNL and on ttbar+HNL (the training-sample study runs; post-freeze,
descriptive; DECISIONS.md 2026-10-12 22:24 UTC, docs/feedback/test_use_ledger.md R11).

The pooled HNL test rows of these runs come from `hnl.eval.bootstrap_tables --extra-test-spec` (results/
bootstrap_ts_test_v1.*). This module gives the other test rows of the release card for the models of a spec in the
format of results/ranking/spec_test_v1.json (one seed per model; RUN/trackbuild_test, configuration 0, written by
slurm/stage1/final_test.sbatch from results/training_sample_test_selection_v1.json):

  per_sample              primary efficiency (HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm) on
                          hnl10_100_test, hnl15_200_test, hnl15_100_external and pooled: the test code path of
                          hnl.eval.per_sample (test_readout: TS.RunTest, TS.efficiency with the primary mask; the
                          per-set efficiency of hnl.eval.strong_baselines.per_set), as results/per_sample_v1.csv;
  published_definitions   Wang et al.'s definitions on their evaluation sample (set test_external = hnl15_100_external:
                          all_track, prompt, displaced, the check populations and the fake rate):
                          hnl.eval.published_defs.seed_counts / summarize with TS.efficiency / TS.event_ratio, as
                          results/published_definitions_comparison.csv;
  second_population       V0 daughters (hnl_joint_test and its halves) and photon-conversion e+- (ttbar_test,
                          hnl15_100_external): hnl.eval.second_population.evaluate / summary, the efficiency rows that
                          results/second_population_v2.csv keeps (main sets; halves for all and r_start >= 100 mm).

Uncertainty: one seed per model, so the event bootstrap alone (sigma = sigma_boot; no seed term; seed_sd empty: the
seed spread of a one-seed model is unknown, not zero), with the preregistered test tables (hnl.eval.bootstrap,
B = 500, seed 12345) that every existing test read-out uses.

Identity check (always, also with --check-only; the run stops before it reads any output of the spec's runs if one
check fails): the same calls on the frozen Canto-L (results/ranking/spec_test_v1.json, three seeds, configuration 0 of
the frozen RUN/trackbuild_test = results/final_selection_v1.json) reproduce its test rows in results/per_sample_v1.csv
and results/published_definitions_comparison.csv (|difference| <= 1e-12, counts exact), in results/second_population_v2.csv
(the file keeps %.10g: every value equal at that precision, counts exact) and in the full-precision
runs/feedback_v3/second_population_v2/test_rows.csv (1e-12). The spec is checked against the committed selection file
(run, configuration, working-point text); both files must be committed and unmodified. Before the read-out of a spec
run: RUN/final_test.done, RUN/final_selection_used.json = the selection file, RUN/trackbuild_test/configs.json = its
single configuration.

Reads: RUN/trackbuild_test (per_particle, per_event, configs.json) of the spec's runs and of the frozen Canto-L, the
provenance files above, the truth tables of the test events, the preregistered bootstrap tables and the reference result
files. No other output of the runs (eval_test, trackbuild_test_llpblind, edge_tpr_test*) is read. Writes only
<out>/<name>.csv, <name>.md and <name>_meta.json, and refuses if any of them exists. Nothing is chosen here.

    python -m hnl.eval.training_sample_test --spec results/training_sample_spec_test_v1.json --out results \
        --name training_sample_test_v1 [--check-only]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import per_sample as PSM
from hnl.eval import published_defs as PD
from hnl.eval import second_population as SP
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table

REPO = BT.REPO
DATA = BT.DATA
TOL = 1e-12
SUB = "trackbuild_test"
SPEC = "results/training_sample_spec_test_v1.json"
SELECTION_TS = "results/training_sample_test_selection_v1.json"
SELECTION = "results/final_selection_v1.json"
FROZEN_SPEC = "results/ranking/spec_test_v1.json"
FROZEN_LABEL = "Canto-L (full budget): learned triplet union, learned merging (rule R6)"
FROZEN_SEEDS = ("0", "1", "2")
CANTO = PSM.CANTO                                   # "Canto-L (frozen: learned merging)", label in all three files
PD_CANTO = "canto_l"                                # its model key in published_definitions_comparison.csv
REF_PS = "results/per_sample_v1.csv"
REF_PD = "results/published_definitions_comparison.csv"
REF_SP = "results/second_population_v2.csv"
REF_SP_FULL = os.path.join(DATA, "runs", "feedback_v3", "second_population_v2", "test_rows.csv")
PD_SET = "test_external"
PD_QS = tuple(PD.POPS) + ("fake_rate",)
PD_FAKE_DEF = "n_fake / n_tracks_ge5 over the set"
PRIMARY_DEF = "HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm (primary population)"
SP_FLOATS = ("value", "seed_sd", "boot_sd", "combined", "boot68_lo", "boot68_hi", "cp68_lo", "cp68_hi")
SP_INTS = ("n_per_seed", "k_pooled", "n_pooled", "n_seeds")
TABLES = f"preregistered test tables (hnl.eval.bootstrap, B {BS.B}, seed {BS.SEED})"
UNCERTAINTY = (f"one seed per model: sigma = event-bootstrap sd alone ({TABLES}; the same tables as every existing "
               "test read-out), no seed term; seed_sd is left empty (the seed spread of a one-seed model is unknown, "
               "not zero)")
COLUMNS = ["block", "set", "population", "quantity", "model", "label", "group", "run", "working_point", "n_seeds",
           "events", "units", "k", "n", "value", "sigma", "sigma_boot", "seed_sd", "B", "boot68_lo", "boot68_hi",
           "cp68_lo", "cp68_hi", "definition", "tables"]


def _repo(p: str) -> str:
    return BT._repo(p)


# ---------------------------------------------------------------------------------------------------------- checks
def _close(mine, ref, tol: float = TOL) -> bool:
    """Numbers equal within tol; NaN equals NaN (empty cells of the reference files)."""
    a, b = float(mine), float(ref)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return abs(a - b) <= tol


def _g10(x) -> str:
    """A number as the %.10g CSV files of this repository write it (NaN: an empty cell)."""
    x = float(x)
    return "" if np.isnan(x) else f"{x:.10g}"


def check(checks: list, what: str, mine, ref, tol: float | None = TOL, how: str = "num") -> bool:
    """Append (what, mine, ref, ok). how: 'num' (|diff| <= tol, NaN = NaN), 'g10' (equal at %.10g), 'str'."""
    if how == "str":
        ok = str(mine) == str(ref)
    elif how == "g10":
        ok = _g10(mine) == _g10(ref)
    else:
        ok = _close(mine, ref, tol)
    checks.append((what, mine, ref, bool(ok)))
    return bool(ok)


def stop_if_failed(checks: list, block: str) -> None:
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:40]:
            print("FAIL", c[0], "| mine", c[1], "| reference", c[2])
        raise SystemExit(f"{block}: {len(bad)} of {len(checks)} checks failed; nothing written")
    print(f"{block}: {len(checks)} of {len(checks)} checks ok", flush=True)


def git_committed(path: str) -> bool:
    """Tracked by git and unmodified against HEAD (the guard of slurm/stage1/final_test.sbatch)."""
    rel = os.path.relpath(os.path.abspath(_repo(path)), REPO)
    run = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True).returncode == 0
    return run("ls-files", "--error-unmatch", "--", rel) and run("diff", "--quiet", "HEAD", "--", rel)


def output_paths(out: str, name: str) -> list[str]:
    return [os.path.join(out, f"{name}{ext}") for ext in (".csv", ".md", "_meta.json")]


def refuse_existing(out: str, name: str) -> None:
    hit = [p for p in output_paths(out, name) if os.path.exists(p)]
    if hit:
        raise SystemExit(f"output exists, refusing to overwrite: {hit}")


# ----------------------------------------------------------------------------------------------------------- models
def load_models(spec_path: str, sel_path: str, checks: list) -> list[dict]:
    """The spec's models (one seed each), every entry checked against the committed selection file: run listed there,
    sub trackbuild_test, configuration index 0, working-point text '<family> <params> (trackbuild_test,
    configuration 0)' of the selection's FR* configuration."""
    spec = json.load(open(_repo(spec_path)))
    sel = json.load(open(_repo(sel_path)))
    by_run = {os.path.normpath(d["run"]): d for p in sel["pipelines"].values() for d in p["seeds"].values()}
    models, labels = [], set()
    for m in spec:
        if m["label"] in labels or m["label"] == CANTO:
            raise SystemExit(f"{spec_path}: label {m['label']!r} repeated or reserved")
        labels.add(m["label"])
        if len(m["entries"]) != 1:
            raise SystemExit(f"{m['label']}: {len(m['entries'])} entries; this read-out is for one-seed models")
        ents = []
        for e in m["entries"]:
            run = os.path.normpath(e["run"])
            d = by_run.get(run)
            if d is None:
                raise SystemExit(f"{m['label']}: {run} is not a run of {sel_path}")
            fam, params = d["config"].split(",", 1)
            check(checks, f"spec {m['short']}: sub", e.get("sub"), SUB, how="str")
            check(checks, f"spec {m['short']}: configuration index", float(e.get("cut", np.nan)), 0.0, tol=0.0)
            check(checks, f"spec {m['short']}: working point = selection FR* configuration", e.get("wp"),
                  f"{fam} {params} ({SUB}, configuration 0)", how="str")
            ents.append({"run": run, "sub": SUB, "cut": 0.0, "wp": e["wp"], "config": d["config"]})
        models.append({"label": m["label"], "short": m["short"], "group": m.get("group", ""), "entries": ents})
    return models


def frozen_canto(checks: list) -> dict:
    """The frozen Canto-L's three test entries (results/ranking/spec_test_v1.json), checked against
    results/final_selection_v1.json and configuration 0 of the stored RUN/trackbuild_test."""
    spec = {m["label"]: m for m in json.load(open(_repo(FROZEN_SPEC)))}
    sel = json.load(open(_repo(SELECTION)))["pipelines"]["l1tl"]["seeds"]
    ents = []
    for s, e in zip(FROZEN_SEEDS, spec[FROZEN_LABEL]["entries"]):
        check(checks, f"frozen Canto-L s{s}: run = selection", os.path.normpath(e["run"]),
              os.path.normpath(sel[s]["run"]), how="str")
        cfgs = json.load(open(os.path.join(e["run"], SUB, "configs.json")))
        check(checks, f"frozen Canto-L s{s}: one configuration in {SUB}", len(cfgs), 1, tol=0.0)
        check(checks, f"frozen Canto-L s{s}: configuration 0 = selection", ",".join(str(x) for x in cfgs[0]),
              sel[s]["config"], how="str")
        ents.append({"run": os.path.normpath(e["run"]), "sub": SUB, "cut": 0.0, "wp": e["wp"],
                     "config": sel[s]["config"]})
    return {"label": CANTO, "short": "Canto-L", "group": "Canto", "entries": ents}


def provenance(models: list[dict], sel_path: str, checks: list) -> None:
    """Each spec run was evaluated once by final_test.sbatch with the committed selection file, at its configuration."""
    sel = json.load(open(_repo(sel_path)))
    for m in models:
        for e in m["entries"]:
            run = e["run"]
            check(checks, f"{m['short']}: final_test.done", int(os.path.exists(os.path.join(run, "final_test.done"))),
                  1, tol=0.0)
            used = os.path.join(run, "final_selection_used.json")
            same = os.path.exists(used) and json.load(open(used)) == sel
            check(checks, f"{m['short']}: final_selection_used.json = {sel_path}", int(same), 1, tol=0.0)
            cp = os.path.join(run, SUB, "configs.json")
            cfgs = json.load(open(cp)) if os.path.exists(cp) else []
            check(checks, f"{m['short']}: one configuration in {SUB}", len(cfgs), 1, tol=0.0)
            check(checks, f"{m['short']}: configuration 0 = selection", ",".join(str(x) for x in cfgs[0]) if cfgs
                  else "", e["config"], how="str")


# ------------------------------------------------------------------------------------------------------- read-outs
def read_per_sample(models: list[dict]) -> tuple[dict, dict]:
    """hnl.eval.per_sample.test_readout on {label: entries}: per model and test set points, replicates, k, n."""
    return PSM.test_readout({m["label"]: m["entries"] for m in models})


def read_published(models: list[dict], info: pd.DataFrame, bs: BS.Bootstrap) -> dict:
    """{(label, quantity): counts (published_defs.summarize), sigma (BS.seed_mean), events} on test_external."""
    kind, sets, _ = PD.SETS[PD_SET]
    out = {}
    for m in models:
        rts = [TS.RunTest(e["run"], e["cut"], sub=e["sub"]) for e in m["entries"]]
        per_seed = [PD.seed_counts(rt, info, PD_SET) for rt in rts]
        if len({c["_events"] for c in per_seed}) != 1:
            raise SystemExit(f"{m['label']}: seeds cover different events on {PD_SET}")
        for q in PD_QS:
            with warnings.catch_warnings():               # std(ddof=1) of one seed: NaN by design
                warnings.simplefilter("ignore", RuntimeWarning)
                r = PD.summarize(per_seed, q)
            if q in PD.POPS:
                p, reps, _ = TS.efficiency(bs, rts, info, TS.masks()[PD.POPS[q][0]], sets)
            else:
                p, reps = TS.event_ratio(bs, rts, "n_fake", "n_tracks_ge5", sets)
            if abs(float(np.mean(p)) - r["mean"]) > TOL:
                raise SystemExit(f"{m['label']} {PD_SET} {q}: bootstrap point differs from the counts")
            out[(m["label"], q)] = {"counts": r, "sig": BS.seed_mean(p, reps), "events": per_seed[0]["_events"]}
    return out


def sp_models(models: list[dict]) -> list[dict]:
    """Model dicts of hnl.eval.second_population.evaluate (test: the per-particle table of RUN/trackbuild_test)."""
    out = []
    for i, m in enumerate(models):
        out.append({"key": "canto_l" if m["label"] == CANTO else f"ts_{i}", "label": m["label"], "short": m["short"],
                    "group": m["group"],
                    "entries": [{"run": e["run"], "dir": os.path.join(e["run"], e["sub"]), "cut": e["cut"],
                                 "wp": e["wp"]} for e in m["entries"]]})
    return out


def read_second_population(models: list[dict], events: np.ndarray, pop: pd.DataFrame) -> dict:
    """{label: second_population.Model} from one evaluate() call (its per-entry progress lines are not printed)."""
    rs = SP.test_resampler()
    with contextlib.redirect_stdout(io.StringIO()):
        ms = SP.evaluate("test", sp_models(models), rs, pop, events)
    return {m.label: m for m in ms}


def sp_keys(m) -> list[tuple]:
    return sorted(m.seeds[0], key=lambda k: (list(SP.EVENT_SETS["test"]).index(k[0]), SP.POPULATIONS.index(k[1]),
                                             SP.QUANTITIES.index(k[2])))


def sp_kept(k: tuple, n_per_seed: int) -> bool:
    """The rows results/second_population_v2.csv keeps (second_population.report)."""
    return (k[0] in SP.MAIN_SETS["test"] or k[2] in ("all", "r_ge100")) and n_per_seed > 0


# ------------------------------------------------------------------------------------------------- identity checks
def check_per_sample(res: dict, units: dict, checks: list) -> None:
    ref = pd.read_csv(_repo(REF_PS), float_precision="round_trip")
    ref = ref[(ref.split == "test") & (ref.label == CANTO)].set_index("set")
    for st in PSM.TEST_SETS:
        r, x = ref.loc[st], res[(CANTO, st)]
        sm = BS.seed_mean(x["pts"], x["reps"])
        w = f"per_sample Canto-L {st}"
        check(checks, f"{w}: n_per_seed", int(x["n"][0]), int(r.n_per_seed), tol=0.0)
        check(checks, f"{w}: units", units[st], int(r.units), tol=0.0)
        check(checks, f"{w}: events", PSM.EVENTS[st], int(r.events), tol=0.0)
        check(checks, f"{w}: B", len(x["reps"][0]), int(r.B), tol=0.0)
        for i, s in enumerate(FROZEN_SEEDS):
            check(checks, f"{w}: k_s{s}", int(round(float(x["k"][i]))), int(r[f"k_s{s}"]), tol=0.0)
            check(checks, f"{w}: eff_s{s}", float(x["pts"][i]), float(r[f"eff_s{s}"]))
        for mine, col in (("mean", "mean"), ("sigma_boot", "sigma_boot"), ("s_seeds", "s_seeds"), ("sigma", "sigma")):
            check(checks, f"{w}: {col}", sm[mine], float(r[col]))


def check_published(out: dict, checks: list) -> None:
    ref = pd.read_csv(_repo(REF_PD), float_precision="round_trip")
    ref = ref[(ref.set == PD_SET) & (ref.model == PD_CANTO)].set_index("quantity")
    check(checks, f"published_definitions Canto-L {PD_SET}: quantities", sorted(ref.index), sorted(PD_QS), how="str")
    for q in PD_QS:
        r, x = ref.loc[q], out[(CANTO, q)]
        c, s = x["counts"], x["sig"]
        w = f"published_definitions Canto-L {PD_SET} {q}"
        for mine, col in ((c["mean"], "mean"), (c["sd_seeds"], "sd_seeds"), (s["sigma"], "sigma_total"),
                          (s["sigma_boot"], "sigma_boot")):
            check(checks, f"{w}: {col}", mine, float(r[col]))
        check(checks, f"{w}: k_pooled", c["k_pooled"], float(r.k_pooled), tol=0.0)
        check(checks, f"{w}: n_pooled", c["n_pooled"], float(r.n_pooled), tol=0.0)
        for i in range(len(FROZEN_SEEDS)):
            check(checks, f"{w}: eff_s{i}", c["per_seed_eff"][i], float(r[f"eff_s{i}"]))
            check(checks, f"{w}: k_s{i}", c["per_seed_k"][i], float(r[f"k_s{i}"]), tol=0.0)
            check(checks, f"{w}: n_s{i}", c["per_seed_n"][i], float(r[f"n_s{i}"]), tol=0.0)


def check_second_population(m, checks: list) -> None:
    """Frozen Canto-L efficiency rows: results/second_population_v2.csv at %.10g (and the same row set), the
    full-precision test_rows.csv of the read-out at 1e-12."""
    key = ["event_set", "population", "quantity"]
    ref = pd.read_csv(_repo(REF_SP), float_precision="round_trip", dtype={"per_seed": str})
    ref = ref[(ref.split == "test") & (ref.kind == "efficiency") & (ref.model == CANTO)].set_index(key)
    full = pd.read_csv(REF_SP_FULL, float_precision="round_trip", dtype={"per_seed": str})
    full = full[(full.kind == "efficiency") & (full.model == CANTO)].set_index(key)
    summ = {k: SP.summary(m, k) for k in sp_keys(m)}
    kept = [k for k, s in summ.items() if sp_kept(k, s["n_per_seed"])]
    check(checks, "second_population Canto-L: rows kept = rows of the results file", sorted(kept),
          sorted(ref.index), how="str")
    check(checks, "second_population Canto-L: rows = rows of test_rows.csv", sorted(summ), sorted(full.index),
          how="str")
    for k, s in summ.items():
        w = f"second_population Canto-L {'/'.join(k)}"
        for tab, how, tag in ((ref, "g10", "results"), (full, "num", "test_rows")):
            if k not in tab.index:
                continue
            r = tab.loc[k]
            for col in SP_INTS:
                check(checks, f"{w} ({tag}): {col}", s[col], float(r[col]), tol=0.0)
            check(checks, f"{w} ({tag}): per_seed", s["per_seed"], r.per_seed, how="str")
            check(checks, f"{w} ({tag}): B", m.seeds[0][k][1].shape[0], float(r.B), tol=0.0)
            for col in SP_FLOATS:
                if col in s:
                    check(checks, f"{w} ({tag}): {col}", s[col], float(r[col]), how=how)


# ------------------------------------------------------------------------------------------------------------ rows
def _base(block: str, m: dict, st: str, population: str, quantity: str) -> dict:
    e = m["entries"][0]
    return {"block": block, "set": st, "population": population, "quantity": quantity, "model": m["short"],
            "label": m["label"], "group": m["group"], "run": e["run"], "working_point": e["wp"],
            "n_seeds": len(m["entries"])}


def _one_seed(sm: dict) -> dict:
    """sigma of a one-seed model: the event bootstrap alone (BS.seed_mean adds no seed term for one seed)."""
    if sm["n_seeds"] != 1 or not _close(sm["sigma"], sm["sigma_boot"], 0.0):
        raise SystemExit(f"not a one-seed estimate: {sm}")
    return {"value": sm["mean"], "sigma": sm["sigma"], "sigma_boot": sm["sigma_boot"], "seed_sd": np.nan}


def per_sample_rows(models: list[dict], res: dict, units: dict) -> list[dict]:
    rows = []
    for m in models:
        for st in PSM.TEST_SETS:
            x = res[(m["label"], st)]
            sm = BS.seed_mean(x["pts"], x["reps"])
            joint = "; joint units of hnl10_100_test and hnl15_200_test" if st in ("hnl10_100_test",
                                                                                   "hnl15_200_test") else ""
            rows.append({**_base("per_sample", m, st, "primary", "eff_primary"), "events": PSM.EVENTS[st],
                         "units": units[st], "k": int(round(float(x["k"][0]))), "n": int(x["n"][0]),
                         **_one_seed(sm), "B": len(x["reps"][0]), "definition": PRIMARY_DEF,
                         "tables": TABLES + joint})
    return rows


def published_rows(models: list[dict], out: dict) -> list[dict]:
    rows = []
    for m in models:
        for q in PD_QS:
            x = out[(m["label"], q)]
            c = x["counts"]
            rows.append({**_base("published_definitions", m, PD_SET, "", q), "events": x["events"],
                         "k": int(c["k_pooled"]), "n": int(c["n_pooled"]), **_one_seed(x["sig"]), "B": BS.B,
                         "definition": PD.POPS[q][1] if q in PD.POPS else PD_FAKE_DEF,
                         "tables": TABLES})
    return rows


def second_population_rows(models: list[dict], ms: dict) -> list[dict]:
    rows = []
    for m in models:
        sm = ms[m["label"]]
        for k in sp_keys(sm):
            s = SP.summary(sm, k)
            if not sp_kept(k, s["n_per_seed"]):
                continue
            if s["n_seeds"] != 1 or not _close(s["combined"], s["boot_sd"], 0.0):
                raise SystemExit(f"{m['label']} {k}: not a one-seed estimate")
            ev_set, popn, q = k
            rows.append({**_base("second_population", m, ev_set, popn, q), "k": s["k_pooled"], "n": s["n_pooled"],
                         "value": s["value"], "sigma": s["combined"], "sigma_boot": s["boot_sd"], "seed_sd": np.nan,
                         "B": sm.seeds[0][k][1].shape[0], "boot68_lo": s["boot68_lo"], "boot68_hi": s["boot68_hi"],
                         "cp68_lo": s["cp68_lo"], "cp68_hi": s["cp68_hi"],
                         "definition": f"{SP.POP_LABEL[popn]}; {SP.Q_LABEL[q]}", "tables": TABLES})
    return rows


# -------------------------------------------------------------------------------------------------------- markdown
def _cell(r) -> str:
    return f"{r.value:.4f} ± {r.sigma:.4f} ({int(r.k)}/{int(r.n)})"


def markdown(df: pd.DataFrame, meta: dict) -> str:
    models = list(dict.fromkeys(df.model))
    L = ["# Test read-out of Canto-L trained on HNL and on ttbar+HNL (one seed each; per sample, Wang et al.'s "
         "definitions, second population)", "",
         f"Generated by `python -m hnl.eval.training_sample_test` (git {meta['git']}; Slurm job {meta['slurm_job']}; "
         f"{meta['finished_utc']}). Post-freeze, descriptive (docs/feedback/test_use_ledger.md R11): configurations "
         f"fixed on validation in {meta['selection']['path']} before the test evaluation; nothing chosen on test; "
         "no claim; results/claims_test_v1.json and results/final_selection_v1.json unchanged.", "",
         f"- Models: {'; '.join(f'{m} ({w})' for m, w in dict(zip(df.model, df.working_point)).items())}.",
         f"- Uncertainty: {UNCERTAINTY}. Cells: value ± sigma (k/n).",
         "- Pooled HNL test rows of the same runs (primary, |d0| bins, W muons, ttbar prompt, fake rate): "
         "results/bootstrap_ts_test_v1.* (hnl.eval.bootstrap_tables --extra-test-spec).",
         f"- Identity check of the code path: {meta['checks']['identity']} (frozen Canto-L, three seeds, against "
         f"{REF_PS}, {REF_PD}, {REF_SP} and the full-precision second-population rows); spec and provenance checks "
         f"{meta['checks']['spec_and_provenance']}.", ""]
    ps = df[df.block == "per_sample"]
    L += ["## Primary efficiency per test sample", "",
          "| model | " + " | ".join(PSM.SET_TITLE[s] for s in PSM.TEST_SETS) + " |",
          "|---|" + "---|" * len(PSM.TEST_SETS)]
    for mod in models:
        g = ps[ps.model == mod].set_index("set")
        L.append(f"| {mod} | " + " | ".join(_cell(g.loc[s]) for s in PSM.TEST_SETS) + " |")
    pdf = df[df.block == "published_definitions"]
    L += ["", f"## Wang et al.'s definitions on hnl15_100_external ({PD_SET})", "",
          "| model | " + " | ".join(PD_QS) + " |", "|---|" + "---|" * len(PD_QS)]
    for mod in models:
        g = pdf[pdf.model == mod].set_index("quantity")
        L.append(f"| {mod} | " + " | ".join(_cell(g.loc[q]) for q in PD_QS) + " |")
    spr = df[(df.block == "second_population") & (df.quantity.isin(["all", "r_ge100"]))]
    combos = list(dict.fromkeys(zip(spr.set, spr.population, spr.quantity)))
    L += ["", "## Second population (V0 daughters, photon-conversion e±): all and r_start >= 100 mm", "",
          "| set | population | quantity | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
    for s, p, q in combos:
        cells = []
        for mod in models:
            r = spr[(spr.model == mod) & (spr.set == s) & (spr.population == p) & (spr.quantity == q)]
            cells.append(_cell(r.iloc[0]) if len(r) else "—")
        L.append(f"| {SP.SET_LABEL[s]} | {SP.POP_LABEL[p]} | {SP.Q_LABEL[q]} | " + " | ".join(cells) + " |")
    L += ["", f"Every row (all start-radius and |d0| bins of the second population included): {meta['name']}.csv; "
          f"inputs with sha256 in {meta['name']}_meta.json.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------ main
def input_files(models: list[dict], frozen: dict, spec: str, sel: str) -> list[str]:
    files = [_repo(p) for p in (spec, sel, SELECTION, FROZEN_SPEC, REF_PS, REF_PD, REF_SP)] + [REF_SP_FULL]
    for m in [frozen] + models:
        for e in m["entries"]:
            files += [os.path.join(e["run"], SUB, f) for f in ("per_particle.parquet", "per_event.parquet",
                                                               "configs.json")]
    files += [os.path.join(BS.DEFAULT_DIR, f"{s}.npz") for s in BS.SETS]
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default=SPEC, help="test spec of the one-seed models (spec_test_v1.json format)")
    ap.add_argument("--selection", default=SELECTION_TS, help="committed selection file the spec runs were tested with")
    ap.add_argument("--out", default=_repo("results"))
    ap.add_argument("--name", default="training_sample_test_v1")
    ap.add_argument("--check-only", action="store_true",
                    help="spec checks and the frozen Canto-L identity checks only: no output of the spec runs is "
                         "read and nothing is written")
    a = ap.parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not a.check_only:
        refuse_existing(a.out, a.name)
    for p in (a.spec, a.selection):
        if not git_committed(p):
            raise SystemExit(f"{p} is not committed as is")

    spec_checks: list = []
    models = load_models(a.spec, a.selection, spec_checks)
    frozen = frozen_canto(spec_checks)
    stop_if_failed(spec_checks, "spec and frozen entries")

    # identity: the same calls on the frozen Canto-L reproduce its stored test rows (before any spec-run output)
    id_checks: list = []
    events = SP.expected_events("test")
    info = particle_table(events)
    bs = BS.Bootstrap()
    pop = SP.population_table(events)
    res_f, units = read_per_sample([frozen])
    check_per_sample(res_f, units, id_checks)
    pd_f = read_published([frozen], info, bs)
    check_published(pd_f, id_checks)
    sp_f = read_second_population([frozen], events, pop)
    check_second_population(sp_f[CANTO], id_checks)
    stop_if_failed(id_checks, "identity (frozen Canto-L)")
    if a.check_only:
        print(json.dumps({"check_only": True, "spec_checks": len(spec_checks), "identity_checks": len(id_checks),
                          "models": [m["short"] for m in models], "nothing written": True}))
        return

    # the spec runs: provenance, then the read-out with the same calls
    prov: list = []
    provenance(models, a.selection, prov)
    stop_if_failed(prov, "provenance of the spec runs")
    res, units_n = read_per_sample(models)
    if units_n != units:
        raise SystemExit(f"resampling units differ: {units_n} vs {units}")
    pd_n = read_published(models, info, bs)
    sp_n = read_second_population(models, events, pop)
    df = pd.DataFrame(per_sample_rows(models, res, units_n) + published_rows(models, pd_n)
                      + second_population_rows(models, sp_n)).reindex(columns=COLUMNS)

    meta = {"name": a.name, "git": PSM.git_state(), "slurm_job": os.environ.get("SLURM_JOB_ID", "local"),
            "host": os.uname().nodename, "started_utc": started,
            "spec": {"path": a.spec, "sha256": PSM.sha256(_repo(a.spec))},
            "selection": {"path": a.selection, "sha256": PSM.sha256(_repo(a.selection))},
            "status": "post-freeze, descriptive one-time test read-out (docs/feedback/test_use_ledger.md R11); "
                      "configurations fixed on validation; nothing chosen on test; no claim",
            "uncertainty": UNCERTAINTY,
            "bootstrap_tables": {"dir": BS.DEFAULT_DIR, "B": BS.B, "seed": BS.SEED},
            "models": [{"label": m["label"], "short": m["short"], "entries": m["entries"]} for m in models],
            "frozen_reference": frozen,
            "checks": {"spec_and_provenance": f"{len(spec_checks) + len(prov)} of {len(spec_checks) + len(prov)} ok",
                       "identity": f"{len(id_checks)} of {len(id_checks)} ok"},
            "check_list": [[c[0], str(c[1]), str(c[2]), c[3]] for c in spec_checks + id_checks + prov],
            "rows": {b: int((df.block == b).sum()) for b in dict.fromkeys(df.block)}}
    meta["inputs"] = {p: PSM.sha256(p) for p in input_files(models, frozen, a.spec, a.selection)}
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = markdown(df, meta)
    refuse_existing(a.out, a.name)                    # again, just before writing
    os.makedirs(a.out, exist_ok=True)
    csv_p, md_p, meta_p = output_paths(a.out, a.name)
    for path, write in ((csv_p, lambda f: df.to_csv(f, index=False)), (md_p, lambda f: f.write(md)),
                        (meta_p, lambda f: json.dump(meta, f, indent=1, default=str))):
        with open(path + ".tmp", "w") as f:
            write(f)
        os.replace(path + ".tmp", path)
    print(md)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("inputs", "check_list")}, indent=1, default=str))


if __name__ == "__main__":
    main()
