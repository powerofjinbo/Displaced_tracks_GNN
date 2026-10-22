"""HNL-daughter efficiency in the |d0| and r_v bins on the pooled HNL test sets, for the four models of paper Fig. 5
(fig:test-samples): ledger R17 (docs/feedback/test_use_ledger.md), post-freeze, descriptive, enters no claim.

Requested by the first author on 2026-10-17 for the redesign of Fig. 5 as efficiency curves. No results file holds
these bins for the acorn baselines with learned merging on test: results/bootstrap_v1.csv has them for Canto-L and the
CC rows only, results/strong_baselines_v3.csv the 100-300 mm and >= 300 mm columns only.

Models (nothing is chosen here; every working point is read from the committed files):
  Canto-L (frozen: learned merging)   the frozen selection (results/final_selection_v1.json); test outputs
        RUN/trackbuild_test, configuration 0 (ledger R2), checked as in hnl.eval.per_sample (configuration 0 = the
        frozen selection, the test job's selection copy = results/final_selection_v1.json);
  acorn (ttbar), acorn (ttbar+HNL), acorn (HNL) + MWS-layer + learned merging (Canto-L's builder)   A2's val-fixed
        spec results/strong_baselines_v3_spec_test.json; test outputs runs/feedback_v3/strong_baselines_a2/
        arm<A>_s<S>/lmerge_test (ledger R8), checked as in hnl.eval.strong_baselines_v3 (configuration, provenance, spec
        and merge-model checksums); every test configuration is the val entry's working point.
Population and bins (docs/definitions.md section 4; hnl.eval.bootstrap_tables.masks): the HNL-daughter targets of the
pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) in the |d0| bins [0, 0.1),
[0.1, 1), [1, 3), [3, 10), [10, 30), [30, 100), [100, 300), >= 300 mm and in the r_v bins [0, 10), [10, 30),
[30, 100), [100, 200), [200, 300), >= 300 mm and "no vertex". The columns of results/strong_baselines_v3.csv (primary
population, HNL daughters, W muons, ttbar prompt, fake rate on the HNL test events) are recomputed as identity checks.
Estimators (hnl.eval.bootstrap_tables.summary, as in results/bootstrap_v1): value = mean of the three seeds; seed sd
(ddof 1); event-bootstrap sd of the seed mean on the preregistered test tables (hnl.eval.bootstrap: B = 500, seed
12345); combined = sqrt(boot sd^2 + seed sd^2 / 3) (Eq. (2) of the paper); 68 % Clopper-Pearson on the per-seed count
(mean k and n per seed); 68 % percentile interval of the bootstrap replicates of the seed mean.
Identity checks (the run stops before writing anything if one fails): per seed, the four efficiencies, the fake rate
and the 100-300 mm / >= 300 mm counts equal results/strong_baselines_v3_entries.csv; per model, the seed means and sds
equal results/strong_baselines_v3.csv, the k/n columns its k_n_d0_100 / k_n_d0_300 and (test) the primary combined and
bootstrap sigma its primary_sigma_total / primary_sigma_boot; every quantity of Canto-L equals its row of
results/bootstrap_v1.csv (value, seed sd, boot sd, combined, intervals, counts, per-seed values; that file keeps ten
significant digits); the |d0| bins, and the r_v bins with "no vertex", add up to the HNL-daughter targets for every
model and seed; all models have the same particles in every bin.

--split val is a dry run of the same code on the validation outputs (results/strong_baselines_v3_val*.csv, the val
tables of hnl.eval.bootstrap_tables, B = 1000, seed 20261005), to test the code before the test read-out; it never opens
a test output and refuses to write into results/.

    python -m hnl.eval.test_binned --split test --out results --name test_binned_v1     (slurm/feedback/test_binned.sbatch)
    python -m hnl.eval.test_binned --split val --out DIR --name val_binned_dryrun
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess

import numpy as np
import pandas as pd

from hnl.eval import bootstrap_tables as BT

REPO = BT.REPO
SEEDS = ("0", "1", "2")
TOL = 1e-12
CANTO = "Canto-L (frozen: learned merging)"
LM = "{} + MWS-layer + learned merging (Canto-L's builder)"
MODELS = ((CANTO, "Canto-L"),
          (LM.format("acorn (ttbar)"), "acorn (ttbar) LM"),
          (LM.format("acorn (ttbar+HNL)"), "acorn (ttbar+HNL) LM"),
          (LM.format("acorn (HNL)"), "acorn (HNL) LM"))
SHORT = dict(MODELS)
BOOT = "results/bootstrap_v1.csv"
BOOT_CANTO = "Canto-L (full budget): learned triplet union, learned merging (rule R6)"
SELECTION = "results/final_selection_v1.json"
FILES = {
    "test": {"spec": "results/strong_baselines_v3_spec_test.json", "entries": "results/strong_baselines_v3_entries.csv",
             "table": "results/strong_baselines_v3.csv", "boot_table": "test_release_card"},
    "val": {"entries": "results/strong_baselines_v3_val_entries.csv", "table": "results/strong_baselines_v3_val.csv",
            "boot_table": "val_ranking_v3"},
}
VAL_ENTRIES = FILES["val"]["entries"]
BIN_QS = BT.D0_BINS + BT.RV_BINS
BOOT_COLS = ("value", "seed_sd", "boot_sd", "combined", "boot68_lo", "boot68_hi", "cp68_lo", "cp68_hi",
             "cp68_pooled_lo", "cp68_pooled_hi")
BOOT_INTS = ("k_pooled", "n_pooled", "n_per_seed", "n_seeds", "B")


def _repo(p: str) -> str:
    return BT._repo(p)


def _norm(p: str) -> str:
    return os.path.normpath(p)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> str:
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    d = git("status", "--porcelain", "--", "src", "slurm")
    return git("rev-parse", "HEAD") + (f" (src/slurm: {len(d.splitlines())} changed or new files)" if d else "")


def _cmp(checks: list, what: str, mine, ref, tol: float = TOL, rel: float = 0.0) -> bool:
    if isinstance(mine, str) or isinstance(ref, str):
        ok = str(mine) == str(ref)
    else:
        a, b = float(mine), float(ref)
        if np.isnan(a) and np.isnan(b):
            ok = True
        else:
            ok = bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= max(tol, rel * abs(b)))
    checks.append((what, mine, ref, ok))
    return ok


def _stop_if_failed(checks: list, block: str) -> None:
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:60]:
            print("FAIL", c)
        raise SystemExit(f"{block}: {len(bad)} of {len(checks)} checks failed; nothing written")
    print(f"{block}: {len(checks)} of {len(checks)} checks ok", flush=True)


def _seed_of(run: str) -> str:
    return re.search(r"_s(\d+)$", _norm(run)).group(1)


def entries(split: str, label: str) -> pd.DataFrame:
    """The committed per-seed entries of one model (seeds 0, 1, 2 in this order)."""
    e = pd.read_csv(_repo(FILES[split]["entries"]))
    e = e[(e.set == split) & (e.label == label)].reset_index(drop=True)
    if [_seed_of(r) for r in e.run] != list(SEEDS):
        raise SystemExit(f"{FILES[split]['entries']}: {split} entries of {label!r} are not seeds 0, 1, 2: {list(e.run)}")
    return e


# --------------------------------------------------------------------------------------------------------------------
# loading: per seed {quantity: {set: (k per unit, n per unit)}} (hnl.eval.bootstrap_tables.counts)


def test_specs(checks: list) -> dict:
    """The test entries (run, sub, cut, wp) of the four models from A2's val-fixed spec, with the output checks of
    hnl.eval.strong_baselines_v3 (acorn) and hnl.eval.per_sample (Canto-L); every entry is its val entry's run and
    working point and the run of the committed test entries."""
    from hnl.eval import strong_baselines_v3 as V3
    spec_file = os.path.abspath(_repo(FILES["test"]["spec"]))
    by = {m["label"]: m for m in json.load(open(spec_file))["models"]}
    sel_all = json.load(open(_repo(SELECTION)))
    sel = sel_all["pipelines"]["l1tl"]["seeds"]
    out = {}
    for lab, short in MODELS:
        tent, vent = entries("test", lab), entries("val", lab)
        es = by[lab]["entries"]
        if len(es) != len(SEEDS):
            raise SystemExit(f"{spec_file}: {len(es)} entries for {lab!r}")
        for s, e, (_, t), (_, v) in zip(SEEDS, es, tent.iterrows(), vent.iterrows()):
            _cmp(checks, f"test {short} seed {s}: spec run = test entry run", _norm(e["run"]), _norm(t.run))
            _cmp(checks, f"test {short} seed {s}: spec run = val entry run", _norm(e["run"]), _norm(v.run))
            _cmp(checks, f"test {short} seed {s}: spec working point = test entry", e["wp"], t.working_point)
            if lab == CANTO:
                cfgs = json.load(open(os.path.join(e["run"], "trackbuild_test", "configs.json")))
                _cmp(checks, f"test {short} seed {s}: run = final selection", _norm(e["run"]), _norm(sel[s]["run"]))
                _cmp(checks, f"test {short} seed {s}: configuration 0 = final selection",
                     ",".join(str(x) for x in cfgs[0]), sel[s]["config"])
                same = json.load(open(os.path.join(e["run"], "final_selection_used.json"))) == sel_all
                _cmp(checks, f"test {short} seed {s}: the test job's selection copy = {SELECTION}", int(same), 1)
                fam, params, _ = re.match(r"^(\S+) (\S+) \((\S+)\)$", v.working_point).groups()
                _cmp(checks, f"test {short} seed {s}: configuration = val working point", sel[s]["config"],
                     f"{fam},{params}")
                _cmp(checks, f"test {short} seed {s}: sub and cut", f"{e['sub']} {float(e['cut']):g}",
                     "trackbuild_test 0")
            else:
                if e.get("reused"):
                    V3._check_reused_output(e)
                else:
                    V3._check_new_output(e, lab, spec_file)
                _cmp(checks, f"test {short} seed {s}: output provenance and checksums (strong_baselines_v3)", 1, 1)
                _cmp(checks, f"test {short} seed {s}: configuration = val working point", e["val_wp"], v.working_point)
                _cmp(checks, f"test {short} seed {s}: learned merging", e["config"].split(",")[0], "mwslmerge")
        out[lab] = [{k: e[k] for k in ("run", "sub", "cut", "wp")} for e in es]
    return out


def load_test(checks: list) -> tuple[BT.Resampler, dict, dict, dict]:
    from hnl.eval import test_stats as TS
    specs = test_specs(checks)
    _stop_if_failed(checks, "test entries")
    rs = BT.Resampler.claims()
    seeds, wps, runs, inputs = {}, {}, {}, []
    events = info = None
    for lab, _ in MODELS:
        seeds[lab], wps[lab], runs[lab] = [], [], []
        for e in specs[lab]:
            rt = TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"])
            ev = np.unique(rt.pe.event_id.to_numpy(np.int64))
            if events is None:
                events = ev
                info = BT.info_table(events)
            elif not np.array_equal(ev, events):
                raise SystemExit(f"{e['run']} {e['sub']}: other test events than the first output")
            seeds[lab].append(BT.counts(rs, rt.pp, rt.pe, info))
            wps[lab].append(e["wp"])
            runs[lab].append(e["run"])
            d = os.path.join(e["run"], e["sub"])
            for f in ("per_particle.parquet", "per_event.parquet"):
                inputs.append((os.path.join(d, f), sha256(os.path.join(d, f))))
            del rt
            print(f"read {SHORT[lab]} seed {_seed_of(e['run'])}: {d}", flush=True)
    meta = {"events": int(len(events)), "inputs": inputs, "tables": rs.label}
    return rs, seeds, {"wps": wps, "runs": runs}, meta


def load_val(checks: list) -> tuple[BT.Resampler, dict, dict, dict]:
    """Dry run on validation: the committed val entries of the same four rows and the val tables of
    hnl.eval.bootstrap_tables (which must exist; never created here)."""
    d = os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}")
    for name in BT.VAL_SETS:
        if not os.path.exists(os.path.join(d, f"{name}.npz")):
            raise SystemExit(f"{d}/{name}.npz missing: the dry run uses the existing val tables only")
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, d, BT.B_VAL, BT.SEED_VAL,
                            f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    seeds, wps, runs = {}, {}, {}
    for lab, _ in MODELS:
        e = entries("val", lab)
        seeds[lab] = [ld.seed(r.run, r.working_point) for _, r in e.iterrows()]
        wps[lab], runs[lab] = list(e.working_point), list(e.run)
        print(f"read {SHORT[lab]} (val)", flush=True)
    return rs, seeds, {"wps": wps, "runs": runs}, {"events": int(len(ld.events)), "inputs": [], "tables": rs.label}


# --------------------------------------------------------------------------------------------------------------------
# checks


def check_models(split: str, models: dict, checks: list) -> None:
    tab = pd.read_csv(_repo(FILES[split]["table"]))
    tab = tab[tab.set == split].set_index("label")
    for lab, short in MODELS:
        m = models[lab]
        for i, (_, r) in enumerate(entries(split, lab).iterrows()):
            BT.check_entry(m, i, r, f"{split} entry", checks)
        BT.check_row(m, tab.loc[lab], f"{split} row", checks)
        for col, q in (("k_n_d0_100", "d0_100_300"), ("k_n_d0_300", "d0_300_inf")):
            s = BT.summary(m, q)
            _cmp(checks, f"{split} {short}: {q} k/n (seeds pooled) = {col}", f"{s['k_pooled']}/{s['n_pooled']}",
                 str(tab.loc[lab, col]))
        if split == "test":
            s = BT.summary(m, "eff_primary")
            _cmp(checks, f"test {short}: primary combined = primary_sigma_total", s["combined"],
                 float(tab.loc[lab, "primary_sigma_total"]))
            _cmp(checks, f"test {short}: primary boot sd = primary_sigma_boot", s["boot_sd"],
                 float(tab.loc[lab, "primary_sigma_boot"]))
        for i in range(len(SEEDS)):
            n_t, k_t = m.den["eff_hnl_targets"][i], m.num["eff_hnl_targets"][i]
            for qs, what in ((BT.D0_BINS, "|d0| bins"), (BT.RV_BINS, "r_v bins and no vertex")):
                _cmp(checks, f"{split} {short} seed {i}: n of the {what} = HNL daughters",
                     sum(m.den[q][i] for q in qs), n_t)
                _cmp(checks, f"{split} {short} seed {i}: k of the {what} = HNL daughters found",
                     sum(m.num[q][i] for q in qs), k_t)
    for q in BIN_QS + ("eff_hnl_targets", "eff_primary"):
        ns = {tuple(float(x) for x in models[lab].den[q]) for lab, _ in MODELS}
        _cmp(checks, f"{split} {q}: the same particles for every model and seed", len(ns), 1)
    boot = pd.read_csv(_repo(BOOT))
    boot = boot[(boot.table == FILES[split]["boot_table"]) & (boot.model == BOOT_CANTO)].set_index("quantity")
    m = models[CANTO]
    for q in BT.QUANTITIES:
        s = BT.summary(m, q)
        r = boot.loc[q]
        for k in BOOT_COLS:
            if k in s:
                _cmp(checks, f"{split} Canto-L {q} {k} = {BOOT}", s[k], float(r[k]), tol=1e-15, rel=5e-10)
        for k in BOOT_INTS:
            if k in s:
                _cmp(checks, f"{split} Canto-L {q} {k} = {BOOT}", int(s[k]), int(r[k]))
        _cmp(checks, f"{split} Canto-L {q} per-seed values = {BOOT}", s["per_seed"], str(r["per_seed"]))


# --------------------------------------------------------------------------------------------------------------------
# output


def _bin(q: str) -> tuple[str, float, float, str]:
    """(variable, low edge, high edge, label) of a quantity."""
    if q in BT.KEYS:
        return "", np.nan, np.nan, q
    if q == "rv_none":
        return "rv_mm", np.nan, np.nan, "no vertex"
    var, lo, hi = q.split("_")
    lo, hi = float(lo), float(hi)
    label = f">= {lo:g}" if np.isinf(hi) else f"{lo:g}-{hi:g}"
    return {"d0": "abs_d0_mm", "rv": "rv_mm"}[var], lo, hi, label


def rows(split: str, models: dict) -> list[dict]:
    out = []
    for lab, short in MODELS:
        m = models[lab]
        for q in BT.QUANTITIES:
            var, lo, hi, label = _bin(q)
            s = BT.summary(m, q)
            out.append({"split": split, "model": lab, "short": short, "quantity": q, "variable": var, "lo": lo,
                        "hi": hi, "bin": label, **s,
                        "k_per_seed": ";".join(f"{x:.10g}" for x in m.num[q]),
                        "n_per_seed_list": ";".join(f"{x:.10g}" for x in m.den[q]),
                        "working_points": "; ".join(m.wps), "runs": "; ".join(m.runs)})
    return out


def pct(x: float, d: int = 2) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def write(split: str, models: dict, meta: dict, checks: list, out: str, name: str) -> tuple[str, str]:
    df = pd.DataFrame(rows(split, models))
    csv = os.path.join(out, f"{name}.csv")
    df.to_csv(csv, index=False, float_format="%.10g")
    here = os.path.abspath(__file__)
    L = [f"# HNL-daughter efficiency in |d0| and r_v bins, four models of paper Fig. 5 ({name})", "",
         f"`python -m hnl.eval.test_binned --split {split}` (module docstring for the method). "
         + ("Ledger R17 (docs/feedback/test_use_ledger.md): post-freeze, descriptive read-out of stored test outputs; "
            "no network run, no track building, nothing chosen on test, no claim." if split == "test" else
            "Dry run on validation outputs (no test output opened)."), "",
         f"- Set: {'pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external)' if split == 'test' else 'pooled HNL val (hnl10_100 + hnl15_200 val)'}"
         f"; {meta['events']} events read (with ttbar_{split} for the identity checks); replicate tables: "
         f"{meta['tables']}.",
         "- Population: HNL-daughter targets (pT > 1 GeV, |eta| < 4, >= 5 space points); |d0| and r_v bins of "
         "docs/definitions.md section 4 (truth-hit fit; r_v = two-track vertex radius, \"no vertex\" when the pair has "
         "none).",
         "- value = mean of the three seeds; combined = sqrt(boot sd^2 + seed sd^2 / 3) (Eq. (2) of the paper); "
         "68 % CP = Clopper-Pearson on the per-seed count.",
         f"- Models: {'; '.join(f'{SHORT[lab]} = {lab}' for lab, _ in MODELS)}; working points per seed in the CSV.",
         f"- Run: job {meta.get('job', '')}, host {meta.get('host', '')}, {meta.get('start', '')} to {meta.get('end', '')}; "
         f"git {meta.get('git', '')}; {os.path.relpath(here, REPO)} sha256 {sha256(here)}"
         + (f"; {meta['sbatch']} sha256 {meta['sbatch_sha256']}" if meta.get("sbatch") else "") + ".", ""]
    for var, title, qs in (("abs_d0_mm", "|d0| bins [mm]", BT.D0_BINS), ("rv_mm", "r_v bins [mm]", BT.RV_BINS)):
        L += [f"## Efficiency [%] in the {title}", "",
              "Cells: value [68 % CP] (combined sigma, points); n = HNL daughters per seed.", "",
              "| bin | n | " + " | ".join(short for _, short in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
        for q in qs:
            _, _, _, label = _bin(q)
            cells = []
            for lab, _ in MODELS:
                s = BT.summary(models[lab], q)
                cells.append(f"{pct(s['value'])} [{pct(s['cp68_lo'])}, {pct(s['cp68_hi'])}] ({pct(s['combined'])})")
            n = BT.summary(models[CANTO], q)["n_per_seed"]
            L.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Identity checks", "",
          f"- {len(checks)} of {len(checks)} passed (per-seed entries and model rows of "
          f"{FILES[split]['entries']} and {FILES[split]['table']}; Canto-L against {BOOT} "
          f"({FILES[split]['boot_table']}); bin sums; the same particles for every model)."]
    if meta["inputs"]:
        L += ["", "## Stored outputs read (sha256)", ""] + [f"- {p} {h}" for p, h in meta["inputs"]]
    md = os.path.join(out, f"{name}.md")
    with open(md, "w") as f:
        f.write("\n".join(L) + "\n")
    return md, csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sbatch", default=None, help="the job script (its sha256 goes into the .md)")
    a = ap.parse_args(argv)
    out = os.path.abspath(_repo(a.out))
    if a.split == "val" and os.path.commonpath([out, os.path.join(REPO, "results")]) == os.path.join(REPO, "results"):
        raise SystemExit("the val dry run does not write into results/")
    for ext in (".csv", ".md"):
        if os.path.exists(os.path.join(out, a.name + ext)):
            raise SystemExit(f"{os.path.join(out, a.name + ext)} exists: refusing to overwrite")
    start = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    checks: list = []
    rs, seeds, info, meta = (load_test if a.split == "test" else load_val)(checks)
    models = {lab: BT.Model(lab, SHORT[lab], seeds[lab], rs, info["wps"][lab], info["runs"][lab]) for lab, _ in MODELS}
    check_models(a.split, models, checks)
    _stop_if_failed(checks, f"{a.split} read-out")
    meta.update({"job": os.environ.get("SLURM_JOB_ID", "none"), "host": socket.gethostname(), "start": start,
                 "end": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git": git_state()})
    if a.sbatch:
        meta.update({"sbatch": os.path.relpath(_repo(a.sbatch), REPO), "sbatch_sha256": sha256(_repo(a.sbatch))})
    os.makedirs(out, exist_ok=True)
    md, csv = write(a.split, models, meta, checks, out, a.name)
    print(open(md).read())
    print(md, csv)


if __name__ == "__main__":
    main()
