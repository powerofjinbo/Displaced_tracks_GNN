"""Primary efficiency per sample, validation and test read out separately (supervisor request on the main comparison).

The main comparison (Canto-L against the acorn baselines, results/strong_baselines_v3) pools the HNL samples. Here the
primary efficiency (HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm) is given per sample:
  validation  hnl10_100 val (1,000 events) and hnl15_200 val (999 events), and pooled (1,999 events);
  test        hnl10_100_test, hnl15_200_test and hnl15_100_external (m_N = 15 GeV, c tau = 100 mm: a sample no model saw
              in training or in any selection), and pooled (6,999 events); post-freeze, descriptive, not claims;
and, per model and labelled sample, the change from validation to test.

Models, each at its committed per-seed working point (nothing is chosen in this module, on val or on test):
  Canto-L (frozen: learned merging)  results/final_selection_v1.json (checked against the v3 val entries and against
                     configuration 0 of the frozen test outputs RUN/trackbuild_test);
  acorn (HNL), acorn (ttbar), acorn (ttbar+HNL) at the val-best configuration of the full builder menu (MWS-layer +
                     learned merging for all nine runs): results/strong_baselines_v3_val_entries.csv (val) and
                     results/strong_baselines_v3_spec_test.json (test; provenance, spec and merge-model checksums checked
                     with hnl.eval.strong_baselines_v3's own checks);
  acorn (HNL) + CC at the preregistered cut (the claims' reference) and acorn (ttbar) + CC (preregistered cut = its
                     best cut <= FR*): cuts of results/working_points_v1.json (checked on val and test);
  validation only    Canto-L classifier + CC on the Canto-L runs' eval_val at the cut per seed of hnl.eval.ranking's
                     rule 'best' (highest primary efficiency with fake rate <= FR*: the rule of the v3 row "Canto-B +
                     CC, best cut <= FR*") and of its rule 's4' (largest fake rate <= FR*: the rule of the preregistered
                     cuts and hnl.eval.ranking's default), both validation choices of those rules; acorn (HNL) + CC,
                     best cut <= FR* (v3 val entries): the comparison of the classifiers with connected components.

Estimators (hnl.eval.bootstrap): a model's number is the seed mean with sigma^2 = sigma_boot^2 + s_seeds^2 / 3
(seed_mean); a difference new - base is linear_combination (paired event bootstrap of the difference of seed means
plus both seed terms).
  test  the code path of hnl.eval.strong_baselines.per_set: the preregistered test tables (hnl.eval.bootstrap, B = 500,
        seed 12345); hnl10_100_test and hnl15_200_test on the joint set's resampling units (event index, both samples'
        event i together), hnl15_100_external on its own units; pooled = the pooled HNL test sets of the v3 tables.
  val   per sample, the sample's own val events are resampled (units = its val event indices; B = 1000;
        numpy SeedSequence(20261008).spawn(2), child 0 hnl10_100, child 1 hnl15_200; one table per sample shared by
        every model and seed, so differences are paired): the subset construction of hnl.eval.round4_checks
        (_subset_resampler / primary_rows). Pooled val: the committed val tables of hnl.eval.bootstrap_tables (B = 1000,
        seed 20261005; both samples' event i together), as results/strong_baselines_v3_val*. Per sample, the bootstrap
        sd read out of that pooled table is given alongside (a cross-check of the choice).
  change  per model and labelled sample, test minus val (mean of the per-seed changes test_i - val_i, seed i the same
        trained network); the val and test tables are independent, so sigma^2 = sigma_boot,val^2 + sigma_boot,test^2 +
        s^2 / 3 with s the seed sd of the per-seed changes; for the difference of two models' changes (new minus base)
        the two bootstrap terms are those of the paired differences on val and on test and the seed term is
        s_new^2 / 3 + s_base^2 / 3 (change_combination).

Identity checks (tolerance 1e-12, counts exact; the run stops before writing any output if one fails): the full-val
per-seed primary efficiencies equal the v3 val entries; the pooled val seed means and sds equal
results/strong_baselines_v3_val.csv and the pooled val differences results/strong_baselines_v3_val_differences.csv; the
pooled test seed means, sds and sigmas equal results/strong_baselines_v3.csv (set test) and the pooled test differences
results/strong_baselines_v3_differences.csv; the per-set test rows equal hnl.eval.strong_baselines.per_set for every
pair and results/strong_baselines_v3_test_per_set.csv for Canto-L against acorn (HNL) val-best; per-sample counts add up
to the pooled counts for every model and seed (test: 603 + 549 + 3861 = 5013 primary particles per seed); Canto-L's
working points equal the frozen selection, the CC cuts the preregistered ones, the val-best test configurations the val
entries; the Canto-L classifier + CC entries of rule 's4' equal the reference read-out given with --cc-ref (made from
a spec without 'rule', hence 's4'; every column).

    python -m hnl.eval.per_sample --out results --name per_sample_v1 --cc-ref FILE      (slurm/feedback/per_sample.sbatch)
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT

REPO = BT.REPO
DATA = BT.DATA
SEEDS = ("0", "1", "2")
TOL = 1e-12
VAL_ENTRIES = "results/strong_baselines_v3_val_entries.csv"
VAL_TABLE = "results/strong_baselines_v3_val.csv"
VAL_DIFFS = "results/strong_baselines_v3_val_differences.csv"
TEST_SPEC = "results/strong_baselines_v3_spec_test.json"
TEST_TABLE = "results/strong_baselines_v3.csv"
TEST_DIFFS = "results/strong_baselines_v3_differences.csv"
TEST_PER_SET = "results/strong_baselines_v3_test_per_set.csv"
SELECTION = "results/final_selection_v1.json"
WP = "results/working_points_v1.json"

CANTO = "Canto-L (frozen: learned merging)"
A_HNL = "acorn (HNL) + val-best builder of the full menu (chosen per seed)"
A_TTBAR = "acorn (ttbar) + val-best builder of the full menu (chosen per seed)"
A_MIX = "acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed)"
A_HNL_CC = "acorn (HNL) + CC, preregistered cut (claims' reference)"
A_TTBAR_CC = "acorn (ttbar) + CC, preregistered cut (= best cut)"
A_HNL_CCB = "acorn (HNL) + CC, best cut <= FR*"
CANTO_CC = "Canto-L classifier + CC, best cut <= FR*"
CANTO_CC4 = "Canto-L classifier + CC, preregistered cut rule"
CC_RULE = {CANTO_CC: "best", CANTO_CC4: "s4"}    # hnl.eval.ranking rules (s4: largest fake rate <= FR*)
MODELS = (  # label, short name, splits
    (CANTO, "Canto-L (frozen: learned merging)", ("val", "test")),
    (A_HNL, "acorn (HNL) val-best (MWS-layer + learned merging)", ("val", "test")),
    (A_MIX, "acorn (ttbar+HNL) val-best (MWS-layer + learned merging)", ("val", "test")),
    (A_TTBAR, "acorn (ttbar) val-best (MWS-layer + learned merging)", ("val", "test")),
    (A_HNL_CC, "acorn (HNL) + CC, preregistered cut (claims' reference)", ("val", "test")),
    (A_TTBAR_CC, "acorn (ttbar) + CC, preregistered cut (= best cut)", ("val", "test")),
    (CANTO_CC, "Canto-L classifier + CC, best cut <= FR* (validation only)", ("val",)),
    (CANTO_CC4, "Canto-L classifier + CC, preregistered cut rule (validation only)", ("val",)),
    (A_HNL_CCB, "acorn (HNL) + CC, best cut <= FR* (validation only)", ("val",)),
)
SHORT = {lab: s for lab, s, _ in MODELS}
SPLITS = {lab: sp for lab, _, sp in MODELS}
PAIRS = (  # new, baseline, splits
    (CANTO, A_HNL, ("val", "test")),
    (CANTO, A_MIX, ("val", "test")),
    (CANTO, A_TTBAR, ("val", "test")),
    (CANTO, A_HNL_CC, ("val", "test")),
    (CANTO, A_TTBAR_CC, ("val", "test")),
    (CANTO_CC, A_HNL_CCB, ("val",)),
    (CANTO_CC, A_TTBAR_CC, ("val",)),
    (CANTO_CC4, A_HNL_CC, ("val",)),
    (CANTO_CC4, A_TTBAR_CC, ("val",)),
)
VAL_SAMPLES = (("hnl10_100", 2), ("hnl15_200", 4))
VAL_SETS = ("hnl10_100_val", "hnl15_200_val", "hnl_val_pooled")
TEST_SETS = ("hnl10_100_test", "hnl15_200_test", "hnl15_100_external", "hnl_test_pooled")
LABELLED = ("hnl10_100", "hnl15_200")
B_SAMPLE, SEED_SAMPLE = 1000, 20261008
N_TEST = {"hnl10_100_test": 603, "hnl15_200_test": 549, "hnl15_100_external": 3861, "hnl_test_pooled": 5013}
EVENTS = {"hnl10_100_val": 1000, "hnl15_200_val": 999, "hnl_val_pooled": 1999, "hnl10_100_test": 1000,
          "hnl15_200_test": 1000, "hnl15_100_external": 4999, "hnl_test_pooled": 6999}
SET_TITLE = {"hnl10_100_val": "hnl10_100 val", "hnl15_200_val": "hnl15_200 val", "hnl_val_pooled": "pooled HNL val",
             "hnl10_100_test": "hnl10_100_test", "hnl15_200_test": "hnl15_200_test",
             "hnl15_100_external": "hnl15_100_external", "hnl_test_pooled": "pooled HNL test"}


def _repo(p: str) -> str:
    return BT._repo(p)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> str:
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    d = git("status", "--porcelain", "--", "src", "slurm")
    return git("rev-parse", "HEAD") + (f" (src/slurm dirty: {len(d.splitlines())} files)" if d else "")


def _cmp(checks: list, what: str, mine, ref, tol: float = TOL) -> bool:
    if isinstance(mine, str) or isinstance(ref, str):
        ok = str(mine) == str(ref)
    else:
        ok = bool(np.isfinite(float(mine)) and abs(float(mine) - float(ref)) <= tol)
    checks.append((what, mine, ref, ok))
    return ok


def _stop_if_failed(checks: list, block: str) -> None:
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:40]:
            print("FAIL", c)
        raise SystemExit(f"{block}: {len(bad)} of {len(checks)} checks failed")
    print(f"{block}: {len(checks)} of {len(checks)} checks ok", flush=True)


def _seed_of(run: str) -> str:
    return re.search(r"_s(\d+)$", os.path.normpath(run)).group(1)


def _arm_of(run: str) -> str:
    return re.search(r"arm(i+)_s\d+$", os.path.normpath(run)).group(1)


def _cc_cut(wp: str) -> float:
    return float(re.match(r"^CC (\d+\.\d+)", wp).group(1))


# --------------------------------------------------------------------------------------------------------------------
# estimators


def change_combination(terms: list[tuple[float, list, list, list, list]]) -> dict:
    """sum_k c_k (test seed mean_k - val seed mean_k) for terms (c_k, val points, val replicates, test points, test
    replicates), seed i of model k on val and on test being the same trained network. The val and test replicate tables
    are independent: the bootstrap variance is that of the val part plus that of the test part, each paired over its own
    replicates (hnl.eval.bootstrap.linear_combination); model k's seed spread enters through its per-seed changes
    test_i - val_i, as c_k^2 s_k^2 / n_k:
        sigma^2 = sigma_boot,val^2 + sigma_boot,test^2 + sum_k c_k^2 s_k^2 / n_k.
    One term (c = 1): a model's change; two terms (1, -1): the change of a difference."""
    lv = BS.linear_combination([(c, pv, rv) for c, pv, rv, _, _ in terms])
    lt = BS.linear_combination([(c, pt, rt) for c, _, _, pt, rt in terms])
    value, seed_var, s_k, per_seed = 0.0, 0.0, [], []
    for c, pv, _, pt, _ in terms:
        pv, pt = np.asarray(pv, dtype=float), np.asarray(pt, dtype=float)
        if pv.shape != pt.shape:
            raise ValueError(f"{len(pv)} val seeds against {len(pt)} test seeds")
        ch = pt - pv
        value += c * float(ch.mean())
        s = float(np.std(ch, ddof=1)) if len(ch) > 1 else 0.0
        seed_var += c ** 2 * s ** 2 / len(ch)
        s_k.append(s)
        per_seed.append([float(x) for x in ch])
    sigma = float(np.sqrt(lv["sigma_boot"] ** 2 + lt["sigma_boot"] ** 2 + seed_var))
    return {"value": value, "val": lv["value"], "test": lt["value"], "sigma_boot_val": lv["sigma_boot"],
            "sigma_boot_test": lt["sigma_boot"], "sigma_seeds": float(np.sqrt(seed_var)), "s_seeds": s_k,
            "per_seed": per_seed, "sigma": sigma, "z": value / sigma if sigma > 0 else np.nan}


def sample_resamplers(b: int = B_SAMPLE, seed: int = SEED_SAMPLE) -> dict:
    """One replicate table per labelled sample on its own val events (units = the sample's val event indices, B
    multinomial draws, numpy SeedSequence(seed).spawn, one child per sample in VAL_SAMPLES order): the construction of
    hnl.eval.round4_checks._subset_resampler with a sample-specific code map (rows of another sample are refused)."""
    from hnl.data.splits import read_split
    out = {}
    for i, ((s, code), child) in enumerate(zip(VAL_SAMPLES, np.random.SeedSequence(seed).spawn(len(VAL_SAMPLES)))):
        units = np.unique(read_split(s, "val").astype(np.int64))
        w = BS.make_weights(len(units), b, np.random.default_rng(child))
        name = f"{s}_val"
        out[s] = BT.Resampler({name: units}, {name: w}, {code: name},
                              f"{name} own events (B = {b}, SeedSequence({seed}) child {i})")
    return out


# --------------------------------------------------------------------------------------------------------------------
# validation


def val_entries(label: str) -> list[tuple[str, str]]:
    """(run, working point) per seed of one row of the committed v3 val entries."""
    e = pd.read_csv(_repo(VAL_ENTRIES))
    e = e[(e.label == label) & (e.set == "val")]
    out = []
    for s in SEEDS:
        r = e[e.run.map(_seed_of) == s]
        if len(r) != 1:
            raise SystemExit(f"{VAL_ENTRIES}: {len(r)} rows for {label} seed {s}")
        out.append((r.run.iloc[0], r.working_point.iloc[0]))
    return out


def canto_runs() -> list[str]:
    sel = json.load(open(_repo(SELECTION)))["pipelines"]["l1tl"]["seeds"]
    return [sel[s]["run"] for s in SEEDS]


REF_COLUMNS = ("working_point", "n_events", "eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt",
               "fake_rate_hnl", "n_d0_100", "k_d0_100", "n_d0_300", "k_d0_300")


def canto_cc_entries(fr: float, ref_file: str, checks: list) -> tuple[dict, pd.DataFrame]:
    """Canto-L classifier + CC on val at the cut per seed of hnl.eval.ranking's rule 'best' (highest primary efficiency
    with fake rate <= FR*; the rule of the v3 rows "... + CC, best cut <= FR*") and of its rule 's4' (largest fake rate
    <= FR*, ties to the lowest cut; the rule of the preregistered cuts and the default of hnl.eval.ranking). The
    reference read-out (ref_file, a spec without 'rule') is checked against the 's4' entries, every column."""
    from hnl.eval import ranking as RK
    spec = [{"label": lab, "short": SHORT[lab], "group": "Canto-L classifier",
             "entries": [{"run": r, "source": "eval_val", "rule": rule} for r in canto_runs()]}
            for lab, rule in CC_RULE.items()]
    _, ent = RK.rank(spec, fr, tie_se=True)
    out = {}
    for lab in CC_RULE:
        e = ent[ent.label == lab]
        out[lab] = [(e[e.run.map(_seed_of) == s].run.iloc[0], e[e.run.map(_seed_of) == s].working_point.iloc[0])
                    for s in SEEDS]
    ref = pd.read_csv(ref_file)
    e = ent[ent.label == CANTO_CC4]
    for s in SEEDS:
        r = e[e.run.map(_seed_of) == s].iloc[0]
        o = ref[ref.run.map(os.path.normpath) == os.path.normpath(r.run)]
        if len(o) != 1:
            raise SystemExit(f"{ref_file}: {len(o)} rows for {r.run}")
        o = o.iloc[0]
        for k in REF_COLUMNS:
            mine, theirs = (r[k], o[k]) if k == "working_point" else (float(r[k]), float(o[k]))
            _cmp(checks, f"Canto-L classifier + CC (rule s4) seed {s}: {k} = reference read-out", mine, theirs)
    return out, ent


def hnl_val_info() -> pd.DataFrame:
    """Targets of the 1,999 HNL val events (the columns of hnl.eval.bootstrap_tables.info_table)."""
    from hnl.eval.particles import particle_table
    ev = BT.expected_events({"hnl": (("hnl10_100", "val"), ("hnl15_200", "val"))})
    p = particle_table(ev)[["event_id", "particle_id", "is_target", "category", "d0", "rv"]]
    return p[p.is_target.astype(bool)].reset_index(drop=True)


def val_primary(run: str, wp: str, info: pd.DataFrame, hnl_events: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """(event id, DM-matched) of the primary particles at one stored val working point (hnl.eval.round4_checks
    primary_rows: hnl.eval.bootstrap_tables val_tables and masks); the table must hold every HNL val event."""
    pp, pe, d = BT.val_tables(run, wp)
    ev = np.unique(pe.event_id.to_numpy(np.int64))
    if not np.array_equal(ev[ev // 100_000_000 != 1], hnl_events):
        raise SystemExit(f"{d} at {wp}: not the {len(hnl_events)} HNL val events")
    t = pp[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    if t.duplicated(["event_id", "particle_id"]).any():
        raise SystemExit(f"{d}: duplicate particles at {wp}")
    t = t[BT.masks(t)["eff_primary"]]
    return t.event_id.to_numpy(np.int64), t.dm_matched.to_numpy(float), d


def val_readout(entries: dict, rs_full: BT.Resampler, rs_s: dict, info: pd.DataFrame) -> tuple[dict, dict]:
    """Per model and val set: points, replicates, k and n per seed (and, per sample, the replicates of the pooled
    table); the stored directory per model and seed."""
    hnl_events = BT.expected_events({"hnl": (("hnl10_100", "val"), ("hnl15_200", "val"))})
    res, dirs = {}, {}
    for lab, ents in entries.items():
        for st in VAL_SETS:
            res[(lab, st)] = {"pts": [], "reps": [], "k": [], "n": [], "reps_full": []}
        for run, wp in ents:
            ev, ok, d = val_primary(run, wp, info, hnl_events)
            dirs[(lab, _seed_of(run))] = d
            one = np.ones(len(ev))
            code = ev // 100_000_000
            parts = {"hnl_val_pooled": (rs_full, np.ones(len(ev), dtype=bool))}
            for s, c in VAL_SAMPLES:
                parts[f"{s}_val"] = (rs_s[s], code == c)
            for st, (rs, m) in parts.items():
                p, r, k, n = rs.ratio(rs.unit_sums(ev[m], ok[m], one[m]))
                x = res[(lab, st)]
                x["pts"].append(p)
                x["reps"].append(r)
                x["k"].append(k)
                x["n"].append(n)
                if st != "hnl_val_pooled":
                    x["reps_full"].append(rs_full.ratio(rs_full.unit_sums(ev[m], ok[m], one[m]))[1])
    return res, dirs


def val_checks(entries: dict, res: dict, checks: list) -> None:
    """Per-seed full-val efficiencies against the v3 entries; pooled means / sds against the v3 val table; counts."""
    ent = pd.read_csv(_repo(VAL_ENTRIES))
    ent = ent[ent.set == "val"]
    tab = pd.read_csv(_repo(VAL_TABLE)).set_index("label")
    n_ref = pd.read_csv(_repo(VAL_DIFFS))
    n_ref = int(n_ref[(n_ref.quantity == "primary") & (n_ref.new == CANTO)].n_per_seed.iloc[0])
    for lab, ents in entries.items():
        full = res[(lab, "hnl_val_pooled")]
        for i, (run, wp) in enumerate(ents):
            s = _seed_of(run)
            if lab not in CC_RULE:
                r = ent[(ent.label == lab) & (ent.run == run)].iloc[0]
                _cmp(checks, f"val {SHORT[lab]} seed {s}: pooled primary = v3 entry", full["pts"][i], float(r.eff_primary))
            k_sum = sum(res[(lab, f"{x}_val")]["k"][i] for x, _ in VAL_SAMPLES)
            n_sum = sum(res[(lab, f"{x}_val")]["n"][i] for x, _ in VAL_SAMPLES)
            _cmp(checks, f"val {SHORT[lab]} seed {s}: k hnl10_100 + hnl15_200 = pooled", k_sum, full["k"][i])
            _cmp(checks, f"val {SHORT[lab]} seed {s}: n hnl10_100 + hnl15_200 = pooled", n_sum, full["n"][i])
            _cmp(checks, f"val {SHORT[lab]} seed {s}: pooled n = v3 primary particles per seed", full["n"][i], n_ref)
        if lab in tab.index:
            sm = BS.seed_mean(full["pts"], full["reps"])
            _cmp(checks, f"val {SHORT[lab]}: pooled seed mean = v3 val table", sm["mean"], float(tab.loc[lab, "eff_primary"]))
            _cmp(checks, f"val {SHORT[lab]}: pooled seed sd = v3 val table", sm["s_seeds"],
                 float(tab.loc[lab, "eff_primary_sd"]))


def val_working_point_checks(entries: dict, checks: list) -> None:
    """Canto-L = the frozen selection; CC rows at the preregistered cuts; the Canto-L + CC runs = Canto-L's runs."""
    sel = json.load(open(_repo(SELECTION)))["pipelines"]["l1tl"]["seeds"]
    wp = json.load(open(_repo(WP)))
    for run, w in entries[CANTO]:
        s = _seed_of(run)
        fam, params, _ = re.match(r"^(\S+) (\S+) \((\S+)\)$", w).groups()
        _cmp(checks, f"val Canto-L seed {s}: run = final selection", os.path.normpath(run), os.path.normpath(sel[s]["run"]))
        _cmp(checks, f"val Canto-L seed {s}: configuration = final selection", f"{fam},{params}", sel[s]["config"])
    for lab in (A_HNL_CC, A_TTBAR_CC):
        for run, w in entries[lab]:
            _cmp(checks, f"val {SHORT[lab]} seed {_seed_of(run)}: cut = preregistered cut", _cc_cut(w),
                 float(wp["cuts_at_FR_star"][_arm_of(run)][_seed_of(run)]), 1e-9)
    for lab in CC_RULE:
        for (run, _), (crun, _) in zip(entries[lab], entries[CANTO]):
            _cmp(checks, f"val {SHORT[lab]} seed {_seed_of(run)}: run = Canto-L's", os.path.normpath(run),
                 os.path.normpath(crun))
    for lab in (A_HNL, A_MIX, A_TTBAR):
        for run, w in entries[lab]:
            _cmp(checks, f"val {SHORT[lab]} seed {_seed_of(run)}: learned merging", w.split(" ")[0], "mwslmerge")


def val_difference_checks(res: dict, checks: list) -> None:
    """Pooled val differences Canto-L minus each acorn row against results/strong_baselines_v3_val_differences.csv."""
    ref = pd.read_csv(_repo(VAL_DIFFS))
    for new, base, sp in PAIRS:
        if new != CANTO:
            continue
        r = ref[(ref.new == new) & (ref.baseline == base) & (ref.quantity == "primary")
                & (ref.block == "Canto-L minus acorn")]
        if len(r) != 1:
            raise SystemExit(f"{VAL_DIFFS}: {len(r)} rows for {new} - {base}")
        r = r.iloc[0]
        a, b = res[(new, "hnl_val_pooled")], res[(base, "hnl_val_pooled")]
        d = BS.linear_combination([(1.0, a["pts"], a["reps"]), (-1.0, b["pts"], b["reps"])])
        sa, sb = BS.seed_mean(a["pts"], a["reps"]), BS.seed_mean(b["pts"], b["reps"])
        mine = {"value_new": sa["mean"], "sigma_new": sa["sigma"], "s_seeds_new": sa["s_seeds"], "value_base": sb["mean"],
                "sigma_base": sb["sigma"], "s_seeds_base": sb["s_seeds"], "difference": d["value"],
                "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"], "sigma": d["sigma"], "z": d["z"]}
        for k, v in mine.items():
            _cmp(checks, f"val pooled Canto-L - {SHORT[base]}: {k} = v3 val differences", v, float(r[k]))


# --------------------------------------------------------------------------------------------------------------------
# test


def test_entries(val: dict, checks: list) -> dict:
    """The test entries (run, sub, cut, wp) of the models with a test read-out, from A2's val-fixed test spec, with the
    checks of hnl.eval.strong_baselines_v3 (new / reused outputs: configuration, provenance, spec and merge-model
    checksums) and of hnl.eval.ranking_test (Canto-L: configuration 0 of the frozen test outputs, the selection copy
    of the test job; CC: the cut is in the stored table); every entry is the val entry's run and working point."""
    from hnl.eval import ranking_test as RT
    from hnl.eval import strong_baselines_v3 as V3
    spec_file = os.path.abspath(_repo(TEST_SPEC))
    by = {m["label"]: m for m in json.load(open(spec_file))["models"]}
    sel_all = json.load(open(_repo(SELECTION)))
    sel = sel_all["pipelines"]["l1tl"]["seeds"]
    wp = json.load(open(_repo(WP)))
    out = {}
    for lab, _, sp in MODELS:
        if "test" not in sp:
            continue
        es = []
        for s, e, (vrun, vwp) in zip(SEEDS, by[lab]["entries"], val[lab]):
            _cmp(checks, f"test {SHORT[lab]} seed {s}: run = val entry's run", os.path.normpath(e["run"]),
                 os.path.normpath(vrun))
            if "config" in e:
                if e.get("reused"):
                    V3._check_reused_output(e)
                else:
                    V3._check_new_output(e, lab, spec_file)
                _cmp(checks, f"test {SHORT[lab]} seed {s}: configuration = val working point", e["val_wp"], vwp)
                _cmp(checks, f"test {SHORT[lab]} seed {s}: output provenance and checksums (strong_baselines_v3)", 1, 1)
            elif lab == CANTO:
                cfgs = json.load(open(os.path.join(e["run"], "trackbuild_test", "configs.json")))
                _cmp(checks, f"test Canto-L seed {s}: configuration 0 = final selection",
                     ",".join(str(x) for x in cfgs[0]), sel[s]["config"])
                same = json.load(open(os.path.join(e["run"], "final_selection_used.json"))) == sel_all
                _cmp(checks, f"test Canto-L seed {s}: the test job's selection copy = {SELECTION}", int(same), 1)
                fam, params, _ = re.match(r"^(\S+) (\S+) \((\S+)\)$", vwp).groups()
                _cmp(checks, f"test Canto-L seed {s}: configuration = val working point", sel[s]["config"],
                     f"{fam},{params}")
            else:
                cut = float(e["cut"])
                _cmp(checks, f"test {SHORT[lab]} seed {s}: cut = preregistered cut", cut,
                     float(wp["cuts_at_FR_star"][_arm_of(e["run"])][s]), 1e-9)
                _cmp(checks, f"test {SHORT[lab]} seed {s}: cut = val working point", cut, _cc_cut(vwp), 1e-9)
                _cmp(checks, f"test {SHORT[lab]} seed {s}: cut held by the stored table",
                     int(RT._holds(os.path.join(e["run"], e["sub"]), cut)), 1)
            es.append({k: e[k] for k in ("run", "sub", "cut", "wp")})
        out[lab] = es
    return out


def test_counts(rt, info: pd.DataFrame, sets) -> tuple[int, int]:
    """(DM-matched, primary particles) of one seed in `sets`: the rows of hnl.eval.test_stats.efficiency."""
    from hnl.eval import test_stats as TS
    t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
    m = np.asarray(TS.masks()["primary"](t), dtype=bool) & TS.in_sets(t.event_id, sets)
    return int(round(float(t.dm_matched[m].astype(float).sum()))), int(m.sum())


def test_readout(entries: dict) -> tuple[dict, dict]:
    """Per model and test set: points, replicates (preregistered tables), k and n per seed."""
    from hnl.eval import test_stats as TS
    from hnl.eval.particles import particle_table
    for f in BS.SETS:
        p = os.path.join(BS.DEFAULT_DIR, f"{f}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: the preregistered test tables must exist (never created here)")
    bs = BS.Bootstrap()
    runs = {lab: [TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"]) for e in es] for lab, es in entries.items()}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for rr in runs.values() for rt in rr]))
    info = particle_table(ev)
    res = {}
    for lab, rr in runs.items():
        for st in TEST_SETS:
            sets = TS.HNL_SETS if st == "hnl_test_pooled" else (st,)
            pts, reps, n = TS.efficiency(bs, rr, info, TS.masks()["primary"], sets)
            kn = [test_counts(rt, info, sets) for rt in rr]
            res[(lab, st)] = {"pts": pts, "reps": reps, "k": [k for k, _ in kn], "n": [x for _, x in kn],
                              "n_first": n}
    units = {"hnl10_100_test": len(bs.units["hnl_joint_test"]), "hnl15_200_test": len(bs.units["hnl_joint_test"]),
             "hnl15_100_external": len(bs.units["hnl15_100_external"]),
             "hnl_test_pooled": len(bs.units["hnl_joint_test"]) + len(bs.units["hnl15_100_external"])}
    return res, units


def test_checks(entries: dict, res: dict, checks: list) -> pd.DataFrame:
    """Counts; pooled rows and differences against the v3 test files; per-set rows against
    hnl.eval.strong_baselines.per_set (every pair) and results/strong_baselines_v3_test_per_set.csv."""
    from hnl.eval import strong_baselines as SB
    for lab in entries:
        for i, s in enumerate(SEEDS):
            for st in TEST_SETS:
                x = res[(lab, st)]
                _cmp(checks, f"test {SHORT[lab]} seed {s} {st}: n = {N_TEST[st]}", x["n"][i], N_TEST[st])
                _cmp(checks, f"test {SHORT[lab]} seed {s} {st}: efficiency = k / n", x["pts"][i], x["k"][i] / x["n"][i])
            for q in ("k", "n"):
                _cmp(checks, f"test {SHORT[lab]} seed {s}: {q} of the three sets = pooled",
                     sum(res[(lab, st)][q][i] for st in TEST_SETS[:3]), res[(lab, "hnl_test_pooled")][q][i])
    tab = pd.read_csv(_repo(TEST_TABLE))
    tab = tab[tab.set == "test"].set_index("label")
    for lab in entries:
        x = res[(lab, "hnl_test_pooled")]
        sm = BS.seed_mean(x["pts"], x["reps"])
        for mine, k in ((sm["mean"], "eff_primary"), (sm["s_seeds"], "eff_primary_sd"), (sm["sigma"], "primary_sigma_total"),
                        (sm["sigma_boot"], "primary_sigma_boot")):
            _cmp(checks, f"test pooled {SHORT[lab]}: {k} = v3 test table", mine, float(tab.loc[lab, k]))
    ref = pd.read_csv(_repo(TEST_DIFFS))
    ref = ref[(ref.set == "test") & (ref.quantity == "primary") & (ref.block == "Canto-L minus acorn")]
    for new, base, sp in PAIRS:
        if "test" not in sp:
            continue
        r = ref[(ref.new == new) & (ref.baseline == base)]
        if len(r) != 1:
            raise SystemExit(f"{TEST_DIFFS}: {len(r)} test rows for {new} - {base}")
        r = r.iloc[0]
        a, b = res[(new, "hnl_test_pooled")], res[(base, "hnl_test_pooled")]
        d = BS.linear_combination([(1.0, a["pts"], a["reps"]), (-1.0, b["pts"], b["reps"])])
        sa, sb = BS.seed_mean(a["pts"], a["reps"]), BS.seed_mean(b["pts"], b["reps"])
        mine = {"value_new": sa["mean"], "sigma_new": sa["sigma"], "s_seeds_new": sa["s_seeds"], "value_base": sb["mean"],
                "sigma_base": sb["sigma"], "s_seeds_base": sb["s_seeds"], "difference": d["value"],
                "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"], "sigma": d["sigma"], "z": d["z"]}
        for k, v in mine.items():
            _cmp(checks, f"test pooled Canto-L - {SHORT[base]}: {k} = v3 test differences", v, float(r[k]))
    # per set: the module function itself, every pair
    spec = [{"label": lab, "entries": es} for lab, es in entries.items()]
    prs = [(a, b) for a, b, sp in PAIRS if "test" in sp]
    ps = SB.per_set(spec, prs)
    for _, r in ps.iterrows():
        a, b = res[(r.new, r.set)], res[(r.baseline, r.set)]
        d = BS.linear_combination([(1.0, a["pts"], a["reps"]), (-1.0, b["pts"], b["reps"])])
        sa, sb = BS.seed_mean(a["pts"], a["reps"]), BS.seed_mean(b["pts"], b["reps"])
        mine = {"n_per_seed": a["n"][0], "value_new": sa["mean"], "sigma_new": sa["sigma"], "value_base": sb["mean"],
                "sigma_base": sb["sigma"], "difference": d["value"], "sigma": d["sigma"], "z": d["z"]}
        for k, v in mine.items():
            _cmp(checks, f"test {r.set} {SHORT[r.new]} - {SHORT[r.baseline]}: {k} = strong_baselines.per_set", v,
                 float(r[k]))
    old = pd.read_csv(_repo(TEST_PER_SET))
    if len(old) != 3:
        raise SystemExit(f"{TEST_PER_SET}: {len(old)} rows, expected 3")
    for _, o in old.iterrows():
        r = ps[(ps.set == o.set) & (ps.new == o.new) & (ps.baseline == o.baseline)]
        if len(r) != 1 or o.new != CANTO or o.baseline != A_HNL:
            raise SystemExit(f"{TEST_PER_SET}: row {o.set} {o.new} - {o.baseline} not found")
        for k in ("n_per_seed", "value_new", "sigma_new", "value_base", "sigma_base", "difference", "sigma", "z"):
            _cmp(checks, f"test {o.set} Canto-L - acorn (HNL) val-best: {k} = {TEST_PER_SET}", float(r.iloc[0][k]),
                 float(o[k]))
    return ps


# --------------------------------------------------------------------------------------------------------------------
# tables


def model_rows(res: dict, split: str, sets: tuple, entries: dict, units: dict) -> list[dict]:
    rows = []
    for lab, ents in entries.items():
        for st in sets:
            x = res[(lab, st)]
            ns = np.asarray(x["n"], dtype=float)
            if not np.all(ns == ns[0]):
                raise SystemExit(f"{lab} {st}: primary particles differ between seeds {ns}")
            sm = BS.seed_mean(x["pts"], x["reps"])
            row = {"split": split, "set": st, "model": SHORT[lab], "label": lab, "events": EVENTS[st], "units": units[st],
                   "n_per_seed": int(ns[0])}
            for i, s in enumerate(SEEDS):
                row[f"k_s{s}"] = int(round(float(x["k"][i])))
            for i, s in enumerate(SEEDS):
                row[f"eff_s{s}"] = float(x["pts"][i])
            row.update({"mean": sm["mean"], "sigma_boot": sm["sigma_boot"], "s_seeds": sm["s_seeds"], "sigma": sm["sigma"],
                        "B": len(x["reps"][0])})
            if split == "val":
                row["tables"] = (f"val tables of hnl.eval.bootstrap_tables (B {BT.B_VAL}, seed {BT.SEED_VAL})"
                                 if st == "hnl_val_pooled" else
                                 f"own events of the sample (B {B_SAMPLE}, SeedSequence({SEED_SAMPLE}) child "
                                 f"{[f'{s}_val' for s, _ in VAL_SAMPLES].index(st)})")
                row["sigma_boot_pooled_table"] = (BS.seed_mean(x["pts"], x["reps_full"])["sigma_boot"]
                                                  if st != "hnl_val_pooled" else np.nan)
            else:
                row["tables"] = (f"preregistered test tables (B {BS.B}, seed {BS.SEED})"
                                 + ("; joint units of hnl10_100_test and hnl15_200_test" if st in
                                    ("hnl10_100_test", "hnl15_200_test") else ""))
                row["sigma_boot_pooled_table"] = np.nan
            row["working_points"] = "; ".join(e[1] if isinstance(e, tuple) else e["wp"] for e in ents)
            rows.append(row)
    return rows


def difference_rows(res: dict, split: str, sets: tuple) -> list[dict]:
    rows = []
    for new, base, sp in PAIRS:
        if split not in sp:
            continue
        for st in sets:
            a, b = res[(new, st)], res[(base, st)]
            d = BS.linear_combination([(1.0, a["pts"], a["reps"]), (-1.0, b["pts"], b["reps"])])
            sa, sb = BS.seed_mean(a["pts"], a["reps"]), BS.seed_mean(b["pts"], b["reps"])
            row = {"split": split, "set": st, "new": SHORT[new], "baseline": SHORT[base], "n_per_seed": int(a["n"][0]),
                   "value_new": sa["mean"], "sigma_new": sa["sigma"], "value_base": sb["mean"], "sigma_base": sb["sigma"],
                   "difference": d["value"], "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"],
                   "sigma": d["sigma"], "z": d["z"], "B": len(a["reps"][0]),
                   "per_seed_new": " / ".join(f"{p:.6f}" for p in a["pts"]),
                   "per_seed_base": " / ".join(f"{p:.6f}" for p in b["pts"]), "label_new": new, "label_base": base}
            if split == "val" and st != "hnl_val_pooled":
                row["sigma_boot_pooled_table"] = BS.linear_combination(
                    [(1.0, a["pts"], a["reps_full"]), (-1.0, b["pts"], b["reps_full"])])["sigma_boot"]
            else:
                row["sigma_boot_pooled_table"] = np.nan
            rows.append(row)
    return rows


def change_rows(vres: dict, tres: dict) -> list[dict]:
    rows = []

    def add(kind, new, base, s, terms):
        c = change_combination(terms)
        row = {"kind": kind, "sample": s, "val_set": f"{s}_val", "test_set": f"{s}_test", "model": SHORT[new],
               "baseline": SHORT[base] if base else "", "val": c["val"], "test": c["test"], "change": c["value"],
               "sigma_boot_val": c["sigma_boot_val"], "sigma_boot_test": c["sigma_boot_test"],
               "sigma_seeds": c["sigma_seeds"], "s_seeds_model": c["s_seeds"][0],
               "s_seeds_baseline": c["s_seeds"][1] if base else np.nan, "sigma": c["sigma"], "z": c["z"],
               "per_seed_change_model": " / ".join(f"{x:+.6f}" for x in c["per_seed"][0]),
               "per_seed_change_baseline": " / ".join(f"{x:+.6f}" for x in c["per_seed"][1]) if base else "",
               "B_val": len(terms[0][2][0]), "B_test": len(terms[0][4][0]), "label": new, "label_baseline": base or ""}
        rows.append(row)

    for s in LABELLED:
        for lab, _, sp in MODELS:
            if "test" not in sp:
                continue
            v, t = vres[(lab, f"{s}_val")], tres[(lab, f"{s}_test")]
            add("model", lab, None, s, [(1.0, v["pts"], v["reps"], t["pts"], t["reps"])])
        for new, base, sp in PAIRS:
            if "test" not in sp:
                continue
            vn, tn = vres[(new, f"{s}_val")], tres[(new, f"{s}_test")]
            vb, tb = vres[(base, f"{s}_val")], tres[(base, f"{s}_test")]
            add("difference", new, base, s, [(1.0, vn["pts"], vn["reps"], tn["pts"], tn["reps"]),
                                             (-1.0, vb["pts"], vb["reps"], tb["pts"], tb["reps"])])
    return rows


# --------------------------------------------------------------------------------------------------------------------
# markdown


def cc_cuts(meta: dict, lab: str) -> str:
    return " / ".join(re.match(r"^CC (\d+\.\d+)", w).group(1) for w in meta["working_points"][SHORT[lab]]["val"])


def eff_cell(mean: float, sigma: float) -> str:
    return f"{100 * mean:.2f} ± {100 * sigma:.2f}"


def diff_cell(value: float, sigma: float, z: float) -> str:
    return f"{100 * value:+.2f} ± {100 * sigma:.2f} (z {z:.1f})"


def _eff_table(mt: pd.DataFrame, sets: tuple) -> list[str]:
    g = mt.set_index(["label", "set"])
    head = " | ".join(f"{SET_TITLE[s]} ({EVENTS[s]:,} events; {int(g.xs(s, level='set').n_per_seed.iloc[0]):,} per seed)"
                      for s in sets)
    lines = [f"| model | {head} |", "|---|" + "---|" * len(sets)]
    for lab in dict.fromkeys(mt.label):
        cells = [eff_cell(g.loc[(lab, s), "mean"], g.loc[(lab, s), "sigma"]) for s in sets]
        lines.append(f"| {SHORT[lab]} | " + " | ".join(cells) + " |")
    return lines


def _diff_table(dt_: pd.DataFrame, sets: tuple) -> list[str]:
    g = dt_.set_index(["label_new", "label_base", "set"])
    lines = ["| new | baseline | " + " | ".join(SET_TITLE[s] for s in sets) + " |", "|---|---|" + "---|" * len(sets)]
    for (a, b) in dict.fromkeys(zip(dt_.label_new, dt_.label_base)):
        cells = [diff_cell(g.loc[(a, b, s), "difference"], g.loc[(a, b, s), "sigma"], g.loc[(a, b, s), "z"]) for s in sets]
        lines.append(f"| {SHORT[a]} | {SHORT[b]} | " + " | ".join(cells) + " |")
    return lines


def _count_table(mt: pd.DataFrame, val: bool) -> list[str]:
    extra = " | bootstrap sd, pooled val table" if val else ""
    lines = [f"| model | set | n per seed | k per seed (s0 / s1 / s2) | efficiency per seed (%) | mean ± sigma (%) | "
             f"bootstrap sd | seed sd{extra} |", "|---|---|---|---|---|---|---|---|" + ("---|" if val else "")]
    for _, r in mt.iterrows():
        k = " / ".join(str(r[f"k_s{s}"]) for s in SEEDS)
        e = " / ".join(f"{100 * r[f'eff_s{s}']:.2f}" for s in SEEDS)
        x = (" | " + (f"{100 * r.sigma_boot_pooled_table:.2f}" if np.isfinite(r.sigma_boot_pooled_table) else "-")
             if val else "")
        lines.append(f"| {r.model} | {SET_TITLE[r.set]} | {r.n_per_seed} | {k} | {e} | {eff_cell(r['mean'], r.sigma)} | "
                     f"{100 * r.sigma_boot:.2f} | {100 * r.s_seeds:.2f}{x} |")
    return lines


def markdown(mt: pd.DataFrame, dt_: pd.DataFrame, ct: pd.DataFrame, fr: float, n_checks: int, meta: dict) -> str:
    vm, tm = mt[mt.split == "val"], mt[mt.split == "test"]
    vd, td = dt_[dt_.split == "val"], dt_[dt_.split == "test"]
    L = ["# Primary efficiency per sample: validation and test read out separately", "",
         "Primary = HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm. Every model at its committed per-seed "
         "working point (three seeds; nothing is chosen here, on validation or on test): Canto-L (frozen: learned "
         "merging; results/final_selection_v1.json); each acorn arm at its val-best configuration of the full builder "
         "menu (MWS-layer + learned merging for all nine runs; results/strong_baselines_v3_val_entries.csv and "
         "results/strong_baselines_v3_spec_test.json); acorn (HNL) + CC at the preregistered cut (the claims' "
         "reference) and acorn (ttbar) + CC (preregistered cut = best cut <= FR*). Validation only: the Canto-L "
         "classifier (Canto-L's scores on its learned triplet union graph) + CC on the Canto-L runs' eval_val, at the "
         f"best cut <= FR* = {fr:.6f} (hnl.eval.ranking rule 'best': highest primary efficiency with fake rate <= FR*; "
         f"cuts {cc_cuts(meta, CANTO_CC)}) and at the cut of the preregistered rule (rule 's4': largest fake rate <= FR*; "
         f"cuts {cc_cuts(meta, CANTO_CC4)}), and acorn (HNL) + CC, best cut <= FR*.", "",
         "Each cell: seed mean ± total sigma, sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (hnl.eval.bootstrap.seed_mean); "
         "differences new minus baseline in percentage points with the paired event bootstrap of the difference of "
         "seed means plus both seed terms (linear_combination). Validation per sample: each sample's own val events "
         f"resampled (B = {B_SAMPLE}, SeedSequence({SEED_SAMPLE}), one table per sample shared by every model and seed; "
         "the subset construction of hnl.eval.round4_checks); pooled validation: the committed val tables (B = "
         f"{BT.B_VAL}, seed {BT.SEED_VAL}; both samples' event i together), as results/strong_baselines_v3_val. Test: "
         f"the preregistered tables (B = {BS.B}, seed {BS.SEED}; hnl10_100_test and hnl15_200_test on the joint "
         "units, as hnl.eval.strong_baselines.per_set); post-freeze, descriptive, not claims "
         "(results/claims_test_v1.json unchanged). hnl15_100_external (m_N = 15 GeV, c tau = 100 mm) entered no "
         "training and no selection.", "",
         "## Validation", "", "### Primary efficiency (%)", ""]
    L += _eff_table(vm, VAL_SETS)
    L += ["", "### Differences (points)", ""]
    L += _diff_table(vd, VAL_SETS)
    L += ["", "### Counts per seed", "",
          "Last column: the bootstrap sd of the seed mean when the sample is read out of the pooled val table (both "
          "samples' event i resampled together) instead of its own table (cross-check of the resampling choice).", ""]
    L += _count_table(vm, True)
    L += ["", "## Test (post-freeze, descriptive; not claims)", "", "### Primary efficiency (%)", ""]
    L += _eff_table(tm, TEST_SETS)
    L += ["", "### Differences (points)", ""]
    L += _diff_table(td, TEST_SETS)
    L += ["", "### Counts per seed", ""]
    L += _count_table(tm, False)
    L += ["", "## Validation to test, per labelled sample", "",
          "Test minus validation (points; the mean of the per-seed changes, seed i the same trained network on both). "
          "The validation and test tables are independent: sigma^2 = sigma_boot,val^2 + sigma_boot,test^2 + s^2 / 3, s "
          "the seed sd of the per-seed changes; for the change of a difference the bootstrap terms are those of the "
          "paired differences on validation and on test and the seed term is s_new^2 / 3 + s_base^2 / 3.", "",
          "### Change of each model", "",
          "| model | hnl10_100: val -> test (%) | change (points) | hnl15_200: val -> test (%) | change (points) |",
          "|---|---|---|---|---|"]
    cm = ct[ct.kind == "model"].set_index(["label", "sample"])
    for lab in dict.fromkeys(ct[ct.kind == "model"].label):
        cells = []
        for s in LABELLED:
            r = cm.loc[(lab, s)]
            cells += [f"{100 * r.val:.2f} -> {100 * r.test:.2f}", diff_cell(r.change, r.sigma, r.z)]
        lines = f"| {SHORT[lab]} | " + " | ".join(cells) + " |"
        L.append(lines)
    L += ["", "### Change of the difference (Canto-L minus baseline)", "",
          "| baseline | hnl10_100: val -> test difference (points) | change (points) | hnl15_200: val -> test difference "
          "(points) | change (points) |", "|---|---|---|---|---|"]
    cd = ct[ct.kind == "difference"].set_index(["label_baseline", "sample"])
    for base in dict.fromkeys(ct[ct.kind == "difference"].label_baseline):
        cells = []
        for s in LABELLED:
            r = cd.loc[(base, s)]
            cells += [f"{100 * r.val:+.2f} -> {100 * r.test:+.2f}", diff_cell(r.change, r.sigma, r.z)]
        L.append(f"| {SHORT[base]} | " + " | ".join(cells) + " |")
    L += ["", "## Checks", "",
          f"{n_checks} of {n_checks} identity checks passed (tolerance 1e-12, counts exact; results/"
          f"{meta['name']}_checks.csv): pooled val and test rows and differences reproduce results/strong_baselines_v3*; "
          "the per-set test rows reproduce hnl.eval.strong_baselines.per_set for every pair and "
          "results/strong_baselines_v3_test_per_set.csv; per-sample counts add up to the pooled counts for every model "
          "and seed; working points, cuts and test-output provenance as committed; the Canto-L classifier + CC entries "
          "of rule 's4' equal the reference read-out of that row (meta: cc_reference; it was made with hnl.eval.ranking's "
          "default rule, 's4'), every column.", "",
          f"Slurm job {meta['slurm_job']}, {meta['started_utc']} - {meta['finished_utc']} (UTC), git {meta['git']}.", ""]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------------------------------
# inputs and main


def input_files(val_dirs: dict, test_entries_: dict, cc_ref: str) -> list[str]:
    files = [_repo(p) for p in (VAL_ENTRIES, VAL_TABLE, VAL_DIFFS, TEST_SPEC, TEST_TABLE, TEST_DIFFS, TEST_PER_SET,
                                SELECTION, WP)]
    files += [_repo(f"splits/{s}.txt") for s in ("hnl10_100_val", "hnl15_200_val", "ttbar_val", "hnl10_100_test",
                                                 "hnl15_200_test", "ttbar_test", "hnl15_100_external")]
    files.append(_repo("docs/dropped_events.tsv"))
    files += [os.path.join(BS.DEFAULT_DIR, f"{s}.npz") for s in BS.SETS]
    files += [os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}", f"{s}.npz") for s in BT.VAL_SETS]
    for d in sorted(set(val_dirs.values())):
        files += [os.path.join(d, f) for f in ("per_particle.parquet", "per_event.parquet", "configs.json",
                                              "merged_from.json") if os.path.exists(os.path.join(d, f))]
    for es in test_entries_.values():
        for e in es:
            d = os.path.join(e["run"], e["sub"])
            files += [os.path.join(d, f) for f in ("per_particle.parquet", "per_event.parquet", "configs.json",
                                                  "provenance.json") if os.path.exists(os.path.join(d, f))]
            if e["sub"] == "trackbuild_test":
                files.append(os.path.join(e["run"], "final_selection_used.json"))
    derived = os.path.join(DATA, "derived")
    for s in ("hnl10_100", "hnl15_200", "hnl15_100", "ttbar"):
        files += sorted(glob.glob(os.path.join(derived, "events", "v1", "particles", s, "*.parquet")))
        files += [p for p in (os.path.join(derived, "truth_fits", "v1", f"particles_{s}.parquet"),
                              os.path.join(derived, "truth_fits", "v1", f"vertices_{s}.parquet")) if os.path.exists(p)]
    files.append(os.path.abspath(cc_ref))
    return list(dict.fromkeys(files))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=_repo("results"))
    ap.add_argument("--name", default="per_sample_v1")
    ap.add_argument("--cc-ref", required=True, help="reference read-out of the Canto-L + CC val entries (CSV)")
    a = ap.parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"name": a.name, "git": git_state(), "slurm_job": os.environ.get("SLURM_JOB_ID", "local"),
            "host": os.uname().nodename, "started_utc": started}
    fr = float(json.load(open(_repo(WP)))["FR_star"])
    checks: list = []

    # validation ------------------------------------------------------------------------------------------------
    entries = {lab: val_entries(lab) for lab, _, _ in MODELS if lab not in CC_RULE}
    cc, cc_ent = canto_cc_entries(fr, a.cc_ref, checks)
    entries.update(cc)
    entries = {lab: entries[lab] for lab, _, _ in MODELS}
    val_working_point_checks(entries, checks)
    for s in BT.VAL_SETS:
        p = os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}", f"{s}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: the committed val tables must exist (never created here)")
    rs_full = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                                 BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    rs_s = sample_resamplers()
    info = hnl_val_info()
    vres, vdirs = val_readout(entries, rs_full, rs_s, info)
    val_checks(entries, vres, checks)
    val_difference_checks(vres, checks)
    _stop_if_failed(checks, "validation")
    vunits = {"hnl_val_pooled": len(rs_full.units["hnl_joint_val"]),
              **{f"{s}_val": len(rs_s[s].units[f"{s}_val"]) for s, _ in VAL_SAMPLES}}

    # test ------------------------------------------------------------------------------------------------------
    n_val = len(checks)
    tentries = test_entries(entries, checks)
    tres, tunits = test_readout(tentries)
    test_checks(tentries, tres, checks)
    _stop_if_failed(checks[n_val:], "test")

    # tables ------------------------------------------------------------------------------------------------------
    mt = pd.DataFrame(model_rows(vres, "val", VAL_SETS, entries, vunits)
                      + model_rows(tres, "test", TEST_SETS, tentries, tunits))
    dtab = pd.DataFrame(difference_rows(vres, "val", VAL_SETS) + difference_rows(tres, "test", TEST_SETS))
    ct = pd.DataFrame(change_rows(vres, tres))
    for _, r in ct.iterrows():                    # the change rows against the model rows they combine
        if r.kind == "model":
            v = mt[(mt.label == r.label) & (mt.set == f"{r['sample']}_val")].iloc[0]
            t = mt[(mt.label == r.label) & (mt.set == f"{r['sample']}_test")].iloc[0]
            _cmp(checks, f"change {r.model} {r['sample']}: val = model row", r.val, v["mean"])
            _cmp(checks, f"change {r.model} {r['sample']}: test = model row", r.test, t["mean"])
            _cmp(checks, f"change {r.model} {r['sample']}: change = test - val", r.change, t["mean"] - v["mean"])
            _cmp(checks, f"change {r.model} {r['sample']}: bootstrap sds = model rows",
                 r.sigma_boot_val ** 2 + r.sigma_boot_test ** 2, v.sigma_boot ** 2 + t.sigma_boot ** 2)
    _stop_if_failed(checks, "all")
    meta["inputs"] = {p: sha256(p) for p in input_files(vdirs, tentries, a.cc_ref)}
    meta.update({
        "test_use": "post-freeze, descriptive read-out of stored test outputs at committed configurations: every "
                    "working point was fixed on validation before (results/final_selection_v1.json, "
                    "results/strong_baselines_v3_val_entries.csv / _spec_test.json, results/working_points_v1.json); "
                    "no configuration, cut, model or rule was chosen on test; results/claims_test_v1.json unchanged",
        "validation_choice": "Canto-L classifier + CC (validation only, no test read-out): cut per seed chosen on "
                             "validation by hnl.eval.ranking's rule 'best' (highest primary efficiency with fake rate "
                             "<= FR*) and, as a second row, by its rule 's4' (largest fake rate <= FR*, ties to the "
                             "lowest cut; the rule of the preregistered cuts and hnl.eval.ranking's default). The "
                             "reference read-out given with --cc-ref was made from a spec without 'rule', i.e. with "
                             "'s4', and equals the 's4' row (checked); the 'best' row has other cuts",
        "cc_entries": {lab: cc_ent[cc_ent.label == lab][["run", "working_point", "eff_primary", "fake_rate_hnl"]]
                       .to_dict("records") for lab in CC_RULE},
        "cc_reference": {"path": os.path.abspath(a.cc_ref), "sha256": sha256(a.cc_ref)},
        "fr_star": fr,
        "per_sample_val_tables": {s: {"units": int(len(rs_s[s].units[f"{s}_val"])), "B": B_SAMPLE,
                                      "seed": f"SeedSequence({SEED_SAMPLE}).spawn({len(VAL_SAMPLES)})[{i}]",
                                      "w_sha256": hashlib.sha256(np.ascontiguousarray(
                                          rs_s[s].w[f"{s}_val"].astype(np.int32)).tobytes()).hexdigest()}
                                  for i, (s, _) in enumerate(VAL_SAMPLES)},
        "checks": f"{sum(c[3] for c in checks)} of {len(checks)}",
        "working_points": {SHORT[lab]: {"val": [w for _, w in entries[lab]],
                                        "test": [e["wp"] for e in tentries[lab]] if lab in tentries else None}
                           for lab, _, _ in MODELS},
    })
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(a.out, exist_ok=True)
    mt.to_csv(os.path.join(a.out, f"{a.name}.csv"), index=False)
    dtab.to_csv(os.path.join(a.out, f"{a.name}_differences.csv"), index=False)
    ct.to_csv(os.path.join(a.out, f"{a.name}_changes.csv"), index=False)
    pd.DataFrame(checks, columns=["check", "value", "reference", "ok"]).to_csv(
        os.path.join(a.out, f"{a.name}_checks.csv"), index=False)
    md = markdown(mt, dtab, ct, fr, len(checks), meta)
    with open(os.path.join(a.out, f"{a.name}.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out, f"{a.name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(md)
    print(json.dumps({k: v for k, v in meta.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
