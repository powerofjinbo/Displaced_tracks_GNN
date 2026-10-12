"""Supervisor feedback, second round: two validation-only checks (no test file and no test output is read).

Check 1, checkpoint-selection overlap (`valmon`). acorn (HNL) (Stage-0 arm ii, runs/baseline_v1/armii_s{0,1,2}) chose
its metric-learning checkpoint (best f1) and its GNN checkpoint (best val_loss) on the valmon events,
hnl.train.acorn_runner.arm_events('ii', 'valmon'): the first N_VAL_MONITOR / 2 = 250 events of the hnl10_100 val split
and of the hnl15_200 val split, 500 of the 1,999 pooled HNL validation events. Canto (arm i) monitored on ttbar val
events. The primary efficiency (HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm) of Canto-L (frozen: learned
merging), acorn (HNL) at its val-best configuration (MWS-layer + learned merging) and acorn (HNL) + MWS-layer + merging
(and, for the same-builder pair, Canto-L + MWS-layer + merging) is computed separately on (a) the 500 valmon events and
(b) the other 1,499 HNL val events, from the stored val outputs at the committed per-seed working points
(results/strong_baselines_v3_val_entries.csv; Canto-L's are those of results/final_selection_v1.json, checked).
Estimators of hnl.eval.bootstrap: a model's number is the seed mean with sigma^2 = sigma_boot^2 + s_seeds^2 / 3
(seed_mean); a difference is linear_combination (paired event bootstrap of the difference of seed means + both seed
terms). Resampling units are event indices (both samples' event i together, as in the val tables of
hnl.eval.bootstrap_tables); each subset is resampled on its own units (B = 1000, SeedSequence(20261007).spawn(2), one
table per subset shared by all models and seeds, so differences are paired). The change of a difference between the
subsets (rest minus valmon) uses the joint replicates of the committed full-val table (B = 1000, seed 20261005; both
subsets from the same replicate) and, per model, the seed term of the per-seed subset differences. Checks: the valmon
list equals the first 250 lines of splits/hnl10_100_val.txt and splits/hnl15_200_val.txt; arm_events and
N_VAL_MONITOR are unchanged at the six training commits (run_*.json) and splits/ is unchanged since; each training
log staged "arm ii valmon: 500 events"; the full-val per-seed efficiencies reproduce the entries and the full-val
differences reproduce results/strong_baselines_v3_val_differences.csv (1e-12); subset counts add up to the full val.

Check 2, far-displaced daughters (`losses`). For the primary-population daughters with |d0| >= 300 mm and >= 200 mm on
val, where each model loses them; every particle is put in the first class that applies:
  found       DM-matched by the builder;
  graph       not DM-matched by CC on the graph's chain-true edges (graph-stage ceiling, as hnl.eval.loss_classes);
  classifier  DM-matched by that ceiling but not by CC on the chain-true edges whose score passes the builder's edge cut
              (s0 of the same-layer Mutex Watershed: its attractive edges are those with score > s0; s0 = 0.2 in the
              first pass of learned merging);
  builder     DM-matched by both ceilings, lost by the track builder; sub-classes of hnl.eval.loss_classes:
              fragmented (split: > 50 % of its hits on tracks, over >= 2 tracks, none holding > 50 %), impure (a track
              holds > 50 % of its hits but is <= 50 % pure; merged_other: another particle holds the majority of that
              track) and short (<= 50 % of its hits on any track of >= 2 hits).
Models: Canto-L with its frozen builder (learned merging) and with MWS-layer + merging, acorn (HNL) at its val-best
configuration (MWS-layer + learned merging) and acorn (HNL) + MWS-layer + merging (same-builder row); the run's stored
val scores (RUN/gnn_scores/valset.tar; only the events holding a population particle are extracted), the learned-merging
model of the val grid, the per-seed configuration of the committed entries. Identity checks: the rebuilt DM match of
every stored particle of the processed events equals the stored val grid's at that configuration; the found counts at
|d0| >= 300 mm equal the entries' k_d0_300.

    python -m hnl.eval.round4_checks all --out results --name round4_checks [--workers N] [--tmp DIR]
    (or `valmon` / `losses` alone)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT

REPO = BT.REPO
DATA = BT.DATA
BASE = os.path.join(DATA, "runs", "baseline_v1")
LOGS = "/work/submit/jinboz1/hnl/logs"
ENTRIES = "results/strong_baselines_v3_val_entries.csv"
DIFFS = "results/strong_baselines_v3_val_differences.csv"
SELECTION = "results/final_selection_v1.json"
SEEDS = ("0", "1", "2")
CANTO = "Canto-L (frozen: learned merging)"
C_MWSM = "Canto-L + MWS-layer + merging (Canto builder), same builder (not the frozen builder)"
A_BEST = "acorn (HNL) + val-best builder of the full menu (chosen per seed)"
A_MWSM = "acorn (HNL) + MWS-layer + merging (Canto builder)"
SHORT = {CANTO: "Canto-L (frozen, learned merging)", C_MWSM: "Canto-L + MWS-layer + merging",
         A_BEST: "acorn (HNL) val-best (MWS-layer + learned merging)", A_MWSM: "acorn (HNL) + MWS-layer + merging"}
MODELS1 = (CANTO, A_BEST, A_MWSM, C_MWSM)
PAIRS1 = ((CANTO, A_BEST), (CANTO, A_MWSM), (C_MWSM, A_MWSM))
MODELS2 = (CANTO, C_MWSM, A_BEST, A_MWSM)
HNL_SETS = {"hnl_joint_val": BT.VAL_SETS["hnl_joint_val"]}
HNL_CODE = {2: "hnl_joint_val", 4: "hnl_joint_val"}
B_SUB, SEED_SUB = 1000, 20261007
SUBSETS = ("valmon", "rest")
FAR = (300.0, 200.0)
TOL = BT.TOL
STAGES = ("found", "graph", "classifier", "fragmented", "impure", "short")


def _repo(p: str) -> str:
    return BT._repo(p)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def git_state() -> str:
    h = _git("rev-parse", "HEAD").stdout.strip()
    d = _git("status", "--porcelain", "--", "src", "slurm").stdout.strip()
    return h + (f" (src/slurm dirty: {len(d.splitlines())} files)" if d else "")


# --------------------------------------------------------------------------------------------------------------------
# shared inputs


_PARTICLES = None


def hnl_val_particles() -> pd.DataFrame:
    """hnl.eval.particles.particle_table of the 1,999 HNL val events (read once)."""
    global _PARTICLES
    if _PARTICLES is None:
        from hnl.eval.particles import particle_table
        _PARTICLES = particle_table(BT.expected_events(HNL_SETS))
    return _PARTICLES


def info_targets() -> pd.DataFrame:
    """The columns of hnl.eval.bootstrap_tables.info_table for the HNL val targets."""
    p = hnl_val_particles()
    p = p[["event_id", "particle_id", "is_target", "category", "d0", "rv"]]
    return p[p.is_target.astype(bool)].reset_index(drop=True)


def entries(label: str) -> list[tuple[str, str]]:
    """(run, working point) per seed of one row of the committed val entries."""
    e = pd.read_csv(_repo(ENTRIES))
    e = e[(e.label == label) & (e.set == "val")]
    out = []
    for s in SEEDS:
        r = e[e.run.str.endswith(f"_s{s}")]
        if len(r) != 1:
            raise SystemExit(f"{ENTRIES}: {len(r)} rows for {label} seed {s}")
        out.append((r.run.iloc[0], r.working_point.iloc[0]))
    return out


def check_canto_selection(rows: list[tuple[str, str]]) -> list[tuple]:
    """Canto-L's working points of the entries are the frozen selection's (results/final_selection_v1.json)."""
    sel = json.load(open(_repo(SELECTION)))["pipelines"]["l1tl"]["seeds"]
    checks = []
    for s, (run, wp) in zip(SEEDS, rows):
        fam, params, _ = parse_wp(wp)
        want = sel[s]["config"]
        ok = os.path.normpath(sel[s]["run"]) == os.path.normpath(run) and f"{fam},{params}" == want
        checks.append((f"Canto-L seed {s} working point = final selection", f"{fam},{params}", want, ok))
    return checks


def parse_wp(wp: str) -> tuple[str, str, str]:
    m = re.match(r"^(\S+) (\S+) \((\S+)\)$", wp)
    if m is None:
        raise ValueError(f"working point {wp!r}")
    return m.group(1), m.group(2), m.group(3)


_TABLES: dict = {}


def stored(run: str, wp: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    key = (os.path.normpath(run), wp)
    if key not in _TABLES:
        _TABLES[key] = BT.val_tables(run, wp)
    return _TABLES[key]


# --------------------------------------------------------------------------------------------------------------------
# check 1: valmon overlap


def valmon_events() -> pd.DataFrame:
    """The valmon events of arm (ii), checked against the split files read directly."""
    from hnl.train.acorn_runner import N_VAL_MONITOR, arm_events
    ev = arm_events("ii", "valmon")
    df = pd.DataFrame(ev, columns=["sample", "index"])
    df["event_id"] = df["sample"].map(BT.SAMPLE_CODE).astype(np.int64) * 100_000_000 + df["index"].astype(np.int64)
    k = N_VAL_MONITOR // 2
    for s in ("hnl10_100", "hnl15_200"):
        lines = [int(x) for x in open(_repo(f"splits/{s}_val.txt")) if x.strip() and not x.startswith("#")]
        if df[df["sample"] == s]["index"].tolist() != lines[:k]:
            raise SystemExit(f"valmon events of {s} are not the first {k} lines of splits/{s}_val.txt")
    if len(df) != N_VAL_MONITOR or df.event_id.duplicated().any():
        raise SystemExit("valmon: not 500 distinct events")
    return df


def _arm_events_source(src: str) -> str:
    m = re.search(r"^def arm_events\(.*?(?=^def )", src, re.M | re.S)
    return m.group(0) if m else ""


def training_provenance() -> pd.DataFrame:
    """Per acorn (HNL) training job (ml, gnn x 3 seeds): the commit it ran (run_*.json), whether arm_events and
    N_VAL_MONITOR at that commit equal HEAD's, whether splits/ is unchanged since, the monitored val count, the
    staging line of its log and the selected checkpoint."""
    head = open(_repo("src/hnl/train/acorn_runner.py")).read()
    head_fn = _arm_events_source(head)
    splits_clean = _git("status", "--porcelain", "--", "splits").stdout.strip() == ""
    rows = []
    for s in SEEDS:
        for st in ("ml", "gnn"):
            js = sorted(glob.glob(os.path.join(BASE, f"armii_s{s}", st, "run_*.json")))
            if len(js) != 1:
                raise SystemExit(f"armii_s{s}/{st}: {len(js)} run_*.json")
            d = json.load(open(js[0]))
            src = _git("show", f"{d['git']}:src/hnl/train/acorn_runner.py").stdout
            nvm = re.search(r"^N_VAL_MONITOR = (\d+)", src, re.M)
            same_splits = _git("diff", "--quiet", d["git"], "HEAD", "--", "splits").returncode == 0 and splits_clean
            log = sorted(glob.glob(os.path.join(LOGS, f"bl_{st}_train_ii_s{s}_{d['slurm_job']}.out")))
            staged = None
            if log:
                staged = next((ln.strip() for ln in open(log[0], errors="replace")
                               if ln.startswith("[stage] arm ii valmon:")), None)
            rows.append({"seed": int(s), "stage": st, "git": d["git"], "slurm_job": d["slurm_job"],
                         "arm_events_same_as_head": _arm_events_source(src) == head_fn and bool(head_fn),
                         "n_val_monitor": int(nvm.group(1)) if nvm else None, "splits_unchanged_since": same_splits,
                         "data_split": json.dumps(d["data_split"]), "log": log[0] if log else "",
                         "log_staging_line": staged or "", "best_ckpt": os.path.basename(d["best_ckpt"]),
                         "monitor": "f1 (max)" if st == "ml" else "val_loss (min)"})
    t = pd.DataFrame(rows)
    ok = (t.arm_events_same_as_head & (t.n_val_monitor == 500) & t.splits_unchanged_since
          & (t.data_split == json.dumps([15999, 500, 0])) & t.log_staging_line.str.contains("valmon: 500 events"))
    if not ok.all():
        print(t.to_string())
        raise SystemExit("training provenance: the valmon events of some acorn (HNL) training job are not confirmed")
    return t


def _subset_resampler(units: np.ndarray, rng: np.random.Generator, label: str) -> BT.Resampler:
    w = BS.make_weights(len(units), B_SUB, rng)
    return BT.Resampler({"hnl_joint_val": units}, {"hnl_joint_val": w}, HNL_CODE, label)


def primary_rows(run: str, wp: str, info: pd.DataFrame) -> pd.DataFrame:
    pp, _, d = stored(run, wp)
    t = pp[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    if t.duplicated(["event_id", "particle_id"]).any():
        raise ValueError(f"{d}: duplicate particles at {wp}")
    return t[BT.masks(t)["eff_primary"]].reset_index(drop=True)


def run_valmon(out: str, name: str) -> dict:
    t0 = time.time()
    checks: list[tuple] = []
    vm = valmon_events()
    prov = training_provenance()
    checks += [(f"training {r.stage} seed {r.seed}: valmon definition confirmed", 1.0, 1.0, True)
               for r in prov.itertuples()]
    ev_all = BT.expected_events(HNL_SETS)
    vm_ids = np.sort(vm.event_id.to_numpy(np.int64))
    rest_ids = np.setdiff1d(ev_all, vm_ids)
    if not np.isin(vm_ids, ev_all).all() or len(rest_ids) != len(ev_all) - len(vm_ids):
        raise SystemExit("valmon events outside the HNL val set")
    idx = {"valmon": np.unique(vm_ids % 100_000_000), "rest": np.unique(rest_ids % 100_000_000)}
    children = np.random.SeedSequence(SEED_SUB).spawn(len(SUBSETS))
    rs_sub = {s: _subset_resampler(idx[s], np.random.default_rng(c), f"{s} (B = {B_SUB}, seed {SEED_SUB})")
              for s, c in zip(SUBSETS, children)}
    rs_full = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                                 BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    info = info_targets()
    ent = pd.read_csv(_repo(ENTRIES))
    rows_by_model = {lab: entries(lab) for lab in MODELS1}
    checks += check_canto_selection(rows_by_model[CANTO])
    res: dict = {}   # (model, scope) -> dict(points, reps, k, n)
    for lab in MODELS1:
        for scope in ("full", "valmon", "rest", "valmon_jt", "rest_jt"):
            res[(lab, scope)] = {"pts": [], "reps": [], "k": [], "n": []}
        for s, (run, wp) in zip(SEEDS, rows_by_model[lab]):
            t = primary_rows(run, wp, info)
            ev, ok = t.event_id.to_numpy(np.int64), t.dm_matched.to_numpy(float)
            one = np.ones(len(t))
            in_vm = np.isin(ev, vm_ids)
            parts = {"full": (rs_full, np.ones(len(t), bool)), "valmon": (rs_sub["valmon"], in_vm),
                     "rest": (rs_sub["rest"], ~in_vm), "valmon_jt": (rs_full, in_vm), "rest_jt": (rs_full, ~in_vm)}
            for scope, (rs, m) in parts.items():
                p, r, k, n = rs.ratio(rs.unit_sums(ev[m], ok[m], one[m]))
                for key, v in (("pts", p), ("reps", r), ("k", k), ("n", n)):
                    res[(lab, scope)][key].append(v)
            ref = ent[(ent.label == lab) & (ent.run == run) & (ent.set == "val")].iloc[0]
            f = res[(lab, "full")]
            checks.append((f"{SHORT[lab]} seed {s}: full-val primary = entry", f["pts"][-1], float(ref.eff_primary),
                           abs(f["pts"][-1] - float(ref.eff_primary)) <= TOL))
            kv, nv = res[(lab, "valmon")]["k"][-1], res[(lab, "valmon")]["n"][-1]
            kr, nr = res[(lab, "rest")]["k"][-1], res[(lab, "rest")]["n"][-1]
            checks.append((f"{SHORT[lab]} seed {s}: valmon + rest counts = full", kv + kr + nv + nr,
                           f["k"][-1] + f["n"][-1], kv + kr == f["k"][-1] and nv + nr == f["n"][-1]))
    # identity: full-val differences against the committed table
    dref = pd.read_csv(_repo(DIFFS))
    for new, base in PAIRS1:
        d = BS.linear_combination([(1.0, res[(new, "full")]["pts"], res[(new, "full")]["reps"]),
                                   (-1.0, res[(base, "full")]["pts"], res[(base, "full")]["reps"])])
        r = dref[(dref.new == new) & (dref.baseline == base) & (dref.quantity == "primary")]
        if len(r) != 1:
            raise SystemExit(f"{DIFFS}: {len(r)} rows for {new} - {base}")
        r = r.iloc[0]
        for mine, theirs in (("value", "difference"), ("sigma_boot", "sigma_boot"), ("sigma_seeds", "sigma_seeds"),
                             ("sigma", "sigma")):
            checks.append((f"full-val {SHORT[new]} - {SHORT[base]}: {theirs}", d[mine], float(r[theirs]),
                           abs(d[mine] - float(r[theirs])) <= TOL))
    # model rows
    mrows = []
    n_events = {"full": len(ev_all), "valmon": len(vm_ids), "rest": len(rest_ids)}
    n_units = {"full": len(rs_full.units["hnl_joint_val"]), "valmon": len(idx["valmon"]), "rest": len(idx["rest"])}
    for lab in MODELS1:
        for scope in ("full", "valmon", "rest"):
            r = res[(lab, scope)]
            sm = BS.seed_mean(r["pts"], r["reps"])
            ns = np.asarray(r["n"])
            if not np.all(ns == ns[0]):
                raise SystemExit(f"{lab} {scope}: particle counts differ between seeds {ns}")
            row = {"model": SHORT[lab], "label": lab, "subset": scope, "events": n_events[scope],
                   "units": n_units[scope], "n_per_seed": int(ns[0]),
                   "k_per_seed": " / ".join(str(int(round(k))) for k in r["k"]),
                   "eff_per_seed": " / ".join(f"{p:.6f}" for p in r["pts"]), "mean": sm["mean"],
                   "sigma_boot": sm["sigma_boot"], "s_seeds": sm["s_seeds"], "sigma": sm["sigma"],
                   "B": rs_full.b if scope == "full" else B_SUB,
                   "tables": (f"val tables (B {BT.B_VAL}, seed {BT.SEED_VAL})" if scope == "full"
                              else f"subset table (B {B_SUB}, SeedSequence({SEED_SUB}) child {SUBSETS.index(scope)})"),
                   "working_points": "; ".join(wp for _, wp in rows_by_model[lab])}
            if scope != "full":
                jt = res[(lab, scope + "_jt")]
                row["sigma_boot_full_table"] = BS.seed_mean(jt["pts"], jt["reps"])["sigma_boot"]
            mrows.append(row)
        # the model's own shift between the subsets (rest - valmon), joint replicates
        a, b = res[(lab, "rest_jt")], res[(lab, "valmon_jt")]
        pts = [x - y for x, y in zip(a["pts"], b["pts"])]
        reps = [x - y for x, y in zip(a["reps"], b["reps"])]
        sm = BS.seed_mean(pts, reps)
        mrows.append({"model": SHORT[lab], "label": lab, "subset": "rest - valmon", "events": None, "units": None,
                      "n_per_seed": None, "k_per_seed": "", "eff_per_seed": " / ".join(f"{p:+.6f}" for p in pts),
                      "mean": sm["mean"], "sigma_boot": sm["sigma_boot"], "s_seeds": sm["s_seeds"],
                      "sigma": sm["sigma"], "B": rs_full.b,
                      "tables": f"joint replicates of the val tables (B {BT.B_VAL}, seed {BT.SEED_VAL})",
                      "working_points": "; ".join(wp for _, wp in rows_by_model[lab])})
    models = pd.DataFrame(mrows)
    # differences
    drows = []
    for new, base in PAIRS1:
        for scope in ("full", "valmon", "rest"):
            rn, rb = res[(new, scope)], res[(base, scope)]
            d = BS.linear_combination([(1.0, rn["pts"], rn["reps"]), (-1.0, rb["pts"], rb["reps"])])
            row = {"new": SHORT[new], "baseline": SHORT[base], "subset": scope,
                   "value_new": float(np.mean(rn["pts"])), "value_base": float(np.mean(rb["pts"])),
                   "difference": d["value"], "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"],
                   "sigma": d["sigma"], "z": d["z"],
                   "per_seed_new": " / ".join(f"{p:.6f}" for p in rn["pts"]),
                   "per_seed_base": " / ".join(f"{p:.6f}" for p in rb["pts"])}
            if scope != "full":
                jn, jb = res[(new, scope + "_jt")], res[(base, scope + "_jt")]
                dj = BS.linear_combination([(1.0, jn["pts"], jn["reps"]), (-1.0, jb["pts"], jb["reps"])])
                row["sigma_boot_full_table"] = dj["sigma_boot"]
            drows.append(row)
        # difference of the difference between the subsets: (new - base) on rest minus on valmon, joint replicates;
        # seed terms per model from its per-seed (rest - valmon) values
        nn = [x - y for x, y in zip(res[(new, "rest_jt")]["pts"], res[(new, "valmon_jt")]["pts"])]
        nr = [x - y for x, y in zip(res[(new, "rest_jt")]["reps"], res[(new, "valmon_jt")]["reps"])]
        bn = [x - y for x, y in zip(res[(base, "rest_jt")]["pts"], res[(base, "valmon_jt")]["pts"])]
        br = [x - y for x, y in zip(res[(base, "rest_jt")]["reps"], res[(base, "valmon_jt")]["reps"])]
        d = BS.linear_combination([(1.0, nn, nr), (-1.0, bn, br)])
        drows.append({"new": SHORT[new], "baseline": SHORT[base], "subset": "rest - valmon",
                      "value_new": float(np.mean(nn)), "value_base": float(np.mean(bn)), "difference": d["value"],
                      "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"], "sigma": d["sigma"],
                      "z": d["z"], "per_seed_new": " / ".join(f"{p:+.6f}" for p in nn),
                      "per_seed_base": " / ".join(f"{p:+.6f}" for p in bn)})
    diffs = pd.DataFrame(drows)
    os.makedirs(out, exist_ok=True)
    vm.to_csv(os.path.join(out, f"{name}_valmon_events.csv"), index=False)
    prov.to_csv(os.path.join(out, f"{name}_valmon_training.csv"), index=False)
    models.to_csv(os.path.join(out, f"{name}_valmon_models.csv"), index=False)
    diffs.to_csv(os.path.join(out, f"{name}_valmon_differences.csv"), index=False)
    ck = pd.DataFrame(checks, columns=["check", "value", "reference", "ok"])
    ck.to_csv(os.path.join(out, f"{name}_valmon_checks.csv"), index=False)
    if not ck.ok.all():
        print(ck[~ck.ok].to_string())
        raise SystemExit(f"{int((~ck.ok).sum())} of {len(ck)} checks failed")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print(models.drop(columns=["label", "working_points", "tables"]).to_string(index=False))
    print(diffs.to_string(index=False))
    sha = hashlib.sha256(" ".join(str(x) for x in vm_ids).encode()).hexdigest()
    meta = {"valmon_events": int(len(vm_ids)), "valmon_units": int(len(idx["valmon"])),
            "rest_events": int(len(rest_ids)), "rest_units": int(len(idx["rest"])),
            "valmon_event_ids_sha256": sha,
            "shared_units": [int(x) for x in np.intersect1d(idx["valmon"], idx["rest"])],
            "checks": f"{int(ck.ok.sum())} of {len(ck)}", "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(meta))
    return meta


# --------------------------------------------------------------------------------------------------------------------
# check 2: far-displaced daughters


def far_population() -> pd.DataFrame:
    """Primary-population HNL daughters on val with |d0| >= 200 mm (flag ge300 for |d0| >= 300 mm)."""
    p = hnl_val_particles()
    t = p[p.is_target.astype(bool)].reset_index(drop=True)
    m = BT.masks(t[["event_id", "particle_id", "is_target", "category", "d0", "rv"]])["eff_primary"]
    f = t[m & (t.d0.abs().to_numpy() >= min(FAR))].copy()
    f["ge300"] = f.d0.abs() >= 300.0
    cols = ["event_id", "particle_id", "sample", "d0", "rv", "n_hits", "n_pixel_hits", "r_first", "pt", "eta", "ge300"]
    return f[cols].sort_values(["event_id", "particle_id"]).reset_index(drop=True)


def merge_model(run: str, fam: str, grid: str) -> str | None:
    """The learned-merging model behind a run's stored mwslmerge val rows."""
    if fam != "mwslmerge":
        return None
    d = os.path.join(run, grid)
    if os.path.exists(os.path.join(d, "merged_from.json")):
        for part in json.load(open(os.path.join(d, "merged_from.json")))["merged_from"]:
            cfgs = json.load(open(os.path.join(part, "configs.json")))
            if any(c[0] == "mwslmerge" for c in cfgs):
                return json.load(open(os.path.join(part, "provenance.json")))["merge_model"]
        raise SystemExit(f"{d}: no part holds mwslmerge")
    if os.path.exists(os.path.join(d, "provenance.json")):
        return json.load(open(os.path.join(d, "provenance.json")))["merge_model"]
    sel = json.load(open(_repo(SELECTION)))["pipelines"]
    for p in sel.values():
        for s in p.get("seeds", {}).values():
            if os.path.normpath(s["run"]) == os.path.normpath(run) and grid == "trackbuild_val_lmerge":
                return s["merge_model"]
    raise SystemExit(f"{d}: merge model unknown")


def build_labels(fam: str, param: float, n, a, b, sc, r, phi, z, geo, merger) -> np.ndarray:
    """Track labels exactly as hnl.eval.trackbuild_val.evaluate_event for the two families used here."""
    if fam == "mwslayermerge":
        from hnl.trackbuild.merge_fragments import merge_fragments
        from hnl.trackbuild.mutex_ws import mws_layer
        lab = mws_layer(n, a, b, sc, geo["layer_id"], param)
        lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
        return merge_fragments(r, phi, z, lab, max_rms=1.0)
    if fam == "mwslmerge":
        from hnl.trackbuild import learned_merge as LM
        first = LM.first_pass(n, a, b, sc, geo["layer_id"])
        pairs, p = merger.prepare(first, r, phi, z, geo)
        return LM.apply(first, pairs, p, param, geo["layer_id"])
    raise ValueError(fam)


def edge_cut(fam: str, param: float) -> float:
    """The builder's edge cut: s0 of the same-layer Mutex Watershed (attractive edges have score > s0)."""
    if fam == "mwslayermerge":
        return float(param)
    if fam == "mwslmerge":
        from hnl.trackbuild.learned_merge import FIRST_S0
        return float(FIRST_S0)
    raise ValueError(fam)


def classify_event(path: str, specs: list[dict], geo: dict, mergers: dict, far_ids: np.ndarray):
    """Per spec: the far particles' classes and diagnostics, and the rebuilt DM match of every particle."""
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval import matching as M
    from hnl.eval.loss_classes import particle_classes
    from hnl.eval.trackbuild_val import load_scored
    from hnl.graphs import ceiling as C
    from hnl.trackbuild import builders as TB
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    eid = int(eid)
    n = len(pid)
    if len(geo["layer_id"]) != n:
        raise ValueError(f"{path}: {len(geo['layer_id'])} hits in the hits table, {n} in the graph")
    g = load_pyg(path[:-3], weights_only=False)
    te = C.unordered_unique(g.track_edges.numpy())
    code = a.astype(np.int64) * n + b
    tcode = te[0] * n + te[1]
    chain = np.isin(code, tcode)
    one = np.ones(int(chain.sum()))
    graph_lab = TB.cc(n, a[chain], b[chain], one, 0.5)
    has_edge = np.zeros(n, dtype=bool)
    has_edge[a] = True
    has_edge[b] = True
    fm = M.FastMatch(pid)
    g_dm = fm.match(graph_lab)["dm_matched"]
    # per true chain edge: in the graph, its score
    order = np.argsort(code, kind="stable")
    pos = np.searchsorted(code[order], tcode)
    pos_c = np.minimum(pos, len(code) - 1)
    in_g = (pos < len(code)) & (code[order][pos_c] == tcode) if len(code) else np.zeros(len(tcode), bool)
    t_sc = np.where(in_g, sc[order][pos_c] if len(code) else 0.0, np.nan)
    t_pid = pid[te[0]]
    out_far, out_dm = [], []
    far = far_ids[np.isin(far_ids, fm.particles)]
    for sp in specs:
        lab = build_labels(sp["family"], sp["param"], n, a, b, sc, r, phi, z, geo, mergers.get(sp["merge_model"]))
        s0 = sp["cut"]
        keep = chain & (sc > s0)
        s_lab = TB.cc(n, a[keep], b[keep], np.ones(int(keep.sum())), 0.5)
        s_dm = fm.match(s_lab)["dm_matched"]
        m = fm.match(lab)
        out_dm.append(pd.DataFrame({"key": sp["key"], "event_id": np.int64(eid), "particle_id": fm.particles,
                                    "dm_rebuilt": m["dm_matched"]}))
        d = particle_classes(pid, lab, graph_lab, has_edge).set_index("particle_id")
        for p in far:
            i = int(np.searchsorted(fm.particles, p))
            hits = np.flatnonzero(pid == p)
            lab_p = lab[hits]
            on = lab_p >= 0
            best_t, maj, best_size, maj_frac = -1, 0, 0, np.nan
            if on.any():
                tu, tc = np.unique(lab_p[on], return_counts=True)
                best_t = int(tu[np.argmax(tc)])
                tp, tcn = np.unique(pid[lab == best_t], return_counts=True)
                best_size = int(tcn.sum())
                k = int(np.argmax(tcn))
                maj = int(tp[k]) if 2 * tcn[k] > tcn.sum() else 0
                maj_frac = float(tcn[k] / tcn.sum())
            row = d.loc[p]
            tm = t_pid == p
            if m["dm_matched"][i]:
                stage = "found"
            elif not g_dm[i]:
                stage = "graph"
            elif not s_dm[i]:
                stage = "classifier"
            else:
                stage = str(row.cls)
                if stage not in ("fragmented", "impure", "short"):
                    raise ValueError(f"{path} {p}: builder loss with loss class {stage}")
            out_far.append({"key": sp["key"], "event_id": np.int64(eid), "particle_id": np.int64(p), "stage": stage,
                            "loss_class": str(row.cls), "graph_ceiling_dm": bool(g_dm[i]),
                            "score_ceiling_dm": bool(s_dm[i]), "edge_cut": s0, "n_hits": int(row.n_hits),
                            "n_hits_no_edge": int(row.n_hits_no_edge), "n_true_chain": int(tm.sum()),
                            "n_chain_in_graph": int(in_g[tm].sum()),
                            "n_chain_above_cut": int((in_g[tm] & (np.nan_to_num(t_sc[tm], nan=-1.0) > s0)).sum()),
                            "min_chain_score": float(np.nanmin(t_sc[tm])) if in_g[tm].any() else np.nan,
                            "best_share": int(row.best_share), "best_purity": float(row.best_purity),
                            "n_tracks": int(row.n_tracks), "n_on_tracks": int(row.n_on_tracks),
                            "one_short": bool(row.one_short), "best_track_size": best_size,
                            "best_track_majority_pid": maj, "best_track_majority_frac": maj_frac,
                            "merged_other": bool(maj != 0 and maj != p)})
    return pd.DataFrame(out_far), pd.concat(out_dm, ignore_index=True)


def _run_task(task: dict):
    """One run: extract the needed val graphs from RUN/gnn_scores/valset.tar, classify every spec on them."""
    from hnl.trackbuild.learned_merge import Merger
    tar_path = os.path.join(task["run"], "gnn_scores", "valset.tar")
    want = {f"event{e}.pyg.gz" for e in task["events"]}
    d = tempfile.mkdtemp(prefix="r4_", dir=task["tmp"])
    n = 0
    with tarfile.open(tar_path, "r") as tar:
        for mem in tar:
            nm = os.path.basename(mem.name)
            if nm in want:
                mem.name = nm
                tar.extract(mem, d, filter="data")
                n += 1
    if n != len(want):
        raise RuntimeError(f"{tar_path}: extracted {n} of {len(want)} graphs")
    mergers = {sp["merge_model"]: Merger(sp["merge_model"]) for sp in task["specs"] if sp["merge_model"]}
    far, dm = [], []
    for e in sorted(task["events"]):
        f, m = classify_event(os.path.join(d, f"event{e}.pyg.gz"), task["specs"], task["geo"][e], mergers,
                              task["far"][e])
        far.append(f)
        dm.append(m)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    os.rmdir(d)
    return pd.concat(far, ignore_index=True), pd.concat(dm, ignore_index=True), tar_path


FAR_COLS = ["model", "seed", "event_id", "particle_id", "sample", "d0", "rv", "n_hits", "n_hits_table",
            "n_pixel_hits", "r_first", "pt", "eta", "ge300", "stage", "loss_class", "graph_ceiling_dm",
            "score_ceiling_dm", "edge_cut", "n_hits_no_edge", "n_true_chain", "n_chain_in_graph", "n_chain_above_cut",
            "min_chain_score", "best_share", "best_purity", "n_tracks", "n_on_tracks", "one_short", "best_track_size",
            "best_track_majority_pid", "best_track_majority_frac", "merged_other", "wp", "label"]


def summarize_far(far: pd.DataFrame) -> pd.DataFrame:
    """Counts per model, seed (and pooled over the seeds) and |d0| threshold."""
    srows = []
    for lo in FAR:
        sel = far[far.d0.abs() >= lo]
        for lab in MODELS2:
            for s in (*[int(x) for x in SEEDS], "pooled"):
                g = sel[sel.label == lab] if s == "pooled" else sel[(sel.label == lab) & (sel.seed == s)]
                c = g.stage.value_counts()
                srows.append({"d0_min_mm": lo, "model": SHORT[lab], "seed": s, "n": len(g),
                              **{k: int(c.get(k, 0)) for k in STAGES},
                              "lost": int(len(g) - c.get("found", 0)),
                              "impure_merged_other": int(((g.stage == "impure") & g.merged_other.astype(bool)).sum()),
                              "found_below_cut_rescued": int(((g.stage == "found")
                                                              & ~g.score_ceiling_dm.astype(bool)).sum()),
                              "graph_one_short": int(((g.stage == "graph") & g.one_short.astype(bool)).sum()),
                              "working_points": "; ".join(dict.fromkeys(g.wp))})
    return pd.DataFrame(srows)


def run_losses(out: str, name: str, workers: int, tmp: str) -> dict:
    from hnl.geometry.resolution import hit_geometry
    t0 = time.time()
    checks: list[tuple] = []
    pop = far_population()
    events = np.unique(pop.event_id.to_numpy(np.int64))
    far_by_event = {int(e): np.sort(g.particle_id.to_numpy(np.int64)) for e, g in pop.groupby("event_id")}
    geo = hit_geometry(events)
    specs_by_run: dict = {}
    specs = []
    ent = pd.read_csv(_repo(ENTRIES))
    rows = {lab: entries(lab) for lab in MODELS2}
    checks += check_canto_selection(rows[CANTO])
    for lab in MODELS2:
        for s, (run, wp) in zip(SEEDS, rows[lab]):
            fam, params, grid = parse_wp(wp)
            mm = merge_model(run, fam, grid)
            sp = {"key": f"{lab}|{s}", "label": lab, "seed": int(s), "run": run, "wp": wp, "family": fam,
                  "param": float(params), "grid": grid, "merge_model": mm, "cut": edge_cut(fam, float(params)),
                  "merge_model_sha256": _sha256(mm) if mm else ""}
            specs.append(sp)
            specs_by_run.setdefault(os.path.normpath(run), []).append(sp)
    tasks = [{"run": run, "specs": sps, "events": [int(e) for e in events], "geo": geo, "far": far_by_event,
              "tmp": tmp} for run, sps in specs_by_run.items()]
    with cf.ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as ex:
        res = list(ex.map(_run_task, tasks))
    far = pd.concat([r[0] for r in res], ignore_index=True)
    dm = pd.concat([r[1] for r in res], ignore_index=True)
    # identity: rebuilt DM match of every stored particle of the processed events = the stored val grid
    for sp in specs:
        pp, _, d = stored(sp["run"], sp["wp"])
        st = pp[pp.event_id.isin(events)][["event_id", "particle_id", "dm_matched"]]
        mine = dm[dm.key == sp["key"]]
        j = st.merge(mine, on=["event_id", "particle_id"], how="left")
        ok = j.dm_rebuilt.notna().all() and bool((j.dm_rebuilt.astype(bool) == j.dm_matched.astype(bool)).all())
        checks.append((f"{SHORT[sp['label']]} seed {sp['seed']}: rebuilt DM = stored ({d}, {len(j)} particles of "
                       f"{len(events)} events)", float(j.dm_rebuilt.fillna(-1).astype(float).sum()),
                       float(j.dm_matched.astype(float).sum()), ok))
    far = far.merge(pd.DataFrame([{k: sp[k] for k in ("key", "label", "seed", "wp", "cut")} for sp in specs]),
                    on="key", how="left").drop(columns="key")
    far = far.merge(pop.rename(columns={"n_hits": "n_hits_table"}), on=["event_id", "particle_id"], how="left")
    far["model"] = far.label.map(SHORT)
    if len(far) != len(pop) * len(specs):
        raise SystemExit(f"{len(far)} classified rows, expected {len(pop)} x {len(specs)}")
    checks.append(("far particles: hits in the graph event = n_hits of the particle table",
                   float((far.n_hits == far.n_hits_table).sum()), float(len(far)),
                   bool((far.n_hits == far.n_hits_table).all())))
    # found counts at |d0| >= 300 mm = the entries' k_d0_300 (and n_d0_300)
    for sp in specs:
        e = ent[(ent.label == sp["label"]) & (ent.run == sp["run"]) & (ent.set == "val")].iloc[0]
        f = far[(far.label == sp["label"]) & (far.seed == sp["seed"]) & far.ge300]
        checks.append((f"{SHORT[sp['label']]} seed {sp['seed']}: |d0| >= 300 found = entry k_d0_300",
                       float((f.stage == "found").sum()), float(e.k_d0_300), int((f.stage == "found").sum()) ==
                       int(e.k_d0_300) and len(f) == int(e.n_d0_300)))
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, f"{name}_far_raw.parquet")    # kept only if a later step fails
    far.to_parquet(raw, index=False)
    summ = summarize_far(far)
    far = far[FAR_COLS].sort_values(["model", "seed", "event_id", "particle_id"])
    far.to_csv(os.path.join(out, f"{name}_far_particles.csv"), index=False)
    summ.to_csv(os.path.join(out, f"{name}_far_summary.csv"), index=False)
    ck = pd.DataFrame(checks, columns=["check", "value", "reference", "ok"])
    ck.to_csv(os.path.join(out, f"{name}_far_checks.csv"), index=False)
    json.dump({"specs": [{k: v for k, v in sp.items()} for sp in specs], "events": [int(e) for e in events],
               "tars": sorted({r[2] for r in res})},
              open(os.path.join(out, f"{name}_far_inputs.json"), "w"), indent=1)
    if not ck.ok.all():
        print(ck[~ck.ok].to_string())
        raise SystemExit(f"{int((~ck.ok).sum())} of {len(ck)} checks failed")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.drop(columns="working_points").to_string(index=False))
    os.remove(raw)
    meta = {"population_ge200": int(len(pop)), "population_ge300": int(pop.ge300.sum()), "events": int(len(events)),
            "checks": f"{int(ck.ok.sum())} of {len(ck)}", "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(meta))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["valmon", "losses", "all"])
    ap.add_argument("--out", default=_repo("results"))
    ap.add_argument("--name", default="round4_checks")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--tmp", default=os.environ.get("LOCAL", tempfile.gettempdir()))
    a = ap.parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    meta = {"git": git_state(), "slurm_job": os.environ.get("SLURM_JOB_ID", "local"), "host": os.uname().nodename,
            "started_utc": started}
    if a.cmd in ("valmon", "all"):
        meta["valmon"] = run_valmon(a.out, a.name)
    if a.cmd in ("losses", "all"):
        meta["losses"] = run_losses(a.out, a.name, a.workers, a.tmp)
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = os.path.join(a.out, f"{a.name}_meta_{a.cmd}.json")
    json.dump(meta, open(path, "w"), indent=1)
    print(path)


if __name__ == "__main__":
    main()
