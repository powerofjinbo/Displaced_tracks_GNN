"""Feedback round 1, supervisor point 2: the strongest acorn baselines on val, at their val-chosen configurations on test
(post-freeze descriptive), and the like-for-like differences against Canto-L.

acorn (HNL) = arm (ii), HNL-trained as in Wang et al.; acorn (ttbar) = arm (i); three seeds each, the GNN scores of the
Stage-0 baseline runs (runs/baseline_v1/arm<A>_s<S>). Track builders on those scores:
  acwalk     acorn's own CC + walkthrough builder (CCandWalk, plain walk as in acorn's CTD 2023 configuration;
             hnl.trackbuild.acorn_walk), grid RUN/trackbuild_val_acwalk_all of hnl.eval.trackbuild_val (`merge`: the
             union of RUN/trackbuild_val_acwalk, 19 configurations per variant, and its lower-edge extension
             RUN/trackbuild_val_acwalk_ext, 8 per variant);
  acwalkg    the same with the score-guided walk (guidance mode 2, as in acorn's Rel. 24 / ACAT 2025 configurations),
             same grid directory;
  walkmerge  our walkthrough + helix fragment merging (RUN/trackbuild_val_ext, the family's first grid);
  mwsm       same-layer Mutex Watershed + merging (mwslayermerge; grids and family exactly as the ranking v3 row
             "acorn IGNN, ... + Canto builder (MWS-layer + merging)").
Configuration per seed: the builder rule at FR* with the one-standard-error tie rule (hnl.eval.ranking with tie_se, as
ranking v3 and the frozen selection). Reference rows: Canto-L and acorn + connected components exactly as in ranking v3
(val, results/ranking/spec_val_v3.json) and ranking_test_v1 (test, results/ranking/spec_test_v1.json).

Test, post-freeze descriptive (not claims; the preregistered claims stay as they are): every configuration fixed on val
is applied unchanged to the run's stored test scores (slurm/feedback/baseline_test.sbatch ->
runs/feedback_v1/strong_baselines/arm<A>_s<S>/<key>_test, configuration index 0, provenance checked) and summarised by
the code path of hnl.eval.ranking_test (hnl.eval.test_stats populations, sets and estimators). Differences Canto-L minus
each baseline with the claims' estimator (hnl.eval.bootstrap.linear_combination: paired event bootstrap of the seed
means, B = 500, plus both seed terms): primary, HNL daughters with |d0| in [100, 300) mm and >= 300 mm, all HNL
daughters, W muons, ttbar prompt and the track fake rate.

    python -m hnl.eval.strong_baselines merge                                            (acwalk grid union)
    python -m hnl.eval.strong_baselines val  --out results --name strong_baselines_v1   (val table + test spec)
    python -m hnl.eval.strong_baselines jobs --out results --name strong_baselines_v1   (sbatch commands for test)
    python -m hnl.eval.strong_baselines test --out results --name strong_baselines_v1   (test table, differences, md)
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import ranking as RK
from hnl.eval import ranking_test as RT
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table

DATA = RT.DATA
BASE = os.path.join(DATA, "runs", "baseline_v1")
FB = os.path.join(DATA, "runs", "feedback_v1", "strong_baselines")
V3_SOURCES = ["trackbuild_val_ext", "trackbuild_val_lmerge", "trackbuild_val_mwslayer", "trackbuild_val_mws"]
SEEDS = ("0", "1", "2")
ARMS = (("ii", "acorn (HNL)"), ("i", "acorn (ttbar)"))
ACWALK_PARTS = ("trackbuild_val_acwalk", "trackbuild_val_acwalk_ext")   # 38 + 16 configurations (lower-edge extension)
ACWALK_ALL = "trackbuild_val_acwalk_all"
BUILDERS = (  # key, family, val grids (first grid holding the family), label, short
    ("acwalk", "acwalk", [ACWALK_ALL], "acorn's CC + walkthrough (CCandWalk)", "CCandWalk"),
    ("acwalkg", "acwalkg", [ACWALK_ALL], "acorn's CC + walkthrough, score-guided (CCandWalk)",
     "CCandWalk-g"),
    ("walkmerge", "walkmerge", V3_SOURCES, "walkthrough + merging (Canto builder family)", "walk+M"),
    ("mwsm", "mwslayermerge", V3_SOURCES, "MWS-layer + merging (Canto builder)", "MWS+M"),
)
REF_VAL = (  # ranking v3 labels -> our labels / shorts
    ("Canto-L (full budget): learned triplet union, learned merging (rule R6)", "Canto-L (frozen: learned merging)",
     "Canto-L"),
    ("acorn IGNN, HNL-trained + CC (best cut <= FR*)", "acorn (HNL) + CC, best cut <= FR*", "acorn (HNL) CC*"),
    ("acorn IGNN, HNL-trained + CC (prereg. cut rule)", "acorn (HNL) + CC, preregistered cut (claims' reference)",
     "acorn (HNL) CC"),
    ("acorn IGNN, ttbar-trained + CC (prereg. cut rule)", "acorn (ttbar) + CC, preregistered cut (= best cut)",
     "acorn (ttbar) CC"),
)
NEW = "Canto-L (frozen: learned merging)"
HEADLINE = "acorn (HNL) + MWS-layer + merging (Canto builder)"
CANTO_SAME = "Canto-L + MWS-layer + merging (same builder as the headline baseline; not the frozen builder)"
SEL_SEEDS = "results/final_selection_v1_seeds.csv"


def _repo(p: str) -> str:
    return RT._repo(p)


def fr_star() -> float:
    return float(json.load(open(_repo(RT.WP)))["FR_star"])


def val_spec(skip_missing: bool = False) -> list[dict]:
    spec = []
    v3 = {m["label"]: m for m in json.load(open(_repo("results/ranking/spec_val_v3.json")))}
    for lab3, lab, short in REF_VAL:
        m = dict(v3[lab3])
        spec.append({**m, "label": lab, "short": short, "group": "reference", "v3_label": lab3})
    for arm, alab in ARMS:
        for key, fam, src, blab, bshort in BUILDERS:
            runs = [os.path.join(BASE, f"arm{arm}_s{s}") for s in SEEDS]
            have = all(any(os.path.exists(os.path.join(r, g, "configs.json")) for g in src) for r in runs)
            if not have:
                if skip_missing:
                    print(f"WARNING: {alab} + {key}: grid {src} missing for some seed, row skipped")
                    continue
                raise SystemExit(f"{alab} + {key}: grid {src} missing")
            spec.append({"label": f"{alab} + {blab}", "short": f"{alab.replace('acorn ', '')} {bshort}",
                         "group": alab, "key": key, "prefix": f"arm{arm}",
                         "entries": [{"run": r, "source": src, "family": fam} for r in runs]})
    # Canto-L with the headline baseline's builder (same-builder check; the frozen Canto-L keeps learned merging)
    cl = v3["Canto-L (full budget): learned triplet union, learned merging (rule R6)"]
    spec.append({"label": CANTO_SAME, "short": "Canto-L MWS+M", "group": "Canto-L, same builder", "key": "mwsm",
                 "prefix": "canto_l", "entries": [{"run": e["run"], "source": e["source"], "family": "mwslayermerge"}
                                                  for e in cl["entries"]]})
    return spec


def merge_grids(run: str, parts=ACWALK_PARTS, out=ACWALK_ALL) -> str:
    """RUN/out = the union of the track-building grids RUN/parts (configuration indices of later parts shifted; same
    events and particles required; no configuration twice). A data copy, nothing recomputed; idempotent."""
    cfgs, pps, pes, off = [], [], [], 0
    for g in parts:
        d = os.path.join(run, g)
        c = json.load(open(os.path.join(d, "configs.json")))
        pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
        pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
        pp["config"] = (pp.config.astype(np.int32) + off).astype(np.int16)
        pe["config"] = (pe.config.astype(np.int32) + off).astype(np.int64)
        cfgs += c
        pps.append(pp)
        pes.append(pe)
        off += len(c)
    keys = [tuple(x) for x in cfgs]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{run}: a configuration appears in two of {parts}")
    ev = [set(pe.event_id) for pe in pes]
    pk = [set(zip(pp.event_id[pp.config == pp.config.min()], pp.particle_id[pp.config == pp.config.min()])) for pp in pps]
    if any(e != ev[0] for e in ev) or any(k != pk[0] for k in pk):
        raise SystemExit(f"{run}: the grids {parts} cover different events or particles")
    d = os.path.join(run, out)
    pp, pe = pd.concat(pps, ignore_index=True), pd.concat(pes, ignore_index=True)
    if os.path.exists(os.path.join(d, "configs.json")):
        old = json.load(open(os.path.join(d, "configs.json")))
        same = old == json.loads(json.dumps(cfgs)) and len(pd.read_parquet(os.path.join(d, "per_particle.parquet"))) == len(pp)
        if not same:
            raise SystemExit(f"{d} exists with other content")
        return d
    os.makedirs(d, exist_ok=True)
    pp.to_parquet(os.path.join(d, "per_particle.parquet"), index=False)
    pe.to_parquet(os.path.join(d, "per_event.parquet"), index=False)
    json.dump(cfgs, open(os.path.join(d, "configs.json"), "w"))
    json.dump({"merged_from": [os.path.join(run, g) for g in parts], "n_configs": len(cfgs),
               "note": "union of the grids (configuration indices of later parts shifted); hnl.eval.strong_baselines merge"},
              open(os.path.join(d, "merged_from.json"), "w"), indent=1)
    return d


def _parse_wp(wp: str) -> tuple[str, str, str]:
    """'mwslayermerge 0.1 (trackbuild_val_ext)' -> (family, params, grid)."""
    m = re.match(r"^(\S+) (\S+) \((\S+)\)$", wp)
    if m is None:
        raise ValueError(f"cannot parse working point {wp!r}")
    return m.group(1), m.group(2), m.group(3)


def run_val(out: str, name: str, skip_missing: bool = False):
    fr = fr_star()
    spec = val_spec(skip_missing)
    tab, ent = RK.rank(spec, fr, tie_se=True)
    tab.insert(0, "set", "val")
    ent.insert(0, "set", "val")
    # the test spec: each builder row's per-seed configuration, fixed here on val
    tspec = []
    for m in spec:
        if "key" not in m:
            continue
        es = []
        for e in m["entries"]:
            r = ent[(ent.label == m["label"]) & (ent.run == e["run"])]
            if len(r) != 1:
                raise SystemExit(f"{len(r)} val entries for {m['label']} / {e['run']}")
            fam, params, grid = _parse_wp(r.working_point.iloc[0])
            seed = re.search(r"_s(\d+)$", e["run"]).group(1)
            es.append({"run": e["run"], "sub": os.path.join(FB, f"{m['prefix']}_s{seed}", f"{m['key']}_test"),
                       "cut": 0, "config": f"{fam},{params}", "val_grid": grid,
                       "wp": f"{fam} {params} (chosen on val in {grid}; test configuration 0)"})
        tspec.append({"label": m["label"], "short": m["short"], "group": m["group"], "key": m["key"],
                      "prefix": m["prefix"], "entries": es})
    # the Canto-L same-builder configurations must be those of the frozen selection's seed table (val)
    ss = pd.read_csv(_repo(SEL_SEEDS))
    ss = ss[(ss.pipeline == "l1tl") & (ss.family == "mwslayermerge")].set_index("seed")
    for e in next(m for m in tspec if m["label"] == CANTO_SAME)["entries"]:
        seed = int(re.search(r"_s(\d+)$", e["run"]).group(1))
        if ss.loc[seed, "config"] != e["config"]:
            raise SystemExit(f"Canto-L MWS-layer + merging seed {seed}: {e['config']} != {ss.loc[seed, 'config']}")
    os.makedirs(out, exist_ok=True)
    json.dump(spec, open(os.path.join(out, f"{name}_spec_val.json"), "w"), indent=1)
    json.dump({"fr_star": fr, "rule": "builder rule at FR* with the one-standard-error tie rule (hnl.eval.ranking, "
                                       "tie_se=True), per seed, on val", "models": tspec},
              open(os.path.join(out, f"{name}_spec_test.json"), "w"), indent=1)
    tab.to_csv(os.path.join(out, f"{name}_val.csv"), index=False)
    ent.to_csv(os.path.join(out, f"{name}_val_entries.csv"), index=False)
    vd = val_differences(spec, ent)
    vd.to_csv(os.path.join(out, f"{name}_val_differences.csv"), index=False)
    pd.set_option("display.width", 250)
    print(RK.markdown(tab.drop(columns="set"), fr))
    for m in tspec:
        print(m["label"], "|", "; ".join(e["config"] for e in m["entries"]))
    return tab, ent, tspec


def pairs(labels: list[str]) -> list[tuple[str, str]]:
    """(new, baseline): the frozen Canto-L against every other row, and Canto-L with the headline baseline's builder
    against that baseline (same builder on both sides)."""
    out = [(NEW, b) for b in labels if b not in (NEW, CANTO_SAME)]
    if CANTO_SAME in labels and HEADLINE in labels:
        out.append((CANTO_SAME, HEADLINE))
    return out


VAL_Q = (("primary", "eff_primary"), ("hnl_d0_100_300", "d0_100_300"), ("hnl_d0_ge300", "d0_300_inf"),
         ("hnl_targets", "eff_hnl_targets"), ("w_muons", "eff_w_muons"), ("ttbar_prompt", "eff_ttbar_prompt"),
         ("fake_rate", "fake_rate_hnl"))


def val_differences(spec: list[dict], ent: pd.DataFrame) -> pd.DataFrame:
    """Val: Canto-L minus each row with the event bootstrap of feedback point 5 (hnl.eval.bootstrap_tables: its val
    replicate tables, B = 1000, seed 20261005, events paired across models; seed terms as the claims), at the working
    points of the val ranking above; also each row's combined sigma."""
    from hnl.eval import bootstrap_tables as BT
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    models = {}
    for m in spec:
        runs = [e["run"] for e in m["entries"]]
        e = ent[ent.label == m["label"]].set_index("run")
        wps = [e.loc[r, "working_point"] for r in runs]
        models[m["label"]] = BT.Model(m["label"], m.get("short", ""), [ld.seed(r, w) for r, w in zip(runs, wps)], rs,
                                      wps, runs, m.get("group", ""))
    rows = []
    for a, lab in pairs(list(models)):
        new, mm = models[a], models[lab]
        for q, bq in VAL_Q:
            d = BT.difference(new, mm, bq)
            sn, sb = BT.summary(new, bq), BT.summary(mm, bq)
            rows.append({"new": a, "baseline": lab, "quantity": q, "n_per_seed": sb.get("n_per_seed"),
                         "value_new": sn["value"], "sigma_new": sn["combined"], "s_seeds_new": sn["seed_sd"],
                         "value_base": sb["value"], "sigma_base": sb["combined"], "s_seeds_base": sb["seed_sd"],
                         "difference": d["value"], "sigma_boot": d["boot_sd"], "sigma_seeds": d["seed_sd"],
                         "sigma": d["combined"], "z": d["z"], "B": d["B"]})
    return pd.DataFrame(rows)


def jobs(out: str, name: str) -> list[str]:
    """sbatch commands (one per run and builder key) for the test outputs that do not exist yet."""
    ts = json.load(open(os.path.join(out, f"{name}_spec_test.json")))["models"]
    cmds = []
    by_run: dict = {}
    for m in ts:
        for e in m["entries"]:
            if os.path.exists(os.path.join(e["sub"], "per_particle.parquet")):
                continue
            by_run.setdefault(e["run"], []).append((os.path.basename(e["sub"]), e["config"], os.path.dirname(e["sub"])))
    for run, items in by_run.items():
        for nm, cfg, root in items:
            cmds.append(f"env -u OMP_NUM_THREADS sbatch -J fb_test_{nm}_{os.path.basename(run)} "
                        f"slurm/feedback/baseline_test.sbatch {run} {root} {nm}={cfg}")
    return cmds


def _check_test_outputs(m: dict):
    for e in m["entries"]:
        prov = json.load(open(os.path.join(e["sub"], "provenance.json")))
        cfgs = json.load(open(os.path.join(e["sub"], "configs.json")))
        want = e["config"].split(",")
        got = [str(x) for x in cfgs[0]]
        same = len(cfgs) == 1 and got[0] == want[0] and np.allclose([float(x) for x in got[1:]],
                                                                    [float(x) for x in want[1:]])
        if not (same and os.path.normpath(prov["run"]) == os.path.normpath(e["run"]) and prov["config"] == e["config"]):
            raise SystemExit(f"{e['sub']}: configuration {cfgs} / provenance {prov} is not the val-chosen {e['config']}")


def test_spec(out: str, name: str) -> list[dict]:
    ts = json.load(open(os.path.join(out, f"{name}_spec_test.json")))["models"]
    ref = {m["label"]: m for m in json.load(open(_repo("results/ranking/spec_test_v1.json")))}
    spec = []
    for lab3, lab, short in REF_VAL:
        m = dict(ref[lab3])
        spec.append({**m, "label": lab, "short": short, "group": "reference"})
    for m in ts:
        _check_test_outputs(m)
        spec.append({k: m[k] for k in ("label", "short", "group")}
                    | {"entries": [{k: e[k] for k in ("run", "sub", "cut", "wp")} for e in m["entries"]]})
    return spec


def _d0_mask(lo: float, hi: float):
    M = TS.masks()
    return lambda t: M["hnl_targets"](t) & (t.d0.abs() >= lo) & (t.d0.abs() < hi)


QUANTITIES = (  # name, kind, population mask (efficiency) or columns (event ratio), test sets
    ("primary", "eff", lambda: TS.masks()["primary"], TS.HNL_SETS),
    ("hnl_d0_100_300", "eff", lambda: _d0_mask(100.0, 300.0), TS.HNL_SETS),
    ("hnl_d0_ge300", "eff", lambda: _d0_mask(300.0, np.inf), TS.HNL_SETS),
    ("hnl_targets", "eff", lambda: TS.masks()["hnl_targets"], TS.HNL_SETS),
    ("w_muons", "eff", lambda: TS.masks()["w_muons"], TS.HNL_SETS),
    ("ttbar_prompt", "eff", lambda: TS.masks()["ttbar_prompt"], ("ttbar_test",)),
    ("fake_rate", "ratio", None, TS.HNL_SETS),
)


def differences(spec: list[dict], prs: list[tuple[str, str]]) -> pd.DataFrame:
    """new minus baseline for each pair, per quantity, with the claims' estimator (paired bootstrap + seed terms)."""
    bs = BS.Bootstrap()
    want = {x for p in prs for x in p}
    models = {m["label"]: [TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"]) for e in m["entries"]]
              for m in spec if m["label"] in want}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for runs in models.values() for rt in runs]))
    info = particle_table(ev)
    vals = {}
    for q, kind, mask, sets in QUANTITIES:
        for lab, runs in models.items():
            if kind == "eff":
                p, r, n = TS.efficiency(bs, runs, info, mask(), sets)
            else:
                p, r = TS.event_ratio(bs, runs, "n_fake", "n_tracks_ge5", sets)
                n = None
            vals[(q, lab)] = (p, r, n)
    rows = []
    for new, b in prs:
        for q, *_ in QUANTITIES:
            pn, rn, nn = vals[(q, new)]
            pb, rb, nb = vals[(q, b)]
            d = BS.linear_combination([(1.0, pn, rn), (-1.0, pb, rb)])
            sn, sb = BS.seed_mean(pn, rn), BS.seed_mean(pb, rb)
            rows.append({"new": new, "baseline": b, "quantity": q, "n_per_seed": nb,
                         "value_new": sn["mean"], "sigma_new": sn["sigma"], "s_seeds_new": sn["s_seeds"],
                         "value_base": sb["mean"], "sigma_base": sb["sigma"], "s_seeds_base": sb["s_seeds"],
                         "difference": d["value"], "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"],
                         "sigma": d["sigma"], "z": d["z"]})
    return pd.DataFrame(rows)


PER_SET = ("hnl10_100_test", "hnl15_200_test", "hnl15_100_external")


def per_set(spec: list[dict], prs: list[tuple[str, str]]) -> pd.DataFrame:
    """Primary efficiency per test set (descriptive) for the given pairs: each model's seed mean and total sigma and the
    paired difference (same estimator; hnl10_100_test and hnl15_200_test on the joint set's resampling units)."""
    bs = BS.Bootstrap()
    want = {x for p in prs for x in p}
    models = {m["label"]: [TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"]) for e in m["entries"]]
              for m in spec if m["label"] in want}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for runs in models.values() for rt in runs]))
    info = particle_table(ev)
    rows = []
    for st in PER_SET:
        v = {lab: TS.efficiency(bs, runs, info, TS.masks()["primary"], (st,)) for lab, runs in models.items()}
        for a, b in prs:
            (pa, ra, na), (pb, rb, nb) = v[a], v[b]
            d = BS.linear_combination([(1.0, pa, ra), (-1.0, pb, rb)])
            sa, sb = BS.seed_mean(pa, ra), BS.seed_mean(pb, rb)
            rows.append({"set": st, "new": a, "baseline": b, "n_per_seed": na, "value_new": sa["mean"],
                         "sigma_new": sa["sigma"], "value_base": sb["mean"], "sigma_base": sb["sigma"],
                         "difference": d["value"], "sigma": d["sigma"], "z": d["z"]})
    return pd.DataFrame(rows)


def _pm(r, k, d=4):
    v = f"{r[k]:.{d}f}"
    return v + (f" ± {r[k + '_sd']:.{d}f}" if r["n_seeds"] > 1 and np.isfinite(r[k + "_sd"]) else "")


def _table(tab: pd.DataFrame) -> list[str]:
    lines = ["| model | seeds | primary | \\|d0\\| 100-300 mm | \\|d0\\| >= 300 mm | W muons | ttbar prompt | fake rate |"
             " working points (seeds 0 / 1 / 2) |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        wps = re.sub(r" \((?:trackbuild_val[^)]*|trackbuild_test, configuration 0|chosen on val in [^)]*)\)", "",
                     r.working_points)
        lines.append(f"| {r.label} | {r.n_seeds} | {_pm(r, 'eff_primary')} | {r['eff_d0_100']:.3f} ({r['k_n_d0_100']}) "
                     f"| {r['k_n_d0_300']} ({r['eff_d0_300']:.3f}) | {_pm(r, 'eff_w_muons')} | {_pm(r, 'eff_ttbar_prompt')} "
                     f"| {_pm(r, 'fake_rate_hnl')} | {wps} |")
    return lines


QLAB = {"primary": "primary", "hnl_d0_100_300": "|d0| 100-300 mm", "hnl_d0_ge300": "|d0| >= 300 mm",
        "hnl_targets": "HNL daughters", "w_muons": "W muons", "ttbar_prompt": "ttbar prompt", "fake_rate": "fake rate"}


def _diff_table(diffs: pd.DataFrame) -> list[str]:
    lines = ["| new | baseline | " + " | ".join(QLAB[q] for q, *_ in QUANTITIES) + " |",
             "|---|---|" + "---|" * len(QUANTITIES)]
    for (a, b), g in diffs.groupby(["new", "baseline"], sort=False):
        g = g.set_index("quantity")
        cells = []
        for q, *_ in QUANTITIES:
            r = g.loc[q]
            z = f"z {r.z:+.1f}" if np.isfinite(r.z) else "identical"
            cells.append(f"{100 * r.difference:+.2f} ± {100 * r.sigma:.2f} ({z})")
        short = "Canto-L" if a == NEW else "Canto-L + MWS-layer + merging"
        lines.append(f"| {short} | {b} | " + " | ".join(cells) + " |")
    return lines


def markdown(vt: pd.DataFrame, tt: pd.DataFrame, more: dict, diffs: pd.DataFrame, vdiffs: pd.DataFrame, fr: float,
             notes: list[str], ps: pd.DataFrame | None = None) -> str:
    L = ["# Strongest acorn baselines (supervisor feedback round 1, point 2)", "",
         "acorn (HNL) = acorn's interaction GNN trained on the HNL signal as in Wang et al. (arm ii); acorn (ttbar) = "
         "trained on ttbar only (arm i); three seeds each, the Stage-0 baseline scores. Builders on those scores: acorn's "
         "own CC + walkthrough (CCandWalk, hnl.trackbuild.acorn_walk: acorn's unmodified functions; plain walk as in "
         "acorn's CTD 2023 configuration, and score-guided as in its Rel. 24 / ACAT 2025 configurations), our walkthrough "
         "+ merging, and the Canto builder (same-layer MWS + merging). Every configuration is chosen per seed on val by "
         f"the builder rule at FR* = {fr:.6f} (pooled HNL val) with the one-standard-error tie rule; nothing is chosen on "
         "test. Cells: mean ± seed sd over three seeds; |d0| bins: HNL-daughter targets pooled over the seeds (k/n).", "",
         "## Validation (pooled HNL val 1,999 events + ttbar val 1,000 events)", ""]
    L += _table(vt.sort_values("eff_primary", ascending=False, kind="stable"))
    L += ["", "## Test, post-freeze descriptive (not claims; the preregistered claims stay as they are)", "",
          "Pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) for primary, |d0|, "
          "W muons and fake rate; ttbar_test (1,000 events) for ttbar prompt. The same configurations as on val, "
          "applied unchanged to the stored test scores; summary code path of hnl.eval.ranking_test.", ""]
    L += _table(tt.sort_values("eff_primary", ascending=False, kind="stable"))
    L += ["", "Primary efficiency with the preregistered total sigma (event bootstrap + seed term):", ""]
    for _, r in tt.sort_values("eff_primary", ascending=False, kind="stable").iterrows():
        p = more[r.label]["primary"]
        L.append(f"- {r.label}: {p['mean']:.4f} ± {p['sigma']:.4f} (bootstrap {p['sigma_boot']:.4f}, seed sd "
                 f"{p['s_seeds']:.4f})")
    L += ["", "## Like-for-like differences on test (Canto-L minus baseline; paired event bootstrap of the preregistered "
              "test tables, B = 500, + both seed terms: the claims' estimator; post-freeze descriptive)", ""]
    L += _diff_table(diffs)
    if ps is not None and len(ps):
        L += ["", "Primary efficiency per test set (descriptive; seed mean ± total sigma; difference with the same "
                  "estimator):", "", "| test set | primary particles per seed | model | model value | acorn (HNL) + MWS-layer + "
                  "merging | difference |",
              "|---|---|---|---|---|---|"]
        for _, r in ps.iterrows():
            nm = "Canto-L" if r.new == NEW else "Canto-L + MWS-layer + merging"
            L.append(f"| {r.set} | {int(r.n_per_seed)} | {nm} | {r.value_new:.4f} ± {r.sigma_new:.4f} | "
                     f"{r.value_base:.4f} ± {r.sigma_base:.4f} | {100 * r.difference:+.2f} ± {100 * r.sigma:.2f} "
                     f"(z {r.z:+.1f}) |")
        L += ["", f"Baseline of these rows: {HEADLINE}."]
    L += ["", "## Differences on val (Canto-L minus baseline; paired event bootstrap of the val tables of feedback "
              "point 5, B = 1000, + both seed terms; validation)", ""]
    L += _diff_table(vdiffs)
    L += ["", "Points = percentage points (fake rate: a negative difference means Canto-L has the lower fake rate). "
              "A row's own sigma (bootstrap + seed term) is in the CSV files.", ""]
    L += ["## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(L) + "\n"


def run_test(out: str, name: str, notes_file: str | None):
    fr = fr_star()
    spec = test_spec(out, name)
    tab, ent, more = RT.rank(spec)
    tab.insert(0, "set", "test")
    ent.insert(0, "set", "test")
    for lab in tab.label:
        p = more[lab]["primary"]
        tab.loc[tab.label == lab, "primary_sigma_total"] = p["sigma"]
        tab.loc[tab.label == lab, "primary_sigma_boot"] = p["sigma_boot"]
    diffs = differences(spec, pairs([m["label"] for m in spec]))
    claims = json.load(open(_repo("results/claims_test_v1.json")))["result"]["primary"]["difference"]
    ref = diffs[(diffs.new == NEW) & (diffs.quantity == "primary")
                & (diffs.baseline == "acorn (HNL) + CC, preregistered cut (claims' reference)")].iloc[0]
    for k in ("value", "sigma_boot", "sigma_seeds", "sigma"):
        mine = ref["difference" if k == "value" else k]
        if abs(float(mine) - float(claims[k])) > 1e-12:
            raise SystemExit(f"identity check: Canto-L - acorn (HNL) CC {k} {mine} != claims {claims[k]}")
    print("identity check against results/claims_test_v1.json primary difference: ok")
    ps = per_set(spec, [(NEW, HEADLINE), (CANTO_SAME, HEADLINE)])
    ps.to_csv(os.path.join(out, f"{name}_test_per_set.csv"), index=False)
    vt = pd.read_csv(os.path.join(out, f"{name}_val.csv"))
    both = pd.concat([vt, tab], ignore_index=True)
    both.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    ve = pd.read_csv(os.path.join(out, f"{name}_val_entries.csv"))
    pd.concat([ve, ent], ignore_index=True).to_csv(os.path.join(out, f"{name}_entries.csv"), index=False)
    diffs.to_csv(os.path.join(out, f"{name}_differences.csv"), index=False)
    notes = [] if notes_file is None else [ln.strip()[2:] for ln in open(notes_file) if ln.startswith("- ")]
    vdiffs = pd.read_csv(os.path.join(out, f"{name}_val_differences.csv"))
    md = markdown(vt, tab, more, diffs, vdiffs, fr, notes, ps)
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        f.write(md)
    print(md)
    return tab, diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["merge", "val", "jobs", "test"])
    ap.add_argument("--out", default=_repo("results"))
    ap.add_argument("--name", default="strong_baselines_v1")
    ap.add_argument("--skip-missing", action="store_true", help="val: skip builder rows whose grids are missing")
    ap.add_argument("--notes", default=None, help="test: a text file of '- ' note lines for the markdown")
    a = ap.parse_args(argv)
    if a.cmd == "merge":                     # the acwalk grid parts of the six acorn runs
        for arm, _ in ARMS:
            for seed in SEEDS:
                print(merge_grids(os.path.join(BASE, f"arm{arm}_s{seed}")))
    elif a.cmd == "val":
        run_val(a.out, a.name, a.skip_missing)
    elif a.cmd == "jobs":
        print("\n".join(jobs(a.out, a.name)))
    else:
        run_test(a.out, a.name, a.notes)


if __name__ == "__main__":
    main()
