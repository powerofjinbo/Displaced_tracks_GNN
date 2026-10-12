"""Our frozen models under the PUBLISHED definitions of Wang et al. (arXiv:2203.08800): post-freeze descriptive
summaries of the one-time test outputs (and of the val outputs at the same frozen configurations), NOT claims.

Wang et al. report three DM track efficiencies (docs/paper_facts.md C7-C9): all tracks 0.9561 +- 0.0006, prompt tracks
from the HNL process (the W muon) 0.990 +- 0.002, displaced tracks from the HNL process (the N daughters)
0.916 +- 0.005; reconstructable = pT > 1 GeV, |eta| < 4, >= 5 space points; no fake rate. The populations below are
those of the earlier reproduction (hnl.eval.test_stats.paper_summary, docs/definitions.md section 4), whose code path
(test_stats.RunTest / masks / efficiency / event_ratio and the preregistered bootstrap) is reused unchanged:

  all-track   every target (pT > 1 GeV, |eta| < 4, >= 5 hits; table values) of the HNL events      mask all_targets
  prompt      the W muon with >= 5 hits, no pT / |eta| cut: on hnl15_100_external this is exactly the
              paper's prompt denominator, 4,744 (docs/paper_facts.md)                               mask w_muons_paper
  displaced   the HNL-daughter targets, no displacement cut: the closest faithful version (the paper's
              set, 2,882 particles, is about half of ours and cannot be reproduced)                  mask hnl_targets
  checks      W-muon targets (with the pT / |eta| cuts); the daughter targets of each charge          w_muons, hnl_mu_*
  fake rate   n_fake / n_tracks_ge5 summed over the events of the set (docs/definitions.md section 5)

Sets: test hnl15_100_external (the paper's evaluation sample, 4,999 events), test pooled HNL (hnl10_100_test +
hnl15_200_test + hnl15_100_external, 6,999 events), val pooled HNL (hnl10_100 + hnl15_200 val, 1,999 events).
Per model (three seeds): mean +- seed sd (ddof 1), the count k/n pooled over the seeds, and for test the
preregistered total sigma (event bootstrap + seed term, hnl.eval.bootstrap).

Models: frozen configurations only, nothing is chosen here. Canto-L / -H / -B at the FR* configuration of
results/final_selection_v1.json (test: RUN/trackbuild_test, index 0; val: the same configuration in the val grid named
by results/final_selection_v1_seeds.csv); acorn (HNL) with CC at its preregistered cuts (working points
cuts_at_FR_star["ii"]), at its best cuts <= FR* on val (results/ranking/ranking_val_v2_entries.csv, re-derived here
from eval_val with the ranking rule) and at s* (the earlier reproduction, NUMBERS B11); acorn (ttbar) with CC at
cuts_at_FR_star["i"]. The best cuts of acorn (HNL) seeds 0 and 2 (0.38, 0.44) were not among the cuts of the one-time
test chain: `bestcut` evaluates them post-freeze on the stored test scores (RUN/gnn_scores_test, CC only, no network
is run) into runs/published_defs_v1/armii_s<seed>/eval_test_bestcut, together with s* as an identity check against
the frozen RUN/eval_test.

    python -m hnl.eval.published_defs bestcut --seeds 0,2          (sbatch; once)
    python -m hnl.eval.published_defs summary --out results         (sbatch)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table, split_event_id

REPO = os.environ.get("HNL_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
BASELINES = os.path.join(DATA, "runs", "baseline_v1")
POSTFREEZE = os.path.join(DATA, "runs", "published_defs_v1")
BESTCUT_SUB = "eval_test_bestcut"
WP = "results/working_points_v1.json"
SELECTION = "results/final_selection_v1.json"
SELECTION_SEEDS = "results/final_selection_v1_seeds.csv"
RANKING_V2 = "results/ranking/ranking_val_v2_entries.csv"
CLAIMS = "results/claims_test_v1.json"
REPRO = os.path.join(BASELINES, "analysis_test", "test_summary.json")      # the earlier reproduction (B11)
SEEDS = (0, 1, 2)
HNL_VAL_CODES = (2, 4)                                                     # hnl10_100, hnl15_200

# quantity -> (test_stats.masks() name, definition)
POPS = {
    "all_track": ("all_targets", "every target (pT > 1 GeV, |eta| < 4, >= 5 hits) of the HNL events"),
    "prompt": ("w_muons_paper", "W muon with >= 5 hits (no pT / |eta| cut; = the paper's 4,744 on the external set)"),
    "displaced": ("hnl_targets", "HNL-daughter targets, no displacement cut"),
    "prompt_targets": ("w_muons", "check: W-muon targets (with the pT / |eta| cuts)"),
    "displaced_mu_plus": ("hnl_mu_plus", "check: HNL-daughter targets with q > 0"),
    "displaced_mu_minus": ("hnl_mu_minus", "check: HNL-daughter targets with q < 0"),
}
SETS = {
    "test_external": ("test", ("hnl15_100_external",),
                      "test, hnl15_100_external (the paper's evaluation sample, 4,999 events)"),
    "test_pooled": ("test", TS.HNL_SETS,
                    "test, pooled HNL (hnl10_100 + hnl15_200 test + hnl15_100_external, 6,999 events)"),
    "val_pooled": ("val", None, "val, pooled HNL (hnl10_100 + hnl15_200 val, 1,999 events)"),
}
PUBLISHED = [  # docs/paper_facts.md (C9 and the Fig. 3 vector data); Wang et al. define and report no fake rate
    ("all_track", 0.9561, 0.0006, "not given; the binomial error implies about 116,600 (99k-139k)"),
    ("prompt", 0.990, 0.002, "4,744 (Fig. 3: 4696/4744 = 0.98988 +- 0.00145)"),
    ("displaced", 0.916, 0.005, "about 2,882 (Fig. 3: 2639/2882 = 0.9157 +- 0.0052, two bins at 1.0 left out)"),
    ("fake_rate", np.nan, np.nan, "not defined or reported"),
]
LABELS = {
    "canto_l": "Canto-L (learned merging)",
    "canto_h": "Canto-H (walkthrough + merging)",
    "canto_b": "Canto-B (learned merging)",
    "acorn_hnl_prereg": "acorn (HNL), CC, preregistered cut",
    "acorn_hnl_best": "acorn (HNL), CC, best cut <= FR* (val)",
    "acorn_hnl_sstar": "acorn (HNL), CC at s* (earlier reproduction)",
    "acorn_ttbar": "acorn (ttbar), CC, working-point cuts",
}
CANTO = {"canto_l": "l1tl", "canto_h": "l1tu", "canto_b": "l1t"}


def _p(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _git() -> str:
    try:
        h = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", "src"], capture_output=True,
                           text=True).stdout.strip()
        return h + (" (src dirty: " + ", ".join(x[3:] for x in d.splitlines()) + ")" if d else "")
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------------------------------- frozen inputs
def ranking_cuts(label: str) -> dict[int, tuple[float, bool]]:
    """seed -> (cut, reachable) of a CC row of the frozen ranking v2 entries (val)."""
    out = {}
    for r in csv.DictReader(open(_p(RANKING_V2))):
        if r["label"] == label:
            seed = int(r["run"].rstrip("/").rsplit("_s", 1)[1])
            m = re.match(r"CC (\d\.\d+)( \(FR\* unreachable)?", r["working_point"])
            out[seed] = (float(m.group(1)), m.group(2) is None)
    if sorted(out) != list(SEEDS):
        raise SystemExit(f"{RANKING_V2}: no three seeds for {label!r}")
    return out


def model_specs() -> dict[str, list[dict]]:
    """Per model and seed: run, test (sub, cut / config index), val (sub, cut / config index), working point."""
    wp = json.load(open(_p(WP)))
    sel = json.load(open(_p(SELECTION)))
    grids = {(r["pipeline"], int(r["seed"]), r["family"]): r for r in csv.DictReader(open(_p(SELECTION_SEEDS)))}
    specs: dict[str, list[dict]] = {}
    for name, pipe in CANTO.items():
        p = sel["pipelines"][pipe]
        es = []
        for s in SEEDS:
            e = p["seeds"][str(s)]
            run, cfg = e["run"], e["config"]
            g = grids[(pipe, s, p["family"])]
            if g["config"] != cfg:
                raise SystemExit(f"{pipe} seed {s}: {SELECTION_SEEDS} has {g['config']}, the selection {cfg}")
            cfgs = json.load(open(os.path.join(run, g["grid"], "configs.json")))
            k = [i for i, c in enumerate(cfgs) if ",".join(str(x) for x in c) == cfg]
            tcfg = json.load(open(os.path.join(run, "trackbuild_test", "configs.json")))
            if len(k) != 1 or len(tcfg) != 1 or ",".join(str(x) for x in tcfg[0]) != cfg:
                raise SystemExit(f"{run}: configuration {cfg} not unique in {g['grid']} or not trackbuild_test[0]")
            es.append({"seed": s, "run": run, "test": ("trackbuild_test", 0.0), "val": (g["grid"], float(k[0])),
                       "wp": cfg.replace(",", " ", 1), "val_eff_primary": float(g["eff_primary"])})
        specs[name] = es

    def acorn(arm: str, cuts: dict, unreach: set, test_sub=None) -> list[dict]:
        es = []
        for s in SEEDS:
            run = os.path.join(BASELINES, f"arm{arm}_s{s}")
            c = float(cuts[s])
            sub, trun = "eval_test", run
            if test_sub is not None and not _has_cut(os.path.join(run, "eval_test"), c):
                trun, sub = os.path.join(POSTFREEZE, f"arm{arm}_s{s}"), test_sub
            es.append({"seed": s, "run": run, "test_run": trun, "test": (sub, c), "val": ("eval_val", c),
                       "wp": f"CC {c:.2f}" + (" (FR* unreachable: minimum-rate cut)" if s in unreach else ""),
                       "new_test_eval": sub == test_sub})
        return es

    unreach = {(a, int(s)) for a, s in wp["unreachable_at_FR_star"]}
    pre = {s: wp["cuts_at_FR_star"]["ii"][str(s)] for s in SEEDS}
    specs["acorn_hnl_prereg"] = acorn("ii", pre, {s for s in SEEDS if ("ii", s) in unreach})
    best = ranking_cuts("acorn IGNN, HNL-trained + CC (best cut <= FR*)")
    specs["acorn_hnl_best"] = acorn("ii", {s: c for s, (c, _) in best.items()},
                                    {s for s, (_, ok) in best.items() if not ok}, test_sub=BESTCUT_SUB)
    specs["acorn_hnl_sstar"] = acorn("ii", {s: wp["s_star"]["ii"] for s in SEEDS}, set())
    tt = {s: wp["cuts_at_FR_star"]["i"][str(s)] for s in SEEDS}
    specs["acorn_ttbar"] = acorn("i", tt, {s for s in SEEDS if ("i", s) in unreach})
    for name in specs:
        for e in specs[name]:
            e.setdefault("test_run", e["run"])
    return specs


def _has_cut(d: str, cut: float) -> bool:
    m = os.path.join(d, "meta.json")
    return os.path.exists(m) and any(abs(float(c) - cut) < 1e-6 for c in json.load(open(m))["cuts"])


# ------------------------------------------------------------------------------------------------------- counting
def set_mask(event_id, set_key: str) -> np.ndarray:
    kind, sets, _ = SETS[set_key]
    if kind == "test":
        return TS.in_sets(event_id, sets)
    return np.isin(split_event_id(np.asarray(event_id))[0], HNL_VAL_CODES)


def seed_counts(rt: TS.RunTest, info: pd.DataFrame, set_key: str) -> dict:
    """k, n of every population and of the fake rate for one seed on one set."""
    t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
    sm = set_mask(t.event_id, set_key)
    M = TS.masks()
    out = {}
    for q, (mask, _) in POPS.items():
        m = np.asarray(M[mask](t), dtype=bool) & sm
        out[q] = (int(t.dm_matched.to_numpy()[m].sum()), int(m.sum()))
    e = rt.pe[set_mask(rt.pe.event_id, set_key)]
    out["fake_rate"] = (int(e.n_fake.sum()), int(e.n_tracks_ge5.sum()))
    out["_events"] = int(e.event_id.nunique())
    return out


def summarize(per_seed: list[dict], q: str) -> dict:
    k = np.array([c[q][0] for c in per_seed], dtype=np.int64)
    n = np.array([c[q][1] for c in per_seed], dtype=np.int64)
    eff = k / n
    return {"mean": float(eff.mean()), "sd_seeds": float(eff.std(ddof=1)), "k_pooled": int(k.sum()),
            "n_pooled": int(n.sum()), "per_seed_eff": eff.tolist(), "per_seed_k": k.tolist(), "per_seed_n": n.tolist()}


def load_runs(specs: dict, kind: str) -> dict[str, list[TS.RunTest]]:
    out = {}
    for name, es in specs.items():
        if kind == "test":
            out[name] = [TS.RunTest(e["test_run"], e["test"][1], sub=e["test"][0]) for e in es]
        else:
            out[name] = [TS.RunTest(e["run"], e["val"][1], sub=e["val"][0], check_events=False) for e in es]
    return out


# --------------------------------------------------------------------------------------------------------- checks
def checks(specs, runs_test, runs_val, info, res, bs) -> list[tuple[str, bool, str]]:
    """Cross-checks against the frozen records (every one must hold)."""
    out = []

    def add(name, ok, detail):
        out.append((name, bool(ok), detail))

    # 1. the earlier reproduction (test_stats.paper_summary, acorn (HNL) at s*, external set) is reproduced exactly
    old = json.load(open(REPRO))["paper_comparison_arm_ii_s_star"]
    new = TS.paper_summary(bs, runs_test["acorn_hnl_sstar"], info)
    for k in ("eff_all_targets", "eff_w_muons_paper", "eff_hnl_targets", "eff_hnl_mu_plus", "eff_hnl_mu_minus",
              "fake_rate"):
        key = f"{k}@hnl15_100_external"
        a, b = old[key], new[key]
        add(f"reproduction {key}", abs(a["mean"] - b["mean"]) < 1e-12 and abs(a["sigma"] - b["sigma"]) < 1e-12,
            f"earlier {a['mean']:.10f} +- {a['sigma']:.10f}, now {b['mean']:.10f} +- {b['sigma']:.10f}")
    r = res[("test_external", "acorn_hnl_sstar", "all_track")]
    add("all-track counting = test_stats.efficiency", abs(r["mean"] - old["eff_all_targets@hnl15_100_external"]["mean"])
        < 1e-12, f"{r['mean']:.10f} ({r['k_pooled']}/{r['n_pooled']})")
    # 2. the frozen claims (pooled HNL test, primary population)
    cl = json.load(open(_p(CLAIMS)))["result"]["primary"]
    for name, ref in (("canto_l", cl["eff_new"]), ("acorn_hnl_prereg", cl["eff_base"])):
        p, _, _ = TS.efficiency(bs, runs_test[name], info, TS.masks()["primary"], TS.HNL_SETS)
        add(f"claims primary {name}", abs(np.mean(p) - ref) < 1e-12, f"{np.mean(p):.10f} vs {ref:.10f}")
    # 3. val: every configuration gives the frozen val primary efficiency of the selection / ranking v2
    rv = {}
    for row in csv.DictReader(open(_p(RANKING_V2))):
        rv[(row["label"], row["run"].rstrip("/"), row["working_point"].split(" (")[0])] = float(row["eff_primary"])
    lab = {"acorn_hnl_prereg": "acorn IGNN, HNL-trained + CC (prereg. cut rule)",
           "acorn_hnl_best": "acorn IGNN, HNL-trained + CC (best cut <= FR*)",
           "acorn_ttbar": "acorn IGNN, ttbar-trained + CC (prereg. cut rule)"}
    M = TS.masks()
    for name, es in specs.items():
        for e, rt in zip(es, runs_val[name]):
            t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
            m = np.asarray(M["primary"](t), dtype=bool) & set_mask(t.event_id, "val_pooled")
            eff = float(t.dm_matched.to_numpy()[m].mean())
            if name in CANTO:
                ref = e["val_eff_primary"]
            elif name in lab:
                ref = rv[(lab[name], e["run"], f"CC {e['val'][1]:.2f}")]
            else:
                continue
            add(f"val primary {name} s{e['seed']}", abs(eff - ref) < 1e-12, f"{eff:.10f} vs {ref:.10f}")
    # 4. the best cuts of acorn (HNL) re-derived from eval_val with the ranking rule ("best")
    from hnl.eval.ranking import _cc
    fr = float(json.load(open(_p(WP)))["FR_star"])
    for e in specs["acorn_hnl_best"]:
        _, _, wp, _ = _cc({"run": e["run"], "source": "eval_val", "rule": "best"}, fr, info)
        add(f"best cut armii_s{e['seed']}", wp == e["wp"], f"ranking rule {wp!r}, frozen {e['wp']!r}")
    # 5. event counts of the sets
    for set_key, n_ev in (("test_external", 4999), ("test_pooled", 6999), ("val_pooled", 1999)):
        got = {res[(set_key, m, "_events")] for m in specs}
        add(f"events {set_key}", got == {n_ev}, f"{sorted(got)} (expected {n_ev})")
    # 6. the prompt denominator on the external set equals the paper's 4,744
    got = {tuple(res[("test_external", m, "prompt")]["per_seed_n"]) for m in specs}
    add("prompt denominator external = 4744", got == {(4744,) * 3}, str(sorted(got)))
    # 7. every model and seed sees the same particles (the populations depend on the truth only)
    for set_key in SETS:
        for q in POPS:
            got = {n for m in specs for n in res[(set_key, m, q)]["per_seed_n"]}
            add(f"same population {set_key} {q}", len(got) == 1, f"n per seed {sorted(got)}")
    return out


# ------------------------------------------------------------------------------------------------------- outputs
def _cell(r: dict, d: int = 4) -> str:
    return f"{r['mean']:.{d}f} ± {r['sd_seeds']:.{d}f} ({r['k_pooled']:,}/{r['n_pooled']:,})"


def _wp(es: list[dict]) -> str:
    """Working points of the three seeds in one cell (* = FR* unreachable on val: minimum-rate cut)."""
    w = [e["wp"] for e in es]
    fam = {x.split(" ")[0] for x in w}
    if len(fam) != 1:
        return " / ".join(w)
    f = fam.pop()
    vals = [x.split(" ", 1)[1].replace(" (FR* unreachable: minimum-rate cut)", "*") for x in w]
    if f == "walkmerge":
        return "walkthrough + merging " + " / ".join(f"({v.replace(',', ', ')})" for v in vals)
    if f == "mwslmerge":
        return "learned merging tau " + " / ".join(vals)
    return f"{f} " + " / ".join(vals)


def _new_seeds(es: list[dict]) -> str:
    return ", ".join(f"s{e['seed']}" for e in es if e.get("new_test_eval"))


def write_outputs(out: str, specs, res, sig, chk, meta) -> None:
    rows = []
    for q, v, e, den in PUBLISHED:
        rows.append({"set": "published", "set_label": "their evaluation sample (5,000 HNL events, 15 GeV, 100 mm)",
                     "model": "wang_et_al", "label": "Wang et al. (arXiv:2203.08800), as published",
                     "status": "published", "working_point": "attention GNN + DBSCAN (minPoints 2, epsilon 0.4)",
                     "quantity": q, "definition": den, "mean": v, "sd_seeds": np.nan, "sigma_total": e})
    for set_key, (kind, _, set_label) in SETS.items():
        for name, es in specs.items():
            for q in list(POPS) + ["fake_rate"]:
                r = res[(set_key, name, q)]
                s = sig.get((set_key, name, q), {})
                rows.append({
                    "set": set_key, "set_label": set_label, "model": name, "label": LABELS[name],
                    "status": "post-freeze descriptive (not a claim)" if kind == "test" else "validation",
                    "working_point": "; ".join(f"s{e['seed']}: {e['wp']}" for e in es),
                    "new_test_evaluation": (_new_seeds(es) if kind == "test" else ""),
                    "quantity": q, "definition": POPS[q][1] if q in POPS else "n_fake / n_tracks_ge5 over the set",
                    "mean": r["mean"], "sd_seeds": r["sd_seeds"], "sigma_total": s.get("sigma", np.nan),
                    "sigma_boot": s.get("sigma_boot", np.nan), "k_pooled": r["k_pooled"], "n_pooled": r["n_pooled"],
                    **{f"eff_s{i}": r["per_seed_eff"][i] for i in range(3)},
                    **{f"k_s{i}": r["per_seed_k"][i] for i in range(3)},
                    **{f"n_s{i}": r["per_seed_n"][i] for i in range(3)}})
    cols = ["set", "set_label", "model", "label", "status", "working_point", "new_test_evaluation", "quantity",
            "definition", "mean", "sd_seeds", "sigma_total", "sigma_boot", "k_pooled", "n_pooled",
            "eff_s0", "eff_s1", "eff_s2", "k_s0", "k_s1", "k_s2", "n_s0", "n_s1", "n_s2"]
    df = pd.DataFrame(rows).reindex(columns=cols)
    for c in ("k_pooled", "n_pooled", "k_s0", "k_s1", "k_s2", "n_s0", "n_s1", "n_s2"):
        df[c] = df[c].astype("Int64")
    df.to_csv(os.path.join(out, "published_definitions_comparison.csv"), index=False)

    L = ["# Our models under the published definitions of Wang et al. (arXiv:2203.08800)", "",
         "**Post-freeze descriptive summaries of the one-time test outputs, NOT claims** (the preregistered claims are "
         "results/claims_test_v1.json). Within this simulation (ACTS generic detector, no pileup). Written by "
         "`python -m hnl.eval.published_defs summary` "
         f"({meta['created_utc']}, git {meta['git']}, Slurm job {meta['slurm_job']}). Frozen configurations only: "
         "nothing was chosen, tuned or re-selected on test. Post-freeze CC evaluations of the stored test scores "
         f"(acorn (HNL) at its val best cut): {meta['postfreeze'] or 'none'}.", "",
         "## Published numbers (Wang et al., p.4; docs/paper_facts.md C9 and the Fig. 3 vector data)", "",
         "| quantity | published | denominator |", "|---|---|---|"]
    names = {"all_track": "all tracks in the event", "prompt": "prompt tracks from the HNL process (the W muon)",
             "displaced": "displaced tracks from the HNL process (the N daughters)", "fake_rate": "track fake rate"}
    for q, v, e, den in PUBLISHED:
        L.append(f"| {names[q]} | {'%.4f' % v if q == 'all_track' else '%.3f' % v if np.isfinite(v) else '-'}"
                 f"{' ± %.4f' % e if q == 'all_track' else ' ± %.3f' % e if np.isfinite(e) else ''} | {den} |")
    L += ["",
          "Their pipeline: metric-learning MLP (3D -> 8D) + filter MLP graph (99.9 % of the true edges at 5.7 % purity), "
          "attention GNN (edges at score 0.5: prompt 93.8 % efficiency / 87.0 % purity, displaced 92.5 % / 84.7 %), "
          "DBSCAN (minPoints 2, epsilon 0.4, the epsilon of highest efficiency and track purity on a 0.1-1.0 scan). "
          "Trained on 10,000 HNL events (15 GeV, c*tau = 100 mm) of their own generation (not released); evaluated on "
          "5,000 events, identified as the complete hnl15_100 file of the Zenodo record (docs/paper_facts.md). "
          "Matching: double majority, reconstructable = pT > 1 GeV, |eta| < 4, >= 5 space points. No fake or "
          "duplicate rate is defined or reported; their uncertainty method is not stated.", "",
          "## Definitions used here (the code path of the earlier reproduction, hnl.eval.test_stats.paper_summary)", "",
          "| quantity | population (per seed) | test_stats mask |", "|---|---|---|"]
    for q, (mask, d) in POPS.items():
        L.append(f"| {q} | {d} | `{mask}` |")
    L += ["| fake rate | tracks with >= 5 hits that DM-match no particle / tracks with >= 5 hits, over the set's events "
          "| `n_fake / n_tracks_ge5` |", "",
          "Cells: mean ± seed standard deviation over the three training seeds (ddof 1), then the count pooled over "
          "the seeds (k/n; n = 3 x the population). Working points: Canto at the FR* configuration of "
          "results/final_selection_v1.json; acorn with connected components (CC) at the cut named in the row "
          "(* = FR* unreachable on val, minimum-rate cut). FR* = 0.029368 was fixed on pooled HNL val; on test the "
          "fake rate is measured, not constrained.", ""]
    order = list(specs)
    for set_key, (kind, _, set_label) in SETS.items():
        tag = " (post-freeze descriptive, not a claim)" if kind == "test" else " (validation)"
        L += [f"## {set_label[0].upper() + set_label[1:]}{tag}", "",
              "| model | working point (seeds 0 / 1 / 2) | all-track | prompt (W muon, >= 5 hits) | displaced "
              "(N-daughter targets) | fake rate |", "|---|---|---|---|---|---|"]
        for name in order:
            r = {q: res[(set_key, name, q)] for q in ("all_track", "prompt", "displaced", "fake_rate")}
            new = f" ({_new_seeds(specs[name])} evaluated post-freeze)" if kind == "test" and _new_seeds(
                specs[name]) else ""
            L.append(f"| {LABELS[name]}{new} | {_wp(specs[name])} | {_cell(r['all_track'])} | {_cell(r['prompt'])} | "
                     f"{_cell(r['displaced'])} | {_cell(r['fake_rate'])} |")
        where = "" if set_key == "test_external" else " (on their evaluation sample, not this set)"
        L.append(f"| Wang et al. (published){where} | attention GNN + DBSCAN | 0.9561 ± 0.0006 | 0.990 ± 0.002 | "
                 f"0.916 ± 0.005 | not reported |")
        L.append("")
        if kind == "test":
            L += [f"Total uncertainty of the three-seed mean on this set (preregistered: event bootstrap B = 500 + "
                  f"seed term):", "", "| model | all-track | prompt | displaced | fake rate |", "|---|---|---|---|---|"]
            for name in order:
                L.append("| " + LABELS[name] + " | " + " | ".join(
                    f"{sig[(set_key, name, q)]['mean']:.4f} ± {sig[(set_key, name, q)]['sigma']:.4f}"
                    for q in ("all_track", "prompt", "displaced", "fake_rate")) + " |")
            L.append("")
    L += ["## Population checks (same outputs)", "",
          "| set | model | prompt: W-muon targets (pT / abs(eta) cuts) | displaced, q > 0 | displaced, q < 0 |",
          "|---|---|---|---|---|"]
    for set_key in SETS:
        for name in order:
            L.append(f"| {set_key} | {LABELS[name]} | " + " | ".join(
                _cell(res[(set_key, name, q)]) for q in ("prompt_targets", "displaced_mu_plus", "displaced_mu_minus"))
                + " |")
    L += ["", "Populations per seed: " + "; ".join(
        f"{set_key}: " + ", ".join(f"{q} {res[(set_key, 'canto_l', q)]['per_seed_n'][0]:,}" for q in POPS)
        for set_key in SETS) + ".", "",
        "## Cross-checks against the frozen records (all must hold)", "", "| check | ok | detail |", "|---|---|---|"]
    for n_, ok, d in chk:
        L.append(f"| {n_} | {'yes' if ok else 'NO'} | {d} |")
    L += ["", "## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    open(os.path.join(out, "published_definitions_comparison.md"), "w").write("\n".join(L))


CAVEATS = [
    "Post-freeze descriptive: every test number here summarises the frozen one-time test outputs (or, for acorn (HNL) "
    "at its best cut, seeds 0 and 2, a post-freeze CC evaluation of the stored test scores at cuts fixed on val); none "
    "is a claim, none was used to choose anything.",
    "Sample: the external set is the paper's evaluation sample (15 GeV, c*tau = 100 mm), 4,999 of its 5,000 events "
    "(corrupted event 4375 dropped). The pooled test sets add 10 GeV / 100 mm and 15 GeV / 200 mm events (2,000 "
    "events), and val holds only those two samples (no 15 GeV / 100 mm event): a different mass and lifetime mix.",
    "Training: Wang et al. trained on 10,000 HNL events of 15 GeV / 100 mm (not released); acorn (HNL) on the 15,999 "
    "training events of the 10 GeV / 100 mm and 15 GeV / 200 mm samples; Canto and acorn (ttbar) on 7,999 ttbar "
    "events only.",
    "Pipelines: Wang et al. used their own metric learning, filter, an attention GNN and DBSCAN; acorn here is the "
    "acorn interaction GNN with connected components (the reproduction of their data setup, not of their pipeline); "
    "Canto is a different pipeline (triplet GNN, helix-triplet graphs, learned merging).",
    "Working points: the paper reports no fake rate, so its efficiencies carry no fake-rate constraint (DBSCAN epsilon "
    "chosen for efficiency and an undefined track purity); ours sit at fake rate <= FR* = 2.94 % on pooled HNL val "
    "(acorn) or at the frozen Canto configurations (about 1.0-1.4 % on val). The comparison is not at equal fake rate.",
    "Prompt population: 'W muon with >= 5 hits' reproduces the paper's denominator (4,744 on the external set) but "
    "not its stated reconstructable definition; with the pT / |eta| cuts the W-muon targets are 4,629 (check table).",
    "Displaced population: the paper's displaced set has 2,882 particles (Fig. 3), about half of our 5,623 daughter "
    "targets in the same events, and its '|d0|' spectrum behaves like our decay radius r_v; no simple half (one "
    "charge, leading pT, >= 3 pixel hits) reproduces it. Our displaced number is on all daughter targets (each charge "
    "separately as a check), so it is not on the paper's population.",
    "All-track population: 116,449 targets on the external set (23.29 per event), consistent with the about 116,600 "
    "implied by the paper's binomial error; dominated by prompt underlying-event tracks.",
    "Uncertainties: the paper's method is not stated (its prompt +-0.002 is larger than the binomial +-0.0015 of 4,744 "
    "tracks); ours are the seed spread of three training seeds, and on test also the preregistered event bootstrap.",
]


# ----------------------------------------------------------------------------------------------------- subcommands
def cmd_summary(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    specs = model_specs()
    runs_test, runs_val = load_runs(specs, "test"), load_runs(specs, "val")
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for d in (runs_test, runs_val)
                                   for runs in d.values() for rt in runs]))
    info = particle_table(ev)
    bs = BS.Bootstrap()
    res, sig = {}, {}
    for set_key, (kind, sets, _) in SETS.items():
        runs = runs_test if kind == "test" else runs_val
        for name in specs:
            per_seed = [seed_counts(rt, info, set_key) for rt in runs[name]]
            for q in list(POPS) + ["fake_rate"]:
                res[(set_key, name, q)] = summarize(per_seed, q)
            res[(set_key, name, "_events")] = per_seed[0]["_events"]
            if {c["_events"] for c in per_seed} != {per_seed[0]["_events"]}:
                raise SystemExit(f"{name}: seeds cover different events on {set_key}")
            if kind == "test":
                for q, (mask, _) in POPS.items():
                    p, r, _ = TS.efficiency(bs, runs[name], info, TS.masks()[mask], sets)
                    sig[(set_key, name, q)] = BS.seed_mean(p, r)
                    if abs(np.mean(p) - res[(set_key, name, q)]["mean"]) > 1e-12:
                        raise SystemExit(f"{name} {set_key} {q}: bootstrap point differs from the counts")
                p, r = TS.event_ratio(bs, runs[name], "n_fake", "n_tracks_ge5", sets)
                sig[(set_key, name, "fake_rate")] = BS.seed_mean(p, r)
    chk = checks(specs, runs_test, runs_val, info, res, bs)
    pf = []
    for e in specs["acorn_hnl_best"]:
        if e.get("new_test_eval"):
            pv = json.load(open(os.path.join(e["test_run"], e["test"][0], "provenance.json")))
            if not all(pv["identity_check_s_star"].values()):
                raise SystemExit(f"{e['test_run']}: the s* identity check of the post-freeze evaluation failed")
            pf.append(f"armii_s{e['seed']} cut {pv['cut_best_val']:.2f} (job {pv['slurm_job']}, "
                      f"{os.path.join(e['test_run'], e['test'][0])}; s* rows identical to the frozen eval_test)")
    meta = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": _git(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", "none"), "postfreeze": "; ".join(pf)}
    write_outputs(a.out, specs, res, sig, chk, meta)
    for n_, ok, d in chk:
        print(("ok   " if ok else "FAIL ") + n_ + ": " + d)
    print(open(os.path.join(a.out, "published_definitions_comparison.md")).read())
    if not all(ok for _, ok, _ in chk):
        raise SystemExit("cross-checks failed (see the table)")


def _same_rows(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> bool:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    cols = sorted(set(a.columns) & set(b.columns) - {"cut"})
    return len(a) == len(b) and a[cols].equals(b[cols])


def cmd_bestcut(a) -> None:
    """CC at the val-fixed best cuts of acorn (HNL) on the stored test scores (post-freeze; refuses to repeat)."""
    from hnl.eval import evaluate_gnn as EG
    wp = json.load(open(_p(WP)))
    s_star = float(wp["s_star"]["ii"])
    best = ranking_cuts("acorn IGNN, HNL-trained + CC (best cut <= FR*)")
    job = os.environ.get("SLURM_JOB_ID", "local")
    for s in (int(x) for x in a.seeds.split(",")):
        run = os.path.join(BASELINES, f"armii_s{s}")
        cut = best[s][0]
        if _has_cut(os.path.join(run, "eval_test"), cut):
            print(f"armii_s{s}: best cut {cut} is in the frozen eval_test; nothing to do")
            continue
        out = os.path.join(POSTFREEZE, f"armii_s{s}", BESTCUT_SUB)
        if os.path.exists(os.path.join(out, "provenance.json")):
            raise SystemExit(f"{out} exists: the post-freeze evaluation is done once")
        local = os.path.join("/tmp", os.environ.get("USER", "u"), f"pd_{job}_s{s}")
        os.makedirs(os.path.join(local, "testset"), exist_ok=True)
        try:
            tar = os.path.join(run, "gnn_scores_test", "testset.tar")
            subprocess.run(["tar", "xf", tar, "-C", os.path.join(local, "testset")], check=True)
            cuts = sorted({cut, s_star})
            EG.main(["--graphs", os.path.join(local, "testset"), "--out", out,
                     "--cuts", ",".join(f"{c:.4f}" for c in cuts)])
        finally:
            shutil.rmtree(local, ignore_errors=True)
        # identity check: s* (also in the frozen one-time eval_test) must give the frozen rows exactly
        ok = {}
        for f, keys in (("per_particle.parquet", ["event_id", "particle_id"]), ("per_event.parquet", ["event_id"])):
            new = pd.read_parquet(os.path.join(out, f))
            old = pd.read_parquet(os.path.join(run, "eval_test", f))
            ok[f] = _same_rows(new[np.isclose(new.cut, s_star, atol=1e-6)], old[np.isclose(old.cut, s_star, atol=1e-6)],
                               keys)
        prov = {"what": "post-freeze CC evaluation of acorn (HNL) on the stored one-time test scores at the best cut "
                        "<= FR* fixed on val (descriptive, not a claim; hnl.eval.published_defs)",
                "run": run, "scores": tar, "cut_best_val": cut, "cut_source": RANKING_V2,
                "cuts_evaluated": cuts, "identity_check_s_star": ok, "git": _git(), "slurm_job": job,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(os.path.join(out, "provenance.json"), "w") as fh:
            json.dump(prov, fh, indent=1)
        print(json.dumps(prov, indent=1))
        if not all(ok.values()):
            raise SystemExit(f"armii_s{s}: s* rows differ from the frozen eval_test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bestcut")
    b.add_argument("--seeds", default="0,2")
    s = sub.add_parser("summary")
    s.add_argument("--out", default=os.path.join(REPO, "results"))
    a = ap.parse_args(argv)
    {"bestcut": cmd_bestcut, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
