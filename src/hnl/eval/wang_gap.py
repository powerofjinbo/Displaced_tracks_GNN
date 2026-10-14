"""Wang et al.'s evaluation sample (hnl15_100_external): the all-track gap between their published pipeline and Canto-L,
split by stage, and the displaced muons selected as in their Fig. 3. Post-freeze, descriptive, not a claim
(docs/feedback/test_use_ledger.md R12, committed before this read-out).

(a) Stage decomposition. Wang et al.'s definitions (hnl.eval.published_defs: all_track = every target, pT > 1 GeV,
|eta| < 4, >= 5 hits; prompt = the W muon with >= 5 hits; displaced = the HNL-daughter targets; fake rate =
n_fake / n_tracks_ge5) for configurations fixed on validation:
  acorn (HNL) + CC at s* (0.40) and at its best cut <= FR*; acorn (ttbar) + CC (preregistered cut = best cut)
                                       stored one-time outputs (R1, R3);
  acorn (HNL) / (ttbar) / (ttbar+HNL) + LM   A2's val-fixed spec (results/strong_baselines_v3_spec_test.json, the
                                       "val-best builder of the full menu", MWS-layer + learned merging for all nine runs);
  Canto-L (ttbar) (frozen), Canto-L (HNL), Canto-L (ttbar+HNL) (R11 spec)   learned merging;
  Canto-L classifiers + CC             `cc`: connected components on the stored scored test graphs
                                       (RUN/gnn_scores/testset.tar; no network is run) at the cut of hnl.eval.ranking's
                                       rule 'best' on the run's eval_val (highest primary efficiency with fake rate <= FR*;
                                       for Canto-L (ttbar) 0.35 / 0.39 / 0.39, as results/per_sample_v1.csv), and at the
                                       run's e* cut as an identity check against its frozen eval_test.
Steps (paired differences, new minus base): the builder (CC -> LM) at a fixed classifier, the classifier and graph
(acorn -> Canto-L) at a fixed builder, in both orders, and the training sample.

(b) Displaced muons after Wang et al.'s Fig. 3 (selection fixed on truth before this read-out,
results/wang_displaced_selection_v2.md): HNL daughters with >= 5 hits and |eta| < 4, no pT requirement; headline the
positive daughter of each event; also all daughters, the negative one, the one with the same / opposite charge as the
W muon, and the same five with pT > 1 GeV (our targets).

Uncertainties: the preregistered test tables (hnl.eval.bootstrap, B = 500, seed 12345) on the external set's own
resampling units; three-seed models sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (bootstrap.seed_mean); one-seed models the
event bootstrap alone ("sigma") and with the seed sd of the three-seed model of the same family in quadrature
("sigma_proxy": Canto-L (ttbar) for the Canto-L runs, as on the release card). Differences: bootstrap.linear_combination
(paired), sigma_proxy adding the proxy seed variance of each one-seed term. Published values enter with their quoted
errors.

Identity checks (the summary stops before writing if one fails): the on-record rows (results/published_definitions_
comparison.csv, set test_external, every quantity: mean and sigma_total to 1e-12, counts exact; results/training_sample_
test_v1.csv, published_definitions block: value and sigma to 1e-12, counts exact); the primary efficiency of the LM
entries per seed against results/per_sample_v1.csv; the CC rows at e* against the frozen eval_test (cc step).

    python -m hnl.eval.wang_gap cc                 (sbatch; once; writes DATA/runs/wang_gap_v1/<name>/eval_test_bestcut)
    python -m hnl.eval.wang_gap summary --out results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import published_defs as PD
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table

REPO = PD.REPO
DATA = PD.DATA
OUT_RUNS = os.path.join(DATA, "runs", "wang_gap_v1")
SUB = "eval_test_bestcut"
SET = "hnl15_100_external"
TOL = 1e-12
WP = "results/working_points_v1.json"
SPEC_V3 = "results/strong_baselines_v3_spec_test.json"
SPEC_TS = "results/training_sample_spec_test_v1.json"
REF_PD = "results/published_definitions_comparison.csv"
REF_TS = "results/training_sample_test_v1.csv"
REF_PS = "results/per_sample_v1.csv"
WANG = {"all_track": (0.9561, 0.0006), "prompt": (0.990, 0.002), "displaced": (0.916, 0.005)}
CANTO_TT = [os.path.join(DATA, "runs", "stage1_final_learned", "l1tl", f"armi_s{s}") for s in (0, 1, 2)]
CANTO_H = os.path.join(DATA, "runs", "stage1_final_learned_ts", "l1tl", "armii_s0")
CANTO_TH = os.path.join(DATA, "runs", "stage1_final_learned_ts", "l1tl", "armiii_s0")
CC_RUNS = {"canto_l_cc": CANTO_TT, "canto_l_hnl_cc": [CANTO_H], "canto_l_tthnl_cc": [CANTO_TH]}
CC_REF_TT = ["CC 0.35", "CC 0.39", "CC 0.39"]       # results/per_sample_v1.csv, rule 'best' (validation only)

LABELS = {
    "acorn_hnl_cc_sstar": "acorn (HNL) + CC at s* (Wang et al.'s reproduction)",
    "acorn_hnl_cc_best": "acorn (HNL) + CC, best cut <= FR*",
    "acorn_hnl_lm": "acorn (HNL) + LM",
    "acorn_tt_cc": "acorn (ttbar) + CC",
    "acorn_tt_lm": "acorn (ttbar) + LM",
    "acorn_tthnl_lm": "acorn (ttbar+HNL) + LM",
    "canto_l_hnl_cc": "Canto-L (HNL) classifier + CC",
    "canto_l_hnl": "Canto-L (HNL) (LM)",
    "canto_l_tthnl_cc": "Canto-L (ttbar+HNL) classifier + CC",
    "canto_l_tthnl": "Canto-L (ttbar+HNL) (LM)",
    "canto_l_cc": "Canto-L (ttbar) classifier + CC",
    "canto_l": "Canto-L (ttbar) (LM, frozen)",
}
PROXY = {"canto_l_hnl": "canto_l", "canto_l_tthnl": "canto_l", "canto_l_hnl_cc": "canto_l_cc",
         "canto_l_tthnl_cc": "canto_l_cc"}
STEPS = [  # (block, step, new, base)
    ("HNL training", "builder: CC (s*) -> LM, acorn (HNL) classifier", "acorn_hnl_lm", "acorn_hnl_cc_sstar"),
    ("HNL training", "builder: CC (best cut) -> LM, acorn (HNL) classifier", "acorn_hnl_lm", "acorn_hnl_cc_best"),
    ("HNL training", "graph and classifier: acorn (HNL) -> Canto-L (HNL), with LM", "canto_l_hnl", "acorn_hnl_lm"),
    ("HNL training", "graph and classifier: acorn (HNL) -> Canto-L (HNL), with CC (best cuts)", "canto_l_hnl_cc",
     "acorn_hnl_cc_best"),
    ("HNL training", "builder: CC -> LM, Canto-L (HNL) classifier", "canto_l_hnl", "canto_l_hnl_cc"),
    ("training sample", "Canto-L with LM: HNL -> ttbar", "canto_l", "canto_l_hnl"),
    ("training sample", "Canto-L with LM: HNL -> ttbar+HNL", "canto_l_tthnl", "canto_l_hnl"),
    ("training sample", "acorn with LM: HNL -> ttbar", "acorn_tt_lm", "acorn_hnl_lm"),
    ("training sample", "acorn with CC: HNL (s*) -> ttbar", "acorn_tt_cc", "acorn_hnl_cc_sstar"),
    ("ttbar training", "builder: CC -> LM, acorn (ttbar) classifier", "acorn_tt_lm", "acorn_tt_cc"),
    ("ttbar training", "graph and classifier: acorn (ttbar) -> Canto-L (ttbar), with LM", "canto_l", "acorn_tt_lm"),
    ("ttbar training", "graph and classifier: acorn (ttbar) -> Canto-L (ttbar), with CC (best cuts)", "canto_l_cc",
     "acorn_tt_cc"),
    ("ttbar training", "builder: CC -> LM, Canto-L (ttbar) classifier", "canto_l", "canto_l_cc"),
    ("total", "Canto-L (ttbar) - acorn (HNL) + CC at s*", "canto_l", "acorn_hnl_cc_sstar"),
]
QS = ("all_track", "prompt", "displaced")
# (b): Fig. 3 displaced-muon selections; headline first
DM_SEL = {
    "pos_h5eta": ("mu+ daughter, >= 5 hits, |eta| < 4 (headline)", lambda t: _h5eta(t) & (t.q > 0)),
    "all_h5eta": ("all daughters, >= 5 hits, |eta| < 4", lambda t: _h5eta(t)),
    "neg_h5eta": ("mu- daughter, >= 5 hits, |eta| < 4", lambda t: _h5eta(t) & (t.q < 0)),
    "samew_h5eta": ("daughter with the W muon's charge, >= 5 hits, |eta| < 4", lambda t: _h5eta(t) & (t.q == t.wq)),
    "oppw_h5eta": ("daughter with the opposite charge, >= 5 hits, |eta| < 4", lambda t: _h5eta(t) & (t.q == -t.wq)),
    "pos_tgt": ("mu+ daughter target (pT > 1 GeV, |eta| < 4, >= 5 hits)", lambda t: _tgt(t) & (t.q > 0)),
    "all_tgt": ("all daughter targets", lambda t: _tgt(t)),
    "neg_tgt": ("mu- daughter target", lambda t: _tgt(t) & (t.q < 0)),
    "samew_tgt": ("daughter target with the W muon's charge", lambda t: _tgt(t) & (t.q == t.wq)),
    "oppw_tgt": ("daughter target with the opposite charge", lambda t: _tgt(t) & (t.q == -t.wq)),
}
DM_MODELS = ("canto_l", "canto_l_hnl", "canto_l_tthnl", "acorn_hnl_lm", "acorn_hnl_cc_sstar", "acorn_hnl_cc_best",
             "acorn_tt_lm", "acorn_tt_cc", "canto_l_cc")


def _h5eta(t):
    return (t.category == "hnl") & (t.n_hits >= 5) & (t.eta.abs() < 4.0)


def _tgt(t):
    return (t.category == "hnl") & t.is_target


def _p(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _git() -> str:
    return PD._git()


def _name(run: str) -> str:
    parts = os.path.normpath(run).split(os.sep)
    return f"{parts[-3]}_{parts[-2]}_{parts[-1]}"


# ------------------------------------------------------------------------------------------------------------- cc
def val_cuts(fr: float) -> dict[str, list[tuple[str, float]]]:
    """Per CC model: [(run, cut)] from hnl.eval.ranking's rule 'best' on each run's eval_val."""
    from hnl.eval import ranking as RK
    spec = [{"label": m, "short": m, "group": "Canto-L classifier",
             "entries": [{"run": r, "source": "eval_val", "rule": "best"} for r in runs]}
            for m, runs in CC_RUNS.items()]
    _, ent = RK.rank(spec, fr, tie_se=True)
    out = {}
    for m, runs in CC_RUNS.items():
        e = ent[ent.label == m].set_index("run")
        wps = [e.loc[r, "working_point"] for r in runs]
        if any("unreachable" in w for w in wps):
            raise SystemExit(f"{m}: FR* unreachable on val ({wps})")
        out[m] = [(r, float(w.split()[1])) for r, w in zip(runs, wps)]
    got = [f"CC {c:.2f}" for _, c in out["canto_l_cc"]]
    if got != CC_REF_TT:
        raise SystemExit(f"Canto-L (ttbar) val cuts {got} differ from {REF_PS} {CC_REF_TT}")
    return out


def cmd_cc(a) -> None:
    from hnl.eval import evaluate_gnn as EG
    fr = float(json.load(open(_p(WP)))["FR_star"])
    cuts = val_cuts(fr)
    print(json.dumps({m: [(_name(r), c) for r, c in v] for m, v in cuts.items()}, indent=1), flush=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    for m, entries in cuts.items():
        for run, cut in entries:
            out = os.path.join(OUT_RUNS, _name(run), SUB)
            if os.path.exists(os.path.join(out, "provenance.json")):
                raise SystemExit(f"{out} exists: the post-freeze evaluation is done once")
            ref_meta = json.load(open(os.path.join(run, "eval_test", "meta.json")))
            if len(ref_meta["cuts"]) != 1:
                raise SystemExit(f"{run}/eval_test: {ref_meta['cuts']} (one e* cut expected)")
            ce = float(ref_meta["cuts"][0])
            local = os.path.join("/tmp", os.environ.get("USER", "u"), f"wg_{job}_{_name(run)}")
            os.makedirs(os.path.join(local, "t"), exist_ok=True)
            try:
                tar = os.path.join(run, "gnn_scores", "testset.tar")
                subprocess.run(["tar", "xf", tar, "-C", os.path.join(local, "t")], check=True)
                EG.main(["--graphs", os.path.join(local, "t"), "--out", out,
                         "--cuts", ",".join(f"{c:.4f}" for c in sorted({cut, ce}))])
            finally:
                shutil.rmtree(local, ignore_errors=True)
            ok = {}
            for f, keys in (("per_particle.parquet", ["event_id", "particle_id"]), ("per_event.parquet", ["event_id"])):
                new = pd.read_parquet(os.path.join(out, f))
                old = pd.read_parquet(os.path.join(run, "eval_test", f))
                ok[f] = PD._same_rows(new[np.isclose(new.cut, ce, atol=1e-6)], old[np.isclose(old.cut, ce, atol=1e-6)],
                                      keys)
            prov = {"what": "post-freeze CC of a Canto-L classifier on its stored scored test graphs at the cut of "
                            "hnl.eval.ranking's rule 'best' on validation (hnl.eval.wang_gap; ledger R12)",
                    "model": m, "run": run, "scores": tar, "cut_best_val": cut, "cut_e_star": ce,
                    "identity_check_e_star": ok, "git": _git(), "slurm_job": job,
                    "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
            print(json.dumps(prov), flush=True)
            if not all(ok.values()):
                raise SystemExit(f"{run}: rows at e* differ from the frozen eval_test")


# -------------------------------------------------------------------------------------------------------- summary
def _spec(label: str) -> list[dict]:
    by = {m["label"]: m for m in json.load(open(_p(SPEC_V3)))["models"]}
    return by[label]["entries"]


def model_entries() -> dict[str, list[tuple[str, str, float, str]]]:
    """name -> [(run, sub, cut, working point)]."""
    wp = json.load(open(_p(WP)))
    ent = {}
    ent["acorn_hnl_cc_sstar"] = [(os.path.join(PD.BASELINES, f"armii_s{s}"), "eval_test", float(wp["s_star"]["ii"]),
                                  f"CC {wp['s_star']['ii']:.2f}") for s in (0, 1, 2)]
    for name, label in (("acorn_hnl_cc_best", "acorn (HNL) + CC, best cut <= FR*"),
                        ("acorn_tt_cc", "acorn (ttbar) + CC, preregistered cut (= best cut)"),
                        ("acorn_hnl_lm", "acorn (HNL) + val-best builder of the full menu (chosen per seed)"),
                        ("acorn_tt_lm", "acorn (ttbar) + val-best builder of the full menu (chosen per seed)"),
                        ("acorn_tthnl_lm", "acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed)"),
                        ("canto_l", "Canto-L (frozen: learned merging)")):
        es = _spec(label)
        if name.endswith("_lm") and not all(e["config"].startswith("mwslmerge,") for e in es):
            raise SystemExit(f"{label}: not learned merging for every seed")
        ent[name] = [(e["run"], e["sub"], float(e["cut"]), e["wp"]) for e in es]
    ts = {m["short"]: m["entries"] for m in json.load(open(_p(SPEC_TS)))}
    ent["canto_l_hnl"] = [(e["run"], e["sub"], float(e["cut"]), e["wp"]) for e in ts["Canto-L (HNL)"]]
    ent["canto_l_tthnl"] = [(e["run"], e["sub"], float(e["cut"]), e["wp"]) for e in ts["Canto-L (ttbar+HNL)"]]
    for m, runs in CC_RUNS.items():
        es = []
        for r in runs:
            pv = json.load(open(os.path.join(OUT_RUNS, _name(r), SUB, "provenance.json")))
            if not all(pv["identity_check_e_star"].values()):
                raise SystemExit(f"{r}: cc identity check failed")
            es.append((r, os.path.join(OUT_RUNS, _name(r), SUB), float(pv["cut_best_val"]),
                       f"CC {pv['cut_best_val']:.2f}"))
        ent[m] = es
    return ent


def w_charge(info: pd.DataFrame) -> pd.DataFrame:
    w = info[info.category == "w"].drop_duplicates("event_id").set_index("event_id").q
    return info.assign(wq=info.event_id.map(w))


def model_rows(bs, runs, info) -> tuple[list[dict], dict]:
    rows, pts = [], {}
    M = TS.masks()
    pops = {q: M[PD.POPS[q][0]] for q in QS}
    pops.update({k: f for k, (_, f) in DM_SEL.items()})
    for name, rr in runs.items():
        for q, mfn in list(pops.items()) + [("fake_rate", None)]:
            if q == "fake_rate":
                p, r = TS.event_ratio(bs, rr, "n_fake", "n_tracks_ge5", (SET,))
                kn = [(int(rt.pe[TS.in_sets(rt.pe.event_id, (SET,))].n_fake.sum()),
                       int(rt.pe[TS.in_sets(rt.pe.event_id, (SET,))].n_tracks_ge5.sum())) for rt in rr]
            else:
                p, r, _ = TS.efficiency(bs, rr, info, mfn, (SET,))
                kn = []
                for rt in rr:
                    t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
                    m = np.asarray(mfn(t), dtype=bool) & TS.in_sets(t.event_id, (SET,))
                    kn.append((int(t.dm_matched.to_numpy()[m].sum()), int(m.sum())))
            pts[(name, q)] = (p, r)
            s = BS.seed_mean(p, r)
            rows.append({"block": "model", "quantity": q, "name": name, "label": LABELS[name], "n_seeds": len(p),
                         "value": s["mean"], "sigma": s["sigma"], "sigma_boot": s["sigma_boot"],
                         "s_seeds": s["s_seeds"] if len(p) > 1 else np.nan,
                         "k": "+".join(str(k) for k, _ in kn), "n": "+".join(str(n) for _, n in kn),
                         "per_seed": " / ".join(f"{x:.6f}" for x in p)})
    by = {(r["name"], r["quantity"]): r for r in rows}
    for r in rows:
        ref = PROXY.get(r["name"])
        sp = by[(ref, r["quantity"])]["s_seeds"] if ref else 0.0
        r["sigma_proxy"] = float(np.hypot(r["sigma"], sp)) if r["n_seeds"] == 1 else r["sigma"]
    return rows, pts


def step_rows(pts, mrows) -> list[dict]:
    by = {(r["name"], r["quantity"]): r for r in mrows}
    out = []
    for q in QS + ("fake_rate",):
        for block, step, new, base in STEPS:
            d = BS.linear_combination([(1.0, *pts[(new, q)]), (-1.0, *pts[(base, q)])])
            extra = sum(by[(PROXY[m], q)]["s_seeds"] ** 2 for m in (new, base)
                        if by[(m, q)]["n_seeds"] == 1 and m in PROXY)
            out.append({"block": block, "quantity": q, "name": step, "label": f"{LABELS[new]} minus {LABELS[base]}",
                        "value": d["value"], "sigma": d["sigma"], "sigma_boot": d["sigma_boot"],
                        "s_seeds": d["sigma_seeds"], "sigma_proxy": float(np.sqrt(d["sigma"] ** 2 + extra))})
        if q in WANG:
            v, e = WANG[q]
            for m in ("acorn_hnl_cc_sstar", "canto_l"):
                r = by[(m, q)]
                out.append({"block": "published", "quantity": q, "name": f"{LABELS[m]} minus Wang et al. (published)",
                            "label": "", "value": r["value"] - v, "sigma": float(np.hypot(r["sigma"], e)),
                            "sigma_boot": np.nan, "s_seeds": np.nan, "sigma_proxy": float(np.hypot(r["sigma"], e))})
    return out


def identity_checks(mrows, runs, info) -> list[tuple[str, bool, str]]:
    out = []
    by = {(r["name"], r["quantity"]): r for r in mrows}
    pdc = pd.read_csv(_p(REF_PD))
    pdc = pdc[pdc.set == "test_external"]
    for name, key in (("canto_l", "canto_l"), ("acorn_hnl_cc_sstar", "acorn_hnl_sstar"),
                      ("acorn_hnl_cc_best", "acorn_hnl_best"), ("acorn_tt_cc", "acorn_ttbar")):
        for q in QS + ("fake_rate",):
            ref = pdc[(pdc.model == key) & (pdc.quantity == q)].iloc[0]
            r = by[(name, q)]
            k = sum(int(x) for x in r["k"].split("+"))
            ok = abs(r["value"] - ref["mean"]) <= TOL and abs(r["sigma"] - ref["sigma_total"]) <= TOL and \
                k == int(ref["k_pooled"])
            out.append((f"{REF_PD} {key} {q}", ok, f"{r['value']:.12f} +- {r['sigma']:.12f} ({k}) vs "
                                                   f"{ref['mean']:.12f} +- {ref['sigma_total']:.12f} ({ref['k_pooled']})"))
    tsc = pd.read_csv(_p(REF_TS))
    tsc = tsc[tsc.block == "published_definitions"]
    for name, key in (("canto_l_hnl", "Canto-L (HNL)"), ("canto_l_tthnl", "Canto-L (ttbar+HNL)")):
        for q in QS + ("fake_rate",):
            ref = tsc[(tsc.model == key) & (tsc.quantity == q)].iloc[0]
            r = by[(name, q)]
            ok = abs(r["value"] - ref["value"]) <= TOL and abs(r["sigma"] - ref["sigma"]) <= TOL and \
                int(r["k"]) == int(ref["k"])
            out.append((f"{REF_TS} {key} {q}", ok, f"{r['value']:.12f} +- {r['sigma']:.12f} vs "
                                                   f"{ref['value']:.12f} +- {ref['sigma']:.12f}"))
    ps = pd.read_csv(_p(REF_PS))
    ps = ps[(ps.split == "test") & (ps.set == SET)]
    for name, key in (("acorn_hnl_lm", "acorn (HNL) val-best (MWS-layer + learned merging)"),
                      ("acorn_tt_lm", "acorn (ttbar) val-best (MWS-layer + learned merging)"),
                      ("acorn_tthnl_lm", "acorn (ttbar+HNL) val-best (MWS-layer + learned merging)")):
        ref = ps[ps.model == key].iloc[0]
        for s, rt in enumerate(runs[name]):
            t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
            m = np.asarray(TS.masks()["primary"](t), dtype=bool) & TS.in_sets(t.event_id, (SET,))
            k = int(t.dm_matched.to_numpy()[m].sum())
            out.append((f"{REF_PS} {key} s{s} primary k", k == int(ref[f"k_s{s}"]), f"{k} vs {ref[f'k_s{s}']}"))
    return out


def _f(v, d=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.{d}f}"


def markdown(mrows, srows, chk, meta) -> tuple[str, str]:
    m = pd.DataFrame(mrows)
    s = pd.DataFrame(srows)
    g = ["# Wang et al.'s evaluation sample: the all-track gap by stage (post-freeze, descriptive)", "",
         f"hnl15_100_external, 4,999 events; Wang et al.'s definitions (hnl.eval.published_defs); configurations fixed "
         f"on validation; ledger R12. Code hnl.eval.wang_gap, git {meta['git']}, job {meta['slurm_job']}, "
         f"{meta['created_utc']}. Identity checks: {sum(ok for _, ok, _ in chk)} of {len(chk)} pass.", "",
         "Percent. sigma: event bootstrap (preregistered test tables) and, for three seeds, the seed term; one-seed "
         "rows: sigma (bootstrap alone) / sigma_proxy (with the seed sd of the three-seed Canto-L (ttbar) model of the "
         "same builder in quadrature).", "",
         "## Models", "", "| model | seeds | all tracks | W muon | HNL daughters | fake rate | working points |",
         "|---|---|---|---|---|---|---|"]
    wps = meta["working_points"]
    for name in LABELS:
        x = m[m.name == name].set_index("quantity")
        cells = []
        for q in QS + ("fake_rate",):
            r = x.loc[q]
            e = f"{_f(r.sigma)}" if r.n_seeds > 1 else f"{_f(r.sigma)} / {_f(r.sigma_proxy)}"
            cells.append(f"{_f(r.value)} ± {e}")
        g.append(f"| {LABELS[name]} | {int(x.loc['all_track'].n_seeds)} | " + " | ".join(cells) + f" | {wps[name]} |")
    g += ["", "Wang et al. (published): all tracks 95.61 ± 0.06, W muon 99.0 ± 0.2, displaced 91.6 ± 0.5 (their "
          "selection), no fake rate.", "", "## Steps (new minus base, points; paired event bootstrap)", "",
          "| block | step | all tracks | W muon | HNL daughters | fake rate |", "|---|---|---|---|---|---|"]
    for block, step, _, _ in STEPS:
        cells = []
        for q in QS + ("fake_rate",):
            r = s[(s.name == step) & (s.quantity == q)].iloc[0]
            sp = "" if abs(r.sigma_proxy - r.sigma) < 1e-12 else f" / {_f(r.sigma_proxy)}"
            cells.append(f"{100 * r.value:+.2f} ± {_f(r.sigma)}{sp}")
        g.append(f"| {block} | {step} | " + " | ".join(cells) + " |")
    pub = s[s.block == "published"]
    g += ["", "## Against the published values (points)", ""]
    for _, r in pub.iterrows():
        g.append(f"- {r.quantity}: {r['name']}: {100 * r.value:+.2f} ± {_f(r.sigma)}")
    g += ["", "## Identity checks", ""] + [f"- {'PASS' if ok else 'FAIL'} {n}: {d}" for n, ok, d in chk]
    dmu = ["# Displaced muons after Wang et al.'s Fig. 3 (post-freeze, descriptive)", "",
           "Selection fixed on truth before this read-out (results/wang_displaced_selection_v2.md; ledger R12): "
           "Wang et al.'s 'displaced' set reads as one HNL daughter per event with >= 5 hits and |eta| < 4 and no pT "
           "requirement (2,887 positive daughters on the 4,999 events against their 2,882 plus two bins whose totals "
           "the figure does not give). Wang et al. published 91.6 ± 0.5 % for their attention GNN + DBSCAN.", "",
           "Percent; sigma as in results/wang_gap_v1.md.", "",
           "| selection | n per seed | " + " | ".join(LABELS[x] for x in DM_MODELS) + " |",
           "|---|---|" + "---|" * len(DM_MODELS)]
    for key, (lab, _) in DM_SEL.items():
        n = m[(m.name == "canto_l") & (m.quantity == key)].iloc[0].n.split("+")[0]
        cells = []
        for name in DM_MODELS:
            r = m[(m.name == name) & (m.quantity == key)].iloc[0]
            e = _f(r.sigma) if r.n_seeds > 1 else f"{_f(r.sigma)} / {_f(r.sigma_proxy)}"
            cells.append(f"{_f(r.value)} ± {e}")
        dmu.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(g) + "\n", "\n".join(dmu) + "\n"


def cmd_summary(a) -> None:
    outs = {k: os.path.join(a.out, f) for k, f in (("csv", "wang_gap_v1.csv"), ("md", "wang_gap_v1.md"),
                                                    ("dcsv", "wang_displaced_muon_v1.csv"),
                                                    ("dmd", "wang_displaced_muon_v1.md"),
                                                    ("meta", "wang_gap_v1_meta.json"))}
    for f in outs.values():
        if os.path.exists(f):
            raise SystemExit(f"{f} exists")
    ent = model_entries()
    runs = {n: [TS.RunTest(r, c, sub=sub) for r, sub, c, _ in es] for n, es in ent.items()}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for rr in runs.values() for rt in rr]))
    info = w_charge(particle_table(ev))
    bs = BS.Bootstrap()
    mrows, pts = model_rows(bs, runs, info)
    chk = identity_checks(mrows, runs, info)
    bad = [c for c in chk if not c[1]]
    if bad:
        for c in bad:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} identity checks failed; nothing written")
    srows = step_rows(pts, mrows)
    meta = {"created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git": _git(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", "local"), "ledger": "docs/feedback/test_use_ledger.md R12",
            "test_use": "post-freeze, descriptive; no configuration, cut, model or rule chosen on test",
            "entries": {n: [list(e) for e in es] for n, es in ent.items()},
            "working_points": {n: " / ".join(e[3].split(" (")[0] for e in es) for n, es in ent.items()},
            "tables": f"hnl.eval.bootstrap B {BS.B} seed {BS.SEED}", "checks": [list(c) for c in chk]}
    df = pd.DataFrame(mrows + srows)
    df[~df.quantity.isin(DM_SEL)].to_csv(outs["csv"], index=False)
    dm = df[(df.block == "model") & df.quantity.isin(DM_SEL)].copy()
    dm["selection"] = dm.quantity.map(lambda k: DM_SEL[k][0])
    dm.to_csv(outs["dcsv"], index=False)
    md1, md2 = markdown(mrows, srows, chk, meta)
    open(outs["md"], "w").write(md1)
    open(outs["dmd"], "w").write(md2)
    json.dump(meta, open(outs["meta"], "w"), indent=1)
    print(md1)
    print(md2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cc")
    s = sub.add_parser("summary")
    s.add_argument("--out", default=os.path.join(REPO, "results"))
    a = ap.parse_args(argv)
    {"cc": cmd_cc, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
