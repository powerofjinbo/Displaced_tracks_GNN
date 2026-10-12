"""Final ttbar-only pipeline from val (rules R6 / R8, DECISIONS.md 2026-10-03 21:59 UTC), frozen before the one-time test.

For every full-budget pipeline (a Stage-1 run directory per seed: classifier + graphs) and every track-builder family of
its val grid (hnl.eval.trackbuild_val, with the learned merging model of the same run): the builder rule's
configuration per seed (pooled-HNL fake rate <= FR*, highest primary efficiency; hnl.eval.trackbuild_select.select)
and its efficiencies; the prompt guard of section 6 per seed against the matching baseline's CC at its section-4 cut
(acorn arm (i), the S1 comparison) and against the reference arm's CC (the primary claim), as 3-seed means
(drop <= 0.2 points). R6: within a pipeline, the family with the highest 3-seed-mean primary efficiency that passes
both guards; R8: the pipeline whose R6 family has the highest 3-seed-mean primary efficiency. Per seed the frozen test
inputs: the FR* configuration, the LLP-blind configuration of the same family (ttbar val only, section 6c), the
classifier's e* cut (RUN/edge_tpr_estar_efpr.json from the same working points) and the merge model.

    python -m hnl.eval.final_select select --wp results/working_points_v1.json --out results/final_selection_v1.json \
        --pipeline l1t=RUNS/stage1_final/l1t:ml_graphs --pipeline l1tu=RUNS/stage1_final_union/l1tu:union_graphs \
        [--name trackbuild_val_lmerge] [--seeds 0,1,2]
    python -m hnl.eval.final_select spec --selection results/final_selection_v1.json --baselines RUNS/baseline_v1 \
        --out SPEC.json                    (the hnl.eval.claims spec of the one-time test: primary, S1, S1b)

The test tables a spec points to: RUN/trackbuild_test and RUN/trackbuild_test_llpblind (trackbuild_val --config, the
configuration index 0 plays the cut), RUN/eval_test (evaluate_gnn at the e* cut) and RUN/edge_tpr_test_events.parquet
(edge_tpr --per-event at the e* cut), written by slurm/stage1/final_test.sbatch; for the baselines their eval_test at
the working points' cuts and edge_tpr_test_events.parquet at their e* cuts (slurm/stage1/edge_dd_test.sbatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import curves
from hnl.eval.particles import particle_table
from hnl.eval.trackbuild_select import GUARD, llp_blind, matching_baseline, reference_row, select, tie_best

KEYS = ("eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt", "fake_rate_hnl", "eff_ttbar_targets",
        "fake_rate_ttbar")
PROMPT = ("eff_w_muons", "eff_ttbar_prompt")


def _grid(run: str, name: str, fr: float):
    d = os.path.join(run, name)
    pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
    cfgs = json.load(open(os.path.join(d, "configs.json")))
    events = np.unique(pe.event_id)
    c = curves(pp, pe, particle_table(events))
    sel, none_ok = select(c, cfgs, fr, tie_se=True)              # one-standard-error rule (2026-10-04)
    sel["grid"] = name
    return sel, none_ok, llp_blind(c, cfgs).assign(grid=name), events


def run_rows(run: str, names: str, wp: dict, ref_arm: str) -> tuple[pd.DataFrame, dict]:
    """Per family of one run's val grids (comma list of trackbuild_val* directories; a family found in several is taken
    from the first, the grids being deterministic): the FR* configuration and the LLP-blind one, with the guard
    deltas."""
    fr = float(wp["FR_star"])
    sels, llps, none_ok, events = [], [], [], None
    for name in names.split(","):
        if not os.path.exists(os.path.join(run, name, "configs.json")):
            continue
        sel, nok, llp, ev = _grid(run, name, fr)
        if events is not None and set(ev) != set(events):
            raise SystemExit(f"{run}: the grids {names} cover different events")
        events = ev
        sels.append(sel)
        llps.append(llp)
        none_ok += nok
    if not sels:
        raise SystemExit(f"{run}: none of the grids {names}")
    sel = pd.concat(sels, ignore_index=True).drop_duplicates("family", keep="first")
    llp = pd.concat(llps, ignore_index=True).drop_duplicates("family", keep="first").set_index("family")
    d = ",".join(os.path.join(run, n) for n in names.split(","))
    base = matching_baseline(run)
    if base is None or not os.path.isdir(base):
        raise SystemExit(f"no matching baseline run for {run}")
    seed = base.rstrip("/").rsplit("_s", 1)[1]
    refs = {"arm_i": reference_row(base, fr, events, wp),
            "ref": reference_row(os.path.join(os.path.dirname(base), f"arm{ref_arm}_s{seed}"), fr, events, wp)}
    rows = []
    for r in sel.to_dict("records"):
        row = {"family": r["family"], "grid": r["grid"], "config": f"{r['family']},{r['params']}",
               "n_primary": r["n_primary"], **{k: r.get(k, np.nan) for k in KEYS}}
        for tag, ref in refs.items():
            for k in PROMPT + ("eff_primary",):
                row[f"d_{k}_vs_{tag}"] = r[k] - ref[k]
        row["llp_blind_config"] = f"{r['family']},{llp.loc[r['family'], 'params']}"
        rows.append(row)
    meta = {"run": run, "grid": d, "families_without_config": none_ok,
            "baseline_cuts": {tag: ref["params"] for tag, ref in refs.items()},
            "baseline_reachable": {tag: ref["reachable"] for tag, ref in refs.items()},
            "baseline_rows": {tag: {k: float(ref[k]) for k in KEYS if k in ref} for tag, ref in refs.items()}}
    return pd.DataFrame(rows), meta


def cut_e(run: str, wp: dict) -> float | None:
    """The classifier's val cut at e* (hnl.eval.edge_tpr --wp, RUN/edge_tpr_estar_efpr.json), None if not computed."""
    p = os.path.join(run, "edge_tpr_estar_efpr.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    if abs(float(m["target_efpr"]) - float(wp["e_star"])) > 1e-12:
        raise SystemExit(f"{p} was computed for edge FPR {m['target_efpr']}, not e* = {wp['e_star']}")
    if not m.get("reachable", True):
        print(f"WARNING: e* unreachable for {run}: minimum-FPR cut {m['cut']}", file=sys.stderr)
    return float(m["cut"])


def choose(tab: pd.DataFrame) -> pd.DataFrame:
    """3-seed means per (pipeline, family) with the guards; R6 marks the best guarded family per pipeline, R8 the
    best pipeline among the R6 families; "best" is the one-standard-error rule (tie_best: within one binomial SE of the
    highest mean primary efficiency, pooled count, the lowest mean fake rate; DECISIONS.md 2026-10-04 20:18 UTC)."""
    num = [c for c in tab.columns if c.startswith(("eff_", "fake_", "d_"))]
    g = tab.groupby(["pipeline", "family"])
    m = g[num].mean()
    m["n_seeds"] = g.size()
    m["n_primary"] = g["n_primary"].sum()                     # pooled over the seeds, for the tie rule's SE
    m["guard_arm_i"] = (m.d_eff_w_muons_vs_arm_i >= -GUARD - 1e-12) & (m.d_eff_ttbar_prompt_vs_arm_i >= -GUARD - 1e-12)
    m["guard_ref"] = (m.d_eff_w_muons_vs_ref >= -GUARD - 1e-12) & (m.d_eff_ttbar_prompt_vs_ref >= -GUARD - 1e-12)
    m = m.reset_index()
    full = m.n_seeds == m.n_seeds.max()
    ok = m[m.guard_arm_i & m.guard_ref & full]
    m["R6"] = False
    for p, gp in ok.groupby("pipeline"):                       # one-standard-error rule on 3-seed means
        best = tie_best(gp, float(gp.loc[gp.eff_primary.idxmax(), "n_primary"]))
        m.loc[best, "R6"] = True
    m["R8"] = False
    if m.R6.any():
        r6 = m[m.R6]
        m.loc[tie_best(r6, float(r6.loc[r6.eff_primary.idxmax(), "n_primary"])), "R8"] = True
    return m.sort_values(["pipeline", "eff_primary"], ascending=[True, False]).reset_index(drop=True)


def claims_spec(sel: dict, baselines: str) -> dict:
    """The hnl.eval.claims spec of the selected candidate against acorn arm (i) and the reference arm."""
    wp = json.load(open(sel["working_points"]))
    if hashlib.sha256(open(sel["working_points"], "rb").read()).hexdigest() != sel["working_points_sha256"]:
        raise SystemExit(f"{sel['working_points']} changed since the selection")
    cand = sel["candidate"]
    if cand is None:
        raise SystemExit("the selection has no candidate")
    seeds = sel["pipelines"][cand["pipeline"]]["seeds"]
    new = [{"run": d["run"], "cut": 0, "sub": "trackbuild_test", "cut_e": d["cut_e"], "sub_e": "eval_test"}
           for _, d in sorted(seeds.items())]
    blind = [{"run": d["run"], "cut": 0, "sub": "trackbuild_test_llpblind"} for _, d in sorted(seeds.items())]
    def acorn(arm):
        return [{"run": os.path.join(baselines, f"arm{arm}_s{s}"), "cut": wp["cuts_at_FR_star"][arm][s],
                 "cut_e": wp["cuts_at_e_star"][arm][s], "sub": "eval_test"} for s in sorted(seeds)]
    ref = sel["reference_arm"]
    return {"models": {"new": new, "new_llp_blind": blind, "arm_i": acorn("i"), "ref": acorn(ref)},
            "dd": {"new": [os.path.join(d["run"], "edge_tpr_test_events.parquet") for _, d in sorted(seeds.items())],
                   "arm_i": [os.path.join(baselines, f"armi_s{s}", "edge_tpr_test_events.parquet")
                             for s in sorted(seeds)]},
            "claims": {"primary": {"new": "new", "base": "ref"}, "S1": {"new": "new", "arm_i": "arm_i", "ref": "ref"},
                       "S1b": {"new": "new", "new_llp_blind": "new_llp_blind", "arm_i": "arm_i"}},
            "candidate": cand, "selection_created_utc": sel["created_utc"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("spec")
    sp.add_argument("--selection", required=True)
    sp.add_argument("--baselines", required=True, help="runs/baseline_v1")
    sp.add_argument("--out", required=True)
    ap_ = sub.add_parser("select")
    ap_.add_argument("--wp", required=True)
    ap_.add_argument("--out", required=True, help="selection JSON (and <out>_table.csv / _seeds.csv)")
    ap_.add_argument("--pipeline", action="append", required=True, help="NAME=DIR:GRAPHS_SUBDIR (DIR/armi_s<seed>)")
    ap_.add_argument("--name", default="trackbuild_val_ext,trackbuild_val_lmerge,trackbuild_val",
                     help="comma list of val grid directories per run (families from the first that has them)")
    ap_.add_argument("--allow-missing-grid", action="store_true",
                     help="previews only: a run may lack some of the --name grids")
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--allow-missing-cut-e", action="store_true", help="previews only")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        spec = claims_spec(json.load(open(a.selection)), a.baselines)
        with open(a.out, "w") as f:
            json.dump(spec, f, indent=1, default=float)
        print(json.dumps(spec, indent=1, default=float))
        return
    wp = json.load(open(a.wp))
    ref_arm = wp["reference_arm"]
    rows, metas, seeds_out = [], {}, {}
    for spec in a.pipeline:
        name, rest = spec.split("=", 1)
        pdir, graphs = rest.rsplit(":", 1)
        for s in a.seeds.split(","):
            run = os.path.join(pdir, f"armi_s{s}")
            missing = [n for n in a.name.split(",") if not os.path.exists(os.path.join(run, n, "configs.json"))]
            if missing and not a.allow_missing_grid:
                raise SystemExit(f"{run}: missing val grids {missing} (every run needs every grid of --name)")
            t, meta = run_rows(run, a.name, wp, ref_arm)
            t.insert(0, "seed", int(s))
            t.insert(0, "pipeline", name)
            rows.append(t)
            ce = cut_e(run, wp)
            if ce is None and not a.allow_missing_cut_e:
                raise SystemExit(f"no e* cut for {run}: run slurm/stage1/edge_tpr_estar.sbatch with {a.wp}")
            metas[f"{name}_s{s}"] = {**meta, "cut_e": ce}
            seeds_out.setdefault(name, {"dir": pdir, "graphs": graphs, "seeds": {}})["seeds"][s] = {
                "run": run, "cut_e": ce, "merge_model": os.path.join(run, "lmerge", "model.joblib")}
    tab = pd.concat(rows, ignore_index=True)
    m = choose(tab)
    for name, p in seeds_out.items():
        sel = m[(m.pipeline == name) & m.R6]
        p["family"] = None if sel.empty else sel.family.iloc[0]
        for s, d in p["seeds"].items():
            if p["family"] is None:
                continue
            r = tab[(tab.pipeline == name) & (tab.seed == int(s)) & (tab.family == p["family"])].iloc[0]
            d["config"], d["llp_blind_config"] = r.config, r.llp_blind_config
    win = m[m.R8]
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "working_points": a.wp,
           "working_points_sha256": hashlib.sha256(open(a.wp, "rb").read()).hexdigest(), "FR_star": wp["FR_star"],
           "e_star": wp["e_star"], "reference_arm": ref_arm, "grid": a.name,
           "rules": "builder rule, R6 (best guarded family per pipeline, 3-seed means) and R8 (best pipeline), each "
                    "with the one-standard-error tie rule: lowest fake rate within one binomial SE of the best",
           "candidate": None if win.empty else {"pipeline": win.pipeline.iloc[0], "family": win.family.iloc[0]},
           "pipelines": seeds_out, "runs": metas}
    base = os.path.splitext(a.out)[0]
    m.to_csv(base + "_table.csv", index=False)
    tab.to_csv(base + "_seeds.csv", index=False)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    pd.set_option("display.width", 250)
    cols = ["pipeline", "family", "n_seeds", "eff_primary", "eff_w_muons", "eff_ttbar_prompt", "fake_rate_hnl",
            "d_eff_primary_vs_arm_i", "d_eff_primary_vs_ref", "guard_arm_i", "guard_ref", "R6", "R8"]
    print(m[cols].round(4).to_string(index=False))
    print(json.dumps(out["candidate"]))
    if win.empty:
        print("WARNING: no pipeline passes the guards", file=sys.stderr)


if __name__ == "__main__":
    main()
