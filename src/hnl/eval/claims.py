"""The preregistered claims on the test sets (docs/preregistration.md sections 6, 6b, 6c), from one model spec.

Every model is a list of seeds, each a RunTest (hnl.eval.test_stats: track level from an evaluate_gnn table at the
cut fixed on val, or from a track-building table of hnl.eval.trackbuild_val --config; edge level from the classifier's
evaluate_gnn table at its e* cut). All numbers use the same fixed bootstrap tables, so differences are paired;
seed terms as in hnl.eval.bootstrap.

  primary (6):  new vs the reference baseline on the primary population pooled over the HNL test sets: inefficiency
                halved (point estimates), difference >= 3 sigma, prompt guard (W muons on the HNL sets and ttbar prompt
                targets: drop <= 0.2 points);
  S1 (6b):      the ttbar-only pipeline vs acorn arm (i) (>= 3 sigma), non-inferior to the reference (difference -
                2 sigma > -1 point), prompt guard against arm (i);
  S1b (6c):     dD (edge TPR at |d0| 100-300 mm minus 0.1-1 mm, HNL-daughter edges of the joint HNL test set, each
                classifier at its e* cut fixed on val; per-event counts of hnl.eval.edge_tpr --per-event) better than
                arm (i) at >= 2 sigma with dD >= -2 points, and at the LLP-blind working point the primary efficiency
                above arm (i) by >= 3 sigma with the fake rate not above arm (i)'s and the prompt guard.

Spec (JSON): {"models": {NAME: [{"run", "cut", "sub"?, "cut_e"?, "sub_e"?}, ...]},
              "dd": {NAME: [per-event parquet, one per seed]},
              "claims": {"primary": {"new", "base"}, "S1": {"new", "arm_i", "ref"},
                         "S1b": {"new", "new_llp_blind", "arm_i"}}}

    python -m hnl.eval.claims --spec SPEC.json --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table

GUARD = 0.002
DD_SETS = ("hnl_joint_test",)


def load_models(spec: dict) -> dict:
    out = {}
    for name, seeds in spec["models"].items():
        out[name] = [TS.RunTest(s["run"], float(s["cut"]), None if s.get("cut_e") is None else float(s["cut_e"]),
                                sub=s.get("sub", "eval_test"), sub_e=s.get("sub_e")) for s in seeds]
    return out


def eff(bs, runs, info, pop, sets):
    p, r, n = TS.efficiency(bs, runs, info, TS.masks()[pop], sets)
    return p, r


def mean(pr) -> float:
    return float(np.mean(pr[0]))


def dd_seeds(bs, files: list[str]):
    """Per seed (point, replicates) of dD from per-event counts (pooled over DD_SETS)."""
    pts, reps = [], []
    for f in files:
        e = pd.read_parquet(f)
        m = TS.in_sets(e.event_id, DD_SETS)
        e = e[m]
        ph, rh = TS.ratio_per_set(bs, e.event_id, e.k_high, e.n_high, DD_SETS)
        pl, rl = TS.ratio_per_set(bs, e.event_id, e.k_low, e.n_low, DD_SETS)
        pts.append(ph - pl)
        reps.append(rh - rl)
    return pts, reps


def guard(bs, new, base, info) -> dict:
    w = mean(eff(bs, new, info, "w_muons", TS.HNL_SETS)) - mean(eff(bs, base, info, "w_muons", TS.HNL_SETS))
    t = (mean(eff(bs, new, info, "ttbar_prompt", ("ttbar_test",)))
         - mean(eff(bs, base, info, "ttbar_prompt", ("ttbar_test",))))
    return {"d_eff_w_muons": w, "d_eff_ttbar_prompt": t, "pass": bool(w >= -GUARD - 1e-12 and t >= -GUARD - 1e-12)}


def fake_rate(bs, runs, sets=TS.HNL_SETS) -> float:
    return mean(TS.event_ratio(bs, runs, "n_fake", "n_tracks_ge5", sets))


def evaluate(spec: dict) -> dict:
    bs = BS.Bootstrap()
    models = load_models(spec)
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for runs in models.values() for rt in runs]))
    info = particle_table(ev)
    out = {"models": {k: [{"run": rt.run, "cut": rt.cut, "cut_e": rt.cut_e} for rt in v] for k, v in models.items()}}
    C = spec.get("claims", {})
    if "primary" in C:
        new, base = models[C["primary"]["new"]], models[C["primary"]["base"]]
        res = BS.primary_claim(eff(bs, new, info, "primary", TS.HNL_SETS), eff(bs, base, info, "primary", TS.HNL_SETS))
        res["criterion3_prompt_guard"] = guard(bs, new, base, info)
        res["holds"] = bool(res["criterion1_halved"] and res["criterion2_3sigma"]
                            and res["criterion3_prompt_guard"]["pass"])
        out["primary"] = res
    if "S1" in C:
        new, ai, ref = (models[C["S1"][k]] for k in ("new", "arm_i", "ref"))
        e_new = eff(bs, new, info, "primary", TS.HNL_SETS)
        d_i = BS.linear_combination([(1.0, *e_new), (-1.0, *eff(bs, ai, info, "primary", TS.HNL_SETS))])
        d_ref = BS.linear_combination([(1.0, *e_new), (-1.0, *eff(bs, ref, info, "primary", TS.HNL_SETS))])
        g = guard(bs, new, ai, info)
        out["S1"] = {"vs_arm_i": d_i, "criterion1_3sigma": bool(d_i["z"] >= 3.0), "vs_reference": d_ref,
                     "criterion2_noninferior": bool(d_ref["value"] - 2 * d_ref["sigma"] > -0.01),
                     "criterion3_prompt_guard": g}
        out["S1"]["holds"] = bool(out["S1"]["criterion1_3sigma"] and out["S1"]["criterion2_noninferior"] and g["pass"])
    if "S1b" in C:
        c = C["S1b"]
        new, blind, ai = models[c["new"]], models[c["new_llp_blind"]], models[c["arm_i"]]
        dn, di = dd_seeds(bs, spec["dd"][c["new"]]), dd_seeds(bs, spec["dd"][c["arm_i"]])
        d = BS.linear_combination([(1.0, *dn), (-1.0, *di)])
        c1 = bool(d["z"] >= 2.0 and float(np.mean(dn[0])) >= -0.02)
        e_b = eff(bs, blind, info, "primary", TS.HNL_SETS)
        d2 = BS.linear_combination([(1.0, *e_b), (-1.0, *eff(bs, ai, info, "primary", TS.HNL_SETS))])
        fr_b, fr_i = fake_rate(bs, blind), fake_rate(bs, ai)
        g = guard(bs, blind, ai, info)
        c2 = bool(d2["z"] >= 3.0 and fr_b <= fr_i + 1e-12 and g["pass"])
        out["S1b"] = {"dD_new": BS.seed_mean(*dn), "dD_arm_i": BS.seed_mean(*di), "dD_difference": d,
                      "criterion1": c1, "llp_blind_vs_arm_i": d2, "fake_rate_llp_blind": fr_b,
                      "fake_rate_arm_i": fr_i, "prompt_guard": g, "criterion2": c2, "holds": bool(c1 and c2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    spec = json.load(open(a.spec))
    res = evaluate(spec)
    with open(os.path.join(a.out, "claims.json"), "w") as f:
        json.dump({"spec": spec, "result": res}, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
