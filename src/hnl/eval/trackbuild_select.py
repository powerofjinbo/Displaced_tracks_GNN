"""Stage-1 C selection on val (DECISIONS.md 2026-10-03, builder selection rule).

For every track-builder family evaluated by hnl.eval.trackbuild_val: the configuration whose pooled-HNL val fake rate
is <= FR* with the highest primary-population efficiency (ties: the lower fake rate). This generalises the per-seed
cut rule to builders with several thresholds; it is not preregistration §4's single-cut rule (largest fake rate <=
FR* on the 0.01 grid), which instead fixes the reference row: CC at the run's own §4 cut, computed from the run's
eval_val (0.01 grid) with the full-precision FR*. Every family's gain and the prompt guard (W-muon and ttbar-prompt
efficiency within 0.2 points of the reference row, preregistration §6) are reported against that row. The "cc"
family of the grid (0.05 steps, builder rule) is listed for information only.

    python -m hnl.eval.trackbuild_select --run RUN_DIR (--wp WORKING_POINTS_JSON | --fr-star FR) [--name trackbuild_val]
        [--baseline-run BASELINE_RUN_DIR]

--wp: FR* (and, for a baseline run, its §4 cut, checked against the recomputed one) from baseline_val's
working_points.json; --fr-star only for previews before the working points exist. For a Stage-1 run
(runs/stage1_<tag>/<NAME>/arm<A>_s<S>) the matching baseline run (runs/baseline_v1/arm<A>_s<S>, or --baseline-run)
gives the preregistered prompt guard (every row against the baseline's CC at its §4 cut) and the screening comparison
(every family against the baseline's same family; DECISIONS.md 2026-10-03 16:16 UTC); the run's own CC row is labelled
as such and its deltas are information only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import curves, cut_at_fake_rate, load_run
from hnl.eval.particles import particle_table

COLS = ["eff_primary", "eff_hnl_targets", "eff_all_targets", "eff_w_muons", "eff_ttbar_prompt", "eff_ttbar_targets",
        "fake_rate_hnl", "fake_rate_ttbar", "dup_rate_all_targets", "pair_eff", "pair_merge_rate"]
GUARD = 0.002


def tie_best(t: pd.DataFrame, n: float, eff: str = "eff_primary", fake: str = "fake_rate_hnl") -> int:
    """Index of the one-standard-error choice (DECISIONS.md 2026-10-04 20:18 UTC): rows with eff >= e_max - SE are
    tied, SE = sqrt(e_max (1 - e_max) / n) (binomial, n particles behind e_max); the tied row with the lowest fake rate
    wins, then the higher efficiency."""
    e = float(t[eff].max())
    se = np.sqrt(max(e * (1.0 - e), 0.0) / float(n))
    tied = t[t[eff] >= e - se - 1e-12]
    return tied.sort_values([fake, eff], ascending=[True, False]).index[0]


def select(c: pd.DataFrame, cfgs: list, fr_star: float, tie_se: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Best configuration per family at fake rate <= FR*; also the families with no such configuration. tie_se: the
    one-standard-error rule (tie_best) instead of the highest primary efficiency (ties: the lower fake rate)."""
    c = c.copy()
    c["family"] = [cfgs[int(i)][0] for i in c.index]
    c["params"] = [",".join(str(x) for x in cfgs[int(i)][1:]) for i in c.index]
    rows, none_ok = [], []
    for fam, g in c.groupby("family"):
        ok = g[g.fake_rate_hnl <= fr_star + 1e-12]
        if ok.empty:
            none_ok.append(fam)
            continue
        if tie_se:
            best = ok.loc[tie_best(ok, float(ok.n_primary.iloc[0]))]
        else:
            best = ok.sort_values(["eff_primary", "fake_rate_hnl"], ascending=[False, True]).iloc[0]
        rows.append({"family": fam, "params": best.params, "n_primary": best.get("n_primary", np.nan),
                     **{k: best.get(k, np.nan) for k in COLS}})
    return pd.DataFrame(rows), none_ok


def llp_blind(c: pd.DataFrame, cfgs: list) -> pd.DataFrame:
    """Per family, the LLP-blind working point (preregistration §6c): the configuration with the highest DM efficiency
    of all ttbar-val targets (the §4 rule on ttbar val; ties: the lower ttbar fake rate); no HNL event is used."""
    c = c.copy()
    c["family"] = [cfgs[int(i)][0] for i in c.index]
    c["params"] = [",".join(str(x) for x in cfgs[int(i)][1:]) for i in c.index]
    rows = []
    for fam, g in c.groupby("family"):
        best = g.sort_values(["eff_ttbar_targets", "fake_rate_ttbar"], ascending=[False, True]).iloc[0]
        rows.append({"family": fam, "params": best.params, **{k: best.get(k, np.nan) for k in COLS}})
    return pd.DataFrame(rows)


def is_baseline_run(run: str) -> bool:
    return os.sep + "baseline_" in os.path.abspath(run)


def matching_baseline(run: str) -> str | None:
    """runs/stage1_<tag>/<NAME>/arm<A>_s<S> -> runs/baseline_v1/arm<A>_s<S> (None if the path does not match)."""
    m = re.search(r"^(.*)/stage1_[^/]+/[^/]+/(arm(i{1,3})_s\d+)$", os.path.abspath(run))
    return os.path.join(m.group(1), "baseline_v1", m.group(2)) if m else None


def reference_row(run: str, fr_star: float, events: np.ndarray, wp: dict | None) -> dict:
    """CC at the run's preregistered §4 cut (0.01 grid of its eval_val), with the reachable flag."""
    bpp, bpe = load_run(run)
    if set(np.unique(bpe.event_id)) != set(events):
        raise SystemExit(f"eval_val of {run} and the track-building table cover different events")
    bc = curves(bpp, bpe, particle_table(np.unique(bpe.event_id)))
    cut, reachable = cut_at_fake_rate(bc, fr_star)
    m = re.search(r"arm(i{1,3})_s(\d+)$", os.path.normpath(run))
    if wp is not None and m and os.sep + "baseline_" in run:
        want = wp["cuts_at_FR_star"].get(m.group(1), {}).get(m.group(2))
        if want is not None and abs(float(want) - cut) > 1e-9:
            raise SystemExit(f"recomputed §4 cut {cut} != working points' {want} for {run}")
    row = bc.loc[cut]
    label = "baseline CC (§4 cut)" if is_baseline_run(run) else "CC (§4 cut, this run)"
    return {"family": label, "params": f"{cut:.2f}", "reachable": bool(reachable),
            **{k: row.get(k, np.nan) for k in COLS}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--wp", default=None, help="working_points.json of baseline_val (FR* at full precision)")
    ap.add_argument("--fr-star", type=float, default=None, help="previews only, without working points")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="trackbuild_val", help="sub-directory written by trackbuild_val")
    ap.add_argument("--baseline-run", default=None, help="for a Stage-1 run: the baseline run of the same arm and seed")
    a = ap.parse_args(argv)
    base_run = None
    if not is_baseline_run(a.run):
        base_run = a.baseline_run or matching_baseline(a.run)
        if base_run is None or not os.path.isdir(base_run):
            raise SystemExit(f"{a.run} is not a baseline run: give --baseline-run (found {base_run})")
    wp = json.load(open(a.wp)) if a.wp else None
    if wp is not None:
        fr = float(wp["FR_star"])
        if a.fr_star is not None and abs(a.fr_star - fr) > 1e-12:
            raise SystemExit(f"--fr-star {a.fr_star} differs from the working points' FR* {fr}")
    elif a.fr_star is not None:
        fr = a.fr_star
    else:
        raise SystemExit("give --wp (or --fr-star for a preview)")
    d = os.path.join(a.run, a.name)
    out = a.out or d
    pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
    cfgs = json.load(open(os.path.join(d, "configs.json")))
    events = np.unique(pe.event_id)
    c = curves(pp, pe, particle_table(events))
    c.to_csv(os.path.join(out, "curves_by_config.csv"))
    sel, none_ok = select(c, cfgs, fr)
    for fam in none_ok:
        print(f"WARNING: family {fam} has no configuration with fake rate <= FR* {fr}", file=sys.stderr)
    ref = reference_row(a.run, fr, events, wp)
    rows = [pd.DataFrame([ref])]
    if len(sel):
        sel["family"] = [f + " (builder rule, info)" if f == "cc" else f for f in sel.family]
        rows.append(sel)
    tab = pd.concat(rows, ignore_index=True)
    keys = ("eff_primary", "eff_w_muons", "eff_ttbar_prompt")
    guard = lambda t, sfx: (t["d_eff_w_muons" + sfx] >= -GUARD - 1e-12) & (t["d_eff_ttbar_prompt" + sfx] >= -GUARD - 1e-12)
    # preregistered prompt guard (§6.3): against the baseline's CC at its §4 cut (for a baseline run: its own CC row)
    bref = ref if base_run is None else reference_row(base_run, fr, events, wp)
    for k in keys:
        tab[f"d_{k}_vs_baseline_cc"] = tab[k] - bref[k]
    tab["prompt_guard_vs_baseline_cc"] = guard(tab, "_vs_baseline_cc")
    meta = {"FR_star": fr, "source": a.wp or "--fr-star (preview)", "reference_cut": ref["params"],
            "reference_reachable": ref["reachable"], "families_without_config": none_ok,
            "baseline_run": base_run, "baseline_cut": bref["params"], "baseline_reachable": bref["reachable"]}
    if base_run is not None:
        for k in keys:                                                 # information: against this run's own CC
            tab[f"d_{k}_vs_own_cc"] = tab[k] - ref[k]
        # screening comparison (DECISIONS 16:16 UTC): each family against the baseline's same family at FR*
        bdir = os.path.join(base_run, a.name)
        if os.path.exists(os.path.join(bdir, "per_particle.parquet")):
            bpe = pd.read_parquet(os.path.join(bdir, "per_event.parquet"))
            if set(np.unique(bpe.event_id)) != set(events):
                raise SystemExit(f"{bdir} covers different events")
            bc = curves(pd.read_parquet(os.path.join(bdir, "per_particle.parquet")), bpe, particle_table(events))
            bsel, _ = select(bc, json.load(open(os.path.join(bdir, "configs.json"))), fr)
            same = {f: r for f, r in bsel.set_index("family").iterrows()} if len(bsel) else {}
            same["CC (§4 cut, this run)"] = pd.Series(bref)
            for k in keys:
                tab[f"d_{k}_same_builder"] = [r[k] - same[f.replace(" (builder rule, info)", "")][k]
                                              if f.replace(" (builder rule, info)", "") in same else np.nan
                                              for f, r in zip(tab.family, tab.to_dict("records"))]
            tab["prompt_guard_same_builder"] = guard(tab, "_same_builder")
            meta["baseline_builder_table"] = bdir
        else:
            print(f"WARNING: no {bdir}: no same-builder comparison", file=sys.stderr)
    tab.to_csv(os.path.join(out, f"selected_at_FR_{fr:.6f}.csv"), index=False)
    with open(os.path.join(out, f"selected_at_FR_{fr:.6f}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    lb = llp_blind(c, cfgs)
    lb.to_csv(os.path.join(out, "llp_blind_working_points.csv"), index=False)
    print("LLP-blind working points (ttbar val only; preregistration §6c):")
    print(lb.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
