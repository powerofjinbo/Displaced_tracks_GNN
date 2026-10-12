"""Stage-0 val analysis (docs/preregistration.md §4, §6): working points of the acorn baseline arms.

Per arm and seed, from <run>/eval_val (evaluate_gnn on the val splits of all three labeled samples, cut grid 0.01):
efficiency per population, track fake rate, duplicate rate, edge efficiency and edge FPR, all as functions of the
edge-score cut. Then:
  * s* per arm: the cut on {0.05, ..., 0.95} maximising the seed-mean DM efficiency of all targets on the pooled HNL
    val splits;
  * FR* and the reference arm: for each candidate arm A, FR*_A = seed-mean fake rate of A at s*_A; every arm is then
    evaluated at FR*_A, each seed at its own cut on the 0.01 grid with the largest fake rate <= FR*_A (ties: the
    lowest cut); A is consistent if it has the highest seed-mean primary-population efficiency at FR*_A. The
    reference is the consistent arm with the highest such efficiency; if none is consistent, the arm with the highest
    primary efficiency at its own s* (DECISIONS.md 2026-10-03, fixed before the baselines were evaluated);
  * e*: the reference's seed-mean edge FPR at s*; per seed, the cut whose edge FPR is the largest value <= e*.

    python -m hnl.eval.baseline_val --runs RUN_ROOT --out OUT [--arms i,ii,iii --seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.particles import particle_table, split_event_id

HNL_CODES = (2, 4)          # hnl10_100, hnl15_200 (val); the pooled HNL val population
TTBAR_CODE = 1
COARSE = np.round(np.arange(0.05, 0.951, 0.05), 4)


def populations(p: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over the particle table (one row per (event_id, particle_id))."""
    code = split_event_id(p.event_id.to_numpy())[0]
    hnl_ev = np.isin(code, HNL_CODES)
    t = p.is_target.to_numpy()
    cat = p.category.to_numpy()
    d0 = p.d0.abs().to_numpy()
    rv = p.rv.to_numpy()
    disp = (d0 >= 10) | (rv >= 100)
    return {
        "all_targets": hnl_ev & t,
        "primary": hnl_ev & t & (cat == "hnl") & np.nan_to_num(disp, nan=False).astype(bool),
        "hnl_targets": hnl_ev & t & (cat == "hnl"),
        "w_muons": hnl_ev & t & (cat == "w"),
        "other_targets_hnl_events": hnl_ev & t & (cat == "other"),
        "ttbar_targets": (code == TTBAR_CODE) & t,
        "ttbar_prompt": (code == TTBAR_CODE) & t & (d0 < 1.0),
        # brief §5d: also the subset with >= 3 pixel hits
        "hnl_targets_pix3": hnl_ev & t & (cat == "hnl") & (p.n_pixel_hits.to_numpy() >= 3),
        "w_muons_pix3": hnl_ev & t & (cat == "w") & (p.n_pixel_hits.to_numpy() >= 3),
    }


def curves(pp: pd.DataFrame, pe: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """One row per cut: efficiencies per population, duplicate rate, fake rate, edge metrics (pooled HNL val; ttbar
    fake rate separately)."""
    pops = populations(info)
    flags = pd.DataFrame({"event_id": info.event_id.to_numpy(), "particle_id": info.particle_id.to_numpy(),
                          **{f"pop_{k}": v for k, v in pops.items()}})
    flags = flags[flags.filter(like="pop_").any(axis=1)]
    t = pp.merge(flags, on=["event_id", "particle_id"], how="inner")
    t["dup"] = (t.n_sm_tracks.astype(np.int32) - 1).clip(lower=0)
    key = "config" if "config" in t else "cut"
    if key == "cut":
        # per-particle cuts are float32, per-event float64: one rounded float64 key for both before aligning
        t["cut"] = np.round(t.cut.astype(np.float64), 4)
        pe = pe.assign(cut=np.round(pe.cut.astype(np.float64), 4))
    rows = {}
    for name in pops:
        g = t[t[f"pop_{name}"]].groupby(key)
        rows[f"eff_{name}"] = g.dm_matched.mean()
        rows[f"n_{name}"] = g.size()
        if "dm_nofp" in t:
            rows[f"eff_nofp_{name}"] = g.dm_nofp.mean()
        if name == "all_targets":
            rows["dup_rate_all_targets"] = g.dup.sum() / g.size()
    out = pd.DataFrame(rows)
    code = split_event_id(pe.event_id.to_numpy())[0]
    edge_cols = [c for c in ("tp", "fp", "n_true", "n_edges", "n_true_in_graph") if c in pe]
    for label, m in (("hnl", np.isin(code, HNL_CODES)), ("ttbar", code == TTBAR_CODE)):
        s = pe[m].groupby(key)[["n_fake", "n_tracks_ge5"] + edge_cols].sum()
        out[f"fake_rate_{label}"] = s.n_fake / s.n_tracks_ge5
        if len(edge_cols) == 5:
            out[f"edge_eff_{label}"] = s.tp / s.n_true
            out[f"edge_fpr_{label}"] = s.fp / (s.n_edges - s.n_true_in_graph)
            out[f"edge_purity_{label}"] = s.tp / (s.tp + s.fp)
    if "tp_mu" in pe:
        for label, m in (("hnl", np.isin(code, HNL_CODES)), ("ttbar", code == TTBAR_CODE)):
            s = pe[m].groupby(key)[["tp_mu", "n_true_mu", "tp_mu_clean", "n_true_mu_clean"]].sum()
            out[f"edge_eff_mu_{label}"] = s.tp_mu / s.n_true_mu
            out[f"edge_eff_mu_clean_{label}"] = s.tp_mu_clean / s.n_true_mu_clean
    hp = pe[np.isin(code, HNL_CODES)]
    if "pair_both" in hp:
        hp = hp.dropna(subset=["pair_both"])
        out["pair_eff"] = hp.groupby(key).pair_both.mean()
        out["pair_merge_rate"] = hp.groupby(key).pair_merged.mean()
    if key == "cut":
        out.index = np.round(out.index.astype(float), 4)
    return out


def cut_at_fake_rate(c: pd.DataFrame, fr_star: float, column: str = "fake_rate_hnl") -> tuple[float, bool]:
    """(cut, reachable): the cut whose pooled-HNL rate in `column` is the largest value <= the target (ties: lowest
    cut). If no cut reaches the target (the rate curve is U-shaped in the cut), the cut with the smallest rate (ties:
    lowest cut) and reachable = False (DECISIONS.md 2026-10-03, review fix): never silently the highest cut."""
    if not np.isfinite(fr_star):
        raise ValueError(f"non-finite target {fr_star!r} for {column}")
    rate = c[column]
    ok = c[rate <= fr_star + 1e-12]
    if ok.empty:
        r = rate.dropna()
        if r.empty:
            raise ValueError(f"no finite {column} values")
        return float(r[r <= r.min() + 1e-12].index.min()), False
    best = ok[column].max()
    return float(ok[ok[column] >= best - 1e-12].index.min()), True


def working_points(cv: dict[tuple[str, int], pd.DataFrame]) -> dict:
    arms = sorted({a for a, _ in cv})
    seeds = {a: sorted(s for b, s in cv if b == a) for a in arms}
    mean = lambda a, col, cuts: float(np.mean([cv[(a, s)].loc[cuts[s] if isinstance(cuts, dict) else cuts, col]
                                               for s in seeds[a]]))
    s_star = {}
    for a in arms:
        m = np.mean([cv[(a, s)].loc[COARSE, "eff_all_targets"].to_numpy() for s in seeds[a]], axis=0)
        s_star[a] = float(COARSE[int(np.argmax(m))])
    fr_at = {a: mean(a, "fake_rate_hnl", s_star[a]) for a in arms}
    cand = {}
    for A in arms:
        res = {b: {s: cut_at_fake_rate(cv[(b, s)], fr_at[A]) for s in seeds[b]} for b in arms}
        cuts = {b: {s: res[b][s][0] for s in seeds[b]} for b in arms}
        unreach = [[b, s] for b in arms for s in seeds[b] if not res[b][s][1]]
        eff = {b: mean(b, "eff_primary", cuts[b]) for b in arms}
        cand[A] = {"fr_star": fr_at[A], "cuts": cuts, "eff_primary": eff, "unreachable": unreach,
                   "consistent": max(eff, key=eff.get) == A}
    consistent = [A for A in arms if cand[A]["consistent"]]
    if consistent:
        ref = max(consistent, key=lambda A: cand[A]["eff_primary"][A])
        rule = "consistent fixed point"
    else:
        ref = max(arms, key=lambda a: mean(a, "eff_primary", s_star[a]))
        rule = "no consistent arm: highest primary efficiency at own s*"
    fr_star = fr_at[ref]
    e_star = mean(ref, "edge_fpr_hnl", s_star[ref])
    res_e = {b: {s: cut_at_fake_rate(cv[(b, s)], e_star, "edge_fpr_hnl") for s in seeds[b]} for b in arms}
    cuts_e = {b: {s: res_e[b][s][0] for s in seeds[b]} for b in arms}
    return {"s_star": s_star, "fake_rate_at_s_star": fr_at, "reference_arm": ref, "reference_rule": rule,
            "FR_star": fr_star, "e_star": e_star, "candidates": cand,
            "cuts_at_FR_star": cand[ref]["cuts"], "cuts_at_e_star": cuts_e,
            "unreachable_at_FR_star": cand[ref]["unreachable"],
            "unreachable_at_e_star": [[b, s] for b in arms for s in seeds[b] if not res_e[b][s][1]]}


def load_run(run: str):
    d = os.path.join(run, "eval_val")
    return pd.read_parquet(os.path.join(d, "per_particle.parquet")), pd.read_parquet(os.path.join(d, "per_event.parquet"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory holding arm<A>_s<S> run directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="i,ii,iii")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--allow-missing", action="store_true",
                    help="previews only: skip requested runs without eval.done (listed in the output)")
    ap.add_argument("--accept-unreachable", default="",
                    help="comma list ARM:SEED of reference-arm seeds whose unreachable FR* was reviewed by hand "
                         "(DECISIONS.md): they keep the documented fallback (minimum-rate cut, flagged) without HOLD")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    data, missing = {}, []
    for arm in a.arms.split(","):
        for s in map(int, a.seeds.split(",")):
            run = os.path.join(a.runs, f"arm{arm}_s{s}")
            if os.path.exists(os.path.join(run, "eval.done")):
                data[(arm, s)] = load_run(run)
            else:
                missing.append(f"arm{arm}_s{s}")
    if missing and not a.allow_missing:
        raise SystemExit(f"requested runs without eval.done: {missing} (use --allow-missing for a preview)")
    if not data:
        raise SystemExit("no evaluated runs")
    from hnl.train.chain_check import check_run, val_chain
    stale = [f"arm{arm}_s{s}: {e}" for arm, s in data
             for e in check_run(os.path.join(a.runs, f"arm{arm}_s{s}"), ["ml", "gnn"])]
    if stale:
        raise SystemExit("val scores do not come from the current best checkpoints:\n  " + "\n  ".join(stale))
    ev_sets = {k: frozenset(pe.event_id.unique()) for k, (_, pe) in data.items()}
    if len(set(ev_sets.values())) != 1:
        raise SystemExit("runs were evaluated on different val events: " +
                         ", ".join(f"arm{k[0]}_s{k[1]}: {len(v)}" for k, v in ev_sets.items()))
    events = np.unique(np.concatenate([pe.event_id.unique() for _, pe in data.values()]))
    info = particle_table(events)
    cv = {k: curves(pp, pe, info) for k, (pp, pe) in data.items()}
    for (arm, s), c in cv.items():
        c.to_csv(os.path.join(a.out, f"curves_arm{arm}_s{s}.csv"))
    wp = working_points(cv)
    wp["runs"] = sorted(f"arm{k[0]}_s{k[1]}" for k in data)
    wp["missing_runs"] = missing
    wp["n_events"] = int(len(events))
    wp["val_chain"] = {f"arm{arm}_s{s}": val_chain(os.path.join(a.runs, f"arm{arm}_s{s}")) for arm, s in data}
    wp["accepted_unreachable_reference"] = [x.split(":") for x in a.accept_unreachable.split(",") if x]
    for key in ("unreachable_at_FR_star", "unreachable_at_e_star"):
        for arm, seed in wp[key]:
            print(f"WARNING {key}: arm {arm} seed {seed} never reaches the target; its cut is the minimum-rate cut")
    for A, c in wp["candidates"].items():
        for arm, seed in c["unreachable"]:
            print(f"WARNING candidate {A} (FR*_A {c['fr_star']:.6f}): arm {arm} seed {seed} never reaches it; the "
                  f"consistency check uses its minimum-rate cut")
    # summary table: each arm at s* and at FR*
    rows = []
    for arm in sorted({x for x, _ in cv}):
        seeds = sorted(s for b, s in cv if b == arm)
        for label, cuts in (("s_star", {s: wp["s_star"][arm] for s in seeds}),
                            ("FR_star", wp["cuts_at_FR_star"][arm])):
            vals = pd.DataFrame([cv[(arm, s)].loc[cuts[s]] for s in seeds])
            r = {"arm": arm, "point": label, "cuts": [cuts[s] for s in seeds],
                 "fr_star_reachable": [[arm, s] not in wp["unreachable_at_FR_star"] for s in seeds]
                 if label == "FR_star" else None}
            for col in vals.columns:
                r[col] = float(vals[col].mean())
                r[col + "_seed_sd"] = float(vals[col].std(ddof=1)) if len(vals) > 1 else np.nan
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "working_points.json"), "w") as f:
        json.dump(wp, f, indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = ["arm", "point", "cuts", "eff_all_targets", "eff_primary", "eff_hnl_targets", "eff_w_muons",
            "eff_ttbar_prompt", "fake_rate_hnl", "dup_rate_all_targets", "edge_fpr_hnl", "pair_eff"]
    print(summ[[c for c in cols if c in summ]].round(4).to_string())
    print(json.dumps({k: v for k, v in wp.items() if k != "candidates"}, indent=1, default=str))
    accepted = {tuple(map(str, x)) for x in wp["accepted_unreachable_reference"]}
    ref_unreach = [x for x in wp["unreachable_at_FR_star"]
                   if x[0] == wp["reference_arm"] and (str(x[0]), str(x[1])) not in accepted]
    if ref_unreach:
        raise SystemExit(f"HOLD: seeds of the reference arm cannot reach FR*: {ref_unreach} (decide by hand)")


if __name__ == "__main__":
    main()
