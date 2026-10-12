"""Test-set counterpart of hnl.eval.ranking: the one-time test outputs of the FROZEN models in the ranking format.

Post-freeze descriptive summaries of the one-time test evaluation, not claims (the preregistered claims are
results/claims_test_v1.json and stay as they are). Nothing is chosen on test: every cut and configuration below was
fixed on val, and the spec is built from the committed val files only.

  Canto-L / -H / -B  RUN/trackbuild_test at configuration index 0 = the frozen FR* configuration of
                     results/final_selection_v1.json (checked against RUN/trackbuild_test/configs.json and against the
                     selection copy the test job stored, RUN/final_selection_used.json); never the builder rule.
  acorn (HNL), acorn (ttbar)  connected components (RUN/eval_test, hnl.eval.evaluate_gnn) at the preregistered cut
                     (results/working_points_v1.json cuts_at_FR_star) and at the best cut <= FR* chosen on val
                     (results/ranking/ranking_val_v3_entries.csv). A val-best cut that the one-time test chain did not
                     evaluate is read from the post-freeze CC evaluation of the run's stored test scores
                     (runs/published_defs_v1/arm<A>_s<S>/eval_test_bestcut, hnl.eval.published_defs bestcut), accepted
                     only with its provenance: same run, same cut, s* rows identical to the frozen RUN/eval_test.
  DeZoort IN / IN-L  never evaluated on test: no row.

Populations, sets and estimators are those of the preregistered test statistics and of the claims (hnl.eval.test_stats
code path: RunTest, masks, efficiency, event_ratio): primary, HNL-daughter and W-muon efficiencies and the track fake
rate (sum n_fake / sum n_tracks_ge5) on the pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external,
6,999 events), the ttbar-prompt efficiency on ttbar_test (1,000 events). Per seed the efficiency pooled over the sets;
per model the mean and seed standard deviation (ddof 1) over the seeds (the claims' estimator); HNL-daughter targets
with |d0| in [100, 300) mm and >= 300 mm pooled over the seeds (k / n). The markdown notes add each model's
working points, the preregistered total sigma of the primary efficiency (event bootstrap + seed term) and, with
--claims, the identity checks against the claims file.

The output columns are those of hnl.eval.ranking (results/ranking/ranking_val_v3.csv) and the labels those of the val
ranking v3 for the same frozen model and working point ("best cut <= FR*" = the cut chosen on val by that rule), so
that hnl.plots.release_card and val / test joins by label read the table.

    python -m hnl.eval.ranking_test spec --out results/ranking/spec_test_v1.json
    python -m hnl.eval.ranking_test rank --spec results/ranking/spec_test_v1.json --out results/ranking \
        --name ranking_test_v1 --claims results/claims_test_v1.json
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
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table

REPO = os.environ.get("HNL_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
SELECTION = "results/final_selection_v1.json"
WP = "results/working_points_v1.json"
VAL_ENTRIES = "results/ranking/ranking_val_v3_entries.csv"
BASELINES = os.path.join(DATA, "runs", "baseline_v1")
POSTFREEZE = os.path.join(DATA, "runs", "published_defs_v1")
POSTFREEZE_SUB = "eval_test_bestcut"
SEEDS = ("0", "1", "2")
CANTO = (  # pipeline of the selection, short name, label (the val ranking v3 label of the same frozen model)
    ("l1tl", "Canto-L", "Canto-L (full budget): learned triplet union, learned merging (rule R6)"),
    ("l1tu", "Canto-H", "Canto-H (full budget): hand-cut helix union, walkthrough + merging (rule R6)"),
    ("l1t", "Canto-B", "Canto-B (full budget): base graph, learned merging (rule R6)"),
)
ACORN = (  # arm, cut rule, short name, label (val ranking v3)
    ("ii", "prereg", "acorn (HNL) CC", "acorn IGNN, HNL-trained + CC (prereg. cut rule)"),
    ("ii", "best", "acorn (HNL) CC*", "acorn IGNN, HNL-trained + CC (best cut <= FR*)"),
    ("i", "prereg", "acorn (ttbar) CC", "acorn IGNN, ttbar-trained + CC (prereg. cut rule)"),
    ("i", "best", "acorn (ttbar) CC*", "acorn IGNN, ttbar-trained + CC (best cut <= FR*)"),
)
POPS = (  # ranking column, test_stats.masks() name, test sets
    ("eff_primary", "primary", TS.HNL_SETS),
    ("eff_hnl_targets", "hnl_targets", TS.HNL_SETS),
    ("eff_w_muons", "w_muons", TS.HNL_SETS),
    ("eff_ttbar_prompt", "ttbar_prompt", ("ttbar_test",)),
)


def _repo(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _holds(d: str, cut: float) -> bool:
    """Whether the evaluate_gnn table in `d` holds `cut` (per-event and per-particle tables)."""
    for f in ("per_event.parquet", "per_particle.parquet"):
        c = pd.read_parquet(os.path.join(d, f), columns=["cut"]).cut.astype(np.float64).unique()
        if not np.isclose(c, cut, atol=1e-6).any():
            return False
    return True


def _postfreeze(run: str, arm: str, seed: str, cut: float) -> str:
    """The post-freeze CC evaluation of the run's stored test scores at `cut` (absolute directory), checked."""
    d = os.path.join(POSTFREEZE, f"arm{arm}_s{seed}", POSTFREEZE_SUB)
    f = os.path.join(d, "provenance.json")
    if not os.path.exists(f):
        raise SystemExit(f"{run}: cut {cut} is in neither eval_test nor {d}: evaluate CC at it on the stored test "
                         f"scores first (RUN/gnn_scores_test/testset.tar; hnl.eval.published_defs bestcut, sbatch)")
    p = json.load(open(f))
    ok = (os.path.normpath(p["run"]) == os.path.normpath(run) and abs(float(p["cut_best_val"]) - cut) < 1e-6
          and bool(p["identity_check_s_star"]) and all(p["identity_check_s_star"].values()) and _holds(d, cut))
    if not ok:
        raise SystemExit(f"{f}: not run {run} at cut {cut}, or its s* identity check against eval_test failed")
    return d


def _wp_text(head: str, flags: list[str]) -> str:
    return head + (f" ({'; '.join(flags)})" if flags else "")


def build_spec() -> list[dict]:
    """The models of the test ranking from the committed val files (selection, working points, val ranking v3)."""
    sel = json.load(open(_repo(SELECTION)))
    wp = json.load(open(_repo(WP)))
    if abs(float(wp["FR_star"]) - float(sel["FR_star"])) > 1e-15:
        raise SystemExit(f"FR* of {WP} and {SELECTION} differ")
    spec = []
    for pipe, short, label in CANTO:
        p = sel["pipelines"][pipe]
        ents = []
        for seed in SEEDS:
            d = p["seeds"][seed]
            run = d["run"]
            cfgs = json.load(open(os.path.join(run, "trackbuild_test", "configs.json")))
            if ",".join(str(x) for x in cfgs[0]) != d["config"]:
                raise SystemExit(f"{run}/trackbuild_test: configuration 0 {cfgs[0]} is not the frozen {d['config']}")
            if json.load(open(os.path.join(run, "final_selection_used.json"))) != sel:
                raise SystemExit(f"{run}: the test job's selection copy differs from {SELECTION}")
            fam, params = d["config"].split(",", 1)
            ents.append({"run": run, "sub": "trackbuild_test", "cut": 0,
                         "wp": f"{fam} {params} (trackbuild_test, configuration 0)"})
        spec.append({"label": label, "short": short, "group": "Canto (full budget)", "entries": ents})
    val = pd.read_csv(_repo(VAL_ENTRIES))
    unreach = {(str(a), str(s)) for a, s in wp["unreachable_at_FR_star"]}
    for arm, rule, short, label in ACORN:
        ents = []
        for seed in SEEDS:
            run = os.path.join(BASELINES, f"arm{arm}_s{seed}")
            if rule == "prereg":
                cut = float(wp["cuts_at_FR_star"][arm][seed])
                unreachable = (arm, seed) in unreach
            else:
                r = val[(val.label == label) & (val.run == run)]
                if len(r) != 1:
                    raise SystemExit(f"{VAL_ENTRIES}: {len(r)} rows for {label} / {run}")
                m = re.match(r"CC (\d+\.\d+)", r.working_point.iloc[0])
                if m is None:
                    raise SystemExit(f"{VAL_ENTRIES}: no CC cut in {r.working_point.iloc[0]!r}")
                cut = float(m.group(1))
                unreachable = "unreachable" in r.working_point.iloc[0]
            flags = ["FR* unreachable on val: minimum-rate cut"] if unreachable else []
            if _holds(os.path.join(run, "eval_test"), cut):
                sub = "eval_test"
            else:
                sub = _postfreeze(run, arm, seed, cut)
                flags.append("post-freeze CC on the stored test scores")
            ents.append({"run": run, "sub": sub, "cut": cut, "wp": _wp_text(f"CC {cut:.2f}", flags)})
        spec.append({"label": label, "short": short, "group": "acorn", "entries": ents})
    return spec


def entry_metrics(bs: BS.Bootstrap, rt: TS.RunTest, e: dict, info: pd.DataFrame) -> tuple[dict, dict]:
    """One seed: the ranking columns (pooled over the test sets) and, per efficiency, (point, bootstrap replicates,
    population size)."""
    M = TS.masks()
    out = {"run": e["run"], "working_point": e["wp"], "n_events": int(rt.pe.event_id.nunique())}
    extra = {}
    for key, pop, sets in POPS:
        p, r, n = TS.efficiency(bs, [rt], info, M[pop], sets)
        out[key] = float(p[0])
        extra[key] = (float(p[0]), r[0], n)
    p, _ = TS.event_ratio(bs, [rt], "n_fake", "n_tracks_ge5", TS.HNL_SETS)
    out["fake_rate_hnl"] = float(p[0])
    t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
    t = t[np.asarray(M["hnl_targets"](t), dtype=bool) & TS.in_sets(t.event_id, TS.HNL_SETS)]
    for lo, hi in RK.D0_BINS:
        b = t[(t.d0.abs() >= lo) & (t.d0.abs() < hi)]
        out[f"n_d0_{lo:g}"], out[f"k_d0_{lo:g}"] = int(len(b)), int(b.dm_matched.sum())
    return out, extra


def rank(spec: list) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(table, per-seed entries, per model: total sigma of the primary efficiency and the seed-pooled k / n)."""
    bs = BS.Bootstrap()
    rts = [[TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"]) for e in m["entries"]] for m in spec]
    evs = {frozenset(rt.pe.event_id.unique()) for runs in rts for rt in runs}
    if len(evs) != 1:
        raise ValueError("the entries were evaluated on different test events")
    info = particle_table(np.array(sorted(next(iter(evs))), dtype=np.int64))
    rows, ent, more = [], [], {}
    for m, runs in zip(spec, rts):
        es, ex = [], []
        for e, rt in zip(m["entries"], runs):
            r, x = entry_metrics(bs, rt, e, info)
            es.append(r)
            ex.append(x)
            ent.append({"label": m["label"], **r})
        df = pd.DataFrame(es)
        row = {"label": m["label"], "short": m.get("short", ""), "group": m.get("group", ""), "n_seeds": len(df),
               "working_points": "; ".join(df.working_point)}
        for k in RK.KEYS:
            row[k] = df[k].mean()
            row[k + "_sd"] = df[k].std(ddof=1) if len(df) > 1 else np.nan
        for lo, _ in RK.D0_BINS:
            n, k = df[f"n_d0_{lo:g}"].sum(), df[f"k_d0_{lo:g}"].sum()
            row[f"eff_d0_{lo:g}"] = k / n if n else np.nan
            row[f"k_n_d0_{lo:g}"] = f"{k}/{n}"
        rows.append(row)
        prim = [x["eff_primary"] for x in ex]
        more[m["label"]] = {"primary": BS.seed_mean([x[0] for x in prim], [x[1] for x in prim]),
                            "primary_k": [int(round(x[0] * x[2])) for x in prim], "primary_n": [x[2] for x in prim]}
    tab = pd.DataFrame(rows).sort_values("eff_primary", ascending=False, kind="stable").reset_index(drop=True)
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    return tab, pd.DataFrame(ent), more


def claims_checks(tab: pd.DataFrame, spec: list, claims: dict) -> list[tuple[str, float, float, bool]]:
    """Every claims number that is a function of this table's rows, recomputed from the table (identity checks)."""
    def key(e):
        return os.path.normpath(e["run"]), round(float(e["cut"]), 6), e.get("sub", "eval_test")
    by_entries = {}                     # first model with these entries (acorn (ttbar): its val-best cuts = prereg.)
    for m in spec:
        by_entries.setdefault(tuple(sorted(key(e) for e in m["entries"])), m["label"])
    cs, res = claims["spec"], claims["result"]
    row = {}
    for name, seeds in cs["models"].items():
        lab = by_entries.get(tuple(sorted(key(e) for e in seeds)))
        if lab is not None:
            row[name] = tab.set_index("label").loc[lab]
    C = cs["claims"]
    out = []

    def add(what, ours, theirs):
        out.append((what, float(ours), float(theirs), bool(abs(float(ours) - float(theirs)) <= 1e-12)))
    if "primary" in C and {C["primary"]["new"], C["primary"]["base"]} <= set(row):
        new, base = row[C["primary"]["new"]], row[C["primary"]["base"]]
        pr = res["primary"]
        add("primary: eff_new (seed mean)", new.eff_primary, pr["eff_new"])
        add("primary: eff_base (seed mean)", base.eff_primary, pr["eff_base"])
        add("primary: difference", new.eff_primary - base.eff_primary, pr["difference"]["value"])
        g = pr["criterion3_prompt_guard"]
        add("primary guard: d eff W muons", new.eff_w_muons - base.eff_w_muons, g["d_eff_w_muons"])
        add("primary guard: d eff ttbar prompt", new.eff_ttbar_prompt - base.eff_ttbar_prompt, g["d_eff_ttbar_prompt"])
    if "S1" in C and {C["S1"]["new"], C["S1"]["arm_i"], C["S1"]["ref"]} <= set(row):
        new, ai, ref = (row[C["S1"][k]] for k in ("new", "arm_i", "ref"))
        s1 = res["S1"]
        add("S1: new - acorn (ttbar)", new.eff_primary - ai.eff_primary, s1["vs_arm_i"]["value"])
        add("S1: new - reference", new.eff_primary - ref.eff_primary, s1["vs_reference"]["value"])
        g = s1["criterion3_prompt_guard"]
        add("S1 guard: d eff W muons", new.eff_w_muons - ai.eff_w_muons, g["d_eff_w_muons"])
        add("S1 guard: d eff ttbar prompt", new.eff_ttbar_prompt - ai.eff_ttbar_prompt, g["d_eff_ttbar_prompt"])
    if "S1b" in C and C["S1b"]["arm_i"] in row:
        add("S1b: fake rate of acorn (ttbar)", row[C["S1b"]["arm_i"]].fake_rate_hnl, res["S1b"]["fake_rate_arm_i"])
    return out


def markdown(tab: pd.DataFrame, spec: list, more: dict, fr: float, checks: list | None = None) -> str:
    head = (f"One-time test, frozen models (post-freeze summaries of the one-time evaluation, not claims; nothing "
            f"chosen on test). Primary / HNL-daughter / W-muon efficiencies and fake rate on the pooled HNL test sets "
            f"(hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events), ttbar prompt on ttbar_test "
            f"(1,000 events); every cut and configuration fixed on val (FR* = {fr:.6f}, pooled HNL val); mean ± "
            f"seed sd over the listed seeds.")
    lines = [head] + RK.markdown(tab, fr).rstrip("\n").split("\n")[1:]
    lines += ["", "Working points (seeds 0 / 1 / 2, all fixed on val; \"best cut <= FR*\" = the cut chosen on val "
                  "by that rule, applied unchanged to test):", ""]
    for m in spec:
        lines.append(f"- {m['short'] or m['label']}: " + "; ".join(e["wp"] for e in m["entries"]))
    lines += ["", "Primary efficiency with the preregistered total sigma (event bootstrap of the test sets + seed "
                  "term, hnl.eval.bootstrap) and the seed-pooled count k / n (every seed has the same particles, so "
                  "k / n equals the seed mean):", ""]
    for _, r in tab.iterrows():
        x = more[r.label]
        p = x["primary"]
        lines.append(f"- {r.short or r.label}: {p['mean']:.4f} ± {p['sigma']:.4f} (bootstrap {p['sigma_boot']:.4f}, "
                     f"seed sd {p['s_seeds']:.4f}); k / n = {sum(x['primary_k'])}/{sum(x['primary_n'])} "
                     f"= {sum(x['primary_k']) / sum(x['primary_n']):.4f}")
    lines += ["", "DeZoort IN / IN-L were never evaluated on test and have no row. The preregistered claims "
                  "(results/claims_test_v1.json) stay as they are."]
    if checks:
        lines += ["", "Identity checks against results/claims_test_v1.json (|difference| <= 1e-12):", ""]
        for what, ours, theirs, ok in checks:
            lines.append(f"- {'ok' if ok else 'FAIL'} {what}: {ours:.16f} (table) vs {theirs:.16f} (claims)")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="build the spec from the committed val files")
    s.add_argument("--out", required=True)
    r = sub.add_parser("rank", help="the test ranking of a spec")
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--name", default="ranking_test")
    r.add_argument("--claims", default=None, help="claims file for the identity checks (results/claims_test_v1.json)")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        spec = build_spec()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(spec, f, indent=1)
        print(json.dumps(spec, indent=1))
        return
    spec = json.load(open(a.spec))
    fr = float(json.load(open(_repo(WP)))["FR_star"])
    tab, ent, more = rank(spec)
    checks = claims_checks(tab, spec, json.load(open(a.claims))) if a.claims else None
    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(os.path.join(a.out, f"{a.name}.csv"), index=False)
    ent.to_csv(os.path.join(a.out, f"{a.name}_entries.csv"), index=False)
    md = markdown(tab, spec, more, fr, checks)
    with open(os.path.join(a.out, f"{a.name}.md"), "w") as f:
        f.write(md)
    print(md)
    if checks and not all(ok for *_, ok in checks):
        raise SystemExit("identity checks against the claims failed (see the table notes)")


if __name__ == "__main__":
    main()
