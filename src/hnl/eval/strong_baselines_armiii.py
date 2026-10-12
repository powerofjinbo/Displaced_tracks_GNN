"""Feedback round 2, task E: acorn (ttbar+HNL) with its strongest track builders (interim answer to supervisor point 2c).

Point 2c asks for acorn trained on ttbar plus a generic displaced sample. That sample does not exist yet. The closest
existing model is acorn (ttbar+HNL) = Stage-0 baseline arm (iii) (runs/baseline_v1/armiii_s{0,1,2}): acorn's
interaction GNN trained on 7,999 ttbar and 7,999 HNL training events together. Its HNL training events are disjoint
from the val and test events but come from the evaluated signal samples, so it is an optimistic stand-in for
"ttbar + a generic displaced sample".

Round 1's strongest-baseline analysis (hnl.eval.strong_baselines) for this arm, three seeds, with round 1's rules:
  acwalk / acwalkg  acorn's own CC + walkthrough (CCandWalk, hnl.trackbuild.acorn_walk; plain and score-guided), the
                    27-configuration grid per variant of round 1: RUN/trackbuild_val_acwalk (19 per variant) and its
                    lower-edge extension RUN/trackbuild_val_acwalk_ext (8 per variant), united in
                    RUN/trackbuild_val_acwalk_all by `merge`;
  walkmerge         our walkthrough + merging (RUN/trackbuild_val_ext, the first grid holding the family);
  mwsm              same-layer MWS + merging (the grids of ranking v3);
  CC                connected components at the best cut <= FR* and at the preregistered cut rule (this arm's rows
                    of ranking v3, RUN/eval_val).
Every track-building configuration per seed on val: the builder rule at FR* with the one-standard-error tie rule
(hnl.eval.ranking, tie_se=True); the CC cuts per seed on val by the CC rules of ranking v3 (no tie rule). Round 1's
lower-edge rule (a CCandWalk choice at the lowest walk threshold of the grid -> extend the
grid downwards) is checked for every chosen configuration (`edges`).

Test (post-freeze descriptive, NOT claims; results/claims_test_v1.json stays as it is): the val-chosen configurations
applied once to the stored test scores (RUN/gnn_scores_test/testset.tar) by slurm/feedback/armiii_test.sbatch ->
runs/feedback_v2/strong_baselines_armiii/armiii_s<S>/<key>_test (configuration index 0, provenance checked: the spec
file the job ran from is unchanged (sha256) and holds the same entry; rows complete before the last CCandWalk grid
finished ran from the first-stage spec of `val --skip-missing`, whose entries equal the final ones); CC from
the frozen RUN/eval_test where the one-time test chain evaluated the cut, otherwise once post-freeze on the stored
test scores (`bestcut` -> runs/feedback_v2/strong_baselines_armiii/armiii_s<S>/eval_test_bestcut, with the s* identity
check against RUN/eval_test). Summaries: the code path of hnl.eval.ranking_test.

Reference rows, reproduced from round 1 to 1e-12: the frozen Canto-L, acorn (HNL) and acorn (ttbar) at their val-best
configuration (the highest val primary efficiency among each arm's rows of results/strong_baselines_v1_val.csv), and,
for the same-builder differences, every round-1 row of both arms. Differences with the claims' estimator (test:
preregistered tables, B = 500; val: the tables of feedback point 5, B = 1000; paired event bootstrap of the seed means
plus both seed terms): Canto-L minus each acorn (ttbar+HNL) row; each acorn (ttbar+HNL) row minus acorn (HNL) and
minus acorn (ttbar) at their val-best configuration; acorn (ttbar+HNL) minus acorn (HNL) / acorn (ttbar) with the same
builder; and, as identity checks, Canto-L minus the two val-best rows (= round 1).

    python -m hnl.eval.strong_baselines_armiii merge                 (acwalk grid union of the arm (iii) runs)
    python -m hnl.eval.strong_baselines_armiii val [--skip-missing]  (val table, differences, test spec;
                                                                      slurm/feedback/armiii_val.sbatch: merge + val)
    python -m hnl.eval.strong_baselines_armiii jobs                  (sbatch commands for missing test outputs)
    python -m hnl.eval.strong_baselines_armiii bestcut --seeds 0,2   (in a Slurm job: CC at the val-best cut on test)
    python -m hnl.eval.strong_baselines_armiii test [--notes FILE]   (test table, differences, markdown)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import ranking as RK
from hnl.eval import ranking_test as RT
from hnl.eval import strong_baselines as SB

DATA = RT.DATA
BASE = SB.BASE
FB2 = os.path.join(DATA, "runs", "feedback_v2", "strong_baselines_armiii")
NAME = "strong_baselines_v2_armiii"
V1 = "strong_baselines_v1"
SEEDS = SB.SEEDS
ARM, ALAB, PREFIX = "iii", "acorn (ttbar+HNL)", "armiii"
NEW = SB.NEW
BESTCUT_SUB = "eval_test_bestcut"
V3_SPEC = "results/ranking/spec_val_v3.json"
V3_RANK = "results/ranking/ranking_val_v3.csv"
V3_ENTRIES = "results/ranking/ranking_val_v3_entries.csv"
ANALYSIS_TEST = os.path.join(BASE, "analysis_test", "test_summary.json")   # the M3 test summary (CC, prereg. cuts)
TEST_SBATCH = "slurm/feedback/armiii_test.sbatch"
EXCLUDE = ("submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,"
           "submit71,submit72,submit73")
TOL = 1e-12
CC_ROWS = (  # ranking v3 label of this arm's CC row, our label, short, rule
    ("acorn IGNN, ttbar+HNL-trained + CC (best cut <= FR*)", f"{ALAB} + CC, best cut <= FR*", "(ttbar+HNL) CC*", "best"),
    ("acorn IGNN, ttbar+HNL-trained + CC (prereg. cut rule)", f"{ALAB} + CC, preregistered cut rule", "(ttbar+HNL) CC",
     "prereg"),
)
V3_MWSM = "acorn IGNN, ttbar+HNL-trained + Canto builder (MWS-layer + merging)"
BLAB = {k: blab for k, _, _, blab, _ in SB.BUILDERS}
PARAM_NAMES = {"acwalk": ("score_cut_cc", "walk min", "walk add"), "acwalkg": ("score_cut_cc", "walk min", "walk add"),
               "walkmerge": ("walk min", "walk add", "merging rms"), "mwslayermerge": ("s0",)}
# builder -> (acorn (ttbar+HNL) row, acorn (HNL) row, acorn (ttbar) row): the same builder on the three arms
SAME = {k: (f"{ALAB} + {BLAB[k]}", f"acorn (HNL) + {BLAB[k]}", f"acorn (ttbar) + {BLAB[k]}") for k in BLAB}
SAME["ccbest"] = (f"{ALAB} + CC, best cut <= FR*", "acorn (HNL) + CC, best cut <= FR*",
                  "acorn (ttbar) + CC, preregistered cut (= best cut)")
SAME["ccprereg"] = (f"{ALAB} + CC, preregistered cut rule", "acorn (HNL) + CC, preregistered cut (claims' reference)",
                    "acorn (ttbar) + CC, preregistered cut (= best cut)")
BLOCKS = ("Canto-L minus acorn (ttbar+HNL)", "acorn (ttbar+HNL) minus acorn (HNL) val-best",
          "acorn (ttbar+HNL) minus acorn (ttbar) val-best", "same builder: acorn (ttbar+HNL) minus acorn (HNL)",
          "same builder: acorn (ttbar+HNL) minus acorn (ttbar)", "round-1 identity")


def _repo(p: str) -> str:
    return RT._repo(p)


# ------------------------------------------------------------------------------------------------- pure helpers
def val_best(tab: pd.DataFrame, arm_label: str) -> str:
    """The val-best row of an arm: the highest seed-mean primary efficiency among the rows whose label starts with
    '<arm_label> + ' (every row at fake rate <= FR* by its own rule); ties: the lower fake rate."""
    t = tab[tab.label.str.startswith(arm_label + " + ")]
    if t.empty:
        raise ValueError(f"no rows of {arm_label}")
    return str(t.sort_values(["eff_primary", "fake_rate_hnl"], ascending=[False, True], kind="stable").label.iloc[0])


def pairs(iii: list[str], hnl_best: str, ttbar_best: str) -> list[tuple[str, str, str]]:
    """(new, baseline, blocks) of the differences; a pair that belongs to several blocks is listed once, its blocks
    joined by '; ' (e.g. acorn (ttbar+HNL) + walkthrough + merging minus acorn (HNL) + walkthrough + merging is both
    'minus acorn (HNL) val-best' and 'same builder')."""
    want = [(NEW, x, BLOCKS[0]) for x in iii]
    want += [(x, hnl_best, BLOCKS[1]) for x in iii]
    want += [(x, ttbar_best, BLOCKS[2]) for x in iii]
    for x, h, t in SAME.values():
        if x in iii:
            want += [(x, h, BLOCKS[3]), (x, t, BLOCKS[4])]
    want += [(NEW, hnl_best, BLOCKS[5]), (NEW, ttbar_best, BLOCKS[5])]
    out: dict = {}
    for a, b, blk in want:
        out.setdefault((a, b), [])
        if blk not in out[(a, b)]:
            out[(a, b)].append(blk)
    return [(a, b, "; ".join(blks)) for (a, b), blks in out.items()]


def edge_flags(cfgs: list, fam: str, values: list[float]) -> list[tuple[str, float, str]]:
    """Where a chosen configuration sits in its family's grid: (parameter, value, 'lowest' | 'highest') for every
    parameter whose chosen value is the lowest / highest value of that parameter among the family's configurations
    (parameters with one value in the grid are fixed, not edges)."""
    names = PARAM_NAMES.get(fam, tuple(f"p{j + 1}" for j in range(len(values))))
    rows = [c[1:] for c in cfgs if c[0] == fam]
    out = []
    for j, v in enumerate(values):
        col = sorted({float(r[j]) for r in rows})
        if len(col) < 2:
            continue
        if np.isclose(v, col[0], rtol=0, atol=1e-12):
            out.append((names[j], float(v), "lowest"))
        elif np.isclose(v, col[-1], rtol=0, atol=1e-12):
            out.append((names[j], float(v), "highest"))
    return out


def lower_edge_rule(fam: str, flags: list[tuple[str, float, str]]) -> str:
    """Round 1's rule for acorn's CCandWalk: a chosen walk min at the grid's lowest value calls for a grid extended
    downwards. The 27-configuration grid already holds round 1's extension, whose lowest value is 0 (every edge kept,
    the walk never stops on score), below which nothing can be added."""
    if fam not in ("acwalk", "acwalkg"):
        return ""
    for name, v, where in flags:
        if name == "walk min" and where == "lowest":
            return ("walk min at the lowest value of the 27-configuration grid" +
                    (" (0: every edge kept; no lower value exists)" if v == 0.0 else ": extend the grid downwards"))
    return "interior (no extension needed)"


# ------------------------------------------------------------------------------------------------- val
def build_val_spec(skip_missing: bool = False) -> tuple[list[dict], list[dict]]:
    """(round 1's val spec, rebuilt and checked against results/strong_baselines_v1_spec_val.json; the acorn
    (ttbar+HNL) rows)."""
    v1 = SB.val_spec()
    ref = json.load(open(_repo(f"results/{V1}_spec_val.json")))
    if json.loads(json.dumps(v1)) != ref:
        raise SystemExit(f"round 1's val spec is not reproduced (results/{V1}_spec_val.json)")
    v3 = {m["label"]: m for m in json.load(open(_repo(V3_SPEC)))}
    iii = []
    for lab3, lab, short, rule in CC_ROWS:
        m = v3[lab3]
        iii.append({"label": lab, "short": short, "group": ALAB, "cc_rule": rule, "prefix": PREFIX, "v3_label": lab3,
                    "entries": [dict(e) for e in m["entries"]]})
    runs = [os.path.join(BASE, f"arm{ARM}_s{s}") for s in SEEDS]
    for key, fam, src, blab, bshort in SB.BUILDERS:
        have = all(any(os.path.exists(os.path.join(r, g, "configs.json")) for g in src) for r in runs)
        if not have:
            if skip_missing:
                print(f"WARNING: {ALAB} + {key}: grid {src} missing for some seed, row skipped")
                continue
            raise SystemExit(f"{ALAB} + {key}: grid {src} missing")
        iii.append({"label": f"{ALAB} + {blab}", "short": f"(ttbar+HNL) {bshort}", "group": ALAB, "key": key,
                    "prefix": PREFIX, "entries": [{"run": r, "source": src, "family": fam} for r in runs]})
    return v1, iii


def _cmp_row(mine: pd.Series, ref: pd.Series, what: str, checks: list) -> None:
    for k in RK.KEYS:
        checks.append((f"{what}: {k}", float(mine[k]), float(ref[k]), abs(float(mine[k]) - float(ref[k])) <= TOL))
        if int(mine["n_seeds"]) > 1:
            a, b = float(mine[k + "_sd"]), float(ref[k + "_sd"])
            checks.append((f"{what}: {k}_sd", a, b, abs(a - b) <= TOL))
    for lo in (100, 300):
        c = f"k_n_d0_{lo}"
        checks.append((f"{what}: {c}", str(mine[c]), str(ref[c]), str(mine[c]) == str(ref[c])))


def _cmp_entries(ent: pd.DataFrame, lab: str, ref: pd.DataFrame, ref_lab: str, what: str, checks: list,
                 wp_of=lambda w: w) -> None:
    a = ent[ent.label == lab].reset_index(drop=True)
    b = ref[ref.label == ref_lab].reset_index(drop=True)
    checks.append((f"{what}: entries", len(a), len(b), len(a) == len(b)))
    for (_, x), (_, y) in zip(a.iterrows(), b.iterrows()):
        ok = os.path.normpath(x.run) == os.path.normpath(y.run) and wp_of(x.working_point) == y.working_point
        checks.append((f"{what}: {os.path.basename(x.run)} working point", x.working_point, y.working_point, ok))
        for k in RK.KEYS:
            checks.append((f"{what}: {os.path.basename(x.run)} {k}", float(x[k]), float(y[k]),
                           abs(float(x[k]) - float(y[k])) <= TOL))


def val_checks(tab: pd.DataFrame, ent: pd.DataFrame, v1: list[dict], iii: list[dict]) -> list:
    """Identity checks of the val ranking: round 1's rows and entries; this arm's CC and MWS-layer + merging rows and
    entries against ranking v3; the preregistered cuts against results/working_points_v1.json."""
    checks: list = []
    t1 = pd.read_csv(_repo(f"results/{V1}_val.csv")).set_index("label")
    e1 = pd.read_csv(_repo(f"results/{V1}_val_entries.csv"))
    t = tab.set_index("label")
    for m in v1:
        _cmp_row(t.loc[m["label"]], t1.loc[m["label"]], f"round 1 val row {m['label']}", checks)
        _cmp_entries(ent, m["label"], e1, m["label"], f"round 1 val entries {m['label']}", checks)
    t3 = pd.read_csv(_repo(V3_RANK)).set_index("label")
    e3 = pd.read_csv(_repo(V3_ENTRIES))
    labs = {m["label"] for m in iii}
    for lab3, lab, *_ in CC_ROWS:
        _cmp_row(t.loc[lab], t3.loc[lab3], f"ranking v3 row {lab3}", checks)
        _cmp_entries(ent, lab, e3, lab3, f"ranking v3 entries {lab3}", checks)
    mw = f"{ALAB} + {BLAB['mwsm']}"
    if mw in labs:
        _cmp_row(t.loc[mw], t3.loc[V3_MWSM], f"ranking v3 row {V3_MWSM}", checks)
        _cmp_entries(ent, mw, e3, V3_MWSM, f"ranking v3 entries {V3_MWSM}", checks)
    wp = json.load(open(_repo(RT.WP)))
    pre = ent[ent.label == f"{ALAB} + CC, preregistered cut rule"]
    for _, r in pre.iterrows():
        seed = re.search(r"_s(\d+)$", r.run).group(1)
        cut = float(re.match(r"CC (\d+\.\d+)", r.working_point).group(1))
        ref = float(wp["cuts_at_FR_star"][ARM][seed])
        checks.append((f"preregistered cut armiii_s{seed} = working_points cuts_at_FR_star", cut, ref,
                       abs(cut - ref) < 1e-9))
    return checks


def _seed(run: str) -> str:
    return re.search(r"_s(\d+)$", run).group(1)


def test_entries(m: dict, ent: pd.DataFrame) -> list[dict]:
    """The test entries of one acorn (ttbar+HNL) row, each fixed by its val working point."""
    es = []
    for e in m["entries"]:
        r = ent[(ent.label == m["label"]) & (ent.run == e["run"])]
        if len(r) != 1:
            raise SystemExit(f"{len(r)} val entries for {m['label']} / {e['run']}")
        wp, seed = r.working_point.iloc[0], _seed(e["run"])
        if "cc_rule" in m:
            cut = float(re.match(r"CC (\d+\.\d+)", wp).group(1))
            flags = ["FR* unreachable on val: minimum-rate cut"] if "unreachable" in wp else []
            if RT._holds(os.path.join(e["run"], "eval_test"), cut):
                sub = "eval_test"
            elif m["cc_rule"] != "best":
                raise SystemExit(f"{e['run']}: preregistered cut {cut} not in the frozen eval_test")
            else:
                sub = os.path.join(FB2, f"{PREFIX}_s{seed}", BESTCUT_SUB)
                flags.append("post-freeze CC on the stored test scores")
            es.append({"run": e["run"], "sub": sub, "cut": cut, "val_wp": wp, "wp": RT._wp_text(f"CC {cut:.2f}", flags)})
        else:
            fam, params, grid = SB._parse_wp(wp)
            es.append({"run": e["run"], "sub": os.path.join(FB2, f"{PREFIX}_s{seed}", f"{m['key']}_test"), "cut": 0,
                       "config": f"{fam},{params}", "val_grid": grid, "val_wp": wp,
                       "wp": f"{fam} {params} (chosen on val in {grid}; test configuration 0)"})
    return es


def edges(iii: list[dict], ent: pd.DataFrame) -> list[dict]:
    """Grid position of every chosen track-building configuration of this arm (round 1's lower-edge rule)."""
    out = []
    for m in iii:
        if "key" not in m:
            continue
        for e in m["entries"]:
            wp = ent[(ent.label == m["label"]) & (ent.run == e["run"])].working_point.iloc[0]
            fam, params, grid = SB._parse_wp(wp)
            cfgs = json.load(open(os.path.join(e["run"], grid, "configs.json")))
            fl = edge_flags(cfgs, fam, [float(x) for x in params.split(",")])
            out.append({"label": m["label"], "seed": _seed(e["run"]), "family": fam, "params": params, "grid": grid,
                        "n_family_configs": sum(c[0] == fam for c in cfgs),
                        "edges": "; ".join(f"{n} {v:g} {w}" for n, v, w in fl) or "interior",
                        "lower_edge_rule": lower_edge_rule(fam, fl)})
    return out


def val_differences(spec: list[dict], ent: pd.DataFrame, prs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Val differences (new minus baseline) with the event bootstrap of feedback point 5 (hnl.eval.bootstrap_tables:
    its val replicate tables, B = 1000, seed 20261005; events paired across models; both seed terms), at the working
    points of the val ranking; as hnl.eval.strong_baselines.val_differences for other pairs."""
    from hnl.eval import bootstrap_tables as BT
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    want = {x for a, b, _ in prs for x in (a, b)}
    models = {}
    for m in spec:
        if m["label"] not in want:
            continue
        runs = [e["run"] for e in m["entries"]]
        e = ent[ent.label == m["label"]].set_index("run")
        wps = [e.loc[r, "working_point"] for r in runs]
        models[m["label"]] = BT.Model(m["label"], m.get("short", ""), [ld.seed(r, w) for r, w in zip(runs, wps)], rs,
                                      wps, runs, m.get("group", ""))
    rows = []
    for a, lab, blocks in prs:
        new, mm = models[a], models[lab]
        for q, bq in SB.VAL_Q:
            d = BT.difference(new, mm, bq)
            sn, sb = BT.summary(new, bq), BT.summary(mm, bq)
            rows.append({"set": "val", "blocks": blocks, "new": a, "baseline": lab, "quantity": q,
                         "n_per_seed": sb.get("n_per_seed"), "value_new": sn["value"], "sigma_new": sn["combined"],
                         "s_seeds_new": sn["seed_sd"], "value_base": sb["value"], "sigma_base": sb["combined"],
                         "s_seeds_base": sb["seed_sd"], "difference": d["value"], "sigma_boot": d["boot_sd"],
                         "sigma_seeds": d["seed_sd"], "sigma": d["combined"], "z": d["z"], "B": d["B"]})
    return pd.DataFrame(rows)


def _diff_identity(d: pd.DataFrame, ref: pd.DataFrame, what: str, checks: list) -> None:
    """Our differences of the round-1 pairs against round 1's differences file (every column, 1e-12)."""
    for (a, b), g in d.groupby(["new", "baseline"], sort=False):
        r = ref[(ref.new == a) & (ref.baseline == b)]
        if r.empty:
            continue
        r = r.set_index("quantity")
        for _, x in g.iterrows():
            y = r.loc[x.quantity]
            for c in ("value_new", "value_base", "difference", "sigma_boot", "sigma_seeds", "sigma"):
                ok = abs(float(x[c]) - float(y[c])) <= TOL
                checks.append((f"{what}: {a} - {b} {x.quantity} {c}", float(x[c]), float(y[c]), ok))


def _rows_for_tables(tab: pd.DataFrame, labels: list[str], roles: dict) -> pd.DataFrame:
    """The rows shown in the tables (ranked among themselves by the val / test primary efficiency)."""
    t = tab[tab.label.isin(labels)].copy()
    t["role"] = t.label.map(roles)
    t = t.sort_values("eff_primary", ascending=False, kind="stable").reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def _roles(hnl_best: str, ttbar_best: str, iii: list[str], iii_best: str | None) -> dict:
    r = {NEW: "Canto-L (frozen), round 1",
         hnl_best: "reference: acorn (HNL) at its val-best configuration (round 1)",
         ttbar_best: "reference: acorn (ttbar) at its val-best configuration (round 1)"}
    for x in iii:
        r[x] = "acorn (ttbar+HNL)" + (", val-best" if x == iii_best else "")
    return r


def run_val(out: str, name: str, skip_missing: bool = False) -> None:
    fr = SB.fr_star()
    v1, iii = build_val_spec(skip_missing)
    spec = v1 + iii
    tab, ent = RK.rank(spec, fr, tie_se=True)
    checks = val_checks(tab, ent, v1, iii)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(checks, columns=["check", "ours", "reference", "ok"]).to_csv(
        os.path.join(out, f"{name}_val_checks.csv"), index=False)
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:20]:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} of {len(checks)} val identity checks failed")
    print(f"val identity checks: {len(checks)} of {len(checks)} ok")
    v1tab = pd.read_csv(_repo(f"results/{V1}_val.csv"))
    hnl_best, ttbar_best = val_best(tab, "acorn (HNL)"), val_best(tab, "acorn (ttbar)")
    if (hnl_best, ttbar_best) != (val_best(v1tab, "acorn (HNL)"), val_best(v1tab, "acorn (ttbar)")):
        raise SystemExit("the val-best rows differ from round 1's")
    labs = [m["label"] for m in iii]
    complete = len(iii) == len(CC_ROWS) + len(SB.BUILDERS)
    iii_best = val_best(tab[tab.label.isin(labs)], ALAB) if complete else None
    roles = _roles(hnl_best, ttbar_best, labs, iii_best)
    show = [NEW, hnl_best, ttbar_best] + labs
    vt = _rows_for_tables(tab, show, roles)
    vt.insert(0, "set", "val")
    vent = ent[ent.label.isin(show)].copy()
    vent.insert(0, "set", "val")
    tspec = []
    for m in iii:
        tspec.append({k: m[k] for k in ("label", "short", "group", "prefix") if k in m}
                     | {k: m[k] for k in ("key", "cc_rule", "v3_label") if k in m}
                     | {"entries": test_entries(m, ent)})
    ed = pd.DataFrame(edges(iii, ent))
    prs = pairs(labs, hnl_best, ttbar_best)
    vd = val_differences(spec, ent, prs)
    _diff_identity(vd, pd.read_csv(_repo(f"results/{V1}_val_differences.csv")), "round 1 val differences", checks)
    bad = [c for c in checks if not c[3]]
    pd.DataFrame(checks, columns=["check", "ours", "reference", "ok"]).to_csv(
        os.path.join(out, f"{name}_val_checks.csv"), index=False)
    if bad:
        for c in bad[:20]:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} of {len(checks)} val identity checks failed (differences)")
    print(f"val identity checks incl. round-1 differences: {len(checks)} of {len(checks)} ok")
    json.dump(spec, open(os.path.join(out, f"{name}_spec_val.json"), "w"), indent=1)
    json.dump({"fr_star": fr, "rule": "builder rule at FR* with the one-standard-error tie rule (hnl.eval.ranking, "
                                       "tie_se=True), per seed, on val; CC: the ranking v3 rules (eval_val)",
               "hnl_best": hnl_best, "ttbar_best": ttbar_best, "iii_best": iii_best, "complete": complete,
               "models": tspec}, open(os.path.join(out, f"{name}_spec_test.json"), "w"), indent=1)
    vt.to_csv(os.path.join(out, f"{name}_val.csv"), index=False)
    vent.to_csv(os.path.join(out, f"{name}_val_entries.csv"), index=False)
    vd.to_csv(os.path.join(out, f"{name}_val_differences.csv"), index=False)
    ed.to_csv(os.path.join(out, f"{name}_edges.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(RK.markdown(vt.drop(columns=["set", "role"]), fr))
    print(ed.to_string())
    for m in tspec:
        print(m["label"], "|", "; ".join(e.get("config", e["wp"]) for e in m["entries"]))
    print("val-best: acorn (HNL)", hnl_best, "| acorn (ttbar)", ttbar_best, "| acorn (ttbar+HNL)", iii_best)
    p = vd[vd.quantity == "primary"]
    for _, r in p.iterrows():
        print(f"{r.new} - {r.baseline}: {100 * r.difference:+.2f} +- {100 * r.sigma:.2f} (z {r.z:+.1f}) [{r.blocks}]")


# ------------------------------------------------------------------------------------------------- test jobs
def jobs(out: str, name: str, cpus: int = 8, mem: str = "12G", tlim: str = "01:00:00") -> list[str]:
    """sbatch commands (one per run and configuration, so that they run in parallel) for the test outputs that do
    not exist yet, and the post-freeze CC job for the best cuts the one-time test chain did not evaluate."""
    spec_file = os.path.abspath(os.path.join(out, f"{name}_spec_test.json"))
    ts = json.load(open(spec_file))["models"]
    cmds, bestcut = [], []
    for m in ts:
        for e in m["entries"]:
            if "config" in e:
                if os.path.exists(os.path.join(e["sub"], "provenance.json")):
                    continue
                nm = os.path.basename(e["sub"])
                cmds.append(f"env -u OMP_NUM_THREADS SPEC={spec_file} sbatch -J fb2E_test_{nm}_"
                            f"{os.path.basename(e['run'])} -p submit,submit-gpu --exclude={EXCLUDE} -c {cpus} "
                            f"--mem={mem} -t {tlim} {TEST_SBATCH} {e['run']} {os.path.dirname(e['sub'])} "
                            f"{nm}={e['config']}")
            elif e["sub"] != "eval_test" and not os.path.exists(os.path.join(e["sub"], "provenance.json")):
                bestcut.append(_seed(e["run"]))
    if bestcut:
        cmds.append(f"env -u OMP_NUM_THREADS sbatch -J fb2E_bestcut -p submit,submit-gpu --exclude={EXCLUDE} -c 8 "
                    f"--mem=12G -t 00:45:00 slurm/run_py.sbatch hnl.eval.strong_baselines_armiii bestcut "
                    f"--seeds {','.join(sorted(set(bestcut)))} --out {os.path.abspath(out)} --name {name}")
    return cmds


def _git() -> str:
    try:
        h = subprocess.run(["git", "-C", RT.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", RT.REPO, "status", "--porcelain", "--", "src"], capture_output=True,
                           text=True).stdout.strip()
        return h + (f" (src dirty: {len(d.splitlines())} files)" if d else "")
    except OSError:
        return "unknown"


def cmd_bestcut(seeds: list[str], out: str, name: str) -> None:
    """CC at the val-fixed best cuts of acorn (ttbar+HNL) on the stored test scores, for the seeds whose cut the
    one-time test chain did not evaluate (post-freeze, descriptive; once; s* rows checked against RUN/eval_test)."""
    from hnl.eval import evaluate_gnn as EG
    from hnl.eval.published_defs import _same_rows
    wp = json.load(open(_repo(RT.WP)))
    s_star = float(wp["s_star"][ARM])
    spec_file = os.path.abspath(os.path.join(out, f"{name}_spec_test.json"))
    ts = json.load(open(spec_file))["models"]
    best = next(m for m in ts if m.get("cc_rule") == "best")
    job = os.environ.get("SLURM_JOB_ID", "local")
    for s in seeds:
        e = next(x for x in best["entries"] if _seed(x["run"]) == str(s))
        run, cut = e["run"], float(e["cut"])
        if e["sub"] == "eval_test":
            print(f"{os.path.basename(run)}: best cut {cut} is in the frozen eval_test; nothing to do")
            continue
        d = e["sub"]
        if os.path.exists(os.path.join(d, "provenance.json")):
            raise SystemExit(f"{d} exists: the post-freeze evaluation is done once")
        local = os.path.join("/tmp", os.environ.get("USER", "u"), f"fb2E_bestcut_{job}_s{s}")
        os.makedirs(os.path.join(local, "testset"), exist_ok=True)
        tar = os.path.join(run, "gnn_scores_test", "testset.tar")
        try:
            subprocess.run(["tar", "xf", tar, "-C", os.path.join(local, "testset")], check=True)
            cuts = sorted({cut, s_star})
            EG.main(["--graphs", os.path.join(local, "testset"), "--out", d, "--cuts", ",".join(f"{c:.4f}" for c in cuts)])
        finally:
            shutil.rmtree(local, ignore_errors=True)
        ok = {}
        for f, keys in (("per_particle.parquet", ["event_id", "particle_id"]), ("per_event.parquet", ["event_id"])):
            new = pd.read_parquet(os.path.join(d, f))
            old = pd.read_parquet(os.path.join(run, "eval_test", f))
            ok[f] = bool(_same_rows(new[np.isclose(new.cut, s_star, atol=1e-6)],
                                    old[np.isclose(old.cut, s_star, atol=1e-6)], keys))
        prov = {"what": "post-freeze CC evaluation of acorn (ttbar+HNL) on the stored one-time test scores at the best "
                        "cut <= FR* fixed on val (descriptive, not a claim; hnl.eval.strong_baselines_armiii bestcut)",
                "run": run, "scores": tar, "cut_best_val": cut, "cut_source": spec_file, "spec": spec_file,
                "spec_sha256": _sha256(spec_file),
                "cuts_evaluated": cuts, "identity_check_s_star": ok, "git": _git(), "slurm_job": job,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(os.path.join(d, "provenance.json"), "w") as fh:
            json.dump(prov, fh, indent=1)
        print(json.dumps(prov, indent=1))
        if not all(ok.values()):
            raise SystemExit(f"{os.path.basename(run)}: s* rows differ from the frozen eval_test")


# ------------------------------------------------------------------------------------------------- test
def _check_bestcut(e: dict) -> None:
    f = os.path.join(e["sub"], "provenance.json")
    p = json.load(open(f))
    ok = (os.path.normpath(p["run"]) == os.path.normpath(e["run"]) and abs(float(p["cut_best_val"]) - e["cut"]) < 1e-6
          and bool(p["identity_check_s_star"]) and all(p["identity_check_s_star"].values()) and RT._holds(e["sub"], e["cut"]))
    if not ok:
        raise SystemExit(f"{f}: not run {e['run']} at cut {e['cut']}, or its s* identity check failed")
    sf = p.get("spec")
    if not sf or not os.path.exists(sf) or _sha256(sf) != p.get("spec_sha256"):
        raise SystemExit(f"{f}: its spec file {sf} is missing or changed since the job (checksum mismatch: stop)")
    ent = [x for m in json.load(open(sf))["models"] if m.get("cc_rule") == "best" for x in m["entries"]
           if os.path.normpath(x["run"]) == os.path.normpath(e["run"])]
    if len(ent) != 1 or abs(float(ent[0]["cut"]) - e["cut"]) > 1e-9:
        raise SystemExit(f"{f}: the spec {sf} of the job does not hold the best cut {e['cut']} for {e['run']}")


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check_spec_of_output(e: dict, label: str) -> None:
    """A test output was written from a val-fixed spec file (path and sha256 in its provenance) that is unchanged
    since (checksum) and holds this very entry (row label, run, configuration, output directory): the outputs of
    rows complete before the last val grid finished may come from the first-stage spec (`val --skip-missing`), whose
    entries must then equal the final ones."""
    p = json.load(open(os.path.join(e["sub"], "provenance.json")))
    f = p.get("spec")
    if not f or not os.path.exists(f):
        raise SystemExit(f"{e['sub']}: provenance names no existing spec file ({f})")
    if _sha256(f) != p.get("spec_sha256"):
        raise SystemExit(f"{e['sub']}: {f} changed since the test job (sha256 {_sha256(f)} != {p.get('spec_sha256')}): "
                         "checksum mismatch, stop")
    ms = [m for m in json.load(open(f))["models"] if m["label"] == label]
    same = [x for m in ms for x in m["entries"] if os.path.normpath(x["run"]) == os.path.normpath(e["run"])
            and x.get("config") == e["config"] and os.path.normpath(x["sub"]) == os.path.normpath(e["sub"])]
    if len(same) != 1:
        raise SystemExit(f"{e['sub']}: the spec {f} of its test job does not hold {label} / {e['run']} / {e['config']}")


def build_test_spec(out: str, name: str) -> tuple[list[dict], dict]:
    """Round 1's test spec (hnl.eval.strong_baselines.test_spec: its outputs and provenance checked) and the acorn
    (ttbar+HNL) rows of our val-fixed spec (outputs and provenance checked; every builder output must have been
    written from this very spec file: sha256 in its provenance)."""
    v1 = SB.test_spec(_repo("results"), V1)
    spec_file = os.path.join(out, f"{name}_spec_test.json")
    meta = json.load(open(spec_file))
    if not meta.get("complete"):
        raise SystemExit("the val spec is incomplete (run `val` after every grid exists)")
    iii = []
    for m in meta["models"]:
        if "key" in m:
            SB._check_test_outputs(m)
            for e in m["entries"]:
                _check_spec_of_output(e, m["label"])
        else:
            for e in m["entries"]:
                if e["sub"] != "eval_test":
                    _check_bestcut(e)
        iii.append({k: m[k] for k in ("label", "short", "group")}
                   | {"entries": [{k: e[k] for k in ("run", "sub", "cut", "wp")} for e in m["entries"]]})
    return v1 + iii, meta


def test_checks(tab: pd.DataFrame, more: dict) -> list:
    """Round 1's test rows reproduced (results/strong_baselines_v1.csv, set test), and this arm's CC row at the
    preregistered cuts against the M3 test summary (runs/baseline_v1/analysis_test/test_summary.json, arm iii)."""
    checks: list = []
    t1 = pd.read_csv(_repo(f"results/{V1}.csv"))
    t1 = t1[t1.set == "test"].set_index("label")
    t = tab.set_index("label")
    for lab in t1.index:
        if lab in t.index:
            _cmp_row(t.loc[lab], t1.loc[lab], f"round 1 test row {lab}", checks)
            for c in ("primary_sigma_total", "primary_sigma_boot"):
                mine = more[lab]["primary"]["sigma" if c == "primary_sigma_total" else "sigma_boot"]
                checks.append((f"round 1 test row {lab}: {c}", mine, float(t1.loc[lab, c]),
                               abs(mine - float(t1.loc[lab, c])) <= TOL))
    a = json.load(open(ANALYSIS_TEST))[ARM]
    lab = f"{ALAB} + CC, preregistered cut rule"
    r = t.loc[lab]
    for k, q in (("eff_primary", "eff_primary@hnl_joint_test+hnl15_100_external"),
                 ("eff_hnl_targets", "eff_hnl_targets@hnl_joint_test+hnl15_100_external"),
                 ("eff_w_muons", "eff_w_muons@hnl_joint_test+hnl15_100_external"),
                 ("eff_ttbar_prompt", "eff_ttbar_prompt@ttbar_test"),
                 ("fake_rate_hnl", "fake_rate@hnl_joint_test+hnl15_100_external")):
        checks.append((f"M3 test summary arm iii: {k} mean", float(r[k]), float(a[q]["mean"]),
                       abs(float(r[k]) - float(a[q]["mean"])) <= TOL))
        checks.append((f"M3 test summary arm iii: {k} seed sd", float(r[k + "_sd"]), float(a[q]["s_seeds"]),
                       abs(float(r[k + "_sd"]) - float(a[q]["s_seeds"])) <= TOL))
    p = more[lab]["primary"]
    for k in ("sigma", "sigma_boot"):
        q = a["eff_primary@hnl_joint_test+hnl15_100_external"][k]
        checks.append((f"M3 test summary arm iii: primary {k}", p[k], float(q), abs(p[k] - float(q)) <= TOL))
    return checks


QLAB = SB.QLAB


def _diff_cells(g: pd.DataFrame) -> list[str]:
    g = g.set_index("quantity")
    cells = []
    for q, *_ in SB.QUANTITIES:
        r = g.loc[q]
        z = f"z {r.z:+.1f}" if np.isfinite(r.z) else "identical"
        cells.append(f"{100 * r.difference:+.2f} ± {100 * r.sigma:.2f} ({z})")
    return cells


def _diff_table(d: pd.DataFrame, block: str) -> list[str]:
    lines = ["| new | baseline | " + " | ".join(QLAB[q] for q, *_ in SB.QUANTITIES) + " |",
             "|---|---|" + "---|" * len(SB.QUANTITIES)]
    sel = d[d.blocks.str.split("; ").apply(lambda b: block in b)]
    for (a, b), g in sel.groupby(["new", "baseline"], sort=False):
        lines.append(f"| {a} | {b} | " + " | ".join(_diff_cells(g)) + " |")
    return lines


def markdown(vt: pd.DataFrame, tt: pd.DataFrame, more: dict, diffs: pd.DataFrame, vdiffs: pd.DataFrame, fr: float,
             meta: dict, ed: pd.DataFrame, ps: pd.DataFrame, notes: list[str]) -> str:
    best = meta["iii_best"]
    L = ["# acorn (ttbar+HNL) with its strongest track builders (supervisor feedback round 2, point 2c, interim)", "",
         "acorn (ttbar+HNL) = acorn's interaction GNN trained on ttbar and HNL events together (Stage-0 baseline arm "
         "(iii): 7,999 ttbar + 7,999 HNL training events; three seeds, the Stage-0 scores). It stands in for the "
         "requested \"acorn trained on ttbar plus a generic displaced sample\", which does not exist yet; its HNL "
         "training events are disjoint from the val and test events but come from the evaluated signal samples (same "
         "masses and lifetimes), so it is an optimistic stand-in. Builders as in round 1 "
         "(results/strong_baselines_v1.md): acorn's own CC + walkthrough (CCandWalk, plain and score-guided; the same "
         "27-configuration grid per variant), our walkthrough + merging, the Canto builder (same-layer MWS + merging), "
         f"and connected components at the best cut <= FR* and at the preregistered cut rule. Every track-building "
         f"configuration is chosen per seed on val by the builder rule at FR* = {fr:.6f} (pooled HNL val) with the "
         "one-standard-error tie rule, and the CC cuts per seed on val by the CC rules of ranking v3 (no tie rule); "
         "nothing is chosen on test. Reference rows (round 1, reproduced to 1e-12): the frozen Canto-L "
         "(trained on ttbar only), acorn (HNL) and acorn (ttbar) at their val-best configuration. Cells: mean ± seed sd "
         "over three seeds; |d0| bins: HNL-daughter targets pooled over the seeds (k/n).", "",
         "## Validation (pooled HNL val 1,999 events + ttbar val 1,000 events)", ""]
    L += SB._table(vt)
    L += ["", f"val-best acorn (ttbar+HNL) row (highest val primary efficiency of its six rows): {best}.", "",
          "## Test, post-freeze descriptive (not claims; the preregistered claims stay as they are)", "",
          "Pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) for primary, "
          "|d0|, W muons and fake rate; ttbar_test (1,000 events) for ttbar prompt. The val-chosen configurations "
          "applied once to the stored test scores; summary code path of hnl.eval.ranking_test.", ""]
    L += SB._table(tt)
    L += ["", "Primary efficiency with the preregistered total sigma (event bootstrap + seed term):", ""]
    for _, r in tt.iterrows():
        p = more[r.label]["primary"]
        L.append(f"- {r.label}: {p['mean']:.4f} ± {p['sigma']:.4f} (bootstrap {p['sigma_boot']:.4f}, seed sd "
                 f"{p['s_seeds']:.4f})")
    L += ["", "## Differences on test (new minus baseline; paired event bootstrap of the preregistered test tables, "
              "B = 500, + both seed terms: the claims' estimator; post-freeze descriptive)", ""]
    for blk in BLOCKS:
        L += [f"**{blk}**", ""] + _diff_table(diffs, blk) + [""]
    if ps is not None and len(ps):
        L += ["Primary efficiency per test set (descriptive; seed mean ± total sigma; difference with the same "
              "estimator):", "", "| test set | primary particles per seed | new | value | baseline | value | "
                                 "difference |", "|---|---|---|---|---|---|---|"]
        for _, r in ps.iterrows():
            L.append(f"| {r.set} | {int(r.n_per_seed)} | {r.new} | {r.value_new:.4f} ± {r.sigma_new:.4f} | "
                     f"{r.baseline} | {r.value_base:.4f} ± {r.sigma_base:.4f} | {100 * r.difference:+.2f} ± "
                     f"{100 * r.sigma:.2f} (z {r.z:+.1f}) |")
        L.append("")
    L += ["## Differences on val (new minus baseline; paired event bootstrap of the val tables of feedback point 5, "
          "B = 1000, + both seed terms; validation)", ""]
    for blk in BLOCKS:
        L += [f"**{blk}**", ""] + _diff_table(vdiffs, blk) + [""]
    L += ["Points = percentage points (fake rate: a negative difference means the new row has the lower fake rate). "
          "A row's own sigma (bootstrap + seed term) is in the CSV files.", "",
          "## Chosen configurations and their place in the grids (val)", "",
          "| row | seed | configuration | grid (configurations of the family) | edge | lower-edge rule (CCandWalk) |",
          "|---|---|---|---|---|---|"]
    for _, r in ed.iterrows():
        L.append(f"| {r.label} | {r.seed} | {r.family} {r.params} | {r.grid} ({r.n_family_configs}) | {r.edges} | "
                 f"{r.lower_edge_rule or '-'} |")
    L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(L) + "\n"


def run_test(out: str, name: str, notes_file: str | None) -> None:
    fr = SB.fr_star()
    spec, meta = build_test_spec(out, name)
    tab, ent, more = RT.rank(spec)
    checks = test_checks(tab, more)
    iii = [m["label"] for m in meta["models"]]
    hnl_best, ttbar_best, best = meta["hnl_best"], meta["ttbar_best"], meta["iii_best"]
    prs = pairs(iii, hnl_best, ttbar_best)
    d = SB.differences(spec, [(a, b) for a, b, _ in prs])
    blk = {(a, b): k for a, b, k in prs}
    d.insert(0, "blocks", [blk[(a, b)] for a, b in zip(d.new, d.baseline)])
    d.insert(0, "set", "test")
    d["B"] = BS.B
    _diff_identity(d, pd.read_csv(_repo(f"results/{V1}_differences.csv")), "round 1 test differences", checks)
    claims = json.load(open(_repo("results/claims_test_v1.json")))["result"]["primary"]["difference"]
    pd.DataFrame(checks, columns=["check", "ours", "reference", "ok"]).to_csv(
        os.path.join(out, f"{name}_test_checks.csv"), index=False)
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:20]:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} of {len(checks)} test identity checks failed")
    print(f"test identity checks: {len(checks)} of {len(checks)} ok (claims' primary difference "
          f"{claims['value']:.6f} untouched)")
    roles = _roles(hnl_best, ttbar_best, iii, best)
    show = [NEW, hnl_best, ttbar_best] + iii
    tt = _rows_for_tables(tab, show, roles)
    tt.insert(0, "set", "test")
    for lab in tt.label:
        p = more[lab]["primary"]
        tt.loc[tt.label == lab, "primary_sigma_total"] = p["sigma"]
        tt.loc[tt.label == lab, "primary_sigma_boot"] = p["sigma_boot"]
    tent = ent[ent.label.isin(show)].copy()
    tent.insert(0, "set", "test")
    ps = SB.per_set(spec, [(NEW, best)])
    ps.to_csv(os.path.join(out, f"{name}_test_per_set.csv"), index=False)
    vt = pd.read_csv(os.path.join(out, f"{name}_val.csv"))
    cols = list(vt.columns) + [c for c in tt.columns if c not in vt.columns]
    pd.concat([vt, tt], ignore_index=True)[cols].to_csv(os.path.join(out, f"{name}.csv"), index=False)
    ve = pd.read_csv(os.path.join(out, f"{name}_val_entries.csv"))
    pd.concat([ve, tent], ignore_index=True).to_csv(os.path.join(out, f"{name}_entries.csv"), index=False)
    vd = pd.read_csv(os.path.join(out, f"{name}_val_differences.csv"))
    pd.concat([vd, d], ignore_index=True).to_csv(os.path.join(out, f"{name}_differences.csv"), index=False)
    ed = pd.read_csv(os.path.join(out, f"{name}_edges.csv"), dtype=str).fillna("")
    notes = [] if notes_file is None else [ln.strip()[2:] for ln in open(notes_file) if ln.startswith("- ")]
    md = markdown(vt, tt, more, d, vd, fr, meta, ed, ps, notes)
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        f.write(md)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["merge", "val", "jobs", "bestcut", "test"])
    ap.add_argument("--out", default=_repo("results"))
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--skip-missing", action="store_true", help="val: skip builder rows whose grids are missing")
    ap.add_argument("--seeds", default="0,2", help="bestcut: seeds")
    ap.add_argument("--notes", default=None, help="test: a text file of '- ' note lines for the markdown")
    ap.add_argument("--cpus", type=int, default=8, help="jobs: CPUs per test job")
    ap.add_argument("--mem", default="12G", help="jobs: memory per test job")
    ap.add_argument("--time", default="01:00:00", help="jobs: time limit per test job")
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        for s in SEEDS:
            print(SB.merge_grids(os.path.join(BASE, f"arm{ARM}_s{s}")))
    elif a.cmd == "val":
        run_val(a.out, a.name, a.skip_missing)
    elif a.cmd == "jobs":
        print("\n".join(jobs(a.out, a.name, a.cpus, a.mem, a.time)))
    elif a.cmd == "bestcut":
        cmd_bestcut(a.seeds.split(","), a.out, a.name)
    else:
        run_test(a.out, a.name, a.notes)


if __name__ == "__main__":
    main()
