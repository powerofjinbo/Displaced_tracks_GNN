"""Training-sample study with three seeds of Canto-L trained on HNL and on ttbar + HNL (docs/training_sample_study.md
section 9), val only.

Per seed: the rows of results/training_sample_val_<family>.csv (hnl.eval.training_sample_board) for the acorn arms and
Canto-L (ttbar), unchanged, and for Canto-L (HNL) and Canto-L (ttbar + HNL) seeds 0-2 the board's own selection
(hnl.eval.final_select.run_rows over the arm (i) selection's grids, builder rule at FR* with the tie rule) and
quantities (training_sample_board.metrics). Seed 0 must reproduce the board's row (working point and every quantity).
Uncertainties as Table III (paper Eq. sigma; hnl.eval.training_sample_paired): per model, the event bootstrap of the
seed mean combined with s^2 / n of the model's own seed spread; differences paired over the bootstrap replicates, each
model's seed spread entering as c^2 s^2 / n. The one-seed values (results/training_sample_paired_v1.csv) are listed
next to the three-seed ones.

    python -m hnl.eval.training_sample_3seed --out results/training_sample_val_3seed_v1.csv \
        --paired-out results/training_sample_paired_3seed_v1.csv --md results/training_sample_val_3seed_v1.md
    (--allow-missing: skip new seeds whose grids do not exist yet; for checking the code before the runs finish)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import training_sample_board as TSB
from hnl.eval import training_sample_paired as TSP

FAMILIES = (("mwslmerge", "MWS-layer + learned merging (LM)"), ("mwslayermerge", "MWS-layer + helix merging (MWS+M)"))
NEW = {"Canto-L (HNL)": "ii", "Canto-L (ttbar+HNL)": "iii"}
SEEDS = (0, 1, 2)
MODELS = ["acorn (ttbar)", "acorn (ttbar+HNL)", "acorn (HNL)", "Canto-L (ttbar)", "Canto-L (ttbar+HNL)", "Canto-L (HNL)"]
Q = "eff_primary"
TOL = 1e-9


def board_rows(fam: str, info: pd.DataFrame, wpj: dict, allow_missing: bool) -> pd.DataFrame:
    from hnl.eval import final_select as FS
    old = pd.read_csv(os.path.join(BT.REPO, "results", f"training_sample_val_{fam}.csv"))
    FS.matching_baseline = lambda run: os.path.join(TSB.R, "baseline_v1", "armi_s0")   # the board's guard baseline

    def fs_wp(run: str) -> str:
        t, _ = FS.run_rows(run, TSB.GRIDS, wpj, wpj["reference_arm"])
        r = t[t.family == fam].iloc[0]
        return f"{fam} {str(r.config).split(',', 1)[1]} ({r.grid})"

    rows = old[~old.model.isin(list(NEW))].to_dict("records")
    for name, arm in NEW.items():
        for s in SEEDS:
            run = os.path.join(TSB.R, "stage1_final_learned_ts", "l1tl", f"arm{arm}_s{s}")
            try:
                wp = fs_wp(run)
            except (SystemExit, IndexError, FileNotFoundError, KeyError) as err:
                if allow_missing and s > 0:
                    print(f"{fam} {name} s{s}: skipped ({err})")
                    continue
                raise
            row = {"model": name, "seed": s, "run": run, "working_point": wp, **TSB.metrics(run, wp, info)}
            if s == 0:
                ref = old[(old.model == name) & (old.seed == 0)]
                if len(ref) != 1:
                    raise SystemExit(f"{fam} {name} s0: {len(ref)} board rows")
                ref = ref.iloc[0]
                if ref.working_point != wp:
                    raise SystemExit(f"{fam} {name} s0: working point {wp} != board {ref.working_point}")
                for k, v in row.items():
                    if isinstance(v, float) and abs(v - float(ref[k])) > TOL:
                        raise SystemExit(f"{fam} {name} s0: {k} {v} != board {ref[k]}")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "family", fam)
    return df


def load_models(df: pd.DataFrame, ld: BT.ValLoader) -> dict:
    out = {}
    for name, g in df.groupby("model", sort=False):
        seeds = [ld.seed(r.run, r.working_point) for r in g.itertuples()]
        m = BT.Model(name, name, seeds, ld.rs, list(g.working_point), list(g.run))
        for i, r in enumerate(g.itertuples()):
            if abs(m.pts[Q][i] - float(r.eff_primary)) > TOL:
                raise ValueError(f"{name} seed {r.seed}: {m.pts[Q][i]} != board {r.eff_primary}")
        out[name] = m
    return out


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def oblique(lines: list) -> None:
    res = os.path.join(BT.REPO, "results")
    parts = [pd.read_csv(os.path.join(res, f"oblique_val_{x}_v1_particles.csv")) for x in ("ref", "ii", "iii")]
    for arm in NEW.values():
        for s in (1, 2):
            p = os.path.join(res, f"oblique_val_{arm}_s{s}_v1_particles.csv")
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
    ob = pd.concat(parts, ignore_index=True)
    ob["fam"] = ob.label.str.replace(r" s\d$", "", regex=True)
    lines += ["## Classifier-stage losses versus |d0| / r_first (graph ceiling matched; s0 = 0.2)", "",
              "| model | seeds | below 0.7 (%) | at >= 0.7 (%) | at >= 0.7 (k of n) |", "|---|---|---|---|---|"]
    for fam, name in (("Canto-L (i)", "Canto-L (ttbar)"), ("acorn (HNL)", "acorn (HNL)"), ("Canto-L (ii)", "Canto-L (HNL)"),
                      ("Canto-L (iii)", "Canto-L (ttbar+HNL)")):
        g = ob[(ob.fam == fam) & (ob.stage != "graph")]
        lo, hi = g[g.ratio < 0.7], g[g.ratio >= 0.7]
        lines.append(f"| {name} | {g.label.nunique()} | {100 * (lo.stage == 'classifier').mean():.1f} | "
                     f"{100 * (hi.stage == 'classifier').mean():.1f} | {int((hi.stage == 'classifier').sum())} of {len(hi)} |")
    lines.append("")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="per-seed rows of both families (CSV)")
    ap.add_argument("--paired-out", required=True, help="differences with paired uncertainties (CSV)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    info = BT.info_table(BT.expected_events(BT.VAL_SETS))
    wpj = json.load(open(os.path.join(BT.REPO, "results", "working_points_v1.json")))
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    one = pd.read_csv(os.path.join(BT.REPO, "results", "training_sample_paired_v1.csv"))
    boards, diffs = [], []
    L = ["# Training-sample study with three seeds of every model (validation)", "",
         "Generated by hnl.eval.training_sample_3seed (docs/training_sample_study.md section 9). Validation events only.",
         "Canto-L trained on HNL and on ttbar + HNL: seeds 0-2 (seed 0 as in results/training_sample_v1.md); every",
         "other model: the three seeds of results/training_sample_v1.md. Each model with the same builder family, its",
         "configuration chosen per seed on validation by the builder rule at FR*. Uncertainties: paper Eq. (sigma),",
         "each model with its own seed spread; differences paired over the bootstrap replicates.", ""]
    for fam, title in FAMILIES:
        df = board_rows(fam, info, wpj, a.allow_missing)
        boards.append(df)
        ms = load_models(df.drop(columns="family"), ld)
        ref_sd = float(np.std(ms["Canto-L (ttbar)"].pts[Q], ddof=1))
        L += [f"## {title}", "", "| model | seeds | primary (%) | seed sd (pts) | inefficiency (%) | 100-300 mm (%) | "
              ">= 300 mm | W muons (%) | ttbar prompt (%) | fake rate (%) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name in MODELS:
            g = df[df.model == name]
            if g.empty:
                continue
            d = BS.seed_mean(*ms[name].pr(Q))
            sig = TSP.total_sigma(ms[name], ref_sd)
            sd = f"{100 * d['s_seeds']:.2f}" if len(g) > 1 else "-"
            L.append(f"| {name} | {len(g)} | {pct(d['mean'])} +- {pct(sig)} | {sd} | {pct(1 - d['mean'])} | "
                     f"{pct(g.d0_100_300.mean(), 1)} | {int(g.d0_300_inf_k.sum())}/{int(g.d0_300_inf_n.sum())} | "
                     f"{pct(g.eff_w_muons.mean(), 1)} | {pct(g.eff_ttbar_prompt.mean(), 1)} | {pct(g.fake_rate_hnl.mean())} |")
        L += ["", "Per seed (primary %, working point):", ""]
        for name in NEW:
            for r in df[df.model == name].itertuples():
                L.append(f"- {name} s{r.seed}: {pct(r.eff_primary)} ({r.working_point}); 100-300 mm {int(r.d0_100_300_k)}/"
                         f"{int(r.d0_100_300_n)}, >= 300 mm {int(r.d0_300_inf_k)}/{int(r.d0_300_inf_n)}, fake {pct(r.fake_rate_hnl)}")
        L += ["", "Differences (points; one seed = results/training_sample_paired_v1.csv):", "",
              "| difference | three seeds | z | one seed | z (one seed) |", "|---|---|---|---|---|"]
        for r in TSP.rows_for(fam, ms):
            o = one[(one.family == fam) & (one.label == r["label"])]
            o1 = (f"{100 * o.value.iloc[0]:+.2f} +- {100 * o.sigma.iloc[0]:.2f} | {o.z.iloc[0]:.1f}" if len(o) == 1 else "- | -")
            L.append(f"| {r['label']} | {100 * r['value']:+.2f} +- {100 * r['sigma']:.2f} (boot {100 * r['sigma_boot']:.2f}, "
                     f"seeds {100 * r['sigma_seeds']:.2f}) | {r['z']:.1f} | {o1} |")
            diffs.append({k: v for k, v in r.items()})
        L.append("")
    oblique(L)
    pd.concat(boards, ignore_index=True).to_csv(a.out, index=False)
    pd.DataFrame(diffs)[["family", "kind", "label", "value", "sigma_boot", "sigma_seeds", "sigma", "z", "terms"]].to_csv(
        a.paired_out, index=False)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
