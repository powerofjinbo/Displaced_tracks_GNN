"""Writes results/training_sample_bins_v1.{md,csv} and paper/tables/tab_training.tex: the training-sample study of
acorn and Canto-L (validation only), every model with MWS-layer + learned merging (LM) at its working point.

Per model (acorn and Canto-L, each trained on ttbar, on ttbar+HNL and on HNL): the primary efficiency and its sigma,
taken from scripts/diagnostics/training_sample_summary.py (table("mwslmerge"), the code path of
results/training_sample_v1.md:11-16, which this script checks line by line); the HNL daughters per abs(d0) bin
(10-30, 30-100, 100-300, >= 300 mm and >= 100 mm) as counts summed over seeds, counts per seed and the mean of the
per-seed efficiencies; the W muons, the ttbar prompt targets and the fake rate (mean and seed sd). Inputs:
results/training_sample_val_mwslmerge.csv (official grids, hnl.eval.training_sample_board),
results/strong_baselines_v3_val_differences.csv (three-seed sigma, through training_sample_summary.py),
results/training_sample_paired_v1.csv (paired differences, Table III). Nothing is read from the test sets.

Table III (paper/tables/tab_training.tex) is generated here from the same numbers; edit this script, not the table.

    source env/activate.sh && python scripts/diagnostics/training_sample_bins.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training_sample_summary as tss  # noqa: E402

REPO = tss.REPO
RES = tss.RES
FAM = "mwslmerge"
GRID = os.path.join(RES, f"training_sample_val_{FAM}.csv")
SUMMARY_MD = os.path.join(RES, "training_sample_v1.md")
SUMMARY_LINES = {m: 11 + i for i, m in enumerate(tss.MODELS)}       # training_sample_v1.md:11-16, as cited in NUMBERS
OUT_MD = os.path.join(RES, "training_sample_bins_v1.md")
OUT_CSV = os.path.join(RES, "training_sample_bins_v1.csv")
OUT_TAB = os.path.join(REPO, "paper", "tables", "tab_training.tex")
CEIL = os.path.join(RES, "ceiling_table2_v1.csv")   # perfect-classifier limits (hnl.eval.ceiling_table2), Table III row
CE_IDS = {"acorn (ttbar)": "CE01", "Canto-L (ttbar)": "CE02", "acorn (ttbar+HNL)": "CE03",
          "Canto-L (ttbar+HNL)": "CE04", "acorn (HNL)": "CE05", "Canto-L (HNL)": "CE06"}
# abs(d0) bins: (key, csv column stem or the stems that are summed, text)
BINS = [("10_30", ["d0_10_30"], "10-30 mm"), ("30_100", ["d0_30_100"], "30-100 mm"),
        ("100_300", ["d0_100_300"], "100-300 mm"), ("300_inf", ["d0_300_inf"], ">= 300 mm"),
        ("100_inf", ["d0_100_300", "d0_300_inf"], ">= 100 mm")]


def arch_sample(m):
    a, s = m.split(" (")
    return a, s.rstrip(")")


def collect():
    """One dict per model, in the order of training_sample_v1.md (tss.MODELS)."""
    t = pd.read_csv(GRID)
    summ, spread = tss.table(FAM)
    rows = []
    for m in tss.MODELS:
        g = t[t.model == m]
        if g.empty:
            raise SystemExit(f"{GRID}: no rows for {m}")
        s = summ[m]
        a, samp = arch_sample(m)
        r = {"model": m, "arch": a, "sample": samp, "n_seeds": len(g), "seeds": ";".join(str(int(x)) for x in g.seed),
             "working_point": s["wp"], "primary_eff": s["primary"], "primary_sigma": s["sigma"],
             "primary_ineff": s["ineff"], "primary_k": int(g.eff_primary_k.sum()), "primary_n": int(g.eff_primary_n.sum())}
        assert abs(float(g.eff_primary.mean()) - s["primary"]) < 1e-12, m
        for key, stems, _ in BINS:
            k = float(sum(g[f"{c}_k"].sum() for c in stems))
            n = float(sum(g[f"{c}_n"].sum() for c in stems))
            n_seed = {float(sum(row[f"{c}_n"] for c in stems)) for _, row in g.iterrows()}
            assert len(n_seed) == 1, (m, key, n_seed)            # the same particles in every seed
            eff_mean = float(np.mean([sum(row[f"{c}_k"] for c in stems) / sum(row[f"{c}_n"] for c in stems)
                                      for _, row in g.iterrows()]))
            assert abs(eff_mean - k / n) < 1e-12, (m, key)        # mean of the per-seed efficiencies = pooled k / n
            if len(stems) == 1:
                assert abs(float(g[stems[0]].mean()) - eff_mean) < 1e-12, (m, key)
            r.update({f"d0_{key}_k": int(round(k)), f"d0_{key}_n": int(round(n)), f"d0_{key}_k_per_seed": k / len(g),
                      f"d0_{key}_n_per_seed": n / len(g), f"d0_{key}_eff": eff_mean, f"d0_{key}_ineff": 1 - eff_mean})
        for col, name in (("eff_w_muons", "w_muons"), ("eff_ttbar_prompt", "ttbar_prompt"), ("fake_rate_hnl", "fake_rate")):
            r[name] = float(g[col].mean())
            r[f"{name}_seed_sd"] = float(g[col].std(ddof=1)) if len(g) > 1 else np.nan
        r["spread_canto_ttbar"] = spread
        rows.append(r)
    return rows, spread


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def summary_line(r):
    """The row of results/training_sample_v1.md, formatted as training_sample_summary.py formats it."""
    return (f"| {r['model']} | {r['n_seeds']} | {pct(r['primary_eff'])} +- {pct(r['primary_sigma'])} | "
            f"{pct(r['primary_ineff'])} | {pct(r['d0_100_300_eff'], 1)} | {r['d0_300_inf_k']}/{r['d0_300_inf_n']} | "
            f"{pct(r['w_muons'], 1)} | {pct(r['ttbar_prompt'], 1)} | {pct(r['fake_rate'])} |")


def check_summary(rows):
    """Every model reproduces its cited line of results/training_sample_v1.md (lines 11-16) character by character."""
    lines = open(SUMMARY_MD).read().splitlines()
    for r in rows:
        ln = SUMMARY_LINES[r["model"]]
        if lines[ln - 1] != summary_line(r):
            raise SystemExit(f"{SUMMARY_MD}:{ln} is\n  {lines[ln - 1]}\nrecomputed\n  {summary_line(r)}")


def paired_per_sample():
    """Canto-L minus acorn at each training sample (paired; results/training_sample_paired_v1.csv, LM)."""
    out = {}
    for s in ("ttbar", "ttbar+HNL", "HNL"):
        r = tss.paired(FAM, f"trained on {s}: Canto-L minus acorn")
        out[s] = (float(r.value), float(r.sigma), float(r.z))
    return out


def write_md(rows, spread):
    by = {r["model"]: r for r in rows}
    L = ["# Training-sample study: HNL daughters per abs(d0) bin and the summary of each model (validation, LM)", "",
         "Generated by scripts/diagnostics/training_sample_bins.py from results/training_sample_val_mwslmerge.csv (official",
         "grids, hnl.eval.training_sample_board); primary efficiency and sigma from scripts/diagnostics/",
         "training_sample_summary.py (table(\"mwslmerge\"), the code path of results/training_sample_v1.md:11-16, which",
         "these numbers reproduce character by character). Validation events only. Every model with MWS-layer + learned",
         "merging (LM) at its working point (fake rate <= FR*), chosen per seed on validation. Canto-L trained on",
         "ttbar+HNL and on HNL: one seed each; every other model: three seeds. Bin efficiencies: the mean over seeds of",
         "the per-seed efficiency (= the pooled k/n, since every seed has the same particles).", "",
         "## Primary population", ""]
    for r in rows:
        L.append(f"- primary, {r['model']}, {r['n_seeds']} seed{'s' if r['n_seeds'] > 1 else ''}: "
                 f"{pct(r['primary_eff'])} +- {pct(r['primary_sigma'])} %, inefficiency {pct(r['primary_ineff'])} % "
                 f"({r['primary_k']}/{r['primary_n']} over seeds)")
    L += [f"- seed spread (sd) of Canto-L (ttbar) primary efficiency: {pct(spread)} points (one-seed sigma term)", "",
          "## Efficiency of the HNL daughters per abs(d0) bin (%, mean over seeds)", ""]
    for key, _, text in BINS:
        n1 = {r[f"d0_{key}_n_per_seed"] for r in rows}
        assert len(n1) == 1, (key, n1)
        n1 = int(n1.pop())
        if key in ("300_inf", "100_inf"):
            vals = "; ".join(f"{r['model']} {r[f'd0_{key}_k']}/{r[f'd0_{key}_n']} ({pct(r[f'd0_{key}_ineff'], 1)} % lost)"
                             for r in rows)
            L.append(f"- abs(d0) {text} ({n1} per seed), k/n over seeds: {vals}")
        else:
            vals = "; ".join(f"{r['model']} {pct(r[f'd0_{key}_eff'], 1)}" for r in rows)
            L.append(f"- abs(d0) {text} ({n1} per seed), efficiency (%): {vals}")
    L += ["", "## Prompt references and fake rate (mean over seeds; seed sd for three seeds)", ""]
    for r in rows:
        sd = (lambda c: f" (sd {pct(r[c + '_seed_sd'])})" if np.isfinite(r[c + "_seed_sd"]) else "")
        L.append(f"- prompt, {r['model']}: W muons {pct(r['w_muons'], 1)} %{sd('w_muons')}, ttbar prompt "
                 f"{pct(r['ttbar_prompt'], 1)} %{sd('ttbar_prompt')}, fake rate {pct(r['fake_rate'])} %{sd('fake_rate')}")
    L += ["", "## Working points (builder setting chosen per seed on validation)", ""]
    for r in rows:
        L.append(f"- working point, {r['model']}: {r['working_point']}")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    return L, by


def tex_pm(v, s):
    return f"${100 * v:.2f}\\pm{100 * s:.2f}$"


def write_table(rows):
    """Table III, transposed: benchmarks in rows, the six models in three column groups by training sample."""
    by = {r["model"]: r for r in rows}
    order = [f"{a} ({s})" for s in ("ttbar", "ttbar+HNL", "HNL") for a in ("acorn", "Canto-L")]
    cols = [by[m] for m in order]
    ts_ids = {"acorn (ttbar)": "TS01", "Canto-L (ttbar)": "TS02", "acorn (ttbar+HNL)": "TS03a",
              "Canto-L (ttbar+HNL)": "TS04a", "acorn (HNL)": "TS05a", "Canto-L (HNL)": "TS06a"}
    ts = ",".join(ts_ids[m] for m in order)
    pr = paired_per_sample()
    for s, a, c in (("ttbar", "acorn (ttbar)", "Canto-L (ttbar)"), ("ttbar+HNL", "acorn (ttbar+HNL)",
                                                                      "Canto-L (ttbar+HNL)"), ("HNL", "acorn (HNL)", "Canto-L (HNL)")):
        assert abs(pr[s][0] - (by[c]["primary_eff"] - by[a]["primary_eff"])) < 1e-9, s
    n_bin = {key: int(rows[0][f"d0_{key}_n_per_seed"]) for key, _, _ in BINS}
    assert n_bin["10_30"] == 508 and n_bin["30_100"] == 502 and n_bin["100_300"] == 167 and n_bin["300_inf"] == 19, n_bin
    seeds = " & ".join(str(r["n_seeds"]) for r in cols)
    prim = " & ".join(tex_pm(r["primary_eff"], r["primary_sigma"]) for r in cols)
    diff = " & ".join(f"\\multicolumn{{2}}{{c}}{{${100 * pr[s][0]:+.2f}\\pm{100 * pr[s][1]:.2f}$}}"
                      for s in ("ttbar", "ttbar+HNL", "HNL"))
    b = lambda key: " & ".join(pct(r[f"d0_{key}_eff"], 1) for r in cols)
    far = " & ".join(f"{r['d0_300_inf_k']}/{r['d0_300_inf_n']}" for r in cols)
    tt = " & ".join(pct(r["ttbar_prompt"], 1) for r in cols)
    fake = " & ".join(pct(r["fake_rate"]) for r in cols)
    w = {pct(r["w_muons"], 1) for r in rows}
    assert w == {"99.8"}, w                                     # the caption states one value for every model
    spread = pct(rows[0]["spread_canto_ttbar"])
    assert spread == "0.20", spread
    n_prim = {r["primary_n"] // r["n_seeds"] for r in rows}
    assert n_prim == {1331}, n_prim
    ce = pd.read_csv(CEIL)
    ce = ce[ce.quantity == "eff_primary"].set_index(["model", "kind"])
    for r in cols:                                              # the limits file holds Table III's own efficiencies
        assert abs(ce.loc[(r["model"], "own"), "value"] - r["primary_eff"]) < 1e-9, r["model"]
    perfect = " & ".join(tex_pm(ce.loc[(m, "lm_wp"), "value"], ce.loc[(m, "lm_wp"), "sigma"]) for m in order)
    ce_ids = ",".join(CE_IDS[m] for m in order)
    T = [
        "% tables/tab_training.tex -- Table III of the Canto paper (training-sample study, validation).",
        "% Generated by scripts/diagnostics/training_sample_bins.py from results/training_sample_val_mwslmerge.csv,",
        "% results/training_sample_v1.md:11-16 (reproduced) and results/training_sample_paired_v1.csv; edit the script,",
        "% not this file. Every number carries its paper/NUMBERS.md row ID in a trailing comment.",
        "% 2026-10-14: rewrite pass (docs/paper_rewrite_brief.md): caption.",
        "% 2026-10-14: perfect-classifier row (results/ceiling_table2_v1.csv; supervisor feedback round 3).",
        "\\begin{table*}[t]",
        "\\caption{\\label{tab:results}%",
        "Validation results of acorn and Canto-L trained on $t\\bar t$ events, on $t\\bar t$ and HNL events ($t\\bar t$+HNL)",
        "and on the HNL signal. Every model uses the same track builder, MWS-layer with learned merging (LM), at its",
        "working point (fake rate $\\le\\mathrm{FR}^*=2.94\\%$). Primary: track efficiency of the HNL daughters (1331 per % N:W02 ; N:D30",
        "seed). Canto-L $-$ acorn: difference at the same training sample, paired over events. The $|d_0|$ bins from 10 to",
        "300\\,mm hold 508, 502 and 167 daughters per seed; above 300\\,mm there are 19 per seed, given as $k/n$ summed over % N:TB02,TB03 ; N:D32 ; N:D33 ; N:W18",
        "seeds. $t\\bar t$ prompt: $t\\bar t$ targets with $|d_0|<1$\\,mm. Every model finds 99.8\\% of the $W$ muons. % N:D14 ; N:" + ts,
        "Uncertainties: Eq.~(\\ref{eq:sigma}); the one-seed models carry the seed spread of Canto-L ($t\\bar t$),",
        f"{spread} points. Canto-L ($t\\bar t$) is the frozen Canto-L. Perfect classifier: the primary efficiency of the same % N:TS07",
        "builder with every true edge of the graph scored one and every other edge zero (Sec.~\\ref{sec:graph}).}",
        "\\begin{ruledtabular}",
        "\\begin{tabular}{lcccccc}",
        "Trained on & \\multicolumn{2}{c}{$t\\bar t$} & \\multicolumn{2}{c}{$t\\bar t$+HNL} & \\multicolumn{2}{c}{HNL} \\\\",
        "\\cline{2-3}\\cline{4-5}\\cline{6-7}",
        " & acorn & Canto-L & acorn & Canto-L & acorn & Canto-L \\\\",
        "\\hline",
        f"Seeds & {seeds} \\\\ % N:{ts}",
        f"Primary (\\%) & {prim} \\\\ % N:{ts}",
        f"Canto-L $-$ acorn (points) & {diff} \\\\ % N:TS20,TS20b,TS20c",
        f"Perfect classifier (\\%) & {perfect} \\\\ % N:{ce_ids}",
        f"$|d_0|$ 10--30\\,mm (\\%) & {b('10_30')} \\\\ % N:TB02 ; N:W18",
        f"$|d_0|$ 30--100\\,mm (\\%) & {b('30_100')} \\\\ % N:TB03 ; N:W18",
        f"$|d_0|$ 100--300\\,mm (\\%) & {b('100_300')} \\\\ % N:{ts}",
        f"$|d_0|\\ge300$\\,mm ($k/n$) & {far} \\\\ % N:{ts}",
        f"$t\\bar t$ prompt (\\%) & {tt} \\\\ % N:{ts}",
        f"Fake rate (\\%) & {fake} \\\\ % N:{ts}",
        "\\end{tabular}",
        "\\end{ruledtabular}",
        "\\end{table*}",
    ]
    os.makedirs(os.path.dirname(OUT_TAB), exist_ok=True)
    open(OUT_TAB, "w").write("\n".join(T) + "\n")
    return T


def main():
    rows, spread = collect()
    check_summary(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, float_format="%.10g")
    L, _ = write_md(rows, spread)
    T = write_table(rows)
    print("\n".join(L))
    print("\n".join(T))
    print("\n".join([OUT_MD, OUT_CSV, OUT_TAB]))


if __name__ == "__main__":
    main()
