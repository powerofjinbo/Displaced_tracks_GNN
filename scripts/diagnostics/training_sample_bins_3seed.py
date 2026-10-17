"""Writes results/training_sample_bins_3seed_v1.{md,csv} and paper/tables/tab_training.tex: the training-sample study of
acorn and Canto-L with three seeds of every model (validation only), every model with MWS-layer + learned merging (LM)
at its working point. The three-seed successor of scripts/diagnostics/training_sample_bins.py (one seed of Canto-L
trained on ttbar+HNL and on HNL), which stays as the reproduction of results/training_sample_bins_v1.*.

Inputs (official outputs of hnl.eval.training_sample_3seed, job 6831564, commit 1d4c751):
- results/training_sample_val_3seed_v1.csv: one row per model, seed and builder family (rows of family mwslmerge here);
- results/training_sample_val_3seed_v1.md:13-18: per model the primary efficiency and its sigma (paper Eq. (sigma),
  the model's own seed spread), which this script reproduces from the CSV (all but the sigma, which only the md holds)
  character by character;
- results/training_sample_paired_3seed_v1.csv: paired differences (Canto-L minus acorn per training sample);
- results/ceiling_table2_v1.csv: perfect-classifier limits (hnl.eval.ceiling_table2), seed 0 only for Canto-L trained
  on ttbar+HNL and on HNL (the new seeds have no limits).
Per model: the HNL daughters per abs(d0) bin (10-30, 30-100, 100-300, >= 300 and >= 100 mm) as counts summed over
seeds, counts per seed and the mean of the per-seed efficiencies; W muons, ttbar prompt targets and fake rate (mean
and seed sd). Nothing is read from the test sets.

The paper's training-sample table (tab:results) is generated here; edit this script, not the table.

    source env/activate.sh && python scripts/diagnostics/training_sample_bins_3seed.py
"""
import os
import re

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
FAM = "mwslmerge"
BOARD = os.path.join(RES, "training_sample_val_3seed_v1.csv")
SUMMARY_MD = os.path.join(RES, "training_sample_val_3seed_v1.md")
PAIRED = os.path.join(RES, "training_sample_paired_3seed_v1.csv")
CEIL = os.path.join(RES, "ceiling_table2_v1.csv")
OUT_MD = os.path.join(RES, "training_sample_bins_3seed_v1.md")
OUT_CSV = os.path.join(RES, "training_sample_bins_3seed_v1.csv")
OUT_TAB = os.path.join(REPO, "paper", "tables", "tab_training.tex")
MODELS = ["acorn (ttbar)", "acorn (ttbar+HNL)", "acorn (HNL)", "Canto-L (ttbar)", "Canto-L (ttbar+HNL)", "Canto-L (HNL)"]
SUMMARY_LINES = {m: 13 + i for i, m in enumerate(MODELS)}     # training_sample_val_3seed_v1.md:13-18 (LM table)
CEIL_ONE_SEED = {"Canto-L (ttbar+HNL)", "Canto-L (HNL)"}      # limits of seed 0 only
CE_IDS = {"acorn (ttbar)": "CE01", "Canto-L (ttbar)": "CE02", "acorn (ttbar+HNL)": "CE03",
          "Canto-L (ttbar+HNL)": "CE04", "acorn (HNL)": "CE05", "Canto-L (HNL)": "CE06"}
TS_IDS = {"acorn (ttbar)": "TS21", "acorn (ttbar+HNL)": "TS22", "acorn (HNL)": "TS23", "Canto-L (ttbar)": "TS24",
          "Canto-L (ttbar+HNL)": "TS25", "Canto-L (HNL)": "TS26"}
PAIR_IDS = {"ttbar": "TS30", "ttbar+HNL": "TS31", "HNL": "TS32"}
# abs(d0) bins: (key, csv column stems summed, text)
BINS = [("10_30", ["d0_10_30"], "10-30 mm"), ("30_100", ["d0_30_100"], "30-100 mm"),
        ("100_300", ["d0_100_300"], "100-300 mm"), ("300_inf", ["d0_300_inf"], ">= 300 mm"),
        ("100_inf", ["d0_100_300", "d0_300_inf"], ">= 100 mm")]
TOL = 1e-12


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def md_row(m):
    """(seeds, primary, sigma, seed sd, inefficiency, 100-300, k300, n300, W, ttbar prompt, fake) as printed."""
    ln = SUMMARY_LINES[m]
    line = open(SUMMARY_MD).read().splitlines()[ln - 1]
    q = re.fullmatch(rf"\| {re.escape(m)} \| (\d) \| ([\d.]+) \+- ([\d.]+) \| ([\d.-]+) \| ([\d.]+) \| ([\d.]+) \| "
                     rf"(\d+)/(\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
    if q is None:
        raise SystemExit(f"{SUMMARY_MD}:{ln}: not the LM row of {m}: {line}")
    return q.groups(), line


def collect():
    """One dict per model, in the order of MODELS; every printed quantity reproduced from the board."""
    t = pd.read_csv(BOARD)
    t = t[t.family == FAM]
    rows = []
    for m in MODELS:
        g = t[t.model == m].sort_values("seed")
        if len(g) != 3 or list(g.seed) != [0, 1, 2]:
            raise SystemExit(f"{BOARD}: {m} has seeds {list(g.seed)}, not 0, 1, 2")
        (seeds, prim, sig, sd, ineff, b100, k300, n300, w, tt, fake), line = md_row(m)
        mean = float(g.eff_primary.mean())
        s_seeds = float(g.eff_primary.std(ddof=1))
        recomputed = (str(len(g)), pct(mean), sig, pct(s_seeds), pct(1 - mean), pct(g.d0_100_300.mean(), 1),
                      str(int(g.d0_300_inf_k.sum())), str(int(g.d0_300_inf_n.sum())), pct(g.eff_w_muons.mean(), 1),
                      pct(g.eff_ttbar_prompt.mean(), 1), pct(g.fake_rate_hnl.mean()))
        if recomputed != (seeds, prim, sig, sd, ineff, b100, k300, n300, w, tt, fake):
            raise SystemExit(f"{SUMMARY_MD}:{SUMMARY_LINES[m]} is\n  {line}\nrecomputed from {BOARD}\n  {recomputed}")
        r = {"model": m, "arch": m.split(" (")[0], "sample": m.split(" (")[1].rstrip(")"), "n_seeds": 3,
             "seeds": "0;1;2", "working_point": "; ".join(g.working_point), "primary_eff": mean,
             "primary_sigma": float(sig) / 100.0, "sigma_source": f"{os.path.relpath(SUMMARY_MD, REPO)}:{SUMMARY_LINES[m]}",
             "primary_seed_sd": s_seeds, "primary_ineff": 1 - mean, "primary_k": int(g.eff_primary_k.sum()),
             "primary_n": int(g.eff_primary_n.sum())}
        for key, stems, _ in BINS:
            k = float(sum(g[f"{c}_k"].sum() for c in stems))
            n = float(sum(g[f"{c}_n"].sum() for c in stems))
            n_seed = {float(sum(row[f"{c}_n"] for c in stems)) for _, row in g.iterrows()}
            assert len(n_seed) == 1, (m, key, n_seed)            # the same particles in every seed
            eff_mean = float(np.mean([sum(row[f"{c}_k"] for c in stems) / sum(row[f"{c}_n"] for c in stems)
                                      for _, row in g.iterrows()]))
            assert abs(eff_mean - k / n) < TOL, (m, key)          # mean of the per-seed efficiencies = pooled k / n
            if len(stems) == 1:
                assert abs(float(g[stems[0]].mean()) - eff_mean) < TOL, (m, key)
            r.update({f"d0_{key}_k": int(round(k)), f"d0_{key}_n": int(round(n)), f"d0_{key}_k_per_seed": k / 3,
                      f"d0_{key}_n_per_seed": n / 3, f"d0_{key}_eff": eff_mean, f"d0_{key}_ineff": 1 - eff_mean})
        for col, name in (("eff_w_muons", "w_muons"), ("eff_ttbar_prompt", "ttbar_prompt"), ("fake_rate_hnl", "fake_rate")):
            r[name] = float(g[col].mean())
            r[f"{name}_seed_sd"] = float(g[col].std(ddof=1))
        rows.append(r)
    return rows


def paired(label):
    p = pd.read_csv(PAIRED)
    r = p[(p.family == FAM) & (p.label == label)]
    if len(r) != 1:
        raise SystemExit(f"{PAIRED}: {len(r)} rows for {FAM} / {label}")
    return r.iloc[0]


def paired_per_sample(by):
    out = {}
    for s, a, c in (("ttbar", "acorn (ttbar)", "Canto-L (ttbar)"), ("ttbar+HNL", "acorn (ttbar+HNL)", "Canto-L (ttbar+HNL)"),
                    ("HNL", "acorn (HNL)", "Canto-L (HNL)")):
        r = paired(f"trained on {s}: Canto-L minus acorn")
        assert abs(float(r.value) - (by[c]["primary_eff"] - by[a]["primary_eff"])) < 1e-9, s
        out[s] = (float(r.value), float(r.sigma), float(r.z))
    return out


def write_md(rows):
    L = ["# Training-sample study, three seeds of every model: HNL daughters per abs(d0) bin and the summary of each "
         "model (validation, LM)", "",
         "Generated by scripts/diagnostics/training_sample_bins_3seed.py from results/training_sample_val_3seed_v1.csv",
         "(hnl.eval.training_sample_3seed, job 6831564; family mwslmerge); primary efficiency and sigma as in",
         "results/training_sample_val_3seed_v1.md:13-18, which these numbers reproduce character by character (the sigma",
         "is read from there). Validation events only. Every model with MWS-layer + learned merging (LM) at its working",
         "point (fake rate <= FR*), chosen per seed on validation; seeds 0-2 of every model. Bin efficiencies: the mean",
         "over seeds of the per-seed efficiency (= the pooled k/n, since every seed has the same particles).", "",
         "## Primary population", ""]
    for r in rows:
        L.append(f"- primary, {r['model']}, 3 seeds: {pct(r['primary_eff'])} +- {pct(r['primary_sigma'])} %, "
                 f"inefficiency {pct(r['primary_ineff'])} % ({r['primary_k']}/{r['primary_n']} over seeds)")
    L += ["", "## Efficiency of the HNL daughters per abs(d0) bin (%, mean over seeds)", ""]
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
    L += ["", "## Prompt references and fake rate (mean over seeds; seed sd)", ""]
    for r in rows:
        L.append(f"- prompt, {r['model']}: W muons {pct(r['w_muons'], 1)} % (sd {pct(r['w_muons_seed_sd'])}), ttbar "
                 f"prompt {pct(r['ttbar_prompt'], 1)} % (sd {pct(r['ttbar_prompt_seed_sd'])}), fake rate "
                 f"{pct(r['fake_rate'])} % (sd {pct(r['fake_rate_seed_sd'])})")
    L += ["", "## Working points (builder setting chosen per seed on validation)", ""]
    for r in rows:
        L.append(f"- working point, {r['model']}: {r['working_point']}")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    return L


def bin_line(key):
    """1-based line of OUT_MD that holds the abs(d0) bin `key` (for the NUMBERS rows TB10, TB11)."""
    text = dict((k, t) for k, _, t in BINS)[key]
    for i, ln in enumerate(open(OUT_MD).read().splitlines(), 1):
        if ln.startswith(f"- abs(d0) {text} "):
            return i
    raise SystemExit(f"{OUT_MD}: no line for {text}")


def tex_pm(v, s):
    return f"${100 * v:.2f}\\pm{100 * s:.2f}$"


def write_table(rows):
    """The training-sample table, transposed: benchmarks in rows, six models in three column groups by sample."""
    by = {r["model"]: r for r in rows}
    order = [f"{a} ({s})" for s in ("ttbar", "ttbar+HNL", "HNL") for a in ("acorn", "Canto-L")]
    cols = [by[m] for m in order]
    ts = ",".join(TS_IDS[m] for m in order)
    pr = paired_per_sample(by)
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
    n_prim = {r["primary_n"] // r["n_seeds"] for r in rows}
    assert n_prim == {1331}, n_prim
    ce = pd.read_csv(CEIL)
    ce = ce[ce.quantity == "eff_primary"].set_index(["model", "kind"])
    t = pd.read_csv(BOARD)
    t = t[t.family == FAM]
    for m in order:                                             # the limits file holds the efficiencies of its seeds
        own = float(ce.loc[(m, "own"), "value"])
        if m in CEIL_ONE_SEED:
            assert int(ce.loc[(m, "own"), "n_seeds"]) == 1, m
            s0 = float(t[(t.model == m) & (t.seed == 0)].eff_primary.iloc[0])
            assert abs(own - s0) < 1e-9, (m, own, s0)
        else:
            assert abs(own - by[m]["primary_eff"]) < 1e-9, m
    perfect = " & ".join(tex_pm(ce.loc[(m, "lm_wp"), "value"], ce.loc[(m, "lm_wp"), "sigma"]) for m in order)
    ce_ids = ",".join(CE_IDS[m] for m in order)
    l10, l30 = bin_line("10_30"), bin_line("30_100")
    assert (l10, l30) == (21, 22), (l10, l30)                  # the lines cited by NUMBERS TB10, TB11
    T = [
        "% tables/tab_training.tex -- the training-sample table of the Canto paper (tab:results; Appendix A), validation.",
        "% Generated by scripts/diagnostics/training_sample_bins_3seed.py from results/training_sample_val_3seed_v1.csv,",
        "% results/training_sample_val_3seed_v1.md:13-18 (reproduced) and results/training_sample_paired_3seed_v1.csv; edit",
        "% the script, not this file. Every number carries its paper/NUMBERS.md row ID in a trailing comment.",
        "% 2026-10-16: three seeds of every model (seeds 1 and 2 of Canto trained on ttbar+HNL and on HNL added; the",
        "% perfect-classifier limits of those two stay seed 0).",
        "\\begin{table*}[t]",
        "\\caption{\\label{tab:results}%",
        "Validation results of acorn and Canto trained on $t\\bar t$ events, on $t\\bar t$ and HNL events ($t\\bar t$+HNL)",
        "and on the HNL signal, three seeds each. Every model uses the same track builder, MWS-layer with learned merging",
        "(LM), at its working point (fake rate $\\le\\mathrm{FR}^*=2.94\\%$). Primary: track efficiency of the HNL daughters % N:W02 ; N:D30",
        "(1331 per seed). Canto $-$ acorn: difference at the same training sample, paired over events. The $|d_0|$ bins",
        "from 10 to 300\\,mm hold 508, 502 and 167 daughters per seed; above 300\\,mm there are 19 per seed, given as $k/n$ % N:TB10,TB11 ; N:D32 ; N:D33 ; N:W18",
        "summed over seeds. $t\\bar t$ prompt: $t\\bar t$ targets with $|d_0|<1$\\,mm. Every model finds 99.8\\% of the $W$ % N:D14 ; N:" + ts,
        "muons. Uncertainties: Eq.~(\\ref{eq:sigma}). Canto trained on $t\\bar t$ events is the model of the main text.",
        "Perfect classifier: the primary efficiency of the same builder with every true edge of the graph scored one and",
        "every other edge zero (Sec.~\\ref{sec:graph}); seed~0 only for Canto trained on $t\\bar t$+HNL and on HNL.}",
        "\\begin{ruledtabular}",
        "\\begin{tabular}{lcccccc}",
        "Trained on & \\multicolumn{2}{c}{$t\\bar t$} & \\multicolumn{2}{c}{$t\\bar t$+HNL} & \\multicolumn{2}{c}{HNL} \\\\",
        "\\cline{2-3}\\cline{4-5}\\cline{6-7}",
        " & acorn & Canto & acorn & Canto & acorn & Canto \\\\",
        "\\hline",
        f"Seeds & {seeds} \\\\ % N:{ts}",
        f"Primary (\\%) & {prim} \\\\ % N:{ts}",
        f"Canto $-$ acorn (points) & {diff} \\\\ % N:{PAIR_IDS['ttbar']},{PAIR_IDS['ttbar+HNL']},{PAIR_IDS['HNL']}",
        f"Perfect classifier (\\%) & {perfect} \\\\ % N:{ce_ids}",
        f"$|d_0|$ 10--30\\,mm (\\%) & {b('10_30')} \\\\ % N:TB10 ; N:W18",
        f"$|d_0|$ 30--100\\,mm (\\%) & {b('30_100')} \\\\ % N:TB11 ; N:W18",
        f"$|d_0|$ 100--300\\,mm (\\%) & {b('100_300')} \\\\ % N:{ts}",
        f"$|d_0|\\ge300$\\,mm ($k/n$) & {far} \\\\ % N:{ts}",
        f"$t\\bar t$ prompt (\\%) & {tt} \\\\ % N:{ts}",
        f"Fake rate (\\%) & {fake} \\\\ % N:{ts}",
        "\\end{tabular}",
        "\\end{ruledtabular}",
        "\\end{table*}",
    ]
    open(OUT_TAB, "w").write("\n".join(T) + "\n")
    return T


def main():
    rows = collect()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, float_format="%.10g")
    L = write_md(rows)
    T = write_table(rows)
    print("\n".join(L))
    print("\n".join(T))
    print("\n".join([OUT_MD, OUT_CSV, OUT_TAB]))


if __name__ == "__main__":
    main()
