"""Read-out of the secondary-particle upweighting test (docs/secondary_upweight_plan.md, section 4; val only).

Runs: ref (the screening-budget L1 + trunc classifier of 2026-10-03, its stored scores rebuilt with the current code
under runs/stage1_secw/ref), secwctrl (the same configuration retrained by the same route as the test run) and
secw18 (displaced true edges weighted 18). For each run and graph (base: RUN/trackbuild_val; union: the learned
triplet-support union graphs of Canto-L, RUN/ltf_union_preview/trackbuild_val_ext):
  * the primary configuration, MWS+M (mwslayermerge) at s0 = 0.2;
  * the mwslayermerge configuration of the builder rule at FR* (highest primary efficiency at a pooled-HNL fake rate
    <= FR*; ties: the lower fake rate);
with primary, HNL-daughter |d0| 100-300 mm and >= 300 mm, W-muon, ttbar-prompt and ttbar 1-100 mm efficiencies and
the HNL and ttbar fake rates. Uncertainties: the val event-bootstrap tables of hnl.eval.bootstrap_tables (B = 1000,
seed 20261005), as 68 % intervals' half width; paired differences use the same replicates. One seed per run: no seed
term. Next to them: classifier-stage losses versus |d0| / r_first (results/<NAME>_oblique_{base,union}_summary.csv
of hnl.eval.oblique_val) and the edge TPR at the edge FPR 0.0122 (RUN/edge_tpr_estar*.json/csv).

    python -m hnl.eval.secondary_upweight --out results --name secondary_upweight_v1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap_tables as BT
from hnl.eval.particles import split_event_id

RUNS_DIR = os.path.join(BT.DATA, "runs")
RUNS = {"ref": os.path.join(RUNS_DIR, "stage1_secw", "ref", "armi_s0"),
        "secwctrl": os.path.join(RUNS_DIR, "stage1_secw", "secwctrl", "armi_s0"),
        "secw18": os.path.join(RUNS_DIR, "stage1_secw", "secw18", "armi_s0")}
GRAPHS = {"base": "trackbuild_val", "base_ext": "trackbuild_val_extm",
          "union": os.path.join("ltf_union_preview", "trackbuild_val_ext")}
# secw18's union graphs were scored on a GPU (slurm/stage1/score_graphs_gpu.sbatch) after its CPU scoring stalled on a
# busy node; for secwctrl the GPU- and CPU-scored union builds are identical (every particle and event row)
UNION_DIR = {"secw18": "ltf_union_preview_gpu"}
OBLIQUE = (("base", ""), ("base", "_efpr"), ("union", ""), ("union", "_efpr"))
PRIMARY = ("mwslayermerge", 0.2)
DIFFS = (("secw18", "secwctrl"), ("secw18", "ref"), ("secwctrl", "ref"))
QS = ("eff_primary", "d0_100_300", "d0_300_inf", "eff_w_muons", "eff_ttbar_prompt", "ttbar_d0_1_100",
      "fake_rate_hnl", "fake_rate_ttbar")


def _cfgs(d: str) -> list:
    return [tuple(c) for c in json.load(open(os.path.join(d, "configs.json")))]


def _index(cfgs: list, want: tuple) -> int:
    for i, c in enumerate(cfgs):
        if c[0] == want[0] and len(c) == len(want) and all(np.isclose(float(a), float(b)) for a, b in zip(c[1:], want[1:])):
            return i
    raise KeyError(f"{want} not in {cfgs}")


def sums(rs: BT.Resampler, pp: pd.DataFrame, pe: pd.DataFrame, info: pd.DataFrame) -> dict:
    out = BT.counts(rs, pp, pe, info)
    t = pp[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    code = split_event_id(t.event_id.to_numpy())[0]
    ad0 = t.d0.abs().to_numpy(float)
    with np.errstate(invalid="ignore"):
        m = (code == BT.TTBAR_CODE) & t.is_target.to_numpy(bool) & (ad0 >= 1.0) & (ad0 < 100.0)
    out["ttbar_d0_1_100"] = rs.unit_sums(t.event_id.to_numpy()[m], t.dm_matched.to_numpy(float)[m], np.ones(int(m.sum())))
    c = split_event_id(pe.event_id.to_numpy())[0] == BT.TTBAR_CODE
    out["fake_rate_ttbar"] = rs.unit_sums(pe.event_id.to_numpy()[c], pe.n_fake.to_numpy(float)[c],
                                          pe.n_tracks_ge5.to_numpy(float)[c])
    out["d0_100_300"] = out[BT.bin_name("d0", 100.0, 300.0)]
    out["d0_300_inf"] = out[BT.bin_name("d0", 300.0, np.inf)]
    return out


def evaluate(rs, info, d: str, i: int) -> dict:
    pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"), filters=[("config", "==", int(i))])
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"), filters=[("config", "==", int(i))])
    s = sums(rs, pp, pe, info)
    return {q: rs.ratio(s[q]) for q in QS}


def rule_index(rs, info, d: str, cfgs: list, fr_star: float) -> tuple[int, pd.DataFrame]:
    rows = []
    for i, c in enumerate(cfgs):
        if c[0] != PRIMARY[0]:
            continue
        r = evaluate(rs, info, d, i)
        rows.append({"config": i, "params": ",".join(map(str, c[1:])), "eff_primary": r["eff_primary"][0],
                     "fake_rate_hnl": r["fake_rate_hnl"][0]})
    g = pd.DataFrame(rows)
    ok = g[g.fake_rate_hnl <= fr_star + 1e-12]
    best = ok.sort_values(["eff_primary", "fake_rate_hnl"], ascending=[False, True]).iloc[0] if len(ok) else None
    return (int(best.config) if best is not None else -1), g


def half(reps: np.ndarray) -> float:
    lo, hi = np.nanpercentile(reps, BT.PCT)
    return float((hi - lo) / 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="secondary_upweight_v1")
    a = ap.parse_args(argv)
    fr_star = float(json.load(open(os.path.join(BT.REPO, "results", "working_points_v1.json")))["FR_star"])
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    info = BT.info_table(BT.expected_events(BT.VAL_SETS))
    rows, res, grids = [], {}, []
    for graph, sub in GRAPHS.items():
        for run, rd in RUNS.items():
            here = sub.replace("ltf_union_preview", UNION_DIR[run]) if graph == "union" and run in UNION_DIR else sub
            d = os.path.join(rd, here)
            cfgs = _cfgs(d)
            ri, g = rule_index(rs, info, d, cfgs, fr_star)
            g.insert(0, "run", run)
            g.insert(0, "graph", graph)
            grids.append(g)
            for wp, i in (("MWS+M s0=0.2", _index(cfgs, PRIMARY)), ("MWS+M at FR*", ri)):
                if i < 0:
                    continue
                r = evaluate(rs, info, d, i)
                res[(graph, wp, run)] = r
                row = {"graph": graph, "wp": wp, "run": run, "params": ",".join(map(str, cfgs[i][1:]))}
                for q, (p, reps, k, n) in r.items():
                    row[q] = p
                    row[f"{q}_err"] = half(reps)
                    row[f"{q}_k"], row[f"{q}_n"] = k, n
                rows.append(row)
    vals = pd.DataFrame(rows)
    drows = []
    for graph in GRAPHS:
        for wp in ("MWS+M s0=0.2", "MWS+M at FR*"):
            for new, base in DIFFS:
                if (graph, wp, new) not in res or (graph, wp, base) not in res:
                    continue
                row = {"graph": graph, "wp": wp, "diff": f"{new} - {base}"}
                for q in QS:
                    pn, rn, _, _ = res[(graph, wp, new)][q]
                    pb, rb, _, _ = res[(graph, wp, base)][q]
                    row[q] = pn - pb
                    row[f"{q}_err"] = half(rn - rb)
                drows.append(row)
    diffs = pd.DataFrame(drows)
    os.makedirs(a.out, exist_ok=True)
    vals.to_csv(os.path.join(a.out, f"{a.name}.csv"), index=False)
    diffs.to_csv(os.path.join(a.out, f"{a.name}_differences.csv"), index=False)
    pd.concat(grids).to_csv(os.path.join(a.out, f"{a.name}_mwsm_grids.csv"), index=False)
    md = markdown(vals, diffs, a.name, fr_star) + curves_md(pd.concat(grids)) + extras(a.out, a.name)
    open(os.path.join(a.out, f"{a.name}.md"), "w").write(md)
    print(md)


def curves_md(g: pd.DataFrame) -> str:
    """Primary efficiency and HNL fake rate along the MWS+M threshold s0, per run (to compare at equal fake rate)."""
    L = ["## Primary efficiency / HNL fake rate [%] along the MWS+M grid", ""]
    for graph in ("base_ext", "union"):
        h = g[g.graph == graph]
        L += [f"{graph} graphs:", "", "| s0 | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
        for prm in sorted(h.params.unique(), key=float):
            cells = []
            for run in RUNS:
                r = h[(h.run == run) & (h.params == prm)]
                cells.append(f"{100 * r.eff_primary.iloc[0]:.2f} / {100 * r.fake_rate_hnl.iloc[0]:.2f}" if len(r) else "")
            L.append(f"| {prm} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def extras(out: str, name: str) -> str:
    """Classifier-stage losses versus |d0| / r_first (hnl.eval.oblique_val outputs) and the edge TPR at 0.0122."""
    L = ["## Classifier-stage losses of the primary HNL daughters (hnl.eval.oblique_val; s0 = 0.2, or each run's edge "
         "cut at the edge FPR 0.0122, hnl.eval.oblique_at_cut)", "",
         "| graphs | cut | run | ratio < 0.7 | ratio >= 0.7 | ratio >= 0.7 and |d0| >= 100 mm | graph-stage losses |",
         "|---|---|---|---|---|---|---|"]
    for graph, suf in OBLIQUE:
        f = os.path.join(out, f"{name}_oblique_{graph}{suf}_particles.csv")
        if not os.path.exists(f):
            L.append(f"| {graph} | {suf} | (missing {f}) | | | | |")
            continue
        p = pd.read_csv(f)
        for run in RUNS:
            g = p[p.label == run]
            ok = g[g.stage != "graph"]

            def cl(m):
                h = ok[m]
                k = int((h.stage == "classifier").sum())
                return f"{k} of {len(h)} ({100 * k / len(h):.1f} %)" if len(h) else ""
            cut = f"{g.cut.iloc[0]:.2f}" if "cut" in g and len(g) else "0.20"
            L.append(f"| {graph} | {cut} | {run} | {cl(ok.ratio < 0.7)} | {cl(ok.ratio >= 0.7)} | "
                     f"{cl((ok.ratio >= 0.7) & (ok.d0.abs() >= 100))} | {int((g.stage == 'graph').sum())} of {len(g)} |")
    L += ["", "## Edge TPR of HNL daughters at the edge FPR 0.0122 (base graphs; hnl.eval.edge_tpr)", "",
          "| run | cut | TPR 0.1-1 mm | TPR 100-300 mm | TPR >= 300 mm | dD [points] |", "|---|---|---|---|---|---|"]
    for run, rd in RUNS.items():
        jf, cf = os.path.join(rd, "edge_tpr_estar_efpr.json"), os.path.join(rd, "edge_tpr_estar.csv")
        if not (os.path.exists(jf) and os.path.exists(cf)):
            L.append(f"| {run} | (missing) | | | | |")
            continue
        j = json.load(open(jf))
        c = pd.read_csv(cf)
        c = c[(c.population == "hnl_daughters") & (c.variable == "d0_bin")]
        col = [x for x in c.columns if x.startswith("tpr_in_graph@")][0]

        def t(b):
            r = c[c.bin == b]
            return f"{100 * float(r[col].iloc[0]):.1f}" if len(r) else ""
        L.append(f"| {run} | {j['cut']} | {t('[0.1, 1.0)')} | {t('[100.0, 300.0)')} | {t('[300.0, inf)')} | "
                 f"{100 * j['dD']:+.1f} |")
    return "\n".join(L) + "\n"


def _pct(x, e=None, d=2):
    if pd.isna(x):
        return ""
    return f"{100 * x:.{d}f}" + (f" ± {100 * e:.{d}f}" if e is not None and not pd.isna(e) else "")


def markdown(vals: pd.DataFrame, diffs: pd.DataFrame, name: str, fr_star: float) -> str:
    L = [f"# Secondary-particle upweighting test, val read-out ({name})", "",
         "docs/secondary_upweight_plan.md. Percent; uncertainties: val event bootstrap (B = 1000), 68 % half width, "
         f"no seed term (one seed per run). FR* = {fr_star:.6f}. ref = screening-budget L1 + trunc (2026-10-03), "
         "secwctrl = the same configuration retrained now by the test's route, secw18 = displaced (|d0| >= 10 mm) "
         "true edges weighted 18 in training.", ""]
    shown = (("base", "MWS+M s0=0.2"), ("base_ext", "MWS+M at FR*"), ("union", "MWS+M s0=0.2"), ("union", "MWS+M at FR*"))
    for graph, wp in shown:
        v = vals[(vals.graph == graph) & (vals.wp == wp)]
        if v.empty:
            continue
        L += [f"## {graph} graphs, {wp}", "",
              "| run | params | primary | 100-300 mm | >= 300 mm | W muons | ttbar prompt | ttbar 1-100 mm | FR HNL | FR ttbar |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in v.iterrows():
            L.append(f"| {r.run} | {r.params} | {_pct(r.eff_primary, r.eff_primary_err)} | "
                     f"{int(r.d0_100_300_k)}/{int(r.d0_100_300_n)} ({_pct(r.d0_100_300, d=1)}) | "
                     f"{int(r.d0_300_inf_k)}/{int(r.d0_300_inf_n)} | {_pct(r.eff_w_muons, r.eff_w_muons_err)} | "
                     f"{_pct(r.eff_ttbar_prompt, r.eff_ttbar_prompt_err)} | "
                     f"{_pct(r.ttbar_d0_1_100, r.ttbar_d0_1_100_err, 1)} | {_pct(r.fake_rate_hnl, r.fake_rate_hnl_err)} | "
                     f"{_pct(r.fake_rate_ttbar, r.fake_rate_ttbar_err)} |")
        d = diffs[(diffs.graph == graph) & (diffs.wp == wp)]
        if len(d):
            L += ["", "| difference [points] | primary | 100-300 mm | >= 300 mm | W muons | ttbar prompt | ttbar 1-100 mm | FR HNL | FR ttbar |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for _, r in d.iterrows():
                L.append("| " + r["diff"] + " | " + " | ".join(
                    f"{100 * r[q]:+.2f} ± {100 * r[q + '_err']:.2f}" for q in QS) + " |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
