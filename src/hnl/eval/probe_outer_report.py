"""Read-out of the mirror / endpoint truncation probes (hnl.eval.probe_outer; supervisor feedback round 1, point 4a).

For every model directory of runs/feedback_v1/probe_outer_v1 (edges.parquet, meta.json) and every mode it computes
the edge TPR at the fixed score cut (0.35, as dT) of the measured chain-true edges, which exist at every depth
m = 0, 2, 4 (M = 4):

  inner   both hits at R rank >= M (the original probe's outer edges; reproduces RUN/truncation_probe/summary.csv)
  outer   both hits at reversed rank n - 1 - rank >= M (the inner edges left by the mirror probe)
  middle  both hits outside the depth-M window [n//2 - 2, n//2 + 2): inner piece (ends at the gap) and outer piece
          (starts after the gap), also reported separately

and dT_mode = TPR(m=4) - TPR(m=0) in points. Distance to the cut end, for the fixed edges of inner and outer: the hop
index d = r_lo - M (inner) or n - 1 - r_hi - M (outer); d = 0 is the edge adjacent to the new start (inner) or new end
(outer) at m = 4 (it is 2 hops from it at m = 2 and 4 hops at m = 0); groups d = 0, 1, 2-3, >= 4.

Uncertainties: event bootstrap (B = 2000 resamples of the 300 probe events, with replacement; the same resamples for
every model, mode and depth, so differences between modes, depths and models and seed means are paired); the
quoted sigma is the bootstrap standard deviation. Seed means (acorn (ttbar), Canto-L) average the seeds' dT with the
same resample; the seed sd is reported next to it.

    python -m hnl.eval.probe_outer_report [--base RUNS/feedback_v1/probe_outer_v1] [--out results/probe_outer_v1]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "runs", "feedback_v1",
                    "probe_outer_v1")
CUT = 0.35
M = 4
B = 2000
BOOT_SEED = 20261005
# tag: (label, group, seed, budget / graph, original probe run (inner reference), role)
MODELS = {
    "acorn_ttbar_s0": ("acorn (ttbar) s0", "acorn (ttbar)", 0, "full budget, metric-learning graph", "baseline_v1/armi_s0", "requested"),
    "acorn_ttbar_s1": ("acorn (ttbar) s1", "acorn (ttbar)", 1, "full budget, metric-learning graph", "baseline_v1/armi_s1", "requested"),
    "acorn_ttbar_s2": ("acorn (ttbar) s2", "acorn (ttbar)", 2, "full budget, metric-learning graph", None, "requested"),
    "acorn_hnl_s0": ("acorn (HNL) s0", "acorn (HNL)", 0, "full budget, metric-learning graph (HNL-trained)", "baseline_v1/armii_s0", "requested"),
    "ignn_scr": ("equal-budget IGNN", "equal-budget IGNN", 0, "screening budget, acorn (ttbar) s0 graph", "stage1_screen_ttbar/ignn/armi_s0", "requested"),
    "l1_scr": ("L1 without truncation", "L1 without truncation", 0, "screening budget, acorn (ttbar) s0 graph", "stage1_screen_ttbar/l1/armi_s0", "requested"),
    "canto_l_s0": ("Canto-L s0", "Canto-L", 0, "full budget, own learned union graph", None, "requested"),
    "canto_l_s1": ("Canto-L s1", "Canto-L", 1, "full budget, own learned union graph", None, "requested"),
    "canto_l_s2": ("Canto-L s2", "Canto-L", 2, "full budget, own learned union graph", None, "requested"),
    "ignn_trunc_scr": ("IGNN + truncation", "IGNN + truncation", 0, "screening budget, acorn (ttbar) s0 graph", "stage1_screen_ttbar/ignn_trunc/armi_s0", "extra control"),
    "l1_trunc_scr": ("L1 + truncation (Canto classifier preview)", "L1 + truncation", 0, "screening budget, acorn (ttbar) s0 graph", "stage1_screen_ttbar/symregion_trunc/armi_s0", "extra control"),
    # supervisor point 4b, added for results/probe_outer_v2 (the v1 tables were made without it)
    "ignn147_scr": ("capacity-matched IGNN (hidden 147)", "capacity-matched IGNN", 0, "screening budget, acorn (ttbar) s0 graph", None, "capacity control"),
    "ignn147_full": ("capacity-matched IGNN (hidden 147), full budget", "capacity-matched IGNN, full", 0, "full budget, acorn (ttbar) s0 graph", None, "capacity control"),
}
SEED_GROUPS = {"acorn (ttbar)": ["acorn_ttbar_s0", "acorn_ttbar_s1", "acorn_ttbar_s2"],
               "Canto-L": ["canto_l_s0", "canto_l_s1", "canto_l_s2"]}
DIST_GROUPS = [("d=0 (adjacent)", 0, 0), ("d=1", 1, 1), ("d=2-3", 2, 3), ("d>=4", 4, 10 ** 6)]


def measured(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks of the measured edges per (mode, subset) on the rows of one depth."""
    rr_lo = df.n_hits - 1 - df.r_hi                        # reversed rank of the edge's outer hit
    c = df.n_hits // 2
    out = {("inner", "all"): df.r_lo >= M, ("outer", "all"): rr_lo >= M}
    mid_in, mid_out = df.r_hi < c - M // 2, df.r_lo >= c - M // 2 + M
    out[("middle", "all")] = mid_in | mid_out
    out[("middle", "inner piece")] = mid_in
    out[("middle", "outer piece")] = mid_out
    for name, lo, hi in DIST_GROUPS:
        d_in, d_out = df.r_lo - M, rr_lo - M
        out[("inner", name)] = (df.r_lo >= M) & (d_in >= lo) & (d_in <= hi)
        out[("outer", name)] = (rr_lo >= M) & (d_out >= lo) & (d_out <= hi)
    return out


def row(df: pd.DataFrame, model: str, mode: str, subset: str) -> pd.Series:
    return df[(df.model == model) & (df["mode"] == mode) & (df.subset == subset)].iloc[0]


def classify(df: pd.DataFrame, model: str) -> str:
    """Verdict for one model (rule fixed before any number was read): significance = the 95 % event-bootstrap
    percentile interval excludes 0. Inner drop significant and larger than the outer change (inner - outer < 0
    significant): start radius, with an endpoint part if the outer drop is significant too; inner and outer drops
    both significant and not significantly different: endpoint."""
    i, o = row(df, model, "inner", "all"), row(df, model, "outer", "all")
    d = row(df, model, "difference", "inner - outer")
    # size qualifier (added for reading, the rule itself is unchanged): both changes below 1 point
    small = "both below 1 point; " if abs(i.dT_pts) < 1.0 and abs(o.dT_pts) < 1.0 else ""
    if i.dT_boot_hi95 >= 0:
        return small + "no significant inner drop"
    o_drop, d_sig = o.dT_boot_hi95 < 0, d.dT_boot_hi95 < 0
    if d_sig and not o_drop:
        return small + "start radius (no significant outer drop)"
    if d_sig and o_drop:
        return small + f"start radius with an endpoint part (outer drop {o.dT_pts / i.dT_pts:.0%} of the inner drop)"
    if o_drop:
        return small + "endpoint (outer drop not significantly smaller than the inner drop)"
    return small + "inconclusive (outer drop not significant, difference not significant)"


def _pm(r, key="dT_pts", sd="dT_boot_sd", digits=1) -> str:
    return f"{r[key]:+.{digits}f} +- {r[sd]:.{digits}f}"


NO_AUG = ["acorn_ttbar_s0", "acorn_ttbar_s1", "acorn_ttbar_s2", "ignn_scr", "l1_scr"]
AUG = ["canto_l_s0", "canto_l_s1", "canto_l_s2", "ignn_trunc_scr", "l1_trunc_scr"]


def _rng(vals, fmt="{:+.1f}") -> str:
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo} to {hi}"


def verdict_text(df: pd.DataFrame) -> list[str]:
    """The verdict paragraphs; every number is read from df, the wording follows classify()."""
    have = set(df.model)
    na = [m for m in NO_AUG if m in have]
    out = []
    if not na:
        return ["(no ttbar-trained classifier without truncation augmentation in the read-out)"]
    I = {m: row(df, m, "inner", "all") for m in na}
    O = {m: row(df, m, "outer", "all") for m in na}
    D = {m: row(df, m, "difference", "inner - outer") for m in na}
    V = {m: classify(df, m) for m in na}
    names = ", ".join(MODELS[m][0] for m in na)
    zmin = min(-D[m].dT_pts / D[m].dT_boot_sd for m in na)
    out.append(
        f"**Classifiers trained on ttbar without truncation augmentation** ({names}): removing the 4 innermost hits "
        f"lowers the TPR of the remaining outer edges by dT_inner = {_rng([I[m].dT_pts for m in na])} points "
        f"(event-bootstrap sd {_rng([I[m].dT_boot_sd for m in na], '{:.1f}')}); removing the 4 outermost hits of the "
        f"same particles changes the TPR of the remaining inner edges by dT_outer = "
        f"{_rng([O[m].dT_pts for m in na], '{:+.2f}')} points (sd {_rng([O[m].dT_boot_sd for m in na], '{:.2f}')}). "
        f"The paired difference dT_inner - dT_outer is {_rng([D[m].dT_pts for m in na])} points, at least "
        f"{zmin:.0f} sd from zero for every model. The cut-independent mean score of the same edges moves by "
        f"{_rng([I[m].dscore for m in na], '{:+.3f}')} (inner) against {_rng([O[m].dscore for m in na], '{:+.4f}')} "
        f"(outer).")
    means = [m for m in ("acorn_(ttbar)_mean",) if m in have]
    if all(V[m].startswith("start radius") for m in na):
        sig_o = [MODELS[m][0] for m in na if O[m].dT_boot_hi95 < 0] + \
                [f"the acorn (ttbar) seed mean ({_pm(row(df, m, 'outer', 'all'), digits=2)})" for m in means
                 if row(df, m, "outer", "all").dT_boot_hi95 < 0]
        frac = max(O[m].dT_pts / I[m].dT_pts for m in na)
        txt = ("**Verdict: start radius, not endpoints.** For every one of these classifiers the inner drop is "
               "significant and significantly larger than the outer change (95 % event-bootstrap intervals). A new "
               f"track end at mid radius moves the TPR by at most {max(abs(O[m].dT_pts) for m in na):.2f} points "
               f"(at most {max(frac, 0):.0%} of the same model's inner drop); a late start lowers it by "
               f"{min(-I[m].dT_pts for m in na):.1f} to {max(-I[m].dT_pts for m in na):.1f} points.")
        if sig_o:
            names_o = sig_o[0] if len(sig_o) == 1 else ", ".join(sig_o[:-1]) + " and " + sig_o[-1]
            txt += (f" The outer change is statistically significant for {names_o}, i.e. an endpoint part of that "
                    "size exists; for the others it is consistent with zero.")
        else:
            txt += " The outer change is consistent with zero for every one of them."
        txt += (" The scores depend on where a track starts (a learned start-radius prior), not on where it ends: the "
                "network did not learn a generic distrust of edges near a track endpoint.")
        out.append(txt)
    else:
        out.append("**Verdict per model** (verdict rule): " + "; ".join(f"{MODELS[m][0]}: {V[m]}" for m in na)
                   + ".")
    # middle gap
    mo = {m: row(df, m, "middle", "outer piece") for m in na}
    mi = {m: row(df, m, "middle", "inner piece") for m in na}
    txt = (f"**Middle gap.** Removing 4 hits in the middle changes the TPR of the outer piece, which now starts after "
           f"the gap, by {_rng([mo[m].dT_pts for m in na])} points (sd {_rng([mo[m].dT_boot_sd for m in na], '{:.1f}')}"
           f"), and that of the inner piece, which now ends at the gap, by "
           f"{_rng([mi[m].dT_pts for m in na], '{:+.2f}')} points (sd {_rng([mi[m].dT_boot_sd for m in na], '{:.2f}')}).")
    if all(mo[m].dT_boot_hi95 < 0 for m in na) and all(abs(mi[m].dT_pts) < 0.25 * abs(mo[m].dT_pts) for m in na):
        txt += (" The piece that starts late loses; the piece that ends early loses at most "
                f"{max(abs(mi[m].dT_pts) / abs(mo[m].dT_pts) for m in na):.0%} of that: the same reading.")
    out.append(txt)
    # distance to the new start / end
    a0 = {m: row(df, m, "inner", "d=0 (adjacent)") for m in na}
    a4 = {m: row(df, m, "inner", "d>=4") for m in na}
    o0 = {m: row(df, m, "outer", "d=0 (adjacent)") for m in na}
    txt = (f"**Adjacent against far edges.** Inner probe: the edge adjacent to the new start at m = 4 (d = 0) changes "
           f"by {_rng([a0[m].dT_pts for m in na])} points, edges at least 4 hops from it (d >= 4) by "
           f"{_rng([a4[m].dT_pts for m in na])} points. Outer probe: the edge adjacent to the new end changes by "
           f"{_rng([o0[m].dT_pts for m in na], '{:+.2f}')} points")
    txt += (f" ({_pm(row(df, means[0], 'outer', 'd=0 (adjacent)'), digits=2)} for the acorn (ttbar) seed mean)."
            if means else ".")
    if all(a4[m].dT_boot_hi95 < 0 for m in na):
        txt += (" The start effect is not confined to the edges next to the new start: edges at least 4 hops away "
                "lose too, so a late start lowers the scores along the remaining track.")
    ratio = max(abs(o0[m].dT_pts) / abs(a0[m].dT_pts) for m in na)
    txt += (f" A local endpoint effect, if any, is at most {max(abs(o0[m].dT_pts) for m in na):.2f} points on the "
            f"edge next to the new end, at most {ratio:.0%} of the change of the edge next to the new start.")
    out.append(txt)
    if "acorn_hnl_s0" in have:
        h, ho = row(df, "acorn_hnl_s0", "inner", "all"), row(df, "acorn_hnl_s0", "outer", "all")
        big = min(-I[m].dT_pts for m in na)
        txt = (f"**acorn (HNL) s0** (the same architecture trained on the HNL signal): dT_inner = {_pm(h, digits=2)}, "
               f"dT_outer = {_pm(ho, digits=2)} points. Both changes are below "
               f"{max(abs(h.dT_pts), abs(ho.dT_pts)) + 0.005:.2f} points")
        if abs(h.dT_pts) > 0:
            txt += f", the inner one {big / abs(h.dT_pts):.0f} times smaller than the smallest ttbar-trained inner drop"
        txt += f" (the verdict rule labels it '{classify(df, 'acorn_hnl_s0')}', on effects of this size)."
        out.append(txt)
    ag = [m for m in AUG if m in have]
    if ag:
        out.append("**Truncation-augmented classifiers** (" + ", ".join(MODELS[m][0] for m in ag) + "): dT_inner = "
                   + _rng([row(df, m, "inner", "all").dT_pts for m in ag], "{:+.2f}") + " points, dT_outer = "
                   + _rng([row(df, m, "outer", "all").dT_pts for m in ag], "{:+.2f}") + " points (event-bootstrap "
                   "sd at most " + f"{max(max(row(df, m, 'inner', 'all').dT_boot_sd, row(df, m, 'outer', 'all').dT_boot_sd) for m in ag):.2f}"
                   + ")." + (" The augmentation, which only removes inner hits in training, removes the start-radius "
                             "sensitivity; the response to a new track end stays below 0.5 points, as without it."
                             if all(abs(row(df, m, "inner", "all").dT_pts) < 1.0 and
                                    abs(row(df, m, "outer", "all").dT_pts) < 0.5 for m in ag) else ""))
        if "Canto-L_mean" in have:
            ci, co = row(df, "Canto-L_mean", "inner", "all"), row(df, "Canto-L_mean", "outer", "all")
            cd = row(df, "Canto-L_mean", "difference", "inner - outer")
            if abs(ci.dT_pts) < 1 and abs(co.dT_pts) < 1 and cd.dT_boot_lo95 <= 0 <= cd.dT_boot_hi95:
                out.append(f"For Canto-L the two changes are equal within the uncertainties (seed mean dT_inner "
                           f"{_pm(ci, digits=2)}, dT_outer {_pm(co, digits=2)}, difference {_pm(cd, digits=2)} points; "
                           f"the verdict rule therefore reads '{classify(df, 'Canto-L_mean')}'): what is left "
                           f"after the augmentation is a change of about {max(abs(ci.dT_pts), abs(co.dT_pts)):.1f} "
                           f"point for a new start or a new end, {min(-row(df, m, 'inner', 'all').dT_pts for m in na) / max(abs(ci.dT_pts), 1e-9):.0f} "
                           "or more times smaller than the start-radius drop of the classifiers without augmentation. "
                           "It sits on the edge next to the new endpoint (d = 0, seed mean: inner "
                           f"{_pm(row(df, 'Canto-L_mean', 'inner', 'd=0 (adjacent)'), digits=2)}, outer "
                           f"{_pm(row(df, 'Canto-L_mean', 'outer', 'd=0 (adjacent)'), digits=2)} points; d >= 4: inner "
                           f"{_pm(row(df, 'Canto-L_mean', 'inner', 'd>=4'), digits=2)}, outer "
                           f"{_pm(row(df, 'Canto-L_mean', 'outer', 'd>=4'), digits=2)}): a small local endpoint "
                           "effect, the same at both ends.")
        gi = [row(df, m, "middle", "inner piece") for m in ag]
        go = [row(df, m, "middle", "outer piece") for m in ag]
        worst = min(ag, key=lambda m: row(df, m, "middle", "all").dT_pts)
        out.append("A 4-hit gap in the middle is a different edit: for these classifiers the outer piece changes by "
                   + _rng([r.dT_pts for r in go]) + " points and the inner piece by " + _rng([r.dT_pts for r in gi])
                   + f" points (largest change of the two pieces together: {MODELS[worst][0]}, "
                   + _pm(row(df, worst, "middle", "all")) + "), against "
                   + _rng([row(df, m, "middle", "outer piece").dT_pts for m in na]) + " points for the outer piece of "
                   "the classifiers without augmentation.")
    return out


def markdown(df: pd.DataFrame, meta: dict, n_events: int, base: str) -> str:
    """results/probe_outer_v1.md: tables of the read-out (numbers only from df)."""
    order = [m for m in MODELS if (df.model == m).any()]
    seq = []                                   # models with each seed mean right after its last seed
    for m in order:
        seq.append(m)
        for g, tags in SEED_GROUPS.items():
            mean = g.replace(" ", "_") + "_mean"
            if m == tags[-1] and (df.model == mean).any():
                seq.append(mean)
    n_part = sorted({mt["n_particles"] for mt in meta.values()})
    n_part = n_part[0] if len(n_part) == 1 else n_part
    means = [m for m in ("acorn_(ttbar)_mean", "Canto-L_mean") if (df.model == m).any()]
    L = []
    w = L.append
    w("# Mirror truncation probe: start radius or track endpoints? (supervisor feedback round 1, point 4a)\n")
    w("Validation only (300 hnl10_100 val events, the events of the original probe); a diagnostic of the edge "
      "classifiers, not a claim; nothing here is chosen on or evaluated on test. Produced by `python -m "
      "hnl.eval.probe_outer` (one job per model, `slurm/stage1/probe_outer.sbatch`) and `python -m "
      f"hnl.eval.probe_outer_report`; per-edge outputs in `{base}/<model>/edges.parquet`.\n")
    w("## Method\n")
    w(f"- Particles and edits as the truncation probe (`hnl.eval.truncation_probe`, paper Sec. IV B): prompt particles "
      f"(truth |d0| < 1 mm, >= 8 hits, pT >= 2 GeV, at most 5 per event; {n_part} particles) of {n_events} HNL val "
      "events (hnl10_100, the first 300 HNL members of the val graph tars); hits removed "
      "together with every graph edge touching them, no edge added (`hnl.eval.truncation_probe.truncate`; for the "
      "metric-learning radius graph this is the graph the graph stage would build without those hits, as its "
      "neighbour cap never binds; for the Canto-L union graphs, union edges whose triplet support used a removed hit "
      "are kept, i.e. the graph stage is not re-run and the probe tests the edge classifier), the edited graph "
      "re-scored by the model's GNN (best checkpoint of the finished training); edge TPR at the fixed score cut 0.35 "
      "of the chain-true edges whose two hits survive at every depth m = 0, 2, 4 (the same edges at every depth; an "
      "edge missing from the graph counts as lost, as in the original probe; column in_graph of the csv); "
      "dT = TPR(m=4) - TPR(m=0) in points.")
    w("- **inner** (the original probe): remove the m innermost hits (R = sqrt(r^2 + z^2)); measured: chain-true edges "
      "with both hits at R rank >= 4 (the outer edges). Reproduces the stored RUN/truncation_probe/summary.csv.")
    w("- **outer** (the mirror): remove the m outermost hits of the same particles in the same events; measured: "
      "chain-true edges with both hits at reversed rank n - 1 - rank >= 4 (the inner edges). The track still starts "
      "at the beam line and points to it; it only ends early.")
    w("- **middle**: remove m hits in the middle (R ranks [n//2 - m/2, n//2 + m/2)); measured: chain-true edges with "
      "both hits outside the m = 4 window: the inner piece (ends at the gap) and the outer piece (starts after the "
      "gap, at a larger radius).")
    w("- Distance to the cut: for the fixed edges of inner (outer), d = rank of the edge's inner hit - 4 (reversed rank "
      "of its outer hit - 4): d = 0 is the edge adjacent to the new start (new end) at m = 4, 2 hops from it at m = 2, "
      "4 hops at m = 0.")
    w(f"- Uncertainties: event bootstrap, B = {B} resamples of the {n_events} events with replacement (all edges of "
      "an event move together, so the correlation of the edges of one track is included); the same resamples for "
      "every model, mode and depth (paired differences); +- is the bootstrap standard deviation; 'sig.' = the 95 % "
      "percentile interval excludes 0. Seed means: mean of the seeds' dT, bootstrap of the mean with the same "
      "resamples; the seed sd is given separately. dT is at a fixed score cut, not at a matched false-positive rate "
      "(as in the paper).")
    w("- Verdict rule (fixed before any number was read): significance = the 95 % event-bootstrap interval excludes 0. "
      "Inner drop significant and significantly larger than the outer change (dT_inner - dT_outer < 0): 'start "
      "radius', with 'an endpoint part' if the outer drop is significant too; inner and outer drops both significant "
      "and not significantly different: 'endpoint'; inner drop not significant: 'no significant inner drop'. Labels "
      "of models whose two changes are both below 1 point carry the prefix 'both below 1 point' (added for reading; "
      "the rule is unchanged).\n")
    w("## Summary: dT_inner against dT_outer [points], +- event-bootstrap sd\n")
    w("| model | dT_inner | dT_outer | dT_inner - dT_outer | verdict |")
    w("|---|---|---|---|---|")
    for m in seq:
        i, o = row(df, m, "inner", "all"), row(df, m, "outer", "all")
        d = row(df, m, "difference", "inner - outer")
        lab = i.label + (f" (seed sd {i.dT_seed_sd:.2f} / {o.dT_seed_sd:.2f})" if m in means else "")
        w(f"| {lab} | {_pm(i, digits=2)} | {_pm(o, digits=2)} | {_pm(d, digits=2)} | {d.verdict} |")
    w("")
    w("## Verdict\n")
    for p in verdict_text(df):
        w(p + "\n")
    w("## Main table: dT_inner against dT_outer [points]\n")
    w("| model | budget, graph | edges (inner / outer) | TPR inner m=0 / 2 / 4 | dT_inner | TPR outer m=0 / 2 / 4 | "
      "dT_outer | dT_inner - dT_outer | verdict (rule in the text) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m in seq:
        i, o = row(df, m, "inner", "all"), row(df, m, "outer", "all")
        d = row(df, m, "difference", "inner - outer")
        lab = i.label + (f" (seed sd: {i.dT_seed_sd:.2f} / {o.dT_seed_sd:.2f})" if m in means else "")
        w(f"| {lab} | {i.budget_graph} | {int(i.n_edges)} / {int(o.n_edges)} | {i.tpr_m0:.3f} / {i.tpr_m2:.3f} / "
          f"{i.tpr_m4:.3f} | {_pm(i, digits=2)} | {o.tpr_m0:.3f} / {o.tpr_m2:.3f} / {o.tpr_m4:.3f} | {_pm(o, digits=2)} | "
          f"{_pm(d, digits=2)} | "
          f"{d.verdict} |")
    w("")
    w("## Middle gap [points]\n")
    w("| model | dT middle, all | inner piece (ends at the gap) | outer piece (starts after the gap) | "
      "outer piece - inner piece |")
    w("|---|---|---|---|---|")
    for m in seq:
        a_, ip, op = (row(df, m, "middle", s) for s in ("all", "inner piece", "outer piece"))
        d = row(df, m, "difference", "middle: outer piece - inner piece")
        w(f"| {a_.label} | {_pm(a_)} ({int(a_.n_edges)} edges) | {_pm(ip, digits=2)} ({int(ip.n_edges)}) | {_pm(op)} "
          f"({int(op.n_edges)}) | {_pm(d)} |")
    w("")
    w("## Where along the track: TPR change m=0 -> 4 by distance d to the new start (inner) or new end (outer) "
      "[points]\n")
    w("| model | probe | " + " | ".join(n for n, _, _ in DIST_GROUPS) + " | adjacent - far (d>=4) |")
    w("|---|---|" + "---|" * (len(DIST_GROUPS) + 1))
    for m in seq:
        for mode in ("inner", "outer"):
            cells = []
            for n, _, _ in DIST_GROUPS:
                r = row(df, m, mode, n)
                cells.append(f"{_pm(r, digits=1 if mode == 'inner' else 2)} ({int(r.n_edges)})")
            d = row(df, m, "difference", f"{mode}: adjacent - far (d>=4)")
            w(f"| {row(df, m, mode, 'all').label} | {mode} | " + " | ".join(cells)
              + f" | {_pm(d, digits=1 if mode == 'inner' else 2)} |")
    w("")
    w("## Reproduction of the stored inner probe\n")
    w("| model | stored TPR m=0 / 2 / 4 (RUN/truncation_probe) | this run | max abs difference |")
    w("|---|---|---|---|")
    for m in order:
        rp = meta.get(m, {}).get("repro")
        if rp:
            w(f"| {MODELS[m][0]} | {' / '.join(f'{x:.4f}' for x in rp['orig_tpr'])} ({rp['orig_n']} edges) | "
              f"{' / '.join(f'{x:.4f}' for x in rp['new_tpr'])} ({rp['new_n']} edges) | {rp['max_abs_diff']:.1e} |")
    w("")
    w("## Files\n")
    w("- `results/probe_outer_v1.csv`: one row per model (seed means: `<group>_mean`), mode (inner, outer, middle, "
      "difference) and subset (all; distance groups d; middle pieces; paired differences): n_edges, in_graph, TPR at "
      "m = 0 / 2 / 4, dT_pts (m=4 - m=0) and dT_m2_pts (m=2 - m=0) in points, dT_boot_sd and the 95 % percentile "
      "interval (dT_boot_lo95, dT_boot_hi95), the mean score of the same in-graph edges at m = 0 / 2 / 4 and its "
      "change (dscore, dscore_boot_sd), dT_seed_sd for seed means, and the verdict of `classify` on the "
      "'inner - outer' rows.")
    w("- `results/probe_outer_v1_meta.json`: per-model run metadata (checkpoint, graphs, job, host, timings) and the "
      "reproduction check against RUN/truncation_probe/summary.csv.")
    w(f"- Per-edge outputs: `{base}/<model>/edges.parquet` (columns mode, depth, event_id, particle_id, r_lo, r_hi, "
      "n_hits, in_graph, score; every chain-true edge of the chosen particles that survives the edit), "
      f"`{base}/events.json` (the 300 event ids).\n")
    w("## Runs\n")
    w("| model | run | graphs | checkpoint | job | host | elapsed |")
    w("|---|---|---|---|---|---|---|")
    for m in order:
        mt = meta[m]
        w(f"| {MODELS[m][0]} | {mt['run'].split('/runs/')[-1]} | {mt['graphs'].split('/runs/')[-1]} | "
          f"{os.path.basename(mt['ckpt'])} | {mt.get('slurm_job')} | {mt.get('host')} | {mt.get('elapsed_s')} s |")
    w("")
    for m in order:
        if meta[m].get("merged"):
            w(f"- {MODELS[m][0]}: {meta[m]['merged']}.")
    w("- CPU only (8-16 cores per job, no GPU); inference time per depth 90-1,500 s depending on the model and node.")
    w("")
    return "\n".join(L) + "\n"


def per_event_counts(e: pd.DataFrame, events: np.ndarray) -> dict:
    """{(mode, subset): (n_e [events], {depth: kept_e [events]}, in_graph count, n)} on the fixed event list."""
    pos = pd.Series(np.arange(len(events)), index=events)
    res = {}
    base = e[e["mode"] == "none"]
    for mode in ("inner", "outer", "middle"):
        rows = {0: base, **{k: e[(e["mode"] == mode) & (e.depth == k)] for k in (2, 4)}}
        masks = {k: measured(d) for k, d in rows.items()}
        for key in [k for k in masks[0] if k[0] == mode]:
            n_e, kept, keys0, ssum = None, {}, None, {}
            for k, d in rows.items():
                sel = d[masks[k][key].to_numpy()]
                keys = sel[["event_id", "particle_id", "r_lo", "r_hi"]].sort_values(
                    ["event_id", "particle_id", "r_lo", "r_hi"]).to_numpy()
                if keys0 is None:
                    keys0 = keys
                elif not np.array_equal(keys, keys0):
                    raise RuntimeError(f"measured edges differ between depths: {mode} {key} depth {k}")
                idx = pos.reindex(sel.event_id.to_numpy()).to_numpy()
                if np.isnan(idx.astype(float)).any():
                    raise RuntimeError("edge of an event outside the event list")
                idx = idx.astype(int)
                sc = sel.score.to_numpy(dtype=float)
                if n_e is None:
                    n_e = np.bincount(idx, minlength=len(events)).astype(float)
                    ing = int(sel.in_graph.sum())
                    g_e = np.bincount(idx, weights=(~np.isnan(sc)).astype(float), minlength=len(events))
                kept[k] = np.bincount(idx, weights=(sc > CUT).astype(float), minlength=len(events))
                ssum[k] = np.bincount(idx, weights=np.nan_to_num(sc), minlength=len(events))
            res[key] = (n_e, kept, ing, int(n_e.sum()), g_e, ssum)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="results/probe_outer_v1")
    a = ap.parse_args(argv)
    events = np.array(json.load(open(os.path.join(a.base, "events.json")))["events"], dtype=np.int64)
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(len(events), np.full(len(events), 1.0 / len(events)), size=B).astype(float)  # (B, events)
    rows, boot, meta = [], {}, {}
    for tag, (label, group, seed, budget, orig, role) in MODELS.items():
        f = os.path.join(a.base, tag, "edges.parquet")
        if not os.path.exists(f):
            print(f"missing {tag}")
            continue
        meta[tag] = json.load(open(os.path.join(a.base, tag, "meta.json")))
        e = pd.read_parquet(f)
        counts = per_event_counts(e, events)
        for (mode, subset), (n_e, kept, ing, n, g_e, ssum) in counts.items():
            tpr = {k: kept[k].sum() / n_e.sum() for k in kept}
            den = W @ n_e
            tb = {k: (W @ kept[k]) / den for k in kept}
            dtb = 100 * (tb[4] - tb[0])
            boot[(tag, mode, subset)] = dtb
            # cut-independent companion: mean score of the same in-graph edges, change m=0 -> 4 (paired)
            ms = {k: ssum[k].sum() / g_e.sum() for k in ssum}
            dsb = (W @ (ssum[4] - ssum[0])) / (W @ g_e)
            rows.append({"model": tag, "label": label, "group": group, "seed": seed, "budget_graph": budget,
                         "role": role, "mode": mode, "subset": subset, "n_edges": n, "in_graph": ing / n,
                         "tpr_m0": tpr[0], "tpr_m2": tpr[2], "tpr_m4": tpr[4],
                         "dT_pts": 100 * (tpr[4] - tpr[0]), "dT_m2_pts": 100 * (tpr[2] - tpr[0]),
                         "dT_boot_sd": dtb.std(ddof=1), "dT_boot_lo95": np.percentile(dtb, 2.5),
                         "dT_boot_hi95": np.percentile(dtb, 97.5),
                         "mean_score_m0": ms[0], "mean_score_m2": ms[2], "mean_score_m4": ms[4],
                         "dscore": ms[4] - ms[0], "dscore_boot_sd": dsb.std(ddof=1)})
        # paired differences between modes
        for m1, s1, m2, s2, name in [("inner", "all", "outer", "all", "inner - outer"),
                                     ("middle", "outer piece", "middle", "inner piece", "middle: outer piece - inner piece"),
                                     ("inner", "d=0 (adjacent)", "inner", "d>=4", "inner: adjacent - far (d>=4)"),
                                     ("outer", "d=0 (adjacent)", "outer", "d>=4", "outer: adjacent - far (d>=4)")]:
            r1 = next(r for r in rows if r["model"] == tag and r["mode"] == m1 and r["subset"] == s1)
            r2 = next(r for r in rows if r["model"] == tag and r["mode"] == m2 and r["subset"] == s2)
            db = boot[(tag, m1, s1)] - boot[(tag, m2, s2)]
            boot[(tag, "diff", name)] = db
            rows.append({"model": tag, "label": label, "group": group, "seed": seed, "budget_graph": budget,
                         "role": role, "mode": "difference", "subset": name, "n_edges": np.nan, "in_graph": np.nan,
                         "tpr_m0": np.nan, "tpr_m2": np.nan, "tpr_m4": np.nan,
                         "dT_pts": r1["dT_pts"] - r2["dT_pts"], "dT_m2_pts": r1["dT_m2_pts"] - r2["dT_m2_pts"],
                         "dT_boot_sd": db.std(ddof=1), "dT_boot_lo95": np.percentile(db, 2.5),
                         "dT_boot_hi95": np.percentile(db, 97.5)})
        # reproduction of the original inner probe
        if orig:
            runs = os.path.dirname(os.path.dirname(os.path.abspath(a.base)))      # RUNS/feedback_v1/<probe> -> RUNS
            s = pd.read_csv(os.path.join(runs, orig, "truncation_probe", "summary.csv")).set_index("depth")
            r = next(r for r in rows if r["model"] == tag and r["mode"] == "inner" and r["subset"] == "all")
            meta[tag]["repro"] = {"orig_tpr": [float(s.tpr[k]) for k in (0, 2, 4)],
                                  "new_tpr": [r["tpr_m0"], r["tpr_m2"], r["tpr_m4"]],
                                  "orig_n": int(s.n_edges[0]), "new_n": int(r["n_edges"]),
                                  "max_abs_diff": float(max(abs(s.tpr[k] - r[f"tpr_m{k}"]) for k in (0, 2, 4)))}
    # seed means (same resamples)
    for group, tags in SEED_GROUPS.items():
        if not all(any(r["model"] == t for r in rows) for t in tags):
            continue
        keys = sorted({(r["mode"], r["subset"]) for r in rows if r["model"] == tags[0]})
        for mode, subset in keys:
            rs = [next(r for r in rows if r["model"] == t and r["mode"] == mode and r["subset"] == subset) for t in tags]
            bk = (lambda t: (t, "diff", subset)) if mode == "difference" else (lambda t: (t, mode, subset))
            mb = np.mean([boot[bk(t)] for t in tags], axis=0)
            vals = np.array([r["dT_pts"] for r in rs])
            row = {"model": group.replace(" ", "_") + "_mean", "label": f"{group}, mean of seeds 0-2", "group": group,
                   "seed": -1, "budget_graph": rs[0]["budget_graph"], "role": "seed mean", "mode": mode,
                   "subset": subset, "n_edges": np.nan if mode == "difference" else rs[0]["n_edges"],
                   "in_graph": np.nan, "dT_pts": vals.mean(), "dT_m2_pts": np.mean([r["dT_m2_pts"] for r in rs]),
                   "dT_seed_sd": vals.std(ddof=1), "dT_boot_sd": mb.std(ddof=1),
                   "dT_boot_lo95": np.percentile(mb, 2.5), "dT_boot_hi95": np.percentile(mb, 97.5)}
            for k in (0, 2, 4):
                row[f"tpr_m{k}"] = np.nan if mode == "difference" else np.mean([r[f"tpr_m{k}"] for r in rs])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["verdict"] = ""
    for i, r in df[(df["mode"] == "difference") & (df.subset == "inner - outer")].iterrows():
        df.loc[i, "verdict"] = classify(df, r["model"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out + ".csv", index=False, float_format="%.6g")
    with open(a.out + "_meta.json", "w") as f:
        json.dump({"cut": CUT, "M": M, "B": B, "boot_seed": BOOT_SEED, "n_events": len(events), "models": meta},
                  f, indent=1, default=str)
    with open(a.out + ".md", "w") as f:
        f.write(markdown(df, meta, len(events), a.base))
    pd.set_option("display.width", 250)
    show = df[df.subset.isin(["all", "inner - outer"])][["model", "mode", "subset", "n_edges", "tpr_m0", "tpr_m2",
                                                          "tpr_m4", "dT_pts", "dT_boot_sd"]]
    print(show.round(4).to_string())
    for t, m in meta.items():
        if "repro" in m:
            print(t, m["repro"])


if __name__ == "__main__":
    main()
