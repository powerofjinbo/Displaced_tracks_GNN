"""Two "model release" style cards of Canto-L on TrackML, one per data phase, each against the published numbers
of that phase. Same look as hnl.plots.release_card (colours, fonts, geometry).

Throughput card (one-time test, 300 throughput-phase events). Every row is one published metric computed with
that reference's own evaluation code: Canto-L's values come from the post-freeze recount of the frozen pipeline's
test tracks (hnl.trackml.refmetrics: $RUN/test_post/refmetrics/test_<cond>/refmetrics.csv, ports of the EggNet and
HierarchicalGNN code), the published values from hnl.trackml.evaluate.REFERENCES (no transcription). Only references
that used the throughput phase, as our test does, are shown (DECISIONS 2026-10-11). A model the paper does not
report gets a dash. Shading: the first place of each row only (the highest value, the lowest for rates; equal
shown values share it), bold and shaded, peach in the highlighted column, grey elsewhere.

Accuracy card (not a release card: Canto-L was not trained on that phase; kept for the record, rendered only on request
with --out-accuracy; one-time T2, 300 accuracy-phase events of Kaggle train_1, the same frozen pipeline):
the only same-phase study, Ju et al. (P1), under variant J' (our port of their evaluation script, as in the test
reports; J' at pT > 0.1 GeV recomputed here from the stored per-event counts with the reports' bootstrap), and
Canto-L's own definitions in the three conditions. A condition whose T2 report is not written yet shows "pending".

    python -m hnl.plots.release_card_trackml --out figures/release/canto_l_trackml_card
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from hnl.plots.release_card import BG, FULL, GREY, INK, MUTED, ORANGE, PEACH
from hnl.trackml.evaluate import REFERENCES as REF

RUN = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1"
COLUMNS = [  # (header, header note)
    ("Canto-L", "1 seed · TEST"),
    ("EggNet", "Calafiura 2024 · Chan 2025"),
    ("EC", "Calafiura et al. 2024"),
    ("BC-HGNN", "Liu et al. 2026"),
    ("EC-GNN", "Liu et al. 2026"),
]
P2, P3, P4 = REF["P2"], REF["P3"], REF["P4"]
# (title, subtitle lines, condition, Canto-L metric of the post-freeze recount, best = max/min,
#  {column: (published value, quoted uncertainty or None)})
ROWS = [
    ("Track efficiency", ("TrackML-1GeV · EggNet code",), "1gev", "eggnet_efficiency", "max",
     {"EggNet": P4["EggNet"]["efficiency"], "EC": P4["EC"]["efficiency"]}),
    ("Fake-track rate", ("TrackML-1GeV · EggNet code", "lower is better"), "1gev", "eggnet_fake", "min",
     {"EggNet": P4["EggNet"]["fake"], "EC": P4["EC"]["fake"]}),
    ("Duplicate rate", ("TrackML-1GeV · EggNet code", "lower is better"), "1gev", "eggnet_duplication", "min",
     {"EggNet": P4["EggNet"]["duplication"], "EC": P4["EC"]["duplication"]}),
    ("Track efficiency", ("TrackML-1GeV · Liu et al. code",), "1gev", "liu_eff", "max",
     {"BC-HGNN": (P2["1gev"]["BC-HGNN"][0], None)}),
    ("Fake rate", ("TrackML-1GeV · Liu et al. code", "lower is better"), "1gev", "liu_fr", "min",
     {"BC-HGNN": (P2["1gev"]["BC-HGNN"][1], None)}),
    ("Track efficiency", ("full TrackML · EggNet code",), "full", "eggnet_efficiency", "max",
     {"EggNet": P3["efficiency"]}),
    ("Fake-track rate", ("full TrackML · EggNet code", "lower is better"), "full", "eggnet_fake", "min",
     {"EggNet": P3["fake"]}),
    ("Duplicate rate", ("full TrackML · EggNet code", "lower is better"), "full", "eggnet_duplication", "min",
     {"EggNet": P3["duplication"]}),
    ("Track efficiency", ("full TrackML · Liu et al. code",), "full", "liu_eff", "max",
     {"BC-HGNN": (P2["full"]["BC-HGNN"][0], None), "EC-GNN": (P2["full"]["EC-GNN"][0], None)}),
    ("Fake rate", ("full TrackML · Liu et al. code", "lower is better"), "full", "liu_fr", "min",
     {"BC-HGNN": (P2["full"]["BC-HGNN"][1], None), "EC-GNN": (P2["full"]["EC-GNN"][1], None)}),
]
TITLE = "Canto-L · TrackML throughput phase"
SUBTITLE = ("Track reconstruction at pileup 200 · frozen pipeline, evaluated once on 300 test events · "
            "each row under the reference's own metric code (ported, checked against the original)")

COLUMNS_ACC = [("Canto-L", "1 seed · T2 · not retrained"), ("Exa.TrkX", "Ju et al. 2021")]
P1 = REF["P1"]
# (title, subtitle lines, Canto-L value source, best = max/min, {column: (published value, quoted uncertainty)}, unit);
# source ("jp", numerator, denominator): variant J' from the T2 C-nonoise per-event counts; ("rep", condition, metric):
# the T2 report
ROWS_ACC = [
    ("Technical efficiency", ("no noise · pT > 0.5 GeV, ≥ 5 hits", "Ju et al. script (J′)"),
     ("jp", "mJp_P1_reconstructable_pt05", "n_P1_reconstructable_pt05"), "max", {"Exa.TrkX": P1["eps_tech_pt05"]}, "%"),
    ("Physics efficiency", ("no noise · pT > 0.5 GeV, |η| < 4", "Ju et al. script (J′)"),
     ("jp", "mJp_P1_selected_pt05", "n_P1_selected_pt05"), "max", {"Exa.TrkX": P1["eps_phys_pt05"]}, "%"),
    ("Technical efficiency", ("no noise · pT > 0.1 GeV, ≥ 5 hits", "Ju et al. script (J′)"),
     ("jp", "mJp_P1_reconstructable_pt01", "n_P1_reconstructable_pt01"), "max", {"Exa.TrkX": P1["eps_tech_pt01"]}, "%"),
    ("Physics efficiency", ("no noise · pT > 0.1 GeV, |η| < 4", "Ju et al. script (J′)"),
     ("jp", "mJp_P1_selected_pt01", "n_P1_selected_pt01"), "max", {"Exa.TrkX": P1["eps_phys_pt01"]}, "%"),
    ("TrackML score", ("no noise · challenge weights",), ("rep", "nonoise", "P0_trackml_score"), "max",
     {"Exa.TrkX": P1["trackml_score"]}, "score"),
    ("Target efficiency", ("all hits · Canto definition",), ("rep", "full", "ours_eff_target"), "max", {}, "%"),
    ("Fake-track rate", ("all hits · Canto definition", "lower is better"), ("rep", "full", "ours_fake_rate"), "min",
     {}, "%"),
    ("TrackML score", ("all hits · challenge weights",), ("rep", "full", "P0_trackml_score"), "max", {}, "score"),
    ("Target efficiency", ("no noise · Canto definition",), ("rep", "nonoise", "ours_eff_target"), "max", {}, "%"),
    ("Fake-track rate", ("no noise · Canto definition", "lower is better"), ("rep", "nonoise", "ours_fake_rate"),
     "min", {}, "%"),
    ("Target efficiency", ("TrackML-1GeV · Canto definition",), ("rep", "1gev", "ours_eff_target"), "max", {}, "%"),
    ("Fake-track rate", ("TrackML-1GeV · Canto definition", "lower is better"), ("rep", "1gev", "ours_fake_rate"),
     "min", {}, "%"),
]
TITLE_ACC = "Canto-L · TrackML accuracy phase"
SUBTITLE_ACC = ("Track reconstruction at pileup 200 · the frozen throughput-phase pipeline, not retrained, evaluated "
                "once on 300 accuracy-phase events (Kaggle train_1)")


def ours(cond: str, metric: str, name: str = "WP-A") -> tuple[float, float]:
    """Canto-L value and event-bootstrap error from the post-freeze recount."""
    m = pd.read_csv(f"{RUN}/test_post/refmetrics/test_{cond}/refmetrics.csv")
    r = m[(m.name == name) & (m.metric == metric)]
    if len(r) != 1:
        raise SystemExit(f"{cond}/{metric}/{name}: {len(r)} rows")
    return float(r.value.iloc[0]), float(r.boot_std.iloc[0])


def report(sset: str, cond: str, metric: str, name: str = "WP-A", err: bool = False):
    """A value (with err: value and bootstrap error) of the test chain's report (frozen definitions); None while the
    report of that condition is not written."""
    f = f"{RUN}/test/report_{sset}/metrics_{cond}.csv"
    if not os.path.exists(f):
        return None
    m = pd.read_csv(f)
    r = m[(m.name == name) & (m.metric == metric)]
    if len(r) != 1:
        raise SystemExit(f"{sset}/{cond}/{metric}/{name}: {len(r)} rows")
    return (float(r.value.iloc[0]), float(r.boot_std.iloc[0])) if err else float(r.value.iloc[0])


def jprime(num: str, den: str, sset: str = "t2", cond: str = "nonoise", name: str = "WP-A",
           B: int = 500, seed: int = 12345) -> tuple[float, float]:
    """A pooled ratio of the stored per-event counts with the reports' event bootstrap (hnl.trackml.evaluate.bootstrap:
    same seed, same resampled event lists); reproduces the reported J' values at pT > 0.5 GeV."""
    d = f"{RUN}/test/{sset}" + ("" if cond == "full" else f"_{cond}")
    i = json.load(open(f"{d}/config_names.json")).index(name)
    c = pd.read_parquet(f"{d}/per_event.parquet")
    g = c[c.config == i].sort_values("event")
    a, b = g[num].to_numpy(float), g[den].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(g), size=(B, len(g)))
    return float(a.sum() / b.sum()), float(pd.Series(a[idx].sum(1) / b[idx].sum(1)).std())


def acc_value(src) -> tuple[float, float] | None:
    return jprime(*src[1:]) if src[0] == "jp" else report("t2", src[1], src[2], err=True)


def fmt_published(v: float) -> str:
    """As quoted: the decimals of the stored fraction, shown in percent."""
    d = len(repr(v).split(".")[1]) - 2 if "." in repr(v) else 0
    return f"{100 * v:.{max(d, 0)}f}%"


def fmt_ours(v: float) -> str:
    return f"{100 * v:.3f}%" if 100 * v < 0.1 else f"{100 * v:.2f}%"


def row_values(cond, key, pub):
    """{column: (value, sigma)} for one row; a published value without a quoted uncertainty has sigma 0."""
    return with_published(ours(cond, key), pub)


def with_published(ve, pub):
    """Canto-L's (value, sigma), absent while pending, and the published values of the row."""
    vals = {} if ve is None else {"Canto-L": (float(ve[0]), float(ve[1]))}
    for col, (pv, pe) in pub.items():
        vals[col] = (float(pv), float(pe) if pe else 0.0)
    return vals


def tied_best(vals: dict, kind: str) -> set:
    """The best column and every column within 2 combined sigma of it (not resolved)."""
    best = (max if kind == "max" else min)(vals, key=lambda c: vals[c][0])
    vb, sb = vals[best]
    return {c for c, (v, s) in vals.items() if abs(v - vb) <= 2 * np.hypot(s, sb)}


def first_place(vals: dict, kind: str, unit: str) -> set:
    """The column with the best value, and any column showing the same value."""
    best = (max if kind == "max" else min)(vals, key=lambda c: vals[c][0])
    shown = lambda c: f"{vals[c][0]:.3f}" if unit == "score" else (fmt_ours(vals[c][0]) if c == "Canto-L"
                                                                   else fmt_published(vals[c][0]))
    return {c for c in vals if shown(c).rstrip("%") == shown(best).rstrip("%")
            or abs(vals[c][0] - vals[best][0]) < 1e-12}


def notes() -> list[str]:
    g = lambda c, k, n="WP-A": ours(c, k, n)
    errs = [ours(c, k)[1] for _t, _s, c, k, _b, _r in ROWS]
    f3, e3 = g("full", "eggnet_fake", "WP-R-P3")
    eff4, s4 = g("1gev", "eggnet_efficiency", "WP-R-P4")
    e2, f2 = g("full", "liu_eff", "WP-R-P2"), g("full", "liu_fr", "WP-R-P2")
    return [n.replace(" %", "\u00a0%").replace(" et al.", "\u00a0et\u00a0al.") for n in [
        "All numbers: TrackML throughput phase (Zenodo 4730157), pileup 200; only references that used the same "
        "phase are shown (established from the authors' code and released data). Canto-L: the frozen pipeline at "
        "working point WP-A, chosen on 150 val events (the highest DM efficiency of targets at a track fake rate ≤ "
        "2.94 %, the HNL value, not re-tuned; configurations within one standard error tied, the lowest fake rate "
        "wins), evaluated once on 300 test events; one training seed. Every row is recomputed after the freeze from "
        "the frozen pipeline's test tracks with ports of the reference's evaluation code, checked against the "
        "unmodified code on hand-counted, randomised and real events (EggNet: github.com/exatrkx/EggNet e2bd117; "
        "Liu et al.: eval_metrics of github.com/ryanliu30/Tracking-ML-Exa.TrkX f107781, strict > 50 % majority, "
        "which reproduces their Truth-CC; HierarchicalGNN a2563af has >= 50 %); nothing was re-chosen. "
        f"Event-bootstrap errors on the Canto-L values are at most {np.ceil(1e4 * max(errs)) / 100:.2f} points. "
        "Published values as quoted, on the authors' own events; where no uncertainty is quoted (Liu et al., the "
        "Calafiura et al. fake rates) the 2-sigma test uses the Canto-L error alone. Bold and shaded: the first "
        "place of each row; a difference below 2 sigma is not resolved.",
        "Working points chosen on val for one reference each, with our readings of its definitions. Full TrackML, EggNet code, at the point chosen for "
        f"Chan et al.'s fake rate: efficiency {100 * g('full', 'eggnet_efficiency', 'WP-R-P3')[0]:.2f} %, "
        f"duplicates {100 * g('full', 'eggnet_duplication', 'WP-R-P3')[0]:.2f} %, fake tracks "
        f"{100 * f3:.3f} ± {100 * e3:.3f} % (EggNet {100 * P3['efficiency'][0]:.2f} %, "
        f"{100 * P3['duplication'][0]:.2f} %, {100 * P3['fake'][0]:.2f} ± {100 * P3['fake'][1]:.2f} %). "
        f"TrackML-1GeV, at the point chosen for the fake rate of Calafiura et al.'s EC: efficiency {100 * eff4:.2f} ± "
        f"{100 * s4:.2f} %, duplicates {100 * g('1gev', 'eggnet_duplication', 'WP-R-P4')[0]:.2f} %, fake tracks "
        f"{100 * g('1gev', 'eggnet_fake', 'WP-R-P4')[0]:.3f} %. "
        f"Full TrackML, Liu et al. code, at the point chosen for the FR of their EC-GNN: efficiency "
        f"{100 * e2[0]:.2f} ± {100 * e2[1]:.2f} %, FR {100 * f2[0]:.2f} ± {100 * f2[1]:.2f} % (EC-GNN "
        f"{100 * P2['full']['EC-GNN'][0]:.2f} %, {100 * P2['full']['EC-GNN'][1]:.2f} %).",
        "Caveats. Liu et al. state that their FR is a model-to-model comparison, not final track-building "
        "performance; Canto-L's is after its final builder. Liu et al.'s code takes a particle's pT as the minimum "
        "hit-level truth pT, our tables hold the initial pT. Our TrackML-1GeV events keep the hits of particles with "
        "pT > 1 GeV (EggNet's hard cut; Liu et al.'s released events cut hits by hit-level pT).",
        "Context, not a comparison: TrackML score on full test events "
        f"{report('test', 'full', 'P0_trackml_score'):.3f}; challenge winner of the throughput phase (2 CPU cores) "
        "Mikado 94.4 % at 0.56 s per event.",
    ]]


def throughput_rows() -> list:
    return [(t, sub, row_values(c, k, pub), kind, "%") for t, sub, c, k, kind, pub in ROWS]


def acc_rows() -> list:
    return [(t, sub, with_published(acc_value(src), pub), kind, unit) for t, sub, src, kind, pub, unit in ROWS_ACC]


def notes_acc(rows: list) -> list[str]:
    pct = lambda v: f"{100 * v:.2f}\u00a0%"
    t2 = lambda c, k: report("t2", c, k)
    te = lambda c, k: report("test", c, k)
    e_pct = max(r[2]["Canto-L"][1] for r in rows if r[4] == "%" and "Canto-L" in r[2])
    e_sc = max(r[2]["Canto-L"][1] for r in rows if r[4] == "score" and "Canto-L" in r[2])
    pending = t2("full", "ours_eff_target") is None
    return [n.replace(" %", "\u00a0%").replace(" et al.", "\u00a0et\u00a0al.") for n in [
        "All numbers: TrackML accuracy phase (Kaggle 2018, train_1), pileup 200. Canto-L: the frozen pipeline of the "
        "throughput-phase card (trained on throughput-phase events, working point WP-A chosen on throughput-phase val "
        "events), applied once and without retraining to 300 accuracy-phase events (T2), whose digitisation differs "
        "(pixel clusters about twice as large). Ju et al. (Exa.TrkX: metric learning, edge filter, interaction "
        "network, connected components): 50 test events of the accuracy phase (established from the hit counts of "
        "their Fig. 9 and the efficiencies of their Fig. 5; the phase of their training events is not stated), trained "
        "without noise hits; the noise condition of their test is not stated. The first five rows count Canto-L with "
        "variant J′, our port of the authors' evaluation script (exatrkx-iml2020 eval_reco_trkx.py: noise hits and "
        "the hits of particles with ≤ 4 hits dropped before matching, strict > 50 % majorities); unlike the ports of "
        "the throughput-phase card it was not run against the unmodified script. "
        f"Event-bootstrap errors on the Canto-L values are at most {np.ceil(1e4 * e_pct) / 100:.2f} points "
        f"(score {np.ceil(1e4 * e_sc) / 1e4:.4f}). Published values as quoted. Bold and shaded: the best value of a "
        "row and every value within 2 sigma of it (not resolved).",
        f"Not shown: Ju et al.'s purity, {100 * P1['purity'][0]:.1f} ± {100 * P1['purity'][1]:.1f} %, whose track "
        "population is not stated (their connected components label every hit, so one-hit tracks may count), and "
        f"their TrackML score with a pipeline trained on events with noise hits, {P1['trackml_score_trained_with_noise'][0]:.3f} "
        f"± {P1['trackml_score_trained_with_noise'][1]:.3f} (test condition not stated). Context, not a comparison: "
        f"accuracy-phase challenge winner (Kaggle 2018) Top Quarks {REF['P0']['kaggle_best_context_only']:.5f}.",
        "The same pipeline on the throughput-phase test (300 events), for the change between phases: target "
        f"efficiency {pct(te('full', 'ours_eff_target'))} with all hits, {pct(te('nonoise', 'ours_eff_target'))} "
        f"without noise, {pct(te('1gev', 'ours_eff_target'))} in TrackML-1GeV; fake-track rate "
        f"{pct(te('full', 'ours_fake_rate'))}, {pct(te('nonoise', 'ours_fake_rate'))}, "
        f"{100 * te('1gev', 'ours_fake_rate'):.3f} %; TrackML score {te('full', 'P0_trackml_score'):.3f} with all "
        f"hits, {te('nonoise', 'P0_trackml_score'):.3f} without noise. The accuracy-phase score weights also count the "
        "particles produced in the simulation (9.1 % of the weight in each of three T2 events; none in three "
        "throughput-phase events); how much of the score change this accounts for is not measured."
        + (" All-hit T2 rows: evaluation running." if pending else ""),
    ]]


def render(out: str, title: str, subtitle: str, columns: list, rows: list, nts: list[str]) -> list[str]:
    """rows: (title, subtitle lines, {column: (value, sigma)}, best = max/min, unit "%" or "score"); a row without
    a Canto-L value shows it as pending."""
    st = dict(FULL)
    W = 4.7 + 2.42 * max(len(columns), len(COLUMNS))
    lines = sum(len(textwrap.wrap(n, int(10.2 * (W - 1.2)), break_on_hyphens=False)) for n in nts)
    H = st["top_gap"] + st["hh"] + st["rh"] * len(rows) + 0.45 + 0.3 * (lines - 1) + 0.12 * (len(nts) - 1) + 0.58
    lw, hh, rh = st["lw"], st["hh"], st["rh"]
    fig = plt.figure(figsize=(W, H), dpi=st["dpi"])
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(W / 2, H - 1.25, title, ha="center", va="center", fontsize=58, family="FreeSerif", color=INK)
    for k, line in enumerate(textwrap.wrap(subtitle, 110)):
        ax.text(W / 2, H - 2.05 - 0.32 * k, line, ha="center", va="center", fontsize=15, family="DejaVu Sans",
                color=MUTED)
    x0, x1 = st["margin"], W - st["margin"]
    cw = (x1 - x0 - lw) / len(columns)
    top = H - st["top_gap"]
    ybot = top - hh - rh * len(rows)
    ax.add_patch(Rectangle((x0, ybot), x1 - x0, top - ybot, fill=False, lw=st["border"], ec=INK))
    for i in range(len(columns) + 1):
        x = x0 + lw + i * cw
        ax.plot([x, x], [ybot, top], color=INK, lw=st["grid"])
    for j in range(len(rows) + 1):
        y = top - hh - j * rh
        ax.plot([x0, x1], [y, y], color=INK, lw=st["grid"])
    for i, (head, note) in enumerate(columns):
        xt, yh = x0 + lw + i * cw + st["cpad"], top - hh / 2
        ax.text(xt, yh + st["name_dy"], head, ha="left", va="center", fontsize=st["name"], family="DejaVu Sans",
                weight="bold" if i == 0 else "normal", color=INK)
        ax.text(xt, yh + st["note_dy"], note, ha="left", va="center", fontsize=st["note"], color=MUTED)
    for j, (rt, rs, vals, kind, unit) in enumerate(rows):
        yc = top - hh - (j + 0.5) * rh
        subs = list(rs)
        dy = (len(subs) - 1) * 0.25 / 2 if len(subs) > 1 else 0.0
        ax.text(x0 + st["lpad"], yc + st["title_dy"] + dy, rt, ha="left", va="center", fontsize=st["title"],
                weight="bold", color=INK)
        for k, line in enumerate(subs):
            ax.text(x0 + st["lpad"], yc + st["sub_dy"] + dy - k * 0.25, line, ha="left", va="center",
                    fontsize=st["sub"], color=MUTED)
        best = first_place(vals, kind, unit) if len(vals) > 1 and "Canto-L" in vals else set()
        for i, (head, _note) in enumerate(columns):
            xl = x0 + lw + i * cw
            v = vals.get(head)
            if v is None:
                txt, is_best = ("pending" if i == 0 else "—"), False
            elif unit == "score":
                txt, is_best = f"{v[0]:.3f}", head in best
            else:
                txt = fmt_ours(v[0]) if i == 0 else fmt_published(v[0])
                is_best = head in best
            if is_best:
                ax.add_patch(Rectangle((xl, yc - rh / 2), cw, rh, color=PEACH if i == 0 else GREY, zorder=0))
            elif i == 0:
                ax.add_patch(Rectangle((xl, yc - rh / 2), cw, rh, color="#F6E7DF", zorder=0))
            ax.text(xl + st["cpad"], yc + st["val_dy"] + dy, txt, ha="left", va="center", fontsize=st["val"],
                    weight="bold" if is_best else "normal", color=INK if v is not None else MUTED)
    ax.add_patch(Rectangle((x0 + lw, ybot), cw, top - ybot, fill=False, lw=st["frame"], ec=ORANGE, zorder=5))
    y = ybot - 0.45
    for note in nts:
        for line in textwrap.wrap(note, int(10.2 * (W - 1.2)), break_on_hyphens=False):
            ax.text(x0 + 0.1, y, line, ha="left", va="top", fontsize=11.2, color=MUTED)
            y -= 0.3
        y -= 0.12
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", facecolor=BG, dpi=st["dpi"])
        files.append(f"{out}.{ext}")
    plt.close(fig)
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="throughput-phase card")
    ap.add_argument("--out-accuracy", help="accuracy-phase card")
    a = ap.parse_args()
    files = render(a.out, TITLE, SUBTITLE, COLUMNS, throughput_rows(), notes())
    if a.out_accuracy:
        rows = acc_rows()
        files += render(a.out_accuracy, TITLE_ACC, SUBTITLE_ACC, COLUMNS_ACC, rows, notes_acc(rows))
    print("\n".join(files))
