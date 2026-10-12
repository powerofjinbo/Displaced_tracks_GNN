"""A "model release" style comparison card of our best model against the published models (val numbers).

Every number is read from a ranking table of hnl.eval.ranking (no transcription); parameter counts from the runs'
training records. The highlighted column is our model; in every row the best value is bold and shaded (peach in
the highlighted column, grey elsewhere).

--slide renders the slide version to OUT_slide.{png,pdf}: the same table (same numbers and shading) without the
title and the notes block, drawn for about 1024 px of width on a 1920 x 1080 slide, with larger header notes, row
subtitles and k/n particle counts. The caveats of the notes then have to be on the slide itself.

--slide-wide renders OUT_slide_wide.{png,pdf}: the same table for the full content width of a 1920 x 1080 slide
(--width, default 1664 slide px; PNG at 2x) in the slides' typeface (Noto Sans, figures/video/fonts), every text at
least 22 slide px (values 30 px), rows 60 slide px high. The caveats go on the slide.

    python -m hnl.plots.release_card --ranking results/ranking/ranking_val_v3.csv --out figures/release/canto_card
    python -m hnl.plots.release_card --ranking results/ranking/ranking_val_v3.csv --out figures/release/canto_card --slide
    python -m hnl.plots.release_card --ranking results/ranking/ranking_val_v3.csv --out figures/release/canto_card \
        --slide-wide [--width 1664]

The columns are the v3 ranking's labels (one-standard-error builder rule, extended grid; Canto-L, Canto-H and Canto-B
at the full budget with three seeds each; docs/canto_facts.md section 7c).

--test renders the card of the one-time blind test from the test ranking of hnl.eval.ranking_test
(results/ranking/ranking_test_v1.csv; docs/canto_facts.md section 7e): the columns COLUMNS_TEST (the val card's
models without DeZoort IN, which was never evaluated on test; acorn (HNL) and acorn at the cut chosen as best on val),
header notes "3 seeds · TEST", the same rows and shading rule, the title "Canto · one-time blind test" and the test
notes (frozen selection, test sets, the acorn cuts and acorn's values at the preregistered cut rule, read from the
same ranking). It refuses a ranking whose Canto rows are not configuration 0 of RUN/trackbuild_test, and the val card
refuses a test ranking. With --slide / --slide-wide the geometry is that of the val versions; the full test card is
narrower by one column and as tall as its notes need. The val card is unchanged.

    python -m hnl.plots.release_card --test --ranking results/ranking/ranking_test_v1.csv \
        --out figures/release/canto_card_test [--slide | --slide-wide]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import textwrap
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath

RUNS = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
PREVIEW = RUNS + "/stage1_screen_ttbar/symregion_trunc/armi_s0"   # the previews' classifier
COLUMNS = [  # (header, header note, ranking label, a run holding the parameter count, training data)
    ("Canto-L", "full budget", "Canto-L (full budget): learned triplet union, learned merging (rule R6)",
     RUNS + "/stage1_final_learned/l1tl/armi_s0", "ttbar only"),
    ("Canto-H", "full budget", "Canto-H (full budget): hand-cut helix union, walkthrough + merging (rule R6)",
     RUNS + "/stage1_final_union/l1tu/armi_s0", "ttbar only"),
    ("Canto-B", "full budget", "Canto-B (full budget): base graph, learned merging (rule R6)",
     RUNS + "/stage1_final/l1t/armi_s0", "ttbar only"),
    ("acorn (HNL)", "HNL-trained", "acorn IGNN, HNL-trained + CC (best cut <= FR*)", RUNS + "/baseline_v1/armii_s0",
     "HNL signal"),
    ("acorn", "ttbar-trained", "acorn IGNN, ttbar-trained + CC (best cut <= FR*)", RUNS + "/baseline_v1/armi_s0",
     "ttbar only"),
    ("DeZoort IN", "ttbar-trained", "DeZoort IN (6.4k par.) + CC [FR* unreachable]",
     RUNS + "/stage1_screen_ttbar/dzin/armi_s0", "ttbar only"),
]
# --test: the same models without DeZoort IN (never evaluated on test), header note "TEST"; hnl.eval.ranking_test keeps
# the val ranking's label for the same frozen model and working point, so the labels (acorn: the cut chosen as best on
# val, applied unchanged to test) and the parameter runs are those of COLUMNS.
COLUMNS_TEST = [(head, "TEST", lab, run, train) for head, _note, lab, run, train in COLUMNS if head != "DeZoort IN"]
PREREG_TEST = {  # test ranking label of the same acorn model at its preregistered cut rule (the claims' reference)
    "acorn (HNL)": "acorn IGNN, HNL-trained + CC (prereg. cut rule)",
    "acorn": "acorn IGNN, ttbar-trained + CC (prereg. cut rule)",
}
TITLE_TEST = "Canto · one-time blind test"
SUBTITLE_TEST = "Displaced-track reconstruction, trained on ttbar only · frozen models, evaluated once on the test sets"
ROWS = [  # (title, subtitle, ranking column, kind)
    ("Displaced tracks", "|d0| ≥ 10 mm or r_v ≥ 100 mm", "eff_primary", "max"),
    ("Far-displaced tracks", "|d0| 100–300 mm", "eff_d0_100", "max"),
    ("Very far-displaced", "|d0| ≥ 300 mm", "eff_d0_300", "max"),
    ("Prompt muons", "W → μ in HNL events", "eff_w_muons", "max"),
    ("Dense prompt tracks", "ttbar, |d0| < 1 mm", "eff_ttbar_prompt", "max"),
    ("Fake-track rate", "lower is better", "fake_rate_hnl", "min"),
    ("Training data", "", "train", None),
    ("Parameters", "edge classifier", "params", None),
]
BG, INK, MUTED = "#FAF9F5", "#141413", "#6B6A64"
PEACH, ORANGE, GREY = "#EDCAB8", "#D97757", "#D3D1C7"

# Geometry in inches (the axes span the figure: one data unit is one inch), font sizes in points. *_dy: offsets of
# a text centre from its cell centre; *_lh: line step of the two-line header notes and wrapped row subtitles.
FULL = dict(dpi=100, width=4.7 + 2.42 * len(COLUMNS), height=19.3, margin=0.5, top_gap=2.9, lw=3.7, hh=0.85,
            rh=1.32, lpad=0.22, cpad=0.22, border=1.3, grid=1.0, frame=4.5,
            name=21, name_dy=0.08, note=10.5, note_dy=-0.26, note_lh=None,          # note_lh None: one-line note
            title=19, title_dy=0.17, sub=13.5, sub_dy=-0.27, sub_lh=None, center_bare=False,
            val=21, val_dy=0.12, text=16, kn=11, kn_dy=-0.33)
# Slide: drawn at 100 slide px per inch for a 1024 px wide image (PNG at 2x); smallest text 13.5 slide px.
SLIDE = dict(dpi=200, width=10.24, height=None, margin=0.06, top_gap=0.06, lw=2.00, hh=0.80,
             rh=0.72, lpad=0.09, cpad=0.065, border=1.2, grid=0.8, frame=2.6,
             name=11.0, name_dy=0.195, note=10.4, note_dy=-0.0275, note_lh=0.175,
             title=11.2, title_dy=0.1075, sub=10.4, sub_dy=-0.117, sub_lh=0.175, center_bare=True,
             val=17.3, val_dy=0.1075, text=11.9, kn=9.7, kn_dy=-0.135)
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures", "video", "fonts")


def wide_geometry(width_px: float = 1664.0) -> dict:
    """--slide-wide: drawn at 100 slide px per inch (PNG at 2x) for `width_px` slide px of width; sizes in points are
    slide px x 0.72, so the smallest text (header notes, row subtitles, k/n counts) is 22 slide px. Noto Sans as in the
    video, with DejaVu Sans for the glyphs it lacks (≥, →)."""
    pt = 0.72
    return dict(dpi=200, width=width_px / 100.0, height=None, margin=0.015, top_gap=0.015, lw=3.70, hh=0.92,
                rh=0.60, lpad=0.14, cpad=0.13, border=1.2, grid=0.8, frame=2.6,
                name=26 * pt, name_dy=0.285, note=22 * pt, note_dy=-0.02, note_lh=0.255,
                title=24 * pt, title_dy=0.125, sub=22 * pt, sub_dy=-0.125, sub_lh=0.25, center_bare=True,
                val=30 * pt, val_dy=0.117, text=24 * pt, kn=22 * pt, kn_dy=-0.143, center_vals=True,
                family=["Noto Sans", "DejaVu Sans"])


def full_geometry(n_columns: int, notes: Sequence[str]) -> dict:
    """FULL for `n_columns` columns (2.42 in each), as tall as `notes` need when card() wraps them: the notes block
    starts 0.45 in below the table, 0.3 in per line and 0.12 in between notes, 0.58 in below the last line's top (the
    val card's six columns and NOTES give FULL's 19.3 in). Used by the full test card; the val card keeps FULL."""
    W = 4.7 + 2.42 * n_columns
    lines = sum(len(textwrap.wrap(n, int(10.2 * (W - 1.2)), break_on_hyphens=False)) for n in notes)
    H = (FULL["top_gap"] + FULL["hh"] + FULL["rh"] * len(ROWS) + 0.45 + 0.3 * (lines - 1)
         + 0.12 * (len(notes) - 1) + 0.58)
    return {**FULL, "width": W, "height": round(H, 2)}


def _use_noto() -> None:
    """Register the Noto Sans files shipped with the video (only the --slide-wide card uses them)."""
    from matplotlib.font_manager import fontManager
    for f in sorted(glob.glob(os.path.join(FONT_DIR, "NotoSans-*.ttf"))):
        fontManager.addfont(f)


def n_params(run: str) -> int:
    f = sorted(glob.glob(os.path.join(run, "gnn", "run_*.json")))[-1]
    return int(json.load(open(f))["n_parameters"])


def _wrap(text: str, size: float, width: float, family: str = "DejaVu Sans") -> list[str]:
    """`text` (`family`, `size` pt) in lines at most `width` inches wide: one line if it fits, else the most
    balanced two-line split, else greedy lines. A number stays with its unit and with a relation sign before it."""
    fp = FontProperties(family=family)
    w = lambda s: TextPath((0, 0), s.replace("\u00a0", " "), size=size, prop=fp).get_extents().width / 72
    glued = re.sub(r"([≥≤<>→]) ", "\\1\u00a0", re.sub(r"(\d) (mm|GeV)\b", "\\1\u00a0\\2", text))
    words = glued.split(" ")
    lines = [glued]
    if w(glued) > width and len(words) > 1:
        a, b = min(((" ".join(words[:k]), " ".join(words[k:])) for k in range(1, len(words))),
                   key=lambda ab: max(w(ab[0]), w(ab[1])))
        lines = [a, b]
        if max(w(a), w(b)) > width:                     # greedy over as many lines as it takes
            lines = []
            for word in words:
                if lines and w(lines[-1] + " " + word) <= width:
                    lines[-1] += " " + word
                else:
                    lines.append(word)
    return [line.replace("\u00a0", " ") for line in lines]


def card(ranking: str, out: str, title: str | None = None, subtitle: str | None = None,
         notes: Sequence[str] = (), slide: bool = False, geometry: dict | None = None,
         columns: Sequence[tuple] | None = None) -> list[str]:
    """Render the card to OUT.png / OUT.pdf: the full card (title, table, notes) or, with slide=True, the table
    alone in the slide geometry (title, subtitle and notes are then not drawn); `geometry` replaces the FULL or SLIDE
    geometry, e.g. wide_geometry() for --slide-wide or full_geometry() for the full test card; `columns` replaces
    COLUMNS (e.g. COLUMNS_TEST); the first column is the highlighted one."""
    cols = COLUMNS if columns is None else list(columns)
    st = geometry if geometry is not None else (SLIDE if slide else FULL)
    fam = st.get("family")                                # None: matplotlib's default family, as before
    tk = {"family": fam} if fam else {}
    if fam and "Noto Sans" in fam:                     # DejaVu Sans fills the glyphs Noto Sans lacks (≥ →)
        _use_noto()
    tab = pd.read_csv(ranking).set_index("label")
    vals = []                                             # per column (two columns may share a header)
    for head, note, lab, run, train in cols:
        r = tab.loc[lab]
        vals.append({**{k: float(r[k]) for _, _, k, kind in ROWS if kind},
                     **{k: str(r[k]) for _, _, k, kind in ROWS if not kind and k in r.index}, "train": train,
                     "params": n_params(run), "n_seeds": int(r.n_seeds), "k_n_300": r.k_n_d0_300})
    lw, hh, rh = st["lw"], st["hh"], st["rh"]
    W = st["width"]
    H = st["height"] or 2 * st["margin"] + hh + rh * len(ROWS)
    fig = plt.figure(figsize=(W, H), dpi=st["dpi"])
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    if not slide:
        ax.text(W / 2, H - 1.25, title, ha="center", va="center", fontsize=58, family="FreeSerif", color=INK)
        ax.text(W / 2, H - 2.15, subtitle, ha="center", va="center", fontsize=15, family="DejaVu Sans", color=MUTED)
    x0, x1 = st["margin"], W - st["margin"]
    cw = (x1 - x0 - lw) / len(cols)
    top = H - st["top_gap"]
    ybot = top - hh - rh * len(ROWS)
    # grid
    ax.add_patch(Rectangle((x0, ybot), x1 - x0, top - ybot, fill=False, lw=st["border"], ec=INK))
    for i in range(len(cols) + 1):
        x = x0 + lw + i * cw
        ax.plot([x, x], [ybot, top], color=INK, lw=st["grid"])
    for j in range(len(ROWS) + 1):
        y = top - hh - j * rh
        ax.plot([x0, x1], [y, y], color=INK, lw=st["grid"])
    # headers
    for i, (head, note, *_rest) in enumerate(cols):
        n = vals[i]["n_seeds"]
        seeds = f"{n} seed{'s' if n > 1 else ''}"
        xt, yh = x0 + lw + i * cw + st["cpad"], top - hh / 2
        ax.text(xt, yh + st["name_dy"], st.get("heads", {}).get(head, head), ha="left", va="center",
                fontsize=st["name"], family=fam or "DejaVu Sans", weight="bold" if i == 0 else "normal", color=INK)
        if st["note_lh"] is None:
            ax.text(xt, yh + st["note_dy"], f"{seeds} · {note}", ha="left", va="center", fontsize=st["note"],
                    color=MUTED, **tk)
        else:                                             # two lines: seeds, then the note
            for k, line in enumerate((seeds, note)):
                ax.text(xt, yh + st["note_dy"] - k * st["note_lh"], line, ha="left", va="center",
                        fontsize=st["note"], color=MUTED, **tk)
    # rows
    for j, (rt, rs, key, kind) in enumerate(ROWS):
        yc = top - hh - (j + 0.5) * rh
        subs = [rs] if rs else []
        if rs and st["sub_lh"] is not None:               # wrap the subtitle into the row-label column
            subs = _wrap(rs, st["sub"], lw - 2 * st["lpad"], fam or "DejaVu Sans")
        # shift of the row's text block: centred when it has several subtitle lines or (slide) none
        dy = (len(subs) - 1) * st["sub_lh"] / 2 if len(subs) > 1 else 0.0
        if not subs and st["center_bare"]:
            dy = -st["title_dy"]
        ax.text(x0 + st["lpad"], yc + st["title_dy"] + dy, rt, ha="left", va="center", fontsize=st["title"],
                weight="bold", color=INK, **tk)
        for k, line in enumerate(subs):
            ax.text(x0 + st["lpad"], yc + st["sub_dy"] + dy - k * (st["sub_lh"] or 0.0), line, ha="left",
                    va="center", fontsize=st["sub"], color=MUTED, **tk)
        fmt = (lambda v: f"{100 * v:.2f}%") if key == "fake_rate_hnl" else (lambda v: f"{100 * v:.1f}%")
        best = None
        if kind:                                          # every column that shows the best value (ties too)
            best = fmt((max if kind == "max" else min)(c[key] for c in vals))
        for i, (head, *_rest) in enumerate(cols):
            xl = x0 + lw + i * cw
            v = vals[i][key]
            if kind:
                txt = fmt(v)
            elif key == "params":
                txt = f"{v / 1000:.0f}k" if v >= 10000 else f"{v / 1000:.1f}k"
            else:
                txt = v
            is_best = kind is not None and txt == best
            if is_best:
                ax.add_patch(Rectangle((xl, yc - rh / 2), cw, rh, color=PEACH if i == 0 else GREY, zorder=0))
            elif i == 0:
                ax.add_patch(Rectangle((xl, yc - rh / 2), cw, rh, color="#F6E7DF", zorder=0))
            vy = yc if st.get("center_vals") and key != "eff_d0_300" else yc + st["val_dy"] + dy  # centred alone
            ax.text(xl + st["cpad"], vy, txt, ha="left", va="center",
                    fontsize=st["val"] if kind else st["text"], weight="bold" if is_best else "normal", color=INK,
                    **tk)
            if key == "eff_d0_300":
                ax.text(xl + st["cpad"], yc + st["kn_dy"] + dy, f"{vals[i]['k_n_300']} particles", ha="left",
                        va="center", fontsize=st["kn"], color=MUTED, **tk)
    # highlighted column frame
    ax.add_patch(Rectangle((x0 + lw, ybot), cw, top - ybot, fill=False, lw=st["frame"], ec=ORANGE, zorder=5))
    # notes
    y = ybot - 0.45
    for note in ([] if slide else notes):
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


NOTES = [
    "All numbers: ACTS generic detector simulation without pileup, validation set (2,999 events of ttbar and the two "
    "HNL samples: the displaced-track, prompt-muon and fake-rate rows from the HNL events, 'Dense prompt tracks' from "
    "the ttbar events); every model at a track fake rate ≤ 2.94 % (pooled HNL val, the preregistered working point FR*) "
    "where it can reach it, otherwise at its lowest fake rate (acorn ttbar-trained seeds 0 and 2, acorn HNL-trained "
    "seed 1, DeZoort IN); track efficiency by double-majority matching; mean over the listed training seeds. Best "
    "value per row bold and shaded (ties included).",
    "Canto (Chained-triplet, origin-Agnostic Networks for Tracking Off-vertex), trained on ttbar only; the branches "
    "differ in the graph: Canto-B the base graph, Canto-H a hand-cut helix-triplet union, Canto-L a learned "
    "triplet-support union. All three at the full budget (30 epochs on the ttbar training graphs), three seeds "
    "each; the builder of each branch chosen on val by rule R6 with the one-standard-error rule (within one "
    "binomial standard error of the best efficiency, the lowest fake rate): Canto-L learned merging, Canto-H walkthrough + merging, "
    "Canto-B learned fragment merging. The one-time test evaluation is pending.",
    "acorn (HNL): acorn's interaction network with connected components, trained on the HNL signal as in Wang et al. "
    "(arXiv:2203.08800), whose own pipeline used an attention GNN and DBSCAN; it reproduces their all-track efficiency "
    "(test: 0.9554 vs 0.9561 in the paper). acorn: the same network trained on ttbar only, like Canto. Both at their "
    "highest-efficiency cut within the fake-rate target where reachable, the most favourable choice for acorn (with "
    "the preregistered cut rule: 93.8 % / 92.3 %).",
    "DeZoort IN: the original 6.4k-parameter interaction network (DeZoort et al. 2021), trained on ttbar here; it "
    "cannot reach the fake-rate target and is shown at its lowest fake rate (12.5 %). |d0| ≥ 300 mm: 19 particles per "
    "seed, statistics limited.",
]


def _pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def notes_test(ranking: str) -> list[str]:
    """The notes of the full test card. The particles per seed, the acorn cuts per seed and acorn's values at the
    preregistered cut rule are read from the test ranking (no transcription)."""
    tab = pd.read_csv(ranking).set_index("label")
    row = {head: tab.loc[lab] for head, _note, lab, _run, _train in COLUMNS_TEST}
    pre = {head: tab.loc[lab] for head, lab in PREREG_TEST.items()}
    n_seeds = int(row["Canto-L"].n_seeds)
    per_seed = {}
    for key in ("k_n_d0_100", "k_n_d0_300"):                # pooled over the seeds: n = n_seeds x the seed's count
        n = int(str(row["Canto-L"][key]).split("/")[1])
        if n % n_seeds:
            raise ValueError(f"{key} = {row['Canto-L'][key]} is not {n_seeds} equal seeds")
        per_seed[key] = n // n_seeds
    cuts = lambda r: " / ".join(re.findall(r"CC (\d+(?:\.\d+)?)", str(r.working_points)))
    wps = str(row["acorn (HNL)"].working_points).split("; ")
    post = [str(i) for i, wp in enumerate(wps) if "post-freeze" in wp]   # seed order, as in hnl.eval.ranking_test
    many = len(post) > 1
    post_txt = (f"; acorn (HNL) seed{'s' if many else ''} {' and '.join(post)}: CC at "
                f"{'these cuts' if many else 'this cut'} on the stored test scores after the freeze, as the one-time "
                f"test chain did not include {'them' if many else 'it'}") if post else ""
    p = pre["acorn (HNL)"]
    acorn_pre = ("acorn the same cuts, so the values shown" if cuts(pre["acorn"]) == cuts(row["acorn"])
                 else f"acorn {cuts(pre['acorn'])}: {_pct(pre['acorn'].eff_primary)} displaced")
    return [
        "All numbers: ACTS generic detector simulation without pileup, the one-time blind test of the frozen models: "
        "the displaced-track, prompt-muon and fake-rate rows from the pooled HNL test sets (hnl10_100_test, "
        "hnl15_200_test and hnl15_100_external, 6,999 events), 'Dense prompt tracks' from ttbar_test (1,000 events); "
        f"track efficiency by double-majority matching; mean over the {n_seeds} training seeds, as in the "
        f"preregistered claims. |d0| 100–300 mm: {per_seed['k_n_d0_100']} and |d0| ≥ 300 mm: "
        f"{per_seed['k_n_d0_300']} HNL daughters per seed, the latter statistics limited. Best value per row bold and "
        "shaded (ties included).",
        "Frozen before the test: the Canto variants, every seed's builder configuration and the working points were "
        "committed before any Canto test job (results/final_selection_v1.json, commit aecb23f), and every acorn cut "
        "was fixed on val; nothing was chosen on test. Canto (trained on ttbar only, full budget; the branches differ "
        "in the graph: Canto-B the base graph, Canto-H a hand-cut helix-triplet union, Canto-L a learned "
        "triplet-support union) at the builder configurations chosen on val by rule R6 with the one-standard-error "
        "rule: Canto-L learned merging, Canto-H walkthrough + merging, Canto-B learned fragment merging. The "
        "highlighted Canto-L is the variant selected on val (rule R8); it is not re-chosen on test. These are "
        "post-freeze summaries of the one-time evaluation; the preregistered claims (results/claims_test_v1.json) are "
        "unchanged.",
        "acorn (HNL): acorn's interaction network with connected components (CC), trained on the HNL signal as in "
        "Wang et al. (arXiv:2203.08800), whose own pipeline used an attention GNN and DBSCAN; it reproduces their "
        "all-track efficiency (test: 0.9554 vs 0.9561 in the paper). acorn: the same network trained on ttbar only, "
        "like Canto. Both at the cut chosen as best on val (the highest efficiency at a fake rate ≤ FR* = 2.94 % on "
        "pooled HNL val, else the lowest-fake-rate cut), applied unchanged to test: acorn (HNL) "
        f"{cuts(row['acorn (HNL)'])}, acorn {cuts(row['acorn'])} (seeds 0 / 1 / 2{post_txt}). With the "
        f"preregistered cut rule, the reference of the claims (acorn (HNL) {cuts(p)}; {acorn_pre}): acorn (HNL) "
        f"{_pct(p.eff_primary)} displaced, {_pct(p.eff_d0_100)} at |d0| 100–300 mm, {p.k_n_d0_300} at ≥ 300 mm, "
        f"{_pct(p.eff_w_muons)} prompt muons, {_pct(p.eff_ttbar_prompt)} dense prompt tracks, fake rate "
        f"{100 * p.fake_rate_hnl:.2f} %.",
        "DeZoort IN (and IN-L) was never evaluated on test and has no column. FR* is the working point on val; the "
        "test fake rates are as measured.",
    ]


def check_ranking(ranking: str, test: bool) -> None:
    """Refuse a ranking of the wrong kind: the test card needs every Canto column at configuration 0 of
    RUN/trackbuild_test for each seed (the frozen configuration, hnl.eval.ranking_test); the val card refuses test
    rows."""
    tab = pd.read_csv(ranking).set_index("label")
    if not test:
        if tab.get("working_points", pd.Series(dtype=str)).astype(str).str.contains("trackbuild_test").any():
            raise SystemExit(f"{ranking} holds test rows: render the test card with --test")
        return
    for head, _note, lab, _run, _train in COLUMNS_TEST:
        r = tab.loc[lab]
        if head.startswith("Canto") and \
                str(r.working_points).count("(trackbuild_test, configuration 0)") != int(r.n_seeds):
            raise SystemExit(f"{ranking}: {head} is not at configuration 0 of RUN/trackbuild_test for every seed "
                             f"({r.working_points})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--out", required=True, help="output stem: OUT.png / OUT.pdf (with --slide: OUT_slide.*)")
    ap.add_argument("--slide", action="store_true",
                    help="the slide version: table only (no title, no notes), larger secondary text")
    ap.add_argument("--slide-wide", action="store_true",
                    help="the full-width slide version (OUT_slide_wide.*): table only, Noto Sans, text >= 22 slide px")
    ap.add_argument("--width", type=float, default=1664.0, help="--slide-wide: width in slide px (default 1664)")
    ap.add_argument("--test", action="store_true",
                    help="the one-time test card (COLUMNS_TEST, test title and notes) from a test ranking of "
                         "hnl.eval.ranking_test, e.g. results/ranking/ranking_test_v1.csv")
    a = ap.parse_args(argv)
    check_ranking(a.ranking, a.test)
    cols = COLUMNS_TEST if a.test else COLUMNS
    if a.slide_wide:
        files = card(a.ranking, a.out + "_slide_wide", slide=True, geometry=wide_geometry(a.width), columns=cols)
    elif a.slide:
        files = card(a.ranking, a.out + "_slide", slide=True, columns=cols)
    elif a.test:
        notes = notes_test(a.ranking)
        files = card(a.ranking, a.out, TITLE_TEST, SUBTITLE_TEST, notes, geometry=full_geometry(len(cols), notes),
                     columns=cols)
    else:
        files = card(a.ranking, a.out, "Canto", "Displaced-track reconstruction, trained on ttbar only", NOTES)
    print("\n".join(files))


if __name__ == "__main__":
    main()
