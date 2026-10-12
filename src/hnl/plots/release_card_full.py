"""One "model release" card with every result of the study, in the look of hnl.plots.release_card (colours, bold row
title over a muted subtitle, the highlighted Canto-L column in an orange frame, the best value of a row bold and
shaded). Seventeen columns: Canto-L/H/B, Canto-L trained on HNL and on ttbar+HNL (validation only), acorn with its
val-best builder (three training samples), acorn + CC at the preregistered cut (two), DeZoort IN / IN-L, Wang et al.'s
published values (external sample only) and the published TrackML references of the throughput phase. Six bands,
each a full-width row:
HNL validation, HNL one-time test, Wang et al.'s external sample, the second population, the TrackML throughput-phase
test and training.

Every number is read from a result file; only labels are typed here:
  results/bootstrap_v1.csv              Canto, acorn + CC and DeZoort on the pooled HNL val and test sets
  results/strong_baselines_v3.csv       acorn with the val-best builder (values, k/n at |d0| >= 300 mm)
  results/strong_baselines_v3_differences.csv   the same acorn rows' combined sigma ("Canto-L minus acorn" block)
  results/per_sample_v1.csv             per test sample (hnl10_100_test, hnl15_200_test, hnl15_100_external)
  results/published_definitions_comparison.csv  Wang et al.'s definitions on the external sample
  results/second_population_v2.csv      V0 daughters and conversion e+-
  results/training_sample_bins_v1.csv   Canto-L trained on HNL / ttbar+HNL (values, primary sigma, |d0|-bin k/n)
  results/training_sample_val_mwslmerge.csv     the same runs' per-seed k/n (cross-checked against the values)
  results/model_table_v1.md             training events, parameters and steps (edge classifier)
  docs/paper_facts.md                   Wang et al.'s published values (one line each, cross-checked against the
                                        "published" rows of results/published_definitions_comparison.csv)
  results/final_selection_v1.json       FR*
  $RUN/test_post/refmetrics, $RUN/test/report_test (via hnl.plots.release_card_trackml: ours(), report()),
                                        $RUN/gnn/run_*.json (training),
  hnl.trackml.evaluate.REFERENCES       published TrackML values
with $RUN the TrackML run canto_l_v1/seed0_x1. No acorn number on TrackML is shown (only Canto-L was run there).

Cells: the value, and on a muted second line the combined uncertainty in percentage points (score units in score
rows), or k/n in the |d0| >= 300 mm rows and where the source gives no sigma; "—" where a model was not evaluated on that
set; "pending" where a report is not written yet. Shading (the user's rule, 2026-10-12): the
first place of each row only (highest value, lowest for rates; cells showing the same value share it), dark in its
column; the Canto-L column is light where it is not first. The 2-sigma rule is stated in the notes.

    python -m hnl.plots.release_card_full --out figures/release/canto_card_full [--dump cells.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from hnl.plots.release_card import BG, GREY, INK, MUTED, ORANGE, PEACH, _wrap
from hnl.plots.release_card_trackml import ROWS as TM_ROWS
from hnl.plots.release_card_trackml import RUN, fmt_ours, fmt_published, ours, report
from hnl.trackml.evaluate import REFERENCES as REF

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EVALUATE_PY = "src/hnl/trackml/evaluate.py"
TITLE = "Canto · all results"
SUBTITLE = ("Displaced-track reconstruction in HNL events (ACTS generic detector) and track reconstruction on TrackML "
            "· validation, one-time test, external sample, second population, training")

# (key, header ("\n": explicit line break), header note lines ({n}: seeds and {wp}: the builder setting, both read
# from the result files), group)
COLUMNS = [
    ("CL", "Canto-L", ("{n} · ttbar", "learned merging"), 0),
    ("CH", "Canto-H", ("{n} · ttbar", "walkthrough + merging"), 0),
    ("CB", "Canto-B", ("{n} · ttbar", "learned merging"), 0),
    ("CLH", "Canto-L (HNL)", ("{n} · HNL", "validation only", "learned merging {wp}"), 0),
    ("CLTH", "Canto-L\n(ttbar+HNL)", ("{n} · ttbar+HNL", "validation only", "learned merging {wp}"), 0),
    ("AH", "acorn (HNL)", ("{n} · HNL", "learned merging"), 1),
    ("ATH", "acorn (ttbar+HNL)", ("{n} · ttbar+HNL", "learned merging"), 1),
    ("A", "acorn", ("{n} · ttbar", "learned merging"), 1),
    ("AHcc", "acorn (HNL)\n+ CC", ("{n} · HNL", "prereg. cut"), 2),
    ("Acc", "acorn + CC", ("{n} · ttbar", "prereg. cut = best cut"), 2),
    ("DZ", "DeZoort IN", ("{n} · ttbar", "lowest fake rate"), 3),
    ("DZL", "DeZoort IN-L", ("{n} · ttbar", "lowest fake rate"), 3),
    ("WG", "Wang et al.", ("ACAT 2021 · HNL-trained", "attention GNN + DBSCAN"), 4),
    ("EG", "EggNet", ("Calafiura et al. 2024", "Chan et al. 2025"), 5),
    ("EC", "EC", ("Calafiura et al. 2024",), 5),
    ("BC", "BC-HGNN", ("Liu et al. 2026",), 5),
    ("ECG", "EC-GNN", ("Liu et al. 2026",), 5),
]
GROUPS = ["Canto", "acorn, val-best builder", "acorn, connected components (CC)", "DeZoort et al.",
          "published, HNL", "published, TrackML"]
PUBLISHED = {"WG", "EG", "EC", "BC", "ECG"}
# Wang et al.'s published values (arXiv:2203.08800, p. 4; NUMBERS.md P09-P11): the line of docs/paper_facts.md that
# starts with each prefix (exactly one), cross-checked against the "published" rows of
# results/published_definitions_comparison.csv
PAPER_FACTS = "docs/paper_facts.md"
WANG_LINES = {"all_track": "| Track efficiency, all tracks in the event |",
              "prompt": "| Track efficiency, prompt tracks from the HNL process (W muon) |",
              "displaced": "| Track efficiency, displaced tracks from the HNL process (N daughters) |",
              "fake_rate": "- **Not defined or reported:**"}
TM_COL = {"EggNet": "EG", "EC": "EC", "BC-HGNN": "BC", "EC-GNN": "ECG"}

# model labels in each result file
BS_SHORT = {"CL": "Canto-L", "CH": "Canto-H", "CB": "Canto-B", "AHcc": "acorn (HNL) CC", "Acc": "acorn (ttbar) CC",
            "DZ": "DeZoort IN", "DZL": "DeZoort IN-L"}
CANTO_L_SBD = "Canto-L (frozen: learned merging)"
LM = {"AH": "acorn (HNL) + val-best builder of the full menu (chosen per seed)",
      "ATH": "acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed)",
      "A": "acorn (ttbar) + val-best builder of the full menu (chosen per seed)"}
SB_KEY = {"primary": "eff_primary", "hnl_d0_100_300": "eff_d0_100", "hnl_d0_ge300": "eff_d0_300",
          "w_muons": "eff_w_muons", "ttbar_prompt": "eff_ttbar_prompt", "fake_rate": "fake_rate_hnl"}
PS_MODEL = {"CL": "Canto-L (frozen: learned merging)",
            "AH": "acorn (HNL) val-best (MWS-layer + learned merging)",
            "ATH": "acorn (ttbar+HNL) val-best (MWS-layer + learned merging)",
            "A": "acorn (ttbar) val-best (MWS-layer + learned merging)",
            "AHcc": "acorn (HNL) + CC, preregistered cut (claims' reference)",
            "Acc": "acorn (ttbar) + CC, preregistered cut (= best cut)"}
PD_MODEL = {"CL": "canto_l", "CH": "canto_h", "CB": "canto_b", "AHcc": "acorn_hnl_prereg", "Acc": "acorn_ttbar"}
SP_MODEL = {"CL": "Canto-L (frozen: learned merging)", "CH": "Canto-H (frozen: walkthrough + merging)",
            "CB": "Canto-B (frozen: learned merging)",
            "AH": "acorn (HNL) + MWS-layer + learned merging (val-best acorn (HNL) configuration, full menu)",
            "ATH": "acorn (ttbar+HNL) + MWS-layer + learned merging (val-best acorn (ttbar+HNL) configuration, "
                   "full menu)",
            "A": "acorn (ttbar) + MWS-layer + learned merging (val-best acorn (ttbar) configuration, full menu)",
            "AHcc": "acorn (HNL) + CC, preregistered cut", "Acc": "acorn (ttbar) + CC (preregistered cut = best cut)"}
TS_MODEL = {"CLH": "Canto-L (HNL)", "CLTH": "Canto-L (ttbar+HNL)"}   # training_sample_bins_v1 / _val_mwslmerge
MT_MODEL = {"CL": "Canto-L", "CH": "Canto-H", "CB": "Canto-B", "CLH": "Canto-L (HNL)",
            "CLTH": "Canto-L (ttbar+HNL)", "AH": "acorn (HNL)", "ATH": "acorn (ttbar+HNL)",
            "A": "acorn (ttbar)", "AHcc": "acorn (HNL)", "Acc": "acorn (ttbar)", "DZ": "DeZoort IN",
            "DZL": "DeZoort IN-L"}
TRAIN_SAMPLE = {"CL": "ttbar", "CH": "ttbar", "CB": "ttbar", "CLH": "HNL signal", "CLTH": "ttbar + HNL",
                "AH": "HNL signal", "ATH": "ttbar + HNL", "A": "ttbar",
                "AHcc": "HNL signal", "Acc": "ttbar", "DZ": "ttbar", "DZL": "ttbar"}

# HNL rows of the val and test bands: (title, subtitle, bootstrap quantity, differences quantity, best, format,
# second line, per-seed count in the subtitle)
HNL_ROWS = [
    ("Displaced tracks", "|d0| ≥ 10 mm or r_v ≥ 100 mm", "eff_primary", "primary", "max", "pct2", "sigma", True),
    ("Far-displaced tracks", "|d0| 100–300 mm", "d0_100_300", "hnl_d0_100_300", "max", "pct2", "sigma", True),
    ("Very far-displaced", "|d0| ≥ 300 mm · k/n", "d0_300_inf", "hnl_d0_ge300", "max", "pct1", "kn", True),
    ("Prompt muons", "W → μ in HNL events", "eff_w_muons", "w_muons", "max", "pct2", "sigma", False),
    ("Dense prompt tracks", "{ttbar}, |d0| < 1 mm", "eff_ttbar_prompt", "ttbar_prompt", "max", "pct2", "sigma", False),
    ("Fake-track rate", "HNL events · lower is better", "fake_rate_hnl", "fake_rate", "min", "pct2", "sigma", False),
]
# the same quantities for Canto-L trained on HNL / ttbar+HNL: bootstrap quantity -> (value and sigma columns of
# training_sample_bins_v1.csv, k and n columns: "b:" that file, "s:" training_sample_val_mwslmerge.csv, value column
# of the latter for the cross-check)
TS_Q = {"eff_primary": ("primary_eff", "primary_sigma", "s:eff_primary_k", "s:eff_primary_n", "eff_primary"),
        "d0_100_300": ("d0_100_300_eff", None, "b:d0_100_300_k", "b:d0_100_300_n", "d0_100_300"),
        "d0_300_inf": ("d0_300_inf_eff", None, "b:d0_300_inf_k", "b:d0_300_inf_n", "d0_300_inf"),
        "eff_w_muons": ("w_muons", None, "s:eff_w_muons_k", "s:eff_w_muons_n", "eff_w_muons"),
        "eff_ttbar_prompt": ("ttbar_prompt", None, "s:eff_ttbar_prompt_k", "s:eff_ttbar_prompt_n",
                             "eff_ttbar_prompt"),
        "fake_rate_hnl": ("fake_rate", None, None, None, "fake_rate_hnl")}   # no sigma and no k/n in the sources
# seed term of a one-seed run in the shading test: the seed sd of Canto-L (ttbar) for the quantity, the convention of
# results/training_sample_v1.md (column of training_sample_bins_v1.csv; the |d0| bins have none)
TS_SEED_SD = {"eff_w_muons": "w_muons_seed_sd", "eff_ttbar_prompt": "ttbar_prompt_seed_sd",
              "fake_rate_hnl": "fake_rate_seed_sd"}

# Canto-definition rows of the TrackML throughput band: (title, subtitle, condition, report metric, best, format)
TM_CANTO = [
    ("Target efficiency", "all hits · Canto definition", "full", "ours_eff_target", "max", "tm"),
    ("Target efficiency", "no noise · Canto definition", "nonoise", "ours_eff_target", "max", "tm"),
    ("Target efficiency", "TrackML-1GeV · Canto definition", "1gev", "ours_eff_target", "max", "tm"),
    ("Fake-track rate", "all hits · Canto definition", "full", "ours_fake_rate", "min", "tm"),
    ("Fake-track rate", "no noise · Canto definition", "nonoise", "ours_fake_rate", "min", "tm"),
    ("Fake-track rate", "TrackML-1GeV · Canto definition", "1gev", "ours_fake_rate", "min", "tm"),
    ("TrackML score", "all hits · challenge weights", "full", "P0_trackml_score", "max", "score"),
    ("TrackML score", "no noise · challenge weights", "nonoise", "P0_trackml_score", "max", "score"),
]

# Geometry in inches (one data unit is one inch), font sizes in points.
GEOM = dict(dpi=100, margin=0.5, lw=4.25, cw=2.16, top_gap=3.0, gh=0.55, hh=1.62, bh=0.8, rh=0.98,
            lpad=0.2, cpad=0.16, border=1.4, grid=0.6, sep=1.5, frame=4.0,
            name=16.0, name_lh=0.29, note=10.5, note_lh=0.2, group=12.0,
            rtitle=15.5, rsub=11.5, rsub_lh=0.21, val=18.5, val_dy=0.14, text=15.0, second=11.0, second_dy=-0.25,
            btitle=17.0, bsub=12.5, bsub_lh=0.25, notes=11.5, notes_head=12.5, notes_lh=0.235, notes_gap=0.5, notes_cols=3)
BAND_FILL = "#ECEAE2"
CANTO_FILL = "#F6E7DF"


@dataclass
class Cell:
    v: float | None = None          # value (fraction, or score)
    s: float | None = None          # combined uncertainty (None: not quoted)
    kn: str | None = None           # pooled k/n
    lines: tuple = ()               # text cells: the first line, then muted lines
    pending: bool = False
    src: str = ""
    s_test: float | None = None     # sigma for the shading test only, where the source gives none (not shown)
    dec: int | None = None          # decimals in percent of a published value (as quoted)
    label: str | None = None        # text in place of a value (e.g. "not reported"), muted
    note: str | None = None         # an extra muted line under the uncertainty


@dataclass
class Row:
    title: str
    subs: tuple
    kind: str | None                # "max" / "min"; None for text rows (no shading)
    fmt: str                        # "pct2", "pct1", "tm" (fmt_ours / fmt_published), "score", "text"
    cells: dict
    second: str = "sigma"           # second line: "sigma" or "kn"


@dataclass
class Band:
    title: str
    sub: str


class Csv:
    """A result CSV with the file line of every row (header = line 1)."""

    def __init__(self, rel: str):
        self.rel = rel
        path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        self.df = pd.read_csv(path)
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        if n_lines != len(self.df) + 1:                  # a quoted field with a line break would shift the lines
            raise SystemExit(f"{rel}: {n_lines} lines for {len(self.df)} rows")
        self.df["_line"] = np.arange(len(self.df)) + 2

    def one(self, **eq):
        m = self.df
        for k, v in eq.items():
            m = m[m[k] == v]
        if len(m) != 1:
            raise SystemExit(f"{self.rel}: {eq}: {len(m)} rows")
        return m.iloc[0]

    def src(self, r) -> str:
        return f"{self.rel.replace(RUN, '$RUN')}:{int(r['_line'])}"


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _pct(v: float, d: int = 2) -> str:
    return f"{100 * v:.{d}f} %"


def _n(x) -> str:
    return f"{int(x):,}"


# ---------------------------------------------------------------------------------------------------------- sources
class Sources:
    def __init__(self):
        self.BS = Csv("results/bootstrap_v1.csv")
        self.SB = Csv("results/strong_baselines_v3.csv")
        self.SBD = Csv("results/strong_baselines_v3_differences.csv")
        self.PS = Csv("results/per_sample_v1.csv")
        self.PD = Csv("results/published_definitions_comparison.csv")
        self.SP = Csv("results/second_population_v2.csv")
        self.TS = Csv("results/training_sample_bins_v1.csv")
        self.TSV = Csv("results/training_sample_val_mwslmerge.csv")
        self.fsel = json.load(open(os.path.join(REPO, "results/final_selection_v1.json")))
        self.split_tp = json.load(open(os.path.join(REPO, "configs/trackml/split_tp_v1.json")))
        self.mt = model_table()
        self.tm_run = trackml_record()

    # HNL cells
    def bs(self, table: str, col: str, q: str) -> Cell:
        r = self.BS.one(table=table, short=BS_SHORT[col], quantity=q)
        return Cell(float(r.value), _f(r.combined), f"{int(r.k_pooled)}/{int(r.n_pooled)}", src=self.BS.src(r))

    def lm(self, split: str, col: str, q: str) -> Cell:
        """acorn with its val-best builder: value and sigma from the differences file, cross-checked against (and k/n
        from) strong_baselines_v3.csv."""
        r = self.SBD.one(set=split, block="Canto-L minus acorn", new=CANTO_L_SBD, baseline=LM[col], quantity=q)
        s = self.SB.one(set=split, label=LM[col])
        if abs(float(s[SB_KEY[q]]) - float(r.value_base)) > 1e-9:
            raise SystemExit(f"{col} {split} {q}: {s[SB_KEY[q]]} in {self.SB.src(s)}, {r.value_base} in "
                             f"{self.SBD.src(r)}")
        kn = str(s.k_n_d0_300) if q == "hnl_d0_ge300" else None
        return Cell(float(r.value_base), _f(r.sigma_base), kn, src=f"{self.SBD.src(r)} (k/n {self.SB.src(s)})")

    def canto_check(self, table: str, split: str, q_bs: str, q_sbd: str) -> None:
        """Canto-L in the bootstrap file and in the differences file must agree."""
        a = self.BS.one(table=table, short="Canto-L", quantity=q_bs)
        b = self.SBD.one(set=split, block="Canto-L minus acorn", new=CANTO_L_SBD, baseline=LM["AH"], quantity=q_sbd)
        if abs(float(a.value) - float(b.value_new)) > 1e-9 or abs(float(a.combined) - float(b.sigma_new)) > 1e-9:
            raise SystemExit(f"Canto-L {split} {q_bs}: {self.BS.src(a)} vs {self.SBD.src(b)}")

    def ts(self, col: str, q: str) -> Cell:
        """Canto-L trained on HNL or on ttbar+HNL (validation, learned merging): value (and, for the primary
        population, the combined sigma) from training_sample_bins_v1.csv, k/n from it or from the per-seed file;
        the per-seed file's value must agree. Only one-seed runs are read here (pooled k/n = the seed's k/n)."""
        r = self.TS.one(model=TS_MODEL[col])
        seeds = self.TSV.df[self.TSV.df.model == TS_MODEL[col]]
        if int(r.n_seeds) != 1 or len(seeds) != 1:
            raise SystemExit(f"{TS_MODEL[col]}: {r.n_seeds} seeds, {len(seeds)} per-seed rows")
        v = seeds.iloc[0]
        vcol, scol, kcol, ncol, vcol_s = TS_Q[q]
        if abs(float(r[vcol]) - float(v[vcol_s])) > 1e-9:
            raise SystemExit(f"{col} {q}: {r[vcol]} in {self.TS.src(r)}, {v[vcol_s]} in {self.TSV.src(v)}")
        pick = lambda c: r[c[2:]] if c.startswith("b:") else v[c[2:]]
        kn = f"{int(pick(kcol))}/{int(pick(ncol))}" if kcol else None
        src = self.TS.src(r) + ("" if kcol is None or kcol.startswith("b:") else f" (k/n {self.TSV.src(v)})")
        sigma = _f(r[scol]) if scol else None
        s_test = None
        if sigma is None:                                # shading test only: binomial error (+) seed term
            binom = 0.0
            if kn:
                k, n = int(pick(kcol)), int(pick(ncol))
                binom = float(np.sqrt(k * (n - k) / n) / n)
            seed = _f(self.TS.one(model="Canto-L (ttbar)")[TS_SEED_SD[q]]) if q in TS_SEED_SD else None
            s_test = float(np.hypot(binom, seed or 0.0))
        return Cell(float(r[vcol]), sigma, kn, src=src, s_test=s_test)

    def ps(self, split: str, sset: str, col: str) -> Cell:
        r = self.PS.one(split=split, set=sset, model=PS_MODEL[col])
        return Cell(float(r["mean"]), _f(r.sigma), src=self.PS.src(r))

    def pd_(self, col: str, q: str) -> Cell:
        r = self.PD.one(set="test_external", model=PD_MODEL[col], quantity=q)
        return Cell(float(r["mean"]), _f(r.sigma_total), f"{int(r.k_pooled)}/{int(r.n_pooled)}", src=self.PD.src(r))

    def sp(self, split: str, event_set: str, pop: str, col: str) -> Cell:
        r = self.SP.one(split=split, kind="efficiency", event_set=event_set, population=pop, quantity="all",
                        model=SP_MODEL[col])
        return Cell(float(r.value), _f(r.combined), f"{int(r.k_pooled)}/{int(r.n_pooled)}", src=self.SP.src(r))


def wang_published(PD: Csv) -> dict:
    """Wang et al.'s published values on their evaluation sample, by quantity of the definitions CSV: the one line of
    docs/paper_facts.md that starts with WANG_LINES[q] (value and quoted error as printed, with their decimals),
    cross-checked against the "published" rows of results/published_definitions_comparison.csv; the fake rate is
    "not reported" (paper_facts lists it as not defined or reported; the CSV row has no value)."""
    lines = open(os.path.join(REPO, PAPER_FACTS)).read().splitlines()
    out = {}
    for q, prefix in WANG_LINES.items():
        hits = [i for i, line in enumerate(lines, 1) if line.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"{PAPER_FACTS}: {len(hits)} lines start with {prefix!r}")
        i = hits[0]
        line = lines[i - 1]
        r = PD.one(set="published", model="wang_et_al", quantity=q)
        src = f"{PAPER_FACTS}:{i} (= {PD.src(r)})"
        if q == "fake_rate":
            if "fake rate" not in line or not pd.isna(r["mean"]) or "not" not in str(r.definition):
                raise SystemExit(f"Wang et al.'s fake rate: {PAPER_FACTS}:{i} or {PD.src(r)} changed")
            out[q] = Cell(label="not reported", src=src)
            continue
        m = re.match(re.escape(prefix) + r" (0\.(\d+)) ± (0\.\d+) \|", line)
        if not m:
            raise SystemExit(f"{PAPER_FACTS}:{i}: no 'value ± error' in {line!r}")
        v, sig = float(m.group(1)), float(m.group(3))
        if abs(v - float(r["mean"])) > 1e-12 or abs(sig - float(r.sigma_total)) > 1e-12:
            raise SystemExit(f"Wang et al. {q}: {PAPER_FACTS}:{i} ({v} ± {sig}) != {PD.src(r)}")
        out[q] = Cell(v, sig, dec=len(m.group(2)) - 2, src=src)
    return out


def model_table(rel: str = "results/model_table_v1.md") -> dict:
    """The training table of results/model_table_v1.md ("| model | seeds | training graphs | ..."), by model."""
    rows, hdr = {}, None
    for i, line in enumerate(open(os.path.join(REPO, rel)).read().splitlines(), 1):
        if line.startswith("| model | seeds | training graphs"):
            hdr = [c.strip() for c in line.strip().strip("|").split("|")]
            continue
        if hdr is None or line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = {**dict(zip(hdr, cells)), "src": f"{rel}:{i}"}
    if not rows:
        raise SystemExit(f"{rel}: no training table")
    return rows


def trackml_record() -> dict:
    """The finished training record of the TrackML edge classifier (the last chunk: state done, most steps)."""
    files = sorted(glob.glob(f"{RUN}/gnn/run_*.json"))
    recs = [(json.load(open(f)), f) for f in files]
    done = [(r, f) for r, f in recs if r.get("state") == "done"]
    if not done:
        raise SystemExit(f"{RUN}/gnn: no finished training record")
    r, f = max(done, key=lambda rf: int(rf[0]["global_step"]))
    return {**r, "src": f.replace(RUN, "$RUN")}


def _csv_line(path: str, **eq) -> str:
    c = Csv(path)
    return c.src(c.one(**eq))


def ref_src(path: tuple) -> str:
    """evaluate.py:line of a REFERENCES entry, e.g. ("P4", "EggNet", "efficiency")."""
    lines = open(os.path.join(REPO, EVALUATE_PY)).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith(f'"{path[0]}":'))
    key = path[-1] if path[0] != "P2" else path[2]
    j = next(k for k in range(i, len(lines)) if f'"{key}"' in lines[k])
    return f"{EVALUATE_PY}:{j + 1} REFERENCES{''.join(f'[{p}]' for p in path)}"


def tm_ref_path(cond: str, key: str, name: str) -> tuple:
    if key.startswith("eggnet"):
        metric = key.split("_", 1)[1]
        return ("P4", name, metric) if cond == "1gev" else ("P3", metric)
    return ("P2", cond, name, 0 if key == "liu_eff" else 1)


# ------------------------------------------------------------------------------------------------------------- rows
def build(S: Sources) -> tuple[list, dict]:
    BS, PS, PD = S.BS, S.PS, S.PD
    items: list = []
    ctx: dict = {}
    hnl_cols = ["CL", "CH", "CB", "CLH", "CLTH", "AH", "ATH", "A", "AHcc", "Acc", "DZ", "DZL"]
    fr_star = float(S.fsel["FR_star"])
    ev = lambda split, sset: int(PS.df[(PS.df.split == split) & (PS.df.set == sset)].events.iloc[0])
    B = lambda table: int(BS.df[BS.df.table == table].B.iloc[0])
    ctx.update(fr_star=fr_star, B_val=B("val_ranking_v3"), B_test=B("test_release_card"))
    ctx["n_seeds"] = {c: int(BS.one(table="val_ranking_v3", short=BS_SHORT[c], quantity="eff_primary").n_seeds)
                      for c in BS_SHORT}
    ctx["n_seeds"].update({c: int(S.SB.one(set="val", label=LM[c]).n_seeds) for c in LM})
    ctx["n_seeds"].update({c: int(S.TS.one(model=TS_MODEL[c]).n_seeds) for c in TS_MODEL})
    ctx["wp"] = {}                                     # builder setting of the one-seed runs (header note)
    for c in TS_MODEL:
        m = re.fullmatch(r"mwslmerge ([\d.]+) \(\w+\)", str(S.TS.one(model=TS_MODEL[c]).working_point))
        if not m:
            raise SystemExit(f"{TS_MODEL[c]}: not learned merging ({S.TS.one(model=TS_MODEL[c]).working_point})")
        ctx["wp"][c] = f"τ {m.group(1)}"
    for c in LM:                                       # every acorn seed at MWS-layer + learned merging
        for split in ("val", "test"):
            r = S.SB.one(set=split, label=LM[c])
            fams = re.findall(r"(?:^|\); )(\w+) [\d.,]+ \(", str(r.working_points))
            if fams != ["mwslmerge"] * int(r.n_seeds):
                raise SystemExit(f"{LM[c]} ({split}): {r.working_points}")
    # acorn + CC: the preregistered cut equals the best cut <= FR* (the header note says so)
    for table in ("val_ranking_v3", "test_release_card"):
        for q in ("eff_primary", "fake_rate_hnl"):
            a = BS.one(table=table, short="acorn (ttbar) CC", quantity=q).value
            b = BS.one(table=table, short="acorn (ttbar) CC*", quantity=q).value
            if abs(a - b) > 1e-12:
                raise SystemExit(f"acorn (ttbar) CC: preregistered cut != best cut ({table}, {q})")

    def hnl_band(split: str, table: str, ttbar: str, val_only: bool):
        rows = []
        for title, sub, q, qd, kind, fmt, second, per_seed in HNL_ROWS:
            S.canto_check(table, split, q, qd)
            cells = {c: S.bs(table, c, q) for c in ("CL", "CH", "CB", "AHcc", "Acc")}
            cells.update({c: S.lm(split, c, qd) for c in LM})
            if val_only:                                  # DeZoort and the Canto-L training-sample runs: val only
                cells.update({c: S.bs(table, c, q) for c in ("DZ", "DZL")})
                cells.update({c: S.ts(c, q) for c in TS_MODEL})
            sub = sub.format(ttbar=ttbar)
            if per_seed:
                sub += f" · {_n(BS.one(table=table, short='Canto-L', quantity=q).n_per_seed)} per seed"
            rows.append(Row(title, (sub,), kind, fmt, cells, second))
        return rows

    # A: HNL validation
    items.append(Band("HNL samples · validation",
                      f"pooled HNL val (hnl10_100, hnl15_200; {_n(ev('val', 'hnl_val_pooled'))} events) and ttbar val "
                      f"· mean over the seeds in the header · every model at a track fake rate ≤ FR* = "
                      f"{_pct(fr_star)} where it can reach it · val values of selected configurations are measured on "
                      "the events they were selected on"))
    items += hnl_band("val", "val_ranking_v3", "ttbar val", True)

    # B: HNL one-time test
    items.append(Band("HNL samples · one-time blind test",
                      f"pooled HNL test (hnl10_100_test, hnl15_200_test, hnl15_100_external; "
                      f"{_n(ev('test', 'hnl_test_pooled'))} events) and ttbar_test · frozen models and working points, "
                      "evaluated once · DeZoort was not evaluated on test"))
    items += hnl_band("test", "test_release_card", "ttbar_test", False)
    for sset in ("hnl10_100_test", "hnl15_200_test"):
        cells = {c: S.ps("test", sset, c) for c in PS_MODEL}
        n = PS.one(split="test", set=sset, model=PS_MODEL["CL"]).n_per_seed
        items.append(Row("Displaced tracks", (f"{sset} only · {_n(n)} per seed",), "max", "pct2", cells))

    # C: external sample
    n_ext = ev("test", "hnl15_100_external")
    ctx["n_ext"] = n_ext
    wang = wang_published(PD)
    ctx["wang"] = wang
    items.append(Band("Wang et al.'s evaluation sample · hnl15_100_external",
                      f"m_N = 15 GeV, cτ = 100 mm, {_n(n_ext)} events · test only, never used in training or selection "
                      "· all-track, prompt-muon and HNL-daughter rows under Wang et al.'s definitions (Canto, the CC "
                      "columns and Wang et al.'s published values)"))
    cells = {c: S.ps("test", "hnl15_100_external", c) for c in PS_MODEL}
    n = PS.one(split="test", set="hnl15_100_external", model=PS_MODEL["CL"]).n_per_seed
    items.append(Row("Displaced tracks", (f"|d0| ≥ 10 mm or r_v ≥ 100 mm · {_n(n)} per seed",), "max", "pct2",
                     cells))
    for title, sub, q, kind in [("All tracks", "Wang et al.'s definition", "all_track", "max"),
                                ("Prompt muons", "Wang et al.'s definition: W\u00a0muons, ≥ 5 hits", "prompt", "max"),
                                ("HNL daughters", "Wang et al.'s 'displaced', all |d0|", "displaced", "max"),
                                ("Fake-track rate", "lower is better", "fake_rate", "min")]:
        cells = {c: S.pd_(c, q) for c in PD_MODEL}
        cells["WG"] = wang[q]
        if q == "displaced":                              # their set is another selection of the same particles
            cells["WG"].note = "their displaced selection"
        if q != "fake_rate":
            n0 = PD.one(set="test_external", model="canto_l", quantity=q).n_s0
            sub = f"{sub} · {_n(n0)} per seed"
        items.append(Row(title, (sub,), kind, "pct2", cells))
    r = PD.one(set="test_external", model="acorn_hnl_sstar", quantity="all_track")
    ctx["sstar"] = (float(r["mean"]), PD.src(r))

    # D: second population
    items.append(Band("Second population · not used in any selection",
                      "K0S and Λ daughters (V0 daughters) in the HNL events, photon-conversion e± in ttbar · targets "
                      "pT > 1 GeV, |η| < 4, ≥ 5 hits · every model frozen at its working point"))
    sp_cols = list(SP_MODEL)
    for title, split, event_set, pop, what in [
            ("V0 daughters", "val", "hnl_joint_val", "v0", "HNL val (hnl10_100 + hnl15_200)"),
            ("V0 daughters", "test", "hnl_joint_test", "v0", "HNL test (hnl10_100 + hnl15_200)"),
            ("Conversion e±", "val", "ttbar_val", "conv", "ttbar val"),
            ("Conversion e±", "test", "ttbar_test", "conv", "ttbar_test")]:
        cells = {c: S.sp(split, event_set, pop, c) for c in sp_cols}
        n = S.SP.one(split=split, kind="efficiency", event_set=event_set, population=pop, quantity="all",
                     model=SP_MODEL["CL"]).n_per_seed
        items.append(Row(title, (f"{what} · {_n(n)} per seed",), "max", "pct2", cells))

    # E: TrackML throughput phase. The Canto-definition rows (TM_CANTO) first, then the
    # rows of the throughput card under the references' own code (without the duplicate rates)
    rm = lambda cond: f"{RUN}/test_post/refmetrics/test_{cond}/refmetrics.csv"
    rep_csv = lambda sset, cond: f"{RUN}/test/report_{sset}/metrics_{cond}.csv"
    n_test = int(pd.read_csv(rm("full")).n_events.iloc[0])
    n_train = int(S.tm_run["data_split"][0])
    ctx.update(n_tm_test=n_test, n_tm_train=n_train, n_tm_val=int(S.split_tp["counts"]["val"]),
               tm_seed=S.tm_run["seed"])
    if int(pd.read_csv(rep_csv("test", "full")).n_events.iloc[0]) != n_test:
        raise SystemExit("report_test and the post-freeze recount differ in the number of test events")
    conditions = ("pileup 200 in every condition; 'no noise' removes only the noise hits, TrackML-1GeV keeps only "
                  "the hits of particles with pT > 1 GeV (both using truth)")

    def canto_rows(sset: str) -> list:
        """The Canto-definition rows of a TrackML test set (Canto-L only; "pending" while the report of a condition
        is not written)."""
        rows = []
        for title, sub, cond, metric, kind, fmt in TM_CANTO:
            ve = report(sset, cond, metric, err=True)
            f = rep_csv(sset, cond)
            where = _csv_line(f, name="WP-A", metric=metric) if ve is not None else f.replace(RUN, "$RUN")
            cells = {"CL": Cell(pending=True, src=where) if ve is None else Cell(ve[0], ve[1], src=where)}
            rows.append(Row(title, (sub,) + (("lower is better",) if kind == "min" else ()), kind, fmt, cells))
        return rows

    items.append(Band("TrackML throughput phase · one-time test",
                      f"{conditions} · {n_test} test events · Canto-L retrained on {_n(n_train)} throughput-phase "
                      "events, one seed, working point WP-A · Canto-definition rows: Canto-L only · then each "
                      "reference's rows under its own evaluation code (ported; recounted after the freeze), published "
                      "values on the authors' own events"))
    items += canto_rows("test")
    for title, subs, cond, key, kind, pub_ in TM_ROWS:
        if key == "eggnet_duplication":
            continue
        v, s_ = ours(cond, key)
        cells = {"CL": Cell(v, s_, src=_csv_line(rm(cond), name="WP-A", metric=key))}
        for name, (pv, pe) in pub_.items():
            cells[TM_COL[name]] = Cell(float(pv), float(pe) if pe else None,
                                       src=ref_src(tm_ref_path(cond, key, name)))
        items.append(Row(title, tuple(subs), kind, "tm", cells))
    ctx["dup"] = {c: ours(c, "eggnet_duplication") for c in ("1gev", "full")}

    # G: training
    items.append(Band("Training · edge classifier",
                      "the CC columns use the networks of the acorn columns · last line of the Canto-L cells: the "
                      "TrackML instance (retrained)"))
    mt, tr = S.mt, S.tm_run
    num = lambda s: int(str(s).replace(",", ""))

    def text_row(title: str, sub: str, lines, trackml: str) -> Row:
        """lines(column, model-table row) -> the cell's lines; Canto-L gets the TrackML value as a last line."""
        cells = {}
        for c in hnl_cols:
            m = mt[MT_MODEL[c]]
            extra = (f"TrackML: {trackml}",) if c == "CL" else ()
            cells[c] = Cell(lines=lines(c, m) + extra, src=m["src"] + (f"; {tr['src']}" if c == "CL" else ""))
        return Row(title, (sub,), None, "text", cells)

    items.append(text_row("Training data", "events, one graph per event",
                          lambda c, m: (TRAIN_SAMPLE[c], f"{_n(num(m['training graphs']))} events"), _n(n_train)))
    items.append(text_row("Parameters", "edge classifier",
                          lambda c, m: (_n(num(m["parameters"])),), _n(tr["n_parameters"])))
    items.append(text_row("Training steps", "one graph (TrackML: one window) per step",
                          lambda c, m: (_n(num(m["steps"])), f"{num(m['epochs'])} epochs"), _n(tr["global_step"])))
    return items, ctx


# ------------------------------------------------------------------------------------------------------------ notes
def notes(ctx: dict) -> list[tuple[str, str]]:
    fr = _pct(ctx["fr_star"])
    w = ctx["wang"]
    dup = ctx["dup"]
    P3, P4 = REF["P3"], REF["P4"]
    out = [
        ("Samples and definitions.",
         "ACTS generic detector simulation without pileup: HNL signal samples (m_N, cτ in the sample name) and ttbar. "
         "Track efficiency by double-majority matching: more than half of the track's hits come from the particle "
         "and more than half of the particle's hits are on the track. Targets: pT > 1 GeV, |η| < 4, at least 5 hits. "
         "Displaced tracks: HNL-daughter targets with |d0| ≥ 10 mm or r_v ≥ 100 mm. Fake-track rate: the fraction of "
         "tracks with at least 5 hits that match no particle. TrackML: pileup 200, with the definitions named in "
         "each row."),
        ("Working points.",
         f"HNL samples: every model at a track fake rate ≤ FR* = {fr} on pooled HNL val (preregistered); a model or "
         "seed that cannot reach FR* is at its lowest fake rate. Canto: the builder of each branch chosen on val by "
         "rule R6 with the one-standard-error rule (within one binomial standard error of the best efficiency, the "
         "lowest fake rate); Canto-L selected by rule R8 and frozen before the test (results/final_selection_v1.json). "
         "acorn (HNL), acorn (ttbar+HNL) and acorn: acorn's interaction network with the val-best builder of the full "
         "menu, chosen per seed (MWS-layer + learned merging for every seed). acorn (HNL) + CC and acorn + CC: "
         "connected components at the preregistered cut rule, the reference of the preregistered claims; for acorn it "
         "equals the best cut ≤ FR*. DeZoort IN and IN-L: one seed at the screening budget; they cannot reach FR*. "
         "Canto-L (HNL) and Canto-L (ttbar+HNL): the Canto-L pipeline trained on HNL signal events or on ttbar + HNL, "
         "one seed each, with learned merging at the setting chosen on val by the builder rule at FR* (header); "
         "evaluated on validation only, not on test, the external sample, the second population or TrackML. "
         f"TrackML: working point WP-A, chosen on {ctx['n_tm_val']} val events (the highest target efficiency at a "
         "track fake rate ≤ FR*, the HNL value, not re-tuned; one-standard-error tie rule)."),
        ("Seeds and uncertainties.",
         "HNL samples: means over the training seeds in the column header; k/n pooled over the seeds. Second line of "
         "a cell: the combined uncertainty σ = (σ_boot² + s²/n)^½ of the mean, in percentage points (score units in "
         f"score rows), with σ_boot from an event bootstrap (B = {_n(ctx['B_val'])} on val, {ctx['B_test']} on test) and "
         "s the standard deviation over the n seeds. The |d0| ≥ 300 mm rows show k/n instead (statistics limited). "
         "Canto-L (HNL) and Canto-L (ttbar+HNL): one seed; the σ of their displaced-track row takes the seed "
         "standard deviation of Canto-L (trained on ttbar) as its seed term (results/training_sample_bins_v1.md); "
         "their other rows have no σ in the source and show k/n, or '1 seed, no σ' for the fake rate. TrackML: "
         "Canto-L has one training seed, σ is its event-bootstrap error (B = 500). Published values carry "
         "the uncertainty quoted by their authors; none is shown where none is quoted (Liu et al.; Calafiura et al.'s "
         "fake rates)."),
        ("Shading.",
         "Bold and shaded: the first place of each row, the highest value (the lowest for rates; cells showing the "
         "same value share it), peach in the Canto-L column and grey elsewhere; the Canto-L column is light where "
         "it is not first; published values take part like any other column. A difference below 2σ, σ the two "
         "cells' uncertainties in quadrature, is not resolved. "
         "Paired differences, with smaller uncertainties, are in results/strong_baselines_v3_differences.csv and "
         "results/bootstrap_v1.csv."),
        ("External sample and Wang et al.",
         f"hnl15_100_external is Wang et al.'s evaluation sample (arXiv:2203.08800), {_n(ctx['n_ext'])} events; it "
         "was evaluated once and never used in training or selection. Column Wang et al.: their published values "
         "on this sample (ACAT 2021; attention GNN + DBSCAN trained on HNL events), as reported in the paper "
         f"(docs/paper_facts.md). Definitions: all tracks, every target, whose count matches the one their binomial "
         "error implies; prompt muons, W muons with at least 5 hits, their prompt denominator. HNL daughters: all "
         "daughter targets for our models; Wang et al.'s displaced value comes from their own selection of these "
         "particles (about half of them, not reproducible from the paper), so their cell is shown as reported and "
         "is not a like-for-like comparison. Wang et al. report no fake rate. Of our models only Canto and the CC "
         "columns were evaluated under these definitions. acorn (HNL) + CC at the fixed cut of our earlier "
         f"reproduction gives {ctx['sstar'][0]:.4f} for all tracks, against their {w['all_track'].v:.4f}."),
        ("Second population.",
         "V0 daughters (charged daughters of K0S and Λ decays) exist only in the HNL samples; the external sample "
         "carries no parent labels and is not used for them. acorn (HNL) and acorn (ttbar+HNL) saw V0 daughters in "
         "training; Canto and acorn, trained on ttbar, did not. Conversion e±: targets from photon conversions in "
         "ttbar. No row of this band entered any selection."),
        ("TrackML throughput phase.",
         f"Canto-L is retrained on TrackML ({_n(ctx['n_tm_train'])} throughput-phase training events, one seed, run "
         "canto_l_v1/seed0_x1); no other column was run on TrackML. Three conditions, all at pileup 200: all hits "
         "(the primary condition: training and every choice were made here), no noise (noise hits removed using "
         "truth) and TrackML-1GeV (only the hits of particles with pT > 1 GeV, using truth). The frozen pipeline was "
         f"evaluated once on {ctx['n_tm_test']} test events. Canto-definition rows (Canto-L only): the HNL "
         "definitions above, targets pT > 1 GeV, |η| < 4, at least 5 hits, from the test reports; the TrackML score "
         "is not computed in TrackML-1GeV. Reference rows: recounted after the freeze from the frozen pipeline's test "
         "tracks with a port of the reference's evaluation code (EggNet; Liu et al.), checked against the unmodified "
         "code. Only references that used the throughput phase are compared. Liu et al.'s FR is a model-to-model "
         "comparison, not final track-building performance; Canto-L's is after its final builder. Not shown as rows, "
         f"duplicate rates under the EggNet code: Canto-L {_pct(dup['1gev'][0])} (TrackML-1GeV) and "
         f"{_pct(dup['full'][0])} (full TrackML); EggNet {_pct(P4['EggNet']['duplication'][0])} and "
         f"{_pct(P3['duplication'][0])}; EC {_pct(P4['EC']['duplication'][0])} (TrackML-1GeV)."),
        ("Training.",
         "Edge classifier only (results/model_table_v1.md; TrackML: the run's training record); the metric-learning "
         "graph stage is not included. One step is one event graph on the HNL samples and one training window on "
         "TrackML. acorn (HNL), acorn (ttbar+HNL), Canto-L (HNL) and Canto-L (ttbar+HNL) train on events of the "
         "evaluated HNL samples (their training split); every other model on ttbar only."),
    ]
    return [(h, t.replace(" %", " %").replace(" et al.", " et al.")) for h, t in out]


# ----------------------------------------------------------------------------------------------------------- render
def _dec(x: float, d: int) -> int:
    """Decimals for an uncertainty shown next to a value with d decimals: d, or more until a digit shows."""
    for k in range(d, d + 3):
        if round(x, k) != 0:
            return k
    return d + 2


def cell_text(row: Row, col: str, c: Cell | None) -> tuple[str, tuple]:
    """(first line, muted lines) of a cell."""
    if c is None:
        return "—", ()
    if c.pending:
        return "pending", ()
    if c.label:
        return c.label, ()
    if row.fmt == "text":
        return c.lines[0], tuple(c.lines[1:])
    val, muted = _value_text(row, col, c)
    return val, muted + ((c.note,) if c.note else ())


def _value_text(row: Row, col: str, c: Cell) -> tuple[str, tuple]:
    """(value, uncertainty or k/n line) of a numeric cell."""
    if row.fmt == "score":
        val, d = f"{c.v:.3f}", 3
    elif row.fmt == "tm":
        val = fmt_published(c.v) if col in PUBLISHED else fmt_ours(c.v)
        d = len(val.rstrip("%").split(".")[1]) if "." in val else 0
    else:
        d = c.dec if c.dec is not None else 1 if row.fmt == "pct1" else 2
        val = f"{100 * c.v:.{d}f}%"
    if row.second == "kn":
        return val, ((c.kn,) if c.kn else ())
    if c.s:
        x = c.s if row.fmt == "score" else 100 * c.s
        return val, (f"± {x:.{_dec(x, d)}f}",)
    if col not in PUBLISHED:                             # no sigma in the source: k/n, or say so
        return val, ((c.kn,) if c.kn else ("1 seed, no σ",))
    return val, ()


def _test_sigma(c: Cell) -> float:
    """sigma for the shading test: the cell's own, else its test-only sigma (one-seed runs), else 0 (a published
    value without a quoted uncertainty)."""
    return c.s or c.s_test or 0.0


def best_cols(row: Row) -> set:
    """The first place of the row: the highest value (lowest for rates), and any cell showing the same value."""
    if row.kind is None:
        return set()
    cells = {k: c for k, c in row.cells.items() if c.v is not None and not c.pending}
    if len(cells) < 2:
        return set()
    top = (max if row.kind == "max" else min)(cells, key=lambda k: cells[k].v)
    shown = cell_text(row, top, cells[top])[0]
    return {k for k, c in cells.items() if cell_text(row, k, c)[0] == shown}


_FONTS: dict = {}


def _width(s: str, size: float, family: str = "DejaVu Sans") -> float:
    """Width in inches of `s` at `size` pt, from the font's advance widths (FT2Font, as the Agg renderer lays out
    text; much faster than TextPath for paragraphs)."""
    from matplotlib import ft2font
    from matplotlib.font_manager import FontProperties, findfont
    flags = ft2font.LoadFlags.NO_HINTING if hasattr(ft2font, "LoadFlags") else ft2font.LOAD_NO_HINTING
    if family not in _FONTS:
        _FONTS[family] = ft2font.FT2Font(findfont(FontProperties(family=family)))
    f = _FONTS[family]
    f.set_size(size, 72)
    f.set_text(s, 0.0, flags=flags)
    return f.get_width_height()[0] / 64 / 72


def _wrap_long(text: str, size: float, width: float) -> list[str]:
    """Greedy lines at most `width` inches wide (non-breaking spaces stay inside a line)."""
    lines: list[str] = []
    for word in text.split(" "):
        if lines and _width(lines[-1] + " " + word, size) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def _notes_layout(nts, width: float, st: dict) -> list[list[tuple[str, list[str]]]]:
    """Wrap every note to the column width and fill the note columns in order, balanced by line count."""
    wrapped = [(h, _wrap_long(t, st["notes"], width)) for h, t in nts]
    total = sum(len(l) + 1.6 for _h, l in wrapped)
    cols, cur, acc = [[]], 0, 0.0
    for h, lines in wrapped:
        n = len(lines) + 1.6
        if cols[cur] and acc + n / 2 > total / st["notes_cols"] * (cur + 1) and cur < st["notes_cols"] - 1:
            cols.append([])
            cur += 1
        cols[cur].append((h, lines))
        acc += n
    return cols


def render(items: list, nts: list, out: str, hdr: dict) -> list[str]:
    """hdr: {"n_seeds": {column: seeds}, "wp": {column: builder setting}} for the header notes."""
    st = GEOM
    lw, cw, hh, bh, rh = st["lw"], st["cw"], st["hh"], st["bh"], st["rh"]
    keys = [c[0] for c in COLUMNS]
    W = 2 * st["margin"] + lw + cw * len(COLUMNS)
    x0, x1 = st["margin"], W - st["margin"]
    note_w = (x1 - x0 - (st["notes_cols"] - 1) * st["notes_gap"]) / st["notes_cols"]
    ncols = _notes_layout(nts, note_w - 0.1, st)
    notes_h = max(sum(st["notes_lh"] * (len(l) + 1.6) for _h, l in col) for col in ncols)
    band_lines = {id(it): _wrap_long(it.sub, st["bsub"], x1 - x0 - 2 * st["lpad"]) for it in items
                  if isinstance(it, Band)}                # a long band subtitle takes more lines
    band_h = lambda it: bh + st["bsub_lh"] * (len(band_lines[id(it)]) - 1)
    table_h = hh + sum(band_h(it) if isinstance(it, Band) else rh for it in items)
    H = st["top_gap"] + st["gh"] + table_h + 0.55 + notes_h + 0.5
    fig = plt.figure(figsize=(W, H), dpi=st["dpi"])
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(W / 2, H - 1.25, TITLE, ha="center", va="center", fontsize=58, family="FreeSerif", color=INK)
    ax.text(W / 2, H - 2.15, SUBTITLE, ha="center", va="center", fontsize=15, family="DejaVu Sans", color=MUTED)
    top = H - st["top_gap"] - st["gh"]
    xc = lambda i: x0 + lw + i * cw                      # left edge of column i
    group_of = [c[3] for c in COLUMNS]
    seps = {i for i in range(1, len(COLUMNS)) if group_of[i] != group_of[i - 1]}

    # group labels over the header, each with a rule over its columns
    for g, name in enumerate(GROUPS):
        idx = [i for i, gi in enumerate(group_of) if gi == g]
        a, b = xc(idx[0]) + 0.08, xc(idx[-1] + 1) - 0.08
        ax.text((a + b) / 2, top + 0.3, name, ha="center", va="center", fontsize=st["group"], color=MUTED,
                weight="bold")
        ax.plot([a, b], [top + 0.1, top + 0.1], color=MUTED, lw=1.0)

    # header
    for i, (key, head, note, _g) in enumerate(COLUMNS):
        xt = xc(i) + st["cpad"]
        names = head.split("\n") if "\n" in head else _wrap(head, st["name"], cw - 2 * st["cpad"])
        y = top - 0.3
        for line in names:
            ax.text(xt, y, line, ha="left", va="center", fontsize=st["name"], color=INK,
                    weight="bold" if i == 0 else "normal")
            y -= st["name_lh"]
        y -= 0.03
        n = hdr["n_seeds"].get(key)
        for part in note:
            if n is not None:
                part = part.format(n=f"{n} seed{'s' if n != 1 else ''}", wp=hdr["wp"].get(key, ""))
            for line in _wrap(part, st["note"], cw - 2 * st["cpad"]):
                ax.text(xt, y, line, ha="left", va="center", fontsize=st["note"], color=MUTED)
                y -= st["note_lh"]
    if not any(isinstance(it, Band) for it in items[:1]):
        raise SystemExit("the first item must be a band")

    # blocks: header, then each band's rows
    y = top - hh
    blocks = [(top, y)]                                   # (top, bottom) of the header and of each band's rows
    for it in items:
        if isinstance(it, Band):
            yb = y - band_h(it)
            ax.add_patch(Rectangle((x0, yb), x1 - x0, y - yb, color=BAND_FILL, zorder=0, lw=0))
            ax.text(x0 + st["lpad"], y - 0.25, it.title, ha="left", va="center", fontsize=st["btitle"],
                    weight="bold", color=INK)
            for k, line in enumerate(band_lines[id(it)]):
                ax.text(x0 + st["lpad"], y - 0.59 - k * st["bsub_lh"], line, ha="left", va="center",
                        fontsize=st["bsub"], color=MUTED)
            for yy in (y, yb):
                ax.plot([x0, x1], [yy, yy], color=INK, lw=1.0, zorder=2)
            y = yb
            blocks.append([y, y])
            continue
        yc = y - rh / 2
        # row label
        subs = [s for sub in it.subs for s in _wrap(sub, st["rsub"], lw - 2 * st["lpad"])]
        dy = (len(subs) - 1) * st["rsub_lh"] / 2
        ax.text(x0 + st["lpad"], yc + 0.15 + dy, it.title, ha="left", va="center", fontsize=st["rtitle"],
                weight="bold", color=INK)
        for k, line in enumerate(subs):
            ax.text(x0 + st["lpad"], yc - 0.17 + dy - k * st["rsub_lh"], line, ha="left", va="center",
                    fontsize=st["rsub"], color=MUTED)
        best = best_cols(it)
        texts = {k: cell_text(it, k, it.cells.get(k)) for k in keys}
        n_lines = max(1 + len(m) for _v, m in texts.values())
        for i, key in enumerate(keys):
            xl = xc(i)
            c = it.cells.get(key)
            val, muted = texts[key]
            is_best = key in best
            if is_best:
                ax.add_patch(Rectangle((xl, y - rh), cw, rh, color=PEACH if i == 0 else GREY, zorder=0, lw=0))
            elif i == 0:
                ax.add_patch(Rectangle((xl, y - rh), cw, rh, color=CANTO_FILL, zorder=0, lw=0))
            missing = c is None or c.pending or bool(c.label)
            if it.fmt == "text":
                y1 = yc + (0.14 if n_lines == 2 else 0.24 if n_lines >= 3 else 0.0)
                ax.text(xl + st["cpad"], y1, val, ha="left", va="center", fontsize=st["text"],
                        color=MUTED if missing else INK)
                for k, line in enumerate(muted):
                    ax.text(xl + st["cpad"], y1 - 0.32 - k * 0.22, line, ha="left", va="center",
                            fontsize=st["second"], color=MUTED)
                continue
            # a row with a two-line muted part (value, uncertainty, note) moves every value of the row up
            two = n_lines >= 3
            ax.text(xl + st["cpad"], yc + (0.21 if two else st["val_dy"]), val, ha="left", va="center",
                    fontsize=st["val"] if not missing else st["text"], weight="bold" if is_best else "normal",
                    color=MUTED if missing else INK)
            for k, line in enumerate(muted):
                ax.text(xl + st["cpad"], yc + (-0.11 - 0.2 * k if two else st["second_dy"]), line, ha="left",
                        va="center", fontsize=st["second"], color=MUTED)
        ax.plot([x0, x1], [y - rh, y - rh], color=INK, lw=st["grid"], zorder=2)
        y -= rh
        blocks[-1][1] = y
    ybot = y
    # vertical lines and the highlighted column's frame, block by block (the band rows stay whole)
    for bt, bb in blocks:
        if bt == bb:
            continue
        for i in range(len(COLUMNS) + 1):
            w = st["sep"] if i in seps else st["grid"]
            ax.plot([xc(i), xc(i)], [bb, bt], color=INK, lw=w, zorder=2)
        ax.add_patch(Rectangle((xc(0), bb), cw, bt - bb, fill=False, lw=st["frame"], ec=ORANGE, zorder=5))
    ax.plot([x0, x1], [top - hh, top - hh], color=INK, lw=1.0, zorder=2)
    ax.add_patch(Rectangle((x0, ybot), x1 - x0, top - ybot, fill=False, lw=st["border"], ec=INK, zorder=4))

    # notes, in columns
    for k, col in enumerate(ncols):
        xn = x0 + k * (note_w + st["notes_gap"]) + 0.05
        yn = ybot - 0.55
        for head, lines in col:
            ax.text(xn, yn, head, ha="left", va="top", fontsize=st["notes_head"], weight="bold", color=INK)
            yn -= st["notes_lh"] * 1.25
            for line in lines:
                ax.text(xn, yn, line, ha="left", va="top", fontsize=st["notes"], color=MUTED)
                yn -= st["notes_lh"]
            yn -= st["notes_lh"] * 0.35
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", facecolor=BG, dpi=st["dpi"])
        files.append(f"{out}.{ext}")
    plt.close(fig)
    return files


def dump(items: list, path: str) -> None:
    """Every cell as rendered, with its raw value and source (for checks)."""
    rows, band = [], ""
    for it in items:
        if isinstance(it, Band):
            band = it.title
            continue
        best = best_cols(it)
        for key, *_ in COLUMNS:
            c = it.cells.get(key)
            val, muted = cell_text(it, key, c)
            rows.append({"band": band, "row": it.title, "sub": " / ".join(it.subs), "column": key, "shown": val,
                         "second": " ".join(muted), "best": key in best, "value": None if c is None else c.v,
                         "sigma": None if c is None else c.s, "kn": None if c is None else c.kn,
                         "source": "" if c is None else c.src})
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output stem: OUT.png and OUT.pdf")
    ap.add_argument("--dump", help="also write every cell (shown text, raw value, source) to this CSV")
    a = ap.parse_args(argv)
    S = Sources()
    items, ctx = build(S)
    if a.dump:
        dump(items, a.dump)
    print("\n".join(render(items, notes(ctx), a.out, ctx)))


if __name__ == "__main__":
    main()
