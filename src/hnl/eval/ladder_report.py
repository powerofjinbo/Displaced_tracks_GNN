"""Ladder read-out table for the Stage-1 decision rules R1-R8 (DECISIONS.md, val only).

For each run directory: CC at the run's own cut at the fake-rate target (eval_val, 0.01 grid, largest fake rate <=
the target; hnl.eval.baseline_val.cut_at_fake_rate, flagged if unreachable) and every track-builder family of each
of its trackbuild_val* outputs at the configuration selected by the builder rule (pooled-HNL fake rate <= the target,
highest primary efficiency; hnl.eval.trackbuild_select.select), with primary / HNL-daughter / W-muon / ttbar-prompt
efficiency and the fake rates. Each grid is reported as it is (one row per output directory and family): grids made
by different code versions are never merged. Next to them, per run:
  * dD: edge TPR at |d0| 100-300 mm minus 0.1-1 mm of HNL-daughter edges at the run's own cut giving the target edge
    FPR (RUN/edge_tpr*_efpr.json of hnl.eval.edge_tpr --efpr / --wp; with --e-star only a file for that target);
  * dT: truncation-probe TPR at depth 4 (else the deepest depth) minus depth 0 (RUN/truncation_probe/summary.csv).
Missing inputs leave blank cells.

    python -m hnl.eval.ladder_report RUN [RUN ...] (--fr-star FR | --wp WORKING_POINTS_JSON) --out OUT_PREFIX
        [--names trackbuild_val,trackbuild_val_mwslayer] [--e-star E]
writes OUT_PREFIX.csv and OUT_PREFIX.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import curves, cut_at_fake_rate, load_run
from hnl.eval.particles import particle_table_cached
from hnl.eval.trackbuild_select import COLS, select

MD_COLS = ["run", "source", "family", "params", "eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt",
           "fake_rate_hnl", "fake_rate_ttbar", "dD", "dT"]


def run_label(run: str) -> str:
    """'<tag>/<NAME>/arm<A>_s<S>' for runs/stage1_<tag>/<NAME>/arm<A>_s<S>, 'baseline_<tag>/arm<A>_s<S>' for a
    baseline run, else the last two path components."""
    p = os.path.normpath(os.path.abspath(run))
    m = re.search(r"/stage1_([^/]+)/([^/]+)/(arm[iv]+_s\d+)$", p)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m = re.search(r"/(baseline_[^/]+)/(arm[iv]+_s\d+)$", p)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "/".join(p.split("/")[-2:])


def delta_t(summary: pd.DataFrame) -> tuple[float, float]:
    """(dT, depth): TPR at depth 4 (else the deepest depth > 0) minus TPR at depth 0 of a truncation-probe summary."""
    s = summary.set_index("depth").tpr if "depth" in summary else summary.tpr
    s.index = s.index.astype(int)
    if 0 not in s.index or s.index.max() <= 0:
        return np.nan, np.nan
    k = 4 if 4 in s.index else int(s.index.max())
    return float(s.loc[k] - s.loc[0]), float(k)


def read_dd(run: str, e_star: float | None = None) -> dict:
    """dD of the newest RUN/edge_tpr*_efpr.json (only files for target e_star if given); {} if none."""
    best = None
    for f in glob.glob(os.path.join(run, "edge_tpr*_efpr.json")):
        m = json.load(open(f))
        if e_star is not None and abs(float(m.get("target_efpr", np.nan)) - e_star) > 1e-12:
            continue
        if best is None or os.path.getmtime(f) > best[0]:
            best = (os.path.getmtime(f), m)
    if best is None:
        return {}
    m = best[1]
    return {"dD": m.get("dD"), "dD_cut": m.get("cut"), "dD_efpr": m.get("efpr"), "dD_reachable": m.get("reachable")}


def read_dt(run: str) -> dict:
    f = os.path.join(run, "truncation_probe", "summary.csv")
    if not os.path.exists(f):
        return {}
    dt, k = delta_t(pd.read_csv(f))
    return {"dT": dt, "dT_depth": k}


def trackbuild_dirs(run: str, names: list[str] | None) -> list[str]:
    cands = names if names else sorted(os.path.basename(d) for d in glob.glob(os.path.join(run, "trackbuild_val*"))
                                       if os.path.isdir(d))
    need = ("per_particle.parquet", "per_event.parquet", "configs.json")
    return [n for n in cands if all(os.path.exists(os.path.join(run, n, x)) for x in need)]


def run_rows(run: str, fr: float, names: list[str] | None, e_star: float | None = None) -> list[dict]:
    rows = []
    if os.path.exists(os.path.join(run, "eval_val", "per_particle.parquet")):
        pp, pe = load_run(run)
        c = curves(pp, pe, particle_table_cached(np.unique(pe.event_id)))
        cut, reachable = cut_at_fake_rate(c, fr)
        rows.append({"source": "eval_val", "family": "CC (own cut)",
                     "params": f"{cut:.2f}" + ("" if reachable else " (target unreachable: minimum-rate cut)"),
                     "reachable": bool(reachable), **{k: c.loc[cut].get(k, np.nan) for k in COLS}})
    for name in trackbuild_dirs(run, names):
        d = os.path.join(run, name)
        pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
        pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
        cfgs = json.load(open(os.path.join(d, "configs.json")))
        sel, none_ok = select(curves(pp, pe, particle_table_cached(np.unique(pe.event_id))), cfgs, fr)
        for r in sel.to_dict("records"):
            rows.append({"source": name, "reachable": True, **r})
        for fam in none_ok:
            rows.append({"source": name, "family": fam, "params": "no configuration <= target", "reachable": False})
    extra = {**read_dd(run, e_star), **read_dt(run)}
    for r in rows:
        r.update({"run": run_label(run), **extra})
    if not rows:
        rows.append({"run": run_label(run), "source": "", "family": "", "params": "", **extra})
    return rows


def markdown(tab: pd.DataFrame) -> str:
    cols = [c for c in MD_COLS if c in tab]
    fmt = lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (
        f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in tab[cols].itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fr-star", type=float, default=None)
    ap.add_argument("--wp", default=None, help="working_points.json (FR_star; e_star for dD files)")
    ap.add_argument("--e-star", type=float, default=None, help="only dD files computed for this target edge FPR")
    ap.add_argument("--names", default=None, help="comma-separated trackbuild_val* directories (default: all)")
    ap.add_argument("--out", required=True, help="output prefix: OUT.csv and OUT.md")
    a = ap.parse_args(argv)
    wp = json.load(open(a.wp)) if a.wp else None
    if wp is not None:
        fr = float(wp["FR_star"])
        if a.fr_star is not None and abs(a.fr_star - fr) > 1e-12:
            raise SystemExit(f"--fr-star {a.fr_star} differs from the working points' FR* {fr}")
        e_star = a.e_star if a.e_star is not None else float(wp["e_star"])
    elif a.fr_star is not None:
        fr, e_star = a.fr_star, a.e_star
    else:
        raise SystemExit("give --wp or --fr-star")
    names = a.names.split(",") if a.names else None
    rows = []
    for run in a.runs:
        if not os.path.isdir(run):
            print(f"WARNING: no run directory {run}", file=sys.stderr)
            continue
        rows += run_rows(run, fr, names, e_star)
    tab = pd.DataFrame(rows)
    for c in ("dD", "dD_cut", "dD_efpr", "dD_reachable", "dT", "dT_depth"):      # stable columns, blank if missing
        if c not in tab:
            tab[c] = np.nan
    first = ["run", "source", "family", "params", "reachable"]
    tab = tab[[c for c in first if c in tab] + [c for c in tab if c not in first]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tab.to_csv(a.out + ".csv", index=False)
    with open(a.out + ".md", "w") as f:
        f.write(f"Fake-rate target {fr:.6f} ({a.wp or '--fr-star'}); dD target edge FPR "
                f"{e_star if e_star is not None else 'any'}\n\n")
        f.write(markdown(tab))
    pd.set_option("display.width", 250)
    print(tab[[c for c in MD_COLS if c in tab]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
