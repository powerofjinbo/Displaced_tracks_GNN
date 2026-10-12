"""Model ranking on the pooled HNL val set at one working point (every model, one methodology).

Spec (JSON list), one item per model: {"label", "group", "entries": [one per seed]}; each entry
  {"run", "source": "eval_val", "rule": "s4" | "best"}   connected components at the run's own cut: the section-4 rule
                                                           (largest fake rate <= FR*, ties to the lowest cut; unreachable
                                                           -> minimum-rate cut, flagged), or the highest primary efficiency
                                                           with fake rate <= FR* (sensitivity; minimum-rate cut if none)
  {"run", "source": [GRID, ...], "family": NAME | "*"}    a track-building family of the first listed trackbuild_val* grid
                                                           holding it, at the builder rule's configuration (fake rate <=
                                                           FR*, highest primary efficiency); "*" = the best family over
                                                           the grids (each family from the first grid holding it; val
                                                           selection). --tie-se: the one-standard-error rule for both
                                                           choices (DECISIONS.md 2026-10-04 20:18 UTC)
Per model: mean and seed standard deviation of the primary / HNL-daughter / W-muon / ttbar-prompt efficiency and the
pooled-HNL fake rate; HNL-daughter targets with |d0| in [100, 300) mm and >= 300 mm pooled over seeds (k / n).
Writes OUT/ranking.csv, OUT/ranking_entries.csv, OUT/ranking.md and the figures ranking_bar and ranking_scatter.

    python -m hnl.eval.ranking --spec SPEC.json --fr FR --out DIR [--name NAME] [--extra "label text"]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import curves, cut_at_fake_rate
from hnl.eval.particles import particle_table
from hnl.eval.trackbuild_select import select, tie_best
from hnl.plots import style

KEYS = ("eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt", "fake_rate_hnl")
D0_BINS = ((100.0, 300.0), (300.0, np.inf))


def _cc(entry: dict, fr: float, info: pd.DataFrame):
    d = os.path.join(entry["run"], "eval_val")
    pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
    c = curves(pp, pe, info)
    if entry.get("rule", "s4") == "best":
        ok = c[c.fake_rate_hnl <= fr + 1e-12]
        if ok.empty:
            cut, reach = float(c.fake_rate_hnl.idxmin()), False
        else:
            cut = float(ok.sort_values(["eff_primary", "fake_rate_hnl"], ascending=[False, True]).index[0])
            reach = True
    else:
        cut, reach = cut_at_fake_rate(c, fr)
    sel = pp[np.isclose(pp.cut, cut, atol=1e-6)]
    return sel, c.loc[cut], f"CC {cut:.2f}" + ("" if reach else " (FR* unreachable: minimum-rate cut)"), set(pe.event_id)


def _grid(entry: dict, fr: float, info: pd.DataFrame, tie_se: bool = False):
    fam = entry.get("family", "*")
    seen, cands = set(), []                    # families already taken from an earlier grid; candidate rows
    for g in entry["source"]:
        d = os.path.join(entry["run"], g)
        if not os.path.exists(os.path.join(d, "configs.json")):
            continue
        cfgs = json.load(open(os.path.join(d, "configs.json")))
        fams = {c[0] for c in cfgs} - seen
        want = fams - {"cc"} if fam == "*" else fams & {fam}
        seen |= fams
        if not want:
            continue
        pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
        pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
        c = curves(pp, pe, info)
        tab, _ = select(c, cfgs, fr, tie_se=tie_se)
        for r in tab[tab.family.isin(want)].to_dict("records"):
            k = [i for i, cfg in enumerate(cfgs)
                 if cfg[0] == r["family"] and ",".join(str(x) for x in cfg[1:]) == r["params"]][0]
            cands.append({**r, "grid": g, "k": k, "pp": pp, "c": c, "events": set(pe.event_id)})
    if not cands:
        raise ValueError(f"{entry['run']}: no grid of {entry['source']} has family {fam} at fake rate <= {fr}")
    t = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pp", "c", "events")} for r in cands])
    if tie_se:
        i = tie_best(t, float(t.loc[t.eff_primary.idxmax(), "n_primary"]))
    else:
        i = t.sort_values(["eff_primary", "fake_rate_hnl"], ascending=[False, True]).index[0]
    b = cands[int(i)]
    return b["pp"][b["pp"].config == b["k"]], b["c"].loc[b["k"]], f"{b['family']} {b['params']} ({b['grid']})", b["events"]


def entry_metrics(entry: dict, fr: float, info: pd.DataFrame, tie_se: bool = False) -> tuple[dict, pd.DataFrame]:
    if entry["source"] == "eval_val":
        sel, row, wp, events = _cc(entry, fr, info)
    else:
        sel, row, wp, events = _grid(entry, fr, info, tie_se)
    t = sel[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    from hnl.eval.baseline_val import populations
    t = t[populations(t)["hnl_targets"]]
    out = {"run": entry["run"], "working_point": wp, "n_events": len(events), **{k: float(row[k]) for k in KEYS}}
    for lo, hi in D0_BINS:
        b = t[(t.d0.abs() >= lo) & (t.d0.abs() < hi)]
        out[f"n_d0_{lo:g}"], out[f"k_d0_{lo:g}"] = int(len(b)), int(b.dm_matched.sum())
    return out, t


def rank(spec: list, fr: float, tie_se: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = spec[0]["entries"][0]
    d = os.path.join(first["run"], "eval_val" if first["source"] == "eval_val" else first["source"][0])
    events = np.unique(pd.read_parquet(os.path.join(d, "per_event.parquet")).event_id)
    info = particle_table(events)
    rows, ent = [], []
    for m in spec:
        es = []
        for e in m["entries"]:
            r, _ = entry_metrics(e, fr, info, tie_se)
            if r["n_events"] != len(events):
                raise ValueError(f"{e['run']}: {r['n_events']} val events, the first model has {len(events)}")
            es.append(r)
            ent.append({"label": m["label"], **r})
        df = pd.DataFrame(es)
        row = {"label": m["label"], "short": m.get("short", ""), "group": m.get("group", ""), "n_seeds": len(df),
               "working_points": "; ".join(df.working_point)}
        for k in KEYS:
            row[k] = df[k].mean()
            row[k + "_sd"] = df[k].std(ddof=1) if len(df) > 1 else np.nan
        for lo, _ in D0_BINS:
            n, k = df[f"n_d0_{lo:g}"].sum(), df[f"k_d0_{lo:g}"].sum()
            row[f"eff_d0_{lo:g}"] = k / n if n else np.nan
            row[f"k_n_d0_{lo:g}"] = f"{k}/{n}"
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("eff_primary", ascending=False).reset_index(drop=True)
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    return tab, pd.DataFrame(ent)


def markdown(tab: pd.DataFrame, fr: float) -> str:
    def pm(r, k, d=4):
        v = f"{r[k]:.{d}f}"
        return v + (f" ± {r[k + '_sd']:.{d}f}" if r["n_seeds"] > 1 and np.isfinite(r[k + "_sd"]) else "")
    lines = [f"Pooled HNL val, fake-rate target FR* = {fr:.6f}; mean ± seed sd over the listed seeds.", "",
             "| rank | model | seeds | primary eff. | HNL daughters | \\|d0\\| 100-300 mm | \\|d0\\| >= 300 mm | W muons | "
             "ttbar prompt | fake rate |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r['rank']} | {r.label} | {r.n_seeds} | {pm(r, 'eff_primary')} | {pm(r, 'eff_hnl_targets')} | "
                     f"{r['eff_d0_100']:.3f} | {r['k_n_d0_300']} | {pm(r, 'eff_w_muons')} | {pm(r, 'eff_ttbar_prompt')} | "
                     f"{pm(r, 'fake_rate_hnl')} |")
    return "\n".join(lines) + "\n"


GROUP_COLORS = {"DeZoort": "#8c564b", "acorn": "#7f7f7f", "acorn + Canto builder": "#1f77b4",
                "ladder (preview)": "#bcbd22", "Canto (preview)": "#e377c2", "Canto (full budget)": "#d62728",
                "HNL-trained triplet (full budget)": "#2ca02c"}


def figures(tab: pd.DataFrame, fr: float, name: str, extra: str | None, subdir: str = "ranking") -> list[str]:
    files = []
    t = tab.iloc[::-1].reset_index(drop=True)                       # best on top
    fig, ax = style.new(figsize=(11, 0.42 * len(t) + 2.6))
    y = np.arange(len(t))
    col = [GROUP_COLORS.get(g, "#17becf") for g in t.group]
    ax.barh(y, t.eff_primary, color=col, alpha=0.85, height=0.62)
    err = np.where(t.n_seeds > 1, t.eff_primary_sd.fillna(0), 0)
    ax.errorbar(t.eff_primary, y, xerr=err, fmt="none", ecolor="k", capsize=2, lw=1)
    ax.plot(t.eff_ttbar_prompt, y, "^", color="#d62728", ms=6, label="ttbar prompt", ls="none")
    ax.plot(t.eff_w_muons, y, "s", color="#ff7f0e", ms=5, label="W muons", ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}  ({n} seed{'s' if n > 1 else ''})" for lab, n in zip(t.label, t.n_seeds)], fontsize=9)
    lo = float(np.nanmin(np.r_[t.eff_primary, t.eff_ttbar_prompt]))
    ax.set_xlim(max(0.0, np.floor((lo - 0.02) * 20) / 20), 1.005)
    ax.set_ylim(-0.7, len(t) + 1.8)
    style.axis_labels(ax, "Primary-population track efficiency (bars), DM", "")
    style.label(ax, extra=(extra or "") + f"\nfake rate $\\leq$ {fr:.4f} (pooled HNL val)", loc="upper left")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    files += style.save(fig, f"{name}_bar", subdir)
    fig, ax = style.new(figsize=(10, 7.5))
    inside = tab[tab.fake_rate_hnl <= 1.25 * fr]
    for _, r in inside.iterrows():
        c = GROUP_COLORS.get(r.group, "#17becf")
        ax.errorbar(r.fake_rate_hnl, r.eff_primary, xerr=r.fake_rate_hnl_sd if r.n_seeds > 1 else None,
                    yerr=r.eff_primary_sd if r.n_seeds > 1 else None, fmt="o", color=c, ms=6, capsize=2)
        if not r.group.startswith("ladder"):
            ax.annotate(r.get("short", "") or str(r["rank"]), (r.fake_rate_hnl, r.eff_primary),
                        textcoords="offset points", xytext=(5, 7 if r["rank"] % 2 else -11), fontsize=8)
    ax.axvline(fr, color="0.5", ls="--", lw=1)
    off = tab[tab.fake_rate_hnl > 1.25 * fr]
    if len(off):
        ax.text(0.98, 0.80, "off scale:\n" + "\n".join(f"{r.label.split(' + ')[0]} (FR {r.fake_rate_hnl:.3f}, "
                                                          f"eff. {r.eff_primary:.3f})" for _, r in off.iterrows()),
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    for g, c in GROUP_COLORS.items():
        if g in set(inside.group):
            ax.plot([], [], "o", color=c, label=g)
    lo = float(np.nanmin(inside.eff_primary - inside.eff_primary_sd.fillna(0)))
    ax.set_xlim(0.0, 1.25 * fr)
    ax.set_ylim(max(0.0, np.floor((lo - 0.01) * 50) / 50), 1.0 + 0.32 * (1.0 - lo))
    style.axis_labels(ax, "Track fake rate (pooled HNL val)", "Primary-population efficiency (DM)")
    style.label(ax, extra=(extra or "") + "\ndashed: FR*; ladder rungs unlabelled (see the table)")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
    files += style.save(fig, f"{name}_scatter", subdir)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--fr", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ranking")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--tie-se", action="store_true", help="one-standard-error rule for the builder choices (v2 on)")
    ap.add_argument("--fig-dir", default="ranking", help="subdirectory of figures/ for the two figures")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    tab, ent = rank(json.load(open(a.spec)), a.fr, a.tie_se)
    tab.to_csv(os.path.join(a.out, f"{a.name}.csv"), index=False)
    ent.to_csv(os.path.join(a.out, f"{a.name}_entries.csv"), index=False)
    md = markdown(tab, a.fr)
    open(os.path.join(a.out, f"{a.name}.md"), "w").write(md)
    print(md)
    print("\n".join(figures(tab, a.fr, a.name, a.extra, subdir=a.fig_dir)))


if __name__ == "__main__":
    main()
