"""Model comparison on the same LLP dataset (pooled HNL val; the test version after the one-time evaluation).

Every model is evaluated at its own working point for one common fake-rate target (pooled-HNL track fake rate <= FR):
a classifier with connected components at its cut from eval_val (hnl.eval.baseline_val.cut_at_fake_rate, the §4
rule), or a track-building family of a trackbuild_val output at the configuration chosen by the builder rule
(hnl.eval.trackbuild_select.select). Plotted: DM efficiency of HNL-daughter targets vs truth |d0| and vs r_v
(preregistered bins; Clopper-Pearson 68 % intervals), and the primary-population / W-muon / ttbar-prompt efficiency
summary with the fake rate; a CSV holds every number.

Spec (JSON list): {"label", "run", "source": "eval_val" | TRACKBUILD_DIR_NAME, "family" (track building only),
                   "color", "marker", "ls"}

    python -m hnl.plots.compare --spec SPEC.json --fr 0.0314 --name NAME [--extra "HNL val, ..."]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta

from hnl.eval.baseline_val import curves, cut_at_fake_rate, populations
from hnl.eval.particles import particle_table
from hnl.eval.trackbuild_select import select
from hnl.plots import style

D0_EDGES = [0.1, 1, 3, 10, 30, 100, 300, 1000]           # mm; the last bin is >= 300 mm (drawn up to 1000)
RV_EDGES = [0, 10, 30, 100, 200, 300, 1000]               # mm; the last bin is >= 300 mm


def _tables(entry: dict):
    d = os.path.join(entry["run"], entry["source"])
    pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
    return pp, pe


def selected(entry: dict, fr: float, info: pd.DataFrame | None = None):
    """(per-particle dm_matched at the entry's working point, the working point's curve row, description)."""
    pp, pe = _tables(entry)
    info = particle_table(np.unique(pe.event_id)) if info is None else info
    c = curves(pp, pe, info)
    if entry["source"] == "eval_val":
        cut, ok = cut_at_fake_rate(c, fr)
        key, wp = "cut", f"CC cut {cut:.2f}" + ("" if ok else " (FR unreachable: minimum-FR cut)")
        sel = pp[np.isclose(pp.cut, cut, atol=1e-6)]
        row = c.loc[cut]
    else:
        cfgs = json.load(open(os.path.join(entry["run"], entry["source"], "configs.json")))
        tab, none_ok = select(c, cfgs, fr)
        fam = entry["family"]
        if fam in none_ok or fam not in set(tab.family):
            raise ValueError(f"{entry['label']}: family {fam} has no configuration with fake rate <= {fr}")
        params = tab.set_index("family").loc[fam, "params"]
        k = [i for i, cfg in enumerate(cfgs) if cfg[0] == fam and ",".join(str(x) for x in cfg[1:]) == params][0]
        key, wp = "config", f"{fam} {params}"
        sel = pp[pp.config == k]
        row = c.loc[k]
    return sel[["event_id", "particle_id", "dm_matched"]], row, wp


def binned(sel: pd.DataFrame, info: pd.DataFrame, var: str, edges) -> pd.DataFrame:
    t = sel.merge(info, on=["event_id", "particle_id"], how="inner")
    t = t[populations(t)["hnl_targets"]]
    x = t.d0.abs() if var == "absd0" else t.rv
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi_eff = np.inf if hi == edges[-1] else hi
        b = t[(x >= lo) & (x < hi_eff)]
        n, k = len(b), int(b.dm_matched.sum())
        lo_ci = beta.ppf(0.16, k, n - k + 1) if k > 0 else 0.0
        hi_ci = beta.ppf(0.84, k + 1, n - k) if k < n else 1.0
        rows.append({"lo": lo, "hi": hi, "n": n, "eff": k / n if n else np.nan, "err_lo": (k / n - lo_ci) if n else 0,
                     "err_hi": (hi_ci - k / n) if n else 0})
    return pd.DataFrame(rows)


def plot(spec: list, fr: float, name: str, extra: str | None, subdir: str = "compare"):
    pes = [_tables(e)[1] for e in spec]
    ev = np.unique(np.concatenate([np.unique(pe.event_id) for pe in pes]))
    info = particle_table(ev)
    rows, curves_ = [], {}
    for e in spec:
        sel, row, wp = selected(e, fr, info)
        for var, edges in (("absd0", D0_EDGES), ("rv", RV_EDGES)):
            curves_[(e["label"], var)] = binned(sel, info, var, edges)
        rows.append({"label": e["label"], "working_point": wp, "eff_primary": row.get("eff_primary"),
                     "eff_hnl_targets": row.get("eff_hnl_targets"), "eff_w_muons": row.get("eff_w_muons"),
                     "eff_ttbar_prompt": row.get("eff_ttbar_prompt"), "fake_rate_hnl": row.get("fake_rate_hnl")})
    summ = pd.DataFrame(rows)
    out_dir = os.path.join(style.FIG_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    summ.to_csv(os.path.join(out_dir, f"{name}_summary.csv"), index=False)
    pd.concat([df.assign(label=lab, variable=var) for (lab, var), df in curves_.items()]).to_csv(
        os.path.join(out_dir, f"{name}_binned.csv"), index=False)
    files = []
    for var, edges, xlab in (("absd0", D0_EDGES, r"Truth $|d_0|$ [mm]"), ("rv", RV_EDGES, r"Truth $r_v$ [mm]")):
        fig, ax = style.new(figsize=(10, 8))
        nm = len(spec)
        nb = len(edges) - 1
        lows = []
        for i, e in enumerate(spec):
            df = curves_[(e["label"], var)]
            xc = np.arange(nb) + 0.6 * (i - (nm - 1) / 2) / max(nm, 1)     # categorical bins, small offsets
            ax.errorbar(xc, df.eff, yerr=[df.err_lo, df.err_hi], color=e.get("color"), marker=e.get("marker", "o"),
                        ls=e.get("ls", "-"), lw=1.5, ms=6, capsize=2, label=e["label"])
            lows.append(np.nanmin(df.eff - df.err_lo))
        names = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-2], edges[1:-1])] + [rf"$\geq${edges[-2]:g}"]
        ax.set_xticks(np.arange(nb))
        ax.set_xticklabels(names, fontsize=14)
        ax.set_xlim(-0.6, nb - 0.4)
        ymin = max(0.0, np.floor((min(lows) - 0.02) * 20) / 20)
        ax.set_ylim(ymin, 1.0 + 0.3 * (1.0 - ymin))           # headroom for the label
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        style.axis_labels(ax, xlab, "Track efficiency (DM), HNL-daughter targets")
        style.label(ax, extra=(extra or "") + f"\nworking point: pooled-HNL fake rate $\\leq$ {fr:.4f}")
        ax.legend(loc="upper center", fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.13), ncol=2)
        files += style.save(fig, f"{name}_eff_vs_{var}", subdir)
    # summary: primary population, W muons, ttbar prompt
    fig, ax = style.new(figsize=(9, 6.5))
    y = np.arange(len(summ))[::-1]
    for col, mk, lab in (("eff_primary", "o", r"primary ($|d_0|\geq10$ mm or $r_v\geq100$ mm)"),
                         ("eff_w_muons", "s", "W muons"), ("eff_ttbar_prompt", "^", r"ttbar prompt ($|d_0|<1$ mm)")):
        ax.plot(summ[col], y, mk, ms=8, label=lab, ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels(summ.label, fontsize=10)
    lo = float(np.nanmin(summ[["eff_primary", "eff_w_muons", "eff_ttbar_prompt"]].to_numpy(dtype=float)))
    ax.set_xlim(min(0.80, np.floor((lo - 0.02) * 20) / 20), 1.005)          # every point on the axis
    ax.set_ylim(-0.6, len(summ) + 1.4)                       # an empty band on top for the label
    style.axis_labels(ax, "Track efficiency (DM)", "")
    style.label(ax, extra=(extra or "") + f"\nfake rate $\\leq$ {fr:.4f}", loc="upper left")
    ax.legend(loc="upper left", fontsize=10, frameon=False, bbox_to_anchor=(0.0, 1.16), ncol=3)
    files += style.save(fig, f"{name}_summary", subdir)
    return summ, files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--fr", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", default=None)
    a = ap.parse_args(argv)
    summ, files = plot(json.load(open(a.spec)), a.fr, a.name, a.extra)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))
    print("\n".join(files))


if __name__ == "__main__":
    main()
