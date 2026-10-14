"""Proper lifetime of the N in every generator-level run of the HNL card (docs/hnl_points_spec.md section 10).

Reads the per-event generator truth of hnl_pythia.cc runs (census parquet of slurm/acts/produce_hnl.sbatch SIMULATE=0,
column n_tau) and of hnl_lifetime_check.cc runs (tau_*.csv), and writes per run: events, mean tau with its standard
error, the ratio to c tau0, a Kolmogorov-Smirnov test against the exponential, and for the check runs the number of
events in which a copy of the N carries a tau different from the process record. Combinations: the full-event runs
(inverse-variance mean, chi2 of their scatter, with and without the run that raised the question).

    python scripts/acts/hnl_lifetime_summary.py --run LABEL=KIND:CTAU0:GLOB [--run ...] --first-run LABEL --out OUT.json
KIND: census (gen_driver_*_events.parquet) or check (tau_*.csv; process-only runs have no event-record copy).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats


def load(kind: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files for {pattern}")
    if kind == "census":
        t = pd.concat([pd.read_parquet(f, columns=["event", "n_tau", "n_tries"]) for f in files], ignore_index=True)
        return t.rename(columns={"n_tau": "tau"})
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t["tau"] = np.where(t.n_evt > 0, t.tau_last, t.tau_proc)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--first-run", help="the run that raised the question (left out of the second combination)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for spec in a.run:
        label, rest = spec.split("=", 1)
        kind, ctau0, pattern = rest.split(":", 2)
        ctau0 = float(ctau0)
        t = load(kind, pattern)
        if t.event.duplicated().any():
            raise SystemExit(f"{label}: duplicate events")
        tau = t.tau.to_numpy(float)
        m, se = float(tau.mean()), float(tau.std(ddof=1) / np.sqrt(len(tau)))
        r = {"label": label, "kind": kind, "pattern": pattern, "ctau0": ctau0, "events": int(len(tau)),
             "mean": m, "se": se, "ratio": m / ctau0, "ratio_se": se / ctau0, "z": (m - ctau0) / se,
             "ks_p_exponential": float(stats.kstest(tau / ctau0, "expon").pvalue),
             "n_tries_gt1": int((t.n_tries > 1).sum()) if "n_tries" in t else int((t.tries > 1).sum())}
        if kind == "check":
            full = t.n_evt > 0
            r["full_event"] = bool(full.all())
            if full.any():
                r["copies_with_other_tau"] = int(((t.tau_last != t.tau_proc) | (t.n_tau_distinct > 1))[full].sum())
                r["events_with_two_or_more_copies"] = int((t.n_evt[full] >= 2).sum())
                r["undecayed"] = int((t.decayed[full] == 0).sum())
                r["max_abs_vertex_minus_tau_mm"] = float(np.max(np.abs(t.ctau_vertex[full] - t.tau_last[full])))
        else:
            r["full_event"] = True
        rows.append(r)
    out = {"runs": rows}
    for name, keep in (("full_event_all", lambda r: True), ("full_event_without_first", lambda r: r["label"] != a.first_run)):
        sel = [r for r in rows if r["full_event"] and keep(r)]
        if len(sel) < 2 or len({r["ctau0"] for r in sel}) != 1:
            continue
        w = np.array([1 / r["se"] ** 2 for r in sel])
        x = np.array([r["mean"] for r in sel])
        mu = float((w * x).sum() / w.sum())
        se = float(1 / np.sqrt(w.sum()))
        chi2 = float((w * (x - mu) ** 2).sum())
        out[name] = {"runs": [r["label"] for r in sel], "events": int(sum(r["events"] for r in sel)), "mean": mu,
                     "se": se, "z": (mu - sel[0]["ctau0"]) / se, "chi2": chi2, "dof": len(sel) - 1,
                     "p": float(stats.chi2.sf(chi2, len(sel) - 1))}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("| run | events | mean tau (mm) | / c tau0 | z | KS p | other |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        other = ", ".join(f"{k} {r[k]}" for k in ("copies_with_other_tau", "events_with_two_or_more_copies",
                                                  "undecayed", "n_tries_gt1") if k in r)
        print(f"| {r['label']} | {r['events']:,} | {r['mean']:.2f} +- {r['se']:.2f} | {r['ratio']:.4f} +- "
              f"{r['ratio_se']:.4f} | {r['z']:+.1f} | {r['ks_p_exponential']:.2g} | {other} |")
    for name in ("full_event_all", "full_event_without_first"):
        if name in out:
            c = out[name]
            print(f"{name}: {c['events']:,} events, {c['mean']:.2f} +- {c['se']:.2f} mm (z {c['z']:+.1f}); "
                  f"chi2 {c['chi2']:.1f} / {c['dof']} (p {c['p']:.2g})")


if __name__ == "__main__":
    main()
