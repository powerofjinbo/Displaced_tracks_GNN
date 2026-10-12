"""Validate the barcode label rules against column 12 on every labeled HNL event (brief §2.3).

    python -m hnl.truth.validate_labels --out <dir>      (CPU Slurm job: slurm/validate_labels.sbatch)

Per event and rule: whether the rule's particle set equals the labeled set; particle-level TP / FP / FN; and the
same restricted to particles with >= 1 hit. Dropped events (docs/dropped_events.tsv) are excluded. Also tabulates
the PDG codes of |parent| = 24 particles in ttbar and the rule's behaviour on the unlabeled 15 GeV 100 mm sample.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.data.npz import COL, SAMPLES, load_event, particle_ids
from hnl.truth import labels as L

RULES = {
    "hnl_daughter": ("hnl_daughter", L.hnl_daughter_rule),
    "hnl_daughter_first_pair": ("hnl_daughter", L.hnl_daughter_rule_first_pair),
    "hnl_daughter_photons": ("hnl_daughter", L.hnl_daughter_rule_photons),
    "w_muon": ("w_muon", L.w_muon_rule),
}


def dropped(repo: str) -> set[tuple[str, int]]:
    df = pd.read_csv(os.path.join(repo, "docs", "dropped_events.tsv"), sep="\t", comment="#")
    return set(zip(df["sample"], df["event"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    drop = dropped(repo)
    rows, ttbar_w = [], {}
    t0 = time.time()
    for s in ("hnl10_100", "hnl15_200", "hnl15_100", "ttbar"):
        n = SAMPLES[s].n_events if a.limit is None else min(a.limit, SAMPLES[s].n_events)
        for i in range(n):
            if (s, i) in drop:
                continue
            ev = load_event(s, i)
            P = ev["particles"]
            ids = particle_ids(P)
            has_hits = np.isin(ids, np.unique(ev["pid"]))
            if s == "ttbar":
                w = np.abs(P[:, COL["parent_pid"]]) == 24
                for pdg in P[w, COL["particle_type"]].astype(int):
                    ttbar_w[pdg] = ttbar_w.get(pdg, 0) + 1
                continue
            row = {"sample": s, "index": i}
            truth = L.labels_from_table(P) if SAMPLES[s].labeled else None
            for name, (cat, rule) in RULES.items():
                pred = rule(P)
                row[f"{name}_n_pred"] = int(pred.sum())
                row[f"{name}_n_pred_hits"] = int((pred & has_hits).sum())
                if truth is None:
                    continue
                t = truth[cat]
                row[f"{name}_n_true"] = int(t.sum())
                row[f"{name}_exact"] = bool(np.array_equal(pred, t))
                row[f"{name}_exact_hits"] = bool(np.array_equal(pred & has_hits, t & has_hits))
                row[f"{name}_tp"] = int((pred & t).sum())
                row[f"{name}_fp"] = int((pred & ~t).sum())
                row[f"{name}_fn"] = int((~pred & t).sum())
            rows.append(row)
        print(f"[labels] {s} done, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(a.out, "label_rules.parquet"), index=False)
    summary = {}
    for s in ("hnl10_100", "hnl15_200"):
        d = df[df["sample"] == s]
        for name in RULES:
            summary[f"{s}/{name}"] = {
                "events": int(len(d)),
                "event_exact_frac": float(d[f"{name}_exact"].mean()),
                "event_exact_frac_hits": float(d[f"{name}_exact_hits"].mean()),
                "tp": int(d[f"{name}_tp"].sum()), "fp": int(d[f"{name}_fp"].sum()), "fn": int(d[f"{name}_fn"].sum()),
            }
    d = df[df["sample"] == "hnl15_100"]
    for name in RULES:
        summary[f"hnl15_100/{name}"] = {"events": int(len(d)),
                                         "n_pred_per_event": {int(k): int(v) for k, v in
                                                                      d[f"{name}_n_pred"].value_counts().sort_index().items()}}
    summary["ttbar_parent24_pdg_counts"] = {int(k): int(v) for k, v in sorted(ttbar_w.items())}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=int)
    print(json.dumps(summary, indent=1, default=int))


if __name__ == "__main__":
    main()
