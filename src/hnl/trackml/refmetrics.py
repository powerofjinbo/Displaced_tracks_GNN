"""Canto-L's test numbers under the same-phase references' own evaluation code (post-freeze definition variants;
DECISIONS 2026-10-11, the user's decision to recount like for like).

Inputs: the post-freeze track labels of hnl.trackml.labels ($RUN/test_post/labels/<set>_<cond>: labels_b*.parquet,
particles_b*.parquet, counts_b*.parquet, meta_b*.json). Per event and frozen configuration:
  * EggNet (Calafiura et al. arXiv:2407.13925 Table 1, TrackML pT > 1 GeV; Chan et al. arXiv:2506.03415, full
    TrackML): hnl.trackml.refports.eggnet.port with the authors' settings (target pT >= 1 GeV and >= 3 hits, no eta
    cut; candidates = our tracks with >= 3 hits, the DBSCAN(min_samples=3) floor; noise-majority tracks are fakes).
    Our C-1GeV events already hold only the hits of pT > 1 GeV particles, as the authors' hard cut does, so no cut
    is applied here. Pooled over events (their eval_stage sums counts).
  * HierarchicalGNN (Liu et al., Tables 2-3): hnl.trackml.refports.liu.port, variant 'fork' (strict > 50 %, which
    reproduces their Truth-CC), aggregated as published (per-event ratios weighted by the event's hit count) and,
    for reference, pooled. Our label -1 ("no track", hnl.trackml.evaluate) stays out of the candidate graph, as the
    authors' callers leave unclustered hits out (HierarchicalGNN embedding_base.py:270); EggNet's port treats
    labels < 0 as unclustered itself. Particle pT is the initial pT (their code takes the minimum of the hit-level truth pT,
    which our test tables do not store).
Errors: event bootstrap, B = 500, seed 12345, indices drawn as in hnl.trackml.evaluate.bootstrap over the events in
ascending order.
Consistency: the rebuild's own event_counts (counts_b*.parquet) against the test chain's per_event.parquet.

    python -m hnl.trackml.refmetrics --labels DIR --test-grid $RUN/test/<set> --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.trackml.refports import eggnet as EG
from hnl.trackml.refports import liu as LIU

LIU_KEYS = ("n_hits_event", "n_rec", "n_matched_rec", "pur_den", "n_cand", "ref_early_exit")


def per_event(labels_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per (event, configuration) with the EggNet and Liu counts."""
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(labels_dir, "meta_b*.json")))]
    names = metas[0]["config_names"]
    if any(m["config_names"] != names for m in metas):
        raise SystemExit("blocks differ in their configurations")
    rows = []
    for f in sorted(glob.glob(os.path.join(labels_dir, "labels_b*.parquet"))):
        lab = pd.read_parquet(f)
        parts = pd.read_parquet(f.replace("labels_b", "particles_b"))
        parts = parts.rename(columns={"n_hits": "nhits"})
        for ev, g in lab.groupby("event", sort=True):
            p = parts[parts.event == ev]
            pid = g.pid.to_numpy()
            for i in range(len(names)):
                lb = g[f"lab_{i}"].to_numpy()
                e = EG.port(pid, lb, p, hard_cut_pt=None)
                li = LIU.port(pid, lb, p, variant="fork", unassigned_label=-1)
                rows.append({"event": int(ev), "config": i, **{f"eg_{k}": int(v) for k, v in e.items()},
                             **{f"liu_{k}": int(li[k]) for k in LIU_KEYS}})
    return pd.DataFrame(rows), names


def _eggnet(c: pd.DataFrame) -> dict:
    s = {k: c[f"eg_{k}"].sum() for k in EG.COUNT_KEYS}
    m = EG.metrics(s)
    return {"eggnet_efficiency": m["efficiency"], "eggnet_duplication": m["duplication_rate"],
            "eggnet_fake": m["fake_rate"]}


def _liu(c: pd.DataFrame) -> dict:
    lst = [{k: int(r[f"liu_{k}"]) for k in LIU_KEYS} for _, r in c.iterrows()]
    pub = LIU.aggregate(lst, mode="pl_weighted", empty="zero")
    poo = LIU.aggregate(lst, mode="pooled")
    return {"liu_eff": pub["eff"], "liu_fr": pub["fr"], "liu_eff_pooled": poo["eff"], "liu_fr_pooled": poo["fr"]}


def summarize(df: pd.DataFrame, names: list[str], B: int = 500, seed: int = 12345) -> pd.DataFrame:
    out = []
    for i, name in enumerate(names):
        c = df[df.config == i].sort_values("event").reset_index(drop=True)
        base = {**_eggnet(c), **_liu(c)}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(c), size=(B, len(c)))
        boots = pd.DataFrame([{**_eggnet(c.iloc[j]), **_liu(c.iloc[j])} for j in idx])
        for k, v in base.items():
            out.append({"config": i, "name": name, "metric": k, "value": float(v),
                        "boot_std": float(boots[k].std()), "n_events": len(c)})
    return pd.DataFrame(out)


def consistency(labels_dir: str, test_grid: str) -> dict:
    """Largest difference between the rebuild's event_counts and the test chain's per_event counts."""
    new = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(labels_dir, "counts_b*.parquet")))])
    old = pd.read_parquet(os.path.join(test_grid, "per_event.parquet"))
    key = ["event", "config"]
    m = new.merge(old, on=key, suffixes=("_new", "_old"))
    cols = [c[:-4] for c in m.columns if c.endswith("_new") and c[:-4] + "_old" in m.columns]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(m[c + "_new"])]
    diff = {c: int((m[c + "_new"] != m[c + "_old"]).sum()) for c in num if c != "trackml_score"}
    sc = (m["trackml_score_new"] - m["trackml_score_old"]).abs().max() if "trackml_score_new" in m else 0.0
    return {"rows_new": len(new), "rows_old": len(old), "rows_matched": len(m),
            "rows_differing_any": int(np.any(np.stack([m[c + "_new"] != m[c + "_old"] for c in num
                                                       if c != "trackml_score"]), axis=0).sum()),
            "columns_with_differences": {c: n for c, n in diff.items() if n},
            "max_abs_trackml_score_diff": float(sc) if sc == sc else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--test-grid", required=True, help="the test chain's grid directory, e.g. $RUN/test/test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df, names = per_event(a.labels)
    df.to_parquet(os.path.join(a.out, "per_event_refcounts.parquet"), index=False)
    s = summarize(df, names)
    s.to_csv(os.path.join(a.out, "refmetrics.csv"), index=False)
    cons = consistency(a.labels, a.test_grid)
    json.dump(cons, open(os.path.join(a.out, "consistency.json"), "w"), indent=1)
    print(s.to_string(index=False))
    print(json.dumps(cons, indent=1))


if __name__ == "__main__":
    main()
