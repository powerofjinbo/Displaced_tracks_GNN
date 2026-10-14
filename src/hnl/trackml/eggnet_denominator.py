"""Canto-L's TrackML test tracks under the EggNet evaluation code with the target denominator changed (ledger R13;
post-freeze, descriptive; nothing is chosen).

The EggNet code (refports.eggnet, exatrkx/EggNet e2bd117) counts as targets the particles with pid != 0, pT >= 1 GeV
and at least 3 hits. Chan et al.'s "About 467K particles" over 400 test events matches pT >= 1 GeV with at least 1 hit
on our non-test events, not >= 3 hits (results/trackml_port_closure_v1.md). This module recounts the stored post-freeze
test tracks of the configurations already evaluated under the EggNet code in Table VI (C-full: WP-A, WP-R-P3; C-1GeV:
WP-A, WP-R-P4) with the target population pT >= 1 GeV and >= 1, >= 2 and >= 3 hits; matching, candidates (>= 3-hit
tracks), duplication and fake rate as in the code. The >= 3-hit rows must equal $RUN/test_post/refmetrics exactly
(value and bootstrap standard deviation). Event bootstrap B = 500, seed 12345, indices over the events in ascending
order, one fresh generator per configuration and definition, as hnl.trackml.refmetrics.

    python -m hnl.trackml.eggnet_denominator --run RUN --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.trackml.refports import eggnet as EG

CONFIGS = {"test_full": ("WP-A", "WP-R-P3"), "test_1gev": ("WP-A", "WP-R-P4")}
MIN_HITS = (1, 2, 3)


def per_event(labels_dir: str, wanted) -> pd.DataFrame:
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(labels_dir, "meta_b*.json")))]
    names = metas[0]["config_names"]
    if any(m["config_names"] != names for m in metas):
        raise SystemExit("blocks differ in their configurations")
    idx = {n: names.index(n) for n in wanted}
    _cfg_index.update(idx)
    rows = []
    for f in sorted(glob.glob(os.path.join(labels_dir, "labels_b*.parquet"))):
        lab = pd.read_parquet(f)
        parts = pd.read_parquet(f.replace("labels_b", "particles_b")).rename(columns={"n_hits": "nhits"})
        for ev, g in lab.groupby("event", sort=True):
            p = parts[parts.event == ev]
            pid = g.pid.to_numpy()
            for name, i in idx.items():
                lb = g[f"lab_{i}"].to_numpy()
                for k in MIN_HITS:
                    c = EG.port(pid, lb, p, hard_cut_pt=None, nhits_min=k)
                    rows.append({"event": int(ev), "config": name, "min_hits": k, **{key: int(v) for key, v in c.items()}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, B: int = 500, seed: int = 12345) -> pd.DataFrame:
    out = []
    for (name, k), c in df.groupby(["config", "min_hits"], sort=False):
        c = c.sort_values("event").reset_index(drop=True)
        pooled = {key: int(c[key].sum()) for key in EG.COUNT_KEYS}
        base = EG.metrics(pooled)
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, len(c), size=(B, len(c)))
        arr = c[list(EG.COUNT_KEYS)].to_numpy()
        boots = pd.DataFrame([EG.metrics(dict(zip(EG.COUNT_KEYS, arr[j].sum(0)))) for j in ix])
        for m, v in base.items():
            out.append({"config": name, "min_hits": k, "metric": m, "value": float(v),
                        "boot_std": float(boots[m].std()), "n_events": len(c),
                        "n_targets": pooled["n_particles"], "n_matched_targets": pooled["n_matched_target_particles"]})
    return pd.DataFrame(out)


def check_counts(df: pd.DataFrame, refparquet: str) -> int:
    """Rows of per-event integer counts (>= 3 hits) that differ from the recount of 2026-10-11."""
    ref = pd.read_parquet(refparquet)
    bad = 0
    for name in sorted(df.config.unique()):
        mine = df[(df.config == name) & (df.min_hits == 3)].sort_values("event").reset_index(drop=True)
        r = ref[ref.config == _cfg_index[name]].sort_values("event").reset_index(drop=True)
        if len(r) != len(mine) or not (r.event.to_numpy() == mine.event.to_numpy()).all():
            return -1
        for k in EG.COUNT_KEYS:
            bad += int((r[f"eg_{k}"].to_numpy() != mine[k].to_numpy()).sum())
    return bad


_cfg_index: dict = {}  # configuration name -> lab_<i> column of the condition being read (set by per_event)


def check_against(s: pd.DataFrame, refcsv: str) -> pd.DataFrame:
    """The >= 3-hit rows against the recount of 2026-10-11 (refmetrics.csv, read with round-trip float parsing):
    exact equality expected."""
    ref = pd.read_csv(refcsv, float_precision="round_trip")
    key = {"efficiency": "eggnet_efficiency", "duplication_rate": "eggnet_duplication", "fake_rate": "eggnet_fake"}
    rows = []
    for _, r in s[s.min_hits == 3].iterrows():
        q = ref[(ref.name == r.config) & (ref.metric == key[r.metric])].iloc[0]
        rows.append({"config": r.config, "metric": r.metric, "value": r.value, "ref_value": q.value,
                     "boot_std": r.boot_std, "ref_boot_std": q.boot_std,
                     "equal": bool(r.value == q.value and r.boot_std == q.boot_std)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the TrackML run directory (canto_l_v1/seed0_x1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    allsum, allchk = [], []
    for cond, wanted in CONFIGS.items():
        df = per_event(os.path.join(a.run, "test_post", "labels", cond), wanted)
        df.to_parquet(os.path.join(a.out, f"per_event_{cond}.parquet"), index=False)
        s = summarize(df).assign(condition=cond)
        chk = check_against(s, os.path.join(a.run, "test_post", "refmetrics", cond, "refmetrics.csv")).assign(condition=cond)
        nbad = check_counts(df, os.path.join(a.run, "test_post", "refmetrics", cond, "per_event_refcounts.parquet"))
        print(f"{cond}: per-event EggNet counts (>= 3 hits) differing from the 2026-10-11 recount: {nbad}")
        chk["per_event_count_mismatches"] = nbad
        allsum.append(s)
        allchk.append(chk)
    s, chk = pd.concat(allsum, ignore_index=True), pd.concat(allchk, ignore_index=True)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    chk.to_csv(os.path.join(a.out, "check_ge3_vs_refmetrics.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.6g}".format):
        print(s.to_string(index=False))
        print(chk.to_string(index=False))
    print("check >= 3 hits reproduces refmetrics exactly:", bool(chk.equal.all()))


if __name__ == "__main__":
    main()
