"""r-z pointing classes of the primary population (Stage-1 read-out, val only): graph ceiling, 'connected' and DM
efficiency of the primary-population HNL-daughter targets (preregistration section 6) by |dz0| / r_v, where dz0 =
z0(daughter) - z0(W muon of the event) from the truth-hit helix fits (derived/truth_fits fix_z0) and r_v the
two-track vertex radius of the daughter pair (hnl.eval.particles). A daughter with |dz0| / r_v >= 1 does not point
back to the primary vertex in r-z; the design panel found most of the ttbar-trained arm's loss there (DECISIONS.md,
decision rules R1-R8). Daughters without a vertex or in an event without a W muon form their own class.

A source is one run's per-particle table at one working point:
  * the run's evaluate_gnn output (RUN/eval_val), CC at --cut, or at the run's own cut at the fake-rate target
    (--fr-star / --wp: the largest fake rate <= FR* on the 0.01 grid, hnl.eval.baseline_val.cut_at_fake_rate);
  * a track-building grid RUN/<--name> of hnl.eval.trackbuild_val: --config K, or the configuration of --family F
    selected by the builder rule at the fake-rate target (hnl.eval.trackbuild_select.select).
The graph ceiling (perfect classifier + CC) and 'connected' (CC at the eval_val cut) come from RUN/eval_val when it
exists ('connected' only for an eval_val source). With --compare, each further run is read the same way (its own cut
or selected configuration; --compare-cc: its eval_val CC whatever --name says, e.g. an acorn arm next to a Stage-1
run's builder) and the table also gives each class's contribution, in points of the whole population, to the
efficiency gap between the first run and that run (paired on the particles common to all runs).

    python -m hnl.eval.rz_gap --run RUN [--compare RUN2 ...] [--compare-cc RUN3 ...] (--cut C | --fr-star FR | --wp WP)
        [--name trackbuild_val_mwslayer (--family mwslayer | --config K)] [--out OUT.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import curves, cut_at_fake_rate, load_run, populations
from hnl.eval.particles import CODE_SAMPLE, DERIVED, particle_table_cached, split_event_id

RZ_EDGES = (0.0, 0.5, 1.0, np.inf)
RZ_LABELS = ("rz < 0.5", "rz 0.5-1", "rz >= 1")
NO_CLASS = "no vertex / no W z0"
CLASSES = RZ_LABELS + (NO_CLASS,)


def rz_classes(dz0, rv) -> np.ndarray:
    """Class label per particle from dz0 [mm] (signed or absolute) and r_v [mm]; NaN or r_v <= 0 -> NO_CLASS. Bins are
    [lo, hi) (a ratio of exactly 0.5 is in 'rz 0.5-1')."""
    dz0 = np.abs(np.asarray(dz0, dtype=np.float64))
    rv = np.asarray(rv, dtype=np.float64)
    ok = np.isfinite(dz0) & np.isfinite(rv) & (rv > 0)
    ratio = np.where(ok, dz0 / np.where(ok, rv, 1.0), np.nan)
    k = np.searchsorted(np.asarray(RZ_EDGES), ratio, side="right") - 1
    lab = np.array(RZ_LABELS + (NO_CLASS,), dtype=object)[np.where(ok, np.clip(k, 0, len(RZ_LABELS) - 1), -1)]
    return lab.astype(str)


def daughter_dz0(event_ids) -> pd.DataFrame:
    """event_id, particle_id, dz0 = fix_z0(particle) - fix_z0(W muon of its event) for every particle of the HNL
    events (NaN if the event has no W muon)."""
    codes, idx = split_event_id(np.unique(np.asarray(event_ids, dtype=np.int64)))
    out = []
    for c in np.unique(codes):
        smp = CODE_SAMPLE[int(c)]
        if smp == "ttbar":
            continue
        p = pd.read_parquet(os.path.join(DERIVED, "truth_fits", "v1", f"particles_{smp}.parquet"),
                            columns=["event", "particle_id", "label_w", "fix_z0"])
        p = p[p.event.isin(idx[codes == c])]
        w = p[p.label_w].groupby("event").fix_z0.first().rename("z0_w")      # one W muon per event (checked)
        p = p.join(w, on="event")
        out.append(pd.DataFrame({"event_id": np.int64(c) * 100_000_000 + p.event.to_numpy(np.int64),
                                 "particle_id": p.particle_id.to_numpy(np.int64),
                                 "dz0": (p.fix_z0 - p.z0_w).to_numpy(np.float64)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["event_id", "particle_id", "dz0"])


def class_table(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean of each column in `cols` per class (rows in CLASSES order, then 'all'), with the counts."""
    g = t.groupby("rz_class")
    tab = g[cols].mean().reindex(list(CLASSES))
    tab.insert(0, "n", g.size().reindex(list(CLASSES)).fillna(0).astype(int))
    tab.loc["all"] = [len(t)] + [t[c].mean() for c in cols]
    return tab


def gap_shares(t: pd.DataFrame, ref: str, other: str) -> pd.Series:
    """Contribution of each class to mean(ref) - mean(other) over all rows of t, in points (they sum to the total)."""
    n = len(t)
    d = (t[ref].astype(float) - t[other].astype(float)).groupby(t.rz_class).sum() / n * 100
    out = d.reindex(list(CLASSES)).fillna(0.0)
    out.loc["all"] = out.sum()
    return out


def config_index(cfgs: list, family: str, params: str) -> int:
    """Index in configs.json of a configuration given as hnl.eval.trackbuild_select reports it (family, params)."""
    hits = [i for i, c in enumerate(cfgs) if c[0] == family and ",".join(str(x) for x in c[1:]) == params]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} configurations match {family} {params}")
    return hits[0]


def _eval_rows(ev_dir: str, cut: float) -> pd.DataFrame:
    cols = ["event_id", "particle_id", "cut", "dm_matched", "connected"]
    path = os.path.join(ev_dir, "per_particle.parquet")
    e = pd.read_parquet(path, columns=cols, filters=[("cut", "==", np.float32(round(cut, 4)))])
    if e.empty:                              # cut values written with another rounding: filter in memory
        e = pd.read_parquet(path, columns=cols)
        e = e[np.isclose(e.cut.astype(np.float64), cut, atol=5e-5)]
    return e


def read_source(run: str, a, use_name: bool = True) -> tuple[pd.DataFrame, str]:
    """(event_id, particle_id, eff[, ceiling, connected]) of one run at its working point, and a description."""
    ev_dir = os.path.join(run, "eval_val")
    have_eval = os.path.exists(os.path.join(ev_dir, "per_particle.parquet"))
    fr = None
    if a.wp:
        fr = float(json.load(open(a.wp))["FR_star"])
    elif a.fr_star is not None:
        fr = a.fr_star
    if a.name and use_name:
        d = os.path.join(run, a.name)
        pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
        cfgs = json.load(open(os.path.join(d, "configs.json")))
        if a.config is not None:
            k = a.config
        else:
            if fr is None or not a.family:
                raise SystemExit("a track-building source needs --config, or --family with --fr-star / --wp")
            from hnl.eval.trackbuild_select import select
            pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
            sel, none_ok = select(curves(pp, pe, particle_table_cached(np.unique(pe.event_id))), cfgs, fr)
            if a.family in none_ok or a.family not in set(sel.family):
                raise SystemExit(f"{d}: family {a.family} has no configuration with fake rate <= {fr}")
            row = sel[sel.family == a.family].iloc[0]
            k = config_index(cfgs, a.family, row.params)
        t = pp[pp.config == k][["event_id", "particle_id", "dm_matched"]].rename(columns={"dm_matched": "eff"})
        desc = f"{a.name} config {k} {cfgs[k]}"
        if have_eval:
            c = pd.read_parquet(os.path.join(ev_dir, "ceiling_particle.parquet"), columns=["event_id", "particle_id",
                                                                                          "dm_ceiling"])
            t = t.merge(c.rename(columns={"dm_ceiling": "ceiling"}), on=["event_id", "particle_id"], how="left")
        return t, desc
    if not have_eval:
        raise SystemExit(f"{run}: no eval_val (give --name for a track-building source)")
    if a.cut is not None:
        cut = a.cut
    else:
        if fr is None:
            raise SystemExit("give --cut, --fr-star or --wp")
        pp, pe = load_run(run)
        cut, reachable = cut_at_fake_rate(curves(pp, pe, particle_table_cached(np.unique(pe.event_id))), fr)
        if not reachable:
            print(f"WARNING {run}: fake-rate target {fr} unreachable; minimum-rate cut {cut}", file=sys.stderr)
    e = _eval_rows(ev_dir, cut)
    if e.empty:
        raise SystemExit(f"{run}: no eval_val rows at cut {cut}")
    t = e[["event_id", "particle_id", "dm_matched", "connected"]].rename(columns={"dm_matched": "eff"})
    c = pd.read_parquet(os.path.join(ev_dir, "ceiling_particle.parquet"), columns=["event_id", "particle_id",
                                                                                  "dm_ceiling"])
    t = t.merge(c.rename(columns={"dm_ceiling": "ceiling"}), on=["event_id", "particle_id"], how="left")
    return t, f"eval_val CC at {cut:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", action="append", default=[], help="further runs, read the same way")
    ap.add_argument("--compare-cc", action="append", default=[], help="further runs, read from their eval_val (CC)")
    ap.add_argument("--cut", type=float, default=None, help="eval_val source: CC at this cut")
    ap.add_argument("--fr-star", type=float, default=None)
    ap.add_argument("--wp", default=None, help="working_points.json (FR_star)")
    ap.add_argument("--name", default=None, help="track-building source: RUN/<name> of hnl.eval.trackbuild_val")
    ap.add_argument("--family", default=None)
    ap.add_argument("--config", type=int, default=None)
    ap.add_argument("--out", default=None, help="CSV (default: print only)")
    a = ap.parse_args(argv)
    runs = [(a.run, True)] + [(r, True) for r in a.compare] + [(r, False) for r in a.compare_cc]
    tabs, descs = [], {}
    for k, (run, use_name) in enumerate(runs):
        t, desc = read_source(run, a, use_name)
        tabs.append(t.set_index(["event_id", "particle_id"]).add_suffix(f"_{k}"))
        descs[f"run{k}"] = {"run": os.path.abspath(run), "source": desc}
    t = pd.concat(tabs, axis=1, join="inner").reset_index()
    if len({len(x) for x in tabs}) > 1:
        print(f"WARNING: runs cover different particles; {len(t)} common particles used", file=sys.stderr)
    info = particle_table_cached(np.unique(t.event_id.to_numpy()))
    prim = info[populations(info)["primary"]][["event_id", "particle_id", "d0", "rv"]]
    t = t.merge(prim, on=["event_id", "particle_id"], how="inner")
    t = t.merge(daughter_dz0(t.event_id.unique()), on=["event_id", "particle_id"], how="left")
    t["rz_class"] = rz_classes(t.dz0, t.rv)
    cols = [c for c in t.columns if c.split("_")[0] in ("eff", "ceiling", "connected") and c[-1].isdigit()]
    tab = class_table(t, cols)
    for k in range(1, len(runs)):
        tab[f"gap_0_minus_{k}_pts"] = gap_shares(t, "eff_0", f"eff_{k}")
    pd.set_option("display.width", 250)
    for k, d in descs.items():
        print(f"{k}: {d['run']} ({d['source']})")
    print(tab.round(4).to_string())
    if a.out:
        tab.to_csv(a.out)
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump({"sources": descs, "n_primary": int(len(t)), "rz_edges": list(RZ_EDGES[:-1]) + ["inf"]}, f,
                      indent=1)


if __name__ == "__main__":
    main()
