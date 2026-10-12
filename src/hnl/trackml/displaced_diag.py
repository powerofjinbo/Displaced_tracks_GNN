"""Where Canto-L loses displaced targets on TrackML: a diagnosis on validation events (post-freeze; nothing is chosen
from it, and the test events are not read).

Per target (pT > 1 GeV, |eta| < 4, >= 5 hits) of a validation event, in one condition:
  particle      d0, rv, pt, eta, n_hits, n_pixel_hits, r_first, vz, category, vprim (from the particles table)
  graph stage   lcc_graph: the largest set of the particle's hits connected by graph edges between its own hits;
                te_graph: the fraction of its true (consecutive) edges that are graph edges; n_orphan: its hits
                without any graph edge to another of its hits
  edge scores   lcc_s<c>, te_s<c>: the same with the edges scored >= c (c = 0.1, 0.5, 0.9); n_wrong05: edges scored
                >= 0.5 from its hits to hits of another particle or to noise
  tracks (WP-A) n_best: its hits on the track holding most of them; t_best: that track's size; noise_best: noise
                hits on it; n_unassigned: its hits on no track; n_tracks: tracks holding at least one of its hits;
                dm: double-majority match (as hnl.trackml.evaluate)
A DM match needs n_best > n_hits / 2, so lcc_graph > n_hits / 2 is the graph-stage condition for the connected-
component builders (merging can join fragments that no edge connects).

    python -m hnl.trackml.displaced_diag run --events val --condition full --block K --out DIR
    python -m hnl.trackml.displaced_diag summary --dir DIR [--grid $RUN/grid/full]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

CUTS = (0.1, 0.5, 0.9)
COLS = ("d0", "rv", "pt", "eta", "n_hits", "n_pixel_hits", "r_first", "vz", "category", "vprim")


def _lcc(n: int, pid: np.ndarray, a: np.ndarray, b: np.ndarray) -> pd.Series:
    """Per particle: the largest number of its hits connected by the given same-particle edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    on = pid != 0
    df = pd.DataFrame({"pid": pid[on], "c": comp[on]})
    return df.groupby(["pid", "c"]).size().groupby("pid").max()


def _per_pid(pid_of_edges: np.ndarray, value: np.ndarray | None = None) -> pd.Series:
    s = pd.Series(np.ones(len(pid_of_edges)) if value is None else value, index=pid_of_edges)
    return s.groupby(level=0).sum()


def diagnose(ev, a: np.ndarray, b: np.ndarray, sc: np.ndarray, lab: np.ndarray) -> pd.DataFrame:
    pid, n = ev.pid, ev.n
    P = ev.particles.set_index("particle_id")
    tg = P.index[P.is_target.to_numpy(bool)]
    tg = tg[np.isin(tg, pid)]
    out = P.loc[tg, list(COLS)].copy()
    same = (pid[a] == pid[b]) & (pid[a] != 0)
    out["lcc_graph"] = _lcc(n, pid, a[same], b[same]).reindex(tg).fillna(1).to_numpy()
    # true edges present in the graph, and their scores
    key = lambda u, v: np.minimum(u, v).astype(np.int64) * n + np.maximum(u, v)
    gk = key(a, b)
    o = np.argsort(gk)
    gk, gs = gk[o], sc[o]
    te = ev.true_edges
    tk = key(te[0], te[1])
    j = np.clip(np.searchsorted(gk, tk), 0, len(gk) - 1)
    present = gk[j] == tk
    tscore = np.where(present, gs[j], -1.0)
    tpid = pid[te[0]]
    n_te = _per_pid(tpid).reindex(tg).fillna(0)
    out["n_true_edges"] = n_te.to_numpy()
    out["te_graph"] = (_per_pid(tpid, present.astype(float)).reindex(tg).fillna(0) / n_te.clip(lower=1)).to_numpy()
    for c in CUTS:
        m = same & (sc >= c)
        tag = f"s{int(round(c * 10)):02d}"
        out[f"lcc_{tag}"] = _lcc(n, pid, a[m], b[m]).reindex(tg).fillna(1).to_numpy()
        out[f"te_{tag}"] = (_per_pid(tpid, (tscore >= c).astype(float)).reindex(tg).fillna(0)
                            / n_te.clip(lower=1)).to_numpy()
    out["te_score_min"] = pd.Series(tscore, index=tpid).groupby(level=0).min().reindex(tg).to_numpy()
    # hits of the particle without any graph edge to another of its hits
    deg = np.bincount(np.concatenate([a[same], b[same]]), minlength=n)
    on = pid != 0
    out["n_orphan"] = pd.Series((deg[on] == 0).astype(int), index=pid[on]).groupby(level=0).sum().reindex(tg).to_numpy()
    # high-score edges into other particles or noise
    w = (~same) & (sc >= 0.5)
    ends = np.concatenate([pid[a[w]], pid[b[w]]])
    ends = ends[ends != 0]
    out["n_wrong05"] = _per_pid(ends).reindex(tg).fillna(0).to_numpy()
    # tracks of the frozen working point
    t_size = pd.Series(lab[lab >= 0]).value_counts()
    t_noise = pd.Series(lab[(lab >= 0) & (pid == 0)]).value_counts()
    h = pd.DataFrame({"pid": pid, "lab": lab})
    h = h[h.pid.isin(tg)]
    out["n_unassigned"] = (h.lab < 0).groupby(h.pid).sum().reindex(tg).fillna(0).to_numpy()
    ht = h[h.lab >= 0]
    sh = ht.groupby(["pid", "lab"]).size().rename("k").reset_index()
    out["n_tracks"] = sh.groupby("pid").size().reindex(tg).fillna(0).to_numpy()
    sh["t"] = t_size.reindex(sh.lab).to_numpy()
    sh = sh.sort_values(["pid", "k", "t"], ascending=[True, False, True]).drop_duplicates("pid")
    sh = sh.set_index("pid").reindex(tg)
    out["n_best"] = sh.k.fillna(0).to_numpy()
    out["t_best"] = sh.t.fillna(0).to_numpy()
    out["noise_best"] = t_noise.reindex(sh.lab.fillna(-1).astype(np.int64)).fillna(0).to_numpy()
    out["dm"] = (out.n_best > out.n_hits / 2) & (out.n_best > out.t_best / 2)
    out.insert(0, "event", np.int32(ev.event))
    return out.reset_index().rename(columns={"index": "particle_id"})


def cmd_run(a) -> None:
    from hnl.trackml import config as CFG
    from hnl.trackml import events as E
    from hnl.trackml import trackbuild as TB
    cfg = CFG.load(a.config)
    events = CFG.event_set(a.events, cfg)
    events = events[a.block * a.block_size:(a.block + 1) * a.block_size]
    cfgs, names = TB.explicit_configs(a.configs)
    wp = cfgs[names.index("WP-A")]
    import joblib
    model = joblib.load(a.merge_model) if wp[0] == "mwslmerge" else None
    sc_all = TB.read_scores(a.scores, events, a.condition)
    rows, t0 = [], time.time()
    for ev in E.iter_events(events):
        ev = E.with_condition(ev, a.condition)
        aa, bb, sc = sc_all[ev.event]
        lab = np.asarray(TB.build(wp, TB.EventState(ev, aa, bb, sc, model)))
        rows.append(diagnose(ev, aa, bb, sc, lab))
        print(json.dumps({"event": ev.event, "seconds": round(time.time() - t0, 1)}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat(rows, ignore_index=True)
    df.insert(1, "condition", a.condition)
    df.to_parquet(os.path.join(a.out, f"diag_{a.condition}_b{a.block}.parquet"), index=False)
    json.dump({"events": [int(e) for e in events], "condition": a.condition, "scores": a.scores, "wp": list(wp),
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - t0, 1)},
              open(os.path.join(a.out, f"meta_{a.condition}_b{a.block}.json"), "w"), indent=1)


D0_BINS = [0, 1, 10, 30, 100, 300, np.inf]
RV_BINS = [0, 1, 30, 100, 200, 400, 700, np.inf]
SPECIES = {11: "e", 13: "mu", 211: "pi", 321: "K", 2212: "p"}


def species(df: pd.DataFrame) -> pd.Series:
    """Particle species from the throughput phase's particle_type (PDG id; side table ptype)."""
    from hnl.trackml.convert import read_table
    from hnl.trackml.events import DERIVED_TP
    t = read_table("ptype", sorted(df.event.unique()), root=DERIVED_TP)
    m = df[["event", "particle_id"]].merge(t, on=["event", "particle_id"], how="left")
    return m.particle_type.abs().map(SPECIES).fillna("other").set_axis(df.index)


def generation(pid) -> np.ndarray:
    """Barcode generation of TrackML particle ids: 0 for the event generator, >= 1 for particles produced in the
    detector simulation. TrackML ids follow the 2018 ACTS barcode, vertex [52, 64), primary [36, 52), generation
    [24, 36), secondary [12, 24), process [0, 12), not the layout of hnl.geometry.ids. Checked on val: [24, 36) is 0
    for every primary and V0 daughter and non-zero together with the secondary and process fields, and generation 0
    gives 1.75 targets per event at |d0| 30-100 mm, close to an independent count on other throughput events."""
    return ((np.asarray(pid).astype(np.uint64) >> np.uint64(24)) & np.uint64(0xFFF)).astype(np.int64)


def cmd_exposure(a) -> None:
    """Targets per |d0| bin, category and species in an event set (particles tables only)."""
    from hnl.trackml import config as CFG
    from hnl.trackml.convert import read_table
    from hnl.trackml.events import DERIVED_TP
    events = CFG.event_set(a.events, CFG.load(a.config))
    p = read_table("particles", events, root=DERIVED_TP)
    p = p[p.is_target.to_numpy(bool)].copy()
    p["species"] = species(p)
    p["ad0"] = pd.cut(p.d0.abs(), D0_BINS, right=False)
    n = len(events)
    print(f"{a.events}: {n} events, {len(p)} targets, {len(p) / n:.1f} per event")
    print(p.groupby("ad0", observed=True).size().rename("targets").to_frame()
          .assign(per_event=lambda x: x.targets / n, frac=lambda x: x.targets / len(p)).to_string())
    d = p[p.d0.abs() >= 10]
    print(f"\n|d0| >= 10 mm: {len(d)} targets ({len(d) / n:.2f} per event, {len(d) / len(p):.4f} of the targets)")
    print(pd.crosstab(d.category, d.species, margins=True).to_string())
    f = p[p.d0.abs() >= 100]
    print(f"\n|d0| >= 100 mm: {len(f)} targets")
    print(pd.crosstab(f.category, f.species, margins=True).to_string())


def outcome(df: pd.DataFrame) -> pd.Series:
    """found / graph (graph LCC <= half) / gnn (graph fine, LCC at 0.5 <= half) / split / contaminated."""
    half = df.n_hits / 2
    o = np.select([df.dm, df.lcc_graph <= half, df.lcc_s05 <= half, df.n_best <= half],
                  ["found", "graph", "gnn", "split"], "contaminated")
    return pd.Series(o, index=df.index)


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    df = df.assign(outcome=outcome(df))
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({"n": g.size(), "eff": g.dm.mean(),
                      "graph_ok": g.apply(lambda x: (x.lcc_graph > x.n_hits / 2).mean()),
                      "s05_ok": g.apply(lambda x: (x.lcc_s05 > x.n_hits / 2).mean()),
                      "te_graph": g.te_graph.mean(), "te_s05": g.te_s05.mean(),
                      "pt_med": g.pt.median(), "nhits_med": g.n_hits.median(), "npix_med": g.n_pixel_hits.median(),
                      "wrong05_per_hit": g.apply(lambda x: x.n_wrong05.sum() / x.n_hits.sum())})
    oc = pd.crosstab(df[by], df.outcome, normalize="index")
    return t.join(oc)


def cmd_summary(a) -> None:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.dir, "diag_*_b*.parquet")))],
                   ignore_index=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    df["species"] = species(df)
    for cond, d in df.groupby("condition"):
        d = d.copy()
        d["ad0"] = pd.cut(d.d0.abs(), D0_BINS, right=False)
        d["rvb"] = pd.cut(d.rv, RV_BINS, right=False)
        print(f"\n=== condition {cond}: {d.event.nunique()} events, {len(d)} targets")
        if a.grid:
            g = pd.read_parquet(os.path.join(a.grid, "per_target.parquet"), filters=[("config", "==", 90)])
            m = d.merge(g[["event", "particle_id", "dm"]], on=["event", "particle_id"], suffixes=("", "_grid"))
            print(f"rebuild vs grid (config 90): {len(m)} matched, {(m.dm != m.dm_grid).sum()} DM differ")
        print("\n-- by |d0| [mm]\n", table(d, "ad0").to_string())
        dd = d[d.d0.abs() >= 10]
        print("\n-- |d0| >= 10 mm, by category\n", table(dd, "category").to_string())
        print("\n-- |d0| >= 10 mm, by species\n", table(dd, "species").to_string())
        print("\n-- |d0| < 1 mm, by species\n", table(d[d.d0.abs() < 1], "species").to_string())
        print("\n-- |d0| >= 10 mm, by production radius rv [mm]\n", table(dd, "rvb").to_string())
        dd = dd.assign(ptb=pd.cut(dd.pt, [1, 1.5, 2, 3, 5, 10, np.inf], right=False))
        print("\n-- |d0| >= 10 mm, by pT [GeV]\n", table(dd, "ptb").to_string())
        dd = dd.assign(nhb=pd.cut(dd.n_hits, [5, 6, 8, 10, 13, 100], right=False))
        print("\n-- |d0| >= 10 mm, by number of hits\n", table(dd, "nhb").to_string())
        far = d[d.d0.abs() >= 100]
        print("\n-- |d0| >= 100 mm, by category\n", table(far, "category").to_string())
        print("\n-- |d0| >= 100 mm, by species\n", table(far, "species").to_string())
        print("\n-- |d0| >= 100 mm, by rv\n", table(far, "rvb").to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--events", default="val")
    r.add_argument("--condition", default="full", choices=["full", "nonoise", "1gev"])
    r.add_argument("--scores", required=True)
    r.add_argument("--configs", required=True, help="frozen configurations, e.g. $RUN/test/frozen_full.json")
    r.add_argument("--merge-model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--block", type=int, default=0)
    r.add_argument("--block-size", type=int, default=10)
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    s.add_argument("--grid", default=None)
    x = sub.add_parser("exposure")
    x.add_argument("--events", default="train")
    a = ap.parse_args(argv)
    if getattr(a, "events", None) in ("test", "t2"):
        raise SystemExit("validation and training events only")
    {"run": cmd_run, "summary": cmd_summary, "exposure": cmd_exposure}[a.cmd](a)


if __name__ == "__main__":
    main()
