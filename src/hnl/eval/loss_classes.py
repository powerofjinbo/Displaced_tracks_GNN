"""Why are particles lost after a given track builder? (Stage 1 C diagnostic, val only.)

For one builder configuration on a run's scored val graphs, every particle of a population is put in one class:
  found      DM-matched;
  graph      not DM-matched by the chain-truth ceiling (CC on the graph's chain-true edges): lost in the graph stage;
  impure     some track holds > 50 % of its hits, but that track is <= 50 % pure (merged with another particle);
  fragmented its hits on tracks cover > 50 % of its hits, spread over >= 2 tracks with none above 50 %;
  short      its hits on tracks cover <= 50 % (the rest unassigned or on other particles' tracks);
with the sub-flag one_short (the best track misses DM by a single hit) and the number of its hits that have no graph
edge at all. A builder that wins only in one class shows where the remaining headroom is (fragment merging, graph-free
attachment, purity), before any model for it is built.

    python -m hnl.eval.loss_classes --graphs DIR --builder mwslayermerge,0.2 --out OUT.parquet [--population primary]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.eval import matching as M
from hnl.graphs import ceiling as C


def particle_classes(pid, labels, truth_labels, has_edge) -> pd.DataFrame:
    """One row per particle (np.unique(pid), pid != 0) with its class and diagnostics."""
    pid = np.asarray(pid)
    fm = M.FastMatch(pid)
    m = fm.match(labels)
    mt = fm.match(truth_labels)
    P = fm.P
    # per particle: hits on tracks, best track's share and that track's size
    on = np.asarray(labels) >= 0
    t, p = np.asarray(labels)[on], fm.pidx[on]
    n_on = np.bincount(p, minlength=P)
    best_share = np.zeros(P, dtype=np.int64)
    best_purity = np.zeros(P)
    n_tracks = np.zeros(P, dtype=np.int64)
    if len(t):
        tu, tinv = np.unique(t, return_inverse=True)
        size = np.bincount(tinv)
        key, shared = np.unique(tinv.astype(np.int64) * P + p, return_counts=True)
        kt, kp = key // P, key % P
        np.add.at(n_tracks, kp, 1)
        order = np.lexsort((-shared, kp))
        first = np.r_[True, kp[order][1:] != kp[order][:-1]]
        bk = order[first]
        best_share[kp[bk]] = shared[bk]
        best_purity[kp[bk]] = shared[bk] / size[kt[bk]]
    n = fm.n_particle
    no_edge = np.bincount(fm.pidx, weights=(~np.asarray(has_edge)).astype(float), minlength=P).astype(int)
    cls = np.full(P, "short", dtype=object)
    frag = (2 * n_on > n) & (2 * best_share <= n) & (n_tracks >= 2)
    impure = (2 * best_share > n) & ~m["dm_matched"]
    cls[frag] = "fragmented"
    cls[impure] = "impure"
    cls[~mt["dm_matched"] & ~m["dm_matched"]] = "graph"
    cls[m["dm_matched"]] = "found"
    return pd.DataFrame({"particle_id": fm.particles.astype(np.int64), "n_hits": n, "cls": cls,
                         "best_share": best_share, "best_purity": best_purity, "n_tracks": n_tracks,
                         "n_on_tracks": n_on, "one_short": ~m["dm_matched"] & (2 * (best_share + 1) > n),
                         "n_hits_no_edge": no_edge})


def _event(path, cfg):
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval.trackbuild_val import build, load_scored
    from hnl.trackbuild.mutex_ws import hit_layers, mws_layer
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    n = len(pid)
    g = load_pyg(path[:-3], weights_only=False)
    te = C.unordered_unique(g.track_edges.numpy())
    code = a.astype(np.int64) * n + b
    chain = np.isin(code, te[0] * n + te[1])
    from hnl.trackbuild import builders as TB
    truth_labels = TB.cc(n, a[chain], b[chain], np.ones(int(chain.sum())), 0.5)
    if cfg[0] in ("mwslayer", "mwslayermerge", "mws2pass", "mws2passmerge"):
        L = hit_layers([int(eid)])[int(eid)]
        if cfg[0].startswith("mws2pass"):
            from hnl.trackbuild.mutex_ws import mws_two_pass
            lab = mws_two_pass(n, a, b, sc, L, cfg[1], cfg[2])
        else:
            lab = mws_layer(n, a, b, sc, L, cfg[1])
        if cfg[0] in ("mwslayermerge", "mws2passmerge"):
            from hnl.trackbuild.merge_fragments import merge_fragments
            lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
            lab = merge_fragments(r, phi, z, lab, max_rms=1.0)
    else:
        lab = build(cfg, n, a, b, sc, R, (r, phi, z))
    has_edge = np.zeros(n, dtype=bool)
    has_edge[a] = True
    has_edge[b] = True
    d = particle_classes(pid, lab, truth_labels, has_edge)
    d.insert(0, "event_id", np.int64(eid))
    return d[d.particle_id != 0]


def _chunk(args):
    paths, cfg = args
    return pd.concat([_event(p, cfg) for p in paths], ignore_index=True)


def main(argv=None):
    import concurrent.futures as cf
    from hnl.eval.baseline_val import populations
    from hnl.eval.particles import particle_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--builder", required=True, help="family,param,... e.g. mwslayermerge,0.2 or walkmerge,0.2,0.5,1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", default="primary")
    ap.add_argument("--hnl-only", action="store_true", default=True, help="only HNL events (code 2 / 4)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    t0 = time.time()
    parts = a.builder.split(",")
    cfg = (parts[0], *[float(x) for x in parts[1:]])
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
    if a.hnl_only:
        paths = [p for p in paths if os.path.basename(p)[5] in "24"]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_chunk, [(paths[k:k + 10], cfg) for k in range(0, len(paths), 10)]))
    d = pd.concat(res, ignore_index=True)
    info = particle_table(np.unique(d.event_id))
    pop = populations(info)[a.population]
    keys = info.loc[pop, ["event_id", "particle_id"]]
    d = d.merge(keys, on=["event_id", "particle_id"], how="inner").merge(
        info[["event_id", "particle_id", "d0", "rv"]], on=["event_id", "particle_id"], how="left")
    d.to_parquet(a.out, index=False)
    summ = d.groupby("cls").agg(n=("cls", "size"), one_short=("one_short", "sum"),
                                no_edge_hits=("n_hits_no_edge", "sum"))
    summ["frac"] = summ.n / len(d)
    print(json.dumps({"builder": cfg, "population": a.population, "n": int(len(d)),
                      "elapsed_s": round(time.time() - t0, 1)}))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
