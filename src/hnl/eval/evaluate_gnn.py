"""Evaluate scored graphs (acorn GNN outputs) with our definitions (docs/definitions.md §5-6, preregistration §4).

For every event and every edge-score cut s on a grid: connected components on the edges with score > s (acorn's
strict cut; isolated hits are no track), double-majority matching against hit_particle_id, fake tracks (>= 5 hits),
duplicates (single majority among >= 5-hit tracks), HNL-daughter pair efficiency and merges; edge TP / FP / FN against
the npz chain truth (unique unordered pairs).

    python -m hnl.eval.evaluate_gnn --graphs DIR_OF_event*.pyg.gz --out OUT [--cuts 0.05:0.95:0.05] [--workers N]

Outputs: per_particle.parquet (event_id, particle_id, cut, dm_matched, dm_nofp (CC on the kept chain-truth edges
only), dm_nocross (CC on the kept edges joining two hits of one particle), connected (some CC track holds more than
half of the particle's hits: DM without the purity condition), n_sm_tracks; rows for targets, HNL daughters and W muons unless --all-particles), ceiling_particle.parquet (graph-stage ceilings: dm_ceiling = CC on every
true edge of the graph, dm_ceiling_samepid = CC on every same-particle edge),
per_event.parquet (event_id, cut, n_tracks_ge5, n_fake, tp, fp, fn, n_edges, n_true_in_graph, n_true, n_dup_pairs,
hnl pair flags; edge FPR = fp / (n_edges - n_true_in_graph); muon-target edge counts n_true_mu, n_true_in_graph_mu,
tp_mu and their *_clean variants without edges that touch an outlier hit, DECISIONS.md); event_id
is the integer acorn event id (sample code * 1e8 + index).
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


def parse_cuts(spec: str) -> list[np.float64]:
    """Cuts as float64 whatever the spec form, so that val (range) and test (comma list) compare identically."""
    if spec == "none":
        return []
    if ":" in spec:
        a, b, st = (float(x) for x in spec.split(":"))
        return [np.float64(round(float(x), 4)) for x in np.arange(a, b + 1e-9, st)]
    return [np.float64(round(float(x), 4)) for x in spec.split(",")]


def _event_id(g) -> str:
    e = g.event_id
    while isinstance(e, (list, tuple)):
        e = e[0]
    return str(e)


def evaluate_event(path: str, cuts: list[float], hnl_pairs: dict, keep_ids: dict | None = None,
                   muons: dict | None = None, labels_out: dict | None = None):
    """keep_ids: event_id (str) -> particle ids whose rows are written (default: every particle); matching always
    uses every hit and particle of the event. muons: event_id -> (muon-target particle ids, outlier hit rows) for the
    muon-target edge counts (*_mu) and their "clean" variant without edges touching an outlier hit (*_mu_clean).
    labels_out: if given, receives the CC track labels of every cut, keyed (event id, cut index)."""
    from acorn.utils.loading_utils import load_pyg
    g = load_pyg(path[:-3] if path.endswith(".gz") else path, weights_only=False)
    eid = _event_id(g)
    pid = g.hit_particle_id.numpy()
    n = len(pid)
    ei = g.edge_index.numpy()
    if "scores" in g:
        sc = g.scores.numpy()
    elif "edge_scores" in g:
        sc = g.edge_scores.numpy()
    else:                                  # unscored graphs (metric-learning output): ceilings only
        if cuts:
            raise ValueError(f"{path}: no edge scores, use --cuts none")
        sc = np.zeros(ei.shape[1], dtype=np.float32)
    truth = C.unordered_unique(g.track_edges.numpy())
    tcode = truth[0].astype(np.int64) * n + truth[1]
    a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    # edge counts are in unique unordered pairs: if a pair appears twice (both directions), keep its highest score
    code = a.astype(np.int64) * n + b
    order = np.lexsort((-sc, code))
    first = np.ones(len(code), dtype=bool)
    first[1:] = code[order][1:] != code[order][:-1]
    n_dup = int((~first).sum())
    keep_idx = order[first]
    a, b, sc, code = a[keep_idx], b[keep_idx], sc[keep_idx], code[keep_idx]
    sc = sc.astype(np.float64)                 # float64 comparison with float64 cuts, on val and on test alike
    is_true = np.isin(code, tcode)
    n_true_in_graph = int(is_true.sum())
    mu_counts = None
    if muons is not None:
        mu_ids, out_rows = muons.get(eid, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        bad = np.zeros(n, dtype=bool)
        bad[np.asarray(out_rows, dtype=np.int64)] = True
        t_mu = np.isin(pid[truth[0]], mu_ids)                      # chain edges are same-particle by construction
        t_clean = t_mu & ~bad[truth[0]] & ~bad[truth[1]]
        g_mu = np.isin(code, tcode[t_mu])
        g_clean = np.isin(code, tcode[t_clean])
        mu_counts = {"n_true_mu": int(t_mu.sum()), "n_true_mu_clean": int(t_clean.sum()),
                     "n_true_in_graph_mu": int(g_mu.sum()), "n_true_in_graph_mu_clean": int(g_clean.sum())}
    fm = M.FastMatch(pid)
    pair = hnl_pairs.get(eid)
    # graph-stage ceilings (cut independent): perfect classifier = CC on every true edge of the graph; and CC on
    # every same-particle edge of the graph
    ceil = fm.match(C.cc_labels(n, np.stack([a[is_true], b[is_true]])))["dm_matched"]
    same = pid[a] == pid[b]
    ceil_same = fm.match(C.cc_labels(n, np.stack([a[same], b[same]])))["dm_matched"]
    dm = np.zeros((len(cuts), fm.P), dtype=bool)
    dm_nofp = np.zeros((len(cuts), fm.P), dtype=bool)
    dm_nocross = np.zeros((len(cuts), fm.P), dtype=bool)
    conn = np.zeros((len(cuts), fm.P), dtype=bool)
    nsm = np.zeros((len(cuts), fm.P), dtype=np.int16)
    evrows = []
    for k, s in enumerate(cuts):
        keep = sc > s
        labels = C.cc_labels(n, np.stack([a[keep], b[keep]]))
        if labels_out is not None:
            labels_out[(eid, k)] = labels
        m = fm.match(labels)
        dm[k], nsm[k], conn[k] = m["dm_matched"], m["n_sm_tracks"], m["connected"]
        # the same cut with every fake edge removed: the loss from missed true edges alone
        dm_nofp[k] = fm.match(C.cc_labels(n, np.stack([a[keep & is_true], b[keep & is_true]])))["dm_matched"]
        # the same cut with only the edges between two different particles removed (a truth-assisted builder that
        # keeps same-particle skip edges, which the chain truth calls fake)
        dm_nocross[k] = fm.match(C.cc_labels(n, np.stack([a[keep & same], b[keep & same]])))["dm_matched"]
        tp = int((keep & is_true).sum())
        row = {"event_id": int(eid), "cut": float(s), "n_tracks_ge5": m["n_tracks_ge5"], "n_fake": m["n_fake"],
               "tp": tp, "fp": int((keep & ~is_true).sum()), "fn": int(len(tcode) - tp), "n_edges": int(len(sc)),
               "n_true_in_graph": n_true_in_graph, "n_true": int(len(tcode)), "n_dup_pairs": n_dup}
        if mu_counts is not None:
            row.update(mu_counts)
            row["tp_mu"] = int((keep & g_mu).sum())
            row["tp_mu_clean"] = int((keep & g_clean).sum())
        if pair is not None:
            pa, pb = pair
            ia, ib = np.searchsorted(fm.particles, [pa, pb])
            row["pair_both"] = bool(m["dm_matched"][ia] and m["dm_matched"][ib])
            row["pair_merged"] = fm.merged(m, pa, pb)
        evrows.append(row)
    if not cuts:                           # ceilings only: one row of graph statistics
        evrows.append({"event_id": int(eid), "cut": np.nan, "n_edges": int(len(sc)), "n_true_in_graph": n_true_in_graph,
                       "n_true": int(len(tcode)), "n_dup_pairs": n_dup, **(mu_counts or {})})
    sel = np.ones(fm.P, dtype=bool) if keep_ids is None else np.isin(fm.particles, keep_ids.get(eid, []))
    P = int(sel.sum())
    part = pd.DataFrame({"event_id": np.int64(eid), "particle_id": np.tile(fm.particles[sel].astype(np.int64), len(cuts)),
                         "cut": np.repeat(np.asarray(cuts, dtype=np.float32), P),
                         "dm_matched": dm[:, sel].ravel(), "dm_nofp": dm_nofp[:, sel].ravel(),
                         "dm_nocross": dm_nocross[:, sel].ravel(), "connected": conn[:, sel].ravel(),
                         "n_sm_tracks": nsm[:, sel].ravel()})
    ceiling = pd.DataFrame({"event_id": np.int64(eid), "particle_id": fm.particles[sel].astype(np.int64),
                            "dm_ceiling": ceil[sel], "dm_ceiling_samepid": ceil_same[sel]})
    return part, pd.DataFrame(evrows), ceiling


def _chunk(args):
    paths, cuts, hnl_pairs, keep, muons = args
    res = [evaluate_event(p, cuts, hnl_pairs, keep, muons) for p in paths]
    return tuple(pd.concat([r[k] for r in res], ignore_index=True) for k in range(3))


def event_truth(event_ids: list[str]) -> tuple[dict, dict]:
    """(pairs, keep): event_id -> (pid_a, pid_b) of the HNL-daughter pair when both daughters are targets; event_id ->
    particle ids written to the per-particle tables (targets, every HNL daughter and W muon)."""
    from hnl.graphs.ml_scan import particle_info
    info = particle_info(event_ids)
    code = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}
    info["event_id"] = [f"{code[s]}{e:08d}" for s, e in zip(info["sample"], info["event"])]
    h = info[(info.category == "hnl") & (info.is_target == True)]
    pairs = {}
    for eid, g in h.groupby("event_id"):
        if len(g) == 2:
            pairs[eid] = (int(g.particle_id.iloc[0]), int(g.particle_id.iloc[1]))
    k = info[(info.is_target == True) | info.category.isin(["hnl", "w"])]
    keep = {eid: np.sort(g.particle_id.to_numpy(np.int64)) for eid, g in k.groupby("event_id")}
    return pairs, keep


def muon_truth(event_ids: list[str]) -> dict:
    """event_id -> (ids of muon targets: HNL-daughter and W-muon targets, outlier hit rows of muons) from
    derived/truth_fits/v1/muon_outlier_hits_<sample>.parquet (hnl.truth.outlier_hits)."""
    from hnl.data.acorn_store import CODE_SAMPLE
    from hnl.graphs.ml_scan import particle_info
    info = particle_info(event_ids)
    code = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}
    info["event_id"] = [f"{code[s]}{e:08d}" for s, e in zip(info["sample"], info["event"])]
    mu = info[(info.is_target == True) & info.category.isin(["hnl", "w"])]
    ids = {eid: g.particle_id.to_numpy(np.int64) for eid, g in mu.groupby("event_id")}
    root = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived",
                        "truth_fits", "v1")
    rows = {}
    for c in sorted({int(e[0]) for e in event_ids}):
        f = pd.read_parquet(os.path.join(root, f"muon_outlier_hits_{CODE_SAMPLE[c]}.parquet"))
        f["event_id"] = [f"{c}{e:08d}" for e in f.event]
        f = f[f.event_id.isin(set(event_ids))]
        for eid, g in f.groupby("event_id"):
            rows[eid] = g.row.to_numpy(np.int64)
    empty = np.zeros(0, np.int64)
    return {eid: (ids.get(eid, empty), rows.get(eid, empty)) for eid in event_ids}


def hnl_pair_map(event_ids: list[str]) -> dict:
    return event_truth(event_ids)[0]


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", default="0.05:0.95:0.05")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--all-particles", action="store_true", help="write rows for every particle, not only targets, "
                                                                     "HNL daughters and W muons")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
    eids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths]
    pairs, keep = event_truth(eids)
    if a.all_particles:
        keep = None
    muons = muon_truth(eids)
    cuts = parse_cuts(a.cuts)
    chunks = [(paths[k:k + a.chunk], cuts, pairs, keep, muons) for k in range(0, len(paths), a.chunk)]
    if a.workers > 1:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(_chunk, chunks))
    else:
        res = [_chunk(c) for c in chunks]
    pp = pd.concat([r[0] for r in res], ignore_index=True)
    pe = pd.concat([r[1] for r in res], ignore_index=True)
    ce = pd.concat([r[2] for r in res], ignore_index=True)
    pp.to_parquet(os.path.join(a.out, "per_particle.parquet"), index=False)
    pe.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    ce.to_parquet(os.path.join(a.out, "ceiling_particle.parquet"), index=False)
    meta = {"graphs": a.graphs, "n_events": len(paths), "cuts": [float(c) for c in cuts],
            "elapsed_s": round(time.time() - t0, 1),
            "slurm_job": os.environ.get("SLURM_JOB_ID")}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
