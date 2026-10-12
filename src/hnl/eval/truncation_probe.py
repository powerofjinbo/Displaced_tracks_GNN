"""Causal inner-hit truncation probe (M4 diagnostic, val only): does a GNN reject displaced tracks because they do not
point to the beam line, or because they start late (no inner pixel hits)?

For prompt particles (truth |d0| < 1 mm, >= 8 hits, pT >= 2 GeV, at most 5 per event) of HNL val events, the k innermost
hits (in R = sqrt(r^2 + z^2)) are removed from the run's metric-learning val graph (with every edge touching them), so
the track starts late but still points to the beam line; the graph is re-scored by the run's GNN. Compared across
depths k (0 = untouched) are the scores of the SAME outer chain-true edges (both hits at R rank >= max k), which exist
in every version: if their TPR drops with k, the classifier uses "where the track starts"; if it does not, its loss on
displaced daughters comes from pointing. All other hits and edges of the event are unchanged.

    python -m hnl.eval.truncation_probe --run RUN --out OUT_DIR [--n-events 300] [--depths 0,2,4] [--cut 0.35]
        [--graphs ML_GRAPHS_VALSET_TAR]   (default RUN/ml_graphs/valset.tar; Stage-1 runs use their baseline's graphs)
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import torch

# attributes of acorn's metric-learning output graphs, by level (explicit, so a track-level tensor that happens to
# have as many entries as hits is never filtered as node-level)
NODE_KEYS = {"z", "phi", "x", "r", "hit_R", "eta", "batch", "hit_particle_id", "hit_id", "hit_y", "hit_x", "hit_z",
             "hit_r", "hit_phi", "hit_eta"}
EDGE_KEYS = {"y", "edge_weights", "edge_y", "scores", "edge_scores"}
TRACK_KEYS = {"eta_particle", "nhits", "particle_id", "pt", "track_particle_id", "track_particle_pt"}
SKIP_KEYS = {"ptr", "num_nodes", "config", "event_id", "edge_index", "track_edges", "truth_map"}


def truncate(g, remove: np.ndarray):
    """Copy of graph g without the hits in `remove` (bool, per hit), with node / edge / track-level tensors filtered,
    nodes renumbered, hit_id reset to the row index and truth_map re-pointed to the new edge list."""
    import copy
    n, E, T = g.num_nodes, g.edge_index.shape[1], g.track_edges.shape[1]
    keep = ~torch.as_tensor(remove, dtype=torch.bool)
    new_id = torch.cumsum(keep.long(), 0) - 1
    h = copy.copy(g)
    ek = keep[g.edge_index[0]] & keep[g.edge_index[1]]
    tk = keep[g.track_edges[0]] & keep[g.track_edges[1]]
    edge_new = torch.full((E,), -1, dtype=torch.long)
    edge_new[ek] = torch.arange(int(ek.sum()))
    for k in g.keys():
        v = g[k]
        if not torch.is_tensor(v) or k in ("ptr",):
            continue
        if k in NODE_KEYS:
            assert v.shape[0] == n, k
            h[k] = v[keep]
        elif k in EDGE_KEYS:
            assert v.shape[-1] == E, k
            h[k] = v[..., ek]
        elif k in TRACK_KEYS:
            assert v.shape[-1] == T, k
            h[k] = v[..., tk]
        elif k not in SKIP_KEYS:
            raise KeyError(f"unknown graph attribute {k} {tuple(v.shape)}")
    h.edge_index = new_id[g.edge_index[:, ek]]
    h.track_edges = new_id[g.track_edges[:, tk]]
    if "truth_map" in g:
        tm = g.truth_map[tk]
        h.truth_map = torch.where(tm >= 0, edge_new[tm.clamp(min=0)], tm)
    m = int(keep.sum())
    h.num_nodes = m
    h.hit_id = torch.arange(m)
    if "batch" in g:
        h.batch = torch.zeros(m, dtype=g.batch.dtype)
    if "ptr" in g:
        h.ptr = torch.tensor([0, m])
    return h


def _save(g, path):
    buf = io.BytesIO()
    torch.save(g, buf)
    with gzip.open(path, "wb") as f:
        f.write(buf.getvalue())


def main(argv=None):
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval.particles import particle_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--n-events", type=int, default=300)
    ap.add_argument("--depths", default="0,2,4")
    ap.add_argument("--cut", type=float, default=0.35)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--graphs", default=None, help="metric-learning val graphs (tar); default RUN/ml_graphs/valset.tar")
    a = ap.parse_args(argv)
    depths = sorted({int(x) for x in a.depths.split(",")})
    kmax = max(depths)
    work = os.path.join(a.work, "truncation_probe")
    os.makedirs(a.out, exist_ok=True)
    # input graphs: the first n HNL val events of the run's metric-learning val graphs
    src = os.path.join(work, "src")
    os.makedirs(src, exist_ok=True)
    got = []
    with tarfile.open(a.graphs or os.path.join(a.run, "ml_graphs", "valset.tar")) as t:
        for mem in t:
            base = os.path.basename(mem.name)
            if base.endswith(".pyg.gz") and base[5] in "24":          # event2... / event4... = HNL samples
                mem.name = base
                t.extract(mem, src, filter="data")
                got.append(os.path.join(src, base))
                if len(got) >= a.n_events:
                    break
    graphs = {int(os.path.basename(p)[5:-7]): load_pyg(p[:-3], weights_only=False) for p in got}
    info = particle_table(np.array(sorted(graphs)))
    sel = info[(info.d0.abs() < 1.0) & (info.n_hits >= 8) & (info.pt >= 2.0)]
    sel = sel.groupby("event_id").head(5)
    chosen = {e: set(g_.particle_id.astype(np.int64)) for e, g_ in sel.groupby("event_id")}
    # per depth: truncated graphs, scored by the run's GNN (best checkpoint of the finished training)
    import glob
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(a.run, "gnn", "run_*.json"))]
    ckpt = max([r for r in runs if r.get("state") == "done"], key=lambda r: r["end_utc"])["best_ckpt"]
    rows = []
    for k in depths:
        d_in = os.path.join(work, f"in_{k}", "valset")
        os.makedirs(d_in, exist_ok=True)
        keep_rank = {}
        for e, g in graphs.items():
            pid = g.hit_particle_id.numpy()
            R = torch.sqrt(g.r.double() ** 2 + g.z.double() ** 2).numpy()
            remove = np.zeros(len(pid), dtype=bool)
            rank = np.full(len(pid), -1)
            for p in chosen.get(e, ()):
                idx = np.flatnonzero(pid == p)
                order = idx[np.argsort(R[idx])]
                rank[order] = np.arange(len(order))
                remove[order[:k]] = True
            keep_rank[e] = rank[~remove]
            _save(truncate(g, remove), os.path.join(d_in, f"event{e}.pyg.gz"))
        d_out = os.path.join(work, f"out_{k}")
        subprocess.run([sys.executable, "-m", "hnl.train.acorn_runner", "infer", "--stage", "gnn", "--ckpt", ckpt,
                        "--input-dir", os.path.dirname(d_in), "--out-dir", d_out, "--accelerator", a.accelerator],
                       check=True)
        for e in graphs:
            s = load_pyg(os.path.join(d_out, "valset", f"event{e}.pyg"), weights_only=False)
            pid = s.hit_particle_id.numpy()
            rk = keep_rank[e]
            ei = s.edge_index.numpy()
            sc = s.scores.numpy()
            te = s.track_edges.numpy()
            # chain-true outer edges of the chosen particles (both hits at R rank >= kmax), unordered, max score
            n = len(pid)
            code = np.minimum(ei[0], ei[1]).astype(np.int64) * n + np.maximum(ei[0], ei[1])
            best = pd.Series(sc).groupby(code).max()
            tcode = np.unique(np.minimum(te[0], te[1]).astype(np.int64) * n + np.maximum(te[0], te[1]))
            for u, v in zip(tcode // n, tcode % n):
                p = pid[u]
                if p not in chosen.get(e, ()) or pid[v] != p or min(rk[u], rk[v]) < kmax:
                    continue
                c = min(u, v) * n + max(u, v)
                rows.append({"depth": k, "event_id": e, "particle_id": int(p), "outer_rank": int(min(rk[u], rk[v])),
                             "in_graph": c in best.index, "score": float(best.get(c, np.nan))})
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(a.out, "edges.parquet"), index=False)
    summ = df.groupby("depth").agg(n_edges=("score", "size"), in_graph=("in_graph", "mean"),
                                   mean_score=("score", "mean"),
                                   tpr=("score", lambda x: float((x > a.cut).mean())))
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    print(json.dumps({"run": a.run, "ckpt": ckpt, "n_events": len(graphs), "n_particles": int(sum(map(len, chosen.values()))),
                      "cut": a.cut}))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
