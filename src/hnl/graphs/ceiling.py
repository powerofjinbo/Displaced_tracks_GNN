"""Graph-stage ceiling (brief §5e): keep exactly the true edges a graph contains (a perfect edge classifier), build
tracks with connected components, and measure double-majority efficiencies; plus edge efficiency / purity, also per
layer pair. Builder-agnostic: a graph is any (2, E) array of hit-row pairs for one event.

True edges are the npz chain edges (docs/definitions.md §2), compared as unique unordered pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from hnl.eval import matching as M


def unordered_unique(edges: np.ndarray) -> np.ndarray:
    """(2, E) -> (2, E') with a < b per column, duplicates and self-loops removed, sorted."""
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    a, b = np.minimum(e[0], e[1]), np.maximum(e[0], e[1])
    keep = a != b
    u = np.unique(np.stack([a[keep], b[keep]], 1), axis=0)
    return u.T.copy()


def _codes(e: np.ndarray, n: int) -> np.ndarray:
    return e[0] * n + e[1]


def cc_labels(n_hits: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component label of every hit; isolated hits get -1 (no track), as in acorn's ConnectedComponents."""
    if edges.shape[1] == 0:
        return np.full(n_hits, -1, dtype=np.int64)
    w = np.ones(edges.shape[1], dtype=np.int8)
    g = coo_matrix((w, (edges[0], edges[1])), shape=(n_hits, n_hits))
    lab = connected_components(g, directed=False)[1].astype(np.int64)
    size = np.bincount(lab)
    lab[size[lab] < 2] = -1
    return lab


def evaluate_graph(n_hits: int, pid: np.ndarray, true_edges: np.ndarray, graph: np.ndarray,
                   layer_id: np.ndarray | None = None):
    """One event. Returns (edge stats dict, per-particle DM frame from the perfect-classifier CC tracks,
    per-layer-pair frame or None). layer_id: an integer id per hit (e.g. volume * 100 + layer)."""
    t = unordered_unique(true_edges)
    g = unordered_unique(graph)
    tc, gc = _codes(t, n_hits), _codes(g, n_hits)
    in_graph = np.isin(tc, gc)
    kept = t[:, in_graph]
    same = pid[g[0]] == pid[g[1]]
    stats = {"n_true": int(t.shape[1]), "n_graph": int(g.shape[1]), "n_true_in_graph": int(in_graph.sum()),
             "n_same_pid_in_graph": int(same.sum())}
    labels = cc_labels(n_hits, kept)
    part, trk = M.match_event(pid, labels)
    # secondary ceiling: every graph edge joining two hits of one particle counts as correct (e.g. an edge that skips
    # an outlier hit of the chain); an upper bound for any edge classifier feeding connected components
    part_same, _ = M.match_event(pid, cc_labels(n_hits, g[:, same]))
    part["dm_matched_samepid"] = part_same.dm_matched.reindex(part.index).fillna(False).to_numpy()
    lp = None
    if layer_id is not None:
        la, lb = layer_id[t[0]], layer_id[t[1]]
        lo, hi = np.minimum(la, lb), np.maximum(la, lb)
        lp = pd.DataFrame({"lay_lo": lo, "lay_hi": hi, "in_graph": in_graph}).groupby(["lay_lo", "lay_hi"]) \
            .in_graph.agg(["size", "sum"]).rename(columns={"size": "n_true", "sum": "n_kept"})
    return stats, part, lp


def summarize_ceiling(per_particle: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """per_particle: concatenated DM frames joined with truth columns (is_target, category, bins...)."""
    t = per_particle[per_particle.is_target]
    g = t.groupby(by, observed=True).dm_matched
    out = pd.DataFrame({"n_targets": g.size(), "n_matched": g.sum()})
    out["efficiency"] = out.n_matched / out.n_targets
    out["err"] = np.sqrt(out.efficiency * (1 - out.efficiency) / out.n_targets)
    return out
