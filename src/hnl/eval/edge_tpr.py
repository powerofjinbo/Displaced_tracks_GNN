"""Edge-level view of the pointing shortcut (M4 diagnostic, val only): for the chain-true edges of HNL-daughter and
W-muon targets (and of ttbar targets), the fraction present in the metric-learning graph (graph efficiency) and the
fraction kept by the GNN at a score cut (edge TPR), binned in the particle's truth |d0| (preregistered |d0| bins) and
pT. Edges are unique unordered pairs with the highest score of their two directions (as in hnl.eval.evaluate_gnn).

Cut at a target edge FPR (--efpr, or e* from a working_points.json with --wp): the run's own cut whose edge FPR on
the fake edges of its pooled HNL val graphs (unique unordered pairs that are not chain edges, as evaluate_gnn; kept =
score > cut) is the largest value <= the target on the 0.01 grid of eval_val (ties: the lowest cut; unreachable ->
the minimum-FPR cut, flagged; hnl.eval.baseline_val.cut_at_fake_rate), or the exact quantile with --efpr-grid 0.
Then dD = TPR(|d0| in [100, 300) mm) - TPR(|d0| in [0.1, 1) mm) of in-graph chain-true edges of HNL-daughter
targets at that cut (preregistration section 6c), written with the cut to OUT_efpr.json.

    python -m hnl.eval.edge_tpr --graphs DIR --out OUT.csv --cut 0.40 [--cut ...]
    python -m hnl.eval.edge_tpr --graphs DIR --out OUT.csv (--efpr E | --wp WORKING_POINTS_JSON) [--cut ...]
    python -m hnl.eval.edge_tpr --graphs TEST_DIR --out OUT.csv --cut C_FIXED_ON_VAL --per-event OUT_EVENTS.parquet

--per-event writes, at the single evaluation cut (the --efpr / --wp cut, else the one --cut), per event the in-graph
chain-true edges of HNL-daughter targets in the two dD ranges and how many are kept (n_low, k_low, n_high, k_high),
for the paired event bootstrap of dD on test (preregistration section 6c: the cut fixed on val is applied unchanged).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.baseline_val import HNL_CODES, cut_at_fake_rate
from hnl.eval.particles import particle_table, split_event_id
from hnl.truth.report import D0_BINS

PT_BINS = [1, 2, 5, 10, 20, np.inf]
CUT_GRID = np.round(np.arange(0.01, 0.991, 0.01), 4)       # eval_val's cut grid (evaluate_gnn --cuts 0.01:0.99:0.01)
DD_LOW, DD_HIGH = (0.1, 1.0), (100.0, 300.0)              # |d0| ranges [mm] of dD (preregistration section 6c)


def event_scores(path: str) -> tuple[int, pd.DataFrame, np.ndarray]:
    """(event id; one row per chain-true edge: event, particle, in_graph, score (NaN if not in the graph); the scores
    of the graph's fake edges, i.e. unique unordered pairs that are not chain edges)."""
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval.evaluate_gnn import _event_id
    g = load_pyg(path[:-3] if path.endswith(".gz") else path, weights_only=False)
    eid = int(_event_id(g))
    n = int(g.hit_particle_id.shape[0])
    pid = g.hit_particle_id.numpy()
    ei = g.edge_index.numpy()
    sc = (g.scores if "scores" in g else g.edge_scores).numpy().astype(np.float64)
    code = np.minimum(ei[0], ei[1]).astype(np.int64) * n + np.maximum(ei[0], ei[1])
    order = np.lexsort((-sc, code))
    first = np.r_[True, code[order][1:] != code[order][:-1]]
    gcode, gsc = code[order][first], sc[order][first]
    te = g.track_edges.numpy()
    tcode = np.unique(np.minimum(te[0], te[1]).astype(np.int64) * n + np.maximum(te[0], te[1]))
    pos = np.searchsorted(gcode, tcode)
    found = (pos < len(gcode)) & (gcode[np.minimum(pos, len(gcode) - 1)] == tcode)
    score = np.where(found, gsc[np.minimum(pos, len(gcode) - 1)], np.nan)
    true = pd.DataFrame({"event_id": eid, "particle_id": pid[tcode // n].astype(np.int64), "in_graph": found,
                         "score": score})
    return eid, true, gsc[~np.isin(gcode, tcode)]


def event_edges(path: str) -> pd.DataFrame:
    """One row per chain-true edge: event, particle, in_graph, score (NaN if not in the graph)."""
    return event_scores(path)[1]


def cut_at_fpr(fake_scores: np.ndarray, target: float, grid=CUT_GRID) -> tuple[float, bool, float]:
    """(cut, reachable, fpr) for kept = score > cut. With a grid: the cut whose FPR is the largest value <= target
    (ties: the lowest cut; none -> the minimum-FPR cut, reachable False), as baseline_val.cut_at_fake_rate. grid None:
    the smallest cut with FPR <= target among the scores themselves (the exact quantile)."""
    f = np.sort(np.asarray(fake_scores, dtype=np.float64))
    if len(f) == 0:
        raise ValueError("no fake edges")
    if not np.isfinite(target) or target < 0:
        raise ValueError(f"bad target FPR {target!r}")
    fpr_of = lambda c: float(len(f) - np.searchsorted(f, c, side="right")) / len(f)
    if grid is None:
        k = int(np.floor(target * len(f) + 1e-9))       # at most k fake edges may stay above the cut
        if k >= len(f):
            return -np.inf, True, 1.0
        cut = float(f[len(f) - k - 1])
        return cut, True, fpr_of(cut)
    grid = np.asarray(grid, dtype=np.float64)
    c = pd.DataFrame({"edge_fpr_hnl": [fpr_of(x) for x in grid]}, index=grid)
    cut, reachable = cut_at_fake_rate(c, target, "edge_fpr_hnl")
    return cut, reachable, float(c.loc[cut, "edge_fpr_hnl"])


def delta_d(t: pd.DataFrame, cut: float) -> dict:
    """dD at a cut from chain-true edges of HNL-daughter targets (columns d0 (signed or absolute), in_graph, score):
    TPR of in-graph edges (score > cut) with |d0| in DD_HIGH minus that with |d0| in DD_LOW."""
    d0 = t.d0.abs().to_numpy()
    ing = t.in_graph.to_numpy(bool)
    keep = t.score.to_numpy(np.float64) > cut                 # NaN scores (not in the graph) are never kept
    out = {}
    for name, (lo, hi) in (("low", DD_LOW), ("high", DD_HIGH)):
        m = ing & (d0 >= lo) & (d0 < hi)
        out[f"n_{name}"] = int(m.sum())
        out[f"tpr_{name}"] = float(keep[m].mean()) if m.any() else np.nan
    out["dD"] = out["tpr_high"] - out["tpr_low"]
    return out


def per_event_dd(t: pd.DataFrame, cut: float) -> pd.DataFrame:
    """Per event: in-graph chain-true edges of HNL-daughter targets with |d0| in DD_LOW / DD_HIGH and how many have
    score > cut (columns event_id, n_low, k_low, n_high, k_high; events without such edges are zero rows)."""
    d0 = t.d0.abs().to_numpy()
    ing = t.in_graph.to_numpy(bool)
    keep = t.score.to_numpy(np.float64) > cut
    ev = t.event_id.to_numpy(np.int64)
    out = pd.DataFrame({"event_id": np.unique(ev)})
    for name, (lo, hi) in (("low", DD_LOW), ("high", DD_HIGH)):
        m = ing & (d0 >= lo) & (d0 < hi)
        n = pd.Series(m.astype(np.int64)).groupby(ev).sum()
        k = pd.Series((m & keep).astype(np.int64)).groupby(ev).sum()
        out[f"n_{name}"] = out.event_id.map(n).fillna(0).astype(np.int64)
        out[f"k_{name}"] = out.event_id.map(k).fillna(0).astype(np.int64)
    return out


def _targets(edges: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    t = edges.merge(info[["event_id", "particle_id", "category", "is_target", "d0", "pt", "sample"]],
                    on=["event_id", "particle_id"], how="inner")
    t = t[t.is_target]
    t["population"] = np.where(t["sample"] == "ttbar", "ttbar_targets",
                               np.where(t.category == "hnl", "hnl_daughters",
                                        np.where(t.category == "w", "w_muons", "other_hnl_events")))
    return t


def summarize(edges: pd.DataFrame, info: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    t = _targets(edges, info)
    t["d0_bin"] = pd.cut(t.d0.abs(), D0_BINS, right=False).astype(str)
    t["pt_bin"] = pd.cut(t.pt, PT_BINS, right=False).astype(str)
    rows = []
    for var in ("d0_bin", "pt_bin"):
        for (pop, b), gg in t.groupby(["population", var], observed=True):
            r = {"population": pop, "variable": var, "bin": b, "n_true_edges": len(gg),
                 "graph_eff": float(gg.in_graph.mean())}
            ing = gg[gg.in_graph]
            for c in cuts:
                r[f"tpr_in_graph@{c:g}"] = float((ing.score > c).mean()) if len(ing) else np.nan
                r[f"edge_eff@{c:g}"] = float((gg.score > c).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def _chunk(paths):
    res = [event_scores(p) for p in paths]
    fake = [f for e, _, f in res if split_event_id(np.array([e]))[0][0] in HNL_CODES]
    return pd.concat([t for _, t, _ in res], ignore_index=True), np.concatenate(fake) if fake else np.zeros(0)


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", type=float, action="append", default=[])
    ap.add_argument("--efpr", type=float, default=None, help="target edge FPR on the pooled HNL val fake edges")
    ap.add_argument("--wp", default=None, help="working_points.json: the target is its --wp-key (default e_star)")
    ap.add_argument("--wp-key", default="e_star")
    ap.add_argument("--efpr-grid", type=float, default=0.01, help="cut grid step (0: the exact quantile)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--per-event", default=None, help="per-event dD counts at the evaluation cut (parquet)")
    a = ap.parse_args(argv)
    target = a.efpr
    if a.wp:
        wv = float(json.load(open(a.wp))[a.wp_key])
        if target is not None and abs(target - wv) > 1e-12:
            raise SystemExit(f"--efpr {target} differs from {a.wp_key} = {wv} of {a.wp}")
        target = wv
    if target is None and not a.cut:
        raise SystemExit("give --cut, --efpr or --wp")
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_chunk, [paths[k:k + 20] for k in range(0, len(paths), 20)]))
    edges = pd.concat([r[0] for r in res], ignore_index=True)
    info = particle_table(edges.event_id.unique())
    cuts = list(a.cut)
    meta = None
    if target is not None:
        fake = np.concatenate([r[1] for r in res])
        if a.efpr_grid > 0:
            grid = np.round(np.arange(a.efpr_grid, 1.0 - 1e-9, a.efpr_grid), 4)
            cut, reachable, fpr = cut_at_fpr(fake, target, grid)
        else:
            cut, reachable, fpr = cut_at_fpr(fake, target, None)
        if not reachable:
            print(f"WARNING: no cut reaches edge FPR {target}; using the minimum-FPR cut {cut}")
        t = _targets(edges, info)
        dd = delta_d(t[t.population == "hnl_daughters"], cut)
        meta = {"graphs": os.path.abspath(a.graphs), "target_efpr": target, "source": a.wp or "--efpr",
                "grid_step": a.efpr_grid, "cut": cut, "reachable": bool(reachable), "efpr": fpr,
                "n_fake_edges_hnl": int(len(fake)), "n_events": int(edges.event_id.nunique()), **dd}
        if cut not in cuts:
            cuts.append(cut)
    if a.per_event:
        if meta is not None:
            c_eval = meta["cut"]
        elif len(a.cut) == 1:
            c_eval = a.cut[0]
        else:
            raise SystemExit("--per-event needs --efpr / --wp or exactly one --cut")
        t = _targets(edges, info)
        pe = per_event_dd(t[t.population == "hnl_daughters"], c_eval)
        # every event of the graphs, also those without daughter edges (zero rows), for the bootstrap tables
        allev = pd.DataFrame({"event_id": np.unique(edges.event_id.to_numpy(np.int64))})
        pe = allev.merge(pe, on="event_id", how="left").fillna(0)
        pe[["n_low", "k_low", "n_high", "k_high"]] = pe[["n_low", "k_low", "n_high", "k_high"]].astype(np.int64)
        pe["cut"] = c_eval
        pe.to_parquet(a.per_event, index=False)
    s = summarize(edges, info, cuts)
    s.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(s[s.variable == "d0_bin"].round(4).to_string(index=False))
    if meta is not None:
        with open(os.path.splitext(a.out)[0] + "_efpr.json", "w") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
