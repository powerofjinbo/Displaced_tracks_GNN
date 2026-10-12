"""acorn's own CC + walkthrough track builder on our scored graphs (feedback round 1, strongest baselines).

acorn's walkthrough builder is `CCandWalk` (acorn/stages/track_building/models/cc_and_walk.py, "chaining of connected
components and walkthrough algorithm (a la L2IT)"; the GNN4ITk builder). Per event, CCandWalk._build_tracks_one_evt
runs these functions of acorn/stages/track_building/models/cc_and_walk_utils.py (unpatched at our pin 99d15341):

  remove_cycles          every edge points outward in R = hit_r^2 + hit_z^2 (float32 as stored; ties by hit index)
  filter_graph           edges with score <= score_cut_cc removed, then the isolated hits (networkx DiGraph)
  topological_sort_graph a copy of the graph with its hits in topological order
  get_simple_path        weakly connected components in which every hit has <= 1 inward and <= 1 outward edge are
                         tracks if they have > 2 hits, and are removed from the graph either way
  walk_through           on the remaining hits (topological order): from each unused hit follow the best outward edge
                         above score_cut_walk.min and also every other one above score_cut_walk.add; the longest road
                         (ties: the last one found) is a track
  walk_through_score_guided  the same with score_guided_walk: True (guidance_mode_walk 0 / 1 / 2)
  add_track_labels       hits of the "cc" tracks, then of the "walk" tracks, numbered in that order (first track wins)

`AcornCCWalk.labels(..., fast=False)` calls these functions themselves, in this order, on the stored graph (stored edge
order, float32 coordinates and scores, as acorn's track-building GraphDataset hands them over), with two deviations
that change no track: add_track_labels is replaced by the identical mapping `track_labels` (acorn's version fails under
pandas 3: fillna(-1) on its string column reco_method), and the filtered graph of one score_cut_cc is built once per
event (topological_sort_graph copies it without modifying it).

`fast=True` (default) is the same except for the two walk loops: `_get_tracks` / `_get_tracks_score_guided` are line by
line copies of acorn's get_tracks / get_tracks_score_guided / build_roads / find_next_hits in which the used hits are a
set instead of a list rebuilt with np.unique at every step (acorn spends most of its time there). Every iteration
order is kept: the neighbour list `list(set(G.neighbors(h)).difference(set(used_hits)))` is rebuilt as CPython's
set.difference builds it (copy-and-discard when len(neighbours) // 4 > len(used_hits), else insert the survivors into a
new set), with len(used_hits) = |used ∪ road| as acorn passes it; the road's own hits are predecessors of h in the
outward-oriented DAG, so never among its neighbours. `python -m hnl.trackbuild.acorn_walk check` requires identical
track partitions from the fast path and from acorn's own CCandWalk._build_tracks_one_evt (tests/test_acorn_walk.py
does the same on synthetic events).

The labels are returned in our convention (hnl.trackbuild.builders: -1 = no track, tracks with >= 2 hits; acorn's
one-hit walk "tracks" become -1, which no efficiency or fake-rate number can see).

acorn's published configurations: CTD 2023 (examples/CTD_2023/track_building_infer.yaml) score_cut_cc 0.01, walk
min 0.1 / add 0.6, plain walk; Rel. 24 and ACAT 2025 (examples/Example_ATLAS-P2-RUN4-03-00-00_Rel.24,
examples/ACAT_2025, CI/track_building/ci_track_building_cc_and_walk.yml): the same thresholds with
score_guided_walk: True and guidance_mode_walk: 2.
"""

from __future__ import annotations

from itertools import chain

import numpy as np

from hnl.trackbuild.builders import _relabel


def _utils():
    from acorn.stages.track_building.models import cc_and_walk_utils as U
    return U


def track_labels(n: int, all_trks: dict) -> np.ndarray:
    """acorn add_track_labels without its method column: label of each hit (0..n-1 = hit_id), -1 if on no track."""
    U = _utils()
    hit, tid, off = [], [], 0
    for trks in all_trks.values():
        lens = U._track_lengths(trks)
        hit += U._flatten_track_collection(trks)
        tid += list(chain.from_iterable([[i + off] * L for i, L in enumerate(lens)]))
        off += len(trks)
    lab = np.full(n, -1, dtype=np.int64)
    if hit:
        h = np.asarray(hit, dtype=np.int64)
        t = np.asarray(tid, dtype=np.int64)
        _, first = np.unique(h, return_index=True)        # drop_duplicates(subset="hit_id"): first occurrence
        lab[h[first]] = t[first]
    return lab


# ---- fast exact copies of acorn's walk loops (cc_and_walk_utils at 99d15341) -------------------------------------

def _find_next_hits(G, current_hit, used: set, n_used: int, score_name: str, th_min: float, th_add: float):
    """acorn find_next_hits with used_hits of n_used distinct hits, of which only `used` can be neighbours."""
    so = set(G.neighbors(current_hit))
    if (len(so) >> 2) > n_used:             # CPython set_difference: copy, then discard the common hits
        res = so.copy()
        res.difference_update(used)
    else:                                   # iterate so, insert the survivors into a new set
        res = set()
        for x in so:
            if x not in used:
                res.add(x)
    neighbors = list(res)
    if len(neighbors) < 1:
        return None
    neighbors_scores = [G.edges[(current_hit, i)][score_name] for i in neighbors]
    if max(neighbors_scores) <= th_min:
        return None
    sorted_idx = list(reversed(np.argsort(neighbors_scores)))
    next_hits = [neighbors[sorted_idx[0]]]
    if len(sorted_idx) > 1:
        for ii in range(1, len(sorted_idx)):
            idx = sorted_idx[ii]
            score = neighbors_scores[idx]
            if score > th_add:
                next_hits.append(neighbors[idx])
            else:
                break
    return next_hits


def _build_roads(G, starting_node, used: set, score_name, th_min, th_add, explore_alternatives=False):
    """acorn build_roads with next_hit_fn = find_next_hits; used_hits = the used hits (first step), then
    np.unique(used_hits + list(pp)) (|used ∪ road| distinct hits)."""
    alternative = False
    next_hits = _find_next_hits(G, starting_node, used, len(used), score_name, th_min, th_add)
    if next_hits is None:
        return [(starting_node, None)], alternative
    path = []
    if not explore_alternatives:
        for hit in next_hits:
            path.append((starting_node, hit))
    else:
        for hit in next_hits:
            bestprob = -1
            thebest = None
            for p in list(G.predecessors(hit)):
                if G.edges[(p, hit)][score_name] > bestprob:
                    bestprob = G.edges[(p, hit)][score_name]
                    thebest = p
            if thebest == starting_node:
                path.append((starting_node, hit))
            else:
                alternative = True
                path.append((thebest, hit))
    while True:
        new_path = []
        is_all_none = True
        for pp in path:
            if pp[-1] is not None:
                is_all_none = False
                break
        if is_all_none:
            break
        for pp in path:
            start = pp[-1]
            if start is None:
                new_path.append(pp)
                continue
            n_used = len(used) + len({x for x in pp if x not in used})
            next_hits = _find_next_hits(G, pp[-1], used, n_used, score_name, th_min, th_add)
            if next_hits is None:
                new_path.append(pp + (None,))
            else:
                for hit in next_hits:
                    new_path.append(pp + (hit,))
        path = new_path
    return path, alternative


def _road_hits(G, road) -> list:
    """Hit ids of a road's edge subgraph (its hits; every consecutive pair of a road is an edge of G)."""
    return [G.nodes[node]["hit_id"] for node in dict.fromkeys(road[:-1])]


def _get_tracks(G, th_min, th_add, score_name) -> list:
    """acorn walk_through (get_tracks), as lists of hit ids."""
    U = _utils()
    used: set = set()
    tracks = []
    for node in G.nodes():
        if node in used:
            continue
        road, alternative = _build_roads(G, node, used, score_name, th_min, th_add)
        a_road = U.choose_longest_road(road)
        if len(a_road) < 3:
            used.add(node)
            tracks.append([G.nodes[node]["hit_id"]])
            continue
        tracks.append(_road_hits(G, a_road))
        used.update(a_road[:-1])
    return tracks


def _get_tracks_score_guided(G, th_min, th_add, score_name, guidance_mode) -> list:
    """acorn walk_through_score_guided (get_tracks_score_guided), as lists of hit ids."""
    U = _utils()
    used: set = set()
    tracks = []
    for node in G.nodes():
        alternative = True
        while alternative:
            if node in used:
                alternative = False
                continue
            road, alternative = _build_roads(G, node, used, score_name, th_min, th_add, explore_alternatives=True)
            if len(road) > 1:
                if guidance_mode == 0:
                    a_road = U.choose_longest_road(road)
                elif guidance_mode == 1:
                    a_road = U.choose_longest_most_likely_road(G, score_name, road)
                elif guidance_mode == 2:
                    a_road = U.choose_longest_most_likely_road_local(G, score_name, road)
                else:
                    raise ValueError("Undefined guidance mode")
            else:
                a_road = road[0]
            if len(a_road) < 3:
                used.add(node)
                tracks.append([G.nodes[node]["hit_id"]])
                continue
            tracks.append(_road_hits(G, a_road))
            used.update(a_road[:-1])
    return tracks


class AcornCCWalk:
    """acorn CCandWalk for one event, shared by any number of (score_cut_cc, min, add, guided, mode) settings."""

    def __init__(self, n: int, edge_index: np.ndarray, score: np.ndarray, r: np.ndarray, z: np.ndarray):
        import torch
        from torch_geometric.data import Data
        U = _utils()
        ei = np.asarray(edge_index, dtype=np.int64)
        a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
        if len(np.unique(a * n + b)) != ei.shape[1]:
            raise ValueError("a hit pair is stored twice: acorn's DiGraph would keep one of its scores")
        g = Data(hit_id=torch.arange(n), hit_r=torch.as_tensor(np.asarray(r, dtype=np.float32)),
                 hit_z=torch.as_tensor(np.asarray(z, dtype=np.float32)),
                 edge_index=torch.as_tensor(ei.copy()), edge_scores=torch.as_tensor(np.asarray(score, dtype=np.float32)),
                 num_nodes=n)
        self.n = n
        self.g = U.remove_cycles(g)
        self._filtered = {}

    @classmethod
    def from_path(cls, path: str) -> "AcornCCWalk":
        """From a stored scored graph (acorn GNN output), in its stored edge order."""
        from acorn.utils.loading_utils import load_pyg
        g = load_pyg(path[:-3] if path.endswith(".gz") else path, weights_only=False)
        n = len(g.hit_particle_id)
        sc = g.scores if "scores" in g else g.edge_scores
        r = g.r if "r" in g else g.hit_r
        z = g.z if "z" in g else g.hit_z
        return cls(n, g.edge_index.numpy(), sc.numpy(), r.numpy(), z.numpy())

    def labels(self, cut: float, th_min: float, th_add: float, guided: bool = False, mode: int = 2,
               fast: bool = True) -> np.ndarray:
        U = _utils()
        cut = float(cut)
        if cut not in self._filtered:
            self._filtered[cut] = U.filter_graph(self.g, "edge_scores", cut)
        G = U.topological_sort_graph(self._filtered[cut])
        trks = {"cc": U.get_simple_path(G)}
        th_min, th_add = float(th_min), float(th_add)
        if th_add < th_min:
            raise ValueError("acorn's walk expects add >= min")
        if fast:
            trks["walk"] = (_get_tracks_score_guided(G, th_min, th_add, "edge_scores", int(mode)) if guided
                            else _get_tracks(G, th_min, th_add, "edge_scores"))
        elif guided:
            trks["walk"] = U.walk_through_score_guided(G, "edge_scores", th_min, th_add, int(mode))
        else:
            trks["walk"] = U.walk_through(G, "edge_scores", th_min, th_add)
        return _relabel(track_labels(self.n, trks))


def acorn_method_labels(path: str, cut: float, th_min: float, th_add: float, guided: bool, mode: int = 2) -> np.ndarray:
    """Reference: acorn's own CCandWalk._build_tracks_one_evt on the stored graph (nothing saved), in our convention.
    Needs pandas' future.infer_string off (set by `check`) so that acorn's add_track_labels runs under pandas 3."""
    import torch
    from torch_geometric.data import Data
    from acorn.stages.track_building.models.cc_and_walk import CCandWalk
    from acorn.utils.loading_utils import load_pyg
    g = load_pyg(path[:-3] if path.endswith(".gz") else path, weights_only=False)
    n = len(g.hit_particle_id)
    gg = Data(hit_id=torch.arange(n), hit_r=g.r if "r" in g else g.hit_r, hit_z=g.z if "z" in g else g.hit_z,
              edge_index=g.edge_index.clone(), edge_scores=g.scores if "scores" in g else g.edge_scores, num_nodes=n)
    obj = CCandWalk.__new__(CCandWalk)       # the method reads only these attributes (no stage directories)
    obj.hparams = {"score_cut_cc": float(cut), "score_cut_walk": {"min": float(th_min), "add": float(th_add)},
                   "guidance_mode_walk": int(mode), "save_tracks": False, "save_graph": False}
    obj.cc_only, obj.score_guided_walk = False, bool(guided)
    obj._build_tracks_one_evt(gg, output_dir=None)
    return _relabel(gg.hit_track_labels.numpy().astype(np.int64))


def _partition(lab: np.ndarray) -> list:
    out: dict = {}
    for i, t in enumerate(lab):
        if t >= 0:
            out.setdefault(int(t), []).append(i)
    return sorted(tuple(v) for v in out.values())


def _check_one(args):
    import time
    path, cfgs = args
    import pandas as pd
    pd.set_option("future.infer_string", False)
    w = AcornCCWalk.from_path(path)
    bad, t_fast, t_ref = [], 0.0, 0.0
    for fam, cut, mn, add in cfgs:
        guided = fam == "acwalkg"
        t0 = time.process_time()
        ours = w.labels(cut, mn, add, guided=guided)
        t1 = time.process_time()
        ref = acorn_method_labels(path, cut, mn, add, guided)
        t2 = time.process_time()
        t_fast, t_ref = t_fast + t1 - t0, t_ref + t2 - t1
        if not (np.array_equal(ours, ref) or _partition(ours) == _partition(ref)):
            bad.append([fam, cut, mn, add])
    return path, len(cfgs), bad, t_fast, t_ref


def main(argv=None):
    """check: the fast AcornCCWalk path against acorn's own CCandWalk._build_tracks_one_evt on every graph of a
    directory and every acwalk / acwalkg configuration of hnl.eval.trackbuild_val (identical track partitions)."""
    import argparse
    import concurrent.futures as cf
    import glob
    import json
    import os
    from hnl.eval.trackbuild_val import configs
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--configs", default=None, help="only these configurations: 'fam,cut,min,add;fam,cut,min,add'")
    a = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
    if a.configs:
        cfgs = [(c.split(",")[0], *[float(x) for x in c.split(",")[1:]]) for c in a.configs.split(";")]
    else:
        cfgs = [tuple(c) for c in configs(["acwalk", "acwalkg"])]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_check_one, [(p, cfgs) for p in paths]))
    n = sum(r[1] for r in res)
    bad = [{"graph": os.path.basename(r[0]), "configs": r[2]} for r in res if r[2]]
    out = {"graphs": a.graphs, "n_events": len(paths), "n_configs": len(cfgs), "n_event_configs": n,
           "n_different": sum(len(b["configs"]) for b in bad), "cpu_s_fast": round(sum(r[3] for r in res), 1),
           "cpu_s_acorn_method": round(sum(r[4] for r in res), 1), "different": bad}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "different"}))


if __name__ == "__main__":
    main()
