"""HelixFilter graph booster, hand-cut control (Stage 1 A; design synthesis step 5, DECISIONS.md): a run's
metric-learning graph united with the wider-radius candidates of the SAME embedding that have helix-triplet support.

The ttbar-trained embedding loses displaced pairs at the baseline radius r* (arm (i) seed 0: primary graph ceiling
0.984 on val, arm (ii) 0.993; RESULTS.md M2), and a wider radius recovers them at 3.3x the size. Candidates =
all pairs of the run's own embedding within r_w (default 0.24); a candidate is kept if it is a leg of a helix-consistent
triplet a -> b -> c of candidate legs: s-z slope continuity |dcot| within nsig (per-hit resolution + multiple
scattering) and a turning angle compatible with pT >= pt_min, ranked among the triplets of both its legs (mutual top
k by a consistency score); same-layer short pairs (module overlaps) are kept if both hits are on surviving legs. The
union with the stored base graph keeps every base edge. Prototype (critique scratch, 2026-10-03): primary ceiling
0.997 vs 0.982 at 16.6k vs 14.4k pairs per HNL event and 35.4k vs 28.8k per ttbar event (324 primary particles).

Invariance: the triplet tests use only leg differences (transverse length, slope cot theta, transverse direction),
per-hit resolutions of the hit's detector volume and the "same detector layer" test, so a given triplet's features are
unchanged by translations of the production point and rotations about z. No layer id, no "in base graph" flag and no
absolute position enters a test. Legs are oriented outward in R = sqrt(r^2 + z^2) to pair an in-leg with an out-leg
at the middle hit (the prototype's convention; a track segment moving inward in R is not supported by it). The
per-volume resolutions are the ttbar-train module-frame table (hnl.geometry.resolution; the M1 W-muon table
before), as the strict S1 pipeline requires (design synthesis step 0f).

Union graphs keep the base graph's acorn format: node attributes unchanged; edge_index oriented outward in hit_R and
unique; y and truth_map recomputed exactly as acorn's MetricLearning.get_truth (graph_intersection, undirected); a
provenance entry appended to config. The embedding runs on CPU (KD-tree radius search instead of FRNN): `--check-base`
recomputes the base graph from the embedding at the stored r_infer / knn_infer and the truth of the stored edges.

    python -m hnl.graphs.helix_union build --ml-run RUN --split val --out DIR [--report DIR] [--max-events N]
        [--r-wide 0.24] [--k 2] [--check-base 20]      (DIR/valset/event*.pyg.gz = an acorn input dir)
    python -m hnl.graphs.helix_union build --ml-run RUN --split test --tar RUN/gnn_scores_test/testset.tar
        --no-metrics --out DIR                        (the one-time test: no truth-derived numbers)
    python -m hnl.graphs.helix_union ckpt --run RUN --stage gnn
    python -m hnl.graphs.helix_union rescued --ceiling DIR --trackbuild DIR [--fr-star FR] [--base-trackbuild DIR]
"""

from __future__ import annotations

import argparse
import copy
import glob
import gzip
import io
import json
import sys
import os
import tarfile
import time

import numpy as np
import pandas as pd

from hnl.graphs import ceiling as C

MM_PER_GEV = 1000.0 / (0.3 * 2.0)            # transverse radius [mm] of a 1 GeV track in the 2 T field
MS0 = 0.0136 * np.sqrt(0.03)                 # Highland angle at 1 GeV for ~3 % X0 per layer
BARREL = (8, 13, 17)
# per-volume module-frame resolutions [mm] measured on ttbar train events only (hnl.geometry.resolution.RES): fine
# (r-phi) and coarse (z in the barrel, r in the endcaps); until 2026-10-03 22:39 UTC the M1 table of hnl10_100 W
# muons (RESULTS.md M1), which made the filter not strictly ttbar-only
from hnl.geometry.resolution import RES as _RES
FINE = {v: float(_RES[v][0]) for v in _RES}
COARSE = {v: float(_RES[v][1]) for v in _RES}
COT_CLIP = 200.0
SAME_LAYER_MAX = 30.0                        # [mm] same-layer pairs (module overlaps) kept up to this 3D distance


# --------------------------------------------------------------------------------------------- triplet support
def hit_sigmas(vol: np.ndarray):
    """(fine, sigma_z, sigma_r) per hit: barrel hits measure z coarsely, endcap hits r."""
    vol = np.asarray(vol, dtype=np.int64)
    fine = np.zeros(vol.max(initial=0) + 1)
    coarse = np.zeros_like(fine)
    for v in FINE:
        if v < len(fine):
            fine[v], coarse[v] = FINE[v], COARSE[v]
    if not np.isin(vol, list(FINE)).all():
        raise ValueError(f"unknown volumes {np.setdiff1d(vol, list(FINE))}")
    barrel = np.isin(vol, BARREL)
    return fine[vol], np.where(barrel, coarse[vol], 0.0), np.where(barrel, 0.0, coarse[vol])


def legs(xyz: np.ndarray, src: np.ndarray, dst: np.ndarray, vol: np.ndarray) -> dict:
    """Per oriented leg src -> dst: transverse length L, dz, cot theta, the slope and direction errors and the
    transverse direction angle."""
    fine, sz, sr = hit_sigmas(vol)
    d = xyz[dst] - xyz[src]
    L = np.hypot(d[:, 0], d[:, 1])
    Lc = np.maximum(L, 1e-3)
    cot = d[:, 2] / Lc
    s_cot = np.sqrt(sz[src] ** 2 + sz[dst] ** 2 + cot ** 2 * (sr[src] ** 2 + sr[dst] ** 2)) / Lc
    s_t = np.sqrt(fine[src] ** 2 + fine[dst] ** 2 + 0.25 * (sr[src] ** 2 + sr[dst] ** 2)) / Lc
    return {"L": L, "dz": d[:, 2], "cot": cot, "s_cot": s_cot, "s_t": s_t, "ang": np.arctan2(d[:, 1], d[:, 0])}


def triplet_features(lg: dict, e_in: np.ndarray, e_out: np.ndarray, nsig: float = 5.0, pt_min: float = 0.5):
    """Tests of the triplets (in-leg e_in = a -> b, out-leg e_out = b -> c): (good, score, pull_cot, turn).
    pull_cot = |dcot| / its error (resolutions + multiple scattering at 1 GeV); turn = transverse turning angle,
    allowed up to the arc of a pT = pt_min helix plus nsig errors; score = pull_cot^2 + (turn / its 2 GeV + error
    scale)^2 (lower = more consistent)."""
    L, cot, s_cot, s_t, ang = lg["L"], lg["cot"], lg["s_cot"], lg["s_t"], lg["ang"]
    msq = MS0 ** 2 * (1 + cot[e_out] ** 2)
    pull_cot = np.abs(cot[e_out] - cot[e_in]) / np.sqrt(s_cot[e_in] ** 2 + s_cot[e_out] ** 2 + msq)
    turn = np.remainder(ang[e_out] - ang[e_in] + np.pi, 2 * np.pi) - np.pi
    s_turn = np.sqrt(s_t[e_in] ** 2 + s_t[e_out] ** 2 + msq)
    half = 0.5 * (L[e_in] + L[e_out])
    good = (pull_cot < nsig) & (np.abs(turn) < half / (MM_PER_GEV * pt_min) + nsig * s_turn)
    score = pull_cot ** 2 + (turn / np.sqrt((half / (MM_PER_GEV * 2.0)) ** 2 + s_turn ** 2)) ** 2
    return good, score, pull_cot, turn


def candidate_triplets(src: np.ndarray, dst: np.ndarray, lg: dict, use: np.ndarray, nsig: float = 5.0):
    """(e_in, e_out) pairs of legs meeting at a hit (dst[e_in] == src[e_out]) whose slopes agree within a generous
    window, by a sorted search over (middle hit, clipped cot). The window is capped at the full clipped-cot range,
    and the key stride exceeds it, so a search never reaches another middle hit (the prototype's fixed stride could)."""
    idx = np.flatnonzero(use)
    cot, s_cot = lg["cot"][idx], lg["s_cot"][idx]
    c = np.clip(cot, -COT_CLIP, COT_CLIP)
    w = nsig * (np.sqrt(s_cot ** 2 + MS0 ** 2 * (1 + cot ** 2)) + 0.05 * (1 + np.abs(cot)))
    w = np.minimum(w, 2 * COT_CLIP + 1.0)
    K = 2 * (COT_CLIP + 2 * COT_CLIP + 1.0) + 10.0
    key_in = dst[idx] * K + c
    o = np.argsort(key_in, kind="stable")
    in_sorted, key_sorted = idx[o], key_in[o]
    kc = src[idx] * K + c
    lo = np.searchsorted(key_sorted, kc - w, side="left")
    hi = np.searchsorted(key_sorted, kc + w, side="right")
    cnt = hi - lo
    tot = int(cnt.sum())
    e_out = np.repeat(idx, cnt)
    e_in = in_sorted[np.repeat(lo, cnt) + np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    assert (dst[e_in] == src[e_out]).all()
    return e_in, e_out


def rank_within(group: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Rank (0 = best) of each element within its group by ascending score (ties: input order)."""
    o = np.lexsort((score, group))
    g = group[o]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    first = np.maximum.accumulate(np.where(start, np.arange(len(g)), 0))
    rk = np.empty(len(g), dtype=np.int64)
    rk[o] = np.arange(len(g)) - first
    return rk


def support_filter(r, phi, z, vol, lid, cand, k: int = 2, nsig: float = 5.0, pt_min: float = 0.5):
    """Candidate pairs (2, E) with helix-triplet support: (kept (2, E') unordered pairs, stats). A leg survives if it
    is in a good triplet that is among the k best (lowest score) of both its legs (k = 0: every good triplet); a
    same-layer pair within SAME_LAYER_MAX of 3D distance survives if both its hits are on surviving legs."""
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    R = np.hypot(r, z)
    a, b = np.asarray(cand[0], dtype=np.int64), np.asarray(cand[1], dtype=np.int64)
    fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    src, dst = np.where(fwd, a, b), np.where(fwd, b, a)
    same_layer = lid[src] == lid[dst]
    lg = legs(xyz, src, dst, vol)
    e_in, e_out = candidate_triplets(src, dst, lg, (~same_layer) & (lg["L"] > 1e-3), nsig)
    good, score, _, _ = triplet_features(lg, e_in, e_out, nsig, pt_min)
    e_in, e_out, score = e_in[good], e_out[good], score[good]
    m = np.ones(len(e_in), dtype=bool) if k <= 0 else \
        (rank_within(e_in, score) < k) & (rank_within(e_out, score) < k)
    keep = np.zeros(len(src), dtype=bool)
    keep[e_in[m]] = True
    keep[e_out[m]] = True
    on = np.zeros(len(r), dtype=bool)
    on[src[keep]] = True
    on[dst[keep]] = True
    sl = same_layer & ~keep & (np.hypot(lg["L"], lg["dz"]) < SAME_LAYER_MAX) & on[src] & on[dst]
    keep |= sl
    stats = {"n_cand": int(len(src)), "n_trip_cand": int(len(good)), "n_trip_good": int(len(e_in)),
             "n_trip_kept": int(m.sum()), "n_same_layer_kept": int(sl.sum()), "n_kept": int(keep.sum())}
    return C.unordered_unique(np.stack([src[keep], dst[keep]])), stats


# --------------------------------------------------------------------------------------------- embedding, graphs
def best_ckpt(run: str, stage: str) -> str:
    """Best checkpoint of the latest finished training of a stage (as slurm/baseline/_common.sh best_ckpt, including its
    RUN/<stage>/best_ckpt_override.json)."""
    o = os.path.join(run, stage, "best_ckpt_override.json")
    if os.path.exists(o):
        d = json.load(open(o))
        if not os.path.exists(d["ckpt"]):
            raise SystemExit(f"missing override {d['ckpt']}")
        print(f"best checkpoint override: {d['ckpt']} ({d['reason']})", file=sys.stderr)
        return d["ckpt"]
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(run, stage, "run_*.json"))]
    runs = [x for x in runs if x.get("state") == "done" and x.get("best_ckpt")]
    if not runs:
        raise SystemExit(f"no finished {stage} training in {run}")
    ck = max(runs, key=lambda x: x["end_utc"])["best_ckpt"]
    if not os.path.exists(ck):
        raise SystemExit(f"missing {ck}")
    return ck


class Embedding:
    """acorn MetricLearning's network (make_mlp + F.normalize) from a checkpoint, on CPU."""

    def __init__(self, ckpt: str):
        import torch
        from acorn.stages.graph_construction.models.utils import make_mlp
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        hp = sd["hyper_parameters"]
        self.features, self.scales = list(hp["node_features"]), list(hp["node_scales"])
        self.net = make_mlp(len(self.features), [hp["emb_hidden"]] * hp["nb_layer"] + [hp["emb_dim"]],
                            hidden_activation=hp["activation"], output_activation=None, layer_norm=True)
        self.net.load_state_dict({k[len("network."):]: v for k, v in sd["state_dict"].items()
                                  if k.startswith("network.")})
        self.net.eval()

    def __call__(self, g) -> np.ndarray:
        import torch
        cols = []
        for f, s in zip(self.features, self.scales):
            name = f if f in g else f.removeprefix("hit_")
            cols.append(g[name].float() / s)
        x = torch.stack(cols, dim=-1).float()
        x[x != x] = 0
        with torch.inference_mode():
            return torch.nn.functional.normalize(self.net(x)).numpy()


def radius_pairs(emb: np.ndarray, r: float, knn: int | None = None) -> np.ndarray:
    """Unordered pairs (2, P), i < j, with embedding distance <= r. knn: FRNN's per-query cap K (which counts the
    point itself): an edge is kept if either end has the other among its K - 1 nearest other points."""
    from scipy.spatial import cKDTree
    t = cKDTree(emb.astype(np.float64))
    p = t.query_pairs(r, output_type="ndarray").astype(np.int64)
    if len(p) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    if knn:
        d = np.linalg.norm(emb[p[:, 0]].astype(np.float64) - emb[p[:, 1]].astype(np.float64), axis=1)
        q = np.r_[p[:, 0], p[:, 1]]
        rank = rank_within(q, np.r_[d, d])
        ok = (rank[:len(p)] < knn - 1) | (rank[len(p):] < knn - 1)
        p = p[ok]
    return C.unordered_unique(p.T)


def acorn_truth(edge_index, track_edges, hit_R):
    """acorn MetricLearning.get_truth on an edge list: orient outward in hit_R (ties: lower index first), unique,
    undirected intersection with the chain truth. Returns (edge_index, y, truth_map) as torch tensors."""
    import torch
    from acorn.stages.graph_construction.models.utils import graph_intersection
    e = torch.as_tensor(edge_index, dtype=torch.long).clone()
    hR = torch.as_tensor(hit_R)
    flip = (hR[e[0]] > hR[e[1]]) | ((hR[e[0]] == hR[e[1]]) & (e[0] > e[1]))
    e[:, flip] = e[:, flip].flip(0)
    e, y, tm = graph_intersection(e, torch.as_tensor(track_edges, dtype=torch.long), return_y_pred=True,
                                  return_truth_to_pred=True, unique_pred=False, undirected=True)
    return e, y, tm


def union_graph(g, added: np.ndarray, info: dict):
    """Copy of the stored graph g with edge_index = base U added (acorn orientation and order) and y / truth_map
    recomputed; info is appended to config. Returns (graph, unordered union pairs)."""
    base = C.unordered_unique(g.edge_index.numpy())
    union = C.unordered_unique(np.concatenate([base, np.asarray(added, dtype=np.int64).reshape(2, -1)], axis=1))
    ei, y, tm = acorn_truth(union, g.track_edges, g.hit_R)
    h = copy.copy(g)
    h.edge_index, h.y, h.truth_map = ei, y, tm
    h.config = list(g.config) + [info]
    return h, union


def union_event(g, emb: Embedding, lid_vol, r_wide: float, k: int, nsig: float, pt_min: float,
                check_base: bool = False, metrics: bool = True, triplet_model=None, tau: float | None = None):
    """(union graph, per-event stats, per-particle base / union ceilings) of one stored metric-learning graph.
    metrics=False (test split): no truth-derived numbers (true-edge counts, ceilings); the per-particle table is empty
    and the stats hold only graph sizes and the support-filter counts. triplet_model (with tau): the learned triplet
    support of hnl.graphs.learned_triplet_filter (mutual top k, p > tau) instead of the hand-cut triplet tests."""
    import torch
    from acorn.utils.loading_utils import flatten_event_id
    eid = int(flatten_event_id(g.event_id))
    vol, lid = lid_vol
    n = int(g.num_nodes)
    if len(vol) != n:
        raise ValueError(f"event {eid}: {len(vol)} hits in the table, {n} in the graph")
    r, phi, z = (g[f].double().numpy() for f in ("r", "phi", "z"))
    e = emb(g)
    base = C.unordered_unique(g.edge_index.numpy())
    wide = radius_pairs(e, r_wide)
    if triplet_model is None:
        kept, st = support_filter(r, phi, z, vol, lid, wide, k=k, nsig=nsig, pt_min=pt_min)
        info = {"stage": "helix_union", "r_wide": r_wide, "k": k, "nsig": nsig, "pt_min": pt_min}
    else:
        from hnl.graphs import learned_triplet_filter as LT
        s_, d_, lg, sl, ei, eo, X = LT.legs_and_triplets(r, phi, z, vol, lid, wide)
        p = triplet_model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
        kept = LT.select(s_, d_, lg, sl, ei, eo, p, tau, k, n)
        st = {"n_cand": int(len(s_)), "n_trip_cand": int(len(ei)), "n_trip_good": int((p > tau).sum()),
              "n_kept": int(kept.shape[1])}
        info = {"stage": "helix_union", "filter": "learned_triplet", "r_wide": r_wide, "k": k, "tau": tau}
    h, union = union_graph(g, kept, info)
    pid = g.hit_particle_id.numpy()
    te = g.track_edges.numpy()
    stats = {"event_id": eid, "n_hits": n, "n_base": base.shape[1], "n_wide": wide.shape[1], "n_union": union.shape[1],
             "n_added": union.shape[1] - base.shape[1], **st}
    parts = [pd.DataFrame(columns=["particle_id", "dm_matched", "dm_matched_samepid", "event_id", "source"])]
    for name, graph in (("base", base), ("union", union)) if metrics else ():
        s, part, _ = C.evaluate_graph(n, pid, te, graph)
        stats[f"n_true_in_{name}"] = s["n_true_in_graph"]
        stats["n_true"] = s["n_true"]
        part = part.reset_index()[["particle_id", "dm_matched", "dm_matched_samepid"]]
        part["event_id"], part["source"] = eid, name
        parts.append(part)
    if check_base:
        # the stored base graph against the CPU embedding at the stored r_infer / knn_infer, and its stored truth
        hp = [c for c in g.config if isinstance(c, dict) and c.get("stage") == "graph_construction"][-1]
        rec = radius_pairs(e, float(hp["r_infer"]), int(hp.get("knn_infer", 500)))
        bc, rc = C._codes(base, n), C._codes(rec, n)
        only_s, only_r = np.setdiff1d(bc, rc), np.setdiff1d(rc, bc)
        dist = lambda codes: np.linalg.norm(e[codes // n].astype(np.float64) - e[codes % n].astype(np.float64), axis=1)
        dev = np.abs(np.r_[dist(only_s), dist(only_r)] - float(hp["r_infer"]))
        e0, y0, tm0 = acorn_truth(g.edge_index, g.track_edges, g.hit_R)
        stats.update({"check_only_stored": len(only_s), "check_only_recomputed": len(only_r),
                      "check_max_dist_dev": float(dev.max()) if len(dev) else 0.0,
                      "check_truth_ok": bool(torch.equal(e0, g.edge_index) and torch.equal(y0, g.y)
                                             and torch.equal(tm0, g.truth_map))})
    return h, stats, pd.concat(parts, ignore_index=True)


def _load_bytes(data: bytes):
    import torch
    return torch.load(io.BytesIO(gzip.decompress(data)), weights_only=False)


GNN_KEYS = ("scores",)                       # attributes acorn's edge-classifier inference adds to a stored graph
COLLATE_KEYS = ("batch", "ptr")              # PyG collation leftovers: kept where present, never required


def reference_keys(tar_path: str) -> list[str]:
    """Attribute names of the first stored graph of a tar (e.g. RUN/ml_graphs/valset.tar: the metric-learning form)."""
    with tarfile.open(tar_path) as t:
        nm = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))[0]
        return sorted(_load_bytes(t.extractfile(nm).read()).keys())


def reference_graph_construction(tar_path: str) -> dict | None:
    """The graph_construction config entry of the first stored graph of a tar (RUN/ml_graphs/valset.tar)."""
    with tarfile.open(tar_path) as t:
        nm = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))[0]
        g = _load_bytes(t.extractfile(nm).read())
    gc = [c for c in g.config if isinstance(c, dict) and c.get("stage") == "graph_construction"]
    return dict(gc[-1]) if gc else None


def ml_form(g, keep: list[str] | None = None, gc: dict | None = None):
    """A stored graph in its metric-learning form: the edge classifier's added attributes (GNN_KEYS; with `keep`, every
    attribute not in it) and config entry removed, so that union graphs built from gnn_scores_test/ have the format of
    the train / val union graphs built from ml_graphs/. Edge list, order and truth are untouched (the build's
    --check-base events recompute the base graph and its truth from the embedding)."""
    drop = [k for k in g.keys() if k in GNN_KEYS or (keep is not None and k not in keep)]
    for key in drop:
        del g[key]
    if keep is not None:
        missing = sorted(set(keep) - set(g.keys()) - set(COLLATE_KEYS))
        if missing:
            raise ValueError(f"graph lacks metric-learning attributes {missing}")
    g.config = [c for c in g.config if not (isinstance(c, dict) and c.get("stage") == "edge_classifier")]
    # the Stage-0 test graphs carry no graph_construction entry: the run's own entry of its val graphs (same metric-
    # learning model, r_infer, knn_infer) is added, marked, so that the base reproduction check can run on them
    if gc is not None and not any(isinstance(c, dict) and c.get("stage") == "graph_construction" for c in g.config):
        g.config = list(g.config) + [{**gc, "copied_from_val_graphs": True}]
    return g


def mlform_dir(d: str, keep: list[str] | None, workers: int = 1, gc: dict | None = None) -> int:
    """ml_form on every event*.pyg.gz of a directory, rewritten in place (only graphs that change); returns that count."""
    import concurrent.futures as cf
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pyg.gz"))
    with cf.ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        return sum(ex.map(_mlform_file, files, [keep] * len(files), [gc] * len(files), chunksize=50))


def _mlform_file(path: str, keep, gc=None) -> int:
    from acorn.utils.loading_utils import save_pyg
    g = _load_bytes(open(path, "rb").read())
    before = (sorted(g.keys()), len(g.config))
    g = ml_form(g, keep, gc)
    if (sorted(g.keys()), len(g.config)) == before:
        return 0
    save_pyg(g, path[:-len(".gz")])                      # atomic (tmp + replace), acorn's compression
    return 1


def _chunk(args):
    from acorn.utils.loading_utils import save_pyg
    from hnl.trackbuild.mutex_ws import hit_layers
    import torch
    src, names, ckpt, out_dir, opt, n_check = args
    torch.set_num_threads(1)
    emb = Embedding(ckpt)
    model = None
    if opt.get("triplet_model"):
        import joblib
        model = joblib.load(opt["triplet_model"])
    if os.path.isdir(src):
        graphs = [_load_bytes(open(os.path.join(src, nm), "rb").read()) for nm in names]
    else:
        with tarfile.open(src) as t:
            graphs = [_load_bytes(t.extractfile(nm).read()) for nm in names]
    graphs = [ml_form(g, opt.get("keep"), opt.get("gc")) for g in graphs]
    eids = [int(os.path.basename(nm)[5:].split(".")[0]) for nm in names]
    L = hit_layers(eids)
    stats, parts = [], []
    for i, (eid, g) in enumerate(zip(eids, graphs)):
        lid = L[eid]
        h, st, pp = union_event(g, emb, (lid // 1000, lid), opt["r_wide"], opt["k"], opt["nsig"], opt["pt_min"],
                                check_base=i < n_check, metrics=opt["metrics"], triplet_model=model,
                                tau=opt.get("tau"))
        save_pyg(h, os.path.join(out_dir, f"event{eid}.pyg"))
        stats.append(st)
        parts.append(pp)
    return pd.DataFrame(stats), pd.concat(parts, ignore_index=True)


def summarize(report: str) -> pd.DataFrame:
    """Per source (base / union): pairs per HNL and ttbar event, true-edge efficiency and the ceilings (chain truth /
    same particle) of the preregistered populations; the rescued set R (primary targets DM-connected in the union
    ceiling but not in the base one) to rescued.csv. Written to summary.csv."""
    from hnl.eval.baseline_val import populations
    from hnl.eval.particles import particle_table, split_event_id
    pp = pd.read_parquet(os.path.join(report, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(report, "per_event.parquet"))
    info = particle_table(pp.event_id.unique())
    pops = populations(info)
    keys = ("primary", "hnl_targets", "w_muons", "ttbar_targets", "ttbar_prompt")
    flags = pd.DataFrame({"event_id": info.event_id, "particle_id": info.particle_id, "d0": info.d0, "rv": info.rv,
                          "n_hits": info.n_hits, **{k: pops[k] for k in keys}})
    t = pp.merge(flags, on=["event_id", "particle_id"], how="inner")
    hnl_ev = np.isin(split_event_id(pe.event_id.to_numpy())[0], (2, 4))
    rows = []
    for src in ("base", "union"):
        g = t[t.source == src]
        row = {"source": src, "pairs_hnl": pe[hnl_ev][f"n_{src}"].mean(), "pairs_ttbar": pe[~hnl_ev][f"n_{src}"].mean(),
               "true_edge_eff": pe[f"n_true_in_{src}"].sum() / pe.n_true.sum()}
        for k in keys:
            row[f"ceil_{k}"] = g.dm_matched[g[k]].mean()
            row[f"ceil_samepid_{k}"] = g.dm_matched_samepid[g[k]].mean()
            row[f"n_{k}"] = int(g[k].sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(report, "summary.csv"), index=False)
    key = ["event_id", "particle_id"]
    both = t[(t.source == "base") & t.primary][key + ["dm_matched", "d0", "rv", "n_hits"]].merge(
        t[t.source == "union"][key + ["dm_matched"]], on=key, suffixes=("_base", "_union"), validate="one_to_one")
    resc = both[both.dm_matched_union.astype(bool) & ~both.dm_matched_base.astype(bool)][key + ["d0", "rv", "n_hits"]]
    resc.to_csv(os.path.join(report, "rescued.csv"), index=False)
    return out


# --------------------------------------------------------------------------------------------- rescued at FR*
def best_configs(c: pd.DataFrame, cfgs: list, fr: float) -> dict:
    """Per builder family: the configuration index with fake rate <= fr and the highest primary efficiency (ties: the
    lower fake rate), as hnl.eval.trackbuild_select.select."""
    out = {}
    fam = pd.Series([cfgs[int(i)][0] for i in c.index], index=c.index)
    for f in fam.unique():
        g = c[(fam == f) & (c.fake_rate_hnl <= fr + 1e-12)]
        if len(g):
            out[f] = int(g.sort_values(["eff_primary", "fake_rate_hnl"], ascending=[False, True]).index[0])
    return out


def rescued_at_fr(ceiling: str, trackbuild: str, fr: float, base_trackbuild: str | None = None) -> pd.DataFrame:
    from hnl.eval.baseline_val import curves
    from hnl.eval.particles import particle_table
    R = pd.read_csv(os.path.join(ceiling, "rescued.csv"))
    rows = []
    events = None
    for label, d in (("union", trackbuild), ("base", base_trackbuild)):
        if d is None:
            continue
        pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
        pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
        if events is None:
            events = np.unique(pe.event_id)
        else:                                # the base grid on the union graphs' events only (same FR* events)
            if not np.isin(events, pe.event_id).all():
                raise SystemExit(f"{d} lacks events of {trackbuild}")
            pp, pe = pp[pp.event_id.isin(events)], pe[pe.event_id.isin(events)]
        cfgs = json.load(open(os.path.join(d, "configs.json")))
        c = curves(pp, pe, particle_table(events))
        key = pd.MultiIndex.from_frame(R[["event_id", "particle_id"]])
        for fam, i in best_configs(c, cfgs, fr).items():
            m = pp[pp.config == i].set_index(["event_id", "particle_id"]).dm_matched
            got = m.reindex(key).fillna(False).astype(bool)
            rows.append({"graph": label, "family": fam, "params": ",".join(str(x) for x in cfgs[i][1:]),
                         "eff_primary": c.loc[i, "eff_primary"], "fake_rate_hnl": c.loc[i, "fake_rate_hnl"],
                         "eff_w_muons": c.loc[i, "eff_w_muons"], "eff_ttbar_prompt": c.loc[i, "eff_ttbar_prompt"],
                         "n_rescued_set": len(R), "n_rescued_matched": int(got.sum()),
                         "frac_rescued_matched": float(got.mean()) if len(R) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ml-run", required=True, help="run holding ml_graphs/ and the metric-learning checkpoint")
    b.add_argument("--split", default="val", choices=["train", "val", "test"])
    b.add_argument("--graphs", default=None, help="a directory of stored graphs instead of ML_RUN/ml_graphs/<split>set.tar")
    b.add_argument("--tar", default=None, help="a tar of stored graphs instead, e.g. RUN/gnn_scores_test/testset.tar "
                   "(GNN-scored graphs are returned to their metric-learning form)")
    b.add_argument("--no-metrics", action="store_true", help="no truth-derived numbers (required for --split test)")
    b.add_argument("--triplet-model", default=None, help="learned triplet support (hnl.graphs.learned_triplet_filter) "
                   "instead of the hand-cut triplet tests; with --tau and --k")
    b.add_argument("--tau", type=float, default=None)
    b.add_argument("--out", required=True, help="acorn input dir: union graphs go to OUT/<split>set/")
    b.add_argument("--report", default=None, help="per-event stats, ceilings, summary, rescued set (default OUT/report)")
    b.add_argument("--max-events", type=int, default=None)
    b.add_argument("--r-wide", type=float, default=0.24)
    b.add_argument("--k", type=int, default=2)
    b.add_argument("--nsig", type=float, default=5.0)
    b.add_argument("--pt-min", type=float, default=0.5)
    b.add_argument("--check-base", type=int, default=20, help="events per chunk-0 with the base reproduction check")
    b.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    m = sub.add_parser("mlform", help="a directory of GNN-scored graphs back to the metric-learning form, in place")
    m.add_argument("--dir", required=True)
    m.add_argument("--like", required=True, help="tar whose first graph gives the attribute set (RUN/ml_graphs/valset.tar)")
    m.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    c = sub.add_parser("ckpt")
    c.add_argument("--run", required=True)
    c.add_argument("--stage", required=True, choices=["ml", "gnn"])
    s = sub.add_parser("rescued")
    s.add_argument("--ceiling", required=True)
    s.add_argument("--trackbuild", required=True)
    s.add_argument("--base-trackbuild", default=None)
    s.add_argument("--fr-star", type=float, default=0.031402365961525895)
    a = ap.parse_args(argv)
    if a.cmd == "ckpt":
        print(best_ckpt(a.run, a.stage))
        return
    if a.cmd == "mlform":
        keep, gc = reference_keys(a.like), reference_graph_construction(a.like)
        print(json.dumps({"dir": a.dir, "keep": keep, "graph_construction": gc,
                          "rewritten": mlform_dir(a.dir, keep, a.workers, gc)}, default=str))
        return
    if a.cmd == "rescued":
        tab = rescued_at_fr(a.ceiling, a.trackbuild, a.fr_star, a.base_trackbuild)
        tab.to_csv(os.path.join(a.ceiling, f"rescued_at_FR_{a.fr_star:.6f}.csv"), index=False)
        pd.set_option("display.width", 250)
        print(tab.round(4).to_string(index=False))
        return
    t0 = time.time()
    split = f"{a.split}set"
    out_dir = os.path.join(a.out, split)
    report = a.report or os.path.join(a.out, "report")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report, exist_ok=True)
    if a.split == "test" and not a.no_metrics:
        raise SystemExit("--split test needs --no-metrics: no test ceilings before the one-time test evaluation")
    if a.graphs:
        src = a.graphs
        names = sorted(f for f in os.listdir(src) if f.endswith(".pyg.gz"))
    else:
        src = a.tar or os.path.join(a.ml_run, "ml_graphs", f"{split}.tar")
        with tarfile.open(src) as t:
            names = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))
    names = names[:a.max_events]
    ckpt = best_ckpt(a.ml_run, "ml")
    if a.triplet_model and a.tau is None:
        raise SystemExit("--triplet-model needs --tau")
    opt = {"r_wide": a.r_wide, "k": a.k, "nsig": a.nsig, "pt_min": a.pt_min, "metrics": not a.no_metrics,
           "triplet_model": a.triplet_model, "tau": a.tau,
           "keep": reference_keys(os.path.join(a.ml_run, "ml_graphs", "valset.tar")) if a.tar else None,
           "gc": reference_graph_construction(os.path.join(a.ml_run, "ml_graphs", "valset.tar")) if a.tar else None}
    chunks = [(src, names[i:i + 25], ckpt, out_dir, opt, a.check_base if i == 0 else 0)
              for i in range(0, len(names), 25)]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_chunk, chunks))
    pe = pd.concat([x[0] for x in res], ignore_index=True)
    pp = pd.concat([x[1] for x in res], ignore_index=True)
    pe.to_parquet(os.path.join(report, "per_event.parquet"), index=False)
    pp.to_parquet(os.path.join(report, "per_particle.parquet"), index=False)
    meta = {"ml_run": a.ml_run, "ml_ckpt": ckpt, "source": src, "n_events": len(names), **opt,
            "check_events": min(a.check_base, len(names)),
            "elapsed_s": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    chk = pe.dropna(subset=["check_truth_ok"]) if "check_truth_ok" in pe else pe.iloc[:0]
    if len(chk):
        meta["check"] = {"events": len(chk), "only_stored": int(chk.check_only_stored.sum()),
                         "only_recomputed": int(chk.check_only_recomputed.sum()),
                         "max_dist_dev": float(chk.check_max_dist_dev.max()),
                         "truth_ok": bool(chk.check_truth_ok.astype(bool).all())}
    with open(os.path.join(report, "build.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))
    pd.set_option("display.width", 250)
    if a.no_metrics:
        print(pe[["n_base", "n_union", "n_added"]].describe().round(1).to_string())
        return
    print(summarize(report).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
