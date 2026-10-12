"""Far-displaced HNL daughters on the HNL TRAINING split: where Canto-L and acorn (ttbar) lose them (diagnostic only).

Population: the primary-population HNL daughters (HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm) with
|d0| >= 200 mm in the hnl10_100 and hnl15_200 TRAIN events (splits/<sample>_train.txt), about five times the
validation statistics (DECISIONS.md, 2026-10-04 01:52 UTC, item 4). Both pipelines studied here are trained on ttbar
only (Stage-0 arm i): the metric-learning embedding, the learned triplet support, the edge classifiers, both
learned-merging GBDTs and the resolution table come from ttbar TRAIN events, the checkpoints were chosen on ttbar val
events and the builder configurations on the val split, so no step of theirs has seen these events. Nothing is chosen
on them. No test event, test graph, test score or test output is read: the feature-store tars are opened one split at
a time (<sample>_train.tar, <sample>_val.tar) and every run file used is a val or train file.

Graphs and scores (slurm/feedback/far_trainsplit_graphs.sbatch SEED: one A30 job per seed, the steps of the val graphs
of these runs): the events are staged from the feature store (`stage`); acorn's MetricLearning inference at the run's
r_infer / knn_infer (hnl.train.acorn_runner infer --stage ml, as slurm/baseline/ml_infer.sbatch); Canto-L's union with
the learned triplet support of the seed's own GBDT (hnl.graphs.helix_union build --triplet-model
RUN/learned_triplet_filter_v2/model.joblib --tau 0.005 --k 5, r_w 0.24, as slurm/stage1/union_graphs.sbatch); the edge
scores of acorn (ttbar) on its metric-learning graph and of Canto-L on its union graph (hnl.train.acorn_runner infer
--stage gnn with each run's best checkpoint, as slurm/baseline/gnn_infer.sbatch and slurm/stage1/variant.sbatch). All
events are staged as "valset", like the val graphs: Canto-L's model class (hnl.models.augment.TruncationGraphDataset)
applies the inner-hit truncation augmentation to the items of a "trainset" directory also at inference (checked here,
`trainset_truncation_check`).

Builders, per seed at the committed val-chosen configurations (results/strong_baselines_v3_val_entries.csv; Canto-L's
equal results/final_selection_v1.json, checked): Canto-L (frozen: MWS-layer + learned merging, LM, tau per seed,
RUN/lmerge/model.joblib), Canto-L + MWS-layer + helix merging (MWS+M, s0 0.2), acorn (ttbar) + LM (tau per seed, the
GBDT of runs/feedback_v3/strong_baselines_a2/armi_s<S>/lmerge) and acorn (ttbar) + MWS+M (s0 0.1). Every particle is
put in the first class that applies, with hnl.eval.round4_checks.classify_event: found / graph / classifier (true
edges in the graph, but CC on the chain-true edges above the builder's edge cut does not match it) / builder
(fragmented, impure, short).

Identity check: the 49 val events of round 4 check 2 go through the same staging, graphs and scoring; their scored
graphs must equal the stored val graphs (node arrays, edge list, truth, scores of RUN/gnn_scores/valset.tar), the
rebuilt DM match of every particle of these events must equal the stored val grid at each configuration, the classes
of the 51 val far daughters must equal results/round4_checks_far_particles.csv (Canto-L LM and MWS+M) and the found
counts at |d0| >= 300 mm the entries' k_d0_300 (all four).

Hypotheses: (a) Canto-L's learned-merging GBDT (ttbar fragment pairs) does not merge the fragments of late-starting
tracks: the particles found by one of LM / MWS+M and lost by the other, with the merge decisions (the GBDT
probability of every pair of the particle's first-pass fragments against tau, the helix fit of their union against
merge_fragments' cuts); (b) the classifier under-scores tracks that start in the outer strips: classifier losses and
chain-edge scores by first-hit radius, against the start radii that the truncation augmentation (hnl.models.augment:
p 0.3, m uniform in 1..n-5) gives ttbar training tracks (`truncation_starts`); (c) graph losses common to every model.

    python -m hnl.eval.far_trainsplit population [--out DIR]
    python -m hnl.eval.far_trainsplit params --seed S          (shell assignments for the graphs job)
    python -m hnl.eval.far_trainsplit stage [--out DIR] --dest LOCAL_DIR
    python -m hnl.eval.far_trainsplit analyze [--out DIR] --results results --name far_trainsplit_v1 [--workers N]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import json
import os
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap_tables as BT
from hnl.eval import round4_checks as R4

REPO = BT.REPO
DATA = BT.DATA
OUT = os.path.join(DATA, "runs", "feedback_v4", "far_trainsplit")
BASE = os.path.join(DATA, "runs", "baseline_v1")
CANTO_DIR = os.path.join(DATA, "runs", "stage1_final_learned", "l1tl")
FS_DIR = os.path.join(DATA, "derived", "acorn_fs", "v1")
SEEDS = R4.SEEDS
TRAIN_SETS = {"hnl_joint_train": (("hnl10_100", "train"), ("hnl15_200", "train"))}
FAR = R4.FAR                                     # (300, 200) mm
R_EDGES = (0.0, 200.0, 500.0, 700.0, np.inf)     # first-hit radius bins [mm]
R_LABELS = ("0-200", "200-500", "500-700", ">=700")
LTF = {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5}   # Canto-L's union (DECISIONS 2026-10-04)
STAGES = R4.STAGES
A_LM = "acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder)"
A_MWSM = "acorn (ttbar) + MWS-layer + merging (Canto builder)"
MODELS = (R4.CANTO, R4.C_MWSM, A_LM, A_MWSM)
SHORT = {R4.CANTO: "Canto-L LM (frozen)", R4.C_MWSM: "Canto-L MWS+M", A_LM: "acorn (ttbar) LM",
         A_MWSM: "acorn (ttbar) MWS+M"}
PIPE = {R4.CANTO: "canto_l", R4.C_MWSM: "canto_l", A_LM: "acorn_ttbar", A_MWSM: "acorn_ttbar"}
BUILDER = {R4.CANTO: "LM", R4.C_MWSM: "MWS+M", A_LM: "LM", A_MWSM: "MWS+M"}
PIPELINES = ("canto_l", "acorn_ttbar")
PIPE_NAME = {"canto_l": "Canto-L", "acorn_ttbar": "acorn (ttbar)"}
ROUND4_PARTICLES = "results/round4_checks_far_particles.csv"
TRUNC_FINE = np.arange(0.0, 1101.0, 100.0)       # 100 mm bins for the start-radius table (last bin open)
SCORE_TOL = 1e-4                                 # rescored vs stored val scores (GPU scoring is not bitwise repeatable)


def _repo(p: str) -> str:
    return R4._repo(p)


def _sha256(path: str) -> str:
    return R4._sha256(path)


def _one(pattern: str) -> str:
    f = sorted(glob.glob(pattern))
    if len(f) != 1:
        raise SystemExit(f"{pattern}: {len(f)} files, expected 1")
    return f[0]


def r_bin(r) -> np.ndarray:
    """First-hit radius bin label (R_LABELS) per value."""
    r = np.asarray(r, dtype=float)
    k = np.clip(np.searchsorted(R_EDGES, r, side="right") - 1, 0, len(R_LABELS) - 1)
    return np.array(R_LABELS, dtype=object)[k]


def runs_of(pipe: str, seed: str) -> str:
    return os.path.join(CANTO_DIR, f"armi_s{seed}") if pipe == "canto_l" else os.path.join(BASE, f"armi_s{seed}")


# --------------------------------------------------------------------------------------------------------------------
# population


def train_population() -> tuple[pd.DataFrame, dict]:
    """Primary-population HNL daughters with |d0| >= 200 mm in the HNL train events (flag ge300), and counts."""
    from hnl.eval.particles import particle_table
    ev = BT.expected_events(TRAIN_SETS)
    val = BT.expected_events(R4.HNL_SETS)
    if np.intersect1d(ev, val).size:
        raise SystemExit("train and val event ids overlap")
    p = particle_table(ev)
    t = p[p.is_target.astype(bool)].reset_index(drop=True)
    prim = BT.masks(t[["event_id", "particle_id", "is_target", "category", "d0", "rv"]])["eff_primary"]
    f = t[prim & (t.d0.abs().to_numpy() >= min(FAR))].copy()
    f["ge300"] = f.d0.abs() >= 300.0
    f["r_bin"] = r_bin(f.r_first)
    cols = ["event_id", "particle_id", "sample", "d0", "rv", "n_hits", "n_pixel_hits", "r_first", "pt", "eta", "ge300",
            "r_bin"]
    f = f[cols].sort_values(["event_id", "particle_id"]).reset_index(drop=True)
    counts = {"train_events": int(len(ev)), "train_events_with_particles": int(p.event_id.nunique()),
              "primary_daughters": int(prim.sum()), "far_ge200": int(len(f)), "far_ge300": int(f.ge300.sum()),
              "events_ge200": int(f.event_id.nunique()), "events_ge300": int(f[f.ge300].event_id.nunique())}
    return f, counts


def val_population() -> pd.DataFrame:
    """The 51 val daughters of round 4 check 2 (|d0| >= 200 mm), checked against its committed particle table."""
    f = R4.far_population()
    f["r_bin"] = r_bin(f.r_first)
    ref = pd.read_csv(_repo(ROUND4_PARTICLES))[["event_id", "particle_id"]].drop_duplicates()
    a = set(map(tuple, f[["event_id", "particle_id"]].to_numpy().tolist()))
    b = set(map(tuple, ref.to_numpy().tolist()))
    if a != b:
        raise SystemExit(f"val population differs from {ROUND4_PARTICLES}: {len(a)} vs {len(b)}")
    return f


def run_population(out: str) -> dict:
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    f, counts = train_population()
    v = val_population()
    f.to_parquet(os.path.join(out, "population_train.parquet"), index=False)
    v.to_parquet(os.path.join(out, "population_val.parquet"), index=False)
    by = f.groupby(["sample", "ge300", "r_bin"]).size().rename("n").reset_index()
    counts.update({"val_far_ge200": int(len(v)), "val_far_ge300": int(v.ge300.sum()),
                   "val_events": int(v.event_id.nunique()), "elapsed_s": round(time.time() - t0, 1)})
    events = {"train": sorted(int(e) for e in f.event_id.unique()), "val": sorted(int(e) for e in v.event_id.unique())}
    json.dump({**counts, "events": events}, open(os.path.join(out, "population.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    print(by.to_string(index=False))
    print(json.dumps(counts))
    return counts


def population_events(out: str) -> dict:
    d = json.load(open(os.path.join(out, "population.json")))
    return {k: [int(e) for e in v] for k, v in d["events"].items()}


# --------------------------------------------------------------------------------------------------------------------
# graphs job: parameters and staging


def graph_params(seed: str) -> dict:
    """The val-graph settings of seed `seed`, read from the runs' own records and checked against each other: the
    metric-learning checkpoint and r_infer / knn_infer of RUN/ml_graphs, the learned-triplet union settings of
    RUN/ltf_union_graphs/report_val/build.json, the GNN checkpoints of both runs' val scores (sha256)."""
    from hnl.graphs.helix_union import best_ckpt
    ml_run = os.path.join(BASE, f"armi_s{seed}")
    canto_run = os.path.join(CANTO_DIR, f"armi_s{seed}")
    ml_ckpt = best_ckpt(ml_run, "ml")
    inf = json.load(open(_one(os.path.join(ml_run, "ml_graphs", "infer_*.json"))))
    if inf["ckpt"] != ml_ckpt or inf["ckpt_sha256"] != _sha256(ml_ckpt):
        raise SystemExit(f"{ml_run}: ml_graphs were not made with {ml_ckpt}")
    ov = inf["overrides"]
    if ov.get("hard_cuts") or not ov.get("undirected"):
        raise SystemExit(f"{ml_run}: unexpected ml inference overrides {ov}")
    ub = json.load(open(os.path.join(ml_run, "ltf_union_graphs", "report_val", "build.json")))
    tm = os.path.join(ml_run, "learned_triplet_filter_v2", "model.joblib")
    want = {"triplet_model": tm, "ml_ckpt": ml_ckpt, "source": os.path.join(ml_run, "ml_graphs", "valset.tar"),
            "metrics": True, **LTF}
    bad = {k: (ub.get(k), v) for k, v in want.items() if ub.get(k) != v}
    if bad:
        raise SystemExit(f"{ml_run}: the stored val union graphs differ from the settings used here: {bad}")
    out = {"ML_RUN": ml_run, "ML_CKPT": ml_ckpt, "R_INFER": ov["r_infer"], "KNN_INFER": ov["knn_infer"],
           "TRIPLET_MODEL": tm, "TAU": LTF["tau"], "KTOP": LTF["k"], "R_WIDE": LTF["r_wide"],
           "NSIG": LTF["nsig"], "PT_MIN": LTF["pt_min"]}
    for key, run in (("ACORN_GNN_CKPT", ml_run), ("CANTO_GNN_CKPT", canto_run)):
        ck = best_ckpt(run, "gnn")
        gi = json.load(open(_one(os.path.join(run, "gnn_scores", "infer_*.json"))))
        if gi["ckpt"] != ck or gi["ckpt_sha256"] != _sha256(ck) or gi["overrides"]["data_split"][0] != 0:
            raise SystemExit(f"{run}: val scores not made with {ck}")
        out[key] = ck
    return out


def stage(out: str, dest: str) -> dict:
    """Extract the population's train events and the identity val events from the feature store into DEST/valset
    (one tar per sample and split, only <sample>_train.tar and <sample>_val.tar are opened)."""
    from hnl.train.acorn_runner import stage_events
    from hnl.data.acorn_store import parse_event_id
    ev = population_events(out)
    d = os.path.join(dest, "valset")
    n = {}
    for split, ids in (("train", ev["train"]), ("val", ev["val"])):
        by: dict = {}
        for e in ids:
            s, i = parse_event_id(str(int(e)))
            by.setdefault(s, []).append((s, int(i)))
        for s, lst in sorted(by.items()):
            tar = os.path.join(FS_DIR, f"{s}_{split}.tar")
            n[f"{s}_{split}"] = stage_events(lst, "tar", d, tar_name=tar)
    print(json.dumps(n))
    return n


# --------------------------------------------------------------------------------------------------------------------
# per-event read-out


def compare_graphs(mine: str, ref: str) -> dict:
    """A rescored graph against the stored one: node arrays, edge list, truth, scores."""
    import torch
    from acorn.utils.loading_utils import load_pyg
    g, h = load_pyg(mine[:-3], weights_only=False), load_pyg(ref[:-3], weights_only=False)
    out = {}
    for k in ("hit_id", "hit_particle_id", "r", "phi", "z", "track_edges", "edge_index", "y", "truth_map"):
        if k in g or k in h:
            out[f"eq_{k}"] = bool(k in g and k in h and torch.equal(g[k], h[k]))
    s1 = (g.scores if "scores" in g else g.edge_scores).double().numpy()
    s2 = (h.scores if "scores" in h else h.edge_scores).double().numpy()
    out["n_edges"] = int(len(s1))
    if out.get("eq_edge_index") and len(s1) == len(s2):
        d = np.abs(s1 - s2)
        out.update({"scores_equal": bool((d == 0).all()), "n_score_diff": int((d > 0).sum()),
                    "max_abs_score_diff": float(d.max()) if len(d) else 0.0,
                    "n_cross_0.1": int(((s1 > 0.1) != (s2 > 0.1)).sum()),
                    "n_cross_0.2": int(((s1 > 0.2) != (s2 > 0.2)).sum())})
    else:
        out.update({"scores_equal": False, "n_score_diff": -1, "max_abs_score_diff": np.nan, "n_cross_0.1": -1,
                    "n_cross_0.2": -1})
    return out


def chain_scores(path: str, far_ids: np.ndarray) -> pd.DataFrame:
    """Per far particle: its chain (consecutive true) edges in R order of the inner hit, whether each is in the graph
    and its score (the highest of both directions, as the builders use), the inner hit's transverse radius."""
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval.trackbuild_val import load_scored
    from hnl.graphs import ceiling as C
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    n = len(pid)
    g = load_pyg(path[:-3], weights_only=False)
    te = C.unordered_unique(g.track_edges.numpy())
    code = a.astype(np.int64) * n + b
    order = np.argsort(code, kind="stable")
    tcode = te[0].astype(np.int64) * n + te[1]
    pos = np.minimum(np.searchsorted(code[order], tcode), max(len(code) - 1, 0))
    in_g = (code[order][pos] == tcode) if len(code) else np.zeros(len(tcode), bool)
    t_sc = np.where(in_g, sc[order][pos] if len(code) else 0.0, np.nan)
    inner = np.where(R[te[0]] <= R[te[1]], te[0], te[1])
    rows = []
    for p in far_ids:
        m = np.flatnonzero(pid[te[0]] == p)
        m = m[np.argsort(R[inner[m]], kind="stable")]
        s = t_sc[m]
        hits = np.flatnonzero(pid == p)
        h0 = hits[np.argmin(R[hits])]
        rows.append({"event_id": np.int64(eid), "particle_id": np.int64(p), "n_chain": int(len(m)),
                     "n_chain_in_graph": int(in_g[m].sum()), "chain_r_inner": " ".join(f"{x:.0f}" for x in r[inner[m]]),
                     "chain_scores": " ".join("nan" if not np.isfinite(x) else f"{x:.4f}" for x in s),
                     "n_chain_gt_0.1": int((np.nan_to_num(s, nan=-1) > 0.1).sum()),
                     "n_chain_gt_0.2": int((np.nan_to_num(s, nan=-1) > 0.2).sum()),
                     "n_chain_gt_0.5": int((np.nan_to_num(s, nan=-1) > 0.5).sum()),
                     "first_edge_score": float(s[0]) if len(s) else np.nan,
                     "min_chain_score_rank": int(np.nanargmin(s)) if np.isfinite(s).any() else -1,
                     "r_first_graph": float(r[h0]), "n_hits_graph": int(len(hits))})
    return pd.DataFrame(rows)


def _fragments(first: np.ndarray, hits: np.ndarray) -> list[tuple[int, int, int]]:
    """(track label, the particle's hits on it, track size) of the first-pass tracks holding >= 2 of its hits,
    largest share first."""
    lab = first[hits]
    tu, tc = np.unique(lab[lab >= 0], return_counts=True)
    out = [(int(t), int(c), int((first == t).sum())) for t, c in zip(tu, tc) if c >= 2]
    return sorted(out, key=lambda x: (-x[1], x[0]))


def merge_diagnostics(path: str, geo: dict, merger, tau: float, s0_mwsm: float, far_ids: np.ndarray,
                      dm_lm: dict, dm_mwsm: dict):
    """Merge decisions for the far particles of one event, the builders recomputed as in hnl.eval.round4_checks
    (learned merging: same-layer MWS at s0 0.2, GBDT p of the candidate fragment pairs, LM.apply at tau; MWS+M:
    same-layer MWS at s0_mwsm, merge_fragments "any" then "sequential", max_rms 1.0). Per particle and first pass, every
    pair of its first-pass fragments (>= 2 of its hits each): the GBDT candidate flag, p and features (LM first pass),
    the helix fit of the union (merge_fragments' fit: rms, zrms, outliers, R span and the pass flag), the two candidate
    windows of merge_fragments and whether the builders put both fragments on one final track. dm_*: the particle's
    DM match by the builders (checked against the recomputation)."""
    from hnl.eval import matching as M
    from hnl.eval.trackbuild_val import load_scored
    from hnl.trackbuild import learned_merge as LM
    from hnl.trackbuild.merge_fragments import _candidates_any, _volume_from_radius, merge_fragments
    from hnl.trackbuild.mutex_ws import mws_layer
    from hnl.vertex.pairing import fit_tracks
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    n = len(pid)
    L = geo["layer_id"]
    first_lm = LM.first_pass(n, a, b, sc, L)
    pairs = LM.candidates(first_lm, r, phi, z)
    X = LM.features(first_lm, pairs, r, phi, z, geo)
    p = merger.model.predict_proba(X)[:, 1] if len(pairs) else np.zeros(0)
    final_lm = LM.apply(first_lm, pairs, p, tau, L)
    first_mw = mws_layer(n, a, b, sc, L, s0_mwsm)
    any_mw = merge_fragments(r, phi, z, first_mw, max_rms=1.0, mode="any")
    final_mw = merge_fragments(r, phi, z, any_mw, max_rms=1.0)
    fm = M.FastMatch(pid)
    dm = {"lm_first": fm.match(first_lm)["dm_matched"], "lm": fm.match(final_lm)["dm_matched"],
          "mw_first": fm.match(first_mw)["dm_matched"], "mw_any": fm.match(any_mw)["dm_matched"],
          "mw": fm.match(final_mw)["dm_matched"]}
    vol = _volume_from_radius(r, z)
    key = {(int(x), int(y)): k for k, (x, y) in enumerate(pairs)}
    col = {f: k for k, f in enumerate(LM.FEATURES)}
    same_first = bool(np.array_equal(first_lm, first_mw))
    passes = [("lm", first_lm)] + ([] if same_first else [("mwsm", first_mw)])
    part_rows, pair_rows = [], []
    for P in far_ids:
        i = int(np.searchsorted(fm.particles, P))
        for name, d_ref in (("lm", dm_lm), ("mw", dm_mwsm)):
            if bool(dm[name][i]) != bool(d_ref[int(P)]):
                raise ValueError(f"{path} {P}: recomputed {name} DM differs from classify_event")
        hits = np.flatnonzero(pid == P)
        prow = {"event_id": np.int64(eid), "particle_id": np.int64(P), "first_passes_equal": same_first,
                **{f"dm_{k}": bool(v[i]) for k, v in dm.items()}}
        for fp, first in passes:
            fr = _fragments(first, hits)
            prow[f"{fp}_n_frag"] = len(fr)
            prow[f"{fp}_frags"] = " ".join(f"{c}/{s}" for _, c, s in fr)
            prow[f"{fp}_hits_on_frags"] = int(sum(c for _, c, _ in fr))
            for x in range(len(fr)):
                for y in range(x + 1, len(fr)):
                    ta, tb = fr[x][0], fr[y][0]
                    ha, hb = np.flatnonzero(first == ta), np.flatnonzero(first == tb)
                    rows = np.r_[ha, hb]
                    f = fit_tracks(r[rows], phi[rows], z[rows], vol[rows], np.zeros(len(rows), dtype=np.int64),
                                   min_hits=3)
                    span = float(R[rows].max() - R[rows].min())
                    if len(f["track"]):
                        rms, zrms, nout = float(f["rms"][0]), float(f["zrms"][0]), int(f["n_out"][0])
                    else:
                        rms, zrms, nout = np.nan, np.nan, -1
                    # merge_fragments' windows: "any" (local directions at the median radius), "sequential" (one
                    # fragment starts outward of where the other ends, small phi gap)
                    win_any = len(_candidates_any([ta, tb], first, r, phi, z, R, 0.05, 0.05)) > 0
                    seq = False
                    for u, v in ((ha, hb), (hb, ha)):
                        o, i_ = u[np.argmax(R[u])], v[np.argmin(R[v])]
                        dR = R[i_] - R[o]
                        dphi = abs((phi[i_] - phi[o] + np.pi) % (2 * np.pi) - np.pi)
                        seq |= bool(-20.0 <= dR <= 400.0 and dphi <= 0.3)
                    row = {"event_id": np.int64(eid), "particle_id": np.int64(P), "first_pass": fp,
                           "s0": 0.2 if fp == "lm" else s0_mwsm, "frag_a": ta, "frag_b": tb,
                           "a_hits_particle": fr[x][1], "a_size": fr[x][2], "b_hits_particle": fr[y][1],
                           "b_size": fr[y][2], "a_r_min": float(r[ha].min()), "b_r_min": float(r[hb].min()),
                           "fit_rms": rms, "fit_zrms": zrms, "fit_n_out": nout, "fit_span_R": span,
                           "fit_pass": bool(np.isfinite(rms) and rms <= 1.0 and zrms <= 5.0 and nout <= 1
                                            and span >= 50.0),
                           "window_any": bool(win_any), "window_sequential": seq}
                    k = key.get((min(ta, tb), max(ta, tb)))
                    if fp == "lm":
                        row.update({"lm_candidate": k is not None, "lm_p": float(p[k]) if k is not None else np.nan,
                                    "lm_tau": tau, "lm_p_gt_tau": bool(k is not None and p[k] > tau),
                                    "lm_merged": bool(final_lm[ha[0]] == final_lm[hb[0]])})
                        for fn in ("u_rms_n", "u_zrms_n", "u_out_frac", "gap_s", "shared_layers", "span_s", "d_min",
                                   "dkappa", "dcot"):
                            row[f"lm_{fn}"] = float(X[k, col[fn]]) if k is not None else np.nan
                    if fp == "mwsm" or same_first:
                        row.update({"mwsm_merged_any": bool(any_mw[ha[0]] == any_mw[hb[0]]),
                                    "mwsm_merged": bool(final_mw[ha[0]] == final_mw[hb[0]])})
                    pair_rows.append(row)
        part_rows.append(prow)
    return pd.DataFrame(part_rows), pd.DataFrame(pair_rows)


def _event_task(task: dict) -> dict:
    """One pipeline, seed and chunk of events: classes (round 4's classify_event), chain-edge scores, merge
    diagnostics, the identity comparison of the val events."""
    from hnl.trackbuild.learned_merge import Merger
    mergers = {sp["merge_model"]: Merger(sp["merge_model"]) for sp in task["specs"] if sp["merge_model"]}
    lm = next(sp for sp in task["specs"] if sp["family"] == "mwslmerge")
    mw = next(sp for sp in task["specs"] if sp["family"] == "mwslayermerge")
    out = {k: [] for k in ("far", "dm", "ident", "chain", "mpart", "mpair")}
    for e in task["events"]:
        path = os.path.join(task["dir"], f"event{e}.pyg.gz")
        far_ids = task["far"][e]
        f, m = R4.classify_event(path, task["specs"], task["geo"][e], mergers, far_ids)
        f["split"] = task["split"][e]
        out["far"].append(f)
        if task["split"][e] == "val":
            out["dm"].append(m)
            c = compare_graphs(path, os.path.join(task["ref_dir"], f"event{e}.pyg.gz"))
            out["ident"].append({"pipeline": task["pipe"], "seed": int(task["seed"]), "event_id": int(e), **c})
        c = chain_scores(path, far_ids)
        c["split"] = task["split"][e]
        out["chain"].append(c)
        dl = {int(r.particle_id): r.stage == "found" for r in f[f.key == lm["key"]].itertuples()}
        dw = {int(r.particle_id): r.stage == "found" for r in f[f.key == mw["key"]].itertuples()}
        mp, mq = merge_diagnostics(path, task["geo"][e], mergers[lm["merge_model"]], lm["param"], mw["param"], far_ids,
                                   dl, dw)
        mp["split"] = task["split"][e]
        if len(mq):
            mq["split"] = task["split"][e]
        out["mpart"].append(mp)
        out["mpair"].append(mq)
    res = {}
    for k, v in out.items():
        if k == "ident":
            res[k] = pd.DataFrame(v)
        else:
            res[k] = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        if k != "ident" and len(res[k]):
            res[k]["pipeline"], res[k]["seed"] = task["pipe"], int(task["seed"])
    return res


def _extract(tar_path: str, events, dest: str) -> int:
    """Extract event<id>.pyg.gz of the listed events from a tar (flat)."""
    want = {f"event{int(e)}.pyg.gz" for e in events}
    os.makedirs(dest, exist_ok=True)
    k = 0
    with tarfile.open(tar_path, "r") as tar:
        for mem in tar:
            nm = os.path.basename(mem.name)
            if nm in want:
                mem.name = nm
                tar.extract(mem, dest, filter="data")
                k += 1
    if k != len(want):
        raise RuntimeError(f"{tar_path}: extracted {k} of {len(want)} graphs")
    return k


# --------------------------------------------------------------------------------------------------------------------
# hypothesis (b): start radii of ttbar training tracks under the truncation augmentation


def start_weights(n: int, p: float, keep: int, min_hits: int) -> np.ndarray:
    """Probability that a particle with n hits starts at its j-th innermost hit (in R) after the augmentation of
    hnl.models.augment.truncation_mask: untruncated (j = 0) with 1 - p, else m uniform in {1, ..., n - keep};
    particles with fewer than min_hits hits are never truncated."""
    w = np.zeros(n)
    if n < min_hits or p <= 0:
        w[0] = 1.0
        return w
    w[0] = 1.0 - p
    w[1:n - keep + 1] = p / (n - keep)
    return w


def start_weights_vec(n: np.ndarray, j: np.ndarray, p: float, keep: int, min_hits: int) -> np.ndarray:
    """start_weights for many (n, j) at once: the probability that a particle with n hits starts at its j-th
    innermost hit after the augmentation."""
    n, j = np.asarray(n, dtype=np.int64), np.asarray(j, dtype=np.int64)
    trunc = (n >= min_hits) & (p > 0)
    w = np.where(j == 0, np.where(trunc, 1.0 - p, 1.0), 0.0)
    lateral = trunc & (j >= 1) & (j <= n - keep)
    return np.where(lateral, p / np.maximum(n - keep, 1), w)


FINE_LABELS = tuple(f"{int(a)}-{int(b)}" for a, b in zip(TRUNC_FINE[:-1], TRUNC_FINE[1:])) + (f">={int(TRUNC_FINE[-1])}",)
COARSE_OF_FINE = {lab: R_LABELS[int(np.clip(np.searchsorted(R_EDGES, lo, side="right") - 1, 0, len(R_LABELS) - 1))]
                  for lab, lo in zip(FINE_LABELS, TRUNC_FINE)}


def _trunc_chunk(events):
    """Start radii of the ttbar train particles (pid != 0, >= 5 hits) of a list of events, aggregated per population
    (all / targets), kind (natural: innermost hit in R; truncated: the starts of truncated tracks; augmented: every
    track of one augmented epoch) and 100 mm bin of the start hit's transverse radius: sum of weights (expected tracks
    per epoch) and of weight x hits left. Hits ordered as hnl.models.augment.truncation_mask (R, ties by row)."""
    from hnl.data.convert import read_table
    from hnl.eval.particles import DERIVED
    from hnl.models.augment import TRUNC_KEEP, TRUNC_MIN_HITS, TRUNC_P
    h = read_table(os.path.join(DERIVED, "events", "v1"), "hits", "ttbar", events=events,
                   columns=["event", "row", "r", "z", "pid"])
    pt = read_table(os.path.join(DERIVED, "events", "v1"), "particles", "ttbar", events=events,
                    columns=["event", "particle_id", "is_target"])
    h = h[h.pid != 0]
    ev, pid, row = (h[c].to_numpy(np.int64) for c in ("event", "pid", "row"))
    r = h.r.to_numpy().astype(np.float64)
    zz = h.z.to_numpy().astype(np.float64)
    R = np.sqrt(r ** 2 + zz ** 2)
    o = np.lexsort((row, R, pid, ev))
    ev, pid, r = ev[o], pid[o], r[o]
    new = np.r_[True, (ev[1:] != ev[:-1]) | (pid[1:] != pid[:-1])]
    gid = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    n = np.bincount(gid)[gid]
    j = np.arange(len(gid)) - first[gid]
    tset = pt[pt.is_target.astype(bool)]
    tkey = set((tset.event.to_numpy(np.int64) * (1 << 32) + tset.particle_id.to_numpy(np.int64)).tolist())
    is_t = np.fromiter(((e * (1 << 32) + q) in tkey for e, q in zip(ev[first], pid[first])), bool, len(first))[gid]
    w_aug = start_weights_vec(n, j, TRUNC_P, TRUNC_KEEP, TRUNC_MIN_HITS)
    keep = n >= 5
    fine = np.clip(np.searchsorted(TRUNC_FINE, r, side="right") - 1, 0, len(FINE_LABELS) - 1)
    rows = []
    for pop_name, pm in (("all", keep), ("targets", keep & is_t)):
        for kind, w in (("natural", np.where(j == 0, 1.0, 0.0)), ("truncated", np.where(j > 0, w_aug, 0.0)),
                        ("augmented", w_aug)):
            m = pm & (w > 0)
            ws = np.bincount(fine[m], weights=w[m], minlength=len(FINE_LABELS))
            wh = np.bincount(fine[m], weights=(w * (n - j))[m], minlength=len(FINE_LABELS))
            for i, lab in enumerate(FINE_LABELS):
                rows.append({"population": pop_name, "kind": kind, "r_bin_100mm": lab, "w": ws[i], "w_hits_left": wh[i]})
    return pd.DataFrame(rows), {"all": int((keep[first]).sum()), "targets": int((keep & is_t)[first].sum()),
                                "events": int(len(np.unique(ev)))}


def _load_bytes(data: bytes):
    import gzip
    import io
    import torch
    return torch.load(io.BytesIO(gzip.decompress(data)), weights_only=False)


def fs_node_check(n_events: int) -> dict:
    """The first n_events graphs of the feature store's ttbar_train.tar against the hits table: same particle id,
    r and z per node (so that the augmentation's start radii can be read from the table)."""
    from hnl.data.convert import read_table
    from hnl.eval.particles import DERIVED
    tar = os.path.join(FS_DIR, "ttbar_train.tar")
    got = {}
    with tarfile.open(tar) as t:
        for mem in t:
            if not mem.name.endswith(".pyg.gz"):
                continue
            g = _load_bytes(t.extractfile(mem).read())
            eid = int(os.path.basename(mem.name)[5:].split("-")[0].split(".")[0])
            got[eid] = g
            if len(got) >= n_events:
                break
    ok = 0
    for eid, g in got.items():
        e = eid % 100_000_000
        h = read_table(os.path.join(DERIVED, "events", "v1"), "hits", "ttbar", events=[e],
                       columns=["event", "row", "r", "z", "pid"]).sort_values("row")
        gp = (g.hit_particle_id if "hit_particle_id" in g else g.particle_id).numpy()
        gr = (g.r if "r" in g else g.hit_r).double().numpy()
        gz = (g.z if "z" in g else g.hit_z).double().numpy()
        same = (len(gp) == len(h) and np.array_equal(gp, h.pid.to_numpy())
                and np.array_equal(gr, h.r.to_numpy().astype(np.float64))
                and np.array_equal(gz, h.z.to_numpy().astype(np.float64)))
        ok += int(same)
    return {"events": len(got), "identical": ok}


def run_truncation(workers: int) -> tuple[pd.DataFrame, dict]:
    """Expected start radii of the ttbar TRAIN tracks (particles with >= 5 hits) in one augmented epoch, read from the
    hits table (graph node order = row order of the table; fs_node_check confirms it on the feature-store graphs)."""
    from hnl.data.splits import read_split
    ev = read_split("ttbar", "train").astype(np.int64)
    chunks = [ev[k:k + 250].tolist() for k in range(0, len(ev), 250)]
    with cf.ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        parts = list(ex.map(_trunc_chunk, chunks))
    t = pd.concat([x[0] for x in parts], ignore_index=True).groupby(
        ["population", "kind", "r_bin_100mm"], sort=False, as_index=False)[["w", "w_hits_left"]].sum()
    meta = {"ttbar_train_events": int(len(ev)), "events_with_tracks": int(sum(x[1]["events"] for x in parts)),
            "tracks_ge5_hits": int(sum(x[1]["all"] for x in parts)),
            "target_tracks_ge5_hits": int(sum(x[1]["targets"] for x in parts)),
            "fs_node_check": fs_node_check(20)}
    return t, meta


POP_LABEL = {"all": "ttbar train tracks (pid != 0, >= 5 hits)",
             "targets": "ttbar train targets (pT > 1 GeV, |eta| < 4, >= 5 hits)"}


def truncation_table(t: pd.DataFrame, pops: dict) -> pd.DataFrame:
    """Per population of ttbar training tracks and start kind (natural: innermost hit, no augmentation; truncated:
    the starts of truncated tracks only; augmented: every track of one augmented epoch), the expected number of track
    starts per epoch and their fraction per first-hit radius bin (R_LABELS and 100 mm bins), with the mean hits left;
    and the first-hit radii of the far daughters (pops: name -> frame with r_first, n_hits)."""
    rows = []
    t = t.copy()
    t["r_bin"] = t.r_bin_100mm.map(COARSE_OF_FINE)
    for (pop_name, kind), g in t.groupby(["population", "kind"], sort=False):
        tot = g.w.sum()
        for scheme, col, labels in (("coarse", "r_bin", R_LABELS), ("100mm", "r_bin_100mm", FINE_LABELS)):
            a = g.groupby(col)[["w", "w_hits_left"]].sum().reindex(list(labels), fill_value=0.0)
            for lab in labels:
                w, wh = float(a.loc[lab, "w"]), float(a.loc[lab, "w_hits_left"])
                rows.append({"population": POP_LABEL[pop_name], "kind": kind, "binning": scheme, "r_bin": lab,
                             "expected_per_epoch": w, "fraction": w / tot if tot else np.nan,
                             "mean_hits": wh / w if w else np.nan, "total_per_epoch": float(tot)})
    for name, f in pops.items():
        for scheme, edges, labels in (("coarse", R_EDGES, R_LABELS), ("100mm", (*TRUNC_FINE, np.inf), FINE_LABELS)):
            k = np.clip(np.searchsorted(edges, f.r_first.to_numpy(), side="right") - 1, 0, len(labels) - 1)
            for i, lab in enumerate(labels):
                sel = k == i
                rows.append({"population": name, "kind": "first hit (particle table)", "binning": scheme, "r_bin": lab,
                             "expected_per_epoch": float(sel.sum()), "fraction": float(sel.mean()) if len(f) else np.nan,
                             "mean_hits": float(f.n_hits.to_numpy()[sel].mean()) if sel.any() else np.nan,
                             "total_per_epoch": float(len(f))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------------
# analysis


def specs_by_pipeline() -> tuple[dict, list]:
    """Per (pipeline, seed): the LM and MWS+M specs of the committed val entries (round 4's spec format)."""
    checks: list[tuple] = []
    rows = {lab: R4.entries(lab) for lab in MODELS}
    checks += R4.check_canto_selection(rows[R4.CANTO])
    specs: dict = {}
    for lab in MODELS:
        for s, (run, wp) in zip(SEEDS, rows[lab]):
            if os.path.normpath(run) != os.path.normpath(runs_of(PIPE[lab], s)):
                raise SystemExit(f"{lab} seed {s}: entry run {run}, expected {runs_of(PIPE[lab], s)}")
            fam, params, grid = R4.parse_wp(wp)
            mm = R4.merge_model(run, fam, grid)
            sp = {"key": f"{lab}|{s}", "label": lab, "seed": int(s), "run": run, "wp": wp, "family": fam,
                  "param": float(params), "grid": grid, "merge_model": mm, "cut": R4.edge_cut(fam, float(params)),
                  "merge_model_sha256": _sha256(mm) if mm else ""}
            specs.setdefault((PIPE[lab], s), []).append(sp)
    for k, v in specs.items():
        fams = sorted(sp["family"] for sp in v)
        if fams != ["mwslayermerge", "mwslmerge"]:
            raise SystemExit(f"{k}: builder families {fams}")
    return specs, checks


def trainset_truncation_check(tmp: str) -> dict:
    """Canto-L's model class at inference (hnl.train.acorn_runner.infer's setup, CPU, no prediction): one stored ttbar
    TRAINING graph of its union graphs staged both as trainset and as valset; node counts of the dataset items."""
    import torch
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config
    from hnl.graphs.helix_union import best_ckpt
    from hnl.train.acorn_runner import checkpoint_class, count_split_files
    ck = best_ckpt(runs_of("canto_l", "0"), "gnn")
    src = os.path.join(BASE, "armi_s0", "ltf_union_graphs", "trainset.tar")
    d = tempfile.mkdtemp(prefix="trunc_", dir=tmp)
    with tarfile.open(src) as t:
        for mem in t:
            if mem.name.endswith(".pyg.gz"):
                data, name = t.extractfile(mem).read(), os.path.basename(mem.name)
                break
    for sp in ("trainset", "valset"):
        os.makedirs(os.path.join(d, sp))
        with open(os.path.join(d, sp, name), "wb") as f:
            f.write(data)
    model = checkpoint_class(ck).load_from_checkpoint(ck, map_location=torch.device("cpu"))
    over = {"input_dir": d, "stage_dir": os.path.join(d, "out"), "data_split": count_split_files(d), "hard_cuts": {},
            "undirected": True, "skip_existing": False, "accelerator": "cpu", "devices": 1, "nodes": 1,
            "num_workers": [0, 0, 0], "undirected_message_passing": True}
    model._hparams = add_variable_name_prefix_in_config({**model._hparams, **over})
    model.setup(stage="predict")
    nodes = lambda g: int(len(g["hit_particle_id"] if "hit_particle_id" in g else g["particle_id"]))
    raw = _load_bytes(data)
    a, b = model.trainset[0], model.valset[0]
    return {"checkpoint": ck, "graph": f"{src}:{name}", "model_class": type(model).__name__,
            "dataset_class": type(model.trainset).__name__, "nodes_stored": int(len(raw.hit_particle_id)),
            "nodes_trainset_item": nodes(a), "nodes_valset_item": nodes(b),
            "edges_stored": int(raw.edge_index.shape[1]), "edges_trainset_item": int(a["edge_index"].shape[1]),
            "edges_valset_item": int(b["edge_index"].shape[1])}


def _count_rows(far: pd.DataFrame, split: str) -> list[dict]:
    rows = []
    for lo in FAR:
        for rb in ("all", *R_LABELS):
            sel = far[(far.d0.abs() >= lo) & ((far.r_bin == rb) if rb != "all" else True)]
            for lab in MODELS:
                for s in (*[int(x) for x in SEEDS], "pooled"):
                    g = sel[sel.label == lab] if s == "pooled" else sel[(sel.label == lab) & (sel.seed == s)]
                    c = g.stage.value_counts()
                    rows.append({"split": split, "d0_min_mm": lo, "r_first_bin": rb, "model": SHORT[lab],
                                 "pipeline": PIPE_NAME[PIPE[lab]], "builder": BUILDER[lab], "seed": s, "n": len(g),
                                 **{k: int(c.get(k, 0)) for k in STAGES}, "lost": int(len(g) - c.get("found", 0)),
                                 "working_points": "; ".join(dict.fromkeys(g.wp)) if len(g) else ""})
    return rows


def discordant(far: pd.DataFrame, mpart: pd.DataFrame) -> pd.DataFrame:
    """Per pipeline, seed and particle: found by one of LM / MWS+M and lost by the other, with both classes and the
    merge summary of the particle (fragments per first pass, DM after each step)."""
    out = []
    key = ["split", "seed", "event_id", "particle_id"]
    for pipe in PIPELINES:
        lm = [lab for lab in MODELS if PIPE[lab] == pipe and BUILDER[lab] == "LM"][0]
        mw = [lab for lab in MODELS if PIPE[lab] == pipe and BUILDER[lab] == "MWS+M"][0]
        a = far[far.label == lm].set_index(key)
        b = far[far.label == mw].set_index(key)
        j = a[["stage", "d0", "r_first", "r_bin", "n_hits", "ge300", "edge_cut"]].join(
            b[["stage", "edge_cut"]], rsuffix="_mwsm", how="inner")
        j = j.rename(columns={"stage": "stage_lm", "edge_cut": "edge_cut_lm"})
        j = j[(j.stage_lm == "found") != (j.stage_mwsm == "found")].reset_index()
        j["pipeline"] = PIPE_NAME[pipe]
        j["found_by"] = np.where(j.stage_lm == "found", "LM only", "MWS+M only")
        out.append(j)
    d = pd.concat(out, ignore_index=True)
    m = mpart.copy()
    m["pipeline"] = m.pipeline.map(PIPE_NAME)
    return d.merge(m, on=["pipeline", "split", "seed", "event_id", "particle_id"], how="left")


def input_files(specs: dict, seeds) -> dict:
    """The input files of this read-out (path -> role)."""
    from hnl.graphs.helix_union import best_ckpt
    f = {}
    for x in (R4.ENTRIES, R4.SELECTION, ROUND4_PARTICLES, "splits/hnl10_100_train.txt", "splits/hnl15_200_train.txt",
              "splits/hnl10_100_val.txt", "splits/hnl15_200_val.txt", "splits/ttbar_train.txt",
              "docs/dropped_events.tsv"):
        f[_repo(x)] = "committed table"
    for s_ in ("hnl10_100", "hnl15_200"):
        for sp in ("train", "val"):
            f[os.path.join(FS_DIR, f"{s_}_{sp}.tar")] = "feature store (staged events)"
    f[os.path.join(FS_DIR, "ttbar_train.tar")] = "feature store (first 20 graphs: node check)"
    f[os.path.join(DATA, "derived", "events", "v1", "meta.json")] = "derived event tables (hits, particles)"
    for s in seeds:
        ml_run = os.path.join(BASE, f"armi_s{s}")
        f[best_ckpt(ml_run, "ml")] = f"metric-learning checkpoint seed {s}"
        f[os.path.join(ml_run, "learned_triplet_filter_v2", "model.joblib")] = f"Canto-L triplet GBDT seed {s}"
        for pipe in PIPELINES:
            f[best_ckpt(runs_of(pipe, s), "gnn")] = f"{PIPE_NAME[pipe]} GNN checkpoint seed {s}"
            f[os.path.join(runs_of(pipe, s), "gnn_scores", "valset.tar")] = f"{PIPE_NAME[pipe]} stored val scores seed {s}"
    for v in specs.values():
        for sp in v:
            if sp["merge_model"]:
                f[sp["merge_model"]] = f"{SHORT[sp['label']]} merging GBDT seed {sp['seed']}"
    return f


def run_analyze(out: str, results: str, name: str, workers: int, tmp: str, seeds=SEEDS, truncation: bool = True) -> dict:
    from hnl.geometry.resolution import hit_geometry
    t0 = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    checks: list[tuple] = []
    pop_t = pd.read_parquet(os.path.join(out, "population_train.parquet"))
    pop_v = pd.read_parquet(os.path.join(out, "population_val.parquet"))
    pop_t["split"], pop_v["split"] = "train", "val"
    pop = pd.concat([pop_t, pop_v], ignore_index=True)
    events = population_events(out)
    split_of = {**{e: "train" for e in events["train"]}, **{e: "val" for e in events["val"]}}
    if set(events["train"]) != set(pop_t.event_id) or set(events["val"]) != set(pop_v.event_id):
        raise SystemExit("population.json and the population tables disagree")
    far_by_event = {int(e): np.sort(g.particle_id.to_numpy(np.int64)) for e, g in pop.groupby("event_id")}
    specs, c0 = specs_by_pipeline()
    specs = {k: v for k, v in specs.items() if k[1] in seeds}
    checks += c0
    geo = hit_geometry(sorted(split_of))
    work = tempfile.mkdtemp(prefix="fts_", dir=tmp)
    # extraction: my scored graphs (all events), the stored val graphs of the identity events
    jobs, prov = [], {}
    for pipe in PIPELINES:
        for s in seeds:
            d = os.path.join(out, f"s{s}")
            if not os.path.exists(os.path.join(d, "graphs.done")):
                raise SystemExit(f"{d}: graphs job not finished")
            mine = os.path.join(d, pipe, "valset.tar")
            ref = os.path.join(runs_of(pipe, s), "gnn_scores", "valset.tar")
            jobs.append((mine, sorted(split_of), os.path.join(work, f"{pipe}_s{s}", "mine")))
            jobs.append((ref, events["val"], os.path.join(work, f"{pipe}_s{s}", "ref")))
            inf = json.load(open(os.path.join(d, pipe, "infer.json")))
            vi = json.load(open(_one(os.path.join(runs_of(pipe, s), "gnn_scores", "infer_*.json"))))
            ok = inf["ckpt_sha256"] == vi["ckpt_sha256"] and inf["gpu"] == vi["gpu"]
            checks.append((f"{PIPE_NAME[pipe]} seed {s}: scoring checkpoint and GPU = the stored val scores'",
                           f"{os.path.basename(inf['ckpt'])} on {inf['gpu']}", f"{os.path.basename(vi['ckpt'])} on "
                           f"{vi['gpu']}", ok))
            mi = json.load(open(os.path.join(d, "ml_infer.json")))
            gp = graph_params(s)
            ok = (mi["ckpt"] == gp["ML_CKPT"] and mi["overrides"]["r_infer"] == gp["R_INFER"]
                  and mi["overrides"]["knn_infer"] == gp["KNN_INFER"])
            if pipe == "canto_l":
                checks.append((f"seed {s}: metric-learning checkpoint, r_infer, knn_infer = ml_graphs'",
                               f"{os.path.basename(mi['ckpt'])} {mi['overrides']['r_infer']} "
                               f"{mi['overrides']['knn_infer']}", f"{os.path.basename(gp['ML_CKPT'])} "
                               f"{gp['R_INFER']} {gp['KNN_INFER']}", ok))
                ub = json.load(open(os.path.join(d, "union_report", "build.json")))
                ok = all(ub.get(k) == v for k, v in LTF.items()) and ub["triplet_model"] == gp["TRIPLET_MODEL"]
                chk = ub.get("check", {})
                checks.append((f"seed {s}: union build settings = the stored val union graphs' (k, tau, r_w, GBDT)",
                               json.dumps({k: ub.get(k) for k in LTF}), json.dumps(LTF), ok))
                checks.append((f"seed {s}: union build base check (CPU radius graph = metric-learning graph, truth)",
                               json.dumps(chk), "only_stored 0, only_recomputed 0, truth_ok",
                               bool(chk) and chk.get("truth_ok") is True and chk.get("only_recomputed") == 0
                               and chk.get("only_stored") == 0))
            prov[f"{pipe}_s{s}"] = {"scored_tar": mine, "scored_tar_sha256": _sha256(mine), "stored_val_tar": ref,
                                    "infer": inf, "ml_infer_ckpt": mi["ckpt"]}
    with cf.ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        list(ex.map(_extract, *zip(*jobs)))
    tasks = []
    for (pipe, s), sps in sorted(specs.items()):
        evs = sorted(split_of)
        for k in range(0, len(evs), 8):
            ch = evs[k:k + 8]
            tasks.append({"pipe": pipe, "seed": s, "specs": sps, "events": ch,
                          "dir": os.path.join(work, f"{pipe}_s{s}", "mine"),
                          "ref_dir": os.path.join(work, f"{pipe}_s{s}", "ref"),
                          "geo": {e: geo[e] for e in ch}, "far": {e: far_by_event[e] for e in ch},
                          "split": {e: split_of[e] for e in ch}})
    with cf.ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(_event_task, tasks))
    cat = lambda k: pd.concat([r[k] for r in res if len(r[k])], ignore_index=True)
    far, dm, ident, chain, mpart, mpair = (cat(k) for k in ("far", "dm", "ident", "chain", "mpart", "mpair"))
    allspecs = [sp for v in specs.values() for sp in v]
    far = far.merge(pd.DataFrame([{k: sp[k] for k in ("key", "label", "wp")} for sp in allspecs]),
                    on="key", how="left").drop(columns="key")
    far = far.rename(columns={"n_hits": "n_hits_graph"}).merge(
        pop.rename(columns={"n_hits": "n_hits_table"}), on=["event_id", "particle_id", "split"], how="left")
    far["n_hits"] = far["n_hits_table"]
    far["model"] = far.label.map(SHORT)
    if len(far) != len(pop) * len(allspecs):
        raise SystemExit(f"{len(far)} classified rows, expected {len(pop)} x {len(allspecs)}")
    checks.append(("far particles: hits in the graph = n_hits of the particle table",
                   float((far.n_hits_graph == far.n_hits_table).sum()) if "n_hits_graph" in far else np.nan,
                   float(len(far)), bool("n_hits_graph" in far and (far.n_hits_graph == far.n_hits_table).all())))
    # ---- identity on the val events
    for pipe in PIPELINES:
        for s in seeds:
            g = ident[(ident.pipeline == pipe) & (ident.seed == int(s))]
            eqc = [c for c in g.columns if c.startswith("eq_")]
            ok_g = len(g) == len(events["val"]) and bool(g[eqc].all(axis=None))
            checks.append((f"{PIPE_NAME[pipe]} seed {s}: val graphs = stored (nodes, edges, truth), "
                           f"{len(events['val'])} events", float(g[eqc].all(axis=1).sum()), float(len(events["val"])),
                           ok_g))
            # GPU scoring is not bitwise repeatable (atomic scatter-adds; slurm/feedback/far_trainsplit_repeat.sbatch):
            # the scores are required within SCORE_TOL of the stored ones and on the same side of every builder cut
            checks.append((f"{PIPE_NAME[pipe]} seed {s}: val scores bitwise equal to the stored ones, events "
                           "(information)", float(g.scores_equal.sum()), float(len(events["val"])), None))
            checks.append((f"{PIPE_NAME[pipe]} seed {s}: largest |score - stored score| <= {SCORE_TOL:g}",
                           float(g.max_abs_score_diff.max()), SCORE_TOL, bool(g.max_abs_score_diff.max() <= SCORE_TOL)))
            checks.append((f"{PIPE_NAME[pipe]} seed {s}: edges whose score is on the other side of a builder cut (0.1, "
                           "0.2) than the stored score", float(g["n_cross_0.1"].sum() + g["n_cross_0.2"].sum()), 0.0,
                           bool(g["n_cross_0.1"].sum() + g["n_cross_0.2"].sum() == 0)))
    for sp in allspecs:
        pp, _, d = R4.stored(sp["run"], sp["wp"])
        st = pp[pp.event_id.isin(events["val"])][["event_id", "particle_id", "dm_matched"]]
        mine = dm[(dm.key == sp["key"])]
        j = st.merge(mine, on=["event_id", "particle_id"], how="left")
        ok = j.dm_rebuilt.notna().all() and bool((j.dm_rebuilt.astype(bool) == j.dm_matched.astype(bool)).all())
        checks.append((f"{SHORT[sp['label']]} seed {sp['seed']}: rebuilt DM of every particle of the "
                       f"{len(events['val'])} val events = stored grid ({os.path.basename(d)}, {len(j)} particles)",
                       float(j.dm_rebuilt.fillna(-1).astype(float).sum()), float(j.dm_matched.astype(float).sum()), ok))
        ent = pd.read_csv(_repo(R4.ENTRIES))
        e = ent[(ent.label == sp["label"]) & (ent.run == sp["run"]) & (ent.set == "val")].iloc[0]
        f = far[(far.split == "val") & (far.label == sp["label"]) & (far.seed == sp["seed"]) & far.ge300]
        checks.append((f"{SHORT[sp['label']]} seed {sp['seed']}: val |d0| >= 300 found = entry k_d0_300",
                       float((f.stage == "found").sum()), float(e.k_d0_300),
                       int((f.stage == "found").sum()) == int(e.k_d0_300) and len(f) == int(e.n_d0_300)))
    r4 = pd.read_csv(_repo(ROUND4_PARTICLES))
    r4 = r4[r4.seed.isin([int(x) for x in seeds])]
    for lab in (R4.CANTO, R4.C_MWSM):
        ref = r4[r4.label == lab].set_index(["seed", "event_id", "particle_id"])
        mine = far[(far.split == "val") & (far.label == lab)].set_index(["seed", "event_id", "particle_id"])
        j = ref[["stage", "n_chain_in_graph", "n_chain_above_cut", "min_chain_score"]].join(
            mine[["stage", "n_chain_in_graph", "n_chain_above_cut", "min_chain_score"]], rsuffix="_mine", how="outer")
        ok = (len(j) == len(mine) == len(ref) and (j.stage == j.stage_mine).all()
              and (j.n_chain_in_graph == j.n_chain_in_graph_mine).all()
              and (j.n_chain_above_cut == j.n_chain_above_cut_mine).all())
        checks.append((f"{SHORT[lab]}: val far-daughter classes and chain-edge counts = round 4 check 2 "
                       f"({ROUND4_PARTICLES}), {len(seeds)} seeds x {len(pop_v)} particles",
                       float((j.stage == j.stage_mine).sum()), float(len(ref)), bool(ok)))
        dmin = (j.min_chain_score - j.min_chain_score_mine).abs()
        checks.append((f"{SHORT[lab]}: val far daughters' lowest chain-edge score within {SCORE_TOL:g} of round 4's",
                       float(dmin.max()), SCORE_TOL, bool((dmin.fillna(0) <= SCORE_TOL).all())))
    # ---- side check: the truncation augmentation at inference
    tc = trainset_truncation_check(tmp)
    # ---- hypothesis (b): start radii under the truncation augmentation
    if truncation:
        tt, tmeta = run_truncation(workers)
        checks.append(("feature-store ttbar train graphs = hits table (particle id, r, z per node), first 20 events",
                       float(tmeta["fs_node_check"]["identical"]), float(tmeta["fs_node_check"]["events"]),
                       tmeta["fs_node_check"]["identical"] == tmeta["fs_node_check"]["events"]))
        trunc = truncation_table(tt, {"far daughters, HNL train, |d0| >= 300 mm": pop_t[pop_t.ge300],
                                      "far daughters, HNL train, |d0| >= 200 mm": pop_t,
                                      "far daughters, HNL val, |d0| >= 200 mm": pop_v})
    else:
        tmeta, trunc = {}, pd.DataFrame()
    # ---- summaries
    summ = pd.DataFrame(_count_rows(far[far.split == "train"], "train") + _count_rows(far[far.split == "val"], "val"))
    disc = discordant(far, mpart)
    chain = chain.merge(pop[["event_id", "particle_id", "split", "d0", "r_first", "r_bin", "ge300"]],
                        on=["event_id", "particle_id", "split"], how="left")
    os.makedirs(results, exist_ok=True)
    cols_p = ["split", "model", "pipeline", "seed", "event_id", "particle_id", "sample", "d0", "rv", "n_hits",
              "n_pixel_hits", "r_first", "r_bin", "pt", "eta", "ge300", "stage", "loss_class", "graph_ceiling_dm",
              "score_ceiling_dm", "edge_cut", "n_hits_no_edge", "n_true_chain", "n_chain_in_graph",
              "n_chain_above_cut", "min_chain_score", "best_share", "best_purity", "n_tracks", "n_on_tracks",
              "one_short", "best_track_size", "best_track_majority_pid", "best_track_majority_frac", "merged_other",
              "wp", "label"]
    far["pipeline"] = far.label.map(lambda x: PIPE_NAME[PIPE[x]])
    far = far[cols_p].sort_values(["split", "model", "seed", "event_id", "particle_id"])
    far.to_csv(os.path.join(results, f"{name}_particles.csv"), index=False)
    summ.to_csv(os.path.join(results, f"{name}.csv"), index=False)
    disc.to_csv(os.path.join(results, f"{name}_discordant.csv"), index=False)
    mp_out = mpart.copy()
    mp_out["pipeline"] = mp_out.pipeline.map(PIPE_NAME)
    mp_out = mp_out.merge(pop[["event_id", "particle_id", "split", "d0", "r_first", "r_bin"]],
                          on=["event_id", "particle_id", "split"], how="left")
    mp_out.to_csv(os.path.join(results, f"{name}_merge_particles.csv"), index=False)
    mpair["pipeline"] = mpair.pipeline.map(PIPE_NAME)
    mpair = mpair.merge(pop[["event_id", "particle_id", "split", "d0", "r_first", "r_bin"]],
                        on=["event_id", "particle_id", "split"], how="left")
    mpair.to_csv(os.path.join(results, f"{name}_merge_pairs.csv"), index=False)
    chain["pipeline"] = chain.pipeline.map(PIPE_NAME)
    chain.to_csv(os.path.join(results, f"{name}_chain_scores.csv"), index=False)
    trunc.to_csv(os.path.join(results, f"{name}_truncation.csv"), index=False)
    ident.to_csv(os.path.join(out, "identity_graphs.csv"), index=False)
    ck = pd.DataFrame(checks, columns=["check", "value", "reference", "ok"])
    ck.to_csv(os.path.join(results, f"{name}_checks.csv"), index=False)
    req = ck[ck.ok.notna()]
    files = input_files(specs, seeds)
    with cf.ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        sums = dict(zip(files, ex.map(_sha256, list(files))))
    inputs = [{"path": k, "role": v, "bytes": os.path.getsize(k), "sha256": sums[k]} for k, v in files.items()]
    inputs.append({"path": os.path.join(BASE, "armi_s0", "ltf_union_graphs", "trainset.tar"),
                   "role": "side check: its first member only", "bytes": os.path.getsize(
                       os.path.join(BASE, "armi_s0", "ltf_union_graphs", "trainset.tar")), "sha256": None})
    meta = {"started_utc": started, "elapsed_s": round(time.time() - t0, 1), "inputs": inputs,
            "code_sha256": {x: _sha256(_repo(x)) for x in ("src/hnl/eval/far_trainsplit.py",
                                                          "slurm/feedback/far_trainsplit.sbatch",
                                                          "slurm/feedback/far_trainsplit_graphs.sbatch",
                                                          "slurm/feedback/far_trainsplit_repeat.sbatch")},
            "git_head": R4.git_state(),
            "checks": f"{int(req.ok.astype(bool).sum())} of {len(req)} (plus {int(ck.ok.isna().sum())} information rows)",
            "failed_checks": req[~req.ok.astype(bool)].check.tolist(), "population": json.load(open(os.path.join(out, "population.json"))),
            "specs": allspecs, "provenance": prov, "trainset_truncation_check": tc, "truncation": tmeta}
    meta["population"].pop("events", None)
    json.dump(meta, open(os.path.join(out, "analyze_meta.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(ck.to_string())
    print(summ[(summ.split == "train") & (summ.seed == "pooled") & (summ.r_first_bin == "all")]
          .drop(columns=["working_points"]).to_string(index=False))
    print(json.dumps({k: meta[k] for k in ("elapsed_s", "checks")}))
    print(json.dumps(tc))
    return meta


# --------------------------------------------------------------------------------------------------------------------
# repeatability of the GPU scores (why the identity check compares the scores with a tolerance)


def repeat_extract(out: str, seed: str, work: str) -> dict:
    """The identity val events from the STORED val graphs of seed `seed` (acorn (ttbar): ml_graphs; Canto-L:
    ltf_union_graphs) into WORK/{acorn_ttbar,canto_l}/valset, as inputs of two identical GPU scorings."""
    ev = population_events(out)["val"]
    src = {"acorn_ttbar": os.path.join(BASE, f"armi_s{seed}", "ml_graphs", "valset.tar"),
           "canto_l": os.path.join(BASE, f"armi_s{seed}", "ltf_union_graphs", "valset.tar")}
    n = {k: _extract(v, ev, os.path.join(work, k, "valset")) for k, v in src.items()}
    print(json.dumps(n))
    return n


def repeat_compare(out: str, seed: str, work: str) -> dict:
    """Scores of two identical GPU scorings of the stored val graphs (WORK/<pipe>_run{1,2}/valset) against each other
    and against the stored val scores (RUN/gnn_scores/valset.tar)."""
    ev = population_events(out)["val"]
    res = {}
    for pipe in PIPELINES:
        ref = os.path.join(work, f"{pipe}_stored")
        _extract(os.path.join(runs_of(pipe, seed), "gnn_scores", "valset.tar"), ev, ref)
        rows = []
        for e in ev:
            f = f"event{e}.pyg.gz"
            a, b = (os.path.join(work, f"{pipe}_run{k}", "valset", f) for k in (1, 2))
            c12, c1s, c2s = compare_graphs(a, b), compare_graphs(a, os.path.join(ref, f)), \
                compare_graphs(b, os.path.join(ref, f))
            rows.append({"event_id": e, "run1_run2_equal": c12["scores_equal"], "run1_run2_max": c12["max_abs_score_diff"],
                         "run1_stored_equal": c1s["scores_equal"], "run1_stored_max": c1s["max_abs_score_diff"],
                         "run2_stored_equal": c2s["scores_equal"], "run2_stored_max": c2s["max_abs_score_diff"],
                         "edges_equal": bool(c12.get("eq_edge_index") and c1s.get("eq_edge_index"))})
        t = pd.DataFrame(rows)
        res[pipe] = {"events": len(t), "edges_equal": int(t.edges_equal.sum()),
                     "run1_run2_bitwise_equal_events": int(t.run1_run2_equal.sum()),
                     "run1_run2_max_abs_diff": float(t.run1_run2_max.max()),
                     "run1_stored_bitwise_equal_events": int(t.run1_stored_equal.sum()),
                     "run1_stored_max_abs_diff": float(t.run1_stored_max.max()),
                     "run2_stored_bitwise_equal_events": int(t.run2_stored_equal.sum()),
                     "run2_stored_max_abs_diff": float(t.run2_stored_max.max())}
    res["seed"] = seed
    res["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    json.dump(res, open(os.path.join(out, f"repeat_s{seed}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------------------------------------------------
# report


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """A pipe table of a frame (index dropped)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(str(c).replace("|", "\\|") for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def sacct_times(jobs: dict) -> dict:
    """Start / end (UTC), elapsed, node and state of each Slurm job (sacct with TZ=UTC)."""
    import subprocess
    out = {}
    for name, j in jobs.items():
        r = subprocess.run(["sacct", "-j", str(j), "-X", "-n", "-P", "--format=JobID,JobName,Start,End,Elapsed,NodeList,"
                            "State,Partition,AllocCPUS"], capture_output=True, text=True, env={**os.environ, "TZ": "UTC"})
        line = next((ln for ln in r.stdout.splitlines() if ln.split("|")[0] == str(j)), "")
        f = line.split("|") if line else [""] * 9
        out[name] = {"job": str(j), "name": f[1], "start_utc": f[2], "end_utc": f[3], "elapsed": f[4], "node": f[5],
                     "state": f[6], "partition": f[7], "cpus": f[8]}
    return out


def count_table(summ: pd.DataFrame, split: str, lo: float, rb: str = "all", per_seed: bool = False) -> pd.DataFrame:
    """Loss classes per model (pooled over the seeds; per_seed: one row per seed and the pooled row)."""
    s = summ[(summ.split == split) & (summ.d0_min_mm == lo) & (summ.r_first_bin == rb)]
    rows = []
    for m in [SHORT[x] for x in MODELS]:
        g = s[s.model == m].set_index("seed")
        for k in ((*SEEDS, "pooled") if per_seed else ("pooled",)):
            p = g.loc[k]
            row = {"model": m}
            if per_seed:
                row["seed"] = k
            else:
                row["found s0 / s1 / s2"] = " / ".join(f"{int(g.loc[x, 'found'])}" for x in SEEDS)
            row.update({"found / n": f"{int(p.found)} / {int(p.n)}", "graph": int(p.graph),
                        "classifier": int(p.classifier), "fragmented": int(p.fragmented), "impure": int(p.impure),
                        "short": int(p.short), "lost": int(p.lost)})
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_text(g: pd.DataFrame, lm_pass: bool) -> str:
    """Compact merge decisions of one particle's first-pass fragment pairs."""
    out = []
    for r in g.itertuples():
        t = f"{r.a_hits_particle}/{r.a_size}+{r.b_hits_particle}/{r.b_size}:"
        if lm_pass:
            t += (f" p {r.lm_p:.3f} {'>' if r.lm_p_gt_tau else '<='} tau {r.lm_tau:g}" if r.lm_candidate is True
                  or r.lm_candidate == "True" else " no LM candidate")
            t += ", LM merged" if str(r.lm_merged) == "True" else ", LM not merged"
        fit = (f"fit rms {r.fit_rms:.2f} zrms {r.fit_zrms:.2f} out {int(r.fit_n_out)} span {r.fit_span_R:.0f}"
               if np.isfinite(r.fit_rms) else "no fit")
        t += f"{';' if lm_pass else ''} {fit} ({'pass' if str(r.fit_pass) == 'True' else 'fail'}"
        win = [w for w, f in (("any", r.window_any), ("seq", r.window_sequential)) if str(f) == "True"]
        t += f", window {'+'.join(win) if win else 'none'})"
        if str(getattr(r, "mwsm_merged", "nan")) in ("True", "False"):
            t += ", MWS+M merged" if str(r.mwsm_merged) == "True" else ", MWS+M not merged"
        out.append(t)
    return " | ".join(out) if out else "(fewer than 2 fragments with >= 2 of its hits)"


def discordant_table(disc: pd.DataFrame, pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    d = disc[disc.split == split].sort_values(["pipeline", "found_by", "event_id", "particle_id", "seed"])
    rows = []
    for r in d.itertuples():
        pp = pairs[(pairs.pipeline == r.pipeline) & (pairs.split == split) & (pairs.seed == r.seed)
                   & (pairs.event_id == r.event_id) & (pairs.particle_id == r.particle_id)]
        lm = pp[pp.first_pass == "lm"]
        mw = pp[pp.first_pass == "mwsm"]
        same = str(r.first_passes_equal) == "True"
        rows.append({"pipeline": r.pipeline, "seed": r.seed, "event": r.event_id, "particle": r.particle_id,
                     "|d0|": abs(r.d0), "r_first": r.r_first, "hits": int(r.n_hits), "found by": r.found_by,
                     "LM class": r.stage_lm, "MWS+M class": r.stage_mwsm,
                     "DM first pass (LM / MWS)": f"{'y' if str(r.dm_lm_first) == 'True' else 'n'} / "
                                                  f"{'y' if str(r.dm_mw_first) == 'True' else 'n'}",
                     "LM first-pass fragments": r.lm_frags if isinstance(r.lm_frags, str) else "",
                     "pairs (LM first pass, s0 0.2)": _pair_text(lm, True),
                     "pairs (MWS+M first pass)": "same first pass" if same else _pair_text(mw, False)})
    return pd.DataFrame(rows)


def classifier_table(chain: pd.DataFrame, parts: pd.DataFrame, lo: float) -> pd.DataFrame:
    """Hypothesis (b): per pipeline and first-hit radius bin (train, pooled over seeds, |d0| >= lo): particle-seeds,
    graph and classifier losses of each builder, chain-edge scores (fraction of the in-graph chain edges above 0.1 /
    0.2 / 0.5, particles with every in-graph chain edge above 0.2, median score of the first chain edge)."""
    rows = []
    c = chain[(chain.split == "train") & (chain.d0.abs() >= lo)]
    p = parts[(parts.split == "train") & (parts.d0.abs() >= lo)]
    for pipe in PIPELINES:
        for rb in R_LABELS[1:]:
            g = c[(c.pipeline == PIPE_NAME[pipe]) & (c.r_bin == rb)]
            q = p[(p.pipeline == PIPE_NAME[pipe]) & (p.r_bin == rb)]
            ing = g.n_chain_in_graph.sum()
            row = {"pipeline": PIPE_NAME[pipe], "r_first [mm]": rb, "particle-seeds": len(g)}
            for lab in MODELS:
                if PIPE[lab] != pipe:
                    continue
                qq = q[q.label == lab]
                row[f"{BUILDER[lab]} graph / classifier"] = f"{int((qq.stage == 'graph').sum())} / " \
                                                            f"{int((qq.stage == 'classifier').sum())}"
            row.update({"chain edges in graph": int(ing),
                        "frac > 0.1": g["n_chain_gt_0.1"].sum() / ing if ing else np.nan,
                        "frac > 0.2": g["n_chain_gt_0.2"].sum() / ing if ing else np.nan,
                        "frac > 0.5": g["n_chain_gt_0.5"].sum() / ing if ing else np.nan,
                        "all in-graph chain edges > 0.2": f"{int((g['n_chain_gt_0.2'] == g.n_chain_in_graph).sum())}"
                                                          f" / {len(g)}",
                        "median first-edge score": float(g.first_edge_score.median()) if len(g) else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def truncation_md(trunc: pd.DataFrame) -> pd.DataFrame:
    t = trunc[trunc.binning == "coarse"]
    rows = []
    for (pop_, kind), g in t.groupby(["population", "kind"], sort=False):
        g = g.set_index("r_bin")
        row = {"tracks": pop_, "start": kind}
        for b in R_LABELS:
            row[f"{b} mm"] = f"{100 * g.loc[b, 'fraction']:.2f} %"
        row["per epoch at 500-700 / >= 700"] = f"{g.loc['500-700', 'expected_per_epoch']:.0f} / " \
                                               f"{g.loc['>=700', 'expected_per_epoch']:.0f}"
        row["mean hits at >= 700"] = g.loc[">=700", "mean_hits"]
        rows.append(row)
    return pd.DataFrame(rows)


def graph_common(parts: pd.DataFrame, lo: float) -> dict:
    """Hypothesis (c): far daughters (train, |d0| >= lo) in the graph class (lost, and the graph's chain-true edges do
    not DM-match them) per model, and in common; and the graph-ceiling failures (whatever the builder did)."""
    p = parts[(parts.split == "train") & (parts.d0.abs() >= lo)].copy()
    key = ["event_id", "particle_id"]
    p["is_graph"] = p.stage == "graph"
    p["ceil_fail"] = p.graph_ceiling_dm.astype(str) != "True"
    out = {}
    for lab in MODELS:
        out[f"{SHORT[lab]}: graph-class losses (particle-seeds)"] = int(p[p.label == lab].is_graph.sum())
    per = p.groupby(key).is_graph.agg(["sum", "size"])
    out[f"particles in the graph class for every model and seed ({len(MODELS) * len(SEEDS)} of "
        f"{len(MODELS) * len(SEEDS)})"] = int((per["sum"] == per["size"]).sum())
    out["particles in the graph class for >= 1 model-seed"] = int((per["sum"] > 0).sum())
    for pipe in PIPELINES:
        q = p[p.pipeline == PIPE_NAME[pipe]].groupby(key).is_graph.agg(["sum", "size"])
        out[f"particles in the graph class for every {PIPE_NAME[pipe]} row (both builders, 3 seeds)"] = \
            int((q["sum"] == q["size"]).sum())
    for lab in (R4.CANTO, A_LM):
        q = p[p.label == lab]
        out[f"{PIPE_NAME[PIPE[lab]]}: graph ceiling fails (particle-seeds; found or not)"] = int(q.ceil_fail.sum())
    c = p[p.label.isin([R4.CANTO, A_LM])].groupby(key).ceil_fail.agg(["sum", "size"])
    out["particles whose graph ceiling fails in every seed of both pipelines"] = int((c["sum"] == c["size"]).sum())
    return out


def merge_recovery(mp: pd.DataFrame, lo: float) -> pd.DataFrame:
    """Per pipeline and first-hit radius bin (train, pooled over seeds, |d0| >= lo): particle-seeds that each builder's
    first pass (same-layer MWS) does not DM-match and that its merging then recovers, and those that the first pass
    matches and the merging loses."""
    m = mp[(mp.split == "train") & (mp.d0.abs() >= lo)]
    tf = lambda x: x.astype(str) == "True"
    rows = []
    for pipe in PIPELINES:
        for rb in ("all", *R_LABELS[1:]):
            g = m[(m.pipeline == PIPE_NAME[pipe]) & ((m.r_bin == rb) if rb != "all" else True)]
            lf, lmf, mf, mwf = tf(g.dm_lm_first), tf(g.dm_lm), tf(g.dm_mw_first), tf(g.dm_mw)
            rows.append({"pipeline": PIPE_NAME[pipe], "r_first [mm]": rb, "particle-seeds": len(g),
                         "LM first pass (s0 0.2): not matched": int((~lf).sum()),
                         "-> recovered by learned merging": int((~lf & lmf).sum()),
                         "matched -> lost by learned merging": int((lf & ~lmf).sum()),
                         "MWS+M first pass: not matched": int((~mf).sum()),
                         "-> recovered by helix merging": int((~mf & mwf).sum()),
                         "matched -> lost by helix merging": int((mf & ~mwf).sum())})
    return pd.DataFrame(rows)


def pair_decisions(pairs: pd.DataFrame, lo: float) -> pd.DataFrame:
    """Canto-L (one first pass for both builders): the pairs of a far daughter's first-pass fragments (>= 2 of its
    hits each) per first-hit radius bin, with the learned-merging and helix-merging decisions."""
    q = pairs[(pairs.split == "train") & (pairs.pipeline == "Canto-L") & (pairs.first_pass == "lm")
              & (pairs.d0.abs() >= lo)]
    tf = lambda x: x.astype(str) == "True"
    rows = []
    for rb in ("all", *R_LABELS[1:]):
        g = q[(q.r_bin == rb) if rb != "all" else np.ones(len(q), bool)]
        rows.append({"r_first [mm]": rb, "fragment pairs (particle-seeds)": f"{len(g)} ({g.groupby(['seed', 'event_id', 'particle_id']).ngroups})",
                     "LM candidate": int(tf(g.lm_candidate).sum()), "GBDT p > tau": int(tf(g.lm_p_gt_tau).sum()),
                     "merged by LM": int(tf(g.lm_merged).sum()), "helix fit passes": int(tf(g.fit_pass).sum()),
                     "in a merge_fragments window": int((tf(g.window_any) | tf(g.window_sequential)).sum()),
                     "merged by MWS+M": int(tf(g.mwsm_merged).sum()),
                     "median p (candidates)": float(g.lm_p.median()) if g.lm_p.notna().any() else np.nan,
                     "median fit rms": float(g.fit_rms.median()) if len(g) else np.nan})
    return pd.DataFrame(rows)


CODE = {"found": ".", "graph": "G", "classifier": "C", "fragmented": "F", "impure": "I", "short": "S"}


def loss_particles(parts: pd.DataFrame, lo: float, pipe_name: str) -> pd.DataFrame:
    """The train far daughters (|d0| >= lo) lost by at least one row of a pipeline: class per model and seed (codes:
    . found, G graph, C classifier, F fragmented, I impure, S short), with the chain edges in the graph and the lowest
    chain-edge score (seeds 0 / 1 / 2) of that pipeline."""
    p = parts[(parts.split == "train") & (parts.d0.abs() >= lo)]
    key = ["event_id", "particle_id"]
    lost = p[(p.pipeline == pipe_name) & (p.stage != "found")][key].drop_duplicates()
    q = p.merge(lost, on=key)
    rows = []
    for (e, pid), g in q.groupby(key):
        r0 = g.iloc[0]
        row = {"event": e, "particle": pid, "|d0| [mm]": abs(r0.d0), "r_first [mm]": r0.r_first, "hits": int(r0.n_hits)}
        for lab in MODELS:
            gg = g[g.label == lab].set_index("seed")
            row[SHORT[lab]] = " ".join(CODE[gg.loc[int(k), "stage"]] for k in SEEDS)
        gg = g[(g.pipeline == pipe_name) & (g.label.map(BUILDER) == "LM")].set_index("seed")
        row[f"{pipe_name} chain edges in graph"] = " / ".join(f"{int(gg.loc[int(k), 'n_chain_in_graph'])}/"
                                                             f"{int(gg.loc[int(k), 'n_true_chain'])}" for k in SEEDS)
        row[f"{pipe_name} lowest chain score"] = " / ".join(
            f"{gg.loc[int(k), 'min_chain_score']:.4f}" if np.isfinite(gg.loc[int(k), "min_chain_score"]) else "-"
            for k in SEEDS)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["r_first [mm]"]).reset_index(drop=True)


def _kept_summary(md_path: str) -> str:
    """The hand-written Summary section of an existing report (kept when the tables are regenerated)."""
    placeholder = "(Written by hand from the tables below.)\n"
    if not os.path.exists(md_path):
        return placeholder
    txt = open(md_path).read()
    a = txt.find("## Summary\n")
    if a < 0:
        return placeholder
    b = txt.find("\n## ", a + 1)
    body = txt[a + len("## Summary\n"):b + 1 if b >= 0 else len(txt)].strip("\n")
    return body + "\n" if body else placeholder


def run_report(out: str, results: str, name: str, jobs: dict) -> dict:
    summ = pd.read_csv(os.path.join(results, f"{name}.csv"), dtype={"seed": str})
    parts = pd.read_csv(os.path.join(results, f"{name}_particles.csv"))
    disc = pd.read_csv(os.path.join(results, f"{name}_discordant.csv"))
    pairs = pd.read_csv(os.path.join(results, f"{name}_merge_pairs.csv"))
    chain = pd.read_csv(os.path.join(results, f"{name}_chain_scores.csv"))
    tpath = os.path.join(results, f"{name}_truncation.csv")
    trunc = pd.read_csv(tpath) if os.path.getsize(tpath) > 2 else pd.DataFrame()
    ck = pd.read_csv(os.path.join(results, f"{name}_checks.csv"))
    am = json.load(open(os.path.join(out, "analyze_meta.json")))
    pj = json.load(open(os.path.join(out, "population.json")))
    pop_t = pd.read_parquet(os.path.join(out, "population_train.parquet"))
    rep = {s: json.load(open(p)) for s in SEEDS if os.path.exists(p := os.path.join(out, f"repeat_s{s}.json"))}
    L = []
    L.append("# Far-displaced HNL daughters on the HNL training split: where Canto-L and acorn (ttbar) lose them\n")
    L.append("Diagnostic only. The sample is the HNL TRAINING split, which neither pipeline has seen (both are trained on "
             "ttbar only); nothing is chosen on it, and no frozen choice (results/final_selection_v1.json) or paper file "
             "is touched. No test event, test graph, test score or test output was read. Code "
             "`src/hnl/eval/far_trainsplit.py` (job scripts `slurm/feedback/far_trainsplit*.sbatch`); jobs, times and "
             f"inputs with sha256 in `results/{name}_meta.json`.\n")
    L.append("## Summary\n")
    L.append(_kept_summary(os.path.join(results, f"{name}.md")))
    L.append("## Population and pipelines\n")
    L.append(f"HNL train events (hnl10_100 + hnl15_200): {pj['train_events']:,}; primary-population daughters "
             f"(HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm): {pj['primary_daughters']:,}; with "
             f"|d0| >= 200 mm: {pj['far_ge200']} in {pj['events_ge200']} events; with |d0| >= 300 mm: "
             f"{pj['far_ge300']} in {pj['events_ge300']} events (validation: {pj['val_far_ge200']} and "
             f"{pj['val_far_ge300']}). r_first is the transverse radius of the particle's first hit.\n")
    rows = []
    for lab_, f in (("|d0| >= 300 mm", pop_t[pop_t.ge300]), ("|d0| >= 200 mm", pop_t)):
        c = f.r_bin.value_counts()
        rows.append({"daughters": lab_, "n": len(f), **{f"r_first {b} mm": int(c.get(b, 0)) for b in R_LABELS},
                     "hnl10_100 / hnl15_200": f"{int((f['sample'] == 'hnl10_100').sum())} / "
                                              f"{int((f['sample'] == 'hnl15_200').sum())}",
                     "median hits": float(f.n_hits.median())})
    L.append(_md(pd.DataFrame(rows), "{:.0f}") + "\n")
    sp = pd.DataFrame(am["specs"]).sort_values(["seed"], kind="stable")
    sp["model"] = sp.label.map(SHORT)
    rows = []
    for m, g in sp.groupby("model", sort=False):
        fam = g.family.iloc[0]
        rows.append({"model": m, "builder (val-chosen, seeds 0 / 1 / 2)": f"{fam} {' / '.join(f'{v:g}' for v in g.param)} "
                     f"({g.grid.iloc[0]})", "edge cut": " / ".join(f"{v:g}" for v in g.cut),
                     "merging GBDT": (os.path.relpath(g.merge_model.iloc[0], DATA).replace("armi_s0", "armi_s<S>")
                                      if isinstance(g.merge_model.iloc[0], str) else "-")})
    sp = pd.DataFrame(rows).set_index("model")
    sp = sp.loc[[SHORT[x] for x in MODELS]]
    L.append("Pipelines (per seed 0 / 1 / 2, the committed val-chosen configurations; Canto-L's learned triplet union "
             "at k 5, tau 0.005, r_w 0.24 with each seed's own triplet GBDT; acorn (ttbar): its metric-learning graph; "
             "the edge cut of the classifier class is the s0 of the first same-layer Mutex Watershed):\n")
    L.append(_md(sp.reset_index()) + "\n")
    # identity
    L.append("## Identity check on the 49 validation events of round 4 check 2\n")
    req = ck[ck.ok.notna()]
    rmax = max((d[p_]["run1_run2_max_abs_diff"] for d in rep.values() for p_ in PIPELINES), default=np.nan)
    L.append(f"{int((req.ok.astype(str) == 'True').sum())} of {len(req)} checks pass, plus {int(ck.ok.isna().sum())} "
             f"information rows (`results/{name}_checks.csv`). The GPU edge scores are not bitwise repeatable: two "
             "identical scorings of the stored val graphs with the same checkpoint on an A30 differ by up to "
             f"{rmax:.1e} (table below). The rescored val scores are therefore compared with the stored ones within "
             f"{SCORE_TOL:g} and on the same side of every builder cut; the graphs, the DM matches and the classes are "
             "compared exactly.\n")
    ck = ck.assign(ok=ck.ok.fillna("information"))
    L.append(_md(ck[["check", "value", "reference", "ok"]], "{:g}") + "\n")
    if rep:
        L.append("Repeatability of the GPU scores (`slurm/feedback/far_trainsplit_repeat.sbatch`: the stored val graphs "
                 "of the 49 events scored twice with the same checkpoint on an A30):\n")
        rr = []
        for s, d in rep.items():
            for pipe in PIPELINES:
                x = d[pipe]
                rr.append({"seed": s, "pipeline": PIPE_NAME[pipe],
                           "run 1 = run 2 (bitwise), events": f"{x['run1_run2_bitwise_equal_events']} / {x['events']}",
                           "max |run 1 - run 2|": x["run1_run2_max_abs_diff"],
                           "run 1 = stored (bitwise), events": f"{x['run1_stored_bitwise_equal_events']} / {x['events']}",
                           "max |run 1 - stored|": x["run1_stored_max_abs_diff"],
                           "max |run 2 - stored|": x["run2_stored_max_abs_diff"]})
        L.append(_md(pd.DataFrame(rr), "{:.2e}") + "\n")
    # counts
    for lo in FAR:
        L.append(f"## Loss classes at |d0| >= {int(lo)} mm (train split)\n")
        L.append("Pooled = particle-seeds over the three seeds.\n")
        L.append(_md(count_table(summ, "train", lo)) + "\n")
        L.append("Per seed:\n")
        L.append(_md(count_table(summ, "train", lo, per_seed=True)) + "\n")
    L.append("The same code path on the 49 validation events (round 4 check 2: Canto-L LM 50 / 57 and 142 / 153, "
             "Canto-L MWS+M 49 / 57 and 141 / 153):\n")
    for lo in FAR:
        L.append(f"|d0| >= {int(lo)} mm:\n")
        L.append(_md(count_table(summ, "val", lo)) + "\n")
    L.append("## By first-hit radius (train split, pooled over the seeds)\n")
    for lo in FAR:
        t = pd.concat([count_table(summ, "train", lo, rb).assign(**{"r_first [mm]": rb}) for rb in R_LABELS[1:]],
                      ignore_index=True)
        t = t[["r_first [mm]", *[c for c in t.columns if c != "r_first [mm]"]]]
        L.append(f"|d0| >= {int(lo)} mm (no daughter at this |d0| has r_first < 200 mm):\n")
        L.append(_md(t) + "\n")
    mpp = pd.read_csv(os.path.join(results, f"{name}_merge_particles.csv"))
    key = ["pipeline", "seed", "event_id", "particle_id"]
    jj = parts[parts.split == "train"].merge(mpp[key + ["dm_lm_first", "dm_mw_first"]], on=key, how="left")
    gm = []
    for lab in MODELS:
        g = jj[(jj.label == lab) & (jj.stage == "graph")]
        fp = (g.dm_lm_first if BUILDER[lab] == "LM" else g.dm_mw_first).astype(str) == "True"
        gm.append(f"{SHORT[lab]} {int(fp.sum())} of {len(g)}")
    L.append("The classes follow round 4 check 2 (first that applies: found, graph, classifier, builder), so a "
             "particle that the builder loses and whose graph ceiling also fails counts as graph. Graph-class "
             "particle-seeds (train, |d0| >= 200 mm) that the builder's own first pass did DM-match: "
             + "; ".join(gm) + f" (`results/{name}_merge_particles.csv`).\n")
    for pipe in PIPELINES:
        L.append(f"Train far daughters with |d0| >= 300 mm lost by at least one {PIPE_NAME[pipe]} row, class per seed "
                 "0 / 1 / 2 (. found, G graph, C classifier, F fragmented, I impure, S short):\n")
        L.append(_md(loss_particles(parts, 300.0, PIPE_NAME[pipe]), "{:.0f}") + "\n")
    # LM vs MWS+M
    L.append("## Learned merging (LM) against MWS-layer + helix merging (MWS+M)\n")
    rows = []
    for lo in FAR:
        d = disc[(disc.split == "train") & (disc.d0.abs() >= lo)]
        for pipe in PIPELINES:
            g = d[d.pipeline == PIPE_NAME[pipe]]
            a, b = g[g.found_by == "LM only"], g[g.found_by == "MWS+M only"]
            rows.append({"|d0| >=": int(lo), "pipeline": PIPE_NAME[pipe],
                         "found by LM only (MWS+M class)": f"{len(a)} ({', '.join(f'{k} {v}' for k, v in a.stage_mwsm.value_counts().items())})",
                         "found by MWS+M only (LM class)": f"{len(b)} ({', '.join(f'{k} {v}' for k, v in b.stage_lm.value_counts().items())})",
                         "r_first of LM-only finds": " ".join(f"{x:.0f}" for x in sorted(a.r_first)),
                         "r_first of MWS+M-only finds": " ".join(f"{x:.0f}" for x in sorted(b.r_first))})
    L.append(_md(pd.DataFrame(rows)) + "\n")
    for lo in FAR:
        L.append(f"Recovery by merging, |d0| >= {int(lo)} mm (train, pooled over seeds; 'matched' = DM-matched; "
                 "Canto-L's two builders share one first pass, acorn (ttbar)'s LM uses s0 0.2 and its MWS+M s0 0.1):\n")
        L.append(_md(merge_recovery(mpp, lo)) + "\n")
        L.append(f"Canto-L, every pair of a far daughter's first-pass fragments (>= 2 of its hits each), |d0| >= "
                 f"{int(lo)} mm (train, pooled over seeds):\n")
        L.append(_md(pair_decisions(pairs, lo)) + "\n")
    L.append("Every train particle-seed with |d0| >= 200 mm that one builder finds and the other loses; fragments are "
             "the first-pass tracks holding >= 2 of the particle's hits (its hits / track size); for every pair of them "
             "the merging GBDT's probability against tau and the helix fit of the union against merge_fragments' cuts "
             "(rms <= 1, zrms <= 5, <= 1 outlier, R span >= 50 mm; candidate windows 'any' and 'seq'):\n")
    L.append(_md(discordant_table(disc, pairs, "train"), "{:.0f}") + "\n")
    # hypotheses (b), (c)
    L.append("## Classifier losses and chain-edge scores by first-hit radius (hypothesis b)\n")
    for lo in FAR:
        L.append(f"|d0| >= {int(lo)} mm (train, pooled over seeds; Canto-L's cut is 0.2 for both builders, acorn "
                 "(ttbar)'s 0.2 for LM and 0.1 for MWS+M):\n")
        L.append(_md(classifier_table(chain, parts, lo)) + "\n")
    if len(trunc):
        L.append("Start radius of ttbar training tracks under the truncation augmentation (hnl.models.augment: each "
                 "particle with >= 6 hits truncated with p = 0.3, its m innermost hits in R removed, m uniform in "
                 "1..n-5; expected values for one epoch over the 7,999 ttbar train events; 'natural' = no "
                 "augmentation, 'truncated' = the starts of truncated tracks only, 'augmented' = every track of one "
                 "augmented epoch) against the first-hit radius of the far daughters:\n")
        L.append(_md(truncation_md(trunc), "{:.1f}") + "\n")
    L.append("## Graph losses common to both pipelines (hypothesis c)\n")
    for lo in FAR:
        L.append(f"|d0| >= {int(lo)} mm (graph ceiling = CC on the graph's chain-true edges; Canto-L's union graph "
                 "contains acorn (ttbar)'s metric-learning graph of the same seed):\n")
        L.append(_md(pd.DataFrame([{"quantity": k, "count": v} for k, v in graph_common(parts, lo).items()])) + "\n")
    tc = am.get("trainset_truncation_check", {})
    L.append("## Side check: Canto-L's model class truncates 'trainset' items at inference\n")
    L.append(f"One stored ttbar training graph of Canto-L seed 0 ({tc.get('graph', '')}), staged both as trainset and as "
             f"valset and loaded through {tc.get('model_class', '')}'s inference setup (hnl.train.acorn_runner.infer): "
             f"{tc.get('nodes_stored')} nodes / {tc.get('edges_stored')} edges stored, {tc.get('nodes_valset_item')} / "
             f"{tc.get('edges_valset_item')} as a valset item, {tc.get('nodes_trainset_item')} / "
             f"{tc.get('edges_trainset_item')} as a trainset item. All events here were therefore staged as valset, "
             "like the val graphs.\n")
    L.append("## Files\n")
    L.append(f"- `results/{name}.csv`: counts per split, |d0| threshold, first-hit radius bin, model and seed (and "
             "pooled).\n"
             f"- `results/{name}_particles.csv`: one row per far daughter, model and seed (class and diagnostics, as "
             "round 4 check 2).\n"
             f"- `results/{name}_discordant.csv`, `results/{name}_merge_particles.csv`, `results/{name}_merge_pairs.csv`: "
             "LM vs MWS+M disagreements, per far daughter the DM match after each building step and its first-pass "
             "fragments, and every pair of these fragments (GBDT probability and features, helix fit, windows, merged "
             "or not).\n"
             f"- `results/{name}_chain_scores.csv`: per far daughter, model and seed, its chain edges in R order with "
             "their scores.\n"
             f"- `results/{name}_truncation.csv`: start-radius distributions (100 mm and coarse bins).\n"
             f"- `results/{name}_checks.csv`, `results/{name}_meta.json`.\n"
             "- Graphs and scores (small, only these 371 events per seed): "
             "`$HNL_DATA/runs/feedback_v4/far_trainsplit/s<S>/{acorn_ttbar,canto_l}/valset.tar`.\n")
    md = "\n".join(L)
    with open(os.path.join(results, f"{name}.md"), "w") as f:
        f.write(md)
    outs = [os.path.join(results, f"{name}{x}") for x in (".csv", "_particles.csv", "_discordant.csv",
                                                          "_merge_particles.csv", "_merge_pairs.csv",
                                                          "_chain_scores.csv", "_truncation.csv", "_checks.csv")]
    outs += sorted(glob.glob(os.path.join(out, "s*", "*", "valset.tar"))) + [os.path.join(out, x) for x in (
        "population_train.parquet", "population_val.parquet", "population.json", "analyze_meta.json",
        "identity_graphs.csv")] + sorted(glob.glob(os.path.join(out, "repeat_s*.json")))
    du = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    meta = {"name": name, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "status": "Diagnostic only, on the HNL TRAINING split (hnl10_100 and hnl15_200 train events), which neither "
                      "ttbar-trained pipeline has seen. Nothing is chosen on it; results/final_selection_v1.json and the "
                      "paper files are untouched.",
            "test_data": "No test event, test graph, test score or test output was read or evaluated: the feature-store "
                         "tars were opened per split (<sample>_train.tar, <sample>_val.tar, ttbar_train.tar), every run "
                         "file used is a train or val file. While locating the learned-triplet union settings, the "
                         "configuration record RUN/ltf_union_graphs/report_test/build.json of the three seeds (settings "
                         "and file paths only, no event data, score or truth-derived number) was displayed once; its "
                         "values were not used.",
            "git": {"head_at_analysis": am.get("git_head"), "code_sha256_at_analysis": am.get("code_sha256"),
                    "note": "the module and job scripts were uncommitted while the jobs ran and are committed together "
                            "with this file. The analysis code is the one of code_sha256_at_analysis (job 6713002); "
                            "afterwards only the report code (run_report and its tables) was extended, and the Polars "
                            "AVX2 compat line of slurm/baseline/_common.sh was added to the three job scripts (the graph "
                            "jobs, submitted before, ran on A30 nodes, which have AVX2)."},
            "jobs": sacct_times(jobs), "population": {k: v for k, v in pj.items() if k != "events"},
            "checks": am.get("checks"), "failed_checks": am.get("failed_checks"), "repeatability": rep,
            "trainset_truncation_check": am.get("trainset_truncation_check"), "truncation": am.get("truncation"),
            "specs": am.get("specs"), "inputs": am.get("inputs"),
            "outputs": [{"path": x, "bytes": os.path.getsize(x), "sha256": _sha256(x)} for x in outs if os.path.exists(x)],
            "storage_bytes_under_out": du, "out": out}
    with open(os.path.join(results, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return {"md": os.path.join(results, f"{name}.md"), "meta": os.path.join(results, f"{name}_meta.json"),
            "storage_bytes": du, "jobs": meta["jobs"]}


# --------------------------------------------------------------------------------------------------------------------
# CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("population")
    p.add_argument("--out", default=OUT)
    q = sub.add_parser("params")
    q.add_argument("--seed", required=True, choices=list(SEEDS))
    s = sub.add_parser("stage")
    s.add_argument("--out", default=OUT)
    s.add_argument("--dest", required=True)
    a_ = sub.add_parser("analyze")
    a_.add_argument("--out", default=OUT)
    a_.add_argument("--results", default=_repo("results"))
    a_.add_argument("--name", default="far_trainsplit_v1")
    a_.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a_.add_argument("--tmp", default=os.environ.get("LOCAL", tempfile.gettempdir()))
    a_.add_argument("--seeds", default=",".join(SEEDS), help="a subset of the seeds (a dry run)")
    a_.add_argument("--no-truncation", action="store_true", help="skip the truncation start-radius table")
    rp = sub.add_parser("repeat")
    rp.add_argument("--out", default=OUT)
    rp.add_argument("--seed", default="0", choices=list(SEEDS))
    rp.add_argument("--work", required=True)
    rp.add_argument("--phase", required=True, choices=["extract", "compare"])
    r_ = sub.add_parser("report")
    r_.add_argument("--out", default=OUT)
    r_.add_argument("--results", default=_repo("results"))
    r_.add_argument("--name", default="far_trainsplit_v1")
    r_.add_argument("--jobs", default="", help="name:jobid,... (Slurm jobs of this read-out)")
    a = ap.parse_args(argv)
    if a.cmd == "population":
        run_population(a.out)
    elif a.cmd == "params":
        for k, v in graph_params(a.seed).items():
            print(f"{k}={v}")
    elif a.cmd == "stage":
        stage(a.out, a.dest)
    elif a.cmd == "repeat":
        (repeat_extract if a.phase == "extract" else repeat_compare)(a.out, a.seed, a.work)
    elif a.cmd == "report":
        jobs = dict(x.split(":", 1) for x in a.jobs.split(",") if x)
        print(json.dumps(run_report(a.out, a.results, a.name, jobs), indent=1))
    else:
        run_analyze(a.out, a.results, a.name, a.workers, a.tmp, tuple(a.seeds.split(",")), not a.no_truncation)


if __name__ == "__main__":
    main()
