"""Learned HelixFilter (Stage 1 A: the learned graph booster; design synthesis step 5c). The hand-cut support filter
of hnl.graphs.helix_union replaced by a gradient-boosted classifier trained on ttbar TRAIN events.

Candidates: the pairs of a run's own metric-learning embedding within r_w (default 0.24) that are not in its base
graph (radius r*): what a union can add. Per candidate (oriented outward in R as in helix_union), production-point-
invariant features from the helix triplets it belongs to (helix_union.candidate_triplets / triplet_features: s-z slope
continuity in units of its error, turning angle, the combined consistency score), as the in-leg (continuations at its
outer hit) and as the out-leg (predecessors at its inner hit): the number of good triplets, the best score, its slope
pull and |turning angle|, and the rank of the candidate among its partners' choices; plus its transverse length,
|cot theta| and the same-layer flag. No absolute position, no layer id, no "in base graph" flag, no embedding
distance. Label: both hits belong to one particle (--label samepid), or the pair is a chain edge (--label chain: the
consecutive pairs the graph ceiling needs, ~0.06 % of the candidates). The union keeps every base edge and adds the
candidates with p > tau.

    python -m hnl.graphs.learned_filter data --ml-run RUN --split train --out SHARD.npz [--max-events N] [--row-frac 0.05]
    python -m hnl.graphs.learned_filter train --data 'DIR/*.npz' --out MODEL.joblib
    python -m hnl.graphs.learned_filter scan --ml-run RUN --split val --model MODEL.joblib --out DIR
        (pairs per event and graph ceilings at a grid of tau, against the hand-cut union and the base graph)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd

from hnl.graphs import ceiling as C
from hnl.graphs import helix_union as HU

FEATURES = ["L_t", "abs_cot", "same_layer", "n_good_in", "n_good_out", "best_score_in", "best_score_out",
            "best_pull_cot_in", "best_pull_cot_out", "best_turn_in", "best_turn_out", "best_rank_in",
            "best_rank_out"]
TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def candidate_features(r, phi, z, vol, lid, cand, nsig: float = 5.0, pt_min: float = 0.5):
    """(src, dst, X) for candidate pairs (2, E) oriented outward in R; X in FEATURES order (NaN: no good triplet)."""
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    R = np.hypot(r, z)
    a, b = np.asarray(cand[0], dtype=np.int64), np.asarray(cand[1], dtype=np.int64)
    fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    src, dst = np.where(fwd, a, b), np.where(fwd, b, a)
    E = len(src)
    same_layer = lid[src] == lid[dst]
    lg = HU.legs(xyz, src, dst, vol)
    X = np.full((E, len(FEATURES)), np.nan)
    col = {f: k for k, f in enumerate(FEATURES)}
    X[:, col["L_t"]] = lg["L"]
    X[:, col["abs_cot"]] = np.minimum(np.abs(lg["cot"]), HU.COT_CLIP)
    X[:, col["same_layer"]] = same_layer
    X[:, col["n_good_in"]] = 0.0
    X[:, col["n_good_out"]] = 0.0
    if E == 0:
        return src, dst, X
    e_in, e_out = HU.candidate_triplets(src, dst, lg, (~same_layer) & (lg["L"] > 1e-3), nsig)
    good, score, pull_cot, turn = HU.triplet_features(lg, e_in, e_out, nsig, pt_min)
    e_in, e_out, score, pull_cot, turn = e_in[good], e_out[good], score[good], pull_cot[good], np.abs(turn[good])
    if len(score) == 0:
        return src, dst, X
    rk_in = HU.rank_within(e_in, score)        # the triplet's rank among the continuations of its in-leg
    rk_out = HU.rank_within(e_out, score)      # ... among the predecessors of its out-leg
    for side, leg, partner_rank in (("in", e_in, rk_out), ("out", e_out, rk_in)):
        X[:, col[f"n_good_{side}"]] = np.bincount(leg, minlength=E)
        o = np.lexsort((score, leg))
        first = np.r_[True, leg[o][1:] != leg[o][:-1]]
        best = o[first]
        X[leg[best], col[f"best_score_{side}"]] = score[best]
        X[leg[best], col[f"best_pull_cot_{side}"]] = pull_cot[best]
        X[leg[best], col[f"best_turn_{side}"]] = turn[best]
        X[leg[best], col[f"best_rank_{side}"]] = partner_rank[best]
    return src, dst, X


def chain_labels(g, src, dst) -> np.ndarray:
    """1 if the candidate pair is a chain edge (consecutive hits of one particle, the stored track_edges)."""
    n = int(g.num_nodes)
    te = C.unordered_unique(g.track_edges.numpy())
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    return np.isin(a.astype(np.int64) * n + b, te[0] * n + te[1])


def _event(g, emb, lid, r_wide: float):
    """(base pairs (2, B), candidate src, dst, X, pid) of one stored metric-learning graph."""
    n = int(g.num_nodes)
    r, phi, z = (g[f].double().numpy() for f in ("r", "phi", "z"))
    base = C.unordered_unique(g.edge_index.numpy())
    wide = HU.radius_pairs(emb(g), r_wide)
    bc, wc = C._codes(base, n), C._codes(wide, n)
    add = wide[:, ~np.isin(wc, bc)]
    src, dst, X = candidate_features(r, phi, z, lid // 1000, lid, add)
    return base, src, dst, X, g.hit_particle_id.numpy()


def _graphs(ml_run: str, split: str, max_events: int | None, seed: int = 0):
    src = os.path.join(ml_run, "ml_graphs", f"{split}set.tar")
    with tarfile.open(src) as t:
        names = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))
    if max_events is not None and max_events < len(names):
        pick = np.random.default_rng(seed).choice(len(names), max_events, replace=False)
        names = [names[k] for k in np.sort(pick)]
    return src, names


def _data_chunk(args):
    from hnl.trackbuild.mutex_ws import hit_layers
    import torch
    src, names, ckpt, r_wide, row_frac, label = args
    torch.set_num_threads(1)
    emb = HU.Embedding(ckpt)
    eids = [int(os.path.basename(nm)[5:].split(".")[0]) for nm in names]
    L = hit_layers(eids)
    Xs, ys, evs = [], [], []
    with tarfile.open(src) as t:
        for eid, nm in zip(eids, names):
            g = HU._load_bytes(t.extractfile(nm).read())
            _, s, d, X, pid = _event(g, emb, L[eid], r_wide)
            y = chain_labels(g, s, d) if label == "chain" else (pid[s] == pid[d]) & (pid[s] != 0)
            keep = y | (np.random.default_rng(eid).random(len(y)) < row_frac)   # every positive, a fraction of negatives
            w = np.where(y, 1.0, 1.0 / row_frac)
            Xs.append(np.c_[X[keep], w[keep]].astype(np.float32))
            ys.append(y[keep])
            evs.append(np.full(int(keep.sum()), eid, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(evs)


def _scan_chunk(args):
    """Per event: base / hand-cut union / learned unions at TAUS: pair counts and per-particle DM ceilings."""
    from hnl.trackbuild.mutex_ws import hit_layers
    import joblib
    import torch
    src, names, ckpt, r_wide, model_path, taus = args
    torch.set_num_threads(1)
    emb = HU.Embedding(ckpt)
    model = joblib.load(model_path)
    eids = [int(os.path.basename(nm)[5:].split(".")[0]) for nm in names]
    L = hit_layers(eids)
    rows, parts = [], []
    with tarfile.open(src) as t:
        for eid, nm in zip(eids, names):
            g = HU._load_bytes(t.extractfile(nm).read())
            n = int(g.num_nodes)
            pid = g.hit_particle_id.numpy()
            te = g.track_edges.numpy()
            base, s, d, X, _ = _event(g, emb, L[eid], r_wide)
            p = model.predict_proba(X)[:, 1] if len(s) else np.zeros(0)
            r, phi, z = (g[f].double().numpy() for f in ("r", "phi", "z"))
            hand, _ = HU.support_filter(r, phi, z, L[eid] // 1000, L[eid], np.stack([s, d]))
            graphs = {"base": base, "hand": np.concatenate([base, hand], axis=1)}
            for tau in taus:
                graphs[f"tau{tau:g}"] = np.concatenate([base, np.stack([s[p > tau], d[p > tau]])], axis=1)
            for name, gr in graphs.items():
                gr = C.unordered_unique(gr)
                st, part, _ = C.evaluate_graph(n, pid, te, gr)
                rows.append({"event_id": eid, "graph": name, "n_pairs": gr.shape[1], "n_true_in_graph": st["n_true_in_graph"],
                             "n_true": st["n_true"]})
                part = part.reset_index()[["particle_id", "dm_matched", "dm_matched_samepid"]]
                part["event_id"], part["graph"] = eid, name
                parts.append(part)
    return pd.DataFrame(rows), pd.concat(parts, ignore_index=True)


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--ml-run", required=True)
    d.add_argument("--split", default="train")
    d.add_argument("--out", required=True)
    d.add_argument("--max-events", type=int, default=2000)
    d.add_argument("--row-frac", type=float, default=0.05)
    d.add_argument("--r-wide", type=float, default=0.24)
    d.add_argument("--label", default="samepid", choices=["samepid", "chain"],
                   help="positives: same particle, or chain edges only (what the graph ceiling needs)")
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("scan")
    s.add_argument("--ml-run", required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-events", type=int, default=None)
    s.add_argument("--r-wide", type=float, default=0.24)
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "data":
        src, names = _graphs(a.ml_run, a.split, a.max_events)
        ckpt = HU.best_ckpt(a.ml_run, "ml")
        chunks = [(src, names[k:k + 20], ckpt, a.r_wide, a.row_frac, a.label) for k in range(0, len(names), 20)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            res = list(ex.map(_data_chunk, chunks))
        X = np.concatenate([x[0] for x in res])
        y = np.concatenate([x[1] for x in res])
        ev = np.concatenate([x[2] for x in res])
        np.savez(a.out, X=X[:, :-1], w=X[:, -1], y=y, event=ev, features=np.array(FEATURES))
        print(json.dumps({"events": len(names), "rows": int(len(y)), "positives": int(y.sum()),
                          "elapsed_s": round(time.time() - t0, 1)}))
    elif a.cmd == "train":
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        Z = [np.load(f) for f in sorted(glob.glob(a.data))]
        X = np.concatenate([z["X"] for z in Z])
        y = np.concatenate([z["y"] for z in Z])
        w = np.concatenate([z["w"] for z in Z])
        ev = np.concatenate([z["event"] for z in Z])
        rng = np.random.default_rng(a.seed)
        uev = np.unique(ev)
        hold = np.isin(ev, rng.choice(uev, size=max(1, len(uev) // 10), replace=False))
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                             l2_regularization=1.0, random_state=a.seed)
        clf.fit(X[~hold], y[~hold], sample_weight=w[~hold])
        clf.hnl_features = FEATURES
        p = clf.predict_proba(X[hold])[:, 1]
        info = {"rows_train": int((~hold).sum()), "holdout_auc_weighted": float(roc_auc_score(y[hold], p,
                sample_weight=w[hold])), "positives": int(y.sum()), "features": FEATURES,
                "elapsed_s": round(time.time() - t0, 1)}
        joblib.dump(clf, a.out)
        with open(a.out + ".json", "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info))
    else:
        from hnl.eval.baseline_val import populations
        from hnl.eval.particles import particle_table
        src, names = _graphs(a.ml_run, a.split, a.max_events)
        ckpt = HU.best_ckpt(a.ml_run, "ml")
        chunks = [(src, names[k:k + 20], ckpt, a.r_wide, a.model, TAUS) for k in range(0, len(names), 20)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            res = list(ex.map(_scan_chunk, chunks))
        pe = pd.concat([x[0] for x in res], ignore_index=True)
        pp = pd.concat([x[1] for x in res], ignore_index=True)
        os.makedirs(a.out, exist_ok=True)
        pe.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
        pp.to_parquet(os.path.join(a.out, "per_particle.parquet"), index=False)
        info = particle_table(np.unique(pe.event_id))
        P = populations(info)
        rows = []
        code = pe.event_id // 100_000_000
        for gname, gg in pe.groupby("graph"):
            q = pp[pp.graph == gname].merge(info, on=["event_id", "particle_id"])
            Q = populations(q)
            big = q[Q["primary"] & (q.d0.abs() >= 300)]
            rows.append({"graph": gname,
                         "pairs_hnl": float(gg[np.isin(code[gg.index], (2, 4))].n_pairs.mean()),
                         "pairs_ttbar": float(gg[code[gg.index] == 1].n_pairs.mean()),
                         "ceil_primary": float(q[Q["primary"]].dm_matched.mean()),
                         "ceil_d0_ge300": float(big.dm_matched.mean()) if len(big) else np.nan,
                         "ceil_w_muons": float(q[Q["w_muons"]].dm_matched.mean()),
                         "ceil_ttbar_targets": float(q[Q["ttbar_targets"]].dm_matched.mean())})
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
        pd.set_option("display.width", 200)
        print(summ.round(4).to_string(index=False))
        print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
