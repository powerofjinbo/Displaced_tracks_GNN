"""Learned triplet support (Stage 1 A, learned graph booster v3): the HelixFilter of hnl.graphs.helix_union with its
hand-set triplet tests (slope pull < nsig, turning angle within the arc of a pT >= pt_min helix, the chi2-like
consistency score) replaced by a gradient-boosted triplet probability trained on ttbar TRAIN events; the structure of
the hand-cut filter is kept: candidates = all pairs of the run's own embedding within r_w, legs oriented outward in R,
triplets a -> b -> c of candidate legs meeting at b within the generous slope window of
helix_union.candidate_triplets, a triplet selected if p > tau and among the k most probable of BOTH its legs (mutual
top k), its legs kept, same-layer pairs kept as in the hand-cut filter, the union with the base graph.

Triplet features (all invariant under translations of the production point and rotations about z, like the hand-cut
tests): slope pull |dcot| / error, |turning angle| and its pull, the pT implied by the turning angle, the two
transverse leg lengths and their ratio, |cot theta| of both legs, the per-leg slope and direction errors (per-hit
resolutions of the ttbar-train module-frame table), |dz| of both legs and the 3D opening angle. Label: both legs are
chain edges (consecutive hits of one particle: what the graph ceiling needs; ~0.08 % of the candidate triplets of a
ttbar event). Training keeps every positive and a fraction row_frac of the negatives, unweighted by default: the
probability is that of the subsampled class ratio (odds x row_frac would restore the true one), a monotonic map that
the mutual top-k ranking ignores and the tau scan absorbs (weighting the negatives back, 1:1200, made the boosting stop
at a constant). Rule R7 (second part): the learned filter replaces the hand cuts if its val primary ceiling at equal
ttbar graph size is <= 0.2 point below.

    python -m hnl.graphs.learned_triplet_filter data --ml-run RUN --split train --out OUT.npz [--max-events 400]
        [--row-frac 0.01]
    python -m hnl.graphs.learned_triplet_filter train --data OUT.npz --out MODEL.joblib
    python -m hnl.graphs.learned_triplet_filter scan --ml-run RUN --split val --model MODEL.joblib --out DIR
        [--taus 0.05,0.1,...] [--ks 1,2,3]     (pairs per event and ceilings against the hand-cut union and the base)
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd

from hnl.graphs import ceiling as C
from hnl.graphs import helix_union as HU
from hnl.graphs.learned_filter import _graphs

FEATURES = ["pull_cot", "abs_turn", "turn_pull", "log_pt_turn", "L_in", "L_out", "log_L_ratio", "abs_cot_in",
            "abs_cot_out", "s_cot_in", "s_cot_out", "s_t_in", "s_t_out", "abs_dz_in", "abs_dz_out", "cos3d"]
TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
KS = (1, 2, 3)


def legs_and_triplets(r, phi, z, vol, lid, cand):
    """Oriented legs (src, dst), leg table, same-layer flags, candidate triplets (e_in, e_out) and their features."""
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    R = np.hypot(r, z)
    a, b = np.asarray(cand[0], dtype=np.int64), np.asarray(cand[1], dtype=np.int64)
    fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    src, dst = np.where(fwd, a, b), np.where(fwd, b, a)
    same_layer = lid[src] == lid[dst]
    lg = HU.legs(xyz, src, dst, vol)
    if len(src) == 0:
        e = np.zeros(0, dtype=np.int64)
        return src, dst, lg, same_layer, e, e, np.zeros((0, len(FEATURES)))
    e_in, e_out = HU.candidate_triplets(src, dst, lg, (~same_layer) & (lg["L"] > 1e-3))
    return src, dst, lg, same_layer, e_in, e_out, triplet_matrix(lg, e_in, e_out)


def triplet_matrix(lg: dict, e_in: np.ndarray, e_out: np.ndarray) -> np.ndarray:
    L, cot, s_cot, s_t, ang, dz = lg["L"], lg["cot"], lg["s_cot"], lg["s_t"], lg["ang"], lg["dz"]
    msq = HU.MS0 ** 2 * (1 + cot[e_out] ** 2)
    pull_cot = np.abs(cot[e_out] - cot[e_in]) / np.sqrt(s_cot[e_in] ** 2 + s_cot[e_out] ** 2 + msq)
    turn = np.abs(np.remainder(ang[e_out] - ang[e_in] + np.pi, 2 * np.pi) - np.pi)
    s_turn = np.sqrt(s_t[e_in] ** 2 + s_t[e_out] ** 2 + msq)
    half = 0.5 * (L[e_in] + L[e_out])
    pt = half / (HU.MM_PER_GEV * np.maximum(turn, 1e-6))
    v_in = np.stack([L[e_in] * np.cos(ang[e_in]), L[e_in] * np.sin(ang[e_in]), dz[e_in]], 1)
    v_out = np.stack([L[e_out] * np.cos(ang[e_out]), L[e_out] * np.sin(ang[e_out]), dz[e_out]], 1)
    cos3d = (v_in * v_out).sum(1) / np.maximum(np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1), 1e-9)
    X = np.stack([pull_cot, turn, turn / s_turn, np.log10(np.clip(pt, 1e-2, 1e4)), L[e_in], L[e_out],
                  np.log(np.maximum(L[e_out], 1e-3) / np.maximum(L[e_in], 1e-3)),
                  np.minimum(np.abs(cot[e_in]), HU.COT_CLIP), np.minimum(np.abs(cot[e_out]), HU.COT_CLIP),
                  s_cot[e_in], s_cot[e_out], s_t[e_in], s_t[e_out], np.abs(dz[e_in]), np.abs(dz[e_out]), cos3d], 1)
    return X.astype(np.float32)


def select(src, dst, lg, same_layer, e_in, e_out, p, tau: float, k: int, n_hits: int) -> np.ndarray:
    """Kept unordered pairs (2, E'): legs of triplets with p > tau among the k most probable of both legs, plus
    same-layer pairs within SAME_LAYER_MAX whose hits are both on kept legs (as helix_union.support_filter)."""
    good = p > tau
    ei, eo, pg = e_in[good], e_out[good], p[good]
    m = (HU.rank_within(ei, -pg) < k) & (HU.rank_within(eo, -pg) < k)
    keep = np.zeros(len(src), dtype=bool)
    keep[ei[m]] = True
    keep[eo[m]] = True
    on = np.zeros(n_hits, dtype=bool)
    on[src[keep]] = True
    on[dst[keep]] = True
    sl = same_layer & ~keep & (np.hypot(lg["L"], lg["dz"]) < HU.SAME_LAYER_MAX) & on[src] & on[dst]
    keep |= sl
    return C.unordered_unique(np.stack([src[keep], dst[keep]]))


def _labels(g, src, dst, e_in, e_out) -> np.ndarray:
    n = int(g.num_nodes)
    te = C.unordered_unique(g.track_edges.numpy())
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    chain = np.isin(lo.astype(np.int64) * n + hi, te[0].astype(np.int64) * n + te[1])
    return chain[e_in] & chain[e_out]


def _event(g, emb, lid, r_wide: float):
    r, phi, z = (g[f].double().numpy() for f in ("r", "phi", "z"))
    wide = HU.radius_pairs(emb(g), r_wide)
    return wide, legs_and_triplets(r, phi, z, lid // 1000, lid, wide)


def _data_chunk(args):
    from hnl.trackbuild.mutex_ws import hit_layers
    import torch
    src_tar, names, ckpt, r_wide, row_frac = args
    torch.set_num_threads(1)
    emb = HU.Embedding(ckpt)
    eids = [int(os.path.basename(nm)[5:].split(".")[0]) for nm in names]
    L = hit_layers(eids)
    Xs, ys, ws, evs = [], [], [], []
    with tarfile.open(src_tar) as t:
        for eid, nm in zip(eids, names):
            g = HU._load_bytes(t.extractfile(nm).read())
            _, (s, d, lg, sl, ei, eo, X) = _event(g, emb, L[eid], r_wide)
            y = _labels(g, s, d, ei, eo)
            keep = y | (np.random.default_rng(eid).random(len(y)) < row_frac)
            Xs.append(X[keep])
            ys.append(y[keep])
            ws.append(np.where(y[keep], 1.0, 1.0 / row_frac).astype(np.float32))
            evs.append(np.full(int(keep.sum()), eid, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(evs)


def _scan_chunk(args):
    from hnl.trackbuild.mutex_ws import hit_layers
    import joblib
    import torch
    src_tar, names, ckpt, r_wide, model_path, taus, ks, radii = args
    torch.set_num_threads(1)
    emb = HU.Embedding(ckpt)
    model = joblib.load(model_path)
    eids = [int(os.path.basename(nm)[5:].split(".")[0]) for nm in names]
    L = hit_layers(eids)
    rows, parts = [], []
    with tarfile.open(src_tar) as t:
        for eid, nm in zip(eids, names):
            g = HU._load_bytes(t.extractfile(nm).read())
            n = int(g.num_nodes)
            pid = g.hit_particle_id.numpy()
            te = g.track_edges.numpy()
            base = C.unordered_unique(g.edge_index.numpy())
            e = emb(g)
            r_, phi_, z_ = (g[f].double().numpy() for f in ("r", "phi", "z"))
            wide = HU.radius_pairs(e, r_wide)
            s, d, lg, sl, ei, eo, X = legs_and_triplets(r_, phi_, z_, L[eid] // 1000, L[eid], wide)
            p = model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
            r, phi, z = (g[f].double().numpy() for f in ("r", "phi", "z"))
            hand, _ = HU.support_filter(r, phi, z, L[eid] // 1000, L[eid], wide)
            graphs = {"base": base, "hand": np.concatenate([base, hand], axis=1)}
            for k in ks:
                for tau in taus:
                    kept = select(s, d, lg, sl, ei, eo, p, tau, k, n)
                    graphs[f"k{k}_tau{tau:g}"] = np.concatenate([base, kept], axis=1)
            for rad in radii:                 # size control: the same embedding at a larger radius, no filter
                graphs[f"radius{rad:g}"] = np.concatenate([base, HU.radius_pairs(e, rad)], axis=1)
            for name, gr in graphs.items():
                gr = C.unordered_unique(gr)
                st, part, _ = C.evaluate_graph(n, pid, te, gr)
                rows.append({"event_id": eid, "graph": name, "n_pairs": gr.shape[1],
                             "n_true_in_graph": st["n_true_in_graph"], "n_true": st["n_true"]})
                part = part.reset_index()[["particle_id", "dm_matched", "dm_matched_samepid"]]
                part["event_id"], part["graph"] = eid, name
                parts.append(part)
    return pd.DataFrame(rows), pd.concat(parts, ignore_index=True)


def summarize(pe: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    from hnl.eval.baseline_val import populations
    from hnl.eval.particles import particle_table
    info = particle_table(np.unique(pe.event_id))
    code = pe.event_id // 100_000_000
    rows = []
    for gname, gg in pe.groupby("graph"):
        q = pp[pp.graph == gname].merge(info, on=["event_id", "particle_id"])
        Q = populations(q)
        big = q[Q["primary"] & (q.d0.abs() >= 300)]
        rows.append({"graph": gname,
                     "pairs_hnl": float(gg[np.isin(code[gg.index], (2, 4))].n_pairs.mean()),
                     "pairs_ttbar": float(gg[code[gg.index] == 1].n_pairs.mean()),
                     "true_edge_eff": float(gg.n_true_in_graph.sum() / gg.n_true.sum()),
                     "ceil_primary": float(q[Q["primary"]].dm_matched.mean()),
                     "ceil_d0_ge300": float(big.dm_matched.mean()) if len(big) else np.nan,
                     "ceil_w_muons": float(q[Q["w_muons"]].dm_matched.mean()),
                     "ceil_ttbar_targets": float(q[Q["ttbar_targets"]].dm_matched.mean())})
    return pd.DataFrame(rows).sort_values("pairs_ttbar").reset_index(drop=True)


def operating_point(pe: pd.DataFrame, summ: pd.DataFrame, ref: str = "hand") -> tuple[str, pd.DataFrame]:
    """The learned configuration at the reference's ttbar graph size, chosen on ttbar val events only (no HNL event):
    among the (k, tau) graphs with at most the reference's mean pairs per ttbar event, the highest chain-edge
    efficiency of the ttbar events (ties: fewer pairs). Returns (graph name, the ttbar table of every graph)."""
    t = pe[pe.event_id // 100_000_000 == 1].groupby("graph").agg(pairs_ttbar=("n_pairs", "mean"),
                                                                 n_true_in_graph=("n_true_in_graph", "sum"),
                                                                 n_true=("n_true", "sum"))
    t["ttbar_edge_eff"] = t.n_true_in_graph / t.n_true
    learned = t[~t.index.isin(["base", ref]) & ~t.index.str.startswith("radius")
                & (t.pairs_ttbar <= t.loc[ref, "pairs_ttbar"] + 1e-9)]
    if learned.empty:
        raise ValueError(f"no learned configuration at or below the {ref} graph size")
    best = learned.sort_values(["ttbar_edge_eff", "pairs_ttbar"], ascending=[False, True]).index[0]
    return best, t.join(summ.set_index("graph")[["ceil_primary", "ceil_d0_ge300", "ceil_w_muons", "ceil_ttbar_targets"]])


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--ml-run", required=True)
    d.add_argument("--split", default="train")
    d.add_argument("--out", required=True)
    d.add_argument("--max-events", type=int, default=400)
    d.add_argument("--row-frac", type=float, default=0.01)
    d.add_argument("--r-wide", type=float, default=0.24)
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--weighted", action="store_true", help="weight the negatives back by 1 / row_frac")
    s = sub.add_parser("scan")
    s.add_argument("--ml-run", required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-events", type=int, default=None)
    s.add_argument("--r-wide", type=float, default=0.24)
    s.add_argument("--taus", default=",".join(str(x) for x in TAUS))
    s.add_argument("--ks", default=",".join(str(x) for x in KS))
    s.add_argument("--radii", default="", help="size controls: unfiltered radius graphs of the same embedding")
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    o = sub.add_parser("choose", help="operating point at the hand-cut ttbar graph size (ttbar val only) and rule R7")
    o.add_argument("--scan", required=True, help="a scan output directory")
    a = ap.parse_args(argv)
    if a.cmd == "choose":
        pe = pd.read_parquet(os.path.join(a.scan, "per_event.parquet"))
        summ = pd.read_csv(os.path.join(a.scan, "summary.csv"))
        best, t = operating_point(pe, summ)
        d = t.loc[best, "ceil_primary"] - t.loc["hand", "ceil_primary"]
        res = {"operating_point": best, "pairs_ttbar": float(t.loc[best, "pairs_ttbar"]),
               "pairs_ttbar_hand": float(t.loc["hand", "pairs_ttbar"]),
               "ttbar_edge_eff": float(t.loc[best, "ttbar_edge_eff"]),
               "ttbar_edge_eff_hand": float(t.loc["hand", "ttbar_edge_eff"]),
               "ceil_primary": float(t.loc[best, "ceil_primary"]), "ceil_primary_hand": float(t.loc["hand", "ceil_primary"]),
               "R7_learned_replaces_hand_cuts": bool(d >= -0.002 - 1e-12)}
        with open(os.path.join(a.scan, "operating_point.json"), "w") as f:
            json.dump(res, f, indent=1)
        pd.set_option("display.width", 220)
        print(t.sort_values("pairs_ttbar").round(4).to_string())
        print(json.dumps(res))
        return
    t0 = time.time()
    if a.cmd == "data":
        if a.split != "train":
            raise SystemExit("training data only from the train split")
        src, names = _graphs(a.ml_run, a.split, a.max_events)
        ckpt = HU.best_ckpt(a.ml_run, "ml")
        chunks = [(src, names[k:k + 10], ckpt, a.r_wide, a.row_frac) for k in range(0, len(names), 10)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            res = list(ex.map(_data_chunk, chunks))
        X = np.concatenate([x[0] for x in res])
        y = np.concatenate([x[1] for x in res])
        w = np.concatenate([x[2] for x in res])
        ev = np.concatenate([x[3] for x in res])
        np.savez(a.out, X=X, y=y, w=w, event=ev, features=np.array(FEATURES))
        codes = np.unique(ev // 100_000_000).tolist()
        print(json.dumps({"events": len(names), "sample_codes": codes, "rows": int(len(y)), "positives": int(y.sum()),
                          "row_frac": a.row_frac, "elapsed_s": round(time.time() - t0, 1)}))
    elif a.cmd == "train":
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        z = np.load(a.data)
        X, y, w, ev = z["X"], z["y"], z["w"], z["event"]
        if list(z["features"]) != FEATURES:
            raise SystemExit("feature list of the data differs from the code's")
        rng = np.random.default_rng(a.seed)
        uev = np.unique(ev)
        hold = np.isin(ev, rng.choice(uev, size=max(1, len(uev) // 10), replace=False))
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                             l2_regularization=1.0, random_state=a.seed)
        clf.fit(X[~hold], y[~hold], sample_weight=w[~hold] if a.weighted else None)
        clf.hnl_features = FEATURES
        p = clf.predict_proba(X[hold])[:, 1]
        info = {"rows_train": int((~hold).sum()), "positives": int(y.sum()), "weighted": a.weighted,
                "n_iter": int(clf.n_iter_), "holdout_auc": float(roc_auc_score(y[hold], p)),
                "holdout_auc_weighted": float(roc_auc_score(y[hold], p, sample_weight=w[hold])),
                "features": FEATURES, "elapsed_s": round(time.time() - t0, 1)}
        joblib.dump(clf, a.out)
        with open(a.out + ".json", "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info))
    else:
        src, names = _graphs(a.ml_run, a.split, a.max_events)
        ckpt = HU.best_ckpt(a.ml_run, "ml")
        taus = tuple(float(x) for x in a.taus.split(","))
        ks = tuple(int(x) for x in a.ks.split(","))
        radii = tuple(float(x) for x in a.radii.split(",") if x)
        chunks = [(src, names[k:k + 10], ckpt, a.r_wide, a.model, taus, ks, radii) for k in range(0, len(names), 10)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            res = list(ex.map(_scan_chunk, chunks))
        pe = pd.concat([x[0] for x in res], ignore_index=True)
        pp = pd.concat([x[1] for x in res], ignore_index=True)
        os.makedirs(a.out, exist_ok=True)
        pe.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
        pp.to_parquet(os.path.join(a.out, "per_particle.parquet"), index=False)
        summ = summarize(pe, pp)
        summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
        pd.set_option("display.width", 220)
        print(summ.round(4).to_string(index=False))
        print(json.dumps({"events": len(names), "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
