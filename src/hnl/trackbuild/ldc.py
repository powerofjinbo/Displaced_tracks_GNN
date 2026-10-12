"""Learned dendrogram cut (LDC): node features, truth labels, the node-quality model q(v) and the builder.

Candidate nodes are the internal nodes of the single-linkage dendrogram (hnl.trackbuild.dendrogram) with 3-40 hits.
Features use only the scores' relative structure and the geometry of the node's own hits (no absolute position, no
distance to the beam line), so the same rule applies to prompt and displaced tracks:
  size, hits per detector region (pixel / short strips / long strips), birth level, death level, persistence, the
  birth level's rank within the root component, the children's sizes, and a robust helix fit of the node's hits
  (hnl.vertex.pairing.fit_tracks: transverse and longitudinal rms normalised by the hits' resolutions, outliers, |kappa|,
  cot theta, span in R) with the merge test: did this union join two different helices (normalised chi2 increase over
  the two children)?
Labels (training events only): the node's hits DM-match a particle (as a track would). Model: sklearn
HistGradientBoostingClassifier. Builder: w(v) = q(v) - theta (the expected number of correct tracks minus theta per
track), exact DP cut, ties to the larger node; theta is the working point. (Until commit 7ed4053, 2026-10-03 21:55
UTC, the weight was (q - theta) * size: a DM-positive union of two particles' tracks, 10 + 6 hits, then ties with its
two children and the DP keeps the union, losing the smaller particle; truth-oracle primary efficiency 0.735 instead
of 0.973.)

    python -m hnl.trackbuild.ldc data --graphs DIR --out SHARD.npz [--max-events N]     (scored train graphs)
    python -m hnl.trackbuild.ldc train --data 'DIR/*.npz' --out MODEL.joblib
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from hnl.geometry.resolution import RES
from hnl.trackbuild import dendrogram as D
from hnl.trackbuild.merge_fragments import _volume_from_radius
from hnl.vertex.pairing import fit_tracks

MIN_SIZE, MAX_SIZE = 3, 40
# Normalisation of the node fit's rms: the fine module axis s_u for the transverse rms; for the z rms the coarse axis
# s_v in the barrel (z) and |cot theta| s_v in the endcaps (z is the disk's; the radial r error moves the arc length,
# radial crossing assumed). Module-frame sigmas of hnl.geometry.resolution (ttbar train); volumes from the radius as
# in hnl.trackbuild.merge_fragments (endcaps 7, 12, 16). Until 2026-10-03 the tables were isotropic (HNL W-muon
# residuals) and used the endcaps' coarse r sigma as a z sigma.
SIG_T = {v: RES[v][0] for v in RES}
SIG_V = {v: RES[v][1] for v in RES}
FEATURES = ["size", "n_pix", "n_short", "n_long", "birth", "death", "persistence", "birth_rank", "child_min",
            "child_max", "fit_ok", "rms_n", "zrms_n", "n_out", "abs_kappa", "cot", "span_R", "merge_chi2"]


def _region_counts(vol: np.ndarray) -> np.ndarray:
    sub = np.where(np.isin(vol, (7, 8, 9)), 0, np.where(np.isin(vol, (12, 13, 14)), 1, 2))
    return np.eye(3, dtype=np.int64)[sub]


def candidates(tree: D.Tree) -> np.ndarray:
    """Internal-node indices k (node id n + k) with MIN_SIZE <= size <= MAX_SIZE."""
    s = tree.size[tree.n:]
    return np.flatnonzero((s >= MIN_SIZE) & (s <= MAX_SIZE))


def _fit(tree: D.Tree, ks: np.ndarray, r, phi, z, vol):
    """Helix fit of every listed node; returns (rms_n, zrms_n, n_out, abs_kappa [1/m], cot, chi2) arrays (NaN if the
    node could not be fitted)."""
    out = {k: np.full(len(ks), np.nan) for k in ("rms_n", "zrms_n", "n_out", "abs_kappa", "cot", "chi2")}
    if len(ks) == 0:
        return out
    rows = [tree.hits(tree.n + k) for k in ks]
    lab = np.repeat(np.arange(len(ks)), [len(x) for x in rows])
    rows = np.concatenate(rows)
    f = fit_tracks(r[rows], phi[rows], z[rows], vol[rows], lab, min_hits=MIN_SIZE)
    if len(f["track"]) == 0:
        return out
    t = f["track"]
    cot = np.zeros(len(ks))
    cot[t] = f["cot"]
    st = np.vectorize(SIG_T.get)(vol[rows]).astype(float)
    sv = np.vectorize(SIG_V.get)(vol[rows]).astype(float)
    sz = np.where(np.isin(vol[rows], (8, 13, 17)), sv, np.maximum(np.abs(cot[lab]), 0.1) * sv)
    mt = np.bincount(lab, weights=st, minlength=len(ks)) / np.maximum(np.bincount(lab, minlength=len(ks)), 1)
    mz = np.bincount(lab, weights=sz, minlength=len(ks)) / np.maximum(np.bincount(lab, minlength=len(ks)), 1)
    out["rms_n"][t] = f["rms"] / mt[t]
    out["zrms_n"][t] = f["zrms"] / mz[t]
    out["n_out"][t] = f["n_out"]
    out["abs_kappa"][t] = 1000.0 / f["R"]
    out["cot"][t] = f["cot"]
    out["chi2"][t] = f["n_hits"] * (out["rms_n"][t] ** 2 + out["zrms_n"][t] ** 2)
    return out


def node_features(tree: D.Tree, ks: np.ndarray, r, phi, z) -> np.ndarray:
    """(len(ks), len(FEATURES)) features of the candidate nodes."""
    n = tree.n
    vol = _volume_from_radius(r, z)
    reg = _region_counts(vol)
    death = tree.death()
    # rank of the birth level among the internal nodes of the same root component (1 = the highest)
    root = np.empty(n + tree.m, dtype=np.int64)
    for v in range(n + tree.m - 1, -1, -1):
        root[v] = v if tree.parent[v] < 0 else root[tree.parent[v]]
    roots_int = root[n:]
    order = np.lexsort((-tree.level, roots_int))
    rank = np.empty(tree.m)
    first = np.r_[True, roots_int[order][1:] != roots_int[order][:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(tree.m), 0))
    cnt = np.bincount(roots_int, minlength=n + tree.m)[roots_int[order]]
    rank[order] = (np.arange(tree.m) - grp_start + 1) / cnt
    sizes = tree.size
    left, right = tree.left[ks], tree.right[ks]
    fit = _fit(tree, ks, r, phi, z, vol)
    # children fits for the merge test (children with >= MIN_SIZE hits are internal nodes)
    kids = np.unique(np.r_[left, right])
    kids = kids[(kids >= n) & (sizes[kids] >= MIN_SIZE)] - n
    kfit = _fit(tree, kids, r, phi, z, vol)
    kchi = dict(zip(kids.tolist(), np.nan_to_num(kfit["chi2"]).tolist()))
    chi_children = np.array([kchi.get(int(x) - n, 0.0) + kchi.get(int(y) - n, 0.0) for x, y in zip(left, right)])
    ndf = np.maximum(sizes[n + ks] - 5, 1)
    span = np.array([np.ptp(np.hypot(r, z)[tree.hits(n + k)]) for k in ks])
    reg_counts = np.array([reg[tree.hits(n + k)].sum(0) for k in ks]).reshape(len(ks), 3)
    feats = {
        "size": sizes[n + ks], "n_pix": reg_counts[:, 0], "n_short": reg_counts[:, 1], "n_long": reg_counts[:, 2],
        "birth": tree.level[ks], "death": death[ks], "persistence": tree.level[ks] - death[ks], "birth_rank": rank[ks],
        "child_min": np.minimum(sizes[left], sizes[right]), "child_max": np.maximum(sizes[left], sizes[right]),
        "fit_ok": np.isfinite(fit["rms_n"]).astype(float), "rms_n": fit["rms_n"], "zrms_n": fit["zrms_n"],
        "n_out": fit["n_out"], "abs_kappa": fit["abs_kappa"], "cot": fit["cot"], "span_R": span,
        "merge_chi2": (fit["chi2"] - chi_children) / ndf,
    }
    return np.stack([np.asarray(feats[f], dtype=np.float64) for f in FEATURES], axis=1)


def node_labels(tree: D.Tree, ks: np.ndarray, pid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is_track, weight): the node's hits DM-match a particle (pid != 0); weight 2 for particles with >= 5 hits."""
    pid = np.asarray(pid)
    upid, inv, n_hits = np.unique(pid, return_inverse=True, return_counts=True)
    y = np.zeros(len(ks), dtype=bool)
    w = np.ones(len(ks))
    for i, k in enumerate(ks):
        h = inv[tree.hits(tree.n + k)]
        c = np.bincount(h)
        j = int(np.argmax(c))
        if upid[j] != 0 and 2 * c[j] > len(h) and 2 * c[j] > n_hits[j]:
            y[i] = True
            w[i] = 2.0 if n_hits[j] >= 5 else 1.0
    return y, w


class LDC:
    """Fitted node-quality model + the builder (cached per event: one tree and one q per event, any theta)."""

    def __init__(self, path: str):
        import joblib
        self.model = joblib.load(path)
        self.path = path

    def scores(self, n, a, b, sc, r, phi, z):
        tree = D.single_linkage(n, a, b, sc)
        ks = candidates(tree)
        q = np.zeros(tree.m)
        if len(ks):
            q[ks] = self.model.predict_proba(node_features(tree, ks, r, phi, z))[:, 1]
        mask = np.zeros(tree.m, dtype=bool)
        mask[ks] = True
        return tree, q, mask

    @staticmethod
    def labels(tree, q, mask, theta: float) -> np.ndarray:
        return D.dp_cut(tree, weights(tree, q, mask, theta))


def weights(tree: D.Tree, q: np.ndarray, mask: np.ndarray, theta: float) -> np.ndarray:
    """w(v) = q(v) - theta for candidate nodes, -1 elsewhere (never a track)."""
    return np.where(mask, q - float(theta), -1.0)


def _event_rows(path: str):
    from hnl.eval.trackbuild_val import load_scored
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    tree = D.single_linkage(len(pid), a, b, sc)
    ks = candidates(tree)
    X = node_features(tree, ks, r, phi, z)
    y, w = node_labels(tree, ks, pid)
    return X, y, w, np.full(len(ks), int(eid), dtype=np.int64)


def _chunk(paths):
    res = [_event_rows(p) for p in paths]
    return [np.concatenate([x[i] for x in res]) for i in range(4)]


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--graphs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-events", type=int, default=None)
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="glob of npz shards")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "data":
        paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))[:a.max_events]
        chunks = [paths[k:k + 20] for k in range(0, len(paths), 20)]
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            parts = list(ex.map(_chunk, chunks))
        X, y, w, ev = (np.concatenate([p[i] for p in parts]) for i in range(4))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez_compressed(a.out, X=X, y=y, w=w, event=ev, features=np.array(FEATURES))
        print(json.dumps({"events": len(paths), "nodes": int(len(y)), "positive_frac": float(y.mean()),
                          "elapsed_s": round(time.time() - t0, 1)}))
    else:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        files = sorted(glob.glob(a.data))
        Z = [np.load(f) for f in files]
        X = np.concatenate([z["X"] for z in Z])
        y = np.concatenate([z["y"] for z in Z])
        w = np.concatenate([z["w"] for z in Z])
        ev = np.concatenate([z["event"] for z in Z])
        # hold out 10 % of the training EVENTS for early stopping (never val)
        rng = np.random.default_rng(a.seed)
        uev = np.unique(ev)
        hold = np.isin(ev, rng.choice(uev, size=max(1, len(uev) // 10), replace=False))
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=63,
                                             l2_regularization=1.0, early_stopping=False, random_state=a.seed)
        clf.fit(X[~hold], y[~hold], sample_weight=w[~hold])
        p = clf.predict_proba(X[hold])[:, 1]
        from sklearn.metrics import roc_auc_score
        info = {"files": files, "nodes_train": int((~hold).sum()), "nodes_holdout": int(hold.sum()),
                "holdout_auc": float(roc_auc_score(y[hold], p, sample_weight=w[hold])), "features": FEATURES,
                "seed": a.seed, "elapsed_s": round(time.time() - t0, 1)}
        joblib.dump(clf, a.out)
        with open(a.out + ".json", "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
