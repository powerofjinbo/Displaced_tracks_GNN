"""Learned, graph-free hit attachment (Stage 1 C, learned track building), v2: module-frame pulls.

After a first-pass builder (walkthrough + fragment merging, or the same-layer Mutex Watershed), every "free" hit (on
no track, or on a track with fewer than 3 hits) may be attached to a track (>= 4 hits) whose fitted helix it lies
on, whether or not the graph has an edge between them (on arm (i) seed 0 val, 61 of the 77 unassigned hits of the
particles that walkthrough + merging still loses have no graph edge to their particle's best track). Candidate pairs
(track, free hit) pass a loose geometric window; a gradient-boosted classifier trained on ttbar TRAIN events gives
p(attach). Decoding: each free hit goes to its best candidate if p > tau (the working point, builder rule).

Tracks are fitted by hnl.trackbuild.helix_fit (weighted by each hit's module-frame resolution, robust). Features of
a candidate, all functions of hit differences, of the track's own helix and of the hit's module axes, so they do not
change under translations of the production point or rotations about z:
  pull_t: distance of the hit to the fitted circle over its module-frame sigma along the circle normal
    (hnl.geometry.resolution: endcap strips measure r coarsely, so a hit of a track that does not cross the disk
    radially has a wide transverse sigma), with multiple scattering (Highland, at the nearest track hit's module,
    over the transverse gap) added in quadrature;
  pull_z: z residual from the track's s-z line over the z sigma of the hit (barrel: the coarse axis; endcap: z is
    the disk's, the r error enters through the arc length: |cot| |t.v| s_v), with scattering;
  gap: arc length to the nearest track hit; inside: within the track's angular span; d_near: 3D distance to the
  nearest track hit; same_layer: the track already has a hit on the hit's layer (a double hit or a competitor);
  the track's normalised transverse and z rms, size and |kappa|; the hit's subsystem; the number of tracks
  competing for the hit.
The GNN score of an edge between the hit and the track (max_score) is not a default feature: it carries the
classifier's pointing slope and would veto exactly the graph-free attaches; it is kept behind --with-score for an
ablation. v1 (commit 9d25ee9) used isotropic per-volume sigmas from HNL W muons and the score.

    python -m hnl.trackbuild.attach data --graphs DIR --out SHARD.npz [--first-pass mwslayer|walkmerge] [--with-score]
    python -m hnl.trackbuild.attach train --data 'DIR/*.npz' --out MODEL.joblib
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from hnl.geometry import resolution as RS
from hnl.trackbuild import helix_fit as HF
from hnl.trackbuild.merge_fragments import _volume_from_radius

FEATURES = ["pull_t", "pull_z", "gap", "inside", "d_near", "same_layer", "trk_rms_n", "trk_zrms_n", "trk_size",
            "trk_kappa", "hit_sub", "n_compete"]
SCORE_FEATURE = "max_score"
WIN_T, WIN_Z, WIN_GAP = 5.0, 60.0, 450.0          # loose candidate window [mm]
FIRST_PASS = ("mwslayer", "walkmerge", "mwslayermerge", "mwslmerge")
MERGE_TAU = 0.1                                   # learned-merging threshold of the 'mwslmerge' first pass
_MERGER = None


def _merger():
    global _MERGER
    if _MERGER is None:
        from hnl.trackbuild.learned_merge import Merger
        path = os.environ.get("HNL_MERGE_MODEL")
        if not path:
            raise ValueError("the mwslmerge first pass needs a learned-merge model ($HNL_MERGE_MODEL / --merge-model)")
        _MERGER = Merger(path)
    return _MERGER


def feature_names(with_score: bool = False) -> list[str]:
    return FEATURES + ([SCORE_FEATURE] if with_score else [])


def first_pass(n, a, b, sc, R, hits, kind: str = "walkmerge", L=None, geo=None):
    """The first-pass builder: walkthrough (0.1, 0.5) + fragment merging, the same-layer Mutex Watershed at s0 = 0.1,
    the same-layer Mutex Watershed at s0 = 0.2 + fragment merging (both need the hits' layer ids L), or the
    same-layer Mutex Watershed at s0 = 0.2 + learned fragment merging at MERGE_TAU (needs the hits' module geometry
    geo and $HNL_MERGE_MODEL)."""
    if kind == "mwslmerge":
        from hnl.trackbuild import learned_merge as LM
        if geo is None:
            raise ValueError("the mwslmerge first pass needs the hits' module geometry")
        r, phi, z = hits
        lab = LM.first_pass(n, a, b, sc, geo["layer_id"])
        pairs, p = _merger().prepare(lab, r, phi, z, geo)
        return LM.apply(lab, pairs, p, MERGE_TAU, geo["layer_id"])
    if kind == "walkmerge":
        from hnl.eval.trackbuild_val import build
        return build(("walkmerge", 0.1, 0.5, 1.0), n, a, b, sc, R, hits)
    if kind in ("mwslayer", "mwslayermerge"):
        from hnl.trackbuild.mutex_ws import mws_layer
        if L is None:
            raise ValueError(f"the {kind} first pass needs the hits' layer ids")
        if kind == "mwslayer":
            return mws_layer(n, np.asarray(a), np.asarray(b), np.asarray(sc), np.asarray(L), 0.1)
        from hnl.trackbuild.merge_fragments import merge_fragments
        r, phi, z = hits
        lab = mws_layer(n, np.asarray(a), np.asarray(b), np.asarray(sc), np.asarray(L), 0.2)
        lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
        return merge_fragments(r, phi, z, lab, max_rms=1.0)
    raise ValueError(kind)


def _geometry(geo, r, z, phi):
    """(volume, phi_c, layer_id) per hit: from hnl.geometry.resolution.hit_geometry, else from the radius (volume),
    the hit's own azimuth (phi_c) and no layer ids."""
    if geo is not None:
        return np.asarray(geo["volume"]), np.asarray(geo["phi_c"], dtype=np.float64), np.asarray(geo["layer_id"])
    return _volume_from_radius(r, z), np.asarray(phi, dtype=np.float64), None


def candidates(labels, r, phi, z, a=None, b=None, sc=None, geo=None, with_score: bool = False):
    """(rows: (track label, hit index) per candidate, X: features in feature_names(with_score) order). Free hits =
    on no track or on a track with < 3 hits; tracks = labels with >= 4 hits."""
    names = feature_names(with_score)
    labels = np.asarray(labels)
    n = len(labels)
    r, phi, z = (np.asarray(v, dtype=np.float64) for v in (r, phi, z))
    x, y = r * np.cos(phi), r * np.sin(phi)
    vol, phi_c, layer = _geometry(geo, r, z, phi)
    trk, cnt = np.unique(labels[labels >= 0], return_counts=True)
    size = dict(zip(trk.tolist(), cnt.tolist()))
    free = np.flatnonzero((labels < 0) | np.isin(labels, trk[cnt < 3]))
    big = trk[cnt >= 4]
    if len(free) == 0 or len(big) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, len(names)))
    f = HF.fit(x, y, z, np.where(np.isin(labels, big), labels, -1), vol, phi_c, min_hits=4)
    sub = np.where(np.isin(vol, RS.PIXEL), 0, np.where(np.isin(vol, (12, 13, 14)), 1, 2))
    best_edge = {}
    if with_score and a is not None:
        for u, v, s_ in zip(a, b, sc):
            for h, o in ((u, v), (v, u)):
                if labels[o] >= 0 and (labels[h] < 0 or size.get(int(labels[h]), 0) < 3):
                    key = (int(labels[o]), int(h))
                    if s_ > best_edge.get(key, 0.0):
                        best_edge[key] = float(s_)
    xf, yf, zf = x[free], y[free], z[free]
    rows, feats = [], []
    for k, t in enumerate(f["track"]):
        mk = f["rows"][k] >= 0
        idx = f["rows"][k][mk]
        dth_t = f["dtheta"][k][mk]
        xc, yc, Rc, cot = f["xc"][k], f["yc"][k], f["R"][k], f["cot"][k]
        dx, dy = xf - xc, yf - yc
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        eps_t = d - Rc
        dth = HF._wrap(np.arctan2(dy, dx) - f["theta_ref"][k])
        j = np.argmin(np.abs(dth[:, None] - dth_t[None, :]), axis=1)
        gap = Rc * np.abs(dth - dth_t[j])
        zres = zf - (f["z_ref"][k] + cot * Rc * dth)
        ok = np.flatnonzero((np.abs(eps_t) < WIN_T) & (np.abs(zres) < WIN_Z) & (gap < WIN_GAP))
        if len(ok) == 0:
            continue
        h = free[ok]
        nvec = np.stack([dx[ok] / d[ok], dy[ok] / d[ok]], -1)
        tvec = np.stack([-nvec[:, 1], nvec[:, 0]], -1)
        s_t = RS.sigma_transverse(vol[h], phi_c[h], nvec)
        s_z = RS.sigma_z(vol[h], phi_c[h], tvec, cot)
        # multiple scattering at the nearest track hit's module, over the transverse gap
        jn = idx[j[ok]]
        nj = np.stack([x[jn] - xc, y[jn] - yc], -1) / np.maximum(Rc, 1e-12)
        tj = np.stack([-nj[:, 1], nj[:, 0]], -1)
        th0 = RS.highland(RS.momentum_gev(Rc, cot), RS.crossing_x_x0(vol[jn], phi_c[jn], tj, cot))
        ms_t, ms_z = RS.ms_sigmas(th0, gap[ok], cot)
        d_near = np.sqrt(((np.stack([x[h], y[h], z[h]], 1)[:, None, :]
                           - np.stack([x[idx], y[idx], z[idx]], 1)[None, :, :]) ** 2).sum(-1)).min(1)
        same = (np.isin(layer[h], layer[idx]).astype(float) if layer is not None else np.full(len(h), np.nan))
        inside = ((dth[ok] >= dth_t.min()) & (dth[ok] <= dth_t.max())).astype(float)
        block = np.stack([np.abs(eps_t[ok]) / np.sqrt(s_t**2 + ms_t**2), np.abs(zres[ok]) / np.sqrt(s_z**2 + ms_z**2),
                          gap[ok], inside, d_near, same, np.full(len(h), f["rms_n"][k]),
                          np.full(len(h), f["zrms_n"][k]), np.full(len(h), size[int(t)], dtype=float),
                          np.full(len(h), 1000.0 / Rc), sub[h].astype(float), np.zeros(len(h))], 1)
        if with_score:
            block = np.c_[block, [best_edge.get((int(t), int(hh)), 0.0) for hh in h]]
        rows.append(np.stack([np.full(len(h), t), h], 1))
        feats.append(block)
    if not rows:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, len(names)))
    rows = np.concatenate(rows).astype(np.int64)
    X = np.concatenate(feats)
    nc = np.bincount(rows[:, 1], minlength=n)
    X[:, names.index("n_compete")] = nc[rows[:, 1]]
    return rows, X


def labels_for(rows, labels, pid):
    """1 if the hit's particle is the track's majority particle (pid != 0)."""
    pid = np.asarray(pid)
    maj = {}
    for t in np.unique(rows[:, 0]):
        p, c = np.unique(pid[labels == t], return_counts=True)
        maj[int(t)] = int(p[np.argmax(c)])
    return np.array([(pid[h] != 0) and (pid[h] == maj[int(t)]) for t, h in rows], dtype=bool)


def apply(labels, rows, p, tau: float):
    """Attach every free hit to its best candidate track if p > tau; returns new labels."""
    out = np.asarray(labels).copy()
    if len(rows) == 0:
        return out
    order = np.lexsort((-p, rows[:, 1]))
    first = np.r_[True, rows[order][1:, 1] != rows[order][:-1, 1]]
    for k in order[first]:
        if p[k] > tau:
            out[rows[k, 1]] = rows[k, 0]
    return out


class Attacher:
    """A trained attach model (v2: the feature list and first pass are stored on the model)."""

    def __init__(self, path: str):
        import joblib
        self.model = joblib.load(path)
        self.features = getattr(self.model, "hnl_features", None)
        if self.features is None:
            raise ValueError(f"{path}: an attach v1 model (isotropic sigmas + GNN score, commit 9d25ee9); retrain")
        self.first_pass = getattr(self.model, "hnl_first_pass", "walkmerge")
        self.with_score = SCORE_FEATURE in self.features

    def prepare(self, labels, r, phi, z, a, b, sc, geo=None):
        rows, X = candidates(labels, r, phi, z, a, b, sc, geo=geo, with_score=self.with_score)
        p = self.model.predict_proba(X)[:, 1] if len(rows) else np.zeros(0)
        return rows, p


def _event_rows(path: str, kind: str, with_score: bool):
    from hnl.eval.trackbuild_val import load_scored
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    geo = RS.hit_geometry([int(eid)])[int(eid)]
    assert len(geo["volume"]) == len(pid)
    lab = first_pass(len(pid), a, b, sc, R, (r, phi, z), kind, geo["layer_id"], geo)
    rows, X = candidates(lab, r, phi, z, a, b, sc, geo=geo, with_score=with_score)
    y = labels_for(rows, lab, pid) if len(rows) else np.zeros(0, dtype=bool)
    return X, y, np.full(len(y), int(eid), dtype=np.int64)


def _chunk(args):
    paths, kind, with_score = args
    res = [_event_rows(p, kind, with_score) for p in paths]
    return [np.concatenate([x[i] for x in res]) for i in range(3)]


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--graphs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-events", type=int, default=None)
    d.add_argument("--first-pass", choices=FIRST_PASS, default="mwslayer")
    d.add_argument("--with-score", action="store_true", help="ablation: add the GNN edge score (max_score)")
    d.add_argument("--merge-model", default=None, help="learned-merge model for the mwslmerge first pass")
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "data":
        paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))
        if a.max_events is not None and a.max_events < len(paths):     # a seeded random subset (all samples)
            pick = np.random.default_rng(0).choice(len(paths), a.max_events, replace=False)
            paths = [paths[k] for k in np.sort(pick)]
        if getattr(a, "merge_model", None):
            os.environ["HNL_MERGE_MODEL"] = os.path.abspath(a.merge_model)    # inherited by the workers
        with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
            parts = list(ex.map(_chunk, [(paths[k:k + 20], a.first_pass, a.with_score)
                                         for k in range(0, len(paths), 20)]))
        X, y, ev = (np.concatenate([p[i] for p in parts]) for i in range(3))
        np.savez_compressed(a.out, X=X, y=y, event=ev, features=np.array(feature_names(a.with_score)),
                            first_pass=np.array(a.first_pass))
        print(json.dumps({"events": len(paths), "candidates": int(len(y)), "positive_frac": float(y.mean()),
                          "first_pass": a.first_pass, "features": feature_names(a.with_score),
                          "elapsed_s": round(time.time() - t0, 1)}))
    else:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        Z = [np.load(f) for f in sorted(glob.glob(a.data))]
        names = [list(z["features"]) for z in Z]
        kinds = {str(z["first_pass"]) if "first_pass" in z else "walkmerge" for z in Z}
        if any(nm != names[0] for nm in names) or len(kinds) != 1:
            raise ValueError("data shards with different feature lists or first passes")
        X = np.concatenate([z["X"] for z in Z])
        y = np.concatenate([z["y"] for z in Z])
        ev = np.concatenate([z["event"] for z in Z])
        rng = np.random.default_rng(a.seed)
        uev = np.unique(ev)
        hold = np.isin(ev, rng.choice(uev, size=max(1, len(uev) // 10), replace=False))
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                             l2_regularization=1.0, random_state=a.seed)
        clf.fit(X[~hold], y[~hold])
        clf.hnl_features = [str(s) for s in names[0]]
        clf.hnl_first_pass = kinds.pop()
        info = {"candidates_train": int((~hold).sum()), "holdout_auc": float(roc_auc_score(y[hold],
                clf.predict_proba(X[hold])[:, 1])), "positive_frac": float(y.mean()), "features": clf.hnl_features,
                "first_pass": clf.hnl_first_pass, "resolution": RS.RES_PROVENANCE,
                "elapsed_s": round(time.time() - t0, 1)}
        joblib.dump(clf, a.out)
        with open(a.out + ".json", "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
