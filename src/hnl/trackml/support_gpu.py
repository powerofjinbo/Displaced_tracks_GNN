"""Torch port of the learned triplet support (docs/trackml_plan.md section 6.2 "GPU port of the support filter", gate
G4): the candidate triplets, the 16 features, the HistGradientBoosting ensemble and the mutual top-k selection on a
torch device (CUDA on an A30; CPU for tests), chunked by middle hits exactly as hnl.trackml.events.support_select_multi,
which it must reproduce pair for pair.

Exactness by construction where possible:
  * the leg table (hnl.graphs.helix_union.legs: transcendental functions per leg) is computed with numpy on the CPU and
    moved to the device, together with the per-leg L cos(ang), L sin(ang) and |v| of the 3D opening-angle feature;
  * the candidate search uses only exactly rounded float64 operations (+, -, *, /, sqrt, min, abs) in the reference's
    order, stable sorts and searchsorted with the same sides;
  * remainder follows numpy's fmod-based definition; sums of three terms keep numpy's left-to-right order;
  * the features are cast to float32 and back to float64 as sklearn sees them; the trees are walked node by node with
    sklearn's rule (x <= threshold goes left, NaN follows missing_go_to_left); leaf values are added in tree order to
    the baseline (as sklearn's _raw_predict); p = scipy.special.expit(raw) on the CPU (the reference's function);
  * ranks: stable sort by score, then stable sort by group (= numpy.lexsort((score, group))).
The only per-triplet transcendental functions left on the device are log10 (log_pt_turn) and log (log_L_ratio): a
1-ulp difference can survive the float32 cast very rarely and then matter only within one float32 step of a split
threshold. `compare` lists every feature and selection difference (the plan allows features within 1e-9 of a split
threshold, listed).

    python -m hnl.trackml.support_gpu compare --events 1000,1001 --ml-ckpt CKPT --r-wide 0.1 --model GBDT \\
        --ops 0.005:5,0.05:3 --out DIR [--device cuda] [--wedge-deg 0]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from hnl.trackml import events as E


class TorchHGB:
    """A fitted sklearn HistGradientBoostingClassifier (binary, numerical features) as padded node tables on a device."""

    def __init__(self, model, device: str = "cpu"):
        trees = [p[0].nodes for p in model._predictors]
        if any(len(p) != 1 for p in model._predictors):
            raise ValueError("binary classifiers only")
        if any(t["is_categorical"].any() for t in trees):
            raise ValueError("categorical splits are not supported")
        T, M = len(trees), max(len(t) for t in trees)
        feat = np.zeros((T, M), dtype=np.int64)
        thr = np.zeros((T, M), dtype=np.float64)
        left = np.zeros((T, M), dtype=np.int64)
        right = np.zeros((T, M), dtype=np.int64)
        value = np.zeros((T, M), dtype=np.float64)
        leaf = np.ones((T, M), dtype=bool)
        mleft = np.zeros((T, M), dtype=bool)
        for i, t in enumerate(trees):
            n = len(t)
            feat[i, :n] = t["feature_idx"]
            thr[i, :n] = t["num_threshold"]
            left[i, :n] = t["left"]
            right[i, :n] = t["right"]
            value[i, :n] = t["value"]
            leaf[i, :n] = t["is_leaf"].astype(bool)
            mleft[i, :n] = t["missing_go_to_left"].astype(bool)
        dev = torch.device(device)
        self.feat, self.thr, self.left, self.right, self.value, self.leaf, self.mleft = (
            torch.as_tensor(x, device=dev) for x in (feat, thr, left, right, value, leaf, mleft))
        self.baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
        self.max_depth = int(max(int(t["depth"].max()) for t in trees)) + 1
        self.device = dev
        self.thresholds = {int(f): np.unique(thr[leaf == 0][feat[leaf == 0] == f]) for f in np.unique(feat[~leaf])}

    @torch.inference_mode()
    def raw(self, X64: torch.Tensor) -> torch.Tensor:
        """Raw scores (float64) of rows X64 (float64, NaN allowed), sklearn's accumulation order."""
        N = X64.shape[0]
        out = torch.full((N,), self.baseline, dtype=torch.float64, device=self.device)
        for t in range(self.feat.shape[0]):
            feat, thr, left, right = self.feat[t], self.thr[t], self.left[t], self.right[t]
            leaf, mleft, value = self.leaf[t], self.mleft[t], self.value[t]
            node = torch.zeros(N, dtype=torch.int64, device=self.device)
            active = torch.arange(N, device=self.device)
            for _ in range(self.max_depth):
                nd = node[active]
                go = ~leaf[nd]                       # rows not yet at a leaf (compacted each level)
                active, nd = active[go], nd[go]
                if active.numel() == 0:
                    break
                x = X64[active, feat[nd]]
                go_left = torch.where(torch.isnan(x), mleft[nd], x <= thr[nd])
                node[active] = torch.where(go_left, left[nd], right[nd])
            out = out + value[node]
        return out


def _remainder(a: torch.Tensor, b: float) -> torch.Tensor:
    """numpy.remainder (fmod, then + b where the signs differ) for b > 0."""
    m = torch.fmod(a, b)
    return torch.where((m != 0) & (m < 0), m + b, m)


def _rank_within(group: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """hnl.graphs.helix_union.rank_within on a device: rank within group by ascending score, ties by input order."""
    o1 = torch.argsort(score, stable=True)
    o = o1[torch.argsort(group[o1], stable=True)]
    g = group[o]
    start = torch.ones_like(g, dtype=torch.bool)
    start[1:] = g[1:] != g[:-1]
    pos = torch.arange(len(g), device=g.device)
    first = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    rk = torch.empty_like(pos)
    rk[o] = pos - first
    return rk


def _candidate_triplets(src, dst, cot, s_cot, use_idx: torch.Tensor, nsig: float = 5.0):
    """hnl.graphs.helix_union.candidate_triplets on a device (same float64 operations and order)."""
    from hnl.graphs import helix_union as HU
    idx = use_idx
    c_ = cot[idx]
    sc = s_cot[idx]
    c = torch.clamp(c_, -HU.COT_CLIP, HU.COT_CLIP)
    w = nsig * (torch.sqrt(sc ** 2 + HU.MS0 ** 2 * (1 + c_ ** 2)) + 0.05 * (1 + torch.abs(c_)))
    w = torch.minimum(w, torch.full_like(w, 2 * HU.COT_CLIP + 1.0))
    K = 2 * (HU.COT_CLIP + 2 * HU.COT_CLIP + 1.0) + 10.0
    key_in = dst[idx].to(torch.float64) * K + c
    o = torch.argsort(key_in, stable=True)
    in_sorted, key_sorted = idx[o], key_in[o]
    kc = src[idx].to(torch.float64) * K + c
    lo = torch.searchsorted(key_sorted, kc - w, right=False)
    hi = torch.searchsorted(key_sorted, kc + w, right=True)
    cnt = hi - lo
    tot = int(cnt.sum())
    e_out = torch.repeat_interleave(idx, cnt)
    base = torch.repeat_interleave(lo - (torch.cumsum(cnt, 0) - cnt), cnt)
    e_in = in_sorted[base + torch.arange(tot, device=idx.device)]
    return e_in, e_out


def _features(lgd: dict, e_in, e_out, B_mid=None) -> torch.Tensor:
    """hnl.graphs.learned_triplet_filter.triplet_matrix on a device (float32 result as the reference)."""
    from hnl.graphs import helix_union as HU
    L, cot, s_cot, s_t, ang, dz = (lgd[k] for k in ("L", "cot", "s_cot", "s_t", "ang", "dz"))
    msq = HU.MS0 ** 2 * (1 + cot[e_out] ** 2)
    pull_cot = torch.abs(cot[e_out] - cot[e_in]) / torch.sqrt(s_cot[e_in] ** 2 + s_cot[e_out] ** 2 + msq)
    turn = torch.abs(_remainder(ang[e_out] - ang[e_in] + np.pi, 2 * np.pi) - np.pi)
    s_turn = torch.sqrt(s_t[e_in] ** 2 + s_t[e_out] ** 2 + msq)
    half = 0.5 * (L[e_in] + L[e_out])
    if B_mid is None:
        pt = half / (HU.MM_PER_GEV * torch.clamp(turn, min=1e-6))
    else:
        mm = 1000.0 / (0.3 * torch.clamp(B_mid, min=1e-3))
        pt = half / (mm * torch.clamp(turn, min=1e-6))
    lc, ls, nv = lgd["Lcos"], lgd["Lsin"], lgd["norm"]
    dot = (lc[e_in] * lc[e_out] + ls[e_in] * ls[e_out]) + dz[e_in] * dz[e_out]
    cos3d = dot / torch.clamp(nv[e_in] * nv[e_out], min=1e-9)
    Lo, Li = torch.clamp(L[e_out], min=1e-3), torch.clamp(L[e_in], min=1e-3)
    X = torch.stack([pull_cot, turn, turn / s_turn, torch.log10(torch.clamp(pt, 1e-2, 1e4)), L[e_in], L[e_out],
                     torch.log(Lo / Li), torch.clamp(torch.abs(cot[e_in]), max=HU.COT_CLIP),
                     torch.clamp(torch.abs(cot[e_out]), max=HU.COT_CLIP), s_cot[e_in], s_cot[e_out], s_t[e_in],
                     s_t[e_out], torch.abs(dz[e_in]), torch.abs(dz[e_out]), cos3d], 1)
    return X.to(torch.float32)


def leg_table(ev: E.Event, cand: np.ndarray, device: str):
    """The reference leg table (numpy) plus the per-leg 3D-angle terms, moved to the device."""
    src, dst, lg, same_layer, use = E.support_legs(ev, cand)
    v = np.stack([lg["L"] * np.cos(lg["ang"]), lg["L"] * np.sin(lg["ang"]), lg["dz"]], 1)
    norm = np.linalg.norm(v, axis=1)
    dev = torch.device(device)
    lgd = {k: torch.as_tensor(lg[k], device=dev) for k in ("L", "cot", "s_cot", "s_t", "ang", "dz")}
    lgd.update({"Lcos": torch.as_tensor(np.ascontiguousarray(v[:, 0]), device=dev),
                "Lsin": torch.as_tensor(np.ascontiguousarray(v[:, 1]), device=dev),
                "norm": torch.as_tensor(norm, device=dev)})
    return src, dst, lg, same_layer, use, lgd


def support_select_torch(ev: E.Event, cand: np.ndarray, hgb: TorchHGB, ops, field_map=None,
                         max_triplets: float = 2e7, return_data: bool = False):
    """support_select_multi on a device: ({(tau, k): kept pairs}, stats[, per-chunk (e_in, e_out, X, p) on the CPU])."""
    from scipy.special import expit
    from hnl.graphs import helix_union as HU
    from hnl.graphs.ceiling import unordered_unique
    ops = [(float(t), int(k)) for t, k in ops]
    dev = hgb.device
    src, dst, lg, same_layer, use, lgd = leg_table(ev, cand, str(dev))
    n = ev.n
    keep = {op: torch.zeros(len(src), dtype=torch.bool, device=dev) for op in ops}
    st = {"n_cand": int(len(src)), "n_trip_cand": 0, "n_chunks": 0, "seconds_trees": 0.0,
          **{f"n_trip_kept_tau{t:g}_k{k}": 0 for t, k in ops}}
    data = []
    if len(src):
        src_t, dst_t = torch.as_tensor(src, device=dev), torch.as_tensor(dst, device=dev)
        use_t = torch.as_tensor(use, device=dev)
        B = None
        if field_map is not None:
            B = torch.as_tensor(field_map(np.asarray(ev.r, dtype=np.float64), np.abs(np.asarray(ev.z, dtype=np.float64))),
                                device=dev)
        for mids in E.middle_hit_chunks(n, src, dst, use, max_triplets):
            inm = torch.zeros(n, dtype=torch.bool, device=dev)
            inm[torch.as_tensor(mids, device=dev)] = True
            use_c = use_t & (inm[dst_t] | inm[src_t])
            idx = torch.nonzero(use_c).squeeze(1)
            if len(idx) == 0:
                continue
            e_in, e_out = _candidate_triplets(src_t, dst_t, lgd["cot"], lgd["s_cot"], idx)
            t = inm[dst_t[e_in]]
            e_in, e_out = e_in[t], e_out[t]
            st["n_chunks"] += 1
            if len(e_in) == 0:
                continue
            X = _features(lgd, e_in, e_out, None if B is None else B[dst_t[e_in]])
            t0 = time.time()
            raw = hgb.raw(X.to(torch.float64))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            st["seconds_trees"] += time.time() - t0
            p = torch.as_tensor(expit(raw.cpu().numpy()), device=dev)
            if return_data:
                data.append((e_in.cpu().numpy(), e_out.cpu().numpy(), X.cpu().numpy(), p.cpu().numpy()))
            st["n_trip_cand"] += int(len(e_in))
            for tau, k in ops:
                good = p > tau
                ei, eo, pg = e_in[good], e_out[good], p[good]
                m = (_rank_within(ei, -pg) < k) & (_rank_within(eo, -pg) < k)
                keep[(tau, k)][ei[m]] = True
                keep[(tau, k)][eo[m]] = True
                st[f"n_trip_kept_tau{tau:g}_k{k}"] += int(m.sum())
    near = np.hypot(lg["L"], lg["dz"]) < HU.SAME_LAYER_MAX
    out = {}
    for op in ops:
        kp = keep[op].cpu().numpy()
        on = np.zeros(n, dtype=bool)
        on[src[kp]] = True
        on[dst[kp]] = True
        sl = same_layer & ~kp & near & on[src] & on[dst]
        out[op] = unordered_unique(np.stack([src[kp | sl], dst[kp | sl]]))
    return (out, st, data) if return_data else (out, st)


def _reference_chunks(ev, cand, model, ops, max_triplets, field_map=None):
    rows = []
    out, st = E.support_select_multi(ev, cand, model, ops, field_map, max_triplets,
                                     triplet_data=lambda ei, eo, X, p, s, d: rows.append((ei, eo, X, p)))
    return out, st, rows


def compare(ev: E.Event, cand: np.ndarray, model, hgb: TorchHGB, ops, max_triplets: float = 2e7,
            field_map=None) -> dict:
    """The device path against the numpy reference: identical kept pairs per operating point, and the triplet rows
    (same triplets in the same order, float32 features, p): differing features listed with their distance to the
    nearest split threshold of that feature."""
    t0 = time.time()
    ref, st_r, rows_r = _reference_chunks(ev, cand, model, ops, max_triplets, field_map)
    t_ref = time.time() - t0
    t0 = time.time()
    got, st_g, rows_g = support_select_torch(ev, cand, hgb, ops, field_map, max_triplets, return_data=True)
    t_dev = time.time() - t0
    res = {"hits": ev.n, "candidates": int(cand.shape[1]), "triplets": st_r["n_trip_cand"], "ref_s": round(t_ref, 1),
           "device_s": round(t_dev, 1), "device_tree_s": round(st_g["seconds_trees"], 1),
           "same_triplets": len(rows_r) == len(rows_g) and all(
               np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(rows_r, rows_g))}
    feat_diff, p_diff, near = 0, 0, []
    if res["same_triplets"]:
        for (_, _, Xr, pr), (_, _, Xg, pg) in zip(rows_r, rows_g):
            d = Xr != Xg
            feat_diff += int(d.any(1).sum())
            p_diff += int((pr != pg).sum())
            for i, j in zip(*np.nonzero(d)):
                th = hgb.thresholds.get(int(j))
                dist = float(np.min(np.abs(th - float(Xr[i, j])))) if th is not None and len(th) else np.inf
                if len(near) < 50:
                    near.append({"feature": int(j), "ref": float(Xr[i, j]), "device": float(Xg[i, j]),
                                 "dist_to_threshold": dist})
    res.update({"rows_with_feature_diff": feat_diff, "p_diff": p_diff, "feature_diffs": near,
                **{f"identical_tau{t:g}_k{k}": bool(np.array_equal(ref[(t, k)], got[(t, k)])) for t, k in ref}})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--events", default="1000")
    c.add_argument("--ml-ckpt", required=True)
    c.add_argument("--r-wide", type=float, required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--ops", default="0.005:5")
    c.add_argument("--out", required=True)
    c.add_argument("--device", default=None)
    c.add_argument("--wedge-deg", type=float, default=0.0, help="restrict to a phi wedge (0: the whole event)")
    c.add_argument("--max-triplets", type=float, default=2e7)
    c.add_argument("--field-map", default=None, help="field map of the run (graph_choice.json field_map); default none")
    a = ap.parse_args(argv)
    import joblib
    from hnl.trackml.config import threads
    threads()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = joblib.load(a.model)
    hgb = TorchHGB(model, device)
    ops = [(float(x.split(":")[0]), int(x.split(":")[1])) for x in a.ops.split(",")]
    emb_f = E.Embedder(a.ml_ckpt, device)
    from hnl.trackml import config as CFG
    fmap = E.FieldMap(CFG.repo_path(a.field_map)) if a.field_map else None
    rows = []
    for ev in E.iter_events([int(x) for x in a.events.split(",")], particles=False):
        emb = emb_f(ev)
        if a.wedge_deg > 0:
            from hnl.trackml.sectors import wrap
            keep = np.abs(wrap(ev.phi.astype(np.float64) - 0.3)) < np.radians(a.wedge_deg) / 2
            ev = E.subset(ev, keep)
            emb = emb[torch.as_tensor(np.flatnonzero(keep), device=emb.device)]
        cand = E.radius_graph(emb.cpu(), a.r_wide, None)
        r = {"event": ev.event, "device": device, "r_wide": a.r_wide, **compare(ev, cand, model, hgb, ops, a.max_triplets, fmap),
             "field_map": a.field_map}
        rows.append(r)
        print(json.dumps(r), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"support_gpu_compare_{device}.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
