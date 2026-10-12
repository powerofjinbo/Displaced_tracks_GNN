"""hnl.trackml.events: the chunked learned-triplet-support selection equals the unchunked
hnl.graphs.learned_triplet_filter path pair for pair (ties included); the field-corrected log_pt_turn; conditions.

    pytest src/hnl/trackml/tests
"""

import numpy as np
import pandas as pd

from hnl.graphs import learned_triplet_filter as LT
from hnl.trackml import events as E

BARREL_R = [(8, 2, 32.0), (8, 4, 72.0), (8, 6, 116.0), (8, 8, 172.0), (13, 2, 260.0), (13, 4, 360.0), (13, 6, 500.0),
            (13, 8, 660.0), (17, 2, 820.0), (17, 4, 1020.0)]


def _toy_event(seed=0, n_tracks=40, n_noise=200) -> E.Event:
    """Helix-like tracks through the barrel layers (2 T, Bz > 0) plus noise hits."""
    rng = np.random.default_rng(seed)
    x, y, z, vol, lay, pid = [], [], [], [], [], []
    for t in range(n_tracks):
        pt = rng.uniform(0.5, 5.0)
        q = rng.choice([-1, 1])
        phi0 = rng.uniform(-np.pi, np.pi)
        cot = rng.uniform(-1.0, 1.0)
        R = 1000.0 / (0.3 * 2.0) * pt
        for v, l, r in BARREL_R:
            if r >= 2 * R:
                break
            a = np.arcsin(r / (2 * R))
            ph = phi0 - q * a
            x.append(r * np.cos(ph))
            y.append(r * np.sin(ph))
            z.append(cot * 2 * R * a + rng.normal(0, 0.05))
            vol.append(v)
            lay.append(l)
            pid.append(t + 1)
    for _ in range(n_noise):
        v, l, r = BARREL_R[rng.integers(len(BARREL_R))]
        ph = rng.uniform(-np.pi, np.pi)
        x.append(r * np.cos(ph))
        y.append(r * np.sin(ph))
        z.append(rng.uniform(-400, 400))
        vol.append(v)
        lay.append(l)
        pid.append(0)
    x, y, z = (np.asarray(a, dtype=np.float32) for a in (x, y, z))
    pid = np.asarray(pid, dtype=np.int64)
    n = len(x)
    r = np.hypot(x.astype(np.float64), y.astype(np.float64)).astype(np.float32)
    phi = np.arctan2(y.astype(np.float64), x.astype(np.float64)).astype(np.float32)
    rank = np.zeros(n, dtype=np.int64)
    for p in np.unique(pid[pid > 0]):
        h = np.flatnonzero(pid == p)
        rank[h[np.argsort(r[h])]] = np.arange(len(h))
    s = pd.DataFrame({"row": np.arange(n), "pid": pid, "rank": rank})
    te = E._true_edges(s)
    parts = pd.DataFrame({"particle_id": np.arange(1, n_tracks + 1), "pt": np.linspace(0.5, 5.0, n_tracks)})
    z0 = np.zeros(n)
    return E.Event(event=1, x=x, y=y, z=z, r=r, phi=phi, volume=np.asarray(vol, dtype=np.int64),
                   layer=np.asarray(lay, dtype=np.int64), module=z0.astype(np.int64), pid=pid, rank=rank,
                   ncells=z0, charge=z0, weight=z0, true_edges=te, particles=parts)


def _candidates(ev: E.Event, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree
    from hnl.graphs.ceiling import unordered_unique
    xyz = np.stack([ev.x, ev.y, ev.z / 4.0], 1).astype(np.float64)
    _, j = cKDTree(xyz).query(xyz, k=k + 1)
    a = np.repeat(np.arange(ev.n), k)
    return unordered_unique(np.stack([a, j[:, 1:].ravel()]))


class _TieModel:
    """A deterministic 'probability' from the triplet features, quantised to few levels (many exact ties)."""

    def predict_proba(self, X):
        s = 1.0 / (1.0 + X[:, 0] + 3.0 * X[:, 1])
        p = np.round(s, 1)
        return np.stack([1 - p, p], 1)


def test_chunked_support_select_equals_reference():
    for seed in range(3):
        ev = _toy_event(seed)
        cand = _candidates(ev)
        m = _TieModel()
        for tau, k in ((0.0, 2), (0.15, 1), (0.05, 5)):
            ref = E.support_select_reference(ev, cand, m, tau, k)
            for mt in (1e12, 50.0, 1.0):
                got, st = E.support_select(ev, cand, m, tau, k, max_triplets=mt, return_stats=True)
                assert np.array_equal(ref, got), (seed, tau, k, mt)
            assert st["n_chunks"] > 5                    # the smallest budget really chunks


def test_field_log_pt_turn_matches_2T():
    ev = _toy_event(1)
    cand = _candidates(ev)
    src, dst, lg, same, use = E.support_legs(ev, cand)
    from hnl.graphs import helix_union as HU
    e_in, e_out = HU.candidate_triplets(src, dst, lg, use)
    X = LT.triplet_matrix(lg, e_in, e_out)
    col = LT.FEATURES.index("log_pt_turn")
    f2 = E.field_log_pt_turn(lg, e_in, e_out, np.full(len(e_in), 2.0))
    assert np.allclose(X[:, col], f2, atol=1e-6)
    f1 = E.field_log_pt_turn(lg, e_in, e_out, np.full(len(e_in), 1.0))
    inner = (X[:, col] > -1.99) & (X[:, col] < 3.99)
    assert np.allclose(f1[inner], np.clip(X[inner, col] - np.log10(2.0), -2, 4), atol=1e-5)


def test_conditions_and_subset():
    ev = _toy_event(2)
    for cond in ("full", "nonoise"):
        sub = E.with_condition(ev, cond)
        assert sub.n == ev.n - (0 if cond == "full" else int(ev.noise.sum()))
        assert np.array_equal(ev.pid[sub.full_rows], sub.pid)
        # true edges map to the same hit pairs
        assert np.array_equal(sub.full_rows[sub.true_edges], ev.true_edges)
    one = E.with_condition(ev, "1gev")
    fast = ev.particles.particle_id[ev.particles.pt > 1.0]
    assert set(np.unique(one.pid)) == set(fast) & set(np.unique(ev.pid))


def test_graph_ceilings_counts():
    from hnl.trackml.graphs import ceilings
    from hnl.graphs.ceiling import unordered_unique
    ev = _toy_event(4)
    ev.particles["is_target"] = True
    ev.particles["is_displaced_target"] = False
    ev.particles["n_hits"] = np.bincount(ev.pid, minlength=len(ev.particles) + 1)[ev.particles.particle_id]
    full = unordered_unique(ev.true_edges)
    c = ceilings(ev, full)
    assert c["n_true"] == c["n_true_in_graph"] == full.shape[1]
    tracks = ev.particles.n_hits.to_numpy()
    assert c["m_targets"] == int((tracks >= 2).sum())         # a perfect graph connects every particle with 2+ hits
    half = full[:, ::2]
    assert ceilings(ev, half)["n_true_in_graph"] == half.shape[1]


def test_torch_support_port_equals_reference():
    """hnl.trackml.support_gpu on the CPU device: same triplets, features, p and kept pairs as the numpy path."""
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from hnl.trackml import support_gpu as SG
    ev = _toy_event(5, n_tracks=60, n_noise=300)
    cand = _candidates(ev, k=14)
    rows = []
    E.support_select_multi(ev, cand, None, [], triplet_data=lambda ei, eo, X, p, s, d: rows.append((ei, eo, X, s, d)))
    X = np.concatenate([r[2] for r in rows])
    s, d = rows[0][3], rows[0][4]
    lo, hi = np.minimum(s, d), np.maximum(s, d)
    chain = E.true_pair_mask(ev.n, np.stack([lo, hi]), ev.true_edges)
    y = np.concatenate([chain[r[0]] & chain[r[1]] for r in rows])
    model = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=15, random_state=0).fit(X, y)
    hgb = SG.TorchHGB(model, "cpu")
    raw = hgb.raw(torch.as_tensor(X.astype(np.float64))).numpy()
    assert np.array_equal(raw, model._raw_predict(X)[:, 0])
    ops = [(0.005, 5), (0.05, 2), (0.3, 1)]
    for mt in (1e12, 200.0):
        ref, _ = E.support_select_multi(ev, cand, model, ops, max_triplets=mt)
        got, st = SG.support_select_torch(ev, cand, hgb, ops, max_triplets=mt)
        for op in ops:
            assert np.array_equal(ref[op], got[op]), (mt, op)
    res = SG.compare(ev, cand, model, hgb, ops, 500.0)
    assert res["same_triplets"] and res["rows_with_feature_diff"] == 0 and res["p_diff"] == 0
