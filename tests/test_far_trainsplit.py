"""Far-displaced daughters on the HNL train split (hnl.eval.far_trainsplit): the pure helpers."""

import numpy as np
import pytest

from hnl.eval import far_trainsplit as F


def test_r_bin_edges():
    r = np.array([0.0, 199.9, 200.0, 499.9, 500.0, 699.9, 700.0, 1020.0])
    assert F.r_bin(r).tolist() == ["0-200", "0-200", "200-500", "200-500", "500-700", "500-700", ">=700", ">=700"]


def test_start_weights_sum_and_support():
    for n in (5, 6, 7, 10, 15):
        w = F.start_weights(n, 0.3, 5, 6)
        assert w.sum() == pytest.approx(1.0)
        if n < 6:
            assert w[0] == 1.0 and (w[1:] == 0).all()
        else:
            assert w[0] == pytest.approx(0.7)
            assert (w[1:n - 4] > 0).all() and (w[n - 4:] == 0).all()   # at least 5 hits are always left
        v = F.start_weights_vec(np.full(n, n), np.arange(n), 0.3, 5, 6)
        assert np.allclose(v, w)


def test_start_weights_match_the_augmentation():
    """The expected start-hit distribution equals the empirical one of hnl.models.augment.truncation_mask."""
    pytest.importorskip("torch")
    pytest.importorskip("acorn")
    from hnl.models.augment import TRUNC_KEEP, TRUNC_MIN_HITS, TRUNC_P, truncation_mask
    sizes = {1: 5, 2: 6, 3: 9, 4: 12}
    pid = np.concatenate([np.full(n, p) for p, n in sizes.items()] + [np.zeros(4, dtype=int)])
    rng0 = np.random.default_rng(1)
    R = rng0.uniform(30, 1000, len(pid))
    trials = 20000
    rng = np.random.default_rng(7)
    counts = {p: np.zeros(n) for p, n in sizes.items()}
    for _ in range(trials):
        rm = truncation_mask(pid, R, rng, TRUNC_P)
        for p, n in sizes.items():
            idx = np.flatnonzero(pid == p)
            idx = idx[np.argsort(R[idx])]
            j = int(np.argmin(rm[idx])) if (~rm[idx]).any() else n
            counts[p][j] += 1
    for p, n in sizes.items():
        w = F.start_weights(n, TRUNC_P, TRUNC_KEEP, TRUNC_MIN_HITS)
        emp = counts[p] / trials
        assert np.abs(emp - w).max() < 4 * np.sqrt(0.25 / trials) + 1e-9, (p, emp, w)


def test_fragments_order_and_threshold():
    first = np.array([3, 3, 3, 7, 7, 9, 2, 2, 2, 2])
    hits = np.array([0, 1, 2, 3, 4, 5, 6, 7])          # particle hits: 3 on track 3, 2 on 7, 1 on 9, 2 on 2
    fr = F._fragments(first, hits)
    assert fr == [(3, 3, 3), (2, 2, 4), (7, 2, 2)]       # >= 2 of its hits; largest share first, then label
