"""Loss decomposition of hnl.eval.headroom on a synthetic particle table."""

import numpy as np
import pandas as pd

from hnl.eval import headroom as H


def test_binned_losses_add_up():
    rng = np.random.default_rng(0)
    n = 400
    ceiling = rng.random(n) < 0.95
    ta = ceiling & (rng.random(n) < 0.95)
    actual = ta & (rng.random(n) < 0.9)
    t = pd.DataFrame({"event_id": 200000000 + np.arange(n) // 2, "particle_id": np.arange(n), "ceiling": ceiling,
                      "ta": ta, "actual": actual, "ceiling_sp": ceiling, "ta_sp": ta, "connected": ta, "d0": rng.uniform(-200, 200, n), "rv": rng.uniform(0, 400, n),
                      "is_target": True, "category": "hnl", "r_first": rng.uniform(30, 900, n),
                      "n_pixel_hits": rng.integers(0, 9, n), "pt": rng.uniform(1, 60, n),
                      "pair_angle": rng.uniform(0, 1, n)})
    mask = np.ones(n, dtype=bool)
    for var in H.BINS:
        r = H.binned_losses(t, mask, var)
        assert r.n.sum() == n
        tot = r.loss_graph + r.loss_classifier + r.loss_trackbuilding
        assert np.allclose(tot, 1 - r.actual)
        assert (r.loss_classifier >= -1e-12).all() and (r.loss_trackbuilding >= -1e-12).all()
        assert np.allclose(r.loss_graph_sp + r.loss_classifier_sp + r.loss_trackbuilding_sp, 1 - r.actual)
        assert np.allclose(r.loss_graph_conn + r.loss_classifier_conn + r.loss_trackbuilding_conn, 1 - r.actual)
    r = H.binned_losses(t.assign(_all=0.5), mask, "_all")
    assert len(r) == 1 and r.n.iloc[0] == n


def test_rv_bins_include_no_vertex():
    n = 10
    t = pd.DataFrame({"event_id": 200000000 + np.arange(n), "particle_id": np.arange(n), "ceiling": True, "ta": True,
                      "actual": True, "d0": 20.0, "rv": [5, 50, 150, np.nan, np.nan, 250, 350, 20, np.nan, 120],
                      "is_target": True, "category": "hnl", "r_first": 100.0, "n_pixel_hits": 2, "pt": 5.0,
                      "pair_angle": 0.1})
    r = H.binned_losses(t, np.ones(n, dtype=bool), "rv")
    assert r.n.sum() == n and r.loc["no vertex", "n"] == 3
