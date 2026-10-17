"""hnl.overlay.conditions: the C-hs truth filter of addendum H (host hard scatter and gun kept; pileup and noise removed;
particles table cut), C-full and C-1GeV through the frozen hnl.trackml.events conditions, and the refusal of closure
events. Synthetic events only."""

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import conditions as CO
from hnl.trackml import events as E
from hnl.trackml import evaluate as EVAL


def pid_of(vertex, k):
    return (int(vertex) << 52) | (int(k) << 36)


def _event(extra_closure=False) -> E.Event:
    """Hard scatter (vertex 1, a primary and a secondary), pileup (vertex 2, 37), two gun particles (1001, 1009, one at
    pT <= 1 GeV), noise; five hits per particle."""
    parts = [(pid_of(1, 1), 5.0), (pid_of(1, 2) | (1 << 24), 0.6), (pid_of(2, 1), 3.0), (pid_of(37, 4), 0.4),
             (pid_of(1001, 1), 4.0), (pid_of(1009, 2), 0.8)]
    if extra_closure:
        parts.append((pid_of(1025, 3) | (1 << 62), 2.0))
    pid, rank = [], []
    for p, _ in parts:
        pid += [p] * 5
        rank += list(range(5))
    pid += [0] * 7
    rank += [255] * 7
    n = len(pid)
    rng = np.random.default_rng(1)
    x, y, z = (rng.normal(0, 300, n).astype(np.float32) for _ in range(3))
    s = pd.DataFrame({"row": np.arange(n), "pid": pid, "rank": rank})
    p = pd.DataFrame({"particle_id": np.array([q for q, _ in parts], np.int64),
                      "pt": np.array([t for _, t in parts], np.float32), "eta": np.float32(0.3),
                      "n_hits": np.int16(5)})
    p["is_target"] = p.pt > 1.0
    return E.Event(event=50_021_117, x=x, y=y, z=z, r=np.hypot(x, y), phi=np.arctan2(y, x),
                   volume=np.full(n, 8), layer=np.full(n, 2), module=np.arange(n), pid=np.asarray(pid, np.int64),
                   rank=np.asarray(rank, np.int64), ncells=np.ones(n), charge=np.ones(n), weight=np.zeros(n),
                   true_edges=E._true_edges(s), particles=p)


def test_hs_keeps_hard_scatter_and_gun_only():
    ev = _event()
    m = CO.hs_mask(ev)
    v = CO.vertex_field(ev.pid)
    assert np.array_equal(m, (ev.pid != 0) & ((v == 1) | ((v >= 1001) & (v <= 1010))))
    sub = CO.with_condition(ev, "hs")
    assert sub.condition == "hs" and sub.n == 4 * 5                     # 2 hard-scatter + 2 gun particles
    assert set(np.unique(sub.pid)) == {pid_of(1, 1), pid_of(1, 2) | (1 << 24), pid_of(1001, 1), pid_of(1009, 2)}
    assert np.array_equal(sub.rows, np.flatnonzero(m))                  # rows map back to the full event
    # whole particles removed: every kept particle keeps its chain of true edges
    assert sub.true_edges.shape[1] == 4 * 4 and (sub.pid[sub.true_edges[0]] == sub.pid[sub.true_edges[1]]).all()
    # the particles table holds the kept particles only, so a removed pileup target is no unmatched target
    assert sorted(sub.particles.particle_id) == sorted(np.unique(sub.pid))
    c, rows = EVAL.event_counts(sub.pid, pd.factorize(sub.pid)[0], sub.particles)       # one track per particle
    assert c["n_target"] == 2 and c["m_target"] == 2 and set(rows.particle_id) == {pid_of(1, 1), pid_of(1001, 1)}


def test_full_and_1gev_are_the_frozen_conditions():
    ev = _event()
    full = CO.with_condition(ev, "full")
    assert full is ev
    one = CO.with_condition(ev, "1gev")
    ref = E.with_condition(ev, "1gev")
    assert np.array_equal(one.pid, ref.pid) and one.particles is ev.particles      # table unchanged, as frozen
    assert set(np.unique(one.pid)) == {pid_of(1, 1), pid_of(2, 1), pid_of(1001, 1)}
    with pytest.raises(ValueError):
        CO.with_condition(ev, "nonoise")
    with pytest.raises(ValueError, match="full event"):
        CO.hs_mask(one)


def test_closure_events_are_not_run_in_hs():
    ev = _event(extra_closure=True)
    with pytest.raises(ValueError, match="closure"):
        CO.with_condition(ev, "hs")
    assert CO.counts(ev)["closure"] == 5 and CO.counts(ev)["kept_hs"] is None
    c = CO.counts(_event())
    assert c == {"hits": 37, "noise": 7, "hard_scatter": 10, "gun": 10, "closure": 0, "kept_hs": 20}
