"""evaluate_gnn.evaluate_event on a hand-made scored graph: strict cut, unique unordered pairs, isolated hits."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")
from torch_geometric.data import Data  # noqa: E402

from hnl.eval import evaluate_gnn as E  # noqa: E402


def _graph(tmp_path):
    # particle 1: hits 0-5 (chain), particle 2: hits 6-10 (chain), hit 11 noise (pid 0)
    pid = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0])
    track = torch.tensor([[0, 1, 2, 3, 4, 6, 7, 8, 9], [1, 2, 3, 4, 5, 7, 8, 9, 10]])
    # scored edges: particle 1 chain (one pair given in both directions with different scores), particle 2 chain with
    # one weak link (8-9 at 0.3), a fake edge 5-6 at 0.6, a fake edge 10-11 at 0.2
    ei = torch.tensor([[0, 1, 2, 3, 4, 2, 6, 7, 8, 9, 5, 10],
                       [1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 6, 11]])
    sc = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.9, 0.9, 0.3, 0.9, 0.6, 0.2])
    g = Data(hit_particle_id=pid, track_edges=track, edge_index=ei, scores=sc, event_id="200000007")
    path = tmp_path / "event200000007.pyg"
    torch.save(g, path)
    return str(path)


def test_cuts_pairs_and_matching(tmp_path):
    path = _graph(tmp_path)
    parts, ev, ceil = E.evaluate_event(path, [0.1, 0.5, 0.7], {"200000007": (1, 2)})
    # ceiling: every true edge of the graph kept -> both particles complete; no fake edges -> dm_nofp >= dm_matched
    assert ceil.set_index("particle_id").dm_ceiling.loc[[1, 2]].all()
    assert (parts.dm_nofp >= parts.dm_matched).all()
    nofp = parts[parts.cut == 0.5].set_index("particle_id").dm_nofp
    assert bool(nofp.loc[1]) and bool(nofp.loc[2])           # without the fake 5-6 edge p2's 6-7-8 is a match
    # the same-particle variant also keeps the 2-1 / 1-2 duplicate and drops only the cross-particle 5-6 edge
    assert (parts.dm_nocross >= parts.dm_matched).all()
    assert parts[parts.cut == 0.5].set_index("particle_id").dm_nocross.loc[[1, 2]].all()
    ev = ev.set_index("cut")
    assert (ev.n_edges == 11).all() and (ev.n_dup_pairs == 1).all()      # 1-2 / 2-1 counted once
    # cut 0.5: kept pairs = 4 (p1) + 3 (p2 without 8-9) + 5-6 fake = 9 -> tp 8? p1 chain 5 pairs (0-1..4-5), all 0.9
    assert ev.loc[0.5, "tp"] == 8 and ev.loc[0.5, "fp"] == 1 and ev.loc[0.5, "fn"] == 1
    # at 0.5 the fake edge 5-6 merges p1 (6 hits) with 6-7-8 (3 hits of p2): 9 hits, p1 majority 6/9 -> p1 matched
    # (6/9 > 0.5 and 6/6 of p1); p2 keeps 9-10 (2 hits, not a track >= 5): p2 not matched
    p = parts[parts.cut == 0.5].set_index("particle_id").dm_matched
    assert bool(p.loc[1]) and not bool(p.loc[2])
    assert ev.loc[0.5, "pair_merged"] and not ev.loc[0.5, "pair_both"]
    # cut 0.7: p1 (6 hits); p2 split into 6-7-8 and 9-10. DM has no track-size cut (definitions §5): 3 of p2's 5
    # hits in a pure 3-hit track is a match. No merge.
    p = parts[parts.cut == 0.7].set_index("particle_id").dm_matched
    assert bool(p.loc[1]) and bool(p.loc[2]) and ev.loc[0.7, "pair_both"] and not ev.loc[0.7, "pair_merged"]
    assert ev.loc[0.7, "n_tracks_ge5"] == 1 and ev.loc[0.7, "n_fake"] == 0
    # cut 0.1: everything connected incl. noise hit 11 -> one 12-hit track: p1 6/12 is not a strict majority
    p = parts[parts.cut == 0.1].set_index("particle_id").dm_matched
    assert not bool(p.loc[1]) and not bool(p.loc[2]) and ev.loc[0.1, "n_tracks_ge5"] == 1
    assert ev.loc[0.1, "fp"] == 2 and ev.loc[0.1, "fn"] == 0


def test_cut_specs_compare_identically(tmp_path):
    """A score exactly equal to float32(cut) must be treated the same with the range spec (val) and the comma list
    (test): both compare in float64."""
    pid = torch.tensor([1, 1, 1, 2, 2, 2])
    ei = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
    sc = torch.tensor([0.9, 0.61, 0.9, 0.9], dtype=torch.float32)          # float32(0.61) > 0.61 in float64
    g = Data(hit_particle_id=pid, track_edges=ei.clone(), edge_index=ei, scores=sc, event_id="200000011")
    path = tmp_path / "event200000011.pyg"
    torch.save(g, path)
    grid = [c for c in E.parse_cuts("0.01:0.99:0.01") if abs(c - 0.61) < 1e-9]
    listed = E.parse_cuts("0.6100")
    assert type(grid[0]) is type(listed[0]) and grid[0] == listed[0]
    _, ev_a, _ = E.evaluate_event(str(path), grid, {})
    _, ev_b, _ = E.evaluate_event(str(path), listed, {})
    assert ev_a.drop(columns="cut").equals(ev_b.drop(columns="cut"))
    assert int(ev_a.tp.iloc[0]) == 4 and int(ev_b.tp.iloc[0]) == 4     # the tied edge is kept (float64 comparison)


def test_muon_edge_counts_and_clean_variant(tmp_path):
    path = _graph(tmp_path)                      # particle 1: hits 0-5, particle 2: hits 6-10, hit 11 noise
    # particle 1 is a muon target with an outlier hit at row 3 (chain edges 2-3 and 3-4 touch it)
    muons = {"200000007": (np.array([1]), np.array([3]))}
    _, ev, _ = E.evaluate_event(path, [0.5], {}, None, muons)
    r = ev.iloc[0]
    assert r.n_true_mu == 5 and r.n_true_mu_clean == 3
    assert r.n_true_in_graph_mu == 5 and r.n_true_in_graph_mu_clean == 3
    assert r.tp_mu == 5 and r.tp_mu_clean == 3
