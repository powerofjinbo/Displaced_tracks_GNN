"""The acorn feature-store events load through acorn's own reader path with the expected fields (marker `data`)."""

import gzip
import io

import numpy as np
import pytest
import torch

from hnl.data import acorn_store as A
from hnl.data.convert import read_table
from hnl.data.npz import load_event

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def event():
    root = A.DERIVED + "/events/v1"
    s, i = "hnl10_100", 6
    hits = read_table(root, "hits", s, events=[i])
    edges = read_table(root, "edges", s, events=[i])
    parts = read_table(root, "particles", s, events=[i], columns=["event", "particle_id", "pt", "eta", "n_hits"])
    return s, i, A.acorn_event(hits, edges, parts, A.event_id(s, i))


def test_ids():
    assert A.event_id("hnl15_200", 42) == "400000042"
    assert A.parse_event_id("400000042") == ("hnl15_200", 42)


def test_fields_and_truth(event):
    s, i, g = event
    ev = load_event(s, i)
    n = len(ev["pid"])
    assert g.num_nodes == n if hasattr(g, "num_nodes") else True
    assert torch.equal(g.hit_id, torch.arange(n))
    assert torch.allclose(torch.hypot(g.x, g.hit_y), g.r, atol=1e-3)
    assert torch.equal(g.hit_particle_id, torch.from_numpy(ev["pid"]))
    te = g.track_edges.numpy()
    # same unordered pairs as the npz chain, oriented outward in r^2 + z^2
    a = np.sort(te, axis=0)
    b = np.sort(ev["layerless_true_edges"], axis=0)
    assert set(map(tuple, a.T)) == set(map(tuple, b.T))
    R = ev["x"][:, 0].astype(np.float64) ** 2 + ev["x"][:, 2].astype(np.float64) ** 2
    assert (R[te[0]] <= R[te[1]]).all()
    assert g.particle_id.shape[0] == te.shape[1] and torch.equal(g.particle_id, g.hit_particle_id[g.track_edges[0]])
    assert g.event_id == "200000006" and g.config == []
    assert not hasattr(g, "y") or g.y is None


def test_acorn_prefixing_round_trip(event):
    """acorn's own loader path (prefix map) yields hit_*, track_* fields of the right lengths and keeps hit_y."""
    from acorn.utils.loading_utils import add_variable_name_prefix_in_pyg
    _, _, g = event
    data = A._serialise(g)
    h = torch.load(io.BytesIO(gzip.decompress(data)), weights_only=False)
    h = add_variable_name_prefix_in_pyg(h)
    n, t = h.hit_id.shape[0], h.track_edges.shape[1]
    for k in ("hit_x", "hit_r", "hit_phi", "hit_z", "hit_eta", "hit_y", "hit_particle_id"):
        assert h[k].shape[0] == n, k
    for k in ("track_particle_id", "track_particle_pt", "track_particle_eta", "track_particle_nhits"):
        assert h[k].shape[0] == t, k
    assert "edge_y" not in h
