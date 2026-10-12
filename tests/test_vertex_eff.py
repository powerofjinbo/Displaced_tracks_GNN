"""Vertex efficiency on a toy scored graph: two daughters of a displaced vertex plus one prompt particle."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")
from torch_geometric.data import Data  # noqa: E402

from hnl.eval import vertex_eff as VE  # noqa: E402

LAYERS = np.array([70.0, 110, 170, 260, 360, 500, 660, 820])


def helix_hits(v, phi0, pt_gev, sigma, cot):
    """Hits of a helix from vertex v = (x, y, z) at the layer radii (exact circle, outward)."""
    rc = 1000.0 * pt_gev / 0.6
    c = np.array([v[0] - sigma * rc * np.sin(phi0), v[1] + sigma * rc * np.cos(phi0)])
    t = np.linspace(0, 2000.0 / rc, 200001)
    x = c[0] + sigma * rc * np.sin(phi0 + sigma * t)
    y = c[1] - sigma * rc * np.cos(phi0 + sigma * t)
    rad = np.hypot(x, y)
    tl = np.interp(LAYERS, rad, t)                     # rad increases along these outward tracks
    xl = c[0] + sigma * rc * np.sin(phi0 + sigma * tl)
    yl = c[1] - sigma * rc * np.cos(phi0 + sigma * tl)
    zl = v[2] + cot * rc * tl
    return np.hypot(xl, yl), np.arctan2(yl, xl), zl


def toy_graph(tmp_path, cross_score, vtx=None):
    v = (50 * np.cos(0.3), 50 * np.sin(0.3), 10.0)
    parts = [helix_hits(v, 0.55, 5.0, 1, 0.3), helix_hits(v, 0.05, 8.0, -1, -0.2),
             helix_hits((0.0, 0.0, -20.0), 2.0, 3.0, 1, 0.5)]
    r = np.concatenate([p[0] for p in parts])
    phi = np.concatenate([p[1] for p in parts])
    z = np.concatenate([p[2] for p in parts])
    pid = np.repeat([1, 2, 3], len(LAYERS))
    src, dst = [], []
    for k in range(3):
        o = k * len(LAYERS)
        src += list(range(o, o + len(LAYERS) - 1))
        dst += list(range(o + 1, o + len(LAYERS)))
    sc = [0.97] * len(src)
    src += [0, 9, 3]                                   # daughter 1 -> daughter 2 near the vertex, one fake to prompt
    dst += [8, 18, 20]
    sc += [cross_score, cross_score, 0.1]
    g = Data(hit_particle_id=torch.tensor(pid), r=torch.tensor(r, dtype=torch.float32),
             phi=torch.tensor(phi, dtype=torch.float32), z=torch.tensor(z, dtype=torch.float32),
             edge_index=torch.tensor([src, dst]), scores=torch.tensor(sc), event_id="200000003")
    if vtx is not None:                                # cross-daughter scores: the two daughter-daughter edges high
        g.edge_vtx_scores = torch.tensor([0.02] * (len(sc) - 3) + [vtx, vtx, 0.02])
    p = tmp_path / "event200000003.pyg"
    torch.save(g, p)
    return str(p)


def test_vertex_found_when_daughters_separate(tmp_path):
    rows = VE.evaluate_event(toy_graph(tmp_path, 0.2), [("cc", 0.5)], {"200000003": (1, 2)})
    assert rows[0]["pair_both"] and rows[0]["vertex_found"]
    assert rows[0]["n_vertices"] == 1
    assert abs(rows[0]["rv_reco"] - 50.0) < 2.0


def test_no_vertex_when_daughters_merged(tmp_path):
    rows = VE.evaluate_event(toy_graph(tmp_path, 0.95), [("cc", 0.5), ("cc", 0.96)], {"200000003": (1, 2)})
    assert not rows[0]["pair_both"] and not rows[0]["vertex_found"]     # CC joins both daughters into one track
    assert rows[1]["pair_both"] and rows[1]["vertex_found"]


def test_summary_bins():
    import pandas as pd
    df = pd.DataFrame({"event_id": [200000001, 200000002, 400000001, 100000001], "builder": 0,
                       "n_vertices": [1, 2, 0, 1], "pair_both": [True, True, False, np.nan],
                       "vertex_found": [True, False, False, np.nan], "rv_reco": [40.0, np.nan, np.nan, np.nan],
                       "rv": [45.0, 150.0, np.nan, np.nan]})
    s = VE.summarize(df).set_index("sample")
    assert s.loc["hnl10_100", "vertex_eff"] == 0.5
    assert s.loc["hnl10_100", "vertex_eff_rv[30,100)"] == 1.0
    assert s.loc["hnl10_100", "fake_vertices_per_event"] == 1.0          # (1 - 1 + 2 - 0) / 2
    assert s.loc["ttbar", "fake_vertices_per_event"] == 1.0
    assert s.loc["hnl15_200", "vertex_eff_rv_no_vertex"] == 0.0


def test_veto_builder_after_other_builders(tmp_path):
    # review finding: the per-edge vertex scores must not be overwritten by the found vertices of the previous builder
    p = toy_graph(tmp_path, 0.95, vtx=0.9)
    alone = VE.evaluate_event(p, [("vetocc", 0.5, 0.5)], {"200000003": (1, 2)})[0]
    rows = VE.evaluate_event(p, [("cc", 0.5), ("vetocc", 0.5, 0.5), ("vetocc", 0.5, 0.95)], {"200000003": (1, 2)})
    assert not rows[0]["pair_both"]                                       # CC merges the daughters
    assert {k: v for k, v in rows[1].items() if k != "builder"} == {k: v for k, v in alone.items() if k != "builder"}
    assert rows[1]["vertex_found"] and not rows[2]["pair_both"]          # veto at 0.95 keeps the cross edges
    q = toy_graph(tmp_path, 0.95)                                         # no vertex scores: the veto must refuse
    with pytest.raises(ValueError):
        VE.evaluate_event(q, [("cc", 0.5), ("vetocc", 0.5, 0.5)], {"200000003": (1, 2)})
