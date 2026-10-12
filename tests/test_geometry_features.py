"""Triplet enumeration and origin-free features."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hnl.models import geometry_features as G  # noqa: E402


def _helix_hits(R, x0, y0, phi_start, dphi_step, n, cot=0.3, z0=5.0):
    t = phi_start + dphi_step * np.arange(n)
    x = x0 + R * np.cos(t)
    y = y0 + R * np.sin(t)
    s = R * np.abs(t - t[0])
    z = z0 + cot * s
    return torch.tensor(np.hypot(x, y)), torch.tensor(np.arctan2(y, x)), torch.tensor(z)


def test_triplet_enumeration_counts():
    # node 1 has in-edges from 0 and 2, out-edges to 3, 4, 5 -> 6 triplets; node 3 -> 6: 1 triplet (1->3->6)
    src = torch.tensor([0, 2, 1, 1, 1, 3])
    dst = torch.tensor([1, 1, 3, 4, 5, 6])
    e_in, e_out = G.build_triplets(src, dst, 7)
    pairs = sorted(zip(e_in.tolist(), e_out.tolist()))
    assert len(pairs) == 7 and (0, 2) in pairs and (1, 4) in pairs and (2, 5) in pairs
    assert all(dst[a] == src[b] for a, b in pairs)
    e_in, e_out = G.build_triplets(src, dst, 7, max_per_node=1)
    assert len(e_in) == 2


def test_curvature_invariant_under_displacement():
    # the same circle of radius 1666.7 mm (pT 1 GeV in 2 T) traversed near the origin and far from it
    for x0, y0 in ((0.0, -1666.7), (300.0, -1500.0)):
        r, phi, z = _helix_hits(1666.7, x0, y0, math.pi / 2 + 0.01, 0.02, 4)
        src, dst = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
        e_in, e_out = G.build_triplets(src, dst, 4)
        f = G.triplet_features(r, phi, z, src, dst, e_in, e_out)
        assert torch.allclose(f[:, 0].abs(), torch.full((2,), 1e3 / 1666.7, dtype=f.dtype), rtol=1e-6)
        assert torch.allclose(f[:, 2], torch.zeros(2, dtype=f.dtype), atol=1e-9)      # helix: same s-z slope
    # rotation about z leaves every feature unchanged
    r, phi, z = _helix_hits(800.0, 100.0, 50.0, 0.3, 0.05, 4)
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    e_in, e_out = G.build_triplets(src, dst, 4)
    f1 = G.triplet_features(r, phi, z, src, dst, e_in, e_out)
    f2 = G.triplet_features(r, phi + 1.234, z + 7.0, src, dst, e_in, e_out)
    assert torch.allclose(f1, f2, atol=1e-9)


def test_select_triplets_topk():
    # edge 0 has continuations 1, 2, 3 with scores 0.3, 0.1, 0.2; k = 1 keeps (0 -> 2); edges 1-3 each have only
    # predecessor 0 and keep it as their best predecessor -> union keeps all three
    e_in = torch.tensor([0, 0, 0])
    e_out = torch.tensor([1, 2, 3])
    score = torch.tensor([0.3, 0.1, 0.2])
    assert G.select_triplets(e_in, e_out, score, 1).tolist() == [True, True, True]
    # two predecessors 4, 5 of edge 6 and two continuations: k=1 per side keeps the best of each
    e_in = torch.tensor([4, 5, 6, 6])
    e_out = torch.tensor([6, 6, 7, 8])
    score = torch.tensor([0.5, 0.1, 0.4, 0.2])
    m = G.select_triplets(e_in, e_out, score, 1)
    # by e_in: groups 4:{0}, 5:{1}, 6:{2,3}->3 ; by e_out: 6:{0,1}->1, 7:{2}, 8:{3}
    assert m.tolist() == [True, True, True, True]
    m = G.select_triplets(torch.tensor([6, 6, 6]), torch.tensor([7, 8, 9]), torch.tensor([0.3, 0.1, 0.2]), 1)
    assert m.tolist() == [True, True, True]         # each continuation is the only one of its e_out group
    m = G.select_triplets(torch.tensor([1, 2, 3]), torch.tensor([9, 9, 9]), torch.tensor([0.3, 0.1, 0.2]), 2)
    assert m.tolist() == [True, True, True]


def test_select_triplets_drops_non_best():
    # (0 -> 2) is neither edge 0's best continuation (0 -> 1 is) nor edge 2's best predecessor (9 -> 2 is)
    m = G.select_triplets(torch.tensor([0, 0, 9]), torch.tensor([1, 2, 2]), torch.tensor([0.1, 0.5, 0.05]), 1)
    assert m.tolist() == [True, False, True]
