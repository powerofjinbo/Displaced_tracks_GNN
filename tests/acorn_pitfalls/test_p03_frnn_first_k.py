"""Pitfall p03 (brief §4.5): FRNN keeps the first K neighbours it finds within r, not the K nearest.

acorn's build_edges (acorn/stages/graph_construction/models/utils.py:61-107; the file is identical at the pin
99d15341 and in the patched tree, and no patch in patches/acorn touches it) calls
frnn.frnn_grid_points(K=k_max, r=r_max, return_sorted=True) and keeps every returned index >= 0. In the pinned FRNN
(murnanedaniel/FRNN c3d7bc3f):

* MinK::add (frnn/csrc/utils/mink.cuh:55-61) only appends while fewer than K entries are stored; later candidates are
  ignored however close they are (FRNN commit 0c610c8 "Remove KNN complexity" deleted the replace-the-farthest branch).
* Candidates are visited cell by cell, x-major over the first three coordinates (find_nbrs.cu:383-395), i.e. in
  increasing linear cell id, which is the order of FRNN's counting-sorted database; inside a cell the order is the
  atomicAdd insertion order (insert_points.cu:107-109, counting_sort.cu:90-101).
* The cell size is r / 2, or bbox_extent / 64 (3-D grid) if that is larger (frnn.py:72-88), and the bounding box is
  that of the database's first three coordinates: the traversal order, hence the kept set, depends on r and on the
  bounding box.
* return_sorted is ignored (frnn.py:181-187). The kept entries are sorted by distance only by the V2 kernel
  (2 <= D <= 8 and K <= 32) and by the 2-D and D > 32 kernels; acorn's usual emb_dim = 12 goes to
  FindNbrsNDKernelV1, whose sort was deleted (find_nbrs.cu:331-410), so rows come back in traversal order.

torch_cluster.radius (acorn backend "PYG", which is also acorn's silent fallback when FRNN is missing or there is no
GPU) keeps the first max_num_neighbors in database index order on the GPU and in nanoflann KD-tree order on the CPU.
In every backend the query itself (distance 0) takes one of the K slots, and build_edges removes it afterwards.

GPU tests (@pytest.mark.gpu) need CUDA and the FRNN extension. The CPU tests run anywhere.
"""

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.nn import radius as pyg_radius

from acorn.stages.graph_construction.models import utils as gc_utils
from acorn.stages.graph_construction.models.utils import build_edges

try:
    import frnn
except ImportError:  # the environment installs FRNN; the GPU tests then fail in test_gpu_acorn_selects_frnn
    frnn = None

_GPU_DEVICE = "cuda"
REL_BAND = 1e-4  # pairs with |d^2 - r^2| <= REL_BAND * r^2 are boundary cases (float32 kernels vs float64 here)

needs_frnn = pytest.mark.skipif(frnn is None, reason="FRNN is not importable")


# ------------------------------------------------------------------------------------------------ toy data / truth
def _gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def uniform_cube(n, dim, seed):
    return torch.rand(n, dim, generator=_gen(seed))


def clustered_sphere(n_clusters, per_cluster, sigma, dim, seed):
    """Track-like clusters on the unit sphere, like acorn's F.normalize'd metric-learning embedding."""
    g = _gen(seed)
    centres = torch.nn.functional.normalize(torch.randn(n_clusters, dim, generator=g), dim=1)
    x = centres.repeat_interleave(per_cluster, 0) + sigma * torch.randn(n_clusters * per_cluster, dim, generator=g)
    return torch.nn.functional.normalize(x, dim=1)


def sqdist64(query, database):
    q, d = query.detach().cpu().double(), database.detach().cpu().double()
    return torch.cdist(q, d, compute_mode="donot_use_mm_for_euclid_dist") ** 2


class Neighbourhoods:
    """Exact (float64) within-r sets; strict: d^2 < r^2 (torch_cluster, nanoflann), else d^2 <= r^2 (FRNN)."""

    def __init__(self, query, database, r, strict):
        self.r2 = float(r) ** 2
        self.d2 = sqdist64(query, database)
        self.inside = self.d2 < self.r2 if strict else self.d2 <= self.r2
        self.ambiguous = ((self.d2 - self.r2).abs() <= REL_BAND * self.r2).any(1)
        self.count = self.inside.sum(1)  # includes the query itself when it is in the database

    def within(self, i):
        return torch.nonzero(self.inside[i]).flatten()  # ascending database index

    def exact_knn(self, i, k):
        """The k nearest database points within r (the query itself first, at distance 0)."""
        w = self.within(i)
        return set(w[torch.argsort(self.d2[i, w], stable=True)][:k].tolist())


def kept_sets(edge_index, n_query):
    out = [set() for _ in range(n_query)]
    for a, b in edge_index.t().tolist():
        out[a].add(b)
    return out


# ------------------------------------------------------------------- binding fraction: what our pipeline must measure
def within_r_counts(x, r):
    """Exact number of OTHER nodes within r (d < r) of each node, in float64."""
    return (sqdist64(x, x) < float(r) ** 2).sum(1) - 1


def binding_fraction(counts, k):
    """A cap K drops a within-r neighbour of a node iff the node has >= K other nodes within r: the node itself
    (distance 0) takes one of the K slots, in FRNN and in torch_cluster alike."""
    return float((counts >= k).double().mean())


def smallest_k(counts, max_fraction=0.05):
    """Smallest cap K for which fewer than max_fraction of the nodes lose a within-r neighbour."""
    k = 1
    while binding_fraction(counts, k) >= max_fraction:
        k += 1
    return k


def binding_from_degrees(edge_index, n, k):
    """Cheap check from acorn's own output: build the graph with cap K + 1; the cap K binds for a node iff its
    out-degree (self loop already removed by build_edges) is >= K."""
    return torch.bincount(edge_index[0].cpu(), minlength=n) >= k


# ------------------------------------------------------------------------------------------- CPU model of FRNN's grid
def frnn_grid_model(database, r, radius_cell_ratio=2.0):
    """CPU replica of the grid set-up in frnn/frnn.py:50-88 (pinned c3d7bc3f) for one point cloud."""
    dim = database.shape[1]
    grid_dim, grid_max_res = (3, 64) if dim >= 3 else (2, 512)
    p = database[:, :grid_dim].detach().cpu().float()
    grid_min = p.min(0).values
    grid_size = p.max(0).values - grid_min
    cell = torch.tensor(float(r), dtype=torch.float32).item() / radius_cell_ratio  # r[i].item() / radius_cell_ratio
    floor_cell = grid_size.max() / grid_max_res
    if cell < floor_cell:  # frnn.py:78-79: cell_size becomes a 0-d tensor
        delta = torch.reciprocal(floor_cell)
        res = torch.floor(grid_size / floor_cell) + 1
        cell = float(floor_cell)
    else:  # 1 / python float is stored as float32; tensor / python float runs on CUDA as tensor * float32(1 / cell)
        delta = torch.tensor(1.0 / cell, dtype=torch.float32)
        res = torch.floor(grid_size * (torch.tensor(1.0) / torch.tensor(cell, dtype=torch.float32))) + 1
    return SimpleNamespace(grid_dim=grid_dim, min=grid_min, delta=delta, res=res, total=torch.prod(res), cell=cell)


def frnn_cell_ids(points, grid):
    """Linear cell id of each point as InsertPointsNDKernel / InsertPoints2DKernel compute it (float32, (int) cast)."""
    p = points[:, : grid.grid_dim].detach().cpu().float()
    c = ((p - grid.min) * grid.delta).to(torch.int64)  # (int) truncates toward zero
    res = grid.res.to(torch.int64)
    c = torch.minimum(torch.maximum(c, torch.zeros_like(c)), res - 1)
    if grid.grid_dim == 3:
        return (c[:, 0] * res[1] + c[:, 1]) * res[2] + c[:, 2]
    return c[:, 0] * res[1] + c[:, 1]


def first_k_bounds(cells, within, k):
    """Kept set of a first-K-in-cell-order search, known up to the order inside the cell where the cap is reached:
    returns (must, upper) with must <= kept <= upper."""
    if len(within) <= k:
        s = set(within.tolist())
        return s, s
    c = cells[within]
    cut = torch.sort(c).values[k - 1]
    must = set(within[c < cut].tolist())
    return must, must | set(within[c == cut].tolist())


def frnn_grid_order(grid):
    """(rank, cell) of every database point from the grid FRNN returns: rank = position in FRNN's counting-sorted
    database (the order in which the search kernels meet within-r candidates), cell = its linear cell id."""
    order = grid.sorted_points2_idxs[0].long().cpu()
    off = grid.pc2_grid_off[0].long().cpu()
    n = len(order)
    rank = torch.empty(n, dtype=torch.long)
    rank[order] = torch.arange(n)
    cell = torch.empty(n, dtype=torch.long)
    cell[order] = torch.searchsorted(off, torch.arange(n), right=True) - 1
    return rank, cell


# --------------------------------------------------------------------------------------------------------- toy event
# q is the query (and is in the database, as in acorn); every point is within r = 1 of q. The 3 nearest points within
# r (q itself included) are q, n1, n2. The far points have the lowest x, so they fill FRNN's first grid cell.
TOY_POINTS = {
    "q": (0.0, 0.0, 0.0),
    "n1": (0.03, 0.0, 0.0),
    "n2": (0.0, 0.04, 0.0),
    "n3": (0.0, 0.0, 0.05),
    "f1": (-0.8, 0.1, 0.0),
    "f2": (-0.8, 0.0, 0.1),
    "f3": (-0.8, -0.1, 0.0),
}
TOY_R, TOY_K = 1.0, 3
NEAR_FIRST = ("q", "n1", "n2", "n3", "f1", "f2", "f3")
FAR_FIRST = ("f1", "f2", "f3", "q", "n1", "n2", "n3")


def toy_event(order):
    return torch.tensor([TOY_POINTS[n] for n in order], dtype=torch.float32), {n: i for i, n in enumerate(order)}


def toy_neighbours(order, backend, device="cpu", k=TOY_K):
    """Names of the points acorn's build_edges links to q (query = q only, as in hard-negative mining)."""
    x, pos = toy_event(order)
    x = x.to(device)
    iq = pos["q"]
    e = build_edges(
        query=x[iq : iq + 1],
        database=x,
        indices=torch.tensor([iq], device=device),
        r_max=TOY_R,
        k_max=k,
        backend=backend,
    )
    assert bool((e[0] == iq).all())
    names = {i: n for n, i in pos.items()}
    return {names[j] for j in e[1].tolist()}


# r-dependence toy (query (0, 0, 0) is not in the database; database index 0 is far outside both radii, so
# build_edges' self-loop cut, which compares query and database indices, removes nothing).
RDEP_DB = torch.tensor([[-0.6, -0.6, -5.0], [-0.30, 0.30, 0.0], [-0.05, -0.05, 0.0]])  # anchor a, far p, near s
RDEP_Q = torch.zeros(1, 3)


# ===================================================================================================== CPU tests
def test_cpu_build_edges_falls_back_to_pyg_radius(monkeypatch):
    """Without a GPU (utils.py:35-37) or without frnn (24-34) acorn silently uses torch_geometric.nn.radius,
    whatever backend the caller asked for (utils.py:71, 93-94)."""
    if not torch.cuda.is_available():
        assert gc_utils.FRNN_AVAILABLE is False
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)
    x = uniform_cube(500, 3, seed=11)
    r, k = 0.15, 6
    ref = pyg_radius(x, x, r=r, max_num_neighbors=k)  # rows: (query, database); self included
    ref = ref[:, ref[0] != ref[1]]
    assert torch.equal(build_edges(x, x, r_max=r, k_max=k, backend="FRNN"), ref)
    assert torch.equal(build_edges(x, x, r_max=r, k_max=k, backend="PYG"), ref)
    assert int((within_r_counts(x, r) >= k).sum()) > 100  # the cap binds for many nodes here


def test_cpu_pyg_cap_keeps_first_k_in_index_order_not_nearest(monkeypatch):
    """torch_cluster.radius keeps the first max_num_neighbors within r it meets: for <= 10 database points (one
    nanoflann leaf) and on the GPU that is database index order. The kept set depends on the hit order, and the
    query takes a slot."""
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)
    assert toy_neighbours(NEAR_FIRST, "PYG") == {"n1", "n2"}  # = exact 3 nearest {q, n1, n2} minus q
    assert toy_neighbours(FAR_FIRST, "PYG") == {"f1", "f2", "f3"}  # the 3 farthest; q itself did not fit
    assert toy_neighbours(FAR_FIRST, "FRNN") == {"f1", "f2", "f3"}  # backend "FRNN" is PyG radius here
    assert toy_neighbours(FAR_FIRST, "PYG", k=7) == {"n1", "n2", "n3", "f1", "f2", "f3"}  # cap not binding


def test_cpu_pyg_cap_random_points_keeps_k_within_r_not_k_nearest(monkeypatch):
    """On random points the CPU PyG backend returns exactly min(K, #within r) points within r (self included),
    which for most capped nodes are not the K nearest."""
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)
    x = uniform_cube(3000, 3, seed=1)
    r, k = 0.07, 6
    edges = pyg_radius(x, x, r=r, max_num_neighbors=k)
    assert torch.equal(build_edges(x, x, r_max=r, k_max=k, backend="PYG"), edges[:, edges[0] != edges[1]])
    kept = kept_sets(edges, len(x))
    nh = Neighbourhoods(x, x, r, strict=True)
    n_rows = n_bind = n_diff = n_index_prefix = n_self_dropped = 0
    for i in range(len(x)):
        if nh.ambiguous[i]:
            continue
        n_rows += 1
        w = nh.within(i)
        assert kept[i] <= set(w.tolist()) and len(kept[i]) == min(k, len(w))
        if len(w) <= k:
            continue
        n_bind += 1
        n_diff += kept[i] != nh.exact_knn(i, k)
        n_index_prefix += kept[i] == set(w[:k].tolist())
        n_self_dropped += i not in kept[i]
    assert n_rows >= 0.99 * len(x)
    assert n_bind / n_rows == pytest.approx(0.2253, abs=0.003)  # = P(>= 6 other nodes within r), see below
    assert n_diff / n_bind > 0.75  # 567 / 676: not the K nearest
    assert n_index_prefix / n_bind < 0.2  # 56 / 676: nor the index-order prefix (nanoflann KD-tree order)
    assert 0 < n_self_dropped < 0.1 * n_bind  # usually q itself takes a slot (K - 1 real neighbours)


def test_cpu_frnn_cpu_reference_returns_exact_k_nearest():
    """FRNN's own CPU validation kernel (bruteforce_cpu.cpp:33-48, priority queue) keeps the K nearest within r,
    sorted. Its CUDA grid kernels do not (see the GPU tests)."""
    if frnn is None:
        pytest.skip("FRNN is not importable")
    x, pos = toy_event(FAR_FIRST)
    names = {i: n for n, i in pos.items()}
    q = x[pos["q"]].view(1, 1, 3)
    idxs, dists = frnn._C.frnn_bf_cpu(q, x[None], torch.tensor([1]), torch.tensor([len(x)]), TOY_K, TOY_R)
    assert [names[j] for j in idxs[0, 0].tolist()] == ["q", "n1", "n2"]
    assert torch.allclose(dists[0, 0], torch.tensor([0.0, 0.03**2, 0.04**2]))


def test_cpu_frnn_grid_model_depends_on_r_and_bbox():
    """The documented grid set-up (frnn.py:72-88): cell = max(r / 2, extent / 64), on the first 3 coordinates of
    the database's bounding box. The GPU tests check this model against FRNN itself."""
    x = torch.tensor([[0.0, 0.0, 0.0], [0.93, 0.47, 0.26]])
    g = frnn_grid_model(x, 0.1)
    assert g.cell == pytest.approx(0.05) and g.res.tolist() == [19.0, 10.0, 6.0]
    g = frnn_grid_model(x, 0.02)  # r / 2 = 0.01 < 0.93 / 64: the bounding box sets the cell size
    assert g.cell == pytest.approx(0.93 / 64) and g.res.tolist() == [65.0, 33.0, 18.0]
    # one extra database point, far outside r of everything, moves the grid and every cell id
    g2 = frnn_grid_model(torch.cat([x, torch.tensor([[-5.0, 0.0, 0.0]])]), 0.1)
    assert not torch.equal(frnn_cell_ids(x, g2), frnn_cell_ids(x, frnn_grid_model(x, 0.1)))
    # r-dependence toy: with K = 1 the first cell reached holds the far point p at r = 0.5, the near point s at 1.4
    for r, first in ((0.5, 1), (1.4, 2)):
        g = frnn_grid_model(RDEP_DB, r)
        within = torch.tensor([1, 2])  # p and s are within both radii, a is not
        assert torch.all(sqdist64(RDEP_Q, RDEP_DB)[0, within] < r * r) and sqdist64(RDEP_Q, RDEP_DB)[0, 0] > r * r
        must, upper = first_k_bounds(frnn_cell_ids(RDEP_DB, g), within, 1)
        assert must == set() and upper == {first}


def test_cpu_binding_fraction_recipe_on_a_hit_chain(monkeypatch):
    """Ten hits on a straight line, spacing 0.1, r = 0.15: inner hits have 2 neighbours within r. With K = 2 the
    cap binds for them, and the index-order backend keeps only the backward neighbour."""
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)
    x = torch.zeros(10, 3)
    x[:, 0] = 0.1 * torch.arange(10)
    r = 0.15
    counts = within_r_counts(x, r)
    assert counts.tolist() == [1] + [2] * 8 + [1]
    assert binding_fraction(counts, 2) == pytest.approx(0.8) and binding_fraction(counts, 3) == 0.0
    assert smallest_k(counts) == 3
    e2 = build_edges(x, x, r_max=r, k_max=2)  # default backend "FRNN" -> PyG radius on this node
    assert sorted(map(tuple, e2.t().tolist())) == [(0, 1)] + [(i, i - 1) for i in range(1, 10)]
    deg2 = torch.bincount(e2[0], minlength=10)
    assert deg2.tolist() == [1] * 10  # out-degree K - 1 = 1 for binding and non-binding nodes alike
    assert torch.equal(binding_from_degrees(build_edges(x, x, r_max=r, k_max=3), 10, 2), counts >= 2)


def test_cpu_cap_binding_fraction_random_points(monkeypatch):
    """(b) Fraction of nodes for which the cap binds, on random points (backend independent), and the K + 1
    out-degree recipe through acorn's build_edges reproduces it exactly."""
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)
    x = uniform_cube(3000, 3, seed=1)  # mean 4.0 other nodes within r = 0.07
    r = 0.07
    counts = within_r_counts(x, r)
    expected = {2: 0.8903, 4: 0.5693, 6: 0.2253, 8: 0.0637, 9: 0.0243, 12: 0.0013, 16: 0.0}
    for k, frac in expected.items():
        assert binding_fraction(counts, k) == pytest.approx(frac, abs=2e-4)
        e = build_edges(x, x, r_max=r, k_max=k + 1, backend="PYG")
        assert torch.equal(binding_from_degrees(e, len(x), k), counts >= k)
    assert smallest_k(counts) == 9 == int(torch.quantile(counts.double(), 0.95, interpolation="higher")) + 1
    # acorn-like embedding: clusters of 15 on the 12-D unit sphere, r = 0.1 (acorn's r_train)
    c = within_r_counts(clustered_sphere(200, 15, 0.02, 12, seed=2), 0.1)
    assert [round(binding_fraction(c, k), 3) for k in (4, 8, 12, 16)] == [0.956, 0.760, 0.298, 0.0]
    assert smallest_k(c) == 15 == int(c.max()) + 1  # K = 14 still binds for 5.7 %: here < 5 % means no cap at all


def test_cpu_build_edges_return_indices_needs_frnn(monkeypatch):
    """build_edges(return_indices=True) returns FRNN's (dists, idxs, ind) for counting full rows, but on the PyG
    fallback those names are never bound (utils.py:93-107)."""
    monkeypatch.setattr(gc_utils, "FRNN_AVAILABLE", False)
    x = uniform_cube(50, 3, seed=3)
    with pytest.raises(UnboundLocalError):
        build_edges(x, x, r_max=0.3, k_max=5, backend="FRNN", return_indices=True)
    assert build_edges(x, x, r_max=0.3, k_max=5, backend="PYG", return_indices=True).shape[0] == 2


# ===================================================================================================== GPU tests
@pytest.mark.gpu
def test_gpu_acorn_selects_frnn():
    assert frnn is not None, "FRNN is not importable on a CUDA node: acorn silently uses PyG radius instead"
    assert gc_utils.FRNN_AVAILABLE is True


@pytest.mark.gpu
@needs_frnn
def test_gpu_frnn_toy_keeps_lowest_cells_not_nearest():
    """(a) FRNN keeps the 3 far points of the first grid cell (lowest x), whatever the hit order; the exact 3 nearest
    are q, n1, n2. torch_cluster on the GPU keeps the first 3 in index order."""
    dev = _GPU_DEVICE
    for order in (NEAR_FIRST, FAR_FIRST):
        assert toy_neighbours(order, "FRNN", device=dev) == {"f1", "f2", "f3"}
        assert toy_neighbours(order, "FRNN", device=dev, k=7) == {"n1", "n2", "n3", "f1", "f2", "f3"}
        assert toy_neighbours(order, "PYG", device=dev, k=7) == {"n1", "n2", "n3", "f1", "f2", "f3"}
    assert toy_neighbours(NEAR_FIRST, "PYG", device=dev) == {"n1", "n2"}
    assert toy_neighbours(FAR_FIRST, "PYG", device=dev) == {"f1", "f2", "f3"}


@pytest.mark.gpu
@needs_frnn
def test_gpu_frnn_kept_neighbour_depends_on_r():
    """The grid follows r: with K = 1 FRNN keeps the far point p at r = 0.5 and the near point s at r = 1.4, although
    both are within both radii. torch_cluster keeps p (lower index) at both radii."""
    dev = _GPU_DEVICE
    db, q = RDEP_DB.to(dev), RDEP_Q.to(dev)
    for r, kept in ((0.5, 1), (1.4, 2)):
        e = build_edges(q, db, r_max=r, k_max=1, backend="FRNN").cpu()
        assert e.tolist() == [[0], [kept]]
        assert build_edges(q, db, r_max=r, k_max=1, backend="PYG").cpu().tolist() == [[0], [1]]
        _, _, _, grid = frnn.frnn_grid_points(points1=q[None], points2=db[None], K=1, r=r)
        _, cell = frnn_grid_order(grid)
        assert torch.equal(cell, frnn_cell_ids(RDEP_DB, frnn_grid_model(RDEP_DB, r)))


@pytest.mark.gpu
@needs_frnn
@pytest.mark.parametrize("dim,r", [(3, 0.1), (3, 0.02), (12, 0.1), (12, 0.6), (2, 0.05), (2, 0.001)])
def test_gpu_frnn_grid_params_follow_r_and_bbox(dim, r, capsys):
    """frnn.py:72-88 as modelled by frnn_grid_model: cell = max(r / 2, extent / 64) (2-D: / 512), on the first 3
    coordinates; and the debug print at frnn.py:84 fires on every call."""
    scale = torch.tensor([1.0, 0.7, 0.4] + [1.0] * max(dim - 3, 0))[:dim]
    x = uniform_cube(2000, dim, seed=5) * scale
    xd = x.to(_GPU_DEVICE)
    _, _, _, grid = frnn.frnn_grid_points(points1=xd[None], points2=xd[None], K=4, r=r)
    params = grid.grid_params[0].cpu()
    m = frnn_grid_model(x, r)
    gd = m.grid_dim
    assert torch.equal(params[:gd], m.min)
    assert params[gd].item() == pytest.approx(m.delta.item(), rel=1e-6)
    assert params[gd + 1 : 2 * gd + 1].tolist() == m.res.tolist()
    assert params[2 * gd + 1].item() == m.total.item()
    _, cell = frnn_grid_order(grid)
    assert torch.equal(cell, frnn_cell_ids(x, m))
    assert "tensor(" in capsys.readouterr().out


def _case(name):
    if name == "cube3_k6_v2":  # D <= 8, K <= 32: V2 kernel, kept entries sorted by distance
        return uniform_cube(3000, 3, seed=1), 0.07, 6, True, 0.80
    if name == "cube3_k40_v1":  # K > 32: FindNbrsNDKernelV1<3>, unsorted
        return uniform_cube(3000, 3, seed=1), 0.15, 40, False, 0.95
    if name == "sphere12_k8_v1":  # acorn-like emb_dim 12: FindNbrsNDKernelV1<12>, unsorted
        return clustered_sphere(200, 15, 0.02, 12, seed=2), 0.1, 8, False, 0.90
    raise ValueError(name)


@pytest.mark.gpu
@needs_frnn
@pytest.mark.parametrize("case", ["cube3_k6_v2", "cube3_k40_v1", "sphere12_k8_v1"])
def test_gpu_frnn_keeps_first_k_in_grid_order_not_k_nearest(case):
    """(a) FRNN's kept set is exactly the first K within-r points in the order of its own counting-sorted database
    (cell id, then insertion order), returned unsorted by FindNbrsNDKernelV1; it is not the K nearest. The
    min_differ thresholds are the cell-level lower bounds of the CPU model (0.86, 0.99, 0.95) minus a margin."""
    x, r, k, kernel_sorts, min_differ = _case(case)
    xd = x.to(_GPU_DEVICE)
    dists, idxs, _, grid = frnn.frnn_grid_points(points1=xd[None], points2=xd[None], K=k, r=r, return_sorted=True)
    dists, idxs = dists[0].cpu(), idxs[0].cpu()
    rank, cell = frnn_grid_order(grid)
    nh = Neighbourhoods(x, x, r, strict=False)  # FRNN keeps sqdist <= r^2
    n_rows = n_bind = n_diff = n_unsorted = 0
    dx0_kept, dx0_all = [], []
    for i in range(len(x)):
        if nh.ambiguous[i]:
            continue
        n_rows += 1
        w = nh.within(i)
        row = idxs[i][idxs[i] >= 0]
        expected = w[torch.argsort(rank[w])][:k]
        assert set(row.tolist()) == set(expected.tolist()), f"row {i}"
        d = dists[i][: len(row)]
        if kernel_sorts:
            assert bool((d[1:] >= d[:-1]).all()), f"row {i} not sorted"
        else:
            assert torch.equal(row, expected), f"row {i} not in traversal order"
            n_unsorted += bool((d[1:] < d[:-1]).any())
        if len(w) > k:
            n_bind += 1
            n_diff += set(row.tolist()) != nh.exact_knn(i, k)
            dx0_kept.append(float((x[row, 0] - x[i, 0]).mean()))
            dx0_all.append(float((x[w, 0] - x[i, 0]).mean()))
    assert n_rows >= 0.99 * len(x)
    assert n_bind >= 0.2 * n_rows
    assert n_diff >= min_differ * n_bind, f"{n_diff} / {n_bind} capped rows differ from the exact K nearest"
    if not kernel_sorts:
        assert n_unsorted > 0.1 * n_bind
    # traversal is x-major from low x: the kept neighbours sit at lower first coordinate than the within-r set
    bias = sum(dx0_kept) / len(dx0_kept) - sum(dx0_all) / len(dx0_all)
    assert bias < -0.03 * r, f"mean dx0 bias {bias:.5f}"
    # return_sorted is ignored (frnn.py:181-187): same sortedness with False
    d2, i2, _, _ = frnn.frnn_grid_points(points1=xd[None], points2=xd[None], K=k, r=r, return_sorted=False)
    d2, i2 = d2[0].cpu(), i2[0].cpu()
    v = i2 >= 0
    if kernel_sorts:
        assert bool(((d2[:, 1:] >= d2[:, :-1]) | ~v[:, 1:]).all())
    else:
        assert bool(((d2[:, 1:] < d2[:, :-1]) & v[:, 1:]).any())
    # rows that are not capped are identical between the two calls; capped rows may differ (atomicAdd order)
    same = [set(idxs[i][idxs[i] >= 0].tolist()) == set(i2[i][i2[i] >= 0].tolist()) for i in range(len(x))]
    free = [i for i in range(len(x)) if not nh.ambiguous[i] and nh.count[i] <= k]
    assert all(same[i] for i in free)
    print(f"{case}: capped rows {n_bind}/{n_rows}, differ from exact K-NN {n_diff}, bias dx0 {bias:.5f}, "
          f"capped rows changed between two calls: {sum(not s for s in same)}")
    # acorn's build_edges on the same points: within the cell-level bounds (its own call may order a cell differently)
    e = build_edges(xd, xd, r_max=r, k_max=k, backend="FRNN").cpu()
    kept = kept_sets(e, len(x))
    for i in range(len(x)):
        if nh.ambiguous[i]:
            continue
        must, upper = first_k_bounds(cell, nh.within(i), k)
        assert must - {i} <= kept[i] <= upper - {i}
        assert len(kept[i]) in (min(k, int(nh.count[i])) - 1, min(k, int(nh.count[i])))


@pytest.mark.gpu
@needs_frnn
def test_gpu_frnn_cap_binding_fraction_on_random_points():
    """(b) FRNN fills a row (K valid indices) iff the node has >= K candidates within r (itself included), and the
    K + 1 out-degree recipe on acorn's edges gives the exact binding fraction."""
    x = uniform_cube(3000, 3, seed=1)
    r = 0.07
    counts = within_r_counts(x, r)
    nh = Neighbourhoods(x, x, r, strict=False)
    ok = ~nh.ambiguous
    xd = x.to(_GPU_DEVICE)
    for k in (2, 4, 6, 8, 9, 12):
        _, idxs, _, _ = frnn.frnn_grid_points(points1=xd[None], points2=xd[None], K=k, r=r)
        full = (idxs[0] >= 0).sum(1).cpu() == k
        assert torch.equal(full[ok], (nh.count >= k)[ok])
        e = build_edges(xd, xd, r_max=r, k_max=k + 1, backend="FRNN")
        assert torch.equal(binding_from_degrees(e, len(x), k)[ok], (counts >= k)[ok])
        print(f"K={k}: binding fraction {binding_fraction(counts, k):.4f}, full FRNN rows {float(full.double().mean()):.4f}")
    assert binding_fraction(counts, 8) > 0.05 > binding_fraction(counts, 9)


@pytest.mark.gpu
def test_gpu_torch_cluster_radius_keeps_first_k_in_index_order():
    """(c) acorn backend "PYG" on the GPU: torch_cluster's radius_kernel walks the database in index order and stops
    after max_num_neighbors hits with d^2 < r^2; output grouped by query, database index ascending."""
    x = uniform_cube(3000, 3, seed=1)
    r, k = 0.07, 6
    xd = x.to(_GPU_DEVICE)
    e = pyg_radius(xd, xd, r=r, max_num_neighbors=k).cpu()
    key = e[0] * len(x) + e[1]
    assert bool((key[1:] > key[:-1]).all())
    nh = Neighbourhoods(x, x, r, strict=True)
    kept = kept_sets(e, len(x))
    n_bind = n_diff = 0
    for i in range(len(x)):
        if nh.ambiguous[i]:
            continue
        w = nh.within(i)
        assert kept[i] == set(w[:k].tolist()), f"row {i}"
        if len(w) > k:
            n_bind += 1
            n_diff += kept[i] != nh.exact_knn(i, k)
    assert n_bind > 0.2 * len(x) and n_diff > 0.75 * n_bind
    assert torch.equal(build_edges(xd, xd, r_max=r, k_max=k, backend="PYG").cpu(), e[:, e[0] != e[1]])
