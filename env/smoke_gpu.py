"""GPU smoke test of the hnl-displaced environment (brief 4.4). Run on an A30 through Slurm:

    bash env/make_env.sh --stage smoke        (slurm/env_build_smoke.sbatch)

Checks, each of which must pass ("ALL OK" at the end):
  1. CUDA device, torch / PyG versions, sm_80 in the compiled arch list.
  2. PyG message passing forward + backward in fp32, TF32 and bf16 autocast.
  3. FRNN radius search == torch_cluster.radius on random points: identical edge SETS, with a neighbour cap K that
     no query reaches (FRNN keeps the first K neighbours in grid order, so a binding cap would make them differ).
  4. acorn's custom CUDA connected components == scipy's (same partition, not just the same count).
  5. One HNL npz event reads with the documented keys and dtypes.
Adapted from the main project's env/smoke_gpu.py (gnn-displaced repo dc603528).
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
t0 = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_geometric  # noqa: E402
from torch_geometric.nn import MessagePassing  # noqa: E402

print(f"python {sys.version.split()[0]} {sys.executable}")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "pyg", torch_geometric.__version__)
assert torch.cuda.is_available(), "no CUDA device"
dev = torch.device("cuda")
prop = torch.cuda.get_device_properties(0)
print(f"device {prop.name}, {prop.total_memory / 2**30:.1f} GiB, cc {prop.major}.{prop.minor}, "
      f"arch list {torch.cuda.get_arch_list()}")
assert "sm_80" in torch.cuda.get_arch_list(), "torch build lacks sm_80"


# ---------------------------------------------------------------------------------------------- 2. message passing
class EdgeNet(MessagePassing):
    def __init__(self, d=64):
        super().__init__(aggr="add")
        self.enc = torch.nn.Linear(3, d)
        self.msg = torch.nn.Sequential(torch.nn.Linear(2 * d, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
        self.upd = torch.nn.Sequential(torch.nn.Linear(2 * d, d), torch.nn.ReLU())
        self.out = torch.nn.Linear(2 * d, 1)

    def message(self, x_i, x_j):
        return self.msg(torch.cat([x_i, x_j], -1))

    def forward(self, x, ei):
        h = self.enc(x)
        for _ in range(3):
            h = self.upd(torch.cat([h, self.propagate(ei, x=h)], -1))
        return self.out(torch.cat([h[ei[0]], h[ei[1]]], -1)).squeeze(-1)


torch.manual_seed(0)
N, E = 100_000, 1_000_000
x = torch.randn(N, 3, device=dev)
ei = torch.randint(0, N, (2, E), device=dev)
y = (torch.rand(E, device=dev) < 0.1).float()
assert torch.cuda.is_bf16_supported(), "bf16 not supported on this device"
for precision in ("fp32", "tf32", "bf16"):
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"
    torch.manual_seed(1)
    model = EdgeNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t = time.time()
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, ei).float(), y)
        loss.backward()
        grad_norm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters()))
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    assert torch.isfinite(loss) and torch.isfinite(grad_norm) and grad_norm > 0, precision
    print(f"MP {precision}: loss {loss.item():.4f} |grad| {grad_norm.item():.3e} "
          f"{1e3 * np.median(times[1:]):.1f} ms/step, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------------------------- 3. FRNN vs torch_cluster
import frnn  # noqa: E402
from torch_cluster import radius as tc_radius  # noqa: E402

from acorn.stages.graph_construction.models.utils import build_edges  # noqa: E402


def edge_set(e):
    e = e[:, e[0] != e[1]].cpu().numpy().astype(np.int64)
    return np.unique(e[0] * (1 << 32) + e[1])


g = torch.Generator(device="cpu").manual_seed(2)
for n_pts, dim, r in ((50_000, 3, 0.08), (100_000, 12, 1.2)):
    pts = (torch.rand(n_pts, dim, generator=g) if dim == 3 else torch.randn(n_pts, dim, generator=g)).to(dev)
    K = 1024
    t = time.time()
    dists, idxs, _, _ = frnn.frnn_grid_points(points1=pts.unsqueeze(0), points2=pts.unsqueeze(0), lengths1=None,
                                              lengths2=None, K=K, r=r, grid=None, return_nn=False, return_sorted=True)
    torch.cuda.synchronize()
    t_frnn = time.time() - t
    idxs = idxs.squeeze(0)
    n_nb = (idxs >= 0).sum(1)
    q = torch.arange(n_pts, device=dev).unsqueeze(1).expand_as(idxs)
    e_frnn = torch.stack([q[idxs >= 0], idxs[idxs >= 0]])
    t = time.time()
    e_tc = tc_radius(pts, pts, r=r, max_num_neighbors=K)          # rows: (query, database)
    torch.cuda.synchronize()
    t_tc = time.time() - t
    s_frnn, s_tc = edge_set(e_frnn), edge_set(e_tc)
    max_nb = int(n_nb.max())
    print(f"radius dim={dim} n={n_pts} r={r}: FRNN {len(s_frnn)} edges ({t_frnn:.3f}s), torch_cluster {len(s_tc)} "
          f"({t_tc:.3f}s), mean nb {n_nb.float().mean():.1f}, max nb {max_nb} (K={K})")
    assert max_nb < K, "neighbour cap binds: comparison would be meaningless"
    assert len(s_frnn) > 0 and np.array_equal(s_frnn, s_tc), "FRNN and torch_cluster.radius edge sets differ"
    # acorn's wrapper, both backends
    e_af = build_edges(pts, pts, r_max=r, k_max=K, backend="FRNN")
    e_ap = build_edges(pts, pts, r_max=r, k_max=K, backend="PYG")
    assert np.array_equal(edge_set(e_af), s_frnn) and np.array_equal(edge_set(e_ap), s_tc), "acorn build_edges differs"
print("FRNN == torch_cluster.radius (edge sets identical, cap not binding); acorn build_edges agrees")

# ---------------------------------------------------------------------------------------------- 4. CC custom_cuda vs scipy
from acorn.stages.track_building.models import cc_backend_utils as ccu  # noqa: E402

for n_nodes, n_edges in ((300_000, 250_000), (2_000, 1_500), (50, 0)):
    src, dst = torch.randint(0, n_nodes, (2, n_edges), device=dev)
    labels = {}
    for backend in ("scipy", "custom_cuda"):
        t = time.time()
        lab, n = ccu.compute_component_labels_from_tensors(src, dst, n_nodes, use_gpu=True, cc_backend=backend)
        labels[backend] = (torch.as_tensor(lab).cpu().numpy(), int(n))
        print(f"CC[{backend}] nodes={n_nodes} edges={n_edges} components={n} {1e3 * (time.time() - t):.1f} ms")
    (la, na), (lb, nb) = labels["scipy"], labels["custom_cuda"]
    pairs = np.unique(np.stack([la, lb]), axis=1)
    assert na == nb == len(np.unique(la)) == len(np.unique(lb)) and pairs.shape[1] == na, "CC partitions differ"
print("CC custom_cuda == scipy (identical partitions)")

# ---------------------------------------------------------------------------------------------- 5. one npz event
data = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
f = f"{data}/unpacked/WmuHNL10GeV100mm_nopileup_10K/0.npz"
with np.load(f) as ev:
    dtypes = {k: (str(ev[k].dtype), ev[k].shape) for k in ev.files}
expected = {"x": "float32", "hid": "int64", "pid": "int64", "layers": "int64", "layerless_true_edges": "int64",
            "particles": "float64"}
assert {k: v[0] for k, v in dtypes.items()} == expected, dtypes
print(f"npz {f}: {dtypes}")
print(f"TOTAL {time.time() - t0:.1f}s  ALL OK")
