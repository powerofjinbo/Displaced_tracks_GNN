"""p12: the filter's evaluation catches out-of-memory errors and retries (patch 0007: fixed chunks instead).

Pin, acorn/stages/edge_classifier/models/filter.py:183-216: Filter.memory_robust_eval runs split_batch(batch, n)
(batch.clone() per piece, a "Splitting batch due to memory error" warning on every call), catches RuntimeError
containing "out of memory", doubles n up to 128 and caches n in self.cached_n for all later batches.
Patched, filter.py:119-128, 185-202: node inputs computed once, per-edge MLP over
torch.split(edge_index, eval_chunk_size=2_000_000); nothing is caught, the batch is never cloned.
Not patched (characterised below, same at pin and patched):
  GNNFilter.memory_robust_eval / split_batch   filter.py:272-301 (pin 286-315): the same OOM catch, cached_n
  MetricLearning.get_distances                  graph_construction/models/metric_learning.py:201-214 (pin 197-210):
                                                any RuntimeError -> recomputed in 10 chunks
"""

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

N_NODES, N_EDGES = 50, 1000
OOM_MSG = "CUDA out of memory. Tried to allocate 20.00 GiB (toy)"


@pytest.fixture(autouse=True)
def highest_matmul_precision():
    """Chunked-vs-single comparisons need full fp32 matmuls; EdgeClassifierStage.setup("fit") elsewhere in the
    session sets "medium" (edge_classifier_stage.py:120-122)."""
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(old)


class Recorder(nn.Module):
    """Wraps the edge MLP: records the number of rows per call, optionally raises a CUDA-style OOM above a size."""

    def __init__(self, net, oom_above=None, error=None):
        super().__init__()
        self.net, self.oom_above, self.error, self.calls = net, oom_above, error, []

    def forward(self, x):
        self.calls.append(x.shape[0])
        if self.error is not None:
            raise self.error
        if self.oom_above is not None and x.shape[0] > self.oom_above:
            raise torch.OutOfMemoryError(OOM_MSG)
        return self.net(x)


def toy_batch(n_nodes=N_NODES, n_edges=N_EDGES, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return Data(
        hit_r=torch.rand(n_nodes, generator=g),
        hit_phi=torch.rand(n_nodes, generator=g),
        hit_z=torch.rand(n_nodes, generator=g),
        edge_index=torch.randint(0, n_nodes, (2, n_edges), generator=g),
    ).to(device)


def filter_hparams(**kw):
    hp = dict(node_features=["hit_r", "hit_phi", "hit_z"], hidden=16, nb_layer=3, layernorm=False,
              batchnorm=False, hidden_activation="ReLU", edge_cut=0.5, ratio=0)
    hp.update(kw)
    return hp


def make_filter(**kw):
    from acorn.stages.edge_classifier.models.filter import Filter

    torch.manual_seed(0)
    return Filter(filter_hparams(**kw)).eval()


# --------------------------------------------------------------------------------------------------------------
# Filter (patch 0007): fixed behaviour, expected to fail at the pin
# --------------------------------------------------------------------------------------------------------------


def test_filter_eval_runs_in_fixed_chunks_and_equals_a_single_pass():
    model = make_filter(eval_chunk_size=128)
    batch = toy_batch()
    with torch.no_grad():
        single = model(batch)  # forward(): one pass over all edges
        model.net = Recorder(model.net)
        chunked = model.memory_robust_eval(batch)
    assert model.net.calls == [128] * 7 + [N_EDGES - 7 * 128]
    torch.testing.assert_close(chunked, single, rtol=0, atol=1e-6)


def test_filter_eval_does_not_catch_oom_and_does_not_retry():
    model = make_filter()  # default eval_chunk_size (2M) > N_EDGES: one chunk
    model.net = Recorder(model.net, oom_above=N_EDGES // 3)
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        model.memory_robust_eval(toy_batch())
    assert model.net.calls == [N_EDGES]  # one attempt, no halving
    assert not hasattr(model, "cached_n")


def test_filter_eval_never_clones_the_batch(monkeypatch):
    model = make_filter(eval_chunk_size=300)
    batch = toy_batch()
    edge_index = batch.edge_index.clone()

    def no_clone(self, *a, **k):
        raise AssertionError("batch.clone() called")

    monkeypatch.setattr(Data, "clone", no_clone)
    with torch.no_grad():
        out = model.memory_robust_eval(batch)
    assert out.shape == (N_EDGES,) and torch.equal(batch.edge_index, edge_index)


def test_filter_eval_chunk_size_default_and_validation(monkeypatch):
    sizes = []
    real_split = torch.split

    def spy(tensor, split_size_or_sections, dim=0):
        sizes.append(split_size_or_sections)
        return real_split(tensor, split_size_or_sections, dim=dim)

    monkeypatch.setattr(torch, "split", spy)
    with torch.no_grad():
        make_filter().memory_robust_eval(toy_batch())
    assert sizes == [2_000_000]
    for bad in (0, -5):
        with pytest.raises(ValueError, match="eval_chunk_size must be positive"):
            make_filter(eval_chunk_size=bad).memory_robust_eval(toy_batch())


def test_filter_shared_evaluation_scores_through_chunked_eval():
    model = make_filter(eval_chunk_size=256)
    batch = toy_batch()
    batch.edge_y = torch.arange(N_EDGES) % 3 == 0
    batch.edge_weights = torch.ones(N_EDGES)
    with torch.no_grad():
        ref = torch.sigmoid(model(batch))
        model.net = Recorder(model.net)
        out = model.shared_evaluation(batch, 0)
    assert model.net.calls == [256] * 3 + [N_EDGES - 3 * 256]
    torch.testing.assert_close(out["batch"].edge_scores, ref, rtol=0, atol=1e-6)


def test_filter_chunked_eval_with_batchnorm_without_running_stats_depends_on_chunking(acorn_origin):
    """Residual risk of patch 0007: BatchNorm1d(track_running_stats=False), Filter's default when batchnorm is on,
    uses batch statistics also in eval mode, so chunked logits differ from a single pass."""
    if acorn_origin == "pin":
        pytest.skip("the pin has no eval_chunk_size (single pass unless an OOM occurs)")
    model = make_filter(batchnorm=True, eval_chunk_size=128)
    batch = toy_batch()
    with torch.no_grad():
        diff = (model.memory_robust_eval(batch) - model(batch)).abs().max().item()
    assert diff > 1e-3
    model = make_filter(batchnorm=True, track_running_stats=True, eval_chunk_size=128)  # running stats: no effect
    with torch.no_grad():
        torch.testing.assert_close(model.memory_robust_eval(batch), model(batch), rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------------------------------
# OOM catching that remains (not touched by the patches): characterisation, same at pin and patched
# --------------------------------------------------------------------------------------------------------------


def make_gnn_filter():
    from acorn.stages.edge_classifier.models.filter import GNNFilter

    torch.manual_seed(0)
    gnn = [{"module_name": "torch.nn", "class_name": "Linear", "init_kwargs": {"in_features": 3, "out_features": 8},
            "inputs": ["x"]}]
    return GNNFilter(filter_hparams(hidden=8, nb_layer=2, gnn_config=gnn)).eval()


def test_gnn_filter_still_catches_oom_halves_and_caches_the_split():
    model = make_gnn_filter()
    batch = toy_batch()
    with torch.no_grad():
        x = model.gnn(model.stack_x(batch), None)
        direct = model.net(torch.cat([x[batch.edge_index[0]], x[batch.edge_index[1]]], dim=-1)).squeeze(-1)
        model.net = Recorder(model.net, oom_above=N_EDGES // 3)
        out = model.memory_robust_eval(x, batch.edge_index)
        assert model.net.calls == [1000, 500, 250, 250, 250, 250]  # n = 1 (OOM), 2 (OOM), 4 (ok)
        assert model.cached_n == 4
        torch.testing.assert_close(out, direct, rtol=0, atol=1e-6)
        # the cached split is sticky: a later, small graph is still evaluated in 4 pieces
        model.net.calls.clear()
        model.memory_robust_eval(x, batch.edge_index[:, :40])
        assert model.net.calls == [10, 10, 10, 10]


def test_gnn_filter_gives_up_after_128_pieces_and_reraises_other_errors():
    model = make_gnn_filter()
    batch = toy_batch()
    with torch.no_grad():
        x = model.gnn(model.stack_x(batch), None)
        model.net = Recorder(model.net, oom_above=0)
        with pytest.raises(RuntimeError, match="Splitting into 128 parts was not enough"):
            model.memory_robust_eval(x, batch.edge_index)
        assert len(model.net.calls) == 8  # n = 1, 2, ..., 128: first piece of each attempt OOMs
        fresh = make_gnn_filter()
        fresh.net = Recorder(fresh.net, error=RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
        with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
            fresh.memory_robust_eval(x, batch.edge_index)
        assert fresh.net.calls == [N_EDGES]


def test_metric_learning_get_distances_swallows_any_runtime_error(monkeypatch):
    from acorn.stages.graph_construction.models.metric_learning import MetricLearning

    model = MetricLearning(dict(node_features=["hit_r", "hit_phi", "hit_z"], emb_hidden=8, nb_layer=2, emb_dim=4,
                                activation="Tanh"))
    g = torch.Generator().manual_seed(1)
    emb = torch.nn.functional.normalize(torch.randn(N_NODES, 4, generator=g))
    edges = toy_batch().edge_index
    expected = ((emb[edges[1]] - emb[edges[0]]) ** 2).sum(-1)
    real_sum, calls = torch.sum, []

    def failing_sum(t, *a, **k):  # a non-OOM error on the full tensor
        calls.append(t.shape[0])
        if t.shape[0] > N_EDGES // 5:
            raise RuntimeError("some unrelated device-side error")
        return real_sum(t, *a, **k)

    monkeypatch.setattr(torch, "sum", failing_sum)
    d = model.get_distances(emb, edges)
    monkeypatch.undo()
    assert calls == [N_EDGES] + [N_EDGES // 10] * 10  # caught silently, recomputed in 10 chunks
    torch.testing.assert_close(d, expected, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------------------------------
# GPU (run in a Slurm job): real CUDA allocations
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.gpu
def test_filter_gpu_chunked_eval_bounds_peak_memory_and_matches_single_pass():
    dev = torch.device("cuda")
    try:
        model = make_filter(hidden=64, nb_layer=2, eval_chunk_size=500_000).to(dev)  # MLP 6 -> 64 -> 32 -> 1
        batch = toy_batch(n_nodes=200_000, n_edges=4_000_000, device=dev)
        with torch.no_grad():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            single = model(batch)
            torch.cuda.synchronize()
            peak_single = torch.cuda.max_memory_allocated(dev) - base
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            chunked = model.memory_robust_eval(batch)
            torch.cuda.synchronize()
            peak_chunked = torch.cuda.max_memory_allocated(dev) - base
        torch.testing.assert_close(chunked, single, rtol=1e-5, atol=1e-5)
        assert peak_chunked < 0.5 * peak_single, (peak_chunked, peak_single)
    finally:
        torch.cuda.empty_cache()


@pytest.mark.gpu
def test_filter_gpu_real_oom_propagates_and_chunking_avoids_it():
    import gc

    dev = torch.device("cuda")
    model = make_filter(hidden=64, nb_layer=2, eval_chunk_size=8_000_000).to(dev)
    batch = toy_batch(n_nodes=200_000, n_edges=8_000_000, device=dev)  # single pass needs > 2 GiB
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(dev).total_memory
    budget = torch.cuda.memory_reserved(dev) + 768 * 2**20
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), torch.cuda.current_device())
    try:
        with torch.no_grad():
            with pytest.raises(torch.OutOfMemoryError):
                model.memory_robust_eval(batch)  # the pin catches this and retries in halves
            assert not hasattr(model, "cached_n")
            gc.collect()  # drop the frames kept alive by the exception traceback
            torch.cuda.empty_cache()
            model.hparams["eval_chunk_size"] = 250_000
            out = model.memory_robust_eval(batch)
        assert out.shape == (8_000_000,) and torch.isfinite(out).all()
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())
        torch.cuda.empty_cache()


def test_source_audit_of_runtime_error_handlers():
    """Static audit of the imported acorn tree (ast): `except RuntimeError` handlers. Patched: GNNFilter's OOM retry,
    the two metric-learning get_distances fallbacks and run_data_tests' old-PyG-format check
    (loading_utils.py:150-160, not OOM related). At the pin filter.py has 2 (Filter and GNNFilter)."""
    import ast
    import glob
    import os
    import warnings
    from collections import Counter

    import acorn

    base = os.path.dirname(os.path.dirname(os.path.abspath(acorn.__file__)))
    handlers = Counter()
    for path in glob.glob(os.path.join(base, "acorn", "**", "*.py"), recursive=True):
        with open(path) as f, warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                if any(n in ast.unparse(node.type) for n in ("RuntimeError", "OutOfMemoryError")):
                    handlers[os.path.relpath(path, base)] += 1
    assert dict(handlers) == {
        "acorn/stages/edge_classifier/models/filter.py": 1,
        "acorn/stages/graph_construction/models/metric_learning.py": 1,
        "acorn/stages/graph_construction/models/double_metric_learning.py": 1,
        "acorn/utils/loading_utils.py": 1,
    }
