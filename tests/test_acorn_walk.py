"""hnl.trackbuild.acorn_walk reproduces acorn's own CCandWalk._build_tracks_one_evt (plain and score-guided walk)."""

import numpy as np
import pytest
import torch

pytest.importorskip("acorn")


def _event(rng, n_tracks=12, n_layers=8, n_noise=40):
    """Straight-ish tracks crossing layers (one hit per layer) plus noise hits; true edges score high, a random set of
    fake edges (also between tracks, making junctions) scores low to high."""
    r, z, pid = [], [], []
    for t in range(n_tracks):
        phi0, cot = rng.uniform(-np.pi, np.pi), rng.uniform(-1, 1)
        start = int(rng.integers(0, 3))                    # some tracks start late (displaced-like)
        for k in range(start, n_layers):
            rr = 30.0 + 60.0 * k + rng.normal(0, 0.5)
            r.append(rr)
            z.append(cot * rr + rng.normal(0, 1.0))
            pid.append(t + 1)
    for _ in range(n_noise):
        r.append(rng.uniform(30, 500))
        z.append(rng.uniform(-500, 500))
        pid.append(0)
    r, z, pid = np.array(r, np.float32), np.array(z, np.float32), np.array(pid, np.int64)
    n = len(r)
    edges, scores = [], []
    for t in np.unique(pid[pid > 0]):
        idx = np.flatnonzero(pid == t)
        idx = idx[np.argsort(r[idx])]
        for i, j in zip(idx[:-1], idx[1:]):
            edges.append((i, j) if rng.random() < 0.5 else (j, i))
            scores.append(rng.uniform(0.3, 1.0))
    for _ in range(4 * n):
        i, j = rng.integers(0, n, 2)
        if i != j:
            edges.append((i, j))
            scores.append(rng.uniform(0.0, 0.9) ** 2)
    e = np.array(edges, dtype=np.int64)
    a, b = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    _, keep = np.unique(a * n + b, return_index=True)       # every pair once, as in our scored graphs
    keep = np.sort(keep)
    return n, r, z, pid, e[keep].T.copy(), np.array(scores, np.float32)[keep]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_acorn_method(tmp_path, seed):
    import pandas as pd
    from torch_geometric.data import Data
    from acorn.utils.loading_utils import save_pyg
    from hnl.trackbuild.acorn_walk import AcornCCWalk, acorn_method_labels, _partition
    old = pd.get_option("future.infer_string")
    pd.set_option("future.infer_string", False)              # acorn's add_track_labels under pandas 3
    try:
        rng = np.random.default_rng(seed)
        n, r, z, pid, ei, sc = _event(rng)
        g = Data(hit_particle_id=torch.as_tensor(pid), r=torch.as_tensor(r), z=torch.as_tensor(z),
                 edge_index=torch.as_tensor(ei), scores=torch.as_tensor(sc), num_nodes=n)
        path = str(tmp_path / "event000000001.pyg")
        save_pyg(g, path)
        w = AcornCCWalk.from_path(path + ".gz")
        n_tracks = 0
        for cut, mn, add in [(0.01, 0.1, 0.6), (0.01, 0.05, 0.4), (0.1, 0.1, 0.6), (0.3, 0.3, 0.6), (0.01, 0.4, 0.8)]:
            for guided in (False, True):
                ours = w.labels(cut, mn, add, guided=guided)
                ref = acorn_method_labels(path + ".gz", cut, mn, add, guided)
                assert _partition(ours) == _partition(ref), (cut, mn, add, guided)
                n_tracks += len(_partition(ours))
        assert n_tracks > 0
    finally:
        pd.set_option("future.infer_string", old)


def test_track_labels_first_track_wins():
    from hnl.trackbuild.acorn_walk import track_labels
    lab = track_labels(6, {"cc": [[0, 1, 2]], "walk": [[2, 3], [4]]})
    assert lab.tolist() == [0, 0, 0, 1, 2, -1]
