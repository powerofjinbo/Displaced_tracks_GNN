"""hnl.trackml.classifier (sector / window dataset, TrackMLCanto), hnl.trackml.graphs.choose and hnl.trackml.freeze.

    pytest src/hnl/trackml/tests
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hnl.trackml import classifier as CL
from hnl.trackml import events as E
from hnl.trackml.split import REPO
from test_events import _candidates, _toy_event


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("store")
    for split in ("trainset", "valset"):
        os.makedirs(d / split)
    for i in range(3):
        ev = _toy_event(i, n_tracks=150, n_noise=400)
        cand = _candidates(ev)
        y = E.true_pair_mask(ev.n, cand, ev.true_edges)
        for split in ("trainset", "valset"):
            CL.save_event_graph(str(d / split / f"event90000{i:04d}.pyg.gz"), f"90000{i:04d}", ev.r, ev.phi, ev.z,
                                ev.pid, cand, y, ev.true_edges)
    return str(d)


def _model(store, **kw):
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "trackml", "classifier_train.yaml")))
    cfg.update({"input_dir": store, "stage_dir": store + "/stage", "data_split": [3, 3, 0], "windows_per_epoch": 12,
                "window_seed": 5, "num_workers": [0, 0, 0], **kw})
    torch.manual_seed(0)
    m = CL.TrackMLCanto(CL.model_hparams(cfg))
    m.setup("fit")
    return m


def test_dataset_lengths_determinism_and_partition(store):
    m = _model(store)
    assert len(m.trainset) == 12 and len(m.valset) == 3 * 8
    a, b = m.trainset.get(17), m.trainset.get(17)            # an epoch-tagged index (epoch 1, window 5)
    assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.edge_global, b.edge_global)
    # every window draws its event from the seeded cycle: the 3 events appear once per cycle of 3 steps
    ks = [CL.cycle_event(g, 3, 5) for g in range(9)]
    assert sorted(ks[:3]) == sorted(ks[3:6]) == sorted(ks[6:9]) == [0, 1, 2]
    # validation sectors: the owned edges of the 8 sectors of an event partition its edges
    ev0 = CL.load_event_graph(m.valset.input_paths[0])
    own = np.concatenate([m.valset.get(s).edge_global[m.valset.get(s).edge_owned].numpy() for s in range(8)])
    assert np.array_equal(np.sort(own), np.arange(ev0["pairs"].shape[1]))
    # the loss sees only owned edges: weight 1 on them, 0 elsewhere
    d = m.valset.get(3)
    assert torch.equal(d.edge_weights.bool(), d.edge_owned)


def test_truncation_only_in_training(store):
    from hnl.models.augment import truncation_mask
    m = _model(store, trunc_p=1.0)
    g = next(g for g in range(30) if CL.cycle_event(g, 3, 5) == 0)
    ev = CL.load_event_graph(m.trainset.input_paths[0])
    rng = np.random.default_rng([5, g, int(ev["eid"])])
    remove = truncation_mask(ev["pid"], np.hypot(ev["r"].astype(np.float64), ev["z"].astype(np.float64)), rng, 1.0)
    d = m.trainset.get(g)
    assert remove.any() and not np.isin(d.hit_global.numpy(), np.flatnonzero(remove)).any()
    # validation sectors are never truncated: together they hold every hit
    hits = np.unique(np.concatenate([m.valset.get(s).hit_global.numpy() for s in range(8)]))
    assert np.array_equal(hits, np.arange(len(ev["r"])))


def test_forward_and_loss_on_owned_edges(store):
    m = _model(store)
    d = m.trainset.get(0)
    from torch_geometric.loader import DataLoader
    b = next(iter(DataLoader([d], batch_size=1)))
    out = m(b)
    loss, pos, neg = m.loss_function(out, b)
    w = b.edge_weights
    manual = torch.nn.functional.binary_cross_entropy_with_logits(out, b.edge_y.float(), weight=w, reduction="sum") / w.sum()
    assert torch.isclose(loss, manual, rtol=1e-5)


def test_choose_rule():
    from hnl.trackml.graphs import choose
    s = pd.DataFrame({"density": [12, 12, 12, 14], "r_star": [0.07, 0.07, 0.07, 0.072], "rw_factor": [0, 1.5, 2.0, 1.5],
                      "tau": [-1, 0.01, 0.01, 0.005], "k": [0, 3, 5, 5], "pairs_per_hit": [12.0, 15.0, 16.0, 18.0],
                      "ceil_targets": [0.980, 0.9912, 0.9915, 0.999], "ceil_displaced_targets": [0.9] * 4,
                      "ceil_ge5": [0.97] * 4, "true_edge_eff": [0.99] * 4, "cap_binding": [0.0] * 4})
    g = {"union_max_pairs_per_hit": 16.5, "tie_ceiling": 0.0005, "gate_g1_ceiling": 0.990, "cap_bind_max": 0.05}
    ch = choose(s, g)
    # the 18-pairs configuration is over budget; 0.9912 is within 0.05 point of 0.9915 and has fewer pairs
    assert ch["pairs_per_hit"] == 15.0 and ch["gate_g1_pass"] and ch["cap_ok"] and np.isclose(ch["r_wide"], 0.105)


def test_frozen_configs(tmp_path):
    from hnl.trackml.freeze import frozen_configs
    sel = {"selection": {"WP-A": {"config": 3, "family": "walkmerge", "params": "0.2,0.7,1"},
                         "WP-S": {"config": 1, "family": "mwslayer", "params": "0.2"},
                         "WP-R": {"P1": {"reached": True, "config": 2, "family": "cc", "params": "0.5"},
                                  "P2": {"reached": False}, "P3": None,
                                  "P4": {"reached": True, "config": 4, "family": "mwslayer", "params": "0.3"}}}}
    p = tmp_path / "sel.json"
    p.write_text(json.dumps(sel))
    # since 5089c9a: WP-A in every condition, WP-S in C-full and C-nonoise, each WP-R in its reference's condition
    assert [c["name"] for c in frozen_configs(str(p), "full")] == ["WP-A", "WP-S"]
    assert [c["name"] for c in frozen_configs(str(p), "nonoise")] == ["WP-A", "WP-S", "WP-R-P1"]
    assert [c["name"] for c in frozen_configs(str(p), "1gev")] == ["WP-A", "WP-R-P4"]
