"""Builder grid evaluation on a toy scored graph (two chains joined by one fake edge)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")
from torch_geometric.data import Data  # noqa: E402

from hnl.eval import trackbuild_val as TV  # noqa: E402


def test_builders_split_merged_chains(tmp_path):
    pid = torch.tensor([1] * 5 + [2] * 5)
    r = torch.tensor([30., 70, 110, 170, 260, 30, 70, 110, 170, 260])
    z = torch.zeros(10)
    ei = torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8, 2], [1, 2, 3, 4, 6, 7, 8, 9, 8]])
    sc = torch.tensor([0.9] * 8 + [0.6])
    g = Data(hit_particle_id=pid, r=r, z=z, phi=torch.zeros(10), edge_index=ei, scores=sc,
             track_edges=torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 3, 4, 6, 7, 8, 9]]), event_id="200000003")
    p = tmp_path / "event200000003.pyg"
    torch.save(g, p)
    cfgs = [("cc", 0.5), ("jr", 0.5, 0.5), ("walk", 0.5, 0.99), ("ccwalk", 0.5, 0.5, 0.99), ("ccmerge", 0.7, 1.0)]
    part, ev = TV.evaluate_event(str(p), cfgs, {"200000003": (1, 2)})
    dm = part.pivot(index="config", columns="particle_id", values="dm_matched")
    assert dm.loc[0].tolist() == [False, False]         # CC merges both into one 10-hit track: 5/10 is no majority
    assert dm.loc[1].all() and dm.loc[2].all() and dm.loc[3].all()
    assert bool(ev.set_index("config").loc[0, "pair_merged"]) and not ev.set_index("config").loc[1, "pair_merged"]


def test_veto_builder_uses_vertex_scores(tmp_path):
    # the bridge 2 -> 8 has a high edge score but a high cross-daughter score: the veto families drop it
    pid = torch.tensor([1] * 5 + [2] * 5)
    r = torch.tensor([30., 70, 110, 170, 260, 30, 70, 110, 170, 260])
    ei = torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8, 2], [1, 2, 3, 4, 6, 7, 8, 9, 8]])
    sc = torch.tensor([0.9] * 8 + [0.95])
    vtx = torch.tensor([0.05] * 8 + [0.8])
    g = Data(hit_particle_id=pid, r=r, z=torch.zeros(10), phi=torch.zeros(10), edge_index=ei, scores=sc,
             edge_vtx_scores=vtx, event_id="200000003")
    p = tmp_path / "event200000003.pyg"
    torch.save(g, p)
    cfgs = [("cc", 0.5), ("vetocc", 0.5, 0.5), ("vetocc", 0.5, 0.9)]
    part, ev = TV.evaluate_event(str(p), cfgs, {"200000003": (1, 2)})
    dm = part.pivot(index="config", columns="particle_id", values="dm_matched")
    assert not dm.loc[0].any() and dm.loc[1].all() and not dm.loc[2].any()
    with pytest.raises(ValueError):
        TV.build(("vetocc", 0.5, 0.5), 10, ei[0].numpy(), ei[1].numpy(), sc.numpy(), r.numpy().astype(float))


def test_select_best_per_family_and_reports_families_without_config():
    import pandas as pd
    from hnl.eval.trackbuild_select import select
    cfgs = [["cc", 0.3], ["cc", 0.5], ["walkmerge", 0.1, 0.5, 1.0], ["walkmerge", 0.2, 0.5, 1.0], ["ccmergeany", 0.5, 0.5]]
    c = pd.DataFrame({"eff_primary": [0.95, 0.94, 0.96, 0.97, 0.99], "fake_rate_hnl": [0.04, 0.03, 0.02, 0.031, 0.05]},
                     index=range(5))
    sel, none_ok = select(c, cfgs, 0.03)
    best = sel.set_index("family")
    assert best.loc["cc", "params"] == "0.5" and best.loc["walkmerge", "params"] == "0.1,0.5,1.0"
    assert none_ok == ["ccmergeany"]


def test_llp_blind_uses_ttbar_targets_only():
    import pandas as pd
    from hnl.eval.trackbuild_select import llp_blind
    cfgs = [["mwslayer", 0.05], ["mwslayer", 0.1], ["mwslayer", 0.2], ["cc", 0.4]]
    c = pd.DataFrame({"eff_ttbar_targets": [0.90, 0.93, 0.93, 0.80], "fake_rate_ttbar": [0.05, 0.05, 0.04, 0.07],
                      "eff_primary": [0.99, 0.90, 0.91, 0.95], "fake_rate_hnl": [0.01, 0.02, 0.02, 0.03]}, index=range(4))
    lb = llp_blind(c, cfgs).set_index("family")
    assert lb.loc["mwslayer", "params"] == "0.2" and lb.loc["cc", "params"] == "0.4"     # ties: lower ttbar fake rate


def test_ldc_family_end_to_end(tmp_path, monkeypatch):
    # a node-quality model fitted on the toy itself (the two tracks are good nodes, their union is not)
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from hnl.trackbuild import dendrogram as D
    from hnl.trackbuild import ldc as L
    layers = np.array([32.0, 72, 116, 172, 260, 360, 500, 660])
    r = np.r_[layers, layers]
    phi = np.r_[0.10 + 1e-4 * layers, 1.20 - 2e-4 * layers]
    z = np.r_[0.3 * layers, -0.2 * layers]
    pid = np.repeat([11, 22], 8)
    a = np.r_[np.arange(7), 8 + np.arange(7), 3]
    b = np.r_[np.arange(1, 8), 9 + np.arange(7), 12]
    # the fake bridge scores above CC's cut but below the track edges (a bridge above them leaves no node holding one
    # track alone: single linkage cannot split that case, the learned assembly step must)
    s = np.r_[np.full(14, 0.9), 0.5]
    t = D.single_linkage(16, a, b, s)
    ks = L.candidates(t)
    X = L.node_features(t, ks, r, phi, z)
    y, _ = L.node_labels(t, ks, pid)
    clf = HistGradientBoostingClassifier(max_iter=50, min_samples_leaf=1).fit(np.r_[X, X], np.r_[y, y])
    joblib.dump(clf, tmp_path / "ldc.joblib")
    monkeypatch.setenv("HNL_LDC_MODEL", str(tmp_path / "ldc.joblib"))
    monkeypatch.setattr(TV, "_LDC", None)
    g = Data(hit_particle_id=torch.tensor(pid), r=torch.tensor(r, dtype=torch.float32),
             phi=torch.tensor(phi, dtype=torch.float32), z=torch.tensor(z, dtype=torch.float32),
             edge_index=torch.tensor(np.stack([a, b])), scores=torch.tensor(s, dtype=torch.float32), event_id="200000003")
    p = tmp_path / "event200000003.pyg"
    torch.save(g, p)
    part, ev = TV.evaluate_event(str(p), [("cc", 0.4), ("ldc", 0.5)], {"200000003": (11, 22)})
    dm = part.pivot(index="config", columns="particle_id", values="dm_matched")
    assert not dm.loc[0].any()                                  # CC merges both tracks through the bridge
    assert dm.loc[1].all()                                      # LDC splits them below the bridge


def test_mwsattach_family_attaches_hits_without_edges(tmp_path, monkeypatch):
    # particle 1: 8 barrel hits, graph edges only between its first 4 (the other 4 have none): mwslayer leaves a
    # 4-hit track (4 / 8, no double majority); a constant attach model (p = 1) recovers the other hits
    import joblib
    from sklearn.dummy import DummyClassifier
    from hnl.trackbuild import attach as AT
    R, phi0 = 6000.0, 0.3
    radii = np.array([32.0, 72, 116, 172, 260, 360, 500, 660])
    c = np.array([-R * np.sin(phi0), R * np.cos(phi0)])
    t = np.linspace(0, 800 / R, 200001)
    tl = np.interp(radii, np.hypot(c[0] + R * np.sin(phi0 + t), c[1] - R * np.cos(phi0 + t)), t)
    x, y = c[0] + R * np.sin(phi0 + tl), c[1] - R * np.cos(phi0 + tl)
    r, phi, z = np.hypot(x, y), np.arctan2(y, x), 0.2 * R * tl
    # particle 2: a far-away 5-hit chain
    r = np.r_[r, radii[:5]]
    phi = np.r_[phi, np.full(5, -2.0)]
    z = np.r_[z, np.zeros(5)]
    pid = np.r_[np.full(8, 1), np.full(5, 2)]
    a = np.array([0, 1, 2, 8, 9, 10, 11])
    b = np.array([1, 2, 3, 9, 10, 11, 12])
    s = np.full(7, 0.9)
    vol = np.r_[[8, 8, 8, 8, 13, 13, 13, 13], [8, 8, 8, 8, 13]]
    geo = {"volume": vol, "phi_c": phi.copy(), "layer_id": vol * 1000 + np.r_[np.arange(8), np.arange(5)]}
    clf = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, len(AT.FEATURES))), [0, 1])
    clf.hnl_features, clf.hnl_first_pass = list(AT.FEATURES), "mwslayer"
    joblib.dump(clf, tmp_path / "att.joblib")
    monkeypatch.setenv("HNL_MWSATTACH_MODEL", str(tmp_path / "att.joblib"))
    monkeypatch.setattr(TV, "_ATT", {})
    g = Data(hit_particle_id=torch.tensor(pid), r=torch.tensor(r, dtype=torch.float32),
             phi=torch.tensor(phi, dtype=torch.float32), z=torch.tensor(z, dtype=torch.float32),
             edge_index=torch.tensor(np.stack([a, b])), scores=torch.tensor(s, dtype=torch.float32),
             event_id="200000003")
    p = tmp_path / "event200000003.pyg"
    torch.save(g, p)
    part, ev = TV.evaluate_event(str(p), [("mwslayer", 0.1), ("mwsattach", 0.5)], {},
                                 layers={200000003: geo["layer_id"]}, geometry={200000003: geo})
    dm = part.pivot(index="config", columns="particle_id", values="dm_matched")
    assert not dm.loc[0, 1] and dm.loc[1, 1] and dm.loc[0, 2] and dm.loc[1, 2]
    # a model trained on the other first pass is refused
    clf.hnl_first_pass = "walkmerge"
    joblib.dump(clf, tmp_path / "att_wm.joblib")
    monkeypatch.setenv("HNL_MWSATTACH_MODEL", str(tmp_path / "att_wm.joblib"))
    monkeypatch.setattr(TV, "_ATT", {})
    with pytest.raises(ValueError):
        TV._attach_model("mwsattach")
