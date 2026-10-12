"""Stage-1 read-out tooling: r-z classes, cut at a target edge FPR, dD, dT, ladder helpers."""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from hnl.eval import edge_tpr as ET
from hnl.eval import ladder_report as LR
from hnl.eval import rz_gap as RZ


def test_rz_classes():
    dz0 = np.array([0.0, -10.0, 50.0, 60.0, -100.0, 150.0, 5.0, np.nan, 10.0, 10.0])
    rv = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, np.nan, 50.0, 0.0, 1e-3])
    got = RZ.rz_classes(dz0, rv).tolist()
    lo, mid, hi = RZ.RZ_LABELS
    assert got == [lo, lo, mid, mid, hi, hi, RZ.NO_CLASS, RZ.NO_CLASS, RZ.NO_CLASS, hi]


def test_class_table_and_gap_shares():
    rng = np.random.default_rng(0)
    n = 400
    t = pd.DataFrame({"rz_class": rng.choice(list(RZ.CLASSES), n), "eff_0": rng.random(n) < 0.9,
                      "eff_1": rng.random(n) < 0.8})
    tab = RZ.class_table(t, ["eff_0", "eff_1"])
    assert list(tab.index) == list(RZ.CLASSES) + ["all"]
    assert tab.loc[list(RZ.CLASSES), "n"].sum() == n and tab.loc["all", "n"] == n
    for c in RZ.CLASSES:
        assert np.isclose(tab.loc[c, "eff_1"], t[t.rz_class == c].eff_1.mean())
    g = RZ.gap_shares(t, "eff_0", "eff_1")
    assert np.isclose(g.loc[list(RZ.CLASSES)].sum(), (t.eff_0.mean() - t.eff_1.mean()) * 100)
    assert np.isclose(g.loc["all"], (t.eff_0.mean() - t.eff_1.mean()) * 100)


def test_config_index():
    cfgs = [["cc", 0.3], ["mwslayer", 0.05], ["mwslayer", 0.1], ["walkmerge", 0.1, 0.5, 1.0]]
    assert RZ.config_index(cfgs, "mwslayer", "0.1") == 2
    assert RZ.config_index(cfgs, "walkmerge", "0.1,0.5,1.0") == 3
    with pytest.raises(ValueError):
        RZ.config_index(cfgs, "mwslayer", "0.2")


def test_cut_at_fpr_grid_exact_and_unreachable():
    f = np.arange(1000) / 1000.0                              # fake-edge scores 0.000 ... 0.999
    cut, ok, fpr = ET.cut_at_fpr(f, 0.1)
    assert ok and abs(cut - 0.90) < 1e-12 and abs(fpr - 0.099) < 1e-12       # 0.89 would give 0.109 > 0.1
    cut, ok, fpr = ET.cut_at_fpr(f, 0.1, None)
    assert ok and abs(cut - 0.899) < 1e-12 and abs(fpr - 0.1) < 1e-12
    assert ET.cut_at_fpr(f, 1.0, None)[0] == -np.inf
    cut, ok, fpr = ET.cut_at_fpr(np.ones(50), 0.1)           # every fake edge above every grid cut
    assert not ok and abs(cut - 0.01) < 1e-12 and fpr == 1.0
    with pytest.raises(ValueError):
        ET.cut_at_fpr(np.zeros(0), 0.1)


def test_delta_d():
    t = pd.DataFrame({
        "d0": np.r_[np.full(10, 0.5), np.full(10, -150.0), np.full(3, 150.0), [1.0, 300.0, 0.05]],
        "in_graph": np.r_[np.ones(20, bool), np.zeros(3, bool), np.ones(3, bool)],
        "score": np.r_[np.r_[np.full(8, 0.9), np.full(2, 0.1)], np.r_[np.full(5, 0.9), np.full(5, 0.2)],
                       np.full(3, np.nan), [0.9, 0.9, 0.9]]})
    d = ET.delta_d(t, 0.5)
    assert d["n_low"] == 10 and d["n_high"] == 10
    assert abs(d["tpr_low"] - 0.8) < 1e-12 and abs(d["tpr_high"] - 0.5) < 1e-12 and abs(d["dD"] + 0.3) < 1e-12


def test_delta_t():
    s = pd.DataFrame({"depth": [0, 2, 4], "tpr": [0.9, 0.85, 0.8]})
    dt, k = LR.delta_t(s)
    assert abs(dt + 0.1) < 1e-12 and k == 4
    dt, k = LR.delta_t(pd.DataFrame({"depth": [0, 2], "tpr": [0.9, 0.87]}))
    assert abs(dt + 0.03) < 1e-12 and k == 2
    assert np.isnan(LR.delta_t(pd.DataFrame({"depth": [2, 4], "tpr": [0.9, 0.8]}))[0])


def test_run_label():
    assert LR.run_label("/x/runs/stage1_screen_ttbar/sym/armi_s0") == "screen_ttbar/sym/armi_s0"
    assert LR.run_label("/x/runs/baseline_v1/armii_s2/") == "baseline_v1/armii_s2"
    assert LR.run_label("/x/y/z") == "y/z"


def test_read_dd_and_trackbuild_dirs(tmp_path):
    run = str(tmp_path)
    assert LR.read_dd(run) == {} and LR.read_dt(run) == {}
    json.dump({"target_efpr": 0.0122, "dD": -0.01, "cut": 0.4, "efpr": 0.012, "reachable": True},
              open(os.path.join(run, "edge_tpr_a_efpr.json"), "w"))
    time.sleep(0.01)
    json.dump({"target_efpr": 0.02, "dD": -0.05, "cut": 0.3, "efpr": 0.019, "reachable": True},
              open(os.path.join(run, "edge_tpr_b_efpr.json"), "w"))
    os.utime(os.path.join(run, "edge_tpr_b_efpr.json"), (time.time() + 5, time.time() + 5))
    assert LR.read_dd(run)["dD"] == -0.05                    # newest
    assert LR.read_dd(run, 0.0122)["dD"] == -0.01            # the file for the target
    for name, files in (("trackbuild_val", ("per_particle.parquet", "per_event.parquet", "configs.json")),
                        ("trackbuild_val_x", ("per_particle.parquet",))):
        os.makedirs(os.path.join(run, name))
        for f in files:
            open(os.path.join(run, name, f), "w").close()
    assert LR.trackbuild_dirs(run, None) == ["trackbuild_val"]
    assert LR.trackbuild_dirs(run, ["trackbuild_val_x", "trackbuild_val"]) == ["trackbuild_val"]


def test_markdown_blanks():
    tab = pd.DataFrame([{"run": "a", "family": "cc", "eff_primary": 0.95, "dD": np.nan},
                        {"run": "b", "family": "mwslayer", "eff_primary": 0.975, "dD": -0.01}])
    md = LR.markdown(tab).splitlines()
    assert md[0] == "| run | family | eff_primary | dD |"
    assert md[2] == "| a | cc | 0.9500 |  |" and md[3].endswith("| -0.0100 |")


def test_per_event_dd_counts():
    import pandas as pd
    from hnl.eval.edge_tpr import per_event_dd, delta_d
    t = pd.DataFrame({"event_id": [1, 1, 1, 2, 2, 3], "d0": [0.5, -150.0, 150.0, 0.2, 5.0, 200.0],
                      "in_graph": [True, True, True, True, True, False],
                      "score": [0.9, 0.2, 0.8, 0.95, 0.9, np.nan]})
    pe = per_event_dd(t, 0.5).set_index("event_id")
    assert pe.loc[1, "n_low"] == 1 and pe.loc[1, "k_low"] == 1 and pe.loc[1, "n_high"] == 2 and pe.loc[1, "k_high"] == 1
    assert pe.loc[2, "n_low"] == 1 and pe.loc[2, "n_high"] == 0 and pe.loc[3, "n_high"] == 0
    d = delta_d(t, 0.5)
    tot = pe.sum()
    assert abs(d["dD"] - (tot.k_high / tot.n_high - tot.k_low / tot.n_low)) < 1e-12
