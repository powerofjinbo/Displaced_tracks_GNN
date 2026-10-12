"""hnl.eval.test_stats on fabricated per-particle results for real test events (no model involved)."""

import numpy as np
import pandas as pd
import pytest

from hnl.data.splits import read_split
from hnl.eval import bootstrap as BS
from hnl.eval import test_stats as TS


def _fake_runs(info, ev, n_seeds=3, with_e=True):
    rng = np.random.default_rng(0)
    runs = []
    for s in range(n_seeds):
        rt = TS.RunTest.__new__(TS.RunTest)
        rt.pp = pd.DataFrame({"event_id": info.event_id, "particle_id": info.particle_id,
                              "dm_matched": rng.random(len(info)) < 0.9 + 0.02 * s})
        rt.pe = pd.DataFrame({"event_id": ev, "n_fake": rng.integers(0, 3, len(ev)),
                              "n_tracks_ge5": rng.integers(5, 30, len(ev))})
        rt.pe_e = None
        if with_e:
            n_true = rng.integers(50, 100, len(ev))
            rt.pe_e = pd.DataFrame({"event_id": ev, "tp": n_true - rng.integers(0, 5, len(ev)), "n_true": n_true,
                                    "fp": rng.integers(0, 50, len(ev)), "n_edges": n_true + 500,
                                    "n_true_in_graph": n_true})
        runs.append(rt)
    return runs


@pytest.fixture(scope="module")
def truth():
    from hnl.eval.particles import particle_table
    ev = np.concatenate([2 * 100_000_000 + read_split("hnl10_100", "test")[:30],
                         4 * 100_000_000 + read_split("hnl15_200", "test")[:30],
                         1 * 100_000_000 + read_split("ttbar", "test")[:10],
                         3 * 100_000_000 + read_split("hnl15_100", "external")[:30]])
    return particle_table(ev), ev


@pytest.mark.data
def test_model_summary_runs(tmp_path, truth):
    info, ev = truth
    bs = BS.Bootstrap(directory=str(tmp_path))
    out = TS.model_summary(bs, _fake_runs(info, ev), info)
    e = out["eff_primary@hnl_joint_test+hnl15_100_external"]
    assert 0.8 < e["mean"] < 1.0 and e["sigma"] > 0 and len(e["per_seed"]) == 3
    assert 0 < out["fake_rate@ttbar_test"]["mean"] < 0.5
    assert 0.9 < out["edge_eff@hnl_joint_test+hnl15_100_external@e_star"]["mean"] < 1.0
    # per-sample halves of the joint set add up to the joint numbers
    n10 = out["eff_primary@hnl10_100_test"]["n"]
    n15 = out["eff_primary@hnl15_200_test"]["n"]
    assert n10 + n15 == out["eff_primary@hnl_joint_test"]["n"] and n10 > 0 and n15 > 0


@pytest.mark.data
def test_binned_counts_and_no_vertex(tmp_path, truth):
    info, ev = truth
    bs = BS.Bootstrap(directory=str(tmp_path))
    runs = _fake_runs(info, ev, with_e=False)
    M = TS.masks()
    b = TS.binned_efficiency(bs, runs, info, M["hnl_targets"], "rv", [0, 10, 30, 100, 200, 300, np.inf], TS.HNL_SETS)
    total = TS.efficiency(bs, runs, info, M["hnl_targets"], TS.HNL_SETS)[2]
    assert b.n.sum() == total and b.label.iloc[-1] == "no vertex"
    # the n column counts only the requested sets
    b_ext = TS.binned_efficiency(bs, runs, info, M["hnl_targets"], "absd0", [0, 50, np.inf], ("hnl15_100_external",))
    assert b_ext.n.sum() == TS.efficiency(bs, runs, info, M["hnl_targets"], ("hnl15_100_external",))[2]


@pytest.mark.data
def test_paper_prompt_mask_has_no_pt_eta_cut(truth):
    info, _ = truth
    M = TS.masks()
    paper = M["w_muons_paper"](info)
    target = M["w_muons"](info)
    assert (paper | ~target).all() and paper.sum() >= target.sum()
    assert (info.n_hits[paper] >= 5).all()


@pytest.mark.data
def test_paper_summary_counts_and_strict_denominator(tmp_path, truth):
    info, ev = truth
    bs = BS.Bootstrap(directory=str(tmp_path))
    runs = _fake_runs(info, ev, with_e=False)
    n_paper = int(TS.masks()["w_muons_paper"](info[info.event_id // 100_000_000 == 3]).sum())
    out = TS.paper_summary(bs, runs, info, expected_w=None)
    assert out["eff_w_muons_paper@hnl15_100_external"]["n"] == n_paper
    with pytest.raises(ValueError, match="denominator"):
        TS.paper_summary(bs, runs, info, expected_w=n_paper + 1)


def test_runtest_refuses_missing_events(tmp_path, monkeypatch):
    d = tmp_path / "armx_s0" / "eval_test"
    d.mkdir(parents=True)
    ev = np.array([200000001, 400000001, 100000001, 300000001, 300000002])
    pd.DataFrame({"event_id": ev, "particle_id": 1, "cut": np.float32(0.4), "dm_matched": True}).to_parquet(
        d / "per_particle.parquet")
    pd.DataFrame({"event_id": ev, "cut": 0.4, "n_fake": 0, "n_tracks_ge5": 1}).to_parquet(d / "per_event.parquet")
    monkeypatch.setattr(TS, "expected_events", lambda: {"hnl_joint_test": 2, "ttbar_test": 1, "hnl15_100_external": 3})
    with pytest.raises(ValueError, match="hnl15_100_external"):
        TS.RunTest(str(tmp_path / "armx_s0"), 0.4)
    monkeypatch.setattr(TS, "expected_events", lambda: {"hnl_joint_test": 2, "ttbar_test": 1, "hnl15_100_external": 2})
    assert len(TS.RunTest(str(tmp_path / "armx_s0"), 0.4).pe) == 5
