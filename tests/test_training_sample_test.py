"""One-seed test read-out of the training-sample runs (hnl.eval.training_sample_test): helpers without data."""

import json

import numpy as np
import pytest

from hnl.eval import bootstrap as BS
from hnl.eval import training_sample_test as T


def test_close_and_g10():
    assert T._close(0.5, 0.5 + 1e-13) and not T._close(0.5, 0.5 + 1e-11)
    assert T._close(np.nan, float("nan")) and not T._close(np.nan, 0.0)
    assert T._g10(0.12345678901234) == "0.123456789" and T._g10(np.nan) == ""
    c = []
    assert T.check(c, "g10", 0.98765432109876, 0.9876543211, how="g10")
    assert not T.check(c, "num", 0.98765432109876, 0.9876543211)
    assert [x[3] for x in c] == [True, False]


def test_one_seed_sigma_is_the_bootstrap_alone():
    rng = np.random.default_rng(0)
    one = BS.seed_mean([0.99], [0.99 + 0.002 * rng.standard_normal(500)])
    row = T._one_seed(one)
    assert row["sigma"] == row["sigma_boot"] == one["sigma_boot"] and np.isnan(row["seed_sd"])
    three = BS.seed_mean([0.99, 0.98, 0.985], [0.99 + 0.002 * rng.standard_normal(500) for _ in range(3)])
    with pytest.raises(SystemExit):
        T._one_seed(three)


def test_refuses_existing_outputs(tmp_path):
    T.refuse_existing(str(tmp_path), "x")
    (tmp_path / "x_meta.json").write_text("{}")
    with pytest.raises(SystemExit):
        T.refuse_existing(str(tmp_path), "x")


def _files(tmp_path, wp="mwslmerge 0.2 (trackbuild_test, configuration 0)", n_entries=1, label="Canto-L (HNL): x"):
    run = "/r/stage1_final_learned_ts/l1tl/armii_s0"
    sel = {"pipelines": {"l1tl_ii": {"seeds": {"0": {"run": run, "config": "mwslmerge,0.2"}}}}}
    spec = [{"label": label, "short": "Canto-L (HNL)", "group": "g",
             "entries": [{"run": run, "sub": "trackbuild_test", "cut": 0, "wp": wp}] * n_entries}]
    (tmp_path / "sel.json").write_text(json.dumps(sel))
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    return str(tmp_path / "spec.json"), str(tmp_path / "sel.json")


def test_load_models(tmp_path):
    c = []
    m = T.load_models(*_files(tmp_path), c)
    assert all(x[3] for x in c) and m[0]["entries"][0]["config"] == "mwslmerge,0.2"
    c = []
    T.load_models(*_files(tmp_path, wp="mwslmerge 0.3 (trackbuild_test, configuration 0)"), c)
    assert not all(x[3] for x in c)
    with pytest.raises(SystemExit):
        T.load_models(*_files(tmp_path, n_entries=2), [])
    with pytest.raises(SystemExit):
        T.load_models(*_files(tmp_path, label=T.CANTO), [])


def test_sp_kept():
    assert T.sp_kept(("hnl_joint_test", "v0", "r_0_50"), 3)
    assert T.sp_kept(("hnl10_100_test", "v0", "all"), 3) and not T.sp_kept(("hnl10_100_test", "v0", "r_0_50"), 3)
    assert not T.sp_kept(("ttbar_test", "conv", "all"), 0)
