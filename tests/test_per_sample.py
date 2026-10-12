"""Pure helpers of hnl.eval.per_sample: the val-to-test change and its uncertainty (independent val and test tables),
the per-sample val replicate tables and the cell formats of the tables."""

import numpy as np
import pytest

from hnl.eval import bootstrap as BS
from hnl.eval import per_sample as P


def _model(rng, p, n_seeds, b, spread=0.002, boot=0.003):
    pts = list(p + spread * rng.standard_normal(n_seeds))
    reps = [x + boot * rng.standard_normal(b) for x in pts]
    return pts, reps


def test_change_of_one_model_is_test_minus_val_with_both_bootstraps_and_the_seed_term_of_the_changes():
    rng = np.random.default_rng(0)
    pv, rv = _model(rng, 0.990, 3, 1000)
    pt, rt = _model(rng, 0.985, 3, 500)                       # another table length: independent tables
    c = P.change_combination([(1.0, pv, rv, pt, rt)])
    ch = np.asarray(pt) - np.asarray(pv)
    sbv, sbt = BS.seed_mean(pv, rv)["sigma_boot"], BS.seed_mean(pt, rt)["sigma_boot"]
    s = np.std(ch, ddof=1)
    assert c["value"] == pytest.approx(np.mean(pt) - np.mean(pv), abs=1e-15)
    assert c["val"] == pytest.approx(np.mean(pv), abs=1e-15) and c["test"] == pytest.approx(np.mean(pt), abs=1e-15)
    assert c["sigma_boot_val"] == pytest.approx(sbv, abs=1e-15)
    assert c["sigma_boot_test"] == pytest.approx(sbt, abs=1e-15)
    assert c["s_seeds"][0] == pytest.approx(s, abs=1e-15)
    assert c["sigma"] == pytest.approx(np.sqrt(sbv ** 2 + sbt ** 2 + s ** 2 / 3), abs=1e-15)
    assert c["z"] == pytest.approx(c["value"] / c["sigma"])
    assert c["per_seed"][0] == pytest.approx(list(ch))


def test_seed_term_uses_the_per_seed_changes_not_the_seed_spread_of_either_split():
    rng = np.random.default_rng(1)
    pv, rv = _model(rng, 0.97, 3, 1000, spread=0.02)          # large seed spread on val ...
    pt = [x + 0.004 for x in pv]                              # ... the same networks shifted by the same amount
    rt = [x + 0.004 + 0.003 * rng.standard_normal(500) for x in pv]
    c = P.change_combination([(1.0, pv, rv, pt, rt)])
    assert c["value"] == pytest.approx(0.004, abs=1e-15)
    assert c["sigma_seeds"] == pytest.approx(0.0, abs=1e-15)
    assert c["sigma"] == pytest.approx(np.hypot(c["sigma_boot_val"], c["sigma_boot_test"]), abs=1e-15)


def test_change_of_a_difference_pairs_each_split_and_adds_both_models_seed_terms():
    rng = np.random.default_rng(2)
    nv, nrv = _model(rng, 0.991, 3, 1000)
    bv, brv = _model(rng, 0.986, 3, 1000)
    nt, nrt = _model(rng, 0.991, 3, 500)
    bt, brt = _model(rng, 0.980, 3, 500)
    c = P.change_combination([(1.0, nv, nrv, nt, nrt), (-1.0, bv, brv, bt, brt)])
    dv = BS.linear_combination([(1.0, nv, nrv), (-1.0, bv, brv)])
    dt = BS.linear_combination([(1.0, nt, nrt), (-1.0, bt, brt)])
    sn = np.std(np.asarray(nt) - np.asarray(nv), ddof=1)
    sb = np.std(np.asarray(bt) - np.asarray(bv), ddof=1)
    assert c["value"] == pytest.approx(dt["value"] - dv["value"], abs=1e-15)
    assert c["val"] == pytest.approx(dv["value"], abs=1e-15) and c["test"] == pytest.approx(dt["value"], abs=1e-15)
    assert c["sigma_boot_val"] == pytest.approx(dv["sigma_boot"], abs=1e-15)
    assert c["sigma_boot_test"] == pytest.approx(dt["sigma_boot"], abs=1e-15)
    assert c["sigma_seeds"] == pytest.approx(np.sqrt(sn ** 2 / 3 + sb ** 2 / 3), abs=1e-15)
    assert c["sigma"] == pytest.approx(np.sqrt(dv["sigma_boot"] ** 2 + dt["sigma_boot"] ** 2 + sn ** 2 / 3
                                               + sb ** 2 / 3), abs=1e-15)
    swapped = P.change_combination([(-1.0, nv, nrv, nt, nrt), (1.0, bv, brv, bt, brt)])
    assert swapped["value"] == pytest.approx(-c["value"], abs=1e-15)
    assert swapped["sigma"] == pytest.approx(c["sigma"], abs=1e-15)


def test_change_needs_the_same_seeds_on_val_and_test():
    rng = np.random.default_rng(3)
    pv, rv = _model(rng, 0.99, 3, 100)
    pt, rt = _model(rng, 0.99, 2, 100)
    with pytest.raises(ValueError):
        P.change_combination([(1.0, pv, rv, pt, rt)])


def test_per_sample_val_tables_are_fixed_and_hold_only_their_own_sample():
    from hnl.data.splits import read_split
    a, b = P.sample_resamplers(b=50), P.sample_resamplers(b=50)
    for (s, code), other in zip(P.VAL_SAMPLES, (4, 2)):
        name = f"{s}_val"
        u = a[s].units[name]
        assert np.array_equal(u, np.unique(read_split(s, "val")))
        w = a[s].w[name]
        assert w.shape == (50, len(u)) and (w.sum(axis=1) == len(u)).all()
        assert np.array_equal(w, b[s].w[name])                      # the same table every time
        ev = np.int64(code) * 100_000_000 + u[:3]
        k, n = a[s].unit_sums(ev, np.ones(3), np.ones(3))[name]
        assert k.sum() == 3 and n.sum() == 3
        with pytest.raises(ValueError):                             # another sample's events are refused
            a[s].unit_sums(np.int64(other) * 100_000_000 + u[:3], np.ones(3), np.ones(3))
    assert len(a["hnl10_100"].units["hnl10_100_val"]) == 1000 and len(a["hnl15_200"].units["hnl15_200_val"]) == 999
    w10, w15 = a["hnl10_100"].w["hnl10_100_val"], a["hnl15_200"].w["hnl15_200_val"]
    assert not np.array_equal(w10[:, :999], w15)                    # independent children of the seed sequence


def test_cell_formats():
    assert P.eff_cell(0.9907573, 0.0015452) == "99.08 ± 0.15"
    assert P.diff_cell(0.0109640, 0.0024375, 4.498) == "+1.10 ± 0.24 (z 4.5)"
    assert P.diff_cell(-0.0035, 0.0064, -0.547) == "-0.35 ± 0.64 (z -0.5)"
