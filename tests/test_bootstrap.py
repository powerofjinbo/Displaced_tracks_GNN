"""Bootstrap statistics of the preregistration (§5): replicate tables, pooled ratios, seed terms, pairing."""

import numpy as np

from hnl.eval import bootstrap as BS


def test_weights_are_multinomial():
    w = BS.make_weights(50, 200, np.random.default_rng(0))
    assert w.shape == (200, 50) and (w.sum(axis=1) == 50).all()
    assert abs(w.mean() - 1.0) < 1e-12 and 0.8 < w.var() < 1.2          # Poisson-like multiplicities


def test_tables_fixed_and_paired(tmp_path):
    a = BS.Bootstrap(directory=str(tmp_path))
    b = BS.Bootstrap(directory=str(tmp_path))                            # reloaded, identical
    for k in BS.SETS:
        assert np.array_equal(a.w[k], b.w[k]) and np.array_equal(a.units[k], b.units[k])
    # joint set: both samples' event i share one unit
    u = a.units["hnl_joint_test"]
    e = u[:5]
    v1 = a.per_unit("hnl_joint_test", e, np.ones(5))
    v2 = a.per_unit("hnl_joint_test", np.concatenate([e, e]), np.ones(10))
    assert np.array_equal(2 * v1, v2)


def test_ratio_and_seed_terms(tmp_path):
    bs = BS.Bootstrap(directory=str(tmp_path))
    u = bs.units["ttbar_test"]
    rng = np.random.default_rng(1)
    den = rng.integers(1, 6, size=len(u)).astype(float)
    num_a = rng.binomial(den.astype(int), 0.9).astype(float)
    num_b = np.minimum(den, num_a + rng.binomial(1, 0.3, size=len(u)))
    pa, ra = bs.ratio([("ttbar_test", num_a, den)])
    pb, rb = bs.ratio([("ttbar_test", num_b, den)])
    assert abs(pa - num_a.sum() / den.sum()) < 1e-12 and ra.shape == (BS.B,)
    # binomial-like spread
    assert 0.5 < np.std(ra) / np.sqrt(pa * (1 - pa) / den.sum()) < 2.0
    # paired difference is much tighter than the unpaired one
    d = BS.linear_combination([(1.0, [pb], [rb]), (-1.0, [pa], [ra])])
    assert d["sigma_boot"] < 0.5 * np.hypot(np.std(ra), np.std(rb))
    # seed term
    m = BS.seed_mean([0.90, 0.92, 0.94], [ra, ra, ra])
    assert abs(m["s_seeds"] - 0.02) < 1e-12 and abs(m["sigma"] ** 2 - (m["sigma_boot"] ** 2 + 0.02 ** 2 / 3)) < 1e-12
    pc = BS.primary_claim(([0.96] * 3, [rb] * 3), ([0.90] * 3, [ra] * 3))
    assert pc["criterion1_halved"] and abs(pc["ineff_ratio"] - 0.4) < 1e-9
    assert abs(pc["halving_margin"]["value"] - (0.5 * 0.10 - 0.04)) < 1e-12


def test_ratio_of_empty_population_is_nan(tmp_path):
    bs = BS.Bootstrap(directory=str(tmp_path))
    p, r = bs.ratio([])
    assert np.isnan(p) and r.shape == (BS.B,) and np.isnan(r).all()
    u = bs.units["ttbar_test"]
    p, r = bs.ratio([("ttbar_test", np.zeros(len(u)), np.zeros(len(u)))])
    assert np.isnan(p) and np.isnan(r).all()
