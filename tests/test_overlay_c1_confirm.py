"""hnl.overlay.c1_confirm: the C1 confirmation of addendum H section 9.2 (matched bins, pooled per volume with the
200-hit count rule and the C1 tolerances, support shares for all gun hits and P_P-target hits, "not informative" pixel
volumes) on synthetic per-hit feature tables."""

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import c1 as C1
from hnl.overlay import c1_confirm as CF


def _feat(volume, n, rng, tu=(-1.0, 1.0), tv=(0.0, 0.3), span0_scale=1.0, pp=None):
    t_u, t_v = rng.uniform(*tu, n), rng.uniform(*tv, n)
    f = C1.features(np.full(n, volume), t_u, t_v, span0_scale * (1 + rng.poisson(2, n)), 1 + rng.poisson(1, n),
                    0.1 + rng.gamma(4, 0.05, n), rng.normal(0, 0.005, n), rng.normal(0, 0.02, n))
    f["is_pp_target"] = np.zeros(n, bool) if pp is None else pp
    return f


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(7)
    ref = pd.concat([_feat(13, 40000, rng), _feat(8, 40000, rng)], ignore_index=True)
    # gun: 3,000 strip-barrel hits in one tan_u bin plus 100 outside the reference's support (|tan_v| 8-16);
    # 50 pixel-barrel hits (not informative)
    pp = np.r_[np.ones(1000, bool), np.zeros(2000, bool)]
    gun = pd.concat([_feat(13, 3000, rng, tu=(0.21, 0.29), pp=pp),
                     _feat(13, 100, rng, tu=(0.21, 0.29), tv=(8.5, 15.0), pp=np.ones(100, bool)),
                     _feat(8, 50, rng, tu=(0.21, 0.29))], ignore_index=True)
    return ref, gun


def test_confirmation_passes_on_the_same_distributions(samples):
    ref, gun = samples
    r = CF.confirm(ref, gun, n_boot=20)
    p1 = r["part1_matched_bins"]
    assert p1["n_bins_counted"] >= 3 and p1["n_bins_failed"] == 0 and p1["pass"]
    p2 = pd.DataFrame(r["part2_pooled"]).set_index(["volume", "quantity"])
    strip = p2.loc[13]
    assert sorted(strip.index) == ["bias_u", "mad_u", "span0", "span1"]          # no pixel-only quantity
    assert strip.counted.all() and strip.passed.all()
    assert (strip.n_gun_on_support >= 3000).all() and (strip.n_gun_on_support < 3100).all()
    pix = p2.loc[8]
    assert len(pix) == 6 and not pix.counted.any() and pix.passed.isna().all()
    assert r["pass"]
    ni = {(x["volume"], x["quantity"]) for x in r["not_informative_pixel"]}
    assert ni == {(v, q) for v in (7, 8, 9) for q in C1.QUANTITIES}         # 50 hits in 8, none in 7 and 9
    assert {x["volume"] for x in r["not_informative"]} == {7, 8, 9, 12, 14, 16, 17, 18}
    assert set(r["volumes"]) == {7, 8, 9, 12, 13, 14, 16, 17, 18}
    assert r["volumes"][13]["failed"] == [] and r["volumes"][8]["not_informative"] == sorted(C1.QUANTITIES)


def test_support_shares(samples):
    ref, gun = samples
    s = CF.support(ref, gun).set_index(["volume", "quantity"])
    row = s.loc[(13, "span0")]
    assert row.gun_hits == 3100 and row.on_support == 3000
    assert row.share_outside == pytest.approx(100 / 3100)
    assert row.pp_hits == 1100 and row.pp_share_outside == pytest.approx(100 / 1100)
    assert s.loc[(7, "span0")].gun_hits == 0 and pd.isna(s.loc[(7, "span0")].share_outside)


def test_a_wider_cluster_fails_both_parts(samples):
    ref, gun = samples
    rng = np.random.default_rng(8)
    wide = pd.concat([_feat(13, 3000, rng, tu=(0.21, 0.29), span0_scale=1.3), gun[gun.volume == 8]],
                     ignore_index=True)
    r = CF.confirm(ref, wide, n_boot=20)
    assert not r["pass"] and not r["part1_matched_bins"]["pass"]
    assert {"volume": 13, "quantity": "span0"} in r["part2_failed"]
    assert all(b["quantity"] == "span0" for b in r["part1_matched_bins"]["failed_bins"])
    assert r["volumes"][13]["failed"] == ["span0"]


def test_bias_rule_in_microns(samples):
    ref, gun = samples
    shifted = gun.assign(du=gun.du + 0.002)                              # 2 um
    p2 = CF.part2(ref, shifted, n_boot=20).set_index(["volume", "quantity"])
    assert p2.loc[(13, "bias_u"), "diff"] == pytest.approx(2.0, abs=0.5) and not p2.loc[(13, "bias_u"), "passed"]
    assert p2.loc[(13, "mad_u"), "passed"]


# ------------------------------------------------------------------------------------------------ D5 rules
def _gun_du(n, offset_um, sd_mm, rng, tu=(0.21, 0.79)):
    """n strip-barrel gun hits whose du is offset + sd x the normal quantiles (shuffled): the mean is offset exactly."""
    from statistics import NormalDist
    z = np.array([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])
    g = _feat(13, n, rng, tu=tu)
    return g.assign(du=offset_um * 1e-3 + sd_mm * rng.permutation(z))


@pytest.fixture(scope="module")
def wide_ref():
    """A strip-barrel reference with a 17 um u residual: at 250 gun hits the standard error of the gun mean is 1.1 um."""
    rng = np.random.default_rng(11)
    f = _feat(13, 40000, rng)
    return f.assign(du=rng.normal(0, 0.017, len(f)))


def test_ref_mean_se_matches_the_reweighted_reference(wide_ref):
    rng = np.random.default_rng(12)
    gun = _gun_du(250, 0.0, 0.017, rng)
    row = {r["quantity"]: r for r in CF.TC.at_test_incidence(wide_ref, gun, n_boot=20) if r["volume"] == 13}
    se = CF.ref_mean_se(wide_ref, gun, 13, "bias_u")
    assert se["mean"] == pytest.approx(row["bias_u"]["ref"], abs=1e-9)
    assert se["se"] == pytest.approx(17.0 / np.sqrt(se["n_eff"]), rel=0.05)
    assert 1000 < se["n_eff"] < 40000
    assert CF.ref_mean_se(wide_ref, gun, 13, "mad_u") is None                  # MAD: no standard error of a mean
    assert CF.ref_mean_se(wide_ref, gun, 8, "bias_u") is None                  # no gun hit in volume 8
    assert CF.bias_limit(0.3) == 1.0 and CF.bias_limit(1.2) == pytest.approx(2.4) and CF.bias_limit(None) == 1.0


def test_u_bias_rule_scales_with_the_standard_error(wide_ref):
    """Pilot volume 13 (204 hits, -1.59 um, sigma_gun 1.16 um) failed the 1 um rule; the rule of D5 passes such a
    difference at the 200-hit floor and still fails one beyond 2 sigma_diff."""
    rng = np.random.default_rng(13)
    for offset, want in ((-1.6, True), (-3.5, False)):
        p2 = CF.part2(wide_ref, _gun_du(250, offset, 0.017, rng), n_boot=200).set_index(["volume", "quantity"])
        b = p2.loc[(13, "bias_u")]
        assert b.counted and b["diff"] == pytest.approx(offset, abs=0.6)
        assert b.sigma_gun == pytest.approx(17.0 / np.sqrt(250), rel=0.2)
        assert b.sigma_diff == pytest.approx(np.hypot(b.sigma_gun, b.sigma_ref))
        assert b["limit"] == pytest.approx(max(1.0, 2 * b.sigma_diff)) and b["limit"] > 2.0
        assert b["rule"] == "|diff| <= max(1 um, 2 sigma_diff)"
        assert b.passed == want and b.outcome == ("pass" if want else "fail")
        assert abs(b["diff"]) > 1.0                                              # the 1 um rule would fail both
        s0 = p2.loc[(13, "span0")]
        assert s0["rule"] == "|rel_diff| <= 0.1" and s0.sigma_diff == pytest.approx(np.hypot(s0.sigma_gun, s0.sigma_ref))
        assert pd.isna(p2.loc[(13, "mad_u")].sigma_diff) and p2.loc[(13, "mad_u")].passed
    # the floor stays: 150 hits do not count, whatever the difference
    p2 = CF.part2(wide_ref, _gun_du(150, -10.0, 0.017, rng), n_boot=20).set_index(["volume", "quantity"])
    assert not p2.loc[(13, "bias_u")].counted and p2.loc[(13, "bias_u")].outcome == "not informative"


def test_not_informative_parts_and_outcome(wide_ref, samples):
    rng = np.random.default_rng(14)
    # 250 hits over four tan_u bins: no matched bin reaches 200 hits (part 1 not informative), the pooled part counts
    r = CF.confirm(wide_ref, _gun_du(250, 0.0, 0.017, rng), n_boot=50)
    p1 = r["part1_matched_bins"]
    assert p1["n_bins_counted"] == 0 and p1["outcome"] == "not informative" and p1["pass"] is None
    assert not p1["few_bins"]
    assert r["part2_outcome"] == "pass" and r["outcome"] == "pass" and r["pass"] is True and not r["blocks"]
    # a failing pooled u bias fails the confirmation although part 1 is not informative
    r = CF.confirm(wide_ref, _gun_du(250, -4.0, 0.017, rng), n_boot=50)
    assert r["part1_matched_bins"]["outcome"] == "not informative"
    assert r["outcome"] == "fail" and r["pass"] is False and r["blocks"]
    assert r["part2_failed"] == [{"volume": 13, "quantity": "bias_u"}] and r["volumes"][13]["failed"] == ["bias_u"]
    # nothing counted anywhere: not informative, neither pass nor fail, does not block
    r = CF.confirm(wide_ref, _gun_du(150, 0.0, 0.017, rng), n_boot=20)
    assert r["outcome"] == "not informative" and r["pass"] is None and not r["blocks"]
    assert r["part2_outcome"] == "not informative"
    # one tan_u bin: span0 and MAD(du) count in part 1 (two bins, "few bins"), span1 spreads over the tan_v bins
    ref, gun = samples
    p1 = CF.part1(ref, _gun_du(250, 0.0, 0.005, rng, tu=(0.21, 0.29)))
    assert p1["n_bins_counted"] == 2 and p1["few_bins"] and p1["outcome"] == "pass"
    p1 = CF.part1(ref, gun)
    assert p1["n_bins_counted"] >= 3 and not p1["few_bins"] and p1["outcome"] == "pass" and p1["pass"] is True
