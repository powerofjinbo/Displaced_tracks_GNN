"""Second displaced population (feedback round 2, task F): population definitions, bins, counting and differences."""

import numpy as np
import pandas as pd
import pytest

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import second_population as SP


def _particles() -> pd.DataFrame:
    rows = [  # parent_pdg, pdg, gen, process, is_target -> expected populations
        (310, 211, 0, 0, True, {"v0", "k0s"}),
        (310, -211, 0, 0, False, set()),             # not a target
        (-3122, -2212, 0, 0, True, {"v0", "lambda"}),
        (3122, -211, 0, 0, True, {"v0", "lambda"}),
        (0, 11, 1, 2, True, {"conv"}),               # conversion e-
        (0, -11, 2, 2, True, {"conv"}),              # conversion e+ of a later generation
        (0, -11, 1, 2, False, set()),                # conversion e+, not a target
        (22, 11, 0, 0, True, set()),                 # generator-level e- with a photon parent: not a conversion
        (0, 22, 2, 3, True, set()),                  # bremsstrahlung photon
        (0, 211, 1, 2, True, set()),                 # (impossible) non-electron with code 2: excluded
        (50, 13, 0, 0, True, set()),                 # HNL daughter
        (311, 130, 0, 0, False, set()),              # K0L
    ]
    df = pd.DataFrame([r[:5] for r in rows], columns=["parent_pdg", "pdg", "gen", "process", "is_target"])
    return df, [r[5] for r in rows]


def test_population_masks():
    df, want = _particles()
    m = SP.population_masks(df)
    for i, w in enumerate(want):
        got = {p for p in SP.POPULATIONS if m[p][i]}
        assert got == w, (i, got, w)
    assert np.array_equal(m["v0"], m["k0s"] | m["lambda"]) and not (m["v0"] & m["conv"]).any()


def test_quantity_masks_bins():
    r = np.array([32.0, 49.99, 50.0, 99.99, 100.0, 199.9, 200.0, 299.99, 300.0, 1000.0])
    d0 = np.array([0.05, -0.1, 0.5, -2.9, 3.0, 9.99, -10.0, 150.0, 299.9, -300.0])
    m = SP.quantity_masks(r, d0)
    rb = np.stack([m[b] for b in SP.R_BINS])
    db = np.stack([m[b] for b in SP.D0_BINS])
    assert (rb.sum(0) == 1).all() and (db.sum(0) == 1).all()           # every particle in exactly one bin of each
    assert list(np.argmax(rb, 0)) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert list(np.argmax(db, 0)) == [0, 1, 1, 2, 3, 3, 4, 6, 6, 7]
    assert np.array_equal(m["r_ge100"], r >= 100) and m["all"].all()
    assert SP.R_BINS == ("r_0_50", "r_50_100", "r_100_200", "r_200_300", "r_300_inf")


def test_innermost_radius():
    h = pd.DataFrame({"event": [1, 1, 1, 1, 2, 2], "pid": [7, 7, 7, 8, 7, 7],
                      "r": np.array([260.0, 116.6, 172.3, 33.0, 500.5, 820.0], dtype=np.float32)})
    t = SP.innermost_radius(h).set_index(["event", "particle_id"]).r_start
    assert t.loc[(1, 7)] == pytest.approx(116.6, abs=1e-4) and t.loc[(1, 8)] == pytest.approx(33.0)
    assert t.loc[(2, 7)] == pytest.approx(500.5, abs=1e-4) and t.dtype == np.float64


def test_parse_val_wp():
    assert SP._parse_val_wp("CC 0.36 (FR* unreachable: minimum-rate cut)") == ("cc", "0.36")
    assert SP._parse_val_wp("CC 0.38") == ("cc", "0.38")
    assert SP._parse_val_wp("mwslmerge 0.05 (trackbuild_val_lmerge)") == ("mwslmerge", "0.05")
    assert SP._parse_val_wp("acwalk 0.01,0.15,0.6 (trackbuild_val_acwalk_all)") == ("acwalk", "0.01,0.15,0.6")


def _resampler(b=300, seed=3) -> BT.Resampler:
    units = {"ttbar_val": np.arange(10, dtype=np.int64), "hnl_joint_val": np.arange(10, dtype=np.int64)}
    rng = np.random.default_rng(seed)
    w = {k: BS.make_weights(len(u), b, rng) for k, u in units.items()}
    return BT.Resampler(units, w, BT.VAL_CODE, "test tables")


def _pop(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    code = rng.choice([1, 2, 4], size=n)
    ev = rng.integers(0, 10, size=n)
    parent = np.where(rng.random(n) < 0.5, rng.choice([310, 3122, -3122], size=n), 0)
    conv = parent == 0
    return pd.DataFrame({
        "event_id": code.astype(np.int64) * 100_000_000 + ev, "particle_id": np.arange(n, dtype=np.int64) + 1000,
        "parent_pdg": parent, "pdg": np.where(conv, 11, 211), "gen": np.where(conv, 1, 0),
        "process": np.where(conv, 2, 0), "is_target": True,
        "r_start": rng.uniform(30, 1000, size=n), "d0": rng.normal(0, 30, size=n)})


def test_counts_differences_and_completeness():
    rs = _resampler()
    pop = _pop()
    rng = np.random.default_rng(5)
    ok_a = [(rng.random(len(pop)) < 0.95).astype(float) for _ in range(3)]
    ok_b = [np.minimum(o, (rng.random(len(pop)) < 0.9).astype(float)) for o in ok_a]   # b <= a particle by particle
    sa = [SP.seed_counts(rs, pop, o, "val") for o in ok_a]
    sb = [SP.seed_counts(rs, pop, o, "val") for o in ok_b]
    code = pop.event_id.to_numpy() // 100_000_000
    pm = SP.population_masks(pop)
    for (s, p, q), (pt, reps, k, n) in sa[0].items():
        m = np.isin(code, SP.EVENT_SETS["val"][s]) & pm[p] & SP.quantity_masks(pop.r_start, pop.d0)[q]
        assert n == m.sum() and k == ok_a[0][m].sum()
        if n:
            assert pt == pytest.approx(ok_a[0][m].mean()) and reps.shape == (rs.b,)
    a = SP.Model("canto_l", "A", "A", "g", ["r0", "r1", "r2"], ["w"] * 3, sa)
    b = SP.Model("x", "B", "B", "g", ["q0", "q1", "q2"], ["w"] * 3, sb)
    k = ("hnl_joint_val", "v0", "all")
    sm = SP.summary(a, k)
    assert sm["value"] == pytest.approx(np.mean([s[k][0] for s in sa]))
    assert sm["combined"] ** 2 == pytest.approx(sm["boot_sd"] ** 2 + sm["seed_sd"] ** 2 / 3)
    assert sm["n_pooled"] == 3 * sm["n_per_seed"]
    d = SP.difference(a, b, k)
    assert d["value"] == pytest.approx(sm["value"] - SP.summary(b, k)["value"]) and d["value"] >= 0
    z = SP.difference(a, a, k)
    assert z["value"] == 0 and z["boot_sd"] == pytest.approx(0, abs=1e-15)
    # a population particle missing in the per-particle table is an error
    pp = pop[["event_id", "particle_id"]].assign(dm_matched=True).iloc[1:]
    with pytest.raises(SystemExit):
        SP.matched_flags(pop, pp, "unit test")
    full = pop[["event_id", "particle_id"]].assign(dm_matched=ok_a[0].astype(bool))
    assert np.array_equal(SP.matched_flags(pop, full.sample(frac=1, random_state=1), "unit test"), ok_a[0])


def test_external_set_has_no_v0_rows():
    units = {"ttbar_test": np.arange(5, dtype=np.int64), "hnl_joint_test": np.arange(5, dtype=np.int64),
             "hnl15_100_external": np.arange(5, dtype=np.int64)}
    rng = np.random.default_rng(0)
    rs = BT.Resampler(units, {k: BS.make_weights(5, 50, rng) for k in units}, {1: "ttbar_test", 2: "hnl_joint_test",
                                                                                4: "hnl_joint_test",
                                                                                3: "hnl15_100_external"}, "t")
    pop = _pop(40)
    pop["event_id"] = (pop.event_id % 100_000_000) % 5 + 300_000_000           # all on the external set
    c = SP.seed_counts(rs, pop, np.ones(len(pop)), "test")
    assert not any(k[0] == "hnl15_100_external" and k[1] in ("v0", "k0s", "lambda") for k in c)
    assert c[("hnl15_100_external", "conv", "all")][3] == SP.population_masks(pop)["conv"].sum()


def test_val_entries_resolve():
    ents = SP.val_entries()
    assert [m["key"] for m in ents] == [m[0] for m in SP.MODELS]
    for m in ents:
        assert len(m["entries"]) == 3 and all(SP._seed_of(e["run"]) == i for i, e in enumerate(m["entries"]))
    canto = next(m for m in ents if m["key"] == "canto_l")
    assert [e["wp"] for e in canto["entries"]] == ["mwslmerge 0.05 (trackbuild_val_lmerge)",
                                                    "mwslmerge 0.03 (trackbuild_val_lmerge)",
                                                    "mwslmerge 0.1 (trackbuild_val_lmerge)"]
    names = " ".join(m["label"] for m in ents)
    assert "arm (" not in names and "acorn (paper)" not in names


def test_a2_rows_are_the_val_best_choices():
    """Round 3 (task F2): the learned-merging rows are task A2's val-best acorn configurations, seed by seed, on val
    and in A2's val-fixed test spec."""
    import json
    ents = {m["key"]: m for m in SP.val_entries()}
    a2v = pd.read_csv(SP._repo(SP.A2_VAL))
    spec = {m["label"]: m for m in json.load(open(SP._repo(SP.A2_TEST)))["models"]}
    tau = {"hnl_lm": ["0.03", "0.2", "0.05"], "ttbar_lm": ["0.03", "0.02", "0.05"], "mix_lm": ["0.2", "0.03", "0.05"]}
    for key, label, short, group, vf, vlab, tsrc, tlab in SP.MODELS:
        if key not in SP.LM_KEYS:
            continue
        wps = [e["wp"] for e in ents[key]["entries"]]
        assert wps == [f"mwslmerge {t} (trackbuild_val_fb3_all)" for t in tau[key]]
        best = a2v[a2v.label == SP._a2_best(group)].sort_values("run")
        assert list(best.working_point) == wps
        te, be = spec[tlab]["entries"], spec[SP._a2_best(group)]["entries"]
        assert [e["config"] for e in te] == [f"mwslmerge,{t}" for t in tau[key]]
        assert [(e["sub"], e["config"], e["merge_model_sha256"]) for e in te] == \
            [(e["sub"], e["config"], e["merge_model_sha256"]) for e in be]
        assert [e["val_wp"] for e in te] == wps
        assert "acorn (paper)" not in label and "arm (" not in label


def test_report_order_puts_val_best_first_in_each_group():
    labels = [m[1] for m in SP.MODELS]
    order = SP._order(labels[::-1] + ["unknown row"])
    groups = [next(m[3] for m in SP.MODELS if m[1] == x) for x in order[:-1]]
    assert groups == sorted(groups, key=SP.GROUP_ORDER.index) and order[-1] == "unknown row"
    for k in SP.LM_KEYS:
        g = next(m[3] for m in SP.MODELS if m[0] == k)
        first = next(x for x in order if next(m[3] for m in SP.MODELS if m[1] == x) == g)
        assert first == next(m[1] for m in SP.MODELS if m[0] == k)
