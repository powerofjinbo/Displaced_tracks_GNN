"""hnl.overlay.binned (addendum H section 7, efficiencies in bins) on synthetic inputs: the neighbour count equals the
addendum-G definition, the bins and their Clopper-Pearson intervals, the committed density tertiles, the summary layout,
and the refusals of explicit inputs. Truth or toy inputs only; no model output and no grant."""

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import binned as BN


def test_neighbours_equal_addendum_g():
    from hnl.eval.addendum_g_summary import neighbours as ag
    rng = np.random.default_rng(3)
    n = 400
    t = pd.DataFrame({"event": rng.integers(0, 4, n), "p_px": rng.normal(0, 1, n), "p_py": rng.normal(0, 1, n),
                      "p_pz": rng.normal(0, 2, n), "is_target": rng.random(n) < 0.6})
    t.loc[:40, ["p_px", "p_py", "p_pz"]] = [1.0, 0.5, 2.0]          # a tight bunch, so that some counts are > 0
    assert np.array_equal(BN.neighbours(t), ag(t)) and BN.neighbours(t).max() > 1
    from hnl.eval.addendum_g_summary import NB_BINS
    assert list(NB_BINS) == list(BN.NB_BINS)


def test_bins_and_intervals():
    t = pd.DataFrame({"x": [0.5, 1.5, 1.6, 3.0, 9.0], "dm": [True, False, True, True, False]})
    rows = BN.binned(t, "dm", ("x", [0, 1, 2, np.inf]))
    assert [(r["bin"], r["n"], r["k"]) for r in rows] == [("[0, 1)", 1, 1), ("[1, 2)", 2, 1), ("[2, inf)", 2, 1)]
    lo, hi = BN.cp68(1, 2)
    assert 0 < lo < 0.5 < hi < 1
    assert BN.cp68(0, 3)[0] == 0.0 and BN.cp68(3, 3)[1] == 1.0
    cat = BN.binned(pd.DataFrame({"s": ["mu", "pi", "mu"], "dm": [True, False, False]}), "dm", ("s", None))
    assert [(r["bin"], r["n"], r["k"]) for r in cat] == [("mu", 2, 1), ("pi", 1, 0)]


def test_density_edges_are_the_committed_tertiles():
    e = BN._density_edges()
    assert e[0] == 0.0 and e[1] == pytest.approx(4.0) and e[2] == pytest.approx(5.714285714285714) and e[3] == np.inf
    v = BN.variables()
    assert v["signed eta"][1] == BN.ETA_BINS and v["species"][1] is None and len(v) == 9


def test_summarize_layout():
    rng = np.random.default_rng(1)
    n = 60
    t = pd.DataFrame({"abs_d0": rng.uniform(0, 400, n), "rv": rng.uniform(0, 800, n), "r_first": rng.uniform(30, 900, n),
                      "ratio": rng.uniform(0, 1, n), "pt": rng.uniform(1, 30, n), "n_neigh": rng.integers(0, 4, n),
                      "eta": rng.uniform(-3.9, 3.9, n), "species": rng.choice(["mu", "pi", "e"], n),
                      "density": rng.uniform(0, 9, n), "population_pp": rng.random(n) < 0.5,
                      "population_pd": rng.random(n) < 0.3})
    for c in BN.CONDS:
        t[f"dm_{c}_WP-A"] = rng.random(n) < 0.8
    s = BN.summarize(t, ("WP-A",))
    assert set(s["WP-A"]) == set(BN.CONDS)
    pp = s["WP-A"]["full"]["P_P"]
    assert pp["all"]["n"] == int(t.population_pp.sum())
    assert sum(r["n"] for r in pp["bins"]["signed eta"]) == pp["all"]["n"]


def test_refusals():
    with pytest.raises(PermissionError):
        BN.run("closure", "/tmp/x")
    with pytest.raises(PermissionError):
        BN.run("pilot", "/tmp/x")
    with pytest.raises(PermissionError):
        BN.run("hval", "/tmp/x", grid_1gev="/tmp/g")
