"""hnl.overlay.m1: the field kick is removed from a segment's momentum change, a polar kick shows up in the polar-plane
width, the binned summary and its comparison, and the material-free propagation and layer crossing in a uniform
field (circle of radius pT / (0.3 B))."""

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import m1 as M1


def uniform(b=2.0):
    return lambda r, z: (np.zeros(np.shape(r)), np.full(np.shape(r), b))


def test_decompose_removes_the_field_and_keeps_a_polar_kick():
    x0 = np.array([[30.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    x1 = np.array([[130.0, 0.0, 50.0], [130.0, 0.0, 50.0]])
    p0 = np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    q = np.array([1.0, 1.0])
    kick = M1.field_kick(x0, x1, q, uniform())
    assert kick[0] == pytest.approx([0.0, -M1.K * 0.1 * 2.0, 0.0])
    polar = np.array([[-0.5, 0.0, 1.0]]) / np.sqrt(1.25) * 1e-3          # 1 MeV along the polar direction
    p1 = p0 + kick + np.vstack([np.zeros((1, 3)), polar])
    d = M1.decompose(x0, x1, p0, p1, q, np.array([211, 211]), uniform())
    assert d["m1"][0] == pytest.approx(0.0, abs=1e-9) and d["m2"][0] == pytest.approx(0.0, abs=1e-9)
    beta = d["p"][1] / np.sqrt(d["p"][1] ** 2 + M1.MASS[211] ** 2)
    assert abs(d["m2"][1]) == pytest.approx(1000 * beta * 1e-3, rel=0.02)
    assert abs(d["m1"][1]) < 0.05 * abs(d["m2"][1])


def test_summarise_and_compare():
    rng = np.random.default_rng(1)
    n = 1000
    t = pd.DataFrame({"m1": rng.normal(0, 1, n), "m2": rng.normal(0, 2, n), "p": np.ones(n), "dloss": np.full(n, -1e-3),
                      "abseta": rng.uniform(0, 0.5, n), "region": "pix_b", "species": "pi"})
    rows = M1.summarise(t, "a")
    assert len(rows) == 1 and rows[0]["abseta"] == "0-0.5" and rows[0]["segments"] == n
    assert rows[0]["W_polar_MeV"] == pytest.approx(2.0, rel=0.1) and rows[0]["median_dp_MeV"] == pytest.approx(-1.0)
    assert M1.summarise(t.iloc[:100], "a") == []                      # below MIN_SEGMENTS
    b = [{**rows[0], "W_polar_MeV": 1.0, "median_dp_MeV": -2.0}]
    c = M1.compare_rows(rows, b)
    assert c[0]["W_ratio"] == pytest.approx(1.0 / rows[0]["W_polar_MeV"], rel=1e-2) and c[0]["dE_ratio"] == 2.0


def test_propagate_and_crossing_follow_the_helix():
    pt, b = 1.0, 2.0
    radius = pt / (M1.K * b) * 1000.0                                  # mm
    traj = M1.propagate(np.zeros((1, 3)), np.array([[pt, 0.0, 0.0]]), np.array([1.0]), uniform(b), h=1.0,
                        nsteps=1500)
    pnt, i = M1.crossing(traj[0], "r", 500.0)
    assert i > 0
    # a circle through the origin with radius R: r = 2 R sin(phi_turn / 2), and its chord makes phi = -phi_turn / 2
    turn = 2 * np.arcsin(500.0 / (2 * radius))
    assert np.arctan2(pnt[1], pnt[0]) == pytest.approx(-turn / 2, abs=2e-4)   # positive charge turns clockwise
    assert M1.crossing(traj[0], "z", 10.0)[1] == -1


def test_region_and_species():
    assert M1.region_of([7, 8, 9, 12, 13, 14, 16, 17, 18]).tolist() == [
        "pix_ec", "pix_b", "pix_ec", "ss_ec", "ss_b", "ss_ec", "ls_ec", "ls_b", "ls_ec"]
    assert M1.species_of([13, -211, 11, 321]).tolist() == ["mu", "pi", "e", "x"]
