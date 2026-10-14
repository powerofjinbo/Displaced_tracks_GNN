"""Synthetic checks of hnl.eval.addendum_g_t1_x1 (no generated data read) and of the tar mode of
scripts/acts/csv_to_npz.py."""

import importlib.util
import os
import tarfile

import numpy as np
import pandas as pd
import pytest

from hnl.data import shards
from hnl.eval import addendum_g_t1_x1 as X

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv_to_npz():
    spec = importlib.util.spec_from_file_location("csv_to_npz", os.path.join(REPO, "scripts", "acts", "csv_to_npz.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _event(rng, n=20):
    return {"x": rng.normal(size=(n, 3)).astype(np.float32), "hid": np.arange(n, dtype=np.int64),
            "pid": rng.integers(1, 5, n).astype(np.int64), "layers": np.sort(rng.integers(1, 1 << 40, n)).astype(np.int64),
            "layerless_true_edges": rng.integers(0, n, (2, n - 2)).astype(np.int64),
            "particles": rng.normal(size=(4, 16)).astype(np.float64)}


def test_same_is_bitwise():
    a = np.array([1.0, np.nan])
    assert X.same(a, a.copy())
    assert not X.same(a, a.astype(np.float32))
    assert not X.same(np.array([0.0]), np.array([-0.0]))
    assert X.compare_arrays({"a": a, "b": a}, {"a": a}) == ["b"]


def test_tar_mode_roundtrip_and_repro(tmp_path, monkeypatch):
    C = _csv_to_npz()
    rng = np.random.default_rng(1)
    first, n = 100, 6
    v1 = tmp_path / "v1"
    v1t = tmp_path / "v1_truth"
    rep = tmp_path / "rep"
    rept = tmp_path / "rep_truth"
    for d in (v1, v1t, rep, rept):
        d.mkdir()
    evs = {i: _event(rng) for i in range(first, first + n)}
    for i, e in evs.items():
        with open(v1 / f"{i}.npz", "wb") as f:
            np.savez(f, **e)
    with tarfile.open(rep / shards.shard_name(first, first + n - 1), "w", format=tarfile.USTAR_FORMAT) as tf:
        for i, e in evs.items():
            C.add_member(tf, f"{i}.npz", C.npz_bytes(e))
    for stem in ("truth", "gun_truth"):
        side = {"p_x": rng.normal(size=5), "xmode": np.array("measured")}
        np.savez(v1t / f"{stem}_{first}.npz", **side)
        np.savez(rept / f"{stem}_{first}.npz", **side)
    monkeypatch.setattr(X, "V1_DIR", str(v1))
    out = tmp_path / "r.json"
    X.main(["repro", "--repro-dir", str(rep), "--pack-dir", str(tmp_path / "pack"), "--first", str(first),
            "--events", str(n), "--out", str(out)])
    import json
    r = json.load(open(out))
    assert r["pass"] and r["events_identical"] == n and r["reader_identity"]["bytes_identical"] == n
    # a one-bit change in one event is caught
    evs[first + 2]["x"][0, 0] = np.nextafter(evs[first + 2]["x"][0, 0], np.float32(10))
    os.remove(rep / shards.shard_name(first, first + n - 1))
    shards.list_shards.cache_clear()
    shards._members.cache_clear()
    with tarfile.open(rep / shards.shard_name(first, first + n - 1), "w", format=tarfile.USTAR_FORMAT) as tf:
        for i, e in evs.items():
            C.add_member(tf, f"{i}.npz", C.npz_bytes(e))
    shards._members.cache_clear()
    with pytest.raises(SystemExit):
        X.main(["repro", "--repro-dir", str(rep), "--pack-dir", str(tmp_path / "pack"), "--first", str(first),
                "--events", str(n), "--out", str(out)])
    r = json.load(open(out))
    assert not r["pass"] and r["n_events_different"] == 1 and r["events_different"][0]["keys"] == ["x"]


def _tables(rng, n1, n2, shift=0.0):
    rows = []
    for ev in range(n1 + n2):
        x1 = ev >= n1
        for k in range(rng.poisson(3)):
            sp = rng.choice(X.SPECIES)
            rv = rng.uniform(1, 800) + (shift if x1 else 0.0)
            for _ in range(2):          # two daughters share r_v and species
                rows.append({"event": ev, "is_target": True, "in_pop": True, "r_v": rv, "d0": rng.exponential(50),
                             "pt": rng.exponential(5), "eta": rng.uniform(-3, 3), "pdg": sp,
                             "n_hits": int(rng.integers(5, 15))})
        for _ in range(rng.poisson(20)):
            rows.append({"event": ev, "is_target": True, "in_pop": False, "r_v": np.nan, "d0": 0.1, "pt": 2.0,
                         "eta": 0.0, "pdg": 211, "n_hits": int(rng.integers(5, 15))})
    t = pd.DataFrame(rows)
    return t[t.event < n1], t[t.event >= n1], np.arange(n1), np.arange(n1, n1 + n2)


def test_compare_null_and_shift(monkeypatch):
    monkeypatch.setattr(X, "N_PERM", 300)
    rng = np.random.default_rng(7)
    out = X.compare(*_tables(rng, 300, 600))
    assert len(out) == X.N_TESTS and (out.p_perm > 0.002).all()
    out = X.compare(*_tables(rng, 300, 600, shift=150.0))
    r = out.set_index("quantity")
    assert r.loc["r_v of P_G targets", "p_perm"] < 0.01
    assert r.loc["targets per event (all)", "p_perm"] > 0.002


def test_shared_collisions():
    def t(events, vz):
        return pd.DataFrame({"event": events, "vsec": 0, "gen": 0, "sub": 0, "p_vx": 0.01, "p_vy": 0.02, "p_vz": vz})
    c = X.shared_collisions({"generic_iso_v1": t([1, 2], [3.0, 4.0]), "generic_iso_x1": t([8000, 8001], [5.0, 6.0]),
                             "generic_dense_v1": t([7, 9], [4.0, 6.0])})
    assert c["coinciding_groups"] == 2
    assert c["groups_with_extension"] == [["generic_dense_v1 9", "generic_iso_x1 8001"]]
