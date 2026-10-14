"""Addendum G step T2 wiring (hnl.eval.addendum_g, addendum_g_readouts, addendum_g_summary): tar-shard inputs, union
split checks, stored tracks, the label-exchange emulation, the reweighting and the claim rules. No data needed."""

import io
import os
import subprocess
import tarfile

import numpy as np
import pandas as pd
import pytest

from hnl.data import acorn_store as AS
from hnl.data import convert as CV
from hnl.data.npz import Sample
from hnl.eval import addendum_g as G
from hnl.eval import addendum_g_readouts as R
from hnl.eval.addendum_g_t1 import split_of
from hnl.eval.matching import FastMatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------------------------- inputs
def chain_event(seed: int) -> dict:
    """A valid Zenodo-format event: 4 particles with 3-6 hits each, one chain per particle, on distinct modules."""
    rng = np.random.default_rng(seed)
    pids, xs, layers, edges = [], [], [], [[], []]
    parts = []
    row = 0
    for k in range(4):
        pid = (k + 1) << 40
        n = int(rng.integers(3, 7))
        r = np.sort(rng.uniform(30, 1000, n))
        for j in range(n):
            pids.append(pid)
            xs.append([r[j], rng.uniform(-np.pi, np.pi), rng.uniform(-1000, 1000)])
            layers.append((8 << 56) | ((2 * j + 2) << 48) | (k + 1))
            if j:
                edges[0].append(row - 1)
                edges[1].append(row)
            row += 1
        p = np.zeros(16)
        p[0], p[1], p[11] = pid, 13, 1.0
        p[7:10] = rng.normal(size=3) * 5
        p[13] = np.hypot(p[7], p[8])
        p[15] = np.arcsinh(p[9] / p[13])
        parts.append(p)
    n = len(pids)
    return {"x": np.array(xs, dtype=np.float32), "hid": np.arange(n, dtype=np.int64),
            "pid": np.array(pids, dtype=np.int64), "layers": np.array(layers, dtype=np.int64),
            "layerless_true_edges": np.array(edges, dtype=np.int64), "particles": np.array(parts)}


def npz_bytes(ev):
    buf = io.BytesIO()
    np.savez(buf, **ev)
    return buf.getvalue()


def test_shard_path_gives_the_npz_tables(tmp_path):
    """_convert_from_shards (tar members through hnl.data.shards) writes the tables the npz path writes."""
    from hnl.data import shards
    info = Sample("fake", "fake_v1", 3, "generic", False)
    files = tmp_path / "files"
    files.mkdir()
    tdir = tmp_path / "tars"
    tdir.mkdir()
    with tarfile.open(tdir / shards.shard_name(0, 2), "w", format=tarfile.USTAR_FORMAT) as tf:
        for i in range(3):
            data = npz_bytes(chain_event(i))
            (files / f"{i}.npz").write_bytes(data)
            ti = tarfile.TarInfo(f"{i}.npz")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    shards.list_shards.cache_clear()
    frames = {"hits": [], "particles": [], "edges": []}
    for i in range(3):
        with np.load(files / f"{i}.npz") as z:
            ev = {k: z[k] for k in z.files}
        for k, v in CV.event_tables(ev, "fake", i, info=info).items():
            frames[k].append(v)
    G._write_frames(frames, str(tmp_path / "a"), "fake", 0)
    G._convert_from_shards(("fake", str(tdir), info, 0, [0, 1, 2], str(tmp_path / "b")))
    for k in ("hits", "particles", "edges"):
        a = CV.read_table(str(tmp_path / "a"), k, "fake")
        b = CV.read_table(str(tmp_path / "b"), k, "fake")
        assert G._frame_diff(a, b) == "", k
        assert len(a)


def test_parse_event_id_multidigit(monkeypatch):
    monkeypatch.setitem(AS.CODE_SAMPLE, 13, "hnl10_100_ref")
    assert AS.parse_event_id("100000017") == ("ttbar", 17)
    assert AS.parse_event_id(AS.event_id("ttbar", 9999)) == ("ttbar", 9999)
    assert AS.parse_event_id("1300002999") == ("hnl10_100_ref", 2999)


def _lists(monkeypatch, v1, x1, union):
    lists = {("generic_iso", "test"): np.array(v1), ("generic_iso", "x1_test"): np.array(x1),
             ("generic_iso", "union_test"): np.array(union)}
    import hnl.data.splits as S
    monkeypatch.setattr(S, "read_split", lambda s, sp, *a, **k: lists[(s, sp)])


def test_union_checks(monkeypatch):
    x1 = [e for e in range(8000, 8200) if split_of("iso", e) == "test"][:5]
    other = [e for e in range(8000, 8200) if split_of("iso", e) != "test"][0]
    v1 = [3, 17, 7999]
    _lists(monkeypatch, v1, x1, sorted(v1 + x1))
    np.testing.assert_array_equal(G.split_events("generic_iso", "union_test"), sorted(v1 + x1))
    _lists(monkeypatch, v1, x1, sorted(v1 + x1[1:]))
    with pytest.raises(SystemExit):
        G.split_events("generic_iso", "union_test")
    _lists(monkeypatch, v1, x1 + [other], sorted(v1 + x1 + [other]))
    with pytest.raises(SystemExit):
        G.split_events("generic_iso", "union_test")
    _lists(monkeypatch, v1 + [8001], x1, sorted(v1 + x1 + [8001]))
    with pytest.raises(SystemExit):
        G.split_events("generic_iso", "union_test")


def test_fs_tars_and_sets():
    assert G.fs_tars("generic_iso", "union_test") == ["generic_iso_test.tar", "generic_iso_x1_test.tar"]
    assert G.SETS["test"][0] == ("generic_iso", "union_test")
    assert [s for s, _ in G.SETS["hnl"]] == list(G.POINTS_ORDER)
    assert set(G.SPLIT_STEM) == set(G.POINTS_ORDER) == set(AS.POINT_CODE)
    for p in G.POINTS_ORDER:
        assert os.path.exists(os.path.join(REPO, "splits", f"{G.SPLIT_STEM[p]}_all.txt"))


def test_tracks_frame_reproduces_matching():
    rng = np.random.default_rng(0)
    pid = rng.integers(1, 30, 400)
    lab = np.where(rng.random(400) < 0.2, -1, pid % 17)
    frame = G.tracks_frame({("500000017", 0): lab})
    assert (frame.event_id == 500000017).all() and (frame.track >= 0).all()
    track = np.full(400, -1)
    track[frame.row.to_numpy()] = frame.track.to_numpy()
    a, b = FastMatch(pid).match(lab), FastMatch(pid).match(track)
    np.testing.assert_array_equal(a["dm_matched"], b["dm_matched"])
    assert a["n_fake"] == b["n_fake"] and a["n_tracks_ge5"] == b["n_tracks_ge5"]


# ------------------------------------------------------------------------------------------------- emulation
def _modules(n_mod, k, seed=0):
    rng = np.random.default_rng(seed)
    geo = np.repeat(np.arange(n_mod) * 7 + 3, k)
    pid = np.arange(n_mod * k) + 100
    order = rng.permutation(len(pid))          # rows in any order
    return pid[order], geo[order]


def test_emulation_keeps_modules_counts_and_is_deterministic():
    pid, geo = _modules(500, 3)
    pid = pid % 40                               # some modules with repeated labels
    a = R.emulate_exchange(pid, geo, 5, 17)
    np.testing.assert_array_equal(a, R.emulate_exchange(pid, geo, 5, 17))
    assert not np.array_equal(a, R.emulate_exchange(pid, geo, 5, 18))
    for g in np.unique(geo):
        m = geo == g
        assert sorted(a[m]) == sorted(pid[m])
    np.testing.assert_array_equal(np.unique(a, return_counts=True)[1], np.unique(pid, return_counts=True)[1])


def test_emulation_rates_and_single_particle_modules():
    for k, p in ((2, 1961 / 17448), (3, 872 / 3542), (4, 202 / 777), (6, 52 / 188)):
        pid, geo = _modules(20000, k, seed=k)
        a = R.emulate_exchange(pid, geo, 6, k)
        changed = pd.Series(a != pid).groupby(geo).any()
        assert abs(changed.mean() - p) < 4 * np.sqrt(p * (1 - p) / len(changed)), k
        assert R.lx_rate(k) == pytest.approx(p)
    geo = np.repeat(np.arange(1000), 2)
    pid = np.repeat(np.arange(1000), 2)          # two hits of one particle per module: nothing to exchange
    np.testing.assert_array_equal(R.emulate_exchange(pid, geo, 5, 1), pid)
    np.testing.assert_array_equal(R.emulate_exchange(np.arange(5), np.arange(5), 5, 1), np.arange(5))


# ------------------------------------------------------------------------------------------------- read-outs
class Stub:
    """Minimal Data: per-model efficiency 'level' on every population, bootstrap replicates around it."""

    def __init__(self, t, level, fake, noise=0.0005):
        self.truth = {"generic_iso": t}
        self.level, self._fake, self.noise = level, fake, noise
        self.set = "test"

    def models(self):
        return list(self.level)

    def eff(self, m, s, mask, what="dm"):
        rng = np.random.default_rng([int(mask.sum()), sum(map(ord, m))])
        pts = [self.level[m] + d for d in (-0.001, 0.0, 0.001)]
        reps = [p + self.noise * rng.standard_normal(500) for p in pts]
        return pts, reps, int(mask.sum())

    def eff_w(self, m, s, mask, w, what="dm"):
        return self.eff(m, s, mask, what)[:2]

    def fake(self, m, s, col="n_fake", exclude_events=()):
        return [self._fake[m]] * 3, [np.full(500, self._fake[m])] * 3


def _truth(n=200):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"event": rng.integers(0, 50, n), "in_pop": True, "d0": rng.uniform(0, 600, n),
                         "r_v": rng.uniform(1, 800, n), "r_first": rng.uniform(30, 900, n),
                         "pt": rng.uniform(1, 30, n), "pdg": 13, "eta": rng.uniform(-3, 3, n)})


def test_claim_rules():
    t = _truth()
    hold = R.paired_claim(Stub(t, {"a": 0.95, "b": 0.955}, {"a": 0.010, "b": 0.012}), "a", "b", "noninferiority")
    assert hold["outcome"] == "holds" and hold["delta_points"] == pytest.approx(-0.5)
    low = R.paired_claim(Stub(t, {"a": 0.93, "b": 0.955}, {"a": 0.010, "b": 0.012}), "a", "b", "noninferiority")
    assert low["outcome"] == "does not hold, resolved below the margin"
    guard = R.paired_claim(Stub(t, {"a": 0.96, "b": 0.955}, {"a": 0.020, "b": 0.012}), "a", "b", "noninferiority")
    assert guard["outcome"] == "does not hold, inconclusive (fake-rate guard fails)"
    wide = R.paired_claim(Stub(t, {"a": 0.95, "b": 0.955}, {"a": 0.01, "b": 0.01}, noise=0.01), "a", "b",
                          "noninferiority")
    assert wide["outcome"] == "does not hold, inconclusive"
    g3 = R.paired_claim(Stub(t, {"a": 0.97, "b": 0.955}, {"a": 0.010, "b": 0.012}), "a", "b", "superiority")
    assert g3["outcome"] == "holds" and g3["z"] >= 3
    g3f = R.paired_claim(Stub(t, {"a": 0.97, "b": 0.955}, {"a": 0.013, "b": 0.012}), "a", "b", "superiority")
    assert g3f["outcome"] == "does not hold"
    ex = R.paired_claim(Stub(t, {"a": 0.95, "b": 0.955}, {"a": 0.010, "b": 0.012}), "a", "b", "noninferiority",
                        exclude_events=(int(t.event.iloc[0]),))
    assert ex["n_targets"] == int((t.event != t.event.iloc[0]).sum())


def test_bin_signs_rank_rules():
    t = _truth()
    rows = pd.DataFrame(R.bin_signs(Stub(t, {"canto_l_ttbar": 0.96, "acorn_hnl_lm": 0.95}, {})))
    d0 = rows[rows.variable == "|d0| [mm]"]
    assert not d0[d0.bin == "[300, inf)"].ranked.iloc[0] and d0[d0.bin == "[300, inf)"].delta_points.isna().all()
    assert (d0[d0.bin != "[300, inf)"].sign == "+").all()
    rv = rows[rows.variable == "r_v [mm]"]
    assert not rv[rv.bin == "[650, inf)"].ranked.iloc[0] and rv[rv.bin == "[450, 650)"].ranked.iloc[0]
    rf = rows[rows.variable == "first-hit radius [mm]"]
    assert not rf[rf.bin == "[700, inf)"].ranked.iloc[0]


def test_rv_weights(monkeypatch):
    import hnl.eval.addendum_g_truth as T
    t = _truth(4000)
    shape = pd.DataFrame({"lo": T.RV_SHAPE_EDGES[:-1], "hi": T.RV_SHAPE_EDGES[1:]})
    f = np.exp(-shape.lo.to_numpy() / 150.0)
    f[-3:] = 0.0
    shape["fraction"] = f / f.sum()
    monkeypatch.setattr(T, "read_shape", lambda: shape)
    W = R.rv_weights(Stub(t, {}, {}))
    w, info = W["Zenodo HNL r_v shape"]
    b = np.searchsorted(T.RV_SHAPE_EDGES, t.r_v.to_numpy(), side="right") - 1
    got = pd.Series(w).groupby(b).sum()
    np.testing.assert_allclose(got.to_numpy(), shape.fraction.to_numpy()[got.index], rtol=1e-12)
    assert info["uncovered_fraction"] == pytest.approx(0.0)
    w100, _ = W["exponential, lambda = 100 mm"]
    np.testing.assert_allclose(w100, np.exp(-t.r_v.to_numpy() / 100.0))


# ------------------------------------------------------------------------------------------------- guard
def test_guard_refuses_without_ledger_entry():
    ledger = open(os.path.join(REPO, "docs", "feedback", "test_use_ledger.md")).read()
    for s in ("test", "hss", "hnl", "train", "val"):
        r = subprocess.run(["bash", os.path.join(REPO, "slurm", "addendum_g", "guard.sh"), s], capture_output=True,
                           text=True)
        if s in ("train", "val") or "### R14. " not in ledger:
            assert r.returncode != 0, (s, r.stdout)
