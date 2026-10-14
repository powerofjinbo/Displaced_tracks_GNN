"""Tar-shard reader of the HNL points (hnl.data.shards): byte-identical events, ranges, extraction (no data needed)."""

import io
import os
import tarfile

import numpy as np
import pytest

from hnl.data import shards
from hnl.data.npz import NPZ_KEYS


def fake_event(i: int) -> dict:
    rng = np.random.default_rng(i)
    n = 5 + i
    return {
        "x": rng.normal(size=(n, 3)).astype(np.float32), "hid": np.arange(n, dtype=np.int64),
        "pid": rng.integers(1, 4, n).astype(np.int64), "layers": rng.integers(0, 9, n).astype(np.int64),
        "layerless_true_edges": np.zeros((2, 0), dtype=np.int64), "particles": rng.normal(size=(3, 16)),
    }


def npz_bytes(ev):
    buf = io.BytesIO()
    np.savez(buf, **ev)
    return buf.getvalue()


def write_shard(directory, first, last):
    path = os.path.join(directory, shards.shard_name(first, last))
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tf:
        for i in range(first, last + 1):
            data = npz_bytes(fake_event(i))
            ti = tarfile.TarInfo(f"{i}.npz")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


@pytest.fixture()
def sample_dir(tmp_path):
    d = tmp_path / "hnl_x_v1"
    d.mkdir()
    write_shard(str(d), 0, 3)
    write_shard(str(d), 4, 6)
    shards.list_shards.cache_clear()
    return str(d)


def test_load_and_iterate(sample_dir):
    ev = shards.load_event_from(sample_dir, 5)
    want = fake_event(5)
    assert {k: str(v.dtype) for k, v in ev.items()} == NPZ_KEYS
    for k in want:
        np.testing.assert_array_equal(ev[k], want[k])
    got = [i for i, _ in shards.iter_events_from(sample_dir)]
    assert got == list(range(7))
    assert [i for i, _ in shards.iter_events_from(sample_dir, [6, 1])] == [1, 6]
    np.testing.assert_array_equal(shards.event_indices_from(sample_dir), np.arange(7))
    assert shards.check(sample_dir)["events"] == 7
    with pytest.raises(IndexError):
        shards.load_event_from(sample_dir, 7)


def test_extract_byte_identical(sample_dir, tmp_path, monkeypatch):
    s = shards.Sample("tmp", "hnl_x_v1", 7, "hnl", True, root=os.path.dirname(sample_dir))
    monkeypatch.setitem(shards.POINTS, "tmp", s)
    out = shards.extract("tmp", str(tmp_path / "x"))
    assert sorted(os.listdir(out)) == sorted(f"{i}.npz" for i in range(7))
    with open(os.path.join(out, "2.npz"), "rb") as f:
        assert f.read() == npz_bytes(fake_event(2))


def test_overlap_rejected(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    write_shard(str(d), 0, 3)
    write_shard(str(d), 3, 5)
    shards.list_shards.cache_clear()
    with pytest.raises(ValueError):
        shards.list_shards(str(d))
