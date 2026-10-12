"""Split integrity (brief §5c): disjoint, joint across the paired samples, dropped events absent, stable on disk."""

import os

import numpy as np
import pytest

from hnl.data import splits as S
from hnl.data.npz import SAMPLES


@pytest.fixture(scope="module")
def sp():
    return S.make_splits(seed=0)


def test_no_event_in_two_splits(sp):
    for s, parts in sp.items():
        names = list(parts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert len(np.intersect1d(parts[a], parts[b])) == 0, (s, a, b)


def test_cover_all_kept_events(sp):
    drop = S.dropped_events()
    for s, parts in sp.items():
        allv = np.sort(np.concatenate(list(parts.values())))
        kept = np.setdiff1d(np.arange(SAMPLES[s].n_events), sorted(drop[s]))
        assert np.array_equal(allv, kept), s


def test_dropped_events_absent(sp):
    drop = S.dropped_events()
    assert sum(len(v) for v in drop.values()) == 4
    for s, parts in sp.items():
        for idx in parts.values():
            assert not set(idx.tolist()) & drop[s]


def test_joint_split_holds(sp):
    a, b = S.JOINT
    drop = S.dropped_events()
    for name in ("train", "val", "test"):
        ia = set(sp[a][name].tolist()) | (drop[a] & set(S.assign(SAMPLES[a].n_events, 1)[name].tolist()))
        ib = set(sp[b][name].tolist()) | (drop[b] & set(S.assign(SAMPLES[b].n_events, 1)[name].tolist()))
        assert ia == ib, name
    # an index never sits in different splits of the two samples
    where = {}
    for s in S.JOINT:
        for name, idx in sp[s].items():
            for i in idx:
                assert where.setdefault(int(i), name) == name


def test_fractions(sp):
    for s in ("ttbar",) + S.JOINT:
        n = sum(len(v) for v in sp[s].values())
        assert abs(len(sp[s]["train"]) / n - 0.8) < 0.002
        assert abs(len(sp[s]["val"]) / n - 0.1) < 0.002


def test_external_sample_never_in_training(sp):
    assert set(sp[S.EXTERNAL]) == {"external"}


def test_files_on_disk_match(sp):
    for s, parts in sp.items():
        for name, idx in parts.items():
            path = os.path.join(S.SPLITS_DIR, f"{s}_{name}.txt")
            if not os.path.exists(path):
                pytest.skip("splits not written yet")
            assert np.array_equal(S.read_split(s, name), idx), path
