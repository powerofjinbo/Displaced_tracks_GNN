"""Sets, hash rule and seeds of addendum H (hnl.overlay.split_h)."""

import json
import os

import numpy as np

from hnl.overlay import split_h as S
from hnl.overlay import writer as W


def test_hash_rule_is_deterministic_and_total():
    ids = list(range(23000, 23900))
    o = S.hash_order(ids)
    assert sorted(o) == ids
    assert o == S.hash_order(list(reversed(ids)))
    r = S.htest_lists(ids)
    assert len(r["htest"]) == S.N_HTEST and r["htest"] == sorted(r["htest_hash_order"])
    assert r["htest_hash_order"] + r["spares_hash_order"] == o
    # a dropped host is replaced by the next spare: removing one H-test id shifts the first spare in
    first_spare = r["spares_hash_order"][0]
    r2 = S.htest_lists([i for i in ids if i != r["htest_hash_order"][0]])
    assert first_spare in r2["htest"]


def test_key_matches_plan_text():
    import hashlib
    assert S.key(S.HTEST_PREFIX, 23001) == int(hashlib.sha256(b"canto-trackml-overlay-v1:23001").hexdigest()[:8], 16)


def test_pilot_hosts_reproduce_hp_dry_run():
    with open(S.SPLIT_TP) as fh:
        val = json.load(fh)["val"]
    assert sorted(S.hash_order(val, S.PILOT_PREFIX)[:5]) == [21289, 21332, 21610, 21753, 21835]


def test_overlay_ids_agree_with_writer():
    for name in ("hval", "closure", "htest", "pilot"):
        assert S.overlay_id(name, 21117) == W.overlay_id(name, 21117)


def test_new_seed_streams_are_disjoint_and_below_2_32():
    streams = {n: S.new_streams(n) for n in S.PRODUCTION}
    for n, v in streams.items():
        assert v.max() < 2 ** 32
    names = list(streams)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.intersect1d(streams[a], streams[b]).size == 0
    # every earlier family (production samples, H-P runs, HNL pilots) ends below the new streams except the ACTS
    # default seed 1234567890, which starts above them
    lo = min(int(v[0]) for v in streams.values())
    hi = max(int(v[-1]) for v in streams.values())
    for f in S.families():
        top = f["base"] + f["offset"] + int(S.cantor(np.array([S.N_ALG]), np.array([f["events"][1]]))[0])
        assert top < lo or f["base"] + f["offset"] > hi, f


def test_committed_split_file_matches_make(tmp_path):
    out = tmp_path / "split.json"
    S.make_split(str(out))
    committed = os.path.join(S.REPO, "configs", "trackml", "split_h_v1.json")
    with open(committed) as fh:
        a = json.load(fh)
    with open(out) as fh:
        b = json.load(fh)
    assert a == b
    assert a["sets"]["htest"]["base_seed"] == S.SETS["htest"][1]
