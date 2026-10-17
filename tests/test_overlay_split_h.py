"""Sets, hash rule and seeds of addendum H (hnl.overlay.split_h)."""

import json
import os

import numpy as np
import pytest

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


H_T1_FIELDS = ("htest", "htest_spares", "htest_dropped")


def _as_at_h_t0(split: dict) -> dict:
    """The split file as `make` wrote it at H-T0: H-T1 adds the H-test lists and their sha256 (hnl.overlay.ht1
    write-split)."""
    s = json.loads(json.dumps(split))
    for k in H_T1_FIELDS:
        s[k] = None
    s["sha256"].pop("htest", None)
    return s


def test_committed_split_file_matches_make(tmp_path):
    out = tmp_path / "split.json"
    S.make_split(str(out))
    committed = os.path.join(S.REPO, "configs", "trackml", "split_h_v1.json")
    with open(committed) as fh:
        a = json.load(fh)
    with open(out) as fh:
        b = json.load(fh)
    assert _as_at_h_t0(a) == b
    assert a["sets"]["htest"]["base_seed"] == S.SETS["htest"][1]
    if a["htest"] is not None:                                     # after H-T1: the lists follow the rule
        assert S.fatras_order("htest", a) and a["sha256"]["htest"] == S.htest_sha256(a)


def test_base_seeds_of_the_split_file():
    split = S.load_split()
    for name, (k, base) in S.SETS.items():
        assert S.base_seed(name, split) == base == split["sets"][name]["base_seed"]
        assert split["sets"][name]["k"] == k
    assert S.base_seed("hval") == 600_000_001 and S.base_seed("pilot") == 990_101
    assert "htrain" not in split["sets"]
    with pytest.raises(KeyError):
        S.base_seed("htrain", split)


def test_fatras_event_numbers_of_the_val_sets_and_the_pilot():
    split = S.load_split()
    val = split["hval"]
    assert S.fatras_event_numbers("hval", [val[149], val[0], val[36]], split) == {val[149]: 149, val[0]: 0,
                                                                                    val[36]: 36}
    assert S.fatras_event_numbers("closure", val, split) == {h: i for i, h in enumerate(val)}
    # the pilot keeps the numbers of the H-P dry run (events 0-4 in increasing id)
    assert S.fatras_event_numbers("pilot", "21835:21289", split) == {21835: 4, 21289: 0}
    assert S.fatras_event_numbers("hval", "21117,21124") == {21117: 0, 21124: 1}       # default: the committed file
    with pytest.raises(KeyError):
        S.fatras_event_numbers("pilot", [21117], split)
    with pytest.raises(KeyError):
        S.fatras_event_numbers("hval", [22000], split)                                   # not a val event
    with pytest.raises(ValueError):
        S.fatras_event_numbers("hval", [val[0], val[0]], split)
    with pytest.raises(LookupError):
        S.fatras_event_numbers("htest", [val[0]], split)                                 # ids exist only at H-T1


def _split_with_htest(eligible):
    r = S.htest_lists(eligible)
    return {**S.load_split(), "htest": r["htest"], "htest_spares": r["spares_hash_order"], "htest_dropped": None}, r


def test_htest_fatras_numbers_are_hash_ranks_that_drops_and_job_splits_keep():
    eligible = list(range(23000, 24200))
    split, r = _split_with_htest(eligible)
    order = r["htest_hash_order"] + r["spares_hash_order"]
    full = S.fatras_event_numbers("htest", order[:900], split)
    assert full == {h: i for i, h in enumerate(order[:900])}
    assert sorted(full[h] for h in r["htest"]) == list(range(S.N_HTEST))
    # a job with a subset of the hosts gives each host the same number
    part = order[100:110] + order[400:405]
    assert S.fatras_event_numbers("htest", part, split) == {h: full[h] for h in part}
    # a dropped host is replaced by the next spare, which keeps its own rank; no other number moves
    dropped, spare = order[17], r["spares_hash_order"][0]
    hosts = [h for h in order[:S.N_HTEST] if h != dropped] + [spare]
    fe = S.fatras_event_numbers("htest", hosts, split)
    assert fe[spare] == S.N_HTEST and all(fe[h] == full[h] for h in hosts if h != spare)
    # ranks beyond the seed check are refused until the check is extended
    with pytest.raises(ValueError, match="seed check"):
        S.fatras_event_numbers("htest", [order[S.N_EVENT_NEW]], split)
    # lists that break the hash rule are refused
    bad = {**split, "htest": sorted(order[1:S.N_HTEST + 1]), "htest_spares": [order[0]] + order[S.N_HTEST + 1:]}
    with pytest.raises(ValueError, match="hash rule"):
        S.fatras_event_numbers("htest", [order[5]], bad)


def test_committed_split_file_states_the_h_t0_rules():
    split = S.load_split()
    fr = split["fatras_event_rule"]
    assert set(fr) >= {"hval", "closure", "htest", "pilot", "fixed", "seed_check", "streams", "code"}
    assert "0-149" in fr["hval"] and "0-599" in fr["htest"] and "keeps its own rank" in fr["fixed"]
    hr = split["htest_rule"]
    assert "without leading zeros" in hr["id_format"]
    assert len(hr["per_event_checks"]["checks"]) == 3 and len(hr["per_set_checks"]["checks"]) == 4
    assert "returns to H-T0.5 once" in hr["return_rule"] and "descriptive" in hr["return_rule"]
    note = split["sets"]["pilot"]["note"]
    assert "6820120" in note and "51,021,289-51,021,835" in note and "990101" in note
    if split["htest"] is None:                                     # before H-T1
        assert split["htest_spares"] is None and split["htest_dropped"] is None
    else:                                                          # after H-T1 (hnl.overlay.ht1 write-split)
        assert len(S.htest_hosts(split)) == S.N_HTEST and split["sha256"]["htest"] == S.htest_sha256(split)
    rec = hr["record_at_h_t1"]
    assert "htest_dropped" in rec and "htest_hosts" in rec and "htest_sha256" in rec and "R15" in rec


def test_htest_hosts_apply_the_drop_records_in_order():
    eligible = list(range(23000, 24200))
    split, r = _split_with_htest(eligible)
    order = r["htest_hash_order"] + r["spares_hash_order"]
    sp = r["spares_hash_order"]
    assert S.htest_hosts(split) == sorted(r["htest"])
    assert S.htest_sha256(split) == S._sha(sorted(r["htest"]))
    # two drops, the second one of the first replacement: each takes the next spare that has not joined the list
    drops = [{"host": order[17], "reason": "no hard-scatter vertex", "replacement": sp[0]},
             {"host": sp[0], "reason": "host weights do not sum to one", "replacement": sp[1]}]
    hosts = S.htest_hosts({**split, "htest_dropped": drops})
    assert hosts == sorted(set(r["htest"]) - {order[17]} | {sp[1]}) and len(hosts) == S.N_HTEST
    # the same hosts as the hash rule applied to the eligible events without the dropped ones
    assert hosts == S.htest_lists([i for i in eligible if i not in (order[17], sp[0])])["htest"]
    # every host keeps its Fatras event number: the spare its own rank
    fe = S.fatras_event_numbers("htest", hosts, split)
    assert fe[sp[1]] == S.N_HTEST + 1 and all(fe[h] == order.index(h) for h in hosts)
    for bad in ([{"host": order[17], "reason": "x", "replacement": sp[1]}],          # not the next spare
                [{"host": sp[5], "reason": "x", "replacement": sp[0]}],             # not in the list
                [{"host": order[17], "reason": " ", "replacement": sp[0]}]):        # no reason
        with pytest.raises(ValueError):
            S.htest_hosts({**split, "htest_dropped": bad})
    with pytest.raises(LookupError):
        S.htest_hosts(_as_at_h_t0(S.load_split()))                             # before H-T1
