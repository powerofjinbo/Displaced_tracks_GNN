"""hnl.overlay.ht1, the code of addendum H step H-T1, on synthetic inputs (truth only; no part03 event exists): the
converter split file that hnl.trackml.throughput accepts, the eligible ids and the 1 % stop, the hash lists, the
per-event screen with its drop chain on synthetic part03-like hosts, the write into a copy of split_h_v1.json, and the
hit digest of the earlier-read check."""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import ht1 as H
from hnl.overlay import split_h as SH
from hnl.trackml import convert as C
from hnl.trackml import throughput as T


def _manifest(d, events, incomplete=()):
    rows = []
    for e in events:
        for k in C.KINDS:
            if e in incomplete and k == "cells":
                continue
            rows.append({"archive": H.PART, "event": e, "kind": k, "member_index": 0, "offset_header": 0,
                         "offset_data": 0, "size": 10, "md5": "0" * 32, "isize": 10,
                         "name": f"event{e:09d}-{k}.csv.gz"})
    tsv, summ = T.manifest_paths(H.PART, d)
    pd.DataFrame(rows, columns=T.MEMBER_COLUMNS).to_csv(tsv, sep="\t", index=False)
    size, md5 = T.TP_ARCHIVES[H.PART + ".tar"]
    with open(summ, "w") as fh:
        json.dump({"archives": {H.PART + ".tar": {"size_match": True, "md5_match": True, "md5_streamed": md5,
                                                  "incomplete_events": list(incomplete)}}}, fh)


def test_split_tp03_is_read_by_the_converter(tmp_path):
    ev = list(range(23000, 23040))
    _manifest(str(tmp_path), ev, incomplete=(23007,))
    out = str(tmp_path / "split_tp03_v1.json")
    rec = H.write_split_tp03(out, str(tmp_path))
    s = T.load_tp_split(out)
    assert s["parts"] == [H.PART] and s["test"] == set(ev) - {23007}
    assert not (s["train"] | s["val"] | s["dev"] | s["calib"])
    assert rec["archives"][H.PART]["n_events"] == 39
    with pytest.raises(SystemExit):
        H.write_split_tp03(out, str(tmp_path))


def test_split_tp03_refuses_an_unverified_pass_one(tmp_path):
    _manifest(str(tmp_path), [23000, 23001])
    _, summ = T.manifest_paths(H.PART, str(tmp_path))
    d = json.load(open(summ))
    d["archives"][H.PART + ".tar"]["md5_match"] = False
    json.dump(d, open(summ, "w"))
    with pytest.raises(RuntimeError):
        H.split_tp03(str(tmp_path))


def _root_with_dq(base, name, ok, failed):
    root = base / name
    root.mkdir()
    rows = [{"archive": H.PART, "event": e, "status": "ok", "problems": ""} for e in ok]
    rows += [{"archive": H.PART, "event": e, "status": "dropped", "problems": "empty member"} for e in failed]
    pd.DataFrame(rows).to_csv(root / f"dq_{H.PART}.tsv", sep="\t", index=False)
    meta = {"n_events_in_archive": len(ok) + len(failed), "shards": {"23000": {"hits": {"events": sorted(ok)}}}}
    json.dump(meta, open(root / f"meta_{H.PART}.json", "w"))
    return str(root)


def test_eligible_and_the_one_percent_stop(tmp_path):
    ok = list(range(23000, 23200))
    r = H.eligible(_root_with_dq(tmp_path, "a", ok, [23500]))
    assert r["n_eligible"] == 200 and r["n_failed"] == 1 and not r["stop"] and r["eligible"] == ok
    r2 = H.eligible(_root_with_dq(tmp_path, "b", ok[:100], [23500, 23501]))
    assert r2["n_failed"] == 2 and r2["stop"]


def test_lists_follow_the_hash_rule():
    ids = list(range(23000, 23900))
    r = H.lists(ids)
    assert r["htest"] == sorted(SH.hash_order(ids)[:SH.N_HTEST])
    assert r["spares_hash_order"] == SH.hash_order(ids)[SH.N_HTEST:]
    assert r["n_eligible"] == 900 and r["fatras_rank_max_htest"] == SH.N_HTEST - 1


def test_screen_drop_chain_and_split_rule():
    ids = list(range(23000, 23900))
    r = H.lists(ids)
    order = SH.hash_order(ids)
    bad = {order[3], order[17], order[SH.N_HTEST]}           # two hosts and the first spare fail
    rec = H.screen(r, root="unused", check=lambda h, root: ["synthetic"] if h in bad else [])
    drops = rec["htest_dropped"]
    # host order[3] -> first spare (fails, dropped in turn), order[17] -> second spare, first spare -> third spare
    assert [d["host"] for d in drops] == [order[3], order[17], order[SH.N_HTEST]]
    assert [d["replacement"] for d in drops] == [order[SH.N_HTEST], order[SH.N_HTEST + 1], order[SH.N_HTEST + 2]]
    hosts = rec["evaluated_hosts"]
    assert len(hosts) == SH.N_HTEST and not (set(hosts) & bad)
    assert hosts == SH.htest_hosts({"htest": r["htest"], "htest_spares": r["spares_hash_order"],
                                    "htest_dropped": drops})


def test_write_split_on_a_copy(tmp_path):
    p = str(tmp_path / "split_h_v1.json")
    before = H.sha256_file(SH.OUT_SPLIT)
    s0 = SH.load_split()                                    # the copy as at H-T0, whatever the repository's state
    for k in ("htest", "htest_spares", "htest_dropped"):
        s0[k] = None
    s0["sha256"].pop("htest", None)
    with open(p, "w") as fh:
        json.dump(s0, fh, indent=1)
        fh.write("\n")
    ids = list(range(23000, 23900))
    rec = H.screen(H.lists(ids), root="unused", check=lambda h, root: [])
    w = H.write_split(rec, p)
    s = SH.load_split(p)
    assert SH.htest_hosts(s) == rec["evaluated_hosts"] and w["htest_sha256"] == SH._sha(rec["evaluated_hosts"])
    assert SH.fatras_event_numbers("htest", rec["evaluated_hosts"][:3], s)
    with pytest.raises(SystemExit):
        H.write_split(rec, p)
    assert H.sha256_file(SH.OUT_SPLIT) == before            # the repository's file is untouched


def test_hits_digest_order_independent_and_sensitive():
    h = pd.DataFrame({"x": np.float32([1.5, -2.25, 3.0]), "y": np.float32([0.5, 0.25, 7.0]),
                      "z": np.float32([10.0, -20.0, 30.0]), "volume": np.uint8([8, 13, 17]),
                      "layer": np.uint8([2, 4, 6]), "module": np.uint16([5, 6, 7])})
    d = H.hits_digest(h)
    assert H.hits_digest(h.iloc[::-1]) == d
    h2 = h.copy()
    h2.loc[1, "module"] = 9
    assert H.hits_digest(h2) != d


@pytest.mark.skipif(not os.path.exists(C.DETECTORS), reason="needs external/trackml/detectors.csv")
def test_host_problems_on_synthetic_part03_hosts(tmp_path):
    from test_overlay_writer import _store_host, host_tables_of
    det, mods = C.load_detectors(), pd.read_csv(C.DETECTORS)
    root = str(tmp_path / "tp03_v1")
    good, _ = host_tables_of(mods, det, 23001)
    bad, _ = host_tables_of(mods, det, 23017)
    w = bad["hits"].column("weight").to_numpy().copy()
    w[0] += 0.01                                             # host weights no longer sum to one
    import pyarrow as pa
    bad["hits"] = bad["hits"].set_column(bad["hits"].schema.get_field_index("weight"), "weight",
                                         pa.array(w, type=bad["hits"].schema.field("weight").type))
    _store_host(root, H.PART, {23001: good, 23017: bad})
    assert H.host_problems(23001, root) == []
    assert any("weight sum" in p for p in H.host_problems(23017, root))


def test_fatras_check_accepts_only_navigator_losses(tmp_path):
    d = tmp_path / "work" / "gun" / "fatras_out"
    d.mkdir(parents=True)
    nav = ("19:32:27    Simulation     ERROR     Step failed with MagneticFieldError:1: Interpolation out of bounds "
           "was requested")
    fail = ("19:32:27    FatrasSimula   ERROR     event 65 particle id=(1|1006|2|0|0)|pdg=K-|q=-1|m=0.493677|p=2.26 "
            "failed to simulate with error MagneticFieldError:1: Interpolation out of bounds was requested")
    (d / "fatras.log").write_text("\n".join(["19:30:00    Sequencer      INFO      start", nav, fail]) + "\n")
    r = H.fatras_check(str(tmp_path / "work"))
    assert r["pass"] and r["error_lines"] == 2 and r["failed_particles"] == 1 and r["failed_navigator_loss"] == 1
    (d / "fatras.log").write_text("\n".join([nav, fail, "19:33:00    Propagator     ERROR     Navigation failed"]) + "\n")
    r = H.fatras_check(str(tmp_path / "work"))
    assert not r["pass"] and r["other_error_lines"] == 1


def test_summary_inputs_must_pass():
    good = {"verify_tp03": {"pass": True}, "earlier_read": {"pass": True}, "eligible": {"stop": False},
            "fatras": {"pass": True}, "longstrip": {"check_nominal_rerun": {"pass": True}},
            "truth_checks": {"pass": {"format": True, "weights": True, "containment": True, "production": True,
                                      "version": True}}}
    assert H.check_inputs(good) == []
    assert H.check_inputs({**good, "truth_checks": {"pass": {**good["truth_checks"]["pass"], "containment": False}}})
    assert H.check_inputs({k: v for k, v in good.items() if k != "truth_checks"})
    for k, flag, v in (("verify_tp03", "pass", False), ("earlier_read", "pass", False), ("eligible", "stop", True),
                       ("fatras", "pass", False)):
        bad = {**good, k: {flag: v}}
        assert H.check_inputs(bad) and k in H.check_inputs(bad)[0]
    assert any("no fatras" in x for x in H.check_inputs({k: v for k, v in good.items() if k != "fatras"}))
    assert H.check_inputs({**good, "longstrip": {"check_nominal_rerun": {"pass": False}}})


def test_converter_against_manifest(tmp_path):
    v = json.load(open(os.path.join(H.REPO, "configs", "trackml", "overlay_version_h_v1.json")))
    man = v["manifest"]["code_sha256"]
    meta = tmp_path / "meta.json"
    json.dump({"code_sha256": {"convert.py": man["src/hnl/trackml/convert.py"],
                               "throughput.py": man["src/hnl/trackml/throughput.py"]}}, open(meta, "w"))
    assert H.converter_against_manifest(str(meta))["pass"]
    json.dump({"code_sha256": {"convert.py": "0" * 64, "throughput.py": man["src/hnl/trackml/throughput.py"]}},
              open(meta, "w"))
    assert not H.converter_against_manifest(str(meta))["pass"]
