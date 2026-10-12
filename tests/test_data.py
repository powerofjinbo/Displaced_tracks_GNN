"""Unit tests of the reader layer (brief §5a) on a few real events per sample (marker `data`)."""

import numpy as np
import pytest

from hnl.data import scans
from hnl.data.npz import COL, SAMPLES, load_event, particle_ids
from hnl.geometry.ids import (BARREL_VOLUMES, PIXEL_VOLUMES, cylindrical_to_cartesian, decode_barcode,
                              decode_geometry_id)
from hnl.truth import labels as L

pytestmark = pytest.mark.data
EVENTS = [(s, i) for s in SAMPLES for i in (1, 2, 3, 500)]


@pytest.fixture(scope="module", params=EVENTS, ids=lambda p: f"{p[0]}-{p[1]}")
def ev(request):
    return request.param, load_event(*request.param)


def test_rphi_xy_round_trip(ev):
    _, e = ev
    x = e["x"]
    xyz = cylindrical_to_cartesian(x)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    assert np.allclose(r, x[:, 0], rtol=0, atol=1e-4)
    assert np.allclose(np.angle(np.exp(1j * (phi - x[:, 1]))), 0, atol=1e-6)
    assert np.array_equal(xyz[:, 2], x[:, 2].astype(np.float64))


def test_geometry_decoding(ev):
    _, e = ev
    d = decode_geometry_id(e["layers"])
    assert set(np.unique(d["volume"])) <= {7, 8, 9, 12, 13, 14, 16, 17, 18}
    assert (d["boundary"] == 0).all() and (d["approach"] == 0).all()
    assert (d["layer"] % 2 == 0).all() and d["layer"].min() >= 2 and d["layer"].max() <= 14
    assert (d["sensitive"] > 0).all()
    # barrel layers sit at their nominal radii; pixel hits are inside 200 mm
    r = e["x"][:, 0]
    barrel = np.isin(d["volume"], BARREL_VOLUMES)
    for v in BARREL_VOLUMES:
        for lay in np.unique(d["layer"][barrel & (d["volume"] == v)]):
            rr = r[barrel & (d["volume"] == v) & (d["layer"] == lay)]
            assert rr.max() - rr.min() < 30.0, (v, lay)
    assert r[np.isin(d["volume"], PIXEL_VOLUMES)].max() < 200.0


def test_edges_hits_particles(ev):
    _, e = ev
    pid, te = e["pid"], e["layerless_true_edges"]
    n_p = len(np.unique(pid))
    assert te.shape[1] == len(pid) - n_p
    assert (pid[te[0]] == pid[te[1]]).all()
    assert np.isin(pid, particle_ids(e["particles"])).all()
    assert np.array_equal(np.sort(e["hid"]), np.arange(len(pid)))


def test_labels_reproduce_column12(ev):
    (s, _), e = ev
    P = e["particles"]
    if not SAMPLES[s].labeled or SAMPLES[s].kind != "hnl":
        pytest.skip("rules are validated on labeled HNL samples")
    truth = L.labels_from_table(P)
    assert np.array_equal(L.w_muon_rule(P), truth["w_muon"])
    if truth["hnl_daughter"].sum() == 2:
        assert np.array_equal(L.hnl_daughter_rule_first_pair(P), truth["hnl_daughter"])


def test_barcode_fields_roundtrip():
    rng = np.random.default_rng(0)
    f = {"vprim": rng.integers(0, 4096, 50), "vsec": rng.integers(0, 4096, 50), "particle": rng.integers(0, 65536, 50),
         "gen": rng.integers(0, 256, 50), "sub": rng.integers(0, 65536, 50)}
    ids = ((f["vprim"].astype(np.uint64) << np.uint64(52)) | (f["vsec"].astype(np.uint64) << np.uint64(40))
           | (f["particle"].astype(np.uint64) << np.uint64(24)) | (f["gen"].astype(np.uint64) << np.uint64(16))
           | f["sub"].astype(np.uint64)).view(np.int64)
    d = decode_barcode(ids)
    for k in f:
        assert np.array_equal(d[k], f[k]), k


def test_corrupted_event_detector():
    """The scan's hard-range check flags the known corrupted events (and the 3 single-hit cases) but not a clean one;
    the dropping rule (envelope excess > 50 mm) selects exactly the 4 corrupted events listed in dropped_events.tsv."""
    import os
    import pandas as pd
    for s, i, bad in (("ttbar", 7500, True), ("hnl15_100", 4375, True), ("hnl15_200", 0, True),
                      ("hnl10_100", 1171, True), ("ttbar", 6759, True), ("hnl10_100", 1, False)):
        st, _ = scans.event_checks(load_event(s, i))
        flagged = (st["n_r_low"] + st["n_r_high"] + st["n_absz_high"]) > 0
        assert flagged == bad, (s, i)
    scan = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"),
                        "derived/scans/v1/event_scan.parquet")
    if not os.path.exists(scan):
        pytest.skip("scan output missing")
    bad = scans.corrupted_events(pd.read_parquet(scan))
    from hnl.data.splits import dropped_events
    assert {(s, int(i)) for s, i in zip(bad["sample"], bad["index"])} == \
        {(s, i) for s, v in dropped_events().items() for i in v}
    assert len(bad) == 4


def test_hash_helpers():
    a = np.arange(10, dtype=np.uint64)
    assert len(np.unique(scans.mix64(a))) == 10
    g = np.array([0, 0, 1, 1])
    v = np.array([5, 7, 7, 5], dtype=np.uint64)
    h = scans.multiset_hash(g, v, 2)
    assert h[0] == h[1]                                # order-independent
    assert scans.multiset_hash(g, np.array([5, 7, 7, 6], dtype=np.uint64), 2)[1] != h[1]
