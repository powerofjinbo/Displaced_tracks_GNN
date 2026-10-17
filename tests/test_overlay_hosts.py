"""hnl.overlay.hosts and the explicit host root of the gun driver and the writer (addendum H section 9.4): H-test hosts
come from derived/tp03_v1, named explicitly, and the other sets from derived/tp_v1. An H-test-like set of two synthetic
part03 hosts is made end to end from a dry tp03_v1 root (the gun driver's functions, synthetic Fatras output, assemble,
verify); another root is refused at every step, and the drivers' command lines take a dry host root for the pilot set
only (decision E4). Truth only; no model."""

import json
import os

import pandas as pd
import pytest

from hnl.overlay import digitise as D
from hnl.overlay import gun as GUN
from hnl.overlay import hosts as HR
from hnl.overlay import writer as W
from hnl.trackml import convert as C
from hnl.trackml import throughput as T

from test_overlay_version import fake_fatras
from test_overlay_writer import _store_host, host_tables_of

pytestmark = pytest.mark.skipif(not os.path.exists(C.DETECTORS), reason="needs external/trackml/detectors.csv")
HOSTS = (23001, 23017)                                  # part03-like ids; no tp_v1 event


@pytest.fixture(scope="module")
def det():
    return C.load_detectors()


@pytest.fixture(scope="module")
def modules():
    return pd.read_csv(C.DETECTORS)


def test_host_root_rules(tmp_path):
    assert HR.host_root("hval") == HR.DERIVED_TP == T.DERIVED_TP and HR.host_root("pilot") == T.DERIVED_TP
    assert HR.host_root("closure", T.DERIVED_TP) == T.DERIVED_TP
    assert HR.DERIVED_TP03.endswith("/derived/tp03_v1")
    with pytest.raises(ValueError, match="explicitly"):
        HR.host_root("htest")
    assert HR.host_root("htest", HR.DERIVED_TP03) == HR.DERIVED_TP03
    with pytest.raises(ValueError, match="tp03_v1"):
        HR.host_root("htest", T.DERIVED_TP)
    with pytest.raises(ValueError):
        HR.host_root("hval", HR.DERIVED_TP03)
    # a dry root: the pilot set only (as hnl.overlay.truth_checks.set_host_root), named like its root
    from hnl.overlay import truth_checks as TC
    assert HR.DRY_SETS == TC.DRY_HOST_ROOT_SETS == ("pilot",)
    dry = tmp_path / "derived" / "tp_v1"
    assert HR.host_root("pilot", str(dry), dry=True) == str(dry)
    with pytest.raises(ValueError, match="named tp_v1"):
        HR.host_root("pilot", str(tmp_path / "tp03_v1"), dry=True)
    for name in ("htest", "hval", "closure"):
        with pytest.raises(ValueError, match="pilot set only"):
            HR.host_root(name, str(tmp_path / "derived" / os.path.basename(HR.SET_HOST_ROOT[name])), dry=True)
    with pytest.raises(KeyError):
        HR.host_root("htrain")


@pytest.fixture(scope="module")
def dry_htest(modules, det, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("htest")
    hroot = str(tmp / "derived" / "tp03_v1")
    _store_host(hroot, "training_part03", {h: host_tables_of(modules, det, h)[0] for h in HOSTS})
    gdir = str(tmp / "gun")
    with pytest.raises(ValueError, match="pilot set only"):              # the driver: no dry root for H-test
        GUN.main(["--hosts", ":".join(map(str, HOSTS)), "--set", "htest", "--base-seed", "620000001",
                  "--fatras-events", "600:601", "--host-root", hroot, "--dry-host-root", "--out", gdir])
    assert not os.path.exists(gdir)
    # the driver's own steps on the dry root, as GUN.main runs them once host_root has resolved the root
    hosts = list(HOSTS)
    base, fe, rec0 = GUN.resolve_seeds("htest", hosts, 620000001, "600:601")
    GUN.make_gun_set(GUN.host_vertices(hosts, hroot), {h: W.overlay_id("htest", h) for h in hosts}, base, gdir,
                     fatras_events=fe, record={**rec0, "host_root": hroot})
    return {"tmp": tmp, "hroot": hroot, "gdir": gdir}


def test_gun_driver_reads_the_named_root(dry_htest):
    with open(os.path.join(dry_htest["gdir"], "gun_events.json")) as fh:
        gev = json.load(fh)
    assert gev["host_root"] == dry_htest["hroot"] and gev["set"] == "htest"
    assert [e["overlay_event"] for e in gev["events"]] == [52_023_001, 52_023_017]
    assert [e["fatras_event"] for e in gev["events"]] == [600, 601]
    hv = GUN.host_vertices(list(HOSTS), dry_htest["hroot"])
    assert [gev["events"][i]["hs_vertex"] for i in range(2)] == [hv[h] for h in HOSTS]
    out = str(dry_htest["tmp"] / "gun_refused")
    with pytest.raises(ValueError, match="explicitly"):
        GUN.main(["--hosts", "23001", "--set", "htest", "--base-seed", "620000001", "--fatras-events", "600",
                  "--out", out])
    with pytest.raises(ValueError, match="tp03_v1"):
        GUN.main(["--hosts", "23001", "--set", "htest", "--base-seed", "620000001", "--fatras-events", "600",
                  "--host-root", T.DERIVED_TP, "--out", out])
    with pytest.raises(KeyError):                         # the hosts are not in the dry root of another set
        GUN.host_vertices([21117], dry_htest["hroot"])
    assert not os.path.exists(out)


def test_writer_assembles_and_verifies_from_the_named_root(dry_htest):
    from hnl.overlay import emulate as EM
    gdir, hroot = dry_htest["gdir"], dry_htest["hroot"]
    sim = fake_fatras(gdir, os.path.join(gdir, "fatras_out"), 620_000_001 + 13)
    oroot = str(dry_htest["tmp"] / "overlay")
    os.makedirs(oroot)
    with pytest.raises(ValueError, match="gun read its hosts"):
        W.assemble("htest", gdir, sim, oroot, EM.TABLES, D.PATCHED_LOOKUP, host_root=T.DERIVED_TP)
    r = W.assemble("htest", gdir, sim, oroot, EM.TABLES, D.PATCHED_LOOKUP, host_root=hroot)
    assert r["meta"]["host_root"] == hroot and r["meta"]["n_events_converted"] == 2
    assert W.verify_set("htest", oroot, host_root=hroot)["problems"] == []
    with pytest.raises(ValueError, match="pilot set only"):
        W.main(["verify", "--set", "htest", "--root", oroot, "--host-root", hroot, "--dry-host-root"])
    with pytest.raises(ValueError, match="explicitly"):
        W.main(["verify", "--set", "htest", "--root", oroot])
    with pytest.raises(ValueError, match="tp03_v1"):
        W.main(["verify", "--set", "htest", "--root", oroot, "--host-root", T.DERIVED_TP])
