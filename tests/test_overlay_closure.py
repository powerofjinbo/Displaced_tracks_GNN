"""hnl.overlay.closure: the V1 closure machinery. Ids (native + 2^62: positive int64, disjoint from host and gun
ids, exact inverse, read as int64 by pandas), barcodes, the population rules of plan section 10 V1, the prompt hash
order, the Fatras input with the native float32 kinematics, the closure members and event (the gun chain with another
id rule: byte-identical to hnl.overlay.writer on gun input), the merge that replaces the natives, the closure
converter gate (weight rule, kinematics equal to the native, mutations flagged), storage and verify. Synthetic hosts
on TrackML's detectors.csv modules."""

import gzip
import io
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from hnl.overlay import closure as CL
from hnl.overlay import digitise as D
from hnl.overlay import fatras_io as FIO
from hnl.overlay import gun as GUN
from hnl.overlay import writer as W
from hnl.trackml import convert as C

from test_overlay_writer import HOSTS, _place, _store_host, gun_event, host_tables_of

pytestmark = pytest.mark.skipif(not os.path.exists(C.DETECTORS), reason="needs external/trackml/detectors.csv")


@pytest.fixture(scope="module")
def det():
    return C.load_detectors()


@pytest.fixture(scope="module")
def modules():
    return pd.read_csv(C.DETECTORS)


# ------------------------------------------------------------------------------------------------ ids
def test_closure_ids_round_trip_and_ranges():
    native = np.array([(1 << 52) | (5 << 36), (226 << 52) | (7 << 36) | (2 << 24) | (3 << 12) | 9,
                       (1000 << 52) | 1], np.int64)
    cid = CL.closure_particle_id(native)
    assert cid.dtype == np.int64 and (cid > 0).all()
    assert np.array_equal(cid - native, np.full(3, 1 << 62))
    assert np.array_equal(cid >> 52, (native >> 52) + 1024)
    assert np.array_equal(CL.native_particle_id(cid), native)
    assert CL.is_closure_id(cid).all() and not CL.is_closure_id(native).any()
    f_n, f_c = GUN.trackml_fields(native), GUN.trackml_fields(cid)
    for k in ("primary", "generation", "secondary", "process"):
        assert np.array_equal(f_n[k], f_c[k])
    gun = GUN.trackml_particle_id(FIO.barcode(1, 1001, 1))
    assert not CL.is_closure_id(gun) and (cid >> 52).min() > W.GUN_VERTEX[1]
    assert CL.closure_particle_id(int(native[0])) == int(cid[0])
    for bad in (0, -5, 1001 << 52):
        with pytest.raises(ValueError):
            CL.closure_particle_id(bad)
    with pytest.raises(ValueError):
        CL.native_particle_id(native[0])
    # the largest closure id still reads back as int64 from a CSV (pandas would switch to uint64 above 2^63 - 1)
    top = CL.closure_particle_id((1000 << 52) | ((1 << 52) - 1))
    df = pd.read_csv(io.StringIO(f"particle_id\n{top}\n{int(cid[0])}\n"))
    assert df.particle_id.dtype == np.int64 and df.particle_id.iloc[0] == top


def test_closure_barcodes():
    from hnl.geometry.ids import decode_barcode
    bc = CL.closure_barcode(np.arange(1, 6))
    d = decode_barcode(bc.view(np.int64))
    assert (d["vprim"] == 2).all() and (d["vsec"] == 0).all() and d["particle"].tolist() == [1, 2, 3, 4, 5]
    assert (d["gen"] == 0).all() and (d["sub"] == 0).all()
    for bad in (0, 1 << 16):
        with pytest.raises(ValueError):
            CL.closure_barcode([bad])


def test_mass_table():
    assert np.allclose(CL.mass_of([211, -321, 2212, -11, 13]), [0.13957, 0.493677, 0.938272, 0.000510999, 0.105658])
    with pytest.raises(KeyError):
        CL.mass_of([211, 1000010020])


# ------------------------------------------------------------------------------------------------ populations
def test_classify_populations_and_regions():
    g = lambda gen: gen << 24                                          # noqa: E731
    p = pd.DataFrame({
        "particle_id": [(1 << 52) | (1 << 36), (1 << 52) | (2 << 36), (1 << 52) | (3 << 36) | g(1),
                        (1 << 52) | (4 << 36) | g(2), (1 << 52) | (5 << 36), (1 << 52) | (6 << 36),
                        (1 << 52) | (7 << 36), (1 << 52) | (8 << 36)],
        "d0": np.array([12.0, -0.5, -10.0, 9.9, 0.2, 0.9, 5.0, -15.0], np.float32),
        "rv": np.array([20.0, 150.0, 300.0, 300.0, 0.01, 0.5, 50.0, 30.0], np.float32),
        "eta": np.array([1.5, -2.0, 0.1, 0.1, -0.3, 2.5, 0.0, 0.0], np.float32),
        "is_target": [True, True, True, True, True, True, True, False]})
    pid = p.particle_id.to_numpy()
    h = pd.DataFrame({"pid": [pid[0], pid[1], pid[1], pid[4], pid[5], 0], "z": [1200.0, -50.0, -1001.0, 999.0,
                                                                                -1500.0, 3000.0]})
    c = CL.classify(p, h)
    assert c.population.tolist() == ["a", "a", "b", "", "c", "c", "", ""]
    assert c.region.tolist()[:6] == ["fwd+", "fwd-", "central", "central", "central", "fwd+"]
    assert c.gen.tolist()[:4] == [0, 0, 1, 2]


def test_prompt_subset_hash_order():
    import hashlib
    assert CL.prompt_key(21289, 4503599627370496) == int(hashlib.sha256(
        b"canto-trackml-overlay-v1-prompt:21289:4503599627370496").hexdigest()[:8], 16)
    c = pd.DataFrame({"event": np.repeat([21000, 21001], 50), "particle_id": np.tile(np.arange(1, 51) << 36, 2)})
    s, rec = CL.prompt_subset(c, n=10)
    assert rec["candidates"] == 100 and rec["n"] == 10 and s.in_subset.sum() == 10 and not rec["tie_at_cut"]
    k = s.prompt_key.to_numpy()
    assert np.all(np.diff(k) >= 0) and s.prompt_rank.tolist() == list(range(100))
    assert s.in_subset.to_numpy()[:10].all() and not s.in_subset.to_numpy()[10:].any()
    s2, _ = CL.prompt_subset(c.sample(frac=1, random_state=3), n=10)       # independent of the input order
    assert s2[["event", "particle_id"]].equals(s[["event", "particle_id"]])


def test_population_table_reads_val_hosts_only():
    with pytest.raises(PermissionError):
        CL.population_table([1000])
    with pytest.raises(PermissionError):
        CL.population_table([21000], require_all_val=False)                # 21000 is not a val event


def test_compare_val_regions(tmp_path):
    ref = {"counts": {"pp_gen0": {"central": 2, "fwd+": 1, "all": 3}, "d10_gen1": {"central": 1, "all": 1},
                      "prompt": {"fwd-": 4, "all": 4}}}
    path = tmp_path / "vr.json"
    path.write_text(json.dumps(ref))
    same = {"pp_gen0": {"central": 2, "fwd+": 1, "all": 3}, "d10_gen1": {"central": 1, "all": 1},
            "prompt": {"fwd-": 4, "all": 4}}
    assert CL.compare_val_regions(same, str(path))["equal"]
    r = CL.compare_val_regions({**same, "prompt": {"fwd-": 3, "all": 3}}, str(path))
    assert not r["equal"] and r["differences"]["prompt"] == {"fwd-": [3, 4], "all": [3, 4]}


# ------------------------------------------------------------------------------------------------ synthetic closure
def natives_of(host: dict, n: int = 4) -> pd.DataFrame:
    """Population-table rows for the first n host particles with hits (stored kinematics, flags, type)."""
    p = host["particles"].to_pandas()
    p["particle_type"] = host["ptype"].column("particle_type").to_numpy()
    p = p[p.n_hits > 0].head(n).copy()
    p["population"] = ["a", "b", "c", "a"][:n]
    p["region"] = "central"
    p["prompt_rank"] = np.where(p.population == "c", 7, -1)
    return p


def closure_digitised(modules, rows: pd.DataFrame, seed: int) -> D.Digitised:
    """Synthetic digitised hits of the closure particles (barcodes), 0-6 per particle at module centres."""
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(rows.itertuples()):
        n = 0 if i == 0 else int(rng.integers(1, 7))                     # the first one 'failed' in Fatras
        v, l, m, c = _place(modules, rng, n)
        for k in range(n):
            out.append((r.particle_id, c[k, 0], c[k, 1], c[k, 2], v[k], l[k], m[k], c[k, 0], c[k, 1], c[k, 2],
                        r.px, r.py, r.pz))
    h = pd.DataFrame(out, columns=["particle_id", "x", "y", "z", "volume_id", "layer_id", "module_id", "tx", "ty",
                                   "tz", "tpx", "tpy", "tpz"]).sample(frac=1, random_state=seed).reset_index(drop=True)
    h["particle_id"] = h.particle_id.astype(np.uint64)
    nc = rng.integers(1, 4, len(h))
    cells = pd.DataFrame({"hit": np.repeat(np.arange(len(h)), nc), "ch0": rng.integers(0, 100, nc.sum()),
                          "ch1": rng.integers(0, 100, nc.sum()), "value": rng.uniform(0.01, 0.2, nc.sum())})
    return D.Digitised(hits=h, cells=cells)


def closure_tables(modules, det, host, host_id, seed, work):
    """(closure-only converted tables, natives, closure rows, record, CSV bytes, overlay id) for one synthetic host;
    work: an empty directory for the Fatras input and the closure truth."""
    nat = natives_of(host)
    oid = W.overlay_id("closure", host_id)
    rec = CL.make_closure_set(nat.assign(event=host_id), {host_id: oid}, 990101, work, fatras_events={host_id: 0})
    truth = pd.read_parquet(os.path.join(work, "closure_truth.parquet"))
    fp = pd.read_csv(W.FIO_path(work, 0), dtype={"particle_id": np.uint64})
    rows = CL.closure_rows(truth, fp)
    dig = closure_digitised(modules, rows, seed)
    gz = W.members_gz(CL.members(dig, rows, rows.tml_id.to_numpy()))
    tab, dq = W.convert_gun(gz, oid, det)
    return tab, nat.particle_id.to_numpy(np.int64), rows, rec, gz, oid


def test_make_closure_set_and_rows(modules, det, tmp_path):
    host, _ = host_tables_of(modules, det, HOSTS[0])
    nat = natives_of(host).assign(event=HOSTS[0])
    oid = W.overlay_id("closure", HOSTS[0])
    rec = CL.make_closure_set(nat, {HOSTS[0]: oid}, 990101, str(tmp_path), fatras_events={HOSTS[0]: 0})
    assert rec["fatras_seed"] == 990114 and rec["events"][0]["overlay_event"] == oid == 51_021_117
    assert rec["events"][0]["fatras_event"] == 0
    assert rec["events"][0]["n_particles"] == 4 and rec["populations"] == {"a": 2, "b": 1, "c": 1}
    fp = pd.read_csv(W.FIO_path(str(tmp_path), 0), dtype={"particle_id": np.uint64})
    assert list(fp.columns) == FIO.PARTICLE_COLUMNS and (fp.vt == 0).all() and (fp.process == 0).all()
    truth = pd.read_parquet(tmp_path / "closure_truth.parquet")
    assert list(truth.columns) == CL.TRUTH_COLUMNS
    assert np.array_equal(truth.native_particle_id.to_numpy(), np.sort(nat.particle_id.to_numpy()))
    assert np.array_equal(truth.particle_id.to_numpy(), truth.native_particle_id.to_numpy() + (1 << 62))
    assert np.array_equal(fp.particle_id.to_numpy(), truth.acts_barcode.to_numpy())
    for k in ("vx", "vy", "vz", "px", "py", "pz"):                       # the native float32 values, exactly
        n = nat.set_index("particle_id")[k].reindex(truth.native_particle_id).to_numpy()
        assert np.array_equal(fp[k].to_numpy().astype(np.float32), n)
    rows = CL.closure_rows(truth, fp)
    assert np.array_equal(rows.tml_id.to_numpy(), truth.particle_id.to_numpy())
    bad = fp.copy()
    bad.loc[0, "px"] += 1e-3
    with pytest.raises(ValueError):
        CL.closure_rows(truth, bad)


def test_members_equal_the_gun_writer_on_gun_input(modules):
    dig, parts = gun_event(modules, 21)
    a = W.gun_members(dig, parts)
    b = CL.members(dig, parts, GUN.trackml_particle_id(parts.particle_id.to_numpy(np.uint64)))
    assert W.members_gz(a) == W.members_gz(b)
    with pytest.raises(ValueError):
        CL.members(dig, parts, np.zeros(len(parts), np.int64))


def _sim_hits(lookup, mods, parts, seed):
    """Synthetic Fatras sim hits: 2-8 crossings per particle at module centres along the module normal."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in parts.itertuples():
        for idx in range(int(rng.integers(2, 9))):
            i = int(rng.integers(0, len(lookup)))
            v, l, a, t = (int(x) for x in (lookup.volume[i], lookup.layer[i], lookup.key[i] % 10_000,
                                         lookup.trackml[i]))
            row = mods.index([v], [l], [t])[0]
            c, n = mods.c[row], mods.R[row][:, 2] + 0.2 * mods.R[row][:, 0]
            gid = (v << 56) | (l << 36) | a
            rows.append((r.particle_id, gid, *c, *(n * 3.0), idx))
    s = pd.DataFrame(rows, columns=["particle_id", "geometry_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "index"])
    s["particle_id"] = s.particle_id.astype(np.uint64)
    s["geometry_id"] = s.geometry_id.astype(np.uint64)
    return s


def test_closure_event_is_the_gun_chain():
    """closure_event with the gun's ids gives the same emulation, digitisation and CSV bytes as writer.gun_event."""
    from hnl.overlay import emulate as EM
    mods, lookup = D.Modules.load(), D.load_lookup(D.PATCHED_LOOKUP)
    tab = EM.Tables.load(EM.TABLES)
    parts, _ = GUN.make_gun_event({"vx": 0.0, "vy": 0.0, "vz": 0.0}, np.random.default_rng(5), 0, 0)
    s = _sim_hits(lookup, mods, parts, 5)
    oid = W.overlay_id("closure", 21289)
    g = W.gun_event(s, parts, lookup, mods, tab, 990101, oid)
    rows = parts.assign(tml_id=GUN.trackml_particle_id(parts.particle_id.to_numpy(np.uint64)))
    c = CL.closure_event(s, rows, lookup, mods, tab, 990101, oid)
    assert g["gz"] == c["gz"] and len(c["digitised"].hits) > 0
    assert np.array_equal(g["emulation"]["keep"], c["emulation"]["keep"])


def test_merge_and_closure_gate(modules, det, tmp_path):
    host, _ = host_tables_of(modules, det, HOSTS[0])
    tab, natives, rows, _, gz, oid = closure_tables(modules, det, host, HOSTS[0], 2, str(tmp_path))
    p = tab["particles"].to_pandas()
    assert (p.vprim >= 1024).all() and not p.is_hard_scatter.any() and (p.n_hits.iloc[0] == 0)
    assert (tab["hits"].column("weight").to_numpy() == 0).all()
    w_rm, n_rm = CL.removed_weight(host, natives)
    assert w_rm > 0 and n_rm > 0
    merged, rm = W.merge_event(host, tab, oid, drop_pids=natives)
    assert rm["info"]["dropped_host_hits"] == n_rm
    assert CL.check_closure_event(merged, host, natives, 1.0 - w_rm) == []
    # the frozen converter recomputes the derived quantities bit-identically from the native float32 values
    hp = host["particles"].to_pandas().set_index("particle_id")
    cl = merged["particles"].to_pandas().iloc[-len(natives):]
    nat = hp.loc[CL.native_particle_id(cl.particle_id.to_numpy())]
    for k in ("pt", "eta", "rv", "d0", "vx", "pz"):
        assert np.array_equal(cl[k].to_numpy(), nat[k].to_numpy()), k
    # problems are flagged
    assert any("weight sum" in x for x in CL.check_closure_event(merged, host, natives, 1.0))
    bad_drop, _ = W.merge_event(host, tab, oid, drop_pids=natives[1:])
    assert "a replaced native keeps hits" in CL.check_closure_event(bad_drop, host, natives, 1.0 - w_rm)

    def mutate(table, col, i, fn):
        m = dict(merged)
        a = m[table].column(col).to_numpy().copy()
        a[i] = fn(a[i])
        m[table] = m[table].set_column(m[table].schema.get_field_index(col), col, pa.array(a))
        return m

    np_ = merged["particles"].num_rows
    assert "closure pt differs from the native" in CL.check_closure_event(
        mutate("particles", "pt", np_ - 1, lambda x: x * 1.001), host, natives, 1.0 - w_rm)
    nh = merged["hits"].num_rows
    assert "closure hit weight != 0" in CL.check_closure_event(
        mutate("hits", "weight", nh - 1, lambda x: 0.5), host, natives, 1.0 - w_rm)
    assert "host hits.x altered" in CL.check_closure_event(mutate("hits", "x", 0, lambda x: x + 1), host, natives,
                                                          1.0 - w_rm)
    assert "closure rows differ from the replaced natives" in CL.check_closure_event(merged, host,
                                                                                     np.sort(natives)[:-1], 1.0 - w_rm)
    # the gun gate refuses the closure event (closure ids are not gun ids), so it cannot pass by mistake
    assert any("vertex field" in x for x in W.check_event(merged, host, len(natives), 1.0 - w_rm, natives))


def test_merged_closure_event_equals_one_conversion(modules, det, tmp_path):
    """Host rows without the natives followed by the closure rows equal the frozen conversion of the host CSV without
    the natives plus the closure-only CSV written as one event."""
    host, hmem = host_tables_of(modules, det, HOSTS[1])
    tab, natives, rows, _, gz, oid = closure_tables(modules, det, host, HOSTS[1], 3, str(tmp_path))
    merged, _ = W.merge_event(host, tab, oid, drop_pids=natives)
    t = hmem["truth"]
    keep_h = ~np.isin(t.particle_id.to_numpy(), natives)
    new_id = np.cumsum(keep_h)                                           # host hit ids renumbered 1..n_kept
    hid_map = dict(zip(t.hit_id.to_numpy()[keep_h], new_id[keep_h]))
    hh = hmem["hits"][keep_h].assign(hit_id=lambda d: d.hit_id.map(hid_map))
    ht = t[keep_h].assign(hit_id=lambda d: d.hit_id.map(hid_map))
    hc = hmem["cells"][hmem["cells"].hit_id.isin(list(hid_map))].assign(hit_id=lambda d: d.hit_id.map(hid_map))
    hpart = hmem["particles"][~hmem["particles"].particle_id.isin(natives)]
    cm = {k: pd.read_csv(io.BytesIO(gzip.decompress(gz[k]))) for k in gz}
    n = int(keep_h.sum())
    comb = {"hits": pd.concat([hh, cm["hits"].assign(hit_id=cm["hits"].hit_id + n)]),
            "cells": pd.concat([hc, cm["cells"].assign(hit_id=cm["cells"].hit_id + n)]),
            "particles": pd.concat([hpart, cm["particles"]]),
            "truth": pd.concat([ht, cm["truth"].assign(hit_id=cm["truth"].hit_id + n)])}
    from hnl.trackml import throughput as T
    ref, dq = T.event_tables_tp(W.members_gz({k: v.reset_index(drop=True) for k, v in comb.items()}), oid, det,
                                keep_weight=True, keep_truth=False)
    assert dq["problems"] == []
    for k in W.TABLES:
        if k == "particles":                     # category is tagged per conversion: compare the other columns
            a, b = merged[k].drop_columns(["category"]), ref[k].drop_columns(["category"])
            assert W.tables_equal(a, b)
        else:
            assert W.tables_equal(merged[k], ref[k]), k


def test_write_and_verify_closure_set(modules, det, tmp_path):
    hroot, oroot = str(tmp_path / "host"), str(tmp_path / "overlay")
    hosts = {h: host_tables_of(modules, det, h)[0] for h in HOSTS}
    _store_host(hroot, "training_part01", hosts)
    merged, side = {}, []
    for i, h in enumerate(HOSTS):
        tab, natives, _, _, _, oid = closure_tables(modules, det, hosts[h], h, 10 + i, str(tmp_path / f"w{i}"))
        w_rm, n_rm = CL.removed_weight(hosts[h], natives)
        merged[oid], rm = W.merge_event(W.host_tables(h, hroot), tab, oid, drop_pids=natives)
        side.append({**rm["info"], "overlay_event": oid, "host": h, "n_gun_particles": 0,
                     "n_closure_particles": len(natives), "native_particle_ids": natives.tolist(),
                     "expected_host_weight_sum": 1.0 - w_rm})
    os.makedirs(oroot)
    W.write_set("closure", merged, oroot)
    with open(os.path.join(oroot, "sidecar_closure.json"), "w") as fh:
        json.dump({"events": side}, fh)
    v = CL.verify_closure_set("closure", oroot, host_root=hroot)
    assert v["problems"] == [] and v["events"] == 2
    assert W.verify_set("closure", oroot, host_root=hroot)["problems"]    # the gun verifier refuses closure ids
    from hnl.trackml import events as EV
    oid = W.overlay_id("closure", HOSTS[0])
    ev = EV.load_events([oid], root=oroot)[oid]
    assert ev.particles.particle_id.is_unique and CL.is_closure_id(ev.pid[ev.pid != 0]).any()
    side[0]["expected_host_weight_sum"] = 1.0
    with open(os.path.join(oroot, "sidecar_closure.json"), "w") as fh:
        json.dump({"events": side}, fh)
    assert any("weight sum" in p for p in CL.verify_closure_set("closure", oroot, host_root=hroot)["problems"])


def test_checks_pair_natives_with_their_closure_rows(modules, det, tmp_path):
    """hnl.overlay.closure_checks on one synthetic closure event: the converter view passes, the pairs and the
    descriptive track and hit tables are built per population."""
    from hnl.overlay import closure_checks as CC
    host, _ = host_tables_of(modules, det, HOSTS[0])
    tab, natives, rows, _, _, oid = closure_tables(modules, det, host, HOSTS[0], 4, str(tmp_path))
    w_rm, _ = CL.removed_weight(host, natives)
    merged, rm = W.merge_event(host, tab, oid, drop_pids=natives)
    truth = pd.read_parquet(tmp_path / "closure_truth.parquet")
    truth["sim_status"] = np.where(np.arange(len(truth)) == 0, "failed", "simulated")
    side = {"host": HOSTS[0], "n_closure_particles": len(natives), "n_closure_hits": rm["info"]["n_gun_hits"],
            "native_particle_ids": natives.tolist(), "expected_host_weight_sum": 1.0 - w_rm}
    ctx = {"events": {oid: side}, "hosts": {oid: host}, "stored": {oid: merged}, "truth": truth}
    pr = CC.pairs(ctx)
    assert len(pr) == len(natives) and pr.population.tolist() == truth.population.tolist()
    v = CC.converter_view(ctx, pr)
    assert v["pass"] and all(v["exactly_equal"].values()) and v["vprim_offset_1024"]
    first = truth.population.iloc[0]                                     # the particle without hits
    st = CC.sim_status(pr)
    assert st[first]["failed"] == 1 and sum(v.get("failed", 0) for v in st.values()) == 1
    assert sum(sum(v.values()) for v in st.values()) == 4
    tp = CC.track_pairs(ctx, pr)
    assert tp.n_hits_n.tolist() == pr.n_hits_n.tolist() and tp.n_hits_c.iloc[0] == 0
    s = {(r["population"], r["region"]): r for r in CC.summarise_tracks(tp)}
    assert s[("all", "all")]["natives"] == 4 and s[(first, "all")]["fatras_failed"] == 1
    hp = CC.hit_pairs(ctx, pr)
    assert set(hp.columns) >= {"region", "d_mm", "ncells_n", "ncells_c"}
    assert CC.summarise_hits(hp)[0]["hit_pairs"] == len(hp)


def test_hits_per_layer_counts_native_sim_and_closure(monkeypatch):
    """closure_checks.hits_per_layer on a hand-made event: one native with two hits on one pixel-barrel layer, its two
    sim hits and one stored closure hit."""
    from hnl.overlay import closure_checks as CC
    nat = (1 << 52) | (3 << 36)
    cid = CL.closure_particle_id(nat)
    bc = int(CL.closure_barcode([1])[0])
    host_hits = pa.table({"pid": np.array([nat, nat, 0], np.int64), "volume": np.array([8, 8, 8], np.uint8),
                          "layer": np.array([2, 2, 4], np.uint8)})
    st_hits = pa.table({"pid": np.array([cid], np.int64), "volume": np.array([8], np.uint8),
                        "layer": np.array([2], np.uint8)})
    truth = pd.DataFrame({"event": [51_000_001], "fatras_event": [0], "native_particle_id": [nat],
                          "acts_barcode": np.array([bc], np.uint64), "eta": [0.5]})
    sim = pd.DataFrame({"event": [0, 0], "particle_id": np.array([bc, bc], np.uint64),
                        "geometry_id": np.array([(8 << 56) | (2 << 36) | 1, (8 << 56) | (2 << 36) | 2], np.uint64)})

    class Lk:
        def from_geometry_id(self, g):
            g = np.asarray(g, np.uint64)
            return ((g >> np.uint64(56)).astype(np.int64), ((g >> np.uint64(36)) & np.uint64(0xFFF)).astype(np.int64),
                    np.ones(len(g), np.int64))

    monkeypatch.setattr(D, "load_lookup", lambda path: Lk())
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns=None: sim)
    monkeypatch.setattr(CC, "closure_slices", lambda ctx, e: {"hits": st_hits})
    ctx = {"meta": {"lookup": "x"}, "sim_path": "x", "events": {51_000_001: {}}, "truth": truth,
           "hosts": {51_000_001: {"hits": host_hits}}}
    r = CC.hits_per_layer(ctx)
    assert len(r) == 1 and r[0]["region"] == "pixel barrel" and r[0]["abs_eta"] == "[0.0, 1.0)"
    assert (r[0]["hits_native"], r[0]["hits_sim"], r[0]["hits_closure"]) == (2, 2, 1)
    assert (r[0]["layers_native"], r[0]["layers_sim"], r[0]["layers_closure"]) == (1, 1, 1)


@pytest.mark.skipif(not os.path.exists(CL.SPLIT_H), reason="needs configs/trackml/split_h_v1.json")
def test_default_base_seed_is_the_split_h_closure_seed():
    with open(CL.SPLIT_H) as fh:
        s = json.load(fh)["sets"]["closure"]
    assert CL.default_base_seed() == int(s["base_seed"]) and s["k"] == W.SETS["closure"][0] == 1
    assert CL.default_base_seed() == W.SETS["closure"][1] == 610_000_001


def test_make_closure_set_numbers_hosts_by_the_closure_list(modules, det, tmp_path):
    """Fatras event number = position of the host in the split file's closure list, also for a job with two hosts far
    apart in that list; a host's Fatras input does not depend on the other hosts of the job."""
    from hnl.overlay import split_h as SH
    val = SH.load_split()["closure"]
    h1, h2 = val[3], val[120]
    host, _ = host_tables_of(modules, det, HOSTS[0])
    nat = natives_of(host)
    pop = pd.concat([nat.assign(event=h1), nat.assign(event=h2)], ignore_index=True)
    fe = SH.fatras_event_numbers("closure", [h1, h2])
    assert fe == {h1: 3, h2: 120}
    rec = CL.make_closure_set(pop, {h: W.overlay_id("closure", h) for h in (h1, h2)}, 610_000_001,
                              str(tmp_path / "two"), fatras_events=fe)
    assert {e["host"]: e["fatras_event"] for e in rec["events"]} == fe
    CL.make_closure_set(pop[pop.event == h2], {h2: W.overlay_id("closure", h2)}, 610_000_001, str(tmp_path / "one"),
                        fatras_events={h2: 120})
    a = open(W.FIO_path(str(tmp_path / "two"), 120), "rb").read()
    assert a == open(W.FIO_path(str(tmp_path / "one"), 120), "rb").read()
    assert not os.path.exists(W.FIO_path(str(tmp_path / "two"), 0))
    t = pd.read_parquet(tmp_path / "two" / "closure_truth.parquet")
    assert dict(zip(t.host, t.fatras_event)) == fe
    with pytest.raises(ValueError):
        CL.make_closure_set(pop, {h: W.overlay_id("closure", h) for h in (h1, h2)}, 610_000_001,
                            str(tmp_path / "bad"), fatras_events={h1: 3})
