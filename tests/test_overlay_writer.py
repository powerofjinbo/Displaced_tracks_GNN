"""hnl.overlay.writer: gun-only TrackML members, the frozen conversion of the gun-only event, the merge with the host's
rows (the converter accepts the merged event unchanged: merging equals converting host and gun as one event), the
converter gate (no host row altered, ids unique and positive, gun vertex 1001-1010, gun weights 0, host weights sum
to one), storage in the converter's layout read back by hnl.trackml.convert.read_table and hnl.trackml.events, and the
overlay ids with the table_root override. Synthetic hosts and guns on TrackML's detectors.csv modules."""

import gzip
import io
import json
import os
import tarfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hnl.data.convert import SHARD
from hnl.overlay import digitise as D
from hnl.overlay import gun as GUN
from hnl.overlay import writer as W
from hnl.trackml import convert as C
from hnl.trackml import throughput as T

pytestmark = pytest.mark.skipif(not os.path.exists(C.DETECTORS), reason="needs external/trackml/detectors.csv")
HOSTS = (21117, 21124)


@pytest.fixture(scope="module")
def det():
    return C.load_detectors()


@pytest.fixture(scope="module")
def modules():
    return pd.read_csv(C.DETECTORS)


def _place(modules, rng, n):
    m = modules.iloc[rng.integers(0, len(modules), n)]
    return m.volume_id.to_numpy(), m.layer_id.to_numpy(), m.module_id.to_numpy(), m[["cx", "cy", "cz"]].to_numpy()


def host_members(modules, seed, n_tracks=40, n_noise=30) -> dict[str, pd.DataFrame]:
    """A synthetic TrackML host event (throughput CSV layout): tracks with hits at module centres, noise, a particle
    without hits, score weights summing to one over the particle hits."""
    rng = np.random.default_rng(seed)
    H, P = [], []
    for j in range(n_tracks + 1):
        vert = int(rng.integers(1, 4))
        pid = (vert << 52) | ((j + 1) << 36)
        q = int(rng.choice([-1, 1]))
        p = rng.normal(0, 2, 3)
        nh = 0 if j == n_tracks else int(rng.integers(3, 9))
        if nh:
            v, l, m, c = _place(modules, rng, nh)
            for i in range(nh):
                H.append((c[i, 0], c[i, 1], c[i, 2], v[i], l[i], m[i], pid, *p, rng.uniform(0.5, 1.5)))
        P.append((pid, 211 * q, *rng.normal(0, 0.01, 2), rng.normal(0, 50), *p, q, nh))
    v, l, m, c = _place(modules, rng, n_noise)
    for i in range(n_noise):
        H.append((c[i, 0], c[i, 1], c[i, 2], v[i], l[i], m[i], 0, 0.0, 0.0, 0.0, 0.0))
    h = pd.DataFrame(H, columns=["x", "y", "z", "volume_id", "layer_id", "module_id", "particle_id", "tpx", "tpy",
                                 "tpz", "w"])
    h = h.iloc[np.lexsort((np.arange(len(h)), h.module_id, h.layer_id, h.volume_id))].reset_index(drop=True)
    h["hit_id"] = np.arange(1, len(h) + 1)
    h["w"] = np.where(h.particle_id != 0, h.w / h.w[h.particle_id != 0].sum(), 0.0)
    hits = h[["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"]]
    truth = pd.DataFrame({"hit_id": h.hit_id, "particle_id": h.particle_id, "tx": h.x, "ty": h.y, "tz": h.z,
                          "tpx": h.tpx, "tpy": h.tpy, "tpz": h.tpz, "weight": h.w})
    cells = pd.DataFrame({"hit_id": np.repeat(h.hit_id, 2), "ch0": np.tile([10, 11], len(h)),
                          "ch1": np.tile([20, 20], len(h)), "value": np.tile([0.1, 0.05], len(h))})
    parts = pd.DataFrame(P, columns=W.CSV_COLUMNS["particles"])
    return {"hits": hits, "cells": cells, "particles": parts, "truth": truth}


def gun_event(modules, seed) -> tuple[D.Digitised, pd.DataFrame]:
    """Gun particles of hnl.overlay.gun and synthetic digitised hits (0-5 per particle at module centres; ACTS
    barcodes, in no particular order)."""
    rng = np.random.default_rng(seed)
    parts, _ = GUN.make_gun_event({"vx": 0.0, "vy": 0.0, "vz": 10.0}, rng, 0, 0)
    rows = []
    for r in parts.itertuples():
        n = int(rng.integers(0, 6))
        v, l, m, c = _place(modules, rng, n)
        for i in range(n):
            rows.append((r.particle_id, c[i, 0], c[i, 1], c[i, 2], v[i], l[i], m[i], c[i, 0], c[i, 1], c[i, 2],
                         r.px, r.py, r.pz))
    h = pd.DataFrame(rows, columns=["particle_id", "x", "y", "z", "volume_id", "layer_id", "module_id", "tx", "ty",
                                    "tz", "tpx", "tpy", "tpz"]).sample(frac=1, random_state=seed)
    h = h.reset_index(drop=True)
    h["particle_id"] = h.particle_id.astype(np.uint64)
    nc = rng.integers(1, 4, len(h))
    cells = pd.DataFrame({"hit": np.repeat(np.arange(len(h)), nc), "ch0": rng.integers(0, 100, nc.sum()),
                          "ch1": rng.integers(0, 100, nc.sum()), "value": rng.uniform(0.01, 0.2, nc.sum())})
    return D.Digitised(hits=h, cells=cells), parts


def _gz(members, fmt=None):
    return W.members_gz(members) if fmt is None else fmt


def host_tables_of(modules, det, host) -> tuple[dict, dict]:
    mem = host_members(modules, host)
    gz = {k: gzip.compress(mem[k].to_csv(index=False, float_format=W.CSV_FLOAT[k]).encode(), mtime=0)
          for k in mem}
    tables, dq = T.event_tables_tp(gz, host, det, keep_weight=True, keep_truth=False)
    assert tables is not None, dq
    return {k: tables[k] for k in W.TABLES}, mem


def test_overlay_ids_and_table_root():
    from hnl.trackml import events as EV
    assert W.overlay_id("hval", 21117) == 50_021_117 and W.overlay_id("pilot", 21117) == 59_021_117
    assert W.overlay_id("htest", 999_999) == 52_999_999
    assert W.split_overlay_id(53_022_001) == (3, 22001)
    with pytest.raises(ValueError):
        W.overlay_id("hval", 1_000_000)
    assert EV.OVERLAY_FIRST_ID == W.OVERLAY_FIRST_ID and EV.DERIVED_OVERLAY == W.DERIVED_OVERLAY
    assert EV.table_root(59_021_117) == EV.DERIVED_OVERLAY and EV.table_root(50_000_000) == EV.DERIVED_OVERLAY
    assert EV.table_root(49_999_999) == EV.DERIVED_TP and EV.table_root(21117) == EV.DERIVED_TP
    assert EV.table_root(1000) == C.DERIVED
    assert max(W.overlay_id(s, 999_999) for s in W.SETS) < 10 ** 8 < 2 ** 31
    assert "htrain" not in W.SETS                                      # no H-train set in addendum H
    with pytest.raises(KeyError):
        W.overlay_id("htrain", 22999)


def test_sets_carry_the_addendum_seeds():
    from hnl.overlay import split_h as SH
    assert W.SETS == {"hval": (0, 600_000_001), "closure": (1, 610_000_001), "htest": (2, 620_000_001),
                      "pilot": (9, 990_101)}
    split = SH.load_split()
    assert {n: (v["k"], v["base_seed"]) for n, v in split["sets"].items()} == W.SETS
    assert all(SH.base_seed(n) == W.SETS[n][1] for n in W.SETS)


def test_gun_members_layout(modules):
    dig, parts = gun_event(modules, 1)
    mem = W.gun_members(dig, parts)
    h, t, c, p = mem["hits"], mem["truth"], mem["cells"], mem["particles"]
    assert {k: list(v.columns) for k, v in mem.items()} == W.CSV_COLUMNS
    assert h.hit_id.tolist() == list(range(1, len(dig.hits) + 1)) and t.hit_id.tolist() == h.hit_id.tolist()
    key = C.module_key(h.volume_id, h.layer_id, h.module_id)
    assert np.all(np.diff(key) >= 0)                                   # sorted by module, as TrackML
    assert (t.weight == 0).all()
    f = GUN.trackml_fields(p.particle_id.to_numpy())
    assert (p.particle_id > 0).all() and f["vertex"].min() >= 1001 and f["vertex"].max() <= 1010
    assert len(p) == len(parts) and (p.nhits == 0).any()               # particles without hits keep their row
    cnt = t.particle_id.value_counts().reindex(p.particle_id).fillna(0).to_numpy()
    assert np.array_equal(cnt, p.nhits.to_numpy())
    assert set(c.hit_id) <= set(h.hit_id) and len(c) == len(dig.cells)
    # the hit of each row keeps its particle: compare positions per particle with the input
    src = dig.hits.assign(tml=GUN.trackml_particle_id(dig.hits.particle_id.to_numpy(np.uint64)))
    a = src.groupby("tml").x.apply(lambda s: sorted(np.round(s, 2))).to_dict()
    b = pd.DataFrame({"pid": t.particle_id, "x": h.x}).groupby("pid").x.apply(lambda s: sorted(np.round(s, 2)))
    assert a == b.to_dict()


def test_gun_event_passes_the_frozen_converter(modules, det):
    dig, parts = gun_event(modules, 2)
    tables, dq = W.convert_gun(W.members_gz(W.gun_members(dig, parts)), 59_021_117, det)
    assert dq["problems"] == [] and dq["max_outside_mm"] <= C.DQ_MAX_OUTSIDE_MM
    p = tables["particles"].to_pandas()
    assert p.particle_id.dtype == np.int64 and (p.particle_id > 0).all() and not p.particle_id.duplicated().any()
    assert (tables["hits"].column("weight").to_numpy() == 0).all()
    assert np.array_equal(tables["ptype"].column("particle_id").to_numpy(), p.particle_id.to_numpy())
    assert (p.vprim >= 1001).all() and (p.vprim <= 1010).all() and not p.is_hard_scatter.any()


def test_merge_equals_conversion_of_the_combined_event(modules, det):
    """The converter accepts the merged event unchanged: host rows + gun rows equal the frozen conversion of host and
    gun written as one TrackML event (gun hit ids after the host's)."""
    host, hmem = host_tables_of(modules, det, HOSTS[0])
    dig, parts = gun_event(modules, 3)
    gmem = W.gun_members(dig, parts)
    oid = W.overlay_id("pilot", HOSTS[0])
    gtab, _ = W.convert_gun(W.members_gz(gmem), oid, det)
    merged, rm = W.merge_event(host, gtab, oid)
    n = len(hmem["hits"])
    comb = {"hits": pd.concat([hmem["hits"], gmem["hits"].assign(hit_id=gmem["hits"].hit_id + n)]),
            "cells": pd.concat([hmem["cells"], gmem["cells"].assign(hit_id=gmem["cells"].hit_id + n)]),
            "particles": pd.concat([hmem["particles"], gmem["particles"]]),
            "truth": pd.concat([hmem["truth"], gmem["truth"].assign(hit_id=gmem["truth"].hit_id + n)])}
    ref, dq = T.event_tables_tp(W.members_gz({k: v.reset_index(drop=True) for k, v in comb.items()}), oid, det,
                                keep_weight=True, keep_truth=False)
    assert dq["problems"] == []
    for k in W.TABLES:
        assert W.tables_equal(merged[k], ref[k]), k
    assert rm["row_map"].tolist() == list(range(n)) and rm["info"]["n_gun_hits"] == len(gmem["hits"])


def test_check_event_passes_and_flags_problems(modules, det):
    host, _ = host_tables_of(modules, det, HOSTS[1])
    dig, parts = gun_event(modules, 4)
    oid = W.overlay_id("pilot", HOSTS[1])
    gtab, _ = W.convert_gun(W.members_gz(W.gun_members(dig, parts)), oid, det)
    merged, _ = W.merge_event(host, gtab, oid)
    ng = gtab["particles"].num_rows
    assert W.check_event(merged, host, ng) == []

    def mutate(table, col, fn):
        m = dict(merged)
        a = m[table].column(col).to_numpy().copy()
        fn(a)
        m[table] = m[table].set_column(m[table].schema.get_field_index(col), col, pa.array(a))
        return m

    assert "host hits.x altered" in W.check_event(mutate("hits", "x", lambda a: a.__setitem__(0, a[0] + 1)), host, ng)
    nh = merged["hits"].num_rows
    assert "gun hit weight != 0" in W.check_event(mutate("hits", "weight", lambda a: a.__setitem__(nh - 1, 0.1)),
                                                  host, ng)
    assert any("host weight sum" in x for x in W.check_event(merged, host, ng, expect_weight_sum=0.5))
    dup = mutate("particles", "particle_id", lambda a: a.__setitem__(len(a) - 1, a[0]))
    assert "duplicate particle_id" in W.check_event(dup, host, ng)
    neg = mutate("particles", "particle_id", lambda a: a.__setitem__(len(a) - 1, -5))
    assert "particle_id <= 0 (as int64)" in W.check_event(neg, host, ng)


def test_merge_drops_host_particles_for_closure(modules, det):
    host, _ = host_tables_of(modules, det, HOSTS[0])
    dig, parts = gun_event(modules, 5)
    oid = W.overlay_id("closure", HOSTS[0])
    gtab, _ = W.convert_gun(W.members_gz(W.gun_members(dig, parts)), oid, det)
    hp = host["hits"].column("pid").to_numpy()
    drop = [int(hp[hp != 0][0])]
    w_drop = float(host["hits"].column("weight").to_numpy()[hp == drop[0]].sum())
    merged, rm = W.merge_event(host, gtab, oid, drop_pids=drop)
    assert drop[0] not in merged["particles"].column("particle_id").to_numpy()
    assert (rm["row_map"][hp == drop[0]] == -1).all() and rm["info"]["dropped_host_hits"] == int((hp == drop[0]).sum())
    assert W.check_event(merged, host, gtab["particles"].num_rows, 1.0 - w_drop, drop_pids=drop) == []


def _store_host(root, archive, tables_by_event):
    shards = {}
    for e in sorted(tables_by_event):
        shards.setdefault(e // SHARD * SHARD, []).append(e)
    meta = {"archive": archive, "shards": {}}
    for first, evs in shards.items():
        meta["shards"][str(first)] = {}
        for k in W.TABLES:
            path = C.shard_path(root, k, archive, first)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with pq.ParquetWriter(path, T.SCHEMAS[k], write_statistics=True) as w:
                for e in evs:
                    w.write_table(tables_by_event[e][k], row_group_size=tables_by_event[e][k].num_rows)
            meta["shards"][str(first)][k] = {"file": os.path.relpath(path, root), "events": evs}
    with open(os.path.join(root, f"meta_{archive}.json"), "w") as fh:
        json.dump(meta, fh)


def test_write_read_and_verify(modules, det, tmp_path):
    hroot, oroot = str(tmp_path / "host"), str(tmp_path / "overlay")
    hosts = {h: host_tables_of(modules, det, h)[0] for h in HOSTS}
    _store_host(hroot, "training_part01", hosts)
    assert all(W.tables_equal(W.host_tables(h, hroot)[k], hosts[h][k]) for h in HOSTS for k in W.TABLES)
    merged, side, gz = {}, [], {}
    for i, h in enumerate(HOSTS):
        dig, parts = gun_event(modules, 10 + i)
        oid = W.overlay_id("pilot", h)
        gz[oid] = W.members_gz(W.gun_members(dig, parts))
        gtab, _ = W.convert_gun(gz[oid], oid, det)
        merged[oid], rm = W.merge_event(W.host_tables(h, hroot), gtab, oid)
        side.append({**rm["info"], "overlay_event": oid, "host": h, "n_gun_particles": gtab["particles"].num_rows,
                     "expected_host_weight_sum": 1.0})
    os.makedirs(oroot)
    W.write_set("pilot", merged, oroot)
    with open(os.path.join(oroot, "sidecar_pilot.json"), "w") as fh:
        json.dump({"events": side}, fh)
    v = W.verify_set("pilot", oroot, host_root=hroot)
    assert v["problems"] == [] and v["events"] == 2
    # the frozen read path: hits in host order, then the gun; rows and hit ids continue
    oid = W.overlay_id("pilot", HOSTS[0])
    hr = C.read_table("hits", [oid], root=oroot)
    nh = hosts[HOSTS[0]]["hits"].num_rows
    assert len(hr) == merged[oid]["hits"].num_rows and hr.hid.tolist() == list(range(1, len(hr) + 1))
    assert np.array_equal(hr.x.to_numpy()[:nh], hosts[HOSTS[0]]["hits"].column("x").to_numpy())
    assert (hr.pid.to_numpy()[nh:] >> 52 >= 1001).all()
    from hnl.trackml import events as EV
    ev = EV.load_events([oid], root=oroot)[oid]
    assert ev.n == len(hr) and ev.particles.particle_id.is_unique
    host_ev = EV._event_from(C.read_table("hits", [HOSTS[0]], EV.HIT_COLUMNS, hroot), None, HOSTS[0])
    gun = (ev.pid >> 52) >= 1001
    te = ev.true_edges
    assert te.shape[1] > host_ev.true_edges.shape[1] and not (gun[te[0]] != gun[te[1]]).any()
    assert np.array_equal(te[:, ~gun[te[0]]], host_ev.true_edges)          # host edges unchanged
    # tampering is caught
    f = os.path.join(oroot, "hits", "pilot", f"shard_{oid // SHARD * SHARD:05d}.parquet")
    with open(f, "ab") as fh:
        fh.write(b"x")
    assert any("sha256" in p for p in W.verify_set("pilot", oroot, host_root=hroot)["problems"])


def test_write_set_refuses_ids_of_another_set(modules, det, tmp_path):
    host, _ = host_tables_of(modules, det, HOSTS[0])
    dig, parts = gun_event(modules, 6)
    oid = W.overlay_id("hval", HOSTS[0])
    gtab, _ = W.convert_gun(W.members_gz(W.gun_members(dig, parts)), oid, det)
    merged, _ = W.merge_event(host, gtab, oid)
    with pytest.raises(ValueError):
        W.write_set("pilot", {oid: merged}, str(tmp_path))


def test_csv_tar(modules, tmp_path):
    dig, parts = gun_event(modules, 7)
    gz = W.members_gz(W.gun_members(dig, parts))
    rec = W.write_csv_tar(str(tmp_path / "gun.tar"), {59_021_117: gz})
    with tarfile.open(rec["tar"]) as tf:
        names = sorted(tf.getnames())
        assert names == [f"event059021117-{k}.csv.gz" for k in ("cells", "hits", "particles", "truth")]
        back = gzip.decompress(tf.extractfile("event059021117-hits.csv.gz").read())
    assert back == gzip.decompress(gz["hits"])
    raw = T.parse_members(gz)
    assert raw["particles"].particle_id.dtype == np.int64 and (raw["particles"].particle_id > 0).all()
