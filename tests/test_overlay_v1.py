"""hnl.overlay.v1: the pair table of the closure set (populations a-d, produced at a sensor along the module normal,
first-hit layers, regions), V1 per population with its rule, the unconditional V1, V1-Delta, the gun-weighted V1 and
the paired event bootstrap, on synthetic truth and truth-made placeholder outputs (no model output); the checks of
every closure grid V1 reads (hnl.overlay.tracks.grid_record, frozen-classifier provenance, placeholders only in the
pilot closure dry run)."""

import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import closure as CL
from hnl.overlay import digitise as D
from hnl.overlay import hstats as HS
from hnl.overlay import v1 as V


def _modules() -> D.Modules:
    """A pixel-barrel module (8, 2, 1) at x = 32 mm with normal +x, and a pixel-endcap module (9, 2, 1) at z = 600 mm
    with normal -z."""
    key = D.module_key([8, 9], [2, 2], [1, 1])
    Rb = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    Re = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return D.Modules(key=key, c=np.array([[32.0, 0, 0], [100.0, 0, 600.0]]), R=np.array([Rb, Re], float),
                     t=np.full(2, 0.15), minhu=np.full(2, 30.0), maxhu=np.full(2, 30.0), hv=np.full(2, 300.0),
                     pu=np.full(2, 0.05), pv=np.full(2, 0.05), volume=np.array([8, 9]))


def test_at_sensor_is_measured_along_the_module_normal():
    m = _modules()
    xyz = np.array([[32.0, 5.0, 12.0], [32.0, 0.2, 0.0], [100.5, 0.0, 600.0]])
    vtx = np.array([[31.5, 3.0, 10.0], [30.5, 0.0, 0.0], [100.0, 0.0, 595.0]])
    at, d = V.at_sensor(xyz, vtx, [8, 8, 9], [2, 2, 2], [1, 1, 1], m)
    assert d.tolist() == pytest.approx([0.5, 1.5, 5.0]) and at.tolist() == [True, False, False]
    # the third is within 1 mm of its vertex in r (0.5 mm), but 5 mm away along the disk's normal
    with pytest.raises(KeyError):
        V.at_sensor(xyz[:1], vtx[:1], [8], [4], [1], m)


def _synthetic(n_events=10, seed=3):
    """Closure truth, stored closure rows, native and closure hits of n_events closure events: per event 30 natives of
    (a) (regions cycling), 6 of (b) (two produced at a sensor), 20 of (c) (five at |eta| >= 3); one (a) native per
    event whose re-simulated version has 3 hits."""
    rng = np.random.default_rng(seed)
    T, CP, NH, CH = [], [], [], []
    for i in range(n_events):
        host = 21300 + i
        ev = CL.CLOSURE_VERTEX[0] * 0 + 50_000_000 + 1_000_000 + host
        k = 0
        for pop, n in (("a", 30), ("b", 6), ("c", 20)):
            for j in range(n):
                k += 1
                gen = 1 if pop == "b" else 0
                nid = (1 << 52) | (k << 36) | (gen << 24)
                eta = (3.2 if (pop == "c" and j < 5) else 0.5 + 0.1 * (j % 10)) * (1 if j % 2 else -1)
                d0 = {"a": 20.0 if j % 3 else 2.0, "b": 15.0, "c": 0.1}[pop]
                rv = {"a": 150.0, "b": 40.0, "c": 0.2}[pop]
                rf = {"a": 160.0, "b": 50.0 if j < 3 else 300.0, "c": 32.0}[pop]
                at = pop == "b" and j < 2
                vtx = (31.5 if at else 20.0, 0.0, 0.0)
                T.append({"event": ev, "host": host, "native_particle_id": nid, "particle_id": nid + (1 << 62),
                          "population": pop, "region": V.REGIONS[j % 3] if pop == "a" else "central", "eta": eta,
                          "d0": d0, "rv": rv, "native_r_first": rf, "native_n_hits": 6, "vx": vtx[0], "vy": vtx[1],
                          "vz": vtx[2]})
                short = pop == "a" and j == 0
                CP.append({"event": ev, "particle_id": nid + (1 << 62), "is_target": not short,
                           "n_hits": 3 if short else 6})
                for r in range(6):
                    NH.append({"event": ev, "pid": nid, "rank": r, "x": 32.0 if r == 0 else 60.0 + r, "y": 0.0,
                               "z": 0.0, "volume": 8 if r < 3 else 13, "layer": 2, "module": 1})
                    if not short or r < 3:
                        CH.append({"event": ev, "pid": nid + (1 << 62), "rank": r,
                                   "volume": 8 if (r < 3 and not (pop == "b" and j < 2)) else 13, "layer": 2})
    del rng
    return pd.DataFrame(T), pd.DataFrame(CP), pd.DataFrame(NH), pd.DataFrame(CH)


@pytest.fixture(scope="module")
def pairs():
    t, cp, nh, ch = _synthetic()
    return V.pair_rows(t, cp, nh, ch, _modules())


def test_pair_rows(pairs):
    p = pairs
    assert len(p) == 10 * 56 and (p.closure_id == p.native_id + (1 << 62)).all()
    assert p.in_a.sum() == 300 and p.in_b_all.sum() == 60 and p.in_b_at_sensor.sum() == 20 and p.in_b.sum() == 40
    assert p.in_c_all.sum() == 200 and p.in_c.sum() == 150
    # (d): |d0| >= 10 and |d0| / r_first >= 0.2 among (a) and (b): (a) 20/160 = 0.125 no; (b) 15/50 = 0.3 yes for j < 3
    assert p.in_d.sum() == 30 and (p[p.in_d].population == "b").all()
    assert p.closure_target.sum() == 550 and (p[~p.closure_target].closure_n_hits == 3).all()
    assert p["in_pixel barrel"].all() and p["in_short-strip barrel"].all() and not p["in_pixel endcap"].any()
    # first-hit layer: the natives produced at a sensor start one layer later in the re-simulation
    assert (~p.first_layer_same).sum() == 20 and (p[~p.first_layer_same].in_b_at_sensor).all()
    assert p[p.in_b_at_sensor].d_normal_mm.max() <= 1.0 and (p[p.in_a].d_normal_mm == 12.0).all()


def _keys(rows):
    """(closure event, native id) of pair rows: native ids repeat across events."""
    return list(zip(rows.event, rows.native_id))


def _outputs(p, closure_miss=(), native_miss=(), c1gev_miss=(), n1gev_miss=()):
    """(event, particle_id, dm) tables: every target found except the listed (event, native id) (closure: of targets
    only)."""
    def tab(side, miss, cond_rows):
        q = cond_rows
        ev = q.event if side == "closure" else q.host
        pid = q.closure_id if side == "closure" else q.native_id
        lost = pd.MultiIndex.from_arrays([q.event, q.native_id]).isin(list(miss))
        return pd.DataFrame({"event": ev.to_numpy(), "particle_id": pid.to_numpy(), "dm": ~lost})
    ct = p[p.closure_target]
    return {("closure", "full"): tab("closure", closure_miss, ct), ("native", "full"): tab("native", native_miss, p),
            ("closure", "1gev"): tab("closure", c1gev_miss, ct), ("native", "1gev"): tab("native", n1gev_miss, p)}


def test_v1_delta_sigma_and_rules(pairs):
    p = pairs
    events = sorted(p.event.unique())
    a = p[p.in_a & p.closure_target]
    # the closure loses 2 (a) targets of the first event (one central, one fwd+), the natives none
    e0 = a[a.event == events[0]]
    miss = _keys(e0[e0.region == "central"].iloc[:1]) + _keys(e0[e0.region == "fwd+"].iloc[:1])
    q = V.attach_outputs(p, _outputs(p, closure_miss=miss))
    r = V.v1(q, events)
    ra = r["a"]
    assert ra["pairs"] == 290 and ra["natives"] == 300
    assert ra["delta"] == pytest.approx(100 * (288 - 290) / 290) and ra["eps_native"] == 100.0
    # the paired bootstrap: the index table of hnl.trackml.evaluate.bootstrap over the 10 events
    idx = HS.index_table(10)
    n = np.array([29] * 10)
    m = n - np.r_[2, np.zeros(9)]
    boot = 100 * (m[idx].sum(1) / n[idx].sum(1) - 1)
    assert ra["sigma_paired"] == pytest.approx(np.std(boot, ddof=1), rel=1e-12)
    assert ra["rule"]["pass"] and ra["rule"]["regions_pass"] == {"central": True, "fwd+": True, "fwd-": True}
    assert ra["regions"]["central"]["delta"] == pytest.approx(100 * (-1) / 90)
    assert ra["regions"]["fwd+"]["delta"] == pytest.approx(-1.0) and ra["regions"]["fwd-"]["delta"] == 0.0
    assert ra["unconditional"]["natives"] == 300 and ra["unconditional"]["delta"] == pytest.approx(
        100 * (288 - 300) / 300)
    assert ra["closure_below_5_hits"] == 10 and ra["closure_below_5_hits_share"] == pytest.approx(10 / 300)
    assert r["decisive_pass"] and r["flags"] == []
    # (c): the rule uses |eta| < 3 only; (b): without the natives produced at a sensor
    assert r["c"]["pairs"] == 150 and r["c_all"]["pairs"] == 200 and r["b"]["pairs"] == 40
    assert r["b_at_sensor"]["pairs"] == 20 and r["at_sensor"]["b_at_sensor"] == 20
    assert r["at_sensor"]["b_radial_within_1mm"] == 20


def test_v1_failures(pairs):
    p = pairs
    events = sorted(p.event.unique())
    a = p[p.in_a & p.closure_target]
    fwd = _keys(a[a.region == "fwd+"].iloc[:2])                          # 2 of 100 fwd+ pairs: -2 points in fwd+
    r = V.v1(V.attach_outputs(p, _outputs(p, closure_miss=fwd)), events)
    assert r["a"]["rule"]["regions_pass"]["fwd+"] is False and not r["a"]["rule"]["pass"] and not r["decisive_pass"]
    c = _keys(p[p.in_c].iloc[:1])                                        # 1 of 150: -0.67 points
    r = V.v1(V.attach_outputs(p, _outputs(p, closure_miss=c)), events)
    lim = max(0.2, 2 * r["c"]["sigma_paired"])
    assert r["c"]["rule"]["limit"] == pytest.approx(lim) and r["c"]["rule"]["pass"] == (abs(r["c"]["delta"]) <= lim)
    b = _keys(p[p.in_b & ~p.in_d].groupby("event").head(2))              # half of (b) lost in every event
    r = V.v1(V.attach_outputs(p, _outputs(p, closure_miss=b)), events)
    assert r["b"]["delta"] == pytest.approx(-50.0) and r["b"]["sigma_paired"] == pytest.approx(0.0, abs=1e-9)
    assert r["b"]["rule"]["limit"] == 3.0 and r["flags"] == ["b"] and r["decisive_pass"]


def test_v1_delta_rule(pairs):
    p = pairs
    events = sorted(p.event.unique())
    a = p[p.in_a & p.closure_target]
    lost = _keys(a.iloc[::29].iloc[:5])
    # the same 5 targets lost in C-1GeV on both sides: Delta_closure = Delta_native = +5/290, difference 0
    q = V.attach_outputs(p, _outputs(p, c1gev_miss=lost, n1gev_miss=lost))
    d = V.v1_delta(q, events)
    assert d["a"]["delta_closure"] == pytest.approx(100 * 5 / 290) and d["a"]["difference"] == pytest.approx(0.0)
    assert d["a"]["rule"]["pass"] and d["decisive_pass"]
    # the closure loses them in C-1GeV only: difference = 5/290 = 1.72 points
    q = V.attach_outputs(p, _outputs(p, c1gev_miss=lost))
    d = V.v1_delta(q, events)
    assert d["a"]["difference"] == pytest.approx(100 * 5 / 290)
    assert d["a"]["rule"]["pass"] == (d["a"]["difference"] <= max(0.5, 2 * d["a"]["sigma_paired"]))
    assert set(d) == {"a", "b", "c", "c_all", "d", "decisive_pass"}


def test_gun_weighted_and_outputs(pairs):
    p = pairs
    events = sorted(p.event.unique())
    b = p[p.in_b & p.closure_target]
    lost = _keys(b[b.r_first == 50.0].iloc[:4])                          # cell (r_first < 200, ratio >= 0.2)
    q = V.attach_outputs(p, _outputs(p, closure_miss=lost))
    gun = pd.DataFrame({"r_first": [60.0, 70.0, 300.0, 310.0], "d0": [20.0, -30.0, 20.0, 30.0]})
    w = V.gun_weights(gun)
    assert w.to_dict() == {2: 0.5, 4: 0.5}
    qb = q[q.in_b & q.closure_target]
    g = V.gun_weighted(qb, events, w)
    # cell 2: (b) natives with r_first 50 (10 pairs, 4 lost: -40 points); cell 4 (r_first 300, ratio 0.05): no loss
    assert g["delta"] == pytest.approx(0.5 * -40.0 + 0.5 * 0.0) and g["coverage"] == pytest.approx(1.0)
    assert {c["cell"]: c["pairs"] for c in g["cells"] if c["pairs"]} == {2: 10, 4: 30}
    half = V.gun_weighted(qb, events, pd.Series({2: 0.5, 3: 0.5}))      # cell 3 holds no pair: renormalised
    assert half["delta"] == pytest.approx(-40.0) and half["coverage"] == pytest.approx(0.5)
    # a target without an output row is refused
    outs = _outputs(p)
    outs[("closure", "full")] = outs[("closure", "full")].iloc[1:]
    with pytest.raises(LookupError, match="closure full"):
        V.attach_outputs(p, outs)


def test_placeholder_outputs_and_evaluate(pairs):
    outs = V.placeholder_outputs(pairs)
    assert set(outs) == {("closure", "full"), ("closure", "1gev"), ("native", "full"), ("native", "1gev")}
    assert len(outs[("native", "full")]) == len(pairs) and len(outs[("closure", "full")]) == pairs.closure_target.sum()
    q = V.attach_outputs(pairs, outs)
    r = V.evaluate(q, sorted(pairs.event.unique()), V.gun_weights(pd.DataFrame({"r_first": [60.0], "d0": [20.0]})))
    assert set(r) == {"events", "pairs_table", "v1", "v1_delta", "decisive_pass"}
    assert "gun_weighted" in r["v1"]["a"] and "gun_weighted" in r["v1"]["b"]
    assert np.isfinite(r["v1"]["a"]["sigma_paired"]) and np.isfinite(r["v1_delta"]["a"]["sigma_paired"])


def test_diag_outputs_and_compare(pairs, tmp_path):
    from hnl.trackml.convert import sha256_file
    run = tmp_path / "run"
    (run / "diag" / "displaced").mkdir(parents=True)
    nat = pairs[["host", "native_id"]].rename(columns={"host": "event", "native_id": "particle_id"})
    half = len(nat) // 2
    lines = []
    for b, part in enumerate((nat.iloc[:half], nat.iloc[half:])):
        f = f"diag/displaced/diag_full_b{b}.parquet"
        part.assign(dm=True, n_best=6).to_parquet(run / f, index=False)
        lines.append(f"{sha256_file(str(run / f))}  {f}")
    (tmp_path / "sha.txt").write_text("\n".join(lines) + "\n")
    d = V.diag_outputs("full", str(run), str(tmp_path / "sha.txt"))
    assert len(d) == len(nat) and d.dm.all()
    per_target = nat.assign(dm=True)
    assert V.compare_dm(pairs, per_target, d) == {"natives": len(nat), "differ": 0, "equal": True}
    assert V.compare_dm(pairs, per_target.assign(dm=np.arange(len(nat)) > 0), d)["differ"] == 1
    with pytest.raises(LookupError):
        V.diag_outputs("1gev", str(run), str(tmp_path / "sha.txt"))
    (run / "diag/displaced/diag_full_b1.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        V.diag_outputs("full", str(run), str(tmp_path / "sha.txt"))
    assert open(V.VAL_DIAG_SHA).read().count("diag_full_b") == 15          # the addendum's list


# ------------------------------------------------------------------------------------------------ grids (R7)
PILOT_CLOSURE = [51_021_289, 51_021_332, 51_021_610, 51_021_753, 51_021_835]


def _dry_root(tmp_path, base_seed=990101, events=PILOT_CLOSURE):
    import json
    root = tmp_path / "root"
    root.mkdir(parents=True)
    (root / "meta_closure.json").write_text(json.dumps({"base_seed": base_seed}))
    (root / "sidecar_closure.json").write_text(json.dumps({"events": [{"overlay_event": e} for e in events]}))
    return str(root)


def _dm(events, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"event": np.repeat(events, 3), "particle_id": np.tile([(1 << 62) | 1, (1 << 62) | 2,
                                                                               (1 << 62) | 3], len(events)),
                         "dm": rng.random(3 * len(events)) < 0.9})


def test_dry_run_reads_placeholder_grids_of_the_pilot_closure_only(tmp_path):
    from hnl.overlay import blind as B
    root = _dry_root(tmp_path)
    tab = _dm(PILOT_CLOSURE)
    d = str(tmp_path / "grids" / "full")
    rec = V.write_placeholder_grid(d, tab, "full", PILOT_CLOSURE, root)
    assert rec["scores_provenance"]["placeholder"] and rec["set"] == "closure" and rec["k"] == 1
    got_rec, got = V.closure_outputs(d, "full", root, PILOT_CLOSURE, dry=True)
    assert got_rec == rec and got.sort_values(["event", "particle_id"]).reset_index(drop=True).equals(
        tab.sort_values(["event", "particle_id"]).reset_index(drop=True))
    with pytest.raises(FileExistsError):
        V.write_placeholder_grid(d, tab, "full", PILOT_CLOSURE, root)        # never over an existing grid
    with pytest.raises(PermissionError):
        V.write_placeholder_grid(os.path.join(B.RUNS_H, "x"), tab, "full", PILOT_CLOSURE, root)
    with pytest.raises(ValueError, match="not closure in 1gev"):
        V.closure_grid(d, "1gev", root, PILOT_CLOSURE, dry=True)              # the condition as named
    with pytest.raises(ValueError):
        V.closure_grid(d, "hs", root, PILOT_CLOSURE, dry=True)                # the closure set has no C-hs
    with pytest.raises(ValueError, match="events"):
        V.closure_grid(d, "full", root, PILOT_CLOSURE[:4], dry=True)
    # a grid of real scores is never read on the pilot closure ids
    import json
    real = json.load(open(os.path.join(d, "grid.json")))
    real["scores_provenance"]["placeholder"] = False
    json.dump(real, open(os.path.join(d, "grid.json"), "w"))
    with pytest.raises(PermissionError, match="placeholder grids only"):
        V.closure_grid(d, "full", root, PILOT_CLOSURE, dry=True)
    # what a dry run is: a scratch root, closure ids of the pilot hosts, the pilot base seed
    with pytest.raises(PermissionError, match="production"):
        V.pilot_closure_dry(B.DERIVED_OVERLAY, PILOT_CLOSURE)
    with pytest.raises(PermissionError, match="base seed"):
        V.pilot_closure_dry(_dry_root(tmp_path / "a", base_seed=610000001), PILOT_CLOSURE)
    other = [B.overlay_id(1, h) for h in B.set_hosts("closure") if h not in {21289, 21332, 21610, 21753, 21835}][:2]
    with pytest.raises(PermissionError, match="pilot"):
        V.pilot_closure_dry(_dry_root(tmp_path / "b", events=other), other)
    with pytest.raises(PermissionError, match="pilot"):
        V.pilot_closure_dry(_dry_root(tmp_path / "c", events=[59_021_289]), [59_021_289])   # k = 9, not closure


def _production_grid(tmp_path, **change):
    """A grid.json as hnl.overlay.tracks writes it for the closure set in C-full from the frozen classifier."""
    import json
    from hnl.overlay import blind as B
    from hnl.overlay import tracks as TR
    sel = json.load(open(V.SELECTION))["inputs"]
    rec = {"set": "closure", "k": 1, "condition": "full", "events": B.set_ids("closure"), "n_events": 150,
           "block": 0, "n_blocks": 1, "root": B.DERIVED_OVERLAY, "scores": "/x/scores",
           "scores_provenance": {"files": ["a.parquet"], "placeholder": False,
                                 "ckpt_sha256": sel["classifier_checkpoint"]["sha256"], "condition": "full"},
           "configs": TR.wp_configs(), "merge_model": sel["merge_model"]["path"]}
    for k, v in change.items():
        if k == "prov":
            rec["scores_provenance"] = {**rec["scores_provenance"], **v}
        else:
            rec[k] = v
    d = tmp_path / f"g{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(rec))
    return str(d)


def test_run_checks_every_closure_grid(tmp_path):
    from hnl.overlay import blind as B
    ev = B.set_ids("closure")
    root = B.DERIVED_OVERLAY
    rec = V.closure_grid(_production_grid(tmp_path), "full", root, ev)
    assert rec["set"] == "closure" and not rec["scores_provenance"]["placeholder"]
    bad = {"placeholder": dict(prov={"placeholder": True}),
           "other classifier": dict(prov={"ckpt_sha256": "0" * 64}),
           "scored in another condition": dict(prov={"condition": "1gev"}),
           "other condition": dict(condition="1gev"),
           "other set": dict(set="hval", k=0),
           "missing events": dict(events=ev[:-1]),
           "other tables": dict(root="/tmp/scratch_overlay"),
           "other merge model": dict(merge_model="/x/model.joblib"),
           "a block of another classifier": dict(prov={"blocks": [
               {"placeholder": False, "ckpt_sha256": "0" * 64, "condition": "full"}]})}
    for why, change in bad.items():
        with pytest.raises((ValueError, PermissionError)):
            V.closure_grid(_production_grid(tmp_path, **change), "full", root, ev)
            pytest.fail(why)
    with pytest.raises(PermissionError):                                       # k = 1 is read from production only
        V.closure_grid(_production_grid(tmp_path, root="/tmp/scratch_overlay"), "full", "/tmp/scratch_overlay", ev)
    with pytest.raises(ValueError, match="events"):                            # tables of other events
        V.closure_grid(_production_grid(tmp_path), "full", root, ev[:5])
