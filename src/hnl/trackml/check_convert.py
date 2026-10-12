"""Checks of the converted TrackML tables (hnl.trackml.convert) against the raw csv members and the feasibility study.

Test discipline (docs/trackml_plan.md section 2.3): test events are never read here (the script refuses them), and
by default only dev and training events are used.

  * per event (raw csv against the stored tables, through hnl.trackml.convert.read_table):
      - hits: row order = hit_id, x / y / z = float32 of the csv, volume / layer / module, pid = truth particle_id,
        weight (stored events) or NaN, ncells / charge / span0 / span1 against an independent pandas groupby of
        cells.csv, r / phi / region / geoid recomputed (region also against hnl.models.triplet_gnn._region);
      - particles: every csv row in order, vertex / momentum = float32 of the csv, q, n_hits = csv nhits = number of
        stored hits of the particle, every non-noise hit's pid in the table, pt / eta / rv / d0 / flags recomputed;
      - chains: rank = 0 .. n - 1 for every particle, non-decreasing distance of the true point (csv, float64) from
        the production vertex along rank; true edges = the independent construction of the feasibility study
        (hits sorted by particle and float64 distance, consecutive pairs); number of edges = non-noise hits minus
        particles with hits;
      - truth table (calibration events): tx ... tpz = float32 of the csv for every non-noise hit.
  * dev events (1000-1099, all of them): per-event counts against work/out/content/events_all.csv and per-particle
    d0 / class / pixel hits against work/out/content/particles_all.parquet of docs/trackml_feasibility.md.
  * field sign: the rotation sense of positive charges from the truth momenta along the chains (SENSE_POS).
  * disk usage of the derived tables and of external/trackml.

    python -m hnl.trackml.check_convert [--events 1000,1001,...] [--source train_1.zip] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.trackml import convert as C

FEAS = os.path.join(C.EXT, "work", "out", "content")
TRAIN_1 = os.path.join(C.EXT, "train_1.zip")


def _raw(zf, names) -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(zf.open(names[k])) for k in C.KINDS}


def check_event(zf, names, event: int, root: str, stored_weight: bool, stored_truth: bool) -> dict:
    raw = _raw(zf, names)
    h_raw, c_raw, p_raw, t_raw = raw["hits"], raw["cells"], raw["particles"], raw["truth"]
    out = {"event": event, "problems": []}

    def need(cond, msg):
        if not bool(cond):
            out["problems"].append(msg)

    h = C.read_table("hits", [event], root=root)
    p = C.read_table("particles", [event], root=root)
    e = C.read_table("edges", [event], root=root)
    n = len(h_raw)
    # ---- hits
    need(len(h) == n, f"hits rows {len(h)} != csv {n}")
    need(np.array_equal(h.hid.to_numpy(), h_raw.hit_id.to_numpy()), "hid != csv hit_id")
    need(np.array_equal(h.row.to_numpy(), np.arange(n)), "row != 0..N-1")
    for c, rc in (("x", "x"), ("y", "y"), ("z", "z")):
        need(np.array_equal(h[c].to_numpy(), h_raw[rc].to_numpy().astype(np.float32)), f"{c} != float32(csv)")
    for c, rc in (("volume", "volume_id"), ("layer", "layer_id"), ("module", "module_id")):
        need(np.array_equal(h[c].to_numpy().astype(np.int64), h_raw[rc].to_numpy()), f"{c} != csv")
    t_raw = t_raw.sort_values("hit_id").reset_index(drop=True)
    need(np.array_equal(h.pid.to_numpy(), t_raw.particle_id.to_numpy()), "pid != truth particle_id")
    if stored_weight:
        need(np.array_equal(h.weight.to_numpy(), t_raw.weight.to_numpy().astype(np.float32)), "weight != csv")
    else:
        need(np.isnan(h.weight.to_numpy()).all(), "weight stored for a non-weight event")
    g = c_raw.groupby("hit_id")
    agg = pd.DataFrame({"n": g.size(), "q": g.value.sum(), "s0": g.ch0.max() - g.ch0.min() + 1,
                        "s1": g.ch1.max() - g.ch1.min() + 1}).reindex(h_raw.hit_id).fillna(0)
    need(np.array_equal(h.ncells.to_numpy(), agg.n.to_numpy().astype(np.int64)), "ncells != cells.csv")
    need(np.array_equal(h.span0.to_numpy(), agg.s0.to_numpy().astype(np.int64)), "span0 != cells.csv")
    need(np.array_equal(h.span1.to_numpy(), agg.s1.to_numpy().astype(np.int64)), "span1 != cells.csv")
    need(np.allclose(h.charge.to_numpy(np.float64), agg.q.to_numpy(), rtol=1e-6, atol=1e-6), "charge != cells.csv")
    r = np.hypot(h_raw.x.to_numpy().astype(np.float32).astype(np.float64),
                 h_raw.y.to_numpy().astype(np.float32).astype(np.float64)).astype(np.float32)
    need(np.array_equal(h.r.to_numpy(), r), "r recomputed differs")
    need(np.abs(h.phi.to_numpy(np.float64) - np.arctan2(h_raw.y, h_raw.x)).max() < 1e-6, "phi differs")
    try:
        import torch
        from hnl.models.triplet_gnn import _region
        oh = _region(torch.from_numpy(h.r.to_numpy()), torch.from_numpy(h.z.to_numpy())).argmax(1).numpy()
        need(np.array_equal(oh, h.region.to_numpy()), "region != hnl.models.triplet_gnn._region")
        out["region_checked_against_model"] = True
    except ImportError:
        out["region_checked_against_model"] = False
    vol = h.volume.to_numpy().astype(np.uint64)
    need(np.array_equal((h.geoid.to_numpy().view(np.uint64) >> np.uint64(56)), vol), "geoid volume bits")
    # ---- particles
    need(len(p) == len(p_raw), f"particles rows {len(p)} != csv {len(p_raw)}")
    need(np.array_equal(p.particle_id.to_numpy(), p_raw.particle_id.to_numpy()), "particle_id order != csv")
    for c in ("vx", "vy", "vz", "px", "py", "pz"):
        need(np.array_equal(p[c].to_numpy(), p_raw[c].to_numpy().astype(np.float32)), f"{c} != float32(csv)")
    need(np.array_equal(p.q.to_numpy().astype(np.int64), p_raw.q.to_numpy()), "q != csv")
    need(np.array_equal(p.n_hits.to_numpy().astype(np.int64), p_raw.nhits.to_numpy()), "n_hits != csv nhits")
    hit_pids = h.pid.to_numpy()
    nz = hit_pids != 0
    need(np.isin(hit_pids[nz], p.particle_id.to_numpy()).all(), "a hit's particle is missing from particles")
    cnt = pd.Series(hit_pids[nz]).value_counts().reindex(p.particle_id.to_numpy()).fillna(0).to_numpy()
    need(np.array_equal(cnt.astype(np.int64), p.n_hits.to_numpy().astype(np.int64)), "n_hits != stored hit count")
    V = {c: p[c].to_numpy().astype(np.float64) for c in ("vx", "vy", "vz", "px", "py", "pz")}
    need(np.array_equal(p.pt.to_numpy(), np.hypot(V["px"], V["py"]).astype(np.float32)), "pt recomputed differs")
    need(np.array_equal(p.rv.to_numpy(), np.hypot(V["vx"], V["vy"]).astype(np.float32)), "rv recomputed differs")
    d0 = C.d0_from_vertex(V["vx"], V["vy"], V["px"], V["py"], p.q.to_numpy()).astype(np.float32)
    need(np.array_equal(p.d0.to_numpy(), d0), "d0 recomputed differs")
    npix = pd.Series(hit_pids[nz & np.isin(h.volume.to_numpy(), (7, 8, 9))]).value_counts()
    need(np.array_equal(npix.reindex(p.particle_id.to_numpy()).fillna(0).to_numpy().astype(np.int64),
                        p.n_pixel_hits.to_numpy().astype(np.int64)), "n_pixel_hits differs")
    fl = C.particle_flags(p.pt.to_numpy(), p.eta.to_numpy(), p.d0.to_numpy(), p.n_hits.to_numpy(),
                          p.n_pixel_hits.to_numpy(), p.vprim.to_numpy())
    for k, v in fl.items():
        need(np.array_equal(p[k].to_numpy().astype(bool), v), f"flag {k} recomputed differs")
    # ---- chains and edges
    hp = pd.DataFrame({"row": np.arange(n)[nz], "pid": hit_pids[nz], "rank": h["rank"].to_numpy()[nz]})
    rk = hp.sort_values(["pid", "rank"])
    pos = rk.groupby("pid").cumcount().to_numpy()
    need(np.array_equal(rk["rank"].to_numpy(), pos), "ranks are not 0..n-1 per particle")
    need((h["rank"].to_numpy()[~nz] == C.NOISE_RANK).all(), "noise rank")
    tv = t_raw[t_raw.particle_id != 0].merge(p_raw[["particle_id", "vx", "vy", "vz"]], on="particle_id")
    tv["dist"] = np.sqrt((tv.tx - tv.vx) ** 2 + (tv.ty - tv.vy) ** 2 + (tv.tz - tv.vz) ** 2)
    tv["row"] = tv.hit_id - 1
    dist = pd.Series(tv.dist.to_numpy(), index=tv.row.to_numpy())
    rk["dist"] = dist.reindex(rk.row.to_numpy()).to_numpy()
    same = rk.pid.to_numpy()[1:] == rk.pid.to_numpy()[:-1]
    dd = np.diff(rk.dist.to_numpy())[same]
    out["n_decreasing_dist_steps"] = int((dd < 0).sum())
    out["max_decrease_mm"] = float(-dd.min()) if len(dd) and dd.min() < 0 else 0.0
    need(out["max_decrease_mm"] < 1e-3, f"distance decreases along rank by {out['max_decrease_mm']} mm")
    # independent edges: the feasibility construction (csv float64 distances)
    tv = tv.sort_values(["particle_id", "dist"], kind="stable")
    pid_s, row_s = tv.particle_id.to_numpy(), tv.row.to_numpy()
    sn = pid_s[1:] == pid_s[:-1]
    ref = set(zip(row_s[:-1][sn].tolist(), row_s[1:][sn].tolist()))
    got = set(zip(e.src.tolist(), e.dst.tolist()))
    out["edges_stored"] = len(got)
    out["edges_reference_mismatch"] = len(got ^ ref)
    need(got == ref, f"edges differ from the reference construction in {len(got ^ ref)} pairs")
    need(len(e) == nz.sum() - (p.n_hits > 0).sum(), "edge count != hits - particles with hits")
    need((h.pid.to_numpy()[e.src] == h.pid.to_numpy()[e.dst]).all(), "an edge joins two particles")
    # ---- truth table
    if stored_truth:
        tt = C.read_table("truth", [event], root=root)
        need(np.array_equal(tt.row.to_numpy(), np.flatnonzero(nz)), "truth rows != non-noise hits")
        for c in ("tx", "ty", "tz", "tpx", "tpy", "tpz"):
            need(np.array_equal(tt[c].to_numpy(), t_raw[c].to_numpy()[nz].astype(np.float32)), f"truth {c}")
    # ---- counts (for the comparison with the feasibility study)
    lay = h.volume.to_numpy().astype(np.int64) * 100 + h.layer.to_numpy()
    p5 = p[p.n_hits >= 5]
    out["counts"] = {
        "hits": int(n), "noise_hits": int((~nz).sum()), "noise_fraction": round(float((~nz).mean()), 4),
        "particle_rows": int(len(p)), "particles_ge1": int((p.n_hits >= 1).sum()), "particles_ge5": int(len(p5)),
        "targets": int(p.is_target.sum()), "true_edges": int(len(e)),
        "same_layer_edges": int((lay[e.src] == lay[e.dst]).sum()),
        "ge5_absd0_ge10": int((p5.d0.abs() >= 10).sum()), "ge5_absd0_ge100": int((p5.d0.abs() >= 100).sum()),
        "ge5_absd0_ge10_by_category": p5[p5.d0.abs() >= 10].category.value_counts().to_dict(),
        "displaced_targets": int(p.is_displaced_target.sum()), "far_displaced_targets":
            int(p.is_far_displaced_target.sum()), "prompt_targets": int(p.is_prompt_target.sum()),
        "soft_displaced": int(p.is_soft_displaced.sum()),
        "hard_scatter_ge5": int((p5.is_hard_scatter).sum()),
        "hard_scatter_targets": int((p.is_target & p.is_hard_scatter).sum()),
        "max_hits_per_particle": int(p.n_hits.max()),
    }
    return out


def sense_from_truth(zf, names) -> tuple[int, float]:
    """Rotation sense of positive charges from the truth momenta along each particle's hits (the feasibility
    study's sense_from_truth): +1 counter-clockwise seen from +z (Bz < 0), -1 clockwise (Bz > 0)."""
    raw = _raw(zf, names)
    h = raw["hits"].merge(raw["truth"], on="hit_id")
    h = h[h.particle_id != 0].merge(raw["particles"][["particle_id", "q", "vx", "vy", "vz"]], on="particle_id")
    h["dist"] = np.sqrt((h.tx - h.vx) ** 2 + (h.ty - h.vy) ** 2 + (h.tz - h.vz) ** 2)
    h = h.sort_values(["particle_id", "dist"])
    phi = np.arctan2(h.tpy.to_numpy(), h.tpx.to_numpy())
    pid = h.particle_id.to_numpy()
    same = pid[1:] == pid[:-1]
    dphi = np.remainder(phi[1:] - phi[:-1] + np.pi, 2 * np.pi) - np.pi
    s = np.sign(dphi[same]) * np.sign(h.q.to_numpy()[1:][same])
    frac = float((s > 0).mean())
    return (1 if frac > 0.5 else -1), frac


def compare_feasibility(root: str) -> dict:
    """All 100 dev events: stored counts and per-particle d0 / class against the feasibility study's outputs."""
    ev = pd.read_csv(os.path.join(FEAS, "events_all.csv")).set_index("event")
    dev = sorted(ev.index)
    p = C.read_table("particles", dev, root=root)
    h = C.read_table("hits", dev, ["event", "row", "volume", "layer", "pid", "rank"], root=root)
    e = C.edges_from_hits(h)
    lay = h.volume.to_numpy().astype(np.int64) * 100 + h.layer.to_numpy()
    hit_base = np.r_[0, np.cumsum(h.groupby("event", sort=True).size().to_numpy())[:-1]]
    base = pd.Series(hit_base, index=sorted(h.event.unique()))
    gs = base.reindex(e.event.to_numpy()).to_numpy() + e.src.to_numpy()
    gd = base.reindex(e.event.to_numpy()).to_numpy() + e.dst.to_numpy()
    mine = pd.DataFrame({
        "n_hits": h.groupby("event").size(),
        "n_noise": h[h.pid == 0].groupby("event").size(),
        "n_particles_rows": p.groupby("event").size(),
        "n_part_ge1": p[p.n_hits >= 1].groupby("event").size(),
        "n_part_ge5": p[p.n_hits >= 5].groupby("event").size(),
        "n_targets": p[p.is_target].groupby("event").size(),
        "n_true_edges": e.groupby("event").size(),
        "n_same_layer_edges": e[lay[gs] == lay[gd]].groupby("event").size(),
    })
    res = {"events": len(dev), "per_event_equal": {}, "means_ours": mine.mean().round(1).to_dict(),
           "means_feasibility": ev[mine.columns].mean().round(1).to_dict()}
    for c in mine.columns:
        res["per_event_equal"][c] = bool((mine[c].reindex(dev).to_numpy() == ev[c].reindex(dev).to_numpy()).all())
    fp = pd.read_parquet(os.path.join(FEAS, "particles_all.parquet"),
                         columns=["event", "particle_id", "d0_vtx", "cls", "n_pixel_hits", "pt", "eta", "rv"])
    m = p.merge(fp, on=["event", "particle_id"], how="outer", suffixes=("", "_feas"), indicator=True)
    res["particles_matched"] = int((m._merge == "both").sum())
    res["particles_unmatched"] = int((m._merge != "both").sum())
    m = m[m._merge == "both"]
    dd = (m.d0.astype(np.float64) - m.d0_vtx).abs()
    res["d0_abs_diff_mm"] = {"max": float(dd.max()), "p99": float(dd.quantile(0.99)),
                             "max_rel": float((dd / np.maximum(m.d0_vtx.abs(), 1e-3)).max())}
    res["category_equal_fraction"] = float((m.category == m.cls).mean())
    conf = m[m.category != m.cls].groupby(["cls", "category"]).size()
    res["category_confusion"] = {f"{a} -> {b}": int(v) for (a, b), v in conf.items()}
    res["n_pixel_hits_equal_fraction"] = float((m.n_pixel_hits.astype(np.int64) == m.n_pixel_hits_feas).mean())
    res["pt_max_rel_diff"] = float(((m.pt.astype(np.float64) - m.pt_feas).abs() / m.pt_feas).max())
    res["eta_max_abs_diff"] = float((m.eta.astype(np.float64) - m.eta_feas).abs().max())
    bins = [10.0, 100.0]
    ge5 = p[p.n_hits >= 5]
    nev = len(dev)
    res["per_event_ge5_absd0"] = {f">={b:g}mm": round(float((ge5.d0.abs() >= b).sum()) / nev, 1) for b in bins}
    res["per_event_ge5_absd0_ge10_by_category"] = (ge5[ge5.d0.abs() >= 10].category.value_counts() / nev).round(
        1).to_dict()
    res["per_event_targets_absd0"] = {f">={b:g}mm": round(float((p.is_target & (p.d0.abs() >= b)).sum()) / nev, 2)
                                      for b in bins}
    res["per_event_hard_scatter"] = {"ge5": round(float((ge5.is_hard_scatter).sum()) / nev, 1),
                                     "targets": round(float((p.is_target & p.is_hard_scatter).sum()) / nev, 1)}
    return res


def disk_usage(root: str) -> dict:
    out = {}
    for t in sorted(os.listdir(root)):
        d = os.path.join(root, t)
        if os.path.isdir(d):
            out[t] = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs)
    out["derived_total"] = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root) for f in fs)
    out["external_trackml_total"] = int(os.getxattr(C.EXT, "ceph.dir.rbytes").decode())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=TRAIN_1)
    ap.add_argument("--root", default=C.DERIVED)
    ap.add_argument("--events", default=None, help="default: dev 1000-1004 and three training events of later shards")
    ap.add_argument("--no-feasibility", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    split = C.load_split()
    if a.events:
        events = [int(x) for x in a.events.split(",")]
    else:
        tr = sorted(split["train_train1"] - split["calib"])
        events = [1000, 1001, 1002, 1003, 1004] + [next(e for e in tr if e >= lo) for lo in (1500, 2000, 2500)]
    bad = [e for e in events if e in split["test"]]
    if bad:
        raise SystemExit(f"refusing test events {bad} (plan section 2.3)")
    zf = C.open_archive(a.source)
    names = C.member_map(zf)
    res = {"events": events, "per_event": []}
    for ev in events:
        sw = ev in split["calib"] or ev in split["val"] or ev in split["test"]
        r = check_event(zf, names[ev], ev, a.root, sw, ev in split["calib"])
        res["per_event"].append(r)
        print(json.dumps(r), flush=True)
    sense, frac = sense_from_truth(zf, names[events[0]])
    res["field_sense"] = {"sense_pos_from_truth": sense, "fraction_ccw_positive": round(frac, 4),
                          "converter_SENSE_POS": C.SENSE_POS, "agree": sense == C.SENSE_POS}
    print(json.dumps(res["field_sense"]), flush=True)
    if not a.no_feasibility:
        res["feasibility"] = compare_feasibility(a.root)
        print(json.dumps(res["feasibility"], indent=1, default=str), flush=True)
    res["disk_bytes"] = disk_usage(a.root)
    print(json.dumps(res["disk_bytes"], indent=1), flush=True)
    n_prob = sum(len(r["problems"]) for r in res["per_event"]) + (0 if res["field_sense"]["agree"] else 1)
    res["n_problems"] = n_prob
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    print(f"[check] {len(events)} events, {n_prob} problems", flush=True)
    if n_prob:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
