"""Barrel-pixel layers crossed by the N daughters, Zenodo HNL samples against our simulation (truth only; no model).

Follows up the one counted closure difference that had no explanation (results/hnl_points_closure_v1.md section 5:
more N-daughter targets with >= 3 pixel hits in our pilots than in Zenodo). For every N daughter (|parent| = 50) it
records which of the four barrel pixel layers (volume 8, layers 2/4/6/8 = L1-L4) hold a hit of the daughter, its
pixel and strip hit counts and the radius of its first hit; the decay radius r_v is the fitted vertex of the truth
census (scripts/acts/hnl_census.py: derived/truth_fits/v1 for Zenodo, the same function on our events, plus the
generator value for ours).

    extract: python scripts/acts/hnl_pixel_layers.py zenodo --sample hnl10_100 --splits train,val --out OUT.parquet
             python scripts/acts/hnl_pixel_layers.py shards --dir SAMPLEDIR --out OUT.parquet
    summary: python scripts/acts/hnl_pixel_layers.py summary --pair LABEL=ZEN.parquet,ZEN_CENSUS,OURS.parquet,OURS_CENSUS
             [--pair ...] --detectors detectors.csv --out OUT.json

Only the train and val events of the Zenodo samples are read (the test splits never).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from hnl.data.npz import COL, load_event, particle_ids  # noqa: E402
from hnl.data.splits import read_split  # noqa: E402
from hnl.geometry.ids import decode_geometry_id  # noqa: E402

BARREL_PIXEL_VOLUME = 8
BARREL_PIXEL_LAYERS = (2, 4, 6, 8)          # L1..L4
STRIP_VOLUMES = (12, 13, 14, 16, 17, 18)


def event_rows(ev: dict, event: int) -> list[dict]:
    P = ev["particles"]
    ids = particle_ids(P)
    parent = P[:, COL["parent_pid"]].astype(np.int64)
    pdg = P[:, COL["particle_type"]].astype(np.int64)
    pid = ev["pid"]
    g = decode_geometry_id(ev["layers"])
    vol, lay = g["volume"], g["layer"]
    r = ev["x"][:, 0].astype(np.float64)
    out = []
    for k in np.flatnonzero((np.abs(parent) == 50) | ((np.abs(parent) == 24) & (np.abs(pdg) == 13))):
        h = pid == ids[k]
        mask = 0
        for b, L in enumerate(BARREL_PIXEL_LAYERS):
            if np.any(h & (vol == BARREL_PIXEL_VOLUME) & (lay == L)):
                mask |= 1 << b
        hs = h & np.isin(vol, STRIP_VOLUMES)
        out.append({
            "event": event, "particle_id": int(ids[k]), "role": "d" if abs(parent[k]) == 50 else "w",
            "mask": mask, "n_hits": int(h.sum()), "n_pix": int((h & np.isin(vol, (7, 8, 9))).sum()),
            "n_pix_barrel": int((h & (vol == BARREL_PIXEL_VOLUME)).sum()), "n_strip": int(hs.sum()),
            "r_first": float(r[h].min()) if h.any() else np.nan,
            "r_first_strip": float(r[hs].min()) if hs.any() else np.nan,
        })
    return out


def _zen(args):
    sample, events = args
    rows = []
    for i in events:
        rows += event_rows(load_event(sample, int(i)), int(i))
    return rows


def _shards(args):
    directory, events = args
    from hnl.data.shards import iter_events_from
    rows = []
    for i, ev in iter_events_from(directory, events):
        rows += event_rows(ev, i)
    return rows


def run(fn, chunks, workers):
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame([r for rows in ex.map(fn, chunks) for r in rows])


def band_edges(detectors: str) -> list[tuple[float, float]]:
    """Radial extent [r_min, r_max] of the sensitive modules of each barrel pixel layer (closest point of the tilted
    module plane to the beam line, farthest corner)."""
    d = pd.read_csv(detectors)
    out = []
    for L in BARREL_PIXEL_LAYERS:
        g = d[(d.volume_id == BARREL_PIXEL_VOLUME) & (d.layer_id == L) & (d.module_id > 0)]
        c = g[["cx", "cy"]].to_numpy()
        u = g[["rot_xu", "rot_yu"]].to_numpy()
        hx = np.abs(g.bound_param0.to_numpy())
        s = np.linspace(-1, 1, 401)
        pts = c[:, None, :] + (s[None, :, None] * hx[:, None, None]) * u[:, None, :]
        rr = np.hypot(pts[..., 0], pts[..., 1])
        out.append((float(rr.min()), float(rr.max())))
    return out


def load_pair(zen_rows, zen_census, our_rows, our_census):
    out = {}
    for tag, rows, census in (("zenodo", zen_rows, zen_census), ("ours", our_rows, our_census)):
        R = pd.read_parquet(rows)
        E = pd.read_parquet(census + "_events.parquet")
        S = pd.read_parquet(census + "_signal.parquet")
        cols = ["event", "rv"] + (["rv_true"] if "rv_true" in E else [])
        R = R.merge(E[cols], on="event", how="left")
        R = R.merge(S[["event", "particle_id", "pt", "eta"]], on=["event", "particle_id"], how="left")
        R["target"] = (R.pt > 1) & (R.eta.abs() < 4) & (R.n_hits >= 5)
        out[tag] = R
    return out


def frac(x, event=None):
    """Fraction with its standard error; clustered by event when `event` is given (the two daughters of an event share
    r_v and fall into the same bins)."""
    x = np.asarray(x, dtype=bool)
    n = len(x)
    k = int(x.sum())
    if not n:
        return {"n": 0, "k": 0, "f": None, "se": None}
    f = k / n
    if event is None:
        se = float(np.sqrt(f * (1 - f) / n))
    else:
        g = pd.DataFrame({"e": np.asarray(event), "x": x}).groupby("e").x.agg(["sum", "count"])
        m = len(g)
        se = float(np.sqrt(((g["sum"] - f * g["count"]) ** 2).sum() * m / max(m - 1, 1)) / n)
    return {"n": n, "k": k, "f": f, "se": se}


def summary(pairs, detectors, out):
    bands = band_edges(detectors)
    res = {"bands_mm": bands, "points": {}}
    for label, spec in pairs:
        zr, zc, orow, oc = spec.split(",")
        D = load_pair(zr, zc, orow, oc)
        pt = {}
        for tag, R in D.items():
            d = R[(R.role == "d") & R.target & (R.eta.abs() < 1.0) & np.isfinite(R.rv)]
            rows = []
            for k, (lo, hi) in enumerate(bands[:3]):
                later = sum(1 << b for b in range(k + 1, 4))
                for name, a, b in (("before", lo - 6.0, lo - 1.0), ("in band", lo, hi), ("after", hi + 1.0, hi + 6.0)):
                    m = (d.rv >= a) & (d.rv < b)
                    x = d[m]
                    rows.append({"layer": f"L{k + 1}", "where": name, "rv_lo": a, "rv_hi": b,
                                 "no_later_pixel_hit": frac((x["mask"] & later) == 0, x.event),
                                 "ge3_pixel_hits": frac(x.n_pix >= 3, x.event),
                                 "mean_strip_hits": float(x.n_strip.mean()) if len(x) else None})
            t = R[(R.role == "d") & R.target & np.isfinite(R.rv)]
            cond = []
            for a, b in ((0, 30), (30, 72), (72, 116), (116, 172), (172, 1e9)):
                m = (t.rv >= a) & (t.rv < b)
                cond.append({"rv_lo": a, "rv_hi": b, "ge3_pixel_hits": frac(t[m].n_pix >= 3, t[m].event)})
            pt[tag] = {"bands": rows, "ge3_by_rv": cond,
                       "ge3_all_targets": frac(R[(R.role == "d") & R.target].n_pix >= 3,
                                                R[(R.role == "d") & R.target].event)}
            if "rv_true" in R:
                g = R[(R.role == "d") & R.target & (R.eta.abs() < 1.0) & np.isfinite(R.rv_true)]
                rows_t = []
                for k, (lo, hi) in enumerate(bands[:3]):
                    later = sum(1 << b for b in range(k + 1, 4))
                    m = (g.rv_true >= lo) & (g.rv_true < hi)
                    rows_t.append({"layer": f"L{k + 1}", "no_later_pixel_hit": frac((g[m]["mask"] & later) == 0)})
                pt[tag]["bands_true_rv"] = rows_t
        # z of the >= 3 pixel-hit fraction per r_v bin, ours minus Zenodo
        for i, c in enumerate(pt["ours"]["ge3_by_rv"]):
            a, b = c["ge3_pixel_hits"], pt["zenodo"]["ge3_by_rv"][i]["ge3_pixel_hits"]
            se = np.hypot(a["se"] or 0, b["se"] or 0)
            c["z_ours_minus_zenodo"] = float((a["f"] - b["f"]) / se) if a["n"] and b["n"] and se > 0 else None
        res["points"][label] = pt
    pooled = {}
    for tag in ("zenodo", "ours"):
        for i, r0 in enumerate(next(iter(res["points"].values()))[tag]["bands"]):
            k = sum(pt[tag]["bands"][i]["no_later_pixel_hit"]["k"] for pt in res["points"].values())
            n = sum(pt[tag]["bands"][i]["no_later_pixel_hit"]["n"] for pt in res["points"].values())
            pooled.setdefault(tag, []).append({"layer": r0["layer"], "where": r0["where"], "k": k, "n": n,
                                               "f": k / n if n else None})
        if all("bands_true_rv" in pt[tag] for pt in res["points"].values()):
            pooled[tag + "_true_rv"] = [{"layer": r0["layer"],
                                         "k": sum(pt[tag]["bands_true_rv"][i]["no_later_pixel_hit"]["k"]
                                                  for pt in res["points"].values()),
                                         "n": sum(pt[tag]["bands_true_rv"][i]["no_later_pixel_hit"]["n"]
                                                  for pt in res["points"].values())}
                                        for i, r0 in enumerate(next(iter(res["points"].values()))[tag]["bands_true_rv"])]
    res["pooled"] = pooled
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("pooled over the points (no later barrel pixel hit):")
    for a, b in zip(pooled["zenodo"], pooled["ours"]):
        print(f"  {a['layer']} {a['where']:8s} Zenodo {a['k']}/{a['n']}  ours {b['k']}/{b['n']}")
    if "ours_true_rv" in pooled:
        print("  ours, generator r_v in band:", [(r["layer"], r["k"], r["n"]) for r in pooled["ours_true_rv"]])
    # printed table
    print("band edges (mm):", [(round(a, 2), round(b, 2)) for a, b in bands])
    for label, pt in res["points"].items():
        print(f"\n{label}: daughter targets, |eta| < 1, fitted r_v")
        print("layer | r_v range | Zenodo: no later pixel hit | ours: no later pixel hit | Zenodo >=3 pix | ours >=3 pix")
        for a, b in zip(pt["zenodo"]["bands"], pt["ours"]["bands"]):
            fa, fb = a["no_later_pixel_hit"], b["no_later_pixel_hit"]
            print(f"{a['layer']} {a['where']:8s} [{a['rv_lo']:.1f}, {a['rv_hi']:.1f}) | "
                  f"{fa['k']}/{fa['n']} = {fa['f'] if fa['f'] is None else round(fa['f'], 3)} | "
                  f"{fb['k']}/{fb['n']} = {fb['f'] if fb['f'] is None else round(fb['f'], 3)} | "
                  f"{a['ge3_pixel_hits']['f']} | {b['ge3_pixel_hits']['f']}")
        if "bands_true_rv" in pt["ours"]:
            print("ours, generator r_v in band:", [(r["layer"], r["no_later_pixel_hit"]["k"], r["no_later_pixel_hit"]["n"])
                                                 for r in pt["ours"]["bands_true_rv"]])
        print(">= 3 pixel hits among targets by fitted r_v (ours vs Zenodo, z):")
        for a, b in zip(pt["ours"]["ge3_by_rv"], pt["zenodo"]["ge3_by_rv"]):
            print(f"  [{a['rv_lo']}, {a['rv_hi']}) {a['ge3_pixel_hits']['f']} ({a['ge3_pixel_hits']['n']}) vs "
                  f"{b['ge3_pixel_hits']['f']} ({b['ge3_pixel_hits']['n']}) z = {a.get('z_ours_minus_zenodo')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("zenodo")
    p.add_argument("--sample", required=True)
    p.add_argument("--splits", default="train,val")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("shards")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("summary")
    p.add_argument("--pair", action="append", required=True, help="LABEL=zen_rows,zen_census,our_rows,our_census")
    p.add_argument("--detectors", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "zenodo":
        splits = a.splits.split(",")
        if "test" in splits:
            raise SystemExit("the test split is not read")
        ev = np.sort(np.concatenate([read_split(a.sample, s) for s in splits]))
        R = run(_zen, [(a.sample, ev[k:k + 250]) for k in range(0, len(ev), 250)], a.workers)
    elif a.cmd == "shards":
        from hnl.data.shards import event_indices_from
        idx = event_indices_from(a.dir)
        R = run(_shards, [(a.dir, idx[k:k + 250].tolist()) for k in range(0, len(idx), 250)], a.workers)
    else:
        summary([s.split("=", 1) for s in a.pair], a.detectors, a.out)
        return
    R.to_parquet(a.out, index=False)
    print(f"{R.event.nunique()} events, {len(R)} rows -> {a.out}")


if __name__ == "__main__":
    main()
