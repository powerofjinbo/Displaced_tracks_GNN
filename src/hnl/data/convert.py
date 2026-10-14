"""Convert the npz events into sharded per-event tables (brief §5a).

Output under <out>/<table>/<sample>/shard_<first event>.parquet, one shard per 500 consecutive event indices
(dropped events omitted), tables:
  hits       event, row, hid, x, y, z (float64 from the float32 (r, phi, z)), r, phi (float32 as stored), volume, layer,
             module (= sensitive id), geoid, pid
  particles  event, particle_id (int64), pdg, q, px, py, pz, pt, eta (table values: final state), vx, vy, vz, vt (end
             state), process, parent_pdg, barcode fields (vprim, vsec, particle, gen, sub), n_hits, n_pixel_hits,
             r_first (radius of the chain head), is_hnl / is_w (column 12 in labeled samples, barcode rules in
             hnl15_100), category ("hnl", "w", "other"), is_target (pT > 1 GeV, |eta| < 4, >= 5 hits),
             is_target_pix3 (target with >= 3 pixel hits)
  edges      event, src, dst (rows; the npz layerless_true_edges, chain order)
Per-particle truth-fit results (d0, z0, r_v) live in derived/truth_fits and join on (event, particle_id).

    python -m hnl.data.convert --out <derived/events/v1> [--samples ...] [--shard 500]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.data.npz import COL, SAMPLES, load_event, particle_ids
from hnl.data.splits import dropped_events
from hnl.geometry.ids import PIXEL_VOLUMES, cylindrical_to_cartesian, decode_barcode, decode_geometry_id
from hnl.truth import helix as hx
from hnl.truth import labels as L

SHARD = 500


def event_tables(ev: dict, sample: str, event: int, info=None) -> dict[str, pd.DataFrame]:
    """info: the sample's hnl.data.npz.Sample when it is not in SAMPLES (the tar-sharded HNL points of
    hnl.data.shards.POINTS); only its labeled / kind fields are used."""
    info = SAMPLES[sample] if info is None else info
    x, pid, te, P = ev["x"], ev["pid"], ev["layerless_true_edges"], ev["particles"]
    n = len(pid)
    xyz = cylindrical_to_cartesian(x)
    g = decode_geometry_id(ev["layers"])
    hits = pd.DataFrame({
        "event": np.full(n, event, dtype=np.int32), "row": np.arange(n, dtype=np.int32), "hid": ev["hid"],
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "r": x[:, 0], "phi": x[:, 1],
        "volume": g["volume"].astype(np.int16), "layer": g["layer"].astype(np.int16),
        "module": g["sensitive"].astype(np.int32), "geoid": ev["layers"], "pid": pid,
    })
    edges = pd.DataFrame({"event": np.full(te.shape[1], event, dtype=np.int32),
                          "src": te[0].astype(np.int32), "dst": te[1].astype(np.int32)})
    ids = particle_ids(P)
    bc = decode_barcode(ids)
    # per-particle hit counts, pixel hits, first-hit radius (chain head)
    upid, cnt = np.unique(pid, return_counts=True)
    n_hits = pd.Series(cnt, index=upid).reindex(ids).fillna(0).astype(np.int32).to_numpy()
    pix = np.isin(g["volume"], PIXEL_VOLUMES)
    upix, cpix = np.unique(pid[pix], return_counts=True)
    n_pix = pd.Series(cpix, index=upix).reindex(ids).fillna(0).astype(np.int32).to_numpy()
    order = hx.order_hits(pid, te)
    heads = order[np.r_[True, pid[order][1:] != pid[order][:-1]]]
    r_first = pd.Series(x[heads, 0].astype(np.float64), index=pid[heads]).reindex(ids).to_numpy()
    if info.labeled:
        lab = L.labels_from_table(P)
        is_hnl, is_w = lab["hnl_daughter"], lab["w_muon"]
    elif info.kind == "generic":    # generated samples: no HNL / W categories (truth in their sidecars)
        is_hnl = is_w = np.zeros(len(P), dtype=bool)
    else:
        is_hnl, is_w = L.hnl_daughter_rule_first_pair(P), L.w_muon_rule(P)
    pt = P[:, COL["pt"]]
    eta = P[:, COL["eta"]]
    is_target = (pt > 1.0) & (np.abs(eta) < 4.0) & (n_hits >= 5)
    parts = pd.DataFrame({
        "event": np.full(len(P), event, dtype=np.int32), "particle_id": ids,
        "pdg": P[:, COL["particle_type"]].astype(np.int64), "q": P[:, COL["q"]].astype(np.float32),
        "px": P[:, COL["px"]], "py": P[:, COL["py"]], "pz": P[:, COL["pz"]], "pt": pt, "eta": eta,
        "vx": P[:, COL["vx"]], "vy": P[:, COL["vy"]], "vz": P[:, COL["vz"]], "vt": P[:, COL["vt"]],
        "process": P[:, COL["process"]].astype(np.int16), "parent_pdg": P[:, COL["parent_pid"]].astype(np.int32),
        **{k: v.astype(np.int32) for k, v in bc.items()},
        "n_hits": n_hits, "n_pixel_hits": n_pix, "r_first": r_first,
        "is_hnl": is_hnl, "is_w": is_w,
        "category": np.where(is_hnl, "hnl", np.where(is_w, "w", "other")),
        "is_target": is_target, "is_target_pix3": is_target & (n_pix >= 3),
    })
    return {"hits": hits, "particles": parts, "edges": edges}


def convert_sample(sample: str, out: str, shard: int = SHARD, limit: int | None = None) -> dict:
    drop = dropped_events()[sample]
    n = SAMPLES[sample].n_events if limit is None else min(limit, SAMPLES[sample].n_events)
    stats = {"events": 0, "hits": 0, "particles": 0, "edges": 0, "shards": 0}
    for first in range(0, n, shard):
        frames = {"hits": [], "particles": [], "edges": []}
        for i in range(first, min(first + shard, n)):
            if i in drop:
                continue
            t = event_tables(load_event(sample, i), sample, i)
            for k, v in t.items():
                frames[k].append(v)
            stats["events"] += 1
        for k, fr in frames.items():
            if not fr:
                continue
            df = pd.concat(fr, ignore_index=True)
            d = os.path.join(out, k, sample)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"shard_{first:05d}.parquet")
            if os.path.exists(path):
                raise FileExistsError(path)
            df.to_parquet(path, index=False, row_group_size=200_000)
            stats[k] += len(df)
        stats["shards"] += 1
    return stats


def read_table(root: str, table: str, sample: str, events=None, columns=None) -> pd.DataFrame:
    """Read one table of one sample (optionally only some events) from the shards."""
    d = os.path.join(root, table, sample)
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    if events is not None:
        ev = np.asarray(events)
        firsts = np.array([int(f[6:11]) for f in files])
        keep = np.unique(ev // SHARD * SHARD)
        files = [f for f, a in zip(files, firsts) if a in set(keep.tolist())]
    flt = [("event", "in", [int(e) for e in np.asarray(events)])] if events is not None else None
    return pd.concat([pd.read_parquet(os.path.join(d, f), columns=columns, filters=flt) for f in files],
                     ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default=",".join(SAMPLES))
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    meta = {"samples": {}, "shard": a.shard, "slurm_job": os.environ.get("SLURM_JOB_ID")}
    for s in a.samples.split(","):
        meta["samples"][s] = convert_sample(s, a.out, a.shard, a.limit)
        print(f"[convert] {s}: {meta['samples'][s]} {time.time() - t0:.0f}s", flush=True)
    meta["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
