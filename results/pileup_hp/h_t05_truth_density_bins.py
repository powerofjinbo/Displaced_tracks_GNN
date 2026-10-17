"""Addendum H, H-T0.5 step 4 (truth only): the local pileup hit density bins of section 7, committed with overlay_hash
in configs/trackml/overlay_version_h_v1.json.

Section 7: "The density of a target is the mean number of host hits, noise included, per module over the modules that
hold its hits, in C-full. Its bins are the tertiles over the H-val P_P targets, computed from truth and committed with
overlay_hash."

Truth only. Reads the stored H-val overlay tables (hits: event, volume, layer, module, pid; particles through
hnl.overlay.h1.load_truth) of the production overlay root. Reads no model output, no sealed directory, no H-test.
  - P_P targets: hnl.overlay.h1.gun_targets (the selection H1 uses).
  - host hit: a hit whose pid is not a gun particle (vertex field 1001-1010); noise hits (pid 0) count as host hits.
    C-full keeps every hit (hnl.overlay.conditions), so the stored event is the C-full event.
  - module: (volume, layer, module). For a target, the modules that hold its hits are counted once each; its density is
    the mean over them of the number of host hits on the module.
  - tertiles: numpy.quantile at 1/3 and 2/3 (method "linear", numpy's default) over the P_P targets; bins
    [0, q1), [q1, q2), [q2, inf).

    python h_t05_truth_density_bins.py --out h_t05_truth_density_bins_v1.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import blind as B
from hnl.overlay import h1 as H
from hnl.trackml.convert import read_table


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=H.DERIVED_OVERLAY)
    a = ap.parse_args()
    t0 = time.time()
    events = H.set_events("hval")
    parts, ghits = H.load_truth(a.root, events)
    g = H.gun_targets(parts, ghits)
    pp = g[g.pp].reset_index(drop=True)
    rows = []
    for i in range(0, len(events), 20):
        part = events[i:i + 20]
        h = read_table("hits", part, ["event", "volume", "layer", "module", "pid"], root=a.root)
        pid = h.pid.to_numpy(np.int64)
        v = pid >> 52
        gun = (v >= H.GUN_FIRST) & (v < H.GUN_FIRST + H.N_ISO + H.N_DENSE)
        key = ["event", "volume", "layer", "module"]
        host = h[~gun].groupby(key, sort=False).size().rename("n_host").reset_index()
        tp = pp[pp.event.isin(part)][["event", "particle_id"]]
        th = h[gun].merge(tp, left_on=["event", "pid"], right_on=["event", "particle_id"])
        mods = th[["event", "particle_id", "volume", "layer", "module"]].drop_duplicates()
        mods = mods.merge(host, on=key, how="left").fillna({"n_host": 0})
        rows.append(mods.groupby(["event", "particle_id"]).agg(density=("n_host", "mean"),
                                                                n_modules=("n_host", "size")).reset_index())
    d = pd.concat(rows, ignore_index=True)
    d = pp[["event", "particle_id", "region"]].merge(d, on=["event", "particle_id"], how="left")
    if d.density.isna().any():
        raise SystemExit(f"{int(d.density.isna().sum())} P_P targets without a hit in the hits table")
    x = d.density.to_numpy(np.float64)
    q1, q2 = (float(q) for q in np.quantile(x, [1 / 3, 2 / 3]))
    edges = [0.0, q1, q2, None]
    b = np.digitize(x, [q1, q2])
    meta = os.path.join(a.root, "meta_hval.json")
    with open(meta) as fh:
        oh = json.load(fh)["overlay_hash"]
    out = {
        "what": "local pileup hit density bins of addendum H section 7 (tertiles over the H-val P_P targets, truth)",
        "set": "hval", "events": len(events), "root": a.root,
        "meta_hval": {"path": meta, "sha256": sha256(meta), "overlay_hash": oh},
        "definition": ("density of a target = mean, over the distinct modules (volume, layer, module) that hold its "
                       "hits, of the number of host hits on the module in C-full (every hit whose particle is not a "
                       "gun particle, vertex field 1001-1010; noise hits, pid 0, included); P_P targets of "
                       "hnl.overlay.h1.gun_targets"),
        "quantile_method": "numpy.quantile, method linear (default), at 1/3 and 2/3",
        "unit": "host hits per module",
        "edges": edges, "bins": ["[0, q1)", "[q1, q2)", "[q2, inf)"],
        "n_pp": int(len(x)), "n_per_bin": [int((b == k).sum()) for k in range(3)],
        "n_per_bin_region": {r: [int(((b == k) & (d.region.to_numpy() == r)).sum()) for k in range(3)]
                             for r in H.REGIONS},
        "summary": {"min": float(x.min()), "median": float(np.median(x)), "mean": float(x.mean()),
                    "max": float(x.max()), "modules_per_target_mean": float(d.n_modules.mean())},
        "code": {"script": os.path.abspath(__file__), "sha256": sha256(os.path.abspath(__file__)),
                 "hnl_git": B._git_head()},
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in ("n_pp", "edges", "n_per_bin", "summary")}))


if __name__ == "__main__":
    main()
