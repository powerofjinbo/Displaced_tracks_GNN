"""Generator-level collision reuse between the TrackML split's training and validation events, and its effect on
Canto-L's val efficiency (post-freeze; val only, the test ids are not read).

The TrackML throughput events reuse generator-level collisions: the same ttbar or pileup collision, translated
rigidly and simulated again, appears in many events (blocks of 50 consecutive ids draw from shared families). The
collision signatures (one per event and vertex: a hash of the generation-0 particles' type, charge and momentum) come
from the read-only gnn-displaced project (gd/signatures/clusters_v1.parquet). This module counts the val collisions
that also occur in our training events and compares the DM efficiency of generation-0 targets whose collision has a
training twin with those whose collision has none, from the per-target tables of hnl.trackml.displaced_diag.
Particles produced in the simulation (generation >= 1) are never copies.

    python -m hnl.trackml.collision_reuse > results/trackml_collision_reuse_v1.txt
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from hnl.trackml.config import event_set
from hnl.trackml.displaced_diag import generation

SIG = "/ceph/submit/data/user/j/jinboz1/gnn-displaced/gd/signatures/clusters_v1.parquet"
DIAG = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/diag/displaced"
BINS = [0, 1, 3, 10, 30, 100, 300, np.inf]


def main() -> None:
    train, val = event_set("train"), event_set("val")
    c = pd.read_parquet(SIG)
    in_train = set(c[c.event_id.isin(train)].sid)
    v = c[c.event_id.isin(val)].assign(twin=lambda d: d.sid.isin(in_train))
    print(f"signatures: {SIG}")
    print(f"train events {c[c.event_id.isin(train)].event_id.nunique()} of {len(train)}, "
          f"val events {v.event_id.nunique()} of {len(val)}")
    print(f"val collisions with a twin in train: {v.twin.sum()} of {len(v)} ({v.twin.mean():.4f}); "
          f"hard scatter (vertex 1): {v[v.vtx == 1].twin.sum()} of {(v.vtx == 1).sum()}")
    for cond in ("full", "1gev"):
        d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DIAG}/diag_{cond}_b*.parquet"))],
                      ignore_index=True)
        d = d[generation(d.particle_id) == 0].copy()
        d["vtx"] = ((d.particle_id.astype(np.uint64) >> np.uint64(52)) & np.uint64(0xFFF)).astype(np.int32)
        d = d.merge(v[["event_id", "vtx", "twin"]], left_on=["event", "vtx"], right_on=["event_id", "vtx"],
                    how="left")
        if d.twin.isna().any():
            raise SystemExit(f"{cond}: {int(d.twin.isna().sum())} targets without a signature")
        d["bin"] = pd.cut(d.d0.abs(), BINS, right=False)
        print(f"\n== {cond}: generation-0 targets of {d.event.nunique()} val events, DM efficiency by training twin")
        t = d.groupby(["bin", "twin"], observed=True).dm.agg(["mean", "sum", "size"]).unstack()
        print(t.to_string(float_format=lambda x: f"{x:.4f}"))
        for tw, g in d.groupby("twin"):
            p, n = g.dm.mean(), len(g)
            print(f"all |d0|, twin={tw}: {100 * p:.3f} +- {100 * np.sqrt(p * (1 - p) / n):.3f} % "
                  f"({int(g.dm.sum())} of {n})")


if __name__ == "__main__":
    main()
