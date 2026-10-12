"""Per-particle truth for Stage-1 training targets and weights (never as network inputs).

build: one parquet per labeled sample under derived/truth_lookup/v1 with (event, particle_id, cat, absd0, rv, vtx)
for every particle with hits:
cat 0 other / 1 HNL daughter / 2 W muon; absd0 = truth-hit |d0| [mm] (NaN if no fit); rv = daughter-pair vertex
radius [mm] (NaN if none); vtx = vprim * 1e6 + vsec for HNL daughters (their shared barcode vertex, RESULTS.md
2026-10-03), -1 otherwise.

TruthLookup(samples).decorate(event) adds hit_cat (int8), hit_absd0 (float32), hit_rv (float32) and hit_vtx (int64)
to an acorn event from its hit_particle_id and event_id.

    python -m hnl.models.truth_lookup build
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from hnl.data.acorn_store import SAMPLE_CODE

DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
OUT = os.path.join(DERIVED, "truth_lookup", "v1")
LABELED = ("ttbar", "hnl10_100", "hnl15_200", "hnl15_100")
CAT = {"other": 0, "hnl": 1, "w": 2}


def build():
    from hnl.data.convert import read_table
    from hnl.graphs.window_study import truth_particles
    os.makedirs(OUT, exist_ok=True)
    for s in LABELED:
        p = read_table(os.path.join(DERIVED, "events", "v1"), "particles", s,
                       columns=["event", "particle_id", "category", "vprim", "vsec", "n_hits"])
        p = p[p.n_hits > 0]                       # only particles that can appear as hit_particle_id
        t = truth_particles(s)
        p = p.merge(t, on=["event", "particle_id"], how="left")
        out = pd.DataFrame({"event": p.event.astype(np.int64), "particle_id": p.particle_id.astype(np.int64),
                            "cat": p.category.map(CAT).astype(np.int8),
                            "absd0": p.d0.abs().astype(np.float32), "rv": p.rv.astype(np.float32)})
        out["vtx"] = np.where(out.cat == 1, p.vprim.astype(np.int64) * 1_000_000 + p.vsec.astype(np.int64), -1)
        out = out.sort_values(["event", "particle_id"]).reset_index(drop=True)
        out.to_parquet(os.path.join(OUT, f"{s}.parquet"), index=False)
        print(s, len(out), "particles", int((out.cat == 1).sum()), "HNL daughters", flush=True)


class TruthLookup:
    def __init__(self, samples=LABELED, directory: str = OUT):
        self.tab = {}
        for s in samples:
            d = pd.read_parquet(os.path.join(directory, f"{s}.parquet"))
            code = SAMPLE_CODE[s]
            ev = d.event.to_numpy()
            starts = np.searchsorted(ev, np.arange(ev.max() + 2))
            self.tab[code] = (starts, d.particle_id.to_numpy(), d.cat.to_numpy(), d.absd0.to_numpy(),
                              d.rv.to_numpy(), d.vtx.to_numpy())

    def lookup(self, event_id: int, pid: np.ndarray):
        code, idx = divmod(int(event_id), 100_000_000)
        starts, ppid, cat, d0, rv, vtx = self.tab[code]
        lo, hi = starts[idx], starts[idx + 1]
        j = lo + np.searchsorted(ppid[lo:hi], pid)
        j = np.minimum(j, max(hi - 1, lo))
        ok = (j < hi) & (ppid[j] == pid)
        return (np.where(ok, cat[j], 0).astype(np.int8), np.where(ok, d0[j], np.nan).astype(np.float32),
                np.where(ok, rv[j], np.nan).astype(np.float32), np.where(ok, vtx[j], -1).astype(np.int64))

    def decorate(self, event):
        import torch
        eid = event.event_id
        while isinstance(eid, (list, tuple)):
            eid = eid[0]
        cat, d0, rv, vtx = self.lookup(int(eid), event.hit_particle_id.cpu().numpy())
        event.hit_cat = torch.from_numpy(cat)
        event.hit_absd0 = torch.from_numpy(d0)
        event.hit_rv = torch.from_numpy(rv)
        event.hit_vtx = torch.from_numpy(vtx)
        return event


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
