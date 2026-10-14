"""acorn feature store: the sharded event tables (`hnl.data.convert`) as acorn PyG events, packed per (sample, split).

Format (docs/acorn_map.md §2, docs/acorn_pitfalls.md): acorn default naming (short names on disk), one event per
file `event<ID>-graph.pyg.gz` (acorn's `save_pyg`), ID = f"{sample code}{index:08d}" (9 digits, unique across
samples). Fields: hit_id = row (arange N), x (-> hit_x), r, phi, z, eta (-> hit_eta), hit_y, hit_particle_id,
track_edges = npz chain edges oriented outward in r^2 + z^2 (ties by index), and per true edge particle_id
(-> track_particle_id), pt (-> track_particle_pt, GeV, table value), eta_particle, nhits; event_id, config = [].

Each (sample, split) becomes one uncompressed tar `<out>/<sample>_<split>.tar` whose members are the flat file names,
so a job extracts it straight into `<local fs>/<trainset|valset|testset>/`.

    python -m hnl.data.acorn_store --out <derived/acorn_fs/v1>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from hnl.data.convert import SHARD, read_table
from hnl.data.splits import read_split

SAMPLE_CODE = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}
SPLITS = {"ttbar": ("train", "val", "test"), "hnl10_100": ("train", "val", "test"),
          "hnl15_200": ("train", "val", "test"), "hnl15_100": ("external",)}
# the generated samples (hnl.data.npz.GENERATED), only with HNL_GENERATED=1; splits of addendum G section 3
GENERATED_CODE = {"generic_iso": 5, "generic_dense": 6, "hss15_100": 7, "hss40_300": 8}
GENERATED_SPLITS = {"generic_iso": ("train", "val", "test"), "generic_dense": ("train", "val", "test"),
                    "hss15_100": ("all",), "hss40_300": ("all",)}
# the HNL points (hnl.data.shards.POINTS; evaluation only, addendum G step T2), also only with HNL_GENERATED=1; their
# split lists are splits/hnl_<m>_<ctau>_all.txt (hnl.eval.addendum_g.SPLIT_STEM). Two-digit codes: their event ids have
# ten digits, and parse_event_id reads every digit before the last eight as the code.
POINT_CODE = {"hnl5_50": 10, "hnl20_200": 11, "hnl15_500": 12, "hnl10_100_ref": 13, "hnl15_200_ref": 14}
if os.environ.get("HNL_GENERATED") == "1":
    SAMPLE_CODE.update(GENERATED_CODE)
    SPLITS.update(GENERATED_SPLITS)
    SAMPLE_CODE.update(POINT_CODE)
CODE_SAMPLE = {v: k for k, v in SAMPLE_CODE.items()}
DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")


def event_id(sample: str, index: int) -> str:
    return f"{SAMPLE_CODE[sample]}{index:08d}"


def parse_event_id(eid: str) -> tuple[str, int]:
    return CODE_SAMPLE[int(eid[:-8])], int(eid[-8:])


def orient_outward(edges: np.ndarray, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Orient each edge from the hit with smaller r^2 + z^2 to the larger (ties: smaller index first)."""
    R = r.astype(np.float64) ** 2 + z.astype(np.float64) ** 2
    a, b = edges
    flip = (R[a] > R[b]) | ((R[a] == R[b]) & (a > b))
    return np.stack([np.where(flip, b, a), np.where(flip, a, b)])


def acorn_event(hits: pd.DataFrame, edges: pd.DataFrame, parts: pd.DataFrame, eid: str) -> Data:
    hits = hits.sort_values("row")
    n = len(hits)
    if not np.array_equal(hits.row.to_numpy(), np.arange(n)):
        raise ValueError(f"event {eid}: rows are not 0..N-1")
    r = hits.r.to_numpy(np.float32)
    phi = hits.phi.to_numpy(np.float32)
    z = hits.z.to_numpy(np.float32)
    pid = hits.pid.to_numpy(np.int64)
    te = orient_outward(np.stack([edges.src.to_numpy(np.int64), edges.dst.to_numpy(np.int64)]), r, z)
    tpid = pid[te[0]]
    if not np.array_equal(tpid, pid[te[1]]):
        raise ValueError(f"event {eid}: a true edge joins two particles")
    p = parts.set_index("particle_id").loc[tpid]
    g = Data(
        hit_id=torch.arange(n, dtype=torch.int64),
        x=torch.from_numpy(hits.x.to_numpy(np.float32)),
        r=torch.from_numpy(r), phi=torch.from_numpy(phi), z=torch.from_numpy(z),
        eta=torch.from_numpy(np.arcsinh(z / np.maximum(r, 1e-6)).astype(np.float32)),
        track_edges=torch.from_numpy(te),
        particle_id=torch.from_numpy(tpid),
        pt=torch.from_numpy(p.pt.to_numpy(np.float32)),
        eta_particle=torch.from_numpy(np.nan_to_num(p.eta.to_numpy(np.float32), nan=99.0)),
        nhits=torch.from_numpy(p.n_hits.to_numpy(np.int64)),
    )
    g.hit_y = torch.from_numpy(hits.y.to_numpy(np.float32))
    g.hit_particle_id = torch.from_numpy(pid)
    g.event_id = eid
    g.config = []
    return g


def _serialise(g: Data) -> bytes:
    """Exactly what acorn's save_pyg writes (torch.save, gzip level 1)."""
    import gzip
    buf = io.BytesIO()
    torch.save(g, buf)
    return gzip.compress(buf.getvalue(), compresslevel=1)


def build_tar(sample: str, split: str, out: str, events=None, name: str | None = None, root: str | None = None) -> dict:
    """The tar of one (sample, split): the split list's events (or the given ones) from the event tables under root
    (default derived/events/v1), as <out>/<name> (default <sample>_<split>.tar)."""
    events = read_split(sample, split) if events is None else np.sort(np.asarray(events, dtype=np.int64))
    path = os.path.join(out, name or f"{sample}_{split}.tar")
    if os.path.exists(path):
        raise FileExistsError(path)
    root = root or os.path.join(DERIVED, "events", "v1")
    n_written = 0
    tmp = path + ".tmp"
    with tarfile.open(tmp, "w") as tar:
        for first in range(0, int(events.max()) + 1, SHARD):
            ev = events[(events >= first) & (events < first + SHARD)]
            if not len(ev):
                continue
            hits = read_table(root, "hits", sample, events=ev)
            edges = read_table(root, "edges", sample, events=ev)
            parts = read_table(root, "particles", sample, events=ev,
                               columns=["event", "particle_id", "pt", "eta", "n_hits"])
            hg, eg, pg = hits.groupby("event"), edges.groupby("event"), parts.groupby("event")
            for e in ev:
                eid = event_id(sample, int(e))
                data = _serialise(acorn_event(hg.get_group(e), eg.get_group(e), pg.get_group(e), eid))
                info = tarfile.TarInfo(name=f"event{eid}-graph.pyg.gz")
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(data))
                n_written += 1
    os.replace(tmp, path)
    return {"events": n_written, "bytes": os.path.getsize(path)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default=",".join(SAMPLE_CODE))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    t0 = time.time()
    for s in a.samples.split(","):
        for sp in SPLITS[s]:
            meta[f"{s}_{sp}"] = build_tar(s, sp, a.out)
            print(f"[acorn_store] {s} {sp}: {meta[f'{s}_{sp}']} {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"tars": meta, "slurm_job": os.environ.get("SLURM_JOB_ID"), "elapsed_s": time.time() - t0}, f, indent=1)


if __name__ == "__main__":
    main()
