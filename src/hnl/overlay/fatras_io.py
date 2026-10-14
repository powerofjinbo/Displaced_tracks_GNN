"""ACTS v15 Fatras files of the pileup overlay (docs/pileup_overlay_plan.md sections 2.1 and 4.2): the particle CSVs that
ActsExampleFatrasGeneric reads with --input-dir, and its CsvSimHitWriter output packed into one parquet file.

Particle CSV: event<NNNNNNNNN>-particles.csv with the columns of the ACTS v15 CsvParticleReader (PARTICLE_COLUMNS;
mm, ns, GeV). Barcodes are written as unsigned integers.

Sim hits: event<NNNNNNNNN>-hits.csv (particle_id, geometry_id, tx, ty, tz, tt, tpx, tpy, tpz, te, deltapx, deltapy,
deltapz, deltae, index). `pack_sim_hits` keeps one row per hit with the Fatras event number, the barcode, the decoded
geometry id (hnl.geometry.ids layout: volume, layer, sensitive), the true position and momentum at the surface and the
hit index along the particle, and can delete the CSV files afterwards (slurm/overlay/fatras.sbatch).

    python -m hnl.overlay.fatras_io pack --sim-dir DIR --out FILE.parquet [--remove-csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hnl.geometry.ids import decode_geometry_id

PARTICLE_COLUMNS = ["particle_id", "particle_type", "process", "vx", "vy", "vz", "vt", "px", "py", "pz", "m", "q"]
MASS = {11: 0.000510999, 13: 0.105658, 211: 0.139570, 321: 0.493677, 2212: 0.938272}   # GeV, as scripts/acts/gun.py
SIM_DTYPES = {"particle_id": np.uint64, "geometry_id": np.uint64}
FATRAS_OUTPUTS = ("hits", "particles_initial", "particles_final")
PACK_SCHEMA = pa.schema([
    ("event", pa.int32()), ("particle_id", pa.uint64()), ("geometry_id", pa.uint64()), ("volume", pa.uint8()),
    ("layer", pa.uint16()), ("sensitive", pa.int32()), ("tx", pa.float32()), ("ty", pa.float32()),
    ("tz", pa.float32()), ("tpx", pa.float32()), ("tpy", pa.float32()), ("tpz", pa.float32()),
    ("index", pa.int32()),
])
_EVENT_RE = re.compile(r"event(\d{9})-hits\.csv$")


def particles_path(directory: str, event: int) -> str:
    return os.path.join(directory, f"event{int(event):09d}-particles.csv")


def write_particles(df: pd.DataFrame, directory: str, event: int) -> str:
    """Write one Fatras input event. particle_id must be unique non-negative integers (written unsigned)."""
    if list(df.columns) != PARTICLE_COLUMNS:
        raise ValueError(f"columns {list(df.columns)} differ from {PARTICLE_COLUMNS}")
    pid = df.particle_id.to_numpy()
    if pid.dtype.kind not in "iu" or (pid.dtype.kind == "i" and (pid < 0).any()):
        raise ValueError("particle_id must be non-negative integers")
    if pd.Series(pid).duplicated().any():
        raise ValueError(f"event {event}: duplicate particle_id")
    out = df.copy()
    out["particle_id"] = pid.astype(np.uint64)
    os.makedirs(directory, exist_ok=True)
    path = particles_path(directory, event)
    out.to_csv(path, index=False, float_format="%.10g")
    return path


def read_sim_hits(directory: str, event: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(directory, f"event{int(event):09d}-hits.csv"), dtype=SIM_DTYPES)


def sim_hit_events(directory: str) -> list[int]:
    return sorted(int(m.group(1)) for f in os.listdir(directory) if (m := _EVENT_RE.match(f)))


def pack_frame(h: pd.DataFrame, event: int) -> pa.Table:
    gid = h.geometry_id.to_numpy(np.uint64)
    g = decode_geometry_id(gid.view(np.int64))
    if (g["boundary"] != 0).any() or (g["approach"] != 0).any():
        raise ValueError(f"event {event}: sim hits on non-sensitive surfaces (run Fatras with --fatras-hits sensitive)")
    f32 = lambda c: pa.array(h[c].to_numpy(np.float32))
    return pa.Table.from_arrays([
        pa.array(np.full(len(h), event, np.int32)), pa.array(h.particle_id.to_numpy(np.uint64)), pa.array(gid),
        pa.array(g["volume"].astype(np.uint8)), pa.array(g["layer"].astype(np.uint16)),
        pa.array(g["sensitive"].astype(np.int32)), f32("tx"), f32("ty"), f32("tz"), f32("tpx"), f32("tpy"),
        f32("tpz"), pa.array(h["index"].to_numpy(np.int32)),
    ], schema=PACK_SCHEMA)


def pack_sim_hits(directory: str, out: str, remove_csv: bool = False) -> dict:
    """All event*-hits.csv of a Fatras output directory -> one parquet file (one row group per event)."""
    events = sim_hit_events(directory)
    if not events:
        raise FileNotFoundError(f"no event*-hits.csv in {directory}")
    n = 0
    tmp = out + ".tmp"
    with pq.ParquetWriter(tmp, PACK_SCHEMA, compression="zstd") as w:
        for e in events:
            t = pack_frame(read_sim_hits(directory, e), e)
            w.write_table(t, row_group_size=max(t.num_rows, 1))
            n += t.num_rows
    os.replace(tmp, out)
    removed = 0
    if remove_csv:
        for e in events:
            for k in FATRAS_OUTPUTS:
                p = os.path.join(directory, f"event{e:09d}-{k}.csv")
                if os.path.exists(p):
                    os.remove(p)
                    removed += 1
    return {"events": len(events), "hits": int(n), "out": out, "csv_removed": removed}


def read_packed(path: str, columns=None, events=None) -> pd.DataFrame:
    flt = None if events is None else [("event", "in", [int(e) for e in events])]
    return pd.read_parquet(path, columns=columns, filters=flt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--remove-csv", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(pack_sim_hits(a.sim_dir, a.out, a.remove_csv), indent=1))


if __name__ == "__main__":
    main()
