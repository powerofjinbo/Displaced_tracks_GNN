"""ACTS v15 Fatras files of the pileup overlay (docs/pileup_overlay_plan.md sections 2.1 and 4.2): the particle CSVs that
ActsExampleFatrasGeneric reads with --input-dir, and its CsvSimHitWriter output packed into one parquet file.

Particle CSV: event<NNNNNNNNN>-particles.csv with the columns of the ACTS v15 CsvParticleReader (PARTICLE_COLUMNS;
mm, ns, GeV). Barcodes are written as unsigned integers.

Sim hits: event<NNNNNNNNN>-hits.csv (particle_id, geometry_id, tx, ty, tz, tt, tpx, tpy, tpz, te, deltapx, deltapy,
deltapz, deltae, index). `pack_sim_hits` keeps one row per hit with the Fatras event number, the barcode, the decoded
geometry id (hnl.geometry.ids layout: volume, layer, sensitive), the true position and momentum at the surface and the
hit index along the particle, and can delete the CSV files afterwards (slurm/overlay/fatras.sbatch).
Beside it, `<out stem>_particles.parquet` lists the particles Fatras simulated (event, particle_id, in the initial and
in the final state), so that the particles it did not simulate stay known after the CSV files are removed, and the
final state of each (final_vx, final_vy, final_vz [mm], final_px, final_py, final_pz [GeV]; NaN for a particle missing
from the final state). The final position shows a particle that ends outside the detector world without a log line
(addendum H section 9.2, content; hnl.overlay.gun_content.silent_exits).

Particles Fatras did not simulate. v15 `ActsFatras::Simulation` erases a particle whose propagation fails from the
initial-state output, keeps none of its hits and only logs it ("event N particle id=(...) failed to simulate with
error ..."). `failed_particles` reads these lines; `simulation_status` classifies every input particle as simulated,
failed (logged) or below pMin (never simulated, by the selection) and checks that the log and the output agree.

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
FINAL_STATE = ("vx", "vy", "vz", "px", "py", "pz")
SIM_PARTICLES_SCHEMA = pa.schema([("event", pa.int32()), ("particle_id", pa.uint64()), ("initial", pa.bool_()),
                                  ("final", pa.bool_()), *[(f"final_{k}", pa.float64()) for k in FINAL_STATE]])
# v15 FatrasSimulation: "event <n> particle id=(<vertex primary>|<vertex secondary>|<particle>|<generation>|<sub>)
# |pdg=...|p=... failed to simulate with error <category>:<code>: <message>"
_FAILED_RE = re.compile(r"event (\d+) particle id=\((\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\)\S* failed to simulate with "
                        r"error (.*)$")


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


def sim_particles_path(out: str) -> str:
    """The simulated-particle list written beside a packed sim-hit file."""
    return os.path.splitext(out)[0] + "_particles.parquet"


def sim_particles_frame(directory: str, event: int) -> pa.Table:
    """(event, particle_id, initial, final, final_vx ... final_pz) of every particle in the initial or final state of
    one Fatras event; the final-state columns are NaN for a particle without a final state (or a file without them)."""
    ids, final = {}, None
    for k in ("initial", "final"):
        p = os.path.join(directory, f"event{int(event):09d}-particles_{k}.csv")
        t = pd.read_csv(p, dtype={"particle_id": np.uint64}) if os.path.exists(p) else None
        ids[k] = t.particle_id.to_numpy(np.uint64) if t is not None else np.zeros(0, np.uint64)
        if k == "final" and t is not None:
            final = t
    allp = np.union1d(ids["initial"], ids["final"]).astype(np.uint64)
    cols = []
    for c in FINAL_STATE:
        v = np.full(len(allp), np.nan)
        if final is not None and c in final:
            f = pd.Series(final[c].to_numpy(np.float64), index=final.particle_id.to_numpy(np.uint64))
            if f.index.duplicated().any():
                raise ValueError(f"event {event}: a particle appears twice in the final state")
            v = f.reindex(allp).to_numpy(np.float64)
        cols.append(pa.array(v))
    return pa.Table.from_arrays([pa.array(np.full(len(allp), event, np.int32)), pa.array(allp),
                                 pa.array(np.isin(allp, ids["initial"])), pa.array(np.isin(allp, ids["final"])),
                                 *cols], schema=SIM_PARTICLES_SCHEMA)


def pack_sim_hits(directory: str, out: str, remove_csv: bool = False) -> dict:
    """All event*-hits.csv of a Fatras output directory -> one parquet file (one row group per event), and the
    simulated particles of every event -> `sim_particles_path(out)`."""
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
    pout = sim_particles_path(out)
    with pq.ParquetWriter(pout + ".tmp", SIM_PARTICLES_SCHEMA, compression="zstd") as w:
        for e in events:
            w.write_table(sim_particles_frame(directory, e))
    os.replace(tmp, out)
    os.replace(pout + ".tmp", pout)
    removed = 0
    if remove_csv:
        for e in events:
            for k in FATRAS_OUTPUTS:
                p = os.path.join(directory, f"event{e:09d}-{k}.csv")
                if os.path.exists(p):
                    os.remove(p)
                    removed += 1
    return {"events": len(events), "hits": int(n), "out": out, "csv_removed": removed}


def barcode(vprim, vsec, particle, gen=0, sub=0) -> np.ndarray:
    """ACTS v15 barcode (vertex primary [52, 64), vertex secondary [40, 52), particle [24, 40), generation [16, 24),
    sub-particle [0, 16); scripts/acts/gun.py `barcode`, hnl.geometry.ids.decode_barcode) as uint64."""
    f = [np.asarray(x, np.uint64) for x in (vprim, vsec, particle, gen, sub)]
    return ((f[0] << np.uint64(52)) | (f[1] << np.uint64(40)) | (f[2] << np.uint64(24)) | (f[3] << np.uint64(16))
            | f[4]).astype(np.uint64)


def failed_particles(log_text: str) -> pd.DataFrame:
    """Every particle a v15 Fatras log reports as failed: event, particle_id (barcode, uint64) and the error text."""
    rows = [m.groups() for m in map(_FAILED_RE.search, log_text.splitlines()) if m]
    if not rows:
        return pd.DataFrame({"event": np.zeros(0, np.int64), "particle_id": np.zeros(0, np.uint64),
                             "error": np.zeros(0, object)})
    a = np.array([r[:6] for r in rows], np.int64)
    return pd.DataFrame({"event": a[:, 0], "particle_id": barcode(*a[:, 1:].T), "error": [r[6] for r in rows]})


def simulation_status(inputs: pd.DataFrame, simulated: pd.DataFrame, failed: pd.DataFrame,
                      pmin_gev: float) -> pd.DataFrame:
    """Status of every input particle (columns event, particle_id, q, px, py, pz): 'simulated' (in Fatras's initial
    state), 'failed' (logged as failed) or 'below_pmin' (charged with p <= pMin, or neutral: not selected).
    simulated: event, particle_id, initial (`sim_particles_frame`); failed: `failed_particles`. Raises if an input
    particle is missing from the output without a reason, if a logged failure was simulated or is not an input, or if
    the output holds a particle that is not an input (v15 makes no secondaries with these settings)."""
    inp =inputs.assign(event=inputs.event.astype(np.int64), particle_id=inputs.particle_id.astype(np.uint64))
    ini = simulated[simulated.initial.astype(bool)][["event", "particle_id"]].astype(
        {"event": np.int64, "particle_id": np.uint64})
    fl = failed[["event", "particle_id"]].astype({"event": np.int64, "particle_id": np.uint64})
    if fl.duplicated().any():
        raise ValueError("a particle is logged as failed twice")
    j = inp.merge(ini.assign(sim=True), on=["event", "particle_id"], how="outer", indicator="_s")
    if (j._s == "right_only").any():
        raise ValueError(f"{int((j._s == 'right_only').sum())} simulated particles are not inputs")
    j = j.drop(columns="_s").merge(fl.assign(fail=True), on=["event", "particle_id"], how="outer", indicator="_f")
    if (j._f == "right_only").any():
        raise ValueError(f"{int((j._f == 'right_only').sum())} logged failures are not inputs")
    sim, fail = j.sim.fillna(False).astype(bool).to_numpy(), j.fail.fillna(False).astype(bool).to_numpy()
    if (sim & fail).any():
        raise ValueError("a particle is both simulated and logged as failed")
    p = np.sqrt(j.px.to_numpy(np.float64) ** 2 + j.py.to_numpy(np.float64) ** 2 + j.pz.to_numpy(np.float64) ** 2)
    below = (j.q.to_numpy() == 0) | (p <= pmin_gev)
    unexplained = ~sim & ~fail & ~below
    if unexplained.any():
        raise ValueError(f"{int(unexplained.sum())} input particles above pMin are neither simulated nor logged as "
                         f"failed")
    status = np.where(sim, "simulated", np.where(fail, "failed", "below_pmin"))
    out = j.drop(columns=["sim", "fail", "_f"]).assign(status=status)
    return out.sort_values(["event", "particle_id"], kind="stable").reset_index(drop=True)


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
