"""ACTS v15 CSV output of one HNL chunk -> one tar shard of per-event npz files in the Zenodo format, plus one truth
sidecar (docs/hnl_points_spec.md section 5).

Each event is converted by scripts/acts/csv_to_npz.py (`convert`: smeared measurements, hit-to-particle links from the
measurement-simhit map, true edges in time order, particles_final as the particle table). Added here:
  - parent_pid (column 12) as in the labeled Zenodo HNL samples: for every generator particle (barcode generation 0)
    the PDG code of its generator mother, from event<N>-parents.csv of hnl_pythia.cc (so W muon +-24, N daughters
    +-50); 0 for particles created in the simulation (generation > 0), as in the Zenodo files;
  - the events are written into one uncompressed tar, member `<event>.npz` (np.savez bytes, identical to a file
    written by np.savez), fixed member metadata (mtime 0, uid/gid 0, mode 0444), written under a temporary name and
    renamed when complete;
  - sidecar truth_<first>.npz: the csv_to_npz sidecar (p_*: initial state of every particle with hits; e_*: per-event
    counts) plus g_*: the generator truth of hnl_pythia.cc per event (W, N, neutrino, decay vertex, proper decay
    length, barcodes of the prompt muon and the N daughters).

    python scripts/acts/hnl_to_npz.py CSVDIR --detectors detectors.csv --tar OUT.tar --sidecar TRUTH.npz
        --first N --events N [--keep-csv]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_to_npz as C  # noqa: E402

from hnl.data.npz import COL, NPZ_KEYS  # noqa: E402  (csv_to_npz put src/ on sys.path)

GEN_STEMS = ("particles_gen", "parents")


def fill_parents(ev: dict, parents: pd.DataFrame) -> int:
    """Set column 12 of ev['particles'] from the generator parents; returns the number of generator rows matched."""
    P = ev["particles"]
    ids = P[:, COL["particle_id"]].astype(np.int64).view(np.uint64)
    gen = (ids >> np.uint64(16)) & np.uint64(0xFF)
    lut = dict(zip(parents.particle_id.to_numpy(np.uint64).tolist(), parents.parent_pdg.to_numpy(np.int64).tolist()))
    col = np.zeros(len(P), dtype=np.float64)
    n = 0
    for k in np.flatnonzero(gen == 0):
        v = lut.get(int(ids[k]))
        if v is None:
            raise ValueError(f"generator particle {int(ids[k])} missing from parents.csv")
        col[k] = v
        n += 1
    P[:, COL["parent_pid"]] = col
    return n


def npz_bytes(ev: dict) -> bytes:
    assert {k: str(v.dtype) for k, v in ev.items()} == NPZ_KEYS
    buf = io.BytesIO()
    np.savez(buf, **ev)
    return buf.getvalue()


def add_member(tf: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    ti.mode = 0o444
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    tf.addfile(ti, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvdir")
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--keep-csv", action="store_true")
    a = ap.parse_args()
    for path in (a.tar, a.sidecar):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists: not overwriting")
    surf = C.surfaces(a.detectors)
    os.makedirs(os.path.dirname(os.path.abspath(a.tar)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.sidecar)), exist_ok=True)
    sides, stats = [], []
    tmp = a.tar + ".part"
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tf:
        for ev in range(a.first, a.first + a.events):
            e, side, st = C.convert(a.csvdir, ev, surf, "measured")
            par = pd.read_csv(os.path.join(a.csvdir, f"event{ev:09d}-parents.csv"), dtype={"particle_id": np.uint64})
            st["n_gen_rows_labeled"] = fill_parents(e, par)
            add_member(tf, f"{ev}.npz", npz_bytes(e))
            sides.append(side.assign(event=ev))
            stats.append({"event": ev, **st})
            if not a.keep_csv:
                for stem in C.STEMS + GEN_STEMS:
                    os.remove(os.path.join(a.csvdir, f"event{ev:09d}-{stem}.csv"))
    S = pd.concat(sides, ignore_index=True)
    T = pd.DataFrame(stats)
    G = pd.read_csv(os.path.join(a.csvdir, f"hnl_gen_{a.first}.csv"))
    G = G[(G.event >= a.first) & (G.event < a.first + a.events)]
    if len(G) != a.events:
        raise ValueError(f"generator truth has {len(G)} rows for {a.events} events")
    U64 = ("particle_id", "mu_w_id", "d1_id", "d2_id")
    INT = ("event", "particle_type", "process", "n_tries", "w_charge", "n_id", "nu_id", "n_visible_daughters",
           "n_written")
    out = {f"p_{c}": S[c].to_numpy(np.uint64 if c in U64 else np.int64 if c in INT else np.float64) for c in S.columns}
    out.update({f"e_{c}": T[c].to_numpy() for c in T.columns})
    out.update({f"g_{c}": G[c].to_numpy(np.uint64 if c in U64 else np.int64 if c in INT else np.float64)
                for c in G.columns})
    stmp = a.sidecar + ".part.npz"
    np.savez(stmp, xmode=np.array("measured"), **out)
    os.replace(stmp, a.sidecar)
    os.replace(tmp, a.tar)
    print(f"{len(T)} events -> {a.tar}; measurements {int(T.n_meas.sum())}; dropped duplicate hits "
          f"{int(T.n_dup.sum())}; |pull| > 5: {int(T.res_abs_gt5.sum())}; generator rows labeled "
          f"{int(T.n_gen_rows_labeled.sum())}")


if __name__ == "__main__":
    main()
