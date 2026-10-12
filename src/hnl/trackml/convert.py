"""TrackML events -> compact per-event tables for the Canto pipeline (docs/trackml_plan.md sections 2.3, 3 and 6.0).

Input. The four csv members of every event (hits, cells, particles, truth), read in place from a TrackML
accuracy-phase archive: the local train_1.zip (downloaded once, md5 checked by slurm/trackml/convert_train_1.sbatch),
or a remote Zenodo zip through HTTP range requests (hnl.trackml.split.HTTPRangeFile; train_2 ... train_5 need not be
stored). Nothing is unpacked; `zipfile` checks the CRC-32 of every member it reads.

Output. <root>/<table>/<archive>/shard_<first:05d>.parquet, zstd, first = event // 500 * 500 (the shard rule of
hnl.data.convert, so `hnl.data.convert.read_table(root, table, archive, events=...)` finds the right file and never
silently misses an event). Events in increasing order, one row group per event, the rows of an event in hit_id
order. Events that fail the data-quality rule are not written (listed in dq_<archive>.tsv and meta_<archive>.json).
`read_table` below returns the tables with the column names and meanings of hnl.data.convert's tables and adds the
derived columns.

Stored columns:
  hits       event int32; x, y, z float32 [mm] (the csv values exactly: the csv has 6 significant digits);
             volume, layer uint8, module uint16 (TrackML ids: the endcap module numbers differ from our ACTS files,
             docs/trackml_feasibility.md section 1); pid int64 (TrackML particle_id, 0 = noise hit); rank uint8
             (position of the hit along its particle, 0 = first; NOISE_RANK = 255 for noise hits); ncells uint16 and
             charge float32 (number of cells and sum of the cell values; float32 because a few cells carry anomalous
             values, e.g. 716,689 in training event 1532 where every other value is <= 1, which overflow float16);
             span0, span1 uint16 (cluster extent in
             cells along ch0 / ch1: max - min + 1; with detectors.csv they give the Exa.TrkX cluster-shape inputs);
             weight float32 (TrackML score weight; stored for the dev, calibration, val and test events, NaN for the
             other training events, plan section 6.0)
  particles  every particles.csv row: event int32; particle_id int64; vx, vy, vz float32 [mm] (production vertex);
             px, py, pz float32 [GeV] (initial momentum); q int8; n_hits int16 (csv nhits, checked against the truth
             count); pt, eta float32 (initial momentum); rv float32 (hypot(vx, vy)); d0 float32 [mm] (production
             vertex helix, our sign convention, |B| = 2 T, Bz > 0: plan section 3.2); category (primary, K_S, Lambda,
             conversion, other: plan section 3.3); vprim int16 (particle_id >> 52; 1 = hard scatter); n_pixel_hits
             int16 (volumes 7, 8, 9); r_first float32 (r of the rank-0 hit, NaN without hits); population flags of
             plan section 3.4: is_target (pT > 1 GeV, |eta| < 4, >= 5 hits), is_target_pix3 (target with >= 3 pixel
             hits), is_prompt_target (|d0| < 1 mm), is_displaced_target (|d0| >= 10 mm), is_far_displaced_target
             (|d0| >= 100 mm), is_soft_displaced (>= 5 hits, |d0| >= 10 mm, 0.5 < pT <= 1 GeV), is_hard_scatter
             (vprim == 1; hard-scatter target = is_target & is_hard_scatter)
  truth      calibration events only (plan section 6.0: dev 1000-1099 and the next 100 train_1 training ids):
             event int32, row int32, tx, ty, tz, tpx, tpy, tpz float32 for every non-noise hit
Derived on read (`read_table`):
  hits       row = hit_id - 1 (graph node order), hid = hit_id, r = hypot(x, y) and phi = atan2(y, x) computed in
             float64 from the stored float32 x, y and stored as float32 (as in our tables), geoid = volume << 56 |
             layer << 36 | module (our layout, TrackML numbering), region = the 6-region id of
             hnl.models.triplet_gnn._region (pixel / short strips / long strips x barrel / endcap, from r and |z|),
             noise = (pid == 0)
  edges      event, src, dst (rows): the true edges, consecutive hits of a particle along `rank` (src the earlier
             hit), as the chain-ordered edges of our tables (docs/definitions.md section 2)

Conventions (plan section 3.1). Hits of a particle are ordered by the distance of their true intersection point
(tx, ty, tz) from the particle's production vertex (ties: smaller hit_id first); true edges join consecutive hits.
Noise hits are nodes but never part of a true edge. Every derived quantity (pt, eta, rv, d0, category, rank,
flags) is computed in float64 from the stored float32 values, so it can be recomputed exactly from the tables; the
flags compare the stored float32 values.

Data-quality rule (plan section 2.3, the HNL criterion): an event is dropped if any hit lies more than 50 mm outside
its module's bounds in detectors.csv, or if the particle hit counts are inconsistent (csv nhits against the truth
count), or if the event's tables are structurally inconsistent (hit ids, truth rows, unknown particle ids).

    python -m hnl.trackml.convert convert --source <zip or URL> --archive train_1 [--nproc N] [--events a,b,...]
    python -m hnl.trackml.convert verify --archive train_1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hnl.data.convert import SHARD
from hnl.geometry.ids import PIXEL_VOLUMES
from hnl.trackml.split import ARCHIVES, EXT, REPO, HTTPRangeFile
from hnl.truth import helix as hx

DERIVED = os.path.join(EXT, "derived", "v1")
DETECTORS = os.path.join(EXT, "detectors.csv")
SPLIT = os.path.join(REPO, "configs", "trackml", "split_v1.json")
USER_AREA = "/ceph/submit/data/user/j/jinboz1"
TRACKML_LIMIT_BYTES = None            # plan section 7 had 40 GB for all of external/trackml; lifted by the user (2026-10-08)
MIN_FREE_BYTES = 15e9                 # plan section 7: quota headroom kept after every write
SENSE_POS = -1                        # rotation sense of positive charges: clockwise seen from +z (Bz > 0)
NOISE_RANK = 255
DQ_MAX_OUTSIDE_MM = 50.0
N_CALIB_EXTRA = 100                   # calibration events = dev + the next 100 train_1 training ids (sorted)
M_PI, M_P, M_E = 0.13957, 0.93827, 0.000511
M_KS, M_L = 0.497611, 1.115683
KINDS = ("hits", "cells", "particles", "truth")
CATEGORIES = ("primary", "K_S", "Lambda", "conversion", "other")

SCHEMAS = {
    "hits": pa.schema([
        ("event", pa.int32()), ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()),
        ("volume", pa.uint8()), ("layer", pa.uint8()), ("module", pa.uint16()), ("pid", pa.int64()),
        ("rank", pa.uint8()), ("ncells", pa.uint16()), ("charge", pa.float32()), ("span0", pa.uint16()),
        ("span1", pa.uint16()), ("weight", pa.float32()),
    ]),
    "particles": pa.schema([
        ("event", pa.int32()), ("particle_id", pa.int64()),
        ("vx", pa.float32()), ("vy", pa.float32()), ("vz", pa.float32()),
        ("px", pa.float32()), ("py", pa.float32()), ("pz", pa.float32()),
        ("q", pa.int8()), ("n_hits", pa.int16()), ("pt", pa.float32()), ("eta", pa.float32()), ("rv", pa.float32()),
        ("d0", pa.float32()), ("category", pa.string()), ("vprim", pa.int16()), ("n_pixel_hits", pa.int16()),
        ("r_first", pa.float32()),
        ("is_target", pa.bool_()), ("is_target_pix3", pa.bool_()), ("is_prompt_target", pa.bool_()),
        ("is_displaced_target", pa.bool_()), ("is_far_displaced_target", pa.bool_()),
        ("is_soft_displaced", pa.bool_()), ("is_hard_scatter", pa.bool_()),
    ]),
    "truth": pa.schema([
        ("event", pa.int32()), ("row", pa.int32()),
        ("tx", pa.float32()), ("ty", pa.float32()), ("tz", pa.float32()),
        ("tpx", pa.float32()), ("tpy", pa.float32()), ("tpz", pa.float32()),
    ]),
}
STORED_TABLES = tuple(SCHEMAS)
# Parquet encodings (measured on event 1000 against the pyarrow defaults: hits 2.52 -> 1.81 MB with a float16 charge,
# particles 0.58 -> 0.42 MB, truth 3.65 -> 1.90 MB): dictionary only for the integer / string / flag columns,
# byte-stream split for positions and charge, delta for the truth row index, plain for the other floats; zstd on top.
WRITE_OPTS = {
    k: {"use_dictionary": [f.name for f in s if not pa.types.is_floating(f.type) and f.name != "row"],
        "column_encoding": {**{c: "BYTE_STREAM_SPLIT" for c in ("x", "y", "z", "charge", "tx", "ty", "tz", "tpx", "tpy",
                                                                "tpz") if c in s.names},
                            **({"row": "DELTA_BINARY_PACKED"} if "row" in s.names else {})}}
    for k, s in SCHEMAS.items()
}
CSV_DTYPES = {
    "hits": {"hit_id": np.int64, "x": np.float64, "y": np.float64, "z": np.float64, "volume_id": np.int64,
             "layer_id": np.int64, "module_id": np.int64},
    "cells": {"hit_id": np.int64, "ch0": np.int64, "ch1": np.int64, "value": np.float64},
    "particles": {"particle_id": np.int64, "vx": np.float64, "vy": np.float64, "vz": np.float64, "px": np.float64,
                  "py": np.float64, "pz": np.float64, "q": np.int64, "nhits": np.int64},
    "truth": {"hit_id": np.int64, "particle_id": np.int64, "tx": np.float64, "ty": np.float64, "tz": np.float64,
              "tpx": np.float64, "tpy": np.float64, "tpz": np.float64, "weight": np.float64},
}


# ------------------------------------------------------------------------------------------------ sources
def open_archive(source: str) -> zipfile.ZipFile:
    """A local zip path, or an http(s) URL of a zip read through HTTP range requests."""
    if source.startswith(("http://", "https://")):
        return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(source), buffer_size=4 << 20))
    return zipfile.ZipFile(source)


def member_map(zf: zipfile.ZipFile) -> dict[int, dict[str, str]]:
    """{event: {kind: member name}} for every event member of the archive."""
    out: dict[int, dict[str, str]] = {}
    for name in zf.namelist():
        base = name.rsplit("/", 1)[-1]
        if not (base.startswith("event") and base.endswith(".csv")):
            continue
        ev, kind = base[5:-4].split("-")
        out.setdefault(int(ev), {})[kind] = name
    return out


def read_event_csv(zf: zipfile.ZipFile, names: dict[str, str]) -> dict[str, pd.DataFrame]:
    out = {}
    for k in KINDS:
        data = zf.read(names[k])                  # whole member: zipfile verifies its CRC-32 (BadZipFile otherwise)
        out[k] = pd.read_csv(io.BytesIO(data), dtype=CSV_DTYPES[k], engine="c")
        if list(out[k].columns) != list(CSV_DTYPES[k]):
            raise ValueError(f"{names[k]}: columns {list(out[k].columns)}")
    return out


# ------------------------------------------------------------------------------------------------ geometry
def module_key(volume, layer, module) -> np.ndarray:
    return (np.asarray(volume, np.int64) * 1_000_000 + np.asarray(layer, np.int64) * 10_000
            + np.asarray(module, np.int64))


def load_detectors(path: str = DETECTORS) -> dict:
    d = pd.read_csv(path)
    key = module_key(d.volume_id, d.layer_id, d.module_id)
    o = np.argsort(key)
    if np.any(np.diff(key[o]) == 0):
        raise ValueError("duplicate module in detectors.csv")
    R = d[["rot_xu", "rot_xv", "rot_xw", "rot_yu", "rot_yv", "rot_yw", "rot_zu", "rot_zv", "rot_zw"]].to_numpy()
    return {"key": key[o], "c": d[["cx", "cy", "cz"]].to_numpy()[o], "R": R.reshape(-1, 3, 3)[o],
            "t": d.module_t.to_numpy()[o], "minhu": d.module_minhu.to_numpy()[o],
            "maxhu": d.module_maxhu.to_numpy()[o], "hv": d.module_hv.to_numpy()[o]}


def module_outside_mm(x, y, z, volume, layer, module, det: dict) -> np.ndarray:
    """Distance [mm] by which each hit lies outside the bounds of its own module (detectors.csv: global = R local + c,
    local (u, v, w); trapezoid half widths minhu at v = -hv to maxhu at v = +hv, half thickness t / 2). 0 inside;
    inf for a module id missing from detectors.csv."""
    key = module_key(volume, layer, module)
    pos = np.minimum(np.searchsorted(det["key"], key), len(det["key"]) - 1)
    known = det["key"][pos] == key
    dlt = np.stack([x, y, z], axis=1).astype(np.float64) - det["c"][pos]
    loc = np.einsum("nji,nj->ni", det["R"][pos], dlt)                       # R^T (x - c)
    u, v, w = loc[:, 0], loc[:, 1], loc[:, 2]
    hv, mn, mx = det["hv"][pos], det["minhu"][pos], det["maxhu"][pos]
    hu = mn + (mx - mn) * (np.clip(v, -hv, hv) + hv) / (2 * hv)
    du = np.maximum(np.abs(u) - hu, 0.0)
    dv = np.maximum(np.abs(v) - hv, 0.0)
    dw = np.maximum(np.abs(w) - det["t"][pos] / 2, 0.0)
    return np.where(known, np.sqrt(du * du + dv * dv + dw * dw), np.inf)


def region_id(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """hnl.models.triplet_gnn._region as an integer (argmax of its one-hot): sub * 2 + endcap with sub = 0 / 1 / 2
    for r < 200 / < 720 / else, endcap = |z| > 600 (r < 200) or > 1200 mm. Inputs float32 as the model sees them."""
    r = np.asarray(r, np.float32)
    z = np.asarray(z, np.float32)
    sub = np.where(r < 200, 0, np.where(r < 720, 1, 2))
    endcap = (np.abs(z) > np.where(r < 200, np.float32(600.0), np.float32(1200.0))).astype(np.int64)
    return (sub * 2 + endcap).astype(np.int8)


def geoid(volume, layer, module) -> np.ndarray:
    """Our ACTS-style geometry id layout (hnl.geometry.ids): volume << 56 | layer << 36 | module."""
    v = np.asarray(volume, dtype=np.uint64)
    lay = np.asarray(layer, dtype=np.uint64)
    m = np.asarray(module, dtype=np.uint64)
    return ((v << np.uint64(56)) | (lay << np.uint64(36)) | m).view(np.int64)


def hit_r_phi(x, y) -> tuple[np.ndarray, np.ndarray]:
    x64 = np.asarray(x, np.float32).astype(np.float64)
    y64 = np.asarray(y, np.float32).astype(np.float64)
    return np.hypot(x64, y64).astype(np.float32), np.arctan2(y64, x64).astype(np.float32)


# ------------------------------------------------------------------------------------------------ physics
def d0_from_vertex(vx, vy, px, py, q) -> np.ndarray:
    """Signed d0 [mm] (docs/definitions.md section 3: d0 = (p x t)_z at the point of closest approach p of the
    transverse circle to the z axis) of the helix through the production vertex (vx, vy) with transverse momentum
    (px, py), |B| = 2 T (hnl.truth.helix.MM_PER_GEV) and Bz > 0 (positive charges turn clockwise seen from +z).
    The feasibility study's helix_d0_from_vertex with sense_pos = -1."""
    pt = np.hypot(px, py)
    R = hx.MM_PER_GEV * np.maximum(pt, 1e-9)
    tx, ty = px / np.maximum(pt, 1e-12), py / np.maximum(pt, 1e-12)
    sigma = (SENSE_POS * np.sign(q)).astype(np.int64)
    sigma = np.where(sigma == 0, 1, sigma)
    cx = vx + sigma * R * (-ty)                 # centre on the left of the motion for counter-clockwise turning
    cy = vy + sigma * R * tx
    return hx.perigee(cx, cy, R, sigma).d0


def inv_mass(p1: np.ndarray, p2: np.ndarray, m1: float, m2: float) -> np.ndarray:
    e1 = np.sqrt((p1 ** 2).sum(1) + m1 ** 2)
    e2 = np.sqrt((p2 ** 2).sum(1) + m2 ** 2)
    s = p1 + p2
    return np.sqrt(np.maximum((e1 + e2) ** 2 - (s ** 2).sum(1), 0))


def classify_secondaries(vx, vy, vz, px, py, pz, q, rv) -> np.ndarray:
    """The feasibility study's classify_secondaries (plan section 3.3), vectorised with the same result.

    primary: rv <= 1 mm. Secondaries are grouped by production vertex (coordinates rounded to 1e-4 mm, compared as
    strings as in the study); a vertex with exactly two secondaries of opposite charge is K_S if m(pi pi) is within
    15 MeV of 497.611 MeV, else Lambda if the smaller of the two m(p pi) assignments is within 8 MeV of 1115.683 MeV,
    else a conversion if m(ee) < 30 MeV; every other secondary is "other"."""
    n = len(vx)
    cls = np.full(n, "primary", dtype=object)
    sec = np.flatnonzero(np.asarray(rv) > 1.0)
    cls[sec] = "other"
    if not len(sec):
        return cls
    key = (pd.Series(np.round(vx[sec], 4)).astype(str) + "_" + pd.Series(np.round(vy[sec], 4)).astype(str) + "_"
           + pd.Series(np.round(vz[sec], 4)).astype(str)).to_numpy()
    codes, _ = pd.factorize(key)
    size = np.bincount(codes)
    two = size[codes] == 2
    if not two.any():
        return cls
    s2 = sec[two]
    c2 = codes[two]
    o = np.argsort(c2, kind="stable")
    a, b = s2[o][0::2], s2[o][1::2]
    if not np.array_equal(c2[o][0::2], c2[o][1::2]):
        raise RuntimeError("pairing of two-particle vertices failed")
    ok = q[a] * q[b] < 0
    a, b = a[ok], b[ok]
    pa_ = np.stack([px[a], py[a], pz[a]], axis=1)
    pb_ = np.stack([px[b], py[b], pz[b]], axis=1)
    m_pipi = inv_mass(pa_, pb_, M_PI, M_PI)
    m_ppi = np.minimum(inv_mass(pa_, pb_, M_P, M_PI), inv_mass(pa_, pb_, M_PI, M_P))
    m_ee = inv_mass(pa_, pb_, M_E, M_E)
    lab = np.where(np.abs(m_pipi - M_KS) < 0.015, "K_S",
                   np.where(np.abs(m_ppi - M_L) < 0.008, "Lambda", np.where(m_ee < 0.03, "conversion", "")))
    keep = lab != ""
    cls[a[keep]] = lab[keep]
    cls[b[keep]] = lab[keep]
    return cls


def particle_flags(pt, eta, d0, n_hits, n_pix, vprim) -> dict[str, np.ndarray]:
    """Population flags of plan section 3.4 from the stored (float32) values."""
    pt = np.asarray(pt, np.float32)
    aeta = np.abs(np.asarray(eta, np.float32))
    ad0 = np.abs(np.asarray(d0, np.float32))
    tgt = (pt > 1.0) & (aeta < 4.0) & (n_hits >= 5)
    return {
        "is_target": tgt,
        "is_target_pix3": tgt & (n_pix >= 3),
        "is_prompt_target": tgt & (ad0 < 1.0),
        "is_displaced_target": tgt & (ad0 >= 10.0),
        "is_far_displaced_target": tgt & (ad0 >= 100.0),
        "is_soft_displaced": (n_hits >= 5) & (ad0 >= 10.0) & (pt > 0.5) & (pt <= 1.0),
        "is_hard_scatter": vprim == 1,
    }


def reference_masks(p: pd.DataFrame) -> dict[str, pd.Series]:
    """Particle selections of the published TrackML definitions (plan section 4.2), from a particles table.

    P0 (TrackML score, Amrouche et al. arXiv:1904.06778 p. 10): particles with >= 4 hits.
    P1 (Ju et al., EPJC 81 (2021) 876 p. 5): selected = |eta| < 4 and pT > 0.5 GeV (headline) or > 0.1 GeV ("without
        any fiducial pT cut"); reconstructable = selected and >= 5 hits (technical efficiency denominator).
    P2 (Liu et al., J. Phys. Conf. Ser. 3206 (2026) 012084 p. 5): reconstructable = more than 5 hits and pT >= 1 GeV
        (no eta cut; also with |eta| < 4).
    P3 / P4 and ours: the target definition (is_target)."""
    pt, aeta, nh = p.pt, p.eta.abs(), p.n_hits
    sel05 = (aeta < 4.0) & (pt > 0.5)
    sel01 = (aeta < 4.0) & (pt > 0.1)
    p2 = (nh > 5) & (pt >= 1.0)
    return {"P0_particle": nh >= 4,
            "P1_selected_pt05": sel05, "P1_reconstructable_pt05": sel05 & (nh >= 5),
            "P1_selected_pt01": sel01, "P1_reconstructable_pt01": sel01 & (nh >= 5),
            "P2_reconstructable": p2, "P2_reconstructable_eta4": p2 & (aeta < 4.0),
            "target": p.is_target.astype(bool)}


# ------------------------------------------------------------------------------------------------ one event
def event_tables(raw: dict[str, pd.DataFrame], event: int, det: dict, keep_weight: bool,
                 keep_truth: bool) -> tuple[dict[str, pa.Table] | None, dict]:
    """Tables of one event, or None if the event fails the data-quality rule; and its data-quality record."""
    h, c, p, t = raw["hits"], raw["cells"], raw["particles"], raw["truth"]
    dq = {"event": int(event), "max_outside_mm": float("nan"), "problems": []}

    def fail(msg):
        dq["problems"].append(msg)

    n = len(h)
    if not h.hit_id.is_monotonic_increasing:
        h = h.sort_values("hit_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(h.hit_id.to_numpy(), np.arange(1, n + 1)):
        fail("hit_id is not 1..N")
        return None, dq
    if not t.hit_id.is_monotonic_increasing:
        t = t.sort_values("hit_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(t.hit_id.to_numpy(), np.arange(1, n + 1)):
        fail("truth hit_id is not 1..N")
        return None, dq
    vol, lay, mod = (h[k].to_numpy() for k in ("volume_id", "layer_id", "module_id"))
    if vol.min() < 0 or vol.max() > 255 or lay.min() < 0 or lay.max() > 255 or mod.min() < 0 or mod.max() > 65535:
        fail("volume / layer / module id out of range")
        return None, dq
    x32, y32, z32 = (h[k].to_numpy().astype(np.float32) for k in ("x", "y", "z"))
    x64, y64, z64 = (a.astype(np.float64) for a in (x32, y32, z32))

    # ---- module containment
    out = module_outside_mm(x64, y64, z64, vol, lay, mod, det)
    dq["max_outside_mm"] = float(out.max()) if n else 0.0
    if not np.all(out <= DQ_MAX_OUTSIDE_MM):
        fail(f"{int((out > DQ_MAX_OUTSIDE_MM).sum())} hits more than {DQ_MAX_OUTSIDE_MM:g} mm outside their module")

    # ---- particles and hit-particle consistency
    P = len(p)
    pid_tab = p.particle_id.to_numpy()
    o_p = np.argsort(pid_tab, kind="stable")
    spid = pid_tab[o_p]
    if P and np.any(spid[1:] == spid[:-1]):
        fail("duplicate particle_id in particles.csv")
    if P and spid[0] <= 0:
        fail("particle_id <= 0 in particles.csv")
    tpid = t.particle_id.to_numpy()
    nz = tpid != 0
    pos = np.minimum(np.searchsorted(spid, tpid[nz]), max(P - 1, 0))
    found = (spid[pos] == tpid[nz]) if P else np.zeros(int(nz.sum()), bool)
    if not np.all(found):
        fail(f"{int((~found).sum())} hits of particles missing from particles.csv")
    if dq["problems"]:
        return None, dq
    prow = np.full(n, -1, dtype=np.int64)
    prow[nz] = o_p[pos]
    cnt = np.bincount(prow[nz], minlength=P)
    nhits = p.nhits.to_numpy()
    if not np.array_equal(cnt, nhits):
        fail(f"nhits differs from the truth count for {int((cnt != nhits).sum())} particles")
    q = p.q.to_numpy()
    if not np.all(np.isin(q, (-1, 1))):
        fail("charge not +-1")
    if dq["problems"]:
        return None, dq

    # ---- particle quantities, from the stored float32 values
    pv32 = {k: p[k].to_numpy().astype(np.float32) for k in ("vx", "vy", "vz", "px", "py", "pz")}
    V = {k: a.astype(np.float64) for k, a in pv32.items()}
    pt64 = np.hypot(V["px"], V["py"])
    pt32 = pt64.astype(np.float32)
    eta32 = np.arcsinh(V["pz"] / np.maximum(pt64, 1e-12)).astype(np.float32)
    rv32 = np.hypot(V["vx"], V["vy"]).astype(np.float32)
    d032 = d0_from_vertex(V["vx"], V["vy"], V["px"], V["py"], q).astype(np.float32)
    cat = classify_secondaries(V["vx"], V["vy"], V["vz"], V["px"], V["py"], V["pz"], q, rv32.astype(np.float64))
    vprim = (pid_tab.astype(np.uint64) >> np.uint64(52)).astype(np.int16)

    # ---- order of the hits along each particle: distance of the true point from the production vertex
    tt32 = {k: t[k].to_numpy().astype(np.float32) for k in ("tx", "ty", "tz", "tpx", "tpy", "tpz")}
    idx = np.flatnonzero(nz)
    pr = prow[idx]
    dist = np.sqrt((tt32["tx"][idx].astype(np.float64) - V["vx"][pr]) ** 2
                   + (tt32["ty"][idx].astype(np.float64) - V["vy"][pr]) ** 2
                   + (tt32["tz"][idx].astype(np.float64) - V["vz"][pr]) ** 2)
    o = np.lexsort((idx, dist, pr))
    spr = pr[o]
    head = np.r_[True, spr[1:] != spr[:-1]] if len(o) else np.zeros(0, bool)
    start = np.maximum.accumulate(np.where(head, np.arange(len(o)), 0)) if len(o) else np.zeros(0, np.int64)
    rank_sorted = np.arange(len(o)) - start
    if len(o) and rank_sorted.max() >= NOISE_RANK:
        fail("a particle with more than 254 hits")
        return None, dq
    rank = np.full(n, NOISE_RANK, dtype=np.uint8)
    rank[idx[o]] = rank_sorted
    r32, _ = hit_r_phi(x32, y32)
    r_first = np.full(P, np.nan, dtype=np.float32)
    r_first[spr[head]] = r32[idx[o][head]]
    pix = np.isin(vol, PIXEL_VOLUMES)
    n_pix = np.bincount(prow[nz & pix], minlength=P)

    # ---- cells: count, charge sum, extent along ch0 / ch1
    crow = c.hit_id.to_numpy() - 1
    if len(crow) and (crow.min() < 0 or crow.max() >= n):
        fail("cells of unknown hits")
        return None, dq
    if len(crow) and np.any(crow[1:] < crow[:-1]):
        oc = np.argsort(crow, kind="stable")
        c = c.iloc[oc]
        crow = crow[oc]
    ncells = np.bincount(crow, minlength=n)
    val = c.value.to_numpy().astype(np.float32).astype(np.float64)
    charge = np.bincount(crow, weights=val, minlength=n)
    span0 = np.zeros(n, dtype=np.int64)
    span1 = np.zeros(n, dtype=np.int64)
    if len(crow):
        st = np.flatnonzero(np.r_[True, crow[1:] != crow[:-1]])
        for span, col in ((span0, "ch0"), (span1, "ch1")):
            ch = c[col].to_numpy()
            span[crow[st]] = np.maximum.reduceat(ch, st) - np.minimum.reduceat(ch, st) + 1
    if ncells.max(initial=0) > 65535 or span0.max(initial=0) > 65535 or span1.max(initial=0) > 65535:
        fail("cell count or cluster extent above 65535")
        return None, dq

    weight = (t.weight.to_numpy().astype(np.float32) if keep_weight else np.full(n, np.nan, dtype=np.float32))
    ev32 = np.full(n, event, dtype=np.int32)
    hits = pa.Table.from_arrays([
        pa.array(ev32), pa.array(x32), pa.array(y32), pa.array(z32),
        pa.array(vol.astype(np.uint8)), pa.array(lay.astype(np.uint8)), pa.array(mod.astype(np.uint16)),
        pa.array(tpid.astype(np.int64)), pa.array(rank), pa.array(ncells.astype(np.uint16)),
        pa.array(charge.astype(np.float32)), pa.array(span0.astype(np.uint16)), pa.array(span1.astype(np.uint16)),
        pa.array(weight),
    ], schema=SCHEMAS["hits"])
    n_hits16 = nhits.astype(np.int16)
    flags = particle_flags(pt32, eta32, d032, n_hits16, n_pix, vprim)
    parts = pa.Table.from_arrays([
        pa.array(np.full(P, event, dtype=np.int32)), pa.array(pid_tab.astype(np.int64)),
        *(pa.array(pv32[k]) for k in ("vx", "vy", "vz", "px", "py", "pz")),
        pa.array(q.astype(np.int8)), pa.array(n_hits16), pa.array(pt32), pa.array(eta32), pa.array(rv32),
        pa.array(d032), pa.array(cat.astype(str).tolist(), type=pa.string()), pa.array(vprim),
        pa.array(n_pix.astype(np.int16)), pa.array(r_first),
        *(pa.array(flags[k]) for k in ("is_target", "is_target_pix3", "is_prompt_target", "is_displaced_target",
                                       "is_far_displaced_target", "is_soft_displaced", "is_hard_scatter")),
    ], schema=SCHEMAS["particles"])
    tables = {"hits": hits, "particles": parts}
    if keep_truth:
        tables["truth"] = pa.Table.from_arrays([
            pa.array(np.full(len(idx), event, dtype=np.int32)), pa.array(idx.astype(np.int32)),
            *(pa.array(tt32[k][idx]) for k in ("tx", "ty", "tz", "tpx", "tpy", "tpz")),
        ], schema=SCHEMAS["truth"])
    return tables, dq


# ------------------------------------------------------------------------------------------------ workers
_W: dict = {}


def _worker_init(source: str, det_path: str):
    _W["zf"] = open_archive(source)               # one handle per process (fork must not share file offsets)
    _W["det"] = load_detectors(det_path)
    _W["pid"] = os.getpid()


def _convert_one(args):
    event, names, keep_weight, keep_truth = args
    t0 = time.time()
    raw = read_event_csv(_W["zf"], names)
    tables, dq = event_tables(raw, event, _W["det"], keep_weight, keep_truth)
    dq["seconds"] = round(time.time() - t0, 2)
    return event, tables, dq


# ------------------------------------------------------------------------------------------------ split sets
def load_split(path: str = SPLIT) -> dict:
    with open(path) as fh:
        s = json.load(fh)
    dev = sorted(s["dev"])
    calib_extra = [e for e in sorted(s["train_train1"]) if e not in set(dev)][:N_CALIB_EXTRA]
    return {"path": path, "sha256": sha256_file(path), "dev": set(dev), "calib": set(dev) | set(calib_extra),
            "val": set(s["val"]), "test": set(s["test"]), "train_train1": set(s["train_train1"]),
            "train_all": set(s["train_all"])}


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    hsh = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            hsh.update(b)
    return hsh.hexdigest()


# ------------------------------------------------------------------------------------------------ storage guard
def _xattr(path: str, name: str) -> int:
    return int(os.getxattr(path, name).decode())


def storage_check(projected_bytes: float) -> dict:
    """Plan section 7 stop rule: >= 15 GB of the user quota free after the projected output (the 40 GB limit on
    external/trackml was lifted by the user on 2026-10-08; TRACKML_LIMIT_BYTES None). Raises RuntimeError otherwise."""
    used = _xattr(USER_AREA, "ceph.dir.rbytes")
    quota = _xattr(USER_AREA, "ceph.quota.max_bytes")
    tml = _xattr(EXT, "ceph.dir.rbytes")
    st = {"user_used": used, "user_quota": quota, "trackml_used": tml, "projected": int(projected_bytes)}
    over = TRACKML_LIMIT_BYTES is not None and tml + projected_bytes > TRACKML_LIMIT_BYTES
    if over or quota - used - projected_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"storage stop rule: {st}")
    return st


# ------------------------------------------------------------------------------------------------ conversion
def shard_path(root: str, table: str, archive: str, first: int) -> str:
    return os.path.join(root, table, archive, f"shard_{first:05d}.parquet")


def convert_archive(source: str, archive: str, root: str = DERIVED, events=None, nproc: int = 1,
                    overwrite: bool = False, level: int = 9, det_path: str = DETECTORS,
                    shard_bytes_estimate: float = 2.5e9) -> dict:
    """Convert every complete event of one archive (or `events`), shard by shard (see the module docstring)."""
    t0 = time.time()
    md5_verified = None
    if os.path.exists(source) and os.path.basename(source) in ARCHIVES:
        # a downloaded Zenodo archive is used only after its md5 was checked (slurm/trackml/convert_train_1.sbatch
        # writes <zip>.md5 when md5sum matches the Zenodo listing)
        expect = ARCHIVES[os.path.basename(source)][1]
        if not os.path.exists(source + ".md5"):
            raise RuntimeError(f"{source}: no verified md5 file {source}.md5")
        with open(source + ".md5") as fh:
            md5_verified = fh.read().split()[0]
        if md5_verified != expect:
            raise RuntimeError(f"{source}: md5 {md5_verified} differs from Zenodo's {expect}")
    zf = open_archive(source)
    names = member_map(zf)
    complete = sorted(e for e, k in names.items() if set(k) >= set(KINDS))
    incomplete = sorted(set(names) - set(complete))
    todo = complete if events is None else sorted(int(e) for e in events)
    missing = sorted(set(todo) - set(complete))
    if missing:
        raise ValueError(f"events not complete in {source}: {missing[:10]}")
    split = load_split()
    shards: dict[int, list[int]] = {}
    for e in todo:
        shards.setdefault(e // SHARD * SHARD, []).append(e)
    dq_rows, shard_meta = [], {}
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, initializer=_worker_init, initargs=(source, det_path)) as pool:
        for first, evs in sorted(shards.items()):
            final = {k: shard_path(root, k, archive, first) for k in STORED_TABLES}
            if any(os.path.exists(f) for f in final.values()) and not overwrite:
                raise FileExistsError(f"shard {first} of {archive} exists (use --overwrite to redo it)")
            st = storage_check(shard_bytes_estimate)
            print(f"[convert] shard {first}: {len(evs)} events; storage {st}", flush=True)
            writers: dict[str, pq.ParquetWriter] = {}
            tmp = {k: f + ".tmp" for k, f in final.items()}
            written = {k: [] for k in STORED_TABLES}
            jobs = [(e, names[e], e in split["calib"] or e in split["val"] or e in split["test"], e in split["calib"])
                    for e in evs]
            ts = time.time()
            try:
                for e, tables, dq in pool.imap(_convert_one, jobs, chunksize=1):
                    dq["archive"] = archive
                    dq_rows.append(dq)
                    if tables is None:
                        print(f"[convert] event {e} DROPPED: {dq['problems']}", flush=True)
                        continue
                    for k, tbl in tables.items():
                        if k not in writers:
                            os.makedirs(os.path.dirname(tmp[k]), exist_ok=True)
                            writers[k] = pq.ParquetWriter(tmp[k], SCHEMAS[k], compression="zstd",
                                                          compression_level=level, write_statistics=True,
                                                          **WRITE_OPTS[k])
                        writers[k].write_table(tbl, row_group_size=max(tbl.num_rows, 1))
                        written[k].append(e)
            finally:
                for w in writers.values():
                    w.close()
            meta = {}
            for k, w in writers.items():
                os.replace(tmp[k], final[k])
                meta[k] = {"file": os.path.relpath(final[k], root), "events": written[k],
                           "bytes": os.path.getsize(final[k])}
            shard_meta[first] = meta
            print(f"[convert] shard {first}: {len(written['hits'])} events written in {time.time() - ts:.0f} s; "
                  + ", ".join(f"{k} {m['bytes'] / 1e6:.0f} MB" for k, m in meta.items()), flush=True)
    for first, meta in shard_meta.items():
        for k, m in meta.items():
            m["sha256"] = sha256_file(os.path.join(root, m["file"]))
    dropped = [d for d in dq_rows if d["problems"]]
    res = {
        "archive": archive, "source": source,
        "source_bytes": os.path.getsize(source) if os.path.exists(source) else None,
        "source_md5_zenodo": ARCHIVES.get(f"{archive}.zip", (None, None))[1], "source_md5_verified": md5_verified,
        "converted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
        "code_sha256": sha256_file(os.path.abspath(__file__)), "split": {"path": os.path.relpath(SPLIT, REPO),
                                                                        "sha256": split["sha256"]},
        "detectors_sha256": sha256_file(det_path), "shard_rule": f"first = event // {SHARD} * {SHARD}",
        "zstd_level": level, "nproc": nproc,
        "n_events_in_archive": len(complete), "incomplete_events_in_archive": incomplete,
        "n_events_converted": sum(len(m["hits"]["events"]) for m in shard_meta.values() if "hits" in m),
        "dropped": [{"event": d["event"], "problems": d["problems"]} for d in dropped],
        "weight_events": "dev + calibration + val + test (plan section 6.0)",
        "truth_events": "calibration: dev 1000-1099 + the next 100 train_1 training ids (plan section 6.0)",
        "shards": {str(k): v for k, v in sorted(shard_meta.items())},
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(root, exist_ok=True)
    tag = "" if events is None else "_partial"
    with open(os.path.join(root, f"meta_{archive}{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    pd.DataFrame([{"archive": d["archive"], "event": d["event"], "max_outside_mm": round(d["max_outside_mm"], 4),
                   "seconds": d["seconds"], "status": "dropped" if d["problems"] else "ok",
                   "problems": "; ".join(d["problems"])} for d in dq_rows]).sort_values("event").to_csv(
        os.path.join(root, f"dq_{archive}{tag}.tsv"), sep="\t", index=False)
    return res


# ------------------------------------------------------------------------------------------------ reading
def _metas(root: str) -> dict[str, dict]:
    """meta_<archive>.json of every converted archive (meta_<archive>_partial.json, written by a conversion of
    selected events, only where no full conversion exists)."""
    out, partial = {}, {}
    for f in sorted(os.listdir(root)):
        if f.startswith("meta_") and f.endswith(".json"):
            with open(os.path.join(root, f)) as fh:
                m = json.load(fh)
            (partial if f.endswith("_partial.json") else out)[m["archive"]] = m
    return {**partial, **out}


def event_index(root: str = DERIVED, table: str = "hits") -> dict[int, str]:
    """{event: archive} for every event stored in `table` (from the meta files)."""
    out = {}
    for a, m in _metas(root).items():
        for sh in m["shards"].values():
            if table in sh:
                for e in sh[table]["events"]:
                    out[int(e)] = a
    return out


HIT_DERIVED = {"row": (), "hid": (), "r": ("x", "y"), "phi": ("x", "y"), "geoid": ("volume", "layer", "module"),
               "region": ("x", "y", "z"), "noise": ("pid",)}


def _read_stored(root: str, table: str, events, columns) -> pd.DataFrame:
    idx = event_index(root, table)
    if events is None:
        groups = {}
        for e, a in idx.items():
            groups.setdefault(a, []).append(e)
    else:
        ev = [int(e) for e in np.unique(np.asarray(events, dtype=np.int64))]
        missing = [e for e in ev if e not in idx]
        if missing:
            raise KeyError(f"{table}: events not stored: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        groups = {}
        for e in ev:
            groups.setdefault(idx[e], []).append(e)
    frames = []
    for a, evs in sorted(groups.items()):
        evs = sorted(evs)
        by_shard = {}
        for e in evs:
            by_shard.setdefault(e // SHARD * SHARD, []).append(e)
        for first, se in sorted(by_shard.items()):
            path = shard_path(root, table, a, first)
            flt = None if events is None else [("event", "in", se)]
            frames.append(pd.read_parquet(path, columns=columns, filters=flt))
    if not frames:
        return pd.DataFrame({c: [] for c in (columns or SCHEMAS[table].names)})
    return pd.concat(frames, ignore_index=True)


def read_table(table: str, events=None, columns=None, root: str = DERIVED) -> pd.DataFrame:
    """The TrackML counterpart of hnl.data.convert.read_table: one table (hits, particles, truth or edges) for the
    given TrackML event ids (all stored events if None), with the derived columns of the module docstring.
    Raises KeyError for an event that is not stored (never returns a silently incomplete table)."""
    if table == "edges":
        h = read_table("hits", events, ["event", "row", "pid", "rank"], root)
        return edges_from_hits(h)
    if table == "hits":
        want = list(columns) if columns is not None else SCHEMAS["hits"].names + list(HIT_DERIVED)
        need = {"event"} | {c for c in want if c in SCHEMAS["hits"].names}
        for c in want:
            need |= set(HIT_DERIVED.get(c, ()))
        stored = [c for c in SCHEMAS["hits"].names if c in need]
        df = _read_stored(root, "hits", events, stored)
        ev = df.event.to_numpy()
        if len(ev) and np.any(ev[1:] < ev[:-1]):
            raise RuntimeError("hits not grouped by event in storage order")
        if "row" in want or "hid" in want:
            first = np.r_[True, ev[1:] != ev[:-1]] if len(ev) else np.zeros(0, bool)
            start = np.maximum.accumulate(np.where(first, np.arange(len(ev)), 0)) if len(ev) else np.zeros(0, int)
            row = (np.arange(len(ev)) - start).astype(np.int32)
            df["row"] = row
            df["hid"] = row.astype(np.int64) + 1
        if "r" in want or "phi" in want or "region" in want:
            r, phi = hit_r_phi(df.x.to_numpy(), df.y.to_numpy())
            df["r"], df["phi"] = r, phi
            if "region" in want:
                df["region"] = region_id(r, df.z.to_numpy())
        if "geoid" in want:
            df["geoid"] = geoid(df.volume.to_numpy(), df.layer.to_numpy(), df.module.to_numpy())
        if "noise" in want:
            df["noise"] = df.pid.to_numpy() == 0
        return df[want]
    df = _read_stored(root, table, events, columns)
    return df


def edges_from_hits(h: pd.DataFrame) -> pd.DataFrame:
    """True edges (event, src, dst as rows) from a hits table with event, row, pid, rank: consecutive hits of each
    particle along `rank` (src the earlier hit), noise hits excluded."""
    s = h[h.pid.to_numpy() != 0]
    ev, pid, rank, row = (s[k].to_numpy() for k in ("event", "pid", "rank", "row"))
    o = np.lexsort((rank, pid, ev))
    ev, pid, rank, row = ev[o], pid[o], rank[o], row[o]
    nxt = (ev[1:] == ev[:-1]) & (pid[1:] == pid[:-1])
    return pd.DataFrame({"event": ev[:-1][nxt].astype(np.int32), "src": row[:-1][nxt].astype(np.int32),
                         "dst": row[1:][nxt].astype(np.int32)})


# ------------------------------------------------------------------------------------------------ verification
def verify_archive(archive: str, root: str = DERIVED) -> dict:
    """Structural check of the stored shards against meta_<archive>.json: sha256, schema, one row group per listed
    event in increasing order (row-group event statistics), hits / particles hold the same events, truth only
    calibration events, every non-dropped complete event stored. No physics quantity is read."""
    with open(os.path.join(root, f"meta_{archive}.json")) as fh:
        m = json.load(fh)
    split = load_split()
    dropped = {d["event"] for d in m["dropped"]}
    problems = []
    seen = {k: [] for k in STORED_TABLES}
    for first, sh in m["shards"].items():
        if sh.get("hits", {}).get("events") != sh.get("particles", {}).get("events"):
            problems.append(f"shard {first}: hits and particles hold different events")
        for k, info in sh.items():
            path = os.path.join(root, info["file"])
            if sha256_file(path) != info["sha256"]:
                problems.append(f"{info['file']}: sha256 differs from meta")
            f = pq.ParquetFile(path)
            if not f.schema_arrow.equals(SCHEMAS[k], check_metadata=False):
                problems.append(f"{info['file']}: schema differs")
            ev_col = f.schema_arrow.get_field_index("event")
            evs = []
            for i in range(f.metadata.num_row_groups):
                stt = f.metadata.row_group(i).column(ev_col).statistics
                if stt is None or stt.min != stt.max:
                    problems.append(f"{info['file']}: row group {i} holds several events")
                evs.append(stt.min if stt is not None else None)
            if evs != info["events"]:
                problems.append(f"{info['file']}: row groups {len(evs)} do not match the listed events")
            if any(b <= a for a, b in zip(info["events"], info["events"][1:])):
                problems.append(f"{info['file']}: events not increasing")
            if any(e // SHARD * SHARD != int(first) for e in info["events"]):
                problems.append(f"{info['file']}: event outside the shard range")
            seen[k] += info["events"]
    with zipfile.ZipFile(m["source"]) if os.path.exists(m["source"]) else open_archive(m["source"]) as zf:
        complete = sorted(e for e, k in member_map(zf).items() if set(k) >= set(KINDS))
    expect = [e for e in complete if e not in dropped]
    if sorted(seen["hits"]) != expect or sorted(seen["particles"]) != expect:
        problems.append(f"stored events differ from the complete, non-dropped archive events "
                        f"({len(seen['hits'])} stored, {len(expect)} expected)")
    calib_expect = sorted(e for e in expect if e in split["calib"])
    if sorted(seen["truth"]) != calib_expect:
        problems.append(f"truth table events differ from the calibration events ({len(seen['truth'])} vs "
                        f"{len(calib_expect)})")
    in_split = split["train_train1"] | split["val"] | split["test"]
    if archive == "train_1" and set(expect) != in_split - dropped:
        problems.append("train_1 events differ from the split file's train_1 events")
    return {"archive": archive, "events_stored": len(seen["hits"]), "events_dropped": len(dropped),
            "calibration_events_with_truth": len(seen["truth"]), "problems": problems,
            "bytes": {k: int(sum(sh[k]["bytes"] for sh in m["shards"].values() if k in sh)) for k in STORED_TABLES}}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--source", required=True, help="local zip or Zenodo URL")
    c.add_argument("--archive", required=True, help="train_1 ... train_5")
    c.add_argument("--root", default=DERIVED)
    c.add_argument("--events", default=None, help="comma-separated event ids (default: all complete events)")
    c.add_argument("--nproc", type=int, default=len(os.sched_getaffinity(0)))
    c.add_argument("--level", type=int, default=9, help="zstd level")
    c.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--archive", required=True)
    v.add_argument("--root", default=DERIVED)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        ev = None if a.events is None else [int(x) for x in a.events.split(",")]
        r = convert_archive(a.source, a.archive, a.root, ev, a.nproc, a.overwrite, a.level)
        print(json.dumps({k: r[k] for k in ("archive", "n_events_in_archive", "n_events_converted", "dropped",
                                            "elapsed_s")}, indent=1))
    else:
        r = verify_archive(a.archive, a.root)
        print(json.dumps(r, indent=1))
        if r["problems"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
