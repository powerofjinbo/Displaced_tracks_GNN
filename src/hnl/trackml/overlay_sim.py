"""Simulation of the gun particles of the TrackML pileup overlay and the truth gates F1 (field) and containment of
step H-P (docs/pileup_overlay_plan.md sections 2.1, 4.1, 4.2 and 10).

Simulation. `ActsExampleFatrasGeneric` of an ACTS v15 build reads one particle CSV per event and writes the sim hits
(position, momentum before the surface and its change) and the initial and final particle states. The settings come
from configs/trackml/overlay_sim_v1.json: the TrackML-like RZ field map (`--bf-map-file ... --bf-map-type rz`), pMin
0.1 GeV (decision D6), bremsstrahlung off (D7), hits on sensitive surfaces. v15 Fatras has no decays, so nothing is
set for them. The build directory is an argument, so that the patched build of plan section 3 replaces build-v15
without code changes. The executable runs in the CentOS 7 container with the LCG_97apython3 view
(slurm/acts/build_v15.sbatch); a particle that fails is only logged by v15 (FatrasSimulation.cpp), so the log is parsed.

F1 (section 10). Consecutive sim hits of a particle form a segment. The momentum change along the segment, projected
orthogonal to the mean momentum (which removes the energy loss), is compared with the kick of the loop field
integrated along the chord. The q-signed component in the bending plane is the bending residual. The gate asks for no
particle failures and a mean bending residual within 0.1 MeV in every |z| band, separately for +z and -z. The
selection is that of the earlier probe (field_material/acts_model_test.py, job 6812221): charged non-electrons,
the two hits on different layers, mean-momentum pT > 1 GeV. The loop field is the node table of
field_material/loops_fit.npz with bilinear interpolation, as in that probe. The kick is integrated with 96 midpoint
samples instead of that probe's 6: `reference_closure` scores RK4 tracks through the same field without Fatras, and
6 samples are off by about 0.15 MeV on the mean for long segments in the -z endcaps, 96 by under 0.01 MeV.

B_eff (descriptive). The model-free bending field of pointing segments, B_eff = pT dphi / (0.2998 L_T), median per
z side, |z| band and r band, as in review_physics/beff_sign.py (job 6812558). The same function runs on Fatras
segments and on the TrackML truth segments of the calib training events, so the two can be compared bin by bin.

Containment (section 10). Each sim hit goes from its ACTS module to the TrackML module through the module map, which
is built here from the two detectors.csv files (nearest centre per volume and layer: in 3D in the barrel, in the
transverse plane in the endcaps, as hit_format/analysis.py modmap). The hit is placed in the TrackML module frame. It
must lie on the plane (|w| <= tol) and inside the bounds (|v| <= hv + tol, |u| <= hu(v) + tol). tol is 4 um, the
tolerance of gate G0 for centres, planes and corners.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex

import numpy as np
import pandas as pd

from hnl.geometry.ids import decode_geometry_id
from hnl.trackml.split import REPO

SETTINGS_PATH = os.path.join(REPO, "configs", "trackml", "overlay_sim_v1.json")
K_GEV_PER_T_M = 0.299792458
Z_BANDS = ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3100))
BEFF_Z_BANDS = ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000))
BEFF_R_BANDS = ((0, 200), (200, 600), (600, 1100))
F1_THRESHOLD_MEV = 0.1
F1_MIN_SEGMENTS = 30
# Midpoint samples of the loop field along each segment chord. The earlier probe used 6; a closure without Fatras
# (`reference_closure`: RK4 tracks through the same field, scored by the same code) shows that 6 samples misjudge the
# kick of 800 mm segments near the coil ends by about 0.15 MeV on the band mean (-z, |z| = 2.5-3.1 m), more than the
# threshold, and 96 samples by under 0.01 MeV.
F1_NSUB = 96
WORLD_R_MM, WORLD_ABSZ_MM = 1030.0, 3002.0      # GenericDetector world: r < 1028.5 mm, |z| < 3001 mm
CONTAIN_TOL_MM = 0.004
TRACKER_VOLUMES = (7, 8, 9, 12, 13, 14, 16, 17, 18)
BARREL = (8, 13, 17)
REGION = {7: "pixel endcap", 8: "pixel barrel", 9: "pixel endcap", 12: "short-strip endcap", 13: "short-strip barrel",
          14: "short-strip endcap", 16: "long-strip endcap", 17: "long-strip barrel", 18: "long-strip endcap"}
ROT = ["rot_xu", "rot_xv", "rot_xw", "rot_yu", "rot_yv", "rot_yw", "rot_zu", "rot_zv", "rot_zw"]


# ------------------------------------------------------------------------------------------------ settings and files
def load_settings(path: str = SETTINGS_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rz_map(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lines 'r z Br Bz' (mm, T) of an ACTS v15 RZ text map. v15 fills its grid in file order with global bin
    iz * nR + ir, so r must run fastest; a file in another order raises (job 6812216 showed that v15 reads a z-fastest
    file without an error and scrambles it). Returns r nodes, z nodes, Br (nr, nz), Bz (nr, nz)."""
    a = np.loadtxt(path, comments="#", dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"{path}: expected 4 columns r z Br Bz")
    r, z = np.unique(a[:, 0]), np.unique(a[:, 1])
    if len(r) * len(z) != len(a):
        raise ValueError(f"{path}: {len(a)} lines for a {len(r)} x {len(z)} grid")
    want_r = np.tile(r, len(z))
    want_z = np.repeat(z, len(r))
    if not (np.array_equal(a[:, 0], want_r) and np.array_equal(a[:, 1], want_z)):
        raise ValueError(f"{path}: lines are not in r-fastest order (global bin = iz * nR + ir)")
    br = a[:, 2].reshape(len(z), len(r)).T.copy()
    bz = a[:, 3].reshape(len(z), len(r)).T.copy()
    return r, z, br, bz


# ------------------------------------------------------------------------------------------------ Fatras command
def _flag(v: bool) -> str:
    return "true" if v else "false"


def fatras_args(settings: dict, input_dir: str, output_dir: str, seed: int, threads: int = 1, skip: int = 0,
                events: int | None = None, field_map: str | None = None) -> list[str]:
    """Command line of ActsExampleFatrasGeneric with the settings of overlay_sim_v1.json."""
    f = settings["fatras"]
    fm = settings["field_map"]
    args = [settings["executable"], "--input-dir", input_dir, "--output-dir", output_dir, "--output-csv",
            "--bf-map-file", field_map or fm["path"], "--bf-map-type", fm["type"],
            "--fatras-pmin-gev", repr(float(f["pmin_gev"])),
            "--fatras-em-scattering", _flag(f["em_scattering"]),
            "--fatras-em-ionisation", _flag(f["em_ionisation"]),
            "--fatras-em-radiation", _flag(f["em_radiation"]),
            "--fatras-em-photonconversion", _flag(f["em_photonconversion"]),
            "--fatras-hits", f["hits"],
            "--rnd-seed", str(int(seed)), "-j", str(int(threads)), "--skip", str(int(skip))]
    if events is not None:
        args += ["-n", str(int(events))]
    return args


def container_command(settings: dict, build_dir: str, args: list[str], paths: list[str]) -> list[str]:
    """apptainer call that runs `args` (executable first) from `build_dir`/bin inside the container. Every path the
    run reads or writes must lie under a bound prefix, otherwise the container would not see it."""
    binds = settings["binds"]
    for p in [build_dir, *paths]:
        ap = os.path.abspath(p)
        if not any(ap == b or ap.startswith(b.rstrip("/") + "/") for b in binds):
            raise ValueError(f"{p} is not under a bound prefix {binds}")
    build = os.path.abspath(build_dir)
    exe = os.path.join(build, "bin", args[0])
    inner = (f"export PATH={shlex.quote(settings['shim'])}:$PATH; "
             f"source {shlex.quote(settings['lcg_view'])} > /dev/null 2>&1; "
             f"export LD_LIBRARY_PATH={shlex.quote(os.path.join(build, 'lib64'))}:$LD_LIBRARY_PATH; "
             f"exec {shlex.quote(exe)} " + " ".join(shlex.quote(a) for a in args[1:]))
    cmd = ["apptainer", "exec", "--cleanenv"]
    for b in binds:
        cmd += ["-B", b]
    return cmd + [settings["container"], "bash", "-c", inner]


def event_files(directory: str, stem: str) -> list[int]:
    """Sorted event numbers of the files event<N>-<stem>.csv in a directory."""
    pat = re.compile(r"^event(\d+)-" + re.escape(stem) + r"\.csv$")
    out = []
    for name in os.listdir(directory):
        m = pat.match(name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def event_range(directory: str, first: int | None = None, events: int | None = None,
                stem: str = "particles") -> tuple[int, int, int]:
    """(skip, first, n) for the v15 sequencer, which numbers events from the smallest file in the directory and
    needs them contiguous. Raises for gaps inside the requested range."""
    ev = event_files(directory, stem)
    if not ev:
        raise FileNotFoundError(f"no event*-{stem}.csv in {directory}")
    lo = ev[0] if first is None else first
    n = (ev[-1] + 1 - lo) if events is None else events
    want = set(range(lo, lo + n))
    missing = sorted(want - set(ev))
    if missing:
        raise FileNotFoundError(f"{len(missing)} event files missing in {directory}, first {missing[:5]}")
    gaps = sorted(set(range(ev[0], lo)) - set(ev))
    if gaps:
        raise FileNotFoundError(f"gap below the first requested event in {directory}: {gaps[:5]}")
    return lo - ev[0], lo, n


def event_blocks(directory: str, stem: str = "particles") -> list[tuple[int, int, int]]:
    """(skip, first, n) of every run of consecutive event files in a directory, for one sequencer call each. The v15
    sequencer numbers events from the smallest file (CsvParticleReader::availableEvents), starts at that number +
    --skip and reads event<N>-<stem>.csv for event N, so each run keeps its own event numbers, and with them its ACTS
    streams seed + Cantor(algorithm, N), however the files of a set are spread over jobs (addendum H section 2)."""
    ev = event_files(directory, stem)
    if not ev:
        raise FileNotFoundError(f"no event*-{stem}.csv in {directory}")
    blocks, lo = [], ev[0]
    for a, b in zip(ev, ev[1:] + [None]):
        if b != a + 1:
            blocks.append((lo - ev[0], lo, a + 1 - lo))
            lo = b
    return blocks


_FAILED = re.compile(r"event (\d+) particle (\S+) failed to simulate with error")
_PROCESSED = re.compile(r"Processed (\d+) events")


def parse_fatras_log(text: str) -> dict:
    """Particle failures (logged at ERROR level by v15 FatrasSimulation), other ERROR / FATAL lines and the number of
    processed events."""
    lines = text.splitlines()
    failed = [m.groups() for m in map(_FAILED.search, lines) if m]
    err = [ln for ln in lines if re.search(r"\sERROR\s", ln)]
    fatal = [ln for ln in lines if re.search(r"\sFATAL\s", ln)]
    proc = [int(m.group(1)) for m in map(_PROCESSED.search, lines) if m]
    return {"particle_failures": len(failed), "failed": [{"event": int(e), "particle": p} for e, p in failed[:50]],
            "error_lines": len(err), "fatal_lines": len(fatal), "first_errors": err[:10],
            "events_processed": proc[-1] if proc else None}


# ------------------------------------------------------------------------------------------------ reading sim output
def read_csv_event(directory: str, event: int, stem: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(directory, f"event{event:09d}-{stem}.csv"),
                       dtype={"particle_id": np.uint64, "geometry_id": np.uint64})


def read_sim_events(directory: str, events: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                              pd.DataFrame]:
    """Sim hits (with volume, layer, module decoded from the v15 geometry id), initial and final particles of the
    events in a Fatras output directory, each with an `event` column."""
    if events is None:
        events = event_files(directory, "hits")
    hs, ps, fs = [], [], []
    for e in events:
        h = read_csv_event(directory, e, "hits")
        g = decode_geometry_id(h.geometry_id.to_numpy().view(np.int64))
        h["volume"], h["layer"], h["module"] = g["volume"], g["layer"], g["sensitive"]
        hs.append(h.assign(event=e))
        ps.append(read_csv_event(directory, e, "particles_initial").assign(event=e))
        fs.append(read_csv_event(directory, e, "particles_final").assign(event=e))
    return (pd.concat(hs, ignore_index=True), pd.concat(ps, ignore_index=True), pd.concat(fs, ignore_index=True))


def segments(hits: pd.DataFrame, particles: pd.DataFrame) -> pd.DataFrame:
    """Consecutive hits of a particle (sim-hit index order): start/end position and momentum before the surface,
    with the momentum after the start surface (tp + deltap) as p0a, and the particle's charge, PDG and vertex."""
    h = hits.sort_values(["event", "particle_id", "index"], kind="stable")
    ev = h.event.to_numpy()
    pid = h.particle_id.to_numpy()
    i0 = np.flatnonzero((pid[1:] == pid[:-1]) & (ev[1:] == ev[:-1]))
    i1 = i0 + 1
    cols = {}
    for c in ("tx", "ty", "tz", "tpx", "tpy", "tpz", "volume", "layer", "module"):
        v = h[c].to_numpy()
        cols[c + "0"], cols[c + "1"] = v[i0], v[i1]
    for c in ("x", "y", "z"):
        cols[f"tp{c}0a"] = h[f"tp{c}"].to_numpy()[i0] + h[f"deltap{c}"].to_numpy()[i0]
    seg = pd.DataFrame(cols)
    seg["event"], seg["particle_id"] = ev[i0], pid[i0]
    p = particles[["event", "particle_id", "particle_type", "q", "vx", "vy", "vz"]].rename(
        columns={"particle_type": "pdg"})
    seg = seg.merge(p, on=["event", "particle_id"], how="left", validate="many_to_one")
    seg["rv"] = np.hypot(seg.vx, seg.vy)
    return seg


# ------------------------------------------------------------------------------------------------ fields
class NodeField:
    """(Br, Bz)(r, z) [T] bilinear on a regular node grid (r nodes x z nodes); outside, the nearest edge value."""

    def __init__(self, r_nodes, z_nodes, br, bz):
        self.r = np.asarray(r_nodes, np.float64)
        self.z = np.asarray(z_nodes, np.float64)
        self.br = np.asarray(br, np.float64)
        self.bz = np.asarray(bz, np.float64)
        if self.br.shape != (len(self.r), len(self.z)) or self.bz.shape != self.br.shape:
            raise ValueError("field arrays must be (n_r, n_z)")

    @classmethod
    def from_loop_fit(cls, path: str) -> "NodeField":
        d = np.load(path)
        return cls(d["r_nodes"], d["z_nodes"], d["Br"], d["Bz"])

    @classmethod
    def from_rz_map(cls, path: str) -> "NodeField":
        r, z, br, bz = read_rz_map(path)
        return cls(r, z, br, bz)

    def __call__(self, r, z):
        r = np.clip(np.asarray(r, np.float64), self.r[0], self.r[-1])
        z = np.clip(np.asarray(z, np.float64), self.z[0], self.z[-1])
        ir = np.clip(np.searchsorted(self.r, r, side="right") - 1, 0, len(self.r) - 2)
        iz = np.clip(np.searchsorted(self.z, z, side="right") - 1, 0, len(self.z) - 2)
        fr = (r - self.r[ir]) / (self.r[ir + 1] - self.r[ir])
        fz = (z - self.z[iz]) / (self.z[iz + 1] - self.z[iz])

        def interp(a):
            return ((1 - fr) * (1 - fz) * a[ir, iz] + fr * (1 - fz) * a[ir + 1, iz]
                    + (1 - fr) * fz * a[ir, iz + 1] + fr * fz * a[ir + 1, iz + 1])

        return interp(self.br), interp(self.bz)


def loop_field(r, z, radius):
    """(Br, Bz) of one circular current loop of the given radius at z = 0, per unit mu0 I (the physical formula with
    the elliptic parameter m = k^2; field_material/fieldlib.loop_field with acts_bug=False)."""
    from scipy.special import ellipe, ellipk
    r = np.abs(np.asarray(r, np.float64))
    z = np.asarray(z, np.float64)
    R = float(radius)
    k2 = 4 * R * r / ((R + r) ** 2 + z * z)
    e1, e2 = ellipk(k2), ellipe(k2)
    k = np.sqrt(k2)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = k * z / (4 * np.pi * np.sqrt(R * r ** 3)) * ((2 - k2) / (2 - 2 * k2) * e2 - e1)
        bz = k / (4 * np.pi * np.sqrt(R * r)) * (((R + r) * k2 - 2 * r) / (2 * r * (1 - k2)) * e2 + e1)
    axis = r == 0
    br = np.where(axis, 0.0, br)
    bz = np.where(axis, 0.5 * R * R / (R * R + z * z) ** 1.5, bz)
    return br, bz


def loop_model(r, z, currents, loops):
    """Sum of the fitted current loops: loops (n, 2) = (radius, z position) in mm, currents (n,)."""
    br = np.zeros(np.broadcast(np.asarray(r), np.asarray(z)).shape)
    bz = np.zeros_like(br)
    for current, (radius, z0) in zip(currents, loops):
        a, b = loop_field(r, np.asarray(z, np.float64) - z0, radius)
        br += current * a
        bz += current * b
    return br, bz


def bending_residual(x0, x1, p0, p1, q, field, nsub: int = 6):
    """Residual of the momentum change between two points against the kick of `field` integrated along the chord
    (nsub midpoint steps), projected orthogonal to the mean momentum; returns the residual vectors (n, 3) [GeV] and
    the q-signed component in the bending plane [GeV] (field_material/fit_loops.model_residuals)."""
    x0, x1, p0, p1 = (np.asarray(a, np.float64) for a in (x0, x1, p0, p1))
    q = np.asarray(q, np.float64)
    dx = (x1 - x0) / 1000.0
    pm = 0.5 * (p0 + p1)
    ph = pm / np.linalg.norm(pm, axis=1, keepdims=True)
    pred = np.zeros_like(p0)
    for j in range(nsub):
        xj = x0 + (j + 0.5) / nsub * (x1 - x0)
        r = np.hypot(xj[:, 0], xj[:, 1])
        br, bz = field(r, xj[:, 2])
        rs = np.maximum(r, 1e-9)
        b = np.stack([br * xj[:, 0] / rs, br * xj[:, 1] / rs, bz], 1)
        pred += K_GEV_PER_T_M * q[:, None] * np.cross(dx / nsub, b)
    res = (p1 - p0) - pred
    res -= np.sum(res * ph, 1, keepdims=True) * ph
    u1 = np.cross(ph, np.array([0.0, 0.0, 1.0]))
    u1 /= np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), 1e-12)
    return res, np.sum(res * u1, 1) * q


def f1_select(seg: pd.DataFrame) -> pd.DataFrame:
    """Segments of the F1 test: charged non-electrons, hits on different layers, mean-momentum pT > 1 GeV."""
    s = seg[(seg.q != 0) & (seg.pdg.abs() != 11)]
    s = s[~((s.volume0 == s.volume1) & (s.layer0 == s.layer1))]
    return s[np.hypot(0.5 * (s.tpx0 + s.tpx1), 0.5 * (s.tpy0 + s.tpy1)) > 1.0]


def segment_arrays(s: pd.DataFrame, after: bool = False):
    x0 = s[["tx0", "ty0", "tz0"]].to_numpy(np.float64)
    x1 = s[["tx1", "ty1", "tz1"]].to_numpy(np.float64)
    p0 = s[["tpx0a", "tpy0a", "tpz0a"] if after else ["tpx0", "tpy0", "tpz0"]].to_numpy(np.float64)
    p1 = s[["tpx1", "tpy1", "tpz1"]].to_numpy(np.float64)
    return x0, x1, p0, p1, s.q.to_numpy(np.float64)


def _stats(v: np.ndarray, threshold: float | None) -> dict:
    n = len(v)
    if n < F1_MIN_SEGMENTS:
        return {"n": int(n), "mean_MeV": None, "sem_MeV": None, "rms_MeV": None, "pass": None}
    mean = 1000 * float(np.mean(v))
    row = {"n": int(n), "mean_MeV": round(mean, 4), "sem_MeV": round(1000 * float(np.std(v) / np.sqrt(n)), 4),
           "rms_MeV": round(1000 * float(np.sqrt(np.mean(v * v))), 3)}
    if threshold is not None:
        row["pass"] = bool(abs(mean) <= threshold)
    return row


def _band_rows(bend: np.ndarray, zm: np.ndarray, threshold: float | None) -> list[dict]:
    side = np.where(zm >= 0, "+z", "-z")
    az = np.abs(zm)
    return [{"side": sd, "absz_mm": f"{lo}-{hi}", **_stats(bend[(side == sd) & (az >= lo) & (az < hi)], threshold)}
            for sd in ("+z", "-z") for lo, hi in Z_BANDS]


def f1_tables(seg: pd.DataFrame, field, threshold: float = F1_THRESHOLD_MEV, nsub: int = F1_NSUB) -> dict:
    """F1 rows per z side and |z| band (the gate), per side and detector region (descriptive, the same rule shown),
    the |z| bands with the momentum after the start surface (multiple scattering at the start hit removed;
    diagnostic) and the |z| bands with the 6-sample integration of the earlier probe (diagnostic)."""
    s = f1_select(seg)
    x0, x1, p0, p1, q = segment_arrays(s)
    _, bend = bending_residual(x0, x1, p0, p1, q, field, nsub)
    _, bend_a = bending_residual(*segment_arrays(s, after=True), field, nsub)
    _, bend_6 = bending_residual(x0, x1, p0, p1, q, field, 6)
    zm = 0.5 * (x0[:, 2] + x1[:, 2])
    side = np.where(zm >= 0, "+z", "-z")
    region = np.array([REGION.get(int(v), "other") for v in s.volume0.to_numpy()])
    bands = _band_rows(bend, zm, threshold)
    regions = [{"side": sd, "region": reg, **_stats(bend[(side == sd) & (region == reg)], threshold)}
               for sd in ("+z", "-z") for reg in sorted(set(REGION.values())) if np.any((side == sd) & (region == reg))]
    populated = all(b["pass"] is not None for b in bands)
    means = [abs(b["mean_MeV"]) for b in bands if b["mean_MeV"] is not None]
    return {"segments": int(len(s)), "segments_beyond_last_band": int(np.sum(np.abs(zm) >= Z_BANDS[-1][1])),
            "nsub": nsub, "bands": bands, "regions": regions,
            "bands_after_start_surface": _band_rows(bend_a, zm, None), "bands_nsub6": _band_rows(bend_6, zm, None),
            "all_bands_populated": populated, "max_abs_mean_MeV": max(means) if means else None,
            "pass": populated and all(b["pass"] for b in bands)}


def rk4_tracks(x0: np.ndarray, p0: np.ndarray, q: np.ndarray, field, step_mm: float = 1.0, nsteps: int = 3600,
               every: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Material-free RK4 of many tracks through field(r, z) -> (Br, Bz) [T]: positions and momenta every `every`
    steps (arrays (n_tracks, n_samples, 3)) and a mask of the samples inside the GenericDetector world."""
    x = np.array(x0, np.float64)
    pm = np.linalg.norm(p0, axis=1)
    t = np.asarray(p0, np.float64) / pm[:, None]
    k = (K_GEV_PER_T_M * np.asarray(q, np.float64) / pm / 1000.0)[:, None]

    def b(xx):
        r = np.hypot(xx[:, 0], xx[:, 1])
        br, bz = field(r, xx[:, 2])
        rs = np.maximum(r, 1e-9)
        return np.stack([br * xx[:, 0] / rs, br * xx[:, 1] / rs, bz], 1)

    def f(xx, tt):
        return tt, k * np.cross(tt, b(xx))

    xs, ts = [x.copy()], [t.copy()]
    h = step_mm
    for i in range(1, nsteps + 1):
        k1x, k1t = f(x, t)
        k2x, k2t = f(x + 0.5 * h * k1x, t + 0.5 * h * k1t)
        k3x, k3t = f(x + 0.5 * h * k2x, t + 0.5 * h * k2t)
        k4x, k4t = f(x + h * k3x, t + h * k3t)
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        t = t + h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        if i % every == 0:
            xs.append(x.copy())
            ts.append(t.copy())
    X, T = np.stack(xs, 1), np.stack(ts, 1)
    inside = (np.hypot(X[..., 0], X[..., 1]) < WORLD_R_MM) & (np.abs(X[..., 2]) < WORLD_ABSZ_MM)
    inside = np.cumprod(inside, axis=1).astype(bool)          # nothing after the first exit
    return X, T * pm[:, None, None], inside


def reference_closure(field, n_tracks: int = 2000, seed: int = 20261014, nsubs=(6, F1_NSUB),
                      spacings=(1, 4, 8, 16)) -> dict:
    """The F1 scoring without Fatras: pointing tracks (pT 1-6 GeV, |eta| < 3, vertex z ~ N(0, 55 mm)) through `field`
    by RK4 without material, sampled every 50 mm of path; segments between samples 1, 4, 8 and 16 apart (50-800 mm
    of path), scored against the same field. Any mean residual is an error of the scoring itself. Per nsub and
    segment length: the band rows; and the largest |mean| over lengths and bands (the long segments dominate)."""
    rng = np.random.default_rng(seed)
    pt = rng.uniform(1.0, 6.0, n_tracks)
    eta = rng.uniform(-3.0, 3.0, n_tracks)
    phi = rng.uniform(-np.pi, np.pi, n_tracks)
    q = rng.choice([-1.0, 1.0], n_tracks)
    p0 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], 1)
    x0 = np.stack([np.zeros(n_tracks), np.zeros(n_tracks), rng.normal(0, 55.0, n_tracks)], 1)
    X, P, inside = rk4_tracks(x0, p0, q, field)
    out = {"tracks": n_tracks, "spacings_mm": [50 * s for s in spacings]}
    for ns in nsubs:
        block, worst = {}, 0.0
        for sp in spacings:
            ii, jj = np.nonzero(inside[:, :-sp] & inside[:, sp:])
            _, bend = bending_residual(X[ii, jj], X[ii, jj + sp], P[ii, jj], P[ii, jj + sp], q[ii], field, ns)
            zm = 0.5 * (X[ii, jj, 2] + X[ii, jj + sp, 2])
            rows = _band_rows(bend, zm, None)
            block[f"path_{50 * sp}mm"] = rows
            worst = max([worst] + [abs(r["mean_MeV"]) for r in rows if r["mean_MeV"] is not None])
        out[f"nsub{ns}"] = {"by_segment_length": block, "max_abs_mean_MeV": round(worst, 4)}
    return out


# ------------------------------------------------------------------------------------------------ model-free B_eff
def beff(x0, x1, p0, p1, q) -> np.ndarray:
    """Transverse bending field B_eff = pT dphi_p / (K L_T) [T], positive for Bz > 0 (review_physics/beff_sign.py)."""
    pt0 = np.hypot(p0[:, 0], p0[:, 1])
    pt1 = np.hypot(p1[:, 0], p1[:, 1])
    ptm = 0.5 * (pt0 + pt1)
    lt = np.hypot(x1[:, 0] - x0[:, 0], x1[:, 1] - x0[:, 1]) / 1000.0
    dphi = np.angle(np.exp(1j * (np.arctan2(p1[:, 1], p1[:, 0]) - np.arctan2(p0[:, 1], p0[:, 0]))))
    return -np.sign(q) * ptm * dphi / (K_GEV_PER_T_M * np.maximum(lt, 1e-6))


def beff_select(x0, x1, p0, p1, q, pdg, rv) -> np.ndarray:
    """Pointing segments of beff_sign.py: pT > 2 GeV, non-electron, prompt (r_v < 1 mm), L_T > 20 mm, charged."""
    ptm = 0.5 * (np.hypot(p0[:, 0], p0[:, 1]) + np.hypot(p1[:, 0], p1[:, 1]))
    lt = np.hypot(x1[:, 0] - x0[:, 0], x1[:, 1] - x0[:, 1]) / 1000.0
    return (ptm > 2) & (np.abs(pdg) != 11) & (rv < 1) & (lt > 0.02) & (q != 0)


def beff_table(x0, x1, p0, p1, q, min_n: int = 200) -> dict:
    """Median B_eff per (z side, |z| band, r band); key 'side|absz|r'."""
    b = beff(x0, x1, p0, p1, q)
    zm = 0.5 * (x0[:, 2] + x1[:, 2])
    rm = 0.5 * (np.hypot(x0[:, 0], x0[:, 1]) + np.hypot(x1[:, 0], x1[:, 1]))
    out = {}
    for sg, name in ((1, "+z"), (-1, "-z")):
        for zl, zh in BEFF_Z_BANDS:
            for rl, rh in BEFF_R_BANDS:
                m = (sg * zm >= zl) & (sg * zm < zh) & (rm >= rl) & (rm < rh)
                n = int(m.sum())
                out[f"{name}|{zl}-{zh}|{rl}-{rh}"] = {"n": n, "median_T": round(float(np.median(b[m])), 4)
                                                     if n >= min_n else None}
    return out


def beff_compare(sim: dict, ref: dict) -> list[dict]:
    """Rows of B_eff (sim, TrackML truth) per bin with both sides and the (+z) - (-z) asymmetry of each."""
    rows = []
    for zl, zh in BEFF_Z_BANDS:
        for rl, rh in BEFF_R_BANDS:
            row = {"absz_mm": f"{zl}-{zh}", "r_mm": f"{rl}-{rh}"}
            for name in ("+z", "-z"):
                k = f"{name}|{zl}-{zh}|{rl}-{rh}"
                a, b = sim[k]["median_T"], ref[k]["median_T"]
                row[f"sim{name}"], row[f"trackml{name}"] = a, b
                row[f"n_sim{name}"], row[f"n_trackml{name}"] = sim[k]["n"], ref[k]["n"]
                row[f"diff{name}"] = round(a - b, 4) if a is not None and b is not None else None
            for src in ("sim", "trackml"):
                a, b = row[f"{src}+z"], row[f"{src}-z"]
                row[f"{src}_asym"] = round(a - b, 4) if a is not None and b is not None else None
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ containment
def module_frames(det: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Centres (n, 3) and rotations (n, 3, 3) of a detectors.csv table: columns of R are u, v, w (global = R l + c)."""
    return det[["cx", "cy", "cz"]].to_numpy(np.float64), det[ROT].to_numpy(np.float64).reshape(-1, 3, 3)


def to_local(xyz: np.ndarray, c: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Local (u, v, w) = R^T (x - c) for points and per-point frames."""
    return np.einsum("nij,ni->nj", rot, np.asarray(xyz, np.float64) - c)


def build_module_map(acts_det: pd.DataFrame, tml_det: pd.DataFrame) -> pd.DataFrame:
    """ACTS sensitive module -> TrackML module of the same volume and layer with the nearest centre (3D in the
    barrel, transverse in the endcaps). Raises unless the map is a bijection of every layer's modules."""
    from scipy.spatial import cKDTree
    a = acts_det[(acts_det.module_id > 0) & acts_det.volume_id.isin(TRACKER_VOLUMES)]
    rows = []
    for (v, lay), g in a.groupby(["volume_id", "layer_id"]):
        t = tml_det[(tml_det.volume_id == v) & (tml_det.layer_id == lay)].reset_index(drop=True)
        if len(t) != len(g):
            raise ValueError(f"volume {v} layer {lay}: {len(g)} ACTS modules, {len(t)} TrackML modules")
        cols = ["cx", "cy", "cz"] if v in BARREL else ["cx", "cy"]
        d, i = cKDTree(t[cols].to_numpy()).query(g[cols].to_numpy())
        if len(np.unique(i)) != len(g):
            raise ValueError(f"volume {v} layer {lay}: nearest-centre map is not bijective")
        d3 = np.linalg.norm(g[["cx", "cy", "cz"]].to_numpy() - t[["cx", "cy", "cz"]].to_numpy()[i], axis=1)
        rows.append(pd.DataFrame({"volume": int(v), "layer": int(lay), "acts_module": g.module_id.to_numpy(),
                                  "trackml_module": t.module_id.to_numpy()[i], "d_centre_mm": d3}))
    m = pd.concat(rows, ignore_index=True)
    missing = set(map(tuple, tml_det[tml_det.volume_id.isin(TRACKER_VOLUMES)][["volume_id", "layer_id",
                                                                                 "module_id"]].to_numpy()))
    missing -= set(map(tuple, m[["volume", "layer", "trackml_module"]].to_numpy()))
    if missing:
        raise ValueError(f"{len(missing)} TrackML modules have no ACTS module")
    return m


def read_module_map(path: str) -> pd.DataFrame:
    """A stored module-id table with columns volume_id, layer_id, acts_module_id, trackml_module_id (as
    configs/trackml/module_map_acts_zmirror_v1.csv) in the layout of `build_module_map` (without centre distances)."""
    t = pd.read_csv(path)
    return pd.DataFrame({"volume": t.volume_id.to_numpy(), "layer": t.layer_id.to_numpy(),
                         "acts_module": t.acts_module_id.to_numpy(), "trackml_module": t.trackml_module_id.to_numpy()})


def module_key(v, lay, mod) -> np.ndarray:
    return (np.asarray(v, np.int64) << 40) | (np.asarray(lay, np.int64) << 24) | np.asarray(mod, np.int64)


def containment(hits: pd.DataFrame, mmap: pd.DataFrame, tml_det: pd.DataFrame,
                tol: float = CONTAIN_TOL_MM) -> pd.DataFrame:
    """Per hit: the TrackML module through the map, local (u, v, w) in its frame, and the flags mapped, present
    (the mapped id is in TrackML's detectors.csv), on_plane (|w| <= tol), inside (bounds + tol) and ok."""
    hk = module_key(hits.volume, hits.layer, hits.module)
    mk = pd.Index(module_key(mmap.volume, mmap.layer, mmap.acts_module))
    if not mk.is_unique:
        raise ValueError("module map has duplicate ACTS ids")
    im = mk.get_indexer(hk)
    mapped = im >= 0
    tml_mod = np.where(mapped, mmap.trackml_module.to_numpy()[np.maximum(im, 0)], -1)
    tk = pd.Index(module_key(tml_det.volume_id, tml_det.layer_id, tml_det.module_id))
    it = tk.get_indexer(module_key(hits.volume, hits.layer, tml_mod))
    present = mapped & (it >= 0)
    c, rot = module_frames(tml_det)
    j = np.maximum(it, 0)
    loc = to_local(hits[["tx", "ty", "tz"]].to_numpy(np.float64), c[j], rot[j])
    hv = tml_det.module_hv.to_numpy(np.float64)[j]
    mn, mx = tml_det.module_minhu.to_numpy(np.float64)[j], tml_det.module_maxhu.to_numpy(np.float64)[j]
    hu = mn + (mx - mn) * (np.clip(loc[:, 1], -hv, hv) + hv) / (2 * hv)
    du = np.abs(loc[:, 0]) - hu                 # > 0 outside in u
    dv = np.abs(loc[:, 1]) - hv
    on_plane = present & (np.abs(loc[:, 2]) <= tol)
    inside = present & (du <= tol) & (dv <= tol)
    return pd.DataFrame({"volume": hits.volume.to_numpy(), "trackml_module": tml_mod, "mapped": mapped,
                         "present": present, "u": loc[:, 0], "v": loc[:, 1], "w": loc[:, 2],
                         "out_u": np.where(present, du, np.nan), "out_v": np.where(present, dv, np.nan),
                         "on_plane": on_plane, "inside": inside, "ok": on_plane & inside})


def containment_summary(flags: pd.DataFrame, particle_keys: list[np.ndarray] | None = None) -> dict:
    """Counts overall and per volume, and per particle when `particle_keys` (arrays that together identify the
    particle of each hit, e.g. [event, particle_id]) are given; the gate passes when every hit is ok."""
    def block(f):
        p = f[f.present]
        return {"hits": int(len(f)), "unmapped": int((~f.mapped).sum()),
                "not_in_trackml": int((f.mapped & ~f.present).sum()),
                "off_plane": int((f.present & ~f.on_plane).sum()), "outside_bounds": int((f.present & ~f.inside).sum()),
                "failed": int((~f.ok).sum()), "frac_failed": round(float((~f.ok).mean()), 6) if len(f) else None,
                "max_abs_w_mm": round(float(np.abs(p.w).max()), 6) if len(p) else None,
                "max_out_of_bounds_mm": round(float(np.nanmax(np.maximum(p.out_u, p.out_v))), 6) if len(p) else None}
    out = {"tol_mm": CONTAIN_TOL_MM, **block(flags), "per_volume": {}}
    for v, f in flags.groupby("volume"):
        out["per_volume"][str(int(v))] = block(f)
    if particle_keys is not None:
        bad = pd.Series(~flags.ok.to_numpy()).groupby([np.asarray(k) for k in particle_keys]).any()
        out["particles"] = int(len(bad))
        out["particles_with_failed_hit"] = int(bad.sum())
    out["pass"] = bool(len(flags) > 0 and out["failed"] == 0)
    return out


def map_agreement(mmap: pd.DataFrame, path: str, acts_col: str, tml_col: str) -> dict:
    """Agreement of the built map with a stored one (columns volume, layer and the two module columns)."""
    o = pd.read_csv(path)
    k = module_key(o.volume, o.layer, o[acts_col])
    ref = pd.Series(o[tml_col].to_numpy(), index=k)
    mine = pd.Series(mmap.trackml_module.to_numpy(), index=module_key(mmap.volume, mmap.layer, mmap.acts_module))
    common = mine.index.intersection(ref.index)
    return {"file": path, "modules_compared": int(len(common)), "modules_in_file": int(len(ref)),
            "agree": int((mine[common].to_numpy() == ref[common].to_numpy()).sum())}
