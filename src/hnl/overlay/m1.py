"""Truth check M1 of the pileup overlay (docs/pileup_overlay_plan.md sections 4.3 and 10; descriptive, no pass rule):
multiple scattering and energy loss per crossing in the patched ACTS v15 GenericDetector (Fatras with the TrackML-like
field map) against TrackML truth, per detector region and |eta| bin, and the last-hit position closure. Truth only;
no model is run and no model output is read. The code is that of the field and material study
(pileup_design/field_material/material.py and closure_material.py, jobs 6812190 and 6812228), here in the repository
and run with one field and one integration on both samples.

Samples.
- TrackML: the truth segments of the calibration training events (field_material/segments_calib200.parquet, made by
  field_material/segments.py from derived/tp_v1; consecutive true hits of a particle in rank order).
- ACTS: the Fatras output of a probe of prompt particles (scripts/acts/overlay_sim.py field-probe: mu, pi and e,
  p ~ logU(0.5, 20) GeV, eta ~ U(-3, 3)), simulated in the build under test (sim hits in index order; the momentum of
  a hit is the one before its surface).
- Field: the loop-model fit to TrackML truth (field_material/loops_fit.npz). The RZ map that Fatras reads tabulates it
  (within 5e-7 T at the nodes inside the tracker), so the same field is removed from both samples.

Per segment (two consecutive hits on different layers of a prompt mu or pi with p = 0.7-5 GeV) the field kick
K q INT(dx x B) along the chord (`NSUB` midpoint samples) is removed from the momentum change, the rest is projected
orthogonal to the mean momentum and split into the bending-plane direction u1 = p x e_z / |.| and the polar-plane
direction u2 = p x u1. W = 1.4826 MAD of beta * 1000 (dp_res . u2) [MeV] is the scattering width per crossing (the
field acts along u2 only through Br, which the subtraction removes); dE = median of |p1| - |p0| [MeV] is the energy
loss per crossing. Region = volume of the first hit (pixel, short-strip and long-strip barrel and endcaps); |eta| of
the particle's initial momentum in bands of 0.5. Bins with fewer than MIN_SEGMENTS segments are not reported.

Last-hit closure: prompt pions and muons with pT 1-2 or 2-5 GeV and at least 5 hits, at most CLOSURE_PER_BIN per
|eta| band (fixed seed), are propagated without material from their true vertex and momentum through the same field
(RK4, 5 mm steps); for each hit in order the trajectory's crossing of the hit's layer surface (r in the barrel
volumes, z in the endcaps) is found, and the |r dphi| distance at the last hit is the closure residual. With the field
right in both samples, what remains is scattering and energy loss, i.e. the material.

    python -m hnl.overlay.m1 run --sim DIR|TAR --out-json J [--segments PQ] [--loop-fit NPZ] [--label NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd

from hnl.trackml import overlay_sim as OS

STUDY = "/work/submit/jinboz1/hnl/pileup_design/field_material"
SEGMENTS = os.path.join(STUDY, "segments_calib200.parquet")
LOOP_FIT = os.path.join(STUDY, "loops_fit.npz")
STUDY_MATERIAL = os.path.join(STUDY, "material.json")
STUDY_CLOSURE = os.path.join(STUDY, "closure_material.json")
K = OS.K_GEV_PER_T_M
MASS = {11: 0.000511, 13: 0.10566, 211: 0.13957, 321: 0.49368, 2212: 0.93827}
REGIONS = ("pix_b", "pix_ec", "ss_b", "ss_ec", "ls_b", "ls_ec")
REGION_NAMES = {"pix_b": "pixel barrel", "pix_ec": "pixel endcap", "ss_b": "short-strip barrel",
                "ss_ec": "short-strip endcap", "ls_b": "long-strip barrel", "ls_ec": "long-strip endcap"}
ETA_BANDS = ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0))    # labels as in the study
CLOSURE_ETA = ((0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0))
CLOSURE_PT = ((1.0, 2.0), (2.0, 5.0))
P_RANGE = (0.7, 5.0)
MIN_SEGMENTS = 200
CLOSURE_PER_BIN = 400
CLOSURE_MIN_TRACKS = 20
NSUB = 32
BARREL_VOLUMES = (8, 13, 17)


# ------------------------------------------------------------------------------------------------ segments
def region_of(volume) -> np.ndarray:
    v = np.asarray(volume)
    return np.select([np.isin(v, (7, 9)), v == 8, np.isin(v, (12, 14)), v == 13, np.isin(v, (16, 18)), v == 17],
                     list(("pix_ec", "pix_b", "ss_ec", "ss_b", "ls_ec", "ls_b")), "other")


def species_of(pdg) -> np.ndarray:
    a = np.abs(np.asarray(pdg))
    return np.select([a == 13, a == 211, a == 11], ["mu", "pi", "e"], "x")


def trackml_segments(path: str = SEGMENTS) -> pd.DataFrame:
    """TrackML truth segments in the common layout (module docstring)."""
    cols = ["event", "pid", "tx0", "ty0", "tz0", "tx1", "ty1", "tz1", "tpx0", "tpy0", "tpz0", "tpx1", "tpy1", "tpz1",
            "volume0", "layer0", "volume1", "layer1", "rank0", "q", "pdg", "eta", "rv", "vx", "vy", "vz", "px", "py",
            "pz", "pt", "n_hits"]
    s = pd.read_parquet(path, columns=cols)
    return s[s.q != 0].reset_index(drop=True)


def acts_segments(directory: str) -> pd.DataFrame:
    """Segments of a Fatras output directory in the layout of `trackml_segments` (momentum before each surface)."""
    hits, init, _ = OS.read_sim_events(directory)
    seg = OS.segments(hits, init)
    p = init[["event", "particle_id", "px", "py", "pz"]]
    seg = seg.merge(p, on=["event", "particle_id"], how="left", validate="many_to_one")
    n = hits.groupby(["event", "particle_id"]).size().rename("n_hits").reset_index()
    seg = seg.merge(n, on=["event", "particle_id"], how="left", validate="many_to_one")
    seg["pt"] = np.hypot(seg.px, seg.py)
    seg["eta"] = np.arcsinh(seg.pz / seg.pt)
    seg = seg.rename(columns={"particle_id": "pid"})
    seg["rank0"] = seg.groupby(["event", "pid"]).cumcount()
    return seg


# ------------------------------------------------------------------------------------------------ material
def field_kick(x0, x1, q, field, nsub: int = NSUB) -> np.ndarray:
    """K q INT(dx x B) along the chord from x0 to x1 [mm] with nsub midpoint samples (GeV)."""
    x0, x1 = np.asarray(x0, np.float64), np.asarray(x1, np.float64)
    dx = (x1 - x0) / 1000.0
    out = np.zeros_like(dx)
    for j in range(nsub):
        xj = x0 + (j + 0.5) / nsub * (x1 - x0)
        r = np.hypot(xj[:, 0], xj[:, 1])
        br, bz = field(r, xj[:, 2])
        rs = np.maximum(r, 1e-9)
        b = np.stack([br * xj[:, 0] / rs, br * xj[:, 1] / rs, bz], 1)
        out += K * np.asarray(q, np.float64)[:, None] * np.cross(dx / nsub, b)
    return out


def decompose(x0, x1, p0, p1, q, pdg, field, nsub: int = NSUB) -> dict[str, np.ndarray]:
    """Bending-plane and polar-plane kicks times beta [MeV] after removing the field, mean momentum [GeV], and the
    momentum change |p1| - |p0| [GeV] of each segment."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    pm = 0.5 * (p0 + p1)
    pa = np.linalg.norm(pm, axis=1)
    ph = pm / pa[:, None]
    res = (p1 - p0) - field_kick(x0, x1, q, field, nsub)
    res = res - np.sum(res * ph, 1, keepdims=True) * ph
    u1 = np.cross(ph, np.array([0.0, 0.0, 1.0]))
    u1 /= np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), 1e-12)
    u2 = np.cross(ph, u1)
    m = np.array([MASS.get(int(abs(x)), MASS[211]) for x in np.asarray(pdg)]) if len(pa) else np.zeros(0)
    beta = pa / np.sqrt(pa * pa + m * m)
    return {"m1": 1000 * beta * np.sum(res * u1, 1), "m2": 1000 * beta * np.sum(res * u2, 1), "p": pa,
            "dloss": np.linalg.norm(p1, axis=1) - np.linalg.norm(p0, axis=1)}


def select_segments(s: pd.DataFrame) -> pd.DataFrame:
    """Prompt (production radius < 1 mm) mu and pi, the two hits on different layers."""
    keep = (s.rv < 1.0) & np.isin(np.abs(s.pdg), (13, 211)) & ~((s.volume0 == s.volume1) & (s.layer0 == s.layer1))
    return s[keep].reset_index(drop=True)


def crossing_table(s: pd.DataFrame, field, nsub: int = NSUB) -> pd.DataFrame:
    """Per selected segment with mean momentum in P_RANGE: kicks, energy loss, region, |eta|, species."""
    s = select_segments(s)
    x0 = s[["tx0", "ty0", "tz0"]].to_numpy(np.float64)
    x1 = s[["tx1", "ty1", "tz1"]].to_numpy(np.float64)
    p0 = s[["tpx0", "tpy0", "tpz0"]].to_numpy(np.float64)
    p1 = s[["tpx1", "tpy1", "tpz1"]].to_numpy(np.float64)
    pm = np.linalg.norm(0.5 * (p0 + p1), axis=1)
    m = (pm > P_RANGE[0]) & (pm < P_RANGE[1])
    d = decompose(x0[m], x1[m], p0[m], p1[m], s.q.to_numpy(np.float64)[m], s.pdg.to_numpy()[m], field, nsub)
    return pd.DataFrame({**d, "abseta": np.abs(s.eta.to_numpy(np.float64)[m]),
                         "region": region_of(s.volume0.to_numpy()[m]), "species": species_of(s.pdg.to_numpy()[m])})


def mad(x) -> float:
    x = np.asarray(x, np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def summarise(t: pd.DataFrame, label: str, min_segments: int = MIN_SEGMENTS) -> list[dict]:
    """Rows per region, |eta| band and species with at least min_segments segments."""
    rows = []
    for reg in REGIONS:
        for lo, hi in ETA_BANDS:
            for sp in ("mu", "pi"):
                d = t[(t.region == reg) & (t.abseta >= lo) & (t.abseta < hi) & (t.species == sp)]
                if len(d) < min_segments:
                    continue
                rows.append({"sample": label, "region": reg, "abseta": f"{lo}-{hi}", "species": sp,
                             "segments": int(len(d)), "W_polar_MeV": round(mad(d.m2), 3),
                             "W_bend_MeV": round(mad(d.m1), 3),
                             "median_dp_MeV": round(float(1000 * d.dloss.median()), 3)})
    return rows


def compare_rows(ref: list[dict], test: list[dict]) -> list[dict]:
    """Bins present in both: W and dE of both samples and their ratios (test / reference)."""
    key = lambda r: (r["region"], r["abseta"], r["species"])   # noqa: E731
    tr = {key(r): r for r in test}
    out = []
    for r in ref:
        t = tr.get(key(r))
        if t is None:
            continue
        out.append({"region": r["region"], "abseta": r["abseta"], "species": r["species"],
                    "segments_ref": r["segments"], "segments_test": t["segments"],
                    "W_ref_MeV": r["W_polar_MeV"], "W_test_MeV": t["W_polar_MeV"],
                    "W_ratio": round(t["W_polar_MeV"] / r["W_polar_MeV"], 3) if r["W_polar_MeV"] else None,
                    "dE_ref_MeV": r["median_dp_MeV"], "dE_test_MeV": t["median_dp_MeV"],
                    "dE_ratio": round(t["median_dp_MeV"] / r["median_dp_MeV"], 3) if r["median_dp_MeV"] else None})
    return out


# ------------------------------------------------------------------------------------------------ last-hit closure
def propagate(x0, p0, q, field, h: float = 5.0, nsteps: int = 760) -> np.ndarray:
    """Material-free RK4 of many tracks (fixed path step h [mm]); trajectory (n, nsteps + 1, 3)."""
    x = np.array(x0, np.float64)
    pm = np.linalg.norm(p0, axis=1)
    t = np.asarray(p0, np.float64) / pm[:, None]
    k = (K * np.asarray(q, np.float64) / pm / 1000.0)[:, None]

    def b(xx):
        r = np.hypot(xx[:, 0], xx[:, 1])
        br, bz = field(r, xx[:, 2])
        rs = np.maximum(r, 1e-9)
        return np.stack([br * xx[:, 0] / rs, br * xx[:, 1] / rs, bz], 1)

    def f(xx, tt):
        return tt, k * np.cross(tt, b(xx))

    traj = np.empty((len(x), nsteps + 1, 3))
    traj[:, 0] = x
    for i in range(nsteps):
        k1x, k1t = f(x, t)
        k2x, k2t = f(x + 0.5 * h * k1x, t + 0.5 * h * k1t)
        k3x, k3t = f(x + 0.5 * h * k2x, t + 0.5 * h * k2t)
        k4x, k4t = f(x + h * k3x, t + h * k3t)
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        t = t + h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        traj[:, i + 1] = x
    return traj


def crossing(traj: np.ndarray, kind: str, value: float, start: int = 0) -> tuple[np.ndarray, int]:
    """First crossing of r = value (kind 'r') or z = value (kind 'z') along one trajectory (m, 3) at or after step
    `start`, linearly interpolated; (NaN point, -1) if none."""
    c = np.hypot(traj[:, 0], traj[:, 1]) if kind == "r" else traj[:, 2]
    d = c[start:] - value
    sg = np.flatnonzero(d[:-1] * d[1:] <= 0)
    if not len(sg):
        return np.full(3, np.nan), -1
    j = sg[0]
    a, b = d[j], d[j + 1]
    fr = a / (a - b) if a != b else 0.0
    i = start + j
    return traj[i] + fr * (traj[i + 1] - traj[i]), i


def closure_tracks(s: pd.DataFrame, per_bin: int = CLOSURE_PER_BIN, seed: int = 3) -> list[tuple]:
    """Tracks of the closure (module docstring) from segments: (vertex, momentum, q, pT, eta, |PDG|), hit positions
    in order and their volumes."""
    s = s[(s.rv < 1.0) & (s.pt > CLOSURE_PT[0][0]) & (s.pt < CLOSURE_PT[-1][1]) & np.isin(np.abs(s.pdg), (13, 211))
          & (s.n_hits >= 5)]
    rng = np.random.default_rng(seed)
    u = s.drop_duplicates(["event", "pid"])[["event", "pid", "eta"]]
    pick = []
    for lo, hi in CLOSURE_ETA:
        idx = np.flatnonzero((u.eta.abs() >= lo) & (u.eta.abs() < hi))
        pick.append(idx if len(idx) <= per_bin else rng.choice(idx, per_bin, replace=False))
    sel = u.iloc[np.concatenate(pick)][["event", "pid"]]
    s = s.merge(sel, on=["event", "pid"]).sort_values(["event", "pid", "rank0"])
    out = []
    for _, g in s.groupby(["event", "pid"], sort=False):
        X = np.vstack([g[["tx0", "ty0", "tz0"]].to_numpy(),
                       g[["tx1", "ty1", "tz1"]].to_numpy()[-1:]]).astype(np.float64)
        V = np.r_[g.volume0.to_numpy(), g.volume1.to_numpy()[-1:]]
        P = np.array([g.vx.iloc[0], g.vy.iloc[0], g.vz.iloc[0], g.px.iloc[0], g.py.iloc[0], g.pz.iloc[0],
                      g.q.iloc[0], g.pt.iloc[0], g.eta.iloc[0], abs(g.pdg.iloc[0])], np.float64)
        out.append((P, X, V))
    return out


def last_hit_residuals(tracks: list[tuple], field, chunk: int = 1000) -> pd.DataFrame:
    """|r dphi| [mm] between each track's last hit and the material-free trajectory at that hit's surface."""
    rows = []
    for c0 in range(0, len(tracks), chunk):
        part = tracks[c0:c0 + chunk]
        P = np.array([t[0] for t in part])
        traj = propagate(P[:, :3], P[:, 3:6], P[:, 6], field)
        for k, (p, X, V) in enumerate(part):
            start, last = 0, None
            for j in range(len(X)):
                rb = np.hypot(X[j, 0], X[j, 1])
                kind, val = ("r", rb) if V[j] in BARREL_VOLUMES else ("z", X[j, 2])
                pt_, idx = crossing(traj[k], kind, val, start)
                if idx < 0:
                    last = None
                    break
                start = idx
                dphi = np.angle(np.exp(1j * (np.arctan2(pt_[1], pt_[0]) - np.arctan2(X[j, 1], X[j, 0]))))
                last = rb * dphi
            if last is not None:
                rows.append((p[7], abs(p[8]), p[9], abs(last)))
    return pd.DataFrame(rows, columns=["pt", "abseta", "apdg", "abs_drphi_last"])


def closure_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi in CLOSURE_ETA:
        for plo, phi in CLOSURE_PT:
            m = (df.abseta >= lo) & (df.abseta < hi) & (df.pt >= plo) & (df.pt < phi)
            if m.sum() < CLOSURE_MIN_TRACKS:
                continue
            rows.append({"abseta": f"{lo:g}-{hi:g}", "pt": f"{plo:g}-{phi:g}", "tracks": int(m.sum()),
                         "median_abs_drphi_last_mm": round(float(df.abs_drphi_last[m].median()), 3),
                         "p90_abs_drphi_last_mm": round(float(df.abs_drphi_last[m].quantile(0.9)), 3)})
    return rows


# ------------------------------------------------------------------------------------------------ driver
def sim_directory(path: str, tmp: str) -> str:
    """A Fatras output directory, or a tar of one (packed by the field gates) extracted into tmp."""
    if os.path.isdir(path):
        return path
    with tarfile.open(path) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".csv")
                   and not m.name.startswith("/") and ".." not in m.name.split("/")]
        tf.extractall(tmp, members=members, filter="data")
    dirs = sorted({os.path.dirname(m.name) for m in members})
    if len(dirs) != 1:
        raise ValueError(f"{path}: expected one directory of CSVs, found {dirs}")
    return os.path.join(tmp, dirs[0])


def cmd_run(a) -> dict:
    t0 = time.time()
    field = OS.NodeField.from_loop_fit(a.loop_fit)
    out = {"plan": "docs/pileup_overlay_plan.md sections 4.3 and 10, M1 (descriptive)", "label": a.label,
           "sim": os.path.abspath(a.sim), "segments": os.path.abspath(a.segments),
           "loop_fit": os.path.abspath(a.loop_fit), "loop_fit_md5": OS.file_md5(a.loop_fit), "nsub": a.nsub,
           "selection": {"p_gev": P_RANGE, "min_segments": MIN_SEGMENTS, "species": ["mu", "pi"], "rv_mm": "< 1"},
           "closure": {"pt_bins": CLOSURE_PT, "eta_bins": CLOSURE_ETA, "per_bin": CLOSURE_PER_BIN}}
    with tempfile.TemporaryDirectory() as tmp:
        d = sim_directory(a.sim, tmp)
        acts = acts_segments(d)
        out["acts_segments"] = int(len(acts))
        print(f"[m1] ACTS segments {len(acts)} ({time.time() - t0:.0f} s)", flush=True)
        ta = crossing_table(acts, field, a.nsub)
        rows_a = summarise(ta, a.label)
        clo_a = closure_rows(last_hit_residuals(closure_tracks(acts), field))
    tml = trackml_segments(a.segments)
    out["trackml_segments"] = int(len(tml))
    print(f"[m1] TrackML segments {len(tml)} ({time.time() - t0:.0f} s)", flush=True)
    tt = crossing_table(tml, field, a.nsub)
    rows_t = summarise(tt, "trackml_tp")
    clo_t = closure_rows(last_hit_residuals(closure_tracks(tml), field))
    del tml, tt
    out["ms"] = {"trackml_tp": rows_t, a.label: rows_a}
    out["compare"] = compare_rows(rows_t, rows_a)
    out["closure"] = {"trackml": clo_t, a.label: clo_a}
    if os.path.exists(STUDY_MATERIAL):           # the design study's rows (unpatched build, constant 2 T) for context
        with open(STUDY_MATERIAL) as fh:
            st = json.load(fh)["ms"]
        out["study_rows"] = {k: [r for r in st if r["sample"] == k] for k in ("trackml_tp", "acts_const_before")}
    if os.path.exists(STUDY_CLOSURE):
        with open(STUDY_CLOSURE) as fh:
            out["study_closure"] = json.load(fh)
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"[m1] wrote {a.out_json} ({out['seconds']} s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sim", required=True, help="Fatras output directory (or tar) of the prompt probe")
    r.add_argument("--out-json", required=True)
    r.add_argument("--segments", default=SEGMENTS)
    r.add_argument("--loop-fit", default=LOOP_FIT)
    r.add_argument("--label", default="acts_zmirror_map")
    r.add_argument("--nsub", type=int, default=NSUB)
    a = ap.parse_args(argv)
    cmd_run(a)


if __name__ == "__main__":
    main()
