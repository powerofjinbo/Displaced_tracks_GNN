"""ACTS v15 CSV output -> npz events in the Zenodo format read by hnl.data.npz (keys and dtypes NPZ_KEYS), plus one
truth sidecar per chunk (docs/new_samples_spec.md section 5).

Per event:
  hits      one row per measurement, in measurement order (= geometry-id order, checked); hid = row index.
            x = (r, phi, z) of the smeared measurement (local0, local1 placed on its surface with detectors.csv) or,
            with --x truth, of the simulated hit.
            pid = particle of the measurement's simulated hit, taken from measurement-simhit-map.csv (never from the
            row order). Each measurement must have exactly one simulated hit and each simulated hit at most one
            measurement (smearing digitisation without merging).
            A particle with more than one hit on a module keeps the earliest (sim-hit time); the others are dropped
            and counted (the Zenodo files have none).
  edges     layerless_true_edges: each particle's kept hits in sim-hit time order, consecutive pairs (row indices).
  particles particles_final.csv (end state, as in the Zenodo files) with columns PARTICLE_COLUMNS; parent_pid = 0
            (unlabeled); pt = hypot(px, py), radius = hypot(vx, vy), eta of the momentum (NaN when pt = 0).
Sidecar truth_<first>.npz: per particle with hits, its INITIAL state (production vertex and momentum,
particles_initial.csv), and per event the numbers of dropped duplicate hits and of measurements.

    python scripts/acts/csv_to_npz.py CSVDIR --detectors detectors.csv --out SAMPLEDIR --first N --events N
        [--x measured|truth] [--sidecar PATH] [--keep-csv]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from hnl.data.npz import NPZ_KEYS, PARTICLE_COLUMNS  # noqa: E402

STEMS = ("particles_initial", "particles_final", "hits", "measurements", "measurement-simhit-map")


def surfaces(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"geometry_id": np.uint64})
    return d.set_index("geometry_id")


def read(csvdir, ev, stem, **kw):
    return pd.read_csv(os.path.join(csvdir, f"event{ev:09d}-{stem}.csv"), **kw)


def convert(csvdir: str, ev: int, surf: pd.DataFrame, xmode: str):
    U64 = {"particle_id": np.uint64, "geometry_id": np.uint64}
    pf = read(csvdir, ev, "particles_final", dtype=U64)
    pi = read(csvdir, ev, "particles_initial", dtype=U64)
    sh = read(csvdir, ev, "hits", dtype=U64)
    ms = read(csvdir, ev, "measurements", dtype=U64)
    mp = read(csvdir, ev, "measurement-simhit-map")
    # links: one sim hit per measurement, at most one measurement per sim hit
    if mp.measurement_id.duplicated().any() or mp.hit_id.duplicated().any():
        raise ValueError(f"event {ev}: measurement-simhit map is not one-to-one")
    if len(mp) != len(ms) or not np.array_equal(np.sort(mp.measurement_id.to_numpy()), ms.measurement_id.to_numpy()):
        raise ValueError(f"event {ev}: map does not cover the measurements")
    ms = ms.merge(mp, on="measurement_id", how="left").sort_values("measurement_id").reset_index(drop=True)
    hit = sh.iloc[ms.hit_id.to_numpy()].reset_index(drop=True)
    if not np.array_equal(hit.geometry_id.to_numpy(), ms.geometry_id.to_numpy()):
        raise ValueError(f"event {ev}: measurement and simulated hit on different surfaces")
    geo = ms.geometry_id.to_numpy()
    if np.any(geo[1:] < geo[:-1]):
        raise ValueError(f"event {ev}: measurements not in geometry-id order")
    # global positions
    s = surf.loc[geo]
    c = s[["cx", "cy", "cz"]].to_numpy()
    ua = s[["rot_xu", "rot_yu", "rot_zu"]].to_numpy()
    va = s[["rot_xv", "rot_yv", "rot_zv"]].to_numpy()
    meas = c + ua * ms.local0.to_numpy()[:, None] + va * ms.local1.to_numpy()[:, None]
    truth = hit[["tx", "ty", "tz"]].to_numpy()
    # self-check: the measurement lies on the sim hit's surface, displaced only in the surface plane
    dl = truth - c
    loc_t = np.c_[(dl * ua).sum(1), (dl * va).sum(1)]
    res = np.c_[ms.local0 - loc_t[:, 0], ms.local1 - loc_t[:, 1]] / np.sqrt(ms[["var_local0", "var_local1"]].to_numpy())
    pos = meas if xmode == "measured" else truth
    pid = hit.particle_id.to_numpy()
    tt = hit.tt.to_numpy()
    # one hit per (particle, module): keep the earliest
    order = np.lexsort((tt, geo, pid))
    first_of = np.ones(len(order), bool)
    first_of[1:] = (pid[order][1:] != pid[order][:-1]) | (geo[order][1:] != geo[order][:-1])
    keep = np.zeros(len(order), bool)
    keep[order[first_of]] = True
    n_dup = int((~keep).sum())
    rows = np.flatnonzero(keep)
    pos, pid, geo, tt = pos[rows], pid[rows], geo[rows], tt[rows]
    n = len(rows)
    r = np.hypot(pos[:, 0], pos[:, 1])
    x = np.c_[r, np.arctan2(pos[:, 1], pos[:, 0]), pos[:, 2]].astype(np.float32)
    # true edges: chain of each particle's hits in time order
    o = np.lexsort((tt, pid))
    same = pid[o][1:] == pid[o][:-1]
    edges = np.vstack([o[:-1][same], o[1:][same]]).astype(np.int64)
    # particle table (end state)
    P = pf[["particle_id", "particle_type", "process", "vx", "vy", "vz", "vt", "px", "py", "pz", "m", "q"]].copy()
    if len(np.setdiff1d(np.unique(pid), P.particle_id.to_numpy())):
        raise ValueError(f"event {ev}: hit particle missing from particles_final")
    pt = np.hypot(P.px, P.py)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.where(pt > 0, np.arcsinh(P.pz / np.where(pt > 0, pt, 1)), np.nan)
    tab = np.c_[P.particle_id.to_numpy().astype(np.float64), P[["particle_type", "process", "vx", "vy", "vz", "vt",
                                                                 "px", "py", "pz", "m", "q"]].to_numpy(np.float64),
                np.zeros(len(P)), pt, np.hypot(P.vx, P.vy), eta]
    if np.any(P.particle_id.to_numpy() >= np.uint64(2 ** 53)):
        raise ValueError(f"event {ev}: particle id not exact in float64")
    ev_npz = {"x": x, "hid": np.arange(n, dtype=np.int64), "pid": pid.astype(np.int64), "layers": geo.astype(np.int64),
              "layerless_true_edges": edges, "particles": tab.astype(np.float64)}
    assert {k: str(v.dtype) for k, v in ev_npz.items()} == NPZ_KEYS and tab.shape[1] == len(PARTICLE_COLUMNS)
    withhits = np.isin(pi.particle_id.to_numpy(), np.unique(pid))
    side = pi[withhits][["particle_id", "particle_type", "process", "vx", "vy", "vz", "vt", "px", "py", "pz", "q"]]
    stats = {"n_meas": len(ms), "n_dup": n_dup, "res_abs_gt5": int((np.abs(res) > 5).any(1).sum()),
             "res_sigma0": float(np.std(res[:, 0])) if len(res) else np.nan,
             "res_sigma1": float(np.std(res[:, 1])) if len(res) else np.nan}
    return ev_npz, side, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvdir")
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--x", choices=["measured", "truth"], default="measured")
    ap.add_argument("--sidecar")
    ap.add_argument("--keep-csv", action="store_true")
    a = ap.parse_args()
    surf = surfaces(a.detectors)
    os.makedirs(a.out, exist_ok=True)
    sides, stats = [], []
    for ev in range(a.first, a.first + a.events):
        e, side, st = convert(a.csvdir, ev, surf, a.x)
        tmp = os.path.join(a.out, f".{ev}.npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **e)
        os.replace(tmp, os.path.join(a.out, f"{ev}.npz"))
        side = side.assign(event=ev)
        sides.append(side)
        stats.append({"event": ev, **st})
        if not a.keep_csv:
            for stem in STEMS:
                os.remove(os.path.join(a.csvdir, f"event{ev:09d}-{stem}.csv"))
    S = pd.concat(sides, ignore_index=True)
    T = pd.DataFrame(stats)
    path = a.sidecar or os.path.join(a.out, f"truth_{a.first}.npz")
    np.savez(path, **{f"p_{c}": S[c].to_numpy(np.uint64 if c == "particle_id" else
                                              np.int64 if c in ("event", "particle_type", "process") else np.float64)
                      for c in S.columns}, **{f"e_{c}": T[c].to_numpy() for c in T.columns}, xmode=np.array(a.x))
    print(f"{len(T)} events; measurements {int(T.n_meas.sum())}; dropped duplicate hits {int(T.n_dup.sum())}; "
          f"|pull| > 5: {int(T.res_abs_gt5.sum())}; pull sd {T.res_sigma0.mean():.3f} / {T.res_sigma1.mean():.3f}")


if __name__ == "__main__":
    main()
