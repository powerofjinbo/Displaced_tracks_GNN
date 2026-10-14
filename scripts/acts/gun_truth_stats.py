"""Truth-level summary of the produced new samples (docs/new_samples_spec.md); no model, no split, no target
selection of addendum G (that is its T1 commit).

Per generic component: gun particles per event, their species, r_v, pT and |d0| (helix in 2 T through the production
vertex), hits per gun particle, and the share with at least 5 hits and pT > 1 GeV by r_v. Per h -> s s sample:
charged particles with hits by production radius. Per sample: hits per event and dropped duplicate hits.

    python scripts/acts/gun_truth_stats.py --out results/new_samples_truth_v1.md
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

DATA = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated"
SAMPLES = {"generic_iso_v1": "iso", "generic_dense_v1": "dense", "hss15_100_v1": "hss", "hss40_300_v1": "hss"}
RV = [0, 10, 100, 200, 300, 450, 650, 800]
D0 = [0, 1, 10, 30, 100, 300, 1e9]


def d0_helix(x, y, px, py, q, b_tesla=2.0):
    pt = np.hypot(px, py)
    R = pt / (0.299792458 * b_tesla) * 1000.0             # mm
    ux, uy = px / pt, py / pt
    cx, cy = x + np.sign(q) * R * uy, y - np.sign(q) * R * ux
    return np.abs(np.hypot(cx, cy) - R)


def load_truth(d, stem):
    out = {}
    for f in sorted(glob.glob(os.path.join(d + "_truth", f"{stem}_*.npz"))):
        z = np.load(f)
        for k in z.files:
            if z[k].ndim:
                out.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def hits_per_particle(d, events, pids):
    """Hits of the given (event, particle id) pairs."""
    n = np.zeros(len(pids), np.int64)
    order = np.argsort(events, kind="stable")
    ev_s = events[order]
    for ev in np.unique(events):
        sel = order[np.searchsorted(ev_s, ev, "left"):np.searchsorted(ev_s, ev, "right")]
        with np.load(os.path.join(d, f"{ev}.npz")) as f:
            pid = f["pid"]
        u, c = np.unique(pid, return_counts=True)
        pos = np.searchsorted(u, pids[sel].astype(np.int64))
        pos = np.clip(pos, 0, len(u) - 1)
        n[sel] = np.where(u[pos] == pids[sel].astype(np.int64), c[pos], 0)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = ["# New samples: truth-level summary", "",
         "Produced per `docs/new_samples_spec.md`; `scripts/acts/gun_truth_stats.py`. Truth only: no model, no split, "
         "no addendum-G target selection.", ""]
    for name, kind in SAMPLES.items():
        d = os.path.join(DATA, name)
        files = glob.glob(os.path.join(d, "*.npz"))
        tr = load_truth(d, "truth")
        L += [f"## {name}", "", f"- events: {len(files)}; measurements per event: "
              f"{tr['e_n_meas'].mean():.0f} (sd {tr['e_n_meas'].std():.0f}); dropped duplicate hits: "
              f"{int(tr['e_n_dup'].sum())}; measurements with |pull| > 5: {int(tr['e_res_abs_gt5'].sum())}"]
        if kind == "hss":
            r = np.hypot(tr["p_vx"], tr["p_vy"])
            gen = (tr["p_particle_id"].astype(np.int64) >> 16) & 0xFF
            q = tr["p_q"]
            pt = np.hypot(tr["p_px"], tr["p_py"])
            prim = (gen == 0) & (q != 0)
            L += ["", "| production radius (mm) | charged primaries with hits per event | with pT > 1 GeV | electrons |",
                  "|---|---|---|---|"]
            ne = len(files)
            for lo, hi in zip([0, 1, 10, 100, 300, 600], [1, 10, 100, 300, 600, 1100]):
                s = prim & (r >= lo) & (r < hi)
                L.append(f"| {lo}-{hi} | {s.sum() / ne:.2f} | {(s & (pt > 1)).sum() / ne:.2f} | "
                         f"{(s & (np.abs(tr['p_particle_type']) == 11)).sum() / ne:.2f} |")
            L.append("")
            continue
        g = load_truth(d, "gun_truth")
        nh = hits_per_particle(d, g["event"], g["particle_id"])
        pt = np.hypot(g["px"], g["py"])
        d0 = d0_helix(g["vx"], g["vy"], g["px"], g["py"], g["q"])
        ne = len(np.unique(g["event"]))
        L.append(f"- gun particles: {len(nh)} ({len(nh) / ne:.2f} per event); with >= 1 hit {np.mean(nh > 0):.3f}; "
                 f"with >= 5 hits and pT > 1 GeV {np.mean((nh >= 5) & (pt > 1)):.3f}")
        sp, cnt = np.unique(np.abs(g["pdg"]), return_counts=True)
        L.append("- species: " + ", ".join(f"{int(s)} {c / len(nh):.3f}" for s, c in zip(sp, cnt)))
        L.append(f"- pT quantiles 10/50/90 %: " + " / ".join(f"{v:.2f}" for v in np.percentile(pt, [10, 50, 90]))
                 + " GeV")
        L += ["", "| r_v (mm) | gun particles | mean hits | >= 5 hits and pT > 1 GeV |", "|---|---|---|---|"]
        for lo, hi in zip(RV[:-1], RV[1:]):
            s = (g["r_v"] >= lo) & (g["r_v"] < hi)
            if s.any():
                L.append(f"| {lo}-{hi} | {s.sum()} | {nh[s].mean():.1f} | {np.mean((nh[s] >= 5) & (pt[s] > 1)):.3f} |")
        L += ["", "| abs(d0) (mm), helix in 2 T | gun particles with >= 5 hits and pT > 1 GeV |", "|---|---|"]
        t = (nh >= 5) & (pt > 1)
        for lo, hi in zip(D0[:-1], D0[1:]):
            L.append(f"| {lo:g}-{hi:g} | {int((t & (d0 >= lo) & (d0 < hi)).sum())} |")
        L.append("")
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
