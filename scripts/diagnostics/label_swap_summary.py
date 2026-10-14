"""Summary of the label-exchange verification (label_swap_ransac.py, label_swap_verify.py) -> markdown on stdout.

    python scripts/diagnostics/label_swap_summary.py DIR > results/label_swap_verify_v1.md
DIR holds pt2/ (RANSAC, pT >= 2 GeV, per sample/split), pt1/ (RANSAC, pT >= 1 GeV, val) and local/ (local-fit pair
test on every same-module pair, val).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1.0


def pct(k, n):
    if n == 0:
        return "-"
    p = k / n
    return f"{100 * p:.2f} +- {100 * np.sqrt(max(p * (1 - p), 1e-12) / n):.2f} % ({k}/{n})"


def main():
    d = sys.argv[1]
    from hnl.data.convert import read_table
    from hnl.geometry.ids import decode_barcode
    from hnl.graphs.window_study import truth_particles
    print("# Verification of the hit-label exchange (dense-jet check follow-up)\n")
    print("Truth only; the authors' npz files read directly (hnl.data.npz.load_event). Scripts: "
          "scripts/diagnostics/label_swap_{ransac,verify,summary}.py; job wrapper slurm/diagnostics/"
          "label_swap_verify.sbatch.\n")
    # 1. per sample/split, RANSAC pT >= 2
    print("## 1. Robust trajectories (RANSAC circle, pT >= 2 GeV, >= 6 hits, tolerance 1 mm), per sample and split\n")
    print("| sample | split | events | clean particles | outlier hits, modules without another hit | outlier hits, "
          "modules shared with another particle | of these, another hit on the module lies on the trajectory | "
          "resolved multi-particle modules with labels not matching positions |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(d, "pt2", "ransac_summary_*.json"))):
        s = json.load(open(f))
        print(f"| {s['sample']} | {s['split']} | {s['events']} | {s['clean_particles']} | "
              f"{pct(s['outlier_unshared'], s['hits_unshared'])} | {pct(s['outlier_shared'], s['hits_shared'])} | "
              f"{pct(s['outlier_shared_explained'], s['outlier_shared'])} | "
              f"{pct(s['modules_resolved_nonidentity'], s['modules_resolved'])} |")
    # 2. ttbar val details
    H = pd.read_parquet(os.path.join(d, "pt2", "ransac_hits_ttbar_val.parquet"))
    M = pd.read_parquet(os.path.join(d, "pt2", "ransac_mods_ttbar_val.parquet"))
    sh = H[H.n_on_module > 1].copy()
    sh["ex"] = (sh.res >= TOL) & (sh.other_res < TOL)
    ex = sh[sh.ex]
    print("\n## 2. ttbar val (pT >= 2 GeV): the exchanged positions\n")
    print(f"- Distance of the position labelled as another particle from the particle's trajectory, for the "
          f"explained outliers: median {ex.other_res.median():.3f} mm, 90 % quantile {ex.other_res.quantile(.9):.3f} mm;"
          f" distance of the labelled hit: median {ex.res.median():.1f} mm.")
    sb = pd.cut(ex.other_sep, [0, 1, 2, 5, 10, 20, 50, 1e5])
    print("- Transverse separation of the two positions: " +
          ", ".join(f"{k}: {v}" for k, v in ex.groupby(sb, observed=False).size().items()))
    sh["sepb"] = pd.cut(sh.other_sep, [0, 2, 5, 10, 20, 50, 1e5])
    print("- Fraction of shared-module hits exchanged, by separation to the nearest other hit on the module: " +
          ", ".join(f"{k}: {100 * v:.1f} %" for k, v in sh.groupby("sepb", observed=False).ex.mean().items()))
    print("- By volume: " + ", ".join(f"{int(k)}: {100 * v:.1f} %" for k, v in sh.groupby("vol").ex.mean().items()))
    R = M[M.resolved].copy()
    R["identity"] = R.identity.astype(bool)
    print("- Resolved modules with n hits, fraction whose labels do not match the positions: " +
          ", ".join(f"n={k}: {pct(int((~g.identity).sum()), len(g))}" for k, g in R.groupby("n")))
    # generations of the particles in non-identity modules
    pids = np.array([int(q) for v in R[~R.identity].pids for q in v.split(",")], dtype=np.int64)
    gen = decode_barcode(pids)["gen"]
    print(f"- Particles in modules with mismatched labels: {len(pids)}, of which generator particles (gen 0) "
          f"{pct(int((gen == 0).sum()), len(pids))}.")
    # 3. local-fit pair test
    print("\n## 3. Local-fit pair test on every same-module pair (quadratic transverse fit + (r, z) line through up "
          "to two chain neighbours on each side), all particles\n")
    for f in sorted(glob.glob(os.path.join(d, "local", "summary_*.json"))):
        s = json.load(open(f))
        tag = f"{s['sample']}_{s['split']}"
        P = pd.read_parquet(os.path.join(d, "local", f"pairs_{tag}.parquet"))
        C = pd.read_parquet(os.path.join(d, "local", f"ctrl_{tag}.parquet"))
        b = pd.cut(P.sepxy, [0, 2, 5, 10, 20, 50, 1e5])
        cb = pd.cut(C.sepxy, [0, 2, 5, 10, 20, 50])
        print(f"- {tag}: same-module pairs {pct(s['pairs_exch'], s['pairs'])} exchanged; control (different modules "
              f"of one layer, <= 50 mm) {pct(s['ctrl_exch'], s['ctrl'])}. By separation: " +
              ", ".join(f"{k} {100 * v:.1f} %" for k, v in P.groupby(b, observed=False).exch.mean().items()) +
              "; control: " + ", ".join(f"{k} {100 * v:.2f} %" for k, v in C.groupby(cb, observed=False).exch.mean().items()))
        lo = np.minimum(P.pt_i, P.pt_j)
        print(f"  Both particles pT >= 2 GeV: {pct(int(P[lo >= 2].exch.sum()), int((lo >= 2).sum()))}; "
              f"any pT below 1 GeV: {pct(int(P[lo < 1].exch.sum()), int((lo < 1).sum()))}.")
        print(f"  Modules holding exactly two hits of two particles: {s['two_hit_modules']}; the hit with the lower hit "
              f"id carries the lower particle id in {pct(s['hid_order_eq_pid_order'], s['two_hit_modules'])}.")
        if "derived" in s:
            print(f"  Derived hit tables (hnl.data.convert) against the npz arrays, row by row (pid, hid, geometry id, "
                  f"r, phi, z): {s['derived']['mismatched_events']} of {s['derived']['events']} events differ.")
    # 4. targets, pT >= 1
    print("\n## 4. Targets (pT >= 1 GeV RANSAC; d0 as in docs/definitions.md section 3)\n")
    for f in sorted(glob.glob(os.path.join(d, "pt1", "ransac_parts_*_val.parquet"))):
        sample = os.path.basename(f)[len("ransac_parts_"):-len("_val.parquet")]
        Q = pd.read_parquet(f)
        ev = np.unique(Q.event)
        parts = read_table(os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"),
                                        "derived", "events", "v1"), "particles", sample, events=ev,
                           columns=["event", "particle_id", "category", "is_target", "pt"])
        tp = truth_particles(sample)
        parts = parts.merge(tp[tp.event.isin(ev)], on=["event", "particle_id"], how="left")
        t = parts[parts.is_target.astype(bool)].merge(Q.rename(columns={"pid": "particle_id"}),
                                                     on=["event", "particle_id"], how="left", suffixes=("", "_r"))
        t["ad0"] = t.d0.abs()
        t["fit"] = t.n_inl.notna()
        groups = {"ttbar": [("prompt (|d0| < 1 mm)", t.ad0 < 1), ("displaced (1 <= |d0| < 100 mm)",
                                                                   (t.ad0 >= 1) & (t.ad0 < 100))]}
        gl = groups.get(sample, [("HNL daughters", t.category == "hnl"), ("W muons", t.category == "w"),
                                 ("other, displaced (1 <= |d0| < 100 mm)", (t.category == "other") & (t.ad0 >= 1)
                                  & (t.ad0 < 100))])
        print(f"### {sample} val ({len(ev)} events)\n")
        for lab, m in gl:
            g = t[m]
            gf = g[g.fit]
            print(f"- {lab}: {len(g)} targets, RANSAC fit {pct(len(gf), len(g))}; with an exchanged hit "
                  f"{pct(int((gf.n_exch > 0).sum()), len(gf))}; |d0| of the circle through the inliers < 1 mm: "
                  f"{pct(int((gf.d0_clean_circle < 1).sum()), len(gf))}.")
        print()


if __name__ == "__main__":
    main()
