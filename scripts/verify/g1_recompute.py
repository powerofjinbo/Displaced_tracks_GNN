"""Independent recomputation of claim G1 (addendum G section 6) from the stored per-particle / per-event tables.

Shares no code with hnl.eval.addendum_g_summary: P_G from the raw truth columns (target: pT > 1 GeV, |eta| < 4,
>= 5 hits; displaced source: barcode secondary vertex >= 1000, generation 0, sub-particle 0 (deviation 4);
|d0| >= 10 mm or r_v >= 100 mm), the bootstrap table regenerated from numpy default_rng(12345) over the union events
sorted by event number, ratio-of-sums replicates, sigma^2 = sd_boot(paired diff of seed means)^2 + s_A^2/3 + s_B^2/3.
Usage: python g1_recompute.py [EVAL_ROOT]
"""
import json
import sys

import numpy as np
import pandas as pd

DATA = "/ceph/submit/data/user/j/jinboz1/hnl_lrt"
REPO = "/work/submit/jinboz1/hnl/hnl-displaced"
ROOT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/runs/addendum_g_v1/eval"
B = 500

ev = np.array(sorted(int(x) for x in open(f"{REPO}/splits/generic_iso_union_test.txt") if not x.startswith("#")))
t = pd.concat([pd.read_parquet(f"{DATA}/derived/generated_v1/t1/truth_{s}.parquet",
                               columns=["event", "particle_id", "pt", "eta", "n_hits", "vsec", "gen", "sub", "d0",
                                        "r_v", "is_target", "gun", "in_pop"])
               for s in ("generic_iso", "generic_iso_x1")], ignore_index=True)
t = t[t.event.isin(ev)].reset_index(drop=True)
tgt = (t.pt > 1.0) & (t.eta.abs() < 4.0) & (t.n_hits >= 5)
src = (t.vsec >= 1000) & (t["gen"] == 0) & (t["sub"] == 0)
pg = tgt & src & ((t.d0.abs() >= 10.0) | (t.r_v >= 100.0))
out = {"events": int(len(ev)), "P_G": int(pg.sum()), "P_G_equals_table_in_pop": bool((pg == t.in_pop.astype(bool)).all()),
       "target_equals_table": bool((tgt == t.is_target.astype(bool)).all()),
       "source_equals_table_gun": bool((src == t.gun.fillna(False).astype(bool)).all())}
p = t[pg].copy()
p["eid"] = 5 * 100_000_000 + p.event.astype(np.int64)
pos = np.searchsorted(ev, p.event.to_numpy())

rng = np.random.default_rng(12345)
idx = rng.integers(0, len(ev), size=(B, len(ev)))
w = np.stack([np.bincount(idx[k], minlength=len(ev)) for k in range(B)]).astype(float)
z = np.load(f"{DATA}/derived/bootstrap/generated_v1/generic_iso_union_test.npz")
out["boot_table_equals_stored"] = bool(np.array_equal(z["units"], ev) and np.array_equal(z["w"], w))
den = np.bincount(pos, minlength=len(ev)).astype(float)


def model(m):
    pts, reps, fr, frr = [], [], [], []
    for s in range(3):
        pp = pd.read_parquet(f"{ROOT}/{m}_s{s}/test/per_particle.parquet")
        pp = pp.drop_duplicates(["event_id", "particle_id"])
        k = p.merge(pp, left_on=["eid", "particle_id"], right_on=["event_id", "particle_id"], how="left")
        assert k.dm_matched.notna().all(), f"{m} s{s}: P_G rows missing"
        num = np.bincount(pos, weights=k.dm_matched.astype(float).to_numpy(), minlength=len(ev))
        pts.append(num.sum() / den.sum())
        reps.append((w @ num) / (w @ den))
        pe = pd.read_parquet(f"{ROOT}/{m}_s{s}/test/per_event.parquet")
        pe = pe[pe.event_id // 100_000_000 == 5]
        assert np.array_equal(np.sort((pe.event_id % 100_000_000).to_numpy()), ev), f"{m} s{s}: per-event rows"
        q = np.searchsorted(ev, (pe.event_id % 100_000_000).to_numpy())
        nf = np.bincount(q, weights=pe.n_fake.to_numpy(float), minlength=len(ev))
        nt = np.bincount(q, weights=pe.n_tracks_ge5.to_numpy(float), minlength=len(ev))
        fr.append(nf.sum() / nt.sum())
    return np.array(pts), np.array(reps), np.array(fr)


A, RA, FA = model("canto_l_ttbar")
Bm, RB, FB = model("acorn_hnl_lm")
d = 100 * (A.mean() - Bm.mean())
sb = 100 * np.std(RA.mean(0) - RB.mean(0), ddof=1)
ss = 100 * np.sqrt(np.var(A, ddof=1) / 3 + np.var(Bm, ddof=1) / 3)
sg = np.hypot(sb, ss)
dfr = 100 * (FA.mean() - FB.mean())
guard = dfr <= 0.5
if d - 2 * sg > -1.0 and guard:
    oc = "holds"
elif d + 2 * sg < -1.0:
    oc = "does not hold, resolved below the margin"
else:
    oc = "does not hold, inconclusive"
out.update({"eff_A_seeds": (100 * A).round(4).tolist(), "eff_B_seeds": (100 * Bm).round(4).tolist(),
            "eff_A": 100 * A.mean(), "eff_B": 100 * Bm.mean(), "delta": d, "sigma_boot": sb, "sigma_seeds": ss,
            "sigma": sg, "delta_minus_2sigma": d - 2 * sg, "delta_plus_2sigma": d + 2 * sg,
            "fake_A": 100 * FA.mean(), "fake_B": 100 * FB.mean(), "fake_diff": dfr, "guard_ok": bool(guard),
            "outcome": oc, "underpowered": int(pg.sum()) < 1000})
print(json.dumps(out, indent=1))
