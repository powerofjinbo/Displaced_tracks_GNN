"""How often do ttbar training particles cross the layers obliquely at high pT? |d0| / r_first ~ sin(angle between
the flight direction and the radial direction at the first hit). Truth tables only (no model output, no test event)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/work/submit/jinboz1/hnl/hnl-displaced/src")
from hnl.eval.particles import particle_table
from hnl.data.acorn_store import SAMPLE_CODE
REPO = "/work/submit/jinboz1/hnl/hnl-displaced"
from hnl.data.splits import read_split
def ids(split, sample, n=None):
    ev = [int(x) for x in read_split(sample, split)]
    ev = ev[:n] if n else ev
    return np.array([SAMPLE_CODE[sample] * 100_000_000 + e for e in ev], dtype=np.int64)
tt = particle_table(ids("train", "ttbar", 2000))
tt = tt[tt.n_hits >= 5].copy().reset_index(drop=True)
tt["ratio"] = tt.d0.abs() / tt.r_first
hn = pd.concat([particle_table(ids("val", s)) for s in ("hnl10_100", "hnl15_200")], ignore_index=True)
hn = hn[(hn.category == "hnl") & (hn.n_hits >= 5)].copy()
hn["ratio"] = hn.d0.abs() / hn.r_first
pb = [0, 0.5, 1.0, 1.5, 3.0, 1e9]; rb = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
for name, t in (("ttbar train (2000 events), particles with >= 5 hits", tt), ("HNL daughters (validation), >= 5 hits", hn),
                ("HNL daughters with |d0| >= 200 mm", hn[hn.d0.abs() >= 200])):
    t = t[np.isfinite(t.ratio)]
    h = pd.crosstab(pd.cut(t.pt, pb), pd.cut(t.ratio, rb))
    print(f"\n== {name}: {len(t)} particles; hit-weighted share with pT > 1.5 GeV and ratio > 0.7: "
          f"{((t.pt > 1.5) & (t.ratio > 0.7)).mul(t.n_hits - 1).sum() / (t.n_hits - 1).sum():.2e}; count {int(((t.pt > 1.5) & (t.ratio > 0.7)).sum())}")
    print(h.to_string())
    out = t[(t.pt > 1.5) & (t.ratio > 0.7)]
    if name.startswith("ttbar"):
        print("pT > 1.5 GeV and ratio > 0.7 in ttbar, by pdg:", out.pdg.abs().value_counts().head(8).to_dict(),
              " median r_first", out.r_first.median(), " median |d0|", out.d0.abs().median())

fb = [0, 50, 100, 200, 350, 500, 700, 5000]
sel_t = tt[(tt.pt > 1.5) & (tt.ratio > 0.7)]
sel_h = hn[(hn.pt > 1.5) & (hn.ratio > 0.7) & (hn.d0.abs() >= 200)]
print("\nfirst-hit radius of pT > 1.5 GeV, ratio > 0.7 particles (count):")
print(pd.DataFrame({"ttbar train (2000 ev)": pd.cut(sel_t.r_first, fb).value_counts().sort_index(),
                    "HNL daughters |d0|>=200 (val)": pd.cut(sel_h.r_first, fb).value_counts().sort_index()}).to_string())
big = tt[(tt.pt > 1.5) & (tt.ratio > 0.7) & (tt.r_first > 200)]
print("\nttbar: pT > 1.5, ratio > 0.7 and r_first > 200 mm:", len(big), "particles in 2000 events; by pdg",
      big.pdg.abs().value_counts().head(6).to_dict())
print("for scale, ttbar particles with >= 5 hits and pT > 1.5 in 2000 events:", int((tt.pt > 1.5).sum()))
