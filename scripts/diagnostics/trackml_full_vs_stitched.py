"""TrackML X1, first-epoch checkpoint, the 100 dev events (training events): edge scores of whole-event forwards
(scores/dev_epoch0_full, score_full.sbatch) against the stitched 8-sector scores (scores/dev_epoch0, score.sbatch, +-22.5
deg halos). Diagnostic for the S5 switch after gate G2 (DECISIONS 2026-10-08); no val or test event is read.

    python scripts/diagnostics/trackml_full_vs_stitched.py > results/diagnostics/trackml_full_vs_stitched.txt
"""
import glob
import os

import numpy as np
import pandas as pd

S = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/scores"


def events(d):
    """(event, DataFrame) per row group (one event each) of every scores file of a directory."""
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(S, d, "scores_*.parquet"))):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i).to_pandas()
            yield int(t.event.iloc[0]), t


TH = (1e-3, 1e-2, 0.05, 0.1, 0.5)
CUTS = (0.1, 0.2, 0.5)
n = 0
big = {t: 0 for t in TH}
per_event = []
eff = {(c, k): [0, 0] for c in CUTS for k in ("st_t", "fu_t", "st_f", "fu_f", "flip_t", "flip_f")}
nt = nf = 0
unmatched = [0, 0]
for (e1, st), (e2, fu) in zip(events("dev_epoch0"), events("dev_epoch0_full")):
    # the two stores were built on different GPUs (A6000 node-local for the stitched scores, A30 for the S4 store):
    # the FRNN radius graph differs in a few pairs at distance ~ r*; compare the common edges
    assert e1 == e2
    m = st.merge(fu, on=["a", "b"], suffixes=("_st", "_fu"), how="inner", validate="one_to_one")
    unmatched[0] += len(st) - len(m)
    unmatched[1] += len(fu) - len(m)
    y = m.y_st.to_numpy()
    assert (y == m.y_fu.to_numpy()).all()
    a, b = m.score_st.to_numpy(), m.score_fu.to_numpy()
    d = np.abs(a - b)
    n += len(d)
    for t in TH:
        big[t] += int((d > t).sum())
    per_event.append(100 * (d > 0.01).mean())
    nt += int(y.sum()); nf += int((~y).sum())
    for c in CUTS:
        eff[c, "st_t"][0] += int((a[y] > c).sum()); eff[c, "fu_t"][0] += int((b[y] > c).sum())
        eff[c, "st_f"][0] += int((a[~y] > c).sum()); eff[c, "fu_f"][0] += int((b[~y] > c).sum())
        fl = (a > c) != (b > c)
        eff[c, "flip_t"][0] += int(fl[y].sum()); eff[c, "flip_f"][0] += int(fl[~y].sum())
pe = np.array(per_event)
print(f"events {len(pe)}, common edges {n} ({nt} true); only in the stitched store {unmatched[0]}, only in the S4 store {unmatched[1]}")
for t in TH:
    print(f"|delta score| > {t}: {100 * big[t] / n:.3f} % of edges")
print(f"per event, % of edges with |delta| > 0.01: min {pe.min():.3f}, median {np.median(pe):.3f}, max {pe.max():.3f}")
for c in CUTS:
    print(f"cut {c}: true-edge efficiency stitched {100 * eff[c, 'st_t'][0] / nt:.3f} %, full {100 * eff[c, 'fu_t'][0] / nt:.3f} %; "
          f"fake edges kept stitched {100 * eff[c, 'st_f'][0] / nf:.4f} %, full {100 * eff[c, 'fu_f'][0] / nf:.4f} %; "
          f"different side: true {100 * eff[c, 'flip_t'][0] / nt:.4f} %, fake {100 * eff[c, 'flip_f'][0] / nf:.4f} %")
