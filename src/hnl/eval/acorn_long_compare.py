"""Robustness check of the acorn training length (DECISIONS 2026-10-04 07:04 UTC): the acorn IGNNs continued to 60
epochs (runs/stage1_acorn_long/ignn60) against the 30-epoch baselines (runs/baseline_v1), val, per arm and seed: CC at
the best cut <= FR* and at the section-4 rule, and the Canto builders on the grid both runs share (MWS-layer + merging
with c in 0.03-0.2 from trackbuild_val_lmerge / trackbuild_val_mwslayer / trackbuild_val, walkthrough + merging from
trackbuild_val_mws / trackbuild_val_mwslayer / trackbuild_val); one-standard-error rule.

    python -m hnl.eval.acorn_long_compare armi,armii results/acorn_long_60ep_val.csv
"""
import os
import sys

import pandas as pd

from hnl.eval.ranking import rank

R = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/"
rows = []
for arm in sys.argv[1].split(","):
    for s in (0, 1, 2):
        for ep, base, grid_ml, grid_w in [("30", "baseline_v1", ["trackbuild_val_lmerge", "trackbuild_val_mwslayer"], ["trackbuild_val_mws", "trackbuild_val_mwslayer"]),
                                           ("60", "stage1_acorn_long/ignn60", ["trackbuild_val"], ["trackbuild_val"])]:
            run = R + f"{base}/{arm}_s{s}"
            for lab, e in [("CC best", {"source": "eval_val", "rule": "best"}),
                           ("CC s4", {"source": "eval_val", "rule": "s4"}),
                           ("MWS-layer+merge", {"source": grid_ml, "family": "mwslayermerge"}),
                           ("walk+merge", {"source": grid_w, "family": "walkmerge"})]:
                srcs = [e["source"]] if isinstance(e["source"], str) else e["source"]
                if not any(os.path.exists(os.path.join(run, g, "per_event.parquet")) for g in srcs):
                    rows.append(dict(arm=arm, seed=s, epochs=ep, method=lab, note="missing"))
                    continue
                if not isinstance(e["source"], str):   # rank() reads the event list from the first listed grid
                    e = {**e, "source": [g for g in srcs if os.path.exists(os.path.join(run, g, "per_event.parquet"))]}
                try:
                    _, ent = rank([{"label": "x", "entries": [{"run": run, **e}]}], 0.029368, tie_se=True)
                    r = ent.iloc[0]
                    rows.append(dict(arm=arm, seed=s, epochs=ep, method=lab, eff_primary=r.eff_primary,
                                     eff_ttbar_prompt=r.eff_ttbar_prompt, fake_rate_hnl=r.fake_rate_hnl,
                                     wp=str(r.working_point)))
                except Exception as ex:
                    rows.append(dict(arm=arm, seed=s, epochs=ep, method=lab, note=str(ex)[-90:]))
t = pd.DataFrame(rows)
t.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(t.round(4).to_string(index=False))
