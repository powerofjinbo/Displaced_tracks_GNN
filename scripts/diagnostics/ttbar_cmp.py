"""ttbar performance of Canto-L (HNL) s0 vs Canto-L (ttbar) s0-s2 at the same builder settings (val)."""
import json, os, sys
import numpy as np, pandas as pd
from hnl.eval import bootstrap_tables as BT
from hnl.eval.particles import particle_table
R = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
runs = {"Canto-L (ttbar) s0": f"{R}/stage1_final_learned/l1tl/armi_s0", "Canto-L (ttbar) s1": f"{R}/stage1_final_learned/l1tl/armi_s1",
        "Canto-L (ttbar) s2": f"{R}/stage1_final_learned/l1tl/armi_s2", "Canto-L (HNL) s0": f"{R}/stage1_final_learned_ts/l1tl/armii_s0"}
ev = BT.expected_events(BT.VAL_SETS)
p = particle_table(ev)
t0 = p[p.is_target.astype(bool)][["event_id", "particle_id", "category", "d0", "rv", "pt", "eta", "is_target"]]
rows = []
for name, run in runs.items():
    for wp in ["mwslayermerge 0.2 (trackbuild_val_ext)", "mwslayermerge 0.3 (trackbuild_val_ext)"]:
        pp, pe, d = BT.val_tables(run, wp)
        t = pp[["event_id", "particle_id", "dm_matched"]].merge(t0, on=["event_id", "particle_id"], how="inner")
        code = BT.split_event_id(t.event_id.to_numpy())[0]
        tt = t[code == BT.TTBAR_CODE]
        ad0 = tt.d0.abs()
        r = {"model": name, "wp": wp}
        for nm, s in [("ttbar_prompt(<1mm)", ad0 < 1), ("ttbar_1_10mm", (ad0 >= 1) & (ad0 < 10)), ("ttbar_10_100mm", (ad0 >= 10) & (ad0 < 100)),
                      ("ttbar_ge100mm", ad0 >= 100), ("ttbar_all_targets", ad0 >= 0), ("ttbar_pt_lt2", (ad0 < 1) & (tt.pt < 2)), ("ttbar_pt_ge10", (ad0 < 1) & (tt.pt >= 10))]:
            r[nm] = f"{tt.dm_matched[s].mean()*100:.2f} ({int(tt.dm_matched[s].sum())}/{int(s.sum())})"
        m = BT.masks(t)
        r["hnl_primary"] = f"{t.dm_matched[m['eff_primary']].mean()*100:.2f}"
        h = BT.split_event_id(pe.event_id.to_numpy())[0]
        r["fake_hnl_%"] = round(100 * pe.n_fake[h != 1].sum() / pe.n_tracks_ge5[h != 1].sum(), 3)
        r["fake_ttbar_%"] = round(100 * pe.n_fake[h == 1].sum() / pe.n_tracks_ge5[h == 1].sum(), 3)
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(df.to_string(index=False))
df.to_csv("/work/submit/jinboz1/hnl/tmp/leaderboard/ttbar_cmp.csv", index=False)
