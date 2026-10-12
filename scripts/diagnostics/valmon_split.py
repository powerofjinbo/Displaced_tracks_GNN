"""Primary efficiency on the 500 HNL val events used for checkpoint selection by the HNL-trained runs (arm ii valmon)
and on the other 1,499, learned-merging builder (val only)."""
import numpy as np, pandas as pd
from hnl.eval import bootstrap_tables as BT
from hnl.train.acorn_runner import arm_events
t = pd.read_csv("/work/submit/jinboz1/hnl/hnl-displaced/results/training_sample_val_mwslmerge.csv")
vm = {BT.SAMPLE_CODE[s] * 100_000_000 + int(i) for s, i in arm_events("ii", "valmon")}
info = BT.info_table(BT.expected_events(BT.VAL_SETS))
rows = []
for _, r in t.iterrows():
    pp, pe, _ = BT.val_tables(r.run, r.working_point)
    x = pp[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    m = BT.masks(x)["eff_primary"]
    v = x.event_id.isin(vm).to_numpy()
    rows.append({"model": r.model, "seed": r.seed, "valmon": x.dm_matched[m & v].mean(), "n_valmon": int((m & v).sum()),
                 "rest": x.dm_matched[m & ~v].mean(), "n_rest": int((m & ~v).sum()), "all": x.dm_matched[m].mean()})
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(5).to_string())
print(d.groupby("model")[["valmon", "rest", "all"]].mean().mul(100).round(2).to_string())
