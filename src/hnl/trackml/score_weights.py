"""Share of the TrackML score weight carried by particles produced in the detector simulation (barcode generation >= 1,
hnl.trackml.displaced_diag.generation), in the first three events of T2 (accuracy phase) and of the throughput test
set. Reads the stored truth weights only (checked against train_1.zip for T2 events 1112 and 1138: same hits, particle
ids and weights to 7e-12); no track or score of Canto-L is read. Output: results/trackml_score_weights_v1.txt.

    python -m hnl.trackml.score_weights
"""
import numpy as np
import pandas as pd

import hnl.trackml.events as E
from hnl.trackml.displaced_diag import generation

TEST = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/test"

for name, d in (("T2 (accuracy phase)", "t2_nonoise"), ("test (throughput phase)", "test_nonoise")):
    evs = sorted(pd.read_parquet(f"{TEST}/{d}/per_event.parquet", columns=["event"]).event.unique())[:3]
    for e, ev in E.load_events(evs, particles=False).items():
        pid, w = ev.pid.astype(np.int64), ev.weight.astype(np.float64)
        sim = (pid != 0) & (generation(pid) > 0)
        print(f"{name} event {e}: hits {len(w)}; weight sum {w.sum():.4f}; noise weight {w[pid == 0].sum():.1e}; "
              f"generation >= 1: {sim.mean():.3f} of hits, {w[sim].sum():.3f} of the weight; "
              f"zero-weight particle hits {np.mean(w[pid != 0] == 0):.3f}")
