"""Quick projection of counted C1 bins on H-val: pilot gun hits per C1 bin times 30 (150 hosts / 5), unmatched upper
bound (the matched count keeps only hits on the common support). Truth only."""
import json, sys
import numpy as np
from hnl.overlay import c1 as C1, digitise as D, truth_checks as TC
ctx = TC.load_context("pilot", "/work/submit/jinboz1/hnl/pileup_design/hp_dryrun/pilot_v1")
gun = TC.gun_hit_features(ctx, D.Modules.load())
out = {}
for v in C1.VOLUMES:
    g = gun[gun.volume == v]
    row = {"hits_pilot": int(len(g)), "hits_x30": int(len(g)) * 30}
    for q, (col, stat, axis, pix) in C1.QUANTITIES.items():
        if pix and v not in D.PIXEL_VOLUMES:
            continue
        x, e = C1.axis_values(g, axis, v)
        n = np.bincount(np.digitize(x, e) - 1, minlength=len(e) + 1)[: len(e) - 1] * 30
        c = n >= 200
        row[q] = {"bins_counted_ub": int(c.sum()), "share_in_counted": round(float(n[c].sum() / max(n.sum(), 1)), 2),
                  "counted_ranges": [(float(e[k]), float(e[k + 1]), int(n[k])) for k in np.flatnonzero(c)]}
    out[int(v)] = row
json.dump(out, open(sys.argv[1], "w"), indent=1)
for v, r in out.items():
    print(v, r["hits_x30"], {q: (r[q]["bins_counted_ub"], r[q]["share_in_counted"]) for q in r if isinstance(r[q], dict)})
    if v == 8:
        print("  vol 8 mad_u ranges", r["mad_u"]["counted_ranges"]); print("  vol 8 span1 ranges", r["span1"]["counted_ranges"])
