"""What the hit-label exchange costs on validation: efficiency of targets with and without an exchanged hit.

Exchange flag per particle: label_swap_ransac.py at pT >= 1 GeV (OUT/pt1/ransac_parts_<sample>_val.parquet): a target
"with an exchanged hit" has a hit off its RANSAC circle whose position is held, on the same module, by another hit
lying on the circle; "without" has a fit and no such hit; "no fit" has fewer than 6 hits or no circle with >= 5 and
>= 60 % inliers. Matching: the stored per-particle tables (dm_matched) of
  Canto-L (ttbar) s0 at its frozen working point, mwslmerge 0.05 (runs/stage1_final_learned/l1tl/armi_s0/
  trackbuild_val_lmerge), and acorn (HNL) s0 with MWS-layer + learned merging, mwslmerge 0.03
  (runs/baseline_v1/armii_s0/trackbuild_val_fb3_all), as in results/strong_baselines_v3_val_entries.csv.
Populations: the HNL daughters of the primary efficiency (hnl.eval.bootstrap_tables.masks), the V0 daughters of
hnl.eval.second_population (HNL val), and the ttbar targets with 1 <= |d0| < 100 mm (ttbar val). Val only.

    python scripts/diagnostics/label_swap_effect.py --swap DIR > results/label_swap_effect_v1.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

RUNS = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
MODELS = {
    "Canto-L (ttbar) s0, LM 0.05": (f"{RUNS}/stage1_final_learned/l1tl/armi_s0/trackbuild_val_lmerge",
                                    ["mwslmerge", 0.05], 0.9887302779864763),
    "acorn (HNL) s0, LM 0.03": (f"{RUNS}/baseline_v1/armii_s0/trackbuild_val_fb3_all", ["mwslmerge", 0.03],
                                0.9879789631855748),
}


def pct(k, n):
    if n == 0:
        return "-"
    p = k / n
    return f"{100 * p:.2f} +- {100 * np.sqrt(max(p * (1 - p), 1e-12) / n):.2f} % ({int(k)}/{int(n)})"


def flags(swap: str, sample: str, code: int) -> pd.DataFrame:
    q = pd.read_parquet(os.path.join(swap, "pt1", f"ransac_parts_{sample}_val.parquet"))
    return pd.DataFrame({"event_id": np.int64(code) * 100_000_000 + q.event.to_numpy(np.int64),
                         "particle_id": q.pid.to_numpy(np.int64), "n_exch": q.n_exch.to_numpy()})


def main(argv=None):
    from hnl.data.splits import read_split
    from hnl.eval import bootstrap_tables as BT
    from hnl.eval import second_population as SP
    from hnl.eval.particles import particle_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", required=True)
    a = ap.parse_args(argv)
    sc = SP.SAMPLE_CODE
    ids = {s: np.int64(sc[s]) * 100_000_000 + read_split(s, "val").astype(np.int64)
           for s in ("ttbar", "hnl10_100", "hnl15_200")}
    allids = np.concatenate(list(ids.values()))
    t = particle_table(allids)
    m = BT.masks(t)
    fl = pd.concat([flags(a.swap, s, sc[s]) for s in ids], ignore_index=True)
    t = t.merge(fl, on=["event_id", "particle_id"], how="left", validate="one_to_one")
    t["grp"] = np.where(t.n_exch.isna(), "no fit", np.where(t.n_exch > 0, "with", "without"))
    v0 = SP.population_table(np.concatenate([ids["hnl10_100"], ids["hnl15_200"]]))
    v0 = v0[SP.population_masks(v0)["v0"]][["event_id", "particle_id"]]
    v0 = v0.merge(t[["event_id", "particle_id", "grp"]], on=["event_id", "particle_id"], how="left",
                  validate="one_to_one")
    code = (t.event_id.to_numpy() // 100_000_000)
    ad0 = t.d0.abs().to_numpy()
    pops = {"HNL daughters (primary population)": t[m["eff_primary"]][["event_id", "particle_id", "grp"]],
            "V0 daughters (HNL val)": v0,
            "ttbar targets, 1 <= |d0| < 100 mm": t[(code == sc["ttbar"]) & t.is_target.to_numpy(bool)
                                                   & (ad0 >= 1) & (ad0 < 100)][["event_id", "particle_id", "grp"]]}
    print("# Cost of the hit-label exchange on validation (val only)\n")
    print("Script: scripts/diagnostics/label_swap_effect.py; exchange flags from label_swap_ransac.py (pT >= 1 GeV, "
          "tolerance 1 mm); matching from the stored per-particle tables.\n")
    checks = []
    for name, (d, cfgkey, ref) in MODELS.items():
        cfg = json.load(open(os.path.join(d, "configs.json")))
        k = [i for i, c in enumerate(cfg) if c == cfgkey]
        assert len(k) == 1, (d, cfgkey)
        pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"), filters=[("config", "=", k[0])],
                             columns=["event_id", "particle_id", "dm_matched"])
        print(f"## {name}\n")
        for pname, P in pops.items():
            x = P.merge(pp, on=["event_id", "particle_id"], how="left", validate="one_to_one")
            if x.dm_matched.isna().any():
                raise SystemExit(f"{name} / {pname}: {int(x.dm_matched.isna().sum())} particles missing")
            ok = x.dm_matched.to_numpy(bool)
            if pname.startswith("HNL"):
                checks.append((name, ok.mean(), ref, abs(ok.mean() - ref) < 1e-12))
            line = [f"all {pct(ok.sum(), len(ok))}"]
            for g in ("without", "with", "no fit"):
                s = (x.grp == g).to_numpy()
                line.append(f"{g} an exchanged hit: {pct(ok[s].sum(), s.sum())}" if g != "no fit"
                            else f"no RANSAC fit: {pct(ok[s].sum(), s.sum())}")
            print(f"- {pname}: " + "; ".join(line) + ".")
        print()
    print("## Identity checks (primary efficiency against results/strong_baselines_v3_val_entries.csv)\n")
    for c in checks:
        print(f"- {c[0]}: {c[1]:.12f} vs {c[2]:.12f}: {'ok' if c[3] else 'FAILED'}")
    print()
    for pname, P in pops.items():
        n = len(P)
        print(f"- {pname}: {n} targets; with an exchanged hit {pct((P.grp == 'with').sum(), n)}, without "
              f"{pct((P.grp == 'without').sum(), n)}, no fit {pct((P.grp == 'no fit').sum(), n)}.")


if __name__ == "__main__":
    main()
