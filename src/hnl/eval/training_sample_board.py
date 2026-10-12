"""Training-sample study (docs/training_sample_study.md), val only: per-seed efficiencies of Canto-L trained on ttbar
(arm i, three seeds), the three acorn arms (three seeds each) and Canto-L trained on HNL (arm ii, seed 0) and on
ttbar + HNL (arm iii, seed 0, when its grids exist), every model with one builder family.

Working points: Canto-L (ttbar) and the acorn arms at their committed per-seed val working points
(results/strong_baselines_v3_val_entries.csv; reproduced to 1e-9); Canto-L (HNL) / (ttbar + HNL) by
hnl.eval.final_select.run_rows over trackbuild_val_ext, trackbuild_val_lmerge, trackbuild_val (the arm (i)
selection's grids; the same call is checked to give Canto-L (ttbar) seed 0's committed entry). Quantities per seed:
the populations of hnl.eval.bootstrap_tables.masks (primary, all HNL targets, W muons, ttbar prompt, the |d0| bins) as
k / n, and the HNL fake rate.

    python -m hnl.eval.training_sample_board --family mwslmerge --out results/training_sample_val_mwslmerge.csv
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from hnl.eval import bootstrap_tables as BT

R = os.path.join(BT.DATA, "runs")
REPO = BT.REPO
ENTRIES = os.path.join(REPO, "results", "strong_baselines_v3_val_entries.csv")
SUFFIX = {"mwslayermerge": (" + MWS-layer + merging (Canto builder)",
                            "Canto-L + MWS-layer + merging (Canto builder), same builder (not the frozen builder)"),
          "mwslmerge": (" + MWS-layer + learned merging (Canto-L's builder)", "Canto-L (frozen: learned merging)")}
GRIDS = "trackbuild_val_ext,trackbuild_val_lmerge,trackbuild_val"
NEW = {"Canto-L (HNL)": "armii_s0", "Canto-L (ttbar+HNL)": "armiii_s0"}
QS = ("eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt") + BT.D0_BINS


def metrics(run: str, wp: str, info: pd.DataFrame) -> dict:
    pp, pe, _ = BT.val_tables(run, wp)
    t = pp[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    m = BT.masks(t)
    row = {}
    for q in QS:
        k, n = float(t.dm_matched[m[q]].sum()), float(m[q].sum())
        row[q + "_k"], row[q + "_n"], row[q] = k, n, k / n
    h = BT.split_event_id(pe.event_id.to_numpy())[0] != BT.TTBAR_CODE
    row["fake_rate_hnl"] = float(pe.n_fake[h].sum() / pe.n_tracks_ge5[h].sum())
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=sorted(SUFFIX), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-grids", default=GRIDS,
                    help="grids of the Canto-L (HNL) / (ttbar + HNL) runs (default: the arm (i) selection's grids)")
    a = ap.parse_args(argv)
    from hnl.eval import final_select as FS
    fam = a.family
    info = BT.info_table(BT.expected_events(BT.VAL_SETS))
    ent = pd.read_csv(ENTRIES)
    ent = ent[ent.set == "val"]
    rows = []
    labels = [(f"acorn ({x})", f"acorn ({x})" + SUFFIX[fam][0]) for x in ("ttbar", "HNL", "ttbar+HNL")]
    for name, label in labels + [("Canto-L (ttbar)", SUFFIX[fam][1])]:
        e = ent[ent.label == label]
        if len(e) != 3:
            raise SystemExit(f"{label}: {len(e)} val entries")
        for _, r in e.iterrows():
            x = metrics(r.run, r.working_point, info)
            for q in ("eff_primary", "eff_ttbar_prompt", "eff_w_muons", "fake_rate_hnl"):
                if abs(x[q] - r[q]) > 1e-9:
                    raise SystemExit(f"{label} {r.run}: {q} {x[q]} != entry {r[q]}")
            rows.append({"model": name, "seed": int(r.run[-1]), "run": r.run, "working_point": r.working_point, **x})
    wpj = json.load(open(os.path.join(REPO, "results", "working_points_v1.json")))
    FS.matching_baseline = lambda run: os.path.join(R, "baseline_v1", "armi_s0")   # the arm (i) selection's guard

    def fs_wp(run: str, grids: str = GRIDS) -> str:
        t, _ = FS.run_rows(run, grids, wpj, wpj["reference_arm"])
        r = t[t.family == fam].iloc[0]
        return f"{fam} {str(r.config).split(',', 1)[1]} ({r.grid})"
    ref = [x for x in rows if x["model"] == "Canto-L (ttbar)" and x["seed"] == 0][0]
    if fs_wp(ref["run"]) != ref["working_point"]:
        raise SystemExit(f"run_rows gives {fs_wp(ref['run'])} for Canto-L (ttbar) s0, entry {ref['working_point']}")
    for name, d in NEW.items():
        run = os.path.join(R, "stage1_final_learned_ts", "l1tl", d)
        try:
            wp = fs_wp(run, a.new_grids)
        except (SystemExit, IndexError, FileNotFoundError, KeyError) as err:
            print(f"{name}: no {fam} row yet ({err})")
            continue
        rows.append({"model": name, "seed": 0, "run": run, "working_point": wp, **metrics(run, wp, info)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(df[["model", "seed", "working_point", "eff_primary", "d0_100_300", "d0_300_inf_k", "eff_ttbar_prompt",
              "fake_rate_hnl"]].to_string())


if __name__ == "__main__":
    main()
