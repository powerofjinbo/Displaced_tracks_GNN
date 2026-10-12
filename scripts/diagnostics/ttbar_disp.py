"""Why ttbar targets with 1 <= |d0| < 100 mm have ~65 % efficiency: composition and efficiency by species, pT, first-hit
radius and n_hits, for several models at their committed val working points (val only)."""
import numpy as np, pandas as pd
from hnl.eval import bootstrap_tables as BT
from hnl.eval.particles import particle_table
R = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
ent = pd.read_csv("/work/submit/jinboz1/hnl/hnl-displaced/results/strong_baselines_v3_val_entries.csv")
ent = ent[ent.set == "val"]
def wp(label, run):
    return ent[(ent.label == label) & (ent.run == run)].working_point.iloc[0]
M = [("Canto-L (ttbar) frozen", f"{R}/stage1_final_learned/l1tl/armi_s0", wp("Canto-L (frozen: learned merging)", f"{R}/stage1_final_learned/l1tl/armi_s0")),
     ("Canto-L (HNL) MWS+M 0.2", f"{R}/stage1_final_learned_ts/l1tl/armii_s0", "mwslayermerge 0.2 (trackbuild_val_ext)"),
     ("acorn (ttbar) LM", f"{R}/baseline_v1/armi_s0", wp("acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder)", f"{R}/baseline_v1/armi_s0")),
     ("acorn (HNL) LM", f"{R}/baseline_v1/armii_s0", wp("acorn (HNL) + MWS-layer + learned merging (Canto-L's builder)", f"{R}/baseline_v1/armii_s0")),
     ("acorn (ttbar) CC", f"{R}/baseline_v1/armi_s0", wp("acorn (ttbar) + CC, preregistered cut (= best cut)", f"{R}/baseline_v1/armi_s0"))]
ev = BT.expected_events(BT.VAL_SETS)
p = particle_table(ev)
tt = p[(BT.split_event_id(p.event_id.to_numpy())[0] == BT.TTBAR_CODE) & p.is_target.astype(bool)].copy()
tt["ad0"] = tt.d0.abs()
sel = tt[(tt.ad0 >= 1) & (tt.ad0 < 100)].copy()
apdg = sel.pdg.abs()
sel["species"] = np.select([apdg == 11, apdg == 13, apdg == 211, apdg == 321, apdg == 2212], ["e", "mu", "pi", "K", "p"], "other")
sel["pt_bin"] = pd.cut(sel.pt, [1, 2, 5, 10, 1e9], right=False).astype(str)
sel["rfirst_bin"] = pd.cut(sel.r_first, [0, 40, 100, 200, 400, 2000], right=False).astype(str)
sel["nhits_bin"] = pd.cut(sel.n_hits, [5, 8, 11, 15, 100], right=False).astype(str)
sel["d0_bin"] = pd.cut(sel.ad0, [1, 3, 10, 30, 100], right=False).astype(str)
print("ttbar targets per val:", len(tt), " with 1<=|d0|<100:", len(sel))
for name, run, w in M:
    pp, pe, d = BT.val_tables(run, w)
    sel = sel.drop(columns=[c for c in sel.columns if c == name], errors="ignore")
    sel = sel.merge(pp[["event_id", "particle_id", "dm_matched"]].rename(columns={"dm_matched": name}), on=["event_id", "particle_id"], how="left")
    print(name, w)
names = [m[0] for m in M]
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
for col in ["species", "pt_bin", "rfirst_bin", "nhits_bin", "d0_bin"]:
    g = sel.groupby(col)
    t = pd.DataFrame({"n": g.size(), "share_%": (100 * g.size() / len(sel)).round(1)})
    for n in names:
        t[n] = (100 * g[n].mean()).round(1)
    print("\n==", col); print(t.to_string())
# cross: species x rfirst for Canto-L (ttbar)
print("\n== species x r_first (n, Canto-L (ttbar) eff)")
print(sel.pivot_table(index="species", columns="rfirst_bin", values=names[0], aggfunc=lambda x: f"{len(x)}:{100*x.mean():.0f}").to_string())
print("\n== pions/kaons/protons with r_first < 40 mm, by pT (Canto-L (ttbar))")
h = sel[sel.species.isin(["pi", "K", "p"]) & (sel.r_first < 40)]
print(h.groupby("pt_bin")[names].mean().mul(100).round(1).assign(n=h.groupby("pt_bin").size()).to_string())
