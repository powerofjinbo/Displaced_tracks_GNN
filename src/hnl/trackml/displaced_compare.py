"""Displaced-target losses of Canto-L on TrackML val (hnl.trackml.displaced_diag output) against the HNL val
graph and classifier ceilings of the frozen Canto-L (results/oblique_val_ref_v1_particles.csv), by |d0| / r_first and pT,
and by barcode generation in the |d0| bins of the main project's acorn (brief section 1: generation-0 targets only).
Post-freeze, validation only; nothing is chosen from it. Output: results/trackml_displaced_diag_v1.txt.

    python -m hnl.trackml.displaced_compare
"""
import glob
import numpy as np, pandas as pd
from hnl.trackml.convert import read_table
from hnl.trackml.events import DERIVED_TP
from hnl.trackml.displaced_diag import generation, species, outcome
D = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1/diag/displaced"
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
RB = [0, 0.2, 0.4, 0.7, 1.0001]
PB = [0, 1.5, 2, 5, 1000]
DB = [1, 3, 10, 30, 100, 300, np.inf]

def load(cond):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/diag_{cond}_b*.parquet"))], ignore_index=True)
    p = read_table("particles", sorted(df.event.unique()), ["event", "particle_id", "vx", "vy", "px", "py"], root=DERIVED_TP)
    df = df.merge(p, on=["event", "particle_id"], how="left")
    df["species"] = species(df)
    df["outcome"] = outcome(df)
    df["ratio"] = (df.d0.abs() / df.r_first).clip(upper=1.0)       # as hnl.eval.oblique_val: |d0| / r_first
    df["graph_ok"] = df.lcc_graph > df.n_hits / 2
    df["inward"] = (df.vx * df.px + df.vy * df.py) < 0
    df["gen"] = np.where(generation(df.particle_id) > 0, "gen>=1", "gen0")
    return df

out = {}
for cond in ("full", "1gev"):
    df = load(cond)
    d = df[df.d0.abs() >= 10]
    out[cond] = d
    print(f"\n===== {cond}: {df.event.nunique()} val events; targets {len(df)}; |d0| >= 10 mm: {len(d)}")
    print("eff all targets %.4f | prompt (|d0|<1) %.4f | |d0|>=10 %.4f | |d0|>=100 %d/%d" % (
        df.dm.mean(), df[df.d0.abs() < 1].dm.mean(), d.dm.mean(), d[d.d0.abs() >= 100].dm.sum(), (d.d0.abs() >= 100).sum()))
    g = d.groupby("category")
    print(pd.DataFrame({"n": g.size(), "share": g.size() / len(d), "eff": g.dm.mean(), "graph_ok": g.graph_ok.mean(),
                        "pt_med": g.pt.median(), "ratio_med": g.ratio.median(), "nhits_med": g.n_hits.median()}).to_string())
    lost = d[~d.dm]
    print("lost |d0|>=10:", len(lost), "by outcome:", lost.outcome.value_counts().to_dict(), "| category:", lost.category.value_counts().to_dict(),
          "| inward:", int(lost.inward.sum()))
    print("high-score (>=0.5) wrong edges per hit, |d0|>=10: found %.4f lost %.4f; prompt %.4f" % (
        d[d.dm].n_wrong05.sum() / d[d.dm].n_hits.sum(), lost.n_wrong05.sum() / lost.n_hits.sum(),
        df[df.d0.abs() < 1].n_wrong05.sum() / df[df.d0.abs() < 1].n_hits.sum()))
    f = d[d.d0.abs() >= 100]
    print("|d0|>=100 by category:", f.groupby("category").dm.agg(["sum", "size"]).to_dict("index"), "outcome of lost:", f[~f.dm].outcome.value_counts().to_dict())
    b = df[df.d0.abs() >= 1]
    print("\neff by |d0| [mm] (main project's bins) x generation (gen0: event generator; gen>=1: detector simulation):")
    print(b.pivot_table(index=pd.cut(b.d0.abs(), DB, right=False), columns="gen", values="dm", aggfunc=["mean", "size"], observed=True).to_string())
    print("|d0|>=10 by generation:", d.groupby("gen").dm.agg(["sum", "size"]).to_dict("index"), "| lost by generation:", lost.gen.value_counts().to_dict(),
          "| gen>=1 by category:", d[d.gen == "gen>=1"].category.value_counts().to_dict())
    print("\neff by |d0|/r_first x pT [GeV]:")
    print(d.pivot_table(index=pd.cut(d.ratio, RB, right=False), columns=pd.cut(d.pt, PB), values="dm", aggfunc=["mean", "size"], observed=True).to_string())
    print("graph-stage ok by |d0|/r_first x pT:")
    print(d.pivot_table(index=pd.cut(d.ratio, RB, right=False), columns=pd.cut(d.pt, PB), values="graph_ok", aggfunc="mean", observed=True).to_string())
    print("eff by |d0|/r_first x category:")
    print(d.pivot_table(index=pd.cut(d.ratio, RB, right=False), columns="category", values="dm", aggfunc=["mean", "size"], observed=True).to_string())

# HNL reference: frozen Canto-L (arm i, ttbar-trained) s0, HNL val primary population, graph / classifier ceilings
h = pd.read_csv("/work/submit/jinboz1/hnl/hnl-displaced/results/oblique_val_ref_v1_particles.csv")
h = h[h.label == "Canto-L (i) s0"].copy()
h["graph_ok"] = h.stage != "graph"
h["pass"] = h.stage == "pass"
print(f"\n===== HNL val, Canto-L (i) s0: {len(h)} primary-population daughters; pT median {h.pt.median():.2f} GeV; "
      f"share with pT <= 1.5 GeV {np.mean(h.pt <= 1.5):.3f}; share with |d0|/r_first >= 0.2 {np.mean(h.ratio >= 0.2):.3f}")
print("graph-stage ok by |d0|/r_first x pT:")
print(h.pivot_table(index=pd.cut(h.ratio.clip(upper=1), RB, right=False), columns=pd.cut(h.pt, PB), values="graph_ok", aggfunc=["mean", "size"], observed=True).to_string())
print("graph + classifier ok by |d0|/r_first x pT:")
print(h.pivot_table(index=pd.cut(h.ratio.clip(upper=1), RB, right=False), columns=pd.cut(h.pt, PB), values="pass", aggfunc="mean", observed=True).to_string())
t = out["full"]
print(f"\nTrackML |d0|>=10 (full): pT median {t.pt.median():.2f}; share pT <= 1.5 {np.mean(t.pt <= 1.5):.3f}; share ratio >= 0.2 {np.mean(t.ratio >= 0.2):.3f}")
