import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, mplhep as hep
hep.style.use("ROOT")
OUT = "docs/feedback/response_andre/"
LAB = "ACTS generic detector simulation, no pileup"
def label(ax, extra=""):
    ax.text(0.0, 1.01, LAB, transform=ax.transAxes, fontsize=12, va="bottom")
    ax.text(1.0, 1.01, extra, transform=ax.transAxes, fontsize=12, va="bottom", ha="right")
d = pd.read_csv("results/strong_baselines_v3.csv")
def get(s, short):
    x = d[(d.set == s) & (d.short == short)]
    if not len(x): return (np.nan, np.nan)
    r = x.iloc[0]; e = r.primary_sigma_total if s == "test" and pd.notna(r.primary_sigma_total) else r.eff_primary_sd
    return 100*r.eff_primary, 100*e
builders = ["connected\ncomponents", "acorn's\nCCandWalk", "walkthrough\n+ merging", "MWS-layer\n+ merging", "two-pass MWS\n+ merging", "MWS-layer +\nlearned merging"]
keys = {"acorn (ttbar)": ["acorn (ttbar) CC", "(ttbar) CCandWalk", "(ttbar) walk+M", "(ttbar) MWS+M", "(ttbar) 2pass+M", "(ttbar) lmerge"],
        "acorn (HNL)": ["acorn (HNL) CC*", "(HNL) CCandWalk", "(HNL) walk+M", "(HNL) MWS+M", "(HNL) 2pass+M", "(HNL) lmerge"],
        "Canto-L": [None, None, "Canto-L walk+M", "Canto-L MWS+M", "Canto-L 2pass+M", "Canto-L"]}
extra = {("val", "Canto-L", 0): (98.12, 0.15), ("test", "Canto-L", 3): (99.10, 0.18)}  # CC at the best cut <= FR* (results/per_sample_v1.md; 2026-10-07: was 96.79, the preregistered cut rule); MWS+M test: strong_baselines_v1
cols = {"acorn (ttbar)": "0.55", "acorn (HNL)": "#1f4e9c", "Canto-L": "#923002"}
fig, axs = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
for ax, s in zip(axs, ("val", "test")):
    for j, (m, ks) in enumerate(keys.items()):
        xs, ys, es = [], [], []
        for i, k in enumerate(ks):
            v, e = extra.get((s, m, i), get(s, k) if k else (np.nan, np.nan))
            if np.isfinite(v): xs.append(i + (j-1)*0.22); ys.append(v); es.append(e)
        ax.errorbar(xs, ys, yerr=es, fmt="o", color=cols[m], ms=9, capsize=0, label=m if s == "val" else None)
        for x, y in zip(xs, ys): ax.text(x, y + 0.35, f"{y:.1f}", ha="center", fontsize=9, color=cols[m])
    ax.set_xticks(range(6)); ax.set_xticklabels(builders, fontsize=11); ax.set_ylim(88, 101)
    ax.set_title("Validation (bars: seed s.d.)" if s == "val" else "Test, post-freeze (bars: seed + event bootstrap)", fontsize=14, pad=26)
    label(ax, "3 seeds each")
axs[0].set_ylabel("Primary efficiency [%]", fontsize=16); axs[0].legend(fontsize=13, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "A_builders.pdf"); fig.savefig(OUT + "A_builders.png", dpi=130)

# B: probe, only acorn and Canto (plus the capacity-matched acorn network)
p = pd.read_csv("results/probe_outer_v3.csv")
rows = [("acorn (ttbar) s0", "acorn (ttbar), seed 0"), ("acorn (ttbar) s1", "acorn (ttbar), seed 1"), ("acorn (ttbar) s2", "acorn (ttbar), seed 2"),
        ("capacity-matched IGNN (hidden 147), full budget", "acorn (ttbar), capacity-matched"), ("acorn (HNL) s0", "acorn (HNL), seed 0"),
        ("Canto-L s0", "Canto-L, seed 0"), ("Canto-L s1", "Canto-L, seed 1"), ("Canto-L s2", "Canto-L, seed 2")]
fig, ax = plt.subplots(figsize=(13, 6.5)); w = 0.27
cols = {"inner": "#923002", "middle": "#d9a441", "outer": "#1f4e9c"}
names = {"inner": "4 innermost hits removed", "middle": "4 middle hits removed (piece after the gap)", "outer": "4 outermost hits removed"}
for j, m in enumerate(("inner", "middle", "outer")):
    sub = "outer piece" if m == "middle" else "all"
    q = p[(p["mode"] == m) & (p.subset == sub)].set_index("label")
    ax.barh(np.arange(len(rows)) + (j-1)*w, [q.loc[k, "dT_pts"] for k, _ in rows], height=w,
            xerr=[q.loc[k, "dT_boot_sd"] for k, _ in rows], color=cols[m], label=names[m], error_kw=dict(lw=1))
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, n in rows], fontsize=12); ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(r"$\Delta T$: change of the true-edge rate of the remaining edges [points]", fontsize=15)
ax.legend(fontsize=12, loc="lower left"); label(ax, "validation, 300 HNL events")
fig.tight_layout(); fig.savefig(OUT + "B_probe.pdf"); fig.savefig(OUT + "B_probe.png", dpi=130)

# D: V0 daughters, Canto-L against the validation-best acorn configurations
v = pd.read_csv("results/second_population_v2.csv")
v = v[(v.kind == "efficiency") & (v.population == "v0") & (v.split == "test") & (v.event_set == "hnl_joint_test")]
bins = ["r_0_50", "r_50_100", "r_100_200", "r_200_300", "r_300_inf"]; bl = ["0-50", "50-100", "100-200", "200-300", ">=300"]
fig, ax = plt.subplots(figsize=(11, 6.3))
for j, (m, lab, col) in enumerate([("Canto-L", "Canto-L", "#923002"), ("(HNL) LM", "acorn (HNL), validation-best builder", "#1f4e9c"),
                                   ("(ttbar) LM", "acorn (ttbar), validation-best builder", "0.5")]):
    y = [100*v[(v.quantity == b) & (v.short == m)].value.iloc[0] for b in bins]
    e = [100*v[(v.quantity == b) & (v.short == m)].combined.iloc[0] for b in bins]
    ax.errorbar(np.arange(5) + (j-1)*0.08, y, yerr=e, fmt="o-", color=col, label=lab, capsize=0, ms=7)
ax.set_xticks(range(5)); ax.set_xticklabels(bl); ax.set_xlabel("Radius of the innermost hit of the V0 daughter [mm]", fontsize=16)
ax.set_ylabel("Efficiency [%]", fontsize=16); ax.set_ylim(90, 101); ax.legend(fontsize=12, loc="lower left"); label(ax, "test, post-freeze")
fig.tight_layout(); fig.savefig(OUT + "D_v0.pdf"); fig.savefig(OUT + "D_v0.png", dpi=130)
print("ok")

# C: capacity-matched control, acorn vs Canto naming
def pick(c, sub, b):
    r = c[c.label.str.contains(sub, regex=False) & c.label.str.contains(b, regex=False) & ~c.label.str.contains("best of", regex=False)]
    return (100*r.eff_primary.iloc[0], 100*(r.eff_primary_sd.iloc[0] if pd.notna(r.eff_primary_sd.iloc[0]) else 0)) if len(r) else (np.nan, 0)
panels = [("results/ranking/ranking_val_capacity_screen_v1.csv", "screening budget (1 seed each)",
           [("acorn network\n263,937 par.", "IGNN (acorn arch."), ("acorn network, widened\n347,803 par.", "capacity-matched IGNN"),
            ("Canto classifier\n347,777 par.", "L1 + trunc. = Canto")]),
          ("results/ranking/ranking_val_capacity_v1.csv", "full budget (widened network: 1 seed; others: 3 seeds)",
           [("acorn network\n263,937 par.", "acorn IGNN (263,937"), ("acorn network, widened\n347,803 par.", "capacity-matched acorn IGNN"),
            ("Canto classifier\n347,777 par.", "Canto-B classifier (347,777")])]
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
for ax, (f, tag, models) in zip(axs, panels):
    c = pd.read_csv(f)
    for j, (b, bn, col) in enumerate([("MWS-layer + merging", "MWS-layer + merging", "#1f4e9c"), ("CC (best cut", "connected components", "0.5")]):
        vv = [pick(c, s, b) for _, s in models]
        ax.bar(np.arange(3) + (j-0.5)*0.36, [v for v, _ in vv], yerr=[e for _, e in vv], width=0.36, color=col, label=bn, error_kw=dict(lw=1))
        for i, (y, e) in enumerate(vv): ax.text(i + (j-0.5)*0.36, y+e+0.25, f"{y:.1f}", ha="center", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels([m for m, _ in models], fontsize=12); ax.set_ylim(85, 100.5)
    label(ax); ax.set_title(tag, fontsize=13, pad=24)
axs[0].set_ylabel("Primary efficiency [%] (validation)", fontsize=16); axs[0].legend(fontsize=12, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "C_capacity.pdf"); fig.savefig(OUT + "C_capacity.png", dpi=130)
print("ok C")
