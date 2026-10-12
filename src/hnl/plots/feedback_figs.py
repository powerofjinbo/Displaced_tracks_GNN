import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, mplhep as hep
hep.style.use("ROOT")
OUT = "figures/feedback/"
LAB = "ACTS generic detector simulation, no pileup"
def label(ax, extra):
    ax.text(0.0, 1.01, LAB, transform=ax.transAxes, fontsize=13, va="bottom")
    ax.text(1.0, 1.01, extra, transform=ax.transAxes, fontsize=13, va="bottom", ha="right")

# A: strongest baselines (v3)
d = pd.read_csv("results/strong_baselines_v3.csv")
rows = [("Canto-L", "Canto-L (frozen, learned merging)"),
        ("(HNL) val-best", "acorn (HNL), val-best = learned merging"),
        ("(HNL) walk+M", "acorn (HNL) + walkthrough + merging"),
        ("(HNL) MWS+M", "acorn (HNL) + MWS-layer + merging"),
        ("(HNL) CCandWalk", "acorn (HNL) + acorn's CCandWalk"),
        ("acorn (HNL) CC*", "acorn (HNL) + CC, best cut"),
        ("(ttbar+HNL) val-best", "acorn (ttbar+HNL), val-best"),
        ("(ttbar) val-best", "acorn (ttbar), val-best"),
        ("(ttbar) CCandWalk", "acorn (ttbar) + acorn's CCandWalk"),
        ("acorn (ttbar) CC", "acorn (ttbar) + CC")]
fig, axs = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
for ax, s in zip(axs, ("val", "test")):
    x = d[d.set == s].set_index("short")
    for i, (k, name) in enumerate(rows):
        if k not in x.index: continue
        r = x.loc[k]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        err = r.primary_sigma_total if s == "test" and pd.notna(r.primary_sigma_total) else r.eff_primary_sd
        c = "#923002" if k == "Canto-L" else ("#1f4e9c" if "val-best" in k else "0.35")
        ax.errorbar(100*r.eff_primary, i, xerr=100*err, fmt="o", color=c, capsize=0, ms=8)
        ax.text(100*r.eff_primary, i-0.32 if i else i+0.42, f"{100*r.eff_primary:.2f}", ha="center", va="center", fontsize=11, color=c)
    ax.set_xlim(89, 100.8)
    ax.set_title("Validation (bars: seed s.d.)" if s == "val" else "Test, post-freeze (bars: seed + event bootstrap)", fontsize=14, pad=26)
fig.supxlabel("Primary efficiency [%] at track fake rate <= 2.94 %", fontsize=16)
axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([n for _, n in rows], fontsize=12); axs[0].invert_yaxis()
label(axs[0], ""); label(axs[1], "3 seeds each")
fig.tight_layout(); fig.savefig(OUT+"A_strongest_baselines_v3.png", dpi=150); fig.savefig(OUT+"A_strongest_baselines_v3.pdf")

# B: outer / middle probe (v3)
p = pd.read_csv("results/probe_outer_v3.csv")
order = ["acorn (ttbar) s0","acorn (ttbar) s1","acorn (ttbar) s2","equal-budget IGNN","capacity-matched IGNN (hidden 147)",
         "capacity-matched IGNN (hidden 147), full budget","L1 without truncation","acorn (HNL) s0","IGNN + truncation",
         "L1 + truncation (Canto classifier preview)","Canto-L s0","Canto-L s1","Canto-L s2"]
order = [o for o in order if o in set(p.label)]
fig, ax = plt.subplots(figsize=(14, 8))
w = 0.27; cols = {"inner": "#923002", "middle": "#d9a441", "outer": "#1f4e9c"}
names = {"inner": "4 innermost hits removed", "middle": "4 middle hits removed (piece after the gap)", "outer": "4 outermost hits removed"}
for j, m in enumerate(("inner", "middle", "outer")):
    sub = "outer piece" if m == "middle" else "all"
    q = p[(p["mode"] == m) & (p.subset == sub)].set_index("label")
    v = [q.loc[o, "dT_pts"] for o in order]; e = [q.loc[o, "dT_boot_sd"] for o in order]
    ax.barh(np.arange(len(order)) + (j-1)*w, v, height=w, xerr=e, color=cols[m], label=names[m], error_kw=dict(lw=1))
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=12); ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(r"$\Delta T$: change of the true-edge rate of the remaining edges [points] (300 HNL val events, cut 0.35)", fontsize=15)
ax.legend(fontsize=12, loc="lower left"); label(ax, "validation")
fig.tight_layout(); fig.savefig(OUT+"B_probe_inner_outer_middle_v3.png", dpi=150); fig.savefig(OUT+"B_probe_inner_outer_middle_v3.pdf")

# C: capacity-matched control, screening budget (1 seed) and full budget (IGNN, Canto-B: 3 seeds; capacity-matched: 1 seed)
def pick(c, sub, b):
    r = c[c.label.str.contains(sub, regex=False) & c.label.str.contains(b, regex=False) & ~c.label.str.contains("best of", regex=False)]
    return (100*r.eff_primary.iloc[0], 100*(r.eff_primary_sd.iloc[0] if "eff_primary_sd" in r and pd.notna(r.eff_primary_sd.iloc[0]) else 0)) if len(r) else (np.nan, 0)
panels = [("results/ranking/ranking_val_capacity_screen_v1.csv", "screening budget, 1 seed each",
           [("IGNN (acorn)\n263,937 par.", "IGNN (acorn arch."), ("capacity-matched IGNN\n347,803 par.", "capacity-matched IGNN"),
            ("Canto classifier\n347,777 par.", "L1 + trunc. = Canto")]),
          ("results/ranking/ranking_val_capacity_v1.csv", "full budget (IGNN, Canto-B: 3 seeds; capacity-matched: 1 seed)",
           [("IGNN (acorn)\n263,937 par.", "acorn IGNN (263,937"), ("capacity-matched IGNN\n347,803 par.", "capacity-matched acorn IGNN"),
            ("Canto-B classifier\n347,777 par.", "Canto-B classifier (347,777")])]
fig, axs = plt.subplots(1, 2, figsize=(17, 6.8), sharey=True)
for ax, (f, tag, models) in zip(axs, panels):
    c = pd.read_csv(f)
    for j, (b, bn, col) in enumerate([("MWS-layer + merging", "MWS-layer + merging", "#1f4e9c"), ("CC (best cut", "connected components, best cut", "0.45")]):
        vv = [pick(c, s, b) for _, s in models]
        ax.bar(np.arange(3) + (j-0.5)*0.36, [v for v, _ in vv], yerr=[e for _, e in vv], width=0.36, color=col, label=bn, error_kw=dict(lw=1))
        for i, (y, e) in enumerate(vv): ax.text(i + (j-0.5)*0.36, y+e+0.25, f"{y:.2f}", ha="center", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels([m for m, _ in models], fontsize=12); ax.set_ylim(85, 100.5)
    label(ax, ""); ax.set_title(tag, fontsize=13, pad=24)
axs[0].set_ylabel("Primary efficiency [%] (validation)", fontsize=16); axs[0].legend(fontsize=12, loc="upper left")
fig.tight_layout(); fig.savefig(OUT+"C_capacity_control.png", dpi=150); fig.savefig(OUT+"C_capacity_control.pdf")

# D: V0 daughters (second displaced population), test
v = pd.read_csv("results/second_population_v2.csv")
v = v[(v.kind == "efficiency") & (v.population == "v0") & (v.split == "test") & (v.event_set == "hnl_joint_test")]
bins = ["r_0_50", "r_50_100", "r_100_200", "r_200_300", "r_300_inf"]; bl = ["0-50", "50-100", "100-200", "200-300", ">=300"]
mods = [("Canto-L", "#923002"), ("(HNL) LM", "#1f4e9c"), ("(HNL) MWS+M", "#6a8fd0"), ("(ttbar) LM", "0.4"), ("(HNL) CC*", "0.7")]
fig, ax = plt.subplots(figsize=(11, 6.5))
for j, (m, col) in enumerate(mods):
    y = []; e = []
    for b in bins:
        r = v[(v.quantity == b) & (v.short == m)]
        y.append(100*r.value.iloc[0] if len(r) else np.nan); e.append(100*r.combined.iloc[0] if len(r) else np.nan)
    ax.errorbar(np.arange(5) + (j-2)*0.08, y, yerr=e, fmt="o-", color=col, label=m.replace("(HNL)", "acorn (HNL)").replace("(ttbar)", "acorn (ttbar)").replace("LM", "MWS-layer + learned merging (validation-best)"), capsize=0, ms=6)
ax.set_xticks(range(5)); ax.set_xticklabels(bl); ax.set_xlabel("Start radius of the V0 daughter (innermost hit) [mm]", fontsize=16)
ax.set_ylabel("Efficiency [%]"); ax.set_ylim(85, 101); ax.legend(fontsize=12, loc="lower left"); label(ax, "test, post-freeze; HNL events")
fig.tight_layout(); fig.savefig(OUT+"D_v0_daughters_test.png", dpi=150); fig.savefig(OUT+"D_v0_daughters_test.pdf")
print("ok")
