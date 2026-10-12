"""Do the classifier losses of displaced ttbar targets sit in dense jet cores? Neighbours within dR < 0.05 / 0.1
(charged particles with pT > 0.5 GeV and >= 3 hits in the same event), for found vs lost, and the same for prompt
high-pT ttbar targets (Canto-L (ttbar) s0 frozen, val)."""
import numpy as np, pandas as pd
from hnl.eval import bootstrap_tables as BT
from hnl.eval.particles import particle_table
from hnl.data.convert import read_table
R = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
d = pd.read_csv("/work/submit/jinboz1/hnl/tmp/leaderboard/ttbar_loss_particles.csv")
ev = np.unique(d.event_id)
p = particle_table(BT.expected_events({"t": (("ttbar", "val"),)}))
p["phi"] = np.arctan2(p.py, p.px)
nb = p[(p.pt > 0.5) & (p.n_hits >= 3) & (p.q != 0)][["event_id", "particle_id", "eta", "phi"]]
def neighbours(t, dr):
    out = np.zeros(len(t), int)
    g = dict(tuple(nb.groupby("event_id")))
    for i, (e, pid, eta, phi) in enumerate(zip(t.event_id, t.particle_id, t.eta, t.phi)):
        o = g.get(e)
        if o is None: continue
        dphi = np.angle(np.exp(1j * (o.phi.to_numpy() - phi)))
        r = np.hypot(o.eta.to_numpy() - eta, dphi)
        out[i] = int(((r < dr) & (o.particle_id.to_numpy() != pid)).sum())
    return out
d = d.merge(p[["event_id", "particle_id", "phi"]], on=["event_id", "particle_id"], how="left")
d["n_dr05"] = neighbours(d, 0.05)
d["lost_clf"] = d["where"] == "classifier"
d["dens"] = pd.cut(d.n_dr05, [-1, 0, 1, 3, 100]).astype(str)
d["pt_bin"] = pd.cut(d.pt, [1, 5, 1e9], right=False).astype(str)
pd.set_option("display.width", 200)
print("displaced ttbar (1-100 mm): classifier-loss rate by neighbours within dR<0.05 and pT")
print(d.pivot_table(index="dens", columns="pt_bin", values="lost_clf", aggfunc=lambda x: f"{100*x.mean():.0f}% of {len(x)}").to_string())
# prompt high-pT for comparison
pp, pe, _ = BT.val_tables(f"{R}/stage1_final_learned/l1tl/armi_s0", "mwslmerge 0.05 (trackbuild_val_lmerge)")
t = p[p.is_target.astype(bool) & (p.d0.abs() < 1)].merge(pp[["event_id", "particle_id", "dm_matched"]], on=["event_id", "particle_id"])
t = t[t.event_id.isin(ev[:400])]
t["n_dr05"] = neighbours(t, 0.05)
t["dens"] = pd.cut(t.n_dr05, [-1, 0, 1, 3, 100]).astype(str)
t["pt_bin"] = pd.cut(t.pt, [1, 5, 1e9], right=False).astype(str)
print("\nprompt ttbar (<1 mm), 400 events: efficiency by neighbours within dR<0.05 and pT")
print(t.pivot_table(index="dens", columns="pt_bin", values="dm_matched", aggfunc=lambda x: f"{100*x.mean():.0f}% of {len(x)}").to_string())
