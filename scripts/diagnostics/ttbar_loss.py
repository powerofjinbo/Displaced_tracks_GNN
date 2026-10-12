"""ttbar targets with 1 <= |d0| < 100 mm, Canto-L (ttbar) s0 frozen: where they are lost (graph ceiling, classifier
ceiling at s0 = 0.2, builder), by pT; plus the production-vertex radius from the truth table (val only)."""
import os, sys, tarfile, tempfile, shutil, concurrent.futures as cf
import numpy as np, pandas as pd
from hnl.eval import bootstrap_tables as BT
from hnl.eval.particles import particle_table
from hnl.eval.oblique_val import classify
R = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
RUN = f"{R}/stage1_final_learned/l1tl/armi_s0"
WP = "mwslmerge 0.05 (trackbuild_val_lmerge)"
ev = BT.expected_events(BT.VAL_SETS)
p = particle_table(ev)
tt = p[(BT.split_event_id(p.event_id.to_numpy())[0] == BT.TTBAR_CODE) & p.is_target.astype(bool)].copy()
tt["ad0"] = tt.d0.abs()
pop = tt[(tt.ad0 >= 1) & (tt.ad0 < 100)].copy()
pp, pe, d = BT.val_tables(RUN, WP)
pop = pop.merge(pp[["event_id", "particle_id", "dm_matched"]], on=["event_id", "particle_id"], how="left")
by_ev = {int(e): g.particle_id.to_numpy(np.int64) for e, g in pop.groupby("event_id")}
tmp = tempfile.mkdtemp(dir=os.environ.get("LOCAL", "/tmp"))
want = {f"event{e}.pyg.gz": e for e in by_ev}
got = {}
with tarfile.open(os.path.join(RUN, "gnn_scores", "valset.tar")) as tar:
    for m in tar:
        nm = os.path.basename(m.name)
        if nm in want:
            m.name = nm; tar.extract(m, tmp, filter="data"); got[want[nm]] = os.path.join(tmp, nm)
def work(items):
    out = []
    for f, ids in items:
        out += classify(f, np.asarray(ids, np.int64), 0.2)
    return out
items = [(got[e], by_ev[e].tolist()) for e in sorted(got)]
W = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
with cf.ProcessPoolExecutor(W) as ex:
    res = list(ex.map(work, [items[k::W] for k in range(W)]))
shutil.rmtree(tmp, ignore_errors=True)
cl = pd.DataFrame([r for rr in res for r in rr])
pop = pop.merge(cl[["event_id", "particle_id", "stage", "n_true_chain", "n_chain_in_graph", "n_chain_above_cut"]], on=["event_id", "particle_id"], how="left")
pop["where"] = np.where(pop.dm_matched.astype(bool), "found", np.where(pop.stage == "pass", "builder", pop.stage))
pop["pt_bin"] = pd.cut(pop.pt, [1, 2, 5, 10, 1e9], right=False).astype(str)
pd.set_option("display.width", 250)
t = pd.crosstab(pop.pt_bin, pop["where"], margins=True)
print(t.to_string())
print((100 * pd.crosstab(pop.pt_bin, pop["where"], normalize="index")).round(1).to_string())
# HNL comparison: same classes for HNL daughters 1-100 mm from the oblique reference file
ob = pd.read_csv("/work/submit/jinboz1/hnl/hnl-displaced/results/oblique_val_ref_v1_particles.csv")
ob = ob[(ob.label == "Canto-L (i) s0") & (ob.d0.abs() >= 1) & (ob.d0.abs() < 100)]
print("HNL daughters 1-100 mm (primary population), stages:", ob.stage.value_counts().to_dict(), len(ob))
pop.to_csv("/work/submit/jinboz1/hnl/tmp/leaderboard/ttbar_loss_particles.csv", index=False)
