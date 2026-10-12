"""Training-sample study (docs/training_sample_study.md), val only: classifier losses of HNL daughters versus how
obliquely the daughter crosses the layers it starts on.

For a straight track from a production point at transverse distance |d0| from the beam line, |d0| / r_first is the sine
of the angle between the track and the radial direction at its first hit (0: radial, as for a prompt track; near 1:
almost tangent to the first layer). On the HNL training split, the ttbar-trained Canto-L's classifier losses of
far-displaced daughters were all at |d0| / r_first > 0.7 (results/far_trainsplit_v1_particles.csv: 16 of 588
particle-seeds above, 0 of 564 below). This module measures the same classes on the HNL val split for any run, so that
Canto-L trained on HNL (arm ii) or on ttbar + HNL (arm iii) can be compared with the ttbar-trained Canto-L (arm i) on
events none of them was trained on.

Population: primary-population HNL-daughter targets (|d0| >= 10 mm or r_v >= 100 mm; hnl.eval.bootstrap_tables.masks)
of the 1,999 HNL val events. Per particle, from the run's stored val scores (RUN/gnn_scores/valset.tar):
  graph       not DM-matched by connected components on the graph's chain-true edges (graph-stage ceiling);
  classifier  DM-matched by that ceiling but not by connected components on the chain-true edges with score > s0,
              s0 = hnl.trackbuild.learned_merge.FIRST_S0 = 0.2 (the edge cut of Canto-L's frozen builder);
  pass        DM-matched by both ceilings (the track builder may still lose it).
The classes are those of hnl.eval.round4_checks (check 2) without the builder step, so they do not depend on the
builder configuration chosen for a run. valmon marks the events a run's checkpoint selection saw
(hnl.train.acorn_runner.arm_events(arm, 'valmon'); arm (i) monitored on ttbar val only, so none of these events).

    python -m hnl.eval.oblique_val --run LABEL=RUN_DIR:ARM [--run ...] --out results --name oblique_val_v1 [--workers N]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap_tables as BT

HNL_SETS = {"hnl_joint_val": BT.VAL_SETS["hnl_joint_val"]}
RATIO_EDGES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0001)
D0_EDGES = (0.0, 10.0, 30.0, 100.0, 200.0, 300.0, np.inf)
STAGES = ("graph", "classifier", "pass")


def population() -> pd.DataFrame:
    from hnl.eval.particles import particle_table
    p = particle_table(BT.expected_events(HNL_SETS))
    t = p[p.is_target.astype(bool)].reset_index(drop=True)
    m = BT.masks(t[["event_id", "particle_id", "is_target", "category", "d0", "rv"]])["eff_primary"]
    f = t[m].copy()
    f["ratio"] = f.d0.abs() / f.r_first
    cols = ["event_id", "particle_id", "sample", "d0", "rv", "r_first", "ratio", "pt", "eta", "n_hits", "n_pixel_hits"]
    return f[cols].sort_values(["event_id", "particle_id"]).reset_index(drop=True)


def valmon_ids(arm: str) -> set[int]:
    if arm == "i":
        return set()
    from hnl.train.acorn_runner import arm_events
    return {BT.SAMPLE_CODE[s] * 100_000_000 + int(i) for s, i in arm_events(arm, "valmon")}


def classify(path: str, ids: np.ndarray, s0: float) -> list[dict]:
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval import matching as M
    from hnl.eval.trackbuild_val import load_scored
    from hnl.graphs import ceiling as C
    from hnl.trackbuild import builders as TB
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    n = len(pid)
    g = load_pyg(path[:-3], weights_only=False)
    te = C.unordered_unique(g.track_edges.numpy())
    code = a.astype(np.int64) * n + b
    chain = np.isin(code, te[0] * n + te[1])
    fm = M.FastMatch(pid)
    g_dm = fm.match(TB.cc(n, a[chain], b[chain], np.ones(int(chain.sum())), 0.5))["dm_matched"]
    keep = chain & (sc > s0)
    s_dm = fm.match(TB.cc(n, a[keep], b[keep], np.ones(int(keep.sum())), 0.5))["dm_matched"]
    tpid = pid[te[0]]
    ins = np.isin(te[0] * n + te[1], code)
    above = np.isin(te[0] * n + te[1], code[keep])
    out = []
    for p in ids:
        if p not in fm.particles:
            raise ValueError(f"{path}: particle {p} has no hits in the graph event")
        i = int(np.searchsorted(fm.particles, p))
        stage = "graph" if not g_dm[i] else ("classifier" if not s_dm[i] else "pass")
        tm = tpid == p
        out.append({"event_id": np.int64(eid), "particle_id": np.int64(p), "stage": stage,
                    "n_true_chain": int(tm.sum()), "n_chain_in_graph": int(ins[tm].sum()),
                    "n_chain_above_cut": int(above[tm].sum())})
    return out


def _work(task: dict) -> list[dict]:
    rows = []
    for f, ids in task["items"]:
        rows += classify(f, np.asarray(ids, dtype=np.int64), task["s0"])
    return rows


def run_one(label: str, run: str, arm: str, pop: pd.DataFrame, workers: int, tmp: str, s0: float) -> pd.DataFrame:
    tar_path = os.path.join(run, "gnn_scores", "valset.tar")
    by_ev = {int(e): g.particle_id.to_numpy(np.int64) for e, g in pop.groupby("event_id")}
    want = {f"event{e}.pyg.gz": e for e in by_ev}
    d = tempfile.mkdtemp(prefix="obl_", dir=tmp)
    try:
        got = {}
        with tarfile.open(tar_path, "r") as tar:
            for mem in tar:
                nm = os.path.basename(mem.name)
                if nm in want:
                    mem.name = nm
                    tar.extract(mem, d, filter="data")
                    got[want[nm]] = os.path.join(d, nm)
        if len(got) != len(want):
            raise SystemExit(f"{tar_path}: {len(got)} of {len(want)} HNL val events with a target")
        items = [(got[e], by_ev[e].tolist()) for e in sorted(got)]
        chunks = [items[k::workers] for k in range(workers)]
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_work, [{"items": c, "s0": s0} for c in chunks if c]))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.insert(0, "arm", arm)
    df.insert(0, "run", run)
    df.insert(0, "label", label)
    vm = valmon_ids(arm)
    df["valmon"] = df.event_id.isin(vm)
    return df


def summarize(parts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rb = pd.cut(parts.ratio, RATIO_EDGES, right=False)
    db = pd.cut(parts.d0.abs(), D0_EDGES, right=False)
    for subset, sel in (("all", np.ones(len(parts), bool)), ("rest", ~parts.valmon.to_numpy(bool))):
        for (lab, kind, b), g in parts[sel].groupby([parts.label[sel], pd.Series("ratio", index=parts.index)[sel],
                                                      rb[sel].astype(str)], sort=False):
            rows.append(_row(subset, lab, kind, b, g))
        for (lab, b), g in parts[sel].groupby([parts.label[sel], db[sel].astype(str)], sort=False):
            rows.append(_row(subset, lab, "abs_d0_mm", b, g))
        for lab, g in parts[sel].groupby(parts.label[sel], sort=False):
            rows.append(_row(subset, lab, "all", "all", g))
            hi = g[(g.ratio >= 0.7) & (g.d0.abs() >= 100.0)]
            rows.append(_row(subset, lab, "ratio>=0.7 and |d0|>=100", "all", hi))
    return pd.DataFrame(rows)


def _row(subset, lab, kind, b, g) -> dict:
    c = g.stage.value_counts()
    ok = int(c.get("classifier", 0) + c.get("pass", 0))
    return {"subset": subset, "label": lab, "binning": kind, "bin": b, "n": len(g),
            **{k: int(c.get(k, 0)) for k in STAGES},
            "classifier_loss_given_graph": (c.get("classifier", 0) / ok) if ok else np.nan}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR:ARM")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="oblique_val_v1")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--tmp", default=os.environ.get("LOCAL", tempfile.gettempdir()))
    a = ap.parse_args(argv)
    from hnl.trackbuild.learned_merge import FIRST_S0
    t0 = time.time()
    pop = population()
    parts = []
    for spec in a.run:
        lab, rest = spec.split("=", 1)
        run, arm = rest.rsplit(":", 1)
        df = run_one(lab, run, arm, pop, a.workers, a.tmp, float(FIRST_S0))
        if len(df) != len(pop):
            raise SystemExit(f"{lab}: {len(df)} classified, {len(pop)} in the population")
        parts.append(df.merge(pop, on=["event_id", "particle_id"], how="left"))
        print(f"{lab}: {len(df)} particles, {time.time() - t0:.0f} s", flush=True)
    parts = pd.concat(parts, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    parts.to_csv(os.path.join(a.out, f"{a.name}_particles.csv"), index=False)
    summ = summarize(parts)
    summ.to_csv(os.path.join(a.out, f"{a.name}_summary.csv"), index=False)
    meta = {"started_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", "local"), "host": os.uname().nodename, "s0": float(FIRST_S0),
            "population": int(len(pop)), "events": int(pop.event_id.nunique()), "runs": a.run,
            "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, f"{a.name}_meta.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(summ[(summ.subset == "all") & summ.binning.isin(["ratio", "all", "ratio>=0.7 and |d0|>=100"])].to_string())
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
