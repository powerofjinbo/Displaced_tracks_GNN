"""Truth content of the ttbar training events for the secondary-particle upweighting test
(docs/secondary_upweight_plan.md; supervisor feedback, 2026-10-14, item 4). Training events only; no val or test event
is read.

Particle level (all 7,999 ttbar training events, derived/events/v1 particles joined with the truth-fit |d0| of
derived/truth_lookup/v1/ttbar.parquet): particles with at least two hits (they have chain edges), their chain edges
(n_hits - 1), and targets, per event and per class of truth |d0|, of generation (barcode gen 0: event generator; >= 1:
detector simulation), of |PDG| and of parent |PDG|; the far, oblique particles of N:TS14 (pT > 1.5 GeV,
|d0| / r_first > 0.7) by r_first. The table has no production point (vx..vt are the end state), so displacement is
the truth-fit |d0|, the quantity that defines the displaced populations of the paper.

Graph level (the first N graphs of the arm (i) seed 0 training tar, the graphs every screening-budget classifier is
trained on): true edges (y = 1) by the |d0| class of their particle, and the share of the positive loss weight that
the |d0| >= D0 class would carry for a set of weights w (acorn's natural weighting: every edge weight 1).

    python -m hnl.eval.secondary_truth --out results --name secondary_upweight_truth_v1 [--graphs 500] [--workers N]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd

HNL_DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
DERIVED = os.path.join(HNL_DATA, "derived")
TRAIN_TAR = os.path.join(HNL_DATA, "runs", "baseline_v1", "armi_s0", "ml_graphs", "trainset.tar")
D0_EDGES = (0.0, 1.0, 10.0, 30.0, 100.0, 300.0, np.inf)
RF_EDGES = (0.0, 50.0, 100.0, 200.0, 300.0, np.inf)
WEIGHTS = (2.0, 3.0, 5.0, 10.0, 20.0)
D0_CUTS = (10.0, 30.0, 100.0)


def particles() -> pd.DataFrame:
    from hnl.data.convert import read_table
    from hnl.data.splits import read_split
    ev = read_split("ttbar", "train")
    p = read_table(os.path.join(DERIVED, "events", "v1"), "particles", "ttbar", events=ev,
                   columns=["event", "particle_id", "pdg", "pt", "eta", "process", "parent_pdg", "gen", "vsec",
                            "n_hits", "r_first", "is_target"])
    t = pd.read_parquet(os.path.join(DERIVED, "truth_lookup", "v1", "ttbar.parquet"),
                        columns=["event", "particle_id", "absd0"])
    p = p.merge(t, on=["event", "particle_id"], how="left")
    p = p[p.n_hits >= 2].reset_index(drop=True)
    p["chain_edges"] = p.n_hits - 1
    p["ratio"] = p.absd0 / p.r_first
    return p, len(ev)


def _summ(g: pd.DataFrame, n_ev: int) -> dict:
    return {"particles_per_event": len(g) / n_ev, "chain_edges_per_event": g.chain_edges.sum() / n_ev,
            "targets_per_event": g.is_target.sum() / n_ev, "particles": len(g), "targets": int(g.is_target.sum())}


def particle_tables(p: pd.DataFrame, n_ev: int) -> dict[str, pd.DataFrame]:
    out = {}
    tot_edges = p.chain_edges.sum()
    d0b = pd.cut(p.absd0, D0_EDGES, right=False).astype(str).where(p.absd0.notna(), "no fit")
    rows = []
    for b, g in p.groupby(d0b, sort=False):
        rows.append({"abs_d0_mm": b, **_summ(g, n_ev), "share_of_chain_edges": g.chain_edges.sum() / tot_edges,
                     "gen0_share": float((g.gen == 0).mean()), "electron_share": float((g.pdg.abs() == 11).mean())})
    out["by_d0"] = pd.DataFrame(rows).sort_values("abs_d0_mm").reset_index(drop=True)
    rows = []
    for cut in D0_CUTS:
        g = p[p.absd0 >= cut]
        for gen_lab, h in (("gen 0", g[g.gen == 0]), ("gen >= 1", g[g.gen >= 1])):
            rows.append({"abs_d0_min_mm": cut, "generation": gen_lab, **_summ(h, n_ev),
                         "share_of_chain_edges": h.chain_edges.sum() / tot_edges})
        for key, col in (("pdg", g.pdg.abs()), ("parent_pdg", g.parent_pdg.abs())):
            vc = g.groupby(col).chain_edges.agg(["size", "sum"]).sort_values("sum", ascending=False).head(8)
            for k, r in vc.iterrows():
                rows.append({"abs_d0_min_mm": cut, "generation": f"{key} {int(k)}", "particles": int(r["size"]),
                             "particles_per_event": r["size"] / n_ev, "chain_edges_per_event": r["sum"] / n_ev,
                             "share_of_chain_edges": r["sum"] / tot_edges})
    out["displaced_composition"] = pd.DataFrame(rows)
    ob = p[(p.pt > 1.5) & (p.ratio > 0.7)]
    rows = []
    for b, g in ob.groupby(pd.cut(ob.r_first, RF_EDGES, right=False).astype(str), sort=False):
        rows.append({"r_first_mm": b, "particles": len(g), "per_event": len(g) / n_ev,
                     "abs_d0_ge_10": int((g.absd0 >= 10).sum()), "electrons": int((g.pdg.abs() == 11).sum()),
                     "gen_ge_1": int((g.gen >= 1).sum()), "targets": int(g.is_target.sum())})
    out["oblique_pt15"] = pd.DataFrame(rows).sort_values("r_first_mm").reset_index(drop=True)
    return out


def _graph_work(members: list[tuple[str, bytes]]) -> list[dict]:
    import torch
    from hnl.models.truth_lookup import TruthLookup
    lk = TruthLookup(samples=("ttbar",))
    rows = []
    for name, raw in members:
        g = torch.load(io.BytesIO(gzip.decompress(raw)), weights_only=False)
        y = g.y.bool().numpy()
        a = g.edge_index[0].numpy()
        _, d0, _, _ = lk.lookup(int(g.event_id), g.hit_particle_id.numpy())
        d0a = np.nan_to_num(d0[a], nan=0.0)
        row = {"graph": name, "edges": int(len(y)), "true": int(y.sum())}
        for cut in D0_CUTS:
            row[f"true_d0_ge_{int(cut)}"] = int((y & (d0a >= cut)).sum())
        rows.append(row)
    return rows


def graph_table(n: int, workers: int) -> pd.DataFrame:
    members = []
    with tarfile.open(TRAIN_TAR, "r") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".pyg.gz"):
                members.append((os.path.basename(m.name), tar.extractfile(m).read()))
                if len(members) >= n:
                    break
    chunks = [members[k::workers] for k in range(workers)]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_graph_work, [c for c in chunks if c]))
    return pd.DataFrame([r for rr in res for r in rr]).sort_values("graph").reset_index(drop=True)


def weight_shares(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    T = gt.true.sum()
    for cut in D0_CUTS:
        d = gt[f"true_d0_ge_{int(cut)}"].sum()
        f = d / T
        rows.append({"abs_d0_min_mm": cut, "w": 1.0, "true_edge_share": f, "positive_weight_share": f,
                     "true_edges_per_graph": d / len(gt)})
        for w in WEIGHTS:
            rows.append({"abs_d0_min_mm": cut, "w": w, "true_edge_share": f,
                         "positive_weight_share": w * f / (w * f + 1 - f), "true_edges_per_graph": d / len(gt)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="secondary_upweight_truth_v1")
    ap.add_argument("--graphs", type=int, default=500)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    t0 = time.time()
    p, n_ev = particles()
    tabs = particle_tables(p, n_ev)
    gt = graph_table(a.graphs, a.workers)
    tabs["graph_shares"] = weight_shares(gt)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    with open(os.path.join(a.out, f"{a.name}.txt"), "w") as fh:
        fh.write(f"ttbar training events: {n_ev}; particles with >= 2 hits: {len(p)}; training graphs read: {len(gt)}"
                 f" (true edges per graph {gt.true.mean():.1f}, edges per graph {gt.edges.mean():.0f})\n")
        for k, t in tabs.items():
            fh.write(f"\n== {k}\n{t.to_string(index=False, float_format=lambda x: f'{x:.5g}')}\n")
    for k, t in tabs.items():
        t.to_csv(os.path.join(a.out, f"{a.name}_{k}.csv"), index=False)
    meta = {"slurm_job": os.environ.get("SLURM_JOB_ID", "local"), "host": os.uname().nodename, "events": n_ev,
            "graphs": int(len(gt)), "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, f"{a.name}_meta.json"), "w"), indent=1)
    print(open(os.path.join(a.out, f"{a.name}.txt")).read())
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
