"""Metric-learning radius scan and graph-stage ceiling on val (brief §5e, §5f; docs/acorn_pitfalls.md p03).

For a trained MetricLearning checkpoint and the staged val feature store:
  * per event, the embedding (acorn's own dataset preprocessing and apply_embedding) and EXACT pair distances (cdist,
    no FRNN cap), so for every r on a grid: true-edge efficiency (all npz chain edges; target particles; HNL-daughter
    and W-muon targets; HNL targets per |d0| bin), graph size in unique unordered pairs, and the per-node neighbour
    counts n_i, giving the fraction of nodes where a knn cap K would bind (n_i >= K) and max n_i;
  * for a few r values, the perfect-classifier connected-components ceiling (chain truth and same-particle truth).
The selected working point is the smallest grid r with >= 99 % efficiency on all true edges, and the smallest K on
a grid that never binds on val (K > max n_i), falling back to < 5 % binding.

    python -m hnl.graphs.ml_scan --ckpt CKPT --input-dir LOCAL/fs_val --out OUT [--ceiling-r 0.05,0.1,0.2]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.data.acorn_store import parse_event_id
from hnl.graphs import ceiling as C
from hnl.truth.report import D0_BINS, RV_BINS

R_GRID = np.round(np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.3, 0.02), np.arange(0.3, 1.01, 0.1)]), 3)
K_GRID = [50, 100, 200, 300, 500, 1000, 2000]


def particle_info(events: list[str]) -> pd.DataFrame:
    """category, is_target, d0, rv per (event_id, particle_id) for the given acorn event ids."""
    from hnl.data.convert import read_table
    from hnl.graphs.window_study import truth_particles
    by = {}
    for eid in events:
        s, i = parse_event_id(eid)
        by.setdefault(s, []).append(i)
    out = []
    root = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived", "events", "v1")
    for s, idx in by.items():
        p = read_table(root, "particles", s, events=idx, columns=["event", "particle_id", "category", "is_target"])
        t = truth_particles(s)
        p = p.merge(t[t.event.isin(idx)], on=["event", "particle_id"], how="left")
        p["sample"] = s
        out.append(p)
    return pd.concat(out, ignore_index=True)


def main(argv=None):
    import torch
    from acorn.stages.graph_construction import MetricLearning
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input-dir", required=True, help="feature store with the val events in valset/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ceiling-r", default="0.05,0.1,0.15,0.2")
    ap.add_argument("--max-events", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda")
    model = MetricLearning.load_from_checkpoint(a.ckpt, map_location="cpu")
    n_val = len([f for f in os.listdir(os.path.join(a.input_dir, "valset")) if f.endswith(".pyg.gz")])
    if a.max_events:
        n_val = min(n_val, a.max_events)
    over = {"input_dir": a.input_dir, "stage_dir": os.path.join(a.out, "_unused"), "data_split": [0, n_val, 0],
            "hard_cuts": {}, "undirected": True, "num_workers": [2, 2, 2]}
    model._hparams = add_variable_name_prefix_in_config({**copy.deepcopy(dict(model._hparams)), **over})
    model.setup(stage="predict")
    model = model.to(dev).eval()
    loader = model.val_dataloader()
    r_ceil = [float(x) for x in a.ceiling_r.split(",")]

    eff_rows, size_rows, node_rows, ceil_parts = [], [], [], []
    eids = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(dev)
            eid = batch.event_id[0] if isinstance(batch.event_id, (list, tuple)) else batch.event_id
            eid = str(eid[0] if isinstance(eid, (list, tuple)) else eid)
            eids.append(eid)
            emb = model.apply_embedding(batch)
            n = emb.shape[0]
            te = batch.track_edges
            d_true = torch.linalg.vector_norm(emb[te[0]] - emb[te[1]], dim=1)
            D = torch.cdist(emb, emb)
            iu = torch.triu_indices(n, n, 1, device=dev)
            dpairs = torch.sort(D[iu[0], iu[1]]).values
            rg = torch.tensor(R_GRID, device=dev, dtype=D.dtype)
            n_pairs = torch.searchsorted(dpairs, rg, right=False)                 # pairs with d < r
            counts = (D.unsqueeze(0) < rg.view(-1, 1, 1)).sum(dim=2) - 1          # (R, N) neighbours within r
            pid = batch.hit_particle_id
            tpid = pid[te[0]].cpu().numpy()
            eff_rows.append(pd.DataFrame({"event_id": eid, "particle_id": tpid,
                                          "d_true": d_true.cpu().numpy()}))
            size_rows.append(pd.DataFrame({"event_id": eid, "r": R_GRID, "n_pairs": n_pairs.cpu().numpy(),
                                           "n_nodes": n, "max_ni": counts.max(dim=1).values.cpu().numpy(),
                                           **{f"bind_{k}": (counts >= k).float().mean(dim=1).cpu().numpy()
                                              for k in K_GRID}}))
            pid_np = pid.cpu().numpy()
            te_np = te.cpu().numpy()
            Dn = D.cpu().numpy()
            for r in r_ceil:
                ii, jj = np.nonzero(np.triu(Dn < r, 1))
                _, part, _ = C.evaluate_graph(n, pid_np, te_np, np.stack([ii, jj]))
                part = part.reset_index()[["particle_id", "dm_matched", "dm_matched_samepid"]]
                part["event_id"], part["r"] = eid, r
                ceil_parts.append(part)
    print(f"[ml_scan] {len(eids)} events embedded and scanned in {time.time() - t0:.0f}s", flush=True)

    info = particle_info(eids)
    info["event_id"] = [f"{ {'ttbar': 1, 'hnl10_100': 2, 'hnl15_100': 3, 'hnl15_200': 4}[s] }{e:08d}"
                        for s, e in zip(info["sample"], info["event"])]
    info["d0_bin"] = pd.cut(info.d0.abs(), D0_BINS, right=False)
    info["rv_bin"] = pd.cut(info.rv, RV_BINS, right=False)
    e = pd.concat(eff_rows, ignore_index=True).merge(info, on=["event_id", "particle_id"], how="left")
    sizes = pd.concat(size_rows, ignore_index=True)
    pops = {"all": e, "targets": e[e.is_target == True], "hnl_targets": e[(e.category == "hnl") & (e.is_target == True)],
            "w_targets": e[(e.category == "w") & (e.is_target == True)],
            "other_targets": e[(e.category == "other") & (e.is_target == True)]}
    curve = pd.DataFrame({"r": R_GRID})
    for name, df in pops.items():
        d = np.sort(df.d_true.to_numpy())
        curve[f"eff_{name}"] = np.searchsorted(d, R_GRID, side="left") / max(len(d), 1)
    agg = sizes.groupby("r").agg(pairs_per_event=("n_pairs", "mean"), max_ni=("max_ni", "max"),
                                 **{f"bind_{k}": (f"bind_{k}", "mean") for k in K_GRID}).reset_index()
    curve = curve.merge(agg, on="r")
    hnl = pops["hnl_targets"]
    d0_curve = pd.DataFrame({"r": R_GRID})
    for b, g in hnl.groupby("d0_bin", observed=True):
        d = np.sort(g.d_true.to_numpy())
        d0_curve[f"hnl_d0_{b.left:g}_{b.right:g}"] = np.searchsorted(d, R_GRID, side="left") / max(len(d), 1)
    curve.to_csv(os.path.join(a.out, "radius_curve.csv"), index=False)
    d0_curve.to_csv(os.path.join(a.out, "radius_curve_hnl_d0.csv"), index=False)

    ok = curve[curve.eff_all >= 0.99]
    r_star = float(ok.r.min()) if len(ok) else None
    ok_t = curve[curve.eff_targets >= 0.99]
    r_star_targets = float(ok_t.r.min()) if len(ok_t) else None
    sel = {"r_star_all_edges": r_star, "r_star_target_edges": r_star_targets}
    for key, r in list(sel.items()):
        if r is None:
            continue
        row = curve[curve.r == r].iloc[0]
        k_never = [k for k in K_GRID if k > row.max_ni]
        k_5pct = [k for k in K_GRID if row[f"bind_{k}"] < 0.05]
        sel[key.replace("r_star", "knn") + "_never_binds"] = k_never[0] if k_never else None
        sel[key.replace("r_star", "knn") + "_binds_lt5pct"] = k_5pct[0] if k_5pct else None
        sel[key + "_pairs_per_event"] = float(row.pairs_per_event)

    cp = pd.concat(ceil_parts, ignore_index=True).merge(info, on=["event_id", "particle_id"], how="left")
    t = cp[cp.is_target == True]
    ceil_cat = t.groupby(["r", "category"]).agg(eff=("dm_matched", "mean"), eff_samepid=("dm_matched_samepid", "mean"),
                                                 n=("dm_matched", "size")).reset_index()
    h = t[t.category == "hnl"]
    ceil_d0 = h.groupby(["r", "d0_bin"], observed=True).agg(eff=("dm_matched", "mean"),
                                                          eff_samepid=("dm_matched_samepid", "mean"),
                                                          n=("dm_matched", "size")).reset_index()
    ceil_rv = h.groupby(["r", "rv_bin"], observed=True).agg(eff=("dm_matched", "mean"),
                                                          eff_samepid=("dm_matched_samepid", "mean"),
                                                          n=("dm_matched", "size")).reset_index()
    ceil_cat.to_csv(os.path.join(a.out, "ceiling_by_category.csv"), index=False)
    ceil_d0.astype({"d0_bin": str}).to_csv(os.path.join(a.out, "ceiling_hnl_by_d0.csv"), index=False)
    ceil_rv.astype({"rv_bin": str}).to_csv(os.path.join(a.out, "ceiling_hnl_by_rv.csv"), index=False)
    meta = {"ckpt": a.ckpt, "n_events": len(eids), "selection": sel, "elapsed_s": round(time.time() - t0, 1),
            "slurm_job": os.environ.get("SLURM_JOB_ID")}
    with open(os.path.join(a.out, "scan.json"), "w") as f:
        json.dump(meta, f, indent=1)
    pd.set_option("display.width", 250)
    print(curve.round(4).to_string())
    print(ceil_cat.round(4).to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
