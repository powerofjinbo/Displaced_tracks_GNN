"""Graph-stage ceiling of the physics-window graphs (brief §5e), val splits only.

    python -m hnl.graphs.window_study --out <dir> [--calib-events 1000] [--val-events N] [--configs default]

Calibration: true edges of the first --calib-events events of each TRAIN split (ttbar, hnl10_100, hnl15_200).
Evaluation: the val splits; per config (d0_max, D_max, slack_scale): unique pairs per event, edge efficiency (all
true edges; per category of the particle), per layer pair, and the perfect-classifier CC + DM efficiency per
category and per |d0| / r_v bin (docs/definitions.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.data.convert import read_table
from hnl.data.splits import read_split
from hnl.graphs import ceiling as C
from hnl.graphs import windows as W
from hnl.truth.report import D0_BINS, RV_BINS, truth_d0

DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
TRAIN_SAMPLES = ("ttbar", "hnl10_100", "hnl15_200")
DEFAULT_CONFIGS = [
    # (d0_max [mm], D_max [mm] or "prompt" or inf, slack_scale); slacks calibrated at 5 robust sigmas
    (0.0, "prompt", 1.0), (0.0, "prompt", 2.0), (0.0, 300.0, 1.0), (0.0, np.inf, 1.0),
    (10.0, 300.0, 1.0), (10.0, np.inf, 1.0), (30.0, np.inf, 1.0), (100.0, np.inf, 1.0),
    (300.0, np.inf, 1.0), (100.0, 1000.0, 1.0), (300.0, 1000.0, 1.0), (300.0, np.inf, 2.0),
]


def truth_particles(sample: str) -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(DERIVED, "truth_fits", "v1", f"particles_{sample}.parquet"),
                        columns=["event", "particle_id", "pdg", "fix_d0", "free_d0"])
    p["d0"] = truth_d0(p)
    vpath = os.path.join(DERIVED, "truth_fits", "v1", f"vertices_{sample}.parquet")
    if os.path.exists(vpath):
        v = pd.read_parquet(vpath)
        v = v[v.ok]
        rv = pd.concat([v[["event", "pid_a", "rv"]].rename(columns={"pid_a": "particle_id"}),
                        v[["event", "pid_b", "rv"]].rename(columns={"pid_b": "particle_id"})])
        p = p.merge(rv, on=["event", "particle_id"], how="left")
    else:
        p["rv"] = np.nan
    return p[["event", "particle_id", "d0", "rv"]]


def load_events(sample: str, events: np.ndarray):
    hits = read_table(os.path.join(DERIVED, "events", "v1"), "hits", sample, events=events,
                      columns=["event", "row", "r", "phi", "z", "volume", "layer", "pid"])
    edges = read_table(os.path.join(DERIVED, "events", "v1"), "edges", sample, events=events)
    parts = read_table(os.path.join(DERIVED, "events", "v1"), "particles", sample, events=events,
                       columns=["event", "particle_id", "category", "is_target", "pt"])
    tp = truth_particles(sample)
    parts = parts.merge(tp[tp.event.isin(events)], on=["event", "particle_id"], how="left")
    return hits, edges, parts


def calibration(n_events: int) -> W.WindowCalibration:
    ef, hf = [], []
    for s in TRAIN_SAMPLES:
        ev = read_split(s, "train")[:n_events]
        hits, edges, parts = load_events(s, ev)
        prompt_ids = parts[parts.is_target & (parts.d0.abs() < 1.0)].set_index(["event", "particle_id"]).index
        for e, h in hits.groupby("event"):
            h = h.sort_values("row")
            lid = W.layer_id(h.volume.to_numpy(), h.layer.to_numpy())
            ed = edges[edges.event == e]
            a, b = ed.src.to_numpy(), ed.dst.to_numpy()
            pid = h.pid.to_numpy()
            prompt = pd.MultiIndex.from_arrays([np.full(len(a), e), pid[a]]).isin(prompt_ids)
            ef.append(pd.DataFrame({"lid_a": lid[a], "lid_b": lid[b], "r_a": h.r.to_numpy()[a], "phi_a": h.phi.to_numpy()[a],
                                    "z_a": h.z.to_numpy()[a], "r_b": h.r.to_numpy()[b], "phi_b": h.phi.to_numpy()[b],
                                    "z_b": h.z.to_numpy()[b], "prompt": prompt}))
            hf.append(pd.DataFrame({"lid": lid, "r": h.r.to_numpy(), "z": h.z.to_numpy()}))
    return W.calibrate(ef, hf)


def _evaluate_chunk(args):
    sample, events, cal, configs = args
    hits, edges, parts = load_events(sample, events)
    rows, pp = [], []
    eg = edges.groupby("event")
    for e, h in hits.groupby("event"):
        h = h.sort_values("row")
        n = len(h)
        lid = W.layer_id(h.volume.to_numpy(), h.layer.to_numpy())
        ed = eg.get_group(e)
        te = np.stack([ed.src.to_numpy(), ed.dst.to_numpy()]).astype(np.int64)
        pid = h.pid.to_numpy()
        for (d0m, dm, sc) in configs:
            g = W.build_graph(h.r.to_numpy(), h.phi.to_numpy(), h.z.to_numpy(), lid, cal, d0m, dm, sc)
            st, part, lp = C.evaluate_graph(n, pid, te, g, layer_id=lid)
            cfg = f"d0={d0m:g},D={dm},s={sc:g}"
            rows.append({"sample": sample, "event": e, "config": cfg, **st})
            part = part.reset_index()
            part["event"], part["config"] = e, cfg
            pp.append(part[["event", "particle_id", "dm_matched", "dm_matched_samepid", "config"]])
    per_event = pd.DataFrame(rows)
    per_particle = pd.concat(pp, ignore_index=True).merge(parts, on=["event", "particle_id"], how="left")
    per_particle["sample"] = sample
    return per_event, per_particle


def evaluate(sample: str, events: np.ndarray, cal: W.WindowCalibration, configs, workers: int = 1,
             chunk: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    import concurrent.futures as cf
    chunks = [(sample, events[k:k + chunk], cal, configs) for k in range(0, len(events), chunk)]
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_evaluate_chunk, chunks))
    else:
        res = [_evaluate_chunk(c) for c in chunks]
    return pd.concat([r[0] for r in res], ignore_index=True), pd.concat([r[1] for r in res], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-events", type=int, default=1000)
    ap.add_argument("--val-events", type=int, default=None)
    ap.add_argument("--samples", default="hnl10_100,hnl15_200,ttbar")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cal = calibration(a.calib_events)
    cal.pairs.to_csv(os.path.join(a.out, "calibration_pairs.csv"))
    cal.layers.to_csv(os.path.join(a.out, "calibration_layers.csv"))
    print(f"[windows] calibrated {len(cal.pairs)} layer pairs in {time.time() - t0:.0f}s", flush=True)
    evs, pps = [], []
    for s in a.samples.split(","):
        ev = read_split(s, "val")
        if a.val_events:
            ev = ev[:a.val_events]
        pe, pp = evaluate(s, ev, cal, DEFAULT_CONFIGS, workers=a.workers)
        evs.append(pe)
        pps.append(pp)
        print(f"[windows] {s}: {len(ev)} val events, {time.time() - t0:.0f}s", flush=True)
    pe = pd.concat(evs, ignore_index=True)
    pp = pd.concat(pps, ignore_index=True)
    pe.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    pp.to_parquet(os.path.join(a.out, "per_particle.parquet"), index=False)
    # summary: graph size, edge efficiency, DM ceiling per category / bins
    size = pe.groupby(["sample", "config"]).agg(pairs_per_event=("n_graph", "mean"),
                                                 edge_eff=("n_true_in_graph", "sum"), n_true=("n_true", "sum"))
    size["edge_eff"] = size.edge_eff / size.n_true
    t = pp[pp.is_target].copy()
    t["d0_bin"] = pd.cut(t.d0.abs(), D0_BINS, right=False)
    t["rv_bin"] = pd.cut(t.rv, RV_BINS, right=False)
    cat = t.groupby(["sample", "config", "category"]).dm_matched.mean().unstack("category")
    cat_same = t.groupby(["sample", "config", "category"]).dm_matched_samepid.mean().unstack("category")
    tgt_edges = None
    d0 = t[t.category == "hnl"].groupby(["config", "d0_bin"], observed=True).dm_matched.agg(["mean", "size"])
    rv = t[t.category == "hnl"].groupby(["config", "rv_bin"], observed=True).dm_matched.agg(["mean", "size"])
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        for name, df in (("graph size / edge efficiency", size), ("DM ceiling by category", cat),
                         ("DM ceiling by category, same-particle edges correct", cat_same),
                         ("HNL DM ceiling by |d0|", d0), ("HNL DM ceiling by r_v", rv)):
            f.write(f"== {name}\n{df.to_string()}\n\n")
    print(open(os.path.join(a.out, "summary.txt")).read())
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"calib_events": a.calib_events, "val_events": a.val_events, "configs": [list(map(str, c)) for c in DEFAULT_CONFIGS],
                   "elapsed_s": time.time() - t0, "slurm_job": os.environ.get("SLURM_JOB_ID")}, f, indent=1)


if __name__ == "__main__":
    main()
