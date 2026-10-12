"""True-edge efficiency per layer pair (brief §5e), val only.

Sources: physics-window graphs for chosen configs (calibration as in window_study: the first 1000 train events of each
labeled sample), and saved metric-learning graphs (an extracted ml_graphs/valset.tar). Layer id = volume * 100 +
layer; a true edge (npz chain, unique unordered pair) belongs to the layer pair (lower id, higher id) of its hits.
Each source is split by sample and by the category of the edge's particle (hnl / w / other; targets only or all).

    python -m hnl.graphs.layer_pairs --out OUT [--windows "d0=100,D=inf,s=1;..."] [--ml-graphs DIR --ml-label NAME]
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from hnl.data.splits import read_split
from hnl.eval.particles import CODE_SAMPLE, split_event_id
from hnl.graphs import ceiling as C
from hnl.graphs import window_study as WS
from hnl.graphs import windows as W


def _edge_rows(sample, e, h, te, graph, parts):
    n = len(h)
    lid = W.layer_id(h.volume.to_numpy(), h.layer.to_numpy())
    pid = h.pid.to_numpy()
    t = C.unordered_unique(te)
    g = C.unordered_unique(graph)
    in_graph = np.isin(t[0].astype(np.int64) * n + t[1], g[0].astype(np.int64) * n + g[1])
    la, lb = lid[t[0]], lid[t[1]]
    p = parts.reindex(pid[t[0]])
    return pd.DataFrame({"sample": sample, "lay_lo": np.minimum(la, lb), "lay_hi": np.maximum(la, lb),
                         "category": p.category.to_numpy(), "is_target": p.is_target.fillna(False).to_numpy(bool),
                         "in_graph": in_graph})


def _aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = []
    for tgt in (False, True):
        d = df[df.is_target] if tgt else df
        g = d.groupby(["sample", "category", "lay_lo", "lay_hi"]).in_graph.agg(["size", "sum"])
        g = g.rename(columns={"size": "n_true", "sum": "n_kept"}).reset_index()
        g["targets_only"] = tgt
        out.append(g)
    r = pd.concat(out, ignore_index=True)
    r["source"] = label
    return r


def windows(configs, cal, max_events=None) -> pd.DataFrame:
    res = []
    for sample in ("hnl10_100", "hnl15_200", "ttbar"):
        ev = read_split(sample, "val")[:max_events]
        hits, edges, parts = WS.load_events(sample, ev)
        eg = edges.groupby("event")
        for cfg in configs:
            rows = []
            for e, h in hits.groupby("event"):
                h = h.sort_values("row")
                ed = eg.get_group(e)
                te = np.stack([ed.src.to_numpy(), ed.dst.to_numpy()]).astype(np.int64)
                lid = W.layer_id(h.volume.to_numpy(), h.layer.to_numpy())
                g = W.build_graph(h.r.to_numpy(), h.phi.to_numpy(), h.z.to_numpy(), lid, cal, *cfg)
                pe = parts[parts.event == e].set_index("particle_id")
                rows.append(_edge_rows(sample, e, h, te, g, pe))
            res.append(_aggregate(pd.concat(rows, ignore_index=True), f"window d0={cfg[0]:g},D={cfg[1]},s={cfg[2]:g}"))
    return pd.concat(res, ignore_index=True)


def ml_graphs(graph_dir: str, label: str) -> pd.DataFrame:
    from acorn.utils.loading_utils import load_pyg
    from hnl.data.convert import read_table
    paths = sorted(glob.glob(os.path.join(graph_dir, "*.pyg.gz")))
    eids = np.array([int(os.path.basename(p).split("event")[-1].split(".")[0]) for p in paths])
    codes, idx = split_event_id(eids)
    rows = []
    for c in np.unique(codes):
        sample = CODE_SAMPLE[int(c)]
        sel = codes == c
        root = os.path.join(WS.DERIVED, "events", "v1")
        hits = read_table(root, "hits", sample, events=idx[sel], columns=["event", "row", "volume", "layer", "pid"])
        parts = read_table(root, "particles", sample, events=idx[sel], columns=["event", "particle_id", "category",
                                                                                 "is_target"])
        hg, pg = hits.groupby("event"), parts.groupby("event")
        for p, e in zip(np.array(paths)[sel], idx[sel]):
            g = load_pyg(p[:-3], weights_only=False)
            h = hg.get_group(e).sort_values("row")
            assert len(h) == len(g.hit_particle_id) and (h.pid.to_numpy() == g.hit_particle_id.numpy()).all()
            rows.append(_edge_rows(sample, e, h, g.track_edges.numpy(), g.edge_index.numpy(),
                                   pg.get_group(e).set_index("particle_id")))
    return _aggregate(pd.concat(rows, ignore_index=True), label)


def parse_configs(spec: str):
    out = []
    for item in spec.split(";"):
        kv = dict(x.split("=") for x in item.split(","))
        D = kv["D"]
        out.append((float(kv["d0"]), D if D == "prompt" else float(D), float(kv["s"])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="")
    ap.add_argument("--calib-events", type=int, default=1000)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--ml-graphs", default="")
    ap.add_argument("--ml-label", default="ml")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.windows:
        cal = WS.calibration(a.calib_events)
        r = windows(parse_configs(a.windows), cal, a.max_events)
        r.to_parquet(os.path.join(a.out, "layer_pairs_windows.parquet"), index=False)
        print(f"[layer_pairs] windows done {time.time() - t0:.0f}s", flush=True)
    if a.ml_graphs:
        r = ml_graphs(a.ml_graphs, a.ml_label)
        r.to_parquet(os.path.join(a.out, f"layer_pairs_{a.ml_label}.parquet"), index=False)
        print(f"[layer_pairs] ml graphs done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
