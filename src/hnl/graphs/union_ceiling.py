"""Graph-stage ceiling of metric-learning graphs united with physics-window graphs (brief §6A, val only).

For a run's saved metric-learning val graphs (an extracted ml_graphs/valset.tar) and window configs (d0_max, D_max,
slack; calibrated as in hnl.graphs.window_study on train events): per event, the union of the ML edges and the window
edges; per particle the perfect-classifier DM after CC (chain truth) and the same-particle variant; per event the graph
sizes (ML only, window only, union).

    python -m hnl.graphs.union_ceiling --graphs DIR --out OUT [--windows "d0=100,D=inf,s=1;..."] [--max-events N]
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from hnl.eval.particles import CODE_SAMPLE, split_event_id
from hnl.graphs import ceiling as C
from hnl.graphs import window_study as WS
from hnl.graphs import windows as W
from hnl.graphs.layer_pairs import parse_configs


def _chunk(args):
    from acorn.utils.loading_utils import load_pyg
    from hnl.data.convert import read_table
    paths, cal, configs = args
    eids = np.array([int(os.path.basename(p).split("event")[-1].split(".")[0]) for p in paths])
    codes, idx = split_event_id(eids)
    pp, pe = [], []
    for c in np.unique(codes):
        sample = CODE_SAMPLE[int(c)]
        sel = codes == c
        hits = read_table(os.path.join(WS.DERIVED, "events", "v1"), "hits", sample, events=idx[sel],
                          columns=["event", "row", "r", "phi", "z", "volume", "layer", "pid"])
        hg = hits.groupby("event")
        for p, e, eid in zip(np.array(paths)[sel], idx[sel], eids[sel]):
            g = load_pyg(p[:-3], weights_only=False)
            h = hg.get_group(e).sort_values("row")
            n = len(h)
            pid = h.pid.to_numpy()
            assert (pid == g.hit_particle_id.numpy()).all()
            te = g.track_edges.numpy()
            ml = C.unordered_unique(g.edge_index.numpy())
            lid = W.layer_id(h.volume.to_numpy(), h.layer.to_numpy())
            for cfg in [None] + configs:
                if cfg is None:
                    graph, name, n_win = ml, "ml", 0
                else:
                    win = W.build_graph(h.r.to_numpy(), h.phi.to_numpy(), h.z.to_numpy(), lid, cal, *cfg)
                    graph = C.unordered_unique(np.concatenate([ml, win], axis=1))
                    name, n_win = f"ml+window d0={cfg[0]:g},D={cfg[1]},s={cfg[2]:g}", win.shape[1]
                st, part, _ = C.evaluate_graph(n, pid, te, graph)
                part = part.reset_index()[["particle_id", "dm_matched", "dm_matched_samepid"]]
                part["event_id"], part["source"] = eid, name
                pp.append(part)
                pe.append({"event_id": eid, "source": name, "n_graph": st["n_graph"], "n_ml": ml.shape[1],
                           "n_window": n_win, "n_true": st["n_true"], "n_true_in_graph": st["n_true_in_graph"]})
    return pd.concat(pp, ignore_index=True), pd.DataFrame(pe)


def calibration_from_study(study_dir: str) -> W.WindowCalibration:
    pairs = pd.read_csv(os.path.join(study_dir, "calibration_pairs.csv"), index_col=[0, 1])
    layers = pd.read_csv(os.path.join(study_dir, "calibration_layers.csv"), index_col=0)
    return W.WindowCalibration(pairs, layers)


def summarize(out_dir: str) -> pd.DataFrame:
    """Per source: pairs per HNL event and the ceilings (chain truth / same particle) of the preregistered
    populations on the HNL val events; written to summary.csv."""
    from hnl.eval.baseline_val import populations
    from hnl.eval.particles import particle_table
    pp = pd.read_parquet(os.path.join(out_dir, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(out_dir, "per_event.parquet"))
    info = particle_table(pp.event_id.unique())
    pops = populations(info)
    flags = pd.DataFrame({"event_id": info.event_id, "particle_id": info.particle_id,
                          **{k: pops[k] for k in ("primary", "hnl_targets", "w_muons", "ttbar_targets")}})
    t = pp.merge(flags, on=["event_id", "particle_id"], how="inner")
    rows = []
    hnl_ev = np.isin(split_event_id(pe.event_id.to_numpy())[0], (2, 4))
    for src, g in t.groupby("source"):
        r = {"source": src, "pairs_hnl": pe[hnl_ev & (pe.source == src)].n_graph.mean(),
             "pairs_ttbar": pe[~hnl_ev & (pe.source == src)].n_graph.mean()}
        for k in ("primary", "hnl_targets", "w_muons", "ttbar_targets"):
            r[f"ceil_{k}"] = g.dm_matched[g[k]].mean()
            r[f"ceil_samepid_{k}"] = g.dm_matched_samepid[g[k]].mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return out


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="d0=30,D=inf,s=1;d0=100,D=inf,s=1;d0=300,D=inf,s=1")
    ap.add_argument("--study", default=os.path.join(WS.DERIVED, "graph_ceiling", "windows_v1"))
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--summarize-only", action="store_true", help="only summarise an existing --out directory")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.summarize_only:
        print(summarize(a.out).round(4).to_string())
        return
    t0 = time.time()
    cal = calibration_from_study(a.study)
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))[:a.max_events]
    chunks = [(paths[k:k + 25], cal, parse_configs(a.windows)) for k in range(0, len(paths), 25)]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_chunk, chunks))
    pp = pd.concat([r[0] for r in res], ignore_index=True)
    pe = pd.concat([r[1] for r in res], ignore_index=True)
    pp.to_parquet(os.path.join(a.out, "per_particle.parquet"), index=False)
    pe.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    print(f"[union] {len(paths)} events in {time.time() - t0:.0f}s")
    print(summarize(a.out).round(4).to_string())


if __name__ == "__main__":
    main()
