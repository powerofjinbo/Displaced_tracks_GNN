"""Mirror and endpoint checks of the truncation probe (supervisor feedback round 1, point 4a; validation only).

hnl.eval.truncation_probe removes the m innermost hits of prompt particles and measures the TPR of the same outer
chain-true edges at a fixed score cut (dT = TPR(m=4) - TPR(m=0), paper Eq. dT). A drop can mean that the classifier
learned where true tracks START (a start-radius prior) or, more generally, that it distrusts edges near any track
ENDPOINT. This module repeats the probe with the same particles, events, graph edit, checkpoint and score cut in three
modes:

  inner   remove the m innermost hits (R = sqrt(r^2 + z^2) order): the original probe (its numbers are reproduced);
  outer   remove the m outermost hits (the mirror): the track keeps its start at the beam line but ends early;
  middle  remove m hits around the middle of the track (R ranks [n//2 - m//2, n//2 - m//2 + m)): a gap that leaves an
          inner piece with a new end and an outer piece with a new start.

m = 0 (the untouched graph, shared by the modes), 2 and 4. Removed hits go with every graph edge touching them (no edge
is added: hnl.eval.truncation_probe.truncate); every other hit and edge of the event is unchanged, and the modified
graph is re-scored by the run's GNN (best checkpoint of the finished training). For every chain-true edge (track_edges)
of the chosen particles whose two hits survive, edges.parquet stores the R ranks of its hits counted from the inner end
(r_lo <= r_hi; ranks of the untouched particle), the particle's number of hits n, whether the unordered pair is in the
graph and its maximum score over the stored directions. The read-out (hnl.eval.probe_outer_report) selects the
measured edges, which exist at every depth (M = largest depth): inner r_lo >= M (as truncation_probe); outer
n - 1 - r_hi >= M; middle both hits outside the depth-M window.

Particles (as truncation_probe): truth |d0| < 1 mm, >= 8 hits, pT >= 2 GeV, at most 5 per event, in a fixed list of HNL
val events (--events: JSON {"events": [...]}; default: the first --n-events HNL members of the graph tar, the original
probe's choice). Union graphs (Canto-H / Canto-L) carry the same attributes as the metric-learning graphs and are
edited the same way. Per-event graphs live only in --work (node-local); only edges.parquet and meta.json are written to
--out (each finished mode and depth is also kept in OUT/partial until then, so a job that hits its time limit resumes
where it stopped when it is submitted again).

    python -m hnl.eval.probe_outer --run RUN --graphs VALSET_TAR --out OUT_DIR [--events EVENTS_JSON]
        [--modes inner,outer,middle] [--depths 0,2,4] [--n-events 300] [--work DIR] [--accelerator cpu]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch

from hnl.eval.truncation_probe import _save, truncate

MODES = ("inner", "outer", "middle")


def removed_ranks(mode: str, n: int, k: int) -> np.ndarray:
    """R ranks (0 = innermost) of the hits removed from a particle with n hits at depth k."""
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k > n:
        raise ValueError(f"depth {k} > {n} hits")
    if mode == "inner":
        return np.arange(k)
    if mode == "outer":
        return np.arange(n - k, n)
    if mode == "middle":
        lo = n // 2 - k // 2
        return np.arange(lo, lo + k)
    raise ValueError(mode)


def first_hnl_events(tar_path: str, n: int) -> list[int]:
    """The first n HNL members (event2* / event4*) of a graph tar in tar order: truncation_probe's event choice."""
    out = []
    with tarfile.open(tar_path) as t:
        for mem in t:
            base = os.path.basename(mem.name)
            if base.endswith(".pyg.gz") and base[5] in "24":
                out.append(int(base[5:-7]))
                if len(out) >= n:
                    break
    return out


def extract_events(tar_path: str, events: list[int], dest: str) -> dict[int, str]:
    """Extract event<ID>.pyg.gz of the listed events (by name) into dest; returns {event id: path}."""
    want = {f"event{e}.pyg.gz": e for e in events}
    got = {}
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path) as t:
        for mem in t:
            base = os.path.basename(mem.name)
            if base in want:
                mem.name = base
                t.extract(mem, dest, filter="data")
                got[want[base]] = os.path.join(dest, base)
                if len(got) == len(want):
                    break
    missing = sorted(set(events) - set(got))
    if missing:
        raise RuntimeError(f"{len(missing)} events not in {tar_path}: {missing[:5]}")
    return got


def best_checkpoint(run: str) -> str:
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(run, "gnn", "run_*.json"))]
    return max([r for r in runs if r.get("state") == "done"], key=lambda r: r["end_utc"])["best_ckpt"]


def chosen_particles(event_ids) -> dict[int, set]:
    """truncation_probe's selection: prompt (|d0| < 1 mm), >= 8 hits, pT >= 2 GeV, the first 5 per event."""
    from hnl.eval.particles import particle_table
    info = particle_table(np.array(sorted(event_ids)))
    sel = info[(info.d0.abs() < 1.0) & (info.n_hits >= 8) & (info.pt >= 2.0)]
    sel = sel.groupby("event_id").head(5)
    return {int(e): set(g_.particle_id.astype(np.int64)) for e, g_ in sel.groupby("event_id")}


def edge_rows(s, chosen: set, rank: np.ndarray, nhit: np.ndarray) -> pd.DataFrame:
    """Chain-true edges of the chosen particles in a scored graph s (nodes = kept hits; rank / nhit per kept node):
    R ranks of the two hits, n, in_graph and the maximum score of the unordered pair (as truncation_probe)."""
    pid = s.hit_particle_id.numpy()
    ei = s.edge_index.numpy()
    sc = s.scores.numpy()
    te = s.track_edges.numpy()
    n = len(pid)
    code = np.minimum(ei[0], ei[1]).astype(np.int64) * n + np.maximum(ei[0], ei[1])
    best = pd.Series(sc).groupby(code).max()
    tcode = np.unique(np.minimum(te[0], te[1]).astype(np.int64) * n + np.maximum(te[0], te[1]))
    u, v = tcode // n, tcode % n
    keep = np.isin(pid[u], np.fromiter(chosen, dtype=np.int64, count=len(chosen))) & (pid[v] == pid[u])
    u, v, tcode = u[keep], v[keep], tcode[keep]
    if (rank[u] < 0).any() or (rank[v] < 0).any() or (nhit[u] != nhit[v]).any():
        raise RuntimeError("chain edge of a chosen particle without a rank")
    score = best.reindex(tcode).to_numpy()
    return pd.DataFrame({"particle_id": pid[u].astype(np.int64),
                         "r_lo": np.minimum(rank[u], rank[v]).astype(np.int16),
                         "r_hi": np.maximum(rank[u], rank[v]).astype(np.int16),
                         "n_hits": nhit[u].astype(np.int16),
                         "in_graph": ~np.isnan(score), "score": score.astype(np.float32)})


def main(argv=None):
    from acorn.utils.loading_utils import load_pyg
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run whose GNN scores the graphs (RUN/gnn/run_*.json)")
    ap.add_argument("--graphs", required=True, help="val graph tar the GNN was trained for (ml_graphs or union)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", default=None, help='JSON {"events": [...]}; default: first --n-events HNL members')
    ap.add_argument("--n-events", type=int, default=300)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--depths", default="0,2,4")
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--accelerator", default="cpu")
    a = ap.parse_args(argv)
    t0 = time.time()
    modes = [m for m in a.modes.split(",") if m]
    if set(modes) - set(MODES):
        raise ValueError(f"unknown modes {set(modes) - set(MODES)}")
    depths = sorted({int(x) for x in a.depths.split(",")})
    if 0 not in depths:
        raise ValueError("depth 0 (the untouched graph) is required")
    events = (json.load(open(a.events))["events"] if a.events else first_hnl_events(a.graphs, a.n_events))
    events = [int(e) for e in events]
    work = os.path.join(a.work, "probe_outer")
    os.makedirs(a.out, exist_ok=True)
    src = extract_events(a.graphs, events, os.path.join(work, "src"))
    graphs = {e: load_pyg(p[:-3], weights_only=False) for e, p in src.items()}
    chosen = chosen_particles(list(graphs))
    ckpt = best_checkpoint(a.run)
    # per event: R order of every chosen particle's hits (as truncation_probe: argsort of R over the particle's hits)
    prep = {}
    for e, g in graphs.items():
        pid = g.hit_particle_id.numpy()
        R = torch.sqrt(g.r.double() ** 2 + g.z.double() ** 2).numpy()
        rank = np.full(len(pid), -1, dtype=np.int64)
        nhit = np.full(len(pid), -1, dtype=np.int64)
        orders = {}
        for p in chosen.get(e, ()):
            idx = np.flatnonzero(pid == p)
            order = idx[np.argsort(R[idx])]
            rank[order] = np.arange(len(order))
            nhit[order] = len(order)
            orders[p] = order
        prep[e] = (rank, nhit, orders)
    jobs = [("none", 0)] + [(m, k) for m in modes for k in depths if k > 0]
    frames, timing = [], {}
    # resume: every finished (mode, depth) is kept in OUT/partial until edges.parquet is written
    partial = os.path.join(a.out, "partial")
    os.makedirs(partial, exist_ok=True)
    fp = {"run": a.run, "graphs": a.graphs, "ckpt": ckpt, "events": events,
          "n_particles": int(sum(map(len, chosen.values())))}
    fpf = os.path.join(partial, "fingerprint.json")
    if os.path.exists(fpf):
        if json.load(open(fpf)) != fp:
            raise RuntimeError(f"{partial} belongs to another configuration")
    else:
        with open(fpf, "w") as f:
            json.dump(fp, f)
    for mode, k in jobs:
        tag = f"{mode}_{k}"
        part = os.path.join(partial, f"{tag}.parquet")
        if os.path.exists(part):
            frames.append(pd.read_parquet(part))
            timing[tag] = "resumed"
            print(json.dumps({"mode": mode, "depth": k, "resumed": part}), flush=True)
            continue
        d_in = os.path.join(work, f"in_{tag}", "valset")
        d_out = os.path.join(work, f"out_{tag}")
        os.makedirs(d_in, exist_ok=True)
        kept, these = {}, []
        n_removed = 0
        for e, g in graphs.items():
            rank, nhit, orders = prep[e]
            remove = np.zeros(len(rank), dtype=bool)
            for p, order in orders.items():
                remove[order[removed_ranks(mode, len(order), k)]] = True
            n_removed += int(remove.sum())
            kept[e] = (rank[~remove], nhit[~remove])
            _save(truncate(g, remove), os.path.join(d_in, f"event{e}.pyg.gz"))
        t1 = time.time()
        subprocess.run([sys.executable, "-m", "hnl.train.acorn_runner", "infer", "--stage", "gnn", "--ckpt", ckpt,
                        "--input-dir", os.path.dirname(d_in), "--out-dir", d_out, "--accelerator", a.accelerator],
                       check=True, stdout=subprocess.DEVNULL)
        timing[tag] = round(time.time() - t1, 1)
        for e in graphs:
            s = load_pyg(os.path.join(d_out, "valset", f"event{e}.pyg"), weights_only=False)
            rk, nk = kept[e]
            if s.num_nodes != len(rk):
                raise RuntimeError(f"event {e}: scored graph has {s.num_nodes} nodes, expected {len(rk)}")
            df = edge_rows(s, chosen.get(e, set()), rk, nk)
            df.insert(0, "event_id", np.int64(e))
            df.insert(0, "depth", np.int8(k))
            df.insert(0, "mode", mode)
            these.append(df)
        these = pd.concat(these, ignore_index=True)
        these.to_parquet(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        frames.append(these)
        shutil.rmtree(os.path.dirname(d_in))
        shutil.rmtree(d_out)
        print(json.dumps({"mode": mode, "depth": k, "removed_hits": n_removed, "infer_s": timing[tag]}), flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(a.out, "edges.parquet"), index=False)
    shutil.rmtree(partial)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        git = ""
    meta = {"run": a.run, "graphs": a.graphs, "ckpt": ckpt, "events": a.events, "n_events": len(graphs),
            "n_particles": int(sum(map(len, chosen.values()))), "modes": modes, "depths": depths,
            "infer_seconds": timing, "elapsed_s": round(time.time() - t0, 1), "host": socket.gethostname(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "git": git, "accelerator": a.accelerator}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
