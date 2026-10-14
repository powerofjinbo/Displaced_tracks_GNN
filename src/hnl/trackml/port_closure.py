"""Closure tests of our ports of the published evaluation codes (hnl.trackml.refports) against the authors' own
published numbers, before the ports are trusted for the comparison of Sec. VIII (supervisor's request, 2026-10-14).
Evaluation code only: no Canto-L output and no test-set event of ours is read.

eggnet-counts
    The EggNet code's efficiency denominator (target particles: pid != 0, pT >= 1 GeV, >= 3 hits, no eta cut;
    refports.eggnet.port) per event, on our throughput-phase events and on our accuracy-phase events, both without
    their test ids. Chan et al. (arXiv:2506.03415v1 p. 4) evaluate "About 467K particles" over 400 test events of
    "an average of 9K particles and 108K spacepoints"; the per-event count of the port's targets is compared with
    467K / 400, with the alternative definitions alongside (>= 1, >= 2 and >= 5 hits, |eta| < 4, and our own
    targets); the spread of 400 consecutive events (collision reuse makes neighbouring events alike) is given. Their
    events are not public (feature store under /global/cfs/cdirs/m2616), so the count is a check of the definition,
    not of the same events.

liu-truthcc
    Truth-CC on the TrackML-1GeV quickstart events released with HierarchicalGNN (portal.nersc.gov/cfs/m3443/
    ryanliu/TrackML1GeV/events.tar.gz; slurm/trackml/port_closure_fetch.sbatch), evaluated with refports.liu.port
    and compared with Liu et al. Table 3 (arXiv:2303.01640v1 p. 6), Truth-CC efficiency and fake rate at 0-50 % of
    the graph edges removed at random. Truth-CC as in the authors' notebook (Tracking-ML-Exa.TrkX f107781,
    notebooks/TrackML_ACAT/track_eval.ipynb cells 4-5): the graph edges whose two hits carry the same pid (noise
    included) are kept, connected components over the hits that touch a kept edge, eval_metrics(pt_cut=1,
    nhits_cut=5, majority_cut=0.5, primary=False); per-event values averaged over the events. Edge removal as in the
    authors' TrackMLDataset (LightningModules/TrackML_ACAT/utils.py:98-102): each edge dropped independently with
    probability f, before the truth pruning. Event split as in the authors' code: sorted files of train/val/test,
    random.seed(42) shuffle, then torch random_split [300, 10, 10] with seed 0 (HGNN_GMM.yaml train_split). The
    per-event identity of refports.liu with the authors' unmodified eval_metrics on these 340 events was shown when
    the port was written (workflow wf_9ddbb9d7-334); cupy is not in our environment, so it is not repeated here.

    python -m hnl.trackml.port_closure eggnet-counts --out DIR
    python -m hnl.trackml.port_closure liu-truthcc --events DIR --out DIR [--draws 20] [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import zlib

import numpy as np
import pandas as pd

from hnl.trackml.convert import DERIVED, read_table
from hnl.trackml.events import DERIVED_TP, SPLIT_THROUGHPUT, event_sets
from hnl.trackml.refports import eggnet as EG
from hnl.trackml.refports import liu as LIU

SPLIT_ACCURACY = os.path.join(os.path.dirname(SPLIT_THROUGHPUT), "split_v1.json")
CHAN_PARTICLES, CHAN_EVENTS = 467_000, 400          # arXiv:2506.03415v1 p. 4: "About 467K particles", 400 test events
CHAN_HITS, CHAN_NPART = 108_000, 9_000              # same page: "an average of 9K particles and 108K spacepoints"
TABLE3 = {  # Liu et al. arXiv:2303.01640v1 Table 3 (p. 6), Truth-CC, percent: removed fraction -> (Eff, FR)
    0.0: (98.72, 5.87), 0.1: (96.21, 15.53), 0.2: (92.31, 24.40),
    0.3: (85.81, 33.48), 0.4: (77.26, 42.99), 0.5: (64.81, 53.12),
}


# ------------------------------------------------------------------------------------------------ EggNet counts
def _non_test_events(phase: str) -> list[int]:
    if phase == "throughput":
        s = event_sets(SPLIT_THROUGHPUT, root=DERIVED_TP)
    else:
        s = event_sets(SPLIT_ACCURACY, root=DERIVED)
    ev = sorted(set(s["train"]) | set(s["val"]))
    if set(ev) & set(s["test"]):
        raise RuntimeError("test ids in the counted events")
    return ev


def eggnet_counts(phase: str, chunk: int = 25) -> pd.DataFrame:
    root = DERIVED_TP if phase == "throughput" else DERIVED
    events = _non_test_events(phase)
    rows = []
    for k in range(0, len(events), chunk):
        part = events[k:k + chunk]
        h = read_table("hits", part, ["event", "pid"], root)
        p = read_table("particles", part, root=root)
        for e, g in h.groupby("event", sort=True):
            pid = g.pid.to_numpy().astype(np.int64)
            pe = p[p.event == e]
            # nhits recounted from the hits (the authors' hit_particle_nhits is the particle's hit count in the event)
            u, c = np.unique(pid[pid != 0], return_counts=True)
            nh = pd.Series(c, index=u)
            pp = pe.set_index("particle_id")
            pp = pp.loc[pp.index.isin(u)].copy()
            pp["nhits"] = nh.reindex(pp.index).to_numpy()
            n_nh_diff = int((pp["nhits"].to_numpy() != pp["n_hits"].to_numpy()).sum())
            pp = pp.reset_index()
            # The port with every hit its own label: only n_particles (the denominator) is used here.
            c_eg = EG.port(pid, np.arange(len(pid)), pp[["particle_id", "pt", "nhits"]], hard_cut_pt=None)
            pt, nhit, eta = pp.pt.to_numpy(), pp.nhits.to_numpy(), pp.eta.to_numpy()
            pt32 = pt.astype(np.float32)
            rows.append({
                "phase": phase, "event": int(e), "hits": len(pid), "noise_hits": int((pid == 0).sum()),
                "particles_with_hits": len(pp), "stored_n_hits_differs": n_nh_diff,
                "eggnet_targets_port": c_eg["n_particles"],
                "pt1_nh1": int(((pt32 >= 1) & (nhit >= 1)).sum()),
                "pt1_nh2": int(((pt32 >= 1) & (nhit >= 2)).sum()),
                "pt1_nh3": int(((pt32 >= 1) & (nhit >= 3)).sum()),
                "pt1_nh5": int(((pt32 >= 1) & (nhit >= 5)).sum()),
                "pt1_nh3_eta4": int(((pt32 >= 1) & (nhit >= 3) & (np.abs(eta) < 4)).sum()),
                "pt1_nh5_eta4": int(((pt32 >= 1) & (nhit >= 5) & (np.abs(eta) < 4)).sum()),
                "liu_rec_code": int(((pt > 1) & (nhit >= 5)).sum()),       # eval_metrics: pt > 1, nhits >= 5
                "liu_rec_text": int(((pt >= 1) & (nhit > 5)).sum()),       # paper text: > 5 spacepoints, pT >= 1
                "our_targets": int(pe.is_target.sum()),
            })
    df = pd.DataFrame(rows)
    if not (df.eggnet_targets_port == df.pt1_nh3).all():
        raise RuntimeError("port denominator differs from the direct count")
    return df


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["hits", "noise_hits", "particles_with_hits", "pt1_nh1", "pt1_nh2", "eggnet_targets_port", "pt1_nh5", "pt1_nh3_eta4",
            "pt1_nh5_eta4", "liu_rec_code", "liu_rec_text", "our_targets"]
    out = []
    for ph, g in df.groupby("phase"):
        for c in cols:
            x = g[c].to_numpy(dtype=float)
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            w = np.convolve(g.sort_values("event")[c].to_numpy(dtype=float), np.ones(400) / 400, mode="valid")
            out.append({"phase": ph, "events": len(g), "quantity": c, "mean_per_event": m, "se_mean": se,
                        "sd_per_event": x.std(ddof=1), "x400": 400 * m,
                        "x400_window_sd": 400 * w.std(), "x400_window_min": 400 * w.min(),
                        "x400_window_max": 400 * w.max()})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ Liu Truth-CC
def liu_split(ev_dir: str, names=("train", "val", "test"), split=(300, 10, 10)) -> dict[str, list[str]]:
    """The authors' split (HierarchicalGNN a2563af, Modules/utils.py:15-26 and the base classes' setup, e.g.
    bipartite_classification_base.py:33-36): sorted files of train/val/test, random.seed(42) shuffle, the first
    sum(split) files, then torch random_split(split, generator=torch.Generator().manual_seed(0))."""
    import torch
    from torch.utils.data import random_split
    allf = []
    for n in names:
        d = os.path.join(ev_dir, n)
        if os.path.isdir(d):
            allf.extend(sorted(os.path.join(d, f) for f in os.listdir(d)))
    random.seed(42)
    random.shuffle(allf)
    paths = allf[:sum(split)]
    tr, va, te = random_split(paths, list(split), generator=torch.Generator().manual_seed(0))
    return {"train": [paths[i] for i in tr.indices], "val": [paths[i] for i in va.indices],
            "test": [paths[i] for i in te.indices], "all": sorted(allf)}


def _load(path: str):
    import torch
    from torch_geometric.data import Data
    ev = torch.load(path, map_location="cpu", weights_only=False)
    d = ev.__dict__ if not isinstance(ev, dict) else ev
    if "_store" in d:                               # a PyG >= 2 Data object
        d = dict(d["_store"])
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in d.items()}


def truth_cc_labels(n: int, edge_index: np.ndarray, pid: np.ndarray, keep_edge: np.ndarray) -> np.ndarray:
    """Truth-CC (track_eval.ipynb cell 4, truth=True): kept edges with equal pid at both ends, connected components
    over the hits that touch such an edge (cugraph builds the graph from the edge list); other hits get -1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = edge_index[:, keep_edge]
    e = e[:, pid[e[0]] == pid[e[1]]]
    lab = np.full(n, -1, np.int64)
    if e.shape[1] == 0:
        return lab
    a = coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n))
    _, cc = connected_components(a, directed=False)
    touched = np.zeros(n, bool)
    touched[e[0]] = touched[e[1]] = True
    lab[touched] = cc[touched]
    return lab


def _truthcc_event(args):
    path, ev_dir, in_test, fractions, draws = args
    ev = _load(path)
    pid = np.asarray(ev["pid"]).astype(np.int64)
    pt = np.asarray(ev["pt"]).astype(np.float64)
    ei = np.asarray(ev["edge_index"]).astype(np.int64)
    n = len(pid)
    hit_pt = pt.copy()
    hit_pt[pid == 0] = 0.0                         # TrackMLDataset: event.pt[event.pid == 0] = 0
    rel = os.path.relpath(path, ev_dir)
    rows = []
    for f in fractions:
        for d in range(draws if f > 0 else 1):
            rng = np.random.default_rng([int(round(f * 100)), d, zlib.crc32(rel.encode())])
            keep = rng.random(ei.shape[1]) >= f if f > 0 else np.ones(ei.shape[1], bool)
            lab = truth_cc_labels(n, ei, pid, keep)
            c = LIU.port(pid, lab, None, variant="fork", pt_cut=1.0, nhits_cut=5, majority_cut=0.5,
                         hit_pt=hit_pt, unassigned_label=-1)
            eff, pur = LIU.per_event_ratios(c)
            rows.append({"file": rel, "event_file": str(ev.get("event_file", "")), "in_test10": in_test,
                         "fraction": f, "draw": d, "n_hits": n, "n_noise": int((pid == 0).sum()),
                         "n_edges": ei.shape[1], "n_edges_kept": int(keep.sum()), "eff": eff, "pur": pur,
                         **{f"c_{k}": int(v) for k, v in c.items()}})
    return rows


def liu_truthcc(ev_dir: str, fractions, draws: int, workers: int = 1) -> pd.DataFrame:
    sp = liu_split(ev_dir)
    test = set(sp["test"])
    jobs = [(p, ev_dir, p in test, list(fractions), draws) for p in sp["all"]]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            parts = pool.map(_truthcc_event, jobs, chunksize=4)
    else:
        parts = [_truthcc_event(j) for j in jobs]
    return pd.DataFrame([r for p in parts for r in p])


def summarize_truthcc(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for subset, g0 in (("test10", df[df.in_test10]), ("all340", df)):
        for f, g in g0.groupby("fraction"):
            per_draw = g.groupby("draw").agg(eff=("eff", "mean"), pur=("pur", "mean"))
            w = g.assign(w_eff=g.eff * g.n_hits, w_pur=g.pur * g.n_hits).groupby("draw").agg(
                we=("w_eff", "sum"), wp=("w_pur", "sum"), w=("n_hits", "sum"))
            pub_e, pub_fr = TABLE3[round(f, 1)]
            out.append({
                "events": subset, "fraction": f, "draws": len(per_draw), "n_events": g.file.nunique(),
                "eff_mean": 100 * per_draw.eff.mean(), "eff_sd_draws": 100 * per_draw.eff.std(ddof=1) if len(per_draw) > 1 else 0.0,
                "fr_mean": 100 * (1 - per_draw.pur.mean()),
                "fr_sd_draws": 100 * per_draw.pur.std(ddof=1) if len(per_draw) > 1 else 0.0,
                "eff_hitw": 100 * (w.we / w.w).mean(), "fr_hitw": 100 * (1 - (w.wp / w.w).mean()),
                "published_eff": pub_e, "published_fr": pub_fr,
            })
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("eggnet-counts")
    a.add_argument("--out", required=True)
    a.add_argument("--phase", choices=("throughput", "accuracy", "both"), default="both")
    b = sub.add_parser("liu-truthcc")
    b.add_argument("--events", required=True, help="directory holding the unpacked train/val/test folders")
    b.add_argument("--out", required=True)
    b.add_argument("--draws", type=int, default=20)
    b.add_argument("--fractions", default="0,0.1,0.2,0.3,0.4,0.5")
    b.add_argument("--workers", type=int, default=1)
    x = ap.parse_args(argv)
    os.makedirs(x.out, exist_ok=True)
    if x.cmd == "eggnet-counts":
        phases = ("throughput", "accuracy") if x.phase == "both" else (x.phase,)
        df = pd.concat([eggnet_counts(p) for p in phases], ignore_index=True)
        df.to_csv(os.path.join(x.out, "eggnet_counts_per_event.csv"), index=False)
        s = summarize_counts(df)
        s.to_csv(os.path.join(x.out, "eggnet_counts_summary.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(s.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"Chan et al.: {CHAN_PARTICLES / CHAN_EVENTS:.1f} particles per event "
              f"({CHAN_PARTICLES} / {CHAN_EVENTS}); {CHAN_HITS} spacepoints, {CHAN_NPART} particles per event")
    else:
        fr = [float(v) for v in x.fractions.split(",")]
        df = liu_truthcc(x.events, fr, x.draws, x.workers)
        df.to_csv(os.path.join(x.out, "liu_truthcc_per_event.csv"), index=False)
        s = summarize_truthcc(df)
        s.to_csv(os.path.join(x.out, "liu_truthcc_summary.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(s.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
