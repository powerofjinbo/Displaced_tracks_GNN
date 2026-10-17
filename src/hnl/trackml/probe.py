"""CPU smoke test and sizing of the sectorised Canto-L pipeline on TrackML dev events (docs/trackml_plan.md sections
6.2-6.4, 8 G0): everything that needs no TrackML-trained network.

graphs   For each event: the metric-learning graph of a given embedding (default: the ttbar-trained arm (i) seed 0
         embedding of Canto-L; `--random` adds an untrained network of the TrackML metric-learning architecture) at the
         radii that give the plan's scan densities (8, 10, 12, 14 pairs per hit, K = 300 cap); its true-edge
         efficiency and the cap-binding fraction; the learned-triplet-support union (the ttbar-trained GBDT of Canto-L
         at its operating point k 5, tau 0.005) on the candidates within r_w = 1.5 r*; then the sector tables of the
         base and union graphs for the INFERENCE layout, its G2 fallback and the 16 training-window positions;
         consistency checks of the sector code on the real event (ownership partition, score stitching, oracle track
         stitching by particle hit count). One graph per event (--store-graph union / base at --store-density) is
         written as an event-store file (hnl.trackml.classifier.save_event_graph) for `memory`.
memory   Peak memory and time of one classifier forward (inference) on the largest INFERENCE sector and of one forward +
         backward (training, activation checkpointing as in training, no optimizer step) on the largest training window,
         each in a fresh subprocess: VmHWM minus the RSS before the pass. The network is the untrained TrackMLCanto of
         configs/trackml/classifier_train.yaml: memory does not depend on the weights.
check-support   The chunked learned-triplet-support selection (hnl.trackml.events.support_select) against the unchunked
         hnl.graphs.learned_triplet_filter path on a phi wedge of a real event, pair for pair.

Dev events only (1000-1099: inspected in docs/trackml_feasibility.md; never val or test). No TrackML-trained network
exists; no performance number is produced beyond graph-level efficiencies of the ttbar-trained embedding.

    python -m hnl.trackml.probe graphs --events 1000,1001 --out DIR [--random] [--max-triplets 2e7]
    python -m hnl.trackml.probe memory --store DIR/store --out DIR [--budget-gb 28]
    python -m hnl.trackml.probe check-support --event 1000 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from hnl.overlay import grant as GRANT
from hnl.trackml import events as E
from hnl.trackml import sectors as S
from hnl.trackml.split import REPO

RUNS = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
TTBAR_ML_CKPT = os.path.join(RUNS, "baseline_v1/armi_s0/ml/artifacts/best-epoch=20-f1=0.356576.ckpt")
TTBAR_LTF_MODEL = os.path.join(RUNS, "baseline_v1/armi_s0/learned_triplet_filter_v2/model.joblib")
TTBAR_LTF_K, TTBAR_LTF_TAU = 5, 0.005        # Canto-L operating point (DECISIONS.md 2026-10-04 07:30 UTC)
DENSITIES = (8.0, 10.0, 12.0, 14.0)          # plan section 6.2 scan: base-graph pairs per hit
K_CAP = 300
UNION_TARGET = 16.5                          # plan section 6.2: union pairs per hit
CLASSIFIER_CONFIG = os.path.join(REPO, "configs", "trackml", "classifier_train.yaml")
ML_CONFIG = os.path.join(REPO, "configs", "trackml", "ml_train.yaml")


def _threads():
    from hnl.trackml.config import threads
    return threads()


# ------------------------------------------------------------------------------------------------ radius calibration
def density_radii(emb: np.ndarray, targets, r_lo=0.02, r_hi=0.2, iters: int = 12) -> dict:
    """Radius giving each target of uncapped pairs per hit (bisection on cKDTree neighbour counts)."""
    from scipy.spatial import cKDTree
    x = np.asarray(emb, dtype=np.float64)
    tree = cKDTree(x)
    cache = {}

    def dens(r):
        if r not in cache:
            c = tree.query_ball_point(x, r, return_length=True, workers=-1)
            cache[r] = float((c.sum() - len(x)) / 2 / len(x))
        return cache[r]
    out = {}
    for t in targets:
        lo, hi = r_lo, r_hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if dens(mid) < t:
                lo = mid
            else:
                hi = mid
        out[t] = round(0.5 * (lo + hi), 5)
    return out


def cap_binding(emb: np.ndarray, r: float, k: int) -> float:
    """Fraction of hits with at least k - 1 other hits within r (where FRNN's cap K = k may drop pairs)."""
    from scipy.spatial import cKDTree
    x = np.asarray(emb, dtype=np.float64)
    c = cKDTree(x).query_ball_point(x, r, return_length=True, workers=-1)
    return float(((c - 1) >= k - 1).mean())


def graph_row(ev: E.Event, pairs: np.ndarray, name: str, extra: dict | None = None) -> dict:
    """Sizes and graph-level true-edge efficiencies (all true edges, targets, displaced targets)."""
    n = ev.n
    y = E.true_pair_mask(n, pairs, ev.true_edges)
    te = ev.true_edges
    p = ev.particles.set_index("particle_id")
    tpid = ev.pid[te[0]]
    tgt = p.is_target.reindex(tpid).to_numpy(bool)
    disp = p.is_displaced_target.reindex(tpid).to_numpy(bool)
    from hnl.graphs.ceiling import unordered_unique
    t = unordered_unique(te)
    tin = np.isin(t[0] * n + t[1], pairs[0].astype(np.int64) * n + pairs[1])
    # map unordered true edges back to target flags
    tt = unordered_unique(te[:, tgt])
    td = unordered_unique(te[:, disp])
    eff = lambda u: float(np.isin(u[0] * n + u[1], pairs[0].astype(np.int64) * n + pairs[1]).mean()) if u.shape[1] else np.nan
    deg = np.bincount(pairs.ravel(), minlength=n)
    ct = S.candidate_triplets(n, pairs)
    return {"event": ev.event, "graph": name, "hits": n, "pairs": int(pairs.shape[1]),
            "pairs_per_hit": pairs.shape[1] / n, "deg_mean": float(deg.mean()), "deg_q99": float(np.quantile(deg, 0.99)),
            "deg_max": int(deg.max()), "candidate_triplets": ct, "cand_triplets_per_pair": ct / max(pairs.shape[1], 1),
            "true_edge_eff_all": float(tin.mean()), "true_edge_eff_targets": eff(tt),
            "true_edge_eff_displaced_targets": eff(td), "purity": float(y.mean()),
            "frac_pairs_touching_noise": float((ev.noise[pairs[0]] | ev.noise[pairs[1]]).mean()), **(extra or {})}


def sector_rows(ev: E.Event, pairs: np.ndarray, graph: str) -> list[pd.DataFrame]:
    eta = ev.eta
    out = []
    for layout, spec in (("inference_8x22.5", S.INFERENCE), ("g2_fallback_8x45", S.INFERENCE_G2_FALLBACK)):
        secs = S.split(ev.phi, eta, pairs, spec)
        S.check_partition(secs, pairs.shape[1])
        t = S.sector_table(secs, ev.n, pairs.shape[1])
        t.insert(0, "layout", layout)
        out.append(t)
    rows = []
    for j in range(16):                       # training windows at 16 fixed centres (the sampler draws them uniformly)
        c = -np.pi + (j + 0.5) * S.TRAIN_WINDOW_CORE
        sec = S.split(ev.phi, eta, pairs, S.window_spec(c), sectors=[0])[0]
        t = S.sector_table([sec], ev.n, pairs.shape[1])
        t["sector"] = j
        t["window_center_deg"] = float(np.degrees(c))
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t.insert(0, "layout", "train_window_22.5+-11.25")
    out.append(t)
    for t in out:
        t.insert(0, "graph", graph)
        t.insert(0, "event", ev.event)
    return out


def sector_checks(ev: E.Event, pairs: np.ndarray) -> dict:
    """On the real event: ownership is a partition; stitching dummy scores returns each edge's own value; oracle
    (true-particle) sector tracks stitched back agree with the event-level oracle tracks."""
    eta = ev.eta
    secs = S.split(ev.phi, eta, pairs, S.INFERENCE)
    S.check_partition(secs, pairs.shape[1])
    ids = np.arange(pairs.shape[1], dtype=np.float64)
    st = S.stitch_scores(pairs.shape[1], secs, [ids[s.edges].astype(np.float32) for s in secs])
    stitch_ok = bool(np.array_equal(st, ids.astype(np.float32)))
    labs = [np.where(ev.pid[s.hits] != 0, ev.pid[s.hits], -1) for s in secs]
    # sector-level oracle labels: small integers per sector
    labs = [np.unique(l, return_inverse=True)[1] - (1 if (l < 0).any() else 0) for l in labs]
    t0 = time.time()
    out = S.stitch_tracks(ev.n, secs, labs, ev.phi, eta, S.INFERENCE)
    dt = time.time() - t0
    from hnl.eval.matching import FastMatch
    fm = FastMatch(ev.pid)
    m = fm.match(out)["dm_matched"]
    p = ev.particles.set_index("particle_id")
    sel = np.isin(fm.particles, p.index[p.is_target.astype(bool)])
    real = fm.particles != 0
    nh = fm.n_particle
    return {"partition_ok": True, "stitch_scores_ok": stitch_ok,
            "oracle_stitch_targets_dm": float(m[sel].mean()),
            **{f"oracle_stitch_dm_ge{k}hits": float(m[real & (nh >= k)].mean()) for k in (2, 5)},
            "oracle_stitch_lost_ge5hits": int((real & (nh >= 5) & ~m).sum()), "n_ge5hits": int((real & (nh >= 5)).sum()),
            "stitch_tracks_s": round(dt, 2), "sector_hits_sum_over_event_hits": sum(s.n_hits for s in secs) / ev.n,
            "sector_edges_sum_over_event_edges": sum(s.n_edges for s in secs) / pairs.shape[1],
            "extra_hits_total": int(sum(s.n_extra for s in secs))}


def _store(ev: E.Event, pairs: np.ndarray, store: str) -> int:
    from hnl.trackml.classifier import save_event_graph
    y = E.true_pair_mask(ev.n, pairs, ev.true_edges)
    return save_event_graph(os.path.join(store, f"event{ev.eid}.pyg.gz"), ev.eid, ev.r, ev.phi, ev.z, ev.pid, pairs, y,
                            ev.true_edges)


def cmd_graphs(a):
    import joblib
    import yaml
    # overlay ids only under a grant of kind model (addendum H; never plumb: no number from a pilot output, D6)
    GRANT.check([int(x) for x in a.events.split(",")], "model", "full", out=a.out)
    nth = _threads()
    os.makedirs(a.out, exist_ok=True)
    store = os.path.join(a.out, "store", "valset")
    os.makedirs(store, exist_ok=True)
    model = joblib.load(a.ltf_model)
    embedders = {"ttbar": E.Embedder(a.ml_ckpt)}
    if a.random:
        embedders["random"] = E.Embedder(None, hparams=yaml.safe_load(open(ML_CONFIG)), seed=0)
    rows, secrows, checks, timing = [], [], [], []
    for ev in E.iter_events([int(x) for x in a.events.split(",")]):
        for name, emb_f in embedders.items():
            t0 = time.time()
            emb = emb_f(ev)
            timing.append({"event": ev.event, "embedding": name, "step": "embed", "s": round(time.time() - t0, 1)})
            x = emb.numpy()
            t0 = time.time()
            radii = density_radii(x, a.densities)
            timing.append({"event": ev.event, "embedding": name, "step": "radius_calibration",
                           "s": round(time.time() - t0, 1)})
            for dens, r in radii.items():
                t0 = time.time()
                base = E.radius_graph(emb, r, K_CAP)
                dt = time.time() - t0
                rows.append(graph_row(ev, base, f"{name}_base_d{dens:g}", {"embedding": name, "r": r, "K": K_CAP,
                                                                         "cap_binding": cap_binding(x, r, K_CAP),
                                                                         "build_s": round(dt, 1)}))
                if name == "ttbar" and a.store_graph == "base" and dens == a.store_density:
                    _store(ev, base, store)
                    checks.append({"event": ev.event, "graph": f"{name}_base_d{dens:g}", **sector_checks(ev, base)})
                if name != "ttbar" or dens not in a.union_at:
                    if name == "ttbar" or dens == 12.0:
                        secrows += sector_rows(ev, base, f"{name}_base_d{dens:g}")
                    continue
                secrows += sector_rows(ev, base, f"{name}_base_d{dens:g}")
                r_w = round(a.rw_factor * r, 5)
                t0 = time.time()
                added, st = E.union_additions(ev, emb, base, r_w, model, a.tau, a.k, max_triplets=a.max_triplets)
                dt = time.time() - t0
                union = E.union_graph(base, added)
                rows.append(graph_row(ev, union, f"{name}_union_d{dens:g}",
                                      {"embedding": name, "r": r, "K": K_CAP, "r_wide": r_w, "k": a.k, "tau": a.tau,
                                       "build_s": round(dt, 1), "added_pairs": int(added.shape[1]),
                                       "added_bytes_int32": int(added.size * 4),
                                       **{f"support_{kk}": v for kk, v in st.items()}}))
                secrows += sector_rows(ev, union, f"{name}_union_d{dens:g}")
                if a.store_graph == "union" and dens == a.store_density:
                    _store(ev, union, store)
                    checks.append({"event": ev.event, "graph": f"{name}_union_d{dens:g}", **sector_checks(ev, union)})
            print(json.dumps(rows[-1], default=float), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "graphs.csv"), index=False)
    pd.concat(secrows, ignore_index=True).to_csv(os.path.join(a.out, "sectors.csv"), index=False)
    with open(os.path.join(a.out, "checks.json"), "w") as fh:
        json.dump({"checks": checks, "timing": timing, "threads": nth, "ml_ckpt": a.ml_ckpt, "ltf_model": a.ltf_model,
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename}, fh, indent=1,
                  default=float)
    print(json.dumps(checks, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ memory
def _status(key):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key):
                return int(line.split()[1]) / 1024
    return float("nan")


def _memory_worker(store_file: str, mode: str, spec_json: str, sector: int):
    import ctypes
    import gc
    import torch
    import yaml
    from hnl.trackml import classifier as CL
    nth = _threads()
    cfg = yaml.safe_load(open(CLASSIFIER_CONFIG))
    hp = CL.model_hparams({**cfg, "input_dir": "/nonexistent", "stage_dir": "/nonexistent", "data_split": [0, 0, 0]})
    torch.manual_seed(0)
    model = CL.TrackMLCanto(hp)
    ev = CL.load_event_graph(store_file)
    spec = S.SectorSpec.from_dict(json.loads(spec_json))
    batch, sec = CL.sector_item(ev, spec, sector)
    n_params = sum(p.numel() for p in model.parameters())
    cand = S.candidate_triplets(sec.n_hits, sec.local_edges)
    tiny, _ = CL.sector_item(ev, S.SectorSpec(n_phi=512, halo_phi=0.0), 0)
    if mode == "infer":
        model.eval()
        with torch.inference_mode():
            model(tiny)
    else:
        model.train()
        out = model(tiny)
        model.loss_function(out, tiny)[0].backward()
        model.zero_grad(set_to_none=True)
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")
    rss0 = _status("VmRSS:")
    t0 = time.time()
    if mode == "infer":
        with torch.inference_mode():
            out = model(batch)
        loss = None
    else:
        out = model(batch)
        loss = model.loss_function(out, batch)[0]
        loss.backward()
    dt = time.time() - t0
    hwm = _status("VmHWM:")
    print(json.dumps({"mode": mode, "sector": sector, "spec": json.loads(spec_json), "hits": sec.n_hits,
                      "pairs": sec.n_edges, "owned_pairs": int(sec.owned.sum()), "candidate_triplets": cand,
                      "n_params": int(n_params), "threads": nth, "rss_before_mb": round(rss0),
                      "rss_peak_mb": round(hwm), "peak_increase_mb": round(hwm - rss0), "seconds": round(dt, 1),
                      "loss": None if loss is None else float(loss), "n_out": int(out.numel())}), flush=True)


def cmd_memory(a):
    store = os.path.join(a.store, "valset")
    files = sorted(f for f in os.listdir(store) if f.endswith(".pyg.gz"))
    GRANT.check_store(files[:a.max_events], "model", out=a.out)   # overlay events only under a grant (addendum H)
    from hnl.trackml import classifier as CL
    jobs = []
    for f in files[:a.max_events]:
        ev = CL.load_event_graph(os.path.join(store, f))
        eta = S.hit_eta(ev["r"], ev["z"])
        inf = S.split(ev["phi"], eta, ev["pairs"], S.INFERENCE)
        sizes = [(S.candidate_triplets(s.n_hits, s.local_edges), s.n_edges, s.index) for s in inf]
        big = max(sizes)
        jobs.append((f, "infer", S.INFERENCE, big[2], big))
        small = min(sizes)
        jobs.append((f, "infer", S.INFERENCE, small[2], small))
        wins = []
        for j in range(16):
            c = -np.pi + (j + 0.5) * S.TRAIN_WINDOW_CORE
            sec = S.split(ev["phi"], eta, ev["pairs"], S.window_spec(c), sectors=[0])[0]
            wins.append((S.candidate_triplets(sec.n_hits, sec.local_edges), sec.n_edges, j, c))
        bw = max(wins)
        jobs.append((f, "train", S.window_spec(bw[3]), 0, bw[:2]))
        sw = min(wins)
        jobs.append((f, "train", S.window_spec(sw[3]), 0, sw[:2]))
    budget_mb = a.budget_gb * 1024
    rows = []
    for f, mode, spec, sector, size in sorted(jobs, key=lambda j: (j[1], j[4][0])):
        pred = (S.memory_estimate(size[1], size[0])["infer_gb" if mode == "infer" else "train_gb"]) * 1024
        if pred > budget_mb:
            rows.append({"file": f, "mode": mode, "skipped": True, "predicted_mb": round(pred), "pairs": size[1],
                         "candidate_triplets": size[0]})
            print(json.dumps(rows[-1]), flush=True)
            continue
        res = subprocess.run([sys.executable, "-m", "hnl.trackml.probe", "_memory_worker", os.path.join(store, f), mode,
                              json.dumps(spec.to_dict()), str(sector)], capture_output=True, text=True)
        line = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
        if res.returncode != 0 or not line:
            rows.append({"file": f, "mode": mode, "failed": True, "rc": res.returncode,
                         "stderr_tail": res.stderr[-2000:]})
        else:
            rows.append({"file": f, **json.loads(line[-1])})
        print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(a.out, "memory.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


# ------------------------------------------------------------------------------------------------ support exactness
def cmd_check_support(a):
    import joblib
    GRANT.check([a.event], "model", "full", out=a.out)            # overlay ids only under a grant (addendum H)
    _threads()
    model = joblib.load(a.ltf_model)
    ev = E.load_events([a.event])[a.event]
    emb = E.Embedder(a.ml_ckpt)(ev)
    # a phi wedge of the event keeps the unchunked reference within memory
    keep = np.abs(S.wrap(ev.phi.astype(np.float64) - 0.3)) < np.radians(a.wedge_deg) / 2
    sub = E.subset(ev, keep)
    x = emb[np.flatnonzero(keep)]
    rows = []
    for r_w in a.radii:
        cand = E.radius_graph(x, r_w, None)
        t0 = time.time()
        ref = E.support_select_reference(sub, cand, model, a.tau, a.k)
        t_ref = time.time() - t0
        res = {}
        for mt in (1e9, 2e5, 2e4):
            t0 = time.time()
            got, st = E.support_select(sub, cand, model, a.tau, a.k, max_triplets=mt, return_stats=True)
            res[mt] = (got, st, time.time() - t0)
        rows.append({"event": a.event, "wedge_deg": a.wedge_deg, "hits": sub.n, "r_wide": r_w, "candidates": cand.shape[1],
                     "kept_reference": int(ref.shape[1]), "ref_s": round(t_ref, 1),
                     **{f"identical_max{mt:g}": bool(np.array_equal(ref, g)) for mt, (g, _, _) in res.items()},
                     **{f"chunks_max{mt:g}": st["n_chunks"] for mt, (_, st, _) in res.items()},
                     **{f"s_max{mt:g}": round(dt, 1) for mt, (_, _, dt) in res.items()},
                     "triplets": res[1e9][1]["n_trip_cand"]})
        print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "check_support.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    if not all(all(v for k, v in r.items() if k.startswith("identical")) for r in rows):
        sys.exit("chunked support selection differs from the reference")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graphs")
    g.add_argument("--events", default="1000,1001")
    g.add_argument("--out", required=True)
    g.add_argument("--ml-ckpt", default=TTBAR_ML_CKPT)
    g.add_argument("--ltf-model", default=TTBAR_LTF_MODEL)
    g.add_argument("--k", type=int, default=TTBAR_LTF_K)
    g.add_argument("--tau", type=float, default=TTBAR_LTF_TAU)
    g.add_argument("--rw-factor", type=float, default=1.5)
    g.add_argument("--union-at", type=lambda s: tuple(float(x) for x in s.split(",") if x), default=(12.0,))
    g.add_argument("--store-density", type=float, default=12.0)
    g.add_argument("--store-graph", default="union", choices=["union", "base"])
    g.add_argument("--densities", type=lambda s: tuple(float(x) for x in s.split(",")), default=DENSITIES)
    g.add_argument("--max-triplets", type=float, default=2e7)
    g.add_argument("--random", action="store_true")
    m = sub.add_parser("memory")
    m.add_argument("--store", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--budget-gb", type=float, default=26.0)
    m.add_argument("--max-events", type=int, default=1)
    w = sub.add_parser("_memory_worker")
    w.add_argument("store_file")
    w.add_argument("mode", choices=["infer", "train"])
    w.add_argument("spec")
    w.add_argument("sector", type=int)
    c = sub.add_parser("check-support")
    c.add_argument("--event", type=int, default=1000)
    c.add_argument("--out", required=True)
    c.add_argument("--wedge-deg", type=float, default=30.0)
    c.add_argument("--radii", type=lambda s: [float(x) for x in s.split(",")], default=[0.1])
    c.add_argument("--ml-ckpt", default=TTBAR_ML_CKPT)
    c.add_argument("--ltf-model", default=TTBAR_LTF_MODEL)
    c.add_argument("--k", type=int, default=TTBAR_LTF_K)
    c.add_argument("--tau", type=float, default=TTBAR_LTF_TAU)
    a = ap.parse_args(argv)
    if a.cmd == "graphs":
        cmd_graphs(a)
    elif a.cmd == "memory":
        cmd_memory(a)
    elif a.cmd == "_memory_worker":
        _memory_worker(a.store_file, a.mode, a.spec, a.sector)
    else:
        cmd_check_support(a)


if __name__ == "__main__":
    main()
