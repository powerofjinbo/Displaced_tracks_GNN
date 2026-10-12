"""Graph stages of Canto-L on TrackML (docs/trackml_plan.md sections 6.1-6.3 and 7). Event sets as in
hnl.trackml.config (train, val, valmon, dev, gbdt, merge_train, smoke, t2 / test with --allow-test, or a comma list).

  fs         S1: acorn feature-store events (node-local) for the metric-learning training (hnl.trackml.ml.stage_fs);
             --ml-config adds the cluster features its node_features list (X1, configs/trackml/ml_train_x1.yaml).
  radii      S2: r* for each density target (uncapped pairs per hit of the trained embedding; bisection per event on the
             first graph.radii_events valmon events, radii averaged) -> radii.json.
  ltf-data   S2: learned-support training rows from the gbdt events (candidates within support.data_rw_factor x
             r*(largest density), the 16 features of hnl.graphs.learned_triplet_filter, field-corrected log_pt_turn
             when a field map is given; label: both legs chain edges; every positive, negatives subsampled to at most
             support.max_negative_rows in total). Train with `python -m hnl.graphs.learned_triplet_filter train`.
  scan       S2: valmon graph scan over density x r_w factor x (tau, k): union pairs per hit, chain-edge efficiency,
             FRNN cap binding, and graph-stage ceilings (CC on the true edges of the graph, double majority): targets,
             displaced targets (|d0| >= 10 mm), all particles with >= 5 hits.
  choose     S2: the configuration with the highest target ceiling at <= graph.union_max_pairs_per_hit (ties within
             graph.tie_ceiling: fewer pairs), cap check, gate G1 -> graph_choice.json (--max-pairs-per-hit 20: D3 X2).
  additions  S3: the union additions (union minus base, int32 pairs) of an event set, 50 events per parquet file under
             <run>/additions/<set>/ (storage stop rule checked before every write).
  stage      S4 / S5: full-event union graphs (base rebuilt from the embedding + stored additions) to a node-local
             event store <dest>/<trainset|valset|testset>/event<eid>.pyg.gz (hnl.trackml.classifier format).

The embedding runs on CUDA when available (FRNN radius search, as hnl.train.acorn_runner infer), else on CPU (cKDTree).
The learned support runs with the numpy reference on the CPU or the torch port (hnl.trackml.support_gpu, gate G4) on
CUDA (--support-backend; on CUDA the wide candidate graph has the FRNN cap WIDE_K_GPU, whose binding is recorded).
Base graphs must be rebuilt with the backend that made the additions (a pair at distance ~ r* may differ between float32
FRNN and the CPU search); the additions' sidecar records it and `stage` refuses a mismatch unless --allow-backend-change.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.trackml import config as CFG
from hnl.trackml import events as E


def _device(arg: str | None = None) -> str:
    import torch
    if arg:
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _backend(device: str) -> str:
    return "frnn_cuda" if device.startswith("cuda") else "ckdtree_cpu"


def _threads():
    CFG.threads()


def _json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(path + ".tmp", path)


def _meta(**kw) -> dict:
    import datetime as dt
    import subprocess
    git = subprocess.run(["git", "-C", CFG.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": git,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, **kw}


# ------------------------------------------------------------------------------------------------ support backends
WIDE_K_GPU = 2000                         # FRNN needs a cap: the wide candidate graph is uncapped on the CPU


class Support:
    """The learned triplet support on a backend: "numpy" (hnl.trackml.events.support_select_multi, the reference) or
    "torch" (hnl.trackml.support_gpu on the embedding's device; gate G4 checks it against the reference)."""

    def __init__(self, model_path: str, backend: str, device: str):
        import joblib
        self.model = joblib.load(model_path)
        self.backend = backend
        self.hgb = None
        if backend == "torch":
            from hnl.trackml.support_gpu import TorchHGB
            self.hgb = TorchHGB(self.model, device)
        elif backend != "numpy":
            raise ValueError(f"support backend {backend!r}")

    def wide(self, emb, r_wide: float) -> tuple[np.ndarray, float]:
        """Candidate pairs within r_wide and the fraction of hits at the FRNN cap (CUDA only; 0 on the CPU)."""
        import torch
        if isinstance(emb, torch.Tensor) and emb.is_cuda:
            w = E.radius_graph(emb, r_wide, WIDE_K_GPU)
            deg = np.bincount(w.ravel(), minlength=emb.shape[0])
            return w, float((deg >= WIDE_K_GPU - 1).mean())
        return E.radius_graph(emb, r_wide, None), 0.0

    def select(self, ev, cand, ops, fmap, max_triplets):
        if self.backend == "torch":
            from hnl.trackml.support_gpu import support_select_torch
            return support_select_torch(ev, cand, self.hgb, ops, fmap, max_triplets)
        return E.support_select_multi(ev, cand, self.model, ops, fmap, max_triplets)


def _support_backend(arg: str | None, device: str) -> str:
    return arg or ("torch" if device.startswith("cuda") else "numpy")


# ------------------------------------------------------------------------------------------------ ceilings
def ceilings(ev: E.Event, pairs: np.ndarray, fm=None) -> dict:
    """Graph-stage ceilings of one graph: connected components on its true edges, double-majority matching; target,
    displaced-target and >= 5-hit particle fractions matched (counts, for pooling)."""
    from hnl.eval.matching import FastMatch
    from hnl.graphs.ceiling import cc_labels, unordered_unique
    fm = fm or FastMatch(ev.pid)
    y = E.true_pair_mask(ev.n, pairs, ev.true_edges)
    dm = fm.match(cc_labels(ev.n, pairs[:, y]))["dm_matched"]
    p = ev.particles.set_index("particle_id")
    out = {"n_true_in_graph": int(y.sum()), "n_true": int(unordered_unique(ev.true_edges).shape[1])}
    for name, col in (("targets", "is_target"), ("displaced_targets", "is_displaced_target")):
        ids = p.index[p[col].astype(bool)]
        sel = np.isin(fm.particles, ids)
        out[f"n_{name}"], out[f"m_{name}"] = int(sel.sum()), int(dm[sel].sum())
    ids = p.index[p.n_hits >= 5]
    sel = np.isin(fm.particles, ids)
    out["n_ge5"], out["m_ge5"] = int(sel.sum()), int(dm[sel].sum())
    return out


# ------------------------------------------------------------------------------------------------ radii
def cmd_radii(a, cfg):
    from hnl.trackml.probe import density_radii
    _threads()
    dens = [float(x) for x in cfg["graph"]["densities"]]
    evs = CFG.event_set(a.events, cfg)[:int(cfg["graph"]["radii_events"])]
    emb_f = E.Embedder(a.ml_ckpt, _device(a.device))
    per = []
    for ev in E.iter_events(evs, particles=False):
        x = emb_f(ev).cpu().numpy()
        r = density_radii(x, dens)
        per.append({"event": ev.event, **{str(k): v for k, v in r.items()}})
        print(json.dumps(per[-1]), flush=True)
    radii = {str(d): float(np.mean([p[str(d)] for p in per])) for d in dens}
    _json(a.out, {"radii": radii, "per_event": per, "ml_ckpt": a.ml_ckpt, "events": evs, **_meta()})
    print(json.dumps(radii))


# ------------------------------------------------------------------------------------------------ support training rows
def cmd_ltf_data(a, cfg):
    from hnl.graphs import learned_triplet_filter as LT
    _threads()
    radii = json.load(open(a.radii))["radii"]
    r_w = float(cfg["support"]["data_rw_factor"]) * max(float(v) for v in radii.values())
    evs = CFG.event_set(a.events, cfg)
    cap = float(cfg["support"]["max_negative_rows"]) / len(evs)
    fmap = E.FieldMap(CFG.repo_path(a.field_map)) if a.field_map else None
    emb_f = E.Embedder(a.ml_ckpt, _device(a.device))
    Xs, ys, ws, eids, info = [], [], [], [], []
    t0 = time.time()
    for ev in E.iter_events(evs, particles=False):
        emb = emb_f(ev)
        wide = E.radius_graph(emb, r_w, WIDE_K_GPU if emb.is_cuda else None)
        n = ev.n
        # pass 1: count the triplets (negatives ~ all of them) to set this event's negative sampling fraction
        counts = {"n": E.count_triplets(ev, wide, float(cfg["graph"]["max_triplets_per_chunk"]))}
        frac = min(1.0, cap / max(counts["n"], 1))
        rng = np.random.default_rng([int(ev.event), 11])
        chain_pairs = None

        def collect(ei, eo, X, p, s, d):
            nonlocal chain_pairs
            if chain_pairs is None:
                lo, hi = np.minimum(s, d), np.maximum(s, d)
                chain_pairs = E.true_pair_mask(n, np.stack([lo, hi]), ev.true_edges)
            y = chain_pairs[ei] & chain_pairs[eo]
            keep = y | (rng.random(len(y)) < frac)
            Xs.append(X[keep])
            ys.append(y[keep])
            ws.append(np.where(y[keep], 1.0, 1.0 / frac).astype(np.float32))
            eids.append(np.full(int(keep.sum()), int(ev.eid), dtype=np.int64))
        E.support_select_multi(ev, wide, None, [], fmap, float(cfg["graph"]["max_triplets_per_chunk"]),
                               triplet_data=collect)
        info.append({"event": ev.event, "wide_pairs": int(wide.shape[1]), "triplets": counts["n"], "neg_frac": frac})
        print(json.dumps(info[-1]), flush=True)
    X, y, w, ev_ = (np.concatenate(v) for v in (Xs, ys, ws, eids))
    np.savez(a.out, X=X, y=y, w=w, event=ev_, features=np.array(LT.FEATURES))
    _json(a.out + ".json", {"rows": int(len(y)), "positives": int(y.sum()), "r_wide": r_w, "events": evs,
                            "per_event": info, "field_map": a.field_map, "ml_ckpt": a.ml_ckpt,
                            "elapsed_s": round(time.time() - t0, 1), **_meta()})


# ------------------------------------------------------------------------------------------------ scan and choice
def cmd_scan(a, cfg):
    from hnl.eval.matching import FastMatch
    from hnl.trackml.probe import cap_binding
    _threads()
    g = cfg["graph"]
    radii = {float(k): float(v) for k, v in json.load(open(a.radii))["radii"].items()}
    ops = [(float(t), int(k)) for t in g["taus"] for k in g["ks"]]
    device = _device(a.device)
    sup = Support(a.ltf_model, _support_backend(a.support_backend, device), device)
    fmap = E.FieldMap(CFG.repo_path(a.field_map)) if a.field_map else None
    emb_f = E.Embedder(a.ml_ckpt, device)
    evs = CFG.event_set(a.events, cfg)[:a.max_events]
    rows = []
    os.makedirs(a.out, exist_ok=True)
    for ev in E.iter_events(evs):
        t0 = time.time()
        emb = emb_f(ev)
        fm = FastMatch(ev.pid)
        for dens, r in sorted(radii.items()):
            base = E.radius_graph(emb, r, int(g["knn"]))
            cb = cap_binding(emb.cpu().numpy(), r, int(g["knn"]))
            rows.append({"event": ev.event, "density": dens, "r_star": r, "rw_factor": 0.0, "tau": np.nan, "k": 0,
                         "hits": ev.n, "pairs": int(base.shape[1]), "cap_binding": cb, **ceilings(ev, base, fm)})
            for f in g["rw_factors"]:
                wide, wide_cap = sup.wide(emb, float(f) * r)
                kept, st = sup.select(ev, wide, ops, fmap, float(g["max_triplets_per_chunk"]))
                for (tau, k), kp in kept.items():
                    union = E.union_graph(base, kp)
                    rows.append({"event": ev.event, "density": dens, "r_star": r, "rw_factor": float(f), "tau": tau,
                                 "k": k, "hits": ev.n, "pairs": int(union.shape[1]), "cap_binding": cb,
                                 "wide_pairs": int(wide.shape[1]), "wide_cap_binding": wide_cap,
                                 "triplets": st["n_trip_cand"],
                                 **ceilings(ev, union, fm)})
        print(json.dumps({"event": ev.event, "rows": len(rows), "seconds": round(time.time() - t0, 1)}), flush=True)
        pd.DataFrame(rows).to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    _json(os.path.join(a.out, "scan.json"), {"events": evs, "radii": radii, "ops": ops, "ml_ckpt": a.ml_ckpt,
                                             "ltf_model": a.ltf_model, "field_map": a.field_map,
                                             "backend": _backend(device), "support_backend": sup.backend, **_meta()})


def summarize_scan(pe: pd.DataFrame) -> pd.DataFrame:
    key = ["density", "r_star", "rw_factor", "tau", "k"]
    pe = pe.fillna({"tau": -1.0})
    g = pe.groupby(key)
    s = g[["pairs", "hits", "n_true_in_graph", "n_true", "n_targets", "m_targets", "n_displaced_targets",
           "m_displaced_targets", "n_ge5", "m_ge5"]].sum()
    s["pairs_per_hit"] = s.pairs / s.hits
    s["true_edge_eff"] = s.n_true_in_graph / s.n_true
    s["ceil_targets"] = s.m_targets / s.n_targets
    s["ceil_displaced_targets"] = s.m_displaced_targets / s.n_displaced_targets
    s["ceil_ge5"] = s.m_ge5 / s.n_ge5
    s["cap_binding"] = g.cap_binding.mean()
    s["n_events"] = g.size()
    return s.reset_index()


def choose(summ: pd.DataFrame, gcfg: dict) -> dict:
    """The plan's rule (section 6.2): among union configurations with <= union_max_pairs_per_hit, the highest target
    ceiling; ceilings within tie_ceiling of the best are tied and the tie goes to fewer pairs per hit."""
    u = summ[(summ.rw_factor > 0) & (summ.pairs_per_hit <= float(gcfg["union_max_pairs_per_hit"]) + 1e-12)]
    if u.empty:
        raise ValueError("no union configuration within the pairs-per-hit budget")
    best = u.ceil_targets.max()
    tied = u[u.ceil_targets >= best - float(gcfg["tie_ceiling"]) - 1e-12]
    row = tied.sort_values(["pairs_per_hit", "ceil_targets"], ascending=[True, False]).iloc[0]
    out = {k: (float(row[k]) if k != "k" else int(row[k])) for k in
           ("density", "r_star", "rw_factor", "tau", "k", "pairs_per_hit", "ceil_targets", "ceil_displaced_targets",
            "ceil_ge5", "true_edge_eff", "cap_binding")}
    out["r_wide"] = out["r_star"] * out["rw_factor"]
    out["n_tied"] = int(len(tied))
    out["cap_ok"] = bool(out["cap_binding"] < float(gcfg["cap_bind_max"]))
    out["gate_g1_pass"] = bool(out["ceil_targets"] >= float(gcfg["gate_g1_ceiling"]))
    return out


def cmd_choose(a, cfg):
    pe = pd.read_parquet(os.path.join(a.scan, "per_event.parquet"))
    summ = summarize_scan(pe)
    summ.to_csv(os.path.join(a.scan, "summary.csv"), index=False)
    scan = json.load(open(os.path.join(a.scan, "scan.json")))
    gcfg = dict(cfg["graph"])
    if a.max_pairs_per_hit is not None:          # D3 X2: the same rule with a larger budget (20 pairs per hit)
        gcfg["union_max_pairs_per_hit"] = float(a.max_pairs_per_hit)
    ch = choose(summ, gcfg)
    ch.update({"ml_ckpt": scan["ml_ckpt"], "ltf_model": scan["ltf_model"], "field_map": scan["field_map"],
               "knn": int(cfg["graph"]["knn"]), "scan": a.scan, "scan_events": int(pe.event.nunique()),
               "union_max_pairs_per_hit": float(gcfg["union_max_pairs_per_hit"]), **_meta()})
    _json(a.out, ch)
    pd.set_option("display.width", 220)
    print(summ.sort_values("pairs_per_hit").round(4).to_string(index=False))
    print(json.dumps(ch, indent=1, default=float))
    if not ch["gate_g1_pass"]:
        print("GATE G1 FAILED: union target ceiling below the target; D3 contingency (X1, then X2) per plan 6.2")


# ------------------------------------------------------------------------------------------------ additions
def cmd_additions(a, cfg):
    from hnl.trackml.convert import storage_check
    _threads()
    ch = json.load(open(a.graph_choice))
    fmap = E.FieldMap(CFG.repo_path(ch["field_map"])) if ch.get("field_map") else None
    device = _device(a.device)
    sup = Support(ch["ltf_model"], _support_backend(a.support_backend, device), device)
    emb_f = E.Embedder(ch["ml_ckpt"], device)
    evs = CFG.event_set(a.events, cfg, allow_test=a.allow_test)
    os.makedirs(a.out, exist_ok=True)
    done = set()
    for f in os.listdir(a.out):
        if f.endswith(".parquet.json"):
            done |= {int(e) for e in json.load(open(os.path.join(a.out, f)))["events"]}
    todo = [e for e in evs if e not in done]
    print(json.dumps({"events": len(evs), "already": len(done & set(evs)), "todo": len(todo), "device": device}),
          flush=True)
    for k in range(0, len(todo), a.per_file):
        part = todo[k:k + a.per_file]
        adds, stats = {}, []
        t0 = time.time()
        for ev in E.iter_events(part, particles=a.condition == "1gev"):
            ev = E.with_condition(ev, a.condition)
            emb = emb_f(ev)
            base = E.radius_graph(emb, ch["r_star"], ch["knn"])
            wide, wide_cap = sup.wide(emb, ch["r_wide"])
            op = (float(ch["tau"]), int(ch["k"]))
            kept, st = sup.select(ev, wide, [op], fmap, float(cfg["graph"]["max_triplets_per_chunk"]))
            union = E.union_graph(base, kept[op])
            n = ev.n
            added = union[:, ~np.isin(union[0] * n + union[1], base[0].astype(np.int64) * n + base[1])]
            adds[ev.event] = added
            stats.append({"event": ev.event, "n_base": int(base.shape[1]), "n_wide": int(wide.shape[1]),
                          "wide_cap_binding": wide_cap, "n_kept": int(kept[op].shape[1]), "n_union": int(union.shape[1]),
                          "n_added": int(added.shape[1]), "n_trip_cand": st["n_trip_cand"], "n_chunks": st["n_chunks"]})
        nbytes = sum(v.size * 4 for v in adds.values())
        st = storage_check(nbytes)
        path = os.path.join(a.out, f"additions_{part[0]:05d}.parquet")
        side = E.write_additions(path, adds, {"graph_choice": a.graph_choice, "backend": _backend(device),
                                              "support_backend": sup.backend,
                                              "condition": a.condition,
                                              "stats": stats, "storage": st, **_meta()})
        print(json.dumps({"file": path, "events": len(part), "bytes": side["bytes"],
                          "mean_added": float(np.mean([s["n_added"] for s in stats])),
                          "seconds": round(time.time() - t0, 1)}), flush=True)


# ------------------------------------------------------------------------------------------------ event store
def cmd_stage(a, cfg):
    from hnl.trackml.classifier import save_event_graph
    _threads()
    ch = json.load(open(a.graph_choice))
    device = _device(a.device)
    evs = CFG.event_set(a.events, cfg, allow_test=a.allow_test)
    sides = [json.load(open(os.path.join(a.additions, f))) for f in os.listdir(a.additions)
             if f.endswith(".parquet.json")]
    backends = {s_["backend"] for s_ in sides}
    conds = {s_.get("condition", "full") for s_ in sides}
    if conds - {a.condition}:
        raise SystemExit(f"additions made for condition(s) {sorted(conds)}, staging condition {a.condition}")
    if backends - {_backend(device)} and not a.allow_backend_change:
        raise SystemExit(f"additions made with {sorted(backends)}, staging on {_backend(device)}: pass "
                         "--allow-backend-change to accept a slightly different base graph")
    emb_f = E.Embedder(ch["ml_ckpt"], device)
    os.makedirs(a.dest, exist_ok=True)
    tmp = os.path.join(a.dest, ".tmp")      # written there, then renamed: a killed job leaves no partial event file
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    tot, skipped = 0, 0
    for k in range(0, len(evs), 50):
        part = evs[k:k + 50]
        adds = E.read_additions(a.additions, part)
        for ev in E.iter_events(part, particles=a.condition == "1gev"):
            name = f"event{ev.eid}.pyg.gz"
            if a.skip_existing and os.path.exists(os.path.join(a.dest, name)):
                skipped += 1
                continue
            ev = E.with_condition(ev, a.condition)
            emb = emb_f(ev)
            base = E.radius_graph(emb, ch["r_star"], ch["knn"])
            union = E.union_graph(base, adds[ev.event])
            y = E.true_pair_mask(ev.n, union, ev.true_edges)
            tot += save_event_graph(os.path.join(tmp, name), ev.eid, ev.r, ev.phi, ev.z, ev.pid,
                                    union, y, ev.true_edges, rows=ev.full_rows)
            os.replace(os.path.join(tmp, name), os.path.join(a.dest, name))
    print(json.dumps({"events": len(evs), "skipped_existing": skipped, "bytes": tot,
                      "seconds": round(time.time() - t0, 1), "dest": a.dest, "condition": a.condition}))


def cmd_fs(a, cfg):
    import yaml
    from hnl.trackml.ml import fs_features, stage_fs
    evs = CFG.event_set(a.events, cfg, allow_test=a.allow_test)
    feats = []
    if a.ml_config:
        with open(CFG.repo_path(a.ml_config)) as fh:
            feats = fs_features(yaml.safe_load(fh)["node_features"])
    print(json.dumps(stage_fs(evs, a.dest, a.condition, feats, a.nproc)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fs")
    f.add_argument("--events", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--condition", default="full")
    f.add_argument("--allow-test", action="store_true")
    f.add_argument("--ml-config", default=None, help="metric-learning config: its cluster node_features are stored")
    f.add_argument("--nproc", type=int, default=1)
    r = sub.add_parser("radii")
    r.add_argument("--events", default="valmon")
    r.add_argument("--ml-ckpt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default=None)
    d = sub.add_parser("ltf-data")
    d.add_argument("--events", default="gbdt")
    d.add_argument("--ml-ckpt", required=True)
    d.add_argument("--radii", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--field-map", default=None)
    d.add_argument("--device", default=None)
    s = sub.add_parser("scan")
    s.add_argument("--events", default="valmon")
    s.add_argument("--ml-ckpt", required=True)
    s.add_argument("--radii", required=True)
    s.add_argument("--ltf-model", required=True)
    s.add_argument("--field-map", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--max-events", type=int, default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--support-backend", default=None, choices=["numpy", "torch"],
                   help="default: torch on CUDA (gate G4), numpy on the CPU")
    c = sub.add_parser("choose")
    c.add_argument("--scan", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-pairs-per-hit", type=float, default=None,
                   help="pairs-per-hit budget (default graph.union_max_pairs_per_hit; 20 for D3 X2)")
    ad = sub.add_parser("additions")
    ad.add_argument("--events", required=True)
    ad.add_argument("--graph-choice", required=True)
    ad.add_argument("--out", required=True)
    ad.add_argument("--per-file", type=int, default=50)
    ad.add_argument("--condition", default="full", choices=E.CONDITIONS)
    ad.add_argument("--support-backend", default=None, choices=["numpy", "torch"])
    ad.add_argument("--device", default=None)
    ad.add_argument("--allow-test", action="store_true")
    st = sub.add_parser("stage")
    st.add_argument("--events", required=True)
    st.add_argument("--graph-choice", required=True)
    st.add_argument("--additions", required=True)
    st.add_argument("--dest", required=True)
    st.add_argument("--condition", default="full", choices=E.CONDITIONS)
    st.add_argument("--device", default=None)
    st.add_argument("--allow-test", action="store_true")
    st.add_argument("--allow-backend-change", action="store_true")
    st.add_argument("--skip-existing", action="store_true",
                    help="keep event files already in --dest (a persistent store filled over several jobs)")
    a = ap.parse_args(argv)
    cfg = CFG.load(a.config)
    {"fs": cmd_fs, "radii": cmd_radii, "ltf-data": cmd_ltf_data, "scan": cmd_scan, "choose": cmd_choose,
     "additions": cmd_additions, "stage": cmd_stage}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
