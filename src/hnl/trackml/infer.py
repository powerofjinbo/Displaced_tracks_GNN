"""Edge scores of whole TrackML events (docs/trackml_plan.md sections 6.3, 8 S5) and gate G2.

score  Each event of a node-local event store (hnl.trackml.graphs stage) is cut into the INFERENCE sectors (8 phi
       sectors, 45 deg cores, +-22.5 deg halos; --sectors g2_fallback for +-45 deg; --sectors full: the whole event in
       one forward, G2's reference, after G2 failed with both halos), the classifier scores every
       sector graph (torch.inference_mode, CUDA when available), and each edge keeps the score of the sector that owns
       it (hnl.trackml.sectors.stitch_scores: every edge scored exactly once). Output, 50 events per file:
       <out>/scores_<first>.parquet (event, a, b int32 with a < b, score float32, y bool; zstd, one row group per
       event) + a JSON sidecar (checkpoint and its sha256, layout, per-event seconds and peak GPU memory, rows of
       truth-filtered conditions). Hit indices are the event store's (the full event's rows in condition full).
       --oracle SIGMA (plumbing and timing tests on dev events only, never a result): no network; each edge gets
       sigmoid(mu + N(0, SIGMA)) with mu = --oracle-true (default 2.2) for true and --oracle-fake (default -4.6) for
       fake edges (with SIGMA 1: about 1.4 % of true edges below 0.5, 0.8 % of fakes above 0.1, 24 % above 0.02),
       seeded per event.
g2     Gate G2 on a few val events (first-epoch checkpoint): the stitched scores against one full-event forward on CPU
       (needs a 64 GB job): the fraction of edges with |delta score| > dscore must be <= max_frac_dscore, and the target
       DM efficiency of the configured builders must agree within max_dmeff_points (in points).

    python -m hnl.trackml.infer score --store DIR/valset --ckpt CKPT --out DIR [--sectors inference|g2_fallback|full]
        [--device cuda] [--events SET] [--block K]
    python -m hnl.trackml.infer g2 --store DIR/valset --ckpt CKPT --out DIR [--events 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from hnl.overlay import grant as GRANT
from hnl.trackml import classifier as CL
from hnl.trackml import config as CFG
from hnl.trackml import sectors as S


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt: str, device: str):
    from hnl.train.acorn_runner import checkpoint_class
    cls = checkpoint_class(ckpt)
    model = cls.load_from_checkpoint(ckpt, map_location=torch.device("cpu"))
    return model.eval().to(device)


def sector_layout(cfg: dict, name: str) -> S.SectorSpec:
    """A layout of the config's sectors block, or "full": the whole event in one forward (hnl.trackml.sectors.FULL_EVENT,
    the reference of gate G2)."""
    if name == "full":
        return S.FULL_EVENT
    return S.SectorSpec.from_config(cfg["sectors"][name])


def event_items(ev: dict, spec: S.SectorSpec, scales=(1000.0, 3.14, 1000.0)):
    """All sector graphs of one stored event: [(Data, Sector)] from one split."""
    eta = S.hit_eta(ev["r"], ev["z"])
    secs = S.split(ev["phi"], eta, ev["pairs"], spec)
    out = []
    for sec in secs:
        h, e = sec.hits, sec.edges
        from torch_geometric.data import Data
        d = Data(hit_r=torch.from_numpy(ev["r"][h].astype(np.float32)) / scales[0],
                 hit_phi=torch.from_numpy(ev["phi"][h].astype(np.float32)) / scales[1],
                 hit_z=torch.from_numpy(ev["z"][h].astype(np.float32)) / scales[2],
                 edge_index=torch.from_numpy(sec.local_edges.astype(np.int64)))
        d.num_nodes = int(len(h))
        out.append((d, sec))
    return out


@torch.inference_mode()
def score_event(model, ev: dict, spec: S.SectorSpec, device: str, chunked: bool = False) -> tuple[np.ndarray, dict]:
    """Stitched per-edge scores (float32, the event's edge order) and per-sector statistics. chunked: the
    memory-bounded forward of hnl.trackml.chunked (CPU) instead of model(d)."""
    items = event_items(ev, spec, tuple(float(s) for s in model.hparams.get("node_scales", [1000.0, 3.14, 1000.0])))
    scores, stats = [], []
    for d, sec in items:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if not sec.n_edges:
            out = np.zeros(0, np.float32)
        elif chunked:
            from hnl.trackml import chunked as CH
            out = torch.sigmoid(CH.forward(model, d)).float().numpy()
        else:
            out = torch.sigmoid(model(d.to(device))).float().cpu().numpy()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        stats.append({"sector": sec.index, "hits": sec.n_hits, "edges": sec.n_edges, "owned": int(sec.owned.sum()),
                      "seconds": round(time.time() - t0, 3),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.startswith("cuda") else None})
        scores.append(out)
    return S.stitch_scores(ev["pairs"].shape[1], [s for _, s in items], scores), {"sectors": stats}


def write_scores(path: str, rows: dict[int, dict], meta: dict) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hnl.trackml.convert import storage_check
    schema = pa.schema([("event", pa.int32()), ("a", pa.int32()), ("b", pa.int32()), ("score", pa.float32()),
                        ("y", pa.bool_())])
    nbytes = sum(r["a"].size * 13 for r in rows.values())
    st = storage_check(nbytes) if path.startswith("/ceph/") else None
    tmp = path + ".tmp"
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=9,
                          column_encoding={"a": "DELTA_BINARY_PACKED", "b": "DELTA_BINARY_PACKED",
                                           "score": "BYTE_STREAM_SPLIT"}, use_dictionary=["event"]) as w:
        for e in sorted(rows):
            r = rows[e]
            t = pa.Table.from_arrays([pa.array(np.full(len(r["a"]), e, dtype=np.int32)), pa.array(r["a"].astype(np.int32)),
                                      pa.array(r["b"].astype(np.int32)), pa.array(r["score"].astype(np.float32)),
                                      pa.array(r["y"].astype(bool))], schema=schema)
            w.write_table(t, row_group_size=max(t.num_rows, 1))
    os.replace(tmp, path)
    side = {**meta, "events": {str(e): int(len(rows[e]["a"])) for e in sorted(rows)}, "bytes": os.path.getsize(path),
            "storage": st}
    with open(path + ".json", "w") as fh:
        json.dump(side, fh)
    return side


def cmd_score(a, cfg):
    from acorn.utils.loading_utils import load_datafiles_in_dir
    from hnl.trackml.events import event_from_eid
    if (a.ckpt is None) == (a.oracle is None):
        raise SystemExit("give exactly one of --ckpt and --oracle")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    CFG.threads()
    spec = sector_layout(cfg, a.sectors)
    paths = sorted(load_datafiles_in_dir(os.path.dirname(a.store.rstrip("/")), os.path.basename(a.store.rstrip("/"))))
    if a.events:            # a named subset of a larger store (e.g. merge_train or dev in the S4 training store)
        want = set(CFG.event_set(a.events, cfg))
        paths = [p for p in paths if event_from_eid(int(os.path.basename(p)[5:].split(".")[0])) in want]
        if len(paths) != len(want):
            raise SystemExit(f"store {a.store}: {len(paths)} of the {len(want)} events of {a.events}")
    if a.block is not None:  # one block of 50 events (the blocks of an unsplit run, same output files)
        paths = paths[50 * a.block:50 * a.block + 50]
        if not paths:
            raise SystemExit(f"block {a.block}: no events")
    GRANT.check([event_from_eid(int(os.path.basename(p)[5:].split(".")[0])) for p in paths], ("model", "plumb"),
                a.condition, out=a.out)        # overlay ids only under a grant (addendum H)
    model = None if a.oracle is not None else load_model(a.ckpt, device)
    os.makedirs(a.out, exist_ok=True)
    meta = {"ckpt": a.ckpt, "ckpt_sha256": _sha256(a.ckpt) if a.ckpt else None, "oracle_sigma": a.oracle,
            "oracle_means": [a.oracle_true, a.oracle_fake] if a.oracle is not None else None,
            "layout": spec.to_dict(), "device": device,
            "store": a.store, "condition": a.condition, "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "forward": "chunked" if a.chunked else "model"}
    for k in range(0, len(paths), 50):
        rows, per = {}, []
        for p in paths[k:k + 50]:
            ev = CL.load_event_graph(p)
            t0 = time.time()
            e = event_from_eid(ev["eid"])
            if model is None:
                if e not in set(range(1000, 1100)):
                    raise SystemExit("--oracle scores are for the dev events 1000-1099 only")
                rng = np.random.default_rng([e, 99])
                z = np.where(ev["y"], a.oracle_true, a.oracle_fake) + rng.normal(0.0, a.oracle, len(ev["y"]))
                sc, st = (1.0 / (1.0 + np.exp(-z))).astype(np.float32), {"sectors": []}
            else:
                sc, st = score_event(model, ev, spec, device, chunked=a.chunked)
            full = len(ev["rows"]) == len(ev["r"]) and np.array_equal(ev["rows"], np.arange(len(ev["r"])))
            if full != (a.condition == "full"):
                raise SystemExit(f"event {e}: the store's hits do not match condition {a.condition}")
            rows[e] = {"a": ev["pairs"][0], "b": ev["pairs"][1], "score": sc, "y": ev["y"]}
            per.append({"event": e, "seconds": round(time.time() - t0, 2), **st})
        first = min(rows)
        side = write_scores(os.path.join(a.out, f"scores_{first:05d}.parquet"), rows, {**meta, "per_event": per})
        print(json.dumps({"file": f"scores_{first:05d}.parquet", "events": len(rows), "bytes": side["bytes"],
                          "max_peak_gb": max([s["peak_gb"] or 0 for q in per for s in q["sectors"]] or [0]),
                          "mean_seconds": float(np.mean([q["seconds"] for q in per]))}), flush=True)


def cmd_g2(a, cfg):
    from acorn.utils.loading_utils import load_datafiles_in_dir
    from hnl.trackml.events import event_from_eid, load_events
    from hnl.trackml.trackbuild import EventState, build
    from hnl.trackml.evaluate import event_counts
    g2 = cfg["classifier"]["gate_g2"]
    CFG.threads()
    paths = sorted(load_datafiles_in_dir(os.path.dirname(a.store.rstrip("/")),
                                         os.path.basename(a.store.rstrip("/"))))[:int(a.events or g2["events"])]
    GRANT.check([event_from_eid(int(os.path.basename(p)[5:].split(".")[0])) for p in paths], "model", "full",
                out=a.out)          # overlay ids only under a grant (addendum H; never plumb, D6)
    model = load_model(a.ckpt, "cpu")
    spec = sector_layout(cfg, a.sectors)
    cfgs = [tuple([c.split(",")[0], *[float(x) for x in c.split(",")[1:]]]) for c in g2["configs"]]
    rows = []
    for p in paths:
        ev = CL.load_event_graph(p)
        e = event_from_eid(ev["eid"])
        t0 = time.time()
        stitched, _ = score_event(model, ev, spec, "cpu")
        t_st = time.time() - t0
        from torch_geometric.data import Data
        d = Data(hit_r=torch.from_numpy(ev["r"].astype(np.float32)) / 1000.0,
                 hit_phi=torch.from_numpy(ev["phi"].astype(np.float32)) / 3.14,
                 hit_z=torch.from_numpy(ev["z"].astype(np.float32)) / 1000.0,
                 edge_index=torch.from_numpy(ev["pairs"].astype(np.int64)))
        d.num_nodes = len(ev["r"])
        t0 = time.time()
        with torch.inference_mode():
            full = torch.sigmoid(model(d)).float().numpy()
        t_full = time.time() - t0
        dsc = np.abs(stitched - full)
        evt = load_events([e])[e]
        out = {"event": e, "edges": int(len(full)), "frac_dscore": float((dsc > float(g2["dscore"])).mean()),
               "max_dscore": float(dsc.max()), "stitched_s": round(t_st, 1), "full_s": round(t_full, 1)}
        for c in cfgs:
            effs = []
            for sc in (stitched, full):
                st = EventState(evt, ev["pairs"][0], ev["pairs"][1], sc)
                cnt, _ = event_counts(evt.pid, build(c, st), evt.particles)
                effs.append(cnt["m_target"] / max(cnt["n_target"], 1))
            out[f"dmeff_points_{c[0]}"] = 100.0 * (effs[0] - effs[1])
        rows.append(out)
        print(json.dumps(out), flush=True)
    df = pd.DataFrame(rows)
    ok_frac = bool((df.frac_dscore <= float(g2["max_frac_dscore"])).all())
    ok_eff = bool(all((df[c].abs() <= float(g2["max_dmeff_points"])).all() for c in df if c.startswith("dmeff_points")))
    res = {"ckpt": a.ckpt, "layout": spec.to_dict(), "per_event": rows, "gate_g2_pass": ok_frac and ok_eff,
           "criteria": g2, "slurm_job": os.environ.get("SLURM_JOB_ID")}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"gate_g2_{a.sectors}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({"gate_g2_pass": res["gate_g2_pass"], "frac_ok": ok_frac, "eff_ok": ok_eff}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--store", required=True, help="event-store split directory, e.g. $LOCAL/store/valset")
    s.add_argument("--ckpt", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--sectors", default="inference", choices=["inference", "g2_fallback", "full"])
    s.add_argument("--device", default=None)
    s.add_argument("--oracle", type=float, default=None, help="truth-based scores with this logit noise (dev events, "
                   "plumbing and timing tests only)")
    s.add_argument("--oracle-true", type=float, default=2.2)
    s.add_argument("--oracle-fake", type=float, default=-4.6)
    s.add_argument("--condition", default="full", choices=["full", "nonoise", "1gev"])
    s.add_argument("--events", default=None, help="score only the store's events of this set (not a test set)")
    s.add_argument("--block", type=int, default=None, help="score only block K of 50 events (sorted store order)")
    s.add_argument("--chunked", action="store_true", help="memory-bounded whole-event forward (hnl.trackml.chunked; "
                   "CPU, --sectors full)")
    g = sub.add_parser("g2")
    g.add_argument("--store", required=True)
    g.add_argument("--ckpt", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--sectors", default="inference", choices=["inference", "g2_fallback"])
    g.add_argument("--events", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = CFG.load(a.config)
    {"score": cmd_score, "g2": cmd_g2}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
