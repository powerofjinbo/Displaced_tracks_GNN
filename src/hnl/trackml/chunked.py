"""Memory-bounded whole-event forward of the SymmetricTripletIGNN edge classifier (inference only).

The model's own forward (hnl.models.invariant_gnn.SymmetricTripletIGNN.forward) enumerates every unordered pair of
edges sharing a hit, keeps the k straightest per edge end, and runs the triplet messages over all kept triplets at
once. On accuracy-phase TrackML events (T2) that needs 140-255 GB (DECISIONS 2026-10-11). This forward computes the
same function in pieces:
  * triplets are enumerated by GF.build_triplets on the directed edges incident to a range of consecutive middle hits,
    and only triplets whose middle hit lies in the range are kept. build_triplets orders triplets by middle hit, then
    by in-edge and out-edge position, so the ranges reproduce the full enumeration in the same order. The top-k
    selection groups triplets by (edge, end at the middle hit), so every group lies inside one range and the kept
    triplets are the same ones, in the same order, ties included;
  * the triplet encoder, the triplet messages and every per-row MLP run on blocks of rows; the triplet messages are
    accumulated into the edge sums block after block (scatter_add with out=), in the original triplet order.
The parameters, inputs and arithmetic are the model's; the results can differ from the one-shot forward only by
floating-point blocking in the matrix products. `check` compares against stored whole-event scores.

    python -m hnl.trackml.chunked check --store STORE --scores SCORES_DIR --ckpt CKPT [--events N] [--condition C]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch_scatter import scatter_add

MAX_TRIPLETS = 8_000_000   # enumerated triplets per range of middle hits (before top-k)
ROWS = 1_000_000           # rows per block of an MLP or of the triplet messages


def _ranges(weight: torch.Tensor, cap: int) -> list[tuple[int, int]]:
    """Consecutive index ranges [lo, hi) with sum(weight) <= cap (a single heavier index gets its own range)."""
    cum = torch.cumsum(weight, 0).tolist()
    out, lo, base = [], 0, 0
    for i, c in enumerate(cum):
        if c - base > cap and i > lo:
            out.append((lo, i))
            lo, base = i, cum[i - 1]
    out.append((lo, len(cum)))
    return out


def _rows(fn, n: int, rows: int = ROWS) -> torch.Tensor:
    """fn(slice) for consecutive slices of n rows, concatenated."""
    return torch.cat([fn(slice(s, min(s + rows, n))) for s in range(0, n, rows)], dim=0)


def kept_triplets(model, xyz: torch.Tensor, a: torch.Tensor, b: torch.Tensor, n: int, cap: int = MAX_TRIPLETS):
    """(e1, e2, tf) of the triplets the model keeps (hnl.models.invariant_gnn.undirected_triplets,
    symmetric_triplet_features, topk_per_edge_end), computed range by range of middle hits."""
    from hnl.models import geometry_features as GF
    from hnl.models import invariant_gnn as IG
    E = a.shape[0]
    eid = torch.arange(E)
    src2, dst2, eid2 = torch.cat([a, b]), torch.cat([b, a]), torch.cat([eid, eid])
    deg_in = torch.bincount(dst2, minlength=n)
    deg_out = torch.bincount(src2, minlength=n)
    k = int(model.hparams["triplet_topk"])
    E1, E2, TF = [], [], []
    for lo, hi in _ranges(deg_in * deg_out, cap):
        sub = torch.nonzero(((dst2 >= lo) & (dst2 < hi)) | ((src2 >= lo) & (src2 < hi))).squeeze(1)   # in order
        s_src, s_dst = src2[sub], dst2[sub]
        e_in, e_out = GF.build_triplets(s_src, s_dst, n)
        mid = s_dst[e_in]
        inside = (mid >= lo) & (mid < hi)
        e_in, e_out = sub[e_in[inside]], sub[e_out[inside]]       # directed-edge ids of the full doubled list
        e1, e2 = eid2[e_in], eid2[e_out]
        keep = e1 < e2
        e_in, e_out = e_in[keep], e_out[keep]
        e1, e2 = eid2[e_in], eid2[e_out]
        i, j, kk = src2[e_in], dst2[e_in], dst2[e_out]
        tf = IG.symmetric_triplet_features(xyz, i, j, kk)
        sel = IG.topk_per_edge_end(e1, e2, j, a, tf[:, 3] + 0.5 * tf[:, 2].clamp(max=10.0), k)
        E1.append(e1[sel])
        E2.append(e2[sel])
        TF.append(tf[sel])
    if not E1:
        z = torch.zeros(0, dtype=torch.long)
        return z, z, torch.zeros(0, 5, dtype=xyz.dtype)
    return torch.cat(E1), torch.cat(E2), torch.cat(TF)


@torch.inference_mode()
def forward(model, batch, cap: int = MAX_TRIPLETS, rows: int = ROWS) -> torch.Tensor:
    """Edge logits of SymmetricTripletIGNN for one graph, as model(batch), with bounded memory (CPU)."""
    from hnl.models import invariant_gnn as IG
    if type(model).__name__ != "TrackMLCanto" and not isinstance(model, IG.SymmetricTripletIGNN):
        raise TypeError(f"chunked forward is written for SymmetricTripletIGNN, not {type(model).__name__}")
    a, b = batch.edge_index
    r, z, xyz = model._xyz(batch)
    n = xyz.shape[0]
    E = a.shape[0]
    x0 = (torch.ones(n, 1, dtype=torch.float32) if model.detector_input == "none" else IG._region(r, z))
    x = model.node_encoder(x0)
    d = xyz[b] - xyz[a]
    lt = torch.linalg.vector_norm(d[:, :2], dim=1).clamp_min(1e-6)
    l3 = torch.linalg.vector_norm(d, dim=1).clamp_min(1e-6)
    geo = torch.stack([torch.log(lt / 10.0), torch.asinh(d[:, 2].abs() / lt), torch.log(l3 / 10.0)], dim=1).float()
    del d, lt, l3
    e = _rows(lambda s: model.edge_encoder(torch.cat([x[a[s]] + x[b[s]], x[a[s]] * x[b[s]], geo[s]], dim=1)), E, rows)
    del geo
    e1, e2, tf = kept_triplets(model, xyz, a, b, n, cap)
    ts = IG._scale_trip_sym(tf).float()
    del tf
    T = ts.shape[0]
    for _ in range(model.hparams["n_graph_iters"]):
        agg = scatter_add(e, a, dim=0, dim_size=n) + scatter_add(e, b, dim=0, dim_size=n)
        x = x + model.node_network(torch.cat([x, agg], dim=1))
        tm1 = torch.zeros_like(e)
        tm2 = torch.zeros_like(e)
        for s0 in range(0, T, rows):
            s = slice(s0, min(s0 + rows, T))
            t = model.trip_encoder(ts[s])
            m = model.trip_network(torch.cat([e[e1[s]] + e[e2[s]], e[e1[s]] * e[e2[s]], t], dim=1))
            scatter_add(m, e1[s], dim=0, out=tm1)
            scatter_add(m, e2[s], dim=0, out=tm2)
        tm = tm1 + tm2
        del tm1, tm2
        e = e + _rows(lambda s: model.edge_network(torch.cat([x[a[s]] + x[b[s]], x[a[s]] * x[b[s]], e[s], tm[s]], dim=1)),
                      E, rows)
        del tm
    return _rows(lambda s: model.output_edge_classifier(torch.cat([x[a[s]] + x[b[s]], x[a[s]] * x[b[s]], e[s]], dim=1)),
                 E, rows).squeeze(-1)


def cmd_check(a) -> None:
    """Chunked scores of the first events of a store against stored whole-event scores (same checkpoint); with
    --also-original, the model's one-shot forward too (memory permitting)."""
    import pyarrow.parquet as pq
    from acorn.utils.loading_utils import load_datafiles_in_dir
    from hnl.trackml import classifier as CL
    from hnl.trackml import infer as INF
    from hnl.trackml.events import event_from_eid
    from hnl.overlay import grant as GRANT
    paths = sorted(load_datafiles_in_dir(os.path.dirname(a.store.rstrip("/")), os.path.basename(a.store.rstrip("/"))))
    # overlay events only under a grant of kind model (addendum H): never plumb, since this compares stored scores of
    # the events (no number from a pilot plumbing output, decision D6)
    GRANT.check_store(paths[:a.events], "model")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    model = INF.load_model(a.ckpt, "cpu")
    spec = INF.sector_layout({}, "full")
    scales = tuple(float(s) for s in model.hparams.get("node_scales", [1000.0, 3.14, 1000.0]))
    stored, sha = {}, INF._sha256(a.ckpt)
    for f in sorted(os.listdir(a.scores)):
        if f.endswith(".parquet") and f.startswith("scores_"):
            side = json.load(open(os.path.join(a.scores, f + ".json")))
            if side.get("ckpt_sha256") != sha:
                raise SystemExit(f"{f}: stored scores come from another checkpoint")
            stored[f] = side
    rows = []
    for p in paths[:a.events]:
        ev = CL.load_event_graph(p)
        e = event_from_eid(ev["eid"])
        f = [k for k, s in stored.items() if str(e) in s["events"]]
        ref = pq.read_table(os.path.join(a.scores, f[0]), filters=[("event", "=", e)]).to_pandas()
        if not (np.array_equal(ref.a.to_numpy(), ev["pairs"][0]) and np.array_equal(ref.b.to_numpy(), ev["pairs"][1])):
            raise SystemExit(f"event {e}: stored edge order differs from the store")
        (d, sec), = INF.event_items(ev, spec, scales)
        t0 = time.time()
        sc = torch.sigmoid(forward(model, d, a.cap, a.rows)).float().numpy()
        out = S_stitch(ev, sec, sc)
        dt = time.time() - t0
        diff = np.abs(out - ref.score.to_numpy())
        row = {"event": e, "edges": int(len(out)), "seconds_chunked": round(dt, 1), "max_abs_diff": float(diff.max()),
               "n_diff_gt_1e-6": int((diff > 1e-6).sum()), "n_identical": int((diff == 0).sum()),
               "threshold_flips": {str(t): int(((out > t) != (ref.score.to_numpy() > t)).sum())
                                   for t in (0.1, 0.2, 0.3, 0.5, 0.8)}}
        if a.also_original:
            t0 = time.time()
            with torch.inference_mode():
                orig = torch.sigmoid(model(d)).float().numpy()
            o = S_stitch(ev, sec, orig)
            row.update({"seconds_original": round(time.time() - t0, 1),
                        "orig_vs_stored_max_abs_diff": float(np.abs(o - ref.score.to_numpy()).max()),
                        "chunked_vs_orig_max_abs_diff": float(np.abs(out - o).max())})
        print(json.dumps(row), flush=True)
        rows.append(row)
    print(json.dumps({"events": len(rows), "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                      "edges": sum(r["edges"] for r in rows),
                      "edges_diff_gt_1e-6": sum(r["n_diff_gt_1e-6"] for r in rows)}), flush=True)


def S_stitch(ev: dict, sec, sc: np.ndarray) -> np.ndarray:
    """Scores in the event's edge order for the single whole-event sector (as hnl.trackml.infer.score_event)."""
    from hnl.trackml import sectors as S
    return S.stitch_scores(ev["pairs"].shape[1], [sec], [sc])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--store", required=True)
    c.add_argument("--scores", required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--events", type=int, default=5)
    c.add_argument("--cap", type=int, default=MAX_TRIPLETS)
    c.add_argument("--rows", type=int, default=ROWS)
    c.add_argument("--also-original", action="store_true")
    a = ap.parse_args(argv)
    {"check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
