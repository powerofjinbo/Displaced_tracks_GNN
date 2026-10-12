"""Gate G0 on an A30 (docs/trackml_plan.md section 8: smoke test and timing before S1-S4; about 2 GPU-h). Untrained
networks only: memory and time do not depend on the weights.

base-check  The base graph built with FRNN on CUDA (as the event store at S4 / S5) against the CPU cKDTree build (as the
            union additions of S3) from the same events: pair counts, symmetric difference and Jaccard, for the
            embedding run on CUDA and on CPU. Accept S3's CPU additions only if the difference is negligible.
sectors     Peak GPU memory (torch.cuda.max_memory_allocated) and time of the classifier on every INFERENCE sector
            (forward, inference mode) and on training windows (forward + backward with activation checkpointing,
            no optimizer step) of event-store files, e.g. those written by hnl.trackml.probe graphs (ttbar embedding at
            the plan's density) or by hnl.trackml.graphs stage (trained embedding). The verdict uses the A30's 24 GiB
            minus a 1 GiB margin.

    python -m hnl.trackml.g0 base-check --events 1000,1001,1002 --ml-ckpt CKPT --r 0.0686 --knn 300 --out DIR
    python -m hnl.trackml.g0 sectors --store DIR/valset --out DIR [--windows 16] [--ckpt CKPT]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from hnl.trackml import events as E
from hnl.trackml import sectors as S

A30_USABLE_GB = 24 * 2**30 / 1e9 - 1.0


def cmd_base_check(a):
    rows = []
    gpu = E.Embedder(a.ml_ckpt, "cuda")
    cpu = E.Embedder(a.ml_ckpt, "cpu")
    for ev in E.iter_events([int(x) for x in a.events.split(",")], particles=False):
        eg = gpu(ev)
        ec = cpu(ev)
        t0 = time.time()
        b_frnn = E.radius_graph(eg, a.r, a.knn)
        t_frnn = time.time() - t0
        t0 = time.time()
        b_cpu = E.radius_graph(ec.cpu(), a.r, a.knn)
        t_cpu = time.time() - t0
        b_cpu_gpuemb = E.radius_graph(eg.cpu(), a.r, a.knn)
        n = ev.n
        code = lambda p: p[0].astype(np.int64) * n + p[1]
        for name, other in (("cpu_embedding", b_cpu), ("gpu_embedding", b_cpu_gpuemb)):
            x, y = code(b_frnn), code(other)
            inter = np.intersect1d(x, y).size
            rows.append({"event": ev.event, "compare": f"frnn_cuda vs ckdtree ({name})", "pairs_frnn": int(len(x)),
                         "pairs_ckdtree": int(len(y)), "only_frnn": int(len(x) - inter), "only_ckdtree": int(len(y) - inter),
                         "jaccard": inter / max(len(np.union1d(x, y)), 1), "frnn_s": round(t_frnn, 2),
                         "ckdtree_s": round(t_cpu, 1),
                         "max_abs_embedding_diff": float((eg.cpu() - ec).abs().max())})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "g0_base_check.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def cmd_sectors(a):
    import yaml
    from acorn.utils.loading_utils import load_datafiles_in_dir
    from hnl.trackml import classifier as CL
    from hnl.trackml.probe import CLASSIFIER_CONFIG
    dev = torch.device("cuda")
    if a.ckpt:
        from hnl.trackml.infer import load_model
        model = load_model(a.ckpt, "cuda")
    else:
        cfg = yaml.safe_load(open(CLASSIFIER_CONFIG))
        hp = CL.model_hparams({**cfg, "input_dir": "/nonexistent", "stage_dir": "/nonexistent", "data_split": [0, 0, 0]})
        torch.manual_seed(0)
        model = CL.TrackMLCanto(hp).to(dev)
    paths = sorted(load_datafiles_in_dir(os.path.dirname(a.store.rstrip("/")), os.path.basename(a.store.rstrip("/"))))
    rows = []
    for p in paths[:a.max_events]:
        ev = CL.load_event_graph(p)
        eta = S.hit_eta(ev["r"], ev["z"])
        jobs = [("infer", S.INFERENCE, s) for s in range(S.INFERENCE.n_sectors)]
        jobs += [("train", S.window_spec(-np.pi + (j + 0.5) * 2 * np.pi / a.windows), 0) for j in range(a.windows)]
        for mode, spec, s in jobs:
            d, sec = CL.sector_item(ev, spec, s)
            d = d.to(dev)
            cand = S.candidate_triplets(sec.n_hits, sec.local_edges)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                if mode == "infer":
                    model.eval()
                    with torch.inference_mode():
                        model(d)
                else:
                    model.train()
                    out = model(d)
                    model.loss_function(out, d)[0].backward()
                    model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                peak = (torch.cuda.max_memory_allocated() - base) / 1e9
                status = "ok"
            except torch.OutOfMemoryError:
                peak, status = float("nan"), "oom"
                model.zero_grad(set_to_none=True)
            rows.append({"file": os.path.basename(p), "mode": mode, "sector": s, "center_deg": float(np.degrees(
                S.sector_bounds(spec, s)["phi_center"])), "hits": sec.n_hits, "pairs": sec.n_edges,
                "owned": int(sec.owned.sum()), "candidate_triplets": cand, "peak_gb": peak,
                "reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "seconds": round(time.time() - t0, 3),
                "status": status, "fits_a30": bool(status == "ok" and peak < A30_USABLE_GB)})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "g0_sectors.json"), "w") as fh:
        json.dump({"rows": rows, "gpu": torch.cuda.get_device_name(0), "a30_usable_gb": A30_USABLE_GB,
                   "slurm_job": os.environ.get("SLURM_JOB_ID")}, fh, indent=1, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base-check")
    b.add_argument("--events", default="1000,1001,1002")
    b.add_argument("--ml-ckpt", required=True)
    b.add_argument("--r", type=float, required=True)
    b.add_argument("--knn", type=int, default=300)
    b.add_argument("--out", required=True)
    s = sub.add_parser("sectors")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--windows", type=int, default=16)
    s.add_argument("--max-events", type=int, default=2)
    a = ap.parse_args(argv)
    {"base-check": cmd_base_check, "sectors": cmd_sectors}[a.cmd](a)


if __name__ == "__main__":
    main()
