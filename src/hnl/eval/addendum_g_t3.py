"""Preregistration addendum G, step T3 (docs/preregistration_new_sample.md section 5.2; plan docs/addendum_g_t3_plan.md):
the retrained models acorn (ttbar+G) and Canto-L (ttbar+G), trained on arm g (hnl.train.arm_g), and their selection on
the generic val split. No test event of any sample is read here.

  scan        metric-learning radius scan on the arm's own val mix (staged by hnl.train.arm_g): the rule of
              hnl.graphs.ml_scan (smallest r on its grid with >= 99 % of all true edges, the smallest knn on its grid
              that never binds, else < 5 %), on the same grids and with the same exact-distance computation; the
              per-population curves of ml_scan need the HNL truth tables and are replaced by per-sample curves
  rk          the (r, knn) of a scan.json, or a stop message (as slurm/baseline/ml_infer.sbatch)
  grid        builder configurations on the generic_iso val events of a run's scored val graphs, block k of n: the
              menu of the model (acorn: Stage-0 menu of feedback round 3 + CC on the 0.01 grid; Canto-L: the frozen
              selection's menu), matching as hnl.eval.trackbuild_val / hnl.eval.evaluate_gnn, particle rows for the
              targets and gun particles (T1 truth)
  grid-merge  the blocks of one grid, checked complete
  select      per run and family the builder rule at FR* (P_G efficiency, fake rate on generic_iso val, one-standard-
              error tie rule); per model the family by the same rule on 3-seed means; writes the selection file
  readout     the selected configurations on generic_dense val and on the val splits of ttbar and the HNL samples
              (W muons, ttbar prompt; addendum G section 7)

    python -m hnl.eval.addendum_g_t3 scan --ckpt CKPT --input-dir DIR --out RUN/ml_scan
    python -m hnl.eval.addendum_g_t3 grid --model acorn --seed 0 --block 0 --n-blocks 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
RUNS = os.path.join(DATA, "runs")
ACORN = os.path.join(RUNS, "baseline_g1")                    # acorn (ttbar+G): ACORN/armg_s<S>
CANTO_GRAPHS = os.path.join(RUNS, "baseline_g1_ts")          # Canto-L pieces: triplet support, union graphs
CANTO = os.path.join(RUNS, "stage1_g1", "l1tl")              # Canto-L (ttbar+G) classifier runs: CANTO/armg_s<S>
T1 = os.path.join(DATA, "derived", "generated_v1", "t1")
FR_STAR = 0.02936834277897644                                # results/working_points_v1.json FR_star
SEEDS = (0, 1, 2)
LABELS = {"acorn": "acorn (ttbar+G)", "canto": "Canto-L (ttbar+G)"}
ACORN_MENU = os.path.join(REPO, "configs", "addendum_g", "acorn_menu_v3.json")
CANTO_BUILDERS = ("walkmergeX", "mwslayermergeX", "mws2passmergeX", "cc", "mwslayer", "mws2pass", "mwslmerge")
CC01 = [round(float(c), 2) for c in np.arange(0.01, 0.995, 0.01)]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _require_generated():
    from hnl.data import acorn_store as AS
    if "generic_iso" not in AS.SAMPLE_CODE:
        raise SystemExit("needs the generated samples: run with HNL_GENERATED=1")


def run_dir(model: str, seed: int) -> str:
    return os.path.join(ACORN if model == "acorn" else CANTO, f"armg_s{seed}")


# ------------------------------------------------------------------------------------------------- radius scan
def cmd_scan(a) -> None:
    """hnl.graphs.ml_scan's embedding loop and selection rule without its HNL truth tables."""
    import torch
    from acorn.stages.graph_construction import MetricLearning
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config
    from hnl.graphs.ml_scan import K_GRID, R_GRID

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda")
    model = MetricLearning.load_from_checkpoint(a.ckpt, map_location="cpu")
    n_val = len([f for f in os.listdir(os.path.join(a.input_dir, "valset")) if f.endswith(".pyg.gz")])
    over = {"input_dir": a.input_dir, "stage_dir": os.path.join(a.out, "_unused"), "data_split": [0, n_val, 0],
            "hard_cuts": {}, "undirected": True, "num_workers": [2, 2, 2]}
    model._hparams = add_variable_name_prefix_in_config({**copy.deepcopy(dict(model._hparams)), **over})
    model.setup(stage="predict")
    model = model.to(dev).eval()
    eff_rows, size_rows = [], []
    with torch.inference_mode():
        for batch in model.val_dataloader():
            batch = batch.to(dev)
            eid = batch.event_id[0] if isinstance(batch.event_id, (list, tuple)) else batch.event_id
            eid = str(eid[0] if isinstance(eid, (list, tuple)) else eid)
            emb = model.apply_embedding(batch)
            n = emb.shape[0]
            te = batch.track_edges
            d_true = torch.linalg.vector_norm(emb[te[0]] - emb[te[1]], dim=1)
            D = torch.cdist(emb, emb)
            iu = torch.triu_indices(n, n, 1, device=dev)
            dpairs = torch.sort(D[iu[0], iu[1]]).values
            rg = torch.tensor(R_GRID, device=dev, dtype=D.dtype)
            n_pairs = torch.searchsorted(dpairs, rg, right=False)
            counts = (D.unsqueeze(0) < rg.view(-1, 1, 1)).sum(dim=2) - 1
            eff_rows.append(pd.DataFrame({"event_id": eid, "d_true": d_true.cpu().numpy()}))
            size_rows.append(pd.DataFrame({"event_id": eid, "r": R_GRID, "n_pairs": n_pairs.cpu().numpy(),
                                           "n_nodes": n, "max_ni": counts.max(dim=1).values.cpu().numpy(),
                                           **{f"bind_{k}": (counts >= k).float().mean(dim=1).cpu().numpy()
                                              for k in K_GRID}}))
    e = pd.concat(eff_rows, ignore_index=True)
    sizes = pd.concat(size_rows, ignore_index=True)
    code = e.event_id.str[0]
    curve = pd.DataFrame({"r": R_GRID})
    for name, df in (("all", e), ("ttbar", e[code == "1"]), ("generic", e[code.isin(["5", "6"])])):
        d = np.sort(df.d_true.to_numpy())
        curve[f"eff_{name}"] = np.searchsorted(d, R_GRID, side="left") / max(len(d), 1)
    agg = sizes.groupby("r").agg(pairs_per_event=("n_pairs", "mean"), max_ni=("max_ni", "max"),
                                 **{f"bind_{k}": (f"bind_{k}", "mean") for k in K_GRID}).reset_index()
    curve = curve.merge(agg, on="r")
    curve.to_csv(os.path.join(a.out, "radius_curve.csv"), index=False)
    ok = curve[curve.eff_all >= 0.99]
    r_star = float(ok.r.min()) if len(ok) else None
    sel = {"r_star_all_edges": r_star}
    if r_star is not None:
        row = curve[curve.r == r_star].iloc[0]
        k_never = [k for k in K_GRID if k > row.max_ni]
        k_5pct = [k for k in K_GRID if row[f"bind_{k}"] < 0.05]
        sel["knn_all_edges_never_binds"] = k_never[0] if k_never else None
        sel["knn_all_edges_binds_lt5pct"] = k_5pct[0] if k_5pct else None
        sel["r_star_all_edges_pairs_per_event"] = float(row.pairs_per_event)
    u, c = np.unique([x[0] for x in e.event_id.unique()], return_counts=True)
    meta = {"ckpt": a.ckpt, "n_events": int(e.event_id.nunique()), "events_per_code": dict(zip(u.tolist(), c.tolist())),
            "selection": sel, "elapsed_s": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "rule": "hnl.graphs.ml_scan: smallest grid r with eff_all >= 0.99, smallest knn never binding, else < 5 %"}
    with open(os.path.join(a.out, "scan.json"), "w") as f:
        json.dump(meta, f, indent=1)
    pd.set_option("display.width", 250)
    print(curve.round(4).to_string())
    print(json.dumps(meta, indent=1))


def cmd_rk(a) -> None:
    s = json.load(open(a.scan))["selection"]
    r = s.get("r_star_all_edges")
    k = s.get("knn_all_edges_never_binds") or s.get("knn_all_edges_binds_lt5pct")
    if r is None or k is None:
        raise SystemExit("no (r, knn) on the grid meets the preregistered rule: stop and decide by hand")
    if s["r_star_all_edges_pairs_per_event"] > 50000:
        raise SystemExit("graph above 50k pairs per event: filter decision needed (DECISIONS.md), stop and decide by hand")
    print(r, k)


# ------------------------------------------------------------------------------------------------- builder grids
def menu(model: str) -> list[tuple]:
    """Track-builder configurations (hnl.eval.trackbuild_val form) of a model's menu, without the CC cuts that
    hnl.eval.evaluate_gnn evaluates (acorn: CC on the 0.01 grid, as Stage 0's eval_val)."""
    from hnl.eval import trackbuild_val as TV
    if model == "acorn":
        return [tuple([c[0], *map(float, c[1:])]) for c in json.load(open(ACORN_MENU))["configs"]]
    seen, out = set(), []
    for c in TV.configs(list(CANTO_BUILDERS)):      # a family found in several grids keeps every distinct configuration
        key = (c[0], *map(float, c[1:]))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def cc_cuts(model: str) -> list[float]:
    return CC01 if model == "acorn" else []


def _keep(eids: list[str]) -> dict:
    """event_id -> sorted particle ids written to the per-particle tables: targets and gun particles (T1 truth)."""
    from hnl.data.acorn_store import SAMPLE_CODE
    out = {}
    for s in ("generic_iso", "generic_dense"):
        mine = [e for e in eids if int(e[0]) == SAMPLE_CODE[s]]
        if not mine:
            continue
        t = pd.read_parquet(os.path.join(T1, f"truth_{s}.parquet"), columns=["event", "particle_id", "is_target", "gun"])
        t = t[t.is_target | t.gun.fillna(False)]
        t["event_id"] = (np.int64(SAMPLE_CODE[s]) * 100_000_000 + t.event.to_numpy(np.int64)).astype(str)
        t = t[t.event_id.isin(set(mine))]
        out.update({e: np.sort(g.particle_id.to_numpy(np.int64)) for e, g in t.groupby("event_id")})
    return out


def _grid_chunk(args):
    paths, cfgs, cuts, keep = args
    from hnl.eval import evaluate_gnn as EG
    from hnl.eval import trackbuild_val as TV
    parts, evs = [], []
    if cfgs:
        p, e = TV._chunk((paths, cfgs, {}, keep))
        parts.append(p[["event_id", "particle_id", "config", "dm_matched", "n_sm_tracks"]])
        evs.append(e[["event_id", "config", "n_tracks_ge5", "n_fake"]])
    res = [EG.evaluate_event(p, cuts, {}, keep) for p in paths]
    ceil = pd.concat([r[2] for r in res], ignore_index=True)
    if cuts:
        off = len(cfgs)
        idx = {round(float(c), 4): off + k for k, c in enumerate(cuts)}
        p = pd.concat([r[0] for r in res], ignore_index=True)
        e = pd.concat([r[1] for r in res], ignore_index=True)
        p["config"] = [idx[round(float(c), 4)] for c in p.cut]
        e["config"] = [idx[round(float(c), 4)] for c in e.cut]
        parts.append(p[["event_id", "particle_id", "config", "dm_matched", "n_sm_tracks"]])
        evs.append(e[["event_id", "config", "n_tracks_ge5", "n_fake"]])
    return pd.concat(parts, ignore_index=True), pd.concat(evs, ignore_index=True), ceil


def grid_configs(model: str) -> list[list]:
    return [list(c) for c in menu(model)] + [["cc", c] for c in cc_cuts(model)]


def scored_val_tar(model: str, seed: int) -> str:
    return os.path.join(run_dir(model, seed), "gnn_scores", "valset.tar")


def merge_model(model: str, seed: int) -> str:
    return os.path.join(run_dir(model, seed), "lmerge", "model.joblib")


def cmd_grid(a) -> None:
    _require_generated()
    from hnl.data.acorn_store import SAMPLE_CODE
    tar = a.tar or scored_val_tar(a.model, a.seed)
    out = a.out or os.path.join(run_dir(a.model, a.seed), a.name)
    blk = os.path.join(out, "blocks", f"{a.block:03d}_of_{a.n_blocks:03d}")
    if os.path.exists(os.path.join(blk, "done")):
        print(f"{blk} is done")
        return
    mm = a.merge_model or merge_model(a.model, a.seed)
    if not os.path.exists(mm):
        raise SystemExit(f"no merge model {mm}")
    os.environ["HNL_MERGE_MODEL"] = os.path.abspath(mm)
    cfgs, cuts = menu(a.model), cc_cuts(a.model)
    code = str(SAMPLE_CODE[a.sample])
    local = os.path.join("/tmp", os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", "local"), "g3")
    os.makedirs(local, exist_ok=True)
    t0 = time.time()
    with tarfile.open(tar) as t:
        names = sorted(m for m in t.getnames() if m.endswith(".pyg.gz") and os.path.basename(m)[5:6] == code)
        sel = [names[i] for i in np.array_split(np.arange(len(names)), a.n_blocks)[a.block]][:a.max_events]
        for nm in sel:
            t.extract(nm, local, filter="data")
    paths = [os.path.join(local, nm) for nm in sel]
    eids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths]
    keep_all = _keep(eids)
    chunks = []
    for k in range(0, len(paths), 5):
        ch = paths[k:k + 5]
        keep = {e: keep_all[e] for e in eids[k:k + 5] if e in keep_all}
        chunks.append((ch, cfgs, cuts, keep))
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_grid_chunk, chunks))
    os.makedirs(blk, exist_ok=True)
    for i, nm in enumerate(("per_particle", "per_event", "ceiling")):
        pd.concat([r[i] for r in res], ignore_index=True).to_parquet(os.path.join(blk, f"{nm}.parquet"), index=False)
    prov = {"model": a.model, "seed": a.seed, "sample": a.sample, "scored_graphs": tar, "block": a.block,
            "n_blocks": a.n_blocks, "n_events": len(sel), "merge_model": mm, "merge_model_sha256": _sha256(mm),
            "n_configs": len(cfgs) + len(cuts), "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename, "elapsed_s": round(time.time() - t0, 1), "finished_utc": _now()}
    with open(os.path.join(blk, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    open(os.path.join(blk, "done"), "w").close()
    shutil.rmtree(local, ignore_errors=True)
    print(json.dumps({k: prov[k] for k in ("model", "seed", "sample", "block", "n_events", "n_configs", "elapsed_s")}))


def cmd_grid_merge(a) -> None:
    _require_generated()
    from hnl.data.acorn_store import SAMPLE_CODE
    out = os.path.join(run_dir(a.model, a.seed), a.name)
    blocks = sorted(glob.glob(os.path.join(out, "blocks", "*")))
    n = {int(os.path.basename(b).split("_of_")[1]) for b in blocks}
    if len(n) != 1 or len(blocks) != n.pop() or not all(os.path.exists(os.path.join(b, "done")) for b in blocks):
        raise SystemExit(f"{out}: incomplete blocks")
    tabs = {nm: pd.concat([pd.read_parquet(os.path.join(b, f"{nm}.parquet")) for b in blocks], ignore_index=True)
            for nm in ("per_particle", "per_event", "ceiling")}
    code = str(SAMPLE_CODE[a.sample])
    with tarfile.open(scored_val_tar(a.model, a.seed)) as t:
        n_graphs = sum(1 for m in t.getnames() if m.endswith(".pyg.gz") and os.path.basename(m)[5:6] == code)
    cfgs = grid_configs(a.model)
    pe = tabs["per_event"]
    if pe.event_id.nunique() != n_graphs or len(pe) != n_graphs * len(cfgs):
        raise SystemExit(f"{out}: {pe.event_id.nunique()} events x configs {len(pe)}, expected {n_graphs} x {len(cfgs)}")
    for nm, df in tabs.items():
        df.to_parquet(os.path.join(out, f"{nm}.parquet"), index=False)
    with open(os.path.join(out, "configs.json"), "w") as f:
        json.dump(cfgs, f)
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump({"blocks": [json.load(open(os.path.join(b, "provenance.json"))) for b in blocks],
                   "n_events": n_graphs, "merged_utc": _now()}, f, indent=1)
    open(os.path.join(out, "done"), "w").close()
    print(f"[grid-merge] {out}: {n_graphs} events, {len(cfgs)} configurations")


# ------------------------------------------------------------------------------------------------- union graphs in blocks
UNION_ARGS = {"tau": 0.005, "k": 5}                         # the fixed operating point (plan section 3)


def _union_dirs(seed: int, split: str, block: int, n_blocks: int) -> tuple[str, str]:
    g = os.path.join(CANTO_GRAPHS, f"armg_s{seed}")
    return g, os.path.join(g, "ltf_union_graphs", "blocks", f"{split}_{block:03d}_of_{n_blocks:03d}")


def cmd_union_block(a) -> None:
    """Block k of n (sorted members, numpy.array_split) of arm g's metric-learning graphs of one split, through
    hnl.graphs.helix_union build with the run's learned triplet support (k 5, tau 0.005, r_w 0.24, --no-metrics),
    packed to ltf_union_graphs/blocks/<split>_<k>_of_<n>/<split>set.tar. The same graphs as one job over the split:
    every event is built on its own."""
    _require_generated()
    g, blk = _union_dirs(a.seed, a.split, a.block, a.n_blocks)
    if os.path.exists(os.path.join(blk, "done")):
        print(f"{blk} is done")
        return
    model = os.path.join(g, "learned_triplet_filter_v2", "model.joblib")
    local = os.path.join("/tmp", os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", "local"), "g3u")
    src = os.path.join(local, "src")
    os.makedirs(src, exist_ok=True)
    t0 = time.time()
    with tarfile.open(os.path.join(g, "ml_graphs", f"{a.split}set.tar")) as t:
        names = sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))
        sel = [names[i] for i in np.array_split(np.arange(len(names)), a.n_blocks)[a.block]]
        for nm in sel:
            t.extract(nm, src, filter="data")
    os.makedirs(blk, exist_ok=True)
    cmd = ["python", "-m", "hnl.graphs.helix_union", "build", "--ml-run", g, "--split", a.split, "--graphs", src,
           "--out", os.path.join(local, "u"), "--report", os.path.join(blk, "report"), "--triplet-model", model,
           "--tau", str(UNION_ARGS["tau"]), "--k", str(UNION_ARGS["k"]), "--no-metrics", "--workers", str(a.workers)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    from hnl.train.acorn_runner import pack
    got = pack(os.path.join(local, "u"), blk)
    if got.get(f"{a.split}set") != len(sel):
        raise SystemExit(f"{blk}: packed {got}, expected {len(sel)} graphs")
    with open(os.path.join(blk, "provenance.json"), "w") as f:
        json.dump({"seed": a.seed, "split": a.split, "block": a.block, "n_blocks": a.n_blocks, "n_events": len(sel),
                   "first": sel[0], "last": sel[-1], "triplet_model": model, "triplet_model_sha256": _sha256(model),
                   "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
                   "elapsed_s": round(time.time() - t0, 1), "finished_utc": _now()}, f, indent=1)
    open(os.path.join(blk, "done"), "w").close()
    shutil.rmtree(local, ignore_errors=True)
    print(f"[union-block] {blk}: {len(sel)} graphs in {time.time() - t0:.0f}s")


def cmd_union_gather(a) -> None:
    """The blocks of one split into ltf_union_graphs/<split>set.tar, checked to hold exactly the events of the
    metric-learning graphs; report_<split>/build.json from block 0 with the block list; ltf_union_graphs_<split>.done."""
    g = os.path.join(CANTO_GRAPHS, f"armg_s{a.seed}")
    d = os.path.join(g, "ltf_union_graphs")
    blocks = sorted(glob.glob(os.path.join(d, "blocks", f"{a.split}_*_of_*")))
    n = {int(os.path.basename(b).rsplit("_of_", 1)[1]) for b in blocks}
    if len(n) != 1 or len(blocks) != n.pop() or not all(os.path.exists(os.path.join(b, "done")) for b in blocks):
        raise SystemExit(f"{d}: incomplete {a.split} blocks")
    with tarfile.open(os.path.join(g, "ml_graphs", f"{a.split}set.tar")) as t:
        want = sorted(os.path.basename(m).split("-")[0].replace(".pyg.gz", "") for m in t.getnames() if m.endswith(".pyg.gz"))
    out = os.path.join(d, f"{a.split}set.tar")
    got = []
    with tarfile.open(out + ".tmp", "w") as o:
        for b in blocks:
            with tarfile.open(os.path.join(b, f"{a.split}set.tar")) as t:
                for m in t.getmembers():
                    o.addfile(m, t.extractfile(m))
                    got.append(m.name.replace(".pyg.gz", ""))
    if sorted(got) != want or len(set(got)) != len(got):
        os.remove(out + ".tmp")
        raise SystemExit(f"{a.split}: {len(got)} union graphs, {len(want)} metric-learning graphs")
    os.replace(out + ".tmp", out)
    rep = os.path.join(d, f"report_{a.split}")
    os.makedirs(rep, exist_ok=True)
    b0 = json.load(open(os.path.join(blocks[0], "report", "build.json")))
    b0.update({"n_events": len(got), "blocks": [json.load(open(os.path.join(b, "provenance.json"))) for b in blocks],
               "source": os.path.join(g, "ml_graphs", f"{a.split}set.tar"), "gathered_utc": _now()})
    with open(os.path.join(rep, "build.json"), "w") as f:
        json.dump(b0, f, indent=1)
    open(os.path.join(g, f"ltf_union_graphs_{a.split}.done"), "w").close()
    print(f"[union-gather] {out}: {len(got)} graphs from {len(blocks)} blocks")


# ------------------------------------------------------------------------------------------------- selection
def truth(sample: str) -> pd.DataFrame:
    from hnl.data.acorn_store import SAMPLE_CODE
    t = pd.read_parquet(os.path.join(T1, f"truth_{sample}.parquet"),
                        columns=["event", "particle_id", "is_target", "gun", "in_pop", "prompt", "d0", "r_v", "pt"])
    t["event_id"] = np.int64(SAMPLE_CODE[sample]) * 100_000_000 + t.event.to_numpy(np.int64)
    return t


def grid_table(model: str, seed: int, name: str = "grid_val_iso", sample: str = "generic_iso") -> pd.DataFrame:
    """One row per configuration: P_G efficiency, displaced-source and prompt efficiencies, fake rate."""
    d = os.path.join(run_dir(model, seed), name)
    if not os.path.exists(os.path.join(d, "done")):
        raise SystemExit(f"no finished grid {d}")
    pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
    pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
    cfgs = json.load(open(os.path.join(d, "configs.json")))
    t = truth(sample)
    t = t[t.event_id.isin(set(pe.event_id.astype(np.int64)))]
    pp = pp.astype({"event_id": np.int64}).merge(
        t[["event_id", "particle_id", "is_target", "gun", "in_pop", "prompt"]], on=["event_id", "particle_id"], how="inner")
    rows = []
    g_pp = pp.groupby("config")
    s = pe.groupby("config")[["n_fake", "n_tracks_ge5"]].sum()
    n_pop = int(t.in_pop.sum())
    for i, c in enumerate(cfgs):
        p = g_pp.get_group(i)
        pop = p[p.in_pop]
        if len(pop) != n_pop:
            raise SystemExit(f"{d}: configuration {i} has {len(pop)} P_G rows, truth {n_pop}")
        disp = p[p.is_target & p.gun]
        pr = p[p.prompt]
        rows.append({"model": model, "seed": seed, "config_index": i, "family": c[0],
                     "params": ",".join(f"{float(x):g}" for x in c[1:]), "config": ",".join([c[0], *[f"{float(x):g}" for x in c[1:]]]),
                     "eff_primary": pop.dm_matched.mean(), "n_primary": n_pop,
                     "eff_displaced_source": disp.dm_matched.mean(), "eff_prompt": pr.dm_matched.mean(),
                     "fake_rate_hnl": s.loc[i, "n_fake"] / s.loc[i, "n_tracks_ge5"],
                     "dup_rate_primary": (pop.n_sm_tracks.astype(int) - 1).clip(lower=0).sum() / len(pop)})
    return pd.DataFrame(rows)


def per_run_choice(tab: pd.DataFrame) -> pd.DataFrame:
    """Per family the builder rule at FR* with the one-standard-error tie rule (hnl.eval.trackbuild_select.select)."""
    from hnl.eval.trackbuild_select import tie_best
    out = []
    for fam, g in tab.groupby("family"):
        ok = g[g.fake_rate_hnl <= FR_STAR + 1e-12]
        if ok.empty:
            continue
        best = ok.loc[tie_best(ok, float(ok.n_primary.iloc[0]))]
        grid = g.params.str.split(",", expand=True).astype(float)
        bp = np.array(best.params.split(","), dtype=float)
        edge = [k for k in range(grid.shape[1]) if grid[k].nunique() > 1 and bp[k] in (grid[k].min(), grid[k].max())]
        out.append({**best.to_dict(), "n_configs": len(g), "n_ok": len(ok), "edge_params": edge})
    return pd.DataFrame(out)


def family_choice(choices: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """3-seed means per family (families with a configuration at FR* in every seed); the one-standard-error rule on
    the means with the pooled count (hnl.eval.final_select.choose without the prompt guards, addendum G 5.2)."""
    from hnl.eval.trackbuild_select import tie_best
    num = ["eff_primary", "fake_rate_hnl", "eff_displaced_source", "eff_prompt"]
    g = choices.groupby("family")
    m = g[num].mean()
    m["n_seeds"] = g.size()
    m["n_primary"] = g["n_primary"].sum()
    m = m.reset_index()
    full = m[m.n_seeds == len(SEEDS)]
    if full.empty:
        raise SystemExit("no family has a configuration at FR* in every seed")
    best = tie_best(full, float(full.loc[full.eff_primary.idxmax(), "n_primary"]))
    m["chosen"] = m.index == best
    return m.sort_values("eff_primary", ascending=False), str(m.loc[best, "family"])


def settings(model: str, seed: int) -> dict:
    """Checkpoints and graph settings of a retrained run, for the selection file (T4 reads them from there)."""
    import yaml
    a = os.path.join(ACORN, f"armg_s{seed}")
    ml = json.load(open(glob.glob(os.path.join(a, "ml_graphs", "infer_*.json"))[0]))
    out = {"acorn_run": a, "ml_ckpt": ml["ckpt"], "ml_sha256": ml["ckpt_sha256"],
           "r_infer": ml["overrides"]["r_infer"], "knn_infer": ml["overrides"]["knn_infer"]}
    if model == "acorn":
        gi = json.load(open(glob.glob(os.path.join(a, "gnn_scores", "infer_*.json"))[0]))
        out.update(run=a, gnn_ckpt=gi["ckpt"], gnn_sha256=gi["ckpt_sha256"])
    else:
        r = run_dir("canto", seed)
        gi = json.load(open(glob.glob(os.path.join(r, "gnn_scores", "infer_*.json"))[0]))
        ub = json.load(open(os.path.join(CANTO_GRAPHS, f"armg_s{seed}", "ltf_union_graphs", "report_val", "build.json")))
        out.update(run=r, graphs_run=os.path.join(CANTO_GRAPHS, f"armg_s{seed}"), gnn_ckpt=gi["ckpt"],
                   gnn_sha256=gi["ckpt_sha256"], triplet_model=ub["triplet_model"],
                   triplet_model_sha256=_sha256(ub["triplet_model"]), tau=ub["tau"], k=ub["k"], r_wide=ub["r_wide"],
                   nsig=ub["nsig"], pt_min=ub["pt_min"], union_ml_ckpt=ub["ml_ckpt"],
                   config_yaml=os.path.join(REPO, "configs", "stage1", "triplet_symregion_trunc.yaml"))
    mm = merge_model(model, seed)
    out.update(merge_model=mm, merge_sha256=_sha256(mm))
    return out


def cmd_select(a) -> None:
    _require_generated()
    res = {"created_utc": _now(), "git": _git(), "fr_star": FR_STAR, "events": "generic_iso val (806 events)",
           "rule": ("per run and family: highest P_G efficiency at fake rate <= FR* on generic_iso val, one-standard-error "
                    "tie rule (hnl.eval.trackbuild_select.tie_best); per model: the family by the same rule on 3-seed "
                    "means with the pooled count (families with a configuration at FR* in every seed); each seed keeps "
                    "its own configuration of that family. Addendum G section 5.2; no prompt guard in the rule (the "
                    "paper's guard against acorn arm (i) is recorded per seed, with a guarded alternative where the "
                    "chosen family fails it; plan D-g2); no test event."),
           "models": {}}
    tabs, fams = [], []
    for model in ("acorn", "canto"):
        ch = []
        for s in SEEDS:
            t = grid_table(model, s)
            tabs.append(t)
            c = per_run_choice(t)
            c["seed"] = s
            ch.append(c)
        ch = pd.concat(ch, ignore_index=True)
        fm, fam = family_choice(ch)
        fm.insert(0, "model", model)
        fams.append(fm)
        guard = {}
        for s in SEEDS:
            gv = json.load(open(os.path.join(run_dir(model, s), "guard_val.json")))
            for r in gv["families"]:
                guard[(s, r["family"])] = {**r, "reference": gv["reference"]}
            got = {r["config"] for r in gv["families"]}
            want = set(ch[ch.seed == s].config)
            if got != want:
                raise SystemExit(f"{model} seed {s}: guard_val.json has {sorted(got)}, the grid's choices {sorted(want)}")

        def entries_of(f):
            out = []
            for s in SEEDS:
                row = ch[(ch.seed == s) & (ch.family == f)].iloc[0]
                out.append({"seed": s, "config": row.config, "family": f, "params": row.params,
                            "kind": "cc" if f == "cc" else "family",
                            "val": {k: float(row[k]) for k in ("eff_primary", "fake_rate_hnl", "eff_displaced_source",
                                                               "eff_prompt", "dup_rate_primary")},
                            "n_primary": int(row.n_primary), "edge_params": list(row.edge_params),
                            "prompt_guard": guard[(s, f)], **settings(model, s)})
            return out

        entries = entries_of(fam)
        # prompt guard of the paper's selection (plan section 4, clarification D-g2): recorded; if the chosen family
        # fails it in any seed, the best family that passes it in every seed, by the same rule, as a sensitivity entry
        ch["guard_arm_i"] = [guard[(s, f)]["guard_arm_i"] for s, f in zip(ch.seed, ch.family)]
        fm["guard_arm_i_every_seed"] = [bool(ch[ch.family == f].guard_arm_i.all()) and int((ch.family == f).sum()) == len(SEEDS)
                                        for f in fm.family]
        fails = [e["seed"] for e in entries if not e["prompt_guard"]["guard_arm_i"]]
        alt = None
        if fails:
            ok = ch.groupby("family").guard_arm_i.transform("all") & (ch.groupby("family").seed.transform("size") == len(SEEDS))
            if ok.any():
                _, afam = family_choice(ch[ok])
                alt = {"family": afam, "seeds": entries_of(afam),
                       "note": "sensitivity entry: evaluated at T4 next to the claim rows, decides no claim"}
            else:
                alt = {"family": None, "note": "no family passes the prompt guard in every seed"}
        res["models"][model] = {"label": LABELS[model], "family": fam, "seeds": entries,
                                "prompt_guard_fails_in_seeds": fails, "guarded_alternative": alt,
                                "family_means": fm.drop(columns="model").to_dict("records"),
                                "per_run_choices": ch.drop(columns=["model"], errors="ignore").to_dict("records")}
    pd.concat(tabs, ignore_index=True).to_csv(os.path.join(REPO, "results", "addendum_g_t3_val_grid_v1.csv"), index=False)
    pd.concat(fams, ignore_index=True).to_csv(os.path.join(REPO, "results", "addendum_g_t3_val_families_v1.csv"),
                                              index=False)
    path = a.out or os.path.join(REPO, "results", "addendum_g_selection_v1.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(path)
    for model, m in res["models"].items():
        print(model, m["family"], [(e["config"], round(e["val"]["eff_primary"], 4), round(e["val"]["fake_rate_hnl"], 4),
                                    e["edge_params"], e["prompt_guard"]["guard_arm_i"]) for e in m["seeds"]],
              "guarded alternative:", (m["guarded_alternative"] or {}).get("family"))


# ------------------------------------------------------------------------------------------------- val read-outs
def _evaluate(model: str, cfgs: list[tuple], paths: list[str], pairs: dict, keep: dict, workers: int):
    """(per_particle, per_event) of configurations, config = index into cfgs, computed as in the grid: acorn's CC
    through hnl.eval.evaluate_gnn, everything else through hnl.eval.trackbuild_val."""
    chunks = []
    for k in range(0, len(paths), 5):
        ids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths[k:k + 5]]
        chunks.append((paths[k:k + 5], model, list(cfgs), {e: pairs[e] for e in ids if e in pairs},
                       {e: keep[e] for e in ids if e in keep}))
    with cf.ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        res = list(ex.map(_eval_one, chunks))
    pp = pd.concat([r[0] for r in res], ignore_index=True)
    pe = pd.concat([r[1] for r in res], ignore_index=True)
    return pp, pe


PE_COLS = ["event_id", "config", "n_tracks_ge5", "n_fake", "pair_both", "pair_merged"]
PP_COLS = ["event_id", "particle_id", "config", "dm_matched", "n_sm_tracks"]


def _eval_one(args):
    paths, model, cfgs, pairs, keep = args
    from hnl.eval import evaluate_gnn as EG
    from hnl.eval import trackbuild_val as TV
    is_cc = [model == "acorn" and c[0] == "cc" for c in cfgs]
    tv = [i for i, x in enumerate(is_cc) if not x]
    cc = [i for i, x in enumerate(is_cc) if x]
    pps, pes = [], []
    if tv:
        pp, pe = TV._chunk((paths, [cfgs[i] for i in tv], pairs, keep))
        pp["config"] = np.asarray(tv)[pp.config.to_numpy(int)]
        pe["config"] = np.asarray(tv)[pe.config.to_numpy(int)]
        pps.append(pp)
        pes.append(pe)
    if cc:
        cuts = [float(cfgs[i][1]) for i in cc]
        idx = {round(c, 4): i for c, i in zip(cuts, cc)}
        res = [EG.evaluate_event(p, cuts, pairs, keep) for p in paths]
        pp = pd.concat([r[0] for r in res], ignore_index=True)
        pe = pd.concat([r[1] for r in res], ignore_index=True)
        pp["config"] = [idx[round(float(c), 4)] for c in pp.cut]
        pe["config"] = [idx[round(float(c), 4)] for c in pe.cut]
        pps.append(pp)
        pes.append(pe)
    pp = pd.concat([x[PP_COLS] for x in pps], ignore_index=True)
    pe = pd.concat([x[[c for c in PE_COLS if c in x]] for x in pes], ignore_index=True)
    return pp, pe


def _extract(model: str, seed: int, codes: set[str], local: str) -> list[str]:
    with tarfile.open(scored_val_tar(model, seed)) as t:
        want = sorted(m for m in t.getnames() if m.endswith(".pyg.gz") and os.path.basename(m)[5:6] in codes)
        for nm in want:
            t.extract(nm, local, filter="data")
    return [os.path.join(local, m) for m in want]


def _labeled_codes() -> set[str]:
    from hnl.data.acorn_store import SAMPLE_CODE
    return {str(SAMPLE_CODE[s]) for s in ("ttbar", "hnl10_100", "hnl15_200")}


def ref_arm_i(seed: int, events: np.ndarray) -> dict:
    """The prompt guard's reference of the paper's selection (hnl.eval.final_select, guard_arm_i): acorn arm (i) of the
    same seed, CC at its preregistered section-4 cut, on the same ttbar and HNL val events."""
    from hnl.eval.trackbuild_select import reference_row
    wp = json.load(open(os.path.join(REPO, "results", "working_points_v1.json")))
    return reference_row(os.path.join(RUNS, "baseline_v1", f"armi_s{seed}"), float(wp["FR_star"]), events, wp)


def cmd_guard(a) -> None:
    """Every family's per-run choice (generic_iso val, section 4 of the plan) on the ttbar and HNL val events, and the
    paper's prompt guard against arm (i): W-muon and ttbar-prompt efficiency at most GUARD below acorn arm (i) CC at
    its section-4 cut (hnl.eval.trackbuild_select.GUARD). Writes RUN/guard_val.json."""
    _require_generated()
    from hnl.eval.baseline_val import curves
    from hnl.eval.evaluate_gnn import event_truth
    from hnl.eval.particles import particle_table
    from hnl.eval.trackbuild_select import GUARD
    ch = per_run_choice(grid_table(a.model, a.seed))
    cfgs = [tuple([c.split(",")[0], *map(float, c.split(",")[1:])]) for c in ch.config]
    mm = merge_model(a.model, a.seed)
    os.environ["HNL_MERGE_MODEL"] = os.path.abspath(mm)
    local = os.path.join("/tmp", os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", "local"), "g3g")
    os.makedirs(local, exist_ok=True)
    t0 = time.time()
    paths = _extract(a.model, a.seed, _labeled_codes(), local)
    ids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths]
    pairs, keep = event_truth(ids)
    pp, pe = _evaluate(a.model, cfgs, paths, pairs, keep, a.workers)
    events = np.unique(pe.event_id.astype(np.int64))
    c = curves(pp, pe, particle_table(events))
    ref = ref_arm_i(a.seed, events)
    rows = []
    for i, cfg in enumerate(ch.config):
        r = c.loc[i]
        d_w, d_p = float(r.eff_w_muons - ref["eff_w_muons"]), float(r.eff_ttbar_prompt - ref["eff_ttbar_prompt"])
        rows.append({"family": ch.family.iloc[i], "config": cfg, "eff_w_muons": float(r.eff_w_muons),
                     "eff_ttbar_prompt": float(r.eff_ttbar_prompt), "eff_hnl_primary": float(r.eff_primary),
                     "fake_rate_hnl": float(r.fake_rate_hnl), "fake_rate_ttbar": float(r.fake_rate_ttbar),
                     "d_eff_w_muons_vs_arm_i": d_w, "d_eff_ttbar_prompt_vs_arm_i": d_p,
                     "guard_arm_i": bool(d_w >= -GUARD - 1e-12 and d_p >= -GUARD - 1e-12)})
    out = {"model": a.model, "seed": a.seed, "guard": GUARD, "n_events": int(len(events)),
           "reference": {"run": os.path.join(RUNS, "baseline_v1", f"armi_s{a.seed}"), "cut": ref["params"],
                         "reachable": ref["reachable"], "eff_w_muons": float(ref["eff_w_muons"]),
                         "eff_ttbar_prompt": float(ref["eff_ttbar_prompt"])},
           "merge_model": mm, "merge_sha256": _sha256(mm), "families": rows, "git": _git(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "elapsed_s": round(time.time() - t0, 1), "finished_utc": _now()}
    with open(os.path.join(run_dir(a.model, a.seed), "guard_val.json"), "w") as f:
        json.dump(out, f, indent=1)
    shutil.rmtree(local, ignore_errors=True)
    print(json.dumps(out, indent=1))


def cmd_readout(a) -> None:
    """The selected configuration of one run on generic_dense val and on the val splits of ttbar and the HNL samples."""
    _require_generated()
    from hnl.data.acorn_store import SAMPLE_CODE
    from hnl.eval.baseline_val import curves
    from hnl.eval.evaluate_gnn import event_truth
    from hnl.eval.particles import particle_table
    sel = json.load(open(a.selection))
    e = [x for x in sel["models"][a.model]["seeds"] if x["seed"] == a.seed][0]
    if _sha256(e["merge_model"]) != e["merge_sha256"]:
        raise SystemExit(f"{e['merge_model']}: sha256 differs from the selection file")
    os.environ["HNL_MERGE_MODEL"] = os.path.abspath(e["merge_model"])
    cfg = tuple([e["config"].split(",")[0], *map(float, e["config"].split(",")[1:])])
    local = os.path.join("/tmp", os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", "local"), "g3r")
    os.makedirs(local, exist_ok=True)
    t0 = time.time()
    paths = {"d": _extract(a.model, a.seed, {str(SAMPLE_CODE["generic_dense"])}, local),
             "l": _extract(a.model, a.seed, _labeled_codes(), local)}
    out = {"model": a.model, "seed": a.seed, "config": e["config"], "git": _git(),
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    # generic_dense val: the T1 truth
    ids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths["d"]]
    pp, pe = _evaluate(a.model, [cfg], paths["d"], {}, _keep(ids), a.workers)
    t = truth("generic_dense")
    t = t[t.event_id.isin(set(pe.event_id.astype(np.int64)))]
    pp = pp.astype({"event_id": np.int64}).merge(t[["event_id", "particle_id", "is_target", "gun", "in_pop", "prompt"]],
                                                on=["event_id", "particle_id"], how="inner")
    out["generic_dense"] = {"n_events": int(pe.event_id.nunique()), "n_pop": int(pp.in_pop.sum()),
                            "eff_pop": float(pp[pp.in_pop].dm_matched.mean()),
                            "eff_displaced_source": float(pp[pp.is_target & pp.gun].dm_matched.mean()),
                            "eff_prompt": float(pp[pp.prompt].dm_matched.mean()),
                            "fake_rate": float(pe.n_fake.sum() / pe.n_tracks_ge5.sum())}
    # ttbar and HNL val: the preregistered populations (hnl.eval.baseline_val.populations)
    ids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths["l"]]
    pairs, keep = event_truth(ids)
    pp, pe = _evaluate(a.model, [cfg], paths["l"], pairs, keep, a.workers)
    c = curves(pp, pe, particle_table(np.unique(pe.event_id))).iloc[0]
    out["labeled_val"] = {k: float(c[k]) for k in ("eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt",
                                                   "eff_ttbar_targets", "fake_rate_hnl", "fake_rate_ttbar") if k in c}
    out["labeled_val"].update({f"n_{k}": int(c[f"n_{k}"]) for k in ("primary", "w_muons", "ttbar_prompt") if f"n_{k}" in c})
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(run_dir(a.model, a.seed), "readout_val.json"), "w") as f:
        json.dump(out, f, indent=1)
    shutil.rmtree(local, ignore_errors=True)
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--input-dir", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("rk")
    r.add_argument("--scan", required=True)
    for nm in ("grid", "grid-merge"):
        g = sub.add_parser(nm)
        g.add_argument("--model", required=True, choices=["acorn", "canto"])
        g.add_argument("--seed", type=int, required=True)
        g.add_argument("--sample", default="generic_iso", choices=["generic_iso"])
        g.add_argument("--name", default="grid_val_iso")
        if nm == "grid":
            g.add_argument("--block", type=int, default=0)
            g.add_argument("--n-blocks", type=int, default=1)
            g.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            g.add_argument("--tar", default=None, help="code smoke tests only: another scored tar")
            g.add_argument("--merge-model", default=None, help="code smoke tests only: another merge model")
            g.add_argument("--out", default=None, help="code smoke tests only: another output directory")
            g.add_argument("--max-events", type=int, default=None, help="code smoke tests only")
    se = sub.add_parser("select")
    se.add_argument("--out", default=None)
    ub = sub.add_parser("union-block")
    ub.add_argument("--seed", type=int, required=True)
    ub.add_argument("--split", required=True, choices=["train", "val"])
    ub.add_argument("--block", type=int, required=True)
    ub.add_argument("--n-blocks", type=int, required=True)
    ub.add_argument("--workers", type=int, default=14)
    ug = sub.add_parser("union-gather")
    ug.add_argument("--seed", type=int, required=True)
    ug.add_argument("--split", required=True, choices=["train", "val"])
    gu = sub.add_parser("guard")
    gu.add_argument("--model", required=True, choices=["acorn", "canto"])
    gu.add_argument("--seed", type=int, required=True)
    gu.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ro = sub.add_parser("readout")
    ro.add_argument("--model", required=True, choices=["acorn", "canto"])
    ro.add_argument("--seed", type=int, required=True)
    ro.add_argument("--selection", default=os.path.join(REPO, "results", "addendum_g_selection_v1.json"))
    ro.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    {"scan": cmd_scan, "rk": cmd_rk, "grid": cmd_grid, "grid-merge": cmd_grid_merge, "select": cmd_select,
     "guard": cmd_guard, "readout": cmd_readout, "union-block": cmd_union_block,
     "union-gather": cmd_union_gather}[a.cmd](a)


if __name__ == "__main__":
    main()
