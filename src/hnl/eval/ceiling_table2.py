"""Ceilings of the training-sample comparison (paper Table tab:results, results/training_sample_val_mwslmerge.csv),
val only: for every run of that table, on its own scored val graphs, the efficiency a perfect edge classifier would
give (supervisor feedback, 2026-10-14: is Canto-L near its limit, so that training on the signal had no room to help?).

Perfect classifier: score 1 for the graph edges that join two consecutive hits of one particle (the chain truth of
docs/definitions.md section 2, the npz track_edges; the classifier's training label y is checked against it per event)
and 0 for every other edge. Quantities per particle (double-majority match, as hnl.eval.trackbuild_val):
  cc_chain        connected components of the true edges in the graph: the graph-stage ceiling of docs/definitions.md
  cc_samepid      connected components of every graph edge joining two hits of one particle (pid != 0): an upper bound
                  for any edge classifier feeding connected components (hnl.graphs.ceiling, secondary ceiling)
  mwsl_chain      the first pass of learned merging (same-layer Mutex Watershed, s0 0.2) on the perfect scores
  lm_chain_<tau>  learned merging (mwslmerge) on the perfect scores with the run's own merge model, every tau of the
                  val grid; the run's working-point tau is the perfect-classifier efficiency with Table 2's builder
  own_<tau>       (identity check, on the first --check HNL events of each block) learned merging on the run's real
                  scores at its working-point tau, compared with the stored per-particle table of that working point
HNL val events get every quantity; ttbar val events only cc_chain and cc_samepid (the merging features are the costly
part, and Table 2's question is about the HNL daughters).

    python -m hnl.eval.ceiling_table2 run --graphs DIR --merge-model M --tau T --out DIR [--block K --nblocks N]
                                          [--check 30] [--max-events N]
    python -m hnl.eval.ceiling_table2 summarize --base DIR --out results/ceiling_table2_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import time

import numpy as np
import pandas as pd

from hnl.eval import matching as M
from hnl.eval.evaluate_gnn import _event_id, event_truth
from hnl.graphs import ceiling as C
from hnl.trackbuild import builders as TB

TAUS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)     # the mwslmerge val grid of hnl.eval.trackbuild_val
S_TRUE, S_FALSE = 0.999, 0.001
TTBAR_CODE = 1


def load(path: str):
    """One scored graph as hnl.eval.trackbuild_val.load_scored (every unordered edge once, highest score of its
    directions), plus per unique edge the chain truth, acorn's label y (max over directions) and the same-particle
    flag."""
    from acorn.utils.loading_utils import load_pyg
    g = load_pyg(path[:-3] if path.endswith(".gz") else path, weights_only=False)
    eid = _event_id(g)
    pid = g.hit_particle_id.numpy()
    n = len(pid)
    r = (g.r if "r" in g else g.hit_r).numpy().astype(np.float64)
    z = (g.z if "z" in g else g.hit_z).numpy().astype(np.float64)
    phi = (g.phi if "phi" in g else g.hit_phi).numpy().astype(np.float64)
    ei = g.edge_index.numpy()
    sc = (g.scores if "scores" in g else g.edge_scores).numpy()
    y = g.y.numpy().astype(bool) if "y" in g else None
    a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    code = a.astype(np.int64) * n + b
    order = np.lexsort((-sc, code))
    first = np.ones(len(code), dtype=bool)
    first[1:] = code[order][1:] != code[order][:-1]
    k = order[first]
    a, b, sc, code = a[k], b[k], sc[k], code[k]
    t = C.unordered_unique(g.track_edges.numpy())
    chain = np.isin(code, t[0].astype(np.int64) * n + t[1])
    yu = None
    if y is not None:                                   # any direction labelled true
        full = np.minimum(ei[0], ei[1]).astype(np.int64) * n + np.maximum(ei[0], ei[1])
        yu = np.isin(code, np.unique(full[y]))
    same = (pid[a] == pid[b]) & (pid[a] != 0)
    return eid, pid, r, phi, z, a, b, sc, chain, yu, same, int(t.shape[1])


def evaluate_event(path, tau_wp, keep, geo, check: bool):
    from hnl.trackbuild import learned_merge as LM
    eid, pid, r, phi, z, a, b, sc, chain, yu, same, n_true = load(path)
    n = len(pid)
    fm = M.FastMatch(pid)
    hnl = int(eid) // 100_000_000 != TTBAR_CODE
    labs = {"cc_chain": TB.cc(n, a, b, np.where(chain, S_TRUE, S_FALSE), 0.5),
            "cc_samepid": TB.cc(n, a, b, np.where(same, S_TRUE, S_FALSE), 0.5)}
    if hnl:
        L = geo["layer_id"]
        assert len(L) == n
        merger = _merge_model()
        first = LM.first_pass(n, a, b, np.where(chain, S_TRUE, S_FALSE), L)
        labs["mwsl_chain"] = first
        pairs, p = merger.prepare(first, r, phi, z, geo)
        for tau in TAUS:
            labs[f"lm_chain_{tau:g}"] = LM.apply(first, pairs, p, tau, L)
        if check:
            f2 = LM.first_pass(n, a, b, sc, L)
            p2 = merger.prepare(f2, r, phi, z, geo)
            labs[f"own_{tau_wp:g}"] = LM.apply(f2, p2[0], p2[1], tau_wp, L)
    sel = np.isin(fm.particles, keep.get(eid, []))
    parts, rows = [], []
    for name, lab in labs.items():
        m = fm.match(lab)
        parts.append(pd.DataFrame({"event_id": np.int64(eid), "particle_id": fm.particles[sel].astype(np.int64),
                                   "config": name, "dm_matched": m["dm_matched"][sel]}))
        rows.append({"event_id": int(eid), "config": name, "n_tracks_ge5": m["n_tracks_ge5"], "n_fake": m["n_fake"]})
    ev = {"event_id": int(eid), "n_edges": len(a), "n_true_edges": n_true, "n_chain_in_graph": int(chain.sum()),
          "n_samepid_in_graph": int(same.sum()),
          "n_y": -1 if yu is None else int(yu.sum()), "n_y_ne_chain": -1 if yu is None else int((yu != chain).sum())}
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows), ev


_MERGE = None


def _merge_model():
    global _MERGE
    if _MERGE is None:
        from hnl.trackbuild.learned_merge import Merger
        _MERGE = Merger(os.environ["HNL_MERGE_MODEL"])
    return _MERGE


def _eid_of(path: str) -> str:
    return os.path.basename(path).split("event")[-1].split(".")[0]


def _chunk(args):
    paths, tau_wp, keep, check = args
    from hnl.geometry.resolution import hit_geometry
    hnl_ids = [int(_eid_of(p)) for p in paths if int(_eid_of(p)) // 100_000_000 != TTBAR_CODE]
    geometry = hit_geometry(hnl_ids) if hnl_ids else {}
    res = [evaluate_event(p, tau_wp, keep, geometry.get(int(_eid_of(p))), _eid_of(p) in check) for p in paths]
    return (pd.concat([x[0] for x in res], ignore_index=True), pd.concat([x[1] for x in res], ignore_index=True),
            pd.DataFrame([x[2] for x in res]))


def cmd_run(a):
    import concurrent.futures as cf
    os.environ["HNL_MERGE_MODEL"] = os.path.abspath(a.merge_model)
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))[:a.max_events]
    paths = paths[a.block::a.nblocks]
    out = os.path.join(a.out, f"block_{a.block}of{a.nblocks}")
    if os.path.exists(os.path.join(out, "meta.json")):
        raise SystemExit(f"{out} is complete: refusing to overwrite")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    eids = [_eid_of(p) for p in paths]
    _, keep = event_truth(eids)
    check = set([e for e in eids if int(e) // 100_000_000 != TTBAR_CODE][:a.check])
    chunks = [(paths[k:k + 5], a.tau, keep, check) for k in range(0, len(paths), 5)]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_chunk, chunks))
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(os.path.join(out, "per_particle.parquet"), index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(os.path.join(out, "per_event.parquet"), index=False)
    pd.concat([r[2] for r in res], ignore_index=True).to_parquet(os.path.join(out, "graph_stats.parquet"), index=False)
    meta = {"graphs": a.graphs, "merge_model": os.environ["HNL_MERGE_MODEL"], "tau_wp": a.tau, "block": a.block,
            "nblocks": a.nblocks, "n_events": len(paths), "n_check": len(check), "elapsed_s": round(time.time() - t0, 1),
            "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "taus": list(TAUS),
            "scores": [S_TRUE, S_FALSE]}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


# --------------------------------------------------------------------------------------------------------------------
# summary

QS = ("eff_primary", "d0_100_300", "d0_300_inf", "eff_w_muons", "eff_ttbar_prompt")
ARCHS = ("acorn", "Canto-L")
SAMPLES = ("ttbar", "ttbar+HNL", "HNL")


def tag(model: str, seed: int) -> str:
    return model.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p") + f"_s{seed}"


def merge_model_of(run: str, wp: str) -> str:
    grid = wp.split("(")[-1].rstrip(")")
    if grid == "trackbuild_val_fb3_all":                    # acorn: the grid part made by fb3_lmerge
        return json.load(open(os.path.join(run, "trackbuild_val_fb3_lmerge", "provenance.json")))["merge_model"]
    if grid == "trackbuild_val_lmerge":                     # Canto-L: slurm/stage1/learned_merge.sbatch
        return os.path.join(run, "lmerge", "model.joblib")
    raise ValueError(f"{run}: no merge model for grid {grid}")


def board() -> pd.DataFrame:
    from hnl.eval import bootstrap_tables as BT
    t = pd.read_csv(os.path.join(BT.REPO, "results", "training_sample_val_mwslmerge.csv"))
    t["tau"] = [float(w.split()[1]) for w in t.working_point]
    t["tag"] = [tag(m, s) for m, s in zip(t.model, t.seed)]
    t["merge_model"] = [merge_model_of(r, w) for r, w in zip(t.run, t.working_point)]
    return t


def cmd_plan(a):
    """One line per run: tag run merge_model tau (read by slurm/diagnostics/ceiling_table2.sh)."""
    for r in board().itertuples():
        print(r.tag, r.run, r.merge_model, f"{r.tau:g}")


def _read_blocks(d: str):
    blocks = sorted(glob.glob(os.path.join(d, "block_*", "meta.json")))
    if not blocks:
        raise SystemExit(f"{d}: no finished block")
    metas = [json.load(open(b)) for b in blocks]
    nb = {m["nblocks"] for m in metas}
    if len(nb) != 1 or len(metas) != nb.pop():
        raise SystemExit(f"{d}: {len(metas)} finished blocks of {[m['nblocks'] for m in metas]}")
    rd = lambda f: pd.concat([pd.read_parquet(os.path.join(os.path.dirname(b), f)) for b in blocks],  # noqa: E731
                             ignore_index=True)
    return rd("per_particle.parquet"), rd("per_event.parquet"), rd("graph_stats.parquet"), metas


def cmd_summarize(a):
    from hnl.eval import bootstrap as BS
    from hnl.eval import bootstrap_tables as BT
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    info = BT.info_table(BT.expected_events(BT.VAL_SETS))
    expected = set(BT.expected_events(BT.VAL_SETS).tolist())
    tb = board()
    seeds, checks, gstats = {}, [], []
    for r in tb.itertuples():
        pp, pe, gs, metas = _read_blocks(os.path.join(a.base, r.tag))
        if set(gs.event_id) != expected or gs.event_id.duplicated().any():
            raise SystemExit(f"{r.tag}: events processed {len(gs)} (unique {gs.event_id.nunique()}), "
                             f"expected {len(expected)}")
        if {m["merge_model"] for m in metas} != {os.path.abspath(r.merge_model)} or \
                {m["tau_wp"] for m in metas} != {r.tau}:
            raise SystemExit(f"{r.tag}: merge model or tau differ from the board")
        own_pp, own_pe, _ = BT.val_tables(r.run, r.working_point)
        # identity check: learned merging on the real scores, recomputed here, against the stored working point
        chk = pp[pp.config == f"own_{r.tau:g}"].merge(own_pp[["event_id", "particle_id", "dm_matched"]],
                                                       on=["event_id", "particle_id"], how="left",
                                                       suffixes=("", "_stored"))
        n_bad = int((chk.dm_matched != chk.dm_matched_stored).sum())
        checks.append({"tag": r.tag, "check_events": int(chk.event_id.nunique()), "check_particles": len(chk),
                       "mismatches": n_bad})
        if n_bad or chk.dm_matched_stored.isna().any():
            raise SystemExit(f"{r.tag}: identity check failed ({n_bad} of {len(chk)})")
        gstats.append({"tag": r.tag, "y_ne_chain_edges": int(gs.n_y_ne_chain.clip(lower=0).sum()),
                       "chain_edges_in_graph": int(gs.n_chain_in_graph.sum()), "true_edges": int(gs.n_true_edges.sum()),
                       "edges": int(gs.n_edges.sum())})
        s = {"own": ld.seed(r.run, r.working_point)}
        for cfg in ("cc_chain", "cc_samepid", "mwsl_chain", f"lm_chain_{r.tau:g}"):
            q = pp[pp.config == cfg]
            e = pe[pe.config == cfg]
            if cfg.startswith(("mwsl", "lm")):              # HNL events only: the ttbar populations are not defined
                q = q[q.event_id // 100_000_000 != TTBAR_CODE]
            s["lm_wp" if cfg.startswith("lm_") else cfg] = BT.counts(rs, q, e, info)
        seeds.setdefault(r.model, []).append((r, s))
    rows, ref_sd = [], {}
    for model, lst in seeds.items():
        for kind in ("own", "cc_chain", "cc_samepid", "mwsl_chain", "lm_wp"):
            m = BT.Model(model, model, [s[kind] for _, s in lst], rs, [r.working_point for r, _ in lst],
                         [r.run for r, _ in lst])
            for q in QS:
                if kind in ("mwsl_chain", "lm_wp") and q == "eff_ttbar_prompt":
                    continue
                pts, reps = m.pr(q)
                d = BS.seed_mean(pts, reps)
                num = sum(float(s[kind][q][x][0].sum()) for _, s in lst for x in s[kind][q])
                den = sum(float(s[kind][q][x][1].sum()) for _, s in lst for x in s[kind][q])
                rows.append({"model": model, "n_seeds": len(lst), "quantity": q, "kind": kind, "value": d["mean"],
                             "sigma_boot": d["sigma_boot"], "seed_sd": d["s_seeds"], "k": num, "n": den,
                             "per_seed": ";".join(f"{x:.6f}" for x in pts), "_m": m})
    df = pd.DataFrame(rows)
    # one-seed models carry the seed sd of Canto-L (ttbar) for the same quantity and kind (Table 2's convention)
    for (q, kind), g in df[df.model == "Canto-L (ttbar)"].groupby(["quantity", "kind"]):
        ref_sd[q, kind] = float(g.seed_sd.iloc[0])
    df["sigma"] = [np.hypot(r.sigma_boot, ref_sd[r.quantity, r.kind]) if r.n_seeds == 1 else
                   np.sqrt(r.sigma_boot ** 2 + r.seed_sd ** 2 / r.n_seeds) for r in df.itertuples()]
    # room = ceiling - own efficiency, paired over the events and within each seed
    room = []
    for (model, q), g in df.groupby(["model", "quantity"], sort=False):
        own = g[g.kind == "own"].iloc[0]
        po, ro = own._m.pr(q)
        for kind in ("cc_chain", "cc_samepid", "lm_wp"):
            h = g[g.kind == kind]
            if h.empty:
                continue
            pc, rc = h.iloc[0]._m.pr(q)
            pts = list(np.asarray(pc) - np.asarray(po))
            d = BS.seed_mean(pts, [x - y for x, y in zip(rc, ro)])
            room.append({"model": model, "quantity": q, "kind": f"room_{kind}", "value": d["mean"],
                         "sigma_boot": d["sigma_boot"], "seed_sd": d["s_seeds"], "n_seeds": len(pts),
                         "per_seed": ";".join(f"{x:.6f}" for x in pts)})
    room = pd.DataFrame(room)
    sd1 = {(r.quantity, r.kind): r.seed_sd for r in room[room.model == "Canto-L (ttbar)"].itertuples()}
    room["sigma"] = [np.hypot(r.sigma_boot, sd1[r.quantity, r.kind]) if r.n_seeds == 1 else
                     np.sqrt(r.sigma_boot ** 2 + r.seed_sd ** 2 / r.n_seeds) for r in room.itertuples()]
    # signal-training changes of efficiency, ceilings and room (primary population), paired over the events
    q, diffs = "eff_primary", []
    dq = df[df.quantity == q]
    mod = {(m, k): x for m, k, x in zip(dq.model, dq.kind, dq["_m"])}
    for arch in ARCHS:
        for s_, b_ in (("HNL", "ttbar"), ("ttbar+HNL", "ttbar")):
            hi, lo = f"{arch} ({s_})", f"{arch} ({b_})"
            for kind in ("own", "cc_chain", "lm_wp", "room_lm_wp"):
                if kind.startswith("room"):
                    terms = [(1.0, mod[hi, "lm_wp"]), (-1.0, mod[hi, "own"]), (-1.0, mod[lo, "lm_wp"]),
                             (1.0, mod[lo, "own"])]
                else:
                    terms = [(1.0, mod[hi, kind]), (-1.0, mod[lo, kind])]
                d = BS.linear_combination([(c, *m.pr(q)) for c, m in terms])
                s1 = 0.0
                if mod[hi, "own"].n_seeds == 1:     # one-seed model: the seed sd of Canto-L (ttbar) for this kind
                    ref = df[(df.model == "Canto-L (ttbar)") & (df.quantity == q)]
                    s1 = (float(ref[ref.kind == kind].seed_sd.iloc[0]) if not kind.startswith("room") else
                          float(room[(room.model == "Canto-L (ttbar)") & (room.quantity == q)
                                     & (room.kind == "room_lm_wp")].seed_sd.iloc[0]))
                seeds = float(np.hypot(d["sigma_seeds"], s1))
                diffs.append({"model": f"{arch}: {s_} minus {b_}", "quantity": q, "kind": f"diff_{kind}",
                              "value": d["value"], "sigma_boot": d["sigma_boot"], "seed_sd": seeds,
                              "sigma": float(np.hypot(d["sigma_boot"], seeds))})
    out = pd.concat([df.drop(columns="_m"), room, pd.DataFrame(diffs)], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out + ".csv", index=False)
    pd.DataFrame(checks).merge(pd.DataFrame(gstats), on="tag").to_csv(a.out + "_checks.csv", index=False)
    write_md(out, pd.DataFrame(checks).merge(pd.DataFrame(gstats), on="tag"), a.out + ".md")
    print(open(a.out + ".md").read())


ORDER = ["acorn (ttbar)", "acorn (ttbar+HNL)", "acorn (HNL)", "Canto-L (ttbar)", "Canto-L (ttbar+HNL)", "Canto-L (HNL)"]


def write_md(df: pd.DataFrame, chk: pd.DataFrame, path: str):
    def cell(model, q, kind, scale=100.0):
        r = df[(df.model == model) & (df.quantity == q) & (df.kind == kind)]
        if r.empty:
            return "-"
        r = r.iloc[0]
        return f"{scale * r.value:.2f} +- {scale * r.sigma:.2f}"

    def frac(model, q, kind):
        r = df[(df.model == model) & (df.quantity == q) & (df.kind == kind)].iloc[0]
        return f"{r.k:.0f}/{r.n:.0f}"
    L = ["# Ceilings of the training-sample comparison (Table 2; validation)", "",
         "Generated by `python -m hnl.eval.ceiling_table2 summarize`. Every run of results/training_sample_val_mwslmerge.csv"
         " on its own scored val graphs (the graphs its classifier scored). Perfect classifier: score 1 for the graph"
         " edges joining consecutive hits of one particle (chain truth, docs/definitions.md section 2), 0 for every other"
         " edge. Efficiency: Table 2's (learned merging, LM, at the run's val working point). Graph ceiling (CC): connected"
         " components of the true edges in the graph (docs/definitions.md section 6). Perfect + LM: the same builder as"
         " Table 2 (first pass: same-layer Mutex Watershed at s0 0.2; then the run's own merge model at its working-point"
         " tau) on the perfect scores; merging can join fragments across a missing true edge, so it can exceed the CC"
         " ceiling. Same-particle CC: connected components of every graph edge joining two hits of one particle. Room:"
         " ceiling minus efficiency, paired over events and seeds. Uncertainties: val event bootstrap (B 1000, seed"
         " 20261005) with the seed spread / n for three-seed models; one-seed models add the seed sd of Canto-L (ttbar) for"
         " the same quantity, as Table 2.", "",
         "## Primary population (percent)", "",
         "| model | seeds | efficiency (Table 2) | graph ceiling (CC) | perfect + LM | same-particle CC | room to perfect + LM (points) | room to CC ceiling (points) |",
         "|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        n = int(df[df.model == m].n_seeds.iloc[0])
        L.append(f"| {m} | {n} | {cell(m, 'eff_primary', 'own')} | {cell(m, 'eff_primary', 'cc_chain')} | "
                 f"{cell(m, 'eff_primary', 'lm_wp')} | {cell(m, 'eff_primary', 'cc_samepid')} | "
                 f"{cell(m, 'eff_primary', 'room_lm_wp')} | {cell(m, 'eff_primary', 'room_cc_chain')} |")
    L += ["", "## Daughter bins and prompt populations (pooled k/n over seeds)", "",
          "| model | quantity | efficiency | graph ceiling (CC) | perfect + LM | same-particle CC |", "|---|---|---|---|---|---|"]
    for m in ORDER:
        for q in QS[1:]:
            L.append(f"| {m} | {q} | {frac(m, q, 'own')} | {frac(m, q, 'cc_chain')} | "
                     + (f"{frac(m, q, 'lm_wp')}" if q != "eff_ttbar_prompt" else "-") + f" | {frac(m, q, 'cc_samepid')} |")
    L += ["", "## Changes from training on ttbar (primary population, points; paired over events)", "",
          "| change | efficiency | graph ceiling (CC) | perfect + LM | room to perfect + LM |", "|---|---|---|---|---|"]
    for lab in sorted({m for m in df.model if ":" in m}):
        L.append(f"| {lab} | {cell(lab, 'eff_primary', 'diff_own')} | {cell(lab, 'eff_primary', 'diff_cc_chain')} | "
                 f"{cell(lab, 'eff_primary', 'diff_lm_wp')} | {cell(lab, 'eff_primary', 'diff_room_lm_wp')} |")
    L += ["", "## Checks", "",
          "Identity: learned merging on the run's real scores at its working point, recomputed on the first HNL events of"
          " each block, against the stored per-particle table (mismatches must be 0). y vs chain: unique graph edges whose"
          " acorn label y (any direction) differs from the chain truth.", "",
          "| run | check events | check particles | mismatches | graph edges | true edges | chain edges in graph | y != chain |",
          "|---|---|---|---|---|---|---|---|"]
    for r in chk.itertuples():
        L.append(f"| {r.tag} | {r.check_events} | {r.check_particles} | {r.mismatches} | {r.edges} | {r.true_edges} | "
                 f"{r.chain_edges_in_graph} | {r.y_ne_chain_edges} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--graphs", required=True)
    p.add_argument("--merge-model", required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--nblocks", type=int, default=1)
    p.add_argument("--check", type=int, default=30)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("plan")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("summarize")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_summarize)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
