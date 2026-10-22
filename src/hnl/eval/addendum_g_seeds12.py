"""Seeds 1 and 2 of Canto-L (HNL) and Canto-L (ttbar+HNL) on the generated samples of addendum G step T2 (ledger
R20; post-freeze, descriptive, no claim): the read-out and the removal of the chain's intermediate files.

The runs are evaluated by the T2 code (hnl.eval.addendum_g) through slurm/ts_seeds12/g_chain.sh with the output root
ADDG_OUT on /work (slurm/ts_seeds12/g_env.sh) and ADDG_TS_SEEDS=0,1,2. In ADDG_OUT, eval/<model>_s<seed> of every R14
run (and acorn_ii_s1, acorn_ii_s2) are links to runs/addendum_g_v1, so that the T2 summary reads all models at once.

  summarize  hnl.eval.addendum_g_summary.summary for the sets test, hss and hnl on ADDG_OUT/eval (every model; Canto-L
             (HNL) and Canto-L (ttbar+HNL) now with three seeds: mean and sigma^2 = sigma_boot^2 + s_seeds^2 / 3, the
             T2 bootstrap tables), written to <work>/<set>/ (the files of results/addendum_g_g1_v1*, hss_eval_v1*,
             hnl_points_eval_v1*). Checks, before anything is written to results/: every row of every other model in
             the main, bins, differences and points tables and the G1 claim equal the committed R14 files; the two
             models' seed-0 efficiencies (eff_s0) and population sizes equal their committed one-seed rows. Then
             <out>/<name>.csv (every row of the two models: main populations and fake rate, bins, each HNL point beside
             its reference) and .md.
  cleanup    remove one intermediate file of these runs, after the next stage's outputs exist and pass a check, and
             log it (ADDG_CLEANUP_LOG: time, kind, path, bytes, the check, job): acorn (acorn_iii_s<S>/<set>/testset.tar
             once the Canto-L (ttbar+HNL) s<S> scored tar holds exactly its events), scored (<model>_s<S>/<set>/scored/
             testset.tar once the merged evaluation covers every scored graph) and testrun (the HNL-test union and
             score tars of slurm/ts_seeds12/chain.sh's mirror runs once the test evaluation is merged and the R19
             read-outs are written). Only regular files under the runs_extra/ts_seeds12 tree, never through a link.
             Permission and scope: DECISIONS.md and ledger R20 (the first author, 2026-10-17, option (a)).

    python -m hnl.eval.addendum_g_seeds12 summarize --out results --name addendum_g_canto_hnl_seeds12_v1
    python -m hnl.eval.addendum_g_seeds12 cleanup scored --model canto_l_hnl --seed 1 --set test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tarfile

import numpy as np
import pandas as pd

from hnl.eval import addendum_g as G

TREE = "/work/submit/jinboz1/hnl/runs_extra/ts_seeds12"
NEW = ("canto_l_hnl", "canto_l_tthnl")
SETS = ("test", "hss", "hnl")
SUFFIXES = (".csv", "_bins.csv", "_differences.csv", "_points.csv")
TOL = 1e-12


def _members(tar: str) -> list[str]:
    with tarfile.open(tar) as t:
        return sorted(m for m in t.getnames() if m.endswith(".pyg.gz"))


# ------------------------------------------------------------------------------------------------------- cleanup
def _remove(path: str, kind: str, check: str) -> None:
    real = os.path.realpath(path)
    if os.path.islink(path) or not real.startswith(TREE + os.sep) or not os.path.isfile(real) or real != path:
        raise SystemExit(f"refusing to remove {path}: not a regular file of {TREE} reached without a link")
    size = os.path.getsize(path)
    os.remove(path)
    log = os.environ.get("ADDG_CLEANUP_LOG", os.path.join(TREE, "cleanup_log.tsv"))
    new = not os.path.exists(log)
    with open(log, "a") as f:
        if new:
            f.write("utc\tkind\tpath\tbytes\tcheck\tslurm_job\n")
        f.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{kind}\t{path}\t{size}\t{check}\t"
                f"{os.environ.get('SLURM_JOB_ID', 'local')}\n")
    print(f"[cleanup] removed {path} ({size} bytes); {check}")


def cleanup(a) -> None:
    if not os.path.realpath(G.OUT).startswith(TREE + os.sep):
        raise SystemExit(f"ADDG_OUT {G.OUT} is not under {TREE}")
    if a.kind == "acorn":
        src = os.path.join(G.OUT, f"acorn_iii_s{a.seed}", a.set, "testset.tar")
        sc = os.path.join(G.OUT, f"canto_l_tthnl_s{a.seed}", a.set, "scored")
        if not os.path.exists(os.path.join(sc, "done")):
            raise SystemExit(f"{sc} not done")
        n_src, n_sc = len(_members(src)), len(_members(os.path.join(sc, "testset.tar")))
        parts = json.load(open(os.path.join(sc, "parts.json")))
        if not (n_src == n_sc == parts["n_graphs"]):
            raise SystemExit(f"acorn source {n_src}, scored {n_sc}, parts.json {parts['n_graphs']}")
        _remove(src, "acorn", f"scored tar and parts.json hold the {n_src} graphs of the acorn source")
    elif a.kind == "scored":
        d = os.path.join(G.OUT, f"{a.model}_s{a.seed}", a.set, "scored")
        ev = os.path.join(G.OUT, "eval", f"{a.model}_s{a.seed}", a.set)
        if os.path.islink(os.path.join(G.OUT, "eval", f"{a.model}_s{a.seed}")):
            raise SystemExit(f"{a.model} s{a.seed}: an R14 run (linked), nothing of it is removed")
        n_sc = len(_members(os.path.join(d, "testset.tar")))
        prov = json.load(open(os.path.join(ev, "provenance.json")))
        pe = pd.read_parquet(os.path.join(ev, "per_event.parquet"))
        nb = len(prov["blocks"])
        tracks = sum(os.path.exists(os.path.join(ev, "blocks", f"{k:03d}_of_{nb:03d}", "tracks.parquet")) for k in range(nb))
        if not (prov["n_events"] == n_sc == len(pe) == pe.event_id.nunique()) or tracks != nb:
            raise SystemExit(f"merged evaluation {prov['n_events']} / {len(pe)} events, scored {n_sc}, tracks {tracks}/{nb}")
        _remove(os.path.join(d, "testset.tar"), "scored",
                f"merged evaluation covers the {n_sc} scored graphs ({nb} blocks with stored tracks)")
    elif a.kind == "testrun":
        sel = json.load(open(os.path.join(G.REPO, G.SEL_TS_SEEDS12)))
        e = sel["pipelines"][f"l1tl_{a.arm}"]["seeds"][str(a.seed)]
        run, grun = e["run"], e["graphs_run"]
        for f in ("final_test.done", "score_test.done", "final_selection_used.json"):
            if not os.path.exists(os.path.join(run, f)):
                raise SystemExit(f"{run}/{f} missing")
        for f in ("training_sample_test_seeds12_v1_meta.json", "bootstrap_ts_test_3seed_v1.csv",
                  "training_sample_paired_test_3seed_v1.csv"):
            if not os.path.exists(os.path.join(G.REPO, "results", f)):
                raise SystemExit(f"results/{f} missing: the R19 read-outs come first")
        union, scored = os.path.join(grun, "ltf_union_graphs", "testset.tar"), os.path.join(run, "gnn_scores", "testset.tar")
        n_u, n_s = _members(union), _members(scored)
        tb = [pd.read_parquet(os.path.join(run, s, "per_event.parquet")).event_id.nunique()
              for s in ("trackbuild_test", "trackbuild_test_llpblind", "eval_test")]
        if n_u != n_s or len(n_u) != 7999 or tb != [7999, 7999, 7999]:
            raise SystemExit(f"union {len(n_u)}, scored {len(n_s)}, evaluated {tb}")
        chk = "test evaluation merged (final_test.done; 7999 events in trackbuild_test, _llpblind and eval_test) and " \
              "the R19 read-outs written; union and scored tars hold the same 7999 graphs"
        _remove(union, "testrun", chk)
        _remove(scored, "testrun", chk)
    else:
        raise SystemExit(a.kind)


# ----------------------------------------------------------------------------------------------------- summarize
def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            x, y = float(a), float(b)
        except (TypeError, ValueError):
            return str(a) == str(b)
        if np.isnan(x) or np.isnan(y):
            return bool(np.isnan(x) and np.isnan(y))
        return abs(x - y) <= TOL
    return (pd.isna(a) and pd.isna(b)) or str(a) == str(b)


def compare(set_: str, work: str, checks: list) -> None:
    """The new summary of a set against the committed R14 files (see the module docstring)."""
    from hnl.eval.addendum_g_summary import STEM
    stem = STEM[set_]
    for suf in SUFFIXES:
        ref_p, new_p = os.path.join(G.REPO, "results", stem + suf), os.path.join(work, stem + suf)
        if not os.path.exists(ref_p):
            checks.append((f"{set_}{suf}: no committed file", True))
            continue
        ref = pd.read_csv(ref_p, float_precision="round_trip", keep_default_na=False, na_values=[""])
        new = pd.read_csv(new_p, float_precision="round_trip", keep_default_na=False, na_values=[""])
        mcol = "model" if "model" in ref.columns else None
        keys = [c for c in ("sample", "population", "variable", "bin", "point", "reference", "A", "B", "model")
                if c in ref.columns]
        old_rows = ref[~ref[mcol].isin(NEW)] if mcol else ref
        new_rows = new[~new[mcol].isin(NEW)] if mcol else new
        checks.append((f"{set_}{suf}: rows of the other models ({len(old_rows)})", len(old_rows) == len(new_rows)))
        o = old_rows.set_index(keys).sort_index()
        n = new_rows.set_index(keys).sort_index()
        same_index = list(o.index) == list(n.index)
        checks.append((f"{set_}{suf}: same keys of the other models", same_index))
        if same_index:
            bad = [(k, c) for c in o.columns if c in n.columns for k, x, y in zip(o.index, o[c], n[c]) if not _same(x, y)]
            missing = [c for c in o.columns if c not in n.columns]
            checks.append((f"{set_}{suf}: every value of the other models identical ({len(bad)} differ, missing "
                           f"columns {missing})", not bad and not missing))
        if mcol and suf in (".csv", "_bins.csv"):
            o1 = ref[ref[mcol].isin(NEW)].set_index(keys).sort_index()
            n1 = new[new[mcol].isin(NEW)].set_index(keys).sort_index()
            ok = list(o1.index) == list(n1.index)
            checks.append((f"{set_}{suf}: Canto-L (HNL) / (ttbar+HNL) rows ({len(o1)})", ok))
            if ok:
                bad = [k for k, x, y, nx, ny, s in zip(o1.index, o1.eff_s0, n1.eff_s0, o1.n, n1.n, n1.n_seeds)
                       if not (_same(x, y) and _same(nx, ny) and int(s) == 3)]
                checks.append((f"{set_}{suf}: their eff_s0 and n equal the committed one-seed rows, three seeds now "
                               f"({len(bad)} differ)", not bad))
    if set_ == "test":
        ref = json.load(open(os.path.join(G.REPO, "results", stem + "_claim.json")))
        new = json.load(open(os.path.join(work, stem + "_claim.json")))
        checks.append(("test: claim G1 record identical", ref == new))


def _pc(x, d=2):
    return "" if pd.isna(x) else f"{100 * float(x):.{d}f}"


def markdown(tabs: dict, checks: list, meta: dict) -> str:
    lab = G.LABELS
    L = ["# Canto (HNL) and Canto (ttbar+HNL) with three seeds on the generated samples (addendum G T2 sets; seeds 1 "
         "and 2 added)", "",
         f"Written {meta['finished_utc']} by `python -m hnl.eval.addendum_g_seeds12 summarize` (git {meta['git']}; job "
         f"{meta['slurm_job']}). Post-freeze, descriptive, no claim (docs/feedback/test_use_ledger.md R20; seed 0 = R14). "
         "Canto-L (HNL) and Canto-L (ttbar+HNL) are the paper's Canto (HNL) and Canto (ttbar+HNL); seeds 1 and 2 as "
         "selected on validation in results/training_sample_test_selection_seeds12_v1.json, run through the T2 code "
         "(hnl.eval.addendum_g, results/addendum_g_models_seeds12_v1.json). Efficiencies in percent, three-seed mean "
         "+- sigma with sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (paper Eq. (2); the T2 bootstrap tables, B = 500); the "
         "other models are the R14 rows, reproduced exactly. These samples carry the simulation's own labels "
         "(R14).", "",
         f"Checks: {sum(c[1] for c in checks)} of {len(checks)} passed (identity of every other model's row and of "
         "claim G1 with the committed R14 files; seed-0 values of the two models).", ""]
    for set_ in SETS:
        main = tabs[set_]["main"]
        for s in dict.fromkeys(main["sample"]):
            sub = main[main["sample"] == s]
            pops = list(dict.fromkeys(sub.population))
            L += [f"## {s} ({set_})", "", "| model | " + " | ".join(pops) + " |", "|---|" + "---|" * len(pops)]
            for m in dict.fromkeys(sub.model):
                cells = []
                for p in pops:
                    r = sub[(sub.model == m) & (sub.population == p)].iloc[0]
                    cells.append(f"{_pc(r['mean'])} +- {_pc(r['sigma'])}")
                L.append(f"| {lab[m]} | " + " | ".join(cells) + " |")
            L += ["", "Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: " + "; ".join(
                f"{lab[m]} {p}: " + " / ".join(_pc(r[f'eff_s{i}']) for i in range(3))
                for m in NEW for p in (pops[0], "track fake rate")
                for r in [sub[(sub.model == m) & (sub.population == p)].iloc[0]]) + ".", ""]
        if set_ == "hnl" and tabs[set_].get("points") is not None:
            pts = tabs[set_]["points"]
            L += ["## Each HNL point beside its in-stack reference (three-seed means; difference point - reference)", ""]
            for (p, ref), g in pts.groupby(["point", "reference"], sort=False):
                pops = list(dict.fromkeys(g.population))
                L += [f"**{p} beside {ref}**", "", "| model | " + " | ".join(pops) + " |", "|---|" + "---|" * len(pops)]
                for m in dict.fromkeys(g.model):
                    cells = []
                    for q in pops:
                        r = g[(g.model == m) & (g.population == q)].iloc[0]
                        cells.append(f"{_pc(r['point_mean'])} / {_pc(r['reference_mean'])} / {r['delta_points']:+.2f} +- "
                                     f"{r['sigma_points']:.2f}")
                    L.append(f"| {lab[m]} | " + " | ".join(cells) + " |")
                L.append("")
    L += ["Bins of the two models (three seeds) and every row: the CSV next to this file; the full T2 summaries of "
          f"this read-out (all models): {meta['work']}.", ""]
    return "\n".join(L)


def summarize(a) -> None:
    from hnl.eval import addendum_g_summary as S
    from hnl.eval.addendum_g_summary import STEM
    from hnl.eval.per_sample import git_state
    if G.TS_SEEDS != (0, 1, 2):
        raise SystemExit("run with ADDG_TS_SEEDS=0,1,2 (slurm/ts_seeds12/g_env.sh)")
    outs = [os.path.join(a.out, a.name + x) for x in (".csv", ".md", "_meta.json")]
    if any(os.path.exists(p) for p in outs):
        raise SystemExit(f"output exists: {outs}")
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_root = os.path.join(G.OUT, "eval")
    checks, tabs = [], {}
    for set_ in SETS:
        work = os.path.join(a.work, set_)
        if os.path.exists(work) and os.listdir(work):
            raise SystemExit(f"{work} is not empty")
        S.summary(set_, out_dir=work, eval_root=eval_root)
        compare(set_, work, checks)
        stem = STEM[set_]
        tabs[set_] = {"main": pd.read_csv(os.path.join(work, stem + ".csv")),
                      "bins": pd.read_csv(os.path.join(work, stem + "_bins.csv"))}
        pp = os.path.join(work, stem + "_points.csv")
        tabs[set_]["points"] = pd.read_csv(pp) if os.path.exists(pp) else None
    bad = [c for c in checks if not c[1]]
    for c in checks:
        print(("ok   " if c[1] else "FAIL ") + c[0])
    if bad:
        raise SystemExit(f"{len(bad)} of {len(checks)} checks failed: nothing written to {a.out}")
    rows = []
    for set_ in SETS:
        for tab in ("main", "bins", "points"):
            t = tabs[set_][tab]
            if t is None:
                continue
            t = t[t.model.isin(NEW)].drop(columns="table", errors="ignore").copy()   # main/bins carry their own
            t.insert(0, "table", tab)
            t.insert(0, "set", set_)
            rows.append(t)
    df = pd.concat(rows, ignore_index=True)
    meta = {"name": a.name, "git": git_state(), "slurm_job": os.environ.get("SLURM_JOB_ID", "local"),
            "started_utc": started, "eval_root": eval_root, "work": a.work, "ts_seeds": list(G.TS_SEEDS),
            "checks": [[c[0], c[1]] for c in checks],
            "finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(outs[0], index=False)
    with open(outs[1], "w") as f:
        f.write(markdown(tabs, checks, meta))
    with open(outs[2], "w") as f:
        json.dump(meta, f, indent=1)
    print(open(outs[1]).read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--out", default=os.path.join(G.REPO, "results"))
    p.add_argument("--name", default="addendum_g_canto_hnl_seeds12_v1")
    p.add_argument("--work", default=os.path.join(TREE, "summary_v1"))
    p = sub.add_parser("cleanup")
    p.add_argument("kind", choices=["acorn", "scored", "testrun"])
    p.add_argument("--model", choices=list(NEW))
    p.add_argument("--arm", choices=["ii", "iii"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", choices=list(SETS) + ["val"])
    a = ap.parse_args(argv)
    {"summarize": summarize, "cleanup": cleanup}[a.cmd](a)


if __name__ == "__main__":
    main()
