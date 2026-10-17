"""Gate G3 checks of the exact pre-filtered sequential fragment merging (hnl.trackml.seqmerge; plan section 6.7).

    python -m hnl.trackml.g3check env --out DIR
        versions, CPU, and whether numpy's row sums depend on the padded width (pairwise summation)
    python -m hnl.trackml.g3check reference --scores DIR --event 1000 --out DIR [--config walkmerge,0.1,0.5,1.0]
        one full event: hnl.trackml.seqmerge.merge_sequential against the unchanged, unfiltered
        hnl.trackbuild.merge_fragments.merge_fragments(mode="sequential"); without --config on the same-layer MWS
        (--s0) fragments (merge_check's input, about 3e6 candidates), with --config on that configuration's first pass
        after the any-order pass (the builder path). Instrumented (call-through wrappers, results unchanged): the
        candidate list of the original, every refit (rms, zrms, n_out, ...), the s-z iteration count K, padded width
        H, LinAlgError / lstsq fallback counts. Reports: identical candidates, the bounds against every candidate's
        actual fit, accepted pairs kept, bit-identical fits of the pinned batch, identical labels; --unpinned also
        refits the filtered pairs alone (no pinning pairs) to show the batch effects.
    python -m hnl.trackml.g3check timing --scores DIR --events 1000,...,1004 --out DIR [--builders ...]
        time per event and configuration (first pass, any-order pass, sequential pass), one process per event
    python -m hnl.trackml.g3check hnl --graphs DIR --out DIR [--max-events 5]
        HNL val events through hnl.eval.trackbuild_val.evaluate_event (the HNL builder code path) with every
        sequential merge_fragments call also done by merge_sequential and compared (the HNL result is the original's)
Scratch only; nothing here is a result. Dev events only on TrackML.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import sys
import time

import numpy as np

import hnl.trackbuild.merge_fragments as MF
import hnl.truth.helix as HX
from hnl.overlay import grant as GRANT
from hnl.trackml import seqmerge as SM

FIT_KEYS = ("rms", "zrms", "n_out", "z0", "cot", "d0", "R", "xc", "yc", "sigma")


# ------------------------------------------------------------------------------------------------ instrumentation
class Probe:
    """Call-through wrappers: record each refit of _merge, fit_line_sz's iteration count and stopping pattern, and
    the LinAlgError / lstsq fallback of fit_perigee. Results are those of the wrapped functions."""

    def __init__(self, line_sz_done: bool = True):
        self.fits, self.line_sz, self.cands = [], [], []
        self.solve_errors = 0
        self.lstsq_calls = 0
        self._orig = {}
        self.line_sz_done = line_sz_done

    def __enter__(self):
        o = self._orig
        o["fit_tracks"], o["fit_line_sz"] = MF.fit_tracks, HX.fit_line_sz
        o["solve"], o["lstsq"], o["_merge"] = np.linalg.solve, np.linalg.lstsq, MF._merge

        def fit_tracks(*a, **k):
            f = o["fit_tracks"](*a, **k)
            self.fits.append({kk: np.asarray(f[kk]).copy() for kk in ("track", *FIT_KEYS) if kk in f})
            return f

        def fit_line_sz(s, z, mask, scale=0.5, n_iter=30):
            out = o["fit_line_sz"](s, z, mask, scale=scale, n_iter=n_iter)
            rec = {"H": int(s.shape[1]), "T": int(s.shape[0])}
            if self.line_sz_done:
                rec.update(_replay_line_sz(s, z, mask, scale, n_iter))
            self.line_sz.append(rec)
            return out

        def solve(*a, **k):
            try:
                return o["solve"](*a, **k)
            except np.linalg.LinAlgError:
                self.solve_errors += 1
                raise

        def lstsq(*a, **k):
            self.lstsq_calls += 1
            return o["lstsq"](*a, **k)

        def _merge(cand, *a, **k):
            self.cands.append(cand)
            return o["_merge"](cand, *a, **k)
        MF.fit_tracks, HX.fit_line_sz = fit_tracks, fit_line_sz
        np.linalg.solve, np.linalg.lstsq, MF._merge = solve, lstsq, _merge
        return self

    def __exit__(self, *exc):
        o = self._orig
        MF.fit_tracks, HX.fit_line_sz = o["fit_tracks"], o["fit_line_sz"]
        np.linalg.solve, np.linalg.lstsq, MF._merge = o["solve"], o["lstsq"], o["_merge"]


def _replay_line_sz(s, z, mask, scale, n_iter):
    """fit_line_sz's loop again (same operations) with its stopping test kept: K (iterations it ran) and, per track,
    a bit per iteration that had not passed the test."""
    m = mask.astype(np.float64)
    w = m.copy()
    z0 = np.zeros(len(s))
    ct = np.zeros(len(s))
    notdone = np.zeros(len(s), dtype=np.uint32)
    K = None
    for it in range(n_iter):
        S0 = np.sum(w, 1)
        S1 = np.sum(w * s, 1)
        S2 = np.sum(w * s * s, 1)
        Z0 = np.sum(w * z, 1)
        Z1 = np.sum(w * s * z, 1)
        det = S0 * S2 - S1**2
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        ct_n = (S0 * Z1 - S1 * Z0) / det
        z0_n = (S2 * Z0 - S1 * Z1) / det
        done = (np.abs(ct_n - ct) < 1e-10) & (np.abs(z0_n - z0) < 1e-7)
        z0, ct = z0_n, ct_n
        res = z - z0[:, None] - ct[:, None] * s
        w = HX.huber_weights(res, scale) * m
        notdone |= (~done).astype(np.uint32) << np.uint32(it)
        if K is None and done.all():
            K = it + 1
    return {"K": K or n_iter, "notdone_bits": notdone}


# ------------------------------------------------------------------------------------------------ inputs
def _event_state(scores: str, event: int):
    from hnl.trackml import events as E
    from hnl.trackml.trackbuild import EventState, read_scores
    ev = next(E.iter_events([event], particles=False))
    a_, b_, sc = read_scores(scores, [event])[ev.event]
    return ev, EventState(ev, a_, b_, sc)


def _parse_cfg(spec: str) -> tuple:
    p = spec.split(",")
    return (p[0], *[float(x) for x in p[1:]])


def first_pass(cfg: tuple, st) -> tuple[np.ndarray, float]:
    """The configuration's labels before any merging, and its max_rms (hnl.trackml.trackbuild.build semantics)."""
    from hnl.trackbuild import builders as TB
    from hnl.trackbuild.mutex_ws import mws_layer, mws_two_pass
    kind = cfg[0]
    if kind == "walkmerge":
        return TB.walkthrough(st.n, st.a, st.b, st.sc, st.R, cfg[1], cfg[2]), float(cfg[3])
    if kind == "mwslayermerge":
        return mws_layer(st.n, st.a, st.b, st.sc, st.L, cfg[1]), 1.0
    if kind == "mws2passmerge":
        return mws_two_pass(st.n, st.a, st.b, st.sc, st.L, cfg[1], cfg[2]), 1.0
    raise ValueError(kind)


TRANSVERSE = ("rms", "n_out", "d0", "R", "xc", "yc", "sigma")     # never depend on the s-z iteration count
SZ = ("zrms", "z0", "cot")


def _mismatches(fm: dict, fr: dict, pos) -> dict:
    """Pairs whose value differs (bit for bit, NaN = NaN) between the pinned batch fm and the full batch fr[pos]."""
    return {k: int((~((fm[k] == fr[k][pos]) | (np.isnan(fm[k]) & np.isnan(fr[k][pos])))).sum()) for k in FIT_KEYS}


def _pinned_ok(mm, how, K_full, K_pinned, sz_at_K_full) -> bool:
    """Transverse fit identical; s-z fit identical with K pinned, or (certificate) K_pinned <= K_full and the
    pinned batch's s-z values at the full batch's K identical."""
    if mm is None:
        return True
    if any(mm[k] for k in TRANSVERSE):
        return False
    if how == "certificate":
        return K_pinned is not None and K_full is not None and K_pinned <= K_full and sz_at_K_full == 0
    return K_full == K_pinned and not any(mm[k] for k in SZ)


def _cands_array(cand) -> np.ndarray:
    return np.array([(int(a), int(b)) for a, b in cand], dtype=np.int64).reshape(-1, 2)


def _code(pairs: np.ndarray, base: int) -> np.ndarray:
    return pairs[:, 0] * base + pairs[:, 1]


def _write(out: str, name: str, obj) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(json.dumps(obj, default=float), flush=True)


# ------------------------------------------------------------------------------------------------ env
def cmd_env(a):
    import numba
    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(line.split(":", 1)[1].strip() for line in fh if line.startswith("model name"))
    except Exception:
        pass
    rng = np.random.default_rng(0)
    res = {}
    for n in (5, 9, 12, 20):
        x = rng.normal(size=(20000, n))
        base = np.sum(x, 1)
        for H in (n, 8, 16, 24, 32, 40):
            if H < n:
                continue
            p = np.zeros((len(x), H))
            p[:, :n] = x
            res[f"n{n}_H{H}_rows_differing_from_H{n}"] = int((np.sum(p, 1) != base).sum())
    t0 = time.time()
    SM.lam4(0.0, 0.0, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2)
    feats = getattr(np, "__cpu_features__", None) or getattr(np._core._multiarray_umath, "__cpu_features__", {})
    _write(a.out, "env.json", {"numpy": np.__version__, "numba": numba.__version__, "python": sys.version.split()[0],
                               "cpu": cpu, "avx512f": bool(feats.get("AVX512F")), "avx2": bool(feats.get("AVX2")),
                               "numba_first_call_s": round(time.time() - t0, 2), "row_sum_vs_padding": res,
                               "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID")})


# ------------------------------------------------------------------------------------------------ reference
def _cuts_of(name: str):
    from hnl.trackml.trackbuild import cut_set
    return SM.BinnedCuts.uniform() if name == "H" else cut_set(name)


def cmd_reference(a):
    """Filtered pass (merge_sequential) against the unfiltered definition on one full event (see the module doc)."""
    GRANT.check([a.event], "eval", out=a.out)        # overlay ids only under a grant (addendum H)
    from hnl.trackml.trackbuild import merge_fragments_fast
    from hnl.trackbuild.merge_fragments import merge_fragments
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    ev, st = _event_state(a.scores, a.event)
    cuts = _cuts_of(a.cuts)
    if a.config:
        cfg = _parse_cfg(a.config)
        lab0, max_rms = first_pass(cfg, st)
        lab0 = merge_fragments_fast(st.r, st.phi, st.z, lab0, max_rms=max_rms, mode="any",
                                    cuts=None if a.cuts == "H" else cuts)
        what = {"input": "builder path: first pass, then the any-order pass", "config": list(cfg)}
    else:
        from hnl.trackbuild.mutex_ws import mws_layer
        lab0, max_rms = mws_layer(st.n, st.a, st.b, st.sc, st.L, a.s0), 1.0
        what = {"input": f"same-layer MWS s0={a.s0} fragments (merge_check's input)"}
    if a.cuts == "H":
        cuts = SM.BinnedCuts.uniform(max_rms, a.max_zrms, a.max_out)
    kw = dict(max_dphi=a.max_dphi, max_gap_r=a.max_gap_r, min_span_mm=a.min_span_mm)
    r, phi, z = st.r, st.phi, st.z
    vol = MF._volume_from_radius(r, z)
    out = {"event": a.event, **what, "cuts": cuts.describe(), "window": kw, "method": a.method,
           "inputs_s": round(time.time() - t0, 1)}
    print(json.dumps(out, default=float), flush=True)

    # the full candidate set C and the complete bounds of every pair
    ok = lab0 >= 0
    trk, cnt = np.unique(lab0[ok], return_counts=True)
    trk = trk[cnt >= 2]
    tab = SM.Table(r, phi, z, lab0, trk, vol)
    ia, ib = SM.sequential_candidates(tab.R_in, tab.R_out, tab.phi_in, tab.phi_out, a.max_dphi, a.max_gap_r)
    n = (tab.fn[ia] + tab.fn[ib]).astype(np.float64)
    t1 = time.time()
    keep_full, lb, tau, span = SM.prefilter_cuts(tab, ia, ib, cuts, a.min_span_mm, full=True)
    out.update({"fragments": int(len(trk)), "candidates": int(len(ia)), "full_bounds_s": round(time.time() - t1, 1),
                "kept_by_filter": int(keep_full.sum()), "pruned_by_span": int((~(span >= a.min_span_mm)).sum()),
                "union_hits_max": int(n.max()) if len(n) else 0})

    # the filtered pass (instrumented)
    info = {}
    with Probe() as pm:
        t1 = time.time()
        mine, batch = SM.merge_sequential(r, phi, z, lab0, cuts=cuts, method=a.method, info=info, return_batch=True,
                                          **kw)
        t_mine = time.time() - t1
    out.update({"mine_s": round(t_mine, 1), "mine_info": info, "mine_lstsq_calls": pm.lstsq_calls,
                "mine_solve_errors": pm.solve_errors})
    print(json.dumps(out, default=float), flush=True)
    if a.skip_original:
        out["slurm_job"] = os.environ.get("SLURM_JOB_ID")
        _write(a.out, f"quick_{a.event}_{a.tag}.json", out)
        return

    # the unfiltered definition (instrumented): H the original code itself, C every candidate with its bin's cuts
    with Probe() as pr:
        t1 = time.time()
        if a.cuts == "H":
            ref = merge_fragments(r, phi, z, lab0, mode="sequential", max_rms=max_rms, max_zrms=a.max_zrms,
                                  max_out=a.max_out, **kw)
        else:
            ref = SM.merge_sequential_reference(r, phi, z, lab0, cuts=cuts, **kw)
        t_ref = time.time() - t1
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    C_mine = np.stack([trk[ia], trk[ib]], 1)
    out.update({"original_s": round(t_ref, 1), "peak_rss_gb": round(rss, 1),
                "original_lstsq_calls": pr.lstsq_calls, "original_solve_errors": pr.solve_errors,
                "identical_labels": bool(np.array_equal(ref, mine)),
                "merged_fragments": int(len(np.unique(lab0[ok])) - len(np.unique(ref[ref >= 0])))})
    if a.cuts == "H":
        out["identical_candidates"] = bool(np.array_equal(_cands_array(pr.cands[0]), C_mine)) if pr.cands else None
    print(json.dumps(out, default=float), flush=True)

    # every candidate's fit in the full batch (the _merge call of the definition): bounds, acceptance, pre-fit values
    fr = pr.fits[-1]
    assert np.array_equal(fr["track"], np.arange(len(ia)))
    cr, cz, co = cuts.for_cot(fr["cot"])
    q_ok = (fr["rms"] <= cr) & (fr["zrms"] <= cz) & (fr["n_out"] <= co) & (span >= a.min_span_mm)
    qf = (fr["rms"] + 0.1 * fr["zrms"])[q_ok]
    out.update({"accepted_by_cuts": int(q_ok.sum()), "accepted_not_kept": int((q_ok & ~keep_full).sum()),
                "lb_violations": int((lb > n * fr["rms"] ** 2 * (1 + 1e-9) + 1e-12).sum()),
                "tau_violations": int((tau > fr["n_out"]).sum()),
                "finite_q_ties_full_batch": int(len(qf) - len(np.unique(qf)))})
    ls_r, ls_m = (pr.line_sz[-1] if pr.line_sz else {}), (pm.line_sz[-1] if pm.line_sz else {})
    out.update({"K_original": ls_r.get("K"), "K_mine": ls_m.get("K"), "H_original": ls_r.get("H"),
                "H_mine": ls_m.get("H")})
    if ls_r:
        nd = ls_r["notdone_bits"]
        full29 = np.uint32((1 << 29) - 1)
        out["never_done_by_29_frac"] = float(np.mean((nd & full29) == full29))
    code_all = _code(C_mine, int(trk.max()) + 1)
    if batch:
        bpairs = _cands_array(batch)
        pos = np.minimum(np.searchsorted(code_all, _code(bpairs, int(trk.max()) + 1)), len(code_all) - 1)
        out["batch_subset_of_C"] = bool(np.array_equal(code_all[pos], _code(bpairs, int(trk.max()) + 1)))
        if pm.fits:
            out["merge_batch_fit_mismatches"] = _mismatches(pm.fits[-1], fr, pos)
    if info.get("method") == "prefit" and "h" in info:
        # the pre-fit values of every kept pair and pinning pair against the full batch, bit for bit
        Fp = np.union1d(np.flatnonzero(keep_full), np.r_[info["h"], info["W"]])
        t1 = time.time()
        vals = SM.prefit(tab, ia, ib, Fp, info["h"], np.asarray(info["W"], dtype=np.int64), r, phi, z, vol)
        out["prefit_check_s"] = round(time.time() - t1, 1)
        out["prefit_mismatches"] = {k: int((~((vals[k] == fr[k][Fp]) | (np.isnan(vals[k]) & np.isnan(fr[k][Fp])))).sum())
                                    for k in ("rms", "zrms", "n_out", "cot", "z0", "d0", "R")}
    # the batch effects without pinning: a random sample of the kept pairs fit alone (own H and K)
    if a.unpinned:
        rng = np.random.default_rng(1)
        k = np.flatnonzero(keep_full)
        sel = np.sort(rng.choice(k, min(50000, len(k)), replace=False))
        rows, plab = SM.union_rows(tab, ia[sel], ib[sel])
        with Probe() as pu:
            fu = MF.fit_tracks(r[rows], phi[rows], z[rows], vol[rows], plab, min_hits=3)
        rel = {kk: float(np.nanmax(np.abs(fu[kk] - fr[kk][sel]) / np.maximum(np.abs(fr[kk][sel]), 1e-300)))
               for kk in ("rms", "zrms")}
        out["unpinned_sample"] = {"pairs": int(len(sel)), "fit_mismatches": _mismatches(fu, fr, sel),
                                  "max_rel_diff": rel, "K": pu.line_sz[0].get("K") if pu.line_sz else None,
                                  "H": pu.line_sz[0].get("H") if pu.line_sz else None}
    mm = out.get("merge_batch_fit_mismatches") or {}
    pm_ = out.get("prefit_mismatches") or {}
    out["ok"] = bool(out["identical_labels"] and out.get("identical_candidates") is not False
                     and out["accepted_not_kept"] == 0 and out["lb_violations"] == 0 and out["tau_violations"] == 0
                     and not any(mm.values()) and not any(pm_.values()) and info.get("consistent", True)
                     and out["finite_q_ties_full_batch"] == 0)
    out["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    np.savez_compressed(os.path.join(a.out, f"labels_{a.event}_{a.tag}.npz"), first=lab0, original=ref, mine=mine)
    _write(a.out, f"reference_{a.event}_{a.tag}.json", out)


# ------------------------------------------------------------------------------------------------ timing
def _timing_group(args):
    """One (event, cut set, family) group of the timing grid: every configuration timed in this process."""
    scores, event, cs, items, check, n_proc = args
    import hnl.trackml.trackbuild as TBM
    from hnl.trackml.trackbuild import CUT_SET_FLAGS, build, cut_set, merge_fragments_fast
    TBM.PREFIT_PROCS = n_proc                           # the any-order pass too (as HNL_TML_PREFIT_PROCS in a build)
    t0 = time.time()
    ev, st = _event_state(scores, event)
    rows = [{"event": event, "config": -1, "family": "load", "cut_set": cs, "seconds": round(time.time() - t0, 2)}]
    cuts = cut_set(cs)
    for i, cfg in items:
        t0 = time.time()
        try:
            lab, max_rms = first_pass(cfg, st)
            t1 = time.time()
            lab_any = merge_fragments_fast(st.r, st.phi, st.z, lab, max_rms=max_rms, mode="any", cuts=cuts)
            t2 = time.time()
            info = {}
            out = SM.merge_sequential(st.r, st.phi, st.z, lab_any, max_rms=max_rms, cuts=cuts, info=info,
                                      n_proc=n_proc)
            t3 = time.time()
        except Exception as exc:                        # record it and go on with the grid
            row = {"event": event, "config": i, "family": cfg[0], "params": [float(x) for x in cfg[1:]],
                   "cut_set": cs, "error": repr(exc), "seconds": round(time.time() - t0, 2)}
            print(json.dumps(row, default=float), flush=True)
            rows.append(row)
            continue
        row = {"event": event, "config": i, "family": cfg[0], "params": [float(x) for x in cfg[1:]], "cut_set": cs,
               "n_proc": n_proc, "first_pass_s": round(t1 - t0, 2), "any_s": round(t2 - t1, 2),
               "sequential_s": round(t3 - t2, 2), "seconds": round(t3 - t0, 2),
               **{f"seq_{k}": v for k, v in info.items() if k not in ("W", "class_widths")},
               "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
        if check:
            row["build_identical"] = bool(np.array_equal(build((*cfg, CUT_SET_FLAGS[cs]), st), out))
        print(json.dumps(row, default=float), flush=True)
        rows.append(row)
    return rows


def cmd_timing(a):
    import concurrent.futures as cf
    import pandas as pd
    from hnl.eval.trackbuild_val import configs
    GRANT.check(a.events, "eval", out=a.out)         # overlay ids only under a grant (addendum H)
    cfgs = [_parse_cfg(c) for c in a.configs.split(";")] if a.configs else configs(a.builders.split(","))
    events = [int(e) for e in a.events.split(",")]
    cut_sets = a.cut_sets.split(",")
    os.makedirs(a.out, exist_ok=True)
    groups = []
    for e in events:
        for cs in cut_sets:
            for fam in sorted({c[0] for c in cfgs}):
                items = [(i, c) for i, c in enumerate(cfgs) if c[0] == fam]
                groups.append((a.scores, e, cs, items, a.check_build, a.n_proc))
    groups.sort(key=lambda g: (-len(g[3]) * (2 if g[2] == "C" else 1), g[1]))     # longest first
    t0 = time.time()
    if a.workers > 1:                                   # workers x n_proc processes in all
        with cf.ProcessPoolExecutor(max_workers=min(a.workers, len(groups))) as ex:
            res = list(ex.map(_timing_group, groups, chunksize=1))
    else:
        res = [_timing_group(g) for g in groups]
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(a.out, "timing.csv"), index=False)
    t = df[df.config >= 0]
    per = t.groupby(["family", "cut_set"]).seconds.describe().round(1)
    seq = t.groupby(["family", "cut_set"]).sequential_s.describe().round(1)
    per.index = [f"{f}/{c}" for f, c in per.index]
    seq.index = [f"{f}/{c}" for f, c in seq.index]
    _write(a.out, "timing_summary.json", {
        "events": events, "configurations": len(cfgs), "cut_sets": cut_sets, "workers": a.workers,
        "n_proc": a.n_proc, "max_seconds": float(t.seconds.max()), "limit_seconds": 180.0,
        "within_limit": bool(t.seconds.max() <= 180.0),
        "max_seconds_by_cut_set": t.groupby("cut_set").seconds.max().to_dict(),
        "per_family_total_s": per.to_dict(orient="index"), "per_family_sequential_s": seq.to_dict(orient="index"),
        "consistent_all": bool(t.get("seq_consistent", pd.Series([True])).fillna(True).all()),
        "ties_total": int(t.get("seq_finite_q_ties", pd.Series([0])).fillna(0).sum()),
        "build_identical_all": bool(t.build_identical.all()) if "build_identical" in t else None,
        "errors": int(t["error"].notna().sum()) if "error" in t else 0,
        "union_hits_max": int(t["seq_H"].max()) if "seq_H" in t else None,
        "wall_s": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID")})


# ------------------------------------------------------------------------------------------------ options
def partition_diff(ref: np.ndarray, other: np.ndarray) -> dict:
    """Hits (label >= 0 in either) whose track, as a set of hits, differs between two labelings; track counts."""
    ref, other = np.asarray(ref, dtype=np.int64), np.asarray(other, dtype=np.int64)
    ok = (ref >= 0) | (other >= 0)
    r_, o_ = ref[ok], other[ok]
    r_ = np.where(r_ >= 0, r_, -1 - np.arange(len(r_)))          # unlabelled hits: singletons
    o_ = np.where(o_ >= 0, o_, -1 - np.arange(len(o_)))
    _, ri, rc = np.unique(r_, return_inverse=True, return_counts=True)
    _, oi, oc = np.unique(o_, return_inverse=True, return_counts=True)
    pair = ri.astype(np.int64) * (oi.max() + 1) + oi
    _, pi, pc = np.unique(pair, return_inverse=True, return_counts=True)
    same = (pc[pi] == rc[ri]) & (pc[pi] == oc[oi])
    return {"hits_in_different_tracks": int((~same).sum()), "hits_labelled": int(ok.sum()),
            "tracks_ref": int(len(np.unique(ref[ref >= 0]))), "tracks_other": int(len(np.unique(other[other >= 0])))}


def chunked_merge(r, phi, z, labels, n_chunks: int, max_rms: float, max_zrms=5.0, max_out=1, min_span_mm=50.0,
                  max_dphi=0.3, max_gap_r=400.0, prefilter: bool = True) -> tuple[np.ndarray, list]:
    """Option 'chunked _merge' (NOT the definition): the candidates (optionally the pre-filtered ones) cut into
    n_chunks contiguous blocks of C's order, each handed to the unchanged _merge with the original labels (each its own
    batch and greedy), and the merges of all blocks combined by union. Returns labels and per-chunk seconds."""
    labels = np.asarray(labels).copy()
    vol = MF._volume_from_radius(r, z)
    ok = labels >= 0
    trk, cnt = np.unique(labels[ok], return_counts=True)
    trk = trk[cnt >= 2]
    tab = SM.Table(r, phi, z, labels, trk, vol)
    ia, ib = SM.sequential_candidates(tab.R_in, tab.R_out, tab.phi_in, tab.phi_out, max_dphi, max_gap_r)
    if prefilter:
        keep, *_ = SM.prefilter(tab, ia, ib, max_rms, max_out, min_span_mm, max_zrms)
        ia, ib = ia[keep], ib[keep]
    parent = {int(t): int(t) for t in trk}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    secs = []
    for ch in np.array_split(np.arange(len(ia)), max(n_chunks, 1)):
        if not len(ch):
            continue
        t0 = time.time()
        out = MF._merge(list(zip(trk[ia[ch]], trk[ib[ch]])), trk, labels.copy(), r, phi, z, vol, max_rms, max_zrms,
                        max_out, one_per_end=True, min_span_mm=min_span_mm)
        secs.append(round(time.time() - t0, 1))
        moved = np.flatnonzero(out != labels)
        for f, root in set(zip(labels[moved].tolist(), out[moved].tolist())):
            ra, rb = find(root), find(f)
            if ra != rb:
                parent[rb] = ra
    res = labels.copy()
    for t in trk.tolist():
        rt = find(t)
        if rt != t:
            res[labels == t] = rt
    return res, secs


def cmd_options(a):
    """Exact pass against the options of plan section 6.7 / task point 5 on one event: tighter sequential windows
    (exact for their window), the any-order pass only, chunked _merge (H only); time, partition differences against
    the exact labels, and DM efficiency / fake rate. Per configuration and cut set."""
    GRANT.check([a.event], "eval", out=a.out)        # overlay ids only under a grant (addendum H)
    from hnl.trackml import events as E
    from hnl.trackml.evaluate import event_counts
    from hnl.trackml.trackbuild import EventState, cut_set, merge_fragments_fast, read_scores
    os.makedirs(a.out, exist_ok=True)
    ev = next(E.iter_events([a.event], particles=True))
    a_, b_, sc = read_scores(a.scores, [a.event])[ev.event]
    st = EventState(ev, a_, b_, sc)
    r, phi, z = st.r, st.phi, st.z

    def metrics(L):
        c, _ = event_counts(ev.pid, L, ev.particles)
        return {"eff_target": round(c["m_target"] / max(c["n_target"], 1), 5),
                "fake_rate": round(c["n_fake_ge5"] / max(c["n_tracks_ge5"], 1), 5), "tracks_ge5": c["n_tracks_ge5"]}
    for spec in a.configs.split(";"):
        cfg = _parse_cfg(spec)
        for cs in a.cut_sets.split(","):
            cuts = cut_set(cs)
            t0 = time.time()
            lab, max_rms = first_pass(cfg, st)
            t1 = time.time()
            lab_any = merge_fragments_fast(r, phi, z, lab, max_rms=max_rms, mode="any", cuts=cuts)
            t2 = time.time()
            res = {"event": a.event, "config": list(cfg), "cut_set": cs, "first_pass_s": round(t1 - t0, 1),
                   "any_s": round(t2 - t1, 1)}
            info = {}
            t0 = time.time()
            exact = SM.merge_sequential(r, phi, z, lab_any, max_rms=max_rms, cuts=cuts, info=info)
            info.pop("W", None)
            res["exact"] = {"sequential_s": round(time.time() - t0, 1), "info": info, **metrics(exact),
                            **partition_diff(lab_any, exact)}
            print(json.dumps(res, default=float), flush=True)
            res["any_only"] = {"sequential_s": 0.0, **metrics(lab_any), **partition_diff(exact, lab_any)}
            for w in a.windows.split(";"):
                dphi, gap = (float(x) for x in w.split(","))
                info = {}
                t0 = time.time()
                L = SM.merge_sequential(r, phi, z, lab_any, max_rms=max_rms, cuts=cuts, max_dphi=dphi, max_gap_r=gap,
                                        info=info)
                info.pop("W", None)
                key = f"window_dphi{dphi:g}_gap{gap:g}"
                res[key] = {"sequential_s": round(time.time() - t0, 1), "info": info, **metrics(L),
                            **partition_diff(exact, L)}
                print(json.dumps({"window": w, "cut_set": cs, **res[key]}, default=float), flush=True)
            if cs == "H":
                for nch in (int(x) for x in a.chunks.split(",") if x):
                    L, secs = chunked_merge(r, phi, z, lab_any, nch, max_rms)
                    key = f"chunked_{nch}_filtered"
                    res[key] = {"chunk_s_max": max(secs) if secs else 0.0, "chunk_s_sum": round(sum(secs), 1),
                                **metrics(L), **partition_diff(exact, L)}
                    print(json.dumps({"chunks": nch, **res[key]}, default=float), flush=True)
            _write(a.out, f"options_{a.event}_{'_'.join(str(x) for x in cfg)}_{cs}.json", res)


def cmd_anycheck(a):
    """The any-order pass (hnl.trackml.trackbuild.merge_fragments_fast mode any: exact pre-fit, acceptance loop, the
    unchanged _merge on the merges) against its definition: every any-order candidate (the candidate search verified
    identical to the original loops) handed to the unchanged _merge, with per-candidate cuts from a fit of that very
    batch for a cut set. Per configuration and cut set on one full event."""
    GRANT.check([a.event], "eval", out=a.out)        # overlay ids only under a grant (addendum H)
    import hnl.trackml.trackbuild as TBM
    os.makedirs(a.out, exist_ok=True)
    TBM.PREFIT_PROCS = a.n_proc
    ev, st = _event_state(a.scores, a.event)
    r, phi, z = st.r, st.phi, st.z
    vol = MF._volume_from_radius(r, z)
    rows_out = []
    for spec in a.configs.split(";"):
        cfg = _parse_cfg(spec)
        lab, max_rms = first_pass(cfg, st)
        for cs in a.cut_sets.split(","):
            cuts = TBM.cut_set(cs)
            t0 = time.time()
            with Probe(line_sz_done=False) as pm:
                fast = TBM.merge_fragments_fast(r, phi, z, lab, max_rms=max_rms, mode="any", cuts=cuts)
            t_fast = time.time() - t0
            # the definition: all candidates to the unchanged _merge (one batch)
            t0 = time.time()
            ok = lab >= 0
            trk, cnt = np.unique(lab[ok], return_counts=True)
            trk = trk[cnt >= 2]
            hits = TBM._tracks_of(lab, trk)
            cand = TBM._candidates_any_fast(trk, hits, r, phi, z, np.hypot(r, z), 0.05, 0.05, n_proc=a.n_proc)
            cset = SM.BinnedCuts.uniform(max_rms) if cuts is None else cuts
            if cset.scalar:
                cr, cz, co = cset.for_cot(None)
            else:
                rows = [np.sort(np.concatenate([hits[int(x)], hits[int(y)]])) for x, y in cand]
                plab = np.concatenate([np.full(len(x), k) for k, x in enumerate(rows)])
                rows = np.concatenate(rows)
                f = MF.fit_tracks(r[rows], phi[rows], z[rows], vol[rows], plab, min_hits=3)
                cr, cz, co = cset.for_cot(f["cot"])
            with Probe(line_sz_done=False) as pr:
                ref = MF._merge(cand, trk, lab.copy(), r, phi, z, vol, cr, cz, co, one_per_end=False)
            t_ref = time.time() - t0
            row = {"event": a.event, "config": list(cfg), "cut_set": cs, "candidates": len(cand),
                   "identical": bool(np.array_equal(fast, ref)), "fast_s": round(t_fast, 1), "definition_s": round(t_ref, 1),
                   "merged": int(len(np.unique(lab[ok])) - len(np.unique(ref[ref >= 0]))),
                   "lstsq_calls": pm.lstsq_calls + pr.lstsq_calls + pm.solve_errors + pr.solve_errors}
            print(json.dumps(row, default=float), flush=True)
            rows_out.append(row)
    _write(a.out, f"anycheck_{a.event}.json", {"rows": rows_out, "identical_all": all(r_["identical"] for r_ in rows_out),
                                               "slurm_job": os.environ.get("SLURM_JOB_ID")})


# ------------------------------------------------------------------------------------------------ profile
def _profile_event(args):
    """Per configuration (builder path): candidates, pairs kept by the filter, and on a random sample of the kept pairs
    (one unpinned batch: statistics only) which of _merge's cuts they pass; the fit cost per pair."""
    scores, event, cfgs, sample, cuts = args
    from hnl.trackml.trackbuild import merge_fragments_fast
    from hnl.vertex.pairing import fit_tracks
    max_zrms, max_out, min_span = cuts
    ev, st = _event_state(scores, event)
    r, phi, z = st.r, st.phi, st.z
    vol = MF._volume_from_radius(r, z)
    rng = np.random.default_rng(event)
    rows = []
    for i, cfg in enumerate(cfgs):
        t0 = time.time()
        lab, max_rms = first_pass(cfg, st)
        t1 = time.time()
        lab = merge_fragments_fast(r, phi, z, lab, max_rms=max_rms, mode="any")
        t2 = time.time()
        ok = lab >= 0
        trk, cnt = np.unique(lab[ok], return_counts=True)
        trk = trk[cnt >= 2]
        tab = SM.Table(r, phi, z, lab, trk, vol)
        ia, ib = SM.sequential_candidates(tab.R_in, tab.R_out, tab.phi_in, tab.phi_out, 0.3, 400.0)
        keep, lb, tau, span = SM.prefilter(tab, ia, ib, max_rms, max_out, min_span, max_zrms)
        t3 = time.time()
        k = np.flatnonzero(keep)
        pick = np.sort(rng.choice(k, min(sample, len(k)), replace=False)) if len(k) else k
        n = tab.fn[ia] + tab.fn[ib]
        row = {"event": event, "config": i, "family": cfg[0], "params": [float(x) for x in cfg[1:]],
               "first_pass_s": round(t1 - t0, 2), "any_s": round(t2 - t1, 2), "fragments": int(len(trk)),
               "candidates": int(len(ia)), "kept": int(len(k)), "filter_s": round(t3 - t2, 2),
               "H_C": int(n.max()) if len(n) else 0, "union_mean_kept": float(n[k].mean()) if len(k) else 0.0}
        if len(pick):
            rr, pl = SM.union_rows(tab, ia[pick], ib[pick])
            t4 = time.time()
            f = fit_tracks(r[rr], phi[rr], z[rr], vol[rr], pl, min_hits=3)
            t5 = time.time()
            o_rms, o_z, o_out = f["rms"] <= max_rms, f["zrms"] <= max_zrms, f["n_out"] <= max_out
            acc = o_rms & o_z & o_out
            frac = float(acc.mean())
            row.update({"sample": int(len(pick)), "sample_H": int(n[pick].max()),
                        "fit_ms_per_pair": round(1e3 * (t5 - t4) / len(pick), 3),
                        "accepted_frac_of_kept": round(frac, 4), "accepted_est": int(round(frac * len(k))),
                        "fail_rms_frac": round(float((~o_rms).mean()), 4),
                        "fail_zrms_frac": round(float((~o_z).mean()), 4),
                        "fail_nout_frac": round(float((~o_out).mean()), 4),
                        "fail_only_zrms_frac": round(float((~o_z & o_rms & o_out).mean()), 4),
                        "union_mean_accepted": float(n[pick][acc].mean()) if acc.any() else 0.0,
                        "accepted_by_union_size": {int(u): int(c) for u, c in
                                                   zip(*np.unique(n[pick][acc], return_counts=True))}})
        print(json.dumps(row, default=float), flush=True)
        rows.append(row)
    return rows


def cmd_profile(a):
    import concurrent.futures as cf
    import pandas as pd
    from hnl.eval.trackbuild_val import configs
    GRANT.check(a.events, "eval", out=a.out)         # overlay ids only under a grant (addendum H)
    cfgs = configs(a.builders.split(","))
    events = [int(e) for e in a.events.split(",")]
    os.makedirs(a.out, exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=min(a.workers, len(events))) as ex:
        res = list(ex.map(_profile_event, [(a.scores, e, cfgs, a.sample, (5.0, 1, 50.0)) for e in events]))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.drop(columns=["accepted_by_union_size"], errors="ignore").to_csv(os.path.join(a.out, "profile.csv"), index=False)
    g = df.groupby("family")
    _write(a.out, "profile_summary.json", {
        "events": events, "configurations": len(cfgs),
        "per_family": {fam: {"candidates_median": float(x.candidates.median()), "kept_median": float(x.kept.median()),
                             "accepted_est_median": float(x.accepted_est.median()),
                             "accepted_est_max": float(x.accepted_est.max()),
                             "fail_only_zrms_frac_median": float(x.fail_only_zrms_frac.median()),
                             "fit_ms_per_pair_median": float(x.fit_ms_per_pair.median()),
                             "any_s_median": float(x.any_s.median())} for fam, x in g},
        "slurm_job": os.environ.get("SLURM_JOB_ID")})


# ------------------------------------------------------------------------------------------------ HNL
def cmd_hnl(a):
    import glob
    import hnl.eval.trackbuild_val as TV
    from hnl.trackbuild.mutex_ws import hit_layers
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))[:a.max_events]
    cfgs = TV.configs(a.builders.split(","))
    orig = MF.merge_fragments
    sig = inspect.signature(orig)
    recs = []
    any_recs = []
    cur = {}

    def wrapped(*args, **kw):
        ba = sig.bind(*args, **kw)
        ba.apply_defaults()
        p = ba.arguments
        seq = p["mode"] != "any"
        with Probe(line_sz_done=seq) as pr:
            t0 = time.time()
            ref = orig(*args, **kw)
            t_ref = time.time() - t0
        if not seq:                                     # the any-order pass: hnl.trackml's exact version too
            from hnl.trackml.trackbuild import merge_fragments_fast
            t0 = time.time()
            mine_any = merge_fragments_fast(p["r"], p["phi"], p["z"], p["labels"], p["volume"], p["min_frag"],
                                            p["max_dphi"], p["max_gap_r"], p["max_rms"], p["max_zrms"], p["max_out"],
                                            mode="any", max_dphi_dir=p["max_dphi_dir"], max_dcot=p["max_dcot"],
                                            min_span_mm=p["min_span_mm"])
            any_recs.append({**cur, "identical": bool(np.array_equal(ref, mine_any)), "original_s": round(t_ref, 3),
                             "mine_s": round(time.time() - t0, 3)})
            return ref
        info = {}
        with Probe() as pm:
            t0 = time.time()
            mine, batch = SM.merge_sequential(p["r"], p["phi"], p["z"], p["labels"], p["volume"], p["min_frag"],
                                              p["max_dphi"], p["max_gap_r"], p["max_rms"], p["max_zrms"],
                                              p["max_out"], min_span_mm=p["min_span_mm"], info=info,
                                              return_batch=True, check_ties=True)
            t_mine = time.time() - t0
        lab = np.asarray(p["labels"])
        rec = {**cur, "call": len(recs), "identical": bool(np.array_equal(ref, mine)), "original_s": round(t_ref, 3),
               "mine_s": round(t_mine, 3), "merged": int(len(np.unique(lab[lab >= 0])) - len(np.unique(ref[ref >= 0]))),
               "lstsq_original": pr.lstsq_calls + pr.solve_errors, "lstsq_mine": pm.lstsq_calls + pm.solve_errors,
               "K_original": pr.line_sz[0]["K"] if pr.line_sz else None,
               "K_mine": pm.line_sz[-1]["K"] if pm.line_sz else None,
               "H_original": pr.line_sz[0]["H"] if pr.line_sz else None,
               "H_mine": pm.line_sz[-1]["H"] if pm.line_sz else None, "fit_mismatches": None,
               **{k: v for k, v in info.items() if not k.endswith("_s") and k not in ("W", "class_widths")}}
        if pr.fits and pm.fits and pr.cands:
            C, B = _cands_array(pr.cands[0]), _cands_array(batch)
            base = int(max(C.max(), B.max())) + 1
            pos = np.minimum(np.searchsorted(_code(C, base), _code(B, base)), len(C) - 1)
            fr, fm = pr.fits[0], pm.fits[0]
            if not np.array_equal(_code(C, base)[pos], _code(B, base)):
                rec["fit_mismatches"] = -1                     # batch not a subset of C (would be a bug)
            else:
                mm = _mismatches(fm, fr, pos)
                sz_at = None
                if info.get("K_proof") == "certificate" and rec["K_original"]:
                    lab_in = np.asarray(p["labels"])
                    rows = [np.flatnonzero((lab_in == a) | (lab_in == b)) for a, b in batch]
                    plab = np.concatenate([np.full(len(x), k) for k, x in enumerate(rows)])
                    rows = np.concatenate(rows)
                    vol = p["volume"] if p["volume"] is not None else MF._volume_from_radius(
                        np.asarray(p["r"], dtype=np.float64), np.asarray(p["z"], dtype=np.float64))
                    rr, pp, zz = (np.asarray(p[k], dtype=np.float64) for k in ("r", "phi", "z"))
                    d = SM._line_sz_done(rr[rows], pp[rows], zz[rows], np.asarray(vol)[rows], plab)
                    zk = d["zrms_hist"][:, rec["K_original"] - 1]
                    sz_at = int((~((zk == fr["zrms"][pos]) | (np.isnan(zk) & np.isnan(fr["zrms"][pos])))).sum())
                rec.update({"fit_mismatches": int(sum(mm.values())),
                            "transverse_mismatches": int(sum(mm[k] for k in TRANSVERSE)),
                            "sz_mismatches": int(sum(mm[k] for k in SZ)), "sz_at_K_original_mismatches": sz_at,
                            "pinned_ok": _pinned_ok(mm, info.get("K_proof"), rec["K_original"], rec["K_mine"],
                                                    sz_at)})
        if a.c_check:                                   # the C cut set: filtered pass against the unfiltered definition
            cuts_c = _cuts_of("C")
            args_c = (p["r"], p["phi"], p["z"], p["labels"], p["volume"], p["min_frag"], p["max_dphi"], p["max_gap_r"])
            t0 = time.time()
            ref_c = SM.merge_sequential_reference(*args_c, min_span_mm=p["min_span_mm"], cuts=cuts_c)
            t_ref_c = time.time() - t0
            info_c = {}
            t0 = time.time()
            mine_c = SM.merge_sequential(*args_c, min_span_mm=p["min_span_mm"], cuts=cuts_c, info=info_c)
            rec.update({"identical_C": bool(np.array_equal(ref_c, mine_c)), "consistent_C": info_c.get("consistent", True),
                        "merged_C": int(len(np.unique(lab[lab >= 0])) - len(np.unique(ref_c[ref_c >= 0]))),
                        "original_C_s": round(t_ref_c, 3), "mine_C_s": round(time.time() - t0, 3),
                        "kept_C": info_c.get("kept"), "finite_q_ties_C": info_c.get("finite_q_ties")})
        recs.append(rec)
        return ref
    MF.merge_fragments = wrapped
    try:
        for path in paths:
            eid = TV._eid_of(path)
            layers = hit_layers([int(eid)])
            cur.clear()
            cur.update({"event": eid})
            t0 = time.time()
            TV.evaluate_event(path, cfgs, {}, None, layers, None)
            print(json.dumps({"event": eid, "configs": len(cfgs), "s": round(time.time() - t0, 1),
                              "identical": all(r["identical"] for r in recs if r["event"] == eid)}), flush=True)
    finally:
        MF.merge_fragments = orig
    import pandas as pd
    df = pd.DataFrame(recs)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "hnl_identity.csv"), index=False)
    _write(a.out, "hnl_identity.json", {
        "events": [TV._eid_of(p) for p in paths], "graphs": a.graphs, "sequential_calls": int(len(df)),
        "identical_all": bool(df.identical.all()), "merged_total": int(df.merged.sum()),
        "candidates_total": int(df.get("candidates", pd.Series([0])).sum()),
        "pairs_to_merge_total": int(df.get("pairs_to_merge", pd.Series([0])).sum()),
        "K_proofs": df.get("K_proof", pd.Series(dtype=str)).value_counts().to_dict(),
        "lstsq_or_solve_errors": int(df.lstsq_original.sum() + df.lstsq_mine.sum()),
        "K_equal_where_pinned": bool(((df.K_original == df.K_mine) | df.fit_mismatches.isna()
                                      | (df.get("K_proof", pd.Series(dtype=str)) == "certificate")).all()),
        "H_equal_all": bool(((df.H_original == df.H_mine) | df.fit_mismatches.isna()).all()),
        "transverse_fit_mismatches_total": int(df.get("transverse_mismatches", pd.Series([0])).fillna(0).sum()),
        "pinned_ok_all": bool(df.get("pinned_ok", pd.Series([True])).fillna(True).all()),
        "calls_with_refit": int(df.fit_mismatches.notna().sum()),
        "finite_q_total": int(df.get("finite_q", pd.Series([0])).fillna(0).sum()),
        "finite_q_ties_total": int(df.get("finite_q_ties", pd.Series([0])).fillna(0).sum()),
        "mutual_pairs_kept": int(df.get("mutual_pairs_kept", pd.Series([0])).fillna(0).sum()),
        "identical_C_all": bool(df.identical_C.all()) if "identical_C" in df else None,
        "any_pass_calls": len(any_recs), "any_pass_identical_all": all(x["identical"] for x in any_recs),
        "consistent_C_all": bool(df.consistent_C.all()) if "consistent_C" in df else None,
        "merged_C_total": int(df.merged_C.sum()) if "merged_C" in df else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID")})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("env")
    e.add_argument("--out", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--scores", required=True)
    r.add_argument("--event", type=int, default=1000)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=None, help="e.g. mwslayermerge,0.2 or walkmerge,0.1,0.5,1.0")
    r.add_argument("--s0", type=float, default=0.2)
    r.add_argument("--tag", default="direct")
    r.add_argument("--max-zrms", type=float, default=5.0)
    r.add_argument("--max-out", type=float, default=1)
    r.add_argument("--min-span-mm", type=float, default=50.0)
    r.add_argument("--max-dphi", type=float, default=0.3)
    r.add_argument("--max-gap-r", type=float, default=400.0)
    r.add_argument("--unpinned", action="store_true")
    r.add_argument("--cuts", default="H", choices=["H", "C"])
    r.add_argument("--method", default="prefit", choices=["prefit", "pinned"])
    r.add_argument("--skip-original", action="store_true", help="only the filtered pass and the bound statistics")
    t = sub.add_parser("timing")
    t.add_argument("--scores", required=True)
    t.add_argument("--events", default="1000,1001,1002,1003,1004")
    t.add_argument("--out", required=True)
    t.add_argument("--builders", default="walkmergeX,mwslayermergeX,mws2passmergeX")
    t.add_argument("--workers", type=int, default=5)
    t.add_argument("--n-proc", type=int, default=1, help="processes for the pre-fit of each configuration (G3: 16 CPUs)")
    t.add_argument("--check-build", action="store_true")
    t.add_argument("--cut-sets", default="H,C")
    t.add_argument("--configs", default=None, help="explicit configurations, e.g. 'mwslayermerge,0.2;walkmerge,0.1,0.5,1'")
    op = sub.add_parser("options")
    op.add_argument("--scores", required=True)
    op.add_argument("--event", type=int, default=1001)
    op.add_argument("--out", required=True)
    op.add_argument("--configs", default="mwslayermerge,0.2")
    op.add_argument("--windows", default="0.1,400;0.05,400;0.1,150")
    op.add_argument("--chunks", default="16")
    op.add_argument("--cut-sets", default="H,C")
    ac = sub.add_parser("anycheck")
    ac.add_argument("--scores", required=True)
    ac.add_argument("--event", type=int, default=1001)
    ac.add_argument("--out", required=True)
    ac.add_argument("--configs", default="walkmerge,0.1,0.5,1;mwslayermerge,0.2;mws2passmerge,0.3,0.02")
    ac.add_argument("--cut-sets", default="H,C")
    ac.add_argument("--n-proc", type=int, default=12)
    pf = sub.add_parser("profile")
    pf.add_argument("--scores", required=True)
    pf.add_argument("--events", default="1000,1001,1002,1003,1004")
    pf.add_argument("--out", required=True)
    pf.add_argument("--builders", default="walkmergeX,mwslayermergeX,mws2passmergeX")
    pf.add_argument("--workers", type=int, default=5)
    pf.add_argument("--sample", type=int, default=20000)
    h = sub.add_parser("hnl")
    h.add_argument("--graphs", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--max-events", type=int, default=5)
    h.add_argument("--builders", default="walkmergeX,mwslayermergeX,mws2passmergeX")
    h.add_argument("--no-c-check", dest="c_check", action="store_false", help="skip the C cut-set identity check")
    a = ap.parse_args(argv)
    {"env": cmd_env, "reference": cmd_reference, "timing": cmd_timing, "hnl": cmd_hnl, "profile": cmd_profile,
     "options": cmd_options, "anycheck": cmd_anycheck}[a.cmd](a)


if __name__ == "__main__":
    main()
