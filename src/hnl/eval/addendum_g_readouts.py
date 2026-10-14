"""Read-outs of RESULTS T1 deviation 15 (fixed before T2, reported without a claim; none changes a claim), the G2 / G3
code path of deviation 14 and the dense bins chosen from truth. Definitions: docs/addendum_g_t2_readouts.md (committed
before any test read); statistics as hnl.eval.addendum_g_summary (3-seed means, sigma^2 = sigma_boot^2 + s_seeds^2 / n,
paired differences on the set's fixed bootstrap tables). G1's models: A = Canto-L (ttbar), B = acorn (HNL) + LM;
"Delta" is A - B in points.

1. Sign of the G1 difference per bin of addendum G section 7 (bins of deviation 10: |d0|, r_v, first-hit radius,
   |d0| / r_first, pT) on P_G of the generic_iso split. Above |d0| = 300 mm only the absolute efficiencies are quoted
   and the models are not ranked (deviation 15); the same holds for r_v >= 650 mm and first-hit radius >= 700 mm
   (results/new_samples_sufficiency_v1.md 6.5).
2. P_G reweighted in the exact r_v: (a) to the Zenodo HNL primary population's fitted-r_v shape
   (results/addendum_g_zenodo_rv_shape_v1.csv, 20 mm bins over [0, 800) mm; weight of a target in bin b =
   f_Zenodo(b) / n_PG(b), the Zenodo fraction not covered by P_G is reported); (b), (c) to exponentials with
   lambda = 100 and 300 mm: weight exp(-r_v / lambda) on the gun's uniform r_v (the gun draws r_v ~ U(1, 800) mm, so
   the weighted sample has an exponential decay-radius distribution truncated to the gun's range, with the detector
   acceptance and the P_G cut acting as before). Weighted efficiency = sum w k / sum w, with the Kish effective count
   n_eff = (sum w)^2 / sum w^2; weights are fixed by truth and the bootstrap resamples events with them.
3. Tables of P_G by species (|PDG| 11, 13, 211, 321, 2212), pT (deviation 10 bins) and |eta| (table eta; bins 0, 0.5,
   1.0, 1.5, 2.0, 2.5, 3.0, 4.0): every model's efficiency and Delta.
4. A second read-out of every model with the Zenodo same-module label exchange emulated on the generated labels,
   from the stored tracks (hnl.eval.addendum_g evaluate, blocks/*/tracks.parquet). Emulation (fixed before any test
   read): in every event, the hits are grouped by module (geometry id); a module whose k >= 2 hits carry at least two
   different particle labels is exchanged with probability p(k), the fraction of resolved k-hit modules whose labels
   do not match the positions in the Zenodo ttbar val events (results/label_swap_verify_v1.md section 2):
   p(2) = 1961/17448 = 0.1124, p(3) = 872/3542 = 0.2462, p(4) = 202/777 = 0.2600, p(>= 5) = (43+7+2)/(154+30+4) =
   0.2766; an exchanged module's labels are permuted among its hits by a uniformly drawn permutation that changes at
   least one hit's label (for two hits: the swap). The measured mechanism (labels in particle-id order of the hit ids,
   positions in another order, results/label_swap_verify_v1.md "Where it comes from") does not depend on the
   separation, pT or generation, so neither does the emulation. Random numbers: numpy default_rng([20261014, sample
   code, event number]); one uniform per eligible module in increasing geometry-id order, then the permutations of
   the exchanged modules in that order (hits of a module in row order); the emulated labels of an event are the same
   for every model and seed. The tracks are not changed; double-majority matching (docs/definitions.md section 5) is
   repeated against the emulated labels, so each particle keeps its hit count (targets and populations are
   unchanged) and only the matches and the fake count change. Before emulating, the stored tracks matched against the
   true labels must reproduce the block's stored matches and fake counts exactly.
5. P_G under the hit-fit |d0| and the fitted r_v of docs/definitions.md (hnl.eval.addendum_g_truth fits): P_G(fit) =
   gun targets with fitted |d0| >= 10 mm or fitted r_v >= 100 mm, beside P_G (exact); overlap counts, efficiencies,
   Delta, and bins in the fitted |d0| and fitted r_v (with "no vertex").
6. Dense component bins chosen from truth, beside the preregistered ones (hnl.eval.addendum_g_summary): number of the
   LLP's displaced-source targets (same-vertex multiplicity 1 / 2 / 3 / 4 / >= 5); other targets within Delta R < 0.2
   of the initial direction 0 / 1 / 2 / 3-4 / >= 5; merged tails r_v >= 450 mm, first-hit radius >= 500 mm and
   Delta R < 0.05 multiplicity >= 2; on P_dense, every model, and Canto-L (ttbar) minus acorn (HNL) + LM.
7. G2 and G3 (step T4; deviation 14): paired_claim(D, A, B, rule, exclude_events) evaluates a claim of addendum G
   section 6 on the generic_iso split, as written and without iso event 1592 (G23_EXCLUDE); the retrained models'
   keys are passed in at T4. With G1's models and no exclusion it reproduces hnl.eval.addendum_g_summary.g1 (checked
   in every read-out).

    python -m hnl.eval.addendum_g_readouts --set val|test [--eval-root DIR] [--out-dir DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.addendum_g import LABELS, OUT, REPO, T2, _git, model_seeds
from hnl.eval.addendum_g_summary import (D0_BINS, FR_GUARD, MARGIN, PT_BINS, RATIO_BINS, RF_BINS, RV_BINS, Data, _pc,
                                         g1, populations)
from hnl.eval.bootstrap import linear_combination, seed_mean

DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
G1_A, G1_B = "canto_l_ttbar", "acorn_hnl_lm"
LX_SEED = 20261014
LX_COUNTS = {2: (1961, 17448), 3: (872, 3542), 4: (202, 777), 5: (43 + 7 + 2, 154 + 30 + 4)}   # 5 means >= 5
ETA_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
SPECIES = {"e": 11, "mu": 13, "pi": 211, "K": 321, "p": 2212}
LAMBDAS = (100.0, 300.0)
NOT_RANKED = {"|d0| [mm]": 300.0, "r_v [mm]": 650.0, "first-hit radius [mm]": 700.0}
MULT_BINS = [1, 2, 3, 4, 5, np.inf]
DR02_BINS = [0, 1, 2, 3, 5, np.inf]
G23_EXCLUDE = (1592,)


def lx_rate(k: int) -> float:
    a, b = LX_COUNTS[min(int(k), 5)]
    return a / b


# ------------------------------------------------------------------------------------------------- emulation
def emulate_exchange(pid: np.ndarray, geoid: np.ndarray, code: int, event: int) -> np.ndarray:
    """The generated labels of one event (rows in hits-table order) with the Zenodo same-module label exchange
    emulated (module docstring, item 4)."""
    pid = np.asarray(pid)
    geoid = np.asarray(geoid)
    n = len(pid)
    out = pid.copy()
    if n == 0:
        return out
    rng = np.random.default_rng([LX_SEED, int(code), int(event)])
    order = np.lexsort((np.arange(n), geoid))
    g = geoid[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, n])
    elig = []
    for j in np.flatnonzero(sizes >= 2):
        rows = order[starts[j]:starts[j] + sizes[j]]
        if len(np.unique(pid[rows])) >= 2:
            elig.append(rows)
    if not elig:
        return out
    u = rng.random(len(elig))
    for rows, uu in zip(elig, u):
        k = len(rows)
        if uu >= lx_rate(k):
            continue
        lab = pid[rows]
        while True:
            perm = rng.permutation(k)
            if (lab[perm] != lab).any():
                break
        out[rows] = lab[perm]
    return out


def _relabel_chunk(args):
    """Label-exchange read-out of some events of one evaluation block (module docstring, item 4)."""
    blk, eids = args
    from hnl.data.acorn_store import CODE_SAMPLE
    from hnl.data.convert import read_table
    from hnl.eval.matching import FastMatch
    tr = pd.read_parquet(os.path.join(blk, "tracks.parquet"), filters=[("event_id", "in", [int(e) for e in eids])])
    pe = pd.read_parquet(os.path.join(blk, "per_event.parquet"))
    pe = pe[pe.event_id.isin(eids)].set_index("event_id")
    pp = pd.read_parquet(os.path.join(blk, "per_particle.parquet"), columns=["event_id", "particle_id", "dm_matched"])
    pp = pp[pp.event_id.isin(eids)]
    eids = np.asarray(eids, dtype=np.int64)
    hits = {}
    for c in np.unique(eids // 100_000_000):
        ev = (eids[eids // 100_000_000 == c] % 100_000_000).tolist()
        h = read_table(os.path.join(DATA, "derived", "events", "v1"), "hits", CODE_SAMPLE[int(c)], events=ev,
                       columns=["event", "row", "geoid", "pid"])
        for e, gr in h.groupby("event"):
            hits[int(c) * 100_000_000 + int(e)] = gr.sort_values("row")
    trg = {int(e): g for e, g in tr.groupby("event_id")}
    ppg = {int(e): g for e, g in pp.groupby("event_id")}
    prow, erow, bad = [], [], 0
    n_exch = n_hits = 0
    for e in eids:
        h = hits[int(e)]
        n = len(h)
        if not np.array_equal(h.row.to_numpy(), np.arange(n)):
            raise RuntimeError(f"event {e}: hits table rows are not 0..N-1")
        pid, geo = h.pid.to_numpy(), h.geoid.to_numpy()
        track = np.full(n, -1, dtype=np.int64)
        if int(e) in trg:
            t = trg[int(e)]
            track[t.row.to_numpy()] = t.track.to_numpy()
        fm = FastMatch(pid)
        m = fm.match(track)
        kp = ppg.get(int(e))
        ids = np.zeros(0, np.int64) if kp is None else kp.particle_id.to_numpy(np.int64)
        stored = np.zeros(0, bool) if kp is None else kp.dm_matched.to_numpy(bool)
        idx = np.searchsorted(fm.particles, ids)
        if len(ids) and not np.array_equal(fm.particles[np.minimum(idx, fm.P - 1)], ids):
            raise RuntimeError(f"event {e}: stored particles missing from the hits table")
        bad += int((m["dm_matched"][idx] != stored).sum())
        bad += int(m["n_fake"] != pe.loc[e, "n_fake"]) + int(m["n_tracks_ge5"] != pe.loc[e, "n_tracks_ge5"])
        lx = emulate_exchange(pid, geo, int(e) // 100_000_000, int(e) % 100_000_000)
        n_exch += int((lx != pid).sum())
        n_hits += n
        fx = FastMatch(lx)
        if not np.array_equal(fx.particles, fm.particles) or not np.array_equal(fx.n_particle, fm.n_particle):
            raise RuntimeError(f"event {e}: the emulation changed the particles' hit counts")
        mx = fx.match(track)
        prow.append(pd.DataFrame({"event_id": np.int64(e), "particle_id": ids, "dm_lx": mx["dm_matched"][idx]}))
        erow.append({"event_id": int(e), "n_tracks_ge5_lx": mx["n_tracks_ge5"], "n_fake_lx": mx["n_fake"]})
    return pd.concat(prow, ignore_index=True), pd.DataFrame(erow), {"mismatches": bad, "hits": n_hits,
                                                                    "hits_exchanged": n_exch, "events": len(eids)}


def relabel(model: str, seed: int, set_: str, eval_root: str, workers: int) -> dict:
    """per_particle_lx.parquet, per_event_lx.parquet and lx.json of one (model, seed, set) from its stored tracks."""
    d = os.path.join(eval_root, f"{model}_s{seed}", set_)
    blocks = sorted(glob.glob(os.path.join(d, "blocks", "*_of_*")))
    if not blocks or not all(os.path.exists(os.path.join(b, "tracks.parquet")) for b in blocks):
        raise SystemExit(f"{d}: no stored tracks")
    tasks = []
    for b in blocks:
        ev = pd.read_parquet(os.path.join(b, "per_event.parquet"), columns=["event_id"]).event_id.to_numpy()
        ev = np.sort(ev)
        tasks += [(b, ev[k:k + 100].tolist()) for k in range(0, len(ev), 100)]
    with cf.ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        res = list(ex.map(_relabel_chunk, tasks))
    pp = pd.concat([r[0] for r in res], ignore_index=True)
    pe = pd.concat([r[1] for r in res], ignore_index=True)
    st = {k: int(sum(r[2][k] for r in res)) for k in ("mismatches", "hits", "hits_exchanged", "events")}
    if st["mismatches"]:
        raise SystemExit(f"{d}: the stored tracks do not reproduce the stored matching ({st['mismatches']} differences)")
    pp.to_parquet(os.path.join(d, "per_particle_lx.parquet"), index=False)
    pe.to_parquet(os.path.join(d, "per_event_lx.parquet"), index=False)
    st.update(model=model, seed=seed, set=set_, git=_git(), slurm_job=os.environ.get("SLURM_JOB_ID"),
              finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    with open(os.path.join(d, "lx.json"), "w") as f:
        json.dump(st, f, indent=1)
    return st


# ------------------------------------------------------------------------------------------------- data
class RData(Data):
    """Data plus the label-exchange results (dm_lx, n_fake_lx), the hit-fit quantities of generic_iso and the dense
    truth bins."""

    def __init__(self, set_: str, eval_root: str | None = None):
        super().__init__(set_, eval_root)
        self.lx = {}
        for (m, seed), (pp, pe, prov) in list(self.res.items()):
            d = os.path.join(self.eval_root, f"{m}_s{seed}", set_)
            if os.path.exists(os.path.join(d, "per_particle_lx.parquet")):
                lp = pd.read_parquet(os.path.join(d, "per_particle_lx.parquet"))
                le = pd.read_parquet(os.path.join(d, "per_event_lx.parquet"))
                pp = pp.merge(lp, on=["event_id", "particle_id"], how="left", validate="one_to_one")
                pe = pe.merge(le, on="event_id", how="left", validate="one_to_one")
                if pp.dm_lx.isna().any() or pe.n_fake_lx.isna().any():
                    raise RuntimeError(f"{d}: label-exchange rows do not cover the results")
                pp["dm_lx"] = pp.dm_lx.astype(bool)
                self.res[(m, seed)] = (pp, pe, prov)
                self.lx[(m, seed)] = json.load(open(os.path.join(d, "lx.json")))
        self.split = dict(self.parts)
        if "generic_iso" in self.truth:
            t = self.truth["generic_iso"]
            path = os.path.join(T2, f"fits_generic_iso_{self.split['generic_iso']}.parquet")
            self.has_fits = os.path.exists(path)
            if self.has_fits:
                f = pd.read_parquet(path, columns=["event", "particle_id", "d0_fit", "rv_fit"])
                t = t.merge(f, on=["event", "particle_id"], how="left", validate="one_to_one")
                g = t.gun & t.is_target
                if t.loc[g, "d0_fit"].isna().any():
                    raise RuntimeError(f"{path}: gun targets without a fit")
                t["in_pop_fit"] = g & ((t.d0_fit >= 10.0) | (t.rv_fit >= 100.0))
                self.truth["generic_iso"] = t
        if "generic_dense" in self.truth:
            t = self.truth["generic_dense"]
            dst = t.gun & t.is_target
            mult = t[dst].groupby(["event", "llp"]).particle_id.transform("size")
            t["n_same_vertex"] = np.nan
            t.loc[dst, "n_same_vertex"] = mult.to_numpy()
            t["n_neigh_02"] = neighbours_dr(t, 0.2)
            self.truth["generic_dense"] = t

    def models_lx(self) -> list[str]:
        return [m for m in self.models() if all((m, s) in self.lx for s in model_seeds(m))]

    def eff_w(self, model: str, sample: str, mask: np.ndarray, w: np.ndarray, what: str = "dm"):
        """Weighted efficiency (points per seed, replicates per seed) with fixed per-particle weights w."""
        t = self.truth[sample]
        pos = t.pos.to_numpy()[mask]
        wm = np.asarray(w, dtype=float)[mask]
        n_units = len(self.units[sample])
        den = np.bincount(pos, weights=wm, minlength=n_units)
        pts, reps = [], []
        for seed in model_seeds(model):
            f = self.flags(model, seed, sample)[what].to_numpy()[mask].astype(float)
            num = np.bincount(pos, weights=wm * f, minlength=n_units)
            pts.append(num.sum() / den.sum())
            reps.append((self.w[sample] @ num) / (self.w[sample] @ den))
        return pts, reps


def neighbours_dr(t: pd.DataFrame, dr: float) -> np.ndarray:
    """Per row: other targets of the same event within Delta R < dr of the initial momentum direction (as
    hnl.eval.addendum_g_summary.neighbours, which uses 0.05)."""
    out = np.zeros(len(t), dtype=np.int64)
    p = np.hypot(t.p_px.to_numpy(), t.p_py.to_numpy())
    eta = np.arcsinh(t.p_pz.to_numpy() / np.maximum(p, 1e-12))
    phi = np.arctan2(t.p_py.to_numpy(), t.p_px.to_numpy())
    tgt = t.is_target.to_numpy(bool)
    ev = t.event.to_numpy()
    order = np.argsort(ev, kind="stable")
    bounds = np.flatnonzero(np.r_[True, ev[order][1:] != ev[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        ti = idx[tgt[idx]]
        if len(ti) < 2:
            continue
        de = eta[idx][:, None] - eta[ti][None, :]
        dp = np.remainder(phi[idx][:, None] - phi[ti][None, :] + np.pi, 2 * np.pi) - np.pi
        close = (de ** 2 + dp ** 2) < dr ** 2
        out[idx] = close.sum(1) - tgt[idx].astype(int)
    return out


# ------------------------------------------------------------------------------------------------- claims
def paired_claim(D: Data, A: str, B: str, rule: str, exclude_events=(), sample: str = "generic_iso",
                 what: str = "dm", fake_col: str = "n_fake") -> dict:
    """A claim of addendum G section 6 on P_G of the generic_iso split: rule "noninferiority" (G1, G2: holds if
    Delta - 2 sigma > -1.0 and A's fake rate is at most 0.5 points above B's; resolved below the margin if
    Delta + 2 sigma < -1.0; else inconclusive) or "superiority" (G3: holds if Delta >= 3 sigma and A's fake rate is not
    above B's). exclude_events: event numbers left out (G2 / G3 without iso event 1592, deviation 14)."""
    t = D.truth[sample]
    mask = t.in_pop.to_numpy(bool) & ~t.event.isin(list(exclude_events)).to_numpy()
    a, b = D.eff(A, sample, mask, what), D.eff(B, sample, mask, what)
    diff = linear_combination([(1.0, a[0], a[1]), (-1.0, b[0], b[1])])
    d, sg = 100 * diff["value"], 100 * diff["sigma"]
    fa, fb = D.fake(A, sample, fake_col, exclude_events), D.fake(B, sample, fake_col, exclude_events)
    dfr = 100 * (np.mean(fa[0]) - np.mean(fb[0]))
    if rule == "noninferiority":
        guard = dfr <= FR_GUARD
        if d - 2 * sg > -MARGIN and guard:
            outcome = "holds"
        elif d + 2 * sg < -MARGIN:
            outcome = "does not hold, resolved below the margin"
        else:
            outcome = "does not hold, inconclusive" + ("" if guard else " (fake-rate guard fails)")
    elif rule == "superiority":
        guard = dfr <= 0.0
        outcome = "holds" if d >= 3 * sg and guard else "does not hold"
    else:
        raise ValueError(rule)
    n = int(mask.sum())
    return {"A": LABELS.get(A, A), "B": LABELS.get(B, B), "rule": rule, "excluded_events": list(exclude_events),
            "n_targets": n, "eff_A": float(np.mean(a[0])), "eff_B": float(np.mean(b[0])), "delta_points": d,
            "sigma_points": sg, "sigma_boot_points": 100 * diff["sigma_boot"],
            "sigma_seeds_points": 100 * diff["sigma_seeds"], "z": d / sg if sg > 0 else np.nan,
            "fake_rate_A": float(np.mean(fa[0])), "fake_rate_B": float(np.mean(fb[0])), "fake_rate_diff_points": dfr,
            "fake_rate_guard_ok": bool(guard), "outcome": outcome, "underpowered": n < 1000}


def g2_g3(D: Data, canto_ttbar: str, acorn_g: str, canto_g: str) -> list[dict]:
    """Step T4: G2 (Canto-L (ttbar) vs acorn (ttbar+G), non-inferiority) and G3 (Canto-L (ttbar+G) vs acorn (ttbar+G),
    superiority), each as written and without iso event 1592 (deviation 14)."""
    out = []
    for name, A, B, rule in (("G2", canto_ttbar, acorn_g, "noninferiority"), ("G3", canto_g, acorn_g, "superiority")):
        for ex in ((), G23_EXCLUDE):
            out.append({"claim": name, **paired_claim(D, A, B, rule, ex)})
    return out


# ------------------------------------------------------------------------------------------------- read-outs
def _diff(D, sample, mask, A=G1_A, B=G1_B, what="dm", w=None):
    if w is None:
        a, b = D.eff(A, sample, mask, what)[:2], D.eff(B, sample, mask, what)[:2]
    else:
        a, b = D.eff_w(A, sample, mask, w, what), D.eff_w(B, sample, mask, w, what)
    d = linear_combination([(1.0, a[0], a[1]), (-1.0, b[0], b[1])])
    sa, sb = seed_mean(*a), seed_mean(*b)
    return {"eff_A": sa["mean"], "sigma_A": sa["sigma"], "eff_B": sb["mean"], "sigma_B": sb["sigma"],
            "delta_points": 100 * d["value"], "sigma_points": 100 * d["sigma"], "z": d["z"]}


def bin_signs(D: RData) -> list[dict]:
    s = "generic_iso"
    t = D.truth[s]
    P = t.in_pop.to_numpy(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = t.d0.to_numpy() / t.r_first.to_numpy()
    rows = []
    for var, x, edges in (("|d0| [mm]", t.d0.to_numpy(), D0_BINS), ("r_v [mm]", t.r_v.to_numpy(), RV_BINS),
                          ("first-hit radius [mm]", t.r_first.to_numpy(), RF_BINS),
                          ("|d0| / r_first", ratio, RATIO_BINS), ("pT [GeV]", t.pt.to_numpy(), PT_BINS)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = P & (x >= lo) & (x < hi)
            if not mask.sum():
                continue
            r = {"table": "bin signs", "sample": s, "population": "P_G", "variable": var, "bin": f"[{lo:g}, {hi:g})",
                 "n": int(mask.sum()), **_diff(D, s, mask)}
            ranked = not (var in NOT_RANKED and lo >= NOT_RANKED[var])
            r["ranked"] = ranked
            if ranked:
                r["sign"] = "+" if r["delta_points"] > 0 else "-" if r["delta_points"] < 0 else "0"
            else:
                r["sign"], r["delta_points"], r["sigma_points"], r["z"] = "", np.nan, np.nan, np.nan
            rows.append(r)
    return rows


def rv_weights(D: RData) -> dict:
    """target name -> (per-row weights of the generic_iso truth table, info)."""
    from hnl.eval.addendum_g_truth import RV_SHAPE_EDGES, read_shape
    t = D.truth["generic_iso"]
    P = t.in_pop.to_numpy(bool)
    rv = t.r_v.to_numpy()
    out = {}
    shape = read_shape()
    edges = RV_SHAPE_EDGES
    b = np.clip(np.searchsorted(edges, rv, side="right") - 1, 0, len(edges) - 2)
    inside = P & (rv >= edges[0]) & (rv < edges[-1])
    n_pg = np.bincount(b[inside], minlength=len(edges) - 1).astype(float)
    fz = shape.fraction.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        wb = np.where(n_pg > 0, fz / n_pg, 0.0)
    w = np.where(inside, wb[b], 0.0)
    out["Zenodo HNL r_v shape"] = (w, {"uncovered_fraction": float(fz[n_pg == 0].sum())})
    for lam in LAMBDAS:
        out[f"exponential, lambda = {lam:g} mm"] = (np.where(P, np.exp(-np.nan_to_num(rv, nan=0.0) / lam), 0.0),
                                                     {"uncovered_fraction": 0.0})
    return out


def reweighted(D: RData) -> list[dict]:
    s = "generic_iso"
    t = D.truth[s]
    P = t.in_pop.to_numpy(bool)
    rows = []
    for name, (w, info) in rv_weights(D).items():
        mask = P & (w > 0)
        wm = w[mask]
        n_eff = float(wm.sum() ** 2 / (wm ** 2).sum())
        base = {"table": "r_v reweighting", "sample": s, "population": "P_G", "target": name, "n": int(mask.sum()),
                "n_eff_kish": n_eff, **info}
        for m in D.models():
            pts, reps = D.eff_w(m, s, mask, w)
            sm = seed_mean(pts, reps)
            rows.append({**base, "model": m, "label": LABELS[m], "mean": sm["mean"], "sigma": sm["sigma"]})
        if {G1_A, G1_B} <= set(D.models()):
            rows.append({**base, "model": "G1 difference", "label": "Delta (A - B)", **_diff(D, s, mask, w=w)})
    return rows


def kinematic_tables(D: RData) -> list[dict]:
    s = "generic_iso"
    t = D.truth[s]
    P = t.in_pop.to_numpy(bool)
    specs = [("species", k, P & (np.abs(t.pdg.to_numpy()) == v)) for k, v in SPECIES.items()]
    specs += [("pT [GeV]", f"[{lo:g}, {hi:g})", P & (t.pt.to_numpy() >= lo) & (t.pt.to_numpy() < hi))
              for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:])]
    ae = np.abs(t.eta.to_numpy())
    specs += [("|eta|", f"[{lo:g}, {hi:g})", P & (ae >= lo) & (ae < hi)) for lo, hi in zip(ETA_BINS[:-1], ETA_BINS[1:])]
    rows = []
    for var, b, mask in specs:
        if not mask.sum():
            continue
        r = {"table": "kinematics", "sample": s, "population": "P_G", "variable": var, "bin": b, "n": int(mask.sum())}
        for m in D.models():
            sm = seed_mean(*D.eff(m, s, mask)[:2])
            r[f"eff_{m}"], r[f"sigma_{m}"] = sm["mean"], sm["sigma"]
        if {G1_A, G1_B} <= set(D.models()):
            d = _diff(D, s, mask)
            r["delta_points"], r["sigma_points"] = d["delta_points"], d["sigma_points"]
        rows.append(r)
    return rows


def exchange_rows(D: RData) -> list[dict]:
    rows = []
    ms = D.models_lx()
    for s, _ in D.parts:
        t = D.truth[s]
        for pop, mask in populations(s, t).items():
            for m in ms:
                c = seed_mean(*D.eff(m, s, mask)[:2])
                x = seed_mean(*D.eff(m, s, mask, "lx")[:2])
                rows.append({"table": "label exchange", "sample": s, "population": pop, "model": m, "label": LABELS[m],
                             "n": int(mask.sum()), "mean": c["mean"], "sigma": c["sigma"], "mean_lx": x["mean"],
                             "sigma_lx": x["sigma"]})
        for m in ms:
            c, x = seed_mean(*D.fake(m, s)), seed_mean(*D.fake(m, s, "n_fake_lx"))
            rows.append({"table": "label exchange", "sample": s, "population": "track fake rate", "model": m,
                         "label": LABELS[m], "n": 0, "mean": c["mean"], "sigma": c["sigma"], "mean_lx": x["mean"],
                         "sigma_lx": x["sigma"]})
    if {G1_A, G1_B} <= set(ms):
        for what, name in (("dm", "true labels"), ("lx", "emulated exchange")):
            c = paired_claim(D, G1_A, G1_B, "noninferiority", what=what,
                             fake_col="n_fake" if what == "dm" else "n_fake_lx")
            rows.append({"table": "label exchange", "sample": "generic_iso", "population": "P_G",
                         "model": f"G1 rule, {name}", "label": c["outcome"], "n": c["n_targets"],
                         "delta_points": c["delta_points"], "sigma_points": c["sigma_points"],
                         "fake_rate_diff_points": c["fake_rate_diff_points"]})
    return rows


def fit_rows(D: RData) -> list[dict]:
    s = "generic_iso"
    t = D.truth[s]
    if not getattr(D, "has_fits", False):
        return []
    P, F = t.in_pop.to_numpy(bool), t.in_pop_fit.to_numpy(bool)
    rows = [{"table": "hit-fit population", "sample": s, "population": "counts", "variable": "overlap",
             "bin": k, "n": int(v)} for k, v in (("P_G (exact) and P_G (fit)", (P & F).sum()),
                                                 ("P_G (exact) only", (P & ~F).sum()),
                                                 ("P_G (fit) only", (~P & F).sum()))]
    specs = [("P_G (exact)", "population", "P_G (exact)", P), ("P_G (fit)", "population", "P_G (fit)", F)]
    d0f, rvf = t.d0_fit.to_numpy(), t.rv_fit.to_numpy()
    for var, x, edges in (("|d0| fitted [mm]", d0f, D0_BINS), ("r_v fitted [mm]", rvf, RV_BINS)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            specs.append(("P_G (fit)", var, f"[{lo:g}, {hi:g})", F & (x >= lo) & (x < hi)))
    specs.append(("P_G (fit)", "r_v fitted [mm]", "no vertex", F & np.isnan(rvf)))
    for pop, var, b, mask in specs:
        if not mask.sum():
            continue
        r = {"table": "hit-fit population", "sample": s, "population": pop, "variable": var, "bin": b,
             "n": int(mask.sum())}
        for m in D.models():
            sm = seed_mean(*D.eff(m, s, mask)[:2])
            r[f"eff_{m}"], r[f"sigma_{m}"] = sm["mean"], sm["sigma"]
        if {G1_A, G1_B} <= set(D.models()):
            ranked = not (var == "|d0| fitted [mm]" and b.startswith("[300"))
            d = _diff(D, s, mask)
            r["delta_points"] = d["delta_points"] if ranked else np.nan
            r["sigma_points"] = d["sigma_points"] if ranked else np.nan
        rows.append(r)
    return rows


def dense_rows(D: RData) -> list[dict]:
    s = "generic_dense"
    if s not in D.truth:
        return []
    t = D.truth[s]
    P = t.in_pop.to_numpy(bool)
    specs = []
    mult = t.n_same_vertex.to_numpy()
    for lo, hi in zip(MULT_BINS[:-1], MULT_BINS[1:]):
        specs.append(("same-vertex multiplicity", f"{lo:g}" if hi == lo + 1 else f">= {lo:g}",
                      P & (mult >= lo) & (mult < hi)))
    nb = t.n_neigh_02.to_numpy()
    for lo, hi in zip(DR02_BINS[:-1], DR02_BINS[1:]):
        lab = f"{lo:g}" if hi == lo + 1 else f">= {lo:g}" if np.isinf(hi) else f"{lo:g}-{hi - 1:g}"
        specs.append(("other targets within dR < 0.2", lab, P & (nb >= lo) & (nb < hi)))
    specs += [("merged tail", "r_v >= 450 mm", P & (t.r_v.to_numpy() >= 450)),
              ("merged tail", "first-hit radius >= 500 mm", P & (t.r_first.to_numpy() >= 500)),
              ("merged tail", "dR < 0.05 multiplicity >= 2", P & (t.n_neigh.to_numpy() >= 2))]
    rows = []
    for var, b, mask in specs:
        if not mask.sum():
            continue
        r = {"table": "dense bins", "sample": s, "population": "P_dense", "variable": var, "bin": b,
             "n": int(mask.sum())}
        for m in D.models():
            sm = seed_mean(*D.eff(m, s, mask)[:2])
            r[f"eff_{m}"], r[f"sigma_{m}"] = sm["mean"], sm["sigma"]
        if {G1_A, G1_B} <= set(D.models()):
            d = _diff(D, s, mask)
            r["delta_points"], r["sigma_points"] = d["delta_points"], d["sigma_points"]
        rows.append(r)
    return rows


def consistency(D: RData) -> dict:
    """paired_claim with G1's models and no exclusion against hnl.eval.addendum_g_summary.g1 (same numbers)."""
    if not {G1_A, G1_B} <= set(D.models()):
        return {}
    a = paired_claim(D, G1_A, G1_B, "noninferiority")
    b = g1(D)
    same = all(np.isclose(a[k], b[k], rtol=0, atol=1e-12) for k in ("delta_points", "sigma_points",
                                                                     "fake_rate_diff_points"))
    if not same or a["outcome"] != b["outcome"]:
        raise RuntimeError(f"paired_claim differs from g1: {a} {b}")
    return {"paired_claim_equals_g1": True, "lx": {f"{m}_s{s}": v for (m, s), v in D.lx.items()}}


# ------------------------------------------------------------------------------------------------- output
def _f(x, d=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def markdown(D: RData, tabs: dict, chk: dict) -> str:
    ms = D.models()
    dry = "DRY RUN on placeholder outputs made from truth (no model; code test only): " if any(
        r[2].get("dry_outputs") for r in D.res.values()) else ""
    L = [f"# {dry}Addendum G: read-outs of T1 deviation 15 ({D.set}; no claim)", "",
         f"Written {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `hnl.eval.addendum_g_readouts` "
         f"(git {_git()[:7]}); definitions in the module docstring and `docs/addendum_g_t2_readouts.md`. "
         "Efficiencies in percent (3-seed means +- sigma); Delta = Canto-L (ttbar) - acorn (HNL) + LM in points. "
         "Descriptive: none of these numbers changes a claim." + (
             " Val read-out: a test of the code on the frozen models' val outputs; nothing is chosen from it."
             if D.set == "val" else ""), ""]
    b = tabs["bin signs"]
    if len(b):
        L += ["## 1. Sign of the G1 difference per bin (P_G, generic_iso)", "",
              "| variable | bin | n | A | B | Delta | sign |", "|---|---|---|---|---|---|---|"]
        for _, r in b.iterrows():
            L.append(f"| {r['variable']} | {r['bin']} | {r['n']} | {_pc(r['eff_A'])} | {_pc(r['eff_B'])} | "
                     + (f"{_f(r['delta_points'])} +- {r['sigma_points']:.2f} | {r['sign']} |" if r["ranked"]
                        else "absolute only | not ranked |"))
        rk = b[b.ranked]
        L += ["", f"Ranked bins: Delta > 0 in {int((rk.sign == '+').sum())}, < 0 in {int((rk.sign == '-').sum())}, "
              f"= 0 in {int((rk.sign == '0').sum())} (bins of one variable are disjoint; the variables overlap).", ""]
    r = tabs["r_v reweighting"]
    if len(r):
        L += ["## 2. P_G reweighted in r_v", "", "| target | n | Kish n_eff | uncovered target fraction | "
              + " | ".join(LABELS[m] for m in ms) + " | Delta |", "|---|---|---|---|" + "---|" * (len(ms) + 1)]
        for name, g in r.groupby("target", sort=False):
            cells = []
            for m in ms:
                x = g[g.model == m].iloc[0]
                cells.append(f"{_pc(x['mean'])} +- {_pc(x['sigma'])}")
            d = g[g.model == "G1 difference"]
            dd = f"{_f(d.delta_points.iloc[0])} +- {d.sigma_points.iloc[0]:.2f}" if len(d) else ""
            x0 = g.iloc[0]
            L.append(f"| {name} | {x0['n']} | {x0['n_eff_kish']:.0f} | {x0['uncovered_fraction']:.3f} | "
                     + " | ".join(cells) + f" | {dd} |")
        L.append("")

    def wide(title, tab, extra=""):
        if not len(tab):
            return []
        out = [title, "", extra, ""] if extra else [title, ""]
        out += ["| variable | bin | n | " + " | ".join(LABELS[m] for m in ms) + " | Delta |",
                "|---|---|---|" + "---|" * (len(ms) + 1)]
        for _, x in tab.iterrows():
            cells = [f"{_pc(x.get(f'eff_{m}'))}" for m in ms]
            dd = f"{_f(x.get('delta_points'))} +- {x.get('sigma_points'):.2f}" if pd.notna(x.get("delta_points")) else ""
            out.append(f"| {x['variable']} | {x['bin']} | {x['n']} | " + " | ".join(cells) + f" | {dd} |")
        return out + [""]
    L += wide("## 3. P_G by species, pT and |eta| (generic_iso)", tabs["kinematics"])
    x = tabs["label exchange"]
    if len(x):
        L += ["## 4. Second read-out with the Zenodo same-module label exchange emulated", "",
              "Efficiency (fake rate) with the generated labels -> with the emulated exchange, from the stored tracks.",
              "", "| sample | population | " + " | ".join(LABELS[m] for m in D.models_lx()) + " |",
              "|---|---|" + "---|" * len(D.models_lx())]
        xm = x[x.model.isin(D.models_lx())]
        for (s, pop), g in xm.groupby(["sample", "population"], sort=False):
            cells = []
            for m in D.models_lx():
                y = g[g.model == m].iloc[0]
                cells.append(f"{_pc(y['mean'])} -> {_pc(y['mean_lx'])}")
            L.append(f"| {s} | {pop} | " + " | ".join(cells) + " |")
        for _, y in x[x.model.str.startswith("G1 rule")].iterrows():
            L.append(f"\n{y['model']}: Delta = {_f(y['delta_points'])} +- {y['sigma_points']:.2f}, fake-rate "
                     f"difference {_f(y['fake_rate_diff_points'])} points; the G1 rule would read: {y['label']}.")
        ex = [v for v in D.lx.values()]
        if ex:
            L.append(f"\nEmulated exchange: {100 * sum(v['hits_exchanged'] for v in ex) / sum(v['hits'] for v in ex):.2f} "
                     "% of the hits carry another label; stored tracks reproduce the stored matching in every event "
                     "(0 differences).")
        L.append("")
    L += wide("## 5. P_G under the hit-fit |d0| and the fitted r_v (generic_iso)", tabs["hit-fit population"])
    L += wide("## 6. Dense component: bins chosen from truth (P_dense, generic_dense)", tabs["dense bins"],
              "Preregistered bins: `hnl.eval.addendum_g_summary` (bins csv). Delta = Canto-L (ttbar) - acorn (HNL) + LM, "
              "descriptive.")
    L += ["## 7. G2 and G3", "", "Evaluated at step T4 on the retrained models (`paired_claim`, `g2_g3`), as written "
          "and without iso event 1592 (deviation 14). Check of the code path: `paired_claim` with G1's models "
          + ("reproduces `addendum_g_summary.g1` exactly." if chk.get("paired_claim_equals_g1") else "was not run."), ""]
    return "\n".join(L)


def readouts(set_: str, out_dir: str | None = None, eval_root: str | None = None, workers: int = 1) -> None:
    if set_ not in ("val", "test"):
        raise SystemExit("the deviation-15 read-outs are defined on the generic sets (val, test)")
    root = eval_root or os.path.join(OUT, "eval")
    for m in LABELS:
        for seed in model_seeds(m):
            d = os.path.join(root, f"{m}_s{seed}", set_)
            if os.path.exists(os.path.join(d, "per_particle.parquet")) and not os.path.exists(
                    os.path.join(d, "lx.json")):
                print(f"[readouts] relabel {m} s{seed}: {relabel(m, seed, set_, root, workers)}", flush=True)
    D = RData(set_, eval_root)
    if set_ == "test" and not D.has_fits:
        raise SystemExit("no hit-fit table for the union test split (hnl.eval.addendum_g_truth fits --list union_test)")
    tabs = {"bin signs": pd.DataFrame(bin_signs(D)), "r_v reweighting": pd.DataFrame(reweighted(D)),
            "kinematics": pd.DataFrame(kinematic_tables(D)), "label exchange": pd.DataFrame(exchange_rows(D)),
            "hit-fit population": pd.DataFrame(fit_rows(D)), "dense bins": pd.DataFrame(dense_rows(D))}
    chk = consistency(D)
    res = out_dir or os.path.join(REPO, "results")
    os.makedirs(res, exist_ok=True)
    stem = f"addendum_g_readouts_{set_}_v1"
    pd.concat([v for v in tabs.values() if len(v)], ignore_index=True).to_csv(os.path.join(res, f"{stem}.csv"),
                                                                              index=False)
    with open(os.path.join(res, f"{stem}.json"), "w") as f:
        json.dump({"checks": chk, "git": _git(), "eval_root": D.eval_root}, f, indent=1, default=float)
    md = markdown(D, tabs, chk)
    with open(os.path.join(res, f"{stem}.md"), "w") as f:
        f.write(md)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["val", "test"])
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    readouts(a.set, a.out_dir, a.eval_root, a.workers)


if __name__ == "__main__":
    main()
