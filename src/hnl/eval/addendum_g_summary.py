"""Read-out of preregistration addendum G on the generated samples (hnl.eval.addendum_g evaluate / merge outputs).

Statistics (addendum G section 6, docs/preregistration.md section 5): a model's number is the mean over its seeds;
uncertainty sigma^2 = sigma^2_boot + s_seeds^2 / n (hnl.eval.bootstrap.seed_mean); a difference uses the paired
bootstrap of the difference of seed means plus both seed terms (bootstrap.linear_combination). Event bootstrap,
B = 500, one fixed table of resampled events per (sample, split), shared by every model and seed:
derived/bootstrap/generated_v1/<sample>_<split>.npz; generic_iso test (the claims) with numpy default_rng(12345), the
other tables with default_rng(12345 + k), k fixed in BOOT_SEED. Populations from the T1 truth tables
(hnl.eval.addendum_g_t1). Efficiencies are double-majority efficiencies; the track fake rate is
sum(n_fake) / sum(n_tracks_ge5) over the events (docs/definitions.md section 5).

Claim G1 (test only): Delta = eps(Canto-L (ttbar)) - eps(acorn (HNL) + LM) on P_G of the generic_iso test split, in
points; holds if Delta - 2 sigma > -1.0 and Canto-L's fake rate on the generic_iso test split is at most 0.5 points
above acorn's (point estimates); does not hold, resolved below the margin, if Delta + 2 sigma < -1.0; otherwise does
not hold, inconclusive. Underpowered if the test split holds fewer than 1,000 P_G targets.

Bins (fixed before the test read-out): |d0| and r_v as docs/preregistration.md section 2, r_v extended above 300 mm
(the gun reaches 800 mm), first-hit radius, |d0| / r_first, pT, and for the dense component the number of other
targets within Delta R < 0.05 (initial directions).

    python -m hnl.eval.addendum_g summary --set val|test|hss
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval.addendum_g import LABELS, OUT, REPO, SETS, T1, _git, _truth, model_seeds
from hnl.eval.bootstrap import linear_combination, make_weights, seed_mean

DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
BOOT_DIR = os.path.join(DATA, "derived", "bootstrap", "generated_v1")
BOOT_SEED = {("generic_iso", "test"): 12345, ("generic_dense", "test"): 12346, ("hss15_100", "all"): 12347,
             ("hss40_300", "all"): 12348, ("generic_iso", "val"): 12349, ("generic_dense", "val"): 12350}
MARGIN = 1.0
FR_GUARD = 0.5
D0_BINS = [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]
RV_BINS = [0, 10, 30, 100, 200, 300, 450, 650, np.inf]
RF_BINS = [0, 50, 100, 200, 300, 500, 700, np.inf]
RATIO_BINS = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]
PT_BINS = [1, 2, 5, 10, 20, np.inf]
NB_BINS = [0, 1, 2, 3, 5, np.inf]
MAIN = ("canto_l_ttbar", "acorn_hnl_lm", "acorn_ttbar_lm", "acorn_hnl_cc")
VAL_REF = {"canto_l_ttbar": "Canto-L (frozen: learned merging)",
           "acorn_hnl_lm": "acorn (HNL) + MWS-layer + learned merging (Canto-L's builder)",
           "acorn_ttbar_lm": "acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder)",
           "acorn_hnl_cc": "acorn (HNL) + CC, preregistered cut (claims' reference)"}


def boot_table(sample: str, split: str, units: np.ndarray) -> np.ndarray:
    path = os.path.join(BOOT_DIR, f"{sample}_{split}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if not np.array_equal(z["units"], units) or z["w"].shape[0] != BS.B:
            raise RuntimeError(f"{path} does not match the split")
        return z["w"]
    w = make_weights(len(units), BS.B, np.random.default_rng(BOOT_SEED[(sample, split)]))
    os.makedirs(BOOT_DIR, exist_ok=True)
    np.savez_compressed(path + ".tmp.npz", units=units, w=w, seed=BOOT_SEED[(sample, split)], b=BS.B)
    os.replace(path + ".tmp.npz", path)
    return w


def neighbours(t: pd.DataFrame) -> np.ndarray:
    """Per row: other targets of the same event within Delta R < 0.05 (initial momentum directions)."""
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
        close = (de ** 2 + dp ** 2) < 0.05 ** 2
        out[idx] = close.sum(1) - tgt[idx].astype(int)
    return out


class Data:
    """Truth, splits, bootstrap tables and every model-seed's per-particle / per-event results of one set."""

    def __init__(self, set_: str):
        self.set = set_
        self.parts = SETS[set_]
        from hnl.data.splits import read_split
        from hnl.data.convert import read_table
        t = _truth([s for s, _ in self.parts])
        rf = []
        for s, _ in self.parts:
            p = read_table(os.path.join(DATA, "derived", "events", "v1"), "particles", s,
                           columns=["event", "particle_id", "r_first"])
            p["sample"] = s
            rf.append(p)
        t = t.merge(pd.concat(rf, ignore_index=True), on=["sample", "event", "particle_id"], how="left")
        self.units, self.w, self.truth = {}, {}, {}
        for s, sp in self.parts:
            u = np.sort(read_split(s, sp))
            self.units[s], self.w[s] = u, boot_table(s, sp, u)
            ts = t[(t["sample"] == s) & t.event.isin(u)].copy()
            ts["pos"] = np.searchsorted(u, ts.event.to_numpy())
            if s.startswith("generic"):
                ts["gun"] = ts.gun.fillna(False).astype(bool)
                ts["n_neigh"] = neighbours(ts)
            self.truth[s] = ts.reset_index(drop=True)
        self.res, self._flags = {}, {}
        for m in LABELS:
            for seed in model_seeds(m):
                d = os.path.join(OUT, "eval", f"{m}_s{seed}", set_)
                if os.path.exists(os.path.join(d, "per_particle.parquet")):
                    self.res[(m, seed)] = (pd.read_parquet(os.path.join(d, "per_particle.parquet")),
                                           pd.read_parquet(os.path.join(d, "per_event.parquet")),
                                           json.load(open(os.path.join(d, "provenance.json"))))

    def models(self) -> list[str]:
        return [m for m in LABELS if all((m, s) in self.res for s in model_seeds(m))]

    def flags(self, model: str, seed: int, sample: str) -> pd.DataFrame:
        key0 = (model, seed, sample)
        if key0 in self._flags:
            return self._flags[key0]
        pp = self.res[(model, seed)][0]
        from hnl.data.acorn_store import SAMPLE_CODE
        t = self.truth[sample]
        eid = np.int64(SAMPLE_CODE[sample]) * 100_000_000 + t.event.to_numpy(np.int64)
        key = pd.MultiIndex.from_arrays([eid, t.particle_id.to_numpy(np.int64)])
        x = pp.set_index(["event_id", "particle_id"])
        x = x[~x.index.duplicated()]
        got = x.reindex(key)
        self._flags[key0] = pd.DataFrame({"dm": got.dm_matched.to_numpy(), "ceil": got.dm_ceiling.to_numpy()})
        return self._flags[key0]

    def eff(self, model: str, sample: str, mask: np.ndarray, what: str = "dm"):
        """(points per seed, replicates per seed, n) of a population of one sample."""
        t = self.truth[sample]
        pos = t.pos.to_numpy()[mask]
        n_units = len(self.units[sample])
        den = np.bincount(pos, minlength=n_units).astype(float)
        pts, reps = [], []
        for seed in model_seeds(model):
            f = self.flags(model, seed, sample)[what].to_numpy()[mask]
            if np.isnan(f.astype(float)).any():
                raise RuntimeError(f"{model} s{seed} {sample}: {int(np.isnan(f.astype(float)).sum())} population "
                                   f"particles without a result row")
            num = np.bincount(pos, weights=f.astype(float), minlength=n_units)
            with np.errstate(invalid="ignore", divide="ignore"):
                pts.append(num.sum() / den.sum() if den.sum() else np.nan)
                reps.append((self.w[sample] @ num) / (self.w[sample] @ den))
        return pts, reps, int(mask.sum())

    def fake(self, model: str, sample: str):
        from hnl.data.acorn_store import SAMPLE_CODE
        u = self.units[sample]
        pts, reps = [], []
        for seed in model_seeds(model):
            pe = self.res[(model, seed)][1]
            pe = pe[pe.event_id // 100_000_000 == SAMPLE_CODE[sample]]
            ev = (pe.event_id % 100_000_000).to_numpy()
            if len(ev) != len(u) or not np.array_equal(np.sort(ev), u):
                raise RuntimeError(f"{model} s{seed} {sample}: per-event rows do not cover the split")
            pos = np.searchsorted(u, ev)
            nf = np.bincount(pos, weights=pe.n_fake.to_numpy(float), minlength=len(u))
            nt = np.bincount(pos, weights=pe.n_tracks_ge5.to_numpy(float), minlength=len(u))
            pts.append(nf.sum() / nt.sum())
            reps.append((self.w[sample] @ nf) / (self.w[sample] @ nt))
        return pts, reps


def _row(label, sample, pop, model, pts, reps, n, **kw) -> dict:
    sm = seed_mean(pts, reps)
    return {"sample": sample, "population": pop, "model": model, "label": LABELS[model], "n": n,
            **{f"eff_s{i}": p for i, p in enumerate(pts)}, "mean": sm["mean"], "sigma_boot": sm["sigma_boot"],
            "s_seeds": sm["s_seeds"], "sigma": sm["sigma"], "n_seeds": sm["n_seeds"], "table": label, **kw}


def populations(sample: str, t: pd.DataFrame) -> dict:
    if sample.startswith("generic"):
        name = "P_G" if sample == "generic_iso" else "P_dense"
        return {name: t.in_pop.to_numpy(bool), "displaced-source targets": (t.gun & t.is_target).to_numpy(bool),
                "all targets": t.is_target.to_numpy(bool), "prompt (ttbar, |d0| < 1 mm)": t.prompt.to_numpy(bool)}
    P = t.in_pop.to_numpy(bool)
    a = np.abs(t.pdg.to_numpy())
    see, v0 = t.s_ee_daughter.to_numpy(bool), t.v0_daughter.to_numpy(bool)
    return {"P_H": P, "P_H electrons": P & (a == 11), "P_H muons": P & (a == 13),
            "P_H hadrons": P & ~np.isin(a, (11, 13)), "P_H s -> ee daughters": P & see,
            "P_H V0-tagged": P & v0, "P_H other (s -> bb chain, untagged decays)": P & ~see & ~v0,
            "all targets": t.is_target.to_numpy(bool), "prompt (|d0| < 1 mm, r_prod < 1 mm)": t.prompt.to_numpy(bool)}


def main_rows(D: Data) -> list[dict]:
    rows = []
    for s, _ in D.parts:
        t = D.truth[s]
        for pop, mask in populations(s, t).items():
            for m in D.models():
                pts, reps, n = D.eff(m, s, mask)
                cp, cr, _ = D.eff(m, s, mask, "ceil")
                rows.append(_row("main", s, pop, m, pts, reps, n, ceiling=seed_mean(cp, cr)["mean"],
                                 ceiling_sigma=seed_mean(cp, cr)["sigma"]))
        for m in D.models():
            pts, reps = D.fake(m, s)
            rows.append(_row("main", s, "track fake rate", m, pts, reps, 0))
    return rows


def bin_rows(D: Data) -> list[dict]:
    rows = []
    specs = []
    for s, _ in D.parts:
        t = D.truth[s]
        P = populations(s, t)
        pname = next(iter(P))
        base = P[pname]
        bins = [("|d0| [mm]", t.d0.to_numpy(), D0_BINS), ("first-hit radius [mm]", t.r_first.to_numpy(), RF_BINS),
                ("pT [GeV]", t.pt.to_numpy(), PT_BINS)]
        if s.startswith("generic"):
            bins.insert(1, ("r_v [mm]", t.r_v.to_numpy(), RV_BINS))
            with np.errstate(invalid="ignore", divide="ignore"):
                bins.append(("|d0| / r_first", t.d0.to_numpy() / t.r_first.to_numpy(), RATIO_BINS))
            if s == "generic_dense":
                bins.append(("other targets within dR < 0.05", t.n_neigh.to_numpy().astype(float), NB_BINS))
        else:
            bins.insert(1, ("production radius [mm]", t.r_prod.to_numpy(), RV_BINS))
        for var, x, edges in bins:
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = base & (x >= lo) & (x < hi)
                if mask.sum() == 0:
                    continue
                specs.append((s, pname, var, f"[{lo:g}, {hi:g})", mask))
    for s, pname, var, b, mask in specs:
        for m in D.models():
            pts, reps, n = D.eff(m, s, mask)
            cp, cr, _ = D.eff(m, s, mask, "ceil")
            rows.append(_row("bins", s, pname, m, pts, reps, n, variable=var, bin=b,
                             ceiling=seed_mean(cp, cr)["mean"]))
    return rows


def g1(D: Data) -> dict:
    s = "generic_iso"
    t = D.truth[s]
    mask = t.in_pop.to_numpy(bool)
    A = D.eff("canto_l_ttbar", s, mask)
    Bm = D.eff("acorn_hnl_lm", s, mask)
    diff = linear_combination([(1.0, A[0], A[1]), (-1.0, Bm[0], Bm[1])])
    d, sg = 100 * diff["value"], 100 * diff["sigma"]
    fa, fb = D.fake("canto_l_ttbar", s), D.fake("acorn_hnl_lm", s)
    dfr = 100 * (np.mean(fa[0]) - np.mean(fb[0]))
    guard = dfr <= FR_GUARD
    if d - 2 * sg > -MARGIN and guard:
        outcome = "holds"
    elif d + 2 * sg < -MARGIN:
        outcome = "does not hold, resolved below the margin"
    else:
        outcome = "does not hold, inconclusive" + ("" if guard else " (fake-rate guard fails)")
    n = int(mask.sum())
    return {"claim": "G1", "A": LABELS["canto_l_ttbar"], "B": LABELS["acorn_hnl_lm"], "population": "P_G",
            "split": "generic_iso test", "n_targets": n, "eff_A": float(np.mean(A[0])), "eff_B": float(np.mean(Bm[0])),
            "eff_A_seeds": A[0], "eff_B_seeds": Bm[0], "delta_points": d, "sigma_points": sg,
            "sigma_boot_points": 100 * diff["sigma_boot"], "sigma_seeds_points": 100 * diff["sigma_seeds"],
            "delta_minus_2sigma": d - 2 * sg, "delta_plus_2sigma": d + 2 * sg, "margin": MARGIN,
            "fake_rate_A": float(np.mean(fa[0])), "fake_rate_B": float(np.mean(fb[0])), "fake_rate_diff_points": dfr,
            "fake_rate_guard_ok": bool(guard), "outcome": outcome, "underpowered": n < 1000}


def differences(D: Data) -> list[dict]:
    rows = []
    for s, _ in D.parts:
        t = D.truth[s]
        for pop, mask in list(populations(s, t).items())[:2]:
            A = D.eff("canto_l_ttbar", s, mask)
            for m in ("acorn_hnl_lm", "acorn_ttbar_lm", "acorn_hnl_cc"):
                if m not in D.models():
                    continue
                Bm = D.eff(m, s, mask)
                diff = linear_combination([(1.0, A[0], A[1]), (-1.0, Bm[0], Bm[1])])
                rows.append({"sample": s, "population": pop, "A": LABELS["canto_l_ttbar"], "B": LABELS[m],
                             "delta_points": 100 * diff["value"], "sigma_points": 100 * diff["sigma"],
                             "z": diff["z"]})
    return rows


def _pc(x, d=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def markdown(D: Data, main: pd.DataFrame, bins: pd.DataFrame, diffs: pd.DataFrame, claim: dict | None) -> str:
    title = {"val": "Addendum G: frozen models on the generic val split (pipeline check; nothing chosen)",
             "test": "Addendum G, step T2: frozen models on the generic test split (claim G1)",
             "hss": "h -> s s samples: frozen models (evaluation only, no claim)"}[D.set]
    L = [f"# {title}", "",
         f"Written {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `hnl.eval.addendum_g_summary` "
         f"(git {_git()[:7]}). Frozen models and settings: `results/addendum_g_models_v1.json`. Populations: "
         "`results/addendum_g_t1_v1.md`. Labels of these samples are the simulation's own (correct); the frozen models "
         "were trained and selected on the Zenodo labels, which carry the same-module label exchange "
         "(`docs/new_samples_spec.md` section 8). Efficiencies in percent, 3-seed mean +- sigma (sigma^2 = "
         "sigma_boot^2 + s_seeds^2 / 3; one-seed models: bootstrap only); B = 500 event resamplings per split.", ""]
    if claim:
        L += ["## Claim G1", "",
              f"- P_G targets on the generic_iso test split: {claim['n_targets']}"
              f"{' (underpowered: fewer than 1,000)' if claim['underpowered'] else ''}.",
              f"- {claim['A']}: {_pc(claim['eff_A'])} % (seeds " + ", ".join(_pc(x) for x in claim['eff_A_seeds']) + ")",
              f"- {claim['B']}: {_pc(claim['eff_B'])} % (seeds " + ", ".join(_pc(x) for x in claim['eff_B_seeds']) + ")",
              f"- Delta = {claim['delta_points']:+.2f} +- {claim['sigma_points']:.2f} points (bootstrap "
              f"{claim['sigma_boot_points']:.2f}, seeds {claim['sigma_seeds_points']:.2f}); Delta - 2 sigma = "
              f"{claim['delta_minus_2sigma']:+.2f} against -{MARGIN:.1f}; Delta + 2 sigma = "
              f"{claim['delta_plus_2sigma']:+.2f}.",
              f"- Track fake rate on the generic_iso test split: {_pc(claim['fake_rate_A'])} % against "
              f"{_pc(claim['fake_rate_B'])} % ({claim['fake_rate_diff_points']:+.2f} points; guard <= +{FR_GUARD} "
              f"{'met' if claim['fake_rate_guard_ok'] else 'NOT met'}).",
              f"- **Outcome: {claim['outcome']}.**", ""]
    for s, _ in D.parts:
        sub = main[main["sample"] == s]
        pops = list(dict.fromkeys(sub.population))
        L += [f"## {s} ({D.set})", "", "| model | " + " | ".join(pops) + " |", "|---|" + "---|" * len(pops)]
        for m in D.models():
            cells = []
            for p in pops:
                r = sub[(sub.model == m) & (sub.population == p)].iloc[0]
                cells.append(f"{_pc(r['mean'])} +- {_pc(r['sigma'])}")
            L.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
        ns = sub[sub.model == D.models()[0]].set_index("population").n
        L += ["", "Targets per population: " + "; ".join(f"{p} {int(ns[p])}" for p in pops if p != "track fake rate")
              + ".", "", "Graph limit (perfect classifier + CC) and losses, first population:", "",
              "| model | efficiency | graph limit | lost in the graph | lost in classifier + builder |", "|---|---|---|---|---|"]
        p0 = pops[0]
        for m in D.models():
            r = sub[(sub.model == m) & (sub.population == p0)].iloc[0]
            L.append(f"| {LABELS[m]} | {_pc(r['mean'])} | {_pc(r['ceiling'])} | {_pc(1 - r['ceiling'])} | "
                     f"{_pc(r['ceiling'] - r['mean'])} |")
        L.append("")
    if len(diffs):
        L += ["## Differences (Canto-L (ttbar) minus each baseline; descriptive except G1)", "",
              "| sample | population | baseline | difference [points] |", "|---|---|---|---|"]
        for _, r in diffs.iterrows():
            L.append(f"| {r['sample']} | {r['population']} | {r['B']} | {r['delta_points']:+.2f} +- "
                     f"{r['sigma_points']:.2f} |")
        L.append("")
    L += ["## Bins", "", "Efficiency (graph limit) per bin of the first population of each sample; full table in the csv.", ""]
    for (s, var), g in bins.groupby(["sample", "variable"], sort=False):
        ms = [m for m in D.models() if m in g.model.unique()]
        L += [f"**{s}, {g.population.iloc[0]}, {var}**", "", "| bin | n | " + " | ".join(LABELS[m] for m in ms) + " |",
              "|---|---|" + "---|" * len(ms)]
        for b, gb in g.groupby("bin", sort=False):
            cells = []
            for m in ms:
                r = gb[gb.model == m].iloc[0]
                cells.append(f"{_pc(r['mean'], 1)} ({_pc(r['ceiling'], 1)})")
            L.append(f"| {b} | {int(gb.n.iloc[0])} | " + " | ".join(cells) + " |")
        L.append("")
    if D.set == "val":
        ref = pd.read_csv(os.path.join(REPO, "results", "strong_baselines_v3_val_entries.csv"))
        L += ["## Pipeline check against the Zenodo val events", "",
              "Same models on Zenodo val (`results/strong_baselines_v3_val_entries.csv`, 3-seed means) against the "
              "generic val split. The Zenodo labels carry the label exchange; the generated ones do not.", "",
              "| model | Zenodo ttbar prompt | generic_iso prompt | generic_dense prompt | generic_iso fake rate | "
              "generic_dense fake rate | Zenodo HNL fake rate |", "|---|---|---|---|---|---|---|"]
        for m in [x for x in MAIN if x in D.models()]:
            r = ref[ref.label == VAL_REF[m]]
            g = lambda s, p: main[(main["sample"] == s) & (main.model == m) & (main.population == p)]["mean"].iloc[0]
            L.append(f"| {LABELS[m]} | {_pc(r.eff_ttbar_prompt.mean())} | {_pc(g('generic_iso', 'prompt (ttbar, |d0| < 1 mm)'))} | "
                     f"{_pc(g('generic_dense', 'prompt (ttbar, |d0| < 1 mm)'))} | {_pc(g('generic_iso', 'track fake rate'))} | "
                     f"{_pc(g('generic_dense', 'track fake rate'))} | {_pc(r.fake_rate_hnl.mean())} |")
        L.append("")
    return "\n".join(L)


def summary(set_: str) -> None:
    D = Data(set_)
    missing = [m for m in (MAIN if set_ != "hss" else MAIN) if m not in D.models()]
    if missing:
        print(f"[summary] models without complete results: {missing}")
    main = pd.DataFrame(main_rows(D))
    bins = pd.DataFrame(bin_rows(D))
    diffs = pd.DataFrame(differences(D)) if "canto_l_ttbar" in D.models() else pd.DataFrame()
    claim = g1(D) if set_ == "test" and {"canto_l_ttbar", "acorn_hnl_lm"} <= set(D.models()) else None
    stem = {"val": "addendum_g_val_v1", "test": "addendum_g_g1_v1", "hss": "hss_eval_v1"}[set_]
    res = os.path.join(REPO, "results")
    main.to_csv(os.path.join(res, f"{stem}.csv"), index=False)
    bins.to_csv(os.path.join(res, f"{stem}_bins.csv"), index=False)
    if len(diffs):
        diffs.to_csv(os.path.join(res, f"{stem}_differences.csv"), index=False)
    if claim:
        with open(os.path.join(res, f"{stem}_claim.json"), "w") as f:
            json.dump(claim, f, indent=1, default=float)
    md = markdown(D, main, bins, diffs, claim)
    with open(os.path.join(res, f"{stem}.md"), "w") as f:
        f.write(md)
    print(md)
