"""Event-bootstrap uncertainties of the headline numbers (supervisor feedback round 1, point 5).

"Mean +- seed sd over three seeds" alone will not hold up in review. For every headline number of the validation
ranking v3, the test release card (ranking_test_v1), the component decomposition (paper Fig. 8) and the |d0| bins,
this module reports, with the estimator of the preregistered claims (hnl.eval.bootstrap, docs/preregistration.md
section 5):

  value      mean of the seeds' numbers, each pooled over the sets (the estimator of the ranking tables and claims);
  seed sd    standard deviation of the seeds' numbers (ddof 1); the "+-" of the ranking tables;
  boot sd    event-bootstrap standard deviation of the seed mean: events resampled with replacement, the same
             resampled events for every seed and every model (one fixed replicate table per set), so differences are
             paired;
  combined   sqrt(boot sd^2 + seed sd^2 / n_seeds), the total sigma of hnl.eval.claims (one seed: no seed term, as
             there; the seed spread of such a row is unknown, not zero);
  intervals  for the k/n rows (efficiencies of a fixed particle population): 68 % Clopper-Pearson on the per-seed
             particle count (paper Fig. 7 convention: k = seed-mean efficiency x n per seed, n = particles per seed;
             the seed-pooled K/N = 3k/3n would count the same particles three times) and the 68 % percentile interval
             (15.87 / 84.13 %) of the bootstrap replicates of the seed mean.
Differences between two models: hnl.eval.bootstrap.linear_combination (paired bootstrap of the difference of seed means
plus both seed terms, as the claims). Where both models are the same trained networks with another track builder
(seed i of one is seed i of the other) the seed term is also given paired by seed: sd of the per-seed differences /
sqrt(n_seeds).

Sets and resampling units (the ttbar-prompt efficiency resamples ttbar events, everything else the HNL events):
  test  the preregistered tables of hnl.eval.bootstrap (derived/bootstrap/v1; B = 500, seed 12345): hnl_joint_test
        (hnl10_100_test + hnl15_200_test resampled by event index, both samples' event i together), hnl15_100_external
        and ttbar_test; pooled HNL = hnl_joint_test + hnl15_100_external, each set with its own resampling. Identical
        replicates to results/claims_test_v1.json (identity checks below). --test-b1000 adds a check with fresh
        B = 1000 tables (Monte Carlo precision of B = 500).
  val   tables built the same way (B = 1000, seed 20261005): hnl_joint_val (hnl10_100_val + hnl15_200_val by event
        index, 1,999 events on 1,000 units) and ttbar_val (1,000 events), written once to
        runs/feedback_v1/bootstrap_v1/tables (refuses to overwrite tables with other units or B).

Rows (nothing is chosen here; every working point is read from the committed files):
  val ranking v3       every row of results/ranking/ranking_val_v3.csv at the per-seed working points of
                       ranking_val_v3_entries.csv (CC cut or track-building configuration); every entry and every row
                       mean and seed sd reproduced (|difference| <= 1e-12, counts exactly);
  test release card    the frozen models of results/ranking/spec_test_v1.json (TS.RunTest), every entry and row of
                       ranking_test_v1*.csv reproduced, and the claims' primary and S1 differences with their sigma
                       (results/claims_test_v1.json); --extra-test-spec adds rows in the same format (e.g. a later
                       acorn (HNL) + MWS-layer + merging test row);
  decomposition (val)  paper Fig. 8: acorn (ttbar) CC -> + MWS-layer + merging -> Canto-B -> Canto-H (MWS-layer +
                       merging) -> Canto-H walkthrough + merging -> Canto-L walkthrough + merging -> Canto-L learned
                       merging (selected); the same-builder rows at the per-seed configurations of
                       results/final_selection_v1_seeds.csv (checked), with paired differences between steps;
  bins                 HNL-daughter targets in the preregistered |d0| bins (and the r_v bins with the "no vertex"
                       category) for the models of paper Fig. 7 (val) and the release-card models (test).
Status: validation numbers; test numbers are post-freeze descriptive summaries of the one-time test outputs, not
claims (results/claims_test_v1.json stays as it is). On val, the working points (CC best cut, builder rule) were
chosen on the same val events; the bootstrap holds them fixed and does not resample that choice.

    python -m hnl.eval.bootstrap_tables --out results --name bootstrap_v1 [--test-b1000]
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import beta

from hnl.eval import bootstrap as BS
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table, split_event_id

REPO = os.environ.get("HNL_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
OUT_RUNS = os.path.join(DATA, "runs", "feedback_v1", "bootstrap_v1")
VAL_SPEC = "results/ranking/spec_val_v3.json"
VAL_RANK = "results/ranking/ranking_val_v3.csv"
VAL_ENTRIES = "results/ranking/ranking_val_v3_entries.csv"
TEST_SPEC = "results/ranking/spec_test_v1.json"
TEST_RANK = "results/ranking/ranking_test_v1.csv"
TEST_ENTRIES = "results/ranking/ranking_test_v1_entries.csv"
SELECTION = "results/final_selection_v1.json"
SEL_SEEDS = "results/final_selection_v1_seeds.csv"
CLAIMS = "results/claims_test_v1.json"
FIG08 = "paper/figures/fig08_decomposition.csv"

VAL_SETS = {"hnl_joint_val": (("hnl10_100", "val"), ("hnl15_200", "val")), "ttbar_val": (("ttbar", "val"),)}
VAL_CODE = {1: "ttbar_val", 2: "hnl_joint_val", 4: "hnl_joint_val"}
SAMPLE_CODE = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}
B_VAL, SEED_VAL = 1000, 20261005
B_TEST_CHECK, SEED_TEST_CHECK = 1000, 20261006
TTBAR_CODE = 1
D0_EDGES = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)
RV_EDGES = (0.0, 10.0, 30.0, 100.0, 200.0, 300.0, np.inf)
CL = 0.6827
PCT = (50.0 * (1.0 - CL), 100.0 - 50.0 * (1.0 - CL))
EFFS = ("eff_primary", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt")
KEYS = EFFS + ("fake_rate_hnl",)
TOL = 1e-12


def _repo(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _num(x: float) -> str:
    return "inf" if np.isinf(x) else f"{x:g}"


def bin_name(var: str, lo: float, hi: float) -> str:
    return f"{var}_{_num(lo)}_{_num(hi)}"


D0_BINS = tuple(bin_name("d0", lo, hi) for lo, hi in zip(D0_EDGES[:-1], D0_EDGES[1:]))
RV_BINS = tuple(bin_name("rv", lo, hi) for lo, hi in zip(RV_EDGES[:-1], RV_EDGES[1:])) + ("rv_none",)
QUANTITIES = KEYS + D0_BINS + RV_BINS


# --------------------------------------------------------------------------------------------------------------------
# replicate tables


class Resampler:
    """Fixed bootstrap replicate tables: per set the sorted resampling units (event indices) and the (B, units)
    multiplicities; event ids map to sets by their sample code."""

    def __init__(self, units: dict, w: dict, set_of_code: dict, label: str):
        self.units = {s: np.asarray(u, dtype=np.int64) for s, u in units.items()}
        self.w = {s: np.asarray(v).astype(np.float64) for s, v in w.items()}
        self.set_of_code = dict(set_of_code)
        self.label = label
        bs = {v.shape[0] for v in self.w.values()}
        if len(bs) != 1:
            raise ValueError(f"tables with different B: {bs}")
        self.b = bs.pop()

    @classmethod
    def claims(cls) -> "Resampler":
        """The preregistered test tables (B = 500, seed 12345), shared with hnl.eval.claims."""
        bs = BS.Bootstrap()
        return cls(bs.units, bs.w, TS.SET_OF_CODE, f"preregistered test tables (B = {BS.B}, seed {BS.SEED})")

    @classmethod
    def build(cls, sets: dict, set_of_code: dict, directory: str, b: int, seed: int, label: str) -> "Resampler":
        """Tables for `sets` ({name: ((sample, split), ...)}): created once with hnl.eval.bootstrap.make_weights
        (SeedSequence(seed).spawn per set, as hnl.eval.bootstrap), reused afterwards; refuses other units or B."""
        from hnl.data.splits import read_split
        units, w = {}, {}
        for (name, parts), ss in zip(sets.items(), np.random.SeedSequence(seed).spawn(len(sets))):
            u = np.unique(np.concatenate([read_split(s, sp) for s, sp in parts]))
            path = os.path.join(directory, f"{name}.npz")
            if os.path.exists(path):
                z = np.load(path)
                if not np.array_equal(z["units"], u) or z["w"].shape[0] != b or int(z["seed"]) != seed:
                    raise RuntimeError(f"{path} does not match the current split / B / seed")
                ww = z["w"]
            else:
                ww = BS.make_weights(len(u), b, np.random.default_rng(ss))
                os.makedirs(directory, exist_ok=True)
                np.savez_compressed(path + ".tmp.npz", units=u, w=ww, seed=seed, b=b)
                os.replace(path + ".tmp.npz", path)
            units[name], w[name] = u, ww
        return cls(units, w, set_of_code, label)

    def unit_sums(self, event_id, num, den) -> dict:
        """{set: (sum of num per unit, sum of den per unit)} over the rows (identified by their event id)."""
        event_id = np.asarray(event_id, dtype=np.int64)
        num = np.asarray(num, dtype=np.float64)
        den = np.asarray(den, dtype=np.float64)
        code, idx = split_event_id(event_id)
        out = {}
        for c in np.unique(code):
            s = self.set_of_code.get(int(c))
            if s is None:
                raise ValueError(f"sample code {c} is not in the sets of {self.label}")
            m = code == c
            u = self.units[s]
            pos = np.searchsorted(u, idx[m])
            if np.any(pos >= len(u)) or np.any(u[np.minimum(pos, len(u) - 1)] != idx[m]):
                raise ValueError(f"events of code {c} outside set {s}")
            k = np.bincount(pos, weights=num[m], minlength=len(u))
            n = np.bincount(pos, weights=den[m], minlength=len(u))
            if s in out:
                out[s] = (out[s][0] + k, out[s][1] + n)
            else:
                out[s] = (k, n)
        return out

    def ratio(self, sums: dict) -> tuple[float, np.ndarray, float, float]:
        """(point, B replicates, numerator, denominator) of sum(num) / sum(den) pooled over the sets."""
        num = den = 0.0
        rnum, rden = np.zeros(self.b), np.zeros(self.b)
        for s, (k, n) in sums.items():
            num += float(k.sum())
            den += float(n.sum())
            rnum += self.w[s] @ k
            rden += self.w[s] @ n
        with np.errstate(invalid="ignore", divide="ignore"):
            return (float(num / den) if den else np.nan), rnum / rden, num, den


# --------------------------------------------------------------------------------------------------------------------
# per-seed counts


def masks(t: pd.DataFrame) -> dict:
    """Particle populations of the ranking tables (hnl.eval.baseline_val.populations on val, hnl.eval.test_stats.masks
    with the pooled sets on test) and the HNL-daughter bins."""
    code = split_event_id(t.event_id.to_numpy())[0]
    hnl_ev = code != TTBAR_CODE
    tgt = t.is_target.to_numpy(dtype=bool)
    cat = t.category.to_numpy()
    ad0 = t.d0.abs().to_numpy(dtype=float)
    rv = t.rv.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        disp = (ad0 >= 10) | (rv >= 100)
        hnl = hnl_ev & tgt & (cat == "hnl")
        m = {"eff_primary": hnl & disp, "eff_hnl_targets": hnl, "eff_w_muons": hnl_ev & tgt & (cat == "w"),
             "eff_ttbar_prompt": (code == TTBAR_CODE) & tgt & (ad0 < 1.0)}
        for name, lo, hi in zip(D0_BINS, D0_EDGES[:-1], D0_EDGES[1:]):
            m[name] = hnl & (ad0 >= lo) & (ad0 < hi)
        for name, lo, hi in zip(RV_BINS[:-1], RV_EDGES[:-1], RV_EDGES[1:]):
            m[name] = hnl & (rv >= lo) & (rv < hi)
    m["rv_none"] = hnl & np.isnan(rv)
    return m


def info_table(event_ids) -> pd.DataFrame:
    """The particle truth columns needed here, for the particles that can enter a population."""
    p = particle_table(np.unique(np.asarray(event_ids, dtype=np.int64)))
    p = p[["event_id", "particle_id", "is_target", "category", "d0", "rv"]]
    return p[p.is_target.astype(bool)].reset_index(drop=True)


def counts(rs: Resampler, pp: pd.DataFrame, pe: pd.DataFrame, info: pd.DataFrame) -> dict:
    """One seed at one working point: {quantity: {set: (k per unit, n per unit)}}; pp = per-particle rows (event_id,
    particle_id, dm_matched), pe = per-event rows (event_id, n_fake, n_tracks_ge5)."""
    t = pp[["event_id", "particle_id", "dm_matched"]].merge(info, on=["event_id", "particle_id"], how="inner")
    if t.duplicated(["event_id", "particle_id"]).any():
        raise ValueError("duplicate particles at one working point")
    ev, ok = t.event_id.to_numpy(), t.dm_matched.to_numpy(dtype=float)
    out = {}
    for q, m in masks(t).items():
        out[q] = rs.unit_sums(ev[m], ok[m], np.ones(int(m.sum())))
    code = split_event_id(pe.event_id.to_numpy())[0]
    h = code != TTBAR_CODE
    out["fake_rate_hnl"] = rs.unit_sums(pe.event_id.to_numpy()[h], pe.n_fake.to_numpy(float)[h],
                                        pe.n_tracks_ge5.to_numpy(float)[h])
    return out


def _read(d: str, f: str, key: str, value: float) -> pd.DataFrame:
    flt = [(key, "==", int(value))] if key == "config" else [(key, ">=", value - 5e-5), (key, "<=", value + 5e-5)]
    df = pd.read_parquet(os.path.join(d, f), filters=flt)
    if key == "config":
        return df[df.config == int(value)].reset_index(drop=True)
    return df[np.isclose(df.cut.astype(np.float64), value, atol=1e-6)].reset_index(drop=True)


def val_tables(run: str, wp: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Per-particle and per-event rows of one val working point as written in the ranking entries ("CC 0.34 ..." or
    "family params (grid)")."""
    m = re.match(r"^CC (\d+\.\d+)", wp)
    if m:
        d, key, val = os.path.join(run, "eval_val"), "cut", float(m.group(1))
    else:
        m = re.match(r"^(\S+) (\S+) \((\S+)\)$", wp)
        if m is None:
            raise ValueError(f"working point {wp!r}")
        fam, params, grid = m.groups()
        d = os.path.join(run, grid)
        cfgs = json.load(open(os.path.join(d, "configs.json")))
        k = [i for i, c in enumerate(cfgs) if c[0] == fam and ",".join(str(x) for x in c[1:]) == params]
        if len(k) != 1:
            raise ValueError(f"{d}: configuration {fam} {params} found {len(k)} times")
        key, val = "config", k[0]
    pp, pe = _read(d, "per_particle.parquet", key, val), _read(d, "per_event.parquet", key, val)
    if pp.empty or pe.empty:
        raise ValueError(f"{d}: no rows at {key} = {val}")
    return pp, pe, d


def expected_events(sets: dict) -> np.ndarray:
    from hnl.data.splits import read_split
    return np.sort(np.concatenate([np.int64(SAMPLE_CODE[s]) * 100_000_000 + read_split(s, sp).astype(np.int64)
                                   for parts in sets.values() for s, sp in parts]))


# --------------------------------------------------------------------------------------------------------------------
# models and summaries


class Model:
    """A list of seeds (each: {quantity: {set: (k, n)}}) with the replicates of every quantity."""

    def __init__(self, name: str, short: str, seeds: list[dict], rs: Resampler, wps: list[str], runs: list[str],
                 group: str = ""):
        self.name, self.short, self.group, self.wps, self.runs = name, short, group, wps, runs
        self.rs = rs
        self.pts, self.reps, self.num, self.den = {}, {}, {}, {}
        for q in QUANTITIES:
            r = [rs.ratio(s[q]) for s in seeds]
            self.pts[q] = [x[0] for x in r]
            self.reps[q] = [x[1] for x in r]
            self.num[q] = [x[2] for x in r]
            self.den[q] = [x[3] for x in r]

    @property
    def n_seeds(self) -> int:
        return len(self.runs)

    def pr(self, q: str) -> tuple[list, list]:
        return self.pts[q], self.reps[q]


def cp(k: float, n: float, cl: float = CL) -> tuple[float, float]:
    """Clopper-Pearson interval (paper/figures/paperstyle.cp; k may be a seed-mean count)."""
    if n <= 0:
        return np.nan, np.nan
    a = (1.0 - cl) / 2.0
    lo = float(beta.ppf(a, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1.0 - a, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def summary(m: Model, q: str) -> dict:
    p, r = m.pr(q)
    d = BS.seed_mean(p, r)
    rm = np.mean(np.stack(r), axis=0)
    lo, hi = (np.nanpercentile(rm, PCT) if np.isfinite(rm).any() else (np.nan, np.nan))
    out = {"value": d["mean"], "seed_sd": d["s_seeds"] if m.n_seeds > 1 else np.nan, "boot_sd": d["sigma_boot"],
           "combined": d["sigma"], "n_seeds": m.n_seeds, "boot68_lo": float(lo), "boot68_hi": float(hi),
           "B": m.rs.b, "per_seed": ";".join(f"{x:.10g}" for x in p)}
    if q != "fake_rate_hnl":
        ns = np.asarray(m.den[q], dtype=float)
        ks = np.asarray(m.num[q], dtype=float)
        if not np.all(ns == ns[0]):
            raise ValueError(f"{m.name} {q}: particle counts differ between seeds {ns}")
        n = float(ns[0])
        out.update({"k_pooled": int(round(ks.sum())), "n_pooled": int(round(ns.sum())), "n_per_seed": int(n)})
        out["cp68_lo"], out["cp68_hi"] = cp(d["mean"] * n, n)
        # the seed-pooled count as if the seeds' particles were independent (too narrow; CSV only)
        out["cp68_pooled_lo"], out["cp68_pooled_hi"] = cp(float(ks.sum()), float(ns.sum()))
    else:
        out.update({"k_pooled": int(round(sum(m.num[q]))), "n_pooled": int(round(sum(m.den[q])))})
    return out


def difference(new: Model, base: Model, q: str = "eff_primary", paired_seeds: bool = False) -> dict:
    """new - base, paired over the events (claims estimator); with paired_seeds also the seed term of the per-seed
    differences (same trained networks)."""
    if new.rs is not base.rs:
        raise ValueError("a difference needs the same replicate tables")
    d = BS.linear_combination([(1.0, *new.pr(q)), (-1.0, *base.pr(q))])
    rep = np.mean(np.stack(new.reps[q]), axis=0) - np.mean(np.stack(base.reps[q]), axis=0)
    lo, hi = np.nanpercentile(rep, PCT)
    out = {"value": d["value"], "boot_sd": d["sigma_boot"], "seed_sd": d["sigma_seeds"], "combined": d["sigma"],
           "z": d["z"], "boot68_lo": float(lo), "boot68_hi": float(hi), "B": new.rs.b,
           "n_seeds": min(new.n_seeds, base.n_seeds)}
    if paired_seeds:
        if new.runs != base.runs:
            raise ValueError(f"{new.name} / {base.name}: not the same runs, no seed pairing")
        ds = np.asarray(new.pts[q]) - np.asarray(base.pts[q])
        st = float(np.std(ds, ddof=1) / np.sqrt(len(ds)))
        out["seed_term_paired"] = st
        out["combined_paired"] = float(np.sqrt(d["sigma_boot"] ** 2 + st ** 2))
        out["per_seed"] = ";".join(f"{x:.10g}" for x in ds)
    return out


# --------------------------------------------------------------------------------------------------------------------
# val


class ValLoader:
    def __init__(self, rs: Resampler):
        self.rs, self.cache, self.info = rs, {}, None
        self.events = expected_events(VAL_SETS)

    def seed(self, run: str, wp: str) -> dict:
        key = (os.path.normpath(run), wp.split(" (FR*")[0])
        if key not in self.cache:
            pp, pe, d = val_tables(run, wp)
            ev = np.unique(pe.event_id.to_numpy())
            if not np.array_equal(ev, self.events) or len(pe) != len(ev):
                raise ValueError(f"{d} at {wp}: {len(ev)} events ({len(pe)} rows), expected the {len(self.events)} "
                                 f"val events")
            if self.info is None:
                self.info = info_table(self.events)
            self.cache[key] = counts(self.rs, pp, pe, self.info)
        return self.cache[key]


def check_entry(m: Model, i: int, ref: pd.Series, where: str, checks: list) -> None:
    """One seed of a model against a ranking entry (efficiencies, fake rate, |d0| counts)."""
    for k in KEYS:
        ok = abs(m.pts[k][i] - float(ref[k])) <= TOL
        checks.append((f"{where}: {m.short or m.name} seed {i} {k}", m.pts[k][i], float(ref[k]), ok))
    for lo, q in ((100, "d0_100_300"), (300, "d0_300_inf")):
        for col, mine in ((f"n_d0_{lo}", m.den[q][i]), (f"k_d0_{lo}", m.num[q][i])):
            ok = int(round(mine)) == int(ref[col])
            checks.append((f"{where}: {m.short or m.name} seed {i} {col}", float(mine), float(ref[col]), ok))


def check_row(m: Model, row: pd.Series, where: str, checks: list) -> None:
    for k in KEYS:
        s = summary(m, k)
        checks.append((f"{where}: {m.short or m.name} {k} mean", s["value"], float(row[k]),
                       abs(s["value"] - float(row[k])) <= TOL))
        if m.n_seeds > 1:
            checks.append((f"{where}: {m.short or m.name} {k} seed sd", s["seed_sd"], float(row[k + "_sd"]),
                           abs(s["seed_sd"] - float(row[k + "_sd"])) <= TOL))


def val_ranking(ld: ValLoader, checks: list) -> list[Model]:
    rank = pd.read_csv(_repo(VAL_RANK))
    ent = pd.read_csv(_repo(VAL_ENTRIES))
    spec = {m["label"]: m for m in json.load(open(_repo(VAL_SPEC)))}
    out = []
    for _, row in rank.iterrows():
        e = ent[ent.label == row.label]
        runs = [m["run"] for m in spec[row.label]["entries"]]
        if list(e.run) != runs:
            raise SystemExit(f"{row.label}: entries {list(e.run)} differ from the spec {runs}")
        seeds = [ld.seed(r.run, r.working_point) for _, r in e.iterrows()]
        short = short_name(row.label, row.short)
        m = Model(row.label, short, seeds, ld.rs, list(e.working_point), runs, row.group)
        for i, (_, r) in enumerate(e.iterrows()):
            check_entry(m, i, r, "val v3 entry", checks)
        check_row(m, row, "val v3 row", checks)
        out.append(m)
    return out


SAME_BUILDER = (  # decomposition rows not in the ranking: (name, short, pipeline, family)
    ("Canto-B (full budget), MWS-layer + merging", "Canto-B MWS+M", "l1t", "mwslayermerge"),
    ("Canto-H (full budget), MWS-layer + merging", "Canto-H MWS+M", "l1tu", "mwslayermerge"),
    ("Canto-L (full budget), walkthrough + merging", "Canto-L WT+M", "l1tl", "walkmerge"),
    ("Canto-L (full budget), MWS-layer + merging", "Canto-L MWS+M", "l1tl", "mwslayermerge"),
)
R6 = {"l1tu": "Canto-H (full budget): hand-cut helix union, walkthrough + merging (rule R6)",
      "l1tl": "Canto-L (full budget): learned triplet union, learned merging (rule R6)",
      "l1t": "Canto-B (full budget): base graph, learned merging (rule R6)"}


def same_builder(ld: ValLoader, checks: list) -> dict:
    sel = json.load(open(_repo(SELECTION)))
    ss = pd.read_csv(_repo(SEL_SEEDS))
    out = {}
    for name, short, pipe, fam in SAME_BUILDER:
        seeds, wps, runs = [], [], []
        for s in sorted(sel["pipelines"][pipe]["seeds"], key=int):
            r = ss[(ss.pipeline == pipe) & (ss.seed == int(s)) & (ss.family == fam)]
            if len(r) != 1:
                raise SystemExit(f"{SEL_SEEDS}: {len(r)} rows for {pipe} seed {s} {fam}")
            r = r.iloc[0]
            run = sel["pipelines"][pipe]["seeds"][s]["run"]
            wp = f"{fam} {r.config.split(',', 1)[1]} ({r.grid})"
            seeds.append(ld.seed(run, wp))
            wps.append(wp)
            runs.append(run)
        m = Model(name, short, seeds, ld.rs, wps, runs, "same builder")
        for i, s in enumerate(sorted(sel["pipelines"][pipe]["seeds"], key=int)):
            r = ss[(ss.pipeline == pipe) & (ss.seed == int(s)) & (ss.family == fam)].iloc[0]
            for k in KEYS:
                checks.append((f"final_selection_v1_seeds: {short} seed {s} {k}", m.pts[k][i], float(r[k]),
                               abs(m.pts[k][i] - float(r[k])) <= TOL))
        out[short] = m
    return out


DECOMP = (  # (step label, model key, facts value in percent)
    ("acorn (ttbar), CC (best cut <= FR*)", "acorn IGNN, ttbar-trained + CC (best cut <= FR*)", "92.3"),
    ("+ MWS-layer + merging (same acorn (ttbar) scores)",
     "acorn IGNN, ttbar-trained + Canto builder (MWS-layer + merging)", "96.8"),
    ("+ Canto classifier (Canto-B, same graph and builder)", "Canto-B MWS+M", "98.25"),
    ("+ helix triplets, hand cuts (Canto-H, same builder)", "Canto-H MWS+M", "98.72"),
    ("walkthrough + merging (Canto-H, rule R6)", R6["l1tu"], "99.12"),
    ("learned triplet filter (Canto-L, walkthrough + merging)", "Canto-L WT+M", "99.22"),
    ("learned merging (Canto-L, rule R6, selected)", R6["l1tl"], "99.10"),
)
REF_HNL_CC = "acorn IGNN, HNL-trained + CC (best cut <= FR*)"
REF_HNL_MWS = "acorn IGNN, HNL-trained + Canto builder (MWS-layer + merging)"
# val differences: (label, new, base, paired seeds (same trained networks), group)
VAL_DIFFS = (
    ("step 1: builder, CC -> MWS-layer + merging (acorn (ttbar) scores)", 1, 0, True, "staircase"),
    ("step 2: classifier, acorn IGNN -> Canto (base graph, MWS-layer + merging)", 2, 1, False, "staircase"),
    ("step 3: graph, base -> helix union with hand cuts (MWS-layer + merging)", 3, 2, False, "staircase"),
    ("step 4: builder, MWS-layer + merging -> walkthrough + merging (Canto-H)", 4, 3, True, "staircase"),
    ("step 5: graph filter, hand cuts -> learned (walkthrough + merging)", 5, 4, False, "staircase"),
    ("step 6: builder, walkthrough + merging -> learned merging (Canto-L)", 6, 5, True, "staircase"),
    ("98.72 -> 99.10: Canto-H MWS-layer + merging -> Canto-L learned merging (selected)", 6, 3, False, "five-step"),
    ("total: acorn (ttbar) CC -> Canto-L learned merging (selected)", 6, 0, False, "total"),
    ("graph filter at the same builder: Canto-L - Canto-H, both MWS-layer + merging", "Canto-L MWS+M",
     "Canto-H MWS+M", False, "same builder"),
    ("Canto-L (selected) - acorn (HNL) + MWS-layer + merging", R6["l1tl"], REF_HNL_MWS, False, "vs acorn (HNL)"),
    ("Canto-L MWS-layer + merging - acorn (HNL) MWS-layer + merging (same builder)", "Canto-L MWS+M", REF_HNL_MWS,
     False, "vs acorn (HNL)"),
    ("Canto-H (R6) - acorn (HNL) + MWS-layer + merging", R6["l1tu"], REF_HNL_MWS, False, "vs acorn (HNL)"),
    ("Canto-B MWS-layer + merging - acorn (HNL) MWS-layer + merging (same builder)", "Canto-B MWS+M", REF_HNL_MWS,
     False, "vs acorn (HNL)"),
    ("Canto-B (R6, learned merging) - acorn (HNL) + MWS-layer + merging", R6["l1t"], REF_HNL_MWS, False,
     "vs acorn (HNL)"),
    ("Canto-L (selected) - acorn (HNL) CC (best cut <= FR*)", R6["l1tl"], REF_HNL_CC, False, "vs acorn (HNL)"),
    ("acorn (HNL): CC (best cut) -> MWS-layer + merging (same scores)", REF_HNL_MWS, REF_HNL_CC, True,
     "builder, acorn (HNL)"),
    ("Canto-H (R6) - Canto-L (selected) (the R8 tie)", R6["l1tu"], R6["l1tl"], False, "Canto variants"),
    ("Canto-L (selected) - Canto-B (R6)", R6["l1tl"], R6["l1t"], False, "Canto variants"),
)
SHORT = {  # short labels of ranking v3 -> the naming of the Canto deliverables
    "acorn ttbar CC": "acorn (ttbar) CC", "acorn ttbar CC*": "acorn (ttbar) CC*",
    "acorn ttbar + MWS": "acorn (ttbar) + MWS+M", "acorn HNL CC": "acorn (HNL) CC", "acorn HNL CC*": "acorn (HNL) CC*",
    "acorn HNL + MWS": "acorn (HNL) + MWS+M", "acorn mix CC": "acorn (ttbar+HNL) CC",
    "acorn mix CC*": "acorn (ttbar+HNL) CC*", "acorn mix + MWS": "acorn (ttbar+HNL) + MWS+M",
    "Canto-H prev.": "Canto-H preview", "Canto-L prev.": "Canto-L preview", "Canto-B prev.": "Canto-B preview",
    "L3 HNL-trained": "L3 triplets, HNL-trained (v1)",
}


def short_name(label: str, short) -> str:
    if isinstance(short, str) and short:
        return SHORT.get(short, short)
    return label.removesuffix(", ttbar, best builder") if label.startswith("ladder: ") else label


FIG7_VAL = (R6["l1tl"], R6["l1tu"], R6["l1t"], REF_HNL_CC, "acorn IGNN, ttbar-trained + CC (best cut <= FR*)",
            REF_HNL_MWS, "DeZoort IN-L (127k par.) + CC [FR* unreachable]")


# --------------------------------------------------------------------------------------------------------------------
# test


def test_models(rs: Resampler, spec: list, checks: list, extra: bool = False) -> list[Model]:
    rts = [[TS.RunTest(e["run"], float(e["cut"]), sub=e.get("sub", "eval_test")) for e in m["entries"]] for m in spec]
    evs = {frozenset(rt.pe.event_id.unique()) for runs in rts for rt in runs}
    if len(evs) != 1:
        raise ValueError("the test entries were evaluated on different events")
    events = np.array(sorted(next(iter(evs))), dtype=np.int64)
    info = info_table(events)
    ent = pd.read_csv(_repo(TEST_ENTRIES))
    rank = pd.read_csv(_repo(TEST_RANK)).set_index("label")
    cache, out = {}, []
    for m, runs in zip(spec, rts):
        seeds = []
        for e, rt in zip(m["entries"], runs):
            key = (os.path.normpath(os.path.join(e["run"], e.get("sub", "eval_test"))), round(float(e["cut"]), 6))
            if key not in cache:
                cache[key] = counts(rs, rt.pp, rt.pe, info)
            seeds.append(cache[key])
        mod = Model(m["label"], m.get("short") or m["label"], seeds, rs, [e.get("wp", "") for e in m["entries"]],
                    [e["run"] for e in m["entries"]], m.get("group", ""))
        if not extra:
            e = ent[ent.label == m["label"]]
            if list(e.run) != mod.runs:
                raise SystemExit(f"{m['label']}: test entries {list(e.run)} differ from the spec")
            for i, (_, r) in enumerate(e.iterrows()):
                check_entry(mod, i, r, "test entry", checks)
            check_row(mod, rank.loc[m["label"]], "test row", checks)
        out.append(mod)
    return out


TEST_DIFFS = (  # (label, new short, base short, group)
    ("Canto-L - acorn (HNL) CC, preregistered cut (= claims primary difference)", "Canto-L", "acorn (HNL) CC",
     "claims"),
    ("Canto-L - acorn (ttbar) CC (= claims S1 vs arm (i))", "Canto-L", "acorn (ttbar) CC", "claims"),
    ("Canto-L - acorn (HNL) CC, val-best cut", "Canto-L", "acorn (HNL) CC*", "vs acorn (HNL)"),
    ("Canto-H - Canto-L", "Canto-H", "Canto-L", "Canto variants"),
    ("Canto-L - Canto-B", "Canto-L", "Canto-B", "Canto variants"),
    ("Canto-H - Canto-B", "Canto-H", "Canto-B", "Canto variants"),
)


def claims_checks(diffs: dict, checks: list) -> None:
    res = json.load(open(_repo(CLAIMS)))["result"]
    pairs = (("Canto-L - acorn (HNL) CC, preregistered cut (= claims primary difference)", res["primary"]["difference"],
              "claims primary difference"),
             ("Canto-L - acorn (ttbar) CC (= claims S1 vs arm (i))", res["S1"]["vs_arm_i"], "claims S1 vs arm (i)"))
    for lab, c, what in pairs:
        d = diffs[lab]
        for mine, theirs in (("value", "value"), ("boot_sd", "sigma_boot"), ("seed_sd", "sigma_seeds"),
                             ("combined", "sigma")):
            checks.append((f"{what}: {theirs}", d[mine], float(c[theirs]), abs(d[mine] - float(c[theirs])) <= TOL))


# --------------------------------------------------------------------------------------------------------------------
# output


def model_rows(table: str, scope: str, models: list, qs=QUANTITIES) -> list[dict]:
    rows = []
    for m in models:
        for q in qs:
            rows.append({"table": table, "scope": scope, "model": m.name, "short": m.short, "quantity": q,
                         **summary(m, q), "working_points": "; ".join(m.wps)})
    return rows


def f4(x, d=4) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pts(x, d=2) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}"


def md_primary(models: list, title: str, rank_col: bool = True) -> list[str]:
    lines = [f"### {title}", "",
             "| model | seeds | value | seed sd | boot sd | combined | 68 % CP | 68 % bootstrap | k/n (seeds pooled) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        s = summary(m, "eff_primary")
        lines.append(f"| {m.short} | {m.n_seeds} | {f4(s['value'])} | {f4(s['seed_sd'])} | {f4(s['boot_sd'])} | "
                     f"{f4(s['combined'])} | [{f4(s['cp68_lo'])}, {f4(s['cp68_hi'])}] | "
                     f"[{f4(s['boot68_lo'])}, {f4(s['boot68_hi'])}] | {s['k_pooled']}/{s['n_pooled']} |")
    return lines + [""]


def md_other(models: list, title: str) -> list[str]:
    lines = [f"### {title}", "", "Cells: value (seed sd / boot sd / combined).", "",
             "| model | HNL daughters | W muons | ttbar prompt | fake rate |", "|---|---|---|---|---|"]
    for m in models:
        cells = []
        for q in ("eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt", "fake_rate_hnl"):
            s = summary(m, q)
            cells.append(f"{f4(s['value'])} ({f4(s['seed_sd']) or '-'} / {f4(s['boot_sd'])} / {f4(s['combined'])})")
        lines.append(f"| {m.short} | " + " | ".join(cells) + " |")
    return lines + [""]


def md_kn(models: list, title: str, qs=("d0_100_300", "d0_300_inf")) -> list[str]:
    lab = {"d0_100_300": "|d0| 100-300 mm", "d0_300_inf": "|d0| >= 300 mm"}
    lines = [f"### {title}", "",
             "| model | bin | k/n (seeds pooled) | n per seed | value | seed sd | boot sd | combined | 68 % CP | "
             "68 % bootstrap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q in qs:
        for m in models:
            s = summary(m, q)
            b = lab.get(q, q).replace("|", chr(92) + "|")
            lines.append(f"| {m.short} | {b} | {s['k_pooled']}/{s['n_pooled']} | "
                         f"{s['n_per_seed']} | {f4(s['value'])} | {f4(s['seed_sd'])} | {f4(s['boot_sd'])} | "
                         f"{f4(s['combined'])} | [{f4(s['cp68_lo'])}, {f4(s['cp68_hi'])}] | "
                         f"[{f4(s['boot68_lo'])}, {f4(s['boot68_hi'])}] |")
    return lines + [""]


def md_bins(models: list, title: str, qs=D0_BINS) -> list[str]:
    lines = [f"### {title}", "",
             "| model | \\|d0\\| bin (mm) | n per seed | value | seed sd | boot sd | combined | 68 % CP | "
             "68 % bootstrap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        for q in qs:
            s = summary(m, q)
            lo, hi = q.split("_")[1:]
            lines.append(f"| {m.short} | {lo}-{hi} | {s['n_per_seed']} | {f4(s['value'])} | {f4(s['seed_sd'])} | "
                         f"{f4(s['boot_sd'])} | {f4(s['combined'])} | [{f4(s['cp68_lo'])}, {f4(s['cp68_hi'])}] | "
                         f"[{f4(s['boot68_lo'])}, {f4(s['boot68_hi'])}] |")
    return lines + [""]


def md_diffs(diffs: list[tuple[str, dict]], title: str, paired_col: bool) -> list[str]:
    head = ("| difference (primary efficiency) | value (points) | boot sd (paired) | seed term | combined | z | "
            "68 % bootstrap |" + (" seed term, paired seeds | combined, paired seeds |" if paired_col else ""))
    lines = [f"### {title}", "", "Points = percentage points; seed term = sqrt(s_new^2 / 3 + s_base^2 / 3) (claims); "
             "paired seeds: sd of the per-seed differences / sqrt(3), only where both rows are the same trained "
             "networks.", "", head, "|---" * (7 + 2 * paired_col) + "|"]
    for lab, d in diffs:
        row = (f"| {lab} | {pts(d['value'])} | {100 * d['boot_sd']:.2f} | {100 * d['seed_sd']:.2f} | "
               f"{100 * d['combined']:.2f} | {d['z']:.1f} | [{pts(d['boot68_lo'])}, {pts(d['boot68_hi'])}] |")
        if paired_col:
            if "seed_term_paired" in d:
                row += f" {100 * d['seed_term_paired']:.2f} | {100 * d['combined_paired']:.2f} |"
            else:
                row += " - | - |"
        lines.append(row)
    return lines + [""]


def run(test_b1000: bool = False, extra_test_spec: str | None = None) -> dict:
    checks: list = []
    # val
    rs_val = Resampler.build(VAL_SETS, VAL_CODE, os.path.join(OUT_RUNS, "tables", f"val_b{B_VAL}"), B_VAL, SEED_VAL,
                             f"val tables (B = {B_VAL}, seed {SEED_VAL})")
    ld = ValLoader(rs_val)
    val = val_ranking(ld, checks)
    sb = same_builder(ld, checks)
    vm = {m.name: m for m in val} | sb
    steps = [vm[k] for _, k, _ in DECOMP]
    fig8 = pd.read_csv(_repo(FIG08))
    by_lab = {lab: (m, facts) for (lab, _, facts), m in zip(DECOMP, steps)}
    for _, f in fig8[fig8["kind"] == "step"].iterrows():  # source: "bootstrap_v1 <n>. <DECOMP step label>"
        m, facts = by_lab[f.source.split(". ", 1)[1]]
        v = summary(m, "eff_primary")
        digits = len(facts.split(".")[1])
        checks.append((f"fig08 step {f.x} ({facts} %)", v["value"], float(f.eff_primary),
                       abs(v["value"] - float(f.eff_primary)) <= 1e-6 and f"{100 * v['value']:.{digits}f}" == facts))
    # the R6 rows of the staircase are the ranking rows themselves (runs and configurations checked in val_ranking)
    vdiff = []
    for lab, a, b, paired, grp in VAL_DIFFS:
        new = steps[a] if isinstance(a, int) else vm[a]
        base = steps[b] if isinstance(b, int) else vm[b]
        vdiff.append((lab, grp, new, base, difference(new, base, paired_seeds=paired)))
    # test
    rs_test = Resampler.claims()
    spec = json.load(open(_repo(TEST_SPEC)))
    test = test_models(rs_test, spec, checks)
    if extra_test_spec:
        test += test_models(rs_test, json.load(open(extra_test_spec)), checks, extra=True)
    tm = {m.short: m for m in test}
    tdiff = []
    for lab, a, b, grp in TEST_DIFFS:
        tdiff.append((lab, grp, tm[a], tm[b], difference(tm[a], tm[b])))
    claims_checks({lab: d for lab, _g, _a, _b, d in tdiff}, checks)
    b1000 = None
    if test_b1000:
        rs2 = Resampler.build(BS.SETS, TS.SET_OF_CODE, os.path.join(OUT_RUNS, "tables", f"test_b{B_TEST_CHECK}"),
                              B_TEST_CHECK, SEED_TEST_CHECK, f"test check tables (B = {B_TEST_CHECK})")
        if any(not np.array_equal(rs2.units[s], rs_test.units[s]) for s in rs_test.units):
            raise ValueError("check tables with other units than the preregistered ones")
        t2 = test_models(rs2, spec, [])
        ratios = []
        for a, b in zip(test, t2):
            for q in KEYS + ("d0_100_300", "d0_300_inf"):
                ratios.append({"model": a.short, "quantity": q, "boot_sd_b500": summary(a, q)["boot_sd"],
                               "boot_sd_b1000": summary(b, q)["boot_sd"]})
        b1000 = pd.DataFrame(ratios)
        b1000["ratio"] = b1000.boot_sd_b1000 / b1000.boot_sd_b500
    return {"val": val, "same_builder": sb, "steps": steps, "val_diffs": vdiff, "test": test, "test_diffs": tdiff,
            "checks": checks, "b1000": b1000, "rs_val": rs_val, "rs_test": rs_test, "vm": vm}


def write(res: dict, out: str, name: str) -> tuple[str, str]:
    rows = model_rows("val_ranking_v3", "val", res["val"])
    rows += model_rows("val_same_builder", "val", list(res["same_builder"].values()))
    rows += model_rows("test_release_card", "test", res["test"])
    for i, ((lab, k, facts), m) in enumerate(zip(DECOMP, res["steps"])):
        r = {"table": "val_decomposition", "scope": "val", "model": m.name, "short": f"{i + 1}. {lab}",
             "quantity": "eff_primary", **summary(m, "eff_primary"), "working_points": "; ".join(m.wps)}
        rows.append(r)
    for scope, key in (("val", "val_diffs"), ("test", "test_diffs")):
        for lab, grp, new, base, d in res[key]:
            rows.append({"table": f"{scope}_differences", "scope": scope, "model": f"{new.name} MINUS {base.name}",
                         "short": lab, "group": grp, "quantity": "eff_primary", **d})
    df = pd.DataFrame(rows)
    csv = os.path.join(out, f"{name}.csv")
    df.to_csv(csv, index=False, float_format="%.10g")

    V, T = res["val"], res["test"]
    vm = res["vm"]
    main3 = [m for m in V if m.n_seeds == 3]
    single = [m for m in V if m.n_seeds == 1]
    L = [f"# Event-bootstrap uncertainties of the headline numbers ({name})", "",
         "Supervisor feedback round 1, point 5 (`python -m hnl.eval.bootstrap_tables`; module docstring for the "
         "method). Validation numbers (val) and post-freeze descriptive summaries of the one-time test outputs (test; "
         "not claims: results/claims_test_v1.json stays as it is). Nothing is chosen here: every working point is "
         "read from the committed val files.", "",
         "- **value**: mean of the seeds' numbers (the ranking tables' estimator); **seed sd**: sd of the seeds' "
         "numbers (ddof 1), the +- of the ranking tables; **boot sd**: event-bootstrap sd of the seed mean (events "
         "resampled with replacement, the same resampled events for every seed and model); **combined**: "
         "sqrt(boot sd^2 + seed sd^2 / n_seeds), the total sigma of hnl.eval.claims.",
         f"- Val: pooled HNL val (hnl10_100 + hnl15_200 val, 1,999 events, resampled by event index on 1,000 units) "
         f"and ttbar val (1,000 events, ttbar-prompt column), B = {res['rs_val'].b} (new tables, seed {SEED_VAL}). "
         f"Test: pooled HNL test sets (hnl_joint_test by event index + hnl15_100_external, 6,999 events) and "
         f"ttbar_test (1,000 events) with the preregistered tables of the claims (B = {res['rs_test'].b}, seed "
         f"{BS.SEED}).",
         "- 68 % CP: Clopper-Pearson on the per-seed particle count (k = value x n per seed; paper Fig. 7 "
         "convention). 68 % bootstrap: 15.87 / 84.13 % percentiles of the replicates of the seed mean.",
         "- Single-seed rows (DeZoort, ladder, previews): no seed term (as in claims.py), so their combined sigma "
         "is the bootstrap sd alone and understates the total.", ""]
    L += ["- Labels: CC = connected components at the preregistered cut rule, CC* = at the best cut <= FR* chosen "
          "on val; MWS+M = MWS-layer + merging, WT+M = walkthrough + merging; Canto-L / -H / -B = the full-budget "
          "models at their rule-R6 builder (Canto-L: learned merging, selected by R8; Canto-H: WT+M; Canto-B: "
          "learned merging).", "",
          "## 0. Headline numbers (primary efficiency)", "",
          "| model | val value | val seed sd | val boot sd | val combined | test value | test seed sd | test boot sd | "
          "test combined |", "|---|---|---|---|---|---|---|---|---|"]
    tby = {m.name: m for m in T}
    for lab in (R6["l1tl"], R6["l1tu"], R6["l1t"], REF_HNL_MWS, REF_HNL_CC,
                "acorn IGNN, HNL-trained + CC (prereg. cut rule)", "acorn IGNN, ttbar-trained + CC (best cut <= FR*)"):
        v = summary(vm[lab], "eff_primary")
        cells = [f4(v["value"]), f4(v["seed_sd"]), f4(v["boot_sd"]), f4(v["combined"])]
        if lab in tby:
            t = summary(tby[lab], "eff_primary")
            cells += [f4(t["value"]), f4(t["seed_sd"]), f4(t["boot_sd"]), f4(t["combined"])]
        else:
            cells += ["no test row"] + [""] * 3
        label = vm[lab].short + (" (= CC)" if lab.startswith("acorn IGNN, ttbar-trained") else "")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "Paired differences (decomposition steps, like-for-like comparisons) are in section 3.", ""]
    L += ["## 1. Validation ranking v3", ""]
    L += md_primary(main3 + single, "Primary efficiency (pooled HNL val; FR* = 0.029368)")
    L += md_other(main3 + single, "Other columns (val)")
    L += md_kn(main3 + [m for m in single if m.short.startswith("DeZoort")], "|d0| columns, HNL-daughter targets "
               "(val; k/n pooled over seeds)")
    L += ["## 2. Test release card (ranking_test_v1; post-freeze descriptive)", ""]
    L += md_primary(T, "Primary efficiency (pooled HNL test sets)")
    L += md_other(T, "Other columns (test)")
    L += md_kn(T, "|d0| columns, HNL-daughter targets (test; k/n pooled over seeds)")
    L += ["## 3. Component decomposition (val; paper Fig. 8)", "",
          "| step | value | seed sd | boot sd | combined | working points (seeds 0 / 1 / 2) |",
          "|---|---|---|---|---|---|"]
    for i, ((lab, k, facts), m) in enumerate(zip(DECOMP, res["steps"])):
        s = summary(m, "eff_primary")
        L.append(f"| {i + 1}. {lab} | {100 * s['value']:.2f} % | {100 * s['seed_sd']:.2f} | {100 * s['boot_sd']:.2f} | "
                 f"{100 * s['combined']:.2f} | {'; '.join(w.split(' (')[0] for w in m.wps)} |")
    for k in ("Canto-L MWS+M", REF_HNL_CC, REF_HNL_MWS):
        m = vm[k]
        s = summary(m, "eff_primary")
        L.append(f"| reference: {m.short} | {100 * s['value']:.2f} % | {100 * s['seed_sd']:.2f} | "
                 f"{100 * s['boot_sd']:.2f} | {100 * s['combined']:.2f} | "
                 f"{'; '.join(w.split(' (')[0] for w in m.wps)} |")
    L += ["", "Percent; uncertainties in percentage points.", ""]
    L += md_diffs([(lab, d) for lab, _g, _n, _b, d in res["val_diffs"]], "Paired differences (val)", True)
    L += md_diffs([(lab, d) for lab, _g, _n, _b, d in res["test_diffs"]],
                  "Paired differences (test; post-freeze descriptive)", False)
    L += ["## 4. |d0| bins (HNL-daughter targets; preregistered bins)", ""]
    L += md_bins([vm[k] for k in FIG7_VAL], "Validation (paper Fig. 7 models and acorn (HNL) + MWS-layer + merging)")
    L += md_bins([m for m in T if m.short != "acorn (ttbar) CC*"], "Test (release-card models; acorn (ttbar) CC* "
                 "= acorn (ttbar) CC)")
    L += ["The r_v bins (and the \"no vertex\" category) of every row are in the CSV (quantities rv_*).", ""]
    L += ["## 5. Checks", ""]
    bad = [c for c in res["checks"] if not c[3]]
    L.append(f"- {len(res['checks'])} identity checks (val ranking v3 entries and rows, final_selection_v1_seeds, "
             f"fig08 rows, test ranking entries and rows, claims differences and sigmas): {len(bad)} failed.")
    for c in res["checks"]:
        if c[0].startswith("claims") or c[0].startswith("fig08"):
            L.append(f"  - {'ok' if c[3] else 'FAIL'} {c[0]}: {c[1]:.16g} (here) vs {c[2]:.16g}")
    for c in bad:
        L.append(f"  - FAIL {c[0]}: {c[1]:.16g} vs {c[2]:.16g}")
    if res["b1000"] is not None:
        r = res["b1000"].ratio
        L.append(f"- Monte Carlo precision of B = 500 (test): with fresh B = {B_TEST_CHECK} tables (seed "
                 f"{SEED_TEST_CHECK}) the boot sd of the {len(r)} test cells (7 rows x efficiencies, fake rate, |d0| "
                 f"columns) changes by a factor {r.min():.3f}-{r.max():.3f} (median {r.median():.3f}).")
    L += ["- The 68 % bootstrap percentile interval collapses to a point when every particle of a population is "
          "found (k = n, e.g. the |d0| < 1 mm bins of Canto): quote the Clopper-Pearson interval there.",
          "- Clopper-Pearson treats the particles as independent; for populations clustered in events (ttbar prompt: "
          "about 53k particles per seed in 1,000 val events; the two HNL daughters of one decay) the event bootstrap "
          "is wider and is the one to quote (boot sd / combined)."]
    L += ["", "Caveat: on val, the working points (CC best cut, builder rule with the tie rule) were chosen on the "
          "same val events; the bootstrap holds them fixed and does not resample that choice, so val values of the "
          "best-cut and builder-rule rows carry an optimistic selection bias that the test rows do not have.", ""]
    md = os.path.join(out, f"{name}.md")
    with open(md, "w") as f:
        f.write("\n".join(L))
    return md, csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="bootstrap_v1")
    ap.add_argument("--test-b1000", action="store_true", help="also the test boot sd with fresh B = 1000 tables")
    ap.add_argument("--extra-test-spec", default=None, help="more test models (spec_test_v1.json format)")
    ap.add_argument("--dump", default=None, help="pickle of the per-seed points and replicates (for later pairs)")
    a = ap.parse_args(argv)
    res = run(a.test_b1000, a.extra_test_spec)
    os.makedirs(_repo(a.out), exist_ok=True)
    md, csv = write(res, _repo(a.out), a.name)
    if a.dump:
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
        keep = {scope: {m.name: {"short": m.short, "runs": m.runs, "wps": m.wps, "pts": m.pts, "reps": m.reps,
                                 "num": m.num, "den": m.den} for m in ms}
                for scope, ms in (("val", res["val"] + list(res["same_builder"].values())), ("test", res["test"]))}
        with open(a.dump, "wb") as f:
            pickle.dump(keep, f)
    bad = [c for c in res["checks"] if not c[3]]
    print(open(md).read())
    print(md, csv)
    if bad:
        raise SystemExit(f"{len(bad)} identity checks failed")


if __name__ == "__main__":
    main()
