"""Test-set statistics with the preregistered bootstrap (docs/preregistration.md §5, §6, §7).

A "model" is a list of runs (one per training seed), each with eval_test outputs (hnl.eval.evaluate_gnn on the test
splits and the external set) and the cuts fixed on val for that seed (the FR* cut for track-level numbers, the e* cut
for edge-level numbers). For a population (a mask over the particle truth table) and a choice of test sets, every seed
gives a point efficiency and B bootstrap replicates (shared resampling tables, so differences between models are
paired); seeds are combined with hnl.eval.bootstrap.

Sets: hnl_joint_test (hnl10_100_test + hnl15_200_test, resampled by event index), ttbar_test, hnl15_100_external;
the two halves of the joint set are also reported separately (hnl10_100_test, hnl15_200_test) on the joint resampling
units, so they stay paired.

    python -m hnl.eval.test_stats --runs RUN_ROOT --wp WORKING_POINTS_JSON --out OUT
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval.headroom import BINS as HEADROOM_BINS
from hnl.eval.particles import particle_table, split_event_id

SET_OF_CODE = {1: "ttbar_test", 2: "hnl_joint_test", 4: "hnl_joint_test", 3: "hnl15_100_external"}
SUBSETS = {"hnl10_100_test": ("hnl_joint_test", 2), "hnl15_200_test": ("hnl_joint_test", 4)}
HNL_SETS = ("hnl_joint_test", "hnl15_100_external")
PAPER_PROMPT_DENOMINATOR = 4744          # W muons with >= 5 hits on the external set (definitions.md §4)
NAN_BIN = {"rv": "no vertex", "pair_angle": "no pair"}    # NaN categories reported as their own bin


def expected_events() -> dict:
    """Events per test set from the split lists (dropped events excluded)."""
    from hnl.data.splits import read_split
    return {"hnl_joint_test": len(read_split("hnl10_100", "test")) + len(read_split("hnl15_200", "test")),
            "ttbar_test": len(read_split("ttbar", "test")),
            "hnl15_100_external": len(read_split("hnl15_100", "external"))}


def _with_cut(df: pd.DataFrame) -> pd.DataFrame:
    """evaluate_gnn tables carry 'cut'; trackbuild_val tables 'config' (used as the cut)."""
    if "cut" not in df.columns and "config" in df.columns:
        df = df.assign(cut=df.config.astype(np.float64))
    return df


class RunTest:
    """One trained seed at its fixed cuts on the test sets."""

    def __init__(self, run: str, cut: float, cut_e: float | None = None, check_events: bool = True,
                 sub: str = "eval_test", sub_e: str | None = None):
        """Track level from RUN/sub at `cut` (evaluate_gnn tables; for a track-building table of
        hnl.eval.trackbuild_val --config the configuration index plays the cut, normally 0); edge level from
        RUN/sub_e (default: sub) at the classifier's e* cut."""
        d = os.path.join(run, sub)
        pp = _with_cut(pd.read_parquet(os.path.join(d, "per_particle.parquet")))
        pe = _with_cut(pd.read_parquet(os.path.join(d, "per_event.parquet")))
        self.pp = pp[np.isclose(pp.cut, cut, atol=1e-6)].reset_index(drop=True)
        self.pe = pe[np.isclose(pe.cut, cut, atol=1e-6)].reset_index(drop=True)
        if self.pp.empty or self.pe.empty:
            raise ValueError(f"cut {cut} not evaluated in {d}")
        self.pe_e = None
        if cut_e is not None:
            de = os.path.join(run, sub_e or sub)
            pee = pe if de == d else _with_cut(pd.read_parquet(os.path.join(de, "per_event.parquet")))
            self.pe_e = pee[np.isclose(pee.cut, cut_e, atol=1e-6)].reset_index(drop=True)
            if self.pe_e.empty:
                raise ValueError(f"e* cut {cut_e} not evaluated in {de}")
        if check_events:
            names, _ = _units(self.pe.event_id.to_numpy())
            for s, n in expected_events().items():
                got = int((names == s).sum())
                if got != n:
                    raise ValueError(f"{d}: {got} events of {s} at cut {cut}, expected {n}")
        self.run, self.cut, self.cut_e = run, cut, cut_e


def _units(event_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code, idx = split_event_id(event_id)
    return np.array([SET_OF_CODE[int(c)] for c in code]), idx


def in_sets(event_id, sets) -> np.ndarray:
    """Rows whose event belongs to any of `sets` (set names or SUBSETS)."""
    event_id = np.asarray(event_id)
    names, _ = _units(event_id)
    code = split_event_id(event_id)[0]
    m = np.zeros(len(event_id), dtype=bool)
    for s in sets:
        if s in SUBSETS:
            parent, c = SUBSETS[s]
            m |= (names == parent) & (code == c)
        else:
            m |= names == s
    return m


def ratio_per_set(bs: BS.Bootstrap, event_id, num, den, sets) -> tuple[float, np.ndarray]:
    event_id = np.asarray(event_id)
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    names, idx = _units(event_id)
    code = split_event_id(event_id)[0]
    parts = []
    for s in sets:
        if s in SUBSETS:
            parent, c = SUBSETS[s]
            m = (names == parent) & (code == c)
        else:
            parent, m = s, names == s
        if m.any():
            parts.append((parent, bs.per_unit(parent, idx[m], num[m]), bs.per_unit(parent, idx[m], den[m])))
    return bs.ratio(parts)


def efficiency(bs, runs: list[RunTest], info: pd.DataFrame, mask_fn, sets) -> tuple[list, list, int]:
    """Per seed point and replicates; also the population size in the requested sets (first seed)."""
    pts, reps, n = [], [], None
    for rt in runs:
        t = rt.pp.merge(info, on=["event_id", "particle_id"], how="inner")
        m = np.asarray(mask_fn(t), dtype=bool) & in_sets(t.event_id, sets)
        p, r = ratio_per_set(bs, t.event_id[m], t.dm_matched[m].astype(float), np.ones(int(m.sum())), sets)
        pts.append(p)
        reps.append(r)
        n = int(m.sum()) if n is None else n
    return pts, reps, n


def event_ratio(bs, runs: list[RunTest], num_col: str, den_col: str, sets, frame: str = "pe",
                den_minus: str | None = None, den_plus: str | None = None) -> tuple[list, list]:
    """sum(num) / sum(den) over events (den = den_col [- den_minus] [+ den_plus]) at the run's cut in `frame`."""
    pts, reps = [], []
    for rt in runs:
        e = getattr(rt, frame)
        den = e[den_col].astype(float)
        if den_minus is not None:
            den = den - e[den_minus].astype(float)
        if den_plus is not None:
            den = den + e[den_plus].astype(float)
        m = in_sets(e.event_id, sets)
        p, r = ratio_per_set(bs, e.event_id[m], e[num_col].astype(float)[m], den[m], sets)
        pts.append(p)
        reps.append(r)
    return pts, reps


def masks():
    def disp(t):
        return ((t.d0.abs() >= 10) | (t.rv >= 100)).fillna(False)
    return {
        "primary": lambda t: t.is_target & (t.category == "hnl") & disp(t),
        "hnl_targets": lambda t: t.is_target & (t.category == "hnl"),
        "w_muons": lambda t: t.is_target & (t.category == "w"),
        "all_targets": lambda t: t.is_target,
        "ttbar_prompt": lambda t: t.is_target & (t.d0.abs() < 1.0).fillna(False),
        "hnl_mu_plus": lambda t: t.is_target & (t.category == "hnl") & (t.q > 0),
        "hnl_mu_minus": lambda t: t.is_target & (t.category == "hnl") & (t.q < 0),
        "hnl_targets_pix3": lambda t: t.is_target & (t.category == "hnl") & (t.n_pixel_hits >= 3),
        "w_muons_pix3": lambda t: t.is_target & (t.category == "w") & (t.n_pixel_hits >= 3),
        # paper comparison only (definitions.md §4): W muons with >= 5 hits, no pT / |eta| requirement
        "w_muons_paper": lambda t: (t.category == "w") & (t.n_hits >= 5),
    }


def _entry(p, r, n):
    return {**BS.seed_mean(p, r), "per_seed": p, "n": n}


def model_summary(bs, runs: list[RunTest], info) -> dict:
    M = masks()
    out = {}
    plan = [("primary", HNL_SETS), ("primary", ("hnl_joint_test",)), ("primary", ("hnl15_100_external",)),
            ("primary", ("hnl10_100_test",)), ("primary", ("hnl15_200_test",)),
            ("hnl_targets", HNL_SETS), ("w_muons", HNL_SETS), ("ttbar_prompt", ("ttbar_test",)),
            ("all_targets", ("ttbar_test",)), ("all_targets", ("hnl_joint_test",)),
            ("all_targets", ("hnl15_100_external",)), ("w_muons", ("hnl15_100_external",)),
            ("w_muons", ("hnl10_100_test",)), ("w_muons", ("hnl15_200_test",)),
            ("hnl_targets", ("hnl15_100_external",)), ("hnl_mu_plus", ("hnl15_100_external",)),
            ("hnl_mu_minus", ("hnl15_100_external",)), ("hnl_targets_pix3", HNL_SETS), ("w_muons_pix3", HNL_SETS)]
    for pop, sets in plan:
        p, r, n = efficiency(bs, runs, info, M[pop], sets)
        out[f"eff_{pop}@{'+'.join(sets)}"] = _entry(p, r, n)
    for sets in (("hnl_joint_test",), ("hnl15_100_external",), ("ttbar_test",), HNL_SETS):
        p, r = event_ratio(bs, runs, "n_fake", "n_tracks_ge5", sets)
        out[f"fake_rate@{'+'.join(sets)}"] = _entry(p, r, None)
    # edge level at each seed's e* cut (preregistration §4)
    if all(rt.pe_e is not None for rt in runs):
        for sets in (HNL_SETS, ("ttbar_test",)):
            tag = "+".join(sets)
            p, r = event_ratio(bs, runs, "tp", "n_true", sets, frame="pe_e")
            out[f"edge_eff@{tag}@e_star"] = _entry(p, r, None)
            p, r = event_ratio(bs, runs, "fp", "n_edges", sets, frame="pe_e", den_minus="n_true_in_graph")
            out[f"edge_fpr@{tag}@e_star"] = _entry(p, r, None)
            p, r = event_ratio(bs, runs, "tp", "tp", sets, frame="pe_e", den_plus="fp")
            out[f"edge_purity@{tag}@e_star"] = _entry(p, r, None)
            if "tp_mu" in runs[0].pe_e:
                p, r = event_ratio(bs, runs, "tp_mu", "n_true_mu", sets, frame="pe_e")
                out[f"edge_eff_mu@{tag}@e_star"] = _entry(p, r, None)
                p, r = event_ratio(bs, runs, "tp_mu_clean", "n_true_mu_clean", sets, frame="pe_e")
                out[f"edge_eff_mu_clean@{tag}@e_star"] = _entry(p, r, None)
    return out


def paper_summary(bs, runs: list[RunTest], info, expected_w: int | None = PAPER_PROMPT_DENOMINATOR) -> dict:
    """The paper comparison (preregistration §7, definitions.md §4) on the external set at s*."""
    M = masks()
    ext = ("hnl15_100_external",)
    out = {}
    for pop in ("all_targets", "w_muons_paper", "hnl_targets", "hnl_mu_plus", "hnl_mu_minus"):
        p, r, n = efficiency(bs, runs, info, M[pop], ext)
        out[f"eff_{pop}@hnl15_100_external"] = _entry(p, r, n)
    n_w = out["eff_w_muons_paper@hnl15_100_external"]["n"]
    if expected_w is not None and n_w != expected_w:
        raise ValueError(f"paper prompt denominator {n_w} != {expected_w} (definitions.md §4)")
    p, r = event_ratio(bs, runs, "n_fake", "n_tracks_ge5", ext)
    out["fake_rate@hnl15_100_external"] = _entry(p, r, None)
    return out


def binned_efficiency(bs, runs: list[RunTest], info, base_mask, var: str, edges, sets) -> pd.DataFrame:
    """Seed-mean efficiency per bin of `var` ("absd0", "rv", "pt", "r_first", ...) with the paired bootstrap
    uncertainty; for r_v also the preregistered "no vertex" category and for the opening angle a "no pair" category
    (last row, labelled)."""
    rows = []

    def x_of(t):
        return t.d0.abs() if var == "absd0" else t[var]
    bins = list(zip(edges[:-1], edges[1:]))
    if var in NAN_BIN:
        bins.append((np.nan, np.nan))
    for lo, hi in bins:
        def mask(t, lo=lo, hi=hi):
            x = x_of(t)
            if np.isnan(lo):
                return base_mask(t) & x.isna()
            return base_mask(t) & (x >= lo) & (x < hi)
        p, r, n = efficiency(bs, runs, info, mask, sets)
        m = BS.seed_mean(p, r)
        rows.append({"lo": lo, "hi": hi, "label": NAN_BIN[var] if np.isnan(lo) else "", "n": n,
                     "mean": m["mean"], "sigma": m["sigma"]})
    return pd.DataFrame(rows)


def load_model(runs_root: str, arm: str, cuts: dict, cuts_e: dict | None = None) -> list[RunTest]:
    return [RunTest(os.path.join(runs_root, f"arm{arm}_s{s}"), float(c),
                    None if cuts_e is None else float(cuts_e[s])) for s, c in sorted(cuts.items())]


def _write_summary(res: dict, out: str) -> pd.DataFrame:
    with open(os.path.join(out, "test_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    rows = []
    for model, d in res.items():
        for k, v in d.items():
            rows.append({"model": model, "quantity": k, "mean": v["mean"], "sigma": v["sigma"],
                         "sigma_boot": v["sigma_boot"], "s_seeds": v["s_seeds"], "n": v.get("n")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "test_summary.csv"), index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--wp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    wp = json.load(open(a.wp))
    bs = BS.Bootstrap()
    models = {arm: load_model(a.runs, arm, cuts, wp.get("cuts_at_e_star", {}).get(arm))
              for arm, cuts in wp["cuts_at_FR_star"].items()}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for runs in models.values() for rt in runs]))
    info = particle_table(ev)
    res = {arm: model_summary(bs, runs, info) for arm, runs in models.items()}
    paper_runs = None
    if "ii" in wp["s_star"]:
        s2 = wp["s_star"]["ii"]
        paper_runs = load_model(a.runs, "ii", {s: s2 for s in wp["cuts_at_FR_star"]["ii"]})
        res["paper_comparison_arm_ii_s_star"] = paper_summary(bs, paper_runs, info)
    df = _write_summary(res, a.out)              # written before the binned tables (a later failure keeps it)
    M = masks()
    binned = []
    for arm, runs in models.items():
        specs = [("absd0", [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]), ("rv", [0, 10, 30, 100, 200, 300, np.inf]),
                 ("pt", [1, 2, 3, 5, 10, 20, 50, np.inf])]
        specs += [(v, HEADROOM_BINS[v][1]) for v in ("r_first", "n_pixel_hits", "pair_angle")]
        for var, edges in specs:
            for pop in ("hnl_targets", "primary"):
                b = binned_efficiency(bs, runs, info, M[pop], var, edges, HNL_SETS)
                b["model"], b["population"], b["variable"] = arm, pop, var
                binned.append(b)
        b = binned_efficiency(bs, runs, info, M["w_muons"], "pt", [1, 10, 20, 30, 40, 50, 70, np.inf], HNL_SETS)
        b["model"], b["population"], b["variable"] = arm, "w_muons", "pt"
        binned.append(b)
    if paper_runs is not None:
        # paper binning (Fig. 3 |d0| panel, 50 mm bins) on the external set at s*
        b = binned_efficiency(bs, paper_runs, info, M["hnl_targets"], "absd0", list(range(0, 551, 50)) + [np.inf],
                              ("hnl15_100_external",))
        b["model"], b["population"], b["variable"] = "paper_comparison_arm_ii_s_star", "hnl_targets", "absd0_paper"
        binned.append(b)
        # the paper's "|d0|" spectrum resembles our r_v (docs/paper_facts.md): the same 50 mm bins in r_v
        b = binned_efficiency(bs, paper_runs, info, M["hnl_targets"], "rv", list(range(0, 551, 50)) + [np.inf],
                              ("hnl15_100_external",))
        b["model"], b["population"], b["variable"] = "paper_comparison_arm_ii_s_star", "hnl_targets", "rv_paper"
        binned.append(b)
    pd.concat(binned, ignore_index=True).to_csv(os.path.join(a.out, "test_binned.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(5).to_string())


if __name__ == "__main__":
    main()
