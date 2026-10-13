"""Training-sample study on test (post-freeze, descriptive; docs/feedback/test_use_ledger.md R11): the differences of
hnl.eval.training_sample_paired between models trained on different samples, on the test sets, with the builder
family mwslmerge (LM; MWS-layer + learned merging) on every side.

Models (test entries fixed before the test, nothing chosen here):
  Canto-L (ttbar)                 the frozen Canto-L, three seeds (results/strong_baselines_v3_spec_test.json);
  Canto-L (HNL), (ttbar+HNL)      the one-seed runs of R11 (results/training_sample_spec_test_v1.json);
  acorn (ttbar), (HNL), (ttbar+HNL)   acorn + MWS-layer + learned merging, chosen per seed on val
                                  (results/strong_baselines_v3_spec_test.json).
Estimator: the claims' (hnl.eval.bootstrap.linear_combination, paired event bootstrap of the preregistered test tables,
B = 500, seed terms c^2 s^2 / n); a one-seed model adds c^2 s1^2 with s1 the seed sd of Canto-L (ttbar) for the same
quantity (paper Sec. II). Per model: value, bootstrap sd, seed term and total sigma (one-seed models: bootstrap and
s1 in quadrature). Checks: the per-model values against results/bootstrap_ts_test_v1.csv (Canto-L (HNL),
(ttbar+HNL)) and results/strong_baselines_v3.csv / _differences.csv (Canto-L (ttbar), acorn).

    python -m hnl.eval.training_sample_paired_test --out results/training_sample_paired_test_v1.csv

The rounded summary that the paper quotes goes next to the CSV (same name, .md).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import strong_baselines as SB
from hnl.eval import test_stats as TS
from hnl.eval.particles import particle_table

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SB3_SPEC = "results/strong_baselines_v3_spec_test.json"
TS_SPEC = "results/training_sample_spec_test_v1.json"
LABELS = {  # name here -> label in its spec file
    "Canto-L (ttbar)": (SB3_SPEC, "Canto-L (frozen: learned merging)"),
    "Canto-L (HNL)": (TS_SPEC, "Canto-L (HNL): learned triplet union, learned merging (training-sample study, one seed, "
                               "post-freeze)"),
    "Canto-L (ttbar+HNL)": (TS_SPEC, "Canto-L (ttbar+HNL): learned triplet union, learned merging (training-sample "
                                     "study, one seed, post-freeze)"),
    "acorn (ttbar)": (SB3_SPEC, "acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder)"),
    "acorn (HNL)": (SB3_SPEC, "acorn (HNL) + MWS-layer + learned merging (Canto-L's builder)"),
    "acorn (ttbar+HNL)": (SB3_SPEC, "acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder)"),
}
SAMPLES = ("ttbar", "ttbar+HNL", "HNL")
ARCHS = ("acorn", "Canto-L")
TOL = 1e-9


def _repo(p: str) -> str:
    return os.path.join(REPO, p)


def entries(name: str) -> list[dict]:
    f, lab = LABELS[name]
    s = json.load(open(_repo(f)))
    ms = s["models"] if isinstance(s, dict) else s
    return next(m for m in ms if m["label"] == lab)["entries"]


def values() -> tuple[dict, dict]:
    """(quantity, model) -> (per-seed points, replicates); model -> population size per seed (efficiencies)."""
    bs = BS.Bootstrap()
    models = {n: [TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"]) for e in entries(n)] for n in LABELS}
    sizes = {}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for rr in models.values() for rt in rr]))
    info = particle_table(ev)
    out = {}
    for q, kind, mask, sets in SB.QUANTITIES:
        for n, rr in models.items():
            if kind == "eff":
                p, r, sizes[(q, n)] = TS.efficiency(bs, rr, info, mask(), sets)
            else:
                p, r = TS.event_ratio(bs, rr, "n_fake", "n_tracks_ge5", sets)
            out[(q, n)] = (list(p), list(r))
    return out, sizes


def combination(terms: list, s1: float) -> dict:
    """sum c * model over [(c, (points, reps))], paired over the events; one-seed models add c^2 s1^2."""
    d = BS.linear_combination([(c, *pr) for c, pr in terms])
    extra = sum(c ** 2 * s1 ** 2 for c, pr in terms if len(pr[0]) == 1)
    seeds = float(np.sqrt(d["sigma_seeds"] ** 2 + extra))
    sigma = float(np.hypot(d["sigma_boot"], seeds))
    return {"value": d["value"], "sigma_boot": d["sigma_boot"], "sigma_seeds": seeds, "sigma": sigma,
            "z": d["value"] / sigma if sigma > 0 else np.nan}


def rows(vals: dict) -> list[dict]:
    out = []
    for q, *_ in SB.QUANTITIES:
        g = {n: vals[(q, n)] for n in LABELS}
        s1 = float(np.std(g["Canto-L (ttbar)"][0], ddof=1))
        for n, pr in g.items():
            r = combination([(1.0, pr)], s1)
            out.append({"quantity": q, "kind": "model", "label": n, "n_seeds": len(pr[0]), "seed_sd_one_seed": s1,
                        "terms": n} | r)

        def add(kind, label, terms):
            r = combination([(c, g[n]) for c, n in terms], s1)
            out.append({"quantity": q, "kind": kind, "label": label, "n_seeds": np.nan, "seed_sd_one_seed": s1,
                        "terms": "; ".join(f"{c:+g} {n} ({len(g[n][0])} seeds)" for c, n in terms)} | r)

        pairs = (("HNL", "ttbar"), ("ttbar+HNL", "ttbar"), ("ttbar+HNL", "HNL"))
        for a in ARCHS:
            for s, b in pairs:
                add("per architecture", f"{a}, trained on {s} minus trained on {b}",
                    [(1.0, f"{a} ({s})"), (-1.0, f"{a} ({b})")])
        for s in SAMPLES:
            add("per sample", f"trained on {s}: Canto-L minus acorn",
                [(1.0, f"Canto-L ({s})"), (-1.0, f"acorn ({s})")])
        for s, b in pairs:
            add("gain difference", f"{s} minus {b}: acorn's change minus Canto-L's change",
                [(1.0, f"acorn ({s})"), (-1.0, f"acorn ({b})"), (-1.0, f"Canto-L ({s})"), (1.0, f"Canto-L ({b})")])
    return out


def checks(df: pd.DataFrame) -> list[tuple[str, float, float, bool]]:
    res = []
    m = df[df.kind == "model"].set_index(["quantity", "label"])
    bt = pd.read_csv(_repo("results/bootstrap_ts_test_v1.csv"))
    bt = bt[bt.table == "test_release_card"]
    for n in ("Canto-L (HNL)", "Canto-L (ttbar+HNL)"):
        for q, qb in (("primary", "eff_primary"), ("hnl_targets", "eff_hnl_targets"), ("w_muons", "eff_w_muons")):
            ref = bt[(bt.short == n) & (bt.quantity == qb)]
            for col, rc in (("value", "value"), ("sigma_boot", "boot_sd")):
                v, mine = float(ref[rc].iloc[0]), float(m.loc[(q, n), col])
                res.append((f"{n} {q} {col} vs bootstrap_ts_test_v1", mine, v, abs(mine - v) <= TOL))
    d = df[df.kind == "per sample"].set_index(["quantity", "label"])
    d3 = pd.read_csv(_repo("results/strong_baselines_v3_differences.csv"))
    d3 = d3[(d3.set == "test") & (d3.new == LABELS["Canto-L (ttbar)"][1])]
    for s in SAMPLES:
        for r in d3[d3.baseline == LABELS[f"acorn ({s})"][1]].itertuples():
            for mine, ref, what in ((m.loc[(r.quantity, f"acorn ({s})"), "value"], r.value_base, "acorn value"),
                                    (m.loc[(r.quantity, "Canto-L (ttbar)"), "value"], r.value_new, "Canto-L value")):
                res.append((f"{what} {s} {r.quantity} vs strong_baselines_v3", mine, ref, abs(mine - ref) <= TOL))
            if s == "ttbar":
                x = d.loc[(r.quantity, "trained on ttbar: Canto-L minus acorn")]
                for col in ("value", "sigma"):
                    ref = r.difference if col == "value" else r.sigma
                    res.append((f"Canto-L - acorn (ttbar) {r.quantity} {col} vs strong_baselines_v3", x[col], ref,
                                abs(x[col] - ref) <= TOL))
    return res


ORDER = ("acorn (ttbar)", "acorn (ttbar+HNL)", "acorn (HNL)", "Canto-L (ttbar)", "Canto-L (ttbar+HNL)", "Canto-L (HNL)")
QLAB = {"primary": "", "hnl_d0_100_300": ", |d0| 100-300 mm", "fake_rate": ", fake rate"}


def markdown(df: pd.DataFrame, vals: dict, sizes: dict) -> str:
    m = df[df.kind == "model"].set_index(["quantity", "label"])
    pm = lambda q, n: f"{100 * m.loc[(q, n), 'value']:.2f} +- {100 * m.loc[(q, n), 'sigma']:.2f}"

    def kn(q, n):
        pts, size = vals[(q, n)][0], sizes[(q, n)]
        return f"{sum(int(round(p * size)) for p in pts)}/{size * len(pts)}"

    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    L = ["# Training-sample study on test: Canto-L and acorn trained on ttbar, on ttbar + HNL and on HNL", "",
         f"Generated by `python -m hnl.eval.training_sample_paired_test` (git {git.strip()}). Test sets, post-freeze, "
         "descriptive (docs/feedback/test_use_ledger.md R11): Canto-L trained on HNL and on ttbar + HNL, one seed "
         "each, at the configurations committed before their test jobs; Canto-L (ttbar) is the frozen Canto-L; acorn: "
         "MWS-layer + learned merging chosen per seed on val (results/strong_baselines_v3_spec_test.json). Values in "
         "percent, +- the total sigma: event bootstrap (preregistered test tables, B = 500) and seed term; a one-seed "
         "model takes the seed sd of Canto-L (ttbar) in the same row as its seed term (paper Sec. II). Primary: the "
         "pooled HNL test sets (hnl10_100_test, hnl15_200_test, hnl15_100_external).", "",
         "## MWS-layer + learned merging (LM), test", "",
         "| model | seeds | primary (%) | inefficiency (%) | 100-300 mm (%) | >= 300 mm | W muons (%) | ttbar prompt (%) "
         "| fake rate (%) |", "|---|---|---|---|---|---|---|---|---|"]
    for n in ORDER:
        L.append(f"| {n} | {int(m.loc[('primary', n), 'n_seeds'])} | {pm('primary', n)} | "
                 f"{100 * (1 - m.loc[('primary', n), 'value']):.2f} | {pm('hnl_d0_100_300', n)} | "
                 f"{kn('hnl_d0_ge300', n)} | {pm('w_muons', n)} | {pm('ttbar_prompt', n)} | {pm('fake_rate', n)} |")
    s1 = 100 * m.loc[("primary", "Canto-L (ttbar)"), "seed_sd_one_seed"]
    L += ["", f"- test, mwslmerge: seed spread (sd) of Canto-L (ttbar) primary efficiency {s1:.2f} points"]
    for s in SAMPLES:
        a = 100 * (1 - m.loc[("primary", f"acorn ({s})"), "value"])
        c = 100 * (1 - m.loc[("primary", f"Canto-L ({s})"), "value"])
        L.append(f"- test, mwslmerge: trained on {s}: inefficiency acorn {a:.2f} %, Canto-L {c:.2f} %, ratio "
                 f"acorn/Canto-L {a / c:.1f}")
    for q, ql in QLAB.items():
        for r in df[(df.quantity == q) & (df.kind != "model")].itertuples():
            L.append(f"- paired, test, mwslmerge{ql}: {r.label}: {100 * r.value:+.2f} +- {100 * r.sigma:.2f} points "
                     f"(boot {100 * r.sigma_boot:.2f}, seeds {100 * r.sigma_seeds:.2f}; z = {r.z:.1f})")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    vals, sizes = values()
    df = pd.DataFrame(rows(vals))[["quantity", "kind", "label", "n_seeds", "value", "sigma_boot", "sigma_seeds",
                                   "sigma", "z", "seed_sd_one_seed", "terms"]]
    ch = checks(df)
    bad = [c for c in ch if not c[3]]
    print(f"{len(ch)} checks, {len(bad)} failed")
    for c in bad:
        print("  FAILED", c)
    if not ch or bad:
        raise SystemExit("checks failed: nothing written")
    df.to_csv(a.out, index=False)
    md = os.path.splitext(a.out)[0] + ".md"
    open(md, "w").write(markdown(df, vals, sizes))
    print(f"wrote {a.out} and {md}")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(df[df.quantity.isin(["primary", "hnl_d0_100_300", "fake_rate"])].drop(columns="terms")
              .to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
