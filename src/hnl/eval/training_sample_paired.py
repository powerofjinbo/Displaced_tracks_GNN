"""Training-sample study (docs/training_sample_study.md), val only: paired event-bootstrap uncertainties of the
differences between models trained on different samples (paper Sec. V A, with the estimator of Eq. (sigma): the
bootstrap term of a difference is that of the paired difference, the same resampled events for every seed and model).

Rows: every seed of results/training_sample_val_<family>.csv at the run and working point chosen there by
hnl.eval.training_sample_board (each seed's primary efficiency reproduced to 1e-9), for the builder families
mwslmerge (LM) and mwslayermerge (MWS+M). Replicate tables: the val tables of hnl.eval.bootstrap_tables (B = 1000,
seed 20261005), as results/strong_baselines_v3_val. Seed terms: c^2 s^2 / n for each three-seed model
(hnl.eval.bootstrap.linear_combination); a one-seed model (Canto-L trained on HNL or on ttbar + HNL) adds c^2 s^2 with s
the seed sd of Canto-L (ttbar) with the same builder (paper Sec. II: one-seed models carry the seed spread of the
corresponding three-seed model).

Combinations (primary efficiency):
  per architecture   trained on HNL minus on ttbar, on ttbar + HNL minus on ttbar, on ttbar + HNL minus on HNL;
  per sample         Canto-L minus acorn, both trained on that sample;
  gain difference    acorn's change minus Canto-L's change between two training samples.
The unpaired sigma (the two models' total sigmas in quadrature, as results/training_sample_v1.md had it) is listed
for comparison.

    python -m hnl.eval.training_sample_paired --out results/training_sample_paired_v1.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT

FAMILIES = ("mwslmerge", "mwslayermerge")
SAMPLES = ("ttbar", "ttbar+HNL", "HNL")
ARCHS = ("acorn", "Canto-L")
Q = "eff_primary"
TOL = 1e-9


def load_models(fam: str, ld: BT.ValLoader) -> dict:
    t = pd.read_csv(os.path.join(BT.REPO, "results", f"training_sample_val_{fam}.csv"))
    out = {}
    for name, g in t.groupby("model", sort=False):
        seeds = [ld.seed(r.run, r.working_point) for r in g.itertuples()]
        m = BT.Model(name, name, seeds, ld.rs, list(g.working_point), list(g.run))
        for i, r in enumerate(g.itertuples()):
            if abs(m.pts[Q][i] - float(r.eff_primary)) > TOL:
                raise ValueError(f"{fam} {name} seed {r.seed}: {m.pts[Q][i]} != board {r.eff_primary}")
        out[name] = m
    return out


def total_sigma(m: BT.Model, s1: float) -> float:
    d = BS.seed_mean(*m.pr(Q))
    return float(np.hypot(d["sigma"], s1)) if m.n_seeds == 1 else d["sigma"]


def combination(terms: list, s1: float) -> dict:
    """sum c * model over [(c, Model)], paired over the events; one-seed models add c^2 s1^2."""
    d = BS.linear_combination([(c, *m.pr(Q)) for c, m in terms])
    extra = sum(c ** 2 * s1 ** 2 for c, m in terms if m.n_seeds == 1)
    seeds = float(np.sqrt(d["sigma_seeds"] ** 2 + extra))
    sigma = float(np.hypot(d["sigma_boot"], seeds))
    return {"value": d["value"], "sigma_boot": d["sigma_boot"], "sigma_seeds": seeds, "sigma": sigma,
            "z": d["value"] / sigma if sigma > 0 else np.nan}


def rows_for(fam: str, ms: dict) -> list[dict]:
    ref = ms["Canto-L (ttbar)"]
    s1 = float(np.std(ref.pts[Q], ddof=1))
    rows = []

    def add(kind, label, terms):
        if not all(m is not None for _, m in terms):
            return
        r = combination(terms, s1)
        r["sigma_unpaired"] = (float(np.hypot(total_sigma(terms[0][1], s1), total_sigma(terms[1][1], s1)))
                               if len(terms) == 2 else np.nan)
        r.update({"family": fam, "kind": kind, "label": label, "seed_sd_one_seed": s1,
                  "terms": "; ".join(f"{c:+g} {m.name} ({m.n_seeds} seeds)" for c, m in terms)})
        rows.append(r)

    g = {(a, s): ms.get(f"{a} ({s})") for a in ARCHS for s in SAMPLES}
    pairs = (("HNL", "ttbar"), ("ttbar+HNL", "ttbar"), ("ttbar+HNL", "HNL"))
    for a in ARCHS:
        for s, b in pairs:
            add("per architecture", f"{a}, trained on {s} minus trained on {b}", [(1.0, g[a, s]), (-1.0, g[a, b])])
    for s in SAMPLES:
        add("per sample", f"trained on {s}: Canto-L minus acorn", [(1.0, g["Canto-L", s]), (-1.0, g["acorn", s])])
    for s, b in pairs:
        add("gain difference", f"{s} minus {b}: acorn's change minus Canto-L's change",
            [(1.0, g["acorn", s]), (-1.0, g["acorn", b]), (-1.0, g["Canto-L", s]), (1.0, g["Canto-L", b])])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    rows = []
    for fam in FAMILIES:
        rows += rows_for(fam, load_models(fam, ld))
    out = pd.DataFrame(rows)[["family", "kind", "label", "value", "sigma_boot", "sigma_seeds", "sigma", "z",
                              "sigma_unpaired", "seed_sd_one_seed", "terms"]]
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(out.drop(columns="terms").to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
