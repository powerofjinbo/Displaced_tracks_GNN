"""Writes results/training_sample_bins_paired_3seed_v1.{csv,md}: the training-sample study per abs(d0) bin with the
uncertainty of paper Eq. (sigma), for paper Fig. 7 (fig:arms). Validation only; nothing is read from a test set.

Models: acorn and Canto-L trained on ttbar, on ttbar+HNL and on HNL, three seeds each, every seed with MWS-layer +
learned merging (LM) at the working point chosen for it on validation: the per-seed rows of family mwslmerge of
results/training_sample_val_3seed_v1.csv (hnl.eval.training_sample_3seed, job 6831564). Estimator, replicate tables
and model loading are those of that job (hnl.eval.training_sample_3seed.load_models, hnl.eval.bootstrap_tables):
the val replicate tables (B = 1000, seed 20261005, read from runs/feedback_v1/bootstrap_v1/tables/val_b1000, never
rewritten), the value of a model is the mean of its seeds, sigma^2 = sigma_boot^2(seed mean) + s_seeds^2 / 3, and a
difference is the bootstrap of the paired difference of seed means plus each model's seed term
(hnl.eval.bootstrap.linear_combination).

Quantities: the HNL-daughter targets in each preregistered abs(d0) bin (docs/preregistration.md:27; the bins from
10 mm up hold only daughters of the primary population), the primary population (HNL daughters), the W muons and the
ttbar targets with abs(d0) < 1 mm. Rows: every model (value, seed sd, bootstrap sd, sigma, counts, 68 % Clopper-Pearson
interval on the per-seed count) and, per quantity, the paired differences of results/training_sample_paired_3seed_v1.csv
(per architecture between training samples, per sample Canto-L minus acorn, and acorn's change minus Canto-L's change).

Checks (stop on a mismatch): every seed's primary efficiency, W-muon and ttbar-prompt efficiency and every abs(d0) bin
count k and n against the board CSV; every model's primary value, sigma and seed sd as printed in
results/training_sample_val_3seed_v1.md:13-18; every primary difference (value, sigma_boot, sigma_seeds, sigma, z)
against results/training_sample_paired_3seed_v1.csv (1e-12).

    sbatch ... --wrap "source env/activate.sh && python -I scripts/diagnostics/training_sample_bins_paired.py"
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import training_sample_3seed as TS3

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
FAM = "mwslmerge"
BOARD = os.path.join(RES, "training_sample_val_3seed_v1.csv")
SUMMARY_MD = os.path.join(RES, "training_sample_val_3seed_v1.md")
PAIRED = os.path.join(RES, "training_sample_paired_3seed_v1.csv")
OUT = os.path.join(RES, "training_sample_bins_paired_3seed_v1")
MODELS = ["acorn (ttbar)", "acorn (ttbar+HNL)", "acorn (HNL)", "Canto-L (ttbar)", "Canto-L (ttbar+HNL)", "Canto-L (HNL)"]
SAMPLES = ("ttbar", "ttbar+HNL", "HNL")
ARCHS = ("acorn", "Canto-L")
SUMMARY_LINES = {m: 13 + i for i, m in enumerate(MODELS)}       # training_sample_val_3seed_v1.md:13-18 (LM table)
D0 = list(zip(BT.D0_BINS, BT.D0_EDGES[:-1], BT.D0_EDGES[1:]))   # (quantity, lo, hi): d0_0_0.1 ... d0_300_inf
QUANT = ["eff_primary"] + [q for q, _, _ in D0] + ["eff_w_muons", "eff_ttbar_prompt"]
BOARD_COL = {**{q: q for q, _, _ in D0}, "eff_primary": "eff_primary", "eff_w_muons": "eff_w_muons",
             "eff_ttbar_prompt": "eff_ttbar_prompt"}
TOL = 1e-12


def text(q: str) -> str:
    for name, lo, hi in D0:
        if name == q:
            return f"abs(d0) {BT._num(lo)}-{BT._num(hi)} mm" if np.isfinite(hi) else f"abs(d0) >= {BT._num(lo)} mm"
    return {"eff_primary": "primary (HNL daughters)", "eff_w_muons": "W muons",
            "eff_ttbar_prompt": "ttbar prompt (abs(d0) < 1 mm)"}[q]


def board() -> pd.DataFrame:
    t = pd.read_csv(BOARD)
    t = t[t.family == FAM].drop(columns="family")
    order = {m: i for i, m in enumerate(MODELS)}
    t = t.assign(_o=t.model.map(order)).sort_values(["_o", "seed"]).drop(columns="_o").reset_index(drop=True)
    for m in MODELS:
        if list(t[t.model == m].seed) != [0, 1, 2]:
            raise SystemExit(f"{BOARD}: {m} has seeds {list(t[t.model == m].seed)}")
    return t


def check_seeds(ms: dict, t: pd.DataFrame) -> None:
    """Every seed of every model against the board: efficiencies (1e-9, as training_sample_3seed) and bin counts."""
    for m in MODELS:
        g = t[t.model == m].reset_index(drop=True)
        mod = ms[m]
        if mod.runs != list(g.run) or mod.wps != list(g.working_point):
            raise SystemExit(f"{m}: runs or working points differ from the board")
        for i, r in g.iterrows():
            for q in ("eff_primary", "eff_w_muons", "eff_ttbar_prompt"):
                if abs(mod.pts[q][i] - float(r[q])) > 1e-9:
                    raise SystemExit(f"{m} seed {r.seed} {q}: {mod.pts[q][i]} != board {r[q]}")
            for q, _, _ in D0:
                k, n = int(round(mod.num[q][i])), int(round(mod.den[q][i]))
                if (k, n) != (int(r[f"{q}_k"]), int(r[f"{q}_n"])):
                    raise SystemExit(f"{m} seed {r.seed} {q}: {k}/{n} != board {r[f'{q}_k']}/{r[f'{q}_n']}")
                if abs(mod.pts[q][i] - float(r[q])) > 1e-9:
                    raise SystemExit(f"{m} seed {r.seed} {q}: {mod.pts[q][i]} != board {r[q]}")


def check_summary(ms: dict) -> None:
    """Primary value, sigma and seed sd of every model as printed in training_sample_val_3seed_v1.md:13-18."""
    lines = open(SUMMARY_MD).read().splitlines()
    for m in MODELS:
        ln = lines[SUMMARY_LINES[m] - 1]
        q = re.match(rf"\| {re.escape(m)} \| 3 \| ([\d.]+) \+- ([\d.]+) \| ([\d.]+) \|", ln)
        if q is None:
            raise SystemExit(f"{SUMMARY_MD}:{SUMMARY_LINES[m]}: not the LM row of {m}: {ln}")
        s = BT.summary(ms[m], "eff_primary")
        got = (f"{100 * s['value']:.2f}", f"{100 * s['combined']:.2f}", f"{100 * s['seed_sd']:.2f}")
        if got != q.groups():
            raise SystemExit(f"{m}: primary {got} != {SUMMARY_MD}:{SUMMARY_LINES[m]} {q.groups()}")


def combos(ms: dict) -> list[tuple[str, str, list]]:
    """(kind, label, terms) in the order and with the labels of hnl.eval.training_sample_paired.rows_for."""
    g = {(a, s): ms[f"{a} ({s})"] for a in ARCHS for s in SAMPLES}
    pairs = (("HNL", "ttbar"), ("ttbar+HNL", "ttbar"), ("ttbar+HNL", "HNL"))
    out = []
    for a in ARCHS:
        for s, b in pairs:
            out.append(("per architecture", f"{a}, trained on {s} minus trained on {b}",
                        [(1.0, g[a, s]), (-1.0, g[a, b])]))
    for s in SAMPLES:
        out.append(("per sample", f"trained on {s}: Canto-L minus acorn", [(1.0, g["Canto-L", s]), (-1.0, g["acorn", s])]))
    for s, b in pairs:
        out.append(("gain difference", f"{s} minus {b}: acorn's change minus Canto-L's change",
                    [(1.0, g["acorn", s]), (-1.0, g["acorn", b]), (-1.0, g["Canto-L", s]), (1.0, g["Canto-L", b])]))
    return out


def main() -> None:
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", os.path.relpath(__file__, REPO)],
                           capture_output=True, text=True).stdout.strip()
    commit = f"HEAD {commit}" + (", this script not yet committed" if dirty else "")
    t = board()
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    ms = TS3.load_models(t, ld)            # checks every seed's primary efficiency against the board (1e-9)
    if set(ms) != set(MODELS) or any(ms[m].n_seeds != 3 for m in MODELS):
        raise SystemExit(f"models {sorted(ms)}")
    check_seeds(ms, t)
    check_summary(ms)

    rows = []
    for m in MODELS:
        for q in QUANT:
            s = BT.summary(ms[m], q)
            rows.append({"kind": "model", "label": m, "terms": "", "quantity": q, "text": text(q),
                         "value": s["value"], "seed_sd": s["seed_sd"], "sigma_boot": s["boot_sd"],
                         "sigma_seeds": s["seed_sd"] / np.sqrt(3), "sigma": s["combined"], "z": np.nan,
                         "n_per_seed": s["n_per_seed"], "k_pooled": s["k_pooled"], "n_pooled": s["n_pooled"],
                         "cp68_lo": s["cp68_lo"], "cp68_hi": s["cp68_hi"], "boot68_lo": s["boot68_lo"],
                         "boot68_hi": s["boot68_hi"], "B": s["B"], "per_seed": s["per_seed"],
                         "working_points": "; ".join(ms[m].wps)})
    for kind, label, terms in combos(ms):
        for q in QUANT:
            d = BS.linear_combination([(c, *mod.pr(q)) for c, mod in terms])
            rows.append({"kind": kind, "label": label, "quantity": q, "text": text(q),
                         "terms": "; ".join(f"{c:+g} {mod.name} ({mod.n_seeds} seeds)" for c, mod in terms),
                         "value": d["value"], "seed_sd": np.nan, "sigma_boot": d["sigma_boot"],
                         "sigma_seeds": d["sigma_seeds"], "sigma": d["sigma"], "z": d["z"], "B": rs.b,
                         "n_per_seed": int(round(terms[0][1].den[q][0]))})
    out = pd.DataFrame(rows)

    # the primary differences are those of results/training_sample_paired_3seed_v1.csv
    p = pd.read_csv(PAIRED)
    p = p[p.family == FAM].set_index("label")
    mine = out[(out.kind != "model") & (out.quantity == "eff_primary")].set_index("label")
    if set(mine.index) != set(p.index):
        raise SystemExit(f"differences differ from {PAIRED}: {sorted(set(mine.index) ^ set(p.index))}")
    for lab, r in mine.iterrows():
        for col in ("value", "sigma_boot", "sigma_seeds", "sigma", "z"):
            if abs(float(r[col]) - float(p.loc[lab, col])) > TOL:
                raise SystemExit(f"{lab} {col}: {r[col]} != {PAIRED} {p.loc[lab, col]}")

    out.to_csv(OUT + ".csv", index=False, float_format="%.10g")
    L = ["# Training-sample study per abs(d0) bin, with the uncertainty of Eq. (sigma) (validation, LM)", "",
         f"Generated by scripts/diagnostics/training_sample_bins_paired.py ({commit}; job "
         f"{os.environ.get('SLURM_JOB_ID', 'none')}) from the per-seed rows of family mwslmerge of",
         "results/training_sample_val_3seed_v1.csv (hnl.eval.training_sample_3seed, job 6831564), with the val replicate",
         f"tables of hnl.eval.bootstrap_tables (B = {rs.b}, seed {BT.SEED_VAL}). Validation events only. Every model: three",
         "seeds, MWS-layer + learned merging at the working point of each seed. Value: mean over the seeds; sigma: paper",
         "Eq. (sigma), sqrt(boot sd^2 + seed sd^2 / 3); differences: bootstrap of the paired difference plus both seed",
         "terms. The primary rows reproduce results/training_sample_val_3seed_v1.md:13-18 and every primary difference",
         "results/training_sample_paired_3seed_v1.csv. Bins: HNL-daughter targets in the preregistered abs(d0) bins",
         "(docs/preregistration.md:27); from 10 mm up every such target is an HNL daughter of the primary population.", "",
         "## Efficiency (%, mean of three seeds +- sigma of Eq. (sigma))", "",
         "| quantity | per seed | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    mod = out[out.kind == "model"].set_index(["label", "quantity"])
    for q in QUANT:
        n = int(mod.loc[(MODELS[0], q), "n_per_seed"])
        L.append(f"| {text(q)} | {n} | " + " | ".join(
            f"{100 * mod.loc[(m, q), 'value']:.2f} +- {100 * mod.loc[(m, q), 'sigma']:.2f}" for m in MODELS) + " |")
    L += ["", "## Differences (points, +- sigma; z in brackets)", ""]
    labels = [lab for _, lab, _ in combos(ms)]
    dif = out[out.kind != "model"].set_index(["label", "quantity"])
    L += ["| quantity | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for q in QUANT:
        L.append(f"| {text(q)} | " + " | ".join(
            f"{100 * dif.loc[(lab, q), 'value']:+.2f} +- {100 * dif.loc[(lab, q), 'sigma']:.2f} "
            f"({dif.loc[(lab, q), 'z']:.1f})" for lab in labels) + " |")
    L += ["", "## Working points (per seed, as the board)", ""]
    L += [f"- {m}: {'; '.join(ms[m].wps)}" for m in MODELS]
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(OUT + ".csv")


if __name__ == "__main__":
    sys.exit(main())
