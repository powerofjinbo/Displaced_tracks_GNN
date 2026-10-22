# Canto (HNL) and Canto (ttbar+HNL) with three seeds on the generated samples (addendum G T2 sets; seeds 1 and 2 added)

Written 2026-10-18T17:37:45Z by `python -m hnl.eval.addendum_g_seeds12 summarize` (git 6fa9d7b893eaaa1e8262f9898b058721afc7a004; job 6854383). Post-freeze, descriptive, no claim (docs/feedback/test_use_ledger.md R20; seed 0 = R14). Canto-L (HNL) and Canto-L (ttbar+HNL) are the paper's Canto (HNL) and Canto (ttbar+HNL); seeds 1 and 2 as selected on validation in results/training_sample_test_selection_seeds12_v1.json, run through the T2 code (hnl.eval.addendum_g, results/addendum_g_models_seeds12_v1.json). Efficiencies in percent, three-seed mean +- sigma with sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (paper Eq. (2); the T2 bootstrap tables, B = 500); the other models are the R14 rows, reproduced exactly. These samples carry the simulation's own labels (R14).

Checks: 43 of 43 passed (identity of every other model's row and of claim G1 with the committed R14 files; seed-0 values of the two models).

## generic_iso (test)

| model | P_G | displaced-source targets | all targets | prompt (ttbar, |d0| < 1 mm) | track fake rate |
|---|---|---|---|---|---|
| Canto-L (ttbar) | 96.94 +- 0.23 | 97.39 +- 0.20 | 97.60 +- 0.24 | 97.71 +- 0.24 | 0.55 +- 0.04 |
| acorn (HNL) + LM | 94.62 +- 0.29 | 95.37 +- 0.24 | 97.37 +- 0.23 | 97.61 +- 0.24 | 1.38 +- 0.08 |
| acorn (ttbar) + LM | 91.37 +- 0.77 | 92.58 +- 0.65 | 96.87 +- 0.31 | 97.12 +- 0.30 | 1.18 +- 0.04 |
| acorn (HNL) + CC | 86.72 +- 0.64 | 87.04 +- 0.75 | 82.87 +- 0.74 | 83.42 +- 0.83 | 4.25 +- 0.30 |
| Canto-L (HNL) | 98.16 +- 0.13 | 98.41 +- 0.11 | 98.60 +- 0.24 | 98.75 +- 0.24 | 0.44 +- 0.04 |
| Canto-L (ttbar+HNL) | 97.44 +- 0.30 | 97.82 +- 0.25 | 98.09 +- 0.09 | 98.19 +- 0.08 | 0.49 +- 0.02 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_G: 98.13 / 98.24 / 98.11; Canto-L (HNL) track fake rate: 0.44 / 0.37 / 0.51; Canto-L (ttbar+HNL) P_G: 97.05 / 97.91 / 97.35; Canto-L (ttbar+HNL) track fake rate: 0.52 / 0.47 / 0.47.

## generic_dense (test)

| model | P_dense | displaced-source targets | all targets | prompt (ttbar, |d0| < 1 mm) | track fake rate |
|---|---|---|---|---|---|
| Canto-L (ttbar) | 97.20 +- 0.23 | 97.61 +- 0.19 | 97.61 +- 0.25 | 97.73 +- 0.25 | 0.57 +- 0.04 |
| acorn (HNL) + LM | 94.24 +- 0.31 | 95.01 +- 0.25 | 97.27 +- 0.24 | 97.60 +- 0.26 | 1.44 +- 0.09 |
| acorn (ttbar) + LM | 92.69 +- 0.84 | 93.73 +- 0.72 | 96.78 +- 0.33 | 97.13 +- 0.32 | 1.23 +- 0.04 |
| acorn (HNL) + CC | 82.77 +- 1.10 | 83.07 +- 1.08 | 82.38 +- 0.69 | 83.02 +- 0.86 | 4.48 +- 0.30 |
| Canto-L (HNL) | 97.19 +- 0.36 | 97.60 +- 0.31 | 98.55 +- 0.24 | 98.75 +- 0.23 | 0.45 +- 0.03 |
| Canto-L (ttbar+HNL) | 97.85 +- 0.22 | 98.16 +- 0.20 | 98.11 +- 0.12 | 98.21 +- 0.11 | 0.49 +- 0.02 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_dense: 97.08 / 97.76 / 96.72; Canto-L (HNL) track fake rate: 0.46 / 0.40 / 0.50; Canto-L (ttbar+HNL) P_dense: 97.56 / 98.15 / 97.83; Canto-L (ttbar+HNL) track fake rate: 0.53 / 0.47 / 0.49.

## hss15_100 (hss)

| model | P_H | P_H electrons | P_H muons | P_H hadrons | P_H s -> ee daughters | P_H V0-tagged | P_H other (s -> bb chain, untagged decays) | all targets | prompt (|d0| < 1 mm, r_prod < 1 mm) | track fake rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 94.42 +- 0.76 | 98.13 +- 0.46 | 95.85 +- 1.14 | 93.95 +- 0.86 | 99.00 +- 0.49 | 97.43 +- 0.65 | 93.52 +- 0.91 | 97.93 +- 0.18 | 98.85 +- 0.14 | 0.26 +- 0.02 |
| acorn (HNL) + LM | 93.06 +- 0.44 | 97.62 +- 0.51 | 94.04 +- 1.23 | 92.52 +- 0.49 | 98.05 +- 0.59 | 95.63 +- 0.58 | 92.21 +- 0.53 | 97.82 +- 0.19 | 99.00 +- 0.15 | 0.83 +- 0.03 |
| acorn (ttbar) + LM | 91.71 +- 0.96 | 96.77 +- 0.66 | 92.65 +- 1.41 | 91.12 +- 1.05 | 98.12 +- 0.57 | 95.74 +- 0.79 | 90.48 +- 1.13 | 97.22 +- 0.29 | 98.60 +- 0.19 | 0.81 +- 0.03 |
| acorn (HNL) + CC | 76.41 +- 0.97 | 89.38 +- 1.13 | 81.04 +- 2.03 | 74.80 +- 1.14 | 93.22 +- 1.09 | 79.09 +- 2.27 | 74.59 +- 1.22 | 87.92 +- 0.33 | 91.64 +- 0.34 | 2.58 +- 0.06 |
| Canto-L (HNL) | 94.17 +- 0.53 | 97.75 +- 0.60 | 95.10 +- 1.17 | 93.74 +- 0.58 | 98.68 +- 0.55 | 97.34 +- 0.67 | 93.24 +- 0.59 | 98.38 +- 0.20 | 99.36 +- 0.17 | 0.25 +- 0.01 |
| Canto-L (ttbar+HNL) | 95.35 +- 0.58 | 98.34 +- 0.40 | 95.95 +- 1.10 | 94.99 +- 0.63 | 99.12 +- 0.33 | 97.54 +- 0.59 | 94.65 +- 0.67 | 98.21 +- 0.17 | 98.98 +- 0.13 | 0.24 +- 0.02 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_H: 93.92 / 94.78 / 93.80; Canto-L (HNL) track fake rate: 0.25 / 0.24 / 0.25; Canto-L (ttbar+HNL) P_H: 94.56 / 95.76 / 95.72; Canto-L (ttbar+HNL) track fake rate: 0.26 / 0.24 / 0.24.

## hss40_300 (hss)

| model | P_H | P_H electrons | P_H muons | P_H hadrons | P_H s -> ee daughters | P_H V0-tagged | P_H other (s -> bb chain, untagged decays) | all targets | prompt (|d0| < 1 mm, r_prod < 1 mm) | track fake rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 96.38 +- 0.33 | 97.98 +- 0.59 | 96.66 +- 1.00 | 96.23 +- 0.37 | 97.53 +- 0.90 | 98.21 +- 0.42 | 96.02 +- 0.40 | 98.31 +- 0.15 | 98.99 +- 0.13 | 0.27 +- 0.02 |
| acorn (HNL) + LM | 89.12 +- 0.78 | 95.16 +- 0.75 | 91.39 +- 1.52 | 88.50 +- 0.84 | 95.45 +- 0.83 | 93.33 +- 0.80 | 88.04 +- 0.87 | 96.98 +- 0.13 | 99.01 +- 0.17 | 1.09 +- 0.03 |
| acorn (ttbar) + LM | 87.60 +- 0.72 | 92.48 +- 0.88 | 87.64 +- 1.78 | 87.17 +- 0.75 | 92.66 +- 1.02 | 93.91 +- 0.73 | 86.27 +- 0.82 | 96.45 +- 0.22 | 98.77 +- 0.16 | 1.04 +- 0.04 |
| acorn (HNL) + CC | 72.61 +- 1.48 | 87.34 +- 1.24 | 78.72 +- 1.87 | 71.09 +- 1.59 | 90.58 +- 1.26 | 76.18 +- 1.84 | 70.89 +- 1.54 | 87.27 +- 0.28 | 91.81 +- 0.31 | 2.84 +- 0.05 |
| Canto-L (HNL) | 96.43 +- 0.42 | 98.20 +- 0.65 | 95.85 +- 0.98 | 96.29 +- 0.43 | 98.12 +- 0.83 | 98.68 +- 0.36 | 95.96 +- 0.45 | 98.67 +- 0.20 | 99.40 +- 0.15 | 0.26 +- 0.01 |
| Canto-L (ttbar+HNL) | 97.15 +- 0.34 | 98.90 +- 0.35 | 96.76 +- 0.83 | 97.01 +- 0.36 | 98.83 +- 0.44 | 98.63 +- 0.36 | 96.80 +- 0.40 | 98.58 +- 0.14 | 99.14 +- 0.12 | 0.24 +- 0.01 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_H: 96.15 / 97.05 / 96.08; Canto-L (HNL) track fake rate: 0.26 / 0.25 / 0.27; Canto-L (ttbar+HNL) P_H: 96.69 / 97.25 / 97.50; Canto-L (ttbar+HNL) track fake rate: 0.25 / 0.25 / 0.24.

## hnl5_50 (hnl)

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | all targets | track fake rate |
|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 99.73 +- 0.12 | 99.73 +- 0.12 | 99.81 +- 0.08 | 100.00 +- 0.00 | 99.92 +- 0.02 | 0.07 +- 0.01 |
| acorn (HNL) + LM | 99.20 +- 0.24 | 99.15 +- 0.25 | 99.36 +- 0.17 | 100.00 +- 0.00 | 99.80 +- 0.03 | 0.33 +- 0.02 |
| acorn (ttbar) + LM | 99.11 +- 0.22 | 99.03 +- 0.23 | 99.24 +- 0.16 | 99.96 +- 0.03 | 99.80 +- 0.02 | 0.45 +- 0.03 |
| acorn (HNL) + CC | 91.75 +- 1.13 | 91.79 +- 1.13 | 91.63 +- 1.23 | 98.04 +- 0.28 | 95.37 +- 0.38 | 1.48 +- 0.05 |
| Canto-L (HNL) | 99.69 +- 0.10 | 99.69 +- 0.10 | 99.81 +- 0.06 | 100.00 +- 0.00 | 99.92 +- 0.02 | 0.06 +- 0.01 |
| Canto-L (ttbar+HNL) | 99.51 +- 0.17 | 99.52 +- 0.17 | 99.68 +- 0.11 | 100.00 +- 0.00 | 99.91 +- 0.01 | 0.07 +- 0.01 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_HNL (fitted |d0|, r_v): 99.65 / 99.71 / 99.71; Canto-L (HNL) track fake rate: 0.07 / 0.05 / 0.07; Canto-L (ttbar+HNL) P_HNL (fitted |d0|, r_v): 99.53 / 99.71 / 99.30; Canto-L (ttbar+HNL) track fake rate: 0.07 / 0.07 / 0.06.

## hnl20_200 (hnl)

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | all targets | track fake rate |
|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 98.32 +- 0.35 | 98.34 +- 0.35 | 98.72 +- 0.26 | 99.98 +- 0.03 | 99.87 +- 0.03 | 0.07 +- 0.01 |
| acorn (HNL) + LM | 96.59 +- 0.50 | 96.63 +- 0.49 | 97.25 +- 0.39 | 99.98 +- 0.02 | 99.68 +- 0.03 | 0.35 +- 0.03 |
| acorn (ttbar) + LM | 93.46 +- 0.76 | 93.50 +- 0.75 | 94.95 +- 0.58 | 99.99 +- 0.02 | 99.62 +- 0.02 | 0.47 +- 0.03 |
| acorn (HNL) + CC | 90.44 +- 0.66 | 90.52 +- 0.65 | 91.34 +- 0.47 | 98.03 +- 0.31 | 95.40 +- 0.31 | 1.46 +- 0.06 |
| Canto-L (HNL) | 98.70 +- 0.27 | 98.71 +- 0.27 | 98.94 +- 0.20 | 99.94 +- 0.04 | 99.86 +- 0.03 | 0.07 +- 0.01 |
| Canto-L (ttbar+HNL) | 98.56 +- 0.26 | 98.58 +- 0.25 | 98.86 +- 0.19 | 99.96 +- 0.03 | 99.85 +- 0.02 | 0.07 +- 0.01 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_HNL (fitted |d0|, r_v): 98.42 / 98.93 / 98.73; Canto-L (HNL) track fake rate: 0.07 / 0.06 / 0.08; Canto-L (ttbar+HNL) P_HNL (fitted |d0|, r_v): 98.58 / 98.78 / 98.32; Canto-L (ttbar+HNL) track fake rate: 0.07 / 0.08 / 0.07.

## hnl15_500 (hnl)

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | all targets | track fake rate |
|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 98.22 +- 0.35 | 98.24 +- 0.35 | 98.48 +- 0.29 | 100.00 +- 0.00 | 99.87 +- 0.03 | 0.07 +- 0.01 |
| acorn (HNL) + LM | 96.80 +- 0.53 | 96.80 +- 0.52 | 97.29 +- 0.43 | 99.96 +- 0.04 | 99.73 +- 0.03 | 0.33 +- 0.03 |
| acorn (ttbar) + LM | 93.91 +- 0.98 | 93.91 +- 0.99 | 94.94 +- 0.82 | 99.99 +- 0.02 | 99.71 +- 0.03 | 0.44 +- 0.03 |
| acorn (HNL) + CC | 91.82 +- 1.04 | 91.80 +- 1.03 | 92.61 +- 0.90 | 98.16 +- 0.33 | 95.50 +- 0.32 | 1.43 +- 0.06 |
| Canto-L (HNL) | 98.57 +- 0.45 | 98.59 +- 0.44 | 98.79 +- 0.37 | 100.00 +- 0.00 | 99.88 +- 0.03 | 0.07 +- 0.01 |
| Canto-L (ttbar+HNL) | 98.99 +- 0.27 | 98.99 +- 0.26 | 99.12 +- 0.21 | 99.99 +- 0.02 | 99.87 +- 0.02 | 0.07 +- 0.00 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_HNL (fitted |d0|, r_v): 98.00 / 98.57 / 99.14; Canto-L (HNL) track fake rate: 0.07 / 0.06 / 0.08; Canto-L (ttbar+HNL) P_HNL (fitted |d0|, r_v): 98.76 / 99.05 / 99.14; Canto-L (ttbar+HNL) track fake rate: 0.07 / 0.07 / 0.07.

## hnl10_100_ref (hnl)

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | all targets | track fake rate |
|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 99.51 +- 0.11 | 99.51 +- 0.11 | 99.64 +- 0.08 | 100.00 +- 0.00 | 99.89 +- 0.03 | 0.07 +- 0.01 |
| acorn (HNL) + LM | 99.03 +- 0.19 | 99.02 +- 0.19 | 99.22 +- 0.13 | 99.95 +- 0.04 | 99.76 +- 0.03 | 0.33 +- 0.03 |
| acorn (ttbar) + LM | 98.39 +- 0.30 | 98.40 +- 0.30 | 98.81 +- 0.22 | 99.94 +- 0.04 | 99.76 +- 0.02 | 0.44 +- 0.03 |
| acorn (HNL) + CC | 93.27 +- 0.55 | 93.24 +- 0.56 | 93.53 +- 0.56 | 98.43 +- 0.23 | 95.40 +- 0.33 | 1.43 +- 0.06 |
| Canto-L (HNL) | 99.53 +- 0.17 | 99.53 +- 0.17 | 99.66 +- 0.13 | 99.99 +- 0.02 | 99.90 +- 0.03 | 0.07 +- 0.01 |
| Canto-L (ttbar+HNL) | 99.46 +- 0.17 | 99.46 +- 0.17 | 99.63 +- 0.12 | 99.98 +- 0.02 | 99.87 +- 0.02 | 0.07 +- 0.01 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_HNL (fitted |d0|, r_v): 99.29 / 99.64 / 99.64; Canto-L (HNL) track fake rate: 0.07 / 0.06 / 0.08; Canto-L (ttbar+HNL) P_HNL (fitted |d0|, r_v): 99.39 / 99.29 / 99.69; Canto-L (ttbar+HNL) track fake rate: 0.07 / 0.06 / 0.06.

## hnl15_200_ref (hnl)

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | all targets | track fake rate |
|---|---|---|---|---|---|---|
| Canto-L (ttbar) | 98.72 +- 0.28 | 98.74 +- 0.27 | 99.03 +- 0.21 | 100.00 +- 0.00 | 99.87 +- 0.03 | 0.08 +- 0.01 |
| acorn (HNL) + LM | 97.22 +- 0.35 | 97.18 +- 0.35 | 97.68 +- 0.29 | 99.89 +- 0.06 | 99.72 +- 0.03 | 0.33 +- 0.02 |
| acorn (ttbar) + LM | 95.52 +- 0.71 | 95.52 +- 0.70 | 96.38 +- 0.56 | 99.88 +- 0.05 | 99.67 +- 0.03 | 0.46 +- 0.03 |
| acorn (HNL) + CC | 91.69 +- 0.72 | 91.68 +- 0.71 | 92.13 +- 0.64 | 98.20 +- 0.28 | 95.56 +- 0.25 | 1.42 +- 0.06 |
| Canto-L (HNL) | 98.98 +- 0.30 | 98.98 +- 0.30 | 99.22 +- 0.23 | 99.99 +- 0.02 | 99.87 +- 0.04 | 0.07 +- 0.01 |
| Canto-L (ttbar+HNL) | 98.93 +- 0.30 | 98.90 +- 0.29 | 99.15 +- 0.22 | 100.00 +- 0.00 | 99.85 +- 0.02 | 0.07 +- 0.00 |

Per seed (Canto-L (HNL) / Canto-L (ttbar+HNL)), first population and fake rate: Canto-L (HNL) P_HNL (fitted |d0|, r_v): 98.67 / 98.89 / 99.39; Canto-L (HNL) track fake rate: 0.07 / 0.06 / 0.08; Canto-L (ttbar+HNL) P_HNL (fitted |d0|, r_v): 98.61 / 98.89 / 99.28; Canto-L (ttbar+HNL) track fake rate: 0.07 / 0.07 / 0.07.

## Each HNL point beside its in-stack reference (three-seed means; difference point - reference)

**hnl5_50 beside hnl10_100_ref**

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | track fake rate |
|---|---|---|---|---|---|
| Canto-L (ttbar) | 99.73 / 99.51 / +0.22 +- 0.16 | 99.73 / 99.51 / +0.22 +- 0.16 | 99.81 / 99.64 / +0.17 +- 0.11 | 100.00 / 100.00 / +0.00 +- 0.00 | 0.07 / 0.07 / -0.00 +- 0.01 |
| acorn (HNL) + LM | 99.20 / 99.03 / +0.17 +- 0.31 | 99.15 / 99.02 / +0.13 +- 0.31 | 99.36 / 99.22 / +0.13 +- 0.21 | 100.00 / 99.95 / +0.05 +- 0.04 | 0.33 / 0.33 / -0.00 +- 0.03 |
| acorn (ttbar) + LM | 99.11 / 98.39 / +0.72 +- 0.37 | 99.03 / 98.40 / +0.64 +- 0.37 | 99.24 / 98.81 / +0.42 +- 0.27 | 99.96 / 99.94 / +0.02 +- 0.06 | 0.45 / 0.44 / +0.01 +- 0.04 |
| acorn (HNL) + CC | 91.75 / 93.27 / -1.51 +- 1.26 | 91.79 / 93.24 / -1.45 +- 1.27 | 91.63 / 93.53 / -1.89 +- 1.36 | 98.04 / 98.43 / -0.39 +- 0.36 | 1.48 / 1.43 / +0.05 +- 0.08 |
| Canto-L (HNL) | 99.69 / 99.53 / +0.16 +- 0.20 | 99.69 / 99.53 / +0.16 +- 0.20 | 99.81 / 99.66 / +0.15 +- 0.14 | 100.00 / 99.99 / +0.01 +- 0.02 | 0.06 / 0.07 / -0.00 +- 0.01 |
| Canto-L (ttbar+HNL) | 99.51 / 99.46 / +0.06 +- 0.25 | 99.52 / 99.46 / +0.06 +- 0.25 | 99.68 / 99.63 / +0.05 +- 0.17 | 100.00 / 99.98 / +0.02 +- 0.02 | 0.07 / 0.07 / +0.00 +- 0.01 |

**hnl20_200 beside hnl10_100_ref**

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | track fake rate |
|---|---|---|---|---|---|
| Canto-L (ttbar) | 98.32 / 99.51 / -1.19 +- 0.36 | 98.34 / 99.51 / -1.17 +- 0.36 | 98.72 / 99.64 / -0.92 +- 0.27 | 99.98 / 100.00 / -0.02 +- 0.03 | 0.07 / 0.07 / +0.00 +- 0.01 |
| acorn (HNL) + LM | 96.59 / 99.03 / -2.44 +- 0.53 | 96.63 / 99.02 / -2.39 +- 0.53 | 97.25 / 99.22 / -1.97 +- 0.41 | 99.98 / 99.95 / +0.02 +- 0.05 | 0.35 / 0.33 / +0.03 +- 0.04 |
| acorn (ttbar) + LM | 93.46 / 98.39 / -4.93 +- 0.82 | 93.50 / 98.40 / -4.89 +- 0.82 | 94.95 / 98.81 / -3.86 +- 0.62 | 99.99 / 99.94 / +0.05 +- 0.05 | 0.47 / 0.44 / +0.03 +- 0.04 |
| acorn (HNL) + CC | 90.44 / 93.27 / -2.83 +- 0.86 | 90.52 / 93.24 / -2.71 +- 0.85 | 91.34 / 93.53 / -2.18 +- 0.74 | 98.03 / 98.43 / -0.40 +- 0.39 | 1.46 / 1.43 / +0.03 +- 0.08 |
| Canto-L (HNL) | 98.70 / 99.53 / -0.83 +- 0.32 | 98.71 / 99.53 / -0.82 +- 0.32 | 98.94 / 99.66 / -0.72 +- 0.24 | 99.94 / 99.99 / -0.05 +- 0.05 | 0.07 / 0.07 / +0.00 +- 0.01 |
| Canto-L (ttbar+HNL) | 98.56 / 99.46 / -0.90 +- 0.31 | 98.58 / 99.46 / -0.88 +- 0.31 | 98.86 / 99.63 / -0.77 +- 0.22 | 99.96 / 99.98 / -0.01 +- 0.04 | 0.07 / 0.07 / +0.01 +- 0.01 |

**hnl15_500 beside hnl15_200_ref**

| model | P_HNL (fitted |d0|, r_v) | P_HNL (exact |d0|, r_v) | N-daughter targets | W-muon targets | track fake rate |
|---|---|---|---|---|---|
| Canto-L (ttbar) | 98.22 / 98.72 / -0.50 +- 0.45 | 98.24 / 98.74 / -0.50 +- 0.45 | 98.48 / 99.03 / -0.55 +- 0.36 | 100.00 / 100.00 / +0.00 +- 0.00 | 0.07 / 0.08 / -0.01 +- 0.02 |
| acorn (HNL) + LM | 96.80 / 97.22 / -0.43 +- 0.65 | 96.80 / 97.18 / -0.39 +- 0.64 | 97.29 / 97.68 / -0.40 +- 0.53 | 99.96 / 99.89 / +0.07 +- 0.07 | 0.33 / 0.33 / -0.01 +- 0.03 |
| acorn (ttbar) + LM | 93.91 / 95.52 / -1.61 +- 1.22 | 93.91 / 95.52 / -1.61 +- 1.22 | 94.94 / 96.38 / -1.44 +- 1.00 | 99.99 / 99.88 / +0.11 +- 0.06 | 0.44 / 0.46 / -0.01 +- 0.04 |
| acorn (HNL) + CC | 91.82 / 91.69 / +0.13 +- 1.26 | 91.80 / 91.68 / +0.12 +- 1.25 | 92.61 / 92.13 / +0.47 +- 1.09 | 98.16 / 98.20 / -0.04 +- 0.43 | 1.43 / 1.42 / +0.02 +- 0.08 |
| Canto-L (HNL) | 98.57 / 98.98 / -0.41 +- 0.54 | 98.59 / 98.98 / -0.39 +- 0.54 | 98.79 / 99.22 / -0.44 +- 0.44 | 100.00 / 99.99 / +0.01 +- 0.02 | 0.07 / 0.07 / -0.00 +- 0.01 |
| Canto-L (ttbar+HNL) | 98.99 / 98.93 / +0.06 +- 0.40 | 98.99 / 98.90 / +0.09 +- 0.40 | 99.12 / 99.15 / -0.03 +- 0.31 | 99.99 / 100.00 / -0.01 +- 0.02 | 0.07 / 0.07 / -0.00 +- 0.01 |

Bins of the two models (three seeds) and every row: the CSV next to this file; the full T2 summaries of this read-out (all models): /work/submit/jinboz1/hnl/runs_extra/ts_seeds12/summary_v2.
