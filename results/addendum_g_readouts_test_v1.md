# Addendum G: read-outs of T1 deviation 15 (test; no claim)

Written 2026-10-15 19:45 UTC by `hnl.eval.addendum_g_readouts` (git 7604541); definitions in the module docstring and `docs/addendum_g_t2_readouts.md`. Efficiencies in percent (3-seed means +- sigma); Delta = Canto-L (ttbar) - acorn (HNL) + LM in points. Descriptive: none of these numbers changes a claim.

## 1. Sign of the G1 difference per bin (P_G, generic_iso)

| variable | bin | n | A | B | Delta | sign |
|---|---|---|---|---|---|---|
| |d0| [mm] | [0, 0.1) | 21 | 95.24 | 95.24 | +0.00 +- 0.00 | 0 |
| |d0| [mm] | [0.1, 1) | 162 | 98.77 | 97.53 | +1.23 +- 0.98 | + |
| |d0| [mm] | [1, 3) | 380 | 98.60 | 97.89 | +0.70 +- 0.54 | + |
| |d0| [mm] | [3, 10) | 1085 | 99.11 | 98.03 | +1.08 +- 0.33 | + |
| |d0| [mm] | [10, 30) | 2495 | 99.19 | 97.45 | +1.74 +- 0.36 | + |
| |d0| [mm] | [30, 100) | 2876 | 97.28 | 94.48 | +2.79 +- 0.40 | + |
| |d0| [mm] | [100, 300) | 1429 | 92.09 | 88.29 | +3.80 +- 0.71 | + |
| |d0| [mm] | [300, inf) | 262 | 86.01 | 83.08 | absolute only | not ranked |
| r_v [mm] | [10, 30) | 154 | 99.78 | 99.13 | +0.65 +- 0.74 | + |
| r_v [mm] | [30, 100) | 1180 | 98.50 | 95.40 | +3.11 +- 0.75 | + |
| r_v [mm] | [100, 200) | 2697 | 97.26 | 94.59 | +2.67 +- 0.49 | + |
| r_v [mm] | [200, 300) | 2077 | 97.00 | 94.99 | +2.01 +- 0.40 | + |
| r_v [mm] | [300, 450) | 1795 | 97.33 | 95.15 | +2.17 +- 0.44 | + |
| r_v [mm] | [450, 650) | 734 | 94.05 | 92.64 | +1.41 +- 0.77 | + |
| r_v [mm] | [650, inf) | 73 | 72.15 | 70.32 | absolute only | not ranked |
| first-hit radius [mm] | [0, 50) | 195 | 98.29 | 96.92 | +1.37 +- 1.31 | + |
| first-hit radius [mm] | [50, 100) | 605 | 97.47 | 94.16 | +3.31 +- 0.92 | + |
| first-hit radius [mm] | [100, 200) | 2132 | 97.87 | 95.61 | +2.27 +- 0.43 | + |
| first-hit radius [mm] | [200, 300) | 2362 | 96.71 | 93.80 | +2.91 +- 0.46 | + |
| first-hit radius [mm] | [300, 500) | 2601 | 97.22 | 95.09 | +2.13 +- 0.31 | + |
| first-hit radius [mm] | [500, 700) | 781 | 93.56 | 92.53 | +1.02 +- 0.62 | + |
| first-hit radius [mm] | [700, inf) | 34 | 94.12 | 97.06 | absolute only | not ranked |
| |d0| / r_first | [0, 0.05) | 2114 | 99.04 | 97.98 | +1.06 +- 0.23 | + |
| |d0| / r_first | [0.05, 0.1) | 1253 | 99.04 | 97.50 | +1.54 +- 0.38 | + |
| |d0| / r_first | [0.1, 0.2) | 1634 | 98.27 | 96.31 | +1.96 +- 0.47 | + |
| |d0| / r_first | [0.2, 0.4) | 1750 | 96.32 | 93.77 | +2.55 +- 0.49 | + |
| |d0| / r_first | [0.4, 0.7) | 1308 | 95.16 | 91.31 | +3.85 +- 0.70 | + |
| |d0| / r_first | [0.7, 1.01) | 651 | 88.02 | 82.90 | +5.12 +- 1.51 | + |
| pT [GeV] | [1, 2) | 1694 | 94.39 | 90.55 | +3.84 +- 0.69 | + |
| pT [GeV] | [2, 5) | 2543 | 95.27 | 92.12 | +3.15 +- 0.51 | + |
| pT [GeV] | [5, 10) | 1737 | 97.54 | 95.80 | +1.75 +- 0.37 | + |
| pT [GeV] | [10, 20) | 1205 | 99.56 | 98.67 | +0.89 +- 0.38 | + |
| pT [GeV] | [20, inf) | 1531 | 99.80 | 98.76 | +1.05 +- 0.25 | + |

Ranked bins: Delta > 0 in 29, < 0 in 0, = 0 in 1 (bins of one variable are disjoint; the variables overlap).

## 2. P_G reweighted in r_v

| target | n | Kish n_eff | uncovered target fraction | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|---|
| Zenodo HNL r_v shape | 8671 | 6783 | 0.000 | 97.67 +- 0.24 | 95.18 +- 0.32 | 92.10 +- 0.73 | 86.92 +- 0.69 | 98.49 +- 0.15 | 97.67 +- 0.17 | +2.50 +- 0.33 |
| exponential, lambda = 100 mm | 8710 | 4249 | 0.000 | 98.00 +- 0.28 | 95.47 +- 0.38 | 92.49 +- 0.70 | 86.82 +- 0.77 | 98.61 +- 0.18 | 98.00 +- 0.19 | +2.53 +- 0.42 |
| exponential, lambda = 300 mm | 8710 | 7486 | 0.000 | 97.49 +- 0.24 | 95.02 +- 0.31 | 91.82 +- 0.76 | 86.86 +- 0.66 | 98.42 +- 0.14 | 97.52 +- 0.18 | +2.47 +- 0.33 |

## 3. P_G by species, pT and |eta| (generic_iso)

| variable | bin | n | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|
| species | e | 1122 | 98.46 | 96.02 | 93.94 | 88.62 | 98.75 | 97.95 | +2.44 +- 0.51 |
| species | mu | 1668 | 97.42 | 95.02 | 91.63 | 86.63 | 98.80 | 97.30 | +2.40 +- 0.52 |
| species | pi | 1688 | 96.54 | 94.39 | 91.43 | 86.79 | 97.63 | 96.56 | +2.15 +- 0.52 |
| species | K | 2109 | 96.70 | 94.66 | 90.93 | 86.91 | 98.20 | 97.44 | +2.04 +- 0.45 |
| species | p | 2123 | 96.33 | 93.72 | 90.19 | 85.54 | 97.60 | 96.37 | +2.61 +- 0.40 |
| pT [GeV] | [1, 2) | 1694 | 94.39 | 90.55 | 84.45 | 82.78 | 96.93 | 94.16 | +3.84 +- 0.69 |
| pT [GeV] | [2, 5) | 2543 | 95.27 | 92.12 | 87.30 | 84.34 | 97.33 | 95.60 | +3.15 +- 0.51 |
| pT [GeV] | [5, 10) | 1737 | 97.54 | 95.80 | 92.25 | 87.56 | 98.39 | 97.81 | +1.75 +- 0.37 |
| pT [GeV] | [10, 20) | 1205 | 99.56 | 98.67 | 98.26 | 91.62 | 99.17 | 99.59 | +0.89 +- 0.38 |
| pT [GeV] | [20, inf) | 1531 | 99.80 | 98.76 | 99.35 | 90.22 | 99.67 | 99.80 | +1.05 +- 0.25 |
| |eta| | [0, 0.5) | 2587 | 97.01 | 94.47 | 90.57 | 84.92 | 98.65 | 96.17 | +2.54 +- 0.53 |
| |eta| | [0.5, 1) | 2122 | 96.65 | 95.11 | 92.84 | 87.29 | 97.41 | 96.14 | +1.54 +- 0.33 |
| |eta| | [1, 1.5) | 1538 | 95.06 | 91.61 | 87.58 | 84.31 | 96.88 | 96.49 | +3.45 +- 0.78 |
| |eta| | [1.5, 2) | 1428 | 97.90 | 95.77 | 92.46 | 89.19 | 98.81 | 98.81 | +2.12 +- 0.56 |
| |eta| | [2, 2.5) | 749 | 98.71 | 97.06 | 95.15 | 90.48 | 99.20 | 99.47 | +1.65 +- 0.53 |
| |eta| | [2.5, 3) | 226 | 99.12 | 95.72 | 93.81 | 89.53 | 99.12 | 100.00 | +3.39 +- 1.41 |
| |eta| | [3, 4) | 60 | 99.44 | 98.89 | 88.33 | 89.44 | 100.00 | 98.33 | +0.56 +- 1.21 |

## 4. Second read-out with the Zenodo same-module label exchange emulated

Efficiency (fake rate) with the generated labels -> with the emulated exchange, from the stored tracks.

| sample | population | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) |
|---|---|---|---|---|---|---|---|
| generic_iso | P_G | 96.94 -> 96.73 | 94.62 -> 94.31 | 91.37 -> 91.10 | 86.72 -> 86.38 | 98.13 -> 97.98 | 97.05 -> 96.85 |
| generic_iso | displaced-source targets | 97.39 -> 97.19 | 95.37 -> 95.05 | 92.58 -> 92.30 | 87.04 -> 86.64 | 98.36 -> 98.23 | 97.50 -> 97.32 |
| generic_iso | all targets | 97.60 -> 95.75 | 97.37 -> 95.16 | 96.87 -> 94.76 | 82.87 -> 79.50 | 98.63 -> 97.02 | 97.97 -> 96.16 |
| generic_iso | prompt (ttbar, |d0| < 1 mm) | 97.71 -> 95.92 | 97.61 -> 95.47 | 97.12 -> 95.05 | 83.42 -> 80.06 | 98.77 -> 97.23 | 98.08 -> 96.33 |
| generic_iso | track fake rate | 0.55 -> 1.55 | 1.38 -> 2.60 | 1.18 -> 2.37 | 4.25 -> 6.48 | 0.44 -> 1.34 | 0.52 -> 1.49 |
| generic_dense | P_dense | 97.20 -> 94.83 | 94.24 -> 91.87 | 92.69 -> 90.16 | 82.77 -> 79.91 | 97.08 -> 94.86 | 97.56 -> 95.44 |
| generic_dense | displaced-source targets | 97.61 -> 95.48 | 95.01 -> 92.91 | 93.73 -> 91.45 | 83.07 -> 80.26 | 97.55 -> 95.55 | 97.89 -> 96.02 |
| generic_dense | all targets | 97.61 -> 95.61 | 97.27 -> 94.88 | 96.78 -> 94.50 | 82.38 -> 78.92 | 98.57 -> 96.83 | 97.96 -> 96.07 |
| generic_dense | prompt (ttbar, |d0| < 1 mm) | 97.73 -> 95.82 | 97.60 -> 95.30 | 97.13 -> 94.96 | 83.02 -> 79.58 | 98.76 -> 97.15 | 98.08 -> 96.28 |
| generic_dense | track fake rate | 0.57 -> 1.64 | 1.44 -> 2.75 | 1.23 -> 2.51 | 4.48 -> 6.77 | 0.46 -> 1.40 | 0.53 -> 1.54 |

G1 rule, true labels: Delta = +2.32 +- 0.28, fake-rate difference -0.83 points; the G1 rule would read: holds.

G1 rule, emulated exchange: Delta = +2.43 +- 0.30, fake-rate difference -1.05 points; the G1 rule would read: holds.

Emulated exchange: 6.70 % of the hits carry another label; stored tracks reproduce the stored matching in every event (0 differences).

## 5. P_G under the hit-fit |d0| and the fitted r_v (generic_iso)

| variable | bin | n | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|
| overlap | P_G (exact) and P_G (fit) | 8498 |  |  |  |  |  |  |  |
| overlap | P_G (exact) only | 212 |  |  |  |  |  |  |  |
| overlap | P_G (fit) only | 13 |  |  |  |  |  |  |  |
| population | P_G (exact) | 8710 | 96.94 | 94.62 | 91.37 | 86.72 | 98.13 | 97.05 | +2.32 +- 0.28 |
| population | P_G (fit) | 8511 | 97.00 | 94.66 | 91.39 | 86.77 | 98.17 | 97.10 | +2.35 +- 0.29 |
| |d0| fitted [mm] | [0, 0.1) | 12 | 100.00 | 100.00 | 100.00 | 77.78 | 100.00 | 100.00 | +0.00 +- 0.00 |
| |d0| fitted [mm] | [0.1, 1) | 137 | 100.00 | 99.03 | 99.51 | 84.91 | 99.27 | 99.27 | +0.97 +- 0.64 |
| |d0| fitted [mm] | [1, 3) | 323 | 99.28 | 98.56 | 98.66 | 86.69 | 99.69 | 99.07 | +0.72 +- 0.54 |
| |d0| fitted [mm] | [3, 10) | 972 | 99.55 | 98.49 | 98.42 | 89.33 | 99.28 | 99.79 | +1.06 +- 0.35 |
| |d0| fitted [mm] | [10, 30) | 2497 | 99.17 | 97.45 | 96.49 | 89.12 | 99.32 | 98.96 | +1.72 +- 0.36 |
| |d0| fitted [mm] | [30, 100) | 2871 | 97.27 | 94.49 | 91.23 | 87.97 | 97.98 | 97.49 | +2.79 +- 0.40 |
| |d0| fitted [mm] | [100, 300) | 1436 | 92.18 | 88.32 | 79.57 | 81.24 | 95.89 | 92.13 | +3.85 +- 0.72 |
| |d0| fitted [mm] | [300, inf) | 263 | 85.93 | 83.14 | 69.71 | 73.38 | 95.06 | 88.59 |  |
| r_v fitted [mm] | [10, 30) | 154 | 99.78 | 99.13 | 98.92 | 87.66 | 99.35 | 100.00 | +0.65 +- 0.74 |
| r_v fitted [mm] | [30, 100) | 1036 | 98.94 | 96.53 | 94.82 | 88.51 | 98.75 | 98.65 | +2.41 +- 0.65 |
| r_v fitted [mm] | [100, 200) | 2249 | 98.70 | 96.50 | 94.07 | 88.93 | 99.07 | 98.62 | +2.19 +- 0.43 |
| r_v fitted [mm] | [200, 300) | 1791 | 97.84 | 96.09 | 93.78 | 88.83 | 98.66 | 97.82 | +1.75 +- 0.39 |
| r_v fitted [mm] | [300, 450) | 1482 | 98.81 | 97.05 | 95.34 | 89.45 | 99.46 | 99.06 | +1.75 +- 0.41 |
| r_v fitted [mm] | [450, 650) | 521 | 97.31 | 96.99 | 93.60 | 86.63 | 98.46 | 96.55 | +0.32 +- 0.77 |
| r_v fitted [mm] | [650, inf) | 12 | 94.44 | 100.00 | 94.44 | 88.89 | 100.00 | 100.00 | -5.56 +- 5.19 |
| r_v fitted [mm] | no vertex | 1266 | 88.68 | 83.46 | 73.96 | 75.36 | 93.60 | 89.65 | +5.21 +- 1.25 |

## 6. Dense component: bins chosen from truth (P_dense, generic_dense)

Preregistered bins: `hnl.eval.addendum_g_summary` (bins csv). Delta = Canto-L (ttbar) - acorn (HNL) + LM, descriptive.

| variable | bin | n | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|
| same-vertex multiplicity | 1 | 610 | 92.46 | 89.23 | 84.59 | 81.37 | 92.95 | 93.11 | +3.22 +- 1.06 |
| same-vertex multiplicity | 2 | 952 | 95.69 | 92.72 | 89.53 | 82.35 | 95.38 | 96.22 | +2.98 +- 0.95 |
| same-vertex multiplicity | 3 | 1467 | 97.21 | 93.98 | 91.84 | 83.30 | 98.02 | 97.21 | +3.23 +- 0.66 |
| same-vertex multiplicity | 4 | 1439 | 97.43 | 94.90 | 93.65 | 85.27 | 97.08 | 97.64 | +2.52 +- 0.69 |
| same-vertex multiplicity | >= 5 | 3314 | 98.41 | 95.43 | 95.05 | 81.82 | 97.92 | 98.88 | +2.98 +- 0.45 |
| other targets within dR < 0.2 | 0 | 4317 | 96.80 | 93.61 | 90.66 | 86.26 | 97.36 | 97.27 | +3.20 +- 0.38 |
| other targets within dR < 0.2 | 1 | 1493 | 98.37 | 96.03 | 94.93 | 82.52 | 97.92 | 98.46 | +2.34 +- 0.55 |
| other targets within dR < 0.2 | 2 | 749 | 97.86 | 96.48 | 96.57 | 80.64 | 96.26 | 98.40 | +1.38 +- 0.77 |
| other targets within dR < 0.2 | 3-4 | 797 | 97.45 | 94.35 | 96.03 | 76.70 | 96.49 | 97.87 | +3.09 +- 0.90 |
| other targets within dR < 0.2 | >= 5 | 426 | 95.54 | 90.30 | 92.41 | 63.30 | 93.90 | 95.31 | +5.24 +- 1.76 |
| merged tail | r_v >= 450 mm | 792 | 90.40 | 89.06 | 86.41 | 78.37 | 88.64 | 91.54 | +1.35 +- 1.51 |
| merged tail | first-hit radius >= 500 mm | 768 | 92.27 | 90.80 | 89.19 | 81.90 | 90.62 | 92.32 | +1.48 +- 1.18 |
| merged tail | dR < 0.05 multiplicity >= 2 | 128 | 89.32 | 84.11 | 87.24 | 44.53 | 83.59 | 87.50 | +5.21 +- 2.81 |

## 7. G2 and G3

Evaluated at step T4 on the retrained models (`paired_claim`, `g2_g3`), as written and without iso event 1592 (deviation 14). Check of the code path: `paired_claim` with G1's models reproduces `addendum_g_summary.g1` exactly.
