# Addendum G: read-outs of T1 deviation 15 (val; no claim)

Written 2026-10-14 20:11 UTC by `hnl.eval.addendum_g_readouts` (git 1c33269); definitions in the module docstring and `docs/addendum_g_t2_readouts.md`. Efficiencies in percent (3-seed means +- sigma); Delta = Canto-L (ttbar) - acorn (HNL) + LM in points. Descriptive: none of these numbers changes a claim. Val read-out: a test of the code on the frozen models' val outputs; nothing is chosen from it.

## 1. Sign of the G1 difference per bin (P_G, generic_iso)

| variable | bin | n | A | B | Delta | sign |
|---|---|---|---|---|---|---|
| |d0| [mm] | [0, 0.1) | 1 | 100.00 | 100.00 | +0.00 +- 0.00 | 0 |
| |d0| [mm] | [0.1, 1) | 14 | 100.00 | 100.00 | +0.00 +- 0.00 | 0 |
| |d0| [mm] | [1, 3) | 48 | 100.00 | 99.31 | +0.69 +- 0.98 | + |
| |d0| [mm] | [3, 10) | 118 | 98.87 | 98.02 | +0.85 +- 1.56 | + |
| |d0| [mm] | [10, 30) | 290 | 98.97 | 97.36 | +1.61 +- 0.88 | + |
| |d0| [mm] | [30, 100) | 310 | 96.99 | 94.30 | +2.69 +- 0.98 | + |
| |d0| [mm] | [100, 300) | 137 | 93.67 | 88.32 | +5.35 +- 1.95 | + |
| |d0| [mm] | [300, inf) | 38 | 75.44 | 62.28 | absolute only | not ranked |
| r_v [mm] | [10, 30) | 9 | 100.00 | 100.00 | +0.00 +- 0.00 | 0 |
| r_v [mm] | [30, 100) | 121 | 99.72 | 96.42 | +3.31 +- 1.68 | + |
| r_v [mm] | [100, 200) | 265 | 98.62 | 96.23 | +2.39 +- 0.86 | + |
| r_v [mm] | [200, 300) | 244 | 96.58 | 94.40 | +2.19 +- 1.21 | + |
| r_v [mm] | [300, 450) | 219 | 95.89 | 93.30 | +2.59 +- 1.09 | + |
| r_v [mm] | [450, 650) | 91 | 92.67 | 87.18 | +5.49 +- 3.03 | + |
| r_v [mm] | [650, inf) | 7 | 47.62 | 42.86 | absolute only | not ranked |
| first-hit radius [mm] | [0, 50) | 21 | 98.41 | 95.24 | +3.17 +- 5.24 | + |
| first-hit radius [mm] | [50, 100) | 63 | 100.00 | 96.83 | +3.17 +- 2.27 | + |
| first-hit radius [mm] | [100, 200) | 214 | 97.51 | 94.86 | +2.65 +- 1.23 | + |
| first-hit radius [mm] | [200, 300) | 244 | 97.54 | 95.49 | +2.05 +- 1.06 | + |
| first-hit radius [mm] | [300, 500) | 330 | 96.87 | 93.84 | +3.03 +- 1.16 | + |
| first-hit radius [mm] | [500, 700) | 83 | 88.35 | 84.34 | +4.02 +- 2.59 | + |
| first-hit radius [mm] | [700, inf) | 1 | 100.00 | 100.00 | absolute only | not ranked |
| |d0| / r_first | [0, 0.05) | 253 | 99.47 | 97.76 | +1.71 +- 1.00 | + |
| |d0| / r_first | [0.05, 0.1) | 127 | 97.64 | 97.64 | +0.00 +- 0.92 | 0 |
| |d0| / r_first | [0.1, 0.2) | 211 | 97.47 | 96.52 | +0.95 +- 0.72 | + |
| |d0| / r_first | [0.2, 0.4) | 169 | 97.83 | 93.49 | +4.34 +- 1.64 | + |
| |d0| / r_first | [0.4, 0.7) | 108 | 94.75 | 88.27 | +6.48 +- 2.43 | + |
| |d0| / r_first | [0.7, 1.01) | 88 | 85.61 | 78.79 | +6.82 +- 3.56 | + |
| pT [GeV] | [1, 2) | 186 | 93.37 | 87.28 | +6.09 +- 1.70 | + |
| pT [GeV] | [2, 5) | 270 | 94.44 | 91.11 | +3.33 +- 1.43 | + |
| pT [GeV] | [5, 10) | 169 | 97.83 | 96.25 | +1.58 +- 1.26 | + |
| pT [GeV] | [10, 20) | 141 | 99.53 | 98.11 | +1.42 +- 1.25 | + |
| pT [GeV] | [20, inf) | 190 | 100.00 | 99.12 | +0.88 +- 0.51 | + |

Ranked bins: Delta > 0 in 26, < 0 in 0, = 0 in 4 (bins of one variable are disjoint; the variables overlap).

## 2. P_G reweighted in r_v

| target | n | Kish n_eff | uncovered target fraction | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|---|
| Zenodo HNL r_v shape | 950 | 675 | 0.001 | 98.26 +- 0.35 | 95.64 +- 0.66 | 93.18 +- 1.23 | 85.87 +- 1.38 | 97.86 +- 0.54 | 97.39 +- 0.55 | +2.62 +- 0.59 |
| exponential, lambda = 100 mm | 956 | 429 | 0.000 | 98.88 +- 0.33 | 96.44 +- 0.67 | 94.35 +- 1.21 | 86.47 +- 1.52 | 97.71 +- 0.75 | 97.87 +- 0.61 | +2.43 +- 0.66 |
| exponential, lambda = 300 mm | 956 | 814 | 0.000 | 97.82 +- 0.41 | 95.23 +- 0.64 | 92.61 +- 1.21 | 85.36 +- 1.33 | 97.73 +- 0.52 | 97.06 +- 0.56 | +2.59 +- 0.54 |

## 3. P_G by species, pT and |eta| (generic_iso)

| variable | bin | n | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|
| species | e | 126 | 98.41 | 95.77 | 94.71 | 84.66 | 98.41 | 99.21 | +2.65 +- 1.58 |
| species | mu | 212 | 96.54 | 93.71 | 92.92 | 82.39 | 98.11 | 94.34 | +2.83 +- 1.50 |
| species | pi | 180 | 95.93 | 93.33 | 91.11 | 82.41 | 96.67 | 95.56 | +2.59 +- 1.01 |
| species | K | 202 | 95.54 | 93.40 | 89.44 | 86.47 | 97.52 | 96.53 | +2.15 +- 1.23 |
| species | p | 236 | 97.46 | 93.93 | 89.69 | 85.03 | 97.03 | 97.03 | +3.53 +- 1.13 |
| pT [GeV] | [1, 2) | 186 | 93.37 | 87.28 | 83.87 | 77.60 | 96.77 | 92.47 | +6.09 +- 1.70 |
| pT [GeV] | [2, 5) | 270 | 94.44 | 91.11 | 87.04 | 81.85 | 95.19 | 94.44 | +3.33 +- 1.43 |
| pT [GeV] | [5, 10) | 169 | 97.83 | 96.25 | 91.91 | 86.19 | 98.22 | 97.04 | +1.58 +- 1.26 |
| pT [GeV] | [10, 20) | 141 | 99.53 | 98.11 | 96.93 | 90.07 | 99.29 | 99.29 | +1.42 +- 1.25 |
| pT [GeV] | [20, inf) | 190 | 100.00 | 99.12 | 99.82 | 87.89 | 99.47 | 100.00 | +0.88 +- 0.51 |
| |eta| | [0, 0.5) | 269 | 97.15 | 94.05 | 91.95 | 81.78 | 97.40 | 95.91 | +3.10 +- 1.20 |
| |eta| | [0.5, 1) | 248 | 95.03 | 92.74 | 90.19 | 79.57 | 96.37 | 93.15 | +2.28 +- 0.99 |
| |eta| | [1, 1.5) | 162 | 94.03 | 90.33 | 86.42 | 83.54 | 96.30 | 96.91 | +3.70 +- 1.40 |
| |eta| | [1.5, 2) | 165 | 98.99 | 95.56 | 92.73 | 89.70 | 99.39 | 98.79 | +3.43 +- 1.44 |
| |eta| | [2, 2.5) | 83 | 100.00 | 98.80 | 98.39 | 93.57 | 98.80 | 100.00 | +1.20 +- 1.22 |
| |eta| | [2.5, 3) | 25 | 98.67 | 98.67 | 96.00 | 90.67 | 100.00 | 100.00 | +0.00 +- 1.89 |
| |eta| | [3, 4) | 4 | 100.00 | 100.00 | 75.00 | 100.00 | 100.00 | 100.00 | +0.00 +- 0.00 |

## 4. Second read-out with the Zenodo same-module label exchange emulated

Efficiency (fake rate) with the generated labels -> with the emulated exchange, from the stored tracks.

| sample | population | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) |
|---|---|---|---|---|---|---|---|
| generic_iso | P_G | 96.69 -> 96.55 | 93.90 -> 93.58 | 91.28 -> 91.04 | 84.21 -> 84.00 | 97.49 -> 97.49 | 96.34 -> 96.23 |
| generic_iso | displaced-source targets | 97.12 -> 97.00 | 94.75 -> 94.48 | 92.47 -> 92.26 | 85.06 -> 84.73 | 97.84 -> 97.84 | 96.85 -> 96.76 |
| generic_iso | all targets | 97.58 -> 95.74 | 97.30 -> 95.14 | 96.91 -> 94.84 | 82.82 -> 79.48 | 98.52 -> 96.90 | 97.96 -> 96.17 |
| generic_iso | prompt (ttbar, |d0| < 1 mm) | 97.70 -> 95.93 | 97.57 -> 95.47 | 97.16 -> 95.15 | 83.51 -> 80.19 | 98.69 -> 97.12 | 98.07 -> 96.35 |
| generic_iso | track fake rate | 0.52 -> 1.49 | 1.37 -> 2.54 | 1.12 -> 2.26 | 4.23 -> 6.43 | 0.46 -> 1.33 | 0.47 -> 1.41 |
| generic_dense | P_dense | 96.86 -> 94.61 | 94.15 -> 91.53 | 91.52 -> 89.03 | 82.54 -> 79.41 | 97.20 -> 94.65 | 97.60 -> 95.41 |
| generic_dense | displaced-source targets | 97.31 -> 95.29 | 94.84 -> 92.39 | 92.65 -> 90.35 | 82.67 -> 79.55 | 97.54 -> 95.29 | 97.94 -> 96.03 |
| generic_dense | all targets | 97.88 -> 96.01 | 97.41 -> 95.21 | 96.97 -> 94.84 | 82.75 -> 79.38 | 98.62 -> 96.93 | 98.30 -> 96.42 |
| generic_dense | prompt (ttbar, |d0| < 1 mm) | 98.00 -> 96.23 | 97.74 -> 95.65 | 97.36 -> 95.32 | 83.39 -> 80.06 | 98.79 -> 97.25 | 98.42 -> 96.63 |
| generic_dense | track fake rate | 0.54 -> 1.55 | 1.43 -> 2.68 | 1.21 -> 2.43 | 4.40 -> 6.66 | 0.48 -> 1.41 | 0.47 -> 1.47 |

G1 rule, true labels: Delta = +2.79 +- 0.58, fake-rate difference -0.85 points; the G1 rule would read: holds.

G1 rule, emulated exchange: Delta = +2.96 +- 0.59, fake-rate difference -1.06 points; the G1 rule would read: holds.

Emulated exchange: 6.71 % of the hits carry another label; stored tracks reproduce the stored matching in every event (0 differences).

## 5. P_G under the hit-fit |d0| and the fitted r_v (generic_iso)

| variable | bin | n | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|
| overlap | P_G (exact) and P_G (fit) | 940 |  |  |  |  |  |  |  |
| overlap | P_G (exact) only | 16 |  |  |  |  |  |  |  |
| overlap | P_G (fit) only | 0 |  |  |  |  |  |  |  |
| population | P_G (exact) | 956 | 96.69 | 93.90 | 91.28 | 84.21 | 97.49 | 96.34 | +2.79 +- 0.58 |
| population | P_G (fit) | 940 | 96.63 | 94.01 | 91.35 | 84.36 | 97.45 | 96.28 | +2.62 +- 0.54 |
| |d0| fitted [mm] | [0.1, 1) | 11 | 100.00 | 100.00 | 100.00 | 96.97 | 100.00 | 100.00 | +0.00 +- 0.00 |
| |d0| fitted [mm] | [1, 3) | 42 | 100.00 | 99.21 | 100.00 | 89.68 | 100.00 | 100.00 | +0.79 +- 1.12 |
| |d0| fitted [mm] | [3, 10) | 113 | 98.82 | 99.12 | 99.12 | 87.02 | 99.12 | 98.23 | -0.29 +- 1.21 |
| |d0| fitted [mm] | [10, 30) | 285 | 98.95 | 97.54 | 96.73 | 86.32 | 97.89 | 97.54 | +1.40 +- 0.83 |
| |d0| fitted [mm] | [30, 100) | 313 | 97.02 | 94.36 | 90.42 | 85.09 | 98.40 | 96.81 | +2.66 +- 0.97 |
| |d0| fitted [mm] | [100, 300) | 138 | 93.72 | 88.41 | 81.40 | 82.37 | 94.20 | 92.75 | +5.31 +- 1.93 |
| |d0| fitted [mm] | [300, inf) | 38 | 75.44 | 62.28 | 59.65 | 53.51 | 89.47 | 84.21 |  |
| r_v fitted [mm] | [10, 30) | 10 | 100.00 | 100.00 | 100.00 | 96.67 | 100.00 | 100.00 | +0.00 +- 0.00 |
| r_v fitted [mm] | [30, 100) | 99 | 100.00 | 98.65 | 98.65 | 88.55 | 97.98 | 100.00 | +1.35 +- 1.08 |
| r_v fitted [mm] | [100, 200) | 223 | 99.40 | 98.51 | 97.46 | 88.79 | 99.10 | 98.65 | +0.90 +- 0.54 |
| r_v fitted [mm] | [200, 300) | 190 | 97.54 | 97.19 | 94.21 | 89.30 | 97.89 | 95.79 | +0.35 +- 0.96 |
| r_v fitted [mm] | [300, 450) | 186 | 98.21 | 96.06 | 93.19 | 85.48 | 98.92 | 97.85 | +2.15 +- 1.06 |
| r_v fitted [mm] | [450, 650) | 72 | 96.30 | 93.06 | 94.91 | 79.63 | 98.61 | 95.83 | +3.24 +- 2.05 |
| r_v fitted [mm] | no vertex | 160 | 87.71 | 78.75 | 70.62 | 69.79 | 91.88 | 89.38 | +8.96 +- 2.32 |

## 6. Dense component: bins chosen from truth (P_dense, generic_dense)

Preregistered bins: `hnl.eval.addendum_g_summary` (bins csv). Delta = Canto-L (ttbar) - acorn (HNL) + LM, descriptive.

| variable | bin | n | Canto-L (ttbar) | acorn (HNL) + LM | acorn (ttbar) + LM | acorn (HNL) + CC | Canto-L (HNL) | Canto-L (ttbar+HNL) | Delta |
|---|---|---|---|---|---|---|---|---|---|
| same-vertex multiplicity | 1 | 231 | 94.52 | 90.62 | 84.56 | 82.83 | 96.10 | 96.97 | +3.90 +- 1.68 |
| same-vertex multiplicity | 2 | 331 | 94.66 | 91.34 | 87.11 | 83.48 | 96.98 | 96.68 | +3.32 +- 1.40 |
| same-vertex multiplicity | 3 | 470 | 98.09 | 95.82 | 93.40 | 84.75 | 98.30 | 98.30 | +2.27 +- 0.83 |
| same-vertex multiplicity | 4 | 590 | 97.85 | 95.37 | 93.16 | 84.01 | 98.14 | 98.47 | +2.49 +- 0.78 |
| same-vertex multiplicity | >= 5 | 1125 | 96.95 | 94.37 | 92.59 | 80.50 | 96.53 | 97.24 | +2.58 +- 0.96 |
| other targets within dR < 0.2 | 0 | 1568 | 96.81 | 93.75 | 89.46 | 85.27 | 97.70 | 97.64 | +3.06 +- 0.60 |
| other targets within dR < 0.2 | 1 | 527 | 97.79 | 95.76 | 94.94 | 84.57 | 98.86 | 98.48 | +2.02 +- 0.80 |
| other targets within dR < 0.2 | 2 | 244 | 98.36 | 96.45 | 95.90 | 83.47 | 97.13 | 97.95 | +1.91 +- 1.29 |
| other targets within dR < 0.2 | 3-4 | 240 | 97.64 | 97.36 | 95.97 | 75.14 | 97.08 | 98.75 | +0.28 +- 1.37 |
| other targets within dR < 0.2 | >= 5 | 168 | 91.07 | 84.92 | 87.30 | 59.92 | 87.50 | 92.26 | +6.15 +- 3.08 |
| merged tail | r_v >= 450 mm | 270 | 90.37 | 90.37 | 83.21 | 81.48 | 90.00 | 92.22 | +0.00 +- 1.60 |
| merged tail | first-hit radius >= 500 mm | 289 | 92.04 | 90.89 | 87.08 | 83.39 | 91.00 | 93.08 | +1.15 +- 1.37 |
| merged tail | dR < 0.05 multiplicity >= 2 | 59 | 87.01 | 81.36 | 81.36 | 44.63 | 77.97 | 89.83 | +5.65 +- 6.38 |

## 7. G2 and G3

Evaluated at step T4 on the retrained models (`paired_claim`, `g2_g3`), as written and without iso event 1592 (deviation 14). Check of the code path: `paired_claim` with G1's models reproduces `addendum_g_summary.g1` exactly.
