# Displaced muons after Wang et al.'s Fig. 3 (post-freeze, descriptive)

Selection fixed on truth before this read-out (results/wang_displaced_selection_v2.md; ledger R12): Wang et al.'s 'displaced' set reads as one HNL daughter per event with >= 5 hits and |eta| < 4 and no pT requirement (2,887 positive daughters on the 4,999 events against their 2,882 plus two bins whose totals the figure does not give). Wang et al. published 91.6 ± 0.5 % for their attention GNN + DBSCAN.

Percent; sigma as in results/wang_gap_v1.md.

| selection | n per seed | Canto-L (ttbar) (LM, frozen) | Canto-L (HNL) (LM) | Canto-L (ttbar+HNL) (LM) | acorn (HNL) + LM | acorn (HNL) + CC at s* (Wang et al.'s reproduction) | acorn (HNL) + CC, best cut <= FR* | acorn (ttbar) + LM | acorn (ttbar) + CC | Canto-L (ttbar) classifier + CC |
|---|---|---|---|---|---|---|---|---|---|---|
| mu+ daughter, >= 5 hits, |eta| < 4 (headline) | 2887 | 99.11 ± 0.15 | 98.79 ± 0.20 / 0.23 | 99.10 ± 0.18 / 0.21 | 98.01 ± 0.23 | 95.08 ± 0.42 | 94.74 ± 0.58 | 96.80 ± 0.48 | 91.44 ± 0.85 | 98.43 ± 0.28 |
| all daughters, >= 5 hits, |eta| < 4 | 5786 | 99.08 ± 0.15 | 98.89 ± 0.16 / 0.23 | 99.03 ± 0.15 / 0.22 | 98.19 ± 0.17 | 94.98 ± 0.44 | 94.71 ± 0.61 | 96.81 ± 0.49 | 91.49 ± 0.81 | 98.21 ± 0.25 |
| mu- daughter, >= 5 hits, |eta| < 4 | 2899 | 99.05 ± 0.21 | 99.00 ± 0.19 / 0.30 | 98.97 ± 0.19 / 0.30 | 98.36 ± 0.22 | 94.88 ± 0.61 | 94.68 ± 0.75 | 96.81 ± 0.61 | 91.54 ± 0.89 | 97.99 ± 0.29 |
| daughter with the W muon's charge, >= 5 hits, |eta| < 4 | 2901 | 99.13 ± 0.17 | 99.00 ± 0.19 / 0.23 | 99.17 ± 0.18 / 0.22 | 98.24 ± 0.24 | 95.23 ± 0.58 | 95.02 ± 0.76 | 96.89 ± 0.55 | 91.61 ± 0.75 | 98.26 ± 0.24 |
| daughter with the opposite charge, >= 5 hits, |eta| < 4 | 2885 | 99.03 ± 0.19 | 98.79 ± 0.20 / 0.28 | 98.89 ± 0.19 / 0.27 | 98.13 ± 0.21 | 94.73 ± 0.41 | 94.38 ± 0.55 | 96.73 ± 0.53 | 91.37 ± 0.99 | 98.15 ± 0.35 |
| mu+ daughter target (pT > 1 GeV, |eta| < 4, >= 5 hits) | 2803 | 99.27 ± 0.15 | 98.89 ± 0.20 / 0.23 | 99.18 ± 0.17 / 0.21 | 98.32 ± 0.22 | 95.49 ± 0.42 | 95.15 ± 0.60 | 97.32 ± 0.42 | 92.18 ± 0.77 | 98.58 ± 0.30 |
| all daughter targets | 5623 | 99.22 ± 0.15 | 98.97 ± 0.15 / 0.24 | 99.13 ± 0.14 / 0.23 | 98.45 ± 0.17 | 95.41 ± 0.43 | 95.13 ± 0.61 | 97.27 ± 0.43 | 92.12 ± 0.74 | 98.39 ± 0.25 |
| mu- daughter target | 2820 | 99.17 ± 0.20 | 99.04 ± 0.18 / 0.31 | 99.08 ± 0.18 / 0.31 | 98.58 ± 0.22 | 95.33 ± 0.59 | 95.12 ± 0.72 | 97.21 ± 0.53 | 92.07 ± 0.83 | 98.20 ± 0.29 |
| daughter target with the W muon's charge | 2817 | 99.24 ± 0.17 | 99.04 ± 0.18 / 0.23 | 99.25 ± 0.17 / 0.22 | 98.41 ± 0.23 | 95.54 ± 0.58 | 95.34 ± 0.78 | 97.21 ± 0.49 | 92.10 ± 0.67 | 98.39 ± 0.24 |
| daughter target with the opposite charge | 2806 | 99.20 ± 0.19 | 98.90 ± 0.19 / 0.29 | 99.00 ± 0.18 / 0.28 | 98.49 ± 0.20 | 95.28 ± 0.38 | 94.93 ± 0.53 | 97.33 ± 0.47 | 92.15 ± 0.94 | 98.40 ± 0.36 |
