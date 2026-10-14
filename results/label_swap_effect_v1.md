# Cost of the hit-label exchange on validation (val only)

Job 6810635 (CPU, express; A30 and GTX1080 nodes excluded). Val only.
Script: scripts/diagnostics/label_swap_effect.py; exchange flags from label_swap_ransac.py (pT >= 1 GeV, tolerance 1 mm); matching from the stored per-particle tables.

## Canto-L (ttbar) s0, LM 0.05

- HNL daughters (primary population): all 98.87 +- 0.29 % (1316/1331); without an exchanged hit: 99.18 +- 0.29 % (971/979); with an exchanged hit: 98.64 +- 0.96 % (145/147); no RANSAC fit: 97.56 +- 1.08 % (200/205).
- V0 daughters (HNL val): all 99.04 +- 0.21 % (2176/2197); without an exchanged hit: 99.63 +- 0.16 % (1354/1359); with an exchanged hit: 98.75 +- 0.41 % (712/721); no RANSAC fit: 94.02 +- 2.19 % (110/117).
- ttbar targets, 1 <= |d0| < 100 mm: all 68.83 +- 0.51 % (5789/8411); without an exchanged hit: 94.22 +- 0.71 % (1027/1090); with an exchanged hit: 81.11 +- 0.54 % (4242/5230); no RANSAC fit: 24.87 +- 0.95 % (520/2091).

## acorn (HNL) s0, LM 0.03

- HNL daughters (primary population): all 98.80 +- 0.30 % (1315/1331); without an exchanged hit: 99.08 +- 0.31 % (970/979); with an exchanged hit: 99.32 +- 0.68 % (146/147); no RANSAC fit: 97.07 +- 1.18 % (199/205).
- V0 daughters (HNL val): all 98.22 +- 0.28 % (2158/2197); without an exchanged hit: 99.41 +- 0.21 % (1351/1359); with an exchanged hit: 97.36 +- 0.60 % (702/721); no RANSAC fit: 89.74 +- 2.80 % (105/117).
- ttbar targets, 1 <= |d0| < 100 mm: all 66.59 +- 0.51 % (5601/8411); without an exchanged hit: 92.02 +- 0.82 % (1003/1090); with an exchanged hit: 78.18 +- 0.57 % (4089/5230); no RANSAC fit: 24.34 +- 0.94 % (509/2091).

## Identity checks (primary efficiency against results/strong_baselines_v3_val_entries.csv)

- Canto-L (ttbar) s0, LM 0.05: 0.988730277986 vs 0.988730277986: ok
- acorn (HNL) s0, LM 0.03: 0.987978963186 vs 0.987978963186: ok

- HNL daughters (primary population): 1331 targets; with an exchanged hit 11.04 +- 0.86 % (147/1331), without 73.55 +- 1.21 % (979/1331), no fit 15.40 +- 0.99 % (205/1331).
- V0 daughters (HNL val): 2197 targets; with an exchanged hit 32.82 +- 1.00 % (721/2197), without 61.86 +- 1.04 % (1359/2197), no fit 5.33 +- 0.48 % (117/2197).
- ttbar targets, 1 <= |d0| < 100 mm: 8411 targets; with an exchanged hit 62.18 +- 0.53 % (5230/8411), without 12.96 +- 0.37 % (1090/8411), no fit 24.86 +- 0.47 % (2091/8411).
