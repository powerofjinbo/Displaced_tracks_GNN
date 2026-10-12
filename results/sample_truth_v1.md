# Truth of the samples (sample_truth_v1)

Written by scripts/diagnostics/sample_truth.py (truth tables only: derived/events/v1/particles, derived/truth_lookup/v1, splits/). Target = pT > 1 GeV, |eta| < 4, >= 5 hits. r_first = transverse radius of the first hit of the particle's chain; "first hit beyond 100 mm" means r_first > 100 mm. Shares are over every target of the population (targets without a |d0| fit count as not >= 10 mm). Fig. 2 (paper/figures/fig15_samples.pdf) plots results/sample_truth_v1_hist.csv.

Generated 2026-10-08 16:38 EDT; Slurm job 6745295.

## Quoted values (one per line)

- SA01 ttbar training targets with |d0| >= 10 mm: 3.9 % (19250 of 494733)
- SA02 ttbar training targets with first hit beyond 100 mm: 2.7 % (13246 of 494733)
- SA03 HNL-daughter targets with first hit beyond 100 mm, all events: 53-62 % (10 GeV 100 mm / 15 GeV 200 mm / 15 GeV 100 mm ext.: 59 / 62 / 53 %, 5522 / 4921 / 2995 of 9322 / 7931 / 5623)
- SA04 targets in Fig. 2: ttbar training events 494733 (7999 events); HNL-daughter targets, all events, 10 GeV 100 mm / 15 GeV 200 mm / 15 GeV 100 mm ext.: 9322 / 7931 / 5623; W-muon targets of the HNL samples, all events: 23069
- SA05 primary population per seed, test, 10 GeV 100 mm / 15 GeV 200 mm / 15 GeV 100 mm ext.: 603 / 549 / 3861 (pooled 5013)

## Checks

- targets per event, ttbar: 61.8 (617744 / 9999 events); reference 61.8 (D36): equal
- targets per event, hnl10_100: 0.932 (9322 / 9999 events); reference 0.932 (D23): equal
- targets per event, hnl15_200: 0.793 (7931 / 9999 events); reference 0.793 (D23): equal
- targets per event, hnl15_100: 1.125 (5623 / 4999 events); reference 1.125 (D23): equal
- primary population, hnl10_100_val: 692; results/per_sample_v1.csv n_per_seed 692: equal
- primary population, hnl15_200_val: 639; results/per_sample_v1.csv n_per_seed 639: equal
- primary population, hnl10_100_test: 603; results/per_sample_v1.csv n_per_seed 603: equal
- primary population, hnl15_200_test: 549; results/per_sample_v1.csv n_per_seed 549: equal
- primary population, hnl15_100_external: 3861; results/per_sample_v1.csv n_per_seed 3861: equal
- primary population, pooled test sets: 5013; results/per_sample_v1.csv hnl_test_pooled n_per_seed 5013: equal
- primary population per seed, validation, 10 GeV 100 mm / 15 GeV 200 mm: 692 / 639 (NUMBERS D31)
- ttbar training events with at least one target: 7999; splits/ttbar_train.txt: 7999
- result: all checks passed

## Per population

| population | targets | \|d0\| >= 10 mm (%) | first hit > 100 mm (%) | first hit > 200 mm (%) | median pT (GeV) | median \|d0\| (mm) | median r_first (mm) | median hits | no \|d0\| fit |
|---|---|---|---|---|---|---|---|---|---|
| ttbar targets, 7999 training events | 494733 | 3.89 | 2.68 | 1.87 | 2.12 | 0.138 | 32.0 | 12 | 0 |
| HNL-daughter targets, 10 GeV, 100 mm (all events) | 9322 | 60.06 | 59.24 | 33.78 | 8.34 | 15.3 | 116.4 | 9 | 0 |
| HNL-daughter targets, 15 GeV, 200 mm (all events) | 7931 | 68.73 | 62.05 | 37.11 | 8.72 | 23.2 | 125.9 | 8 | 0 |
| HNL-daughter targets, 15 GeV, 100 mm, external (all events) | 5623 | 63.79 | 53.26 | 28.67 | 8.94 | 17.6 | 115.3 | 9 | 0 |
| W-muon targets of the three HNL samples (all events) | 23069 | 0.12 | 0.00 | 0.00 | 30.29 | 0.0186 | 32.0 | 12 | 0 |

Targets per |d0| bin (mm):

| population | [0, 1) | [1, 10) | [10, 30) | [30, 100) | [100, 300) | [300, inf) | max \|d0\| (mm) |
|---|---|---|---|---|---|---|---|
| ttbar targets, 7999 training events | 424626 | 50857 | 13238 | 4926 | 964 | 122 | 655 |
| HNL-daughter targets, 10 GeV, 100 mm (all events) | 805 | 2918 | 2580 | 2362 | 626 | 31 | 811 |
| HNL-daughter targets, 15 GeV, 200 mm (all events) | 494 | 1986 | 2020 | 2429 | 908 | 94 | 830 |
| HNL-daughter targets, 15 GeV, 100 mm, external (all events) | 447 | 1589 | 1533 | 1595 | 433 | 26 | 619 |
| W-muon targets of the three HNL samples (all events) | 22914 | 127 | 21 | 7 | 0 | 0 | 41 |

Outside the plotted ranges of Fig. 2 (counted in the denominators, not drawn): ttbar_train_targets: |d0| < 1e-3 mm 2702, |d0| >= 1e3 mm 0, pT >= 316 GeV 1, max r_first 825 mm; hnl10_100_daughters: |d0| < 1e-3 mm 2, |d0| >= 1e3 mm 0, pT >= 316 GeV 0, max r_first 876 mm; hnl15_200_daughters: |d0| < 1e-3 mm 1, |d0| >= 1e3 mm 0, pT >= 316 GeV 0, max r_first 1020 mm; hnl15_100_daughters: |d0| < 1e-3 mm 1, |d0| >= 1e3 mm 0, pT >= 316 GeV 0, max r_first 768 mm; w_muons_hnl: |d0| < 1e-3 mm 669, |d0| >= 1e3 mm 0, pT >= 316 GeV 0, max r_first 100 mm.

Histogram file: results/sample_truth_v1_hist.csv (5785 rows; panels absd0 and pt: fraction of the population's targets per bin, value = count / n_total; panel r_first_survival: fraction with r_first > x_lo on a 1 mm grid).
