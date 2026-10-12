# Primary efficiency per sample: validation and test read out separately

Primary = HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm. Every model at its committed per-seed working point (three seeds; nothing is chosen here, on validation or on test): Canto-L (frozen: learned merging; results/final_selection_v1.json); each acorn arm at its val-best configuration of the full builder menu (MWS-layer + learned merging for all nine runs; results/strong_baselines_v3_val_entries.csv and results/strong_baselines_v3_spec_test.json); acorn (HNL) + CC at the preregistered cut (the claims' reference) and acorn (ttbar) + CC (preregistered cut = best cut <= FR*). Validation only: the Canto-L classifier (Canto-L's scores on its learned triplet union graph) + CC on the Canto-L runs' eval_val, at the best cut <= FR* = 0.029368 (hnl.eval.ranking rule 'best': highest primary efficiency with fake rate <= FR*; cuts 0.35 / 0.39 / 0.39) and at the cut of the preregistered rule (rule 's4': largest fake rate <= FR*; cuts 0.26 / 0.45 / 0.44), and acorn (HNL) + CC, best cut <= FR*.

Each cell: seed mean ± total sigma, sigma^2 = sigma_boot^2 + s_seeds^2 / 3 (hnl.eval.bootstrap.seed_mean); differences new minus baseline in percentage points with the paired event bootstrap of the difference of seed means plus both seed terms (linear_combination). Validation per sample: each sample's own val events resampled (B = 1000, SeedSequence(20261008), one table per sample shared by every model and seed; the subset construction of hnl.eval.round4_checks); pooled validation: the committed val tables (B = 1000, seed 20261005; both samples' event i together), as results/strong_baselines_v3_val. Test: the preregistered tables (B = 500, seed 12345; hnl10_100_test and hnl15_200_test on the joint units, as hnl.eval.strong_baselines.per_set); post-freeze, descriptive, not claims (results/claims_test_v1.json unchanged). hnl15_100_external (m_N = 15 GeV, c tau = 100 mm) entered no training and no selection.

## Validation

### Primary efficiency (%)

| model | hnl10_100 val (1,000 events; 692 per seed) | hnl15_200 val (999 events; 639 per seed) | pooled HNL val (1,999 events; 1,331 per seed) |
|---|---|---|---|
| Canto-L (frozen: learned merging) | 99.61 ± 0.23 | 98.54 ± 0.46 | 99.10 ± 0.27 |
| acorn (HNL) val-best (MWS-layer + learned merging) | 99.13 ± 0.29 | 98.12 ± 0.50 | 98.65 ± 0.31 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | 98.89 ± 0.33 | 96.82 ± 0.68 | 97.90 ± 0.40 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | 98.17 ± 0.44 | 96.09 ± 0.68 | 97.17 ± 0.44 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 94.27 ± 0.83 | 93.38 ± 0.84 | 93.84 ± 0.60 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 93.16 ± 1.11 | 91.39 ± 1.22 | 92.31 ± 0.97 |
| Canto-L classifier + CC, best cut <= FR* (validation only) | 98.41 ± 0.43 | 97.81 ± 0.51 | 98.12 ± 0.36 |
| Canto-L classifier + CC, preregistered cut rule (validation only) | 97.16 ± 1.41 | 96.40 ± 1.32 | 96.79 ± 1.32 |
| acorn (HNL) + CC, best cut <= FR* (validation only) | 96.34 ± 1.04 | 94.84 ± 0.98 | 95.62 ± 0.88 |

### Differences (points)

| new | baseline | hnl10_100 val | hnl15_200 val | pooled HNL val |
|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) val-best (MWS-layer + learned merging) | +0.48 ± 0.24 (z 2.0) | +0.42 ± 0.52 (z 0.8) | +0.45 ± 0.29 (z 1.6) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | +0.72 ± 0.29 (z 2.5) | +1.72 ± 0.68 (z 2.5) | +1.20 ± 0.39 (z 3.1) |
| Canto-L (frozen: learned merging) | acorn (ttbar) val-best (MWS-layer + learned merging) | +1.45 ± 0.38 (z 3.8) | +2.45 ± 0.65 (z 3.8) | +1.93 ± 0.39 (z 5.0) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, preregistered cut (claims' reference) | +5.35 ± 0.83 (z 6.5) | +5.16 ± 0.85 (z 6.1) | +5.26 ± 0.59 (z 8.8) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + CC, preregistered cut (= best cut) | +6.45 ± 1.09 (z 5.9) | +7.15 ± 1.22 (z 5.9) | +6.79 ± 0.94 (z 7.2) |
| Canto-L classifier + CC, best cut <= FR* (validation only) | acorn (HNL) + CC, best cut <= FR* (validation only) | +2.07 ± 1.09 (z 1.9) | +2.97 ± 0.96 (z 3.1) | +2.50 ± 0.89 (z 2.8) |
| Canto-L classifier + CC, best cut <= FR* (validation only) | acorn (ttbar) + CC, preregistered cut (= best cut) | +5.25 ± 1.09 (z 4.8) | +6.42 ± 1.19 (z 5.4) | +5.81 ± 0.93 (z 6.3) |
| Canto-L classifier + CC, preregistered cut rule (validation only) | acorn (HNL) + CC, preregistered cut (claims' reference) | +2.89 ± 1.54 (z 1.9) | +3.03 ± 1.43 (z 2.1) | +2.96 ± 1.38 (z 2.1) |
| Canto-L classifier + CC, preregistered cut rule (validation only) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.00 ± 1.70 (z 2.3) | +5.01 ± 1.68 (z 3.0) | +4.48 ± 1.56 (z 2.9) |

### Counts per seed

Last column: the bootstrap sd of the seed mean when the sample is read out of the pooled val table (both samples' event i resampled together) instead of its own table (cross-check of the resampling choice).

| model | set | n per seed | k per seed (s0 / s1 / s2) | efficiency per seed (%) | mean ± sigma (%) | bootstrap sd | seed sd | bootstrap sd, pooled val table |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | hnl10_100 val | 692 | 689 / 690 / 689 | 99.57 / 99.71 / 99.57 | 99.61 ± 0.23 | 0.22 | 0.08 | 0.23 |
| Canto-L (frozen: learned merging) | hnl15_200 val | 639 | 627 / 631 / 631 | 98.12 / 98.75 / 98.75 | 98.54 ± 0.46 | 0.41 | 0.36 | 0.40 |
| Canto-L (frozen: learned merging) | pooled HNL val | 1331 | 1316 / 1321 / 1320 | 98.87 / 99.25 / 99.17 | 99.10 ± 0.27 | 0.24 | 0.20 | - |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl10_100 val | 692 | 687 / 685 / 686 | 99.28 / 98.99 / 99.13 | 99.13 ± 0.29 | 0.28 | 0.14 | 0.29 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl15_200 val | 639 | 628 / 625 / 628 | 98.28 / 97.81 / 98.28 | 98.12 ± 0.50 | 0.47 | 0.27 | 0.45 |
| acorn (HNL) val-best (MWS-layer + learned merging) | pooled HNL val | 1331 | 1315 / 1310 / 1314 | 98.80 / 98.42 / 98.72 | 98.65 ± 0.31 | 0.29 | 0.20 | - |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | hnl10_100 val | 692 | 683 / 684 / 686 | 98.70 / 98.84 / 99.13 | 98.89 ± 0.33 | 0.31 | 0.22 | 0.32 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | hnl15_200 val | 639 | 614 / 622 / 620 | 96.09 / 97.34 / 97.03 | 96.82 ± 0.68 | 0.57 | 0.65 | 0.57 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | pooled HNL val | 1331 | 1297 / 1306 / 1306 | 97.45 / 98.12 / 98.12 | 97.90 ± 0.40 | 0.34 | 0.39 | - |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl10_100 val | 692 | 681 / 678 / 679 | 98.41 / 97.98 / 98.12 | 98.17 ± 0.44 | 0.43 | 0.22 | 0.43 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl15_200 val | 639 | 615 / 612 / 615 | 96.24 / 95.77 / 96.24 | 96.09 ± 0.68 | 0.66 | 0.27 | 0.67 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | pooled HNL val | 1331 | 1296 / 1290 / 1294 | 97.37 / 96.92 / 97.22 | 97.17 ± 0.44 | 0.41 | 0.23 | - |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl10_100 val | 692 | 654 / 655 / 648 | 94.51 / 94.65 / 93.64 | 94.27 ± 0.83 | 0.77 | 0.55 | 0.72 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl15_200 val | 639 | 593 / 598 / 599 | 92.80 / 93.58 / 93.74 | 93.38 ± 0.84 | 0.79 | 0.50 | 0.79 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | pooled HNL val | 1331 | 1247 / 1253 / 1247 | 93.69 / 94.14 / 93.69 | 93.84 ± 0.60 | 0.58 | 0.26 | - |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl10_100 val | 692 | 656 / 638 / 640 | 94.80 / 92.20 / 92.49 | 93.16 ± 1.11 | 0.74 | 1.43 | 0.78 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl15_200 val | 639 | 590 / 574 / 588 | 92.33 / 89.83 / 92.02 | 91.39 ± 1.22 | 0.94 | 1.36 | 0.95 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | pooled HNL val | 1331 | 1246 / 1212 / 1228 | 93.61 / 91.06 / 92.26 | 92.31 ± 0.97 | 0.63 | 1.28 | - |
| Canto-L classifier + CC, best cut <= FR* (validation only) | hnl10_100 val | 692 | 680 / 682 / 681 | 98.27 / 98.55 / 98.41 | 98.41 ± 0.43 | 0.42 | 0.14 | 0.41 |
| Canto-L classifier + CC, best cut <= FR* (validation only) | hnl15_200 val | 639 | 624 / 626 / 625 | 97.65 / 97.97 / 97.81 | 97.81 ± 0.51 | 0.50 | 0.16 | 0.49 |
| Canto-L classifier + CC, best cut <= FR* (validation only) | pooled HNL val | 1331 | 1304 / 1308 / 1306 | 97.97 / 98.27 / 98.12 | 98.12 ± 0.36 | 0.35 | 0.15 | - |
| Canto-L classifier + CC, preregistered cut rule (validation only) | hnl10_100 val | 692 | 654 / 683 / 680 | 94.51 / 98.70 / 98.27 | 97.16 ± 1.41 | 0.46 | 2.30 | 0.44 |
| Canto-L classifier + CC, preregistered cut rule (validation only) | hnl15_200 val | 639 | 601 / 624 / 623 | 94.05 / 97.65 / 97.50 | 96.40 ± 1.32 | 0.61 | 2.03 | 0.61 |
| Canto-L classifier + CC, preregistered cut rule (validation only) | pooled HNL val | 1331 | 1255 / 1307 / 1303 | 94.29 / 98.20 / 97.90 | 96.79 ± 1.32 | 0.41 | 2.17 | - |
| acorn (HNL) + CC, best cut <= FR* (validation only) | hnl10_100 val | 692 | 675 / 655 / 670 | 97.54 / 94.65 / 96.82 | 96.34 ± 1.04 | 0.56 | 1.50 | 0.56 |
| acorn (HNL) + CC, best cut <= FR* (validation only) | hnl15_200 val | 639 | 607 / 598 / 613 | 94.99 / 93.58 / 95.93 | 94.84 ± 0.98 | 0.70 | 1.18 | 0.72 |
| acorn (HNL) + CC, best cut <= FR* (validation only) | pooled HNL val | 1331 | 1282 / 1253 / 1283 | 96.32 / 94.14 / 96.39 | 95.62 ± 0.88 | 0.48 | 1.28 | - |

## Test (post-freeze, descriptive; not claims)

### Primary efficiency (%)

| model | hnl10_100_test (1,000 events; 603 per seed) | hnl15_200_test (1,000 events; 549 per seed) | hnl15_100_external (4,999 events; 3,861 per seed) | pooled HNL test (6,999 events; 5,013 per seed) |
|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 99.23 ± 0.33 | 98.48 ± 0.47 | 99.14 ± 0.17 | 99.08 ± 0.15 |
| acorn (HNL) val-best (MWS-layer + learned merging) | 98.45 ± 0.47 | 97.02 ± 0.70 | 98.04 ± 0.22 | 97.98 ± 0.20 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | 97.51 ± 0.75 | 96.17 ± 0.90 | 97.51 ± 0.28 | 97.37 ± 0.31 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | 97.24 ± 0.65 | 94.54 ± 1.02 | 96.37 ± 0.61 | 96.28 ± 0.56 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 92.70 ± 1.13 | 91.56 ± 1.06 | 92.70 ± 0.56 | 92.57 ± 0.47 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 91.43 ± 1.23 | 87.55 ± 2.05 | 90.62 ± 0.98 | 90.38 ± 1.02 |

### Differences (points)

| new | baseline | hnl10_100_test | hnl15_200_test | hnl15_100_external | pooled HNL test |
|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) val-best (MWS-layer + learned merging) | +0.77 ± 0.46 (z 1.7) | +1.46 ± 0.65 (z 2.2) | +1.10 ± 0.24 (z 4.5) | +1.10 ± 0.22 (z 5.1) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | +1.71 ± 0.75 (z 2.3) | +2.31 ± 0.90 (z 2.6) | +1.62 ± 0.29 (z 5.6) | +1.71 ± 0.32 (z 5.4) |
| Canto-L (frozen: learned merging) | acorn (ttbar) val-best (MWS-layer + learned merging) | +1.99 ± 0.65 (z 3.1) | +3.95 ± 0.99 (z 4.0) | +2.76 ± 0.61 (z 4.5) | +2.80 ± 0.56 (z 5.0) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, preregistered cut (claims' reference) | +6.52 ± 1.14 (z 5.7) | +6.92 ± 1.05 (z 6.6) | +6.44 ± 0.57 (z 11.3) | +6.50 ± 0.48 (z 13.5) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + CC, preregistered cut (= best cut) | +7.79 ± 1.23 (z 6.3) | +10.93 ± 2.04 (z 5.3) | +8.52 ± 0.98 (z 8.7) | +8.70 ± 1.02 (z 8.5) |

### Counts per seed

| model | set | n per seed | k per seed (s0 / s1 / s2) | efficiency per seed (%) | mean ± sigma (%) | bootstrap sd | seed sd |
|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | hnl10_100_test | 603 | 598 / 599 / 598 | 99.17 / 99.34 / 99.17 | 99.23 ± 0.33 | 0.33 | 0.10 |
| Canto-L (frozen: learned merging) | hnl15_200_test | 549 | 540 / 541 / 541 | 98.36 / 98.54 / 98.54 | 98.48 ± 0.47 | 0.47 | 0.11 |
| Canto-L (frozen: learned merging) | hnl15_100_external | 3861 | 3819 / 3834 / 3830 | 98.91 / 99.30 / 99.20 | 99.14 ± 0.17 | 0.13 | 0.20 |
| Canto-L (frozen: learned merging) | pooled HNL test | 5013 | 4957 / 4974 / 4969 | 98.88 / 99.22 / 99.12 | 99.08 ± 0.15 | 0.12 | 0.17 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl10_100_test | 603 | 595 / 594 / 592 | 98.67 / 98.51 / 98.18 | 98.45 ± 0.47 | 0.45 | 0.25 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl15_200_test | 549 | 536 / 530 / 532 | 97.63 / 96.54 / 96.90 | 97.02 ± 0.70 | 0.62 | 0.56 |
| acorn (HNL) val-best (MWS-layer + learned merging) | hnl15_100_external | 3861 | 3784 / 3779 / 3793 | 98.01 / 97.88 / 98.24 | 98.04 ± 0.22 | 0.19 | 0.18 |
| acorn (HNL) val-best (MWS-layer + learned merging) | pooled HNL test | 5013 | 4915 / 4903 / 4917 | 98.05 / 97.81 / 98.08 | 97.98 ± 0.20 | 0.18 | 0.15 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | hnl10_100_test | 603 | 585 / 585 / 594 | 97.01 / 97.01 / 98.51 | 97.51 ± 0.75 | 0.56 | 0.86 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | hnl15_200_test | 549 | 522 / 529 / 533 | 95.08 / 96.36 / 97.09 | 96.17 ± 0.90 | 0.68 | 1.01 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | hnl15_100_external | 3861 | 3754 / 3769 / 3772 | 97.23 / 97.62 / 97.69 | 97.51 ± 0.28 | 0.24 | 0.25 |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | pooled HNL test | 5013 | 4861 / 4883 / 4899 | 96.97 / 97.41 / 97.73 | 97.37 ± 0.31 | 0.21 | 0.38 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl10_100_test | 603 | 587 / 584 / 588 | 97.35 / 96.85 / 97.51 | 97.24 ± 0.65 | 0.62 | 0.35 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl15_200_test | 549 | 524 / 513 / 520 | 95.45 / 93.44 / 94.72 | 94.54 ± 1.02 | 0.84 | 1.01 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | hnl15_100_external | 3861 | 3755 / 3682 / 3726 | 97.25 / 95.36 / 96.50 | 96.37 ± 0.61 | 0.26 | 0.95 |
| acorn (ttbar) val-best (MWS-layer + learned merging) | pooled HNL test | 5013 | 4866 / 4779 / 4834 | 97.07 / 95.33 / 96.43 | 96.28 ± 0.56 | 0.23 | 0.88 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl10_100_test | 603 | 552 / 559 / 566 | 91.54 / 92.70 / 93.86 | 92.70 ± 1.13 | 0.91 | 1.16 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl15_200_test | 549 | 507 / 502 / 499 | 92.35 / 91.44 / 90.89 | 91.56 ± 1.06 | 0.97 | 0.74 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | hnl15_100_external | 3861 | 3550 / 3603 / 3584 | 91.95 / 93.32 / 92.83 | 92.70 ± 0.56 | 0.39 | 0.70 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | pooled HNL test | 5013 | 4609 / 4664 / 4649 | 91.94 / 93.04 / 92.74 | 92.57 ± 0.47 | 0.34 | 0.57 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl10_100_test | 603 | 560 / 545 / 549 | 92.87 / 90.38 / 91.04 | 91.43 ± 1.23 | 0.98 | 1.29 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl15_200_test | 549 | 499 / 470 / 473 | 90.89 / 85.61 / 86.16 | 87.55 ± 2.05 | 1.18 | 2.90 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | hnl15_100_external | 3861 | 3557 / 3437 / 3502 | 92.13 / 89.02 / 90.70 | 90.62 ± 0.98 | 0.40 | 1.56 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | pooled HNL test | 5013 | 4616 / 4452 / 4524 | 92.08 / 88.81 / 90.25 | 90.38 ± 1.02 | 0.37 | 1.64 |

## Validation to test, per labelled sample

Test minus validation (points; the mean of the per-seed changes, seed i the same trained network on both). The validation and test tables are independent: sigma^2 = sigma_boot,val^2 + sigma_boot,test^2 + s^2 / 3, s the seed sd of the per-seed changes; for the change of a difference the bootstrap terms are those of the paired differences on validation and on test and the seed term is s_new^2 / 3 + s_base^2 / 3.

### Change of each model

| model | hnl10_100: val -> test (%) | change (points) | hnl15_200: val -> test (%) | change (points) |
|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 99.61 -> 99.23 | -0.39 ± 0.40 (z -1.0) | 98.54 -> 98.48 | -0.06 ± 0.64 (z -0.1) |
| acorn (HNL) val-best (MWS-layer + learned merging) | 99.13 -> 98.45 | -0.68 ± 0.55 (z -1.2) | 98.12 -> 97.02 | -1.10 ± 0.81 (z -1.3) |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | 98.89 -> 97.51 | -1.38 ± 0.74 (z -1.9) | 96.82 -> 96.17 | -0.64 ± 0.96 (z -0.7) |
| acorn (ttbar) val-best (MWS-layer + learned merging) | 98.17 -> 97.24 | -0.93 ± 0.77 (z -1.2) | 96.09 -> 94.54 | -1.55 ± 1.15 (z -1.3) |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 94.27 -> 92.70 | -1.56 ± 1.52 (z -1.0) | 93.38 -> 91.56 | -1.81 ± 1.44 (z -1.3) |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 93.16 -> 91.43 | -1.73 ± 1.23 (z -1.4) | 91.39 -> 87.55 | -3.84 ± 1.99 (z -1.9) |

### Change of the difference (Canto-L minus baseline)

| baseline | hnl10_100: val -> test difference (points) | change (points) | hnl15_200: val -> test difference (points) | change (points) |
|---|---|---|---|---|
| acorn (HNL) val-best (MWS-layer + learned merging) | +0.48 -> +0.77 | +0.29 ± 0.51 (z 0.6) | +0.42 -> +1.46 | +1.04 ± 0.77 (z 1.4) |
| acorn (ttbar+HNL) val-best (MWS-layer + learned merging) | +0.72 -> +1.71 | +0.99 ± 0.72 (z 1.4) | +1.72 -> +2.31 | +0.59 ± 0.94 (z 0.6) |
| acorn (ttbar) val-best (MWS-layer + learned merging) | +1.45 -> +1.99 | +0.54 ± 0.73 (z 0.8) | +2.45 -> +3.95 | +1.49 ± 1.10 (z 1.4) |
| acorn (HNL) + CC, preregistered cut (claims' reference) | +5.35 -> +6.52 | +1.18 ± 1.52 (z 0.8) | +5.16 -> +6.92 | +1.76 ± 1.43 (z 1.2) |
| acorn (ttbar) + CC, preregistered cut (= best cut) | +6.45 -> +7.79 | +1.34 ± 1.22 (z 1.1) | +7.15 -> +10.93 | +3.78 ± 1.97 (z 1.9) |

## Checks

745 of 745 identity checks passed (tolerance 1e-12, counts exact; results/per_sample_v1_checks.csv): pooled val and test rows and differences reproduce results/strong_baselines_v3*; the per-set test rows reproduce hnl.eval.strong_baselines.per_set for every pair and results/strong_baselines_v3_test_per_set.csv; per-sample counts add up to the pooled counts for every model and seed; working points, cuts and test-output provenance as committed; the Canto-L classifier + CC entries of rule 's4' equal the reference read-out of that row (meta: cc_reference; it was made with hnl.eval.ranking's default rule, 's4'), every column.

Slurm job 6712588, 2026-10-07T04:55:00Z - 2026-10-07T05:13:07Z (UTC), git 3d0c22bd3e68bb45886bc61c7825ada1ef8b5d7c (src/slurm dirty: 3 files).

