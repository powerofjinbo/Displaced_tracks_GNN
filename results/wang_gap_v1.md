# Wang et al.'s evaluation sample: the all-track gap by stage (post-freeze, descriptive)

hnl15_100_external, 4,999 events; Wang et al.'s definitions (hnl.eval.published_defs); configurations fixed on validation; ledger R12. Code hnl.eval.wang_gap, git 50408b6646797b4f442ccf36b4081c034cc05b42 (src dirty: src/hnl/eval/secondary_truth.py, src/hnl/eval/secondary_upweight.py, src/hnl/eval/wang_gap.py, src/hnl/models/secondary_weight.py, src/hnl/trackml/eggnet_denominator.py, src/hnl/trackml/port_closure.py), job 6810554, 2026-10-14T09:56:05Z. Identity checks: 33 of 33 pass.

Percent. sigma: event bootstrap (preregistered test tables) and, for three seeds, the seed term; one-seed rows: sigma (bootstrap alone) / sigma_proxy (with the seed sd of the three-seed Canto-L (ttbar) model of the same builder in quadrature).

## Models

| model | seeds | all tracks | W muon | HNL daughters | fake rate | working points |
|---|---|---|---|---|---|---|
| acorn (HNL) + CC at s* (Wang et al.'s reproduction) | 3 | 95.54 ± 0.34 | 98.30 ± 0.19 | 95.41 ± 0.43 | 2.92 ± 0.14 | CC 0.40 / CC 0.40 / CC 0.40 |
| acorn (HNL) + CC, best cut <= FR* | 3 | 95.47 ± 0.37 | 98.25 ± 0.24 | 95.13 ± 0.61 | 2.90 ± 0.04 | CC 0.38 / CC 0.36 / CC 0.44 |
| acorn (HNL) + LM | 3 | 98.93 ± 0.04 | 99.56 ± 0.09 | 98.45 ± 0.17 | 1.36 ± 0.03 | mwslmerge 0.03 / mwslmerge 0.2 / mwslmerge 0.05 |
| acorn (ttbar) + CC | 3 | 95.66 ± 0.14 | 98.17 ± 0.18 | 92.12 ± 0.74 | 2.95 ± 0.05 | CC 0.34 / CC 0.33 / CC 0.31 |
| acorn (ttbar) + LM | 3 | 98.87 ± 0.05 | 99.55 ± 0.10 | 97.27 ± 0.43 | 1.53 ± 0.04 | mwslmerge 0.03 / mwslmerge 0.02 / mwslmerge 0.05 |
| acorn (ttbar+HNL) + LM | 3 | 98.89 ± 0.06 | 99.53 ± 0.10 | 98.05 ± 0.21 | 1.39 ± 0.05 | mwslmerge 0.2 / mwslmerge 0.03 / mwslmerge 0.05 |
| Canto-L (HNL) classifier + CC | 1 | 96.93 ± 0.08 / 0.22 | 99.05 ± 0.15 / 0.17 | 98.22 ± 0.19 / 0.40 | 2.53 ± 0.04 / 0.14 | CC 0.38 |
| Canto-L (HNL) (LM) | 1 | 99.15 ± 0.04 / 0.07 | 99.64 ± 0.09 / 0.09 | 98.97 ± 0.15 / 0.24 | 1.02 ± 0.02 / 0.04 | mwslmerge 0.2 |
| Canto-L (ttbar+HNL) classifier + CC | 1 | 96.32 ± 0.09 / 0.22 | 98.82 ± 0.16 / 0.18 | 98.27 ± 0.19 / 0.40 | 3.00 ± 0.04 / 0.14 | CC 0.40 |
| Canto-L (ttbar+HNL) (LM) | 1 | 99.12 ± 0.04 / 0.07 | 99.62 ± 0.09 / 0.09 | 99.13 ± 0.14 / 0.23 | 1.01 ± 0.02 / 0.04 | mwslmerge 0.3 |
| Canto-L (ttbar) classifier + CC | 3 | 97.06 ± 0.14 | 99.14 ± 0.14 | 98.39 ± 0.25 | 2.54 ± 0.09 | CC 0.35 / CC 0.39 / CC 0.39 |
| Canto-L (ttbar) (LM, frozen) | 3 | 99.14 ± 0.05 | 99.65 ± 0.09 | 99.22 ± 0.15 | 1.03 ± 0.03 | mwslmerge 0.05 / mwslmerge 0.03 / mwslmerge 0.1 |

Wang et al. (published): all tracks 95.61 ± 0.06, W muon 99.0 ± 0.2, displaced 91.6 ± 0.5 (their selection), no fake rate.

## Steps (new minus base, points; paired event bootstrap)

| block | step | all tracks | W muon | HNL daughters | fake rate |
|---|---|---|---|---|---|
| HNL training | builder: CC (s*) -> LM, acorn (HNL) classifier | +3.39 ± 0.34 | +1.26 ± 0.17 | +3.04 ± 0.40 | -1.56 ± 0.14 |
| HNL training | builder: CC (best cut) -> LM, acorn (HNL) classifier | +3.45 ± 0.37 | +1.31 ± 0.22 | +3.32 ± 0.59 | -1.54 ± 0.04 |
| HNL training | graph and classifier: acorn (HNL) -> Canto-L (HNL), with LM | +0.23 ± 0.02 / 0.06 | +0.08 ± 0.04 / 0.04 | +0.52 ± 0.19 / 0.26 | -0.34 ± 0.02 / 0.04 |
| HNL training | graph and classifier: acorn (HNL) -> Canto-L (HNL), with CC (best cuts) | +1.46 ± 0.36 / 0.42 | +0.80 ± 0.22 / 0.24 | +3.09 ± 0.61 / 0.71 | -0.37 ± 0.03 / 0.14 |
| HNL training | builder: CC -> LM, Canto-L (HNL) classifier | +2.22 ± 0.06 / 0.22 | +0.59 ± 0.12 / 0.15 | +0.75 ± 0.20 / 0.45 | -1.51 ± 0.03 / 0.14 |
| training sample | Canto-L with LM: HNL -> ttbar | -0.01 ± 0.03 / 0.06 | +0.01 ± 0.02 / 0.03 | +0.25 ± 0.18 / 0.25 | +0.01 ± 0.02 / 0.04 |
| training sample | Canto-L with LM: HNL -> ttbar+HNL | -0.04 ± 0.01 / 0.08 | -0.02 ± 0.04 / 0.04 | +0.16 ± 0.16 / 0.30 | -0.01 ± 0.01 / 0.05 |
| training sample | acorn with LM: HNL -> ttbar | -0.06 ± 0.02 | -0.01 ± 0.05 | -1.19 ± 0.41 | +0.17 ± 0.03 |
| training sample | acorn with CC: HNL (s*) -> ttbar | +0.12 ± 0.35 | -0.13 ± 0.18 | -3.29 ± 0.80 | +0.03 ± 0.14 |
| ttbar training | builder: CC -> LM, acorn (ttbar) classifier | +3.21 ± 0.13 | +1.38 ± 0.16 | +5.15 ± 0.81 | -1.42 ± 0.05 |
| ttbar training | graph and classifier: acorn (ttbar) -> Canto-L (ttbar), with LM | +0.27 ± 0.04 | +0.10 ± 0.05 | +1.96 ± 0.44 | -0.51 ± 0.03 |
| ttbar training | graph and classifier: acorn (ttbar) -> Canto-L (ttbar), with CC (best cuts) | +1.40 ± 0.17 | +0.97 ± 0.17 | +6.27 ± 0.76 | -0.42 ± 0.09 |
| ttbar training | builder: CC -> LM, Canto-L (ttbar) classifier | +2.08 ± 0.13 | +0.51 ± 0.10 | +0.83 ± 0.26 | -1.51 ± 0.09 |
| total | Canto-L (ttbar) - acorn (HNL) + CC at s* | +3.60 ± 0.34 | +1.35 ± 0.17 | +3.81 ± 0.43 | -1.90 ± 0.14 |

## Against the published values (points)

- all_track: acorn (HNL) + CC at s* (Wang et al.'s reproduction) minus Wang et al. (published): -0.07 ± 0.35
- all_track: Canto-L (ttbar) (LM, frozen) minus Wang et al. (published): +3.53 ± 0.08
- prompt: acorn (HNL) + CC at s* (Wang et al.'s reproduction) minus Wang et al. (published): -0.70 ± 0.28
- prompt: Canto-L (ttbar) (LM, frozen) minus Wang et al. (published): +0.65 ± 0.22
- displaced: acorn (HNL) + CC at s* (Wang et al.'s reproduction) minus Wang et al. (published): +3.81 ± 0.66
- displaced: Canto-L (ttbar) (LM, frozen) minus Wang et al. (published): +7.62 ± 0.52

## Identity checks

- PASS results/published_definitions_comparison.csv canto_l all_track: 0.991403962250 +- 0.000506174906 (346344) vs 0.991403962250 +- 0.000506174906 (346344.0)
- PASS results/published_definitions_comparison.csv canto_l prompt: 0.996486790332 +- 0.000898804568 (14182) vs 0.996486790332 +- 0.000898804568 (14182.0)
- PASS results/published_definitions_comparison.csv canto_l displaced: 0.992234275891 +- 0.001493413184 (16738) vs 0.992234275891 +- 0.001493413184 (16738.0)
- PASS results/published_definitions_comparison.csv canto_l fake_rate: 0.010255129268 +- 0.000298169041 (17609) vs 0.010255129268 +- 0.000298169041 (17609.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_sstar all_track: 0.955399645625 +- 0.003403951776 (333766) vs 0.955399645625 +- 0.003403951776 (333766.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_sstar prompt: 0.982996065205 +- 0.001896727529 (13990) vs 0.982996065205 +- 0.001896727529 (13990.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_sstar displaced: 0.954117019385 +- 0.004282659132 (16095) vs 0.954117019385 +- 0.004282659132 (16095.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_sstar fake_rate: 0.029247976004 +- 0.001384731844 (49281) vs 0.029247976004 +- 0.001384731844 (49281.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_best all_track: 0.954744136918 +- 0.003699632013 (333537) vs 0.954744136918 +- 0.003699632013 (333537.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_best prompt: 0.982504215852 +- 0.002395562269 (13983) vs 0.982504215852 +- 0.002395562269 (13983.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_best displaced: 0.951330843559 +- 0.006074833292 (16048) vs 0.951330843559 +- 0.006074833292 (16048.0)
- PASS results/published_definitions_comparison.csv acorn_hnl_best fake_rate: 0.029023970931 +- 0.000419255169 (48885) vs 0.029023970931 +- 0.000419255169 (48885.0)
- PASS results/published_definitions_comparison.csv acorn_ttbar all_track: 0.956550363965 +- 0.001421583429 (334168) vs 0.956550363965 +- 0.001421583429 (334168.0)
- PASS results/published_definitions_comparison.csv acorn_ttbar prompt: 0.981731309725 +- 0.001834966963 (13972) vs 0.981731309725 +- 0.001834966963 (13972.0)
- PASS results/published_definitions_comparison.csv acorn_ttbar displaced: 0.921216432509 +- 0.007428393657 (15540) vs 0.921216432509 +- 0.007428393657 (15540.0)
- PASS results/published_definitions_comparison.csv acorn_ttbar fake_rate: 0.029532335519 +- 0.000498072667 (49674) vs 0.029532335519 +- 0.000498072667 (49674.0)
- PASS results/training_sample_test_v1.csv Canto-L (HNL) all_track: 0.991532774004 +- 0.000390147350 vs 0.991532774004 +- 0.000390147350
- PASS results/training_sample_test_v1.csv Canto-L (HNL) prompt: 0.996416526138 +- 0.000921891046 vs 0.996416526138 +- 0.000921891046
- PASS results/training_sample_test_v1.csv Canto-L (HNL) displaced: 0.989685221412 +- 0.001541187315 vs 0.989685221412 +- 0.001541187315
- PASS results/training_sample_test_v1.csv Canto-L (HNL) fake_rate: 0.010189358208 +- 0.000228715554 vs 0.010189358208 +- 0.000228715554
- PASS results/training_sample_test_v1.csv Canto-L (ttbar+HNL) all_track: 0.991180688542 +- 0.000408704478 vs 0.991180688542 +- 0.000408704478
- PASS results/training_sample_test_v1.csv Canto-L (ttbar+HNL) prompt: 0.996205733558 +- 0.000938331413 vs 0.996205733558 +- 0.000938331413
- PASS results/training_sample_test_v1.csv Canto-L (ttbar+HNL) displaced: 0.991285790503 +- 0.001434187459 vs 0.991285790503 +- 0.001434187459
- PASS results/training_sample_test_v1.csv Canto-L (ttbar+HNL) fake_rate: 0.010134066714 +- 0.000232489509 vs 0.010134066714 +- 0.000232489509
- PASS results/per_sample_v1.csv acorn (HNL) val-best (MWS-layer + learned merging) s0 primary k: 3784 vs 3784
- PASS results/per_sample_v1.csv acorn (HNL) val-best (MWS-layer + learned merging) s1 primary k: 3779 vs 3779
- PASS results/per_sample_v1.csv acorn (HNL) val-best (MWS-layer + learned merging) s2 primary k: 3793 vs 3793
- PASS results/per_sample_v1.csv acorn (ttbar) val-best (MWS-layer + learned merging) s0 primary k: 3755 vs 3755
- PASS results/per_sample_v1.csv acorn (ttbar) val-best (MWS-layer + learned merging) s1 primary k: 3682 vs 3682
- PASS results/per_sample_v1.csv acorn (ttbar) val-best (MWS-layer + learned merging) s2 primary k: 3726 vs 3726
- PASS results/per_sample_v1.csv acorn (ttbar+HNL) val-best (MWS-layer + learned merging) s0 primary k: 3754 vs 3754
- PASS results/per_sample_v1.csv acorn (ttbar+HNL) val-best (MWS-layer + learned merging) s1 primary k: 3769 vs 3769
- PASS results/per_sample_v1.csv acorn (ttbar+HNL) val-best (MWS-layer + learned merging) s2 primary k: 3772 vs 3772
