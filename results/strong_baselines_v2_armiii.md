# acorn (ttbar+HNL) with its strongest track builders (supervisor feedback round 2, point 2c, interim)

acorn (ttbar+HNL) = acorn's interaction GNN trained on ttbar and HNL events together (Stage-0 baseline arm (iii): 7,999 ttbar + 7,999 HNL training events; three seeds, the Stage-0 scores). It stands in for the requested "acorn trained on ttbar plus a generic displaced sample", which does not exist yet; its HNL training events are disjoint from the val and test events but come from the evaluated signal samples (same masses and lifetimes), so it is an optimistic stand-in. Builders as in round 1 (results/strong_baselines_v1.md): acorn's own CC + walkthrough (CCandWalk, plain and score-guided; the same 27-configuration grid per variant), our walkthrough + merging, the Canto builder (same-layer MWS + merging), and connected components at the best cut <= FR* and at the preregistered cut rule. Every track-building configuration is chosen per seed on val by the builder rule at FR* = 0.029368 (pooled HNL val) with the one-standard-error tie rule, and the CC cuts per seed on val by the CC rules of ranking v3 (no tie rule); nothing is chosen on test. Reference rows (round 1, reproduced to 1e-12): the frozen Canto-L (trained on ttbar only), acorn (HNL) and acorn (ttbar) at their val-best configuration. Cells: mean ± seed sd over three seeds; |d0| bins: HNL-daughter targets pooled over the seeds (k/n).

## Validation (pooled HNL val 1,999 events + ttbar val 1,000 events)

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 3 | 0.9910 ± 0.0020 | 0.976 (489/501) | 50/57 (0.877) | 0.9979 ± 0.0000 | 0.9631 ± 0.0037 | 0.0103 ± 0.0004 | mwslmerge 0.05; mwslmerge 0.03; mwslmerge 0.1 |
| acorn (HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9857 ± 0.0008 | 0.968 (485/501) | 51/57 (0.895) | 0.9975 ± 0.0006 | 0.9438 ± 0.0020 | 0.0154 ± 0.0009 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9787 ± 0.0035 | 0.952 (477/501) | 47/57 (0.825) | 0.9977 ± 0.0003 | 0.9495 ± 0.0065 | 0.0149 ± 0.0012 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9782 ± 0.0020 | 0.956 (479/501) | 47/57 (0.825) | 0.9971 ± 0.0003 | 0.9570 ± 0.0007 | 0.0150 ± 0.0019 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.2 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9740 ± 0.0017 | 0.950 (476/501) | 41/57 (0.719) | 0.9977 ± 0.0003 | 0.9487 ± 0.0065 | 0.0159 ± 0.0014 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9737 ± 0.0015 | 0.950 (476/501) | 41/57 (0.719) | 0.9977 ± 0.0003 | 0.9487 ± 0.0065 | 0.0159 ± 0.0014 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6 |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | 3 | 0.9677 ± 0.0047 | 0.930 (466/501) | 42/57 (0.737) | 0.9970 ± 0.0015 | 0.9603 ± 0.0026 | 0.0166 ± 0.0003 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.1 |
| acorn (ttbar+HNL) + CC, best cut <= FR* | 3 | 0.9504 ± 0.0072 | 0.922 (462/501) | 42/57 (0.737) | 0.9913 ± 0.0015 | 0.8584 ± 0.0109 | 0.0287 ± 0.0003 | CC 0.37; CC 0.38; CC 0.38 |
| acorn (ttbar+HNL) + CC, preregistered cut rule | 3 | 0.9394 ± 0.0227 | 0.916 (459/501) | 42/57 (0.737) | 0.9888 ± 0.0065 | 0.8579 ± 0.0084 | 0.0289 ± 0.0002 | CC 0.26; CC 0.38; CC 0.39 |

val-best acorn (ttbar+HNL) row (highest val primary efficiency of its six rows): acorn (ttbar+HNL) + walkthrough + merging (Canto builder family).

## Test, post-freeze descriptive (not claims; the preregistered claims stay as they are)

Pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) for primary, |d0|, W muons and fake rate; ttbar_test (1,000 events) for ttbar prompt. The val-chosen configurations applied once to the stored test scores; summary code path of hnl.eval.ranking_test.

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 3 | 0.9908 ± 0.0017 | 0.986 (1727/1752) | 94/108 (0.870) | 0.9977 ± 0.0001 | 0.9632 ± 0.0034 | 0.0102 ± 0.0003 | mwslmerge 0.05; mwslmerge 0.03; mwslmerge 0.1 |
| acorn (HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9799 ± 0.0012 | 0.962 (1686/1752) | 103/108 (0.954) | 0.9970 ± 0.0002 | 0.9425 ± 0.0022 | 0.0149 ± 0.0006 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9738 ± 0.0019 | 0.953 (1669/1752) | 99/108 (0.917) | 0.9969 ± 0.0002 | 0.9493 ± 0.0046 | 0.0148 ± 0.0011 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9729 ± 0.0032 | 0.953 (1669/1752) | 99/108 (0.917) | 0.9965 ± 0.0003 | 0.9566 ± 0.0013 | 0.0148 ± 0.0019 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.2 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9693 ± 0.0017 | 0.939 (1646/1752) | 99/108 (0.917) | 0.9966 ± 0.0004 | 0.9481 ± 0.0049 | 0.0156 ± 0.0013 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9691 ± 0.0015 | 0.939 (1645/1752) | 99/108 (0.917) | 0.9964 ± 0.0006 | 0.9481 ± 0.0049 | 0.0156 ± 0.0013 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6 |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | 3 | 0.9600 ± 0.0075 | 0.914 (1602/1752) | 96/108 (0.889) | 0.9962 ± 0.0005 | 0.9594 ± 0.0021 | 0.0163 ± 0.0001 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.1 |
| acorn (ttbar+HNL) + CC, best cut <= FR* | 3 | 0.9401 ± 0.0060 | 0.914 (1602/1752) | 96/108 (0.889) | 0.9878 ± 0.0024 | 0.8612 ± 0.0126 | 0.0289 ± 0.0007 | CC 0.37 (post-freeze CC on the stored test scores); CC 0.38; CC 0.38 (post-freeze CC on the stored test scores) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | 3 | 0.9300 ± 0.0248 | 0.911 (1596/1752) | 95/108 (0.880) | 0.9842 ± 0.0081 | 0.8604 ± 0.0096 | 0.0292 ± 0.0004 | CC 0.26; CC 0.38; CC 0.39 |

Primary efficiency with the preregistered total sigma (event bootstrap + seed term):

- Canto-L (frozen: learned merging): 0.9908 ± 0.0015 (bootstrap 0.0012, seed sd 0.0017)
- acorn (HNL) + walkthrough + merging (Canto builder family): 0.9799 ± 0.0019 (bootstrap 0.0018, seed sd 0.0012)
- acorn (ttbar+HNL) + walkthrough + merging (Canto builder family): 0.9738 ± 0.0024 (bootstrap 0.0021, seed sd 0.0019)
- acorn (ttbar+HNL) + MWS-layer + merging (Canto builder): 0.9729 ± 0.0028 (bootstrap 0.0021, seed sd 0.0032)
- acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk): 0.9693 ± 0.0025 (bootstrap 0.0022, seed sd 0.0017)
- acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk): 0.9691 ± 0.0024 (bootstrap 0.0022, seed sd 0.0015)
- acorn (ttbar) + MWS-layer + merging (Canto builder): 0.9600 ± 0.0050 (bootstrap 0.0024, seed sd 0.0075)
- acorn (ttbar+HNL) + CC, best cut <= FR*: 0.9401 ± 0.0046 (bootstrap 0.0030, seed sd 0.0060)
- acorn (ttbar+HNL) + CC, preregistered cut rule: 0.9300 ± 0.0147 (bootstrap 0.0032, seed sd 0.0248)

## Differences on test (new minus baseline; paired event bootstrap of the preregistered test tables, B = 500, + both seed terms: the claims' estimator; post-freeze descriptive)

**Canto-L minus acorn (ttbar+HNL)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, best cut <= FR* | +5.07 ± 0.46 (z +11.0) | +7.13 ± 0.95 (z +7.5) | -1.85 ± 6.35 (z -0.3) | +4.39 ± 0.38 (z +11.4) | +0.99 ± 0.17 (z +5.7) | +10.20 ± 0.78 (z +13.1) | -1.87 ± 0.05 (z -34.1) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, preregistered cut rule | +6.08 ± 1.47 (z +4.1) | +7.48 ± 1.18 (z +6.3) | -0.93 ± 6.57 (z -0.1) | +5.42 ± 1.40 (z +3.9) | +1.36 ± 0.48 (z +2.8) | +10.29 ± 0.61 (z +16.7) | -1.90 ± 0.04 (z -46.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | +2.15 ± 0.26 (z +8.4) | +4.62 ± 0.91 (z +5.1) | -4.63 ± 5.97 (z -0.8) | +1.57 ± 0.21 (z +7.6) | +0.11 ± 0.04 (z +2.7) | +1.51 ± 0.35 (z +4.3) | -0.54 ± 0.08 (z -6.8) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +2.16 ± 0.25 (z +8.7) | +4.68 ± 0.91 (z +5.1) | -4.63 ± 5.97 (z -0.8) | +1.58 ± 0.20 (z +7.9) | +0.13 ± 0.05 (z +2.7) | +1.52 ± 0.35 (z +4.4) | -0.54 ± 0.08 (z -6.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | +1.70 ± 0.25 (z +6.8) | +3.31 ± 0.88 (z +3.7) | -4.63 ± 5.64 (z -0.8) | +1.21 ± 0.20 (z +6.2) | +0.08 ± 0.04 (z +2.3) | +1.39 ± 0.34 (z +4.1) | -0.46 ± 0.07 (z -7.0) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | +1.79 ± 0.29 (z +6.1) | +3.31 ± 0.94 (z +3.5) | -4.63 ± 5.88 (z -0.8) | +1.30 ± 0.24 (z +5.5) | +0.12 ± 0.04 (z +2.8) | +0.67 ± 0.22 (z +3.0) | -0.46 ± 0.11 (z -4.1) |

**acorn (ttbar+HNL) minus acorn (HNL) val-best**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (HNL) + walkthrough + merging (Canto builder family) | -3.98 ± 0.42 (z -9.5) | -4.79 ± 0.75 (z -6.4) | -6.48 ± 3.57 (z -1.8) | -3.62 ± 0.35 (z -10.3) | -0.92 ± 0.17 (z -5.3) | -8.13 ± 0.76 (z -10.7) | +1.40 ± 0.06 (z +22.8) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (HNL) + walkthrough + merging (Canto builder family) | -4.99 ± 1.45 (z -3.4) | -5.14 ± 1.02 (z -5.1) | -7.41 ± 3.80 (z -1.9) | -4.66 ± 1.39 (z -3.3) | -1.28 ± 0.48 (z -2.7) | -8.21 ± 0.59 (z -13.9) | +1.43 ± 0.05 (z +28.7) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (HNL) + walkthrough + merging (Canto builder family) | -1.06 ± 0.19 (z -5.6) | -2.28 ± 0.68 (z -3.4) | -3.70 ± 3.16 (z -1.2) | -0.80 ± 0.15 (z -5.2) | -0.04 ± 0.04 (z -1.0) | +0.56 ± 0.31 (z +1.8) | +0.07 ± 0.08 (z +0.9) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (HNL) + walkthrough + merging (Canto builder family) | -1.07 ± 0.18 (z -5.9) | -2.34 ± 0.66 (z -3.5) | -3.70 ± 3.16 (z -1.2) | -0.81 ± 0.14 (z -5.7) | -0.06 ± 0.04 (z -1.3) | +0.56 ± 0.31 (z +1.8) | +0.07 ± 0.09 (z +0.9) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (HNL) + walkthrough + merging (Canto builder family) | -0.61 ± 0.18 (z -3.4) | -0.97 ± 0.70 (z -1.4) | -3.70 ± 2.50 (z -1.5) | -0.44 ± 0.13 (z -3.3) | -0.01 ± 0.03 (z -0.3) | +0.68 ± 0.30 (z +2.3) | -0.00 ± 0.07 (z -0.0) |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (HNL) + walkthrough + merging (Canto builder family) | -0.70 ± 0.23 (z -3.0) | -0.97 ± 0.73 (z -1.3) | -3.70 ± 2.96 (z -1.3) | -0.53 ± 0.19 (z -2.8) | -0.05 ± 0.04 (z -1.2) | +1.41 ± 0.16 (z +8.8) | -0.01 ± 0.11 (z -0.1) |

**acorn (ttbar+HNL) minus acorn (ttbar) val-best**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (ttbar) + MWS-layer + merging (Canto builder) | -1.99 ± 0.60 (z -3.3) | +0.00 ± 1.62 (z +0.0) | +0.00 ± 3.83 (z +0.0) | -2.18 ± 0.44 (z -4.9) | -0.84 ± 0.18 (z -4.7) | -9.82 ± 0.77 (z -12.8) | +1.26 ± 0.05 (z +24.5) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (ttbar) + MWS-layer + merging (Canto builder) | -3.00 ± 1.52 (z -2.0) | -0.34 ± 1.77 (z -0.2) | -0.93 ± 3.79 (z -0.2) | -3.22 ± 1.42 (z -2.3) | -1.20 ± 0.48 (z -2.5) | -9.91 ± 0.60 (z -16.6) | +1.29 ± 0.04 (z +35.7) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +0.93 ± 0.47 (z +2.0) | +2.51 ± 1.61 (z +1.6) | +2.78 ± 3.34 (z +0.8) | +0.64 ± 0.31 (z +2.0) | +0.04 ± 0.05 (z +0.8) | -1.13 ± 0.31 (z -3.6) | -0.07 ± 0.08 (z -0.9) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +0.92 ± 0.47 (z +2.0) | +2.45 ± 1.61 (z +1.5) | +2.78 ± 3.34 (z +0.8) | +0.63 ± 0.31 (z +2.0) | +0.03 ± 0.06 (z +0.4) | -1.14 ± 0.31 (z -3.6) | -0.07 ± 0.08 (z -0.9) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.38 ± 0.47 (z +2.9) | +3.82 ± 1.65 (z +2.3) | +2.78 ± 2.61 (z +1.1) | +1.00 ± 0.31 (z +3.2) | +0.07 ± 0.05 (z +1.4) | -1.01 ± 0.30 (z -3.4) | -0.14 ± 0.06 (z -2.3) |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.29 ± 0.50 (z +2.6) | +3.82 ± 1.66 (z +2.3) | +2.78 ± 3.06 (z +0.9) | +0.91 ± 0.34 (z +2.7) | +0.04 ± 0.05 (z +0.7) | -0.29 ± 0.15 (z -2.0) | -0.15 ± 0.11 (z -1.4) |

**same builder: acorn (ttbar+HNL) minus acorn (HNL)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (HNL) + walkthrough + merging (Canto builder family) | -0.61 ± 0.18 (z -3.4) | -0.97 ± 0.70 (z -1.4) | -3.70 ± 2.50 (z -1.5) | -0.44 ± 0.13 (z -3.3) | -0.01 ± 0.03 (z -0.3) | +0.68 ± 0.30 (z +2.3) | -0.00 ± 0.07 (z -0.0) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | -0.80 ± 0.17 (z -4.7) | -1.77 ± 0.67 (z -2.6) | -3.70 ± 3.20 (z -1.2) | -0.60 ± 0.14 (z -4.4) | -0.02 ± 0.04 (z -0.4) | +0.89 ± 0.31 (z +2.9) | +0.04 ± 0.08 (z +0.5) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | -0.82 ± 0.16 (z -5.1) | -1.83 ± 0.65 (z -2.8) | -3.70 ± 3.20 (z -1.2) | -0.62 ± 0.13 (z -4.9) | -0.04 ± 0.04 (z -0.8) | +0.81 ± 0.29 (z +2.8) | +0.05 ± 0.08 (z +0.6) |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (HNL) + MWS-layer + merging (Canto builder) | -0.46 ± 0.23 (z -2.0) | -0.80 ± 0.75 (z -1.1) | -1.85 ± 3.00 (z -0.6) | -0.30 ± 0.18 (z -1.6) | -0.05 ± 0.05 (z -1.0) | +0.88 ± 0.27 (z +3.3) | +0.06 ± 0.12 (z +0.5) |
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (HNL) + CC, best cut <= FR* | -0.47 ± 0.88 (z -0.5) | -1.66 ± 0.79 (z -2.1) | -2.78 ± 3.77 (z -0.7) | -0.11 ± 0.73 (z -0.1) | +0.26 ± 0.25 (z +1.0) | +1.64 ± 1.51 (z +1.1) | -0.01 ± 0.05 (z -0.3) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (HNL) + CC, preregistered cut (claims' reference) | +0.43 ± 1.49 (z +0.3) | -0.63 ± 1.18 (z -0.5) | -2.78 ± 4.35 (z -0.6) | +0.68 ± 1.44 (z +0.5) | +0.79 ± 0.56 (z +1.4) | +4.37 ± 0.61 (z +7.2) | +0.05 ± 0.05 (z +1.0) |

**same builder: acorn (ttbar+HNL) minus acorn (ttbar)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.29 ± 0.50 (z +2.6) | +3.82 ± 1.66 (z +2.3) | +2.78 ± 3.06 (z +0.9) | +0.91 ± 0.34 (z +2.7) | +0.04 ± 0.05 (z +0.7) | -0.29 ± 0.15 (z -2.0) | -0.15 ± 0.11 (z -1.4) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | +2.39 ± 0.75 (z +3.2) | +4.51 ± 1.71 (z +2.6) | +3.70 ± 4.57 (z +0.8) | +1.74 ± 0.55 (z +3.2) | +0.19 ± 0.07 (z +2.6) | +0.41 ± 0.37 (z +1.1) | -0.83 ± 0.10 (z -8.7) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | +2.25 ± 0.55 (z +4.1) | +4.28 ± 1.36 (z +3.1) | +2.78 ± 4.24 (z +0.7) | +1.65 ± 0.43 (z +3.9) | +0.16 ± 0.08 (z +2.0) | +0.56 ± 0.36 (z +1.6) | -0.84 ± 0.09 (z -9.1) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (ttbar) + walkthrough + merging (Canto builder family) | +2.07 ± 0.67 (z +3.1) | +5.71 ± 1.98 (z +2.9) | +1.85 ± 3.80 (z +0.5) | +1.47 ± 0.47 (z +3.1) | +0.01 ± 0.04 (z +0.1) | -0.18 ± 0.34 (z -0.5) | -0.30 ± 0.07 (z -4.3) |
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (ttbar) + CC, preregistered cut (= best cut) | +3.63 ± 1.04 (z +3.5) | +6.56 ± 2.59 (z +2.5) | +9.26 ± 4.55 (z +2.0) | +2.95 ± 0.80 (z +3.7) | +0.34 ± 0.20 (z +1.7) | -1.29 ± 0.88 (z -1.5) | -0.06 ± 0.05 (z -1.1) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (ttbar) + CC, preregistered cut (= best cut) | +2.62 ± 1.74 (z +1.5) | +6.22 ± 2.69 (z +2.3) | +8.33 ± 4.21 (z +2.0) | +1.91 ± 1.56 (z +1.2) | -0.02 ± 0.49 (z -0.0) | -1.37 ± 0.74 (z -1.8) | -0.03 ± 0.04 (z -0.7) |

**round-1 identity**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) + walkthrough + merging (Canto builder family) | +1.09 ± 0.21 (z +5.1) | +2.34 ± 0.69 (z +3.4) | -8.33 ± 5.63 (z -1.5) | +0.77 ± 0.17 (z +4.6) | +0.07 ± 0.03 (z +2.4) | +2.07 ± 0.25 (z +8.4) | -0.47 ± 0.04 (z -11.1) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +3.08 ± 0.50 (z +6.1) | +7.13 ± 1.72 (z +4.1) | -1.85 ± 6.07 (z -0.3) | +2.21 ± 0.34 (z +6.5) | +0.16 ± 0.05 (z +3.3) | +0.38 ± 0.24 (z +1.6) | -0.61 ± 0.02 (z -24.8) |

Primary efficiency per test set (descriptive; seed mean ± total sigma; difference with the same estimator):

| test set | primary particles per seed | new | value | baseline | value | difference |
|---|---|---|---|---|---|---|
| hnl10_100_test | 603 | Canto-L (frozen: learned merging) | 0.9923 ± 0.0033 | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 0.9773 ± 0.0062 | +1.49 ± 0.61 (z +2.4) |
| hnl15_200_test | 549 | Canto-L (frozen: learned merging) | 0.9848 ± 0.0047 | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 0.9642 ± 0.0073 | +2.06 ± 0.68 (z +3.0) |
| hnl15_100_external | 3861 | Canto-L (frozen: learned merging) | 0.9914 ± 0.0017 | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 0.9746 ± 0.0025 | +1.67 ± 0.26 (z +6.4) |

## Differences on val (new minus baseline; paired event bootstrap of the val tables of feedback point 5, B = 1000, + both seed terms; validation)

**Canto-L minus acorn (ttbar+HNL)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, best cut <= FR* | +4.06 ± 0.65 (z +6.2) | +5.39 ± 2.04 (z +2.6) | +14.04 ± 9.57 (z +1.5) | +3.47 ± 0.56 (z +6.2) | +0.66 ± 0.22 (z +2.9) | +10.47 ± 0.70 (z +15.0) | -1.84 ± 0.07 (z -27.5) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, preregistered cut rule | +5.16 ± 1.42 (z +3.6) | +5.99 ± 2.47 (z +2.4) | +14.04 ± 9.57 (z +1.5) | +4.55 ± 1.46 (z +3.1) | +0.91 ± 0.44 (z +2.1) | +10.52 ± 0.56 (z +18.7) | -1.87 ± 0.06 (z -31.3) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | +1.70 ± 0.41 (z +4.2) | +2.59 ± 1.39 (z +1.9) | +15.79 ± 10.13 (z +1.6) | +1.26 ± 0.32 (z +3.9) | +0.02 ± 0.03 (z +0.7) | +1.45 ± 0.44 (z +3.3) | -0.56 ± 0.08 (z -6.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +1.73 ± 0.41 (z +4.3) | +2.59 ± 1.39 (z +1.9) | +15.79 ± 10.13 (z +1.6) | +1.28 ± 0.32 (z +4.1) | +0.02 ± 0.03 (z +0.7) | +1.45 ± 0.44 (z +3.3) | -0.56 ± 0.09 (z -6.5) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | +1.23 ± 0.42 (z +2.9) | +2.40 ± 1.47 (z +1.6) | +5.26 ± 8.23 (z +0.6) | +0.89 ± 0.33 (z +2.7) | +0.02 ± 0.03 (z +0.7) | +1.37 ± 0.44 (z +3.1) | -0.46 ± 0.08 (z -5.9) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | +1.28 ± 0.36 (z +3.6) | +2.00 ± 1.49 (z +1.3) | +5.26 ± 7.58 (z +0.7) | +1.02 ± 0.30 (z +3.4) | +0.07 ± 0.05 (z +1.3) | +0.62 ± 0.23 (z +2.7) | -0.48 ± 0.11 (z -4.2) |

**acorn (ttbar+HNL) minus acorn (HNL) val-best**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (HNL) + walkthrough + merging (Canto builder family) | -3.53 ± 0.61 (z -5.8) | -4.59 ± 1.80 (z -2.5) | -15.79 ± 9.52 (z -1.7) | -3.12 ± 0.52 (z -6.0) | -0.62 ± 0.23 (z -2.7) | -8.54 ± 0.67 (z -12.8) | +1.32 ± 0.08 (z +16.8) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (HNL) + walkthrough + merging (Canto builder family) | -4.63 ± 1.40 (z -3.3) | -5.19 ± 2.29 (z -2.3) | -15.79 ± 9.52 (z -1.7) | -4.20 ± 1.45 (z -2.9) | -0.87 ± 0.45 (z -2.0) | -8.59 ± 0.53 (z -16.3) | +1.35 ± 0.07 (z +18.5) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (HNL) + walkthrough + merging (Canto builder family) | -1.18 ± 0.35 (z -3.4) | -1.80 ± 1.06 (z -1.7) | -17.54 ± 9.78 (z -1.8) | -0.91 ± 0.27 (z -3.4) | +0.02 ± 0.06 (z +0.3) | +0.49 ± 0.40 (z +1.2) | +0.05 ± 0.10 (z +0.5) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (HNL) + walkthrough + merging (Canto builder family) | -1.20 ± 0.35 (z -3.5) | -1.80 ± 1.06 (z -1.7) | -17.54 ± 9.78 (z -1.8) | -0.93 ± 0.26 (z -3.5) | +0.02 ± 0.06 (z +0.3) | +0.49 ± 0.40 (z +1.2) | +0.05 ± 0.10 (z +0.5) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (HNL) + walkthrough + merging (Canto builder family) | -0.70 ± 0.36 (z -1.9) | -1.60 ± 1.13 (z -1.4) | -7.02 ± 8.05 (z -0.9) | -0.55 ± 0.28 (z -1.9) | +0.02 ± 0.06 (z +0.3) | +0.57 ± 0.40 (z +1.4) | -0.06 ± 0.09 (z -0.6) |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (HNL) + walkthrough + merging (Canto builder family) | -0.75 ± 0.31 (z -2.4) | -1.20 ± 1.21 (z -1.0) | -7.02 ± 7.31 (z -1.0) | -0.68 ± 0.26 (z -2.6) | -0.04 ± 0.07 (z -0.5) | +1.32 ± 0.14 (z +9.4) | -0.04 ± 0.12 (z -0.3) |

**acorn (ttbar+HNL) minus acorn (ttbar) val-best**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (ttbar) + MWS-layer + merging (Canto builder) | -1.73 ± 0.67 (z -2.6) | -0.80 ± 2.17 (z -0.4) | +0.00 ± 8.10 (z +0.0) | -1.75 ± 0.56 (z -3.2) | -0.57 ± 0.25 (z -2.3) | -10.18 ± 0.68 (z -14.9) | +1.21 ± 0.06 (z +19.3) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (ttbar) + MWS-layer + merging (Canto builder) | -2.83 ± 1.43 (z -2.0) | -1.40 ± 2.60 (z -0.5) | +0.00 ± 8.10 (z +0.0) | -2.83 ± 1.46 (z -1.9) | -0.82 ± 0.46 (z -1.8) | -10.23 ± 0.54 (z -18.8) | +1.24 ± 0.06 (z +22.3) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +0.63 ± 0.43 (z +1.4) | +2.00 ± 1.73 (z +1.2) | -1.75 ± 8.94 (z -0.2) | +0.46 ± 0.33 (z +1.4) | +0.07 ± 0.11 (z +0.7) | -1.16 ± 0.41 (z -2.8) | -0.07 ± 0.08 (z -0.8) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +0.60 ± 0.43 (z +1.4) | +2.00 ± 1.73 (z +1.2) | -1.75 ± 8.94 (z -0.2) | +0.44 ± 0.32 (z +1.4) | +0.07 ± 0.11 (z +0.7) | -1.16 ± 0.41 (z -2.8) | -0.06 ± 0.09 (z -0.8) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.10 ± 0.46 (z +2.4) | +2.20 ± 1.71 (z +1.3) | +8.77 ± 7.40 (z +1.2) | +0.82 ± 0.35 (z +2.3) | +0.07 ± 0.11 (z +0.7) | -1.08 ± 0.41 (z -2.6) | -0.17 ± 0.08 (z -2.3) |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.05 ± 0.41 (z +2.6) | +2.59 ± 1.81 (z +1.4) | +8.77 ± 5.90 (z +1.5) | +0.69 ± 0.33 (z +2.1) | +0.02 ± 0.12 (z +0.2) | -0.33 ± 0.16 (z -2.0) | -0.15 ± 0.11 (z -1.4) |

**same builder: acorn (ttbar+HNL) minus acorn (HNL)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (HNL) + walkthrough + merging (Canto builder family) | -0.70 ± 0.36 (z -1.9) | -1.60 ± 1.13 (z -1.4) | -7.02 ± 8.05 (z -0.9) | -0.55 ± 0.28 (z -1.9) | +0.02 ± 0.06 (z +0.3) | +0.57 ± 0.40 (z +1.4) | -0.06 ± 0.09 (z -0.6) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | -0.95 ± 0.32 (z -3.0) | -1.80 ± 1.06 (z -1.7) | -10.53 ± 6.87 (z -1.5) | -0.75 ± 0.25 (z -3.0) | +0.04 ± 0.05 (z +0.7) | +0.68 ± 0.40 (z +1.7) | +0.03 ± 0.10 (z +0.3) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | -1.00 ± 0.31 (z -3.3) | -1.80 ± 1.06 (z -1.7) | -10.53 ± 6.87 (z -1.5) | -0.79 ± 0.24 (z -3.2) | +0.04 ± 0.05 (z +0.7) | +0.66 ± 0.38 (z +1.7) | +0.04 ± 0.10 (z +0.4) |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (HNL) + MWS-layer + merging (Canto builder) | -0.55 ± 0.31 (z -1.8) | -1.20 ± 1.21 (z -1.0) | -7.02 ± 6.80 (z -1.0) | -0.47 ± 0.27 (z -1.8) | -0.04 ± 0.07 (z -0.5) | +0.80 ± 0.24 (z +3.3) | +0.07 ± 0.12 (z +0.6) |
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (HNL) + CC, best cut <= FR* | -0.58 ± 0.92 (z -0.6) | -2.20 ± 1.76 (z -1.2) | -7.02 ± 5.47 (z -1.3) | -0.16 ± 0.89 (z -0.2) | +0.37 ± 0.24 (z +1.5) | +1.29 ± 1.36 (z +0.9) | -0.05 ± 0.04 (z -1.3) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (HNL) + CC, preregistered cut (claims' reference) | +0.10 ± 1.38 (z +0.1) | -1.80 ± 2.24 (z -0.8) | -7.02 ± 5.47 (z -1.3) | +0.75 ± 1.46 (z +0.5) | +0.73 ± 0.52 (z +1.4) | +3.69 ± 0.55 (z +6.8) | -0.04 ± 0.05 (z -0.8) |

**same builder: acorn (ttbar+HNL) minus acorn (ttbar)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.05 ± 0.41 (z +2.6) | +2.59 ± 1.81 (z +1.4) | +8.77 ± 5.90 (z +1.5) | +0.69 ± 0.33 (z +2.1) | +0.02 ± 0.12 (z +0.2) | -0.33 ± 0.16 (z -2.0) | -0.15 ± 0.11 (z -1.4) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | +1.98 ± 0.88 (z +2.2) | +3.59 ± 2.26 (z +1.6) | +8.77 ± 8.22 (z +1.1) | +1.53 ± 0.68 (z +2.3) | +0.05 ± 0.06 (z +0.9) | +0.44 ± 0.49 (z +0.9) | -0.87 ± 0.11 (z -8.1) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | +2.08 ± 0.63 (z +3.3) | +3.39 ± 1.72 (z +2.0) | +7.02 ± 7.10 (z +1.0) | +1.61 ± 0.50 (z +3.2) | +0.05 ± 0.05 (z +1.0) | +0.55 ± 0.47 (z +1.2) | -0.89 ± 0.10 (z -8.5) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | acorn (ttbar) + walkthrough + merging (Canto builder family) | +1.88 ± 0.59 (z +3.2) | +5.19 ± 2.36 (z +2.2) | +8.77 ± 7.40 (z +1.2) | +1.50 ± 0.47 (z +3.2) | -0.02 ± 0.03 (z -0.7) | -0.24 ± 0.46 (z -0.5) | -0.30 ± 0.08 (z -3.7) |
| acorn (ttbar+HNL) + CC, best cut <= FR* | acorn (ttbar) + CC, preregistered cut (= best cut) | +2.73 ± 0.97 (z +2.8) | +2.99 ± 3.25 (z +0.9) | +8.77 ± 5.90 (z +1.5) | +2.41 ± 0.88 (z +2.7) | +0.43 ± 0.27 (z +1.6) | -1.39 ± 0.84 (z -1.7) | -0.10 ± 0.04 (z -2.4) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | acorn (ttbar) + CC, preregistered cut (= best cut) | +1.63 ± 1.58 (z +1.0) | +2.40 ± 3.54 (z +0.7) | +8.77 ± 5.90 (z +1.5) | +1.33 ± 1.60 (z +0.8) | +0.18 ± 0.45 (z +0.4) | -1.43 ± 0.73 (z -2.0) | -0.07 ± 0.03 (z -2.1) |

**round-1 identity**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) + walkthrough + merging (Canto builder family) | +0.53 ± 0.30 (z +1.8) | +0.80 ± 1.20 (z +0.7) | -1.75 ± 7.08 (z -0.2) | +0.35 ± 0.22 (z +1.6) | +0.04 ± 0.05 (z +0.7) | +1.93 ± 0.26 (z +7.5) | -0.51 ± 0.06 (z -8.4) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +2.33 ± 0.48 (z +4.9) | +4.59 ± 1.72 (z +2.7) | +14.04 ± 7.05 (z +2.0) | +1.72 ± 0.36 (z +4.8) | +0.09 ± 0.10 (z +0.9) | +0.29 ± 0.27 (z +1.1) | -0.63 ± 0.04 (z -16.7) |

Points = percentage points (fake rate: a negative difference means the new row has the lower fake rate). A row's own sigma (bootstrap + seed term) is in the CSV files.

## Chosen configurations and their place in the grids (val)

| row | seed | configuration | grid (configurations of the family) | edge | lower-edge rule (CCandWalk) |
|---|---|---|---|---|---|
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 0 | acwalk 0.01,0.15,0.6 | trackbuild_val_acwalk_all (27) | interior | interior (no extension needed) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 1 | acwalk 0.01,0.15,0.6 | trackbuild_val_acwalk_all (27) | interior | interior (no extension needed) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 2 | acwalk 0.01,0.15,0.6 | trackbuild_val_acwalk_all (27) | interior | interior (no extension needed) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 0 | acwalkg 0.01,0.15,0.6 | trackbuild_val_acwalk_all (27) | interior | interior (no extension needed) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 1 | acwalkg 0.01,0.15,0.6 | trackbuild_val_acwalk_all (27) | interior | interior (no extension needed) |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 2 | acwalkg 0.01,0.15,0.6 | trackbuild_val_acwalk_all (27) | interior | interior (no extension needed) |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 0 | walkmerge 0.2,0.5,1.0 | trackbuild_val_ext (10) | walk add 0.5 lowest | - |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 1 | walkmerge 0.2,0.5,1.0 | trackbuild_val_ext (10) | walk add 0.5 lowest | - |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 2 | walkmerge 0.2,0.5,1.0 | trackbuild_val_ext (10) | walk add 0.5 lowest | - |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 0 | mwslayermerge 0.1 | trackbuild_val_ext (6) | s0 0.1 lowest | - |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 1 | mwslayermerge 0.1 | trackbuild_val_ext (6) | s0 0.1 lowest | - |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 2 | mwslayermerge 0.2 | trackbuild_val_ext (6) | interior | - |

## Notes

- acorn (ttbar+HNL): the Stage-0 baseline arm (iii) (DECISIONS.md 2026-10-02): acorn's IGNN recipe (metric-learning graph, interaction GNN with hidden 128 and 8 iterations) trained on the 7,999 ttbar training events and 7,999 HNL training events (hnl10_100 train index < 5000, hnl15_200 train index >= 5000); three seeds, each retraining metric learning and GNN; the Stage-0 scores RUN/gnn_scores (val) and RUN/gnn_scores_test (one-time test chain). For every seed the val and test scores come from the same checkpoint (the sha256 recorded at inference equals the checkpoint file's sha256). The HNL training events are disjoint from the val and test events but come from the evaluated signal samples: an optimistic stand-in for "ttbar plus a generic displaced sample".
- Code: hnl.eval.strong_baselines_armiii (round 1's hnl.eval.strong_baselines functions; builders of hnl.eval.trackbuild_val and hnl.trackbuild.acorn_walk). CCandWalk grids: RUN/trackbuild_val_acwalk (19 configurations per variant; jobs 6693443, 6693457, 6693459) and the lower-edge extension RUN/trackbuild_val_acwalk_ext (8 per variant; jobs 6693456, 6693458, 6693460), both run for every seed and united in RUN/trackbuild_val_acwalk_all (`merge`, job 6693556). Walkthrough + merging and MWS-layer + merging: RUN/trackbuild_val_ext (the first grid holding each family, as ranking v3). CC: RUN/eval_val (0.01 grid).
- Chosen configurations (val): CCandWalk (0.01, 0.15, 0.6) for both variants and every seed (interior in all three parameters of the 27-configuration grid: round 1's lower-edge rule does not apply); walkthrough + merging (0.2, 0.5, 1.0) for every seed; MWS-layer + merging 0.1 / 0.1 / 0.2 (= ranking v3); CC best cut 0.37 / 0.38 / 0.38, preregistered cut 0.26 / 0.38 / 0.39 (= results/working_points_v1.json). As in round 1, walk add 0.5 is the lower of the two values of the walkthrough + merging grid and s0 0.1 the lowest MWS-layer + merging value of trackbuild_val_ext; these grids are not extended (round 1's rule extends only acorn's CCandWalk).
- Identity checks (val; |difference| <= 1e-12, counts exact): every row, entry and difference of round 1 (results/strong_baselines_v1_val*.csv), the arm (iii) CC and MWS-layer + merging rows and entries of ranking v3, and the preregistered cuts: 583 of 583 (results/strong_baselines_v2_armiii_val_checks.csv). hnl.trackbuild.acorn_walk against acorn's own CCandWalk._build_tracks_one_evt on acorn (ttbar+HNL) seed 0: identical track partitions in 5,400 of 5,400 val event-configurations (every 30th val graph, 100 events, 54 configurations; job 6693461) and in 800 of 800 test event-configurations at the val-chosen configurations (every 20th test graph, 400 events, the two CCandWalk configurations of seed 0; job 6693596).
- Test: the val-chosen configurations applied once to the stored test scores (RUN/gnn_scores_test/testset.tar) by slurm/feedback/armiii_test.sbatch, one job per run and configuration (walkthrough + merging 6693512-6693514, MWS-layer + merging 6693515-6693517, CCandWalk 6693589-6693591, score-guided 6693592-6693594) -> runs/feedback_v2/strong_baselines_armiii/armiii_s<S>/<key>_test. CC at the preregistered cuts: the frozen RUN/eval_test of the one-time test chain. CC at the best cut: seed 1's 0.38 is in the frozen eval_test; 0.37 (seed 0) and 0.38 (seed 2) were evaluated once, post-freeze, on the stored test scores (job 6693518; s* = 0.35 rows identical to the frozen eval_test for both seeds).
- Two-stage spec: CC, walkthrough + merging and MWS-layer + merging were fixed first by `val --skip-missing` (runs/feedback_v2/strong_baselines_armiii/stage1/stage1_spec_test.json, sha256 b89e9a124495...) so that their test jobs could run while the last CCandWalk grids were computed; the final spec (results/strong_baselines_v2_armiii_spec_test.json, sha256 1beecdb1c93b...) holds identical entries for these rows. Each test output records the spec file it ran from and its sha256; the summary requires that file unchanged and its entry equal to the final one. No test efficiency or summary was computed before the final spec existed; before it, test outputs were read only by the s* identity check of the CC job 6693518 and to look up which cuts RUN/eval_test holds.
- Test identity checks: every round 1 test row and difference (results/strong_baselines_v1.csv and _differences.csv) reproduced, and the arm (iii) CC row at the preregistered cuts reproduces the M3 test summary (runs/baseline_v1/analysis_test/test_summary.json); |difference| <= 1e-12, every check of results/strong_baselines_v2_armiii_test_checks.csv passed (the summary stops otherwise).
- Status: validation numbers and post-freeze descriptive test summaries; nothing is chosen on test; the preregistered claims (results/claims_test_v1.json) stay as they are. The val numbers of every row chosen on val are optimistic for that row (Canto-L included); |d0| >= 300 mm holds 19 (val) and 36 (test) HNL daughters per seed, so no difference in that column is resolved.
