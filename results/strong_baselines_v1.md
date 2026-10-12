# Strongest acorn baselines (supervisor feedback round 1, point 2)

acorn (HNL) = acorn's interaction GNN trained on the HNL signal as in Wang et al. (arm ii); acorn (ttbar) = trained on ttbar only (arm i); three seeds each, the Stage-0 baseline scores. Builders on those scores: acorn's own CC + walkthrough (CCandWalk, hnl.trackbuild.acorn_walk: acorn's unmodified functions; plain walk as in acorn's CTD 2023 configuration, and score-guided as in its Rel. 24 / ACAT 2025 configurations), our walkthrough + merging, and the Canto builder (same-layer MWS + merging). Every configuration is chosen per seed on val by the builder rule at FR* = 0.029368 (pooled HNL val) with the one-standard-error tie rule; nothing is chosen on test. Cells: mean ± seed sd over three seeds; |d0| bins: HNL-daughter targets pooled over the seeds (k/n).

## Validation (pooled HNL val 1,999 events + ttbar val 1,000 events)

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| Canto-L + MWS-layer + merging (same builder as the headline baseline; not the frozen builder) | 3 | 0.9915 ± 0.0017 | 0.976 (489/501) | 49/57 (0.860) | 0.9979 ± 0.0000 | 0.9606 ± 0.0035 | 0.0112 ± 0.0009 | mwslayermerge 0.2; mwslayermerge 0.2; mwslayermerge 0.2 |
| Canto-L (frozen: learned merging) | 3 | 0.9910 ± 0.0020 | 0.976 (489/501) | 50/57 (0.877) | 0.9979 ± 0.0000 | 0.9631 ± 0.0037 | 0.0103 ± 0.0004 | mwslmerge 0.05; mwslmerge 0.03; mwslmerge 0.1 |
| acorn (HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9857 ± 0.0008 | 0.968 (485/501) | 51/57 (0.895) | 0.9975 ± 0.0006 | 0.9438 ± 0.0020 | 0.0154 ± 0.0009 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9837 ± 0.0011 | 0.968 (485/501) | 47/57 (0.825) | 0.9973 ± 0.0005 | 0.9420 ± 0.0009 | 0.0155 ± 0.0010 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.2,0.4 |
| acorn (HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9837 ± 0.0004 | 0.968 (485/501) | 51/57 (0.895) | 0.9975 ± 0.0006 | 0.9490 ± 0.0040 | 0.0143 ± 0.0007 | mwslayermerge 0.1; mwslayermerge 0.2; mwslayermerge 0.1 |
| acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9835 ± 0.0015 | 0.968 (485/501) | 47/57 (0.825) | 0.9973 ± 0.0005 | 0.9418 ± 0.0018 | 0.0156 ± 0.0009 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.2,0.4 |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | 3 | 0.9677 ± 0.0047 | 0.930 (466/501) | 42/57 (0.737) | 0.9970 ± 0.0015 | 0.9603 ± 0.0026 | 0.0166 ± 0.0003 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.1 |
| acorn (ttbar) + walkthrough + merging (Canto builder family) | 3 | 0.9599 ± 0.0068 | 0.900 (451/501) | 42/57 (0.737) | 0.9979 ± 0.0000 | 0.9518 ± 0.0047 | 0.0179 ± 0.0006 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.7,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (HNL) + CC, best cut <= FR* | 3 | 0.9562 ± 0.0128 | 0.944 (473/501) | 46/57 (0.807) | 0.9875 ± 0.0027 | 0.8456 ± 0.0208 | 0.0292 ± 0.0005 | CC 0.38; CC 0.36 (FR* unreachable: minimum-rate cut); CC 0.44 |
| acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9542 ± 0.0137 | 0.914 (458/501) | 36/57 (0.632) | 0.9971 ± 0.0003 | 0.9443 ± 0.0053 | 0.0247 ± 0.0012 | acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6 |
| acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9529 ± 0.0087 | 0.916 (459/501) | 37/57 (0.649) | 0.9971 ± 0.0003 | 0.9431 ± 0.0047 | 0.0248 ± 0.0010 | acwalkg 0.01,0.01,0.6; acwalkg 0.005,0.005,0.6; acwalkg 0.01,0.01,0.6 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 3 | 0.9384 ± 0.0026 | 0.934 (468/501) | 46/57 (0.807) | 0.9815 ± 0.0048 | 0.8210 ± 0.0034 | 0.0293 ± 0.0005 | CC 0.26; CC 0.36 (FR* unreachable: minimum-rate cut); CC 0.27 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 3 | 0.9231 ± 0.0128 | 0.892 (447/501) | 37/57 (0.649) | 0.9870 ± 0.0031 | 0.8723 ± 0.0094 | 0.0297 ± 0.0003 | CC 0.34 (FR* unreachable: minimum-rate cut); CC 0.33; CC 0.31 (FR* unreachable: minimum-rate cut) |

## Test, post-freeze descriptive (not claims; the preregistered claims stay as they are)

Pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) for primary, |d0|, W muons and fake rate; ttbar_test (1,000 events) for ttbar prompt. The same configurations as on val, applied unchanged to the stored test scores; summary code path of hnl.eval.ranking_test.

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| Canto-L + MWS-layer + merging (same builder as the headline baseline; not the frozen builder) | 3 | 0.9910 ± 0.0025 | 0.983 (1723/1752) | 101/108 (0.935) | 0.9978 ± 0.0001 | 0.9610 ± 0.0039 | 0.0112 ± 0.0007 | mwslayermerge 0.2; mwslayermerge 0.2; mwslayermerge 0.2 |
| Canto-L (frozen: learned merging) | 3 | 0.9908 ± 0.0017 | 0.986 (1727/1752) | 94/108 (0.870) | 0.9977 ± 0.0001 | 0.9632 ± 0.0034 | 0.0102 ± 0.0003 | mwslmerge 0.05; mwslmerge 0.03; mwslmerge 0.1 |
| acorn (HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9799 ± 0.0012 | 0.962 (1686/1752) | 103/108 (0.954) | 0.9970 ± 0.0002 | 0.9425 ± 0.0022 | 0.0149 ± 0.0006 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9775 ± 0.0002 | 0.961 (1683/1752) | 101/108 (0.935) | 0.9970 ± 0.0005 | 0.9478 ± 0.0043 | 0.0141 ± 0.0007 | mwslayermerge 0.1; mwslayermerge 0.2; mwslayermerge 0.1 |
| acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9774 ± 0.0006 | 0.957 (1677/1752) | 103/108 (0.954) | 0.9968 ± 0.0002 | 0.9400 ± 0.0006 | 0.0151 ± 0.0005 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.2,0.4 |
| acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9773 ± 0.0008 | 0.957 (1677/1752) | 103/108 (0.954) | 0.9967 ± 0.0002 | 0.9392 ± 0.0021 | 0.0152 ± 0.0006 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.2,0.4 |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | 3 | 0.9600 ± 0.0075 | 0.914 (1602/1752) | 96/108 (0.889) | 0.9962 ± 0.0005 | 0.9594 ± 0.0021 | 0.0163 ± 0.0001 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.1 |
| acorn (ttbar) + walkthrough + merging (Canto builder family) | 3 | 0.9531 ± 0.0110 | 0.896 (1569/1752) | 97/108 (0.898) | 0.9968 ± 0.0004 | 0.9511 ± 0.0035 | 0.0179 ± 0.0005 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.7,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9467 ± 0.0089 | 0.896 (1570/1752) | 96/108 (0.889) | 0.9949 ± 0.0008 | 0.9425 ± 0.0036 | 0.0240 ± 0.0008 | acwalkg 0.01,0.01,0.6; acwalkg 0.005,0.005,0.6; acwalkg 0.01,0.01,0.6 |
| acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9454 ± 0.0123 | 0.894 (1567/1752) | 95/108 (0.880) | 0.9947 ± 0.0006 | 0.9440 ± 0.0040 | 0.0239 ± 0.0010 | acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6 |
| acorn (HNL) + CC, best cut <= FR* | 3 | 0.9448 ± 0.0136 | 0.931 (1631/1752) | 99/108 (0.917) | 0.9851 ± 0.0033 | 0.8448 ± 0.0229 | 0.0290 ± 0.0002 | CC 0.38 (post-freeze CC on the stored test scores); CC 0.36 (FR* unreachable on val: minimum-rate cut); CC 0.44 (post-freeze CC on the stored test scores) |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 3 | 0.9257 ± 0.0057 | 0.917 (1607/1752) | 98/108 (0.907) | 0.9762 ± 0.0047 | 0.8167 ± 0.0032 | 0.0286 ± 0.0007 | CC 0.26; CC 0.36 (FR* unreachable on val: minimum-rate cut); CC 0.27 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 3 | 0.9038 ± 0.0164 | 0.849 (1487/1752) | 86/108 (0.796) | 0.9844 ± 0.0016 | 0.8741 ± 0.0085 | 0.0295 ± 0.0005 | CC 0.34 (FR* unreachable on val: minimum-rate cut); CC 0.33; CC 0.31 (FR* unreachable on val: minimum-rate cut) |

Primary efficiency with the preregistered total sigma (event bootstrap + seed term):

- Canto-L + MWS-layer + merging (same builder as the headline baseline; not the frozen builder): 0.9910 ± 0.0018 (bootstrap 0.0011, seed sd 0.0025)
- Canto-L (frozen: learned merging): 0.9908 ± 0.0015 (bootstrap 0.0012, seed sd 0.0017)
- acorn (HNL) + walkthrough + merging (Canto builder family): 0.9799 ± 0.0019 (bootstrap 0.0018, seed sd 0.0012)
- acorn (HNL) + MWS-layer + merging (Canto builder): 0.9775 ± 0.0018 (bootstrap 0.0018, seed sd 0.0002)
- acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk): 0.9774 ± 0.0020 (bootstrap 0.0019, seed sd 0.0006)
- acorn (HNL) + acorn's CC + walkthrough (CCandWalk): 0.9773 ± 0.0020 (bootstrap 0.0019, seed sd 0.0008)
- acorn (ttbar) + MWS-layer + merging (Canto builder): 0.9600 ± 0.0050 (bootstrap 0.0024, seed sd 0.0075)
- acorn (ttbar) + walkthrough + merging (Canto builder family): 0.9531 ± 0.0069 (bootstrap 0.0026, seed sd 0.0110)
- acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk): 0.9467 ± 0.0059 (bootstrap 0.0029, seed sd 0.0089)
- acorn (ttbar) + acorn's CC + walkthrough (CCandWalk): 0.9454 ± 0.0077 (bootstrap 0.0029, seed sd 0.0123)
- acorn (HNL) + CC, best cut <= FR*: 0.9448 ± 0.0083 (bootstrap 0.0028, seed sd 0.0136)
- acorn (HNL) + CC, preregistered cut (claims' reference): 0.9257 ± 0.0047 (bootstrap 0.0034, seed sd 0.0057)
- acorn (ttbar) + CC, preregistered cut (= best cut): 0.9038 ± 0.0102 (bootstrap 0.0037, seed sd 0.0164)

## Like-for-like differences on test (Canto-L minus baseline; paired event bootstrap of the preregistered test tables, B = 500, + both seed terms: the claims' estimator; post-freeze descriptive)

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L | acorn (HNL) + CC, best cut <= FR* | +4.59 ± 0.84 (z +5.5) | +5.48 ± 0.92 (z +5.9) | -4.63 ± 6.27 (z -0.7) | +4.28 ± 0.69 (z +6.2) | +1.26 ± 0.22 (z +5.6) | +11.85 ± 1.35 (z +8.8) | -1.88 ± 0.03 (z -54.8) |
| Canto-L | acorn (HNL) + CC, preregistered cut (claims' reference) | +6.50 ± 0.48 (z +13.5) | +6.85 ± 1.11 (z +6.2) | -3.70 ± 6.52 (z -0.6) | +6.10 ± 0.48 (z +12.7) | +2.15 ± 0.31 (z +6.8) | +14.65 ± 0.34 (z +43.1) | -1.84 ± 0.05 (z -36.5) |
| Canto-L | acorn (ttbar) + CC, preregistered cut (= best cut) | +8.70 ± 1.02 (z +8.5) | +13.70 ± 2.73 (z +5.0) | +7.41 ± 7.89 (z +0.9) | +7.33 ± 0.77 (z +9.5) | +1.33 ± 0.15 (z +8.7) | +8.91 ± 0.55 (z +16.2) | -1.93 ± 0.04 (z -45.1) |
| Canto-L | acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | +1.34 ± 0.22 (z +6.2) | +2.85 ± 0.78 (z +3.7) | -8.33 ± 5.81 (z -1.4) | +0.97 ± 0.17 (z +5.7) | +0.10 ± 0.03 (z +3.1) | +2.40 ± 0.25 (z +9.8) | -0.50 ± 0.04 (z -12.6) |
| Canto-L | acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +1.34 ± 0.21 (z +6.3) | +2.85 ± 0.78 (z +3.7) | -8.33 ± 5.81 (z -1.4) | +0.96 ± 0.17 (z +5.7) | +0.09 ± 0.03 (z +2.9) | +2.32 ± 0.22 (z +10.7) | -0.49 ± 0.04 (z -13.0) |
| Canto-L | acorn (HNL) + walkthrough + merging (Canto builder family) | +1.09 ± 0.21 (z +5.1) | +2.34 ± 0.69 (z +3.4) | -8.33 ± 5.63 (z -1.5) | +0.77 ± 0.17 (z +4.6) | +0.07 ± 0.03 (z +2.4) | +2.07 ± 0.25 (z +8.4) | -0.47 ± 0.04 (z -11.1) |
| Canto-L | acorn (HNL) + MWS-layer + merging (Canto builder) | +1.33 ± 0.20 (z +6.6) | +2.51 ± 0.70 (z +3.6) | -6.48 ± 5.91 (z -1.1) | +1.00 ± 0.16 (z +6.1) | +0.07 ± 0.04 (z +1.8) | +1.54 ± 0.33 (z +4.7) | -0.39 ± 0.05 (z -8.3) |
| Canto-L | acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | +4.53 ± 0.77 (z +5.9) | +9.13 ± 1.84 (z +5.0) | -0.93 ± 7.30 (z -0.1) | +3.30 ± 0.57 (z +5.8) | +0.30 ± 0.07 (z +4.3) | +1.93 ± 0.31 (z +6.1) | -1.37 ± 0.06 (z -22.2) |
| Canto-L | acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | +4.41 ± 0.59 (z +7.4) | +8.96 ± 1.52 (z +5.9) | -1.85 ± 6.76 (z -0.3) | +3.23 ± 0.46 (z +7.0) | +0.28 ± 0.07 (z +3.8) | +2.07 ± 0.30 (z +6.9) | -1.38 ± 0.05 (z -25.3) |
| Canto-L | acorn (ttbar) + walkthrough + merging (Canto builder family) | +3.76 ± 0.69 (z +5.4) | +9.02 ± 2.03 (z +4.4) | -2.78 ± 6.66 (z -0.4) | +2.68 ± 0.49 (z +5.5) | +0.09 ± 0.04 (z +2.2) | +1.21 ± 0.29 (z +4.2) | -0.77 ± 0.04 (z -19.6) |
| Canto-L | acorn (ttbar) + MWS-layer + merging (Canto builder) | +3.08 ± 0.50 (z +6.1) | +7.13 ± 1.72 (z +4.1) | -1.85 ± 6.07 (z -0.3) | +2.21 ± 0.34 (z +6.5) | +0.16 ± 0.05 (z +3.3) | +0.38 ± 0.24 (z +1.6) | -0.61 ± 0.02 (z -24.8) |
| Canto-L + MWS-layer + merging | acorn (HNL) + MWS-layer + merging (Canto builder) | +1.35 ± 0.22 (z +6.1) | +2.28 ± 0.72 (z +3.2) | +0.00 ± 2.97 (z +0.0) | +1.02 ± 0.18 (z +5.6) | +0.08 ± 0.04 (z +2.0) | +1.32 ± 0.35 (z +3.8) | -0.29 ± 0.06 (z -4.9) |

Primary efficiency per test set (descriptive; seed mean ± total sigma; difference with the same estimator):

| test set | primary particles per seed | model | model value | acorn (HNL) + MWS-layer + merging | difference |
|---|---|---|---|---|---|
| hnl10_100_test | 603 | Canto-L | 0.9923 ± 0.0033 | 0.9829 ± 0.0049 | +0.94 ± 0.48 (z +1.9) |
| hnl10_100_test | 603 | Canto-L + MWS-layer + merging | 0.9934 ± 0.0031 | 0.9829 ± 0.0049 | +1.05 ± 0.43 (z +2.4) |
| hnl15_200_test | 549 | Canto-L | 0.9848 ± 0.0047 | 0.9672 ± 0.0072 | +1.76 ± 0.64 (z +2.8) |
| hnl15_200_test | 549 | Canto-L + MWS-layer + merging | 0.9860 ± 0.0047 | 0.9672 ± 0.0072 | +1.88 ± 0.62 (z +3.0) |
| hnl15_100_external | 3861 | Canto-L | 0.9914 ± 0.0017 | 0.9781 ± 0.0021 | +1.33 ± 0.23 (z +5.8) |
| hnl15_100_external | 3861 | Canto-L + MWS-layer + merging | 0.9913 ± 0.0019 | 0.9781 ± 0.0021 | +1.32 ± 0.24 (z +5.4) |

Baseline of these rows: acorn (HNL) + MWS-layer + merging (Canto builder).

## Differences on val (Canto-L minus baseline; paired event bootstrap of the val tables of feedback point 5, B = 1000, + both seed terms; validation)

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L | acorn (HNL) + CC, best cut <= FR* | +3.48 ± 0.88 (z +3.9) | +3.19 ± 1.61 (z +2.0) | +7.02 ± 9.59 (z +0.7) | +3.31 ± 0.86 (z +3.9) | +1.03 ± 0.30 (z +3.5) | +11.76 ± 1.23 (z +9.5) | -1.89 ± 0.07 (z -27.2) |
| Canto-L | acorn (HNL) + CC, preregistered cut (claims' reference) | +5.26 ± 0.59 (z +8.8) | +4.19 ± 1.70 (z +2.5) | +7.02 ± 9.59 (z +0.7) | +5.30 ± 0.59 (z +9.0) | +1.64 ± 0.41 (z +4.0) | +14.21 ± 0.35 (z +40.6) | -1.90 ± 0.06 (z -31.7) |
| Canto-L | acorn (ttbar) + CC, preregistered cut (= best cut) | +6.79 ± 0.94 (z +7.2) | +8.38 ± 3.13 (z +2.7) | +22.81 ± 9.89 (z +2.3) | +5.88 ± 0.85 (z +6.9) | +1.09 ± 0.32 (z +3.4) | +9.08 ± 0.61 (z +15.0) | -1.94 ± 0.06 (z -32.9) |
| Canto-L | acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | +0.75 ± 0.32 (z +2.3) | +0.80 ± 1.20 (z +0.7) | +5.26 ± 9.72 (z +0.5) | +0.51 ± 0.25 (z +2.1) | +0.05 ± 0.05 (z +1.1) | +2.13 ± 0.26 (z +8.3) | -0.53 ± 0.06 (z -8.4) |
| Canto-L | acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +0.73 ± 0.32 (z +2.3) | +0.80 ± 1.20 (z +0.7) | +5.26 ± 9.72 (z +0.5) | +0.49 ± 0.24 (z +2.0) | +0.05 ± 0.05 (z +1.1) | +2.11 ± 0.24 (z +8.9) | -0.52 ± 0.06 (z -8.2) |
| Canto-L | acorn (HNL) + walkthrough + merging (Canto builder family) | +0.53 ± 0.30 (z +1.8) | +0.80 ± 1.20 (z +0.7) | -1.75 ± 7.08 (z -0.2) | +0.35 ± 0.22 (z +1.6) | +0.04 ± 0.05 (z +0.7) | +1.93 ± 0.26 (z +7.5) | -0.51 ± 0.06 (z -8.4) |
| Canto-L | acorn (HNL) + MWS-layer + merging (Canto builder) | +0.73 ± 0.30 (z +2.4) | +0.80 ± 1.20 (z +0.7) | -1.75 ± 6.94 (z -0.3) | +0.55 ± 0.23 (z +2.4) | +0.04 ± 0.05 (z +0.7) | +1.41 ± 0.32 (z +4.4) | -0.41 ± 0.05 (z -8.0) |
| Canto-L | acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | +3.68 ± 0.92 (z +4.0) | +6.19 ± 2.37 (z +2.6) | +24.56 ± 11.28 (z +2.2) | +2.79 ± 0.71 (z +4.0) | +0.07 ± 0.05 (z +1.4) | +1.89 ± 0.38 (z +5.0) | -1.44 ± 0.08 (z -17.9) |
| Canto-L | acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | +3.81 ± 0.70 (z +5.5) | +5.99 ± 1.86 (z +3.2) | +22.81 ± 10.64 (z +2.1) | +2.89 ± 0.54 (z +5.3) | +0.07 ± 0.05 (z +1.5) | +2.00 ± 0.35 (z +5.7) | -1.45 ± 0.07 (z -20.0) |
| Canto-L | acorn (ttbar) + walkthrough + merging (Canto builder family) | +3.11 ± 0.60 (z +5.2) | +7.58 ± 2.40 (z +3.2) | +14.04 ± 7.05 (z +2.0) | +2.39 ± 0.48 (z +5.0) | +0.00 ± 0.00 (identical) | +1.13 ± 0.35 (z +3.2) | -0.76 ± 0.05 (z -15.9) |
| Canto-L | acorn (ttbar) + MWS-layer + merging (Canto builder) | +2.33 ± 0.48 (z +4.9) | +4.59 ± 1.72 (z +2.7) | +14.04 ± 7.05 (z +2.0) | +1.72 ± 0.36 (z +4.8) | +0.09 ± 0.10 (z +0.9) | +0.29 ± 0.27 (z +1.1) | -0.63 ± 0.04 (z -16.7) |
| Canto-L + MWS-layer + merging | acorn (HNL) + MWS-layer + merging (Canto builder) | +0.78 ± 0.29 (z +2.6) | +0.80 ± 1.20 (z +0.7) | -3.51 ± 7.24 (z -0.5) | +0.58 ± 0.23 (z +2.6) | +0.04 ± 0.05 (z +0.7) | +1.16 ± 0.32 (z +3.7) | -0.31 ± 0.07 (z -4.6) |

Points = percentage points (fake rate: a negative difference means Canto-L has the lower fake rate). A row's own sigma (bootstrap + seed term) is in the CSV files.

## Notes

- acorn's walkthrough builder is CCandWalk (acorn/stages/track_building/models/cc_and_walk.py: connected components for the simple-path components plus a walkthrough on the branching ones, "a la L2IT", the GNN4ITk builder; exported by acorn.stages.track_building, unlike the older Walkthrough class; FastWalkthrough re-implements the same algorithm for GPUs). acorn's example configurations: CTD 2023 score_cut_cc 0.01, walk min 0.1 / add 0.6, plain walk; Rel. 24, ACAT 2025 and the CI the same thresholds with score_guided_walk True and guidance_mode_walk 2. Both variants are rows here.
- Code: hnl.trackbuild.acorn_walk runs acorn's own functions (remove_cycles, filter_graph, topological_sort_graph, get_simple_path, the choose_* road functions) on the stored scored graphs (stored edge order, float32 values); the two walk loops are line-by-line copies that keep the used hits in a set (acorn rebuilds a list with np.unique at every step) with every iteration order unchanged; acorn's add_track_labels fails under pandas 3 (fillna(-1) on its string column) and is replaced by the identical hit-to-track mapping. Identical track partitions to acorn's unmodified CCandWalk._build_tracks_one_evt in 7,600 of 7,600 val event-configurations (every 30th val graph, 100 events (34 ttbar, 66 HNL), of acorn (HNL) seed 0 and of acorn (ttbar) seed 0, all 38 configurations of the first grid; CPU 487 s against 9,287 s), 800 of 800 for the lower-edge extension (acorn (ttbar) seed 0, every 60th val graph, 16 configurations) and 1,600 of 1,600 on test at the val-chosen configurations (every 20th test graph, seed 0 of both arms); tests/test_acorn_walk.py checks the same on synthetic events.
- Grid per variant (score_cut_cc, min, add), 27 configurations: (0.01, m, 0.6) for m = 0.02, 0.05, 0.1, 0.15, ..., 0.4, 0.5, 0.6; (0.01, m, a) for m = 0.05, 0.1, 0.2 and a = 0.4, 0.8; (c, c, 0.6) for c = 0.1, 0.3; and, added after acorn (ttbar) chose min 0.02 (the first grid's lowest value) on val, the lower-edge extension (c, c, 0.6) for c = 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0 (c = 0: every edge kept, the walk never stops on score). acorn's default (0.01, 0.1, 0.6) is in the grid. Grids RUN/trackbuild_val_acwalk + RUN/trackbuild_val_acwalk_ext, united in RUN/trackbuild_val_acwalk_all (hnl.eval.strong_baselines merge; a data copy).
- Our walkthrough + merging: the extended grid of 2026-10-04 (RUN/trackbuild_val_ext: min 0.1-0.5 x add 0.5 / 0.7, merging rms 1.0). MWS-layer + merging: the grids and the per-seed configurations of ranking v3 (reproduced exactly).
- Configuration per seed on val: the builder rule at FR* with the one-standard-error tie rule (hnl.eval.ranking tie_se; hnl.eval.trackbuild_select.tie_best). Test: the val-chosen configurations (results/strong_baselines_v1_spec_test.json) applied unchanged to the stored test scores (acorn: RUN/gnn_scores_test/testset.tar of the one-time test chain; Canto-L: RUN/gnn_scores/testset.tar) with slurm/feedback/baseline_test.sbatch -> runs/feedback_v1/strong_baselines/<arm or canto_l>_s<seed>/<key>_test, one configuration per output with provenance.json; configuration and provenance checked against the spec before the summary.
- Test summaries: the hnl.eval.ranking_test code path (hnl.eval.test_stats.RunTest with the event-count check, the preregistered populations and sets). Differences: hnl.eval.bootstrap.linear_combination on the preregistered test tables (B = 500, seed 12345), the claims' estimator; identity check: Canto-L minus acorn (HNL) + CC at the preregistered cut reproduces results/claims_test_v1.json (value, sigma_boot, sigma_seeds and sigma to 1e-12). Val differences: the val tables of feedback point 5 (hnl.eval.bootstrap_tables; B = 1000, seed 20261005).
- Canto-L + MWS-layer + merging: the headline baseline's builder on Canto-L's scores (mwslayermerge 0.2 for every seed: the builder rule on val, identical to results/final_selection_v1_seeds.csv); the frozen Canto-L keeps learned merging (R6/R8 are not revisited).
- Status: validation numbers and post-freeze descriptive test summaries; nothing is chosen on test; the preregistered claims (results/claims_test_v1.json) stay as they are.
- Jobs: val grids slurm/feedback/acwalk_val.sbatch (jobs 6692378-6692383, extension 6692467-6692472), identity checks slurm/feedback/acwalk_check.sbatch (6692384, 6692385, 6692482, 6692920, 6692921), test slurm/feedback/baseline_test.sbatch (6692386-6692394, 6692914-6692919 and the dependent follow-ups 6693052-6693057 / 6693231-6693232, which re-ran the configurations whose first job ran out of time; every output's provenance.json names the job that wrote it). Code: the working tree on top of c98b573 (uncommitted at run time).
