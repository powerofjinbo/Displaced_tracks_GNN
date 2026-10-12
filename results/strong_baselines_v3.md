# The acorn baselines with Canto's full track-builder menu (feedback round 3, task A2)

acorn (ttbar), acorn (HNL), acorn (ttbar+HNL) = acorn's interaction GNN trained on ttbar only, on the HNL signal (Wang et al.) and on both (Stage-0 arms i, ii, iii; three seeds each, the Stage-0 scores). Builders on those scores: connected components (CC), acorn's own CC + walkthrough (CCandWalk, plain and score-guided), walkthrough + merging, MWS-layer + merging, two-pass MWS-layer + merging and MWS-layer + learned merging (a fragment-merging GBDT per run, trained on that run's own GNN scores of 2000 of its training graphs, as Canto's). Grids: those of rounds 1-2 and Canto's extended grid, extended on val past every edge at which a choice sat (edge-extension rule, docs/feedback/round3_A2_decisions.md). Every configuration is chosen per seed on val by the builder rule at FR* = 0.029368 (pooled HNL val) with the one-standard-error tie rule; the val-best row of an arm applies the same rule per seed over the families' choices and CC at its best cut. Nothing is chosen on test. Cells: mean ± seed sd over three seeds; |d0| bins: HNL-daughter targets pooled over the seeds (k/n).

## Validation (pooled HNL val 1,999 events + ttbar val 1,000 events)

**acorn (ttbar)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar) + val-best builder of the full menu (chosen per seed) | 3 | 0.9717 ± 0.0023 | 0.938 (470/501) | 44/57 (0.772) | 0.9979 ± 0.0000 | 0.9549 ± 0.0035 | 0.0153 ± 0.0006 | mwslmerge 0.03; mwslmerge 0.02; mwslmerge 0.05 |
| acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | 3 | 0.9717 ± 0.0023 | 0.938 (470/501) | 44/57 (0.772) | 0.9979 ± 0.0000 | 0.9549 ± 0.0035 | 0.0153 ± 0.0006 | mwslmerge 0.03; mwslmerge 0.02; mwslmerge 0.05 |
| acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9715 ± 0.0042 | 0.936 (469/501) | 43/57 (0.754) | 0.9970 ± 0.0015 | 0.9595 ± 0.0029 | 0.0168 ± 0.0002 | mws2passmerge 0.1,0.05; mws2passmerge 0.1,0.02; mws2passmerge 0.1,0.02 |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | 3 | 0.9677 ± 0.0047 | 0.930 (466/501) | 42/57 (0.737) | 0.9970 ± 0.0015 | 0.9603 ± 0.0026 | 0.0166 ± 0.0003 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.1 |
| acorn (ttbar) + walkthrough + merging (Canto builder family) | 3 | 0.9599 ± 0.0068 | 0.902 (452/501) | 42/57 (0.737) | 0.9979 ± 0.0000 | 0.9519 ± 0.0046 | 0.0179 ± 0.0005 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.7,1.0; walkmerge 0.2,0.3,1.0 |
| acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9542 ± 0.0137 | 0.914 (458/501) | 36/57 (0.632) | 0.9971 ± 0.0003 | 0.9443 ± 0.0053 | 0.0247 ± 0.0012 | acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6 |
| acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9529 ± 0.0087 | 0.916 (459/501) | 37/57 (0.649) | 0.9971 ± 0.0003 | 0.9431 ± 0.0047 | 0.0248 ± 0.0010 | acwalkg 0.01,0.01,0.6; acwalkg 0.005,0.005,0.6; acwalkg 0.01,0.01,0.6 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 3 | 0.9231 ± 0.0128 | 0.892 (447/501) | 37/57 (0.649) | 0.9870 ± 0.0031 | 0.8723 ± 0.0094 | 0.0297 ± 0.0003 | CC 0.34 (FR* unreachable: minimum-rate cut); CC 0.33; CC 0.31 (FR* unreachable: minimum-rate cut) |

**acorn (HNL)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| acorn (HNL) + MWS-layer + learned merging (Canto-L's builder) | 3 | 0.9865 ± 0.0020 | 0.972 (487/501) | 52/57 (0.912) | 0.9975 ± 0.0006 | 0.9491 ± 0.0012 | 0.0137 ± 0.0004 | mwslmerge 0.03; mwslmerge 0.2; mwslmerge 0.05 |
| acorn (HNL) + val-best builder of the full menu (chosen per seed) | 3 | 0.9865 ± 0.0020 | 0.972 (487/501) | 52/57 (0.912) | 0.9975 ± 0.0006 | 0.9491 ± 0.0012 | 0.0137 ± 0.0004 | mwslmerge 0.03; mwslmerge 0.2; mwslmerge 0.05 |
| acorn (HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9852 ± 0.0011 | 0.968 (485/501) | 51/57 (0.895) | 0.9975 ± 0.0006 | 0.9434 ± 0.0020 | 0.0154 ± 0.0009 | walkmerge 0.2,0.3,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (HNL) + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9850 ± 0.0013 | 0.968 (485/501) | 52/57 (0.912) | 0.9975 ± 0.0006 | 0.9472 ± 0.0046 | 0.0146 ± 0.0006 | mws2passmerge 0.1,0.05; mws2passmerge 0.2,0.1; mws2passmerge 0.2,0.1 |
| acorn (HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9837 ± 0.0004 | 0.968 (485/501) | 51/57 (0.895) | 0.9975 ± 0.0006 | 0.9490 ± 0.0040 | 0.0143 ± 0.0007 | mwslayermerge 0.1; mwslayermerge 0.2; mwslayermerge 0.1 |
| acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9837 ± 0.0011 | 0.968 (485/501) | 47/57 (0.825) | 0.9973 ± 0.0005 | 0.9420 ± 0.0009 | 0.0155 ± 0.0010 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.2,0.4 |
| acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9835 ± 0.0015 | 0.968 (485/501) | 47/57 (0.825) | 0.9973 ± 0.0005 | 0.9418 ± 0.0018 | 0.0156 ± 0.0009 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.2,0.4 |
| acorn (HNL) + CC, best cut <= FR* | 3 | 0.9562 ± 0.0128 | 0.944 (473/501) | 46/57 (0.807) | 0.9875 ± 0.0027 | 0.8456 ± 0.0208 | 0.0292 ± 0.0005 | CC 0.38; CC 0.36 (FR* unreachable: minimum-rate cut); CC 0.44 |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 3 | 0.9384 ± 0.0026 | 0.934 (468/501) | 46/57 (0.807) | 0.9815 ± 0.0048 | 0.8210 ± 0.0034 | 0.0293 ± 0.0005 | CC 0.26; CC 0.36 (FR* unreachable: minimum-rate cut); CC 0.27 |

**acorn (ttbar+HNL)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder) | 3 | 0.9790 ± 0.0039 | 0.956 (479/501) | 48/57 (0.842) | 0.9975 ± 0.0003 | 0.9518 ± 0.0056 | 0.0139 ± 0.0009 | mwslmerge 0.2; mwslmerge 0.03; mwslmerge 0.05 |
| acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | 3 | 0.9790 ± 0.0039 | 0.956 (479/501) | 48/57 (0.842) | 0.9975 ± 0.0003 | 0.9518 ± 0.0056 | 0.0139 ± 0.0009 | mwslmerge 0.2; mwslmerge 0.03; mwslmerge 0.05 |
| acorn (ttbar+HNL) + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9787 ± 0.0019 | 0.958 (480/501) | 47/57 (0.825) | 0.9971 ± 0.0003 | 0.9569 ± 0.0005 | 0.0152 ± 0.0018 | mws2passmerge 0.1,0.05; mws2passmerge 0.1,0.05; mws2passmerge 0.2,0.1 |
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9787 ± 0.0035 | 0.952 (477/501) | 47/57 (0.825) | 0.9977 ± 0.0003 | 0.9495 ± 0.0065 | 0.0149 ± 0.0012 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9782 ± 0.0020 | 0.956 (479/501) | 47/57 (0.825) | 0.9971 ± 0.0003 | 0.9570 ± 0.0007 | 0.0150 ± 0.0019 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.2 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9740 ± 0.0017 | 0.950 (476/501) | 41/57 (0.719) | 0.9977 ± 0.0003 | 0.9487 ± 0.0065 | 0.0159 ± 0.0014 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9737 ± 0.0015 | 0.950 (476/501) | 41/57 (0.719) | 0.9977 ± 0.0003 | 0.9487 ± 0.0065 | 0.0159 ± 0.0014 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6 |
| acorn (ttbar+HNL) + CC, best cut <= FR* | 3 | 0.9504 ± 0.0072 | 0.922 (462/501) | 42/57 (0.737) | 0.9913 ± 0.0015 | 0.8584 ± 0.0109 | 0.0287 ± 0.0003 | CC 0.37; CC 0.38; CC 0.38 |
| acorn (ttbar+HNL) + CC, preregistered cut rule | 3 | 0.9394 ± 0.0227 | 0.916 (459/501) | 42/57 (0.737) | 0.9888 ± 0.0065 | 0.8579 ± 0.0084 | 0.0289 ± 0.0002 | CC 0.26; CC 0.38; CC 0.39 |

**Canto (reference, decomposition and same-builder rows)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| Canto-L + walkthrough + merging (Canto builder family), same builder (not the frozen builder) | 3 | 0.9922 ± 0.0004 | 0.970 (486/501) | 48/57 (0.842) | 0.9979 ± 0.0000 | 0.9604 ± 0.0035 | 0.0113 ± 0.0004 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| Canto-L + two-pass MWS-layer + merging (Canto builder family), same builder (not the frozen builder) | 3 | 0.9920 ± 0.0016 | 0.978 (490/501) | 50/57 (0.877) | 0.9979 ± 0.0000 | 0.9633 ± 0.0014 | 0.0125 ± 0.0018 | mws2passmerge 0.2,0.05; mws2passmerge 0.2,0.1; mws2passmerge 0.2,0.1 |
| Canto-L + MWS-layer + merging (Canto builder), same builder (not the frozen builder) | 3 | 0.9915 ± 0.0017 | 0.976 (489/501) | 49/57 (0.860) | 0.9979 ± 0.0000 | 0.9606 ± 0.0035 | 0.0112 ± 0.0009 | mwslayermerge 0.2; mwslayermerge 0.2; mwslayermerge 0.2 |
| Canto-L (frozen: learned merging) | 3 | 0.9910 ± 0.0020 | 0.976 (489/501) | 50/57 (0.877) | 0.9979 ± 0.0000 | 0.9631 ± 0.0037 | 0.0103 ± 0.0004 | mwslmerge 0.05; mwslmerge 0.03; mwslmerge 0.1 |
| Canto-B + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9850 ± 0.0023 | 0.962 (482/501) | 45/57 (0.789) | 0.9975 ± 0.0003 | 0.9648 ± 0.0003 | 0.0157 ± 0.0007 | mws2passmerge 0.2,0.1; mws2passmerge 0.2,0.1; mws2passmerge 0.2,0.05 |
| Canto-B (frozen: learned merging) | 3 | 0.9845 ± 0.0016 | 0.962 (482/501) | 45/57 (0.789) | 0.9971 ± 0.0003 | 0.9657 ± 0.0006 | 0.0130 ± 0.0005 | mwslmerge 0.05; mwslmerge 0.2; mwslmerge 0.1 |
| Canto-B + walkthrough + merging (Canto builder family) | 3 | 0.9835 ± 0.0034 | 0.946 (474/501) | 42/57 (0.737) | 0.9979 ± 0.0000 | 0.9636 ± 0.0007 | 0.0145 ± 0.0008 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.7,1.0 |
| Canto-B + MWS-layer + merging (Canto builder) | 3 | 0.9825 ± 0.0037 | 0.958 (480/501) | 43/57 (0.754) | 0.9971 ± 0.0003 | 0.9668 ± 0.0047 | 0.0144 ± 0.0007 | mwslayermerge 0.2; mwslayermerge 0.1; mwslayermerge 0.2 |
| Canto-B + CC, best cut <= FR* | 3 | 0.9704 ± 0.0022 | 0.954 (478/501) | 37/57 (0.649) | 0.9952 ± 0.0000 | 0.8844 ± 0.0006 | 0.0276 ± 0.0005 | CC 0.40; CC 0.38; CC 0.40 |

## Val-best configuration per arm and seed (full menu)

| arm | seed | chosen | primary | fake rate | candidates within one standard error (tied) |
|---|---|---|---|---|---|
| acorn (ttbar) | 0 | mwslmerge 0.03 | 0.9737 | 0.0160 | mwslmerge 0.03 0.9737 / 0.0160; mws2passmerge 0.1,0.05 0.9722 / 0.0170; mwslayermerge 0.1 0.9715 / 0.0169 |
| acorn (ttbar) | 1 | mwslmerge 0.02 | 0.9692 | 0.0149 | mwslmerge 0.02 0.9692 / 0.0149; mws2passmerge 0.1,0.02 0.9669 / 0.0165 |
| acorn (ttbar) | 2 | mwslmerge 0.05 | 0.9722 | 0.0150 | mws2passmerge 0.1,0.02 0.9752 / 0.0169; mwslmerge 0.05 0.9722 / 0.0150 |
| acorn (HNL) | 0 | mwslmerge 0.03 | 0.9880 | 0.0134 | mwslmerge 0.03 0.9880 / 0.0134 |
| acorn (HNL) | 1 | mwslmerge 0.2 | 0.9842 | 0.0142 | walkmerge 0.2,0.5,1.0 0.9850 / 0.0163; mws2passmerge 0.2,0.1 0.9842 / 0.0152; mwslmerge 0.2 0.9842 / 0.0142; acwalk 0.01,0.15,0.6 0.9835 / 0.0166; acwalkg 0.01,0.15,0.6 0.9835 / 0.0165; mwslayermerge 0.2 0.9835 / 0.0151 |
| acorn (HNL) | 2 | mwslmerge 0.05 | 0.9872 | 0.0135 | mwslmerge 0.05 0.9872 / 0.0135; walkmerge 0.2,0.5,1.0 0.9865 / 0.0146; mws2passmerge 0.2,0.1 0.9865 / 0.0146 |
| acorn (ttbar+HNL) | 0 | mwslmerge 0.2 | 0.9745 | 0.0129 | walkmerge 0.2,0.5,1.0 0.9767 / 0.0137; mws2passmerge 0.1,0.05 0.9767 / 0.0137; mwslayermerge 0.1 0.9760 / 0.0135; mwslmerge 0.2 0.9745 / 0.0129; acwalk 0.01,0.15,0.6 0.9730 / 0.0147 |
| acorn (ttbar+HNL) | 1 | mwslmerge 0.03 | 0.9812 | 0.0143 | mwslmerge 0.03 0.9812 / 0.0143; mws2passmerge 0.1,0.05 0.9805 / 0.0146; mwslayermerge 0.1 0.9797 / 0.0144 |
| acorn (ttbar+HNL) | 2 | mwslmerge 0.05 | 0.9812 | 0.0146 | walkmerge 0.2,0.5,1.0 0.9827 / 0.0161; mwslmerge 0.05 0.9812 / 0.0146 |

For comparison, round 2's rule (the arm's row with the highest three-seed-mean val primary efficiency, ties to the lower fake rate): acorn (ttbar): MWS-layer + learned merging (Canto-L's builder); acorn (HNL): MWS-layer + learned merging (Canto-L's builder); acorn (ttbar+HNL): MWS-layer + learned merging (Canto-L's builder).

## Grid edges after the extension (val)

| run | family | chosen | primary | fake rate | configurations | extension steps | edges | status |
|---|---|---|---|---|---|---|---|---|
| armi_s0 | acwalk | 0.01,0.01,0.6 | 0.9624 | 0.0240 | 30 | 1 | interior | interior |
| armi_s0 | acwalkg | 0.01,0.01,0.6 | 0.9609 | 0.0240 | 30 | 1 | interior | interior |
| armi_s0 | walkmerge | 0.2,0.5,1.0 | 0.9677 | 0.0176 | 26 | 1 | interior | interior |
| armi_s0 | mwslayermerge | 0.1 | 0.9715 | 0.0169 | 8 | 1 | interior | interior |
| armi_s0 | mws2passmerge | 0.1,0.05 | 0.9722 | 0.0170 | 30 | 2 | interior | interior |
| armi_s0 | mwslmerge | 0.03 | 0.9737 | 0.0160 | 9 | 0 | interior | interior |
| armi_s1 | acwalk | 0.01,0.01,0.6 | 0.9384 | 0.0240 | 30 | 1 | interior | interior |
| armi_s1 | acwalkg | 0.005,0.005,0.6 | 0.9437 | 0.0244 | 30 | 1 | interior | interior |
| armi_s1 | walkmerge | 0.2,0.7,1.0 | 0.9549 | 0.0175 | 26 | 1 | interior | interior |
| armi_s1 | mwslayermerge | 0.1 | 0.9624 | 0.0163 | 8 | 1 | interior | interior |
| armi_s1 | mws2passmerge | 0.1,0.02 | 0.9669 | 0.0165 | 30 | 2 | interior | interior |
| armi_s1 | mwslmerge | 0.02 | 0.9692 | 0.0149 | 9 | 0 | interior | interior |
| armi_s2 | acwalk | 0.01,0.01,0.6 | 0.9617 | 0.0260 | 30 | 1 | interior | interior |
| armi_s2 | acwalkg | 0.01,0.01,0.6 | 0.9542 | 0.0259 | 30 | 1 | interior | interior |
| armi_s2 | walkmerge | 0.2,0.3,1.0 | 0.9572 | 0.0185 | 26 | 1 | interior | interior |
| armi_s2 | mwslayermerge | 0.1 | 0.9692 | 0.0166 | 8 | 1 | interior | interior |
| armi_s2 | mws2passmerge | 0.1,0.02 | 0.9752 | 0.0169 | 30 | 2 | interior | interior |
| armi_s2 | mwslmerge | 0.05 | 0.9722 | 0.0150 | 9 | 0 | interior | interior |
| armii_s0 | acwalk | 0.01,0.15,0.6 | 0.9850 | 0.0154 | 30 | 1 | interior | interior |
| armii_s0 | acwalkg | 0.01,0.15,0.6 | 0.9850 | 0.0154 | 30 | 1 | interior | interior |
| armii_s0 | walkmerge | 0.2,0.3,1.0 | 0.9842 | 0.0152 | 26 | 1 | interior | interior |
| armii_s0 | mwslayermerge | 0.1 | 0.9842 | 0.0140 | 8 | 1 | interior | interior |
| armii_s0 | mws2passmerge | 0.1,0.05 | 0.9842 | 0.0140 | 30 | 2 | interior | interior |
| armii_s0 | mwslmerge | 0.03 | 0.9880 | 0.0134 | 9 | 0 | interior | interior |
| armii_s1 | acwalk | 0.01,0.15,0.6 | 0.9835 | 0.0166 | 30 | 1 | interior | interior |
| armii_s1 | acwalkg | 0.01,0.15,0.6 | 0.9835 | 0.0165 | 30 | 1 | interior | interior |
| armii_s1 | walkmerge | 0.2,0.5,1.0 | 0.9850 | 0.0163 | 26 | 1 | interior | interior |
| armii_s1 | mwslayermerge | 0.2 | 0.9835 | 0.0151 | 8 | 1 | interior | interior |
| armii_s1 | mws2passmerge | 0.2,0.1 | 0.9842 | 0.0152 | 30 | 2 | interior | interior |
| armii_s1 | mwslmerge | 0.2 | 0.9842 | 0.0142 | 9 | 0 | interior | interior |
| armii_s2 | acwalk | 0.01,0.2,0.4 | 0.9820 | 0.0147 | 30 | 1 | interior | interior |
| armii_s2 | acwalkg | 0.01,0.2,0.4 | 0.9827 | 0.0146 | 30 | 1 | interior | interior |
| armii_s2 | walkmerge | 0.2,0.5,1.0 | 0.9865 | 0.0146 | 26 | 1 | interior | interior |
| armii_s2 | mwslayermerge | 0.1 | 0.9835 | 0.0139 | 8 | 1 | interior | interior |
| armii_s2 | mws2passmerge | 0.2,0.1 | 0.9865 | 0.0146 | 30 | 2 | interior | interior |
| armii_s2 | mwslmerge | 0.05 | 0.9872 | 0.0135 | 9 | 0 | interior | interior |
| armiii_s0 | acwalk | 0.01,0.15,0.6 | 0.9730 | 0.0147 | 30 | 1 | interior | interior |
| armiii_s0 | acwalkg | 0.01,0.15,0.6 | 0.9722 | 0.0147 | 30 | 1 | interior | interior |
| armiii_s0 | walkmerge | 0.2,0.5,1.0 | 0.9767 | 0.0137 | 26 | 1 | interior | interior |
| armiii_s0 | mwslayermerge | 0.1 | 0.9760 | 0.0135 | 8 | 1 | interior | interior |
| armiii_s0 | mws2passmerge | 0.1,0.05 | 0.9767 | 0.0137 | 30 | 2 | interior | interior |
| armiii_s0 | mwslmerge | 0.2 | 0.9745 | 0.0129 | 9 | 0 | interior | interior |
| armiii_s1 | acwalk | 0.01,0.15,0.6 | 0.9730 | 0.0156 | 30 | 1 | interior | interior |
| armiii_s1 | acwalkg | 0.01,0.15,0.6 | 0.9737 | 0.0156 | 30 | 1 | interior | interior |
| armiii_s1 | walkmerge | 0.2,0.5,1.0 | 0.9767 | 0.0148 | 26 | 1 | interior | interior |
| armiii_s1 | mwslayermerge | 0.1 | 0.9797 | 0.0144 | 8 | 1 | interior | interior |
| armiii_s1 | mws2passmerge | 0.1,0.05 | 0.9805 | 0.0146 | 30 | 2 | interior | interior |
| armiii_s1 | mwslmerge | 0.03 | 0.9812 | 0.0143 | 9 | 0 | interior | interior |
| armiii_s2 | acwalk | 0.01,0.15,0.6 | 0.9760 | 0.0174 | 30 | 1 | interior | interior |
| armiii_s2 | acwalkg | 0.01,0.15,0.6 | 0.9752 | 0.0174 | 30 | 1 | interior | interior |
| armiii_s2 | walkmerge | 0.2,0.5,1.0 | 0.9827 | 0.0161 | 26 | 1 | interior | interior |
| armiii_s2 | mwslayermerge | 0.2 | 0.9790 | 0.0171 | 8 | 1 | interior | interior |
| armiii_s2 | mws2passmerge | 0.2,0.1 | 0.9790 | 0.0172 | 30 | 2 | interior | interior |
| armiii_s2 | mwslmerge | 0.05 | 0.9812 | 0.0146 | 9 | 0 | interior | interior |

## Test, post-freeze descriptive (not claims; the preregistered claims stay as they are)

Pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) for primary, |d0|, W muons and fake rate; ttbar_test (1,000 events) for ttbar prompt. The val-chosen configurations applied once to the stored test scores (an output of rounds 1-2 is reused where the run and configuration are the same); summary code path of hnl.eval.ranking_test.

**acorn (ttbar)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9642 ± 0.0056 | 0.928 (1626/1752) | 98/108 (0.907) | 0.9962 ± 0.0005 | 0.9589 ± 0.0024 | 0.0165 ± 0.0002 | mws2passmerge 0.1,0.05; mws2passmerge 0.1,0.02; mws2passmerge 0.1,0.02 |
| acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | 3 | 0.9628 ± 0.0088 | 0.918 (1609/1752) | 96/108 (0.889) | 0.9968 ± 0.0007 | 0.9540 ± 0.0028 | 0.0152 ± 0.0004 | mwslmerge 0.03; mwslmerge 0.02; mwslmerge 0.05 |
| acorn (ttbar) + val-best builder of the full menu (chosen per seed) | 3 | 0.9628 ± 0.0088 | 0.918 (1609/1752) | 96/108 (0.889) | 0.9968 ± 0.0007 | 0.9540 ± 0.0028 | 0.0152 ± 0.0004 | mwslmerge 0.03; mwslmerge 0.02; mwslmerge 0.05 |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | 3 | 0.9600 ± 0.0075 | 0.914 (1602/1752) | 96/108 (0.889) | 0.9962 ± 0.0005 | 0.9594 ± 0.0021 | 0.0163 ± 0.0001 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.1 |
| acorn (ttbar) + walkthrough + merging (Canto builder family) | 3 | 0.9533 ± 0.0111 | 0.896 (1569/1752) | 96/108 (0.889) | 0.9968 ± 0.0005 | 0.9508 ± 0.0040 | 0.0178 ± 0.0005 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.7,1.0; walkmerge 0.2,0.3,1.0 |
| acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9467 ± 0.0089 | 0.896 (1570/1752) | 96/108 (0.889) | 0.9949 ± 0.0008 | 0.9425 ± 0.0036 | 0.0240 ± 0.0008 | acwalkg 0.01,0.01,0.6; acwalkg 0.005,0.005,0.6; acwalkg 0.01,0.01,0.6 |
| acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9454 ± 0.0123 | 0.894 (1567/1752) | 95/108 (0.880) | 0.9947 ± 0.0006 | 0.9440 ± 0.0040 | 0.0239 ± 0.0010 | acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6; acwalk 0.01,0.01,0.6 |
| acorn (ttbar) + CC, preregistered cut (= best cut) | 3 | 0.9038 ± 0.0164 | 0.849 (1487/1752) | 86/108 (0.796) | 0.9844 ± 0.0016 | 0.8741 ± 0.0085 | 0.0295 ± 0.0005 | CC 0.34 (FR* unreachable on val: minimum-rate cut); CC 0.33; CC 0.31 (FR* unreachable on val: minimum-rate cut) |

**acorn (HNL)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| acorn (HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9801 ± 0.0009 | 0.963 (1687/1752) | 102/108 (0.944) | 0.9970 ± 0.0002 | 0.9420 ± 0.0018 | 0.0148 ± 0.0006 | walkmerge 0.2,0.3,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (HNL) + MWS-layer + learned merging (Canto-L's builder) | 3 | 0.9798 ± 0.0015 | 0.967 (1695/1752) | 101/108 (0.935) | 0.9969 ± 0.0004 | 0.9484 ± 0.0023 | 0.0136 ± 0.0004 | mwslmerge 0.03; mwslmerge 0.2; mwslmerge 0.05 |
| acorn (HNL) + val-best builder of the full menu (chosen per seed) | 3 | 0.9798 ± 0.0015 | 0.967 (1695/1752) | 101/108 (0.935) | 0.9969 ± 0.0004 | 0.9484 ± 0.0023 | 0.0136 ± 0.0004 | mwslmerge 0.03; mwslmerge 0.2; mwslmerge 0.05 |
| acorn (HNL) + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9783 ± 0.0012 | 0.962 (1685/1752) | 102/108 (0.944) | 0.9969 ± 0.0004 | 0.9462 ± 0.0049 | 0.0144 ± 0.0007 | mws2passmerge 0.1,0.05; mws2passmerge 0.2,0.1; mws2passmerge 0.2,0.1 |
| acorn (HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9775 ± 0.0002 | 0.961 (1683/1752) | 101/108 (0.935) | 0.9970 ± 0.0005 | 0.9478 ± 0.0043 | 0.0141 ± 0.0007 | mwslayermerge 0.1; mwslayermerge 0.2; mwslayermerge 0.1 |
| acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9774 ± 0.0006 | 0.957 (1677/1752) | 103/108 (0.954) | 0.9968 ± 0.0002 | 0.9400 ± 0.0006 | 0.0151 ± 0.0005 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.2,0.4 |
| acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9773 ± 0.0008 | 0.957 (1677/1752) | 103/108 (0.954) | 0.9967 ± 0.0002 | 0.9392 ± 0.0021 | 0.0152 ± 0.0006 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.2,0.4 |
| acorn (HNL) + CC, best cut <= FR* | 3 | 0.9448 ± 0.0136 | 0.931 (1631/1752) | 99/108 (0.917) | 0.9851 ± 0.0033 | 0.8448 ± 0.0229 | 0.0290 ± 0.0002 | CC 0.38 (post-freeze CC on the stored test scores); CC 0.36 (FR* unreachable on val: minimum-rate cut); CC 0.44 (post-freeze CC on the stored test scores) |
| acorn (HNL) + CC, preregistered cut (claims' reference) | 3 | 0.9257 ± 0.0057 | 0.917 (1607/1752) | 98/108 (0.907) | 0.9762 ± 0.0047 | 0.8167 ± 0.0032 | 0.0286 ± 0.0007 | CC 0.26; CC 0.36 (FR* unreachable on val: minimum-rate cut); CC 0.27 |

**acorn (ttbar+HNL)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | 3 | 0.9738 ± 0.0019 | 0.953 (1669/1752) | 99/108 (0.917) | 0.9969 ± 0.0002 | 0.9493 ± 0.0046 | 0.0148 ± 0.0011 | walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0; walkmerge 0.2,0.5,1.0 |
| acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder) | 3 | 0.9737 ± 0.0038 | 0.955 (1673/1752) | 98/108 (0.907) | 0.9965 ± 0.0001 | 0.9517 ± 0.0054 | 0.0138 ± 0.0008 | mwslmerge 0.2; mwslmerge 0.03; mwslmerge 0.05 |
| acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | 3 | 0.9737 ± 0.0038 | 0.955 (1673/1752) | 98/108 (0.907) | 0.9965 ± 0.0001 | 0.9517 ± 0.0054 | 0.0138 ± 0.0008 | mwslmerge 0.2; mwslmerge 0.03; mwslmerge 0.05 |
| acorn (ttbar+HNL) + two-pass MWS-layer + merging (Canto builder family) | 3 | 0.9732 ± 0.0032 | 0.954 (1671/1752) | 99/108 (0.917) | 0.9965 ± 0.0003 | 0.9566 ± 0.0009 | 0.0148 ± 0.0019 | mws2passmerge 0.1,0.05; mws2passmerge 0.1,0.05; mws2passmerge 0.2,0.1 |
| acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | 3 | 0.9729 ± 0.0032 | 0.953 (1669/1752) | 99/108 (0.917) | 0.9965 ± 0.0003 | 0.9566 ± 0.0013 | 0.0148 ± 0.0019 | mwslayermerge 0.1; mwslayermerge 0.1; mwslayermerge 0.2 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | 3 | 0.9693 ± 0.0017 | 0.939 (1646/1752) | 99/108 (0.917) | 0.9966 ± 0.0004 | 0.9481 ± 0.0049 | 0.0156 ± 0.0013 | acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6; acwalk 0.01,0.15,0.6 |
| acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | 3 | 0.9691 ± 0.0015 | 0.939 (1645/1752) | 99/108 (0.917) | 0.9964 ± 0.0006 | 0.9481 ± 0.0049 | 0.0156 ± 0.0013 | acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6; acwalkg 0.01,0.15,0.6 |
| acorn (ttbar+HNL) + CC, best cut <= FR* | 3 | 0.9401 ± 0.0060 | 0.914 (1602/1752) | 96/108 (0.889) | 0.9878 ± 0.0024 | 0.8612 ± 0.0126 | 0.0289 ± 0.0007 | CC 0.37 (post-freeze CC on the stored test scores); CC 0.38; CC 0.38 (post-freeze CC on the stored test scores) |
| acorn (ttbar+HNL) + CC, preregistered cut rule | 3 | 0.9300 ± 0.0248 | 0.911 (1596/1752) | 95/108 (0.880) | 0.9842 ± 0.0081 | 0.8604 ± 0.0096 | 0.0292 ± 0.0004 | CC 0.26; CC 0.38; CC 0.39 |

**Canto (reference, decomposition and same-builder rows)**

| model | seeds | primary | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate | working points (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | 3 | 0.9908 ± 0.0017 | 0.986 (1727/1752) | 94/108 (0.870) | 0.9977 ± 0.0001 | 0.9632 ± 0.0034 | 0.0102 ± 0.0003 | mwslmerge 0.05; mwslmerge 0.03; mwslmerge 0.1 |
| Canto-B (frozen: learned merging) | 3 | 0.9843 ± 0.0038 | 0.970 (1700/1752) | 97/108 (0.898) | 0.9978 ± 0.0001 | 0.9658 ± 0.0004 | 0.0129 ± 0.0007 | mwslmerge 0.05; mwslmerge 0.2; mwslmerge 0.1 |
| Canto-B + CC, best cut <= FR* | 3 | 0.9709 ± 0.0057 | 0.941 (1649/1752) | 100/108 (0.926) | 0.9933 ± 0.0003 | 0.8864 ± 0.0011 | 0.0283 ± 0.0008 | CC 0.40 (post-freeze CC on the stored test scores); CC 0.38 (post-freeze CC on the stored test scores); CC 0.40 (post-freeze CC on the stored test scores) |

Primary efficiency with the preregistered total sigma (event bootstrap + seed term):

- Canto-L (frozen: learned merging): 0.9908 ± 0.0015 (bootstrap 0.0012, seed sd 0.0017)
- Canto-B (frozen: learned merging): 0.9843 ± 0.0026 (bootstrap 0.0015, seed sd 0.0038)
- acorn (HNL) + walkthrough + merging (Canto builder family): 0.9801 ± 0.0019 (bootstrap 0.0018, seed sd 0.0009)
- acorn (HNL) + MWS-layer + learned merging (Canto-L's builder): 0.9798 ± 0.0020 (bootstrap 0.0018, seed sd 0.0015)
- acorn (HNL) + val-best builder of the full menu (chosen per seed): 0.9798 ± 0.0020 (bootstrap 0.0018, seed sd 0.0015)
- acorn (HNL) + two-pass MWS-layer + merging (Canto builder family): 0.9783 ± 0.0020 (bootstrap 0.0018, seed sd 0.0012)
- acorn (HNL) + MWS-layer + merging (Canto builder): 0.9775 ± 0.0018 (bootstrap 0.0018, seed sd 0.0002)
- acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk): 0.9774 ± 0.0020 (bootstrap 0.0019, seed sd 0.0006)
- acorn (HNL) + acorn's CC + walkthrough (CCandWalk): 0.9773 ± 0.0020 (bootstrap 0.0019, seed sd 0.0008)
- acorn (ttbar+HNL) + walkthrough + merging (Canto builder family): 0.9738 ± 0.0024 (bootstrap 0.0021, seed sd 0.0019)
- acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder): 0.9737 ± 0.0031 (bootstrap 0.0021, seed sd 0.0038)
- acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed): 0.9737 ± 0.0031 (bootstrap 0.0021, seed sd 0.0038)
- acorn (ttbar+HNL) + two-pass MWS-layer + merging (Canto builder family): 0.9732 ± 0.0028 (bootstrap 0.0021, seed sd 0.0032)
- acorn (ttbar+HNL) + MWS-layer + merging (Canto builder): 0.9729 ± 0.0028 (bootstrap 0.0021, seed sd 0.0032)
- Canto-B + CC, best cut <= FR*: 0.9709 ± 0.0040 (bootstrap 0.0022, seed sd 0.0057)
- acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk): 0.9693 ± 0.0025 (bootstrap 0.0022, seed sd 0.0017)
- acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk): 0.9691 ± 0.0024 (bootstrap 0.0022, seed sd 0.0015)
- acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family): 0.9642 ± 0.0040 (bootstrap 0.0023, seed sd 0.0056)
- acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder): 0.9628 ± 0.0056 (bootstrap 0.0023, seed sd 0.0088)
- acorn (ttbar) + val-best builder of the full menu (chosen per seed): 0.9628 ± 0.0056 (bootstrap 0.0023, seed sd 0.0088)
- acorn (ttbar) + MWS-layer + merging (Canto builder): 0.9600 ± 0.0050 (bootstrap 0.0024, seed sd 0.0075)
- acorn (ttbar) + walkthrough + merging (Canto builder family): 0.9533 ± 0.0069 (bootstrap 0.0026, seed sd 0.0111)
- acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk): 0.9467 ± 0.0059 (bootstrap 0.0029, seed sd 0.0089)
- acorn (ttbar) + acorn's CC + walkthrough (CCandWalk): 0.9454 ± 0.0077 (bootstrap 0.0029, seed sd 0.0123)
- acorn (HNL) + CC, best cut <= FR*: 0.9448 ± 0.0083 (bootstrap 0.0028, seed sd 0.0136)
- acorn (ttbar+HNL) + CC, best cut <= FR*: 0.9401 ± 0.0046 (bootstrap 0.0030, seed sd 0.0060)
- acorn (ttbar+HNL) + CC, preregistered cut rule: 0.9300 ± 0.0147 (bootstrap 0.0032, seed sd 0.0248)
- acorn (HNL) + CC, preregistered cut (claims' reference): 0.9257 ± 0.0047 (bootstrap 0.0034, seed sd 0.0057)
- acorn (ttbar) + CC, preregistered cut (= best cut): 0.9038 ± 0.0102 (bootstrap 0.0037, seed sd 0.0164)

## Differences on test (new minus baseline; paired event bootstrap of the preregistered test tables, B = 500, + both seed terms: the claims' estimator; post-freeze descriptive)

**Canto-L minus acorn**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) + walkthrough + merging (Canto builder family) | +1.07 ± 0.21 (z +5.2) | +2.28 ± 0.66 (z +3.5) | -7.41 ± 5.92 (z -1.3) | +0.76 ± 0.16 (z +4.6) | +0.07 ± 0.03 (z +2.4) | +2.12 ± 0.24 (z +9.0) | -0.46 ± 0.04 (z -11.1) |
| Canto-L (frozen: learned merging) | acorn (HNL) + MWS-layer + learned merging (Canto-L's builder) | +1.10 ± 0.22 (z +5.1) | +1.83 ± 0.62 (z +2.9) | -6.48 ± 5.91 (z -1.1) | +0.81 ± 0.17 (z +4.8) | +0.08 ± 0.03 (z +2.2) | +1.48 ± 0.25 (z +6.0) | -0.34 ± 0.03 (z -11.2) |
| Canto-L (frozen: learned merging) | acorn (HNL) + val-best builder of the full menu (chosen per seed) | +1.10 ± 0.22 (z +5.1) | +1.83 ± 0.62 (z +2.9) | -6.48 ± 5.91 (z -1.1) | +0.81 ± 0.17 (z +4.8) | +0.08 ± 0.03 (z +2.2) | +1.48 ± 0.25 (z +6.0) | -0.34 ± 0.03 (z -11.2) |
| Canto-L (frozen: learned merging) | acorn (HNL) + two-pass MWS-layer + merging (Canto builder family) | +1.24 ± 0.21 (z +5.9) | +2.40 ± 0.68 (z +3.5) | -7.41 ± 6.37 (z -1.2) | +0.93 ± 0.18 (z +5.3) | +0.08 ± 0.04 (z +2.2) | +1.70 ± 0.35 (z +4.8) | -0.42 ± 0.05 (z -9.0) |
| Canto-L (frozen: learned merging) | acorn (HNL) + MWS-layer + merging (Canto builder) | +1.33 ± 0.20 (z +6.6) | +2.51 ± 0.70 (z +3.6) | -6.48 ± 5.91 (z -1.1) | +1.00 ± 0.16 (z +6.1) | +0.07 ± 0.04 (z +1.8) | +1.54 ± 0.33 (z +4.7) | -0.39 ± 0.05 (z -8.3) |
| Canto-L (frozen: learned merging) | acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +1.34 ± 0.21 (z +6.3) | +2.85 ± 0.78 (z +3.7) | -8.33 ± 5.81 (z -1.4) | +0.96 ± 0.17 (z +5.7) | +0.09 ± 0.03 (z +2.9) | +2.32 ± 0.22 (z +10.7) | -0.49 ± 0.04 (z -13.0) |
| Canto-L (frozen: learned merging) | acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | +1.34 ± 0.22 (z +6.2) | +2.85 ± 0.78 (z +3.7) | -8.33 ± 5.81 (z -1.4) | +0.97 ± 0.17 (z +5.7) | +0.10 ± 0.03 (z +3.1) | +2.40 ± 0.25 (z +9.8) | -0.50 ± 0.04 (z -12.6) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | +1.70 ± 0.25 (z +6.8) | +3.31 ± 0.88 (z +3.7) | -4.63 ± 5.64 (z -0.8) | +1.21 ± 0.20 (z +6.2) | +0.08 ± 0.04 (z +2.3) | +1.39 ± 0.34 (z +4.1) | -0.46 ± 0.07 (z -7.0) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder) | +1.71 ± 0.32 (z +5.4) | +3.08 ± 1.02 (z +3.0) | -3.70 ± 6.00 (z -0.6) | +1.23 ± 0.24 (z +5.1) | +0.12 ± 0.04 (z +3.5) | +1.15 ± 0.37 (z +3.1) | -0.36 ± 0.05 (z -7.3) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | +1.71 ± 0.32 (z +5.4) | +3.08 ± 1.02 (z +3.0) | -3.70 ± 6.00 (z -0.6) | +1.23 ± 0.24 (z +5.1) | +0.12 ± 0.04 (z +3.5) | +1.15 ± 0.37 (z +3.1) | -0.36 ± 0.05 (z -7.3) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + two-pass MWS-layer + merging (Canto builder family) | +1.76 ± 0.29 (z +6.0) | +3.20 ± 1.01 (z +3.2) | -4.63 ± 5.88 (z -0.8) | +1.27 ± 0.23 (z +5.4) | +0.12 ± 0.04 (z +2.8) | +0.67 ± 0.21 (z +3.1) | -0.46 ± 0.11 (z -4.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | +1.79 ± 0.29 (z +6.1) | +3.31 ± 0.94 (z +3.5) | -4.63 ± 5.88 (z -0.8) | +1.30 ± 0.24 (z +5.5) | +0.12 ± 0.04 (z +2.8) | +0.67 ± 0.22 (z +3.0) | -0.46 ± 0.11 (z -4.1) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | +2.15 ± 0.26 (z +8.4) | +4.62 ± 0.91 (z +5.1) | -4.63 ± 5.97 (z -0.8) | +1.57 ± 0.21 (z +7.6) | +0.11 ± 0.04 (z +2.7) | +1.51 ± 0.35 (z +4.3) | -0.54 ± 0.08 (z -6.8) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +2.16 ± 0.25 (z +8.7) | +4.68 ± 0.91 (z +5.1) | -4.63 ± 5.97 (z -0.8) | +1.58 ± 0.20 (z +7.9) | +0.13 ± 0.05 (z +2.7) | +1.52 ± 0.35 (z +4.4) | -0.54 ± 0.08 (z -6.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | +2.66 ± 0.40 (z +6.6) | +5.76 ± 1.26 (z +4.6) | -3.70 ± 5.82 (z -0.6) | +1.91 ± 0.28 (z +6.8) | +0.16 ± 0.05 (z +3.3) | +0.43 ± 0.25 (z +1.7) | -0.63 ± 0.02 (z -25.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | +2.80 ± 0.56 (z +5.0) | +6.74 ± 1.79 (z +3.8) | -1.85 ± 5.16 (z -0.4) | +1.98 ± 0.39 (z +5.1) | +0.09 ± 0.05 (z +1.8) | +0.92 ± 0.26 (z +3.6) | -0.50 ± 0.03 (z -15.4) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + val-best builder of the full menu (chosen per seed) | +2.80 ± 0.56 (z +5.0) | +6.74 ± 1.79 (z +3.8) | -1.85 ± 5.16 (z -0.4) | +1.98 ± 0.39 (z +5.1) | +0.09 ± 0.05 (z +1.8) | +0.92 ± 0.26 (z +3.6) | -0.50 ± 0.03 (z -15.4) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +3.08 ± 0.50 (z +6.1) | +7.13 ± 1.72 (z +4.1) | -1.85 ± 6.07 (z -0.3) | +2.21 ± 0.34 (z +6.5) | +0.16 ± 0.05 (z +3.3) | +0.38 ± 0.24 (z +1.6) | -0.61 ± 0.02 (z -24.8) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + walkthrough + merging (Canto builder family) | +3.74 ± 0.70 (z +5.4) | +9.02 ± 2.03 (z +4.4) | -1.85 ± 6.67 (z -0.3) | +2.65 ± 0.49 (z +5.4) | +0.09 ± 0.04 (z +2.1) | +1.24 ± 0.31 (z +4.0) | -0.76 ± 0.04 (z -21.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | +4.41 ± 0.59 (z +7.4) | +8.96 ± 1.52 (z +5.9) | -1.85 ± 6.76 (z -0.3) | +3.23 ± 0.46 (z +7.0) | +0.28 ± 0.07 (z +3.8) | +2.07 ± 0.30 (z +6.9) | -1.38 ± 0.05 (z -25.3) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | +4.53 ± 0.77 (z +5.9) | +9.13 ± 1.84 (z +5.0) | -0.93 ± 7.30 (z -0.1) | +3.30 ± 0.57 (z +5.8) | +0.30 ± 0.07 (z +4.3) | +1.93 ± 0.31 (z +6.1) | -1.37 ± 0.06 (z -22.2) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, best cut <= FR* | +4.59 ± 0.84 (z +5.5) | +5.48 ± 0.92 (z +5.9) | -4.63 ± 6.27 (z -0.7) | +4.28 ± 0.69 (z +6.2) | +1.26 ± 0.22 (z +5.6) | +11.85 ± 1.35 (z +8.8) | -1.88 ± 0.03 (z -54.8) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, best cut <= FR* | +5.07 ± 0.46 (z +11.0) | +7.13 ± 0.95 (z +7.5) | -1.85 ± 6.35 (z -0.3) | +4.39 ± 0.38 (z +11.4) | +0.99 ± 0.17 (z +5.7) | +10.20 ± 0.78 (z +13.1) | -1.87 ± 0.05 (z -34.1) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, preregistered cut rule | +6.08 ± 1.47 (z +4.1) | +7.48 ± 1.18 (z +6.3) | -0.93 ± 6.57 (z -0.1) | +5.42 ± 1.40 (z +3.9) | +1.36 ± 0.48 (z +2.8) | +10.29 ± 0.61 (z +16.7) | -1.90 ± 0.04 (z -46.2) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, preregistered cut (claims' reference) | +6.50 ± 0.48 (z +13.5) | +6.85 ± 1.11 (z +6.2) | -3.70 ± 6.52 (z -0.6) | +6.10 ± 0.48 (z +12.7) | +2.15 ± 0.31 (z +6.8) | +14.65 ± 0.34 (z +43.1) | -1.84 ± 0.05 (z -36.5) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + CC, preregistered cut (= best cut) | +8.70 ± 1.02 (z +8.5) | +13.70 ± 2.73 (z +5.0) | +7.41 ± 7.89 (z +0.9) | +7.33 ± 0.77 (z +9.5) | +1.33 ± 0.15 (z +8.7) | +8.91 ± 0.55 (z +16.2) | -1.93 ± 0.04 (z -45.1) |

**val-best against val-best**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | acorn (HNL) + val-best builder of the full menu (chosen per seed) | -0.61 ± 0.27 (z -2.3) | -1.26 ± 0.85 (z -1.5) | -2.78 ± 3.40 (z -0.8) | -0.42 ± 0.20 (z -2.1) | -0.05 ± 0.04 (z -1.2) | +0.33 ± 0.34 (z +1.0) | +0.02 ± 0.05 (z +0.4) |
| acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | acorn (ttbar) + val-best builder of the full menu (chosen per seed) | +1.09 ± 0.58 (z +1.9) | +3.65 ± 1.81 (z +2.0) | +1.85 ± 3.54 (z +0.5) | +0.75 ± 0.40 (z +1.9) | -0.03 ± 0.05 (z -0.6) | -0.23 ± 0.35 (z -0.7) | -0.14 ± 0.05 (z -2.8) |
| acorn (HNL) + val-best builder of the full menu (chosen per seed) | acorn (ttbar) + val-best builder of the full menu (chosen per seed) | +1.70 ± 0.54 (z +3.1) | +4.91 ± 1.71 (z +2.9) | +4.63 ± 3.58 (z +1.3) | +1.17 ± 0.37 (z +3.2) | +0.02 ± 0.05 (z +0.3) | -0.56 ± 0.22 (z -2.6) | -0.16 ± 0.03 (z -5.0) |

**decomposition, builder first: builder on acorn (ttbar) scores**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.16 ± 1.22 (z +3.4) [paired seeds ± 0.58] | +4.57 ± 3.00 (z +1.5) [paired seeds ± 1.63] | +8.33 ± 5.85 (z +1.4) [paired seeds ± 5.23] | +4.03 ± 0.91 (z +4.4) [paired seeds ± 0.42] | +1.03 ± 0.16 (z +6.4) [paired seeds ± 0.15] | +6.99 ± 0.56 (z +12.4) [paired seeds ± 0.38] | -0.56 ± 0.07 (z -8.4) [paired seeds ± 0.03] |
| acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.29 ± 1.12 (z +3.8) [paired seeds ± 0.61] | +4.74 ± 2.81 (z +1.7) [paired seeds ± 1.75] | +9.26 ± 5.48 (z +1.7) [paired seeds ± 4.99] | +4.10 ± 0.85 (z +4.9) [paired seeds ± 0.44] | +1.05 ± 0.16 (z +6.4) [paired seeds ± 0.14] | +6.84 ± 0.55 (z +12.4) [paired seeds ± 0.35] | -0.55 ± 0.06 (z -9.2) [paired seeds ± 0.03] |
| acorn (ttbar) + walkthrough + merging (Canto builder family) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.95 ± 1.18 (z +4.2) [paired seeds ± 0.48] | +4.68 ± 3.09 (z +1.5) [paired seeds ± 1.10] | +9.26 ± 4.63 (z +2.0) [paired seeds ± 4.04] | +4.68 ± 0.87 (z +5.4) [paired seeds ± 0.38] | +1.24 ± 0.15 (z +8.1) [paired seeds ± 0.14] | +7.67 ± 0.56 (z +13.7) [paired seeds ± 0.31] | -1.17 ± 0.05 (z -25.6) [paired seeds ± 0.03] |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | acorn (ttbar) + CC, preregistered cut (= best cut) | +5.62 ± 1.07 (z +5.2) [paired seeds ± 0.62] | +6.56 ± 2.91 (z +2.3) [paired seeds ± 1.51] | +9.26 ± 4.34 (z +2.1) [paired seeds ± 4.03] | +5.13 ± 0.79 (z +6.5) [paired seeds ± 0.51] | +1.18 ± 0.16 (z +7.6) [paired seeds ± 0.15] | +8.53 ± 0.53 (z +16.1) [paired seeds ± 0.44] | -1.32 ± 0.04 (z -35.0) [paired seeds ± 0.04] |
| acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | acorn (ttbar) + CC, preregistered cut (= best cut) | +6.04 ± 1.04 (z +5.8) [paired seeds ± 0.78] | +7.93 ± 2.71 (z +2.9) [paired seeds ± 2.08] | +11.11 ± 4.59 (z +2.4) [paired seeds ± 4.68] | +5.43 ± 0.77 (z +7.0) [paired seeds ± 0.62] | +1.18 ± 0.16 (z +7.6) [paired seeds ± 0.15] | +8.48 ± 0.54 (z +15.8) [paired seeds ± 0.42] | -1.30 ± 0.04 (z -34.5) [paired seeds ± 0.04] |
| acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | acorn (ttbar) + CC, preregistered cut (= best cut) | +5.90 ± 1.11 (z +5.3) [paired seeds ± 0.54] | +6.96 ± 2.99 (z +2.3) [paired seeds ± 1.22] | +9.26 ± 4.44 (z +2.1) [paired seeds ± 4.14] | +5.35 ± 0.82 (z +6.5) [paired seeds ± 0.44] | +1.24 ± 0.16 (z +7.9) [paired seeds ± 0.13] | +7.99 ± 0.54 (z +14.9) [paired seeds ± 0.37] | -1.43 ± 0.04 (z -33.0) [paired seeds ± 0.05] |
| acorn (ttbar) + val-best builder of the full menu (chosen per seed) | acorn (ttbar) + CC, preregistered cut (= best cut) | +5.90 ± 1.11 (z +5.3) [paired seeds ± 0.54] | +6.96 ± 2.99 (z +2.3) [paired seeds ± 1.22] | +9.26 ± 4.44 (z +2.1) [paired seeds ± 4.14] | +5.35 ± 0.82 (z +6.5) [paired seeds ± 0.44] | +1.24 ± 0.16 (z +7.9) [paired seeds ± 0.13] | +7.99 ± 0.54 (z +14.9) [paired seeds ± 0.37] | -1.43 ± 0.04 (z -33.0) [paired seeds ± 0.05] |

**decomposition, builder first: then the Canto-B classifier**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-B (frozen: learned merging) | acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | +2.15 ± 0.58 (z +3.7) | +5.19 ± 1.86 (z +2.8) | +0.93 ± 3.84 (z +0.2) | +1.51 ± 0.40 (z +3.8) | +0.10 ± 0.05 (z +1.8) | +1.17 ± 0.17 (z +6.9) | -0.23 ± 0.05 (z -4.8) |

**decomposition, classifier first: Canto-B classifier with CC**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-B + CC, best cut <= FR* | acorn (ttbar) + CC, preregistered cut (= best cut) | +6.72 ± 1.05 (z +6.4) | +9.25 ± 2.71 (z +3.4) | +12.96 ± 5.34 (z +2.4) | +5.67 ± 0.79 (z +7.1) | +0.90 ± 0.15 (z +5.8) | +1.23 ± 0.50 (z +2.4) | -0.12 ± 0.06 (z -2.1) |

**decomposition, classifier first: then the builder**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-B (frozen: learned merging) | Canto-B + CC, best cut <= FR* | +1.34 ± 0.43 (z +3.1) [paired seeds ± 0.20] | +2.91 ± 1.15 (z +2.5) [paired seeds ± 0.70] | -2.78 ± 4.14 (z -0.7) [paired seeds ± 4.54] | +1.19 ± 0.32 (z +3.7) [paired seeds ± 0.17] | +0.44 ± 0.08 (z +5.2) [paired seeds ± 0.08] | +7.94 ± 0.21 (z +38.3) [paired seeds ± 0.21] | -1.54 ± 0.07 (z -23.6) [paired seeds ± 0.04] |


## Differences on val (new minus baseline; paired event bootstrap of the val tables of feedback point 5, B = 1000, + both seed terms; validation)

**Canto-L minus acorn**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L (frozen: learned merging) | acorn (HNL) + MWS-layer + learned merging (Canto-L's builder) | +0.45 ± 0.29 (z +1.6) | +0.40 ± 1.03 (z +0.4) | -3.51 ± 6.41 (z -0.5) | +0.31 ± 0.24 (z +1.3) | +0.04 ± 0.05 (z +0.7) | +1.40 ± 0.24 (z +5.9) | -0.34 ± 0.04 (z -8.9) |
| Canto-L (frozen: learned merging) | acorn (HNL) + val-best builder of the full menu (chosen per seed) | +0.45 ± 0.29 (z +1.6) | +0.40 ± 1.03 (z +0.4) | -3.51 ± 6.41 (z -0.5) | +0.31 ± 0.24 (z +1.3) | +0.04 ± 0.05 (z +0.7) | +1.40 ± 0.24 (z +5.9) | -0.34 ± 0.04 (z -8.9) |
| Canto-L (frozen: learned merging) | acorn (HNL) + walkthrough + merging (Canto builder family) | +0.58 ± 0.31 (z +1.9) | +0.80 ± 1.20 (z +0.7) | -1.75 ± 7.08 (z -0.2) | +0.38 ± 0.23 (z +1.7) | +0.04 ± 0.05 (z +0.7) | +1.97 ± 0.26 (z +7.6) | -0.51 ± 0.06 (z -8.3) |
| Canto-L (frozen: learned merging) | acorn (HNL) + two-pass MWS-layer + merging (Canto builder family) | +0.60 ± 0.30 (z +2.0) | +0.80 ± 1.20 (z +0.7) | -3.51 ± 6.41 (z -0.5) | +0.44 ± 0.23 (z +1.9) | +0.04 ± 0.05 (z +0.7) | +1.59 ± 0.35 (z +4.5) | -0.43 ± 0.05 (z -9.0) |
| Canto-L (frozen: learned merging) | acorn (HNL) + MWS-layer + merging (Canto builder) | +0.73 ± 0.30 (z +2.4) | +0.80 ± 1.20 (z +0.7) | -1.75 ± 6.94 (z -0.3) | +0.55 ± 0.23 (z +2.4) | +0.04 ± 0.05 (z +0.7) | +1.41 ± 0.32 (z +4.4) | -0.41 ± 0.05 (z -8.0) |
| Canto-L (frozen: learned merging) | acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +0.73 ± 0.32 (z +2.3) | +0.80 ± 1.20 (z +0.7) | +5.26 ± 9.72 (z +0.5) | +0.49 ± 0.24 (z +2.0) | +0.05 ± 0.05 (z +1.1) | +2.11 ± 0.24 (z +8.9) | -0.52 ± 0.06 (z -8.2) |
| Canto-L (frozen: learned merging) | acorn (HNL) + acorn's CC + walkthrough (CCandWalk) | +0.75 ± 0.32 (z +2.3) | +0.80 ± 1.20 (z +0.7) | +5.26 ± 9.72 (z +0.5) | +0.51 ± 0.25 (z +2.1) | +0.05 ± 0.05 (z +1.1) | +2.13 ± 0.26 (z +8.3) | -0.53 ± 0.06 (z -8.4) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder) | +1.20 ± 0.39 (z +3.1) | +2.00 ± 1.49 (z +1.3) | +3.51 ± 7.66 (z +0.5) | +0.86 ± 0.30 (z +2.9) | +0.04 ± 0.04 (z +0.9) | +1.13 ± 0.39 (z +2.9) | -0.37 ± 0.06 (z -6.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | +1.20 ± 0.39 (z +3.1) | +2.00 ± 1.49 (z +1.3) | +3.51 ± 7.66 (z +0.5) | +0.86 ± 0.30 (z +2.9) | +0.04 ± 0.04 (z +0.9) | +1.13 ± 0.39 (z +2.9) | -0.37 ± 0.06 (z -6.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + two-pass MWS-layer + merging (Canto builder family) | +1.23 ± 0.35 (z +3.5) | +1.80 ± 1.39 (z +1.3) | +5.26 ± 7.58 (z +0.7) | +0.99 ± 0.30 (z +3.3) | +0.07 ± 0.05 (z +1.3) | +0.63 ± 0.23 (z +2.8) | -0.49 ± 0.11 (z -4.4) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + walkthrough + merging (Canto builder family) | +1.23 ± 0.42 (z +2.9) | +2.40 ± 1.47 (z +1.6) | +5.26 ± 8.23 (z +0.6) | +0.89 ± 0.33 (z +2.7) | +0.02 ± 0.03 (z +0.7) | +1.37 ± 0.44 (z +3.1) | -0.46 ± 0.08 (z -5.9) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + MWS-layer + merging (Canto builder) | +1.28 ± 0.36 (z +3.6) | +2.00 ± 1.49 (z +1.3) | +5.26 ± 7.58 (z +0.7) | +1.02 ± 0.30 (z +3.4) | +0.07 ± 0.05 (z +1.3) | +0.62 ± 0.23 (z +2.7) | -0.48 ± 0.11 (z -4.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough (CCandWalk) | +1.70 ± 0.41 (z +4.2) | +2.59 ± 1.39 (z +1.9) | +15.79 ± 10.13 (z +1.6) | +1.26 ± 0.32 (z +3.9) | +0.02 ± 0.03 (z +0.7) | +1.45 ± 0.44 (z +3.3) | -0.56 ± 0.08 (z -6.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided (CCandWalk) | +1.73 ± 0.41 (z +4.3) | +2.59 ± 1.39 (z +1.9) | +15.79 ± 10.13 (z +1.6) | +1.28 ± 0.32 (z +4.1) | +0.02 ± 0.03 (z +0.7) | +1.45 ± 0.44 (z +3.3) | -0.56 ± 0.09 (z -6.5) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + val-best builder of the full menu (chosen per seed) | +1.93 ± 0.39 (z +5.0) | +3.79 ± 1.79 (z +2.1) | +10.53 ± 6.30 (z +1.7) | +1.37 ± 0.29 (z +4.7) | +0.00 ± 0.00 (identical) | +0.82 ± 0.30 (z +2.8) | -0.50 ± 0.05 (z -10.9) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | +1.93 ± 0.39 (z +5.0) | +3.79 ± 1.79 (z +2.1) | +10.53 ± 6.30 (z +1.7) | +1.37 ± 0.29 (z +4.7) | +0.00 ± 0.00 (identical) | +0.82 ± 0.30 (z +2.8) | -0.50 ± 0.05 (z -10.9) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | +1.95 ± 0.45 (z +4.3) | +3.99 ± 1.59 (z +2.5) | +12.28 ± 8.33 (z +1.5) | +1.44 ± 0.33 (z +4.3) | +0.09 ± 0.10 (z +0.9) | +0.36 ± 0.28 (z +1.3) | -0.65 ± 0.04 (z -17.3) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +2.33 ± 0.48 (z +4.9) | +4.59 ± 1.72 (z +2.7) | +14.04 ± 7.05 (z +2.0) | +1.72 ± 0.36 (z +4.8) | +0.09 ± 0.10 (z +0.9) | +0.29 ± 0.27 (z +1.1) | -0.63 ± 0.04 (z -16.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + walkthrough + merging (Canto builder family) | +3.11 ± 0.60 (z +5.2) | +7.39 ± 2.32 (z +3.2) | +14.04 ± 7.05 (z +2.0) | +2.39 ± 0.48 (z +5.0) | +0.00 ± 0.00 (identical) | +1.13 ± 0.35 (z +3.2) | -0.76 ± 0.05 (z -16.7) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, best cut <= FR* | +3.48 ± 0.88 (z +3.9) | +3.19 ± 1.61 (z +2.0) | +7.02 ± 9.59 (z +0.7) | +3.31 ± 0.86 (z +3.9) | +1.03 ± 0.30 (z +3.5) | +11.76 ± 1.23 (z +9.5) | -1.89 ± 0.07 (z -27.2) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | +3.68 ± 0.92 (z +4.0) | +6.19 ± 2.37 (z +2.6) | +24.56 ± 11.28 (z +2.2) | +2.79 ± 0.71 (z +4.0) | +0.07 ± 0.05 (z +1.4) | +1.89 ± 0.38 (z +5.0) | -1.44 ± 0.08 (z -17.9) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | +3.81 ± 0.70 (z +5.5) | +5.99 ± 1.86 (z +3.2) | +22.81 ± 10.64 (z +2.1) | +2.89 ± 0.54 (z +5.3) | +0.07 ± 0.05 (z +1.5) | +2.00 ± 0.35 (z +5.7) | -1.45 ± 0.07 (z -20.0) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, best cut <= FR* | +4.06 ± 0.65 (z +6.2) | +5.39 ± 2.04 (z +2.6) | +14.04 ± 9.57 (z +1.5) | +3.47 ± 0.56 (z +6.2) | +0.66 ± 0.22 (z +2.9) | +10.47 ± 0.70 (z +15.0) | -1.84 ± 0.07 (z -27.5) |
| Canto-L (frozen: learned merging) | acorn (ttbar+HNL) + CC, preregistered cut rule | +5.16 ± 1.42 (z +3.6) | +5.99 ± 2.47 (z +2.4) | +14.04 ± 9.57 (z +1.5) | +4.55 ± 1.46 (z +3.1) | +0.91 ± 0.44 (z +2.1) | +10.52 ± 0.56 (z +18.7) | -1.87 ± 0.06 (z -31.3) |
| Canto-L (frozen: learned merging) | acorn (HNL) + CC, preregistered cut (claims' reference) | +5.26 ± 0.59 (z +8.8) | +4.19 ± 1.70 (z +2.5) | +7.02 ± 9.59 (z +0.7) | +5.30 ± 0.59 (z +9.0) | +1.64 ± 0.41 (z +4.0) | +14.21 ± 0.35 (z +40.6) | -1.90 ± 0.06 (z -31.7) |
| Canto-L (frozen: learned merging) | acorn (ttbar) + CC, preregistered cut (= best cut) | +6.79 ± 0.94 (z +7.2) | +8.38 ± 3.13 (z +2.7) | +22.81 ± 9.89 (z +2.3) | +5.88 ± 0.85 (z +6.9) | +1.09 ± 0.32 (z +3.4) | +9.08 ± 0.61 (z +15.0) | -1.94 ± 0.06 (z -32.9) |

**val-best against val-best**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | acorn (HNL) + val-best builder of the full menu (chosen per seed) | -0.75 ± 0.37 (z -2.0) | -1.60 ± 1.31 (z -1.2) | -7.02 ± 7.19 (z -1.0) | -0.55 ± 0.29 (z -1.9) | +0.00 ± 0.06 (z +0.0) | +0.27 ± 0.34 (z +0.8) | +0.03 ± 0.06 (z +0.5) |
| acorn (ttbar+HNL) + val-best builder of the full menu (chosen per seed) | acorn (ttbar) + val-best builder of the full menu (chosen per seed) | +0.73 ± 0.39 (z +1.9) | +1.80 ± 1.93 (z +0.9) | +7.02 ± 9.26 (z +0.8) | +0.51 ± 0.29 (z +1.7) | -0.04 ± 0.04 (z -0.9) | -0.31 ± 0.38 (z -0.8) | -0.13 ± 0.06 (z -2.1) |
| acorn (HNL) + val-best builder of the full menu (chosen per seed) | acorn (ttbar) + val-best builder of the full menu (chosen per seed) | +1.48 ± 0.35 (z +4.3) | +3.39 ± 1.74 (z +2.0) | +14.04 ± 7.50 (z +1.9) | +1.06 ± 0.27 (z +3.9) | -0.04 ± 0.05 (z -0.7) | -0.58 ± 0.22 (z -2.6) | -0.16 ± 0.04 (z -3.6) |

**decomposition, builder first: builder on acorn (ttbar) scores**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar) + acorn's CC + walkthrough (CCandWalk) | acorn (ttbar) + CC, preregistered cut (= best cut) | +3.11 ± 1.18 (z +2.6) [paired seeds ± 0.63] | +2.20 ± 3.35 (z +0.7) [paired seeds ± 1.83] | -1.75 ± 7.07 (z -0.2) [paired seeds ± 3.61] | +3.09 ± 1.01 (z +3.1) [paired seeds ± 0.63] | +1.02 ± 0.32 (z +3.1) [paired seeds ± 0.31] | +7.20 ± 0.64 (z +11.3) [paired seeds ± 0.33] | -0.50 ± 0.08 (z -6.0) [paired seeds ± 0.08] |
| acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk) | acorn (ttbar) + CC, preregistered cut (= best cut) | +2.98 ± 1.01 (z +2.9) [paired seeds ± 0.54] | +2.40 ± 3.00 (z +0.8) [paired seeds ± 2.11] | +0.00 ± 5.60 (z +0.0) [paired seeds ± 2.60] | +3.00 ± 0.90 (z +3.3) [paired seeds ± 0.54] | +1.02 ± 0.32 (z +3.2) [paired seeds ± 0.33] | +7.08 ± 0.62 (z +11.4) [paired seeds ± 0.34] | -0.49 ± 0.08 (z -6.4) [paired seeds ± 0.07] |
| acorn (ttbar) + walkthrough + merging (Canto builder family) | acorn (ttbar) + CC, preregistered cut (= best cut) | +3.68 ± 0.96 (z +3.8) [paired seeds ± 0.61] | +1.00 ± 3.18 (z +0.3) [paired seeds ± 2.46] | +8.77 ± 7.94 (z +1.1) [paired seeds ± 6.68] | +3.49 ± 0.87 (z +4.0) [paired seeds ± 0.55] | +1.09 ± 0.32 (z +3.4) [paired seeds ± 0.32] | +7.96 ± 0.62 (z +12.8) [paired seeds ± 0.33] | -1.18 ± 0.06 (z -21.4) [paired seeds ± 0.05] |
| acorn (ttbar) + MWS-layer + merging (Canto builder) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.46 ± 0.91 (z +4.9) [paired seeds ± 0.66] | +3.79 ± 2.80 (z +1.4) [paired seeds ± 1.94] | +8.77 ± 7.94 (z +1.1) [paired seeds ± 6.68] | +4.16 ± 0.81 (z +5.1) [paired seeds ± 0.65] | +1.00 ± 0.34 (z +2.9) [paired seeds ± 0.37] | +8.80 ± 0.59 (z +14.9) [paired seeds ± 0.47] | -1.31 ± 0.05 (z -24.9) [paired seeds ± 0.05] |
| acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.83 ± 0.90 (z +5.4) [paired seeds ± 0.77] | +4.39 ± 2.78 (z +1.6) [paired seeds ± 2.03] | +10.53 ± 8.84 (z +1.2) [paired seeds ± 7.31] | +4.44 ± 0.81 (z +5.5) [paired seeds ± 0.74] | +1.00 ± 0.34 (z +2.9) [paired seeds ± 0.37] | +8.72 ± 0.59 (z +14.7) [paired seeds ± 0.46] | -1.28 ± 0.05 (z -23.9) [paired seeds ± 0.05] |
| acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.86 ± 0.87 (z +5.6) [paired seeds ± 0.76] | +4.59 ± 2.90 (z +1.6) [paired seeds ± 1.90] | +12.28 ± 9.40 (z +1.3) [paired seeds ± 7.59] | +4.51 ± 0.80 (z +5.7) [paired seeds ± 0.72] | +1.09 ± 0.32 (z +3.4) [paired seeds ± 0.32] | +8.26 ± 0.60 (z +13.8) [paired seeds ± 0.39] | -1.44 ± 0.06 (z -24.4) [paired seeds ± 0.05] |
| acorn (ttbar) + val-best builder of the full menu (chosen per seed) | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.86 ± 0.87 (z +5.6) [paired seeds ± 0.76] | +4.59 ± 2.90 (z +1.6) [paired seeds ± 1.90] | +12.28 ± 9.40 (z +1.3) [paired seeds ± 7.59] | +4.51 ± 0.80 (z +5.7) [paired seeds ± 0.72] | +1.09 ± 0.32 (z +3.4) [paired seeds ± 0.32] | +8.26 ± 0.60 (z +13.8) [paired seeds ± 0.39] | -1.44 ± 0.06 (z -24.4) [paired seeds ± 0.05] |

**decomposition, builder first: then the Canto-B classifier**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-B + walkthrough + merging (Canto builder family) | acorn (ttbar) + walkthrough + merging (Canto builder family) | +2.35 ± 0.56 (z +4.2) | +4.39 ± 2.05 (z +2.1) | +0.00 ± 5.64 (z +0.0) | +1.79 ± 0.46 (z +3.9) | +0.00 ± 0.00 (identical) | +1.17 ± 0.28 (z +4.3) | -0.34 ± 0.06 (z -5.7) |
| Canto-B + MWS-layer + merging (Canto builder) | acorn (ttbar) + MWS-layer + merging (Canto builder) | +1.48 ± 0.45 (z +3.3) | +2.79 ± 1.53 (z +1.8) | +1.75 ± 5.45 (z +0.3) | +1.04 ± 0.35 (z +3.0) | +0.02 ± 0.10 (z +0.2) | +0.65 ± 0.32 (z +2.1) | -0.22 ± 0.05 (z -4.7) |
| Canto-B + two-pass MWS-layer + merging (Canto builder family) | acorn (ttbar) + two-pass MWS-layer + merging (Canto builder family) | +1.35 ± 0.38 (z +3.6) | +2.59 ± 1.26 (z +2.1) | +3.51 ± 6.08 (z +0.6) | +0.97 ± 0.27 (z +3.5) | +0.05 ± 0.11 (z +0.5) | +0.52 ± 0.18 (z +2.9) | -0.11 ± 0.05 (z -2.4) |
| Canto-B (frozen: learned merging) | acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder) | +1.28 ± 0.32 (z +4.0) | +2.40 ± 1.53 (z +1.6) | +1.75 ± 8.15 (z +0.2) | +0.91 ± 0.24 (z +3.8) | -0.07 ± 0.05 (z -1.6) | +1.08 ± 0.21 (z +5.1) | -0.23 ± 0.05 (z -5.1) |

**decomposition, classifier first: Canto-B classifier with CC**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-B + CC, best cut <= FR* | acorn (ttbar) + CC, preregistered cut (= best cut) | +4.73 ± 0.93 (z +5.1) | +6.19 ± 2.94 (z +2.1) | +0.00 ± 5.36 (z +0.0) | +4.18 ± 0.84 (z +5.0) | +0.82 ± 0.28 (z +3.0) | +1.21 ± 0.55 (z +2.2) | -0.20 ± 0.05 (z -4.1) |

**decomposition, classifier first: then the builder**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-B + walkthrough + merging (Canto builder family) | Canto-B + CC, best cut <= FR* | +1.30 ± 0.38 (z +3.4) [paired seeds ± 0.44] | -0.80 ± 1.36 (z -0.6) [paired seeds ± 0.88] | +8.77 ± 7.34 (z +1.2) [paired seeds ± 6.68] | +1.10 ± 0.30 (z +3.7) [paired seeds ± 0.33] | +0.27 ± 0.16 (z +1.7) [paired seeds ± 0.16] | +7.92 ± 0.20 (z +38.9) [paired seeds ± 0.21] | -1.31 ± 0.08 (z -17.0) [paired seeds ± 0.06] |
| Canto-B + MWS-layer + merging (Canto builder) | Canto-B + CC, best cut <= FR* | +1.20 ± 0.40 (z +3.0) [paired seeds ± 0.46] | +0.40 ± 0.95 (z +0.4) [paired seeds ± 0.44] | +10.53 ± 8.01 (z +1.3) [paired seeds ± 7.61] | +1.02 ± 0.31 (z +3.3) [paired seeds ± 0.34] | +0.20 ± 0.17 (z +1.2) [paired seeds ± 0.17] | +8.25 ± 0.34 (z +24.1) [paired seeds ± 0.32] | -1.33 ± 0.07 (z -17.9) [paired seeds ± 0.06] |
| Canto-B + two-pass MWS-layer + merging (Canto builder family) | Canto-B + CC, best cut <= FR* | +1.45 ± 0.36 (z +4.1) [paired seeds ± 0.39] | +0.80 ± 0.84 (z +1.0) [paired seeds ± 0.43] | +14.04 ± 8.34 (z +1.7) [paired seeds ± 8.34] | +1.22 ± 0.27 (z +4.5) [paired seeds ± 0.29] | +0.23 ± 0.16 (z +1.4) [paired seeds ± 0.16] | +8.04 ± 0.20 (z +39.6) [paired seeds ± 0.20] | -1.19 ± 0.07 (z -15.9) [paired seeds ± 0.06] |
| Canto-B (frozen: learned merging) | Canto-B + CC, best cut <= FR* | +1.40 ± 0.34 (z +4.1) [paired seeds ± 0.37] | +0.80 ± 0.84 (z +0.9) [paired seeds ± 0.44] | +14.04 ± 9.21 (z +1.5) [paired seeds ± 8.70] | +1.24 ± 0.27 (z +4.6) [paired seeds ± 0.28] | +0.20 ± 0.14 (z +1.4) [paired seeds ± 0.14] | +8.14 ± 0.21 (z +39.4) [paired seeds ± 0.21] | -1.47 ± 0.07 (z -21.7) [paired seeds ± 0.06] |

**same builder: Canto-L minus acorn (HNL)**

| new | baseline | primary | |d0| 100-300 mm | |d0| >= 300 mm | HNL daughters | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|
| Canto-L + walkthrough + merging (Canto builder family), same builder (not the frozen builder) | acorn (HNL) + walkthrough + merging (Canto builder family) | +0.70 ± 0.28 (z +2.5) | +0.20 ± 1.30 (z +0.2) | -5.26 ± 6.77 (z -0.8) | +0.47 ± 0.21 (z +2.3) | +0.04 ± 0.05 (z +0.7) | +1.69 ± 0.25 (z +6.8) | -0.41 ± 0.06 (z -6.5) |
| Canto-L + MWS-layer + merging (Canto builder), same builder (not the frozen builder) | acorn (HNL) + MWS-layer + merging (Canto builder) | +0.78 ± 0.29 (z +2.6) | +0.80 ± 1.20 (z +0.7) | -3.51 ± 7.24 (z -0.5) | +0.58 ± 0.23 (z +2.6) | +0.04 ± 0.05 (z +0.7) | +1.16 ± 0.32 (z +3.7) | -0.31 ± 0.07 (z -4.6) |
| Canto-L + two-pass MWS-layer + merging (Canto builder family), same builder (not the frozen builder) | acorn (HNL) + two-pass MWS-layer + merging (Canto builder family) | +0.70 ± 0.28 (z +2.5) | +1.00 ± 1.24 (z +0.8) | -3.51 ± 5.23 (z -0.7) | +0.51 ± 0.22 (z +2.4) | +0.04 ± 0.05 (z +0.7) | +1.61 ± 0.29 (z +5.6) | -0.22 ± 0.11 (z -1.9) |

Primary efficiency per test set (descriptive; seed mean ± total sigma; difference with the same estimator):

| test set | primary particles per seed | new | value | baseline | value | difference |
|---|---|---|---|---|---|---|
| hnl10_100_test | 603 | Canto-L (frozen: learned merging) | 0.9923 ± 0.0033 | acorn (HNL) + val-best builder of the full menu (chosen per seed) | 0.9845 ± 0.0047 | +0.77 ± 0.46 (z +1.7) |
| hnl15_200_test | 549 | Canto-L (frozen: learned merging) | 0.9848 ± 0.0047 | acorn (HNL) + val-best builder of the full menu (chosen per seed) | 0.9702 ± 0.0070 | +1.46 ± 0.65 (z +2.2) |
| hnl15_100_external | 3861 | Canto-L (frozen: learned merging) | 0.9914 ± 0.0017 | acorn (HNL) + val-best builder of the full menu (chosen per seed) | 0.9804 ± 0.0022 | +1.10 ± 0.24 (z +4.5) |

Points = percentage points (fake rate: a negative difference means the new row has the lower fake rate). [paired seeds]: the same trained networks on both sides, seed term of the per-seed differences. A row's own sigma (bootstrap + seed term) is in the CSV files.

## Notes

- Menu per acorn run (acorn (ttbar), acorn (HNL), acorn (ttbar+HNL) = Stage-0 arms i, ii, iii; seeds 0, 1, 2; the Stage-0 GNN scores): CC at the best cut <= FR* (and the preregistered cut rule, as rounds 1-2), acorn's CCandWalk plain and score-guided, walkthrough + merging, MWS-layer + merging, two-pass MWS-layer + merging, MWS-layer + learned merging. Canto's grids also hold MWS-layer and two-pass without merging; they were never a Canto choice and are not in the menu.
- Learned merging on acorn scores, as Canto's (slurm/feedback/fb3_lmerge.sbatch; the acorn (ttbar+HNL) seed-0 model with fb3_lmerge_split.sbatch): 2000 of the run's own training graphs (runs/baseline_v1/arm<A>_s<S>/ml_graphs/trainset.tar; numpy default_rng(0) over the sorted names, the rule of score_split_sample.sbatch: arm (ii) 965 hnl10_100 + 1035 hnl15_200, arm (iii) 962 ttbar + 520 + 518 HNL, arm (i) 2000 ttbar), scored with the run's best GNN checkpoint (sha256 equal to the checkpoint of its stored val and test scores), fragment pairs after the same-layer MWS at s0 0.2 (hnl.trackbuild.learned_merge data), the GBDT of learned_merge train (fixed settings, seed 0): 0.95-1.64 M pairs, 0.15-0.71 % positive, holdout AUC 0.977-0.994; val grid tau 0.005-0.5 (RUN/trackbuild_val_fb3_lmerge). Models and provenance: runs/feedback_v3/strong_baselines_a2/arm<A>_s<S>/lmerge.
- Edge extension (docs/feedback/round3_A2_decisions.md): a choice with a parameter at the lowest or highest value of that parameter in the family's grid is extended past that edge on val (the edge slice with the next two values of the parameter's progression, within the family's range), on all nine runs, until interior or the range ends. Step 1: walkthrough + merging walk add 0.3 / 0.1 and 0.9 / 1.0, MWS-layer + merging s0 0.05 / 0.02, two-pass s1 0.1 / 0.05 and s2 0.2 / 0.5, CCandWalk walk add 0.2 (walk add below 0.2 is not computable: acorn's walk enumerates every road whose branches pass walk add). Step 2: two-pass s2 0.01 / 0.005. Step 3: none. Every final choice of the 54 run-family pairs is interior. Grid parts RUN/trackbuild_val_fb3_ext<step>_<part>, united in RUN/trackbuild_val_fb3_all.
- Selection (val): per run and family the builder rule at FR* = 0.029368 with the one-standard-error tie rule; the val-best row of an arm: per seed the same rule over the families' choices and CC at its best cut. Learned merging is the val-best family for all nine runs (and also the arm's row with the highest three-seed mean, round 2's rule).
- Test (post-freeze descriptive): the val-fixed spec results/strong_baselines_v3_spec_test.json; configurations evaluated on test in rounds 1-2 (same run and configuration) reuse those outputs, the 20 others were evaluated once (slurm/feedback/fb3_test.sbatch, runs/feedback_v3/strong_baselines_a2/arm<A>_s<S>/<key>_test; provenance with the spec's and the merge model's sha256 checked before the summary). Canto-B with CC at its val-best cut: post-freeze CC on its stored test scores (bestcut; its frozen e* rows reproduced).
- Differences: test with the claims' estimator (paired event bootstrap of the preregistered test tables, B = 500, + both seed terms); val with the tables of feedback point 5 (B = 1000). Rows of the same trained networks (decomposition steps on one classifier) also carry the seed term paired by seed.
- Checks: rounds 1-2 rows, entries and differences of unchanged choices reproduced to 1e-12; the frozen selection's per-seed Canto-B and Canto-L family rows; configurations computed twice (extension against older grids of the same run) identical; hnl.trackbuild.acorn_walk against acorn's own CCandWalk on the new CCandWalk configurations (600 of 600 event-configurations on each of acorn (HNL) seed 2 and acorn (ttbar) seed 0: 100 val events, 6 configurations); test: the claims' primary difference reproduced to 1e-12.
- Status: validation numbers and post-freeze descriptive test summaries; nothing is chosen on test; the preregistered claims (results/claims_test_v1.json) stay as they are. Every val number of a row chosen on val is optimistic for that row.
