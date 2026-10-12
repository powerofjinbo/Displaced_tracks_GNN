# Our models under the published definitions of Wang et al. (arXiv:2203.08800)

**Post-freeze descriptive summaries of the one-time test outputs, NOT claims** (the preregistered claims are results/claims_test_v1.json). Within this simulation (ACTS generic detector, no pileup). Written by `python -m hnl.eval.published_defs summary` (2026-10-05T20:58:04Z, git 6c38a0df79e246cabe8603143e43830dc09a2054 (src dirty: src/hnl/eval/published_defs.py), Slurm job 6690268). Frozen configurations only: nothing was chosen, tuned or re-selected on test. Post-freeze CC evaluations of the stored test scores (acorn (HNL) at its val best cut): armii_s0 cut 0.38 (job 6690254, /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/published_defs_v1/armii_s0/eval_test_bestcut; s* rows identical to the frozen eval_test); armii_s2 cut 0.44 (job 6690254, /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/published_defs_v1/armii_s2/eval_test_bestcut; s* rows identical to the frozen eval_test).

## Published numbers (Wang et al., p.4; docs/paper_facts.md C9 and the Fig. 3 vector data)

| quantity | published | denominator |
|---|---|---|
| all tracks in the event | 0.9561 ± 0.0006 | not given; the binomial error implies about 116,600 (99k-139k) |
| prompt tracks from the HNL process (the W muon) | 0.990 ± 0.002 | 4,744 (Fig. 3: 4696/4744 = 0.98988 +- 0.00145) |
| displaced tracks from the HNL process (the N daughters) | 0.916 ± 0.005 | about 2,882 (Fig. 3: 2639/2882 = 0.9157 +- 0.0052, two bins at 1.0 left out) |
| track fake rate | - | not defined or reported |

Their pipeline: metric-learning MLP (3D -> 8D) + filter MLP graph (99.9 % of the true edges at 5.7 % purity), attention GNN (edges at score 0.5: prompt 93.8 % efficiency / 87.0 % purity, displaced 92.5 % / 84.7 %), DBSCAN (minPoints 2, epsilon 0.4, the epsilon of highest efficiency and track purity on a 0.1-1.0 scan). Trained on 10,000 HNL events (15 GeV, c*tau = 100 mm) of their own generation (not released); evaluated on 5,000 events, identified as the complete hnl15_100 file of the Zenodo record (docs/paper_facts.md). Matching: double majority, reconstructable = pT > 1 GeV, |eta| < 4, >= 5 space points. No fake or duplicate rate is defined or reported; their uncertainty method is not stated.

## Definitions used here (the code path of the earlier reproduction, hnl.eval.test_stats.paper_summary)

| quantity | population (per seed) | test_stats mask |
|---|---|---|
| all_track | every target (pT > 1 GeV, |eta| < 4, >= 5 hits) of the HNL events | `all_targets` |
| prompt | W muon with >= 5 hits (no pT / |eta| cut; = the paper's 4,744 on the external set) | `w_muons_paper` |
| displaced | HNL-daughter targets, no displacement cut | `hnl_targets` |
| prompt_targets | check: W-muon targets (with the pT / |eta| cuts) | `w_muons` |
| displaced_mu_plus | check: HNL-daughter targets with q > 0 | `hnl_mu_plus` |
| displaced_mu_minus | check: HNL-daughter targets with q < 0 | `hnl_mu_minus` |
| fake rate | tracks with >= 5 hits that DM-match no particle / tracks with >= 5 hits, over the set's events | `n_fake / n_tracks_ge5` |

Cells: mean ± seed standard deviation over the three training seeds (ddof 1), then the count pooled over the seeds (k/n; n = 3 x the population). Working points: Canto at the FR* configuration of results/final_selection_v1.json; acorn with connected components (CC) at the cut named in the row (* = FR* unreachable on val, minimum-rate cut). FR* = 0.029368 was fixed on pooled HNL val; on test the fake rate is measured, not constrained.

## Test, hnl15_100_external (the paper's evaluation sample, 4,999 events) (post-freeze descriptive, not a claim)

| model | working point (seeds 0 / 1 / 2) | all-track | prompt (W muon, >= 5 hits) | displaced (N-daughter targets) | fake rate |
|---|---|---|---|---|---|
| Canto-L (learned merging) | learned merging tau 0.05 / 0.03 / 0.1 | 0.9914 ± 0.0005 (346,344/349,347) | 0.9965 ± 0.0001 (14,182/14,232) | 0.9922 ± 0.0018 (16,738/16,869) | 0.0103 ± 0.0003 (17,609/1,717,097) |
| Canto-H (walkthrough + merging) | walkthrough + merging (0.2, 0.7, 1.0) / (0.2, 0.5, 1.0) / (0.2, 0.5, 1.0) | 0.9915 ± 0.0001 (346,372/349,347) | 0.9963 ± 0.0001 (14,180/14,232) | 0.9935 ± 0.0011 (16,760/16,869) | 0.0144 ± 0.0006 (24,738/1,717,979) |
| Canto-B (learned merging) | learned merging tau 0.05 / 0.2 / 0.1 | 0.9913 ± 0.0003 (346,319/349,347) | 0.9966 ± 0.0001 (14,183/14,232) | 0.9877 ± 0.0024 (16,661/16,869) | 0.0130 ± 0.0007 (22,377/1,719,453) |
| acorn (HNL), CC, preregistered cut | CC 0.26 / 0.36* / 0.27 | 0.9426 ± 0.0057 (329,303/349,347) | 0.9746 ± 0.0041 (13,870/14,232) | 0.9331 ± 0.0084 (15,741/16,869) | 0.0289 ± 0.0006 (48,176/1,669,757) |
| acorn (HNL), CC, best cut <= FR* (val) (s0, s2 evaluated post-freeze) | CC 0.38 / 0.36* / 0.44 | 0.9547 ± 0.0062 (333,537/349,347) | 0.9825 ± 0.0031 (13,983/14,232) | 0.9513 ± 0.0096 (16,048/16,869) | 0.0290 ± 0.0003 (48,885/1,684,316) |
| acorn (HNL), CC at s* (earlier reproduction) | CC 0.40 / 0.40 / 0.40 | 0.9554 ± 0.0057 (333,766/349,347) | 0.9830 ± 0.0017 (13,990/14,232) | 0.9541 ± 0.0061 (16,095/16,869) | 0.0292 ± 0.0023 (49,281/1,685,212) |
| acorn (ttbar), CC, working-point cuts | CC 0.34* / 0.33 / 0.31* | 0.9566 ± 0.0020 (334,168/349,347) | 0.9817 ± 0.0013 (13,972/14,232) | 0.9212 ± 0.0116 (15,540/16,869) | 0.0295 ± 0.0006 (49,674/1,682,043) |
| Wang et al. (published) | attention GNN + DBSCAN | 0.9561 ± 0.0006 | 0.990 ± 0.002 | 0.916 ± 0.005 | not reported |

Total uncertainty of the three-seed mean on this set (preregistered: event bootstrap B = 500 + seed term):

| model | all-track | prompt | displaced | fake rate |
|---|---|---|---|---|
| Canto-L (learned merging) | 0.9914 ± 0.0005 | 0.9965 ± 0.0009 | 0.9922 ± 0.0015 | 0.0103 ± 0.0003 |
| Canto-H (walkthrough + merging) | 0.9915 ± 0.0004 | 0.9963 ± 0.0009 | 0.9935 ± 0.0011 | 0.0144 ± 0.0004 |
| Canto-B (learned merging) | 0.9913 ± 0.0004 | 0.9966 ± 0.0009 | 0.9877 ± 0.0019 | 0.0130 ± 0.0005 |
| acorn (HNL), CC, preregistered cut | 0.9426 ± 0.0034 | 0.9746 ± 0.0030 | 0.9331 ± 0.0057 | 0.0289 ± 0.0005 |
| acorn (HNL), CC, best cut <= FR* (val) | 0.9547 ± 0.0037 | 0.9825 ± 0.0024 | 0.9513 ± 0.0061 | 0.0290 ± 0.0004 |
| acorn (HNL), CC at s* (earlier reproduction) | 0.9554 ± 0.0034 | 0.9830 ± 0.0019 | 0.9541 ± 0.0043 | 0.0292 ± 0.0014 |
| acorn (ttbar), CC, working-point cuts | 0.9566 ± 0.0014 | 0.9817 ± 0.0018 | 0.9212 ± 0.0074 | 0.0295 ± 0.0005 |

## Test, pooled HNL (hnl10_100 + hnl15_200 test + hnl15_100_external, 6,999 events) (post-freeze descriptive, not a claim)

| model | working point (seeds 0 / 1 / 2) | all-track | prompt (W muon, >= 5 hits) | displaced (N-daughter targets) | fake rate |
|---|---|---|---|---|---|
| Canto-L (learned merging) | learned merging tau 0.05 / 0.03 / 0.1 | 0.9912 ± 0.0006 (475,800/480,048) | 0.9970 ± 0.0001 (19,774/19,833) | 0.9916 ± 0.0016 (21,705/21,888) | 0.0102 ± 0.0003 (24,327/2,384,875) |
| Canto-H (walkthrough + merging) | walkthrough + merging (0.2, 0.7, 1.0) / (0.2, 0.5, 1.0) / (0.2, 0.5, 1.0) | 0.9912 ± 0.0001 (475,835/480,048) | 0.9969 ± 0.0001 (19,772/19,833) | 0.9930 ± 0.0010 (21,734/21,888) | 0.0142 ± 0.0006 (33,989/2,386,096) |
| Canto-B (learned merging) | learned merging tau 0.05 / 0.2 / 0.1 | 0.9911 ± 0.0003 (475,790/480,048) | 0.9971 ± 0.0001 (19,775/19,833) | 0.9869 ± 0.0026 (21,602/21,888) | 0.0129 ± 0.0007 (30,796/2,388,062) |
| acorn (HNL), CC, preregistered cut | CC 0.26 / 0.36* / 0.27 | 0.9423 ± 0.0054 (452,329/480,048) | 0.9752 ± 0.0049 (19,341/19,833) | 0.9306 ± 0.0067 (20,370/21,888) | 0.0286 ± 0.0007 (66,441/2,319,245) |
| acorn (HNL), CC, best cut <= FR* (val) (s0, s2 evaluated post-freeze) | CC 0.38 / 0.36* / 0.44 | 0.9542 ± 0.0065 (458,077/480,048) | 0.9843 ± 0.0032 (19,521/19,833) | 0.9488 ± 0.0112 (20,768/21,888) | 0.0290 ± 0.0002 (67,857/2,339,605) |
| acorn (HNL), CC at s* (earlier reproduction) | CC 0.40 / 0.40 / 0.40 | 0.9548 ± 0.0058 (458,364/480,048) | 0.9848 ± 0.0020 (19,531/19,833) | 0.9513 ± 0.0065 (20,821/21,888) | 0.0292 ± 0.0023 (68,432/2,340,788) |
| acorn (ttbar), CC, working-point cuts | CC 0.34* / 0.33 / 0.31* | 0.9561 ± 0.0020 (458,988/480,048) | 0.9833 ± 0.0018 (19,501/19,833) | 0.9183 ± 0.0123 (20,100/21,888) | 0.0295 ± 0.0005 (68,827/2,336,314) |
| Wang et al. (published) (on their evaluation sample, not this set) | attention GNN + DBSCAN | 0.9561 ± 0.0006 | 0.990 ± 0.002 | 0.916 ± 0.005 | not reported |

Total uncertainty of the three-seed mean on this set (preregistered: event bootstrap B = 500 + seed term):

| model | all-track | prompt | displaced | fake rate |
|---|---|---|---|---|
| Canto-L (learned merging) | 0.9912 ± 0.0005 | 0.9970 ± 0.0007 | 0.9916 ± 0.0014 | 0.0102 ± 0.0003 |
| Canto-H (walkthrough + merging) | 0.9912 ± 0.0004 | 0.9969 ± 0.0007 | 0.9930 ± 0.0011 | 0.0142 ± 0.0004 |
| Canto-B (learned merging) | 0.9911 ± 0.0004 | 0.9971 ± 0.0007 | 0.9869 ± 0.0019 | 0.0129 ± 0.0005 |
| acorn (HNL), CC, preregistered cut | 0.9423 ± 0.0032 | 0.9752 ± 0.0033 | 0.9306 ± 0.0047 | 0.0286 ± 0.0005 |
| acorn (HNL), CC, best cut <= FR* (val) | 0.9542 ± 0.0038 | 0.9843 ± 0.0023 | 0.9488 ± 0.0068 | 0.0290 ± 0.0004 |
| acorn (HNL), CC at s* (earlier reproduction) | 0.9548 ± 0.0034 | 0.9848 ± 0.0018 | 0.9513 ± 0.0044 | 0.0292 ± 0.0014 |
| acorn (ttbar), CC, working-point cuts | 0.9561 ± 0.0014 | 0.9833 ± 0.0017 | 0.9183 ± 0.0077 | 0.0295 ± 0.0005 |

## Val, pooled HNL (hnl10_100 + hnl15_200 val, 1,999 events) (validation)

| model | working point (seeds 0 / 1 / 2) | all-track | prompt (W muon, >= 5 hits) | displaced (N-daughter targets) | fake rate |
|---|---|---|---|---|---|
| Canto-L (learned merging) | learned merging tau 0.05 / 0.03 / 0.1 | 0.9902 ± 0.0005 (140,068/141,456) | 0.9974 ± 0.0000 (5,745/5,760) | 0.9918 ± 0.0014 (5,430/5,475) | 0.0103 ± 0.0004 (7,197/701,048) |
| Canto-H (walkthrough + merging) | walkthrough + merging (0.2, 0.7, 1.0) / (0.2, 0.5, 1.0) / (0.2, 0.5, 1.0) | 0.9905 ± 0.0002 (140,110/141,456) | 0.9974 ± 0.0000 (5,745/5,760) | 0.9921 ± 0.0016 (5,432/5,475) | 0.0143 ± 0.0005 (10,051/701,415) |
| Canto-B (learned merging) | learned merging tau 0.05 / 0.2 / 0.1 | 0.9903 ± 0.0003 (140,087/141,456) | 0.9967 ± 0.0003 (5,741/5,760) | 0.9872 ± 0.0013 (5,405/5,475) | 0.0130 ± 0.0005 (9,108/702,006) |
| acorn (HNL), CC, preregistered cut | CC 0.26 / 0.36* / 0.27 | 0.9400 ± 0.0050 (132,972/141,456) | 0.9804 ± 0.0047 (5,647/5,760) | 0.9388 ± 0.0046 (5,140/5,475) | 0.0293 ± 0.0005 (19,953/680,914) |
| acorn (HNL), CC, best cut <= FR* (val) | CC 0.38 / 0.36* / 0.44 | 0.9529 ± 0.0076 (134,787/141,456) | 0.9863 ± 0.0026 (5,681/5,760) | 0.9587 ± 0.0129 (5,249/5,475) | 0.0292 ± 0.0005 (20,052/686,933) |
| acorn (HNL), CC at s* (earlier reproduction) | CC 0.40 / 0.40 / 0.40 | 0.9537 ± 0.0069 (134,908/141,456) | 0.9875 ± 0.0031 (5,688/5,760) | 0.9605 ± 0.0086 (5,259/5,475) | 0.0294 ± 0.0026 (20,180/687,253) |
| acorn (ttbar), CC, working-point cuts | CC 0.34* / 0.33 / 0.31* | 0.9554 ± 0.0025 (135,150/141,456) | 0.9859 ± 0.0032 (5,679/5,760) | 0.9330 ± 0.0119 (5,108/5,475) | 0.0297 ± 0.0003 (20,341/685,969) |
| Wang et al. (published) (on their evaluation sample, not this set) | attention GNN + DBSCAN | 0.9561 ± 0.0006 | 0.990 ± 0.002 | 0.916 ± 0.005 | not reported |

## Population checks (same outputs)

| set | model | prompt: W-muon targets (pT / abs(eta) cuts) | displaced, q > 0 | displaced, q < 0 |
|---|---|---|---|---|
| test_external | Canto-L (learned merging) | 0.9975 ± 0.0001 (13,852/13,887) | 0.9927 ± 0.0011 (8,348/8,409) | 0.9917 ± 0.0025 (8,390/8,460) |
| test_external | Canto-H (walkthrough + merging) | 0.9973 ± 0.0001 (13,850/13,887) | 0.9939 ± 0.0016 (8,358/8,409) | 0.9931 ± 0.0007 (8,402/8,460) |
| test_external | Canto-B (learned merging) | 0.9976 ± 0.0001 (13,853/13,887) | 0.9872 ± 0.0026 (8,301/8,409) | 0.9882 ± 0.0023 (8,360/8,460) |
| test_external | acorn (HNL), CC, preregistered cut | 0.9760 ± 0.0040 (13,554/13,887) | 0.9345 ± 0.0097 (7,858/8,409) | 0.9318 ± 0.0071 (7,883/8,460) |
| test_external | acorn (HNL), CC, best cut <= FR* (val) | 0.9839 ± 0.0033 (13,663/13,887) | 0.9515 ± 0.0087 (8,001/8,409) | 0.9512 ± 0.0111 (8,047/8,460) |
| test_external | acorn (HNL), CC at s* (earlier reproduction) | 0.9844 ± 0.0018 (13,670/13,887) | 0.9549 ± 0.0047 (8,030/8,409) | 0.9533 ± 0.0085 (8,065/8,460) |
| test_external | acorn (ttbar), CC, working-point cuts | 0.9833 ± 0.0013 (13,655/13,887) | 0.9218 ± 0.0112 (7,751/8,409) | 0.9207 ± 0.0123 (7,789/8,460) |
| test_pooled | Canto-L (learned merging) | 0.9977 ± 0.0001 (19,282/19,326) | 0.9927 ± 0.0013 (10,876/10,956) | 0.9906 ± 0.0019 (10,829/10,932) |
| test_pooled | Canto-H (walkthrough + merging) | 0.9976 ± 0.0001 (19,280/19,326) | 0.9933 ± 0.0011 (10,883/10,956) | 0.9926 ± 0.0010 (10,851/10,932) |
| test_pooled | Canto-B (learned merging) | 0.9978 ± 0.0001 (19,283/19,326) | 0.9865 ± 0.0027 (10,808/10,956) | 0.9874 ± 0.0027 (10,794/10,932) |
| test_pooled | acorn (HNL), CC, preregistered cut | 0.9762 ± 0.0047 (18,867/19,326) | 0.9299 ± 0.0077 (10,188/10,956) | 0.9314 ± 0.0057 (10,182/10,932) |
| test_pooled | acorn (HNL), CC, best cut <= FR* (val) | 0.9851 ± 0.0033 (19,039/19,326) | 0.9481 ± 0.0114 (10,387/10,956) | 0.9496 ± 0.0114 (10,381/10,932) |
| test_pooled | acorn (HNL), CC at s* (earlier reproduction) | 0.9857 ± 0.0021 (19,049/19,326) | 0.9511 ± 0.0056 (10,420/10,956) | 0.9514 ± 0.0085 (10,401/10,932) |
| test_pooled | acorn (ttbar), CC, working-point cuts | 0.9844 ± 0.0016 (19,024/19,326) | 0.9163 ± 0.0130 (10,039/10,956) | 0.9203 ± 0.0115 (10,061/10,932) |
| val_pooled | Canto-L (learned merging) | 0.9979 ± 0.0000 (5,595/5,607) | 0.9941 ± 0.0017 (2,702/2,718) | 0.9895 ± 0.0013 (2,728/2,757) |
| val_pooled | Canto-H (walkthrough + merging) | 0.9979 ± 0.0000 (5,595/5,607) | 0.9926 ± 0.0023 (2,698/2,718) | 0.9917 ± 0.0023 (2,734/2,757) |
| val_pooled | Canto-B (learned merging) | 0.9971 ± 0.0003 (5,591/5,607) | 0.9890 ± 0.0011 (2,688/2,718) | 0.9855 ± 0.0017 (2,717/2,757) |
| val_pooled | acorn (HNL), CC, preregistered cut | 0.9815 ± 0.0048 (5,503/5,607) | 0.9360 ± 0.0022 (2,544/2,718) | 0.9416 ± 0.0072 (2,596/2,757) |
| val_pooled | acorn (HNL), CC, best cut <= FR* (val) | 0.9875 ± 0.0027 (5,537/5,607) | 0.9547 ± 0.0152 (2,595/2,718) | 0.9626 ± 0.0110 (2,654/2,757) |
| val_pooled | acorn (HNL), CC at s* (earlier reproduction) | 0.9886 ± 0.0029 (5,543/5,607) | 0.9581 ± 0.0112 (2,604/2,718) | 0.9630 ± 0.0068 (2,655/2,757) |
| val_pooled | acorn (ttbar), CC, working-point cuts | 0.9870 ± 0.0031 (5,534/5,607) | 0.9286 ± 0.0132 (2,524/2,718) | 0.9373 ± 0.0142 (2,584/2,757) |

Populations per seed: test_external: all_track 116,449, prompt 4,744, displaced 5,623, prompt_targets 4,629, displaced_mu_plus 2,803, displaced_mu_minus 2,820; test_pooled: all_track 160,016, prompt 6,611, displaced 7,296, prompt_targets 6,442, displaced_mu_plus 3,652, displaced_mu_minus 3,644; val_pooled: all_track 47,152, prompt 1,920, displaced 1,825, prompt_targets 1,869, displaced_mu_plus 906, displaced_mu_minus 919.

## Cross-checks against the frozen records (all must hold)

| check | ok | detail |
|---|---|---|
| reproduction eff_all_targets@hnl15_100_external | yes | earlier 0.9553996456 +- 0.0034039518, now 0.9553996456 +- 0.0034039518 |
| reproduction eff_w_muons_paper@hnl15_100_external | yes | earlier 0.9829960652 +- 0.0018967275, now 0.9829960652 +- 0.0018967275 |
| reproduction eff_hnl_targets@hnl15_100_external | yes | earlier 0.9541170194 +- 0.0042826591, now 0.9541170194 +- 0.0042826591 |
| reproduction eff_hnl_mu_plus@hnl15_100_external | yes | earlier 0.9549292425 +- 0.0042105791, now 0.9549292425 +- 0.0042105791 |
| reproduction eff_hnl_mu_minus@hnl15_100_external | yes | earlier 0.9533096927 +- 0.0058530407, now 0.9533096927 +- 0.0058530407 |
| reproduction fake_rate@hnl15_100_external | yes | earlier 0.0292479760 +- 0.0013847318, now 0.0292479760 +- 0.0013847318 |
| all-track counting = test_stats.efficiency | yes | 0.9553996456 (333766/349347) |
| claims primary canto_l | yes | 0.9907573642 vs 0.9907573642 |
| claims primary acorn_hnl_prereg | yes | 0.9257264446 vs 0.9257264446 |
| val primary canto_l s0 | yes | 0.9887302780 vs 0.9887302780 |
| val primary canto_l s1 | yes | 0.9924868520 vs 0.9924868520 |
| val primary canto_l s2 | yes | 0.9917355372 vs 0.9917355372 |
| val primary canto_h s0 | yes | 0.9932381668 vs 0.9932381668 |
| val primary canto_h s1 | yes | 0.9902329076 vs 0.9902329076 |
| val primary canto_h s2 | yes | 0.9902329076 vs 0.9902329076 |
| val primary canto_b s0 | yes | 0.9849737040 vs 0.9849737040 |
| val primary canto_b s1 | yes | 0.9827197596 vs 0.9827197596 |
| val primary canto_b s2 | yes | 0.9857250188 vs 0.9857250188 |
| val primary acorn_hnl_prereg s0 | yes | 0.9368895567 vs 0.9368895567 |
| val primary acorn_hnl_prereg s1 | yes | 0.9413974455 vs 0.9413974455 |
| val primary acorn_hnl_prereg s2 | yes | 0.9368895567 vs 0.9368895567 |
| val primary acorn_hnl_best s0 | yes | 0.9631855748 vs 0.9631855748 |
| val primary acorn_hnl_best s1 | yes | 0.9413974455 vs 0.9413974455 |
| val primary acorn_hnl_best s2 | yes | 0.9639368896 vs 0.9639368896 |
| val primary acorn_ttbar s0 | yes | 0.9361382419 vs 0.9361382419 |
| val primary acorn_ttbar s1 | yes | 0.9105935387 vs 0.9105935387 |
| val primary acorn_ttbar s2 | yes | 0.9226145755 vs 0.9226145755 |
| best cut armii_s0 | yes | ranking rule 'CC 0.38', frozen 'CC 0.38' |
| best cut armii_s1 | yes | ranking rule 'CC 0.36 (FR* unreachable: minimum-rate cut)', frozen 'CC 0.36 (FR* unreachable: minimum-rate cut)' |
| best cut armii_s2 | yes | ranking rule 'CC 0.44', frozen 'CC 0.44' |
| events test_external | yes | [4999] (expected 4999) |
| events test_pooled | yes | [6999] (expected 6999) |
| events val_pooled | yes | [1999] (expected 1999) |
| prompt denominator external = 4744 | yes | [(4744, 4744, 4744)] |
| same population test_external all_track | yes | n per seed [116449] |
| same population test_external prompt | yes | n per seed [4744] |
| same population test_external displaced | yes | n per seed [5623] |
| same population test_external prompt_targets | yes | n per seed [4629] |
| same population test_external displaced_mu_plus | yes | n per seed [2803] |
| same population test_external displaced_mu_minus | yes | n per seed [2820] |
| same population test_pooled all_track | yes | n per seed [160016] |
| same population test_pooled prompt | yes | n per seed [6611] |
| same population test_pooled displaced | yes | n per seed [7296] |
| same population test_pooled prompt_targets | yes | n per seed [6442] |
| same population test_pooled displaced_mu_plus | yes | n per seed [3652] |
| same population test_pooled displaced_mu_minus | yes | n per seed [3644] |
| same population val_pooled all_track | yes | n per seed [47152] |
| same population val_pooled prompt | yes | n per seed [1920] |
| same population val_pooled displaced | yes | n per seed [1825] |
| same population val_pooled prompt_targets | yes | n per seed [1869] |
| same population val_pooled displaced_mu_plus | yes | n per seed [906] |
| same population val_pooled displaced_mu_minus | yes | n per seed [919] |

## Caveats

- Post-freeze descriptive: every test number here summarises the frozen one-time test outputs (or, for acorn (HNL) at its best cut, seeds 0 and 2, a post-freeze CC evaluation of the stored test scores at cuts fixed on val); none is a claim, none was used to choose anything.
- Sample: the external set is the paper's evaluation sample (15 GeV, c*tau = 100 mm), 4,999 of its 5,000 events (corrupted event 4375 dropped). The pooled test sets add 10 GeV / 100 mm and 15 GeV / 200 mm events (2,000 events), and val holds only those two samples (no 15 GeV / 100 mm event): a different mass and lifetime mix.
- Training: Wang et al. trained on 10,000 HNL events of 15 GeV / 100 mm (not released); acorn (HNL) on the 15,999 training events of the 10 GeV / 100 mm and 15 GeV / 200 mm samples; Canto and acorn (ttbar) on 7,999 ttbar events only.
- Pipelines: Wang et al. used their own metric learning, filter, an attention GNN and DBSCAN; acorn here is the acorn interaction GNN with connected components (the reproduction of their data setup, not of their pipeline); Canto is a different pipeline (triplet GNN, helix-triplet graphs, learned merging).
- Working points: the paper reports no fake rate, so its efficiencies carry no fake-rate constraint (DBSCAN epsilon chosen for efficiency and an undefined track purity); ours sit at fake rate <= FR* = 2.94 % on pooled HNL val (acorn) or at the frozen Canto configurations (about 1.0-1.4 % on val). The comparison is not at equal fake rate.
- Prompt population: 'W muon with >= 5 hits' reproduces the paper's denominator (4,744 on the external set) but not its stated reconstructable definition; with the pT / |eta| cuts the W-muon targets are 4,629 (check table).
- Displaced population: the paper's displaced set has 2,882 particles (Fig. 3), about half of our 5,623 daughter targets in the same events, and its '|d0|' spectrum behaves like our decay radius r_v; no simple half (one charge, leading pT, >= 3 pixel hits) reproduces it. Our displaced number is on all daughter targets (each charge separately as a check), so it is not on the paper's population.
- All-track population: 116,449 targets on the external set (23.29 per event), consistent with the about 116,600 implied by the paper's binomial error; dominated by prompt underlying-event tracks.
- Uncertainties: the paper's method is not stated (its prompt +-0.002 is larger than the binomial +-0.0015 of 4,744 tracks); ours are the seed spread of three training seeds, and on test also the preregistered event bootstrap.
