# Far-displaced HNL daughters on the HNL training split: where Canto-L and acorn (ttbar) lose them

Diagnostic only. The sample is the HNL TRAINING split, which neither pipeline has seen (both are trained on ttbar only); nothing is chosen on it, and no frozen choice (results/final_selection_v1.json) or paper file is touched. No test event, test graph, test score or test output was read. Code `src/hnl/eval/far_trainsplit.py` (job scripts `slurm/feedback/far_trainsplit*.sbatch`); jobs, times and inputs with sha256 in `results/far_trainsplit_v1_meta.json`.

## Summary

(Written by hand from the tables below; counts are particle-seeds pooled over seeds 0-2 unless stated.)

- Sample: 96 primary-population HNL daughters with |d0| >= 300 mm (288 particle-seeds) and 333 with |d0| >= 200 mm
  (999), in 322 HNL train events, 5.1 and 6.5 times the validation numbers (19 and 51). The same code path on the 49
  validation events of round 4 check 2 reproduces the stored graphs exactly, the stored scores within 1e-5 (the GPU
  scoring itself differs by up to 6e-6 between two identical runs), every stored DM match of the four builders and
  round 4's classes (66 of 66 checks).
- |d0| >= 300 mm, found / 288: Canto-L LM 269 (lost 19: graph 11, classifier 6, impure 2), Canto-L MWS+M 267 (lost 21:
  graph 14, classifier 6, fragmented 1), acorn (ttbar) LM 241 (lost 47: graph 28, classifier 12, fragmented 1,
  impure 6), acorn (ttbar) MWS+M 244 (lost 44: graph 27, classifier 12, fragmented 1, impure 4). |d0| >= 200 mm,
  found / 999: 970 (graph 15, classifier 11, impure 3), 969 (graph 18, classifier 11, fragmented 1), 879 (graph 55,
  classifier 53, fragmented 1, impure 11), 881 (graph 58, classifier 47, fragmented 2, impure 11).
- (a) Learned merging and late starts: not supported. At |d0| >= 300 mm Canto-L LM finds 5 particle-seeds that
  MWS+M loses (3 particles, r_first 657-872 mm, each split by the shared first pass into 2-3 fragments that the GBDT
  joins while the helix fit of the union fails, rms 1.4-4.4) and loses 3 that MWS+M finds (2 particles, r_first 870 mm
  in two seeds and 360 mm in one: LM joins the particle's hits to another particle's track of 17-18 hits, purity
  0.24-0.28, where MWS+M keeps a pure track). For the daughters that start beyond 500 mm the GBDT gives p > tau to 11
  of the 13 candidate pairs of their first-pass fragments, the helix fit passes for 3 of 15 pairs, and LM recovers 6 of
  the 15 particle-seeds that the first pass does not match there, helix merging 1. The three LM-specific losses are
  over-merges, not missed merges. The test-set difference (LM 94, MWS+M 101 of 108) does not recur: on these 288
  particle-seeds LM is ahead by 2, where a deficit at the test rate would be about 19.
- (b) Classifier and late starts: not supported between 500 and 700 mm, too few daughters beyond 700 mm. Canto-L
  LM's 6 classifier-class losses at |d0| >= 300 mm are 5 at r_first 200-500 mm (3 particles) and 1 at 872 mm (MWS+M:
  4 and 2), none among the 99 particle-seeds at 500-700 mm, where Canto-L's in-graph chain edges score higher (97.9 %
  above 0.2) than at 200-500 mm (95.0 %). Beyond 700 mm (6 daughters, 18 particle-seeds) 86.7 % of Canto-L's chain
  edges are above 0.2, against 83.3 % for acorn (ttbar), whose classifier was trained without truncation. The
  augmentation makes 2.8 % of the truncated ttbar tracks start at 500-700 mm and 0.04 % beyond 700 mm (0.9 % and
  0.02 % of the tracks of an augmented epoch), against 34 % and 6 % of the |d0| >= 300 mm daughters.
- (c) Graph losses common to all models: supported, the largest class. Three daughters (r_first 362, 502, 581 mm; 1-2
  of their 4-5 chain edges in either graph) are in the graph class for every model and seed: 9 of Canto-L LM's 19
  losses at |d0| >= 300 mm and 9 of acorn (ttbar) LM's 47. One more (r_first 816 mm) is lost in Canto-L's graph in two
  seeds and in acorn (ttbar)'s in all three. Of acorn (ttbar) LM's other 17 graph-class particle-seeds, Canto-L's
  union graph holds the missing chain edges for 14 (Canto-L LM finds 9 of them and loses 5 at its classifier) and
  Canto-L LM finds the remaining 3 by merging fragments that its graph does not connect.
- Side check: Canto-L's model class (hnl.models.augment.TruncationGraphDataset) truncates the items of a 'trainset'
  directory also at inference (one ttbar training graph: 971 nodes stored and as a valset item, 856 as a trainset
  item in the analysis job and 906 in a dry run, because the augmentation's base seed is torch.initial_seed(), drawn
  at random in a process that sets no seed). The 2,000 ttbar training graphs from which Canto-L's merging GBDT learned
  its fragment pairs were scored that way (RUN/gnn_scores/train_infer: data_split [2000, 0, 0],
  slurm/stage1/score_split.sbatch), so that GBDT was trained on truncation-augmented graphs, with a truncation draw
  that was not recorded; acorn (ttbar)'s merging GBDT was trained on unaugmented graphs. All events here were staged
  as valset.

## Population and pipelines

HNL train events (hnl10_100 + hnl15_200): 15,999; primary-population daughters (HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm): 9,759; with |d0| >= 200 mm: 333 in 322 events; with |d0| >= 300 mm: 96 in 95 events (validation: 51 and 19). r_first is the transverse radius of the particle's first hit.

| daughters | n | r_first 0-200 mm | r_first 200-500 mm | r_first 500-700 mm | r_first >=700 mm | hnl10_100 / hnl15_200 | median hits |
|---|---|---|---|---|---|---|---|
| \|d0\| >= 300 mm | 96 | 0 | 57 | 33 | 6 | 18 / 78 | 6 |
| \|d0\| >= 200 mm | 333 | 0 | 263 | 60 | 10 | 104 / 229 | 6 |

Pipelines (per seed 0 / 1 / 2, the committed val-chosen configurations; Canto-L's learned triplet union at k 5, tau 0.005, r_w 0.24 with each seed's own triplet GBDT; acorn (ttbar): its metric-learning graph; the edge cut of the classifier class is the s0 of the first same-layer Mutex Watershed):

| model | builder (val-chosen, seeds 0 / 1 / 2) | edge cut | merging GBDT |
|---|---|---|---|
| Canto-L LM (frozen) | mwslmerge 0.05 / 0.03 / 0.1 (trackbuild_val_lmerge) | 0.2 / 0.2 / 0.2 | runs/stage1_final_learned/l1tl/armi_s<S>/lmerge/model.joblib |
| Canto-L MWS+M | mwslayermerge 0.2 / 0.2 / 0.2 (trackbuild_val_ext) | 0.2 / 0.2 / 0.2 | - |
| acorn (ttbar) LM | mwslmerge 0.03 / 0.02 / 0.05 (trackbuild_val_fb3_all) | 0.2 / 0.2 / 0.2 | runs/feedback_v3/strong_baselines_a2/armi_s<S>/lmerge/model.joblib |
| acorn (ttbar) MWS+M | mwslayermerge 0.1 / 0.1 / 0.1 (trackbuild_val_fb3_all) | 0.1 / 0.1 / 0.1 | - |

## Identity check on the 49 validation events of round 4 check 2

66 of 66 checks pass, plus 6 information rows (`results/far_trainsplit_v1_checks.csv`). The GPU edge scores are not bitwise repeatable: two identical scorings of the stored val graphs with the same checkpoint on an A30 differ by up to 6.4e-06 (table below). The rescored val scores are therefore compared with the stored ones within 0.0001 and on the same side of every builder cut; the graphs, the DM matches and the classes are compared exactly.

| check | value | reference | ok |
|---|---|---|---|
| Canto-L seed 0 working point = final selection | mwslmerge,0.05 | mwslmerge,0.05 | True |
| Canto-L seed 1 working point = final selection | mwslmerge,0.03 | mwslmerge,0.03 | True |
| Canto-L seed 2 working point = final selection | mwslmerge,0.1 | mwslmerge,0.1 | True |
| Canto-L seed 0: scoring checkpoint and GPU = the stored val scores' | best-epoch=28-val_loss=0.052177.ckpt on NVIDIA A30 | best-epoch=28-val_loss=0.052177.ckpt on NVIDIA A30 | True |
| seed 0: metric-learning checkpoint, r_infer, knn_infer = ml_graphs' | best-epoch=20-f1=0.356576.ckpt 0.16 300 | best-epoch=20-f1=0.356576.ckpt 0.16 300 | True |
| seed 0: union build settings = the stored val union graphs' (k, tau, r_w, GBDT) | {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5} | {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5} | True |
| seed 0: union build base check (CPU radius graph = metric-learning graph, truth) | {"events": 20, "only_stored": 0, "only_recomputed": 0, "max_dist_dev": 0.0, "truth_ok": true} | only_stored 0, only_recomputed 0, truth_ok | True |
| Canto-L seed 1: scoring checkpoint and GPU = the stored val scores' | best-epoch=29-val_loss=0.051051.ckpt on NVIDIA A30 | best-epoch=29-val_loss=0.051051.ckpt on NVIDIA A30 | True |
| seed 1: metric-learning checkpoint, r_infer, knn_infer = ml_graphs' | best-epoch=19-f1=0.358315.ckpt 0.16 300 | best-epoch=19-f1=0.358315.ckpt 0.16 300 | True |
| seed 1: union build settings = the stored val union graphs' (k, tau, r_w, GBDT) | {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5} | {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5} | True |
| seed 1: union build base check (CPU radius graph = metric-learning graph, truth) | {"events": 20, "only_stored": 0, "only_recomputed": 0, "max_dist_dev": 0.0, "truth_ok": true} | only_stored 0, only_recomputed 0, truth_ok | True |
| Canto-L seed 2: scoring checkpoint and GPU = the stored val scores' | best-epoch=28-val_loss=0.044359.ckpt on NVIDIA A30 | best-epoch=28-val_loss=0.044359.ckpt on NVIDIA A30 | True |
| seed 2: metric-learning checkpoint, r_infer, knn_infer = ml_graphs' | best-epoch=12-f1=0.354774.ckpt 0.18 500 | best-epoch=12-f1=0.354774.ckpt 0.18 500 | True |
| seed 2: union build settings = the stored val union graphs' (k, tau, r_w, GBDT) | {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5} | {"k": 5, "tau": 0.005, "r_wide": 0.24, "nsig": 5.0, "pt_min": 0.5} | True |
| seed 2: union build base check (CPU radius graph = metric-learning graph, truth) | {"events": 20, "only_stored": 0, "only_recomputed": 0, "max_dist_dev": 0.0, "truth_ok": true} | only_stored 0, only_recomputed 0, truth_ok | True |
| acorn (ttbar) seed 0: scoring checkpoint and GPU = the stored val scores' | best-epoch=23-val_loss=0.058278.ckpt on NVIDIA A30 | best-epoch=23-val_loss=0.058278.ckpt on NVIDIA A30 | True |
| acorn (ttbar) seed 1: scoring checkpoint and GPU = the stored val scores' | best-epoch=29-val_loss=0.056590.ckpt on NVIDIA A30 | best-epoch=29-val_loss=0.056590.ckpt on NVIDIA A30 | True |
| acorn (ttbar) seed 2: scoring checkpoint and GPU = the stored val scores' | best-epoch=28-val_loss=0.046033.ckpt on NVIDIA A30 | best-epoch=28-val_loss=0.046033.ckpt on NVIDIA A30 | True |
| far particles: hits in the graph = n_hits of the particle table | 4608.0 | 4608.0 | True |
| Canto-L seed 0: val graphs = stored (nodes, edges, truth), 49 events | 49.0 | 49.0 | True |
| Canto-L seed 0: val scores bitwise equal to the stored ones, events (information) | 0.0 | 49.0 | information |
| Canto-L seed 0: largest \|score - stored score\| <= 0.0001 | 3.159046173095703e-06 | 0.0001 | True |
| Canto-L seed 0: edges whose score is on the other side of a builder cut (0.1, 0.2) than the stored score | 0.0 | 0.0 | True |
| Canto-L seed 1: val graphs = stored (nodes, edges, truth), 49 events | 49.0 | 49.0 | True |
| Canto-L seed 1: val scores bitwise equal to the stored ones, events (information) | 0.0 | 49.0 | information |
| Canto-L seed 1: largest \|score - stored score\| <= 0.0001 | 2.86102294921875e-06 | 0.0001 | True |
| Canto-L seed 1: edges whose score is on the other side of a builder cut (0.1, 0.2) than the stored score | 0.0 | 0.0 | True |
| Canto-L seed 2: val graphs = stored (nodes, edges, truth), 49 events | 49.0 | 49.0 | True |
| Canto-L seed 2: val scores bitwise equal to the stored ones, events (information) | 0.0 | 49.0 | information |
| Canto-L seed 2: largest \|score - stored score\| <= 0.0001 | 1.6987323760986328e-06 | 0.0001 | True |
| Canto-L seed 2: edges whose score is on the other side of a builder cut (0.1, 0.2) than the stored score | 0.0 | 0.0 | True |
| acorn (ttbar) seed 0: val graphs = stored (nodes, edges, truth), 49 events | 49.0 | 49.0 | True |
| acorn (ttbar) seed 0: val scores bitwise equal to the stored ones, events (information) | 0.0 | 49.0 | information |
| acorn (ttbar) seed 0: largest \|score - stored score\| <= 0.0001 | 8.58306884765625e-06 | 0.0001 | True |
| acorn (ttbar) seed 0: edges whose score is on the other side of a builder cut (0.1, 0.2) than the stored score | 0.0 | 0.0 | True |
| acorn (ttbar) seed 1: val graphs = stored (nodes, edges, truth), 49 events | 49.0 | 49.0 | True |
| acorn (ttbar) seed 1: val scores bitwise equal to the stored ones, events (information) | 0.0 | 49.0 | information |
| acorn (ttbar) seed 1: largest \|score - stored score\| <= 0.0001 | 4.410743713378906e-06 | 0.0001 | True |
| acorn (ttbar) seed 1: edges whose score is on the other side of a builder cut (0.1, 0.2) than the stored score | 0.0 | 0.0 | True |
| acorn (ttbar) seed 2: val graphs = stored (nodes, edges, truth), 49 events | 49.0 | 49.0 | True |
| acorn (ttbar) seed 2: val scores bitwise equal to the stored ones, events (information) | 0.0 | 49.0 | information |
| acorn (ttbar) seed 2: largest \|score - stored score\| <= 0.0001 | 6.556510925292969e-06 | 0.0001 | True |
| acorn (ttbar) seed 2: edges whose score is on the other side of a builder cut (0.1, 0.2) than the stored score | 0.0 | 0.0 | True |
| Canto-L LM (frozen) seed 0: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_lmerge, 1129 particles) | 1117.0 | 1117.0 | True |
| Canto-L LM (frozen) seed 0: val \|d0\| >= 300 found = entry k_d0_300 | 16.0 | 16.0 | True |
| Canto-L MWS+M seed 0: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_ext, 1129 particles) | 1116.0 | 1116.0 | True |
| Canto-L MWS+M seed 0: val \|d0\| >= 300 found = entry k_d0_300 | 16.0 | 16.0 | True |
| Canto-L LM (frozen) seed 1: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_lmerge, 1129 particles) | 1117.0 | 1117.0 | True |
| Canto-L LM (frozen) seed 1: val \|d0\| >= 300 found = entry k_d0_300 | 17.0 | 17.0 | True |
| Canto-L MWS+M seed 1: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_ext, 1129 particles) | 1116.0 | 1116.0 | True |
| Canto-L MWS+M seed 1: val \|d0\| >= 300 found = entry k_d0_300 | 16.0 | 16.0 | True |
| Canto-L LM (frozen) seed 2: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_lmerge, 1129 particles) | 1116.0 | 1116.0 | True |
| Canto-L LM (frozen) seed 2: val \|d0\| >= 300 found = entry k_d0_300 | 17.0 | 17.0 | True |
| Canto-L MWS+M seed 2: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_ext, 1129 particles) | 1116.0 | 1116.0 | True |
| Canto-L MWS+M seed 2: val \|d0\| >= 300 found = entry k_d0_300 | 17.0 | 17.0 | True |
| acorn (ttbar) LM seed 0: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_fb3_all, 1129 particles) | 1111.0 | 1111.0 | True |
| acorn (ttbar) LM seed 0: val \|d0\| >= 300 found = entry k_d0_300 | 15.0 | 15.0 | True |
| acorn (ttbar) MWS+M seed 0: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_fb3_all, 1129 particles) | 1108.0 | 1108.0 | True |
| acorn (ttbar) MWS+M seed 0: val \|d0\| >= 300 found = entry k_d0_300 | 14.0 | 14.0 | True |
| acorn (ttbar) LM seed 1: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_fb3_all, 1129 particles) | 1109.0 | 1109.0 | True |
| acorn (ttbar) LM seed 1: val \|d0\| >= 300 found = entry k_d0_300 | 13.0 | 13.0 | True |
| acorn (ttbar) MWS+M seed 1: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_fb3_all, 1129 particles) | 1102.0 | 1102.0 | True |
| acorn (ttbar) MWS+M seed 1: val \|d0\| >= 300 found = entry k_d0_300 | 13.0 | 13.0 | True |
| acorn (ttbar) LM seed 2: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_fb3_all, 1129 particles) | 1111.0 | 1111.0 | True |
| acorn (ttbar) LM seed 2: val \|d0\| >= 300 found = entry k_d0_300 | 16.0 | 16.0 | True |
| acorn (ttbar) MWS+M seed 2: rebuilt DM of every particle of the 49 val events = stored grid (trackbuild_val_fb3_all, 1129 particles) | 1106.0 | 1106.0 | True |
| acorn (ttbar) MWS+M seed 2: val \|d0\| >= 300 found = entry k_d0_300 | 15.0 | 15.0 | True |
| Canto-L LM (frozen): val far-daughter classes and chain-edge counts = round 4 check 2 (results/round4_checks_far_particles.csv), 3 seeds x 51 particles | 153.0 | 153.0 | True |
| Canto-L LM (frozen): val far daughters' lowest chain-edge score within 0.0001 of round 4's | 8.940696716308594e-07 | 0.0001 | True |
| Canto-L MWS+M: val far-daughter classes and chain-edge counts = round 4 check 2 (results/round4_checks_far_particles.csv), 3 seeds x 51 particles | 153.0 | 153.0 | True |
| Canto-L MWS+M: val far daughters' lowest chain-edge score within 0.0001 of round 4's | 8.940696716308594e-07 | 0.0001 | True |
| feature-store ttbar train graphs = hits table (particle id, r, z per node), first 20 events | 20.0 | 20.0 | True |

Repeatability of the GPU scores (`slurm/feedback/far_trainsplit_repeat.sbatch`: the stored val graphs of the 49 events scored twice with the same checkpoint on an A30):

| seed | pipeline | run 1 = run 2 (bitwise), events | max \|run 1 - run 2\| | run 1 = stored (bitwise), events | max \|run 1 - stored\| | max \|run 2 - stored\| |
|---|---|---|---|---|---|---|
| 0 | Canto-L | 0 / 49 | 2.62e-06 | 0 / 49 | 3.76e-06 | 2.56e-06 |
| 0 | acorn (ttbar) | 0 / 49 | 6.38e-06 | 0 / 49 | 7.87e-06 | 8.46e-06 |

## Loss classes at |d0| >= 300 mm (train split)

Pooled = particle-seeds over the three seeds.

| model | found s0 / s1 / s2 | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|
| Canto-L LM (frozen) | 92 / 87 / 90 | 269 / 288 | 11 | 6 | 0 | 2 | 0 | 19 |
| Canto-L MWS+M | 90 / 87 / 90 | 267 / 288 | 14 | 6 | 1 | 0 | 0 | 21 |
| acorn (ttbar) LM | 84 / 77 / 80 | 241 / 288 | 28 | 12 | 1 | 6 | 0 | 47 |
| acorn (ttbar) MWS+M | 85 / 76 / 83 | 244 / 288 | 27 | 12 | 1 | 4 | 0 | 44 |

Per seed:

| model | seed | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|
| Canto-L LM (frozen) | 0 | 92 / 96 | 3 | 1 | 0 | 0 | 0 | 4 |
| Canto-L LM (frozen) | 1 | 87 / 96 | 4 | 4 | 0 | 1 | 0 | 9 |
| Canto-L LM (frozen) | 2 | 90 / 96 | 4 | 1 | 0 | 1 | 0 | 6 |
| Canto-L LM (frozen) | pooled | 269 / 288 | 11 | 6 | 0 | 2 | 0 | 19 |
| Canto-L MWS+M | 0 | 90 / 96 | 4 | 2 | 0 | 0 | 0 | 6 |
| Canto-L MWS+M | 1 | 87 / 96 | 5 | 3 | 1 | 0 | 0 | 9 |
| Canto-L MWS+M | 2 | 90 / 96 | 5 | 1 | 0 | 0 | 0 | 6 |
| Canto-L MWS+M | pooled | 267 / 288 | 14 | 6 | 1 | 0 | 0 | 21 |
| acorn (ttbar) LM | 0 | 84 / 96 | 10 | 0 | 0 | 2 | 0 | 12 |
| acorn (ttbar) LM | 1 | 77 / 96 | 10 | 6 | 0 | 3 | 0 | 19 |
| acorn (ttbar) LM | 2 | 80 / 96 | 8 | 6 | 1 | 1 | 0 | 16 |
| acorn (ttbar) LM | pooled | 241 / 288 | 28 | 12 | 1 | 6 | 0 | 47 |
| acorn (ttbar) MWS+M | 0 | 85 / 96 | 9 | 0 | 0 | 2 | 0 | 11 |
| acorn (ttbar) MWS+M | 1 | 76 / 96 | 10 | 8 | 0 | 2 | 0 | 20 |
| acorn (ttbar) MWS+M | 2 | 83 / 96 | 8 | 4 | 1 | 0 | 0 | 13 |
| acorn (ttbar) MWS+M | pooled | 244 / 288 | 27 | 12 | 1 | 4 | 0 | 44 |

## Loss classes at |d0| >= 200 mm (train split)

Pooled = particle-seeds over the three seeds.

| model | found s0 / s1 / s2 | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|
| Canto-L LM (frozen) | 325 / 321 / 324 | 970 / 999 | 15 | 11 | 0 | 3 | 0 | 29 |
| Canto-L MWS+M | 323 / 322 / 324 | 969 / 999 | 18 | 11 | 1 | 0 | 0 | 30 |
| acorn (ttbar) LM | 304 / 277 / 298 | 879 / 999 | 55 | 53 | 1 | 11 | 0 | 120 |
| acorn (ttbar) MWS+M | 302 / 275 / 304 | 881 / 999 | 58 | 47 | 2 | 11 | 0 | 118 |

Per seed:

| model | seed | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|
| Canto-L LM (frozen) | 0 | 325 / 333 | 4 | 4 | 0 | 0 | 0 | 8 |
| Canto-L LM (frozen) | 1 | 321 / 333 | 5 | 5 | 0 | 2 | 0 | 12 |
| Canto-L LM (frozen) | 2 | 324 / 333 | 6 | 2 | 0 | 1 | 0 | 9 |
| Canto-L LM (frozen) | pooled | 970 / 999 | 15 | 11 | 0 | 3 | 0 | 29 |
| Canto-L MWS+M | 0 | 323 / 333 | 5 | 5 | 0 | 0 | 0 | 10 |
| Canto-L MWS+M | 1 | 322 / 333 | 6 | 4 | 1 | 0 | 0 | 11 |
| Canto-L MWS+M | 2 | 324 / 333 | 7 | 2 | 0 | 0 | 0 | 9 |
| Canto-L MWS+M | pooled | 969 / 999 | 18 | 11 | 1 | 0 | 0 | 30 |
| acorn (ttbar) LM | 0 | 304 / 333 | 19 | 6 | 0 | 4 | 0 | 29 |
| acorn (ttbar) LM | 1 | 277 / 333 | 21 | 30 | 0 | 5 | 0 | 56 |
| acorn (ttbar) LM | 2 | 298 / 333 | 15 | 17 | 1 | 2 | 0 | 35 |
| acorn (ttbar) LM | pooled | 879 / 999 | 55 | 53 | 1 | 11 | 0 | 120 |
| acorn (ttbar) MWS+M | 0 | 302 / 333 | 20 | 5 | 0 | 6 | 0 | 31 |
| acorn (ttbar) MWS+M | 1 | 275 / 333 | 23 | 30 | 1 | 4 | 0 | 58 |
| acorn (ttbar) MWS+M | 2 | 304 / 333 | 15 | 12 | 1 | 1 | 0 | 29 |
| acorn (ttbar) MWS+M | pooled | 881 / 999 | 58 | 47 | 2 | 11 | 0 | 118 |

The same code path on the 49 validation events (round 4 check 2: Canto-L LM 50 / 57 and 142 / 153, Canto-L MWS+M 49 / 57 and 141 / 153):

|d0| >= 300 mm:

| model | found s0 / s1 / s2 | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|
| Canto-L LM (frozen) | 16 / 17 / 17 | 50 / 57 | 3 | 4 | 0 | 0 | 0 | 7 |
| Canto-L MWS+M | 16 / 16 / 17 | 49 / 57 | 4 | 4 | 0 | 0 | 0 | 8 |
| acorn (ttbar) LM | 15 / 13 / 16 | 44 / 57 | 10 | 3 | 0 | 0 | 0 | 13 |
| acorn (ttbar) MWS+M | 14 / 13 / 15 | 42 / 57 | 12 | 3 | 0 | 0 | 0 | 15 |

|d0| >= 200 mm:

| model | found s0 / s1 / s2 | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|
| Canto-L LM (frozen) | 47 / 47 / 48 | 142 / 153 | 6 | 5 | 0 | 0 | 0 | 11 |
| Canto-L MWS+M | 47 / 46 / 48 | 141 / 153 | 7 | 5 | 0 | 0 | 0 | 12 |
| acorn (ttbar) LM | 45 / 41 / 46 | 132 / 153 | 16 | 5 | 0 | 0 | 0 | 21 |
| acorn (ttbar) MWS+M | 43 / 40 / 44 | 127 / 153 | 19 | 7 | 0 | 0 | 0 | 26 |

## By first-hit radius (train split, pooled over the seeds)

|d0| >= 300 mm (no daughter at this |d0| has r_first < 200 mm):

| r_first [mm] | model | found s0 / s1 / s2 | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|---|
| 200-500 | Canto-L LM (frozen) | 55 / 53 / 55 | 163 / 171 | 3 | 5 | 0 | 0 | 0 | 8 |
| 200-500 | Canto-L MWS+M | 55 / 54 / 55 | 164 / 171 | 3 | 4 | 0 | 0 | 0 | 7 |
| 200-500 | acorn (ttbar) LM | 53 / 49 / 47 | 149 / 171 | 11 | 9 | 1 | 1 | 0 | 22 |
| 200-500 | acorn (ttbar) MWS+M | 53 / 49 / 50 | 152 / 171 | 11 | 7 | 1 | 0 | 0 | 19 |
| 500-700 | Canto-L LM (frozen) | 31 / 31 / 31 | 93 / 99 | 6 | 0 | 0 | 0 | 0 | 6 |
| 500-700 | Canto-L MWS+M | 30 / 30 / 30 | 90 / 99 | 9 | 0 | 0 | 0 | 0 | 9 |
| 500-700 | acorn (ttbar) LM | 28 / 25 / 30 | 83 / 99 | 9 | 2 | 0 | 5 | 0 | 16 |
| 500-700 | acorn (ttbar) MWS+M | 28 / 25 / 30 | 83 / 99 | 9 | 3 | 0 | 4 | 0 | 16 |
| >=700 | Canto-L LM (frozen) | 6 / 3 / 4 | 13 / 18 | 2 | 1 | 0 | 2 | 0 | 5 |
| >=700 | Canto-L MWS+M | 5 / 3 / 5 | 13 / 18 | 2 | 2 | 1 | 0 | 0 | 5 |
| >=700 | acorn (ttbar) LM | 3 / 3 / 3 | 9 / 18 | 8 | 1 | 0 | 0 | 0 | 9 |
| >=700 | acorn (ttbar) MWS+M | 4 / 2 / 3 | 9 / 18 | 7 | 2 | 0 | 0 | 0 | 9 |

|d0| >= 200 mm (no daughter at this |d0| has r_first < 200 mm):

| r_first [mm] | model | found s0 / s1 / s2 | found / n | graph | classifier | fragmented | impure | short | lost |
|---|---|---|---|---|---|---|---|---|---|
| 200-500 | Canto-L LM (frozen) | 257 / 257 / 258 | 772 / 789 | 7 | 10 | 0 | 0 | 0 | 17 |
| 200-500 | Canto-L MWS+M | 257 / 258 / 258 | 773 / 789 | 7 | 9 | 0 | 0 | 0 | 16 |
| 200-500 | acorn (ttbar) LM | 244 / 219 / 235 | 698 / 789 | 38 | 49 | 1 | 3 | 0 | 91 |
| 200-500 | acorn (ttbar) MWS+M | 241 / 218 / 241 | 700 / 789 | 42 | 41 | 2 | 4 | 0 | 89 |
| 500-700 | Canto-L LM (frozen) | 58 / 57 / 58 | 173 / 180 | 6 | 0 | 0 | 1 | 0 | 7 |
| 500-700 | Canto-L MWS+M | 57 / 57 / 57 | 171 / 180 | 9 | 0 | 0 | 0 | 0 | 9 |
| 500-700 | acorn (ttbar) LM | 54 / 51 / 57 | 162 / 180 | 9 | 3 | 0 | 6 | 0 | 18 |
| 500-700 | acorn (ttbar) MWS+M | 54 / 51 / 57 | 162 / 180 | 9 | 4 | 0 | 5 | 0 | 18 |
| >=700 | Canto-L LM (frozen) | 10 / 7 / 8 | 25 / 30 | 2 | 1 | 0 | 2 | 0 | 5 |
| >=700 | Canto-L MWS+M | 9 / 7 / 9 | 25 / 30 | 2 | 2 | 1 | 0 | 0 | 5 |
| >=700 | acorn (ttbar) LM | 6 / 7 / 6 | 19 / 30 | 8 | 1 | 0 | 2 | 0 | 11 |
| >=700 | acorn (ttbar) MWS+M | 7 / 6 / 6 | 19 / 30 | 7 | 2 | 0 | 2 | 0 | 11 |

The classes follow round 4 check 2 (first that applies: found, graph, classifier, builder), so a particle that the builder loses and whose graph ceiling also fails counts as graph. Graph-class particle-seeds (train, |d0| >= 200 mm) that the builder's own first pass did DM-match: Canto-L LM (frozen) 0 of 15; Canto-L MWS+M 0 of 18; acorn (ttbar) LM 1 of 55; acorn (ttbar) MWS+M 0 of 58 (`results/far_trainsplit_v1_merge_particles.csv`).

Train far daughters with |d0| >= 300 mm lost by at least one Canto-L row, class per seed 0 / 1 / 2 (. found, G graph, C classifier, F fragmented, I impure, S short):

| event | particle | \|d0\| [mm] | r_first [mm] | hits | Canto-L LM (frozen) | Canto-L MWS+M | acorn (ttbar) LM | acorn (ttbar) MWS+M | Canto-L chain edges in graph | Canto-L lowest chain score |
|---|---|---|---|---|---|---|---|---|---|---|
| 400005373 | 4515697291952128 | 353 | 360 | 6 | . C . | . . . | . . . | . . . | 4/5 / 4/5 / 4/5 | 0.0041 / 0.0974 / 0.2541 |
| 400006901 | 4510198005760000 | 330 | 362 | 5 | G G G | G G G | G G G | G G G | 2/4 / 2/4 / 2/4 | 0.4706 / 0.6854 / 0.7770 |
| 400005148 | 4522295754227712 | 422 | 485 | 5 | C C C | C C C | G G G | G G G | 3/4 / 4/4 / 3/4 | 0.0002 / 0.0044 / 0.0001 |
| 400006307 | 4512397398114304 | 387 | 500 | 5 | . C . | . C . | G G C | G G . | 3/4 / 3/4 / 3/4 | 0.0002 / 0.0304 / 0.2310 |
| 400001111 | 4513496439980032 | 394 | 502 | 6 | G G G | G G G | G G G | G G G | 1/5 / 1/5 / 1/5 | 0.0382 / 0.2479 / 0.3130 |
| 400007239 | 4513496540643328 | 319 | 581 | 5 | G G G | G G G | G G G | G G G | 1/4 / 1/4 / 1/4 | 0.8168 / 0.8085 / 0.8028 |
| 400004228 | 4515697979817984 | 615 | 657 | 6 | . . . | G G G | G G G | G G G | 4/5 / 4/5 / 4/5 | 0.7642 / 0.7933 / 0.7515 |
| 400005076 | 4515696151101440 | 431 | 816 | 8 | . G G | . G G | G G G | G G G | 5/7 / 2/7 / 2/7 | 0.0006 / 0.9717 / 0.9339 |
| 400004536 | 4510198525853696 | 830 | 844 | 8 | . . . | . F . | . . . | . C C | 7/7 / 7/7 / 7/7 | 0.4011 / 0.8039 / 0.9232 |
| 400001647 | 4531091243270144 | 472 | 870 | 5 | . I I | . . . | . . . | . . . | 4/4 / 4/4 / 4/4 | 0.2741 / 0.7887 / 0.0022 |
| 200006441 | 4525594473660416 | 811 | 872 | 7 | . C . | C C . | G G C | . G . | 6/6 / 6/6 / 6/6 | 0.0018 / 0.0016 / 0.0373 |

Train far daughters with |d0| >= 300 mm lost by at least one acorn (ttbar) row, class per seed 0 / 1 / 2 (. found, G graph, C classifier, F fragmented, I impure, S short):

| event | particle | \|d0\| [mm] | r_first [mm] | hits | Canto-L LM (frozen) | Canto-L MWS+M | acorn (ttbar) LM | acorn (ttbar) MWS+M | acorn (ttbar) chain edges in graph | acorn (ttbar) lowest chain score |
|---|---|---|---|---|---|---|---|---|---|---|
| 400004774 | 4529992000077824 | 310 | 352 | 6 | . . . | . . . | . . F | . . F | 5/5 / 5/5 / 5/5 | 0.6475 / 0.5649 / 0.0322 |
| 400005560 | 4521195839946752 | 348 | 357 | 5 | . . . | . . . | . C . | . C . | 2/4 / 2/4 / 2/4 | 0.7636 / 0.0039 / 0.9921 |
| 200003937 | 4515697761714176 | 310 | 359 | 6 | . . . | . . . | . . I | . . . | 4/5 / 4/5 / 4/5 | 0.0123 / 0.0032 / 0.9898 |
| 400006901 | 4510198005760000 | 330 | 362 | 5 | G G G | G G G | G G G | G G G | 2/4 / 2/4 / 2/4 | 0.2004 / 0.0029 / 0.0029 |
| 400005329 | 4506900343291904 | 349 | 416 | 5 | . . . | . . . | . C C | . C C | 4/4 / 4/4 / 4/4 | 0.0580 / 0.0029 / 0.0048 |
| 400000291 | 4507999687147520 | 409 | 428 | 5 | . . . | . . . | . . C | . . C | 3/4 / 4/4 / 4/4 | 0.0602 / 0.0021 / 0.0072 |
| 400005976 | 4511297332838400 | 319 | 440 | 5 | . . . | . . . | . C C | . C . | 4/4 / 4/4 / 4/4 | 0.0578 / 0.0002 / 0.0095 |
| 400005148 | 4522295754227712 | 422 | 485 | 5 | C C C | C C C | G G G | G G G | 1/4 / 1/4 / 2/4 | 0.9971 / 0.9092 / 0.0277 |
| 200008685 | 4510198743957504 | 331 | 497 | 6 | . . . | . . . | . C C | . C C | 4/5 / 4/5 / 4/5 | 0.1677 / 0.0016 / 0.0806 |
| 400004668 | 4516796367372288 | 338 | 498 | 5 | . . . | . . . | G G G | G G G | 1/4 / 1/4 / 1/4 | 0.9935 / 0.9673 / 0.9986 |
| 400006307 | 4512397398114304 | 387 | 500 | 5 | . C . | . C . | G G C | G G . | 1/4 / 1/4 / 2/4 | 0.9387 / 0.0015 / 0.1045 |
| 400002849 | 4509098661904384 | 378 | 502 | 5 | . . . | . . . | . . . | . C . | 3/4 / 3/4 / 3/4 | 0.2807 / 0.0113 / 0.1507 |
| 400001111 | 4513496439980032 | 394 | 502 | 6 | G G G | G G G | G G G | G G G | 1/5 / 1/5 / 1/5 | 0.0328 / 0.8494 / 0.8653 |
| 400007416 | 4510198173532160 | 485 | 531 | 6 | . . . | . . . | I I . | I I . | 5/5 / 5/5 / 5/5 | 0.7647 / 0.5138 / 0.6236 |
| 200009540 | 4506899252772864 | 428 | 546 | 5 | . . . | . . . | . . . | . C . | 4/4 / 4/4 / 4/4 | 0.8481 / 0.0344 / 0.0187 |
| 400001909 | 4507998546296832 | 543 | 563 | 5 | . . . | . . . | . C . | . C . | 4/4 / 4/4 / 4/4 | 0.1201 / 0.0007 / 0.0133 |
| 400007239 | 4513496540643328 | 319 | 581 | 5 | G G G | G G G | G G G | G G G | 1/4 / 1/4 / 1/4 | 0.8495 / 0.8610 / 0.7992 |
| 400004514 | 4511297718714368 | 393 | 655 | 6 | . . . | . . . | . C . | . . . | 4/5 / 4/5 / 4/5 | 0.8330 / 0.1533 / 0.0362 |
| 400004228 | 4515697979817984 | 615 | 657 | 6 | . . . | G G G | G G G | G G G | 4/5 / 4/5 / 4/5 | 0.8864 / 0.1650 / 0.0445 |
| 400009235 | 4524495599566848 | 430 | 660 | 5 | . . . | . . . | . I . | . . . | 3/4 / 3/4 / 3/4 | 0.0492 / 0.0033 / 0.0149 |
| 400006203 | 4514598887620608 | 602 | 661 | 5 | . . . | . . . | I I . | I I . | 3/4 / 4/4 / 3/4 | 0.7415 / 0.0187 / 0.0121 |
| 400005076 | 4515696151101440 | 431 | 816 | 8 | . G G | . G G | G G G | G G G | 2/7 / 2/7 / 2/7 | 0.9802 / 0.9968 / 0.9961 |
| 400004536 | 4510198525853696 | 830 | 844 | 8 | . . . | . F . | . . . | . C C | 6/7 / 5/7 / 7/7 | 0.6296 / 0.0942 / 0.0124 |
| 200006441 | 4525594473660416 | 811 | 872 | 7 | . C . | C C . | G G C | . G . | 4/6 / 4/6 / 6/6 | 0.6864 / 0.8485 / 0.0010 |
| 400005139 | 4513496020549632 | 782 | 878 | 7 | . . . | . . . | G G G | G G G | 3/6 / 4/6 / 3/6 | 0.7487 / 0.7591 / 0.7472 |

## Learned merging (LM) against MWS-layer + helix merging (MWS+M)

| \|d0\| >= | pipeline | found by LM only (MWS+M class) | found by MWS+M only (LM class) | r_first of LM-only finds | r_first of MWS+M-only finds |
|---|---|---|---|---|---|
| 300 | Canto-L | 5 (graph 3, classifier 1, fragmented 1) | 3 (impure 2, classifier 1) | 657 657 657 844 872 | 360 870 870 |
| 300 | acorn (ttbar) | 4 (classifier 4) | 7 (classifier 4, impure 2, graph 1) | 502 546 844 844 | 359 440 500 655 660 872 872 |
| 200 | Canto-L | 5 (graph 3, classifier 1, fragmented 1) | 4 (impure 3, classifier 1) | 657 657 657 844 872 | 360 596 870 870 |
| 200 | acorn (ttbar) | 13 (graph 4, impure 4, classifier 4, fragmented 1) | 15 (classifier 10, impure 4, graph 1) | 262 357 357 403 414 500 500 500 502 546 789 844 844 | 262 271 273 296 329 359 372 419 440 500 655 660 816 872 872 |

Recovery by merging, |d0| >= 300 mm (train, pooled over seeds; 'matched' = DM-matched; Canto-L's two builders share one first pass, acorn (ttbar)'s LM uses s0 0.2 and its MWS+M s0 0.1):

| pipeline | r_first [mm] | particle-seeds | LM first pass (s0 0.2): not matched | -> recovered by learned merging | matched -> lost by learned merging | MWS+M first pass: not matched | -> recovered by helix merging | matched -> lost by helix merging |
|---|---|---|---|---|---|---|---|---|
| Canto-L | all | 288 | 23 | 6 | 2 | 23 | 2 | 0 |
| Canto-L | 200-500 | 171 | 8 | 0 | 0 | 8 | 1 | 0 |
| Canto-L | 500-700 | 99 | 9 | 3 | 0 | 9 | 0 | 0 |
| Canto-L | >=700 | 18 | 6 | 3 | 2 | 6 | 1 | 0 |
| acorn (ttbar) | all | 288 | 58 | 15 | 4 | 54 | 10 | 0 |
| acorn (ttbar) | 200-500 | 171 | 30 | 9 | 1 | 27 | 8 | 0 |
| acorn (ttbar) | 500-700 | 99 | 18 | 4 | 2 | 18 | 2 | 0 |
| acorn (ttbar) | >=700 | 18 | 10 | 2 | 1 | 9 | 0 | 0 |

Canto-L, every pair of a far daughter's first-pass fragments (>= 2 of its hits each), |d0| >= 300 mm (train, pooled over seeds):

| r_first [mm] | fragment pairs (particle-seeds) | LM candidate | GBDT p > tau | merged by LM | helix fit passes | in a merge_fragments window | merged by MWS+M | median p (candidates) | median fit rms |
|---|---|---|---|---|---|---|---|---|---|
| all | 34 (28) | 23 | 21 | 22 | 22 | 13 | 7 | 0.355 | 0.238 |
| 200-500 | 19 (17) | 10 | 10 | 10 | 19 | 4 | 4 | 0.414 | 0.027 |
| 500-700 | 4 (4) | 4 | 3 | 3 | 0 | 0 | 0 | 0.416 | 1.422 |
| >=700 | 11 (7) | 9 | 8 | 9 | 3 | 9 | 3 | 0.146 | 1.730 |

Recovery by merging, |d0| >= 200 mm (train, pooled over seeds; 'matched' = DM-matched; Canto-L's two builders share one first pass, acorn (ttbar)'s LM uses s0 0.2 and its MWS+M s0 0.1):

| pipeline | r_first [mm] | particle-seeds | LM first pass (s0 0.2): not matched | -> recovered by learned merging | matched -> lost by learned merging | MWS+M first pass: not matched | -> recovered by helix merging | matched -> lost by helix merging |
|---|---|---|---|---|---|---|---|---|
| Canto-L | all | 999 | 32 | 6 | 3 | 32 | 2 | 0 |
| Canto-L | 200-500 | 789 | 17 | 0 | 0 | 17 | 1 | 0 |
| Canto-L | 500-700 | 180 | 9 | 3 | 1 | 9 | 0 | 0 |
| Canto-L | >=700 | 30 | 6 | 3 | 2 | 6 | 1 | 0 |
| acorn (ttbar) | all | 999 | 144 | 30 | 6 | 136 | 18 | 0 |
| acorn (ttbar) | 200-500 | 789 | 112 | 23 | 2 | 105 | 16 | 0 |
| acorn (ttbar) | 500-700 | 180 | 21 | 5 | 2 | 20 | 2 | 0 |
| acorn (ttbar) | >=700 | 30 | 11 | 2 | 2 | 11 | 0 | 0 |

Canto-L, every pair of a far daughter's first-pass fragments (>= 2 of its hits each), |d0| >= 200 mm (train, pooled over seeds):

| r_first [mm] | fragment pairs (particle-seeds) | LM candidate | GBDT p > tau | merged by LM | helix fit passes | in a merge_fragments window | merged by MWS+M | median p (candidates) | median fit rms |
|---|---|---|---|---|---|---|---|---|---|
| all | 61 (55) | 47 | 42 | 42 | 40 | 23 | 15 | 0.351 | 0.216 |
| 200-500 | 42 (40) | 30 | 27 | 27 | 36 | 12 | 11 | 0.373 | 0.126 |
| 500-700 | 8 (8) | 8 | 7 | 6 | 1 | 2 | 1 | 0.323 | 0.937 |
| >=700 | 11 (7) | 9 | 8 | 9 | 3 | 9 | 3 | 0.146 | 1.730 |

Every train particle-seed with |d0| >= 200 mm that one builder finds and the other loses; fragments are the first-pass tracks holding >= 2 of the particle's hits (its hits / track size); for every pair of them the merging GBDT's probability against tau and the helix fit of the union against merge_fragments' cuts (rms <= 1, zrms <= 5, <= 1 outlier, R span >= 50 mm; candidate windows 'any' and 'seq'):

| pipeline | seed | event | particle | \|d0\| | r_first | hits | found by | LM class | MWS+M class | DM first pass (LM / MWS) | LM first-pass fragments | pairs (LM first pass, s0 0.2) | pairs (MWS+M first pass) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Canto-L | 0 | 200006441 | 4525594473660416 | 811 | 872 | 7 | LM only | found | classifier | n / n | 3/3 2/2 2/2 | 3/3+2/2: no LM candidate, LM not merged; fit rms 1.73 zrms 2.36 out 1 span 872 (fail, window none), MWS+M not merged \| 3/3+2/2: p 0.101 > tau 0.05, LM merged; fit rms 4.39 zrms 5.89 out 1 span 630 (fail, window seq), MWS+M not merged \| 2/2+2/2: p 0.030 <= tau 0.05, LM not merged; fit rms 0.87 zrms 11.02 out 0 span 259 (fail, window seq), MWS+M not merged | same first pass |
| Canto-L | 0 | 400004228 | 4515697979817984 | 615 | 657 | 6 | LM only | found | graph | n / n | 3/3 3/3 | 3/3+3/3: p 0.381 > tau 0.05, LM merged; fit rms 1.42 zrms 1.50 out 1 span 659 (fail, window none), MWS+M not merged | same first pass |
| Canto-L | 1 | 400004228 | 4515697979817984 | 615 | 657 | 6 | LM only | found | graph | n / n | 3/3 3/3 | 3/3+3/3: p 0.589 > tau 0.03, LM merged; fit rms 1.42 zrms 1.50 out 1 span 659 (fail, window none), MWS+M not merged | same first pass |
| Canto-L | 2 | 400004228 | 4515697979817984 | 615 | 657 | 6 | LM only | found | graph | n / n | 3/3 3/3 | 3/3+3/3: p 0.450 > tau 0.1, LM merged; fit rms 1.42 zrms 1.50 out 1 span 659 (fail, window none), MWS+M not merged | same first pass |
| Canto-L | 1 | 400004536 | 4510198525853696 | 830 | 844 | 8 | LM only | found | fragmented | n / n | 4/4 4/4 | 4/4+4/4: p 0.833 > tau 0.03, LM merged; fit rms 2.36 zrms 4.22 out 2 span 1266 (fail, window seq), MWS+M not merged | same first pass |
| Canto-L | 1 | 200008725 | 4517896784969728 | 241 | 596 | 5 | MWS+M only | impure | found | y / y | 3/3 2/2 | 3/3+2/2: p 0.264 > tau 0.03, LM not merged; fit rms 0.45 zrms 5.35 out 0 span 1479 (fail, window any), MWS+M not merged | same first pass |
| Canto-L | 1 | 400001647 | 4531091243270144 | 472 | 870 | 5 | MWS+M only | impure | found | y / y | 5/5 | (fewer than 2 fragments with >= 2 of its hits) | same first pass |
| Canto-L | 2 | 400001647 | 4531091243270144 | 472 | 870 | 5 | MWS+M only | impure | found | y / y | 3/3 2/2 | 3/3+2/2: p 0.272 > tau 0.1, LM merged; fit rms 4.06 zrms 10.61 out 1 span 811 (fail, window seq), MWS+M not merged | same first pass |
| Canto-L | 1 | 400005373 | 4515697291952128 | 353 | 360 | 6 | MWS+M only | classifier | found | n / n | 2/2 2/2 2/2 | 2/2+2/2: no LM candidate, LM not merged; fit rms 0.11 zrms 0.27 out 0 span 740 (pass, window none), MWS+M not merged \| 2/2+2/2: no LM candidate, LM not merged; fit rms 0.25 zrms 2.18 out 0 span 1107 (pass, window none), MWS+M not merged \| 2/2+2/2: p 0.068 > tau 0.03, LM merged; fit rms 0.30 zrms 2.25 out 0 span 370 (pass, window seq), MWS+M merged | same first pass |
| acorn (ttbar) | 0 | 200001863 | 4509099148443648 | 276 | 357 | 5 | LM only | found | graph | n / n | 2/2 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 1 | 200001863 | 4509099148443648 | 276 | 357 | 5 | LM only | found | graph | n / n | 2/2 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 1 | 200002895 | 4513498168033280 | 238 | 403 | 5 | LM only | found | fragmented | n / n | 2/2 2/2 | 2/2+2/2: p 0.084 > tau 0.02, LM merged; fit rms 0.25 zrms 1.17 out 0 span 1043 (pass, window none) | 2/2+2/2: fit rms 0.25 zrms 1.17 out 0 span 1043 (pass, window none), MWS+M not merged |
| acorn (ttbar) | 0 | 200005219 | 4509098242473984 | 253 | 500 | 6 | LM only | found | graph | n / n | 3/3 2/2 | 3/3+2/2: p 0.257 > tau 0.03, LM merged; fit rms 0.02 zrms 2.14 out 0 span 760 (pass, window none) | 3/3+2/2: fit rms 0.02 zrms 2.14 out 0 span 760 (pass, window none), MWS+M not merged |
| acorn (ttbar) | 1 | 200005219 | 4509098242473984 | 253 | 500 | 6 | LM only | found | graph | n / n | 3/3 2/2 | 3/3+2/2: p 0.329 > tau 0.02, LM merged; fit rms 0.02 zrms 2.14 out 0 span 760 (pass, window none) | 3/3+2/2: fit rms 0.02 zrms 2.14 out 0 span 760 (pass, window none), MWS+M not merged |
| acorn (ttbar) | 0 | 200006901 | 4510198005760000 | 285 | 414 | 6 | LM only | found | impure | y / n | 6/6 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 1 | 200009540 | 4506899252772864 | 428 | 546 | 5 | LM only | found | classifier | n / n | 2/2 2/2 | 2/2+2/2: p 0.077 > tau 0.02, LM merged; fit rms 0.08 zrms 0.37 out 0 span 921 (pass, window none) | 2/2+2/2: fit rms 0.08 zrms 0.37 out 0 span 921 (pass, window none), MWS+M not merged |
| acorn (ttbar) | 2 | 400002382 | 4511299111223296 | 252 | 789 | 5 | LM only | found | impure | y / n | 4/4 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 1 | 400002849 | 4509098661904384 | 378 | 502 | 5 | LM only | found | classifier | n / n | 2/2 2/2 | 2/2+2/2: p 0.101 > tau 0.02, LM merged; fit rms 0.00 zrms 5.63 out 0 span 360 (fail, window any+seq) | 2/2+2/2: fit rms 0.00 zrms 5.63 out 0 span 360 (fail, window any+seq), MWS+M not merged |
| acorn (ttbar) | 1 | 400004245 | 4522295938777088 | 255 | 500 | 5 | LM only | found | impure | y / n | 3/3 2/2 | 3/3+2/2: p 0.290 > tau 0.02, LM merged; fit rms 0.01 zrms 0.49 out 0 span 599 (pass, window any+seq) | 3/6+2/2: fit rms 94.27 zrms 39.10 out 5 span 644 (fail, window seq), MWS+M not merged |
| acorn (ttbar) | 1 | 400004536 | 4510198525853696 | 830 | 844 | 8 | LM only | found | classifier | n / n | 4/4 2/2 | 4/4+2/2: p 0.207 > tau 0.02, LM merged; fit rms 2.16 zrms 4.32 out 1 span 743 (fail, window seq) | 4/4+2/2: fit rms 2.16 zrms 4.32 out 1 span 743 (fail, window seq), MWS+M not merged |
| acorn (ttbar) | 2 | 400004536 | 4510198525853696 | 830 | 844 | 8 | LM only | found | classifier | n / n | 4/4 2/2 2/2 | 4/4+2/2: p 0.286 > tau 0.05, LM merged; fit rms 2.36 zrms 3.14 out 1 span 534 (fail, window seq) \| 4/4+2/2: p 0.286 > tau 0.05, LM merged; fit rms 2.16 zrms 4.32 out 1 span 743 (fail, window seq) \| 2/2+2/2: p 0.114 > tau 0.05, LM merged; fit rms 0.01 zrms 1.63 out 0 span 1266 (pass, window none) | 4/4+2/2: fit rms 2.36 zrms 3.14 out 1 span 534 (fail, window seq), MWS+M not merged \| 4/4+2/2: fit rms 2.16 zrms 4.32 out 1 span 743 (fail, window seq), MWS+M not merged \| 2/2+2/2: fit rms 0.01 zrms 1.63 out 0 span 1266 (pass, window none), MWS+M not merged |
| acorn (ttbar) | 0 | 400006876 | 4518996229488640 | 216 | 262 | 6 | LM only | found | impure | y / n | 5/5 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 2 | 200003937 | 4515697761714176 | 310 | 359 | 6 | MWS+M only | impure | found | y / y | 4/4 2/9 | 4/4+2/9: p 0.209 > tau 0.05, LM merged; fit rms 48.87 zrms 7.40 out 10 span 1086 (fail, window none) | 4/4+2/9: fit rms 48.87 zrms 7.40 out 10 span 1086 (fail, window none), MWS+M not merged |
| acorn (ttbar) | 0 | 200006441 | 4525594473660416 | 811 | 872 | 7 | MWS+M only | graph | found | y / y | 5/5 2/2 | 5/5+2/2: p 0.227 > tau 0.03, LM merged; fit rms 3.48 zrms 3.91 out 3 span 872 (fail, window seq) | 5/5+2/2: fit rms 3.48 zrms 3.91 out 3 span 872 (fail, window seq), MWS+M not merged |
| acorn (ttbar) | 2 | 200006441 | 4525594473660416 | 811 | 872 | 7 | MWS+M only | classifier | found | n / y | 3/3 2/2 2/2 | 3/3+2/2: no LM candidate, LM merged; fit rms 1.73 zrms 2.36 out 1 span 872 (fail, window none) \| 3/3+2/2: p 0.349 > tau 0.05, LM merged; fit rms 4.39 zrms 5.89 out 1 span 630 (fail, window seq) \| 2/2+2/2: p 0.116 > tau 0.05, LM merged; fit rms 0.87 zrms 11.02 out 0 span 259 (fail, window seq) | 5/5+2/2: fit rms 3.48 zrms 3.91 out 3 span 872 (fail, window seq), MWS+M not merged |
| acorn (ttbar) | 1 | 200007936 | 4517896449425408 | 224 | 271 | 6 | MWS+M only | classifier | found | n / y | 3/4 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 2 | 200009939 | 4513497798934528 | 290 | 296 | 9 | MWS+M only | classifier | found | n / n | 4/4 2/2 2/2 | 4/4+2/2: no LM candidate, LM not merged; fit rms 0.18 zrms 0.65 out 0 span 1712 (pass, window none) \| 4/4+2/2: no LM candidate, LM not merged; fit rms 0.31 zrms 0.53 out 0 span 1440 (pass, window none) \| 2/2+2/2: p 0.093 > tau 0.05, LM merged; fit rms 0.01 zrms 2.01 out 0 span 275 (pass, window seq) | 4/4+3/3: fit rms 0.31 zrms 0.53 out 0 span 1440 (pass, window seq), MWS+M merged \| 4/4+2/2: fit rms 0.18 zrms 0.65 out 0 span 1712 (pass, window none), MWS+M merged \| 3/3+2/2: fit rms 0.38 zrms 0.66 out 0 span 568 (pass, window seq), MWS+M merged |
| acorn (ttbar) | 1 | 400001515 | 4518996028162048 | 245 | 273 | 5 | MWS+M only | impure | found | y / y | 3/5 2/5 | 3/5+2/5: p 0.000 <= tau 0.02, LM not merged; fit rms 30.70 zrms 94.99 out 5 span 1476 (fail, window none) | 3/5+2/5: fit rms 30.70 zrms 94.99 out 5 span 1476 (fail, window none), MWS+M not merged |
| acorn (ttbar) | 0 | 400003332 | 4517895996440576 | 279 | 419 | 5 | MWS+M only | classifier | found | n / y | 2/2 2/4 | 2/2+2/4: p 0.015 <= tau 0.03, LM not merged; fit rms 83.51 zrms 255.05 out 3 span 1509 (fail, window seq) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 1 | 400003625 | 4518996095270912 | 263 | 372 | 6 | MWS+M only | classifier | found | n / n | 2/2 | (fewer than 2 fragments with >= 2 of its hits) | 2/2+2/2: fit rms 0.16 zrms 1.79 out 0 span 1723 (pass, window any), MWS+M merged |
| acorn (ttbar) | 1 | 400004514 | 4511297718714368 | 393 | 655 | 6 | MWS+M only | classifier | found | n / y | 3/3 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 2 | 400004955 | 4511297836154880 | 253 | 329 | 5 | MWS+M only | classifier | found | n / n | 2/2 | (fewer than 2 fragments with >= 2 of its hits) | 2/2+2/2: fit rms 0.28 zrms 0.71 out 0 span 625 (pass, window seq), MWS+M merged |
| acorn (ttbar) | 2 | 400005632 | 4516796266708992 | 272 | 816 | 5 | MWS+M only | impure | found | y / y | 3/3 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 2 | 400005976 | 4511297332838400 | 319 | 440 | 5 | MWS+M only | classifier | found | n / n | 2/2 | (fewer than 2 fragments with >= 2 of its hits) | 2/2+2/2: fit rms 0.03 zrms 0.75 out 0 span 889 (pass, window any), MWS+M merged |
| acorn (ttbar) | 2 | 400006307 | 4512397398114304 | 387 | 500 | 5 | MWS+M only | classifier | found | n / y |  | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |
| acorn (ttbar) | 2 | 400006876 | 4518996229488640 | 216 | 262 | 6 | MWS+M only | classifier | found | n / n | 3/3 | (fewer than 2 fragments with >= 2 of its hits) | 3/3+2/2: fit rms 0.13 zrms 2.22 out 0 span 1447 (pass, window any+seq), MWS+M merged |
| acorn (ttbar) | 1 | 400009235 | 4524495599566848 | 430 | 660 | 5 | MWS+M only | impure | found | y / y | 3/3 | (fewer than 2 fragments with >= 2 of its hits) | (fewer than 2 fragments with >= 2 of its hits) |

## Classifier losses and chain-edge scores by first-hit radius (hypothesis b)

|d0| >= 300 mm (train, pooled over seeds; Canto-L's cut is 0.2 for both builders, acorn (ttbar)'s 0.2 for LM and 0.1 for MWS+M):

| pipeline | r_first [mm] | particle-seeds | LM graph / classifier | MWS+M graph / classifier | chain edges in graph | frac > 0.1 | frac > 0.2 | frac > 0.5 | all in-graph chain edges > 0.2 | median first-edge score |
|---|---|---|---|---|---|---|---|---|---|---|
| Canto-L | 200-500 | 171 | 3 / 5 | 3 / 4 | 806 | 0.974 | 0.950 | 0.908 | 137 / 171 | 0.819 |
| Canto-L | 500-700 | 99 | 6 / 0 | 9 / 0 | 424 | 0.986 | 0.979 | 0.965 | 90 / 99 | 0.932 |
| Canto-L | >=700 | 18 | 2 / 1 | 2 / 2 | 90 | 0.911 | 0.867 | 0.822 | 10 / 18 | 0.867 |
| acorn (ttbar) | 200-500 | 171 | 11 / 9 | 11 / 7 | 752 | 0.916 | 0.884 | 0.828 | 114 / 171 | 0.742 |
| acorn (ttbar) | 500-700 | 99 | 9 / 2 | 9 / 3 | 402 | 0.933 | 0.910 | 0.843 | 69 / 99 | 0.833 |
| acorn (ttbar) | >=700 | 18 | 8 / 1 | 7 / 2 | 72 | 0.903 | 0.833 | 0.778 | 10 / 18 | 0.831 |

|d0| >= 200 mm (train, pooled over seeds; Canto-L's cut is 0.2 for both builders, acorn (ttbar)'s 0.2 for LM and 0.1 for MWS+M):

| pipeline | r_first [mm] | particle-seeds | LM graph / classifier | MWS+M graph / classifier | chain edges in graph | frac > 0.1 | frac > 0.2 | frac > 0.5 | all in-graph chain edges > 0.2 | median first-edge score |
|---|---|---|---|---|---|---|---|---|---|---|
| Canto-L | 200-500 | 789 | 7 / 10 | 7 / 9 | 3813 | 0.972 | 0.950 | 0.922 | 634 / 789 | 0.907 |
| Canto-L | 500-700 | 180 | 6 / 0 | 9 / 0 | 797 | 0.984 | 0.975 | 0.955 | 163 / 180 | 0.950 |
| Canto-L | >=700 | 30 | 2 / 1 | 2 / 2 | 135 | 0.926 | 0.859 | 0.763 | 15 / 30 | 0.646 |
| acorn (ttbar) | 200-500 | 789 | 38 / 49 | 42 / 41 | 3551 | 0.914 | 0.875 | 0.823 | 514 / 789 | 0.841 |
| acorn (ttbar) | 500-700 | 180 | 9 / 3 | 9 / 4 | 773 | 0.920 | 0.886 | 0.799 | 120 / 180 | 0.838 |
| acorn (ttbar) | >=700 | 30 | 8 / 1 | 7 / 2 | 117 | 0.863 | 0.761 | 0.701 | 11 / 30 | 0.651 |

Start radius of ttbar training tracks under the truncation augmentation (hnl.models.augment: each particle with >= 6 hits truncated with p = 0.3, its m innermost hits in R removed, m uniform in 1..n-5; expected values for one epoch over the 7,999 ttbar train events; 'natural' = no augmentation, 'truncated' = the starts of truncated tracks only, 'augmented' = every track of one augmented epoch) against the first-hit radius of the far daughters:

| tracks | start | 0-200 mm | 200-500 mm | 500-700 mm | >=700 mm | per epoch at 500-700 / >= 700 | mean hits at >= 700 |
|---|---|---|---|---|---|---|---|
| ttbar train tracks (pid != 0, >= 5 hits) | natural | 97.53 % | 2.36 % | 0.11 % | 0.00 % | 1532 / 40 | 5.0 |
| ttbar train tracks (pid != 0, >= 5 hits) | truncated | 71.01 % | 26.13 % | 2.82 % | 0.04 % | 11274 / 168 | 5.1 |
| ttbar train tracks (pid != 0, >= 5 hits) | augmented | 89.50 % | 9.55 % | 0.94 % | 0.02 % | 12696 / 207 | 5.1 |
| ttbar train targets (pT > 1 GeV, \|eta\| < 4, >= 5 hits) | natural | 98.20 % | 1.69 % | 0.11 % | 0.00 % | 558 / 16 | 5.0 |
| ttbar train targets (pT > 1 GeV, \|eta\| < 4, >= 5 hits) | truncated | 64.33 % | 32.07 % | 3.55 % | 0.04 % | 5449 / 69 | 5.1 |
| ttbar train targets (pT > 1 GeV, \|eta\| < 4, >= 5 hits) | augmented | 87.98 % | 10.85 % | 1.16 % | 0.02 % | 5963 / 85 | 5.1 |
| far daughters, HNL train, \|d0\| >= 300 mm | first hit (particle table) | 0.00 % | 59.38 % | 34.38 % | 6.25 % | 33 / 6 | 6.7 |
| far daughters, HNL train, \|d0\| >= 200 mm | first hit (particle table) | 0.00 % | 78.98 % | 18.02 % | 3.00 % | 60 / 10 | 6.0 |
| far daughters, HNL val, \|d0\| >= 200 mm | first hit (particle table) | 0.00 % | 78.43 % | 19.61 % | 1.96 % | 10 / 1 | 5.0 |

## Graph losses common to both pipelines (hypothesis c)

|d0| >= 300 mm (graph ceiling = CC on the graph's chain-true edges; Canto-L's union graph contains acorn (ttbar)'s metric-learning graph of the same seed):

| quantity | count |
|---|---|
| Canto-L LM (frozen): graph-class losses (particle-seeds) | 11 |
| Canto-L MWS+M: graph-class losses (particle-seeds) | 14 |
| acorn (ttbar) LM: graph-class losses (particle-seeds) | 28 |
| acorn (ttbar) MWS+M: graph-class losses (particle-seeds) | 27 |
| particles in the graph class for every model and seed (12 of 12) | 3 |
| particles in the graph class for >= 1 model-seed | 10 |
| particles in the graph class for every Canto-L row (both builders, 3 seeds) | 3 |
| particles in the graph class for every acorn (ttbar) row (both builders, 3 seeds) | 8 |
| Canto-L: graph ceiling fails (particle-seeds; found or not) | 14 |
| acorn (ttbar): graph ceiling fails (particle-seeds; found or not) | 34 |
| particles whose graph ceiling fails in every seed of both pipelines | 4 |

|d0| >= 200 mm (graph ceiling = CC on the graph's chain-true edges; Canto-L's union graph contains acorn (ttbar)'s metric-learning graph of the same seed):

| quantity | count |
|---|---|
| Canto-L LM (frozen): graph-class losses (particle-seeds) | 15 |
| Canto-L MWS+M: graph-class losses (particle-seeds) | 18 |
| acorn (ttbar) LM: graph-class losses (particle-seeds) | 55 |
| acorn (ttbar) MWS+M: graph-class losses (particle-seeds) | 58 |
| particles in the graph class for every model and seed (12 of 12) | 4 |
| particles in the graph class for >= 1 model-seed | 25 |
| particles in the graph class for every Canto-L row (both builders, 3 seeds) | 4 |
| particles in the graph class for every acorn (ttbar) row (both builders, 3 seeds) | 14 |
| Canto-L: graph ceiling fails (particle-seeds; found or not) | 18 |
| acorn (ttbar): graph ceiling fails (particle-seeds; found or not) | 67 |
| particles whose graph ceiling fails in every seed of both pipelines | 5 |

## Side check: Canto-L's model class truncates 'trainset' items at inference

One stored ttbar training graph of Canto-L seed 0 (/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/baseline_v1/armi_s0/ltf_union_graphs/trainset.tar:event100000000.pyg.gz), staged both as trainset and as valset and loaded through TruncSymmetricTripletIGNN's inference setup (hnl.train.acorn_runner.infer): 971 nodes / 12164 edges stored, 971 / 12164 as a valset item, 856 / 9116 as a trainset item. All events here were therefore staged as valset, like the val graphs.

## Files

- `results/far_trainsplit_v1.csv`: counts per split, |d0| threshold, first-hit radius bin, model and seed (and pooled).
- `results/far_trainsplit_v1_particles.csv`: one row per far daughter, model and seed (class and diagnostics, as round 4 check 2).
- `results/far_trainsplit_v1_discordant.csv`, `results/far_trainsplit_v1_merge_particles.csv`, `results/far_trainsplit_v1_merge_pairs.csv`: LM vs MWS+M disagreements, per far daughter the DM match after each building step and its first-pass fragments, and every pair of these fragments (GBDT probability and features, helix fit, windows, merged or not).
- `results/far_trainsplit_v1_chain_scores.csv`: per far daughter, model and seed, its chain edges in R order with their scores.
- `results/far_trainsplit_v1_truncation.csv`: start-radius distributions (100 mm and coarse bins).
- `results/far_trainsplit_v1_checks.csv`, `results/far_trainsplit_v1_meta.json`.
- Graphs and scores (small, only these 371 events per seed): `$HNL_DATA/runs/feedback_v4/far_trainsplit/s<S>/{acorn_ttbar,canto_l}/valset.tar`.
