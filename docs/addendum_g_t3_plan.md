# Addendum G, step T3: acorn (ttbar+G) and Canto-L (ttbar+G)

Status: written 2026-10-14, after the T1 commit (967a64f) and before any retrained model exists. It fixes how the two
retrained models of `docs/preregistration_new_sample.md` section 5.2 are trained and selected. The user approved the
GPU budget (G0, about 190-210 A30 GPU-hours; DECISIONS 4d4a08b). No test event of any sample is read in T3; the
selection file is committed before any retrained model reads a test event (T4).

## 1. Training sample: arm g (`hnl.train.arm_g`)

Stage-0 arm (iii) with the HNL events replaced by the generic train list of T1.

| list | events | content |
|---|---|---|
| train | 15,998 | ttbar train 7,999 + `splits/generic_train_g7999.txt` 7,999 (generic_iso 4,005, generic_dense 3,994) |
| valmon | 498 | the first 166 val events of ttbar, generic_iso and generic_dense each (checkpoint monitoring) |
| val | 2,641 | ttbar val 1,000 + generic_iso val 806 + generic_dense val 835 (the arm's own val mix) |
| val_all | 4,640 | val of ttbar, hnl10_100, hnl15_200, generic_iso, generic_dense (scored for the read-outs) |

Training on 15,998 graphs per epoch, as arm (iii) (15,998), so the arm-(iii) epoch budget gives the same number of
steps.

## 2. acorn (ttbar+G): the arm-(iii) recipe (`docs/preregistration.md` section 1b)

Three seeds (0, 1, 2), each retraining every stage. Same code and configs as Stage 0:
- **Metric learning:** `configs/baseline/ml_train.yaml`, 45 epochs (arm (iii)'s budget), best checkpoint by valmon f1.
- **Graph:** (r, knn) by the Stage-0 rule on the arm's own val mix: the smallest r on `hnl.graphs.ml_scan`'s grid with
  at least 99 % of all true edges, and the smallest knn on its grid that never binds, else binds on fewer than 5 % of
  the nodes; a hold if no grid point qualifies or the graph exceeds 50k pairs per event.
- **Edge classifier:** `configs/baseline/gnn_train.yaml` (InteractionGNN), 30 epochs, best checkpoint by valmon loss.
- **Learned merging:** a merge model per run from that run's GNN scores of 2,000 training graphs drawn at random
  (numpy rng 0; `slurm/stage1/score_split_sample.sbatch`, `hnl.trackbuild.learned_merge`), as in feedback round 3.

## 3. Canto-L (ttbar+G): the frozen Canto-L recipe (aecb23f) with arm g's events

As Canto-L (ttbar+HNL) in `docs/training_sample_study.md` sections 4 and 9, on the embedding and graphs of acorn
(ttbar+G) of the same seed:
- **Triplet support:** GBDT on 400 training graphs drawn at random (rng 0), every positive and 1 % of the negatives,
  the same 16 features, hyperparameters and holdout; **operating point fixed at k 5, tau 0.005, r_w 0.24**.
- **Union graphs:** `hnl.graphs.helix_union build` with that model, of the train and val_all graphs.
- **Classifier:** `configs/stage1/triplet_symregion_trunc.yaml` unchanged (inner-hit truncation on the training
  graphs), 30 epochs, best checkpoint by valmon loss.
- **Learned merging:** as for acorn (2,000 scored training graphs at random, rng 0).

## 4. Selection on the generic val split (addendum G section 5.2)

- **Events:** generic_iso val (806 events). P_G as defined at T1 (`in_pop` of the T1 truth table: gun targets with
  |d0| >= 10 mm or r_v >= 100 mm; 956 targets). Fake rate: tracks with at least 5 hits on the same events.
- **Per run and family:** the configurations with fake rate <= FR* = 0.029368 (`results/working_points_v1.json`), then
  the one-standard-error rule (`hnl.eval.trackbuild_select.tie_best`: tied within one binomial standard error of the
  best P_G efficiency, the lowest fake rate wins).
- **Per model:** the family by the same rule on the three-seed means, with the pooled count, among the families that
  reach FR* in every seed; each seed keeps its own configuration of that family.
- **Menus:**
  - acorn: the menu of feedback round 3, the 133 configurations of `configs/addendum_g/acorn_menu_v3.json` (CCandWalk
    plain and score-guided, walkthrough + merging, MWS-layer + merging, two-pass MWS + merging, learned merging) plus
    CC on the 0.01 grid;
  - Canto-L: the menu of the frozen selection (`hnl.eval.final_select` grids): CC on the 0.05 grid, walkthrough +
    merging, MWS-layer with and without merging, two-pass MWS with and without merging, learned merging.
- **Edges:** a chosen configuration with a parameter at the edge of its grid is reported. If it is the chosen family's,
  that family's grid is extended past the edge for all three seeds, as in feedback round 3, and the rule is applied
  again; logged in RESULTS.md before T4.
- **Prompt guard:** recorded, with a guarded alternative where needed (D-g2).
- **Output:** `results/addendum_g_selection_v1.json`: per model the family, and per seed the configuration, the val
  numbers, the prompt-guard record, every checkpoint and model file with its sha256, and the graph settings T4 needs.

## 5. Read-outs on validation (addendum G section 7)

After the selection, from the val_all scores: the selected configuration on generic_dense val, and the W-muon,
ttbar-prompt and HNL-daughter efficiencies on the ttbar and HNL val splits. They are reported; none of them enters the
selection.

## 6. Differences from the Stage-0 and training-sample runs, and why

- **D-g1 Events of the rule.** P_G is a population of the isolated component, so the rule's efficiency and fake rate are
  both taken on generic_iso val, as the claims' fake-rate guard is on generic_iso test (RESULTS.md, T1 deviation 7).
- **D-g2 No prompt guard in the deciding rule; the guard recorded, with a guarded alternative.** Addendum G section
  5.2 states the rule without the prompt guards of the HNL selection (rules R6/R8 of `hnl.eval.final_select`), so the
  unguarded selection is the one that decides G2 and G3. A reader may take "the selection rule of the paper" to include
  the guard, so (clarification added 2026-10-14, before any val number of a retrained model existed):
  - for every model, seed and family, the per-run choice is evaluated on the ttbar and HNL val events and checked
    against the paper's guard against arm (i): W-muon and ttbar-prompt efficiency at most `GUARD` = 0.2 points below
    acorn arm (i) of the same seed with CC at its section-4 cut (`hnl.eval.trackbuild_select.GUARD`,
    `reference_row`; the `guard_arm_i` definition of `final_select`), recorded per seed in the selection file
    (`hnl.eval.addendum_g_t3 guard`, `RUN/guard_val.json`);
  - if the chosen family of a model fails the guard in any seed, the best family that passes it in every seed is
    also selected by the same rule and written to the selection file as `guarded_alternative`. At T4 it is evaluated
    and reported next to the claim rows; it decides no claim. If no family passes in every seed, the file says so.
  This is logged in RESULTS.md before T4.
- **D-g3 Truth-table steps.** The diagnostic val scan of the GBDT step (never used for a choice: the operating point is
  fixed) and the graph-level metrics of the union build read the HNL truth tables, so they are not run (`--no-metrics`;
  the graphs are the same). The Stage-0 CC curves on val (`evaluate_gnn`) are replaced by CC on the 0.01 grid inside
  the menu evaluation, with the T1 truth.
- **D-g4 Metric learning in one run of 45 epochs.** Arm (iii) ran 30 epochs, then 15 more by the Stage-0 convergence
  rule (DECISIONS 2026-10-03); arm g is given the 45 epochs from the start.
- **D-g5 Chunks and GPU types.** Training runs as resumable 1-h chunks (docs/training_sample_study.md D5) on
  submit-gpu-express: metric learning on A30 or V100 (FRNN has no sm_61 or sm_75 kernel); acorn's edge classifier on
  A30, V100, RTX 6000 or GTX 1080 Ti; the Canto-L classifier on A30, V100 or RTX 6000 (arm (iii)'s ran out of memory on
  the 11 GB GTX 1080 Ti). The GTX 1080 Ti was dropped from acorn's chunks on 2026-10-15 00:30 UTC, after its first
  chunks, because its chunks take about twice as long and count against the GPU-hour stop (section 8). Stage 0 ran on A30s in one job per stage. The GPU of every chunk is in its run record.
- **D-g6 CPU jobs** exclude the GTX 1080 nodes (and, until 2026-10-15 ~18:53 UTC, the A30 nodes) and run as 1-h pieces on the express tier first (the CPU
  partition is saturated): the union graphs in blocks of at most 320 events on 12 CPUs (first 700 on 24 CPUs; changed on
  2026-10-15 before any block ran, because free memory on the CPU nodes was too fragmented for 48 GB jobs) (`hnl.eval.addendum_g_t3 union-block`;
  each event is built on its own, so the gathered graphs equal a single job's), the menu grids in 4 blocks. While the G1 test arrays of T2 ran (coordinator, 2026-10-15; submit37 drained for an
  upgrade since 2026-10-14 20:27 UTC, so these CPUs were scarce), T3's CPU jobs and training-graph scores were
  submitted at --nice=20 (the first 32 pending ones were set to it by the coordinator), so that G1 ran first. The GTX
  1080 nodes cannot run the union step (probe 6821144 on submit60: worker processes die; Polars needs AVX2) and stay
  excluded for CPU work. Two more nice-20 periods (coordinator), while G1's last jobs waited on submit77-1:
  2026-10-15 ~17:50-18:15 UTC, until g_summary_test 6820173 started; and from 18:38 UTC until g_readouts_test 6820174
  started (period ended at 19:37:35 UTC). The user lifted the A30 restriction for CPU jobs on 2026-10-15 at ~18:53 UTC (relayed
  by the coordinator; it supersedes a narrower lane exception that was never put in place): from then on T3's CPU
  jobs exclude only submit00-01 and the GTX 1080 nodes (Polars needs AVX2), and the coordinator removed the A30 nodes
  from the exclude lists of the 70 pending jobs (`logs/a30_unexclude.log`). Where a job runs does not change its
  output: every event is built, scored and evaluated on its own, deterministically. After each split's union blocks were gathered into ltf_union_graphs/<split>set.tar (exact membership
  checked by `union-gather`), the 195 block tars (3 seeds x 65 blocks) were removed on 2026-10-16 ~07:30 UTC after a
  size check against the gathered tars (they differ only by the per-tar end-of-archive padding), freeing 20.27 GB
  (20,268,738,560 bytes); the blocks' reports, provenance and done markers stay, each marked `pruned` (the T2 chain's
  gather step removes its union blocks the same way).
- **D-g7 Training-graph scores for learned merging on a GPU** (`score_split_sample.sbatch` with ACCEL=gpu): the CPU
  was the fallback when no GPU was free (`hnl.train.acorn_runner.infer`); float32 numerics differ at the 1e-6 level.

## 7. Directories and code

- acorn (ttbar+G): `runs/baseline_g1/armg_s<S>`; Canto-L pieces: `runs/baseline_g1_ts/armg_s<S>` (`ml`, `ml_graphs`
  linked to the acorn run); Canto-L classifier: `runs/stage1_g1/l1tl/armg_s<S>`; driver state
  `runs/_addendum_g_t3` (`runs` = `/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs`).
- Code: `hnl.train.arm_g` (event lists, staging), `hnl.eval.addendum_g_t3` (scan, menus, selection),
  `slurm/addendum_g_t3/` (stage scripts and `driver.sh`, a login-node submitter).

## 8. Cost and stops

- GPU (A30-equivalent): metric learning about 5 h, acorn classifier about 9 h, Canto-L classifier about 30 h, inference
  about 1 h per seed: about 135-150 GPU-hours for both models. The driver submits no new GPU job once T3's GPU jobs
  reach 230 hours.
- CPU: union graphs about 200-250 CPU-hours per seed, GBDT, train scores, merging and menus about 100 per run: about
  1,300-1,500 CPU-hours.
- Storage on /ceph: about 15 GB per seed. Files: a few hundred per run.
- Stops: /ceph free space near 15 GB, `hnl_lrt` near 80k files, any item of the stop-and-ask list.
