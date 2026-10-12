# Pre-registration (brief §7)

Committed on 2026-10-03, with its git hash recorded in RESULTS.md, before the first evaluation of anything on a test
split or on the external 15 GeV / 100 mm set, including the Stage-0 baselines. The primary claim (section 6) was
approved by the user on 2026-10-03. All results are "within this simulation (ACTS generic detector, no pileup)".

## 1. Definitions

All metric definitions are those of `docs/definitions.md` at commit `89eb354`: targets
(pT > 1 GeV, |eta| < 4, >= 5 hits; table values), categories (HNL daughter / W muon / other), double-majority
matching, efficiency, track fake rate (tracks with >= 5 hits), duplicates (single majority), pair efficiency, merge
rate, true edges (npz chain edges), edge metrics in unique unordered pairs.

## 1b. Stage-0 baselines (brief §5f; DECISIONS.md 2026-10-03)

- Pipeline: acorn metric learning on node features (r, phi, z) only, radius graph with (r, knn) chosen on val for
  >= 99 % true-edge efficiency with the FRNN cap not binding (< 5 % of nodes; target 0), optional filter (used only if
  the val graph size requires it, decided before any test evaluation), InteractionGNN (hidden 128, 8 iterations,
  Example_3 recipe), connected components. No truth-based node cut anywhere.
- Natural weighting (every true edge positive) in metric learning and the GNN.
- Arms, 3 seeds each, every seed retraining the whole pipeline: (i) ttbar train (7,999 events); (ii) HNL joint train
  split (15,999 events); (iii) 7,999 ttbar + 7,999 HNL train events (hnl10_100 index < 5000, hnl15_200 index >= 5000).
- Checkpoints: metric learning best val f1, GNN best val loss, each on the arm's own val mix.

## 2. Bins

- Truth-hit |d0| [mm]: [0, 0.1), [0.1, 1), [1, 3), [3, 10), [10, 30), [30, 100), [100, 300), [300, inf) overflow.
- HNL-daughter r_v [mm]: [0, 10), [10, 30), [30, 100), [100, 200), [200, 300), >= 300, and "no vertex".
- Also reported: first-hit radius, number of pixel hits, pT, opening angle of the daughter pair (binning fixed
  in the code before the first test evaluation).

## 3. Datasets

- Selection and tuning: val splits only (`splits/*_val.txt`).
- Test sets, each evaluated once per final model: `hnl10_100_test`, `hnl15_200_test` (joint split; events with the
  same index are resampled together), `ttbar_test`, and the external `hnl15_100_external` (4,999 events).

## 4. Working points (fixed on val, applied unchanged to test)

- Baseline working point: for each baseline arm, the edge-score cut s* of connected components is the value on the
  grid {0.05, 0.10, ..., 0.95} that maximises the DM efficiency of all targets on the pooled HNL val splits.
- Fixed track fake rate FR*: the fake rate of the reference baseline (section 6) at s* on the pooled HNL val
  splits. Every other model is compared at its own cut chosen on the same val events so that its fake rate is the
  largest value <= FR* on a cut grid of step 0.01.
- Fixed edge false-positive rate e*: the reference baseline's edge FPR (fake edges kept / fake edges in the graph)
  at s* on val; edge-level comparisons use each model's cut that gives e*.

## 5. Statistics

- Event bootstrap, B = 500, with one fixed list of resampled event indices per test set (seed 12345), shared by every
  model and every seed so that differences are paired; joint-split events of the same index resampled together.
- A model's number is the mean over its 3 training seeds; total uncertainty of a 3-seed mean
  sigma^2 = sigma^2_boot(mean) + s^2_seeds / 3. For a difference between two models, the bootstrap term is computed on
  the paired difference and both models' seed terms are added.
- Selection rule: architecture, hyperparameters, checkpoint (section 1b) and working point are chosen on val only;
  no test number is ever used to choose anything. Stage-1 models are selected on val by the primary population's
  efficiency at FR* subject to the prompt guard of section 6.

## 6. Primary claim (approved by the user, 2026-10-03)

- Reference baseline: the acorn arm (i), (ii) or (iii) with the highest val efficiency for the population below at
  FR* (the strongest baseline, so the comparison is conservative).
- Population: HNL-daughter targets with truth |d0| >= 10 mm OR r_v >= 100 mm, pooled over `hnl10_100_test`,
  `hnl15_200_test` and `hnl15_100_external`; each set also reported separately.
- Success requires all of:
  1. the new model's inefficiency is at most half the baseline's: (1 - eps_new) <= 0.5 (1 - eps_base)
     (point estimates of the 3-seed means);
  2. the improvement eps_new - eps_base is positive at >= 3 sigma (section 5);
  3. prompt guard: W-muon efficiency and ttbar prompt-target efficiency (|d0| < 1 mm) drop by no more than
     0.2 percentage points (point estimates) at the same working point; the fake rate is equal by construction.
- Also reported (not part of the success criterion): the significance of the halving itself,
  0.5 (1 - eps_base) - (1 - eps_new) in units of its uncertainty.

## 6b. Secondary claim S1: training on ttbar only (added 2026-10-03, before any test or external-set evaluation)

Added at the user's direction (research focus: a pipeline trained on prompt-dominated ttbar alone that stays robust on
displaced tracks); the user delegated the wording to the agent. The primary claim (section 6) is unchanged.

- Candidate pipelines: every learned stage (metric-learning graph, edge classifier, track builder, any learned
  vertexing) is trained on the ttbar training split only (7,999 events); no LLP sample enters training. Selection
  among candidates, checkpoints and working points follow sections 4-5 on val (pooled HNL val, FR* rule); the
  working point chosen on ttbar val alone is reported next to it (not part of the claim).
- Population, test sets and statistics as in sections 5-6: primary population pooled over `hnl10_100_test`,
  `hnl15_200_test` and `hnl15_100_external`, 3-seed means, sigma^2 = sigma^2_boot + s^2_seeds / 3 per model, paired
  bootstrap for differences.
- S1 holds if all of:
  1. the ttbar-only pipeline beats acorn trained on ttbar (arm (i), at its own FR* cut, flagged if unreachable) by
     >= 3 sigma: eps_S1 - eps_(i) >= 3 sigma_diff;
  2. it is non-inferior to the reference baseline of section 6 (acorn selected by that rule, trained on its own arm,
     possibly with HNL samples): eps_S1 - eps_ref - 2 sigma_diff > -1.0 percentage point (one-sided 2 sigma,
     margin 1 point);
  3. prompt guard against arm (i): W-muon and ttbar prompt-target efficiency not lower than arm (i)'s by more than
     0.2 percentage points (point estimates).
- Also reported (not part of S1): efficiency vs truth |d0| and r_v for both, the edge TPR vs |d0| of the classifiers
  (`hnl.eval.edge_tpr`), and the inner-hit truncation probe (`hnl.eval.truncation_probe`).

## 6c. Secondary claim S1b: production-point invariance of the ttbar-only pipeline (added 2026-10-03, before any test or external-set evaluation)

Added by the agent under the user's delegation (section 6b); written after the val diagnostics of RESULTS.md (edge TPR
vs |d0|, the inner-hit truncation probe, the r-z decomposition of the design panel) had been seen. Sections 6 and 6b
are unchanged.

- P is the pipeline selected for S1 (section 6b), 3 seeds; the comparator is acorn arm (i) (3 seeds). Test sets,
  event bootstrap, seed terms and paired differences as in section 5.
- Edge level: each pipeline's classifier on its own graph, at its cut giving the edge false-positive rate e*
  (section 4). TPR = kept / in-graph chain-true edges of HNL-daughter targets (unique unordered pairs, as
  `hnl.eval.edge_tpr`). dD = TPR(truth |d0| in [100, 300) mm) - TPR(truth |d0| in [0.1, 1) mm), pooled over
  `hnl10_100_test` and `hnl15_200_test`.
- LLP-blind working point of P: every working-point parameter of P's track building chosen by the section-4 rule
  applied to ttbar val instead of pooled HNL val (the grid value with the highest DM efficiency of all ttbar-val
  targets). No HNL event enters P's training, checkpoints or this working point. Arm (i) stays at its section-4 cut
  s*, which was chosen on HNL val (conservative for P).
- S1b holds if both:
  1. dD(P) - dD(i) > 0 at >= 2 sigma, and dD(P) >= -2.0 percentage points (point estimate of the 3-seed mean);
  2. at P's LLP-blind working point, P's primary-population efficiency (section 6 population and test sets) exceeds
     arm (i)'s by >= 3 sigma, with P's pooled-HNL-test track fake rate not above arm (i)'s and its W-muon and ttbar
     prompt-target efficiencies not lower than arm (i)'s by more than 0.2 percentage points (point estimates).
- Also reported (not part of S1b): the inner-hit truncation sensitivity dT (`hnl.eval.truncation_probe`, depth 4
  vs 0, prompt particles of the HNL test events) for P and arms (i), (ii), (iii) (not a criterion: a pipeline trained
  with inner-hit truncation augmentation would make it circular); efficiency in the r-z classes |z0 - z0(W muon)| /
  r_v < 0.5, 0.5-1 and >= 1; dD of arms (ii) and (iii); and the transfer gap of P's architecture trained on the HNL
  train split, if trained.

## 7. Reporting rules

- Every final model is evaluated on each test set once; its test numbers are reported whatever they are.
- Prompt performance (W muon, ttbar) is always reported next to displaced results.
- The comparison with the paper (0.990 / 0.916 / 0.9561) uses arm (ii) on `hnl15_100_external` with the paper's
  definitions, notes the one dropped event (4,999 vs 5,000) and that the pipelines differ; a gap above about
  2 points is investigated and reported, never removed by changing definitions.
