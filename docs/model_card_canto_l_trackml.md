# Model card: Canto-L on TrackML (run canto_l_v1/seed0_x1)

Status 2026-10-12 04:50 UTC: the selection is frozen and the one-time test read-out is running. Filled in: the test
set in all three conditions and T2 C-1GeV and C-nonoise. Pending: T2 C-full (scored with the memory-bounded forward,
array 6773056; grid 6771390). Test numbers were read once, after the freeze; nothing was
chosen on them. The comparisons with published numbers (section 5.2) use each reference's own evaluation code,
recomputed after the freeze from the frozen pipeline's test tracks (DECISIONS 2026-10-11).

## 1. Model

- **Name.** Canto-L, TrackML instance, run `canto_l_v1/seed0_x1`. "X1" means the metric learning also reads four
  cluster features (DECISIONS 2026-10-06). One training seed (0).
- **Developer.** Jinbo Zhang, undergraduate thesis project on graph-neural-network tracking of displaced particles.
- **Family.** Canto-B, Canto-H and Canto-L were built for displaced heavy-neutral-lepton tracks on the ACTS generic
  detector (docs/canto_facts.md). In Canto-L the graph, the edge classifier and the fragment merging are all learned.
  On TrackML every learned piece is retrained. Architectures, losses and selection rules are unchanged, except for the
  four cluster features (X1) and one feature of the triplet support, log_pt_turn, which takes the measured field map
  instead of 2 T (docs/trackml_plan.md section 6.5).
- **Frozen.** Selection made at 5089c9a, committed in 5098f25 (configs/trackml/selection_v1.json, sha256
  a2b30f4f...3524). Test chain launched once, 2026-10-11 11:20 UTC.

### 1.1 Pipeline

| Stage | Component | Size and setting | Training |
|---|---|---|---|
| Embedding | MLP, 4 hidden layers of 1024 (tanh), 12-D output. Inputs: r/1000 mm, phi/3.14 (phi in rad), z/1000 mm, and the logs of the cluster ncells, charge, span0, span1 | 3,169,292 parameters | training events, 30,000 steps (30 virtual epochs); best epoch 23 by valmon F1; 3.1 GPU-h on a V100 |
| Graph | radius graph at r* = 0.1157 (FRNN, at most 300 neighbours), united with the pairs up to r_w = 1.2 r* that the triplet support keeps at (tau, k) = (0.05, 3) | 15.7 pairs per hit; graph ceiling 99.43 % of targets (50 valmon events) | chosen on val by the plan's rule (gate G1: ceiling >= 99.0 %) |
| Triplet support | gradient-boosted trees (scikit-learn HistGradientBoosting, 300 iterations, 63 leaves) on 16 invariant triplet features | | 100 training events (90 for training, 10 held out), 23.50 M triplets (21.15 M training rows); hold-out AUC 0.9994; 10.0 h on 16 CPU cores |
| Edge classifier | SymmetricTripletIGNN (L1): hidden 128, 8 iterations, triplet top-k 4, region one-hot, inner-hit truncation augmentation. Inference: one forward pass over the whole event, on CPU | 347,777 parameters | 120,000 training windows (30 virtual epochs of 4,000); best epoch 27 by val loss (0.003794); 45.6 GPU-h on A30 |
| Track building | same-layer Mutex Watershed (attraction threshold 0.2), then learned fragment merging: gradient-boosted trees (400 iterations, 63 leaves, 14 features) give the probability p that two fragments belong to one particle; fragments merge greedily in decreasing p while p > tau_merge and the two share at most one layer | tau_merge = 0.3 at WP-A | 200 merge_train events (180 for training, 20 held out), 20.67 M fragment pairs (1.18 % positive); hold-out AUC 0.99927; 0.56 h on 60 CPU cores |

### 1.2 Working points (chosen on val, frozen)

| Working point | Rule on val | Configuration |
|---|---|---|
| WP-A (primary) | highest DM efficiency of targets with track fake rate <= 0.029368 (the committed HNL value FR*, not re-tuned on TrackML); configurations within one binomial standard error of the best are tied and the lowest fake rate wins, per builder family and then across families; C-full | Mutex Watershed + learned merging, tau_merge = 0.3 |
| WP-S | highest TrackML score, C-full | same builder, tau_merge = 0.1 |
| WP-R-P1 | Ju et al.: highest eps_tech (pT > 0.5 GeV) with purity >= 58.3 %, C-nonoise (frozen; Ju et al. later found to be accuracy-phase and not compared, section 5.2) | same builder, tau_merge = 0.5 |
| WP-R-P2 | Liu et al.: highest Eff with FR <= 55.58 %, C-full | same builder, tau_merge = 0.5 |
| WP-R-P3 | Chan et al.: highest efficiency with fake rate <= 0.19 %, C-full | connected components at edge score 0.8 |
| WP-R-P4 | Calafiura et al.: highest efficiency with fake rate <= 0.12 %, C-1GeV | walkthrough + helix merging (0.1, 0.5, 1; HNL merge cuts) |

The WP-R rules were applied with our readings of the published definitions, as frozen; section 5.2 evaluates the
frozen configurations with the references' own code.

## 2. Intended use

- Research on track reconstruction with TrackML (pileup <mu> = 200, TrackML detector), and the generalisation study of
  a model built for displaced tracks.
- Out of scope:
  - production or time-limited reconstruction: the one-shot edge classifier peaks at 50-60 GB per throughput-phase
    event block and takes about a minute per event on 48 CPU cores (the memory-bounded forward needs about 10 GB),
    and track building takes minutes per event;
  - data with another digitisation than the throughput phase without retraining (section 6);
  - claims in the long-lived-particle regime: the throughput phase has about 0.25 targets per event with
    |d0| >= 100 mm (352 in the 1,399 training events; 69 in the 300 test events).

## 3. Data

- **Training, validation, test.** TrackML throughput phase (Zenodo 10.5281/zenodo.4730157, CC-BY-4.0),
  training_part01 and training_part02 (1,850 events, ids 21000-22999). Split configs/trackml/split_tp_v1.json (sha256
  36875d20...), fixed by rule when part01 was first streamed, before any model saw an event:
  - test 300 (ids 21102-21996);
  - val 150 (valmon: its first 50);
  - train 1,400, including dev 100 and the subsets calib 200, gbdt 100, merge_train 200. Event 22500 has an empty
    file in the archive and is dropped, so 1,399 training events are used.
- **T2 (cross-phase test).** 300 accuracy-phase events (Kaggle train_1, the test set of configs/trackml/split_v1.json,
  ids 1112-2817), evaluated once with the frozen pipeline.
- **Conditions** (docs/trackml_plan.md section 3.5):
  - C-full: every hit, noise included. Primary condition; every choice is made here.
  - C-nonoise: noise hits removed using truth.
  - C-1GeV: only the hits of particles with pT > 1 GeV, using truth (the "TrackML-1GeV" of the references).
  Training is always in C-full, and no trained stage uses a truth-based cut. C-nonoise and C-1GeV are truth-filtered.

## 4. Metrics

- **Ours (section 5.1).**
  - Matching: double majority (DM), more than half of the track's hits come from the particle and more than half of
    the particle's hits are on the track.
  - Target: pT > 1 GeV, |eta| < 4, >= 5 hits. Efficiency is the fraction of targets with a DM match.
  - Fake rate: tracks with >= 5 hits that DM-match no particle, over all tracks with >= 5 hits.
  - Duplicate rate: extra single-majority matches per target.
  - Displaced target: |d0| >= 10 mm.
- **TrackML score.** The official trackml-library score with the phase's own hit weights. Not computed in C-1GeV.
- **References' own code (section 5.2).** Faithful ports of the authors' evaluation code, checked against the
  unmodified code on hand-counted and randomised events (src/hnl/trackml/refports; workflow wf_9ddbb9d7-334) and on
  real test events (20 event-configuration rows per code, identical counts; workflow wf_ee104fbf-a85):
  - EggNet (github.com/exatrkx/EggNet e2bd117; Calafiura et al. and Chan et al.): candidates are tracks with >= 3
    hits (their DBSCAN min_samples = 3); a track matches the particle holding more than half of its hits; targets
    are particles with pT >= 1 GeV and >= 3 hits, no eta cut; efficiency = matched targets / targets; duplication =
    (tracks matched to targets - matched targets) / matched targets; fake = unmatched tracks / matched particles.
    Counts pooled over events.
  - HierarchicalGNN (Liu et al.): eval_metrics as in github.com/ryanliu30/Tracking-ML-Exa.TrkX f107781, whose
    strict > 50 % majority reproduces their Truth-CC (HierarchicalGNN a2563af has the same code with >= 50 %);
    our "no track" hits (label -1) are left out of the candidate graph, as the authors' callers leave unclustered
    hits out; candidates with >= 3 hits; both-sided majority, each particle's best candidate; pairs with
    <= 2 shared hits or matched to noise dropped; reconstructable = pT > 1 GeV and >= 5 hits; Eff = matched
    reconstructable / reconstructable; FR = 1 - matched reconstructable / (candidates - dropped pairs - matched
    non-reconstructable). Per-event ratios averaged with the event's hit count as weight, as published.
- **Errors.** Event bootstrap, B = 500, one fixed resampling (seed 12345). One training seed, so no seed term.

## 5. Results

### 5.1 WP-A, our metrics

Percent, except the TrackML score. Val has 150 events, test and T2 have 300. Val numbers carry no bootstrap error.

| Condition | Set | Target efficiency | Fake rate | Duplicate rate | Displaced efficiency | TrackML score |
|---|---|---|---|---|---|---|
| C-full | val | 99.64 | 1.70 | 0.03 | 86.8 | 0.9662 |
| C-full | test | 99.65 +- 0.01 | 1.68 +- 0.01 | 0.021 +- 0.002 | 88.2 +- 0.5 | 0.9660 +- 0.0001 |
| C-full | T2 | pending | | | | |
| C-nonoise | val | 99.64 | 1.30 | 0.02 | 87.0 | 0.9678 |
| C-nonoise | test | 99.66 +- 0.01 | 1.29 +- 0.01 | 0.015 +- 0.002 | 88.5 +- 0.5 | 0.9676 +- 0.0001 |
| C-nonoise | T2 | 94.67 +- 0.05 | 7.94 +- 0.05 | 0.43 +- 0.01 | 86.6 +- 0.6 | 0.6135 +- 0.0005 |
| C-1GeV | val | 99.75 | 0.07 | 0.01 | 88.2 | - |
| C-1GeV | test | 99.77 +- 0.01 | 0.055 +- 0.004 | 0.009 +- 0.001 | 89.0 +- 0.5 | - |
| C-1GeV | T2 | 95.98 +- 0.04 | 0.80 +- 0.02 | 0.033 +- 0.003 | 87.8 +- 0.6 | - |

Far-displaced targets (|d0| >= 100 mm) on test: 31 of 69 found in C-full (45 +- 6 %), 29 of 69 in C-nonoise
(42 +- 7 %) and 33 of 69 in C-1GeV (48 +- 6 %); on T2, 29 of 63 in C-nonoise and 30 of 63 in C-1GeV. WP-S, the
val-score optimum, scores 0.9663 +- 0.0001 on test C-full.

The TrackML score weights differ between the phases (results/trackml_score_weights_v1.txt, `python -m
hnl.trackml.score_weights`; truth weights only): particles produced in the detector simulation (barcode generation
>= 1) carry 9.1 % of the weight in each of three T2 events and none in three throughput test events, where they hold
7-8 % of the hits. The T2 score therefore also counts particles that carry no weight in the test score.

### 5.2 Comparison with published numbers (test, each reference's own code)

Only references that used the throughput phase, as our test does, are compared. The phase was established from the
authors' code and released data (DECISIONS 2026-10-11): Liu et al., Chan et al. and Calafiura et al. read the
group's CodaLab tree of throughput events (event000021000 and up). Ju et al. (Exa.TrkX) evaluated on accuracy-phase
events (their Fig. 9 data are events 1000-1499) and is not compared.

Percent. Canto-L at WP-A unless stated; errors from the event bootstrap; published values and uncertainties as
quoted. A difference below 2 sigma is not resolved.

**TrackML-1GeV, EggNet code (Calafiura et al. 2024, Table 1).**

| Model | Efficiency | Duplication | Fake rate |
|---|---|---|---|
| EC | 98.98 +- 0.09 | 4.21 +- 0.11 | 0.12 |
| EggNet | 99.56 +- 0.03 | 1.29 +- 0.02 | 0.06 |
| Canto-L, WP-A | 99.56 +- 0.01 | 0.13 +- 0.01 | 0.008 +- 0.001 |
| Canto-L, WP-R-P4 | 99.67 +- 0.01 | 0.77 +- 0.02 | 0.039 +- 0.003 |

At WP-A the efficiency is the same as EggNet's within the errors (not resolved), the duplication and fake rate are
lower. WP-R-P4, chosen on val for EC's fake rate (0.12 %), is higher in efficiency by 3.4 sigma and lower in
duplication and fake rate.

**TrackML-1GeV, HierarchicalGNN code (Liu et al., Table 3).**

| Model | Eff | FR |
|---|---|---|
| BC-HGNN | 98.55 | 1.23 |
| Canto-L, WP-A | 99.77 +- 0.01 | 0.26 +- 0.01 |

**Full TrackML, EggNet code (Chan et al. 2025).**

| Model | Efficiency | Duplication | Fake rate |
|---|---|---|---|
| EggNet | 95.87 +- 0.03 | 2.76 +- 0.02 | 0.19 +- 0.01 |
| Canto-L, WP-A | 99.41 +- 0.01 | 0.17 +- 0.01 | 2.40 +- 0.01 |
| Canto-L, WP-R-P3 | 99.37 +- 0.01 | 1.70 +- 0.04 | 0.173 +- 0.003 |

At WP-A the efficiency is higher and the duplication lower, and the fake rate is 12.6 times EggNet's. WP-R-P3,
chosen on val for Chan et al.'s fake rate (connected components at 0.8), has a higher efficiency, a lower duplication
and a fake rate equal to EggNet's within the errors (1.7 sigma, not resolved).

**Full TrackML, HierarchicalGNN code (Liu et al., Table 2).**

| Model | Eff | FR |
|---|---|---|
| EC-GNN | 96.35 | 55.58 |
| BC-HGNN | 97.86 | 36.71 |
| Canto-L, WP-A | 99.65 +- 0.01 | 34.94 +- 0.11 |
| Canto-L, WP-R-P2 | 99.66 +- 0.01 | 35.46 +- 0.11 |

Liu et al. quote no uncertainties, and state that their FR is a model-to-model comparison, not final track-building
performance (footnote 2, p. 6); Canto-L's FR is after its final builder.

Our earlier readings of the published definitions (the test chain's comparison_<condition>.csv) are superseded for
quoting: they used our target population and the paper texts, which differ from the authors' code (EggNet counts
particles with >= 3 hits and no eta cut, and divides duplicates and fakes differently; Liu et al. count only
>= 3-hit candidates and drop matched non-reconstructable ones from the FR denominator). They are kept in the reports.

### 5.3 Context: TrackML challenge results (not a comparison)

- Accuracy phase (Kaggle, 2018; score on 125 hidden test events, noise included): Top Quarks 0.92182, outrunner
  0.90302, S. Gorbunov 0.8935 (Amrouche et al., arXiv:1904.06778).
- Throughput phase (CodaLab, 2018-19; 2 CPU cores, 4 GB): Mikado (S. Gorbunov) accuracy 94.4 % at 0.56 s per
  event, FASTrack (D. Emeliyanov) 94.4 % at 1.11 s, cloudkitchen (M. Kunze) 92.8 % at 7.28 s (Amrouche et al.,
  arXiv:2105.01160).
- Canto-L's test C-full score, 0.9660 +- 0.0001, uses the same weights as the throughput accuracy score, but other
  events, no time limit, a GPU and up to 60 GB of memory per event. The numbers are context only.

## 6. Limitations

- **Digitisation and phase.** The two TrackML phases have different digitisation: pixel barrel mean cluster size 7.4
  cells (throughput) against 13.8 (accuracy), strip 2.6 against 4.1 (DECISIONS 2026-10-06, recorded before any X1
  result). The embedding reads these cluster features, so T2 tests them across the digitisation change as well as
  across the phase. In T2 the graphs are denser (C-nonoise: 1.90 M against 1.20 M edges per event over all 300
  events; the sum of squared hit degrees, which sets the triplet count, about three times larger in sampled events),
  and in C-1GeV (section 5.1 metrics) the target efficiency falls from 99.77 % to 95.98 % and the fake rate rises
  from 0.055 % to 0.80 %. In C-nonoise the target efficiency falls from 99.66 % to 94.67 %, the fake rate rises from
  1.29 % to 7.94 % and the TrackML score falls from 0.9676 to 0.6135; the T2 score weights also count the particles
  produced in the simulation (section 5.1), so the two scores are not on the same footing.
  Both phases were converted by the same code (convert.py sha256 ea286e65...). None of the compared studies
  reports results on both phases, so this robustness cannot be ranked.
- **One training seed.**
- **Published comparisons.**
  - Computed on our test events with the references' own code, not on their events.
  - Liu et al.'s code takes a particle's pT as the minimum hit-level truth pT; our test tables hold the initial pT.
    Their released TrackML-1GeV events cut hits by hit-level pT; ours keep the hits of particles with pT > 1 GeV.
  - Liu et al. quote no uncertainties.
  - The WP-R points were chosen on val with our readings of the published definitions.
- **Memory.** The one-shot whole-event CPU inference peaks at 50-60 GB per 50-event block on test C-full events
  (Slurm MaxRSS 48.7-58.4 M KiB). Accuracy-phase events are projected to need about four times more (the triplet count);
  block 0 of T2 C-nonoise was killed at 96 GiB. The memory-bounded forward (hnl.trackml.chunked) gives the same scores
  to the run-to-run level of the one-shot forward on val and needs about 10 GB; T2 C-nonoise and C-full use it.
  Sector inference with halos did not reproduce the whole-event scores (gate G2).
- **Reproducibility.** The merging builders are not bit-identical across nodes. Between two runs over the 150 val
  events, no summary metric moved by more than 1.2e-5; the post-freeze rebuild of the test tracks differs from the
  test chain in 5 of 1,200 event-configuration rows (C-full) and 1 of 600 (C-1GeV), by a few tracks (DECISIONS
  2026-10-11).
- **Far-displaced tracks.** TrackML has few of them; the |d0| >= 100 mm bin is reported, not claimed.
- **Collision reuse.** Throughput-phase events reuse generator-level collisions (shifted and simulated again):
  76 % of the val collisions also occur in a training event, 140 of the 150 ttbar collisions among them. On val
  (C-full), generation-0 targets whose collision occurs in training are found at 99.809 +- 0.012 %, the others at
  99.815 +- 0.023 % (results/trackml_collision_reuse_v1.txt); simulation-produced particles never recur.

## 7. Compute

- **Training.**
  - Embedding: 3.1 GPU-h (V100).
  - Triplet support: 10.0 h on 16 CPU cores (13.5 CPU-h used), mostly building the training rows.
  - Edge classifier: 45.6 GPU-h (A30).
  - Fragment merging: 0.56 h on 60 CPU cores (16.6 CPU-h used).
- **Inference per test event (C-full).**
  - Union graph: about 37 s on one GPU (A30 or V100).
  - Edge classifier: median 73 s on 48 CPU cores (one-shot forward), with a peak of 50-60 GB per 50-event block;
    75-160 s on 16 cores with the memory-bounded forward on val events.
  - Track building: minutes in Python on CPU; not optimised.

## 8. Provenance

- Code: repository hnl-displaced. Plan: docs/trackml_plan.md, fixed before any TrackML result. Records:
  DECISIONS.md, RESULTS.md. Same-convention recount: src/hnl/trackml/labels.py (track rebuild),
  src/hnl/trackml/refports (ports), src/hnl/trackml/refmetrics.py; outputs in $RUN/test_post.
- Files in /ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1 (sha256 prefixes from
  the frozen selection; `python -m hnl.trackml.freeze verify` checks the full hashes):

  | File | sha256 prefix |
  |---|---|
  | embedding: ml/artifacts/best-epoch=23-f1=0.224451.ckpt | 87d07e0357b28481 |
  | triplet support: support/model.joblib | d39d1e7e1b11d660 |
  | graph operating point: graph/graph_choice.json | 08897b2ee1b73076 |
  | edge classifier: gnn/artifacts/best-epoch=27-val_loss=0.003794.ckpt | 306d6fdfc83938e8 |
  | fragment merging: lmerge/model.joblib | c99beec1c610aac5 |
  | environment: env/lock.txt (repository) | d60265adb0d93a83 |

- Licence: the TrackML throughput data are CC-BY-4.0. The model's licence is not decided.
