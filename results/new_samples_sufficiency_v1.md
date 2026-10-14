# Are the generated samples sufficient? Power, per-bin counts and referee objections (v1)

Written 2026-10-14, 13:30 EDT, while the G1 test read (T2) is on hold (DECISIONS 2026-10-14, 996c734). No model was
run for this note and no model output on any test split was read. It combines three assessments, with their scripts
and outputs under `/work/submit/jinboz1/hnl/samples_v2/`: a power study (`power/`), per-bin truth counts (`bins/`) and
a referee-style reading (`referee/`). It also adds the spot checks of section 8. The model inputs are the frozen models
on generic_iso val (`results/addendum_g_val_v1.md`, `results/addendum_g_val_sanity_v1.md`) and the published HNL val
read-outs. T3 is still in metric learning (driver state `runs/_addendum_g_t3`), so no retrained model has a val number
yet. Section 6.3 holds proposed deviation text, which binds only once it is committed to RESULTS.md.

## 1. Question

The user asked whether the events generated today are enough to make the paper convincing. In the terms of
addendum G (`docs/preregistration_new_sample.md`) this note asks four things:

1. Can the test split decide claims G1, G2 and G3 as preregistered?
2. Which bins of section 7, and which of the paper's statements, have enough targets?
3. What would a referee still object to?
4. Should the isolated component get more events before T2?

Short answer. With the current events G1 is decided, and G2 very probably is. G3 has 80 % power only if the true
difference is at least 1.75 points, whereas the plausible range is 1.1 to 3.7 points. Tripling the isolated component
lowers that threshold to 1.30 points, and we recommend doing so (section 6). Most of the cost is about 260 CPU-hours of
inference on the added test events. The far, forward and electron bins can carry absolute efficiencies but not model
rankings. The dense component does not test collimated vertices.

## 2. Current counts

| sample | events | train / val / test events | population | targets on test (val) |
|---|---|---|---|---|
| generic_iso_v1 | 8,000 | 4,810 / 806 / 2,384 | P_G | 2,914 (956); 5,799 on train |
| generic_dense_v1 | 8,000 | 4,780 / 835 / 2,385 | P_G definition on the dense component | 7,782 (2,747) |
| hss15_100_v1 | 1,500 | all evaluation | P_H | 8,294 |
| hss40_300_v1 | 1,500 | all evaluation | P_H | 9,719 |
| HNL test sets (reference) | 6,999 | - | HNL daughters, per seed | 5,013 (val 1,331) |

- P_G holds 1.21 targets per event (1.22 on test).
- The T3 training list (`splits/generic_train_g7999.txt`) has 7,999 events, 4,005 iso and 3,994 dense, all with event
  number at most 6,677.
- Targets with |d0| >= 300 mm on test: 96 in generic_iso and 163 in generic_dense. The HNL test sets have 36 per seed
  and HNL val has 19.
- Source: `results/addendum_g_t1_v1.md` and RESULTS.md (T1). All ten reference counts are reproduced in
  `bins/checks.txt`.

## 3. Power of G1, G2 and G3, now and with more isolated events

**Method** (`power/power_v2.py`, job 6812318).

- The uncertainty is that of addendum G section 6: sigma^2 = sigma_boot^2 + s_A^2/3 + s_B^2/3.
- The bootstrap term scales as sigma_boot^2 = c (L_A + L_B) / N, where L is the inefficiency and N the number of P_G
  test targets. The constant c is calibrated on the stored val bootstrap tables: 0.33 for the G1 pair, 0.375 for G2,
  and 0.42 for G3 after converting to three-seed means.
- Seed floors:
  - frozen models, measured on generic_iso val: Canto-L (ttbar) 0.06 and acorn (HNL)+LM 0.12;
  - retrained models, from their in-distribution HNL analogues: Canto-L 0.1-0.2 and acorn (ttbar+HNL) 0.27-0.35,
    with a test seed spread of 0.38;
  - G3 uses floors of 0.2 / 0.4 as the central case and 0.4 / 0.7 as the pessimistic one.
- The decision rules are simulated with 4 x 10^5 draws, including the noise in s estimated from three seeds. The test
  size is N = 2,914 x E / 8,000 for E isolated events.
- The "x1.3" variant inflates sigma_boot by 30 %. That is the largest underprediction seen when the same val scaling
  is applied to the HNL test sets (predicted 0.14 / 0.16 / 0.20, published 0.17 / 0.21 / 0.21).

| quantity (nominal; x1.3 in brackets) | 8,000 events (now) | 16,000 | 24,000 | 32,000 | no limit |
|---|---|---|---|---|---|
| P_G targets on test | 2,914 | 5,828 | 8,742 | 11,656 | - |
| **G1** sigma [points] | 0.34 (0.43) | 0.24 (0.31) | 0.20 (0.26) | 0.18 (0.23) | 0.08 |
| G1 P(holds) at the val difference +2.79 | 1.00 (1.00) | 1.00 | 1.00 | 1.00 | 1.00 |
| G1 P(holds) if the true difference were 0 | 0.79 (0.58) | 0.97 (0.87) | 1.00 (0.96) | 1.00 (0.99) | 1.00 |
| **G2** difference for 80 % power, acorn floor 0.4 | 0.13 (0.34) | -0.08 (0.05) | -0.15 (-0.06) | -0.19 (-0.12) | -0.32 |
| G2 difference for 80 % power, acorn floor 1.0 | 0.93 (1.07) | 0.81 | 0.76 | 0.74 | 0.68 |
| G2 P(holds) at +0.85, floor 0.4 / 1.0 | 0.99 / 0.77 | 1.00 / 0.81 | 1.00 / 0.83 | 1.00 / 0.83 | 1.00 / 0.85 |
| **G3** sigma at a difference of 1.35, floors 0.2 / 0.4 | 0.42 | 0.35 | 0.32 | 0.31 | 0.26 |
| G3 difference for 50 % / 80 % power | 1.32 / 1.75 | 1.05 / 1.42 | 0.94 / 1.30 | 0.89 / 1.24 | 0.70 / 1.04 |
| G3 difference for 80 % power, x1.3 | 2.08 | 1.61 | 1.44 | 1.34 | 1.04 |
| G3 difference for 80 % power, floors 0.4 / 0.7 | 2.37 | 2.13 | 2.04 | 2.00 | 1.87 |
| G3 difference for 80 % power, floors 0.2 / 0.2 | 1.55 | 1.16 | 1.01 | 0.93 | 0.66 |
| G3 P(holds) at 0.53 / 1.10 / 1.35 / 2.0 points | 0.05 / 0.33 / 0.53 / 0.90 | 0.11 / 0.55 / 0.75 / 0.98 | 0.15 / 0.64 / 0.83 / 0.99 | 0.18 / 0.70 / 0.86 / 0.99 | 0.33 / 0.84 / 0.94 / 1.00 |

**G1** (frozen Canto-L (ttbar) against frozen acorn (HNL)+LM) is decided by the current events.

- On generic_iso val the difference is +2.79 +- 0.58 points on 956 targets.
- On the 2,914 test targets the projected sigma is 0.34: bootstrap 0.33 (0.572 x sqrt(956/2914)) plus a seed term of
  0.08. The same value appears in `results/addendum_g_val_sanity_v1.md`.
- G1 fails only if the test difference falls below -0.32, which is 3.1 points under the val value.
- The fake-rate guard is not close: Canto-L's rate is 0.85 points lower on val, with a paired sigma of 0.04.
- Adding events does nothing for G1.

**G2** (frozen Canto-L (ttbar) against acorn (ttbar+G)) is very probably decided at 8,000 events.

- The projected difference is +0.85 to +3.2 points, from the two HNL training-gain ratios. The HNL analogue is +1.20 on
  val and +1.71 on test.
- With a floor of 0.4 for acorn (ttbar+G), as acorn (ttbar+HNL) had on HNL, P(holds) is 0.99 at the low end of that
  range.
- More events matter only if the true difference is near zero and the floor is low. If the floor is about 1 point,
  they do not help.
- P(resolved below) is at most 0.03 for any difference of -0.5 or more.

**G3** (Canto-L (ttbar+G) against acorn (ttbar+G)) depends on an effect that is not yet known. There are four
reference values:

- 0.53 points: the HNL val analogue with signal-only training;
- 1.35 points: the HNL val analogue with ttbar+signal training;
- 1.1 to 3.7 points: those HNL training effects carried over to G, whose inefficiencies are 3-4 times larger;
- +3.6 points: both models trained on HNL and read on G val (Canto-L with one seed).

T3 trains on ttbar plus G, which is closest to the ttbar+HNL analogue, so 1.1-2 points is the realistic window.
- In that window P(holds) is 0.33-0.90 with 8,000 events and 0.64-0.99 with 24,000.
- At about 0.5 points no affordable sample decides G3. The seed term alone keeps sigma at 0.26 or more, so P(holds)
  is 0.33 even with unlimited events.
- More seeds do not change this (section 6.5).

## 4. Per-bin adequacy

**How the counts were made.** The counts come from truth only (`bins/per_bin_counts.py`, job 6812387), with the bins
and populations imported from `hnl.eval.addendum_g_summary` (T1 deviation 10).

**Uncertainties.** The table quotes sigma = sqrt(eps(1-eps)/n) at eps = 0.90. This is a lower bound: the seed term adds
to it. In clustered bins the event clustering inflates it by up to 1.36 (98 dense targets in 53 events). In bins with
fewer than 5 expected misses we quote Clopper-Pearson intervals instead.

**Columns.** The "after" column scales the generic_iso test counts by 3, which assumes the same gun and the same hash
rule. Bold marks fewer than 100 targets.

| bin | HNL test, per seed | HNL val | generic_iso test now (sigma) | after the extension (sigma) | generic_dense test | what the counts support |
|---|---|---|---|---|---|---|
| all P_G | 5,013 | 1,331 | 2,914 (0.6) | about 8,700 (0.3) | 7,782 | the claims |
| \|d0\| 100-300 mm | 584 | 167 | 494 (1.3) | about 1,480 (0.8) | 1,132 | absolute and paired |
| \|d0\| >= 300 mm | **36** | **19** | **96** (3.1) | about 290 (1.8) | 163 | absolute efficiency only |
| r_v 300-450 mm | 397 | 167 | 589 (1.2) | about 1,770 (0.7) | 1,861 | yes |
| r_v 450-650 mm | **69** | **32** | 249 (1.9) | about 750 (1.1) | 727 | yes |
| r_v >= 650 mm | **2** | **0** | **29** (5.6) | **about 87** (3.2) | **65** | report merged as r_v >= 450 (278 now) |
| r_first < 50 mm | 449 | **68** | **65** (3.7) | about 195 (2.1) | 146 | descriptive |
| r_first 500-700 mm | 122 | **44** | 260 (1.9) | about 780 (1.1) | 724 | yes |
| r_first >= 700 mm | **6** | **3** | **14** (8.0) | **about 42** (4.6) | **44** | report merged as r_first >= 500 (274 now) |
| \|d0\|/r_first >= 0.7 | 348 | **75** | 226 (2.0) | about 680 (1.2) | 431 | yes |
| \|d0\| >= 300 and \|d0\|/r_first >= 0.7 | **24** | **14** | **56** (4.0) | about 170 (2.3) | 116 | descriptive |
| r_first >= 200, ratio > 0.7, pT > 1.5 GeV (paper's cut) | **67** | **18** | **91** (3.1) | about 270 (1.8) | 178 | yes after the extension |
| \|d0\| < 1 mm | **5 + 45** | - | **6 + 49** | **about 18 + 147** | - | structural: P_G admits them only through r_v >= 100 mm |
| r_v 10-30 mm | - | - | **53** (4.1) | about 160 | - | descriptive |
| smallest pT bin | 428 | 108 | 408 (1.5) | about 1,220 (0.9) | 1,008 | yes |
| \|eta\| > 2.5 | about 570 (11.3 %) | - | **about 88** (3.0 %) | about 265 | - | state the forward coverage |
| electrons | - | - | about 381 (13.1 %) | about 1,140 | - | inclusive and binned |

**Far decays.**
- Above 300 mm the current split gives an absolute efficiency to +-3.1 points at eps = 0.90, against +-5.0 on the HNL
  test sets.
- A model ranking needs a paired sigma, sqrt(f_d/n), where f_d is the share of targets that one model finds and the
  other misses. The paper's val misses give f_d of about 0.105 (6 of 57). With that, n = 96 resolves only gaps of
  6.6 points or more at 2 sigma; n of about 290 resolves gaps of 3.8 points or more. A 3-point gap needs about 470
  targets.
- On G val the far-bin gap has opposite signs in the two components: +13.1 points in iso (75.4 against 62.3,
  n = 38) and -4.7 points in dense (83.3 against 88.0, n = 50).
- The bin therefore stays descriptive at any size we would produce.

**Dense component.**
- In the preregistered Delta R < 0.05 binning, the test counts for 0 / 1 / 2 / 3-4 / >= 5 other targets are
  7,088 / 566 / 98 / 29 / 1. The last three bins hold 128 targets in 71 events.
- More events of this gun cannot fill those bins: 100 targets in the >= 5 bin would need about 800,000 events.
- Two alternative binnings are well populated, both chosen from truth only:
  - same-vertex target multiplicity 1 / 2 / 3 / 4 / 5+: 610 / 952 / 1,467 / 1,439 / 3,314 (680 of the 5+ at
    r_v >= 300 mm);
  - other targets within Delta R < 0.2, for 0 / 1 / 2 / 3-4 / >= 5: 4,317 / 1,493 / 749 / 797 / 426.
- "Dense and far" is not covered: of the 1,295 dense targets with |d0| >= 100 mm, 11 have five or more targets within
  0.2.

**h -> s s (P_H, all 1,500 events of each sample).**
- Inclusive P_H, its hadrons and the V0-tagged daughters (1,167 / 1,265) are well populated.
- Electrons (785 / 758), s -> ee daughters (531 / 513, +-1.1 to 1.3 points) and muons (313 / 329, +-1.7) support
  inclusive numbers only.
- A far statement can come only from hss40_300, which has 130 targets at |d0| >= 300 mm. hss15_100 has 3.

**Thin after the extension.** Four sets of bins stay below 100 targets even with 24,000 events: r_v >= 650 mm,
r_first >= 700 mm, |d0| < 1 mm, and the dense Delta R < 0.05 bins with two or more neighbours. They are reported
merged, next to the preregistered bins.

## 5. Referee objections, most severe first

| # | objection | severity | what we measured | remedy | cost |
|---|---|---|---|---|---|
| 1 | G3, which tests the first aim on G, may be underpowered. On HNL the first aim rests on one seed evaluated after the freeze. | high | Section 3. Canto-L (HNL) against acorn (HNL): 0.96 +- 0.29 on test (one seed), 0.53 +- 0.37 on val. | Extend generic_iso to 24,000 events (section 6). State G3's minimum detectable difference in the paper. Optionally, fresh HNL events at the two trained points, with a claim committed first, give a blind three-seed test (section 6.6). | about 31 CPU-h for production and T1; inference about 260 CPU-h and 4 GPU-h |
| 2 | The averaged efficiency has no physical weight: r_v and eta are flat and the species have equal shares. | high | r_v >= 300 mm: 29.8 % of gun P_G against 11.9 % of HNL daughters. Median pT: 5.1 against 7.9 GeV. Species e / mu / pi / K / p: 13 / 21 / 19 / 24 / 23 %. | Report the per-bin sign of the G1 difference as a fixed item; on val Canto-L >= acorn (HNL)+LM in every iso r_v, \|d0\|, r_first and pT bin. Report P_G reweighted in r_v to the Zenodo HNL shape and to exponentials with lambda = 100 and 300 mm (Kish n_eff 2,046 / 1,424 / 2,509 of 2,914 now). Give per-species, pT and \|eta\| tables. | text plus about 1 CPU-h |
| 3 | The labels are correct in the new samples but exchanged in the training files. | high | Prompt-track gap, Canto-L minus acorn (HNL)+LM: 1.40 points on Zenodo val and 0.13 on generic_iso val, a change of the size of the 1-point margin. Labels, d0 definition and events all change at once. | A second read-out with the Zenodo exchange emulated on the generated labels, re-matching the stored tracks (no inference). Fix it before T2. | 1-2 CPU-h |
| 4 | We produced the sample ourselves (designer advantage). | high to medium | Spec 69a6f11 at 06:21:44; production arrays from 06:22:55. Models frozen in aecb23f. All timestamps are in a private repository. | Before T2, deposit addendum G, the spec, the deviations, the split lists and the checksums on Zenodo or OSF. Release the gun code and the seeds. State that the gun enriches the bins where acorn led on the HNL test (\|d0\| >= 300 mm: 96 iso test targets against 36). | about 1 h |
| 5 | The paper says the external HNL sample has a "mass and lifetime no model saw". | high (wording) | 15 GeV / 100 mm shares its mass with hnl15_200 and its lifetime with hnl10_100. The phrase is in intro.tex:79-80 and conclusions.tex:31; test.tex:16 is already correct. | Change it to "a combination of mass and lifetime not used in training". New HNL points outside the trained range (section 6.6). | text; about 7 CPU-h per point |
| 6 | There is no pileup, and everything comes from one simulation. | high for HL-LHC relevance | TrackML with pileup: Canto-L finds 88.2 % of targets with \|d0\| >= 10 mm and 31 of 69 at \|d0\| >= 100 mm. | Scope every generic-sample statement to "this simulation, without pileup", with the TrackML numbers beside it. A pileup sample is outlook only. | text (pileup: 100 GPU-h or more, not now) |
| 7 | The far, forward and shallow bins are too thin for model comparisons. | medium | Section 4 | Absolute efficiencies only above 300 mm; report merged tails. The extension triples the counts. | part of the extension |
| 8 | The "dense" component is not dense. | medium | 91 % of dense test targets have no other target within Delta R < 0.05 (h -> s s: 84-87 %). The decays are flat phase space without fragmentation, and the digitisation has no cluster merging. | Call it "multi-prong". Keep the Delta R < 0.05 read-out and add the truth-chosen bins of section 4. Base dense-vertex statements on h -> s s, without a claim. | text and truth |
| 9 | h -> s s carries no claim and has no decay ancestry. | medium | P_H includes V0 daughters (14.1 / 13.0 %) and s -> ee daughters (6.4 / 5.3 %). | Optional (the user decides): preregister, before any model reads it, a non-inferiority claim on P_H (frozen Canto-L (ttbar) against acorn (HNL)+LM, delta = 1). Rerun Pythia alone with the same seeds to recover the ancestry. | text; minutes of CPU |
| 10 | The truth definitions differ: exact d0 and r_v here, fitted values on Zenodo. | medium-low | T1 deviation 5 | Compute the hit-fit d0 on the generated samples and report P_G under both definitions. | about 1 CPU-h |
| 11 | Displaced propagation has not been closed against the Zenodo HNL samples. | medium | The stack was validated on 1,000 ttbar events and in a 67-particle gun pilot. | The HNL closure pilot, which the HNL points need anyway. | 1-2 CPU-h |
| 12 | The gun covers \|eta\| < 3, the targets \|eta\| < 4. | medium-low | \|eta\| > 2.5: 3.0 % of gun P_G, 11.3 % of HNL daughters. | Report \|eta\| bins and state the coverage. Optionally, a forward evaluation set. | text; about 10 CPU-h |
| 13 | The samples use 14 TeV; the Zenodo record says 13 TeV. | low | Spec section 2. The validation, at +-1.5 % in hits per event, cannot tell the two apart. | State it in the spec and the paper. For the HNL points use the energy the closure pilot selects. | text |
| 14 | Four iso/dense event pairs share their ttbar collision. | low | Section 8, check 6 (new) | Log it as a T1 deviation and report G2 and G3 also without iso event 1592. | truth only |
| 15 | Smaller points: uniform field; K_S, Lambda, nuclear interactions and in-flight decays absent in ttbar; uncorrelated, isotropic LLPs; working points fixed on HNL val (acorn (HNL)+CC runs at a 4.2 % fake rate); acorn (HNL) runs outside its training environment. | low | val read-out | Text. Prompt efficiency on generic_iso val is level (97.70 against 97.57), so the 2.8-point displaced gap is not an environment effect. | text |

## 6. Recommendation

### 6.1 Extra events: 16,000 more generic_iso events, and no other sample

We recommend one new production of the isolated component: 16,000 events numbered 8000-23999, for 24,000 isolated
events in all.

**What it buys.**
- Under the hash rule about 4,770 of the new events land in test. That raises P_G on test from 2,914 to about 8,700.
- G3's difference for 80 % power falls from 1.75 to 1.30 points (from 2.08 to 1.44 in the pessimistic variant).
  P(holds) in the 1.1-1.6 point window rises by 0.2-0.3.
- The |d0| >= 300 mm test bin grows from 96 to about 290 targets.
- Going further gains little, because the seed term dominates: 32,000 events give 1.24 points.

**Why no other sample.**
- The dense Delta R < 0.05 bins cannot be filled by this gun.
- The h -> s s counts already support the inclusive and hadron statements.

Production details:

- **Name and place.** `generic_iso_x1`, written to `hnl_lrt/generated/generic_iso_x1/` with sidecars in
  `generic_iso_x1_truth/`. generic_iso_v1 is not touched.
- **Settings.**
  - Use the isolated component of `docs/new_samples_spec.md` section 3 unchanged: Pythia8 ttbar at 14 TeV plus two gun
    LLPs per event with the same distributions, on the stack of section 2.
  - Keep 14 TeV. The extension must be the same sample, so the 13-TeV question concerns the HNL points only.
  - The spec's section 9 records the extension (seed, directory, storage) before production starts.
- **Seed.** The extension gets its own base seed, 260001:
  - Pythia uses 260001;
  - the gun uses `default_rng([260008, event])`;
  - Fatras and digitisation use 260014.
  We enumerated the ACTS streams, base + Cantor(algorithm, event) mod 2^32, for algorithm numbers 1-12, both ACTS
  steps and events 8000-23999: 384,000 streams in all. None coincides with a stream of generic_iso_v1,
  generic_dense_v1, hss15_100_v1, hss40_300_v1 or the ttbar validation events.
- **Storage.**
  - Each 500-event chunk becomes one uncompressed tar, `events_008000_008499.tar` and so on, whose members are
    `<event>.npz` in the Zenodo format. This is the layout of `hnl.data.shards`, which is being written for the HNL
    points.
  - Each chunk also gets its two sidecars, `truth_<first>.npz` and `gun_truth_<first>.npz`.
  - In all: 32 tars plus 64 sidecars, 96 files and about 2.9 GB (generic_iso_v1 takes 1.46 GB for 8,000 events).
  - hnl_lrt goes from 66,152 to about 66,250 files, against the stop line of 80,000. The /ceph quota has 163 GB free.
- **Code.** Produce from a committed state and record its hash.
  - `slurm/acts/produce.sbatch` gets a `generic_iso_x1` case with the seed, the output directory and tar packing.
  - `scripts/acts/acts_sim.py` currently has uncommitted edits for the HNL points. They leave the ttbar path at 14 TeV,
    but whichever version is used is committed first.
- **Reproduction check before production.**
  - Rerun chunk 7500-7999 of generic_iso_v1 (base seed 250001) with the extension's code into scratch on /work.
  - Require every array to equal the stored npz. Spec section 4 says a chunk reproduces its events.
  - Any difference stops the production. This shows the code path has not changed since the first production.
- **Reader.**
  - The tar reader must pass a byte-identity test: pack 500 generic_iso_v1 events into a scratch tar and read them
    through both readers.
  - The addendum-G code then takes the generic_iso test split as the union of the two productions.

### 6.2 Splits, and how the hash rule's extra train and val events are handled

- **Split rule.** Section 3's hash rule applies unchanged, with component string `iso`. Among events 8000-23999 we
  expect about 9,600 train, 1,600 val and 4,770 test events.
- **Train events.** These are not used, and the T3 training list stays as committed.
  - Section 5.2, with T1 deviation 9, takes the first 7,999 train events by event number. The last of them is number
    6,677.
  - Every extension event has a higher number, so the rule itself excludes them.
  - No model reads them under addendum G. They are stored as a reserve, and any later use needs its own commit first.
- **Val events.** No model reads them under addendum G.
  - T3 selects on the 806 T1 val events (`docs/addendum_g_t3_plan.md` section 4). That choice was fixed before the
    extension existed.
  - The frozen models' val read-out is not extended.
- **Test events.**
  - The claims use the union of the 2,384 T1 test events and the extension's test events, evaluated once at T2 (frozen
    models) and once at T4 (retrained).
  - The bootstrap table (B = 500, `default_rng(12345)`) is drawn over the union, ordered by event number. No
    generic_iso test table exists yet; `derived/bootstrap/generated_v1` holds val and h -> s s tables only.
  - No claim is evaluated on either part alone.
- **Alternative considered: all new events as test.** About 4,800 events would give the same test statistics for
  6.3 CPU-h of production instead of 21, at the same inference cost. We do not recommend it. It changes the split
  rule for the new event numbers, and it saves about 15 CPU-h out of roughly 300.

### 6.3 Text to log in RESULTS.md as T1 deviations 13 and 14, before production and before T2

> 13. **Size of the isolated component** (sections 2, 3 and 6). Decided 2026-10-14, before any model read a generic
> test event and before any retrained model of T3 had a validation number. The power study
> `results/new_samples_sufficiency_v1.md` used truth, the frozen models on generic_iso val and the HNL val read-outs.
> It found that with 8,000 events G3 has 80 % power only for a difference of at least 1.75 points, against a
> plausible range of 1.1-3.7 points.
>
> generic_iso is therefore extended by 16,000 events with event numbers 8000-23999, produced as `generic_iso_x1`:
> - generator settings as in `docs/new_samples_spec.md` section 3 (ttbar at 14 TeV, the same gun);
> - its own base seed 260001 (gun 260008, Fatras and digitisation 260014). No per-event ACTS stream of it coincides
>   with a stream of the existing samples; this was checked by enumeration.
> - storage as one tar of Zenodo-format npz files per 500-event chunk, plus its two truth sidecars, under
>   `hnl_lrt/generated/generic_iso_x1/`.
>
> The component string stays `iso`, so section 3 assigns these events with the same hash. T1's checks are repeated on
> them and committed with their split lists before T2. Consequences:
>
> (a) The generic_iso test split of the claims is the union of the 2,384 T1 test events and the extension's test
> events, and it is evaluated once. The bootstrap table (B = 500, default_rng(12345)) is drawn over the union, ordered
> by event number. No claim is evaluated on either part alone.
>
> (b) The T3 training list is unchanged. Section 5.2 takes the first 7,999 train events by event number
> (deviation 9). The last of these is number 6,677, so no extension event qualifies.
>
> (c) No model reads the extension's train or val events under this addendum. T3 selects on the 806 T1 val events
> (`docs/addendum_g_t3_plan.md` section 4), which were fixed before the extension existed.
>
> (d) The decision stands whatever the extension's truth shows, with one exception: events that fail the T1 checks
> are listed and dropped, as section 3 provides.
>
> 14. **Shared ttbar collisions between the two components** (truth only, found at the sufficiency check). ACTS forms
> each per-event seed as base + Cantor(algorithm, event), and the base seeds of generic_iso_v1 and generic_dense_v1
> differ by 100,000. For four event pairs the resulting Pythia seeds coincide:
> - iso 470 and dense 150;
> - iso 860 and dense 735;
> - iso 1592 and dense 1528;
> - iso 4010 and dense 3985.
>
> Each pair has the same generated ttbar collision: 188-251 particles per pair share their production vertex
> exactly, against none in control pairs. The pairs were simulated with different Fatras and digitisation streams,
> and their gun particles differ. Only iso 1592, a test event, has its twin (dense 1528) in the T3 training list, and
> none of its P_G targets was in training. G2 and G3 are evaluated as written, and their difference without iso
> event 1592 is reported next to them.

### 6.4 Truth and text checks to add before T2

These go in one pre-T2 commit. None of them changes a claim.

1. **T1 on the extension.**
   - Checksums, format and corruption checks, split lists, and the P_G count per split.
   - A comparison with generic_iso_v1 of the P_G targets per event (1.21 now), the distributions of r_v, |d0|, pT and
     eta, the species shares, and the hits per target.
   - A difference beyond the statistical expectation stops the work before any model runs.
2. **Reported items fixed before the read:**
   - the per-bin sign of the G1 difference;
   - P_G reweighted in r_v to the Zenodo HNL shape and to exponentials with lambda = 100 and 300 mm, with Kish n_eff;
   - per-species, pT and |eta| tables.
3. **A second read-out with the Zenodo label exchange** emulated on the generated labels, using the stored tracks.
4. **P_G under the hit-fit |d0| and fitted r_v**, next to the exact definitions.
5. **Dense bins chosen from truth.**
   - Same-vertex multiplicity: 1 / 2 / 3 / 4 / >= 5.
   - Other targets within Delta R < 0.2: 0 / 1 / 2 / 3-4 / >= 5.
   - Merged tails: r_v >= 450 mm, r_first >= 500 mm, and Delta R < 0.05 multiplicity >= 2.
   - The preregistered bins are kept alongside.
6. **G2 and G3 also without iso event 1592** (deviation 14).
7. **An external timestamp.** Deposit on Zenodo or OSF: addendum G, the spec, deviations 1-14, the split lists and
   the checksums.
8. **Paper text.**
   - intro.tex:79-80 and conclusions.tex:31: "a combination of mass and lifetime that no model was trained on".
   - State 14 TeV against the record's 13 TeV.
   - Scope the generic-sample results to this simulation without pileup, with the TrackML displaced numbers beside
     them.
   - Call the dense component "multi-prong".
   - Say that the gun enriches the bins where acorn led on the HNL test.
   - Give G3's minimum detectable difference: 1.30 points at 80 % power after the extension.
   - Above 300 mm, quote absolute efficiencies only.

### 6.5 What not to do

- **Do not extend for G1, and do not extend beyond 24,000 isolated events.** G1 is decided. Beyond 24,000 the gain per
  event is small: 1.30 against 1.24 points at 32,000.
- **Do not add training seeds to rescue G3.** The power study puts five seeds at about +130 GPU-h. A rough
  re-simulation with the same rule gives:
  - with five seeds, a difference for 80 % power of about 1.5 points at 8,000 events, against 1.3 with the extension
    and three seeds;
  - P(holds) of at most 0.44 at 0.53 points, even with unlimited events.

  The re-simulation omits the flip noise. It reproduces the power study's 1.04 with unlimited events and gives 1.26
  against 1.30 at 24,000.
- **Do not extend generic_dense or the h -> s s samples** for adequacy. Do not put a far-decay gun component into P_G
  or G1: it would be a new population, reported separately.
- **Do not change T3.** Its training list, val events and plan stay as committed. No extension event goes into any
  training or selection.
- **Do not look early.** Do not read any generic test output before deviation 13 and the extension's T1 are
  committed. Do not evaluate a claim on the T1 test events alone.
- **Do not change the extension's generator settings.** This includes the energy and the gun ranges. Do not use the
  frozen-model val numbers for anything except this size decision (DECISIONS 2026-10-14).
- **Do not rank models** above 300 mm, at r_v >= 650 mm or at r_first >= 700 mm.
- **Do not describe the generic sample as testing collimated (Delta R < 0.05) vertices.**
- **Do not start now on a pileup sample, a non-uniform field or nuclear interactions.** State them as limitations.
- **Do not write per-event files on /ceph,** and do not touch generic_iso_v1 or the Zenodo data.

### 6.6 For the HNL points the user asked for

The Zenodo yields per event (`bins/per_event_yields.csv`) are 0.59-0.77 HNL daughters, 0.063-0.091 at |d0| of
100-300 mm, and 0.003-0.009 above 300 mm.

**Per-point statistics.**
- 3,000 events per point give 1,800-2,300 daughters.
- The paired sigma is then about 0.3 points per point, against the frozen lead of 1.10 +- 0.22 on the HNL test sets.
- 1,500 events would give about 0.4.

**Far bins.** These must be pooled over points, because 100 targets above 300 mm per point would take 10,000-32,000
events.

**Order of work.**
1. Run the closure pilot first (objection 11).
2. Choose the energy the closure selects and state it. The uncommitted card `scripts/acts/hnl_card.py` notes that
   the closure prefers 14 TeV.
3. For a blind three-seed test of the first aim, add fresh events at the two trained points (10 GeV / 100 mm and
   15 GeV / 200 mm). Its claim must be committed before production.

## 7. Cost and timeline of the recommendation

| item | CPU-h | GPU-h | files on /ceph | storage |
|---|---|---|---|---|
| production, 32 chunks of 500 (4.7 CPU-s per event) | 21 | 0 | 96 | 2.9 GB |
| reproduction check (one chunk) | 0.7 | 0 | 0 (scratch on /work) | - |
| T1 on the extension (truth scan, splits, prepare) | about 10 | 0 | a few | small |
| T2 + T4 on about 4,770 extra test events: Canto-L union graphs, 6 seed-runs at about 33 CPU-s per event | about 260 | - | as for the T1 split | as for the T1 split |
| T2 + T4: Canto-L scoring and acorn inference | small | about 4 | - | - |
| optional: the one-seed Canto-L (HNL) and (ttbar+HNL) rows on the extension | 90-170 | under 1 | - | - |
| tar reader and union-split code | - | - | - | - |

Wall time: production takes about 1 h with 16 chunks in parallel. The extension's T1 takes about an hour more, and the
reader needs a few hours of code and tests. T2 would then start about a day later than without the extension. The
inference CPU adds 20-30 % to T3's 1,300-1,500 CPU-h.

## 8. Spot checks of the three assessments

1. **G3 threshold.**
   - The referee reading gave a G3 threshold of about 1.05 points now and 0.7 after an extension. It used the G1
     pair's sigma (0.35) and a seed term of 0.15.
   - The G3 pair has larger inefficiencies (c = 0.42, acorn L of about 4-5 %) and a seed term of 0.26 at the central
     floors. The power study's sigma is therefore 0.42 at 8,000 events and 0.32 at 24,000. The passing value 3 sigma
     is about 1.3 and 0.96 points, and the difference for 80 % power is 1.75 and 1.30.
   - We use the power study. A rough independent simulation of the same rule (2 x 10^5 draws, flip noise omitted, not
     saved) gives 1.63 / 1.26 / 1.04 at 8,000 / 24,000 / unlimited events.
2. **Production cost.**
   - The referee reading put 4,800 isolated events at 11 CPU-h. That figure uses the all-sample average from DECISIONS
     (41.9 CPU-h for 19,000 events), which includes the dense component.
   - The logs give 548-634 s per 500-event iso chunk on 4 CPUs, 9,442 s for all 16 chunks
     (`logs/acts_produce_6810705_*`). That is 10.5 CPU-h per 8,000 iso events, or 4.7 CPU-s per event.
   - Dense chunks took 1,161-1,944 s.
3. **Inference cost.**
   - The power study quoted "about 19 GPU-node-s per test event", about 25 GPU-h for 16,000 events.
   - Slurm accounting of today's `g_*` val jobs shows that most of that node time is CPU. Over 1,641 val events:
     - the Canto-L (ttbar) union graphs took 45.8 CPU-h on 16-CPU allocations (2.9 h elapsed);
     - scoring and acorn inference took about 0.8 h on GPU nodes.
   - We therefore quote about 260 CPU-h and 4 GPU-h. The total node time agrees with the power study's 274 node-min
     for the four G1 chains.
4. **Shallow crossings.**
   - The referee reading projected about 270 test targets at |d0|/r_first >= 0.7 from the val share (88 of 956).
   - The truth count is 226 of 2,914. The difference is a 1.4-sigma fluctuation of the val share.
   - The val count of 88 matches `results/addendum_g_val_v1.md`.
5. **G1 sigma.** Three routes agree on 0.34:
   - the sanity file's projection (0.33 / 0.34);
   - the power study (0.34);
   - recomputation, 0.572 x sqrt(956/2914) = 0.328 in quadrature with 0.078.
6. **Shared ttbar collisions** (new; the basis of deviation 14).
   - ACTS v15 seeds each algorithm per event with base + (k1+k2)(k1+k2+1)/2 + k2, where k1 is the algorithm number and
     k2 the event number (`Examples/Framework/src/Framework/RandomNumbers.cpp`). The Pythia generator is algorithm 1.
   - With bases 250001 and 350001 the Pythia seeds coincide for the four pairs listed in deviation 14.
   - The npz particle tables confirm it. Per pair, 188-251 particles share id and production vertex to 1e-6. Control
     pairs (iso 860 with dense 736, iso 100 with dense 100) have none.
   - Splits:
     - pair 860/735: both test;
     - pairs 470/150 and 4010/3985: all train;
     - pair 1592/1528: iso 1592 is test, dense 1528 is train and in the T3 list.
7. **Paper wording and numbers quoted by the referee reading.** All of the following were checked at the cited lines:
   - intro.tex:79-80 and conclusions.tex:31 say "mass and lifetime no model saw";
   - test.tex:16 says "combination";
   - results.tex:50-51 gives 1.93 / 1.35 / 0.53;
   - docs/paper_facts.md:324 gives 13 TeV;
   - the val bins (Canto-L >= acorn (HNL)+LM in every iso r_v, |d0|, r_first and pT bin; dense |d0| >= 300 mm
     83.3 against 88.0) appear in `results/addendum_g_val_v1.md`.
8. **Storage and file counts.**
   - generic_iso_v1 plus its truth: 1.3 GB + 0.16 GB.
   - `ceph.dir.rfiles` of hnl_lrt: 66,152 at 13:10 EDT, against 66,112 in the assessments, because other jobs are
     writing.

## 9. Sources

- Power study: `/work/submit/jinboz1/hnl/samples_v2/power/`.
  - Val-only per-target flags: `extract.py`, job 6812208.
  - Calibrations: `power.py`, giving `power_v1.json`, job 6812234.
  - Decision-rule simulations: `power_v2.py`, giving `power_v2.json` and `power_v2_tables.md`, job 6812318.
- Per-bin counts: `/work/submit/jinboz1/hnl/samples_v2/bins/`.
  - Counts: `per_bin_counts.py`, job 6812387, giving `per_bin_counts.csv`, `tables.md`, `projections.csv` and
    `per_event_yields.csv`.
  - Reference checks: `checks.txt`, 10 of 10 equal.
  - Neighbour counts: `density.py` and `density_cross.py`, jobs 6812437 and 6812442.
- Referee reading: `/work/submit/jinboz1/hnl/samples_v2/referee/truth_compare.py`, job 6812275, giving
  `truth_compare_v1.json`.
- Repository inputs:
  - `results/addendum_g_t1_v1.md`, `results/addendum_g_val_v1.md`, `results/addendum_g_val_sanity_v1.md`;
  - `docs/preregistration_new_sample.md`, `docs/addendum_g_t3_plan.md`, `docs/new_samples_spec.md`;
  - RESULTS.md (Addendum G, step T1), `splits/generic_*`.
- Spot checks of section 8, all on the login node and each a few seconds of CPU: the seed enumeration, the particle
  tables of the four event pairs and two control pairs, the split lookups, the production logs and sacct.
