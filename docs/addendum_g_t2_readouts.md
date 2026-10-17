# Addendum G step T2: inputs, read-outs and definitions fixed before the test read

Status: written 2026-10-14, before any model has read a generic test event, an h -> s s event, an event of the HNL
points or any event of the extension `generic_iso_x1`. It fixes how the frozen chains of step T2
(`hnl.eval.addendum_g`, `slurm/addendum_g/`) read the new inputs, and defines every read-out that RESULTS.md T1
deviation 15 lists, together with the HNL-point read-out. Nothing here changes a claim: G1 is decided by
`hnl.eval.addendum_g_summary.g1` exactly as in commit c3dc402, on the union test split of deviation 13a. A change
after this commit is logged in RESULTS.md with its reason before the affected read-out.

Governing documents: `docs/preregistration_new_sample.md` (addendum G, claims G1-G3), RESULTS.md "Addendum G, step T1"
(deviations 1-12) and "T1 deviations 13 and 14, and read-outs fixed before T2" (13-16), `docs/new_samples_spec.md`,
`docs/hnl_points_spec.md`, `results/new_samples_sufficiency_v1.md` sections 6.1-6.5.

## 1. Inputs

- **Sets.** `val` = generic_iso val + generic_dense val (pipeline check, done; nothing chosen). `test` = the generic_iso
  union test split + generic_dense test. `hss` = hss15_100 + hss40_300, every event. `hnl` = the five HNL points
  (hnl5_50, hnl20_200, hnl15_500 and the in-stack references hnl10_100_ref, hnl15_200_ref), every event of
  `splits/hnl_<m>_<ctau>_all.txt`; evaluation only, no claim.
- **Union test split** (deviation 13a). `splits/generic_iso_union_test.txt` (written by `hnl.eval.addendum_g_t1_x1`)
  must equal `splits/generic_iso_test.txt` (numbers 0-7999) plus `splits/generic_iso_x1_test.txt` (numbers
  8000-23999), disjoint, sorted by event number, and every extension event in it must be a test event under the hash
  rule of addendum G section 3 (`hnl.eval.addendum_g.check_union`, run whenever the list is read). The claims, the
  bootstrap table and every read-out of this document use only the union; no claim or read-out is computed on either
  part alone.
- **Extension events** keep the sample key `generic_iso` and its code 5: event ids are 5 x 10^8 + event number, their
  event tables are shards 08000-23500 of `derived/events/v1/<table>/generic_iso/` (next to the T1 shards
  00000-07500, which are not touched), and their truth rows are `derived/generated_v1/t1/truth_generic_iso_x1
  .parquet` (`hnl.eval.addendum_g_t1_x1`), appended to the T1 table of generic_iso. Their feature-store tar is
  `generic_iso_x1_test.tar`; the acorn stage extracts `generic_iso_test.tar` and `generic_iso_x1_test.tar` and checks
  that exactly the union's events were staged.
- **No model reads the extension's train or val events** (deviation 13c): `prepare-inputs` converts only the events of
  `splits/generic_iso_x1_test.txt` and reads only their tar members; no table, tar or graph of another extension event
  is written.
- **HNL points** have codes 10-14 (`hnl.data.acorn_store.POINT_CODE`; ten-digit event ids,
  `parse_event_id` reads all digits before the last eight as the code). Their event tables and feature-store tars
  (`<point>_all.tar`) are made by the same functions as every other sample (`hnl.data.convert.event_tables` with the
  shard sample's record, `hnl.data.acorn_store.build_tar`).
- **Reader identity.** `hnl.eval.addendum_g shard-identity` packs the first 500 events of the generic_iso val list, as
  the exact bytes of their npz files, into one scratch tar and converts them through both readers. Required and
  recorded in `results/addendum_g_t2_inputs_v1.json`: member bytes and arrays identical; the three event tables of the
  two paths identical to each other and to the production tables the val chain read; every feature-store graph built
  from the shard path identical, field by field, to the production val tar.

## 2. Execution changes (no effect on any number)

- **Canto-L scoring in parts.** A test set of 9,618 events would take about 42 min of prediction on one GPU, close
  to the 1-h express limit. The union blocks are split into N contiguous parts (test 4, hnl 4, hss 1), one GPU job
  each (`canto_score.sbatch`); `gather-scored` packs the parts into the one scored tar the evaluation reads, in sorted
  member order, after checking that the parts hold every acorn source event exactly once.
- **Union-block tars are removed** after the scored tar is complete (storage: the scored graphs contain them); their
  build reports and done markers stay.
- **Stored tracks.** Every evaluation block writes `tracks.parquet` (event id, hit row, track label for each hit on a
  track) besides its per-particle and per-event tables. The labels are the ones the frozen builder produced and that
  the block's matching used (`labels_out` of `hnl.eval.trackbuild_val.evaluate_event` and
  `hnl.eval.evaluate_gnn.evaluate_event`). Re-running the val evaluation with this code must reproduce the stored val
  per-particle and per-event tables exactly, and the val summary must reproduce `results/addendum_g_val_v1.*`.
- **Guard.** `slurm/addendum_g/guard.sh SET` refuses every set but val unless the T2 code, this document, the
  set's T1 records and split lists and the ledger are committed unchanged, and the ledger entry R14 names every input
  of the set (test: `generic_iso_union_test`, `generic_iso_x1`, `generic_dense`; hss: both samples; hnl: the five
  point keys). Every job that runs a model or a builder on a set calls it, and `chain.sh` calls it before submitting.

## 3. Read-outs of T1 deviation 15 (`hnl.eval.addendum_g_readouts`)

Statistics as `hnl.eval.addendum_g_summary`: 3-seed means, sigma^2 = sigma_boot^2 + s_seeds^2 / n, paired differences
on the set's fixed bootstrap table (B = 500). Delta = Canto-L (ttbar) - acorn (HNL) + LM (the models of G1), in
points, on the generic_iso part of the set (test: the union). Descriptive; none changes a claim.

1. **Sign of the G1 difference per bin** of addendum G section 7 with the bins of deviation 10: |d0|, r_v, first-hit
   radius, |d0| / r_first, pT, on P_G. Delta +- sigma and its sign per bin; the counts of positive and negative signs
   over the ranked bins. Not ranked (absolute efficiencies only): |d0| >= 300 mm (deviation 15), r_v >= 650 mm and
   first-hit radius >= 700 mm (sufficiency report 6.5).
2. **P_G reweighted in r_v** (exact r_v). Weighted efficiency = sum(w k) / sum(w) with weights fixed by truth, Kish
   n_eff = (sum w)^2 / sum(w^2):
   - to the **Zenodo HNL shape**: `results/addendum_g_zenodo_rv_shape_v1.csv` (`hnl.eval.addendum_g_truth
     zenodo-shape`): N-daughter targets of the labeled hnl10_100 and hnl15_200 samples, train and val splits only,
     with fitted |d0| >= 10 mm or fitted r_v >= 100 mm, histogram of the fitted r_v in 20-mm bins over [0, 800) mm.
     A P_G target in bin b gets w = f_Zenodo(b) / n_PG(b); the Zenodo fraction in bins without P_G targets is
     reported as uncovered (daughters without a vertex or beyond 800 mm are counted in the csv header);
   - to **exponentials with lambda = 100 and 300 mm**: w = exp(-r_v / lambda). The gun draws r_v uniformly in
     [1, 800] mm, so the weighted sample has an exponential decay radius truncated to the gun's range; acceptance and
     the P_G cut act as before.
3. **Tables by species, pT and |eta|** on P_G: species by |PDG| (e 11, mu 13, pi 211, K 321, p 2212); pT bins of
   deviation 10; |eta| (table value) bins 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0. Every model's efficiency and Delta.
4. **Second read-out with the Zenodo same-module label exchange emulated** on the generated labels, from the stored
   tracks, for every model, seed and population of the summary (and the fake rate), with G1's rule applied to the
   emulated numbers beside the true ones. Emulation:
   - per event, hits grouped by module (geometry id); a module whose k >= 2 hits carry at least two different particle
     labels is exchanged with probability p(k), the fraction of resolved k-hit modules whose labels do not match the
     positions in the Zenodo ttbar val events (`results/label_swap_verify_v1.md` section 2): p(2) = 1961/17448 =
     0.1124, p(3) = 872/3542 = 0.2462, p(4) = 202/777 = 0.2600, p(k >= 5) = (43 + 7 + 2) / (154 + 30 + 4) = 0.2766;
   - an exchanged module's labels are permuted among its hits by a uniform random permutation that changes at least
     one hit's label (two hits: the swap);
   - independent of separation, pT and generation, as the measured mechanism (labels in particle-id order of the hit
     ids, positions in another order; same file, "Where it comes from");
   - random numbers: numpy `default_rng([20261014, sample code, event number])`; one uniform per eligible module in
     increasing geometry-id order, then the permutations of the exchanged modules in that order, hits of a module in
     row order. The emulated labels of an event are the same for every model and seed;
   - the tracks are not changed; double-majority matching and the fake count (docs/definitions.md section 5) are
     repeated against the emulated labels. Each particle keeps its number of hits, so targets and populations are
     unchanged. Before emulating, the stored tracks matched against the true labels must reproduce the block's stored
     matches and fake counts exactly (any difference stops the read-out).
5. **P_G under the hit-fit |d0| and fitted r_v** of docs/definitions.md (`hnl.eval.addendum_g_truth fits`): the truth
   fit of docs/definitions.md section 3 on the whole event (`hnl.truth.fit_events.fit_event`, field sign -1, as
   derived/truth_fits/v1); |d0| from the fixed-R fit for muons and the free-curvature fit otherwise; fitted r_v of an
   LLP = the two-track vertex of its two prongs' circles (each from the fit that gives its |d0|), with the choice rule
   of docs/definitions.md; NaN ("no vertex") if a prong has no fit or no candidate is accepted. P_G(fit) = gun targets
   with fitted |d0| >= 10 mm or fitted r_v >= 100 mm. Reported: overlap with P_G (exact), efficiencies and Delta on
   both, bins in fitted |d0| and fitted r_v ("no vertex" separately; |d0| >= 300 mm not ranked).
6. **Dense component, bins chosen from truth** (P_dense on generic_dense), beside the preregistered bins of the
   summary: same-vertex multiplicity = number of displaced-source targets of the same LLP (1 / 2 / 3 / 4 / >= 5);
   other targets within Delta R < 0.2 of the initial direction (0 / 1 / 2 / 3-4 / >= 5); merged tails r_v >= 450 mm,
   first-hit radius >= 500 mm, Delta R < 0.05 multiplicity >= 2. Every model; Delta descriptive.
7. **G2 and G3 without iso event 1592** (deviation 14): `paired_claim(D, A, B, rule, exclude_events)` and `g2_g3`
   evaluate G2 (non-inferiority, as G1) and G3 (Delta >= 3 sigma and a fake rate not above acorn's) on the union,
   as written and with event 1592 removed (its targets and per-event rows; the bootstrap table unchanged). Run at
   step T4 on the retrained models. With G1's models and no exclusion `paired_claim` reproduces `g1` exactly; every
   read-out checks it.

## 4. HNL points (`hnl.eval.addendum_g_summary`, set hnl; descriptive, no claim)

- **Truth** (`hnl.eval.addendum_g_truth points`, every event of the point read through the tar reader): categories
  from column 12 as in the labeled Zenodo samples (N daughter: |parent| = 50; W muon: |parent| = 24 and |PDG| = 13);
  targets as docs/definitions.md section 4.
  - **P_HNL (fitted)** = N-daughter targets with fitted |d0| >= 10 mm or fitted r_v >= 100 mm: the population of
    docs/definitions.md as in the Zenodo samples (truth fit on the whole event; r_v = two-track vertex of the
    daughter pair, `fit_events.pair_vertices`). Its counts must reproduce `results/hnl_points_truth_v1.md` (1,966,
    1,714, 1,967, 1,800, 1,051).
  - **P_HNL (exact)** = N-daughter targets with exact |d0| >= 10 mm (helix through the production vertex with the
    initial momentum, as T1) or exact r_v >= 100 mm (the daughters' production radius, i.e. the N decay radius).
- **Read-out** per point and model: efficiency on P_HNL (fitted), P_HNL (exact), all N-daughter targets, W-muon targets
  and all targets; track fake rate; graph limit; bins of all N-daughter targets in fitted and exact |d0| (bins of
  docs/definitions.md) and r_v (docs/definitions.md bins with the >= 300 mm overflow split at 450 and 650 mm; fitted
  r_v with "no vertex"), first-hit radius and pT.
- **Each new point beside its in-stack reference**: hnl5_50 and hnl20_200 beside hnl10_100_ref, hnl15_500 beside
  hnl15_200_ref (docs/hnl_points_spec.md section 11: m_N or c tau0 effects are read only against them). Per model and
  population: point, reference, and point minus reference with independent bootstrap tables
  (`default_rng(12351-12355)`, B = 500) and the seed terms of both.

## 5. Cost (estimate before the test read)

From the val run (1,641 events; jobs 6810987-6812441): acorn inference 0.37 GPU-h for seven runs and Canto-L scoring
0.75 GPU-h for five runs, i.e. 6.8 x 10^-4 GPU-h per event for the whole model set, job overheads included.
- test: 7,233 union + 2,385 dense = 9,618 events, about 6.6 GPU-h;
- hss: 3,000 events, about 2.1 GPU-h;
- hnl: 15,000 events, about 7.5 GPU-h at their size (0.73 of a generic event), 10.2 GPU-h unscaled;
- the extra scoring parts add about 15 job overheads per set (about 0.4 GPU-h each for test and hnl).

In all about 20 GPU-h, well below the workflow's 75 GPU-h: test, h -> s s and the HNL points can all run. CPU
(allocated cores, the union graphs dominate; val took about 100 CPU-h): about 600 CPU-h for test, 200 for hss and 700
for the HNL points. Storage on /ceph at the peak of a chain, before the union tars are removed: test 45 GB, hss 12 GB,
hnl 52 GB (from the val outputs: 0.24 MB per event and acorn run, 0.28 + 0.26 MB per event and Canto-L run);
`chain.sh` refuses to start when the free quota minus the peak would fall below 15 GB, so with about 150 GB free the
sets run one after the other, or test and hss together. Files: about 1,000 per set under runs/addendum_g_v1
(hnl_lrt holds 66,806 files before T2, against the stop line of 80,000).

## 6. Order of the T2 run

1. Truth only, before the ledger (done 2026-10-14, `results/addendum_g_t2_inputs_v1.json`): `prepare-inputs --inputs
   points`, `addendum_g_truth points` (five points), `zenodo-shape`; after the extension's T1: `prepare-inputs
   --inputs x1` (on submit37, the host of the T1 feature-store tars, so that even the unused float32 hit-eta feature
   is bit-identical), `addendum_g_truth fits --list union_test`.
2. Commit the ledger entry R14 naming every input.
3. `bash slurm/addendum_g/chain.sh test`, then `hss`, then `hnl` (or together if storage allows), each once.
4. Every number is reported, whatever it is: `results/addendum_g_g1_v1.*`, `results/addendum_g_readouts_test_v1.*`,
   `results/hss_eval_v1.*`, `results/hnl_points_eval_v1.*`.
