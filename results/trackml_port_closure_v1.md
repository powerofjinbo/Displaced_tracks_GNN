# Closure of the reference-code ports against the authors' published numbers (v1)

Supervisor's request (2026-10-14): before Canto-L is recounted with the ported evaluation codes, reproduce the
references' own baseline numbers with the ports and check that the efficiency definitions match. Evaluation code only:
no Canto-L output and no test event of ours was read. Code: `hnl.trackml.port_closure` (this commit's working tree),
`slurm/trackml/port_closure_fetch.sbatch`, `slurm/trackml/port_closure.sbatch`. Numbers: `trackml_port_closure_v1.csv`.
Jobs (CPU only, express partition, A30 and GTX 1080 nodes excluded): 6810501 (fetch), 6810527 (Liu), 6810539 (EggNet
counts); 6810518 used a wrong event split (no torch random_split) and is superseded; 0 GPU-h.

## What can be reproduced

| Reference | Checkpoints | Predictions / scored graphs | Evaluation events | Reproducible number |
|---|---|---|---|---|
| EggNet (Calafiura et al. 2024, Chan et al. 2025; github.com/exatrkx/EggNet e2bd117) | none released | none | feature store under /global/cfs/cdirs/m2616 (not public) | none of the model rows; the particle count of Chan et al. (a check of the target definition) |
| Liu et al. (HierarchicalGNN a2563af, Tracking-ML-Exa.TrkX f107781) | none released | none | TrackML-1GeV quickstart: 340 filter-processed events (portal.nersc.gov/cfs/m3443/ryanliu/TrackML1GeV/events.tar.gz, 459 MB); TrackML-full store not public | Truth-CC row of Table 3 (6 removal fractions, efficiency and FR) |

Retraining a model row is outside a ~10 GPU-h check and would test training, not the evaluation code: EggNet
(GAT with repeated KNN on full events, 2,000 training events) has no released data or weights; BC-HGNN needs RAPIDS
22.04 (cudf, cugraph, cuml), frnn and PyTorch Lightning 1.6 and up to 500 epochs on the 300 quickstart events, with no
published training-to-training spread to compare against; EC-GNN (Table 2) needs the TrackML-full store.

The released quickstart events are throughput-phase events: their `event_file` attributes are
`.../trackml-codalab/train_all/event000021000` to `event000021339`.

## Liu et al.: Truth-CC, Table 3 (TrackML-1GeV)

Truth-CC as in the authors' notebook (track_eval.ipynb cells 4-5): graph edges whose two hits carry the same pid,
connected components, `eval_metrics(pt_cut=1, nhits_cut=5, majority_cut=0.5)`, per-event values averaged over the
events. Random edge removal as in the authors' dataset class (each edge dropped with probability f); the published row
is one draw, ours the mean and standard deviation of 20 draws. Event split as in the authors' code (seeded shuffle,
then `random_split([300, 10, 10])` with seed 0): test events 21130, 21172, 21175, 21179, 21218, 21230, 21233, 21250,
21281, 21308.

| Edges removed | Eff published | Eff port (sd of draws) | FR published | FR port (sd of draws) |
|---|---|---|---|---|
| 0 % | 98.72 | **98.72** | 5.87 | 6.09 |
| 10 % | 96.21 | 96.41 (0.17) | 15.53 | 15.79 (0.31) |
| 20 % | 92.31 | 92.39 (0.26) | 24.40 | 25.10 (0.43) |
| 30 % | 85.81 | 86.21 (0.22) | 33.48 | 34.29 (0.22) |
| 40 % | 77.26 | 77.42 (0.20) | 42.99 | 43.64 (0.28) |
| 50 % | 64.81 | 64.96 (0.35) | 53.12 | 53.84 (0.37) |

- Efficiency: exact without removal; with removal the port is 0.08 to 0.40 points above the published single draw,
  0.3 to 1.8 standard deviations of the draws.
- FR: 0.22 points above without removal (deterministic), 0.26 to 0.81 points above with removal, always on the high
  side. The offset was not traced: the authors' unmodified eval_metrics gave the same per-event values as the port on
  these events when the port was written (workflow wf_9ddbb9d7-334), so the difference lies upstream of the metric
  (the graph pruning or event handling of the run that produced Table 3), not in the port.
- Weighting: with the events weighted by their hit count, as PyTorch Lightning averages the logged values of the model
  rows, the 0 % row is 98.79 / 5.96; the published 98.72 is the unweighted mean of the notebook. Canto-L's rows in Table
  VI use the hit-weighted mean, as the published model rows do.
- All 340 events: 98.93 / 5.95 without removal.

## EggNet: the efficiency denominator

The EggNet code counts as targets the particles with pid != 0, pT >= 1 GeV (initial momentum) and at least 3 hits, with
no eta cut. Chan et al. evaluate "About 467K particles" over 400 test events (arXiv:2506.03415v1 p. 4), 1,167.5 per event.
Counted on our non-test events (particles per 400 events, thousands; range over all windows of 400 consecutive events,
since neighbouring throughput events reuse generator collisions):

| Definition | Throughput (1,549 events) | Accuracy (1,470 events) |
|---|---|---|
| pT >= 1 GeV, >= 1 hit | **469.9** (465.3-472.5) | 468.7 (463.0-471.9) |
| pT >= 1 GeV, >= 2 hits | 461.3 (456.8-463.9) | 459.9 (454.2-463.1) |
| **the code's: pT >= 1 GeV, >= 3 hits** | 453.7 (449.3-456.2) | 452.3 (446.8-455.4) |
| pT >= 1 GeV, >= 5 hits | 438.6 (434.2-441.1) | 437.3 (431.9-440.2) |
| published | 467 | 467 |

The code's definition gives 2.9 % fewer particles than quoted, and no window of 400 events reaches 467K. Only the count
without a hit requirement (pT >= 1 GeV and at least one hit) agrees with the quoted number. Hits and particles per event
(109.2K and 9.26K on our throughput events) agree with the quoted 108K and 9K. Either the published efficiency of
95.87 % counts the particles with one or two hits in its denominator, or the evaluated events differ from ours by 3 % in
the number of pT > 1 GeV particles. In the first case Canto-L's 99.41 % with all hits (EggNet code, >= 3 hits, Table VI)
is not like for like: 3.45 % of the pT >= 1 GeV particles with hits have one or two hits (our non-test throughput
events), so under that denominator Canto-L would find between 95.98 % (none of them found) and 99.43 % (all found),
still not below EggNet's 95.87 %. A post-freeze recount of Canto-L's test tracks with the >= 1-hit denominator would
settle the number; it was not run here (test outputs are out of scope of this check). It was run afterwards as
ledger R13 (results/trackml_eggnet_denominator_v1.md): 95.99 +- 0.04 % at WP-A with the >= 1-hit denominator.

## Efficiency definitions side by side

| | EggNet paper text | EggNet code (= refports.eggnet) | Liu et al. paper text | Liu et al. code (= refports.liu, fork) | Our metric (Secs. VI, VIII) |
|---|---|---|---|---|---|
| Candidates | "all track candidates extracted from the DBSCAN clusters" | DBSCAN clusters, min_samples 3: >= 3 hits; ours: tracks with >= 3 hits | "track candidates" | label groups with >= 3 hits | all tracks for matching; tracks with >= 5 hits for the fake rate |
| Matching | > 50 % of the track's hits from the particle | same (strict); noise-majority tracks unmatched | > 50 % on both sides | > 50 % on both sides (strict), best candidate per particle; pairs with <= 2 shared hits or noise dropped | > 50 % on both sides (double majority) |
| Efficiency population | Calafiura: pT > 1 GeV; Chan: not stated ("About 467K particles") | pid != 0, pT >= 1 GeV, >= 3 hits, no eta cut | > 5 spacepoints and pT >= 1 GeV | pT > 1 GeV (minimum over the particle's hits) and >= 5 hits, no eta cut | pT > 1 GeV, abs(eta) < 4, >= 5 hits |
| Efficiency | fraction of particles with >= 1 matched candidate | matched targets / targets, counts pooled over events | N(matched, rec) / N(rec) | same per event; events averaged, weighted by hit count (model rows) or unweighted (Truth-CC notebook) | matched targets / targets, pooled |
| Duplication | "fraction of track candidates that are matched to the same particles" | (tracks matched to targets - matched targets) / matched targets | none | none | extra single-majority matches per target |
| Fake | "fraction of track candidates that are not matched to any particle" | unmatched tracks / matched particles | 1 - N(matched) / N(candidates) | 1 - matched rec / (candidates - dropped pairs - matched non-rec) | unmatched tracks with >= 5 hits / tracks with >= 5 hits |

Mismatches:
1. **EggNet population.** The code's >= 3-hit targets do not reproduce Chan et al.'s particle count; a >= 1-hit
   denominator does (above). Open; matters for the all-hits row of Table VI.
2. **EggNet duplication and fake rate.** The code divides by matched particles, the papers' text by track candidates.
   The port and Canto-L's rows follow the code.
3. **Liu et al. reconstructable particles.** Text: more than 5 hits and pT >= 1 GeV; code: at least 5 hits and pT > 1
   GeV. 2.0 % more particles under the code (1,096.5 against 1,075.1 per throughput event). The port follows the code;
   Truth-CC reproduces Table 3 with it.
4. **Liu et al. FR.** The code's FR leaves the matched non-reconstructable particles' candidates and the dropped pairs
   out of the denominator; it is not the text's 1 - N(matched)/N(candidates). Truth-CC FR is reproduced to 0.2-0.8
   points (above).
5. **Liu et al. particle pT.** The code takes the minimum hit-level pT of each particle; our test tables hold only the
   initial pT, which Canto-L's recount uses. Affects particles whose pT falls below 1 GeV along the track; not
   quantified.
6. **Liu et al. averaging.** Model rows are hit-weighted means of per-event ratios, Truth-CC an unweighted mean.

## The 0.858

0.858 is the graph-stage ceiling for displaced targets (abs(d0) >= 10 mm): the fraction of them that a perfect edge
classifier could reconstruct from Canto-L's TrackML graph (X1 embedding plus triplet support, 15.73 pairs per hit), on
the 50 validation monitoring events of gate G1 (DECISIONS.md "TrackML gate G1 passed with X1", 2026-10-07; jobs
6707285, 6707286), against 0.9943 for all targets. Exact value 0.8578. It was sent to the supervisor in
docs/feedback/response_andre_r2/response_v2.tex, line 214. It is not the accuracy-phase displaced efficiency 85.8 % of
TM48. On the 150 validation events 214 of the 265 lost displaced targets are lost in the graph
(results/trackml_displaced_diag_v1.txt).
