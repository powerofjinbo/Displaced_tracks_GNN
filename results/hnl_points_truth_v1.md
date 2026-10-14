# HNL points: truth summary of the produced samples

Produced per `docs/hnl_points_spec.md` (sections 4-6, changes of section 11); checks of its section 5 by `scripts/acts/hnl_points_check.py` (scan per sample, then summary over the census of `scripts/acts/hnl_census.py shards`). Truth only: no model was run on these events and nothing was chosen on them. All five samples are evaluation samples (spec section 7).

## Samples and checks

| sample | m_N / c tau0 | events | shards / sidecars | size (GB) | reader and format invariants | corrupted (largest envelope excess) | one W muon in the table | generator: W muon and two N daughters | next() > 1 |
|---|---|---|---|---|---|---|---|---|---|
| `hnl10_100_ref` (`hnl_10_100_v1`) | 10 GeV / 100 mm | 3,000 | 6 / 6 | 0.39 | 3,000 of 3,000 pass | 0 (7.4 mm) | 2,999 | 3,000 | 0 |
| `hnl5_50` (`hnl_5_50_v1`) | 5 GeV / 50 mm | 3,000 | 6 / 6 | 0.40 | 3,000 of 3,000 pass | 0 (5.4 mm) | 3,000 | 3,000 | 0 |
| `hnl20_200` (`hnl_20_200_v1`) | 20 GeV / 200 mm | 3,000 | 6 / 6 | 0.40 | 3,000 of 3,000 pass | 0 (3.7 mm) | 2,999 | 3,000 | 0 |
| `hnl15_200_ref` (`hnl_15_200_v1`) | 15 GeV / 200 mm | 3,000 | 6 / 6 | 0.39 | 3,000 of 3,000 pass | 0 (5.9 mm) | 3,000 | 3,000 | 0 |
| `hnl15_500` (`hnl_15_500_v1`) | 15 GeV / 500 mm | 3,000 | 6 / 6 | 0.40 | 3,000 of 3,000 pass | 0 (4.2 mm) | 3,000 | 3,000 | 0 |

Daughters in the particle table per event (0 / 1 / 2) and lifetime of the generator truth (each sample's own mean; the card's c tau is fixed in spec section 10):

| sample | events with 0 / 1 / 2 N daughters in the table | daughters that are not muons | mean proper lifetime (mm) | / c tau0 |
|---|---|---|---|---|
| `hnl10_100_ref` | 0 / 32 / 2968 | 0 | 100.4 +- 1.9 | 1.004 +- 0.019 |
| `hnl5_50` | 0 / 37 / 2963 | 0 | 50.7 +- 1.0 | 1.014 +- 0.019 |
| `hnl20_200` | 0 / 5 / 2995 | 0 | 196.8 +- 3.6 | 0.984 +- 0.018 |
| `hnl15_200_ref` | 0 / 15 / 2985 | 0 | 196.9 +- 3.7 | 0.984 +- 0.018 |
| `hnl15_500` | 0 / 9 / 2991 | 0 | 514.5 +- 9.4 | 1.029 +- 0.019 |

Label check (same-module label-exchange test of `hnl.eval.dense_jet_check` through `scripts/acts/sample_stats.py`, events 0-499 of each sample read through the tar reader; flagged pairs with transverse separation > 2 mm):

| sample | same module | different modules of one layer (control) | chains monotone in sqrt(r^2+z^2) |
|---|---|---|---|
| `hnl10_100_ref` | 0.33 % (511 / 157,154) | 0.07 % (2,697 / 3,855,334) | 98.6 % |
| `hnl5_50` | 0.34 % (560 / 165,771) | 0.08 % (3,141 / 4,136,676) | 98.9 % |
| `hnl20_200` | 0.32 % (488 / 153,724) | 0.07 % (2,808 / 3,778,092) | 98.7 % |
| `hnl15_200_ref` | 0.31 % (503 / 160,034) | 0.07 % (2,725 / 3,841,328) | 99.0 % |
| `hnl15_500` | 0.34 % (558 / 163,623) | 0.07 % (2,950 / 4,016,120) | 98.7 % |
| Zenodo hnl10_100 val (reference, first 500 of the val list) | 14.25 % (23,539 / 165,227) | 0.40 % (16,569 / 4,141,911) | 94.0 % |

## Populations

Targets: pT > 1 GeV, |eta| < 4, >= 5 hits; primary population: N-daughter targets with fitted |d0| >= 10 mm or fitted r_v >= 100 mm (`docs/definitions.md` sections 3-4; truth fits of `hnl.truth.fit_events`, as in the closure census). Expected: spec section 4. Zenodo test sets for scale: 603 (hnl10_100), 549 (hnl15_200).

| sample | kept events | N-daughter targets | primary population | per event | expected (spec) | of which r_v >= 300 mm | of which abs(d0) >= 100 mm | with >= 3 pixel hits | W-muon targets |
|---|---|---|---|---|---|---|---|---|---|
| `hnl10_100_ref` | 3,000 | 2,868 | 1,966 | 0.655 +- 0.015 | 1,880 | 282 | 223 | 550 | 2,726 |
| `hnl5_50` | 3,000 | 2,793 | 1,714 | 0.571 +- 0.015 | 1,635 | 299 | 43 | 330 | 2,726 |
| `hnl20_200` | 3,000 | 2,715 | 1,967 | 0.656 +- 0.015 | 2,350 | 180 | 346 | 705 | 2,748 |
| `hnl15_200_ref` | 3,000 | 2,403 | 1,800 | 0.600 +- 0.015 | 1,970 | 212 | 254 | 565 | 2,738 |
| `hnl15_500` | 3,000 | 1,290 | 1,051 | 0.350 +- 0.012 | 1,136 | 192 | 262 | 277 | 2,737 |

N-daughter targets (primary in parentheses) by fitted r_v (mm):

| r_v | `hnl10_100_ref` | `hnl5_50` | `hnl20_200` | `hnl15_200_ref` | `hnl15_500` |
|---|---|---|---|---|---|
| [0, 10) | 213 (0) | 174 (0) | 201 (0) | 144 (0) | 44 (0) |
| [10, 30) | 363 (80) | 286 (19) | 398 (144) | 283 (91) | 94 (22) |
| [30, 100) | 840 (479) | 863 (270) | 778 (560) | 712 (489) | 343 (245) |
| [100, 200) | 643 (643) | 678 (678) | 582 (582) | 575 (575) | 286 (286) |
| [200, 300) | 385 (385) | 415 (415) | 303 (303) | 287 (287) | 186 (186) |
| >= 300 | 282 (282) | 299 (299) | 180 (180) | 212 (212) | 192 (192) |
| no vertex | 142 (97) | 78 (33) | 273 (198) | 190 (146) | 145 (120) |

N-daughter targets (primary in parentheses) by fitted |d0| (mm):

| abs(d0) | `hnl10_100_ref` | `hnl5_50` | `hnl20_200` | `hnl15_200_ref` | `hnl15_500` |
|---|---|---|---|---|---|
| [0, 0.1) | 36 (4) | 47 (4) | 29 (2) | 20 (1) | 9 (0) |
| [0.1, 1) | 207 (18) | 300 (46) | 141 (19) | 124 (11) | 46 (3) |
| [1, 3) | 294 (51) | 418 (99) | 197 (20) | 183 (28) | 67 (17) |
| [3, 10) | 601 (163) | 757 (294) | 495 (73) | 398 (82) | 184 (47) |
| [10, 30) | 790 (790) | 782 (782) | 673 (673) | 651 (651) | 291 (291) |
| [30, 100) | 717 (717) | 446 (446) | 834 (834) | 773 (773) | 431 (431) |
| [100, 300) | 210 (210) | 43 (43) | 316 (316) | 232 (232) | 228 (228) |
| >= 300 | 13 (13) | 0 (0) | 30 (30) | 22 (22) | 34 (34) |

Generator truth of the N (all kept events): decay-radius fractions, median opening angle of the pair (end-state momenta) and median |d0| of the daughter targets:

| sample | r_v < 100 mm | 100-300 mm | 300-700 mm | > 1 m | median opening angle (rad) | median abs(d0) of daughter targets (mm) |
|---|---|---|---|---|---|---|
| `hnl10_100_ref` | 0.338 | 0.315 | 0.234 | 0.053 | 0.151 | 14.9 |
| `hnl5_50` | 0.334 | 0.323 | 0.235 | 0.052 | 0.074 | 8.5 |
| `hnl20_200` | 0.347 | 0.334 | 0.224 | 0.042 | 0.263 | 23.5 |
| `hnl15_200_ref` | 0.275 | 0.295 | 0.260 | 0.082 | 0.232 | 23.6 |
| `hnl15_500` | 0.127 | 0.188 | 0.225 | 0.353 | 0.226 | 34.3 |

## Statistical reach

Binomial standard error of an efficiency on the primary population (events are independent; the two daughters of an event can be correlated, so these are lower bounds), and of the difference between a new point and its in-stack reference (independent samples). A comparison of two models on the same events is paired and has a smaller error.

| sample | primary population | SE at eff 0.90 (points) | SE at eff 0.80 | reference | SE of the difference at 0.90 / 0.80 |
|---|---|---|---|---|---|
| `hnl10_100_ref` | 1,966 | 0.68 | 0.90 | - | - |
| `hnl5_50` | 1,714 | 0.72 | 0.97 | `hnl10_100_ref` | 0.99 / 1.32 |
| `hnl20_200` | 1,967 | 0.68 | 0.90 | `hnl10_100_ref` | 0.96 / 1.28 |
| `hnl15_200_ref` | 1,800 | 0.71 | 0.94 | - | - |
| `hnl15_500` | 1,051 | 0.93 | 1.23 | `hnl15_200_ref` | 1.16 / 1.55 |

## Files

- `hnl10_100_ref`: `/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_10_100_v1/SHA256SUMS` (copy: `results/hnl_points_truth_v1/hnl_10_100_v1.sha256`); kept events: `splits/hnl_10_100_all.txt`.
- `hnl5_50`: `/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_5_50_v1/SHA256SUMS` (copy: `results/hnl_points_truth_v1/hnl_5_50_v1.sha256`); kept events: `splits/hnl_5_50_all.txt`.
- `hnl20_200`: `/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_20_200_v1/SHA256SUMS` (copy: `results/hnl_points_truth_v1/hnl_20_200_v1.sha256`); kept events: `splits/hnl_20_200_all.txt`.
- `hnl15_200_ref`: `/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_15_200_v1/SHA256SUMS` (copy: `results/hnl_points_truth_v1/hnl_15_200_v1.sha256`); kept events: `splits/hnl_15_200_all.txt`.
- `hnl15_500`: `/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_15_500_v1/SHA256SUMS` (copy: `results/hnl_points_truth_v1/hnl_15_500_v1.sha256`); kept events: `splits/hnl_15_500_all.txt`.
## Production record and remarks

The sections above are written by `scripts/acts/hnl_points_check.py summary`; this section by hand from the job logs
and Slurm accounting.

- **Production**: after the spec changes were committed (b73121e), arrays 6819868 (hnl5_50), 6819869 (hnl15_500),
  6819870 (hnl20_200), 6819871 (hnl10_100_ref) and 6819872 (hnl15_200_ref), six 500-event chunks each, all
  COMPLETED, 49-107 s per chunk on 4 CPUs, CPU only (A30 and GTX 1080 nodes excluded); 2.9 CPU-h allocated. The
  converter logs give 20,756,395 measurements (1,384 per event), 0 dropped duplicate hits and 16 measurements with
  |pull| > 5 over the 15,000 events. Pythia's `next()` never needed a second call.
- **Checks**: scans 6819912, 6819921, 6819923, 6819925, 6819927; census (truth fits) 6819913, 6819922, 6819924,
  6819926, 6819928; Zenodo label-check reference 6819914; summary 6819932. No event is dropped: every event passes the
  reader and format invariants, and the largest excess over a module's padded envelope is 7.4 mm (rule: 50 mm).
- **Storage**: 1.78 GB of shards (118-121 kB per event) and 0.21 GB of sidecars. The `hnl_lrt` area held 66,309
  files before production and 66,401 after (2026-10-14 19:13 UTC; this production adds 65: 30 shards, 30 sidecars,
  5 `SHA256SUMS`; the rest came from concurrent jobs), far below the 80,000 stop line; about 155 GB remain under the
  1 TB quota.
- **W muon not in the particle table** in one event of hnl10_100_ref (324) and one of hnl20_200 (2619): the
  generator recorded it, the simulated table does not hold it. The 4,000-event closure pilot had one such event too.
  These events are kept; the W muon is only the prompt reference.
- **Primary population against the plan.** hnl20_200 holds 1,967 primary daughters where the generator-level estimate
  of spec section 4 gave 2,350 (-16 %; the estimate had been checked to 7 % on the existing points only); the other
  points are within 9 % of their estimates. Every sample holds at least 1.7 times the 603 of the hnl10_100 test set.
- **Reference points against the closure.** Primary daughters per event: hnl10_100_ref 0.655 +- 0.015, closure pilot
  0.642 +- 0.013, Zenodo train + val 0.638 +- 0.009; hnl15_200_ref 0.600 +- 0.015, 0.615 +- 0.013 and 0.595 +- 0.009.
- **Lifetimes.** Each sample's mean proper lifetime is within 1.6 standard errors of c tau0 (hnl15_500 +1.5,
  hnl20_200 and hnl15_200_ref -0.9); the card's own c tau is given with 0.2-0.3 % precision in spec section 10.
- **Labels.** The same-module exchange rate is 0.31-0.34 % in every sample against 14.25 % on Zenodo hnl10_100 val
  events, with the different-module control at 0.07-0.08 % against 0.40 %; our labels carry no exchange beyond the
  geometric false-positive rate of the test (the generic samples gave 0.24-0.39 %, `results/new_samples_truth_v1.md`).
