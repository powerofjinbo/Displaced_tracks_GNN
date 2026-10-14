# New samples: truth-level summary

Produced per `docs/new_samples_spec.md`; `scripts/acts/gun_truth_stats.py`. Truth only: no model, no split, no addendum-G target selection.

## generic_iso_v1

- events: 8000; measurements per event: 2082 (sd 673); dropped duplicate hits: 0; measurements with |pull| > 5: 14
- gun particles: 32000 (4.00 per event); with >= 1 hit 0.887; with >= 5 hits and pT > 1 GeV 0.365
- species: 11 0.196, 13 0.202, 211 0.203, 321 0.201, 2212 0.198
- pT quantiles 10/50/90 %: 0.47 / 3.36 / 27.23 GeV

| r_v (mm) | gun particles | mean hits | >= 5 hits and pT > 1 GeV |
|---|---|---|---|
| 0-10 | 404 | 10.7 | 0.725 |
| 10-100 | 4098 | 9.2 | 0.726 |
| 100-200 | 4600 | 6.2 | 0.636 |
| 200-300 | 4586 | 4.9 | 0.533 |
| 300-450 | 6250 | 3.7 | 0.339 |
| 450-650 | 7294 | 2.8 | 0.117 |
| 650-800 | 4768 | 1.9 | 0.017 |

| abs(d0) (mm), helix in 2 T | gun particles with >= 5 hits and pT > 1 GeV |
|---|---|
| 0-1 | 638 |
| 1-10 | 3000 |
| 10-30 | 2893 |
| 30-100 | 3223 |
| 100-300 | 1631 |
| 300-1e+09 | 302 |

## generic_dense_v1

- events: 8000; measurements per event: 2102 (sd 689); dropped duplicate hits: 0; measurements with |pull| > 5: 17
- gun particles: 88076 (11.01 per event); with >= 1 hit 0.906; with >= 5 hits and pT > 1 GeV 0.366
- species: 11 0.051, 13 0.049, 211 0.699, 321 0.150, 2212 0.050
- pT quantiles 10/50/90 %: 0.37 / 2.68 / 19.60 GeV

| r_v (mm) | gun particles | mean hits | >= 5 hits and pT > 1 GeV |
|---|---|---|---|
| 0-10 | 1167 | 11.0 | 0.695 |
| 10-100 | 11347 | 9.6 | 0.713 |
| 100-200 | 12146 | 6.5 | 0.625 |
| 200-300 | 12680 | 5.1 | 0.545 |
| 300-450 | 17499 | 4.0 | 0.359 |
| 450-650 | 20598 | 3.0 | 0.114 |
| 650-800 | 12639 | 1.9 | 0.015 |

| abs(d0) (mm), helix in 2 T | gun particles with >= 5 hits and pT > 1 GeV |
|---|---|
| 0-1 | 1828 |
| 1-10 | 8460 |
| 10-30 | 8317 |
| 30-100 | 9292 |
| 100-300 | 3830 |
| 300-1e+09 | 493 |

## hss15_100_v1

- events: 1500; measurements per event: 1649 (sd 702); dropped duplicate hits: 0; measurements with |pull| > 5: 7

| production radius (mm) | charged primaries with hits per event | with pT > 1 GeV | electrons |
|---|---|---|---|
| 0-1 | 122.44 | 30.50 | 0.76 |
| 1-10 | 3.31 | 0.68 | 0.09 |
| 10-100 | 9.81 | 3.44 | 0.47 |
| 100-300 | 6.52 | 4.32 | 0.54 |
| 300-600 | 3.25 | 2.45 | 0.31 |
| 600-1100 | 1.49 | 1.23 | 0.15 |

## hss40_300_v1

- events: 1500; measurements per event: 1705 (sd 712); dropped duplicate hits: 0; measurements with |pull| > 5: 2

| production radius (mm) | charged primaries with hits per event | with pT > 1 GeV | electrons |
|---|---|---|---|
| 0-1 | 124.87 | 30.94 | 0.81 |
| 1-10 | 3.52 | 0.81 | 0.06 |
| 10-100 | 11.63 | 4.00 | 0.45 |
| 100-300 | 7.66 | 4.49 | 0.49 |
| 300-600 | 4.32 | 2.83 | 0.30 |
| 600-1100 | 1.43 | 1.01 | 0.11 |


## Label check (swap test of hnl.eval.dense_jet_check; events 0-499 of each sample; `scripts/acts/sample_stats.py`)

| sample | same-module pairs flagged, separation > 2 mm | different-module control, > 2 mm | chains monotone in sqrt(r^2+z^2) |
|---|---|---|---|
| generic_iso_v1 | 0.24 % (992 / 418,475) | 0.06 % | 98.4 % |
| generic_dense_v1 | 0.26 % (1,094 / 428,930) | 0.06 % | 98.5 % |
| hss40_300_v1 | 0.39 % (1,058 / 269,894) | 0.08 % | 98.8 % |
| Zenodo ttbar (reference, results/acts_validation_v1.md) | 10-16 % per bin | 0.2-2.4 % | 90.7 % |

## Reader check

`scripts/acts/check_reader.py` (job 6810858): all 19,000 events load with `hnl.data.npz.load_event(check=True)`
(keys and dtypes of the Zenodo files); hid is a permutation, every pid is in the particle table, ids are exact in
float64, every particle's hits form one chain of true edges between different modules, and E = hits - particles with
hits. Production: arrays 6810705-6810708 (38 chunks of 500 events, all completed); one ttbar positron in
generic_iso_v1 event 776 failed to propagate (Fatras step limit) and has no hits; Pythia regenerated events after
hadronisation errors (20 times; Pythia generates a new event).
