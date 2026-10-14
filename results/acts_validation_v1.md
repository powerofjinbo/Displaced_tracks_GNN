# ACTS v15 stack against the Zenodo ttbar sample (truth only)

Validation of the simulation stack of `docs/new_samples_spec.md` (2026-10-14). No model is run. Inputs: 1,000 ttbar
events simulated here (`slurm/acts/produce.sbatch`, SAMPLE=ttbar_val, jobs 6810641-6810642; kept on /work only) and
the first 1,000 events of the Zenodo ttbar training split (`splits/ttbar_train.txt`; job 6810623). Statistics:
`scripts/acts/sample_stats.py`; table: `scripts/acts/compare_stats.py`.

## Build

ACTS v15.0.0 (0fef9e0), Fatras, examples, Pythia8 244 and Python bindings, built in `atlas/centos7-atlasos:latest-gcc8`
with the `LCG_97apython3/x86_64-centos7-gcc9-opt` view (job 6810588, 408 build steps, 10 min on 5 cores). The bare
`centos:centos7` image lacks glibc-devel and make, and the LCG gcc setup calls `which`, so the build uses the ATLAS
CentOS 7.9 OS image (compilers and libraries still from the LCG view). The Python bindings must keep the detector
object alive while its tracking geometry is used (`scripts/acts/acts_sim.py`).

## Geometry

`detectors.csv` of the v15 GenericDetector has 18,728 sensitive surfaces (volumes 7-9, 12-14, 16-18, module ids up to
3192). All 18,636 distinct geometry ids of 300 Zenodo ttbar training events are among them, in the same (pre-v21) bit
layout.

## Truth-level comparison

| quantity | ACTS v15 (ours) (1000 events) | Zenodo ttbar (1000 events) | difference / sigma |
|---|---|---|---|
| hits per event | 2038.9 +- 21.9 | 2041.5 +- 21.4 | -0.1 |
| particle-table rows per event | 491.7 +- 5.0 | 492.1 +- 4.9 | -0.1 |
| particles with hits per event | 200.1 +- 2.2 | 201.3 +- 2.1 | -0.4 |
| primaries with hits per event | 168.1 +- 1.8 | 169.4 +- 1.8 | -0.5 |
| secondaries with hits per event | 32.0 +- 0.4 | 31.9 +- 0.4 | +0.3 |
| secondary fraction among particles with hits | 16.01 % | 15.85 % | |
| true edges pointing outward in r | 99.32 % | 94.94 % | |
| true edges pointing outward in sqrt(r^2+z^2) | 99.77 % | 98.75 % | |
| chains monotone in sqrt(r^2+z^2) | 98.06 % | 90.68 % | |
| true edges within one module (total) | 0 | 0 | |
| particles with two hits on one module (total) | 0 | 0 | |
| edges minus (hits - particles with hits) (total) | 0 | 0 | |
| mean hits per particle with hits | 10.19 | 10.14 | KS 0.005 |
| pT of particles with hits: shape | | | KS 0.008 |
| pT of primaries with hits: shape | | | KS 0.009 |
| eta of particles with hits: shape | | | KS 0.003 |
| p of primaries with hits, 0-2 GeV: shape | | | KS 0.005 |
| z residual of barrel short-strip hits, sigma_MAD (mm) | 0.378 (44865) | 4.513 (44759) | |
| z residual of barrel long-strip hits, sigma_MAD (mm) | 3.134 (14478) | 9.426 (14184) | |

Hits per (volume, layer), share of all hits:

| volume_layer | ACTS v15 (ours) | Zenodo ttbar |
|---|---|---|
| 7_2 | 1.89 % | 1.91 % |
| 7_4 | 1.96 % | 1.98 % |
| 7_6 | 2.05 % | 2.06 % |
| 7_8 | 2.13 % | 2.13 % |
| 7_10 | 2.22 % | 2.20 % |
| 7_12 | 2.32 % | 2.29 % |
| 7_14 | 2.47 % | 2.44 % |
| 8_2 | 8.89 % | 8.93 % |
| 8_4 | 6.51 % | 6.56 % |
| 8_6 | 5.46 % | 5.52 % |
| 8_8 | 4.55 % | 4.60 % |
| 9_2 | 2.60 % | 2.62 % |
| 9_4 | 2.55 % | 2.53 % |
| 9_6 | 2.47 % | 2.43 % |
| 9_8 | 2.39 % | 2.33 % |
| 9_10 | 2.33 % | 2.25 % |
| 9_12 | 2.26 % | 2.17 % |
| 9_14 | 2.19 % | 2.12 % |
| 12_2 | 1.11 % | 1.07 % |
| 12_4 | 1.15 % | 1.12 % |
| 12_6 | 1.20 % | 1.17 % |
| 12_8 | 1.23 % | 1.21 % |
| 12_10 | 1.25 % | 1.24 % |
| 12_12 | 1.31 % | 1.32 % |
| 13_2 | 5.64 % | 5.70 % |
| 13_4 | 4.86 % | 4.88 % |
| 13_6 | 4.01 % | 4.01 % |
| 13_8 | 3.26 % | 3.26 % |
| 14_2 | 1.34 % | 1.36 % |
| 14_4 | 1.36 % | 1.39 % |
| 14_6 | 1.36 % | 1.38 % |
| 14_8 | 1.34 % | 1.36 % |
| 14_10 | 1.33 % | 1.34 % |
| 14_12 | 1.31 % | 1.31 % |
| 16_2 | 0.38 % | 0.40 % |
| 16_4 | 0.38 % | 0.40 % |
| 16_6 | 0.38 % | 0.40 % |
| 16_8 | 0.39 % | 0.39 % |
| 16_10 | 0.39 % | 0.39 % |
| 16_12 | 0.38 % | 0.38 % |
| 17_2 | 2.61 % | 2.62 % |
| 17_4 | 2.22 % | 2.21 % |
| 18_2 | 0.39 % | 0.39 % |
| 18_4 | 0.42 % | 0.42 % |
| 18_6 | 0.43 % | 0.44 % |
| 18_8 | 0.44 % | 0.46 % |
| 18_10 | 0.45 % | 0.46 % |
| 18_12 | 0.44 % | 0.45 % |

primaries with hits: share by |PDG| (or code), top of Zenodo ttbar:

| code | ACTS v15 (ours) | Zenodo ttbar |
|---|---|---|
| 211 | 80.26 % | 80.38 % |
| 321 | 11.75 % | 11.48 % |
| 2212 | 4.89 % | 5.08 % |
| 11 | 1.13 % | 1.06 % |
| 3112 | 0.61 % | 0.67 % |
| 3222 | 0.67 % | 0.64 % |
| 13 | 0.51 % | 0.50 % |
| 3312 | 0.17 % | 0.18 % |

secondaries with hits: share by |PDG| (or code), top of Zenodo ttbar:

| code | ACTS v15 (ours) | Zenodo ttbar |
|---|---|---|
| 11 | 100.00 % | 100.00 % |

process code of particles with hits: share by |PDG| (or code), top of Zenodo ttbar:

| code | ACTS v15 (ours) | Zenodo ttbar |
|---|---|---|
| 0 | 83.99 % | 84.15 % |
| 2 | 16.01 % | 15.85 % |

Label-exchange test (hnl.eval.dense_jet_check): exchanged / tested pairs by transverse separation (mm):

| pairs | 0-1 | 1-2 | 2-5 | 5-10 | 10-20 | 20-50 | 50-100000 |
|---|---|---|---|---|---|---|---|
| ACTS v15 (ours), same module | 0.7 % (573/83394) | 1.4 % (1047/73524) | 0.5 % (839/182779) | 0.1 % (291/215886) | 0.2 % (324/189133) | 0.2 % (302/183968) | 0.3 % (203/59058) |
| ACTS v15 (ours), different modules of one layer (control) | 1.0 % (1686/173692) | 1.8 % (4705/261179) | 0.5 % (5016/930074) | 0.2 % (2607/1468648) | 0.0 % (1545/3307891) | 0.0 % (850/11478707) | - |
| Zenodo ttbar, same module | 1.7 % (1338/78529) | 6.5 % (4596/71133) | 10.3 % (18510/179632) | 14.6 % (30576/210051) | 15.6 % (28712/183473) | 15.3 % (27480/179506) | 14.5 % (8418/58220) |
| Zenodo ttbar, different modules of one layer (control) | 0.5 % (1186/227075) | 2.0 % (5664/276992) | 2.4 % (22038/921058) | 2.3 % (33698/1450686) | 1.2 % (37601/3267453) | 0.2 % (20086/11457090) | - |


## Reading

- **Agreement.** Hits, particle-table rows, particles with hits, primaries and secondaries per event agree within 0.5
  standard errors of the mean; the shapes of pT, eta, hits per particle and the momentum of primaries near the 0.5 GeV
  threshold agree (Kolmogorov distance at most 0.009); the share of hits per (volume, layer) agrees to about 0.1
  point in every layer; PDG mix of primaries and secondaries, the secondary fraction (16.0 % against 15.9 %, all
  electrons) and the process codes agree. The ttbar settings (14 TeV, generator decays only for c tau0 < 10 mm, Fatras
  EM processes on, pMin 0.5 GeV, beam spot sigma_z 55.5 mm) therefore reproduce the Zenodo sample.
- **Hit positions.** The Zenodo positions are smeared measurements. A Gaussian plus a flat term fitted to the z
  residuals of barrel strip hits (above) gives a Gaussian width of 0.414 mm (short strips) and 3.143 mm (long strips)
  in the Zenodo ttbar and 0.391 mm and 3.141 mm in ours, the smearing of the digitisation config (0.346 and 3.12 mm in
  loc1, widened by the extrapolation); simulated-hit (truth) positions would give sigma_MAD of 0.06 mm and 0.39 mm
  (5 test events, `--x truth`). The new samples use measured positions. In the Zenodo sample the flat term is large
  (next item); in ours it is zero.
- **Labels (the coordinator's swap test).** On the same module, the label-exchange test of hnl.eval.dense_jet_check
  flags 0.1-1.4 % of pairs in our sample, at the level of its own different-module control (0.0-1.8 %), against
  10-16 % above 2 mm in the Zenodo ttbar. The exchange shows up in other truth quantities only in the Zenodo files:
  36 % of the barrel short-strip hits and 44 % of the long-strip hits of primaries with pT > 5 GeV lie more than 10 mm
  in z from the straight r-z line through the particle's pixel hits (ours: 0.0 % and 0.2 %); 9 % of their true-edge
  chains are not monotone in sqrt(r^2+z^2) (ours: 2 %); 5.1 % of their true edges point inward in r (ours: 0.7 %).
- **Where the exchange is not.** In v15 the smearing digitisation (no merging) creates one measurement per simulated
  hit, module by module in the order of the simulated-hit container, so `measurement-simhit-map.csv` is the identity
  in every event checked (20 events), and `hits.csv` is already in geometry-id order: re-sorting it by geometry id
  with quicksort, mergesort or heapsort and joining by row gives 0 wrong labels in 39,844 hits. The simulation,
  digitisation and CSV writers of v15 therefore do not produce the exchange; it must arise in the step that built the
  Zenodo npz files from the ACTS output (or in a different stack version), which we cannot inspect. Our converter takes
  each hit's particle from the measurement-simhit map. The new samples thus carry correct labels, whereas the frozen
  models were trained and selected on the Zenodo labels; this difference is part of the new samples and is stated
  wherever their results are reported.

## Gun and h -> s s pilots (test directories, not part of the samples)

40 events of each generic component and of h -> s s (m_s = 40 GeV): no particle with two hits on one module, pulls of
the smeared measurements with unit width and none beyond 5 sigma. Fatras propagates particles that start inside the
tracker: of the gun particles with pT > 2 GeV, |eta| < 0.8, moving outward from r_prod < 600 mm (67), none has zero
hits, and their first hit lies outside r_prod. The h -> s s pilot has about 13 charged particles per event with
pT > 1 GeV produced beyond 10 mm.
