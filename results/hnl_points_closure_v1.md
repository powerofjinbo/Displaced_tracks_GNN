# HNL generator against the Zenodo HNL samples (truth only)

Written 2026-10-14 for the new HNL points (`docs/hnl_points_spec.md`), before any event of those points was produced.
No model was run. The Zenodo events read are the train and val splits of `hnl10_100` and `hnl15_200`; no test-split
event was read for this closure. Amended the same day, before any production event existed: section 5 (the ">= 3 pixel
hits" difference is the Zenodo pixel-layer artifact) and section 6 (lifetime check at 20 GeV / 200 mm). All tables below
are produced by `scripts/acts/hnl_closure.py` from the censuses of `scripts/acts/hnl_census.py`; machine-readable copies
are `results/hnl_points_closure_v1.csv` (every test of every pilot variant), `_summary.json` (per point, with the Zenodo
truth facts of section 2), `_genvar.csv` and `_driver_vs_acts.csv`.

## 1. Inputs and method

- **Zenodo**: 8,999 train + val events per sample (7,999 + 1,000 for `hnl10_100`, 8,000 + 999 for `hnl15_200`),
  census jobs 6812301 and 6812302. The truth fits (d0, r_v) are those of `derived/truth_fits/v1`; refitting the first
  200 events with the same function (`hnl.truth.fit_events.fit_event`) reproduces them to 2e-6 mm. Event i of
  `hnl10_100` and of `hnl15_200` share the hard process and underlying event (`docs/brief.md` section 2.6), so the
  W-muon and multiplicity numbers of the two samples are one measurement, not two.
- **Pilots**: our generator (`scripts/acts/hnl_card.py`, `scripts/acts/hnl_pythia.cc`), then the ACTS v15 Fatras
  simulation and smearing digitisation of `docs/new_samples_spec.md` section 2, converted to Zenodo-format npz in tar
  shards (`scripts/acts/hnl_to_npz.py`, `slurm/acts/produce_hnl.sbatch`). Pilot events live on scratch
  (`/work/submit/jinboz1/hnl/samples_v2/hnl_points/pilot/`), use seeds above 990,000 and are never used for anything
  else. Our files carry the generator parent codes, so both sides use the same labels (|parent| = 50: N daughter,
  |parent| = 24 and |PDG| = 13: W muon).
- **Tests** (pilot against Zenodo, per point): means with combined standard errors, binomial fractions, two-sample
  Kolmogorov-Smirnov tests for continuous shapes and chi-square tests for hit counts; 60 tests per point. Two of them
  have a known cause outside the generator (section 5) and are shown but not counted. A setting closes when no
  counted test has p < 0.001 and the number with p < 0.01 is compatible with chance.
- **Generator-level variants** (5,000 events each, no simulation) separate the N decay matrix element, the beam
  energy and the initial-state shower; they are compared with Zenodo only through quantities the 2 T field leaves
  unchanged between production and the end state stored in the Zenodo tables (|p|, pT, eta).

## 2. What the Zenodo truth establishes

From the particle tables of the 17,998 train and val events (`_summary.json`, key `zenodo_truth`):

- **One W muon per event.** Every event holds exactly one muon whose parent code is +-24, at generator vertex 1 and
  alone there among the muons. The W is a W+ in 5,217 of 8,999 events (58.0 %). No other particle has parent +-24
  (no photon either).
- **N -> mu+ mu- (nu), always.** Every particle with parent +-50 is a muon (17,911 and 17,948). 8,912 / 8,949 events
  have two of them, the other 87 / 50 one; every pair has opposite charges, shares one generator vertex, and no other
  generator particle sits at that vertex (0 events). Together with the paper (Eq. 1, N -> nu_mu mu+ mu-) and the
  Zenodo record ("forced to decay to two muons and a muon neutrino"), the decay-mode mix is 100 % mu mu nu.
- **N has a distinct antiparticle tied to the W charge**: the parent code of the daughters is +50 in every W+ event
  and -50 in every W- event (8,999 of 8,999).
- **Hadron decays**: no row with PDG 310 or +-3122, but 9.94 K_S daughters and 5.36 Lambda daughters per event
  (parent codes 310, 3122) and 8.18 K_L rows: the generator decayed every particle with c tau0 < 1 m (Pythia's
  default), unlike in the Zenodo ttbar sample, whose tables hold K_S and Lambda.
- **Parent codes**: column 12 is the PDG code of the generator mother, followed up through radiating copies (W muon:
  +-24); particles created in the simulation carry 0 (0 exceptions). Photon parents: pi0 1,074,550, eta 62,011, and
  muons 1,161 (final-state radiation of the muons).
- **Vertex time**: the W-muon end times give a production-time spread of 4.90 / 4.91 ns (4,240 muons ending at the
  barrel boundary, path length / c subtracted), the 5 ns default of the ACTS v15 Pythia8 example; the transverse and
  longitudinal beam-spot widths are the ACTS defaults (W-muon z0 spread 54.8 mm against 54.8 mm in our pilot).
- **Event content**: 1,362 hits, 364 table rows and 138 particles with hits per event.
- **End-state momenta**: the tables hold momenta at the end of the simulation, whose azimuth has turned in the field.
  m(mu mu) computed from them exceeds m_N in 218 / 180 pairs, 7.1 / 5.0 % of the pairs with a daughter that ends
  beyond r = 1.1 m and 0.25 / 0.02 % of the others. Every comparison of angles and masses below is therefore made
  with our simulated end state, not with generator momenta.

The paper states m_N and c tau (15 GeV, 100 mm) and calls N "a right-handed Majorana neutrino"; the record states
ACTS v15, the generic detector, a constant 2 T field, Fatras, smearing digitisation and Pythia8 at 13 TeV. Neither
gives the W production process, the N decay matrix element, the Pythia version or tune, or the shower settings. These
are inferred in section 3.

## 3. Settings inferred by the closure

| setting | inferred value | evidence |
|---|---|---|
| W production | f fbar' -> W, W forced to mu N | one W muon per event, its pT Jacobian peak (median 30.0 GeV) |
| N decay matrix element | V-A, half lepton-number conserving (N -> mu- W+*), half violating (N -> mu+ W-*) | phase space gives m(mu mu)/m_N of 0.550-0.568 against 0.528-0.539 (z = +3.5 to +5.0, four pilots); a single V-A form gives a ratio of the mean daughter momenta (charge opposite to the W muon over same charge) of 0.86-0.87 or 1.15-1.19 against 0.98 +- 0.03 and 0.97 +- 0.03; the 50/50 sum matches both |
| beam energy | 14 TeV | with the default tune at 13 TeV the table rows per event are 2.4-4.5 % low (z = -2.3 to -5.3, four pilots); at 14 TeV z = 0.0 to +1.0 (six pilots) |
| initial-state shower | emissions up to the factorization scale (SpaceShower:pTmaxMatch = 1) | Pythia's default for a hard process without final-state partons (emissions up to the kinematic limit) gives a harder W-muon pT tail: 99 % quantile 72.5 GeV against 60.8 GeV at generator level, KS p 0.0016 and 0.007 in the full-simulation pilots; with the limit 62.7 GeV and KS p 0.42 and 0.16 |
| tune, PDF | Pythia 8.244 defaults (Monash 2013, NNPDF2.3 QCD+QED LO) | the A14 tune at 13 TeV matches the multiplicity and the W-muon acceptance but gives more daughter targets than Zenodo in hnl15_200 (0.420 against 0.400 per daughter; 6 tests at p < 0.01, 2 at p < 0.001) |

The Zenodo record's 13 TeV is not confirmed: no 13 TeV setting tried reproduces both the multiplicity and the
daughter acceptance. The ttbar validation of `results/acts_validation_v1.md` also matched the Zenodo ttbar at
14 TeV.

## 4. Results

Every full-simulation pilot against Zenodo (counted tests; z of the difference, or the KS p-value):

| variant | point | events | p < 0.01 | p < 0.001 | rows / event (z) | m(mu mu)/m_N (z) | W-muon pT KS p | daughter pT KS p | W muon >= 5 hits (z) | daughter targets >= 3 pixel hits (z) |
|---|---|---|---|---|---|---|---|---|---|---|
| phase space, 13 TeV | hnl10_100 | 2,000 | 10 | 6 | -2.3 | +3.5 | 0.0082 | 0.00016 | -0.7 | -1.2 |
| phase space, 13 TeV | hnl15_200 | 2,000 | 9 | 4 | -2.6 | +5.0 | 0.15 | 0.016 | -1.1 | +3.0 |
| Majorana V-A, 13 TeV | hnl10_100 | 4,000 | 12 | 6 | -2.9 | +0.4 | 0.011 | 0.25 | -0.2 | +1.8 |
| Majorana V-A, 13 TeV | hnl15_200 | 4,000 | 17 | 11 | -5.3 | +0.6 | 0.0032 | 0.041 | -1.5 | +3.2 |
| phase space, 14 TeV | hnl10_100 | 4,000 | 10 | 6 | +0.3 | +5.0 | 0.0069 | 0.00014 | -3.3 | +2.3 |
| phase space, 14 TeV | hnl15_200 | 4,000 | 6 | 6 | +0.4 | +4.6 | 0.025 | 4.2e-05 | -2.1 | +1.9 |
| Majorana V-A, 14 TeV | hnl10_100 | 4,000 | 6 | 3 | +0.6 | -0.0 | 0.0016 | 0.023 | -1.4 | +3.8 |
| Majorana V-A, 14 TeV | hnl15_200 | 4,000 | 9 | 3 | +1.0 | -0.2 | 0.007 | 0.042 | -3.1 | +4.6 |
| Majorana V-A, 13 TeV, A14 tune, ISR up to mu_F | hnl10_100 | 4,000 | 2 | 0 | +0.7 | +0.3 | 0.066 | 0.2 | -1.3 | +2.9 |
| Majorana V-A, 13 TeV, A14 tune, ISR up to mu_F | hnl15_200 | 4,000 | 6 | 2 | +0.0 | +0.4 | 0.19 | 0.19 | -0.6 | +2.5 |
| Majorana V-A, 14 TeV, ISR up to mu_F (final card) | hnl10_100 | 4,000 | 1 | 0 | +0.5 | -0.2 | 0.42 | 0.14 | -2.1 | +3.1 |
| Majorana V-A, 14 TeV, ISR up to mu_F (final card) | hnl15_200 | 4,000 | 1 | 1 | +0.0 | +1.0 | 0.16 | 0.089 | -3.5 | +1.5 |

The final card in full (z for means and fractions, the KS or chi-square statistic otherwise; ** marks p < 0.01):

#### hnl10_100: pilot (4,000 events) against Zenodo train + val (8,999 events)

| group | quantity | test | pilot | Zenodo | z or statistic | p |
|---|---|---|---|---|---|---|
| event | n_hits | mean | 1,379 +- 11.161 | 1,362 +- 7.453 | 1.30 | 0.192 |
| event | n_hits | KS (median shown) | 1,344 | 1,320 | 0.02 | 0.145 |
| event | n_rows | mean | 365.6 +- 2.613 | 364.0 +- 1.743 | 0.49 | 0.628 |
| event | n_rows | KS (median shown) | 365.0 | 363.0 | 0.01 | 0.829 |
| event | n_phits | mean | 138.7 +- 1.114 | 138.1 +- 0.748 | 0.47 | 0.641 |
| event | n_phits | KS (median shown) | 135.0 | 134.0 | 0.02 | 0.371 |
| event | n_gen0_phits | mean | 119.9 +- 0.955 | 119.5 +- 0.643 | 0.35 | 0.728 |
| event | n_gen0_phits | KS (median shown) | 117.0 | 116.0 | 0.02 | 0.482 |
| event | n_sec_phits | mean | 18.803 +- 0.183 | 18.579 +- 0.120 | 1.02 | 0.306 |
| event | n_sec_phits | KS (median shown) | 18.000 | 17.000 | 0.02 | 0.219 |
| event | n_rows_k0s | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| event | n_rows_lambda | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| event | n_rows_k0l | mean | 8.385 +- 0.079 | 8.184 +- 0.051 | 2.14 | 0.0323 |
| event | n_parent_k0s | mean | 10.143 +- 0.101 | 9.943 +- 0.067 | 1.64 | 0.1 |
| event | n_parent_lambda | mean | 5.257 +- 0.063 | 5.355 +- 0.043 | -1.28 | 0.201 |
| event | exactly one W muon | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | -1.50 | 0.134 |
| event | W+ (prompt mu+) | fraction | 0.575 +- 0.008 | 0.580 +- 0.005 | -0.50 | 0.614 |
| event | two N daughters in the table | fraction | 0.993 +- 0.001 | 0.990 +- 0.001 | 1.35 | 0.176 |
| event | N daughters all muons | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| event | N daughters opposite charge | fraction | 0.993 +- 0.001 | 0.990 +- 0.001 | 1.35 | 0.176 |
| event | N sign = W sign | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | -1.50 | 0.134 |
| event | W muon alone at vertex 1 | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | -2.12 | 0.0339 |
| event | other particle at the N vertex | fraction | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| event | fitted N vertex (both daughters fitted) | fraction | 0.550 +- 0.008 | 0.540 +- 0.005 | 1.10 | 0.273 |
| event | r_v (fitted) [mm] | KS (median shown) | 118.3 | 114.9 | 0.02 | 0.667 |
| event | r_v (fitted) [mm] | mean | 164.0 +- 3.196 | 159.7 +- 2.088 | 1.13 | 0.258 |
| event | r_v >= 100 mm among fitted | fraction | 0.552 +- 0.011 | 0.543 +- 0.007 | 0.69 | 0.491 |
| event | m(mu mu) / m_N | KS (median shown) | 0.525 | 0.525 | 0.01 | 0.773 |
| event | m(mu mu) / m_N | mean | 0.538 +- 0.004 | 0.539 +- 0.003 | -0.19 | 0.85 |
| event | opening angle [rad] | KS (median shown) | 0.158 | 0.163 | 0.01 | 0.648 |
| event | Delta R of the pair | KS (median shown) | 0.697 | 0.707 | 0.02 | 0.516 |
| event | pair pT [GeV] | KS (median shown) | 18.303 | 18.142 | 0.02 | 0.0726 |
| event | daughter targets per event | mean | 0.927 +- 0.015 | 0.934 +- 0.010 | -0.36 | 0.721 |
| event | primary daughters per event | mean | 0.642 +- 0.013 | 0.638 +- 0.009 | 0.26 | 0.794 |
| W muon | pT [GeV] | KS (median shown) | 29.926 | 30.000 | 0.02 | 0.421 |
| W muon | eta | KS (median shown) | 0.009 | -0.021 | 0.01 | 0.766 |
| W muon | p [GeV] | KS (median shown) | 100.2 | 96.848 | 0.02 | 0.318 |
| W muon | pT [GeV] | mean | 29.179 +- 0.197 | 28.975 +- 0.131 | 0.86 | 0.388 |
| W muon | hits | chi2 (20 dof; means shown) | 11.455 +- 0.051 | 11.519 +- 0.033 | 1.12 | 0.318 |
| W muon | pixel hits | chi2 (15 dof; means shown) | 6.471 +- 0.050 | 6.448 +- 0.033 | 1.37 | 0.151 |
| W muon | >= 5 hits | fraction | 0.939 +- 0.004 | 0.948 +- 0.002 | -2.08 | 0.0376 |
| W muon | target (pT > 1, |eta| < 4, >= 5 hits) | fraction | 0.916 +- 0.004 | 0.923 +- 0.003 | -1.36 | 0.173 |
| W muon | |d0| of targets [mm] | KS (median shown) | 0.016 | 0.019 | 0.07 | 1.73e-10 (known) |
| W muon | first-hit radius of targets [mm] | KS (median shown) | 31.970 | 31.987 | 0.05 | 5.87e-05 (known) |
| W muon | >= 3 pixel hits among targets | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| N daughter | pT [GeV] | KS (median shown) | 7.951 | 7.917 | 0.02 | 0.137 |
| N daughter | eta | KS (median shown) | 0.010 | -0.006 | 0.01 | 0.724 |
| N daughter | p [GeV] | KS (median shown) | 31.423 | 31.079 | 0.01 | 0.609 |
| N daughter | pT [GeV] | mean | 10.253 +- 0.095 | 10.046 +- 0.061 | 1.84 | 0.0662 |
| N daughter | hits | chi2 (17 dof; means shown) | 4.939 +- 0.051 | 4.918 +- 0.034 | 1.18 | 0.272 |
| N daughter | pixel hits | chi2 (15 dof; means shown) | 1.605 +- 0.031 | 1.525 +- 0.020 | 1.27 | 0.209 |
| N daughter | >= 5 hits | fraction | 0.496 +- 0.006 | 0.497 +- 0.004 | -0.16 | 0.873 |
| N daughter | target (pT > 1, |eta| < 4, >= 5 hits) | fraction | 0.465 +- 0.006 | 0.469 +- 0.004 | -0.56 | 0.579 |
| N daughter | |d0| of targets [mm] | KS (median shown) | 15.344 | 15.404 | 0.01 | 0.899 |
| N daughter | first-hit radius of targets [mm] | KS (median shown) | 116.1 | 116.5 | 0.02 | 0.385 |
| N daughter | >= 3 pixel hits among targets | fraction | 0.483 +- 0.008 | 0.453 +- 0.005 | 3.08 | 0.00204 ** |
| N daughter | |d0| >= 10 mm among targets | fraction | 0.603 +- 0.008 | 0.601 +- 0.005 | 0.21 | 0.833 |
| N daughter | primary among targets | fraction | 0.692 +- 0.008 | 0.683 +- 0.005 | 1.00 | 0.319 |
| N daughter | p of the daughter with charge opposite to the W muon [GeV] | mean | 114.4 +- 3.522 | 107.2 +- 2.124 | 1.73 | 0.0831 |
| N daughter | p of the daughter with charge same to the W muon [GeV] | mean | 111.6 +- 3.336 | 109.4 +- 2.184 | 0.55 | 0.58 |

Tests 58 (without the 2 marked known), p < 0.01: 1 (chance: 0.6), p < 0.001: 0.

#### hnl15_200: pilot (4,000 events) against Zenodo train + val (8,999 events)

| group | quantity | test | pilot | Zenodo | z or statistic | p |
|---|---|---|---|---|---|---|
| event | n_hits | mean | 1,376 +- 11.306 | 1,361 +- 7.450 | 1.10 | 0.27 |
| event | n_hits | KS (median shown) | 1,328 | 1,318 | 0.01 | 0.785 |
| event | n_rows | mean | 364.1 +- 2.641 | 363.9 +- 1.743 | 0.05 | 0.961 |
| event | n_rows | KS (median shown) | 359.0 | 363.0 | 0.01 | 0.938 |
| event | n_phits | mean | 138.3 +- 1.128 | 138.0 +- 0.747 | 0.27 | 0.79 |
| event | n_phits | KS (median shown) | 134.0 | 134.0 | 0.01 | 0.945 |
| event | n_gen0_phits | mean | 119.6 +- 0.967 | 119.4 +- 0.642 | 0.25 | 0.803 |
| event | n_gen0_phits | KS (median shown) | 116.0 | 116.0 | 0.01 | 0.997 |
| event | n_sec_phits | mean | 18.690 +- 0.184 | 18.619 +- 0.121 | 0.32 | 0.75 |
| event | n_sec_phits | KS (median shown) | 17.000 | 17.000 | 0.01 | 0.91 |
| event | n_rows_k0s | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| event | n_rows_lambda | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| event | n_rows_k0l | mean | 8.306 +- 0.079 | 8.176 +- 0.051 | 1.38 | 0.168 |
| event | n_parent_k0s | mean | 10.081 +- 0.104 | 9.938 +- 0.067 | 1.15 | 0.249 |
| event | n_parent_lambda | mean | 5.337 +- 0.065 | 5.371 +- 0.043 | -0.45 | 0.656 |
| event | exactly one W muon | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| event | W+ (prompt mu+) | fraction | 0.571 +- 0.008 | 0.580 +- 0.005 | -0.92 | 0.36 |
| event | two N daughters in the table | fraction | 0.996 +- 0.001 | 0.994 +- 0.001 | 0.77 | 0.441 |
| event | N daughters all muons | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| event | N daughters opposite charge | fraction | 0.996 +- 0.001 | 0.994 +- 0.001 | 0.77 | 0.441 |
| event | N sign = W sign | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| event | W muon alone at vertex 1 | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | -1.50 | 0.134 |
| event | other particle at the N vertex | fraction | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| event | fitted N vertex (both daughters fitted) | fraction | 0.464 +- 0.008 | 0.448 +- 0.005 | 1.74 | 0.0825 |
| event | r_v (fitted) [mm] | KS (median shown) | 128.2 | 123.1 | 0.02 | 0.463 |
| event | r_v (fitted) [mm] | mean | 171.6 +- 3.510 | 168.6 +- 2.375 | 0.72 | 0.47 |
| event | r_v >= 100 mm among fitted | fraction | 0.572 +- 0.011 | 0.561 +- 0.008 | 0.78 | 0.435 |
| event | m(mu mu) / m_N | KS (median shown) | 0.523 | 0.523 | 0.01 | 0.623 |
| event | m(mu mu) / m_N | mean | 0.532 +- 0.004 | 0.528 +- 0.003 | 0.97 | 0.331 |
| event | opening angle [rad] | KS (median shown) | 0.223 | 0.234 | 0.02 | 0.53 |
| event | Delta R of the pair | KS (median shown) | 1.035 | 1.032 | 0.02 | 0.452 |
| event | pair pT [GeV] | KS (median shown) | 18.674 | 18.070 | 0.02 | 0.0911 |
| event | daughter targets per event | mean | 0.816 +- 0.014 | 0.798 +- 0.010 | 1.05 | 0.292 |
| event | primary daughters per event | mean | 0.615 +- 0.013 | 0.595 +- 0.009 | 1.31 | 0.19 |
| W muon | pT [GeV] | KS (median shown) | 29.552 | 29.572 | 0.02 | 0.164 |
| W muon | eta | KS (median shown) | 0.069 | -0.014 | 0.02 | 0.345 |
| W muon | p [GeV] | KS (median shown) | 94.587 | 95.315 | 0.02 | 0.3 |
| W muon | pT [GeV] | mean | 28.831 +- 0.197 | 28.617 +- 0.127 | 0.91 | 0.361 |
| W muon | hits | chi2 (20 dof; means shown) | 11.309 +- 0.052 | 11.535 +- 0.033 | 1.29 | 0.17 |
| W muon | pixel hits | chi2 (15 dof; means shown) | 6.313 +- 0.049 | 6.463 +- 0.033 | 1.57 | 0.0735 |
| W muon | >= 5 hits | fraction | 0.934 +- 0.004 | 0.950 +- 0.002 | -3.50 | 0.000466 ** |
| W muon | target (pT > 1, |eta| < 4, >= 5 hits) | fraction | 0.911 +- 0.004 | 0.924 +- 0.003 | -2.56 | 0.0104 |
| W muon | |d0| of targets [mm] | KS (median shown) | 0.017 | 0.019 | 0.06 | 4.78e-08 (known) |
| W muon | first-hit radius of targets [mm] | KS (median shown) | 31.981 | 31.985 | 0.04 | 0.00026 (known) |
| W muon | >= 3 pixel hits among targets | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| N daughter | pT [GeV] | KS (median shown) | 8.458 | 8.286 | 0.02 | 0.0886 |
| N daughter | eta | KS (median shown) | 0.046 | 0.007 | 0.01 | 0.34 |
| N daughter | p [GeV] | KS (median shown) | 32.015 | 32.459 | 0.01 | 0.185 |
| N daughter | pT [GeV] | mean | 10.614 +- 0.095 | 10.362 +- 0.061 | 2.24 | 0.0254 |
| N daughter | hits | chi2 (17 dof; means shown) | 4.244 +- 0.049 | 4.131 +- 0.032 | 1.90 | 0.0137 |
| N daughter | pixel hits | chi2 (14 dof; means shown) | 1.252 +- 0.028 | 1.202 +- 0.018 | 1.28 | 0.209 |
| N daughter | >= 5 hits | fraction | 0.428 +- 0.006 | 0.417 +- 0.004 | 1.65 | 0.0983 |
| N daughter | target (pT > 1, |eta| < 4, >= 5 hits) | fraction | 0.409 +- 0.006 | 0.400 +- 0.004 | 1.35 | 0.176 |
| N daughter | |d0| of targets [mm] | KS (median shown) | 24.957 | 23.549 | 0.02 | 0.609 |
| N daughter | first-hit radius of targets [mm] | KS (median shown) | 122.9 | 129.6 | 0.02 | 0.227 |
| N daughter | >= 3 pixel hits among targets | fraction | 0.436 +- 0.009 | 0.421 +- 0.006 | 1.45 | 0.146 |
| N daughter | |d0| >= 10 mm among targets | fraction | 0.699 +- 0.008 | 0.689 +- 0.005 | 1.03 | 0.305 |
| N daughter | primary among targets | fraction | 0.754 +- 0.008 | 0.745 +- 0.005 | 0.92 | 0.357 |
| N daughter | p of the daughter with charge opposite to the W muon [GeV] | mean | 114.9 +- 3.465 | 108.3 +- 2.128 | 1.62 | 0.106 |
| N daughter | p of the daughter with charge same to the W muon [GeV] | mean | 112.5 +- 3.511 | 112.2 +- 2.213 | 0.08 | 0.936 |

Tests 58 (without the 2 marked known), p < 0.01: 1 (chance: 0.6), p < 0.001: 1.

#### Generator-level variants (N decay matrix element, beam energy, initial-state shower)

| point | variant | events | m(mu mu)/m_N at production | p(opp)/p(same) | Zenodo | mean daughter p [GeV] | Zenodo | daughter p KS p | daughter pT KS p | daughter eta KS p | W+ | Zenodo | W-muon eta KS p | W-muon pT KS p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hnl10_100 | phase space, 13 TeV | 5,000 | 0.532 +- 0.003 | 1.042 +- 0.040 | 0.980 +- 0.028 | 114.1 | 108.3 | 0.107 | 0.000217 | 0.174 | 0.601 | 0.580 | 0.171 | 0.0124 |
| hnl10_100 | meMode 22, nu paired with N, 13 TeV | 5,000 | 0.611 +- 0.002 | 1.012 +- 0.038 | 0.980 +- 0.028 | 114.9 | 108.3 | 1.05e-06 | 1.14e-12 | 0.203 | 0.588 | 0.580 | 0.946 | 0.0168 |
| hnl10_100 | meMode 22, nu paired with N, 14 TeV | 5,000 | 0.609 +- 0.002 | 0.937 +- 0.036 | 0.980 +- 0.028 | 119.1 | 108.3 | 3.44e-05 | 3.56e-14 | 0.232 | 0.580 | 0.580 | 0.333 | 0.0324 |
| hnl10_100 | V-A, mu+ paired with N (lepton number conserved only), 13 TeV | 5,000 | 0.510 +- 0.003 | 1.151 +- 0.045 | 0.980 +- 0.028 | 105.3 | 108.3 | 0.113 | 0.00828 | 0.124 | 0.573 | 0.580 | 0.6 | 0.0217 |
| hnl10_100 | V-A, mu- paired with N (lepton number violated only), 13 TeV | 5,000 | 0.511 +- 0.003 | 0.864 +- 0.033 | 0.980 +- 0.028 | 108.5 | 108.3 | 0.135 | 0.0473 | 0.495 | 0.580 | 0.580 | 0.682 | 0.00118 |
| hnl10_100 | Majorana V-A, 13 TeV | 5,000 | 0.510 +- 0.003 | 0.978 +- 0.038 | 0.980 +- 0.028 | 108.3 | 108.3 | 0.649 | 0.00437 | 0.0468 | 0.577 | 0.580 | 0.142 | 0.019 |
| hnl10_100 | Majorana V-A, 14 TeV | 5,000 | 0.510 +- 0.003 | 0.957 +- 0.036 | 0.980 +- 0.028 | 108.7 | 108.3 | 0.452 | 0.1 | 0.102 | 0.573 | 0.580 | 0.392 | 0.000691 |
| hnl10_100 | Majorana V-A, 14 TeV, no ME correction of the first ISR emission | 5,000 | 0.512 +- 0.003 | 0.989 +- 0.039 | 0.980 +- 0.028 | 109.6 | 108.3 | 0.735 | 0.0733 | 0.503 | 0.570 | 0.580 | 0.244 | 0.122 |
| hnl10_100 | Majorana V-A, 14 TeV, A14 tune | 5,000 | 0.511 +- 0.003 | 1.045 +- 0.040 | 0.980 +- 0.028 | 113.2 | 108.3 | 0.258 | 0.274 | 0.58 | 0.575 | 0.580 | 0.856 | 0.106 |
| hnl10_100 | Majorana V-A, 13 TeV, A14 tune | 5,000 | 0.510 +- 0.003 | 0.932 +- 0.036 | 0.980 +- 0.028 | 111.4 | 108.3 | 0.956 | 0.0482 | 0.27 | 0.594 | 0.580 | 0.635 | 0.0728 |
| hnl10_100 | Majorana V-A, 13 TeV, ISR up to the factorization scale | 5,000 | 0.512 +- 0.003 | 0.980 +- 0.039 | 0.980 +- 0.028 | 108.1 | 108.3 | 0.393 | 0.503 | 0.00232 | 0.585 | 0.580 | 0.0604 | 0.0706 |
| hnl10_100 | Majorana V-A, 13 TeV, A14 tune, ISR up to the factorization scale | 5,000 | 0.513 +- 0.003 | 0.944 +- 0.038 | 0.980 +- 0.028 | 105.7 | 108.3 | 0.00962 | 0.49 | 0.249 | 0.579 | 0.580 | 0.763 | 0.707 |
| hnl10_100 | Majorana V-A, 14 TeV, ISR up to the factorization scale (the final card) | 5,000 | 0.506 +- 0.003 | 1.027 +- 0.041 | 0.980 +- 0.028 | 109.8 | 108.3 | 0.92 | 0.323 | 0.477 | 0.573 | 0.580 | 0.408 | 0.358 |
| hnl10_100 | final card, second sample | 5,000 | 0.508 +- 0.003 | 1.018 +- 0.040 | 0.980 +- 0.028 | 116.0 | 108.3 | 0.192 | 0.583 | 0.0512 | 0.586 | 0.580 | 0.921 | 0.166 |
| hnl10_100 | final card, ACTS v15 Pythia8Generator | 5,000 | 0.507 +- 0.003 | 0.958 +- 0.038 | 0.980 +- 0.028 | 111.5 | 108.3 | 0.809 | 0.326 | 0.108 | 0.580 | 0.580 | 0.873 | 0.416 |
| hnl15_200 | phase space, 13 TeV | 5,000 | 0.538 +- 0.003 | 0.974 +- 0.038 | 0.966 +- 0.027 | 109.0 | 110.2 | 0.0638 | 3.05e-06 | 0.0943 | 0.579 | 0.580 | 0.124 | 0.00402 |
| hnl15_200 | meMode 22, nu paired with N, 13 TeV | 5,000 | 0.612 +- 0.002 | 0.996 +- 0.038 | 0.966 +- 0.027 | 117.9 | 110.2 | 0.0046 | 8.39e-13 | 0.226 | 0.571 | 0.580 | 0.374 | 0.0339 |
| hnl15_200 | meMode 22, nu paired with N, 14 TeV | 5,000 | 0.607 +- 0.002 | 0.964 +- 0.037 | 0.966 +- 0.027 | 119.5 | 110.2 | 1.81e-05 | 7.7e-15 | 0.304 | 0.582 | 0.580 | 0.959 | 0.0118 |
| hnl15_200 | V-A, mu+ paired with N (lepton number conserved only), 13 TeV | 5,000 | 0.507 +- 0.003 | 1.194 +- 0.045 | 0.966 +- 0.027 | 105.7 | 110.2 | 0.351 | 0.229 | 0.0812 | 0.568 | 0.580 | 0.728 | 0.0003 |
| hnl15_200 | V-A, mu- paired with N (lepton number violated only), 13 TeV | 5,000 | 0.506 +- 0.003 | 0.871 +- 0.033 | 0.966 +- 0.027 | 106.3 | 110.2 | 0.0173 | 0.000611 | 0.25 | 0.597 | 0.580 | 0.513 | 0.0428 |
| hnl15_200 | Majorana V-A, 13 TeV | 5,000 | 0.512 +- 0.003 | 1.019 +- 0.040 | 0.966 +- 0.027 | 103.1 | 110.2 | 0.00106 | 0.0204 | 0.0453 | 0.588 | 0.580 | 0.107 | 0.00185 |
| hnl15_200 | Majorana V-A, 14 TeV | 5,000 | 0.514 +- 0.003 | 1.008 +- 0.039 | 0.966 +- 0.027 | 113.2 | 110.2 | 0.975 | 0.034 | 0.157 | 0.597 | 0.580 | 0.342 | 0.00151 |
| hnl15_200 | final card, second sample | 5,000 | 0.509 +- 0.003 | 1.034 +- 0.039 | 0.966 +- 0.027 | 114.8 | 110.2 | 0.322 | 0.759 | 0.495 | 0.572 | 0.580 | 0.784 | 0.473 |
| hnl15_200 | final card, ACTS v15 Pythia8Generator | 5,000 | 0.508 +- 0.003 | 0.995 +- 0.038 | 0.966 +- 0.027 | 112.6 | 110.2 | 0.191 | 0.2 | 0.917 | 0.575 | 0.580 | 0.391 | 0.0574 |

"p(opp)/p(same)": mean momentum of the N daughter whose charge is opposite to the W muon over that of the other one. "meMode 22" is Pythia's weak matrix element, weight (P.p1)(p2.p3) for products (p1, p2, p3).

## 5. What is not matched

With the final card, 2 of the 116 counted tests have p < 0.01 (0.6 expected per point) and 1 has p < 0.001:

- **W-muon acceptance.** The fraction of W muons with at least 5 hits is 0.939 +- 0.004 and 0.934 +- 0.004 against
  0.948 and 0.950 (z = -2.1 and -3.5; both pilots together -3.4 standard deviations, about one point). The muons
  lost are those beyond |eta| of about 4: |eta| > 4 holds 8.2 and 8.8 % of our W muons against 7.6 and 7.4 % in
  Zenodo. Every 14 TeV pilot shows it (pooled over the three, 0.938 +- 0.002); the 13 TeV pilots do not (0.945 +-
  0.002), but at 13 TeV the multiplicity and the daughter acceptance fail (section 4, variant table). The W-muon
  target fraction differs by 0.7 and 1.3 points (z = -1.4 and -2.6). The W muon enters the evaluation only as the
  prompt reference; the difference moves its population at the edge of the acceptance, not the N daughters.
- **N daughters with at least 3 pixel hits among the targets: the Zenodo pixel-layer artifact.** 0.483 +- 0.008
  against 0.453 +- 0.005 in hnl10_100 (z = +3.1), 0.436 against 0.421 in hnl15_200 (z = +1.5). The number of targets,
  their |d0|, first-hit radius, the fitted r_v and the primary population per event all agree (table above); this is
  a subset reported next to the targets (`docs/definitions.md` section 4), not part of any population definition.
  The cause is in the Zenodo files, not in the generator: a Zenodo daughter whose decay vertex lies within the radial
  extent of the modules of a barrel pixel layer loses every later barrel pixel hit. `scripts/acts/hnl_pixel_layers.py`
  (jobs 6819803_0-3; `results/hnl_points_pixel_layers_v1.json`) records which of the barrel pixel layers L1-L4 hold a
  hit of each daughter and takes r_v from the census (truth fits). The layer bands are the radial extent of the
  modules in `detectors.csv` (L1 31.4-34.4 mm, L2 71.1-73.9 mm, L3 114.9-117.7 mm). Daughter targets with |eta| < 1,
  train and val of both points pooled:

  | layer | r_v in the band: no later barrel pixel hit, Zenodo | ours (fitted r_v) | ours (generator r_v) | 1-6 mm before the band, Zenodo / ours | 1-6 mm after, Zenodo / ours |
  |---|---|---|---|---|---|
  | L1 | 54 / 59 (92 %) | 0 / 29 | 0 / 34 | 12 / 131 / 4 / 59 | 9 / 127 / 1 / 55 |
  | L2 | 49 / 51 (96 %) | 0 / 19 | 0 / 21 | 21 / 105 / 10 / 57 | 18 / 118 / 1 / 33 |
  | L3 | 37 / 39 (95 %) | 0 / 15 | 1 / 24 | 15 / 87 / 4 / 24 | 8 / 94 / 6 / 34 |

  Per point the in-band Zenodo fractions are 31/32 and 23/27 (L1), 23/25 and 26/26 (L2), 18/20 and 19/19 (L3). The
  ">= 3 pixel hits" difference follows the decay radius: with standard errors clustered by event, ours minus Zenodo
  is z = +7.4 / +5.3 at fitted r_v in [30, 72) mm (93.9 / 88.9 % against 83.4 / 79.1 %), +3.6 / +1.9 in
  [72, 116) mm, and |z| <= 1.4 at r_v < 30 mm and >= 116 mm. An independent extraction by a reviewer (own vertex
  fit, measure "fewer pixel hits than barrel layers outward of r_v", windows 30-34, 70-74 and 114-118 mm,
  |eta| < 0.8) found 95-100 %, 86-87 % and 54-62 % in Zenodo, 0-13 % in our pilots, and z = +7.6 / +5.7 in
  [30, 72) mm. Every sample of our ACTS v15 chain is free of the artifact.
- **Known, not counted: the hit-label exchange of the Zenodo files.** The fitted |d0| of W-muon targets (median
  0.016 mm here, 0.019 mm in Zenodo; KS p 1.7e-10 and 4.8e-8) and their first-hit radius (KS p 5.9e-5 and 2.6e-4)
  differ because the Zenodo chains contain hits of other particles: the rms of the fixed-radius fit of W muons has an
  84 % quantile of 1.20 mm in Zenodo against 0.044 mm here, and of prompt ttbar tracks with pT > 5 GeV 7.94 mm against
  0.072 mm (`scripts/acts/prompt_fit_check.py`, jobs 6812462 and 6812533; `results/hnl_points_prompt_fit_v1.json`).
  This is the exchange documented in `results/acts_validation_v1.md` section "Labels"; our files have correct labels.

## 6. Checks of the generator itself

- **Driver against the ACTS generator.** `hnl_pythia.cc` reproduces the particles and barcodes of the ACTS v15
  `Pythia8Generator` and adds the parent codes. With the final card, 5,000 generator-level events of each (jobs
  6812597-6812600) agree in all 17 compared quantities per point (lowest p 0.058): particles per event, vertices per
  event, the W muon alone at vertex 1, one two-muon vertex per event, and the pT, eta, p and production radius of the
  W muon and the N daughters (table at the end).
- **Lifetime.** The proper decay length L / (beta gamma) of the generator truth averages 101.9 +- 1.6 mm and 197.5 +-
  3.1 mm in the final pilots (c tau0 = 100 and 200 mm). At generator level, 5,000 events per point: 50.2 +- 0.7
  (50), 100.6 +- 1.4 (100), 201.3 +- 2.9 (200), 296.9 +- 4.2 (300), 501 +- 7 (500), 1016 +- 15 mm (1000); for
  20 GeV and 200 mm, 190.9 +- 2.7 mm in the first 5,000 events and 197.4 +- 2.0 mm in 10,000 more (job 6817275),
  together 195.2 +- 1.6 mm (-3.0 standard deviations; Kolmogorov-Smirnov against the exponential, p = 0.02). A
  review run of 20,000 more events (seed 31000001) gave 196.7 +- 1.4 mm, so 35,000 events gave 196.1 +- 1.1 mm
  (0.980 +- 0.005 c tau0, -3.7 standard deviations). Follow-up before hnl20_200 was produced
  (`scripts/acts/hnl_lifetime_check.cc`, `slurm/acts/hnl_lifetime_check.sbatch`, jobs 6819714-6819720 and 6819830;
  `scripts/acts/hnl_lifetime_summary.py`, `results/hnl_points_lifetime_v1.json`):

  | run (20 GeV / 200 mm unless stated) | events | mean tau (mm) | / c tau0 | z | KS p |
  |---|---|---|---|---|---|
  | planning, seed 10230001 (the run that raised the question) | 5,000 | 190.9 +- 2.7 | 0.955 +- 0.014 | -3.3 | 0.019 |
  | planning, seed 10610001 | 10,000 | 197.4 +- 2.0 | 0.987 +- 0.010 | -1.3 | 0.34 |
  | review, seed 31000001 | 20,000 | 196.7 +- 1.4 | 0.984 +- 0.007 | -2.4 | 0.034 |
  | check, seed 10710001 | 40,000 | 199.6 +- 1.0 | 0.998 +- 0.005 | -0.4 | 0.74 |
  | check, Pythia's own random numbers, seed 10740001 | 20,000 | 201.8 +- 1.4 | 1.009 +- 0.007 | +1.3 | 0.48 |
  | process level only, seed 10720001 | 400,000 | 199.93 +- 0.32 | 0.9997 +- 0.0016 | -0.2 | 0.67 |
  | process level, 5 / 50, 10 / 100, 15 / 200, 15 / 500 | 200,000 each | | 0.999, 1.003, 0.997, 0.997 (+- 0.002) | -0.4, +1.4, -1.2, -1.3 | 0.19-0.99 |

  Pythia draws tau once, in the process record, when the W decays. In the 60,000 events of the two check runs every
  copy of the N in the event record (the N is copied in 47 % of the events, when it takes the recoil of a photon
  emitted by the W muon) carries the process-record value, every N decays at the vertex tau p / m from its production
  point, and the driver never needed a second `next()`. Whether an internal regeneration inside Pythia selects on tau
  is tested by the full-event runs themselves: the two check runs together give 200.3 +- 0.8 mm (60,000 events), in
  agreement with the process-level draw, whose mean is 0.9997 +- 0.0016 c tau0. The check program reproduces the review run event by event (events 0-1999 of seed
  31000001: 2,000 of 2,000 lifetimes identical), so the runs differ only by their events. The five full-event runs
  give 198.7 +- 0.6 mm together (95,000 events, chi2 16.0 for 4 degrees of freedom); without the run that raised the
  question, 199.1 +- 0.7 mm (90,000 events, chi2 7.5 for 3, p = 0.056). The effective c tau of the card is quoted in
  `docs/hnl_points_spec.md` section 10; each produced sample reports its own mean with its error.
- **Truth fits on our events.** Fitted against generator values on the final pilots: r_v differs by a median of
  0.002 / 0.006 mm (median absolute deviation 0.19 / 0.15 mm) and |d0| of daughter targets by 0.0004 / 0.001 mm
  (0.053 / 0.056 mm), so the d0 and r_v definitions of `docs/definitions.md` apply unchanged.
- **Vertex time.** Spread of the W-muon production time 4.99 / 5.05 ns (Zenodo 4.90 / 4.91 ns).
- **Retries.** Pythia's `next()` never failed in the final pilots (0 events with more than one call); the driver
  would retry a failed event instead of writing a partial one, which the ACTS generator does.

## 7. Reproduction

```
# Zenodo census (train + val only), one job per sample
sbatch slurm/acts/hnl_census.sbatch zenodo --sample hnl10_100 --splits train,val --refit-check 200 --out CENSUS/zenodo_hnl10_100
# a full-simulation pilot (final card = the defaults of produce_hnl.sbatch) and its census
sbatch --array=0-7 --export=ALL,MASS=10,CTAU=100,SEED=10110001,OUT=PILOT/F_hnl10_100,CHUNK=500 slurm/acts/produce_hnl.sbatch
sbatch slurm/acts/hnl_census.sbatch shards --dir PILOT/F_hnl10_100 --out CENSUS/F_hnl10_100
# variants: DECAY=phase_space|majorana_va|..., ECM=13000, EXTRA="Tune:pp = 21" (pilot studies only), SIMULATE=0 for
# generator level, GEN=acts for the ACTS v15 Pythia8Generator
python scripts/acts/hnl_closure.py --census CENSUS --pilot F --genvar GENVAR --driver-acts F,Facts \
    --variants "v0=...;M13=...;PS14=...;M14=...;A=...;F=..." --out OUT
```

Pilot seeds: v0 991001-992001 (2,000 events per point), M14 9710001 / 9720001, M13 9730001 / 9740001, PS14
9750001 / 9760001, A 10510001 / 10520001, F 10110001 / 10120001 (4,000 events per point each); generator-level
variants 961001-972001 (phase space, meMode 22 orderings, 14 TeV), 9810001-9930001 (Majorana at 13 / 14 TeV and the
ACTS generator), 9960001-10000001 (shower and tune), 10410001-10420001 (A14 with and without the shower limit),
10130001-10160001 (final card, driver and ACTS generator); planning runs 10210001-10280001 and 10610001.

## Appendix: driver against the ACTS v15 Pythia8Generator

| point | quantity | test | driver | ACTS generator | z or statistic | p |
|---|---|---|---|---|---|---|
| hnl10_100 | n_gen per event | mean | 400.2 +- 2.599 | 398.9 +- 2.524 | 0.37 | 0.712 |
| hnl10_100 | n_gen_p05 per event | mean | 319.4 +- 2.036 | 318.1 +- 1.976 | 0.47 | 0.639 |
| hnl10_100 | n_ch_p05 per event | mean | 170.4 +- 1.091 | 169.7 +- 1.063 | 0.46 | 0.643 |
| hnl10_100 | n_vsec per event | mean | 120.9 +- 0.790 | 120.6 +- 0.765 | 0.35 | 0.725 |
| hnl10_100 | vsec_max per event | mean | 119.9 +- 0.790 | 119.6 +- 0.765 | 0.35 | 0.725 |
| hnl10_100 | n_k0s per event | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| hnl10_100 | n_lambda per event | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| hnl10_100 | W muon alone at vertex 1 | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 1.00 | 0.317 |
| hnl10_100 | exactly one two-muon vertex (the brief's rule) | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | -1.00 | 0.317 |
| hnl10_100 | W muon pt | KS (median shown) | 30.066 | 29.976 | 0.01 | 0.741 |
| hnl10_100 | W muon eta | KS (median shown) | -0.057 | -0.060 | 0.01 | 0.755 |
| hnl10_100 | W muon p | KS (median shown) | 97.098 | 94.170 | 0.01 | 0.912 |
| hnl10_100 | W muon r_prod | KS (median shown) | 0.015 | 0.015 | 0.02 | 0.339 |
| hnl10_100 | N daughter pt | KS (median shown) | 7.780 | 7.987 | 0.01 | 0.546 |
| hnl10_100 | N daughter eta | KS (median shown) | -0.048 | -0.043 | 0.02 | 0.119 |
| hnl10_100 | N daughter p | KS (median shown) | 32.438 | 30.992 | 0.02 | 0.0969 |
| hnl10_100 | N daughter r_prod | KS (median shown) | 177.8 | 183.5 | 0.02 | 0.0583 |
| hnl15_200 | n_gen per event | mean | 399.7 +- 2.540 | 400.0 +- 2.561 | -0.07 | 0.941 |
| hnl15_200 | n_gen_p05 per event | mean | 318.8 +- 1.991 | 319.0 +- 2.004 | -0.06 | 0.956 |
| hnl15_200 | n_ch_p05 per event | mean | 169.8 +- 1.068 | 170.2 +- 1.074 | -0.28 | 0.776 |
| hnl15_200 | n_vsec per event | mean | 121.0 +- 0.771 | 120.8 +- 0.779 | 0.15 | 0.878 |
| hnl15_200 | vsec_max per event | mean | 120.0 +- 0.771 | 119.8 +- 0.779 | 0.15 | 0.878 |
| hnl15_200 | n_k0s per event | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| hnl15_200 | n_lambda per event | mean | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.00 | 1 |
| hnl15_200 | W muon alone at vertex 1 | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | -1.41 | 0.157 |
| hnl15_200 | exactly one two-muon vertex (the brief's rule) | fraction | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.00 | 1 |
| hnl15_200 | W muon pt | KS (median shown) | 29.459 | 29.585 | 0.01 | 0.678 |
| hnl15_200 | W muon eta | KS (median shown) | -0.058 | 0.001 | 0.01 | 0.95 |
| hnl15_200 | W muon p | KS (median shown) | 95.439 | 95.462 | 0.01 | 0.964 |
| hnl15_200 | W muon r_prod | KS (median shown) | 0.015 | 0.015 | 0.02 | 0.097 |
| hnl15_200 | N daughter pt | KS (median shown) | 8.266 | 8.119 | 0.01 | 0.322 |
| hnl15_200 | N daughter eta | KS (median shown) | -0.013 | 0.027 | 0.01 | 0.523 |
| hnl15_200 | N daughter p | KS (median shown) | 33.280 | 31.353 | 0.02 | 0.0872 |
| hnl15_200 | N daughter r_prod | KS (median shown) | 240.7 | 241.2 | 0.01 | 0.581 |
