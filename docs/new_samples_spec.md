# New samples: generic displaced sample and a second LLP topology (spec v1)

Status: fixed 2026-10-14, before any event of these samples was produced. Every setting below is chosen from the
physics requirements in sections 1-3 and from a truth-level comparison of the simulation stack with the Zenodo ttbar
sample (section 8); no model output enters any choice. A change after this commit is logged in RESULTS.md with its
reason before the affected events are produced or read by a model.

## 1. Purpose and governing documents

- **Generic displaced sample** (supervisor, 2026-10-14: the priority). The sample of preregistration addendum G
  (`docs/preregistration_new_sample.md`, a28c647, claims G1-G3): the missing training baseline (acorn and Canto-L trained
  on ttbar plus this sample) and an independent test of generalization. Two components, as addendum G section 2 asks:
  isolated displaced tracks, and hadronic-like decays with several collimated tracks per vertex.
- **Second topology** (supervisor: "hadronic LLP decay with several tracks per vertex, or electrons"; fixes the
  "one simulation, one topology" limitation of paper Sec. IX). A physics generator sample: gg -> H(125) -> s s with a
  long-lived scalar s decaying to b bbar or e+ e-. Evaluation only (section 6).
- **Deviation from addendum G section 2.** Addendum G expected a sample "produced outside this project". These samples
  are produced in this project (user's decision, 2026-10-14). What protects the claims is the order of work: this spec
  is committed before production, the samples are produced and validated on truth only, and addendum G's T1 commit
  (truth mapping, split lists, P_G counts) precedes any model run. The gun deliberately covers the regimes where the
  paper found losses (far decays, shallow crossings, dense vertices); it is not tuned to any model.
- Route and toolchain study: `docs/displaced_sample_search.md` section 3.

## 2. Simulation stack (same as the Zenodo HNL sample)

- **ACTS v15.0.0** (commit 0fef9e0), built from source with Fatras, the examples, Pythia8 and the Python bindings
  (`slurm/acts/build_v15.sbatch`): base OS `atlas/centos7-atlasos:latest-gcc8` from /cvmfs/unpacked.cern.ch (CentOS
  7.9 with the devel packages; the bare `centos:centos7` image has no glibc-devel), compilers and libraries from
  `LCG_97apython3/x86_64-centos7-gcc9-opt` (gcc 9.2.0, ROOT 6.20/06, Boost 1.72, Eigen 3.3.7, Pythia8 244), the
  environment of the v15 CI.
- **Detector**: the GenericDetector with its default configuration (geometry ids in the pre-v21 layout of the Zenodo
  files; checked in section 8). Constant 2 T field along z.
- **Generator**: Pythia8 244 at sqrt(s) = 14 TeV (the ACTS default), one collision per event, no pileup. Beam spot
  Gaussian with sigma = (0.0125, 0.0125, 55.5) mm in (x, y, z) (sigma_z of the Zenodo samples is about 55 mm).
- **Fatras**: multiple scattering, ionisation and radiative energy loss, and photon conversion on; hits on sensitive
  surfaces; pMin = 0.5 GeV (the momentum edge of the Zenodo samples); no nuclear interactions; Fatras decays nothing.
- **Digitisation**: smearing with `Examples/Algorithms/Digitization/share/default-smearing-config-generic.json`
  (Gaussian, sigma_loc0 / sigma_loc1 = 14.4 / 14.4 um in the pixels, 23.1 um / 0.346 mm in the short strips, 34.6 um
  / 3.12 mm in the long strips), no hit merging (the Python default), so every measurement has exactly one simulated
  hit.

## 3. Samples

| Directory (under `hnl_lrt/generated/`) | Events | Content |
|---|---|---|
| `generic_iso_v1` | 8,000 | Pythia8 ttbar + 2 gun LLPs with two-body decays (isolated component) |
| `generic_dense_v1` | 8,000 | Pythia8 ttbar + 2 gun LLPs with 3-8 charged decay products (dense component) |
| `hss15_100_v1` | 1,500 | gg -> H -> s s, m_s = 15 GeV, c tau_s = 100 mm |
| `hss40_300_v1` | 1,500 | gg -> H -> s s, m_s = 40 GeV, c tau_s = 300 mm |

**ttbar** (underlying collision of both generic components): `Top:qqbar2ttbar = on`, `Top:gg2ttbar = on`,
`ParticleDecays:limitTau0 = on` (particles with c tau0 > 10 mm, e.g. K_S and Lambda, are not decayed, as in the
Zenodo ttbar, whose particle tables hold K_S, K_L and Lambda).

**Gun** (`scripts/acts/gun.py`; per event 2 LLPs at the ttbar collision vertex; the LLP itself is not simulated):
- direction eta ~ U(-3, 3), phi ~ U(0, 2 pi); decay at transverse radius r_v ~ U(1, 800) mm along the flight line,
  z_v = z_PV + r_v sinh(eta), eta and r_v redrawn while |z_v| > 2900 mm; decay time from the flight length and the boost;
- decay isotropic in the LLP rest frame (two-body or unweighted N-body phase space), then boosted;
- isolated component: e+e-, mu+mu-, pi+pi-, K+K- or p pbar with equal shares; m ~ logU(max(0.5 GeV, 2.2 m_d), 20 GeV);
  betagamma ~ logU(0.3, 30) (low boosts give wide opening angles and shallow layer crossings);
- dense component: N_ch ~ U{3, ..., 8} charged products, each pi (70 %), K (15 %), p (5 %), e (5 %) or mu (5 %),
  charges summing to 0 (even N_ch) or +-1 (odd N_ch); m ~ logU(max(2 GeV, 1.2 sum m_i), 40 GeV); betagamma ~
  logU(1, 20) (collimated vertices);
- barcodes: primary vertex 1 (the ttbar collision), secondary vertex 1000 + LLP index (1 or 2), particle 1..n,
  generation 0; the generator's own secondary vertices stay below 1000.

**h -> s s** (Pythia8; `scripts/acts/acts_sim.py`): `HiggsSM:gg2H = on`, `25:onMode = off`,
`25:addChannel = 1 1.0 100 35 35`, `35:m0` = 15 or 40 GeV (width 1 MeV, mass window +-50 MeV), `35:tau0` = 100 or
300 mm, `35:onMode = off`, `35:oneChannel = 1 0.7 101 5 -5`, `35:addChannel = 1 0.3 101 11 -11`. Every unstable
hadron is decayed by the generator (Pythia default), K_S and Lambda included, unlike the ttbar of the training
sample. (A first form with `25:oneChannel` gives a vanishing cross section in Pythia 244; checked in a pilot before
this commit.)

## 4. Seeds and event numbers

Event numbers run 0..N-1 within a sample and name the files (`<i>.npz`). Each step draws from a per-event stream:
ACTS seeds `seed + Cantor(algorithm, event)`; the gun uses `numpy.random.default_rng([seed + 7, event])`.
Base seeds: ttbar validation 150001, `generic_iso_v1` 250001, `generic_dense_v1` 350001, `hss15_100_v1` 450001,
`hss40_300_v1` 550001; Fatras and digitisation use base + 13. A chunk can be rerun alone and reproduces its events.
Production runs as chunks of 500 events (`slurm/acts/produce.sbatch`; CPU only, A30 and GTX 1080 nodes excluded).

## 5. File format

Per event, `<i>.npz` with exactly the keys and dtypes of the Zenodo files (`hnl.data.npz.NPZ_KEYS`, PARTICLE_COLUMNS),
written by `scripts/acts/csv_to_npz.py`:
- `x`: (r, phi, z) of the smeared measurement (local0, local1 placed on its surface with `detectors.csv`), float32,
  as in the Zenodo files (section 8);
- `pid`: the particle of the measurement's simulated hit, from `measurement-simhit-map.csv` (never from row order);
- `layers`: the geometry id; rows in measurement order, which is geometry-id order; `hid` = row index;
- a particle with two hits on one module keeps the earlier (the Zenodo files have none; dropped hits are counted per
  event in the sidecar);
- `layerless_true_edges`: each particle's hits in simulated-time order, consecutive pairs;
- `particles`: `particles_final.csv` (end state, as in the Zenodo files), parent_pid = 0 everywhere (unlabeled), pt,
  radius, eta derived as in the Zenodo files.

Truth sidecars in `<dir>_truth/`: `truth_<first>.npz` per chunk (initial state, i.e. production vertex and momentum, of
every particle with hits; per-event counts) and, for the generic samples, `gun_truth_<first>.npz` (per gun particle:
LLP index, prongs, mass, boost, direction, r_v, decay vertex, PDG, charge, momentum).

## 6. Use under addendum G

- Components: `iso` = `generic_iso_v1` (isolated), `dense` = `generic_dense_v1` (dense). Addendum G's split hash uses
  these component strings and the event number in decimal as `<id>`: `sha256("canto-generic-v1:iso:17")`, etc.
- Displaced source (addendum G section 4): the gun particles, i.e. barcode secondary vertex >= 1000, listed in
  `gun_truth_*.npz`; r_v is the exact decay radius, d0 follows from the stored production vertex and momentum.
  These are the definitions T1 commits; nothing here reads a model.
- `hss15_100_v1` and `hss40_300_v1` are evaluation only: no training, no selection, all events evaluated; any other use
  needs its own commit first.
- The ttbar validation events (1,000, section 8) stay on /work and are used for nothing else.

## 7. Storage and file budget

About 170 kB per event (Zenodo ttbar mean 166 kB), so about 3.2 GB on /ceph for 19,000 events, plus the sidecars
(about 0.2 GB). Files: 19,000 npz + at most 76 sidecars, so `hnl_lrt` goes from 46,367 to about 65,500 files, below the
80k stop line. Scratch CSVs live under /work/submit/jinboz1/hnl/acts/work and are deleted chunk by chunk after
conversion.

## 8. Validation of the stack against the Zenodo ttbar (truth only)

`results/acts_validation_v1.md` (1,000 ttbar events simulated with section 2's settings against 1,000 Zenodo ttbar
training events; no model run):
- every Zenodo geometry id is a sensitive surface of the v15 GenericDetector, in the same bit layout;
- hits, particles and secondaries per event agree within 0.5 standard errors; pT, eta, hits-per-particle and
  near-threshold momentum shapes agree (Kolmogorov distance at most 0.009); occupancy per layer, PDG mix, process codes
  and the secondary fraction (16.0 % against 15.9 %) agree;
- the Zenodo hit positions are smeared measurements with the smearing of section 2 (Gaussian core of the barrel-strip
  z residuals 0.414 / 3.143 mm there, 0.391 / 3.141 mm here);
- **one difference, kept on purpose:** in the Zenodo files about 15 % of the pairs of hits of two particles on one
  module carry exchanged labels (swap test of hnl.eval.dense_jet_check; 36-44 % of the barrel-strip hits of
  pT > 5 GeV primaries lie > 10 mm off their track in z). The v15 simulation, digitisation and CSV writers do not
  produce this (one-to-one, identity measurement-simhit map; 0.1-1.4 % flagged, the test's own control level), so our
  labels are correct. The frozen models were trained and selected on the Zenodo labels; results on the new samples
  are reported with this difference stated.

## 9. Changes after the first commit (69a6f11)

- 2026-10-14, before any event was produced: the samples are written to `hnl_lrt/generated/` instead of
  `hnl_lrt/unpacked/`, which holds the Zenodo data and is read-only (the first production jobs 6810666-6810669 stopped
  at `mkdir`, before generating anything). Nothing else changes.

- 2026-10-14, before any event of it exists: **extension `generic_iso_x1`** (addendum G T1 deviation 13, RESULTS.md).
  16,000 events numbered 8000-23999 of the isolated component, settings of section 3 unchanged (Pythia8 ttbar at
  14 TeV plus the same gun). Base seed 260001 (Pythia), gun `default_rng([260008, event])`, Fatras and digitisation
  260014; no per-event ACTS stream coincides with one of the existing samples (enumerated for algorithms 1-12 and both
  ACTS steps). Storage: one uncompressed tar of Zenodo-format `<event>.npz` members per 500-event chunk
  (`events_008000_008499.tar`, ...; the layout of `hnl.data.shards`) plus the two truth sidecars per chunk, under
  `hnl_lrt/generated/generic_iso_x1/` and `generic_iso_x1_truth/`. Produced from a committed state; before production,
  chunk 7500-7999 of generic_iso_v1 is regenerated with the same code and must equal the stored events array by array.
- 2026-10-14, found at the sufficiency check: four iso/dense event pairs share their Pythia ttbar collision because
  the per-event seed is base + Cantor(algorithm, event) and the base seeds differ by 100,000 (iso 470 / dense 150,
  860 / 735, 1592 / 1528, 4010 / 3985); the extension's seed was checked against this (addendum G T1 deviation 14).
