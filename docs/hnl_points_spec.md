# HNL points: three new mass and lifetime points for evaluation (spec v1)

Status: fixed 2026-10-14, before any event of these points was produced. The generator settings come from a
truth-level closure against the train and val events of the Zenodo HNL samples (section 8,
`results/hnl_points_closure_v1.md`); the points and event counts come from the aims of section 4 and from
generator-level studies. No model was run for this spec and no model output entered any choice. A change after this
commit is logged in RESULTS.md with its reason before the affected events are produced or read by a model.

## 1. Purpose

The supervisor asked for "more HNL mass points if easy" (2026-10-14); the user approved the work the same day. The
benchmark holds two labeled HNL points, 10 GeV / 100 mm and 15 GeV / 200 mm, and the unlabeled external set
15 GeV / 100 mm. They differ little: the two labeled points have nearly the same decay-radius distribution and
opening angles within a factor of 1.6. The new points change one property at a time, so that an efficiency
difference can be traced to it:

- a lighter N at the decay-radius distribution of hnl10_100 (collimated pairs that point back towards the beam line);
- a heavier N at the same decay-radius distribution (wide pairs, tracks that cross the layers at shallow angles);
- a longer lifetime at m_N = 15 GeV (decays spread over the tracker, targets that start in the strip layers).

Each new point is compared with an in-stack reference point: the two Zenodo points regenerated with the same card,
driver, simulation, digitisation and labels (`hnl10_100_ref` for the first two, `hnl15_200_ref` for the third;
section 4). A difference between a new point and its reference then comes from the changed property alone. A
comparison with the Zenodo points themselves is confounded by the hit-label exchange of the Zenodo files, by the
Zenodo pixel-layer artifact (section 10), and by the beam energy and Pythia version (section 2), so differences
between our points and the Zenodo points are reported but not attributed to m_N or c tau0.

All five samples are evaluation samples only (section 7).

## 2. How the Zenodo HNL samples were generated

**Established** (paper, Zenodo record, and the truth of the 17,998 train and val events of hnl10_100 and hnl15_200;
`results/hnl_points_closure_v1.md` section 2):

- process pp -> W -> mu N, N -> mu+ mu- nu_mu (paper Eq. 1; record: N "forced to decay to two muons and a muon
  neutrino"); simulated with ACTS v15 Fatras on the generic detector in a constant 2 T field, smearing digitisation,
  no pileup (record);
- every event holds exactly one muon with parent +-24 (the prompt muon), at generator vertex 1; the W is a W+ in 58.0 %
  of the events;
- every particle with parent +-50 is a muon; 99.0 / 99.4 % of the events hold two of them, always of opposite charge,
  at one generator vertex shared with no other particle; so the decay-mode mix is 100 % mu mu nu;
- the N has code 50 and a distinct antiparticle: +50 with a W+, -50 with a W- in every event;
- every particle with c tau0 < 1 m was decayed by the generator (no K_S or Lambda rows, 9.9 K_S and 5.4 Lambda daughters
  per event), unlike in the Zenodo ttbar sample;
- column 12 holds the PDG code of the generator mother (followed up through radiating copies), 0 for particles created
  in the simulation;
- the production time of the collision has a spread of 4.9 ns (the ACTS v15 Pythia8 example default, 5 ns), the
  longitudinal beam spot 54.8 mm;
- the tables hold end-state momenta, whose azimuth has turned in the field: angles and invariant masses of the pairs can
  only be compared after the same simulation.

**Inferred** by the closure (section 8): W production f fbar' -> W; the N decay with the V-A matrix element, half
lepton-number conserving and half violating (a Majorana N; the paper calls N "a right-handed Majorana neutrino"); a
beam energy of 14 TeV (the record says 13 TeV, but no 13 TeV setting reproduces both the particle multiplicity and the
daughter acceptance); initial-state showers limited to the factorization scale; Pythia defaults otherwise. The Pythia
version and tune of the Zenodo production are unknown.

## 3. Simulation stack and generator

- **Simulation and digitisation**: exactly `docs/new_samples_spec.md` section 2 (ACTS v15.0.0, GenericDetector, 2 T
  along z, Fatras with multiple scattering, ionisation and radiation losses and photon conversion, pMin 0.5 GeV,
  hits on sensitive surfaces, smearing digitisation without merging), run by `scripts/acts/acts_sim.py sim`.
- **Generator**: Pythia 8.244 (LCG_97apython3), pp at sqrt(s) = 14 TeV, one collision per event, no pileup, Pythia's
  default tune (Monash 2013) and PDF. Beam spot Gaussian with sigma (x, y, z, t) = (0.0125 mm, 0.0125 mm, 55.5 mm,
  5 ns). The card (`scripts/acts/hnl_card.py`, `settings(m_N, ctau)`):

  ```
  WeakSingleBoson:ffbar2W = on
  24:onMode = off
  24:addChannel = 1 1.0 100 -13 50          ! W+ -> mu+ N (W- -> mu- Nbar by conjugation)
  SpaceShower:pTmaxMatch = 1                ! initial-state emissions up to the factorization scale
  50:new = N Nbar 2 0 0 <m_N> 0.0 0.0 0.0 <c tau0 in mm>
  50:isResonance = false                    ! decayed after hadronization, at its displaced vertex
  50:mayDecay = on                          ! also for c tau0 >= 1000 mm
  50:addChannel = 1 0.5 22 -13 13 14        ! N -> mu- W+*, W+* -> mu+ nu_mu: weight (P.p_mu+)(p_mu-.p_nu)
  50:addChannel = 1 0.5 22 13 -13 -14       ! N -> mu+ W-*, W-* -> mu- nubar_mu: weight (P.p_mu-)(p_mu+.p_nubar)
  ```

  Pythia's meMode 22 weight for products (p1, p2, p3) is (P.p1)(p2.p3); this was measured on generator-level samples
  (the first product takes the smaller energy share), not taken from the manual.
- **Driver**: `scripts/acts/hnl_pythia.cc`, compiled in each production job against the same Pythia8 244 inside the
  ACTS v15 build container. It writes `event<N>-particles_gen.csv` with the particles and barcodes of the ACTS v15
  `Pythia8Generator` and `EventGenerator` (final visible particles in event-record order, beam particles skipped,
  secondary vertex = index of the exact production 4-position, primary vertex 1, position = production vertex + beam
  spot, nominal mass, float32 CSV values, rows in barcode order) and adds what the ACTS generator drops: the parent code
  of every particle and the generator truth of the W, the N and the neutrino. On the same card it agrees with the ACTS
  generator in all 17 compared quantities per point (closure appendix). The ACTS-generator form of the same card is
  `acts_sim.py pythia --process hnl --mass M --ctau C` (vertex time spread 0 there; used only for that comparison).
- **Conversion**: `scripts/acts/hnl_to_npz.py` (event conversion of `scripts/acts/csv_to_npz.py`, plus the parent
  column, tar packing and the sidecar of section 6).

## 4. Points and event counts

| sample key | directory (under `hnl_lrt/generated/`) | m_N | c tau0 | events | what it tests |
|---|---|---|---|---|---|
| `hnl5_50` | `hnl_5_50_v1` | 5 GeV | 50 mm | 3,000 | collimation at the decay radii of hnl10_100 |
| `hnl20_200` | `hnl_20_200_v1` | 20 GeV | 200 mm | 3,000 | wide pairs and shallow crossings at the decay radii of hnl10_100 |
| `hnl15_500` | `hnl_15_500_v1` | 15 GeV | 500 mm | 3,000 | far decays at the mass of hnl15_200 |
| `hnl10_100_ref` | `hnl_10_100_v1` | 10 GeV | 100 mm | 3,000 | in-stack reference of hnl5_50 and hnl20_200 |
| `hnl15_200_ref` | `hnl_15_200_v1` | 15 GeV | 200 mm | 3,000 | in-stack reference of hnl15_500 |

The two reference points repeat the Zenodo points with our card, driver, simulation and labels (added 2026-10-14
before any production event existed; section 11). They share no event with the 4,000-event closure pilots of
section 8 (different seeds) and are produced and stored exactly as the new points.

Generator level, 5,000 events per point with this card (`results/hnl_points_plan_v1.csv`; existing points for
reference):

| point | median beta gamma | r_v < 100 mm | 100-300 mm | 300-700 mm | > 1 m | median opening angle | median true abs(d0) of daughters (pT > 1 GeV, abs(eta) < 4) |
|---|---|---|---|---|---|---|---|
| 10 GeV / 100 mm | 12.6 | 0.34 | 0.32 | 0.23 | 0.05 | 0.150 rad | 28.6 mm |
| 15 GeV / 200 mm | 8.0 | 0.27 | 0.30 | 0.26 | 0.09 | 0.239 rad | 55.4 mm |
| **5 GeV / 50 mm** | 25.2 | 0.32 | 0.33 | 0.23 | 0.05 | 0.078 rad | 15.2 mm |
| **20 GeV / 200 mm** | 6.3 | 0.36 | 0.33 | 0.23 | 0.04 | 0.305 rad | 47.9 mm |
| **15 GeV / 500 mm** | 8.4 | 0.13 | 0.18 | 0.24 | 0.34 | 0.226 rad | 139.6 mm |

- **hnl5_50.** At fixed W kinematics the N momentum hardly depends on m_N, so beta gamma scales as 1/m_N and
  c tau0 / m_N = 10 mm/GeV keeps the decay-length distribution of hnl10_100: the r_v fractions agree within 0.02. The
  opening angle and the daughters' |d0| halve. The point tests close pairs and displaced tracks that point nearly back
  to the beam line (large r_v, small |d0|), which separates a dependence on r_v from one on |d0|.
- **hnl20_200.** Same c tau0 / m_N, so again the decay radii of hnl10_100, with twice the opening angle and 1.7 times
  the |d0|: wide pairs, daughters at large angles to the radial direction that cross the layers obliquely. At
  m_N = 20 GeV Pythia would make the N a resonance by default; the card sets it to a normal particle.
- **hnl15_500.** The mass of hnl15_200 with 2.5 times its lifetime: 34 % of the decays fall beyond 1 m and the
  rest spread over the tracker, so the targets weight towards far decays and large |d0| (27 % of the primary
  population at |d0| >= 100 mm, against 19 % at 200 mm). A lifetime of 1000 mm was also studied: the share of the
  primary population at r_v >= 300 mm is the same (22 %), because targets need five hits, while 3,000 events would give
  only about 685 primary daughters (154 at r_v >= 300 mm) against about 1,136 (256) at 500 mm.

**Event counts.** The primary population (HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm) of the existing
test sets, counted from the split lists and the truth fits: 603 in hnl10_100 test, 549 in hnl15_200 test, 3,861 in
the external set. Each new point must reach at least 603. The expected primary population per event is estimated at
generator level (`scripts/acts/hnl_yield.py`): the probability of at least 5 hits is taken from the full-simulation
pilot of the existing points in bins of true r_v and |eta|, applied to the generator-level daughters with
pT > 1 GeV, |eta| < 4 and true |d0| >= 10 mm or r_v >= 100 mm. The estimate gives 0.627 and 0.657 per event for the
existing points, where the pilot counts 0.642 and 0.615 (agreement within 7 %).

| point | expected primary daughters per event | events needed for 603 | expected at 3,000 events | of which r_v >= 300 mm | of which abs(d0) >= 100 mm |
|---|---|---|---|---|---|
| hnl5_50 | 0.545 | 1,107 | 1,635 | 281 | 60 |
| hnl20_200 | 0.783 | 770 | 2,350 | 302 | 515 |
| hnl15_500 | 0.379 | 1,593 | 1,136 | 256 | 304 |
| hnl10_100_ref | 0.627 (pilot count 0.642) | 963 | 1,880 | 257 | 207 |
| hnl15_200_ref | 0.657 (pilot count 0.615) | 919 | 1,970 | 310 | 374 |

Every point is produced with the cap of 3,000 events: each then holds at least 1.9 times the required population,
enough to report the efficiency in the r_v and |d0| bins of `docs/definitions.md`, at a cost of six 500-event chunks
of about 1.5 minutes each. The reference points have the same size, so that a difference between a new point and
its reference has comparable statistical errors on both sides.

## 5. Seeds, production and checks

- **Seeds** (new bases): `hnl5_50` 650001, `hnl15_500` 750001, `hnl20_200` 850001, `hnl10_100_ref` 1050001,
  `hnl15_200_ref` 1150001 (950001 is taken by the pileup overlay plan). The generator draws the beam spot
  and Pythia's random numbers from one `std::mt19937` per event, seeded with `std::seed_seq{seed, event}`; the
  simulation and digitisation use ACTS seeds `(seed + 13) + Cantor(algorithm, event)`. Event numbers run 0..2999 within
  a point; a chunk can be rerun alone and reproduces its events. Pilot, planning and lifetime-check seeds lie in
  961001-992001 or at 9710001 and above (closure section 7, section 10 here); none is a production base.
- **Production** (CPU only; A30 and GTX 1080 nodes excluded):

  ```
  sbatch --array=0-5 --export=ALL,MASS=5,CTAU=50,SEED=650001,OUT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_5_50_v1,CHUNK=500 slurm/acts/produce_hnl.sbatch
  sbatch --array=0-5 --export=ALL,MASS=15,CTAU=500,SEED=750001,OUT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_15_500_v1,CHUNK=500 slurm/acts/produce_hnl.sbatch
  sbatch --array=0-5 --export=ALL,MASS=20,CTAU=200,SEED=850001,OUT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_20_200_v1,CHUNK=500 slurm/acts/produce_hnl.sbatch
  sbatch --array=0-5 --export=ALL,MASS=10,CTAU=100,SEED=1050001,OUT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_10_100_v1,CHUNK=500 slurm/acts/produce_hnl.sbatch
  sbatch --array=0-5 --export=ALL,MASS=15,CTAU=200,SEED=1150001,OUT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_15_200_v1,CHUNK=500 slurm/acts/produce_hnl.sbatch
  ```

  The production settings are the defaults of `produce_hnl.sbatch` (14 TeV, sigma_t = 5 ns, `majorana_va`, the
  driver, 500-event chunks); the script refuses any override (pilot options such as `EXTRA`, `ECM`, `DECAY`) when the
  output lies under `hnl_lrt/generated/`, and never overwrites a shard or sidecar.
- **Checks after production, truth only, before any model reads an event** (written to
  `results/hnl_points_truth_v1.md` and RESULTS.md; `scripts/acts/hnl_points_check.py`):
  `python -m hnl.data.shards check <dir>` (six contiguous shards, 3,000 members, Zenodo keys and dtypes); SHA-256 of
  every shard and sidecar in `SHA256SUMS` in the sample directory; every event read through the tar reader
  (`hnl.data.shards`, check=True) with the format invariants of `hnl.data.scans.event_checks` and
  `scripts/acts/check_reader.py`; per event exactly one W muon and the N daughters present in the generator truth;
  the corruption rule of `docs/definitions.md` section 1 (dropped events listed in the results file); the
  same-module label-exchange test of `scripts/acts/sample_stats.py` on events 0-499 (same-module flag rate at the
  level of the different-module control, as in `results/new_samples_truth_v1.md`); the mean proper lifetime of the
  generator truth, reported with its standard error (about 1.8 % of c tau0 at 3,000 events; the card's own c tau is
  fixed far more precisely in section 10); the primary population per point and its r_v and |d0| tables, counted
  from truth. Then `splits/hnl_<m>_<ctau>_all.txt` lists every kept event.

## 6. File format and reader

- **Events**: per event an npz with exactly the keys and dtypes of the Zenodo files (`hnl.data.npz.NPZ_KEYS`), made as
  in `docs/new_samples_spec.md` section 5 (smeared measurement positions, hit-to-particle links from the
  measurement-simhit map, true edges in time order, `particles_final` as the particle table). Column 12 holds the
  generator parent code for generation-0 particles and 0 for simulation secondaries, as in the labeled Zenodo samples,
  so `hnl.truth.labels.labels_from_table` gives the N daughters (|parent| = 50) and the W muon (|parent| = 24).
- **Shards**: one uncompressed tar per 500-event chunk,
  `/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated/hnl_<m>_<ctau>_v1/events_<first>_<last>.tar` (six digits,
  inclusive), members `<event>.npz` holding the bytes `np.savez` writes, fixed member metadata, written under a
  temporary name and renamed when complete.
- **Sidecars**: `hnl_<m>_<ctau>_v1_truth/truth_<first>.npz` per chunk: `p_*` the initial state (production vertex,
  momentum) of every particle with hits, `e_*` per-event counts (measurements, dropped duplicate hits, measurement
  pulls), `g_*` the generator truth (W charge and momentum, N four-momentum, production and decay vertex, proper decay
  length, neutrino, barcodes of the W muon and the N daughters, number of `next()` calls).
- **Reader** (`src/hnl/data/shards.py`, an addition; `hnl.data.npz` and every existing sample are unchanged):
  `shards.load_event(sample, i)` and `shards.load_event_from(dir, i)` return the same dict as
  `hnl.data.npz.load_event`; `shards.iter_events(sample, indices)` reads each shard once; `shards.extract(sample, dest)`
  writes `<dest>/<dirname>/<i>.npz` byte-identical to the members for file-based tools; `shards.check(dir)` verifies
  ranges, members and dtypes. The points (the three new ones and the two reference points) are registered in
  `shards.POINTS`, not in `hnl.data.npz.SAMPLES`, so code that iterates over SAMPLES does not see them. Tests:
  `tests/test_shards.py`.

## 7. Use

- **Evaluation only**: no training, no selection (no model, configuration, threshold, working point, bin or
  hyperparameter may be chosen on these events), all kept events evaluated. Any other use needs its own commit first.
- Categories, targets, matching and the primary population as in `docs/definitions.md` for labeled samples; results on
  these points are descriptive, reported per point, and enter no preregistered claim unless a later commit says so.
- Our labels are correct, whereas the Zenodo files used for training and selection have the hit-label exchange
  (`docs/new_samples_spec.md` section 8) and the pixel-layer artifact (section 10); results on these points state
  this, as for the other generated samples.
- A property is attributed to m_N or c tau0 only through a difference between a new point and its in-stack reference
  (section 1): hnl5_50 and hnl20_200 against hnl10_100_ref, hnl15_500 against hnl15_200_ref. The comparison of a
  reference point with its Zenodo counterpart measures the effect of the simulation and labelling differences
  (section 10) and is reported as such.

## 8. Closure against the Zenodo samples

`results/hnl_points_closure_v1.md`: 4,000 fully simulated pilot events per existing point against the 8,999 Zenodo
train and val events, 58 counted tests per point (event multiplicities, PDG and parent content, the W muon and the N
daughters: pT, eta, p, hits, pixel hits, target fractions, |d0|, first-hit radius, fitted r_v, opening angle, Delta R,
pair pT, m(mu mu), charge asymmetry, targets and primary population per event). With the final card 2 of 116 tests
have p < 0.01 (1.2 expected by chance):

- the W muon has at least 5 hits in 0.939 / 0.934 of the events against 0.948 / 0.950 (one point, -3.4 standard
  deviations for both points together), because 0.6-1.3 points more W muons fall beyond |eta| = 4 at 14 TeV; no
  13 TeV setting tried fixes this without breaking the multiplicity or the daughter acceptance;
- among the N-daughter targets of 10 GeV / 100 mm, 48.3 % have at least 3 pixel hits against 45.3 % (+3.1 standard
  deviations; +1.5 at 15 GeV / 200 mm). This is the Zenodo pixel-layer artifact of section 10, not a generator
  setting: in Zenodo, a daughter whose decay vertex lies within the radial extent of a barrel pixel layer's modules
  loses every later barrel pixel hit; ours keep them. The difference sits only at decay radii between the inner
  pixel layers: at fitted r_v in [30, 72) mm, 93.9 / 88.9 % of our daughter targets have at least 3 pixel hits
  against 83.4 / 79.1 % in Zenodo (z = +7.4 / +5.3, standard errors clustered by event), at [72, 116) mm
  z = +3.6 / +1.9, and at r_v < 30 mm or >= 116 mm |z| <= 1.4 (closure section 5).

The primary population per event agrees (0.642 against 0.638, 0.615 against 0.595), as do the number of daughter
targets, their |d0| and first-hit radii, the fitted decay radii and the multiplicities. Two further differences, the
W-muon |d0| and first-hit radius, come from the hit-label exchange of the Zenodo files and are not generator settings
(closure section 5).

## 9. Storage and file budget

About 119 kB per event (pilot shards), so 5 x 3,000 events take about 1.8 GB on /ceph, plus about 0.2 GB of sidecars.
Files: 30 shards, 30 sidecars and 5 `SHA256SUMS`, so `hnl_lrt` goes from 66,309 files (2026-10-14, before
production) to about 66,374, far below the 80k stop line; /ceph has about 158 GB free. Scratch CSVs live under
`/work/submit/jinboz1/hnl/acts/work` and are deleted chunk by chunk.

## 10. Known differences from the Zenodo samples

- Labels: correct here, with the hit-label exchange in the Zenodo files.
- W-muon acceptance about one point lower (section 8).
- Beam energy 14 TeV where the record says 13 TeV (section 2); Pythia 8.244, while the Zenodo Pythia version is
  unknown.
- A failed `Pythia::next()` is retried by the driver (none happened in 8,000 pilot events); the ACTS generator would
  write the incomplete event.
- **Pixel-layer artifact of the Zenodo files.** An N daughter of the Zenodo HNL samples whose decay vertex lies within
  the radial extent of the modules of a barrel pixel layer (L1 31.4-34.4 mm, L2 71.1-73.9 mm, L3 114.9-117.7 mm, from
  `detectors.csv`) has no hit on any later barrel pixel layer in 92 % (L1, 54 of 59), 96 % (L2, 49 of 51) and 95 % (L3,
  37 of 39) of the cases (daughter targets with |eta| < 1, fitted r_v, train and val of both points; its strip hits
  remain). In our pilots: 0 of 29, 0 of 19 and 0 of 15 (generator r_v: 0 of 34, 0 of 21, 1 of 24). Daughters born 1-6 mm
  before or after a layer have no later barrel pixel hit in 7-20 % of the cases in Zenodo and 2-18 % in ours (pooled
  over both points; closure section 5): the artifact is confined to the module extent. An independent extraction with a
  different measure (fewer pixel hits than barrel layers outward of r_v, r_v windows 30-34, 70-74 and 114-118 mm, |eta|
  < 0.8) gives 95-100 %, 86-87 % and 54-62 % in Zenodo and 0-13 % in ours. It changes the ">= 3 pixel hits" subset of
  the daughter targets (section 8), not the targets or the primary population. The new points, the reference points and
  every sample of our ACTS v15 chain (`docs/new_samples_spec.md`) do not have it: a model evaluated on our samples sees
  daughters born in a pixel layer with all their later pixel hits.
- **Lifetime (20 GeV / 200 mm checked in detail).** Pythia draws the proper lifetime of the N once, at the process
  level, as c tau0 times an exponential variate; in 60,000 full events every copy of the N in the event record carries
  that value, every N decays, no event was regenerated by the driver, and the full events give 200.3 +- 0.8 mm. The
  effective c tau of the card at 20 GeV / 200 mm is 199.93 +- 0.32 mm at the process level
  (400,000 events, 0.9997 +- 0.0016 c tau0) and 199.1 +- 0.7 mm in the four full-event runs that did not raise the
  question (90,000 events, 0.996 +- 0.003 c tau0; chi2 7.5 for 3 degrees of freedom); with the first 5,000-event run
  (190.9 +- 2.7 mm), 198.7 +- 0.6 mm (95,000 events; chi2 16.0 for 4). The 35,000 events of the planning and review
  runs gave 196.1 +- 1.1 mm (0.980 +- 0.005 c tau0, -3.7 standard deviations), of which the review run alone
  (20,000 events) 196.7 +- 1.4 mm; the two new full-event runs give 199.6 +- 1.0 mm (40,000 events) and
  201.8 +- 1.4 mm (20,000, Pythia's own random numbers). The check program reproduces the review run event by event
  (2,000 of 2,000 lifetimes identical). At the process level the other points give 0.997-1.003 c tau0 (200,000
  events each; `results/hnl_points_lifetime_v1.json`). The produced samples report their own mean with its error
  (section 5).

## 11. Changes after the first commit

Logged here and in RESULTS.md; all made on 2026-10-14 before any production event of any point existed.

1. **The ">= 3 pixel hits" closure difference explained** (sections 8 and 10; closure section 5). The difference left
   open in v1 is the Zenodo pixel-layer artifact, measured with `scripts/acts/hnl_pixel_layers.py` (jobs
   6819803_0-3) on the census of the closure. Reason: a review of the closure found it; the measure, the counts and the
   conditional z values above are ours. No generator setting changes.
2. **In-stack reference points added** (sections 1, 4, 5, 7, 9): `hnl10_100_ref` and `hnl15_200_ref`, 3,000 events
   each, seeds 1050001 and 1150001, same card, driver, simulation and storage (+26 files). Reason: without them, a
   new point could only be compared with a Zenodo point, and such a difference mixes the changed property with the
   label exchange, the pixel-layer artifact, the beam energy and the Pythia version.
3. **Lifetime check** (sections 5 and 10): the 20 GeV / 200 mm deficit of the planning and review runs (35,000
   events, 196.1 +- 1.1 mm) was followed up with `scripts/acts/hnl_lifetime_check.cc` (jobs 6819714-6819720,
   6819830) before hnl20_200 was produced; the effective c tau of section 10 replaces the post-production test "within
   3 standard errors of c tau0", which at 3,000 events (1.8 % per standard error) could not detect a 2 % bias. The
   produced samples report their measured mean with its error.
