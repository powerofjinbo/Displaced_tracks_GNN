# Step H-P: gates, end-to-end dry run, and what H-T0 needs

Status: 2026-10-14. This note closes step H-P of `docs/pileup_overlay_plan.md` (design v2, bd7e357). It lists every
truth gate of section 10 that has run, with its numbers and verdict. It then reports an end-to-end dry run of the
overlay chain on five tp_v1 val hosts and the truth-only checks that can run before any model run. All work was CPU
only, on `submit-gpu-express` with the A30 and GTX 1080 nodes excluded. No model ran and no stored model output was
read. Nothing from `training_part03` and no test event was read. The thresholds are those of section 10, unchanged.

## 1. Gates of step H-P

These four checks must run before addendum H is committed (section 10, "Before H-T0").

| Gate | Rule (section 10) | Verdict | Numbers | Record |
|---|---|---|---|---|
| G0, geometry | centres, planes and corners within 4 um for every module; bijective id table | pass | 18,728 of 18,728 modules pass. Largest differences 3.744 um (centre), 3.735 um (plane), 3.754 um (corner), all in volume 17, which the patch does not touch; every patched centre rounded to 6 significant digits equals TrackML's. Id table bijective in every (volume, layer), sha256 88178fd6... | `docs/pileup_overlay_hp_g0.md`, commit 9c070f3 (jobs 6819551, 6819558) |
| F1, field | field probe in the patched build with the map: no particle failures, \|mean bending residual\| <= 0.1 MeV in every \|z\| band and z side | pass | 0 of 100,000 particles failed; largest band \|mean\| 0.033 MeV; scoring closure 0.0024 MeV with 96 samples per segment | `f1_containment_v1.md`, `f1_containment_build-v15-zmirror.json`, commit d27ed1a |
| Containment | every gun hit on its TrackML module plane and inside its bounds; every module in TrackML's `detectors.csv` | pass | 0 of 84,269 gun-probe hits fail, 0 unmapped; largest \|w\| 3.8 um (volume 17) | same |
| M1, material | descriptive | reported | section 1.1 below | `m1_v1.json` (job 6819947) |

Controls on the unpatched `build-v15`, run through the same code, fail as the plan expected. F1 fails on one failed
particle of 100,000. Containment fails for 6,083 of 84,290 gun hits, all in volumes 12 and 16. G0 fails for 2,364
modules, all in the -z endcaps.

The F1 rule applied to the gun probe, which is not the gate, fails on every build. 337 of 16,174 gun particles (2.1 %)
are lost to the v15 navigator ("Interpolation out of bounds"). Among isolated gun particles with pT > 1 GeV and
r_v >= 100 mm the share is 2.9 %. The band means of the gun probe are not significant (largest 0.130 +- 0.081 MeV).
Section 4 treats these losses as an open decision.

### 1.1 M1: material of the patched build against TrackML

We ran the field study's material code (`hnl.overlay.m1`, a port of `field_material/material.py` and
`closure_material.py`) on the field probe of the patched build. The probe has 1,000 events of 100 prompt mu, pi and
e with the map (`hp_field/build-v15-zmirror/field_sim.tar`). The TrackML side uses the truth segments of the
calibration training events. The same loop-model field is removed from both, integrated with 32 samples per
segment. With it the TrackML rows reproduce the plan's section 4.3 values: W within 0.004 MeV, dE exactly.

W is 1.4826 times the MAD of beta p theta in the polar plane per crossing; dE is the median energy change per
crossing. Pions with p = 0.7-5 GeV, prompt, by region of the first hit and |eta| of the particle. The last column is
the unpatched build in a constant 2 T field from the plan (`material.json`, `acts_const_before`).

| Region | \|eta\| | W, TrackML [MeV] | W, patched build [MeV] | ratio | dE, TrackML [MeV] | dE, patched build [MeV] | ratio | W, unpatched (plan) [MeV] |
|---|---|---|---|---|---|---|---|---|
| pixel barrel | 0-0.5 | 1.67 | 1.39 | 0.84 | -0.62 | -1.04 | 1.68 | 1.42 |
| pixel barrel | 0.5-1.0 | 1.66 | 1.60 | 0.96 | -0.62 | -1.35 | 2.18 | 1.55 |
| pixel barrel | 1.0-1.5 | 1.66 | 1.98 | 1.20 | -0.62 | -2.02 | 3.27 | 1.98 |
| pixel barrel | 1.5-2.0 | 1.70 | 2.42 | 1.43 | -0.63 | -3.16 | 5.02 | 2.36 |
| pixel barrel | 2.0-2.5 | 1.76 | 2.74 | 1.56 | -0.79 | -4.66 | 5.88 | 2.55 |
| pixel barrel | 2.5-3.0 | 2.01 | 3.45 | 1.72 | -1.11 | -7.42 | 6.71 | 3.56 |
| pixel endcap | 1.5-2.0 | 2.27 | 3.20 | 1.41 | -1.57 | -7.11 | 4.53 | - |
| pixel endcap | 2.0-2.5 | 1.59 | 0.77 | 0.48 | -0.60 | -0.91 | 1.51 | 0.87 |
| pixel endcap | 2.5-3.0 | 1.45 | 0.55 | 0.38 | -0.52 | -0.11 | 0.21 | 0.52 |
| short-strip barrel | 0-0.5 | 1.82 | 1.51 | 0.83 | -0.84 | -1.39 | 1.66 | 1.50 |
| short-strip barrel | 0.5-1.0 | 1.85 | 1.67 | 0.90 | -0.84 | -1.79 | 2.14 | 1.64 |
| short-strip barrel | 1.0-1.5 | 2.05 | 1.55 | 0.76 | -0.83 | -2.25 | 2.71 | 1.56 |
| short-strip barrel | 1.5-2.0 | 2.47 | 1.41 | 0.57 | -1.08 | -2.06 | 1.92 | 1.44 |
| short-strip barrel | 2.0-2.5 | 2.25 | 0.90 | 0.40 | -0.95 | -0.57 | 0.60 | - |
| short-strip endcap | 1.0-1.5 | 3.72 | 0.79 | 0.21 | -1.09 | -1.17 | 1.07 | - |
| short-strip endcap | 1.5-2.0 | 4.62 | 0.66 | 0.14 | -0.92 | -0.16 | 0.18 | 0.63 |
| short-strip endcap | 2.0-2.5 | 6.24 | 0.69 | 0.11 | -0.97 | -0.19 | 0.19 | 0.80 |
| short-strip endcap | 2.5-3.0 | 9.48 | 0.78 | 0.08 | -1.00 | -0.46 | 0.46 | 0.74 |
| long-strip barrel | 0-0.5 | 2.02 | 1.68 | 0.83 | -0.99 | -1.72 | 1.73 | 1.52 |
| long-strip barrel | 0.5-1.0 | 2.19 | 1.87 | 0.85 | -1.01 | -2.14 | 2.13 | 1.84 |
| long-strip barrel | 1.0-1.5 | 3.40 | 1.14 | 0.34 | -2.68 | -2.04 | 0.76 | - |
| long-strip endcap | 1.0-1.5 | 4.50 | 1.31 | 0.29 | -2.62 | -2.58 | 0.98 | - |
| long-strip endcap | 1.5-2.0 | 11.61 | 1.09 | 0.09 | -2.42 | -1.49 | 0.61 | 1.10 |

The patch and the map leave the material where the plan found it. In the pixel barrel above |eta| = 1 the patched
build scatters 1.2-1.7 times more than TrackML and loses 3-7 times more energy. In the strip endcaps it scatters 3 to
12 times less. The muon rows, where TrackML has enough muons, give the same pattern (`m1_v1.json`, key `compare`).

Last-hit closure: the median |r dphi| between the last true hit and a material-free propagation through the same
field, for prompt pions and muons.

| \|eta\| | pT [GeV] | TrackML [mm] | patched build [mm] | tracks, TrackML / patched |
|---|---|---|---|---|
| 0-1 | 1-2 | 2.00 | 1.96 | 325 / 178 |
| 0-1 | 2-5 | 1.12 | 0.79 | 75 / 222 |
| 1-1.5 | 1-2 | 1.60 | 2.28 | 313 / 172 |
| 1-1.5 | 2-5 | 0.86 | 0.93 | 87 / 228 |
| 1.5-2 | 1-2 | 2.47 | 2.23 | 345 / 165 |
| 1.5-2 | 2-5 | 1.49 | 1.25 | 55 / 235 |
| 2-2.5 | 1-2 | 2.37 | 1.64 | 324 / 193 |
| 2-2.5 | 2-5 | 1.13 | 0.77 | 76 / 207 |
| 2.5-3 | 1-2 | 1.80 | 0.81 | 325 / 291 |
| 2.5-3 | 2-5 | 0.79 | 0.54 | 75 / 109 |

Beyond |eta| = 2 the probe tracks of the patched build end closer to the material-free path than TrackML's, by up to
a factor of two at pT = 1-2 GeV. This is the material systematic that V1 measures on the model (decision D5).

## 2. Truth gates for H-T0.5 that ran during H-P

| Gate | Verdict | Numbers | Record |
|---|---|---|---|
| C1, cells and resolution | pass at matched incidence | 514 counted bins, 0 failed. Largest \|relative difference\|: span0 3.1 %, span1 2.8 %, pixel charge x cos 0.8 %, MAD(du) 18.9 % (pixel barrel, tan_u -2.4 to -1.6, 249 gun hits), MAD(dv) 12.0 %; u bias within 0.35 um; t_eff within 0.002 mm | `c1_v1.md`, commit abf1b0e |
| C1, first evaluation (bins of one angle) | fail, kept as a record | 7 of 525 bins fail, all in the pixel barrel: 6 MAD(du), 1 span1. Traced to the second incidence angle | same |
| Converter | pass | 3 val hosts (`overlay_dryrun_v1.json`) and the 5 hosts of section 3 | commit f070086 and this note |
| Emulation, full part02 sample | pass (closure: tables and gate on the same particles) | reach of the last barrel layer pi -0.26, K -0.33, p +0.16 points (rule 1); 26 hole rates within 0.059 points (rule 0.2) | `emulation_overlay_v1.md`, `emulation_gate_v1.json` |
| Emulation, split half | fail | reach passes; 4 of 26 disk hole rates miss by 0.21-0.37 points (disks 7,6; 9,12; 14,4; 14,6). chi2/ndf = 0.91 over 26, largest \|z\| 2.2 | `emulation_gate_half_v1.json` |
| Content | descriptive | section 3.6 | this note |

The matched-incidence reading of C1 was adopted after the one-angle result was seen. It needs the user's acceptance
before H-T0 (section 4).

## 3. End-to-end dry run on five val hosts

### 3.1 Setup

The chain is that of plan section 2.1. `slurm/overlay/overlay_set.sbatch` runs it as one job: the overlay gun
(`hnl.overlay.gun`), Fatras of the patched build `build-v15-zmirror` with the RZ map (`scripts/acts/overlay_sim.py
sim`), packing of the sim hits, then `hnl.overlay.writer assemble`. Assemble applies the emulation, the digitiser,
the frozen converter (`throughput.event_tables_tp`) and the merge with the host rows. `verify` then re-reads the
stored set.

| Item | Value |
|---|---|
| Hosts | tp_v1 val events 21289, 21332, 21610, 21753, 21835: the first five val ids in the order of sha256("canto-trackml-overlay-v1-pilot:<id>") |
| Set | pilot (set index 9), overlay ids 59,021,289 ... 59,021,835 |
| Base seed | 990101. The plan gives the pilot set no base seed. Fatras seed 990114 |
| Build, map | `acts/build-v15-zmirror` (executable md5 eebe8fae...), `trackml_like_rz_map_v1.txt` (md5 b5ad1cc5...) |
| Tables | `external/trackml/derived/overlay_v1/{hits,particles,ptype}/pilot/`, two shards each, plus meta, sidecar and gun truth: 9 files, 11 MB |
| Gun CSV members | `/work/submit/jinboz1/hnl/pileup_design/hp_dryrun/pilot_v1/gun_csv_pilot.tar` |
| Emulation hash / overlay hash | 2be64cc0... / 3e23b363... (section 3.2) |
| Jobs | 6819945 (chain, 14 s), 6819946 (checks, 17 s), 6819947 (M1, 93 s), 6819948 (tests: 110 passed) |

### 3.2 Integration bugs found and fixed

- **Gun particles lost in Fatras were silent.** v15 erases a particle whose propagation fails and only logs it. The
  writer kept such a particle as a row with zero hits and no record, so a lost target looked like a short track that
  never qualified. `hnl.overlay.fatras_io` now reads every failure from `fatras.log`, the packer writes the list of
  simulated particles, and `simulation_status` classifies each gun particle as simulated, failed or below pMin. It
  refuses a particle that is missing without a reason. Assemble lists the lost particles per event in the sidecar and
  writes `sim_status` into the gun truth table.
- **The module table was not tied to the build.** Assemble now reads the Fatras run record and refuses a Fatras
  seed other than base + 13. It also refuses the patched table on hits of `build-v15`, and the reverse.
- **Defaults pointed at the unpatched build.** `slurm/acts/overlay_sim.sbatch`, `slurm/overlay/fatras.sbatch` and
  `configs/trackml/overlay_sim_v1.json` now default to `build-v15-zmirror`. `fatras.sbatch` also keeps `fatras.log`.
- **The emulation hash missed the charge spread.** After C1 the pixel charge spread lives in the digitiser
  (CELL_SIGMA = 0.100), which `emulation_hash` does not cover. `writer.overlay_hash` hashes the emulation hash,
  `digitise.py` and the seed rules together, and meta and sidecar carry both hashes.
- **sbatch splits `--export` at commas.** The first submission ran one host instead of five without an error. The
  set driver takes colon-separated host ids.

The earlier conflict between the writer and the digitiser over the per-hit charge factor was already resolved in
commit f070086: the writer passes only the per-cell generator. Commits: 17dc854 (fixes, driver, tests) and 2e48c42
(checks, M1, tests).

### 3.3 Fatras

Five of 133 gun particles failed with "Interpolation out of bounds" (3.8 %). Two belong to one dense LLP that
decayed at r_v = 611 mm. The others come from LLPs at r_v = 70, 130 and 509 mm. The gun probe's 2.1 % rests on
16,174 particles and is the better estimate. No particle fell below pMin.

### 3.4 Format, ids and weights

| Check | Result |
|---|---|
| `writer.verify_set` (sha256, schema, one row group per event, converter gate against the stored host rows) | 0 problems |
| Read path: `events.load_events` with the default `table_root`; `config.event_set` on the explicit id list | works for all 5 ids |
| Rows = host + gun; gun hits are the last rows; gun CSV hit ids run 1..M | all 5 events |
| Host rows (every column) and host true edges identical to tp_v1 | all 5 events |
| Gun true edges = sum of (hits - 1) over gun particles; mixed true edges | 488 = 488; 0 |
| X1 cluster features (`events.cluster_feature`) finite for every hit | all 5 events |
| Hits on modules missing from TrackML's `detectors.csv` | 0 |
| Particle ids, read back as int64 | unique and positive; gun vertex fields 1001-1010; host vertex fields at most 226 |
| Gun particles in the tables = gun truth table; n_hits = hit count; ptype rows = particle rows | all 5 events |
| Gun particles lost in Fatras carry no hit | 5 of 5 |
| Stored gun positions = the CSV members | all 5 events |
| Gun weights | all 0 |
| Host weight sums (H-val rule, tolerance 1e-6) | largest \|sum - 1\| = 1.4e-7 |
| Closure rule: per event, the native gen-0 targets with \|d0\| >= 10 mm or r_v >= 100 mm (V1 (a)) dropped in memory | 20-24 particles and 144-190 hits per event, removed weight 0.51-0.69 %. Every event passes with the expected sum 1 - removed weight, and the check flags every event when the expected sum is left at 1 |

The converter labels gun particles by its mass-hypothesis tagging. In this run all 133 are labelled "other". An
isolated pion LLP lies in the K_S window in about 0.7 % of cases (mass log-uniform from 0.5 GeV), so a few gun pairs
of H-test may carry the K_S label. Gun populations are selected by the vertex field 1001-1010, as the sidecar and
the gun truth table do, so P_P and P_D are not affected.

### 3.5 Production path and C1 on the overlaid hits

Every gun event was recomputed from the packed sim hits with `writer.gun_event` and the set's seeds. The CSV members
are byte-identical to the tar, and their frozen conversion equals the stored gun rows in all three tables. The
digitiser's cell counts and spans equal the stored ones. Charges agree to 2.7e-6 and positions to 4.5 um, which is
the rounding to 6 significant digits of the TrackML CSV format.

C1 needs at least 200 hits per bin on each side. The dry run has 610 gun hits, so no bin counts and C1 cannot be
applied to it; the gate rests on the dedicated C1 sample of section 2. As a descriptive check we reweighted the 40
calibration events of C1 (3,726,919 hits) to the gun's own joint distribution of both incidence angles, per volume
and quantity. The 42 comparisons give chi2 = 23.8 against the bootstrap errors of the gun values, with a largest pull
of 1.85 (u bias in volume 9, five hits). The overlaid hits are consistent with TrackML at their incidence
(`hp_dryrun_checks_v1.json`, key `c1_overlay.at_gun_incidence`).

### 3.6 V3 and content (descriptive)

tp_v1 stores true hit positions and momenta for the 200 calibration events only. Val hosts have hits and particles
but no truth kinematics. The native side of V3 with incidence and residuals is therefore the C1 reference events,
and the hosts' own targets enter with cluster features only (`v3.same_event_clusters`).

| Region | gun hits | gun mean ncells | native targets, mean ncells | native displaced targets, mean ncells | gun MAD u [um] | native targets MAD u [um] |
|---|---|---|---|---|---|---|
| pixel barrel | 43 | 4.14 | 7.09 | 5.37 | 5.1 | 3.6 |
| pixel endcap | 20 | 2.90 | 1.54 | 2.27 | 9.7 | 17.9 |
| short-strip barrel | 204 | 1.91 | 1.52 | 1.50 | 19.2 | 17.4 |
| short-strip endcap | 128 | 1.88 | 1.22 | 1.22 | 19.3 | 23.1 |
| long-strip barrel | 116 | 1.66 | 1.57 | 1.50 | 28.1 | 25.7 |
| long-strip endcap | 99 | 1.83 | 1.43 | 1.35 | 27.5 | 25.4 |

Gun hits sit mostly in the strips, because most LLPs decay beyond the pixels. Their clusters follow their
incidence, which differs from that of prompt tracks. The table pools all incidences; the binned tables are in
`hp_dryrun_checks_v1.json` (key `v3`). The C1 comparison at matched incidence above is the test of the cluster
model.

Same-module near-coincidences count hits with another particle's hit (noise included) on the same module.

| Hits | another particle on the module | within 1 mm | within 0.1 mm |
|---|---|---|---|
| gun (610) | 95.6 % | 1.8 % | 0 |
| gun, host hits only | 95.6 % | 1.0 % | 0 |
| native targets of the same hosts (68,727) | 98.5 % | 5.9 % | 0.11 % |

The overlay keeps a gun hit and a host hit as two hits even when their cells overlap. TrackML's own targets show 0.11 %
of hits with a neighbour within 0.1 mm, and the gun hits of this run show none.

| Content per host (5 hosts, 50 LLPs) | Value |
|---|---|
| gun particles (charged daughters) | 133, of which 122 have hits |
| gun hits | 610 (12.2 per LLP); 632 sim hits before emulation, 14 removed by stops, 8 by inefficiency |
| targets (pT > 1 GeV, \|eta\| < 4, >= 5 hits) | 38 |
| P_P targets | 18, i.e. 0.45 per isolated LLP (plan section 7: 0.603 in the 2 T setup without emulation; 24 expected, Poisson sigma about 5) |
| P_D targets | 16, i.e. 1.6 per dense LLP (plan: 1.687) |
| smallest Delta R between targets of different LLPs | 0.19; none within 0.05 |
| modules holding gun and host hits | 531 |
| azimuth redraws per LLP | 0.38 |

## 4. What is ready for H-T0, and what remains

**Ready.** The patched build passes G0, F1 and containment, and its module table is committed. The digitiser passes
C1 at matched incidence. The emulation tables pass their closure gate. The chain from gun to stored overlay tables
runs as one job (`slurm/overlay/overlay_set.sbatch`, 14 s for five hosts), and every converter and format check
passes on its output. The truth checks of a stored set run as one job (`slurm/overlay/truth_checks.sbatch
STEP=overlay`), and M1 is measured. Scaled from the dry run, a full H-val set (150 hosts) takes minutes and fits in
one express job.

**Decisions for the user before H-T0.**

1. **Gun particles lost by the v15 navigator.** About 2 % of gun particles (2.9 % of P_P-like ones) fail in Fatras,
   on every build. The chain now records them, so the first option of the field report works as it stands: keep the
   losses and define the targets on what was simulated. The other options are patch 0002 (1.8 % remain, Core
   navigation change, not in the Python bindings) or a test of a larger overstep tolerance. The spec-v1 generic
   samples ran through the same navigator in a constant field, where 6.2 % of gun-probe particles left the world
   without a log line. They were not checked.
2. **C1 reading.** Accept the matched-incidence comparison (514 bins, 0 failed) as the reading of "bins of local
   incidence", with the one-angle result kept as a record.
3. **Emulation split half.** Read the gate as the full-sample closure, which passes, or measure on more training
   events (part01). The rule stays at 0.2 points either way.
4. **Which hash is fixed at H-T0.5.** The plan's emulation hash no longer covers the pixel charge spread.
   `overlay_hash` covers emulation, digitiser and seed rules. We suggest that addendum H fix the latter.
5. **Pilot base seed.** The plan has none. This run used 990101.

**Plan text to update at H-T0** (not edited here, since others edit the plan):
- section 3, item 2, and risk 2: the patch is built and verified (G0), and the -z id order is known;
- sections 4.4 and 5.2: the per-cell charge factor replaces the per-hit factor, and no resolution smear is needed;
- section 7: the implemented gun needs 0.58 azimuth redraws per LLP (overlay report, 30,000 LLPs), not 0.88; these
  five hosts gave 0.38;
- section 10, V3: native residuals and incidence come from the calibration events, because val events carry no truth
  kinematics;
- section 4.2: the navigator losses and their record in the sidecar.

**Work that remains.**
- V1 closure: the id scheme for re-simulated native targets is not defined. The merge and the weight rule are ready.
- C1 on the overlay at scale: H-val gives about 18,000 gun hits, enough for a share of the bins to count.
- Long-strip disk inefficiency stays an assumption (0.98 %, the pooled strip rate).
- V2 and every model run belong to H-T0.5. Streaming of part03 belongs to H-T1.

## 5. Files

| Path | Content |
|---|---|
| `src/hnl/overlay/truth_checks.py` | format, weight, production, C1-on-overlay, V3 and content checks of a stored set |
| `src/hnl/overlay/m1.py` | material check M1 |
| `src/hnl/overlay/fatras_io.py`, `writer.py` | Fatras failure record, simulation status, provenance check, `gun_event`, `overlay_hash` |
| `slurm/overlay/overlay_set.sbatch`, `truth_checks.sbatch` | set driver; checks, M1 and tests |
| `tests/test_overlay_sim_record.py`, `test_overlay_truth_checks.py`, `test_overlay_m1.py` | unit tests (110 overlay and TrackML tests pass, job 6819948) |
| `results/pileup_hp/hp_dryrun_checks_v1.json` | every number of section 3 |
| `results/pileup_hp/m1_v1.json` | M1 tables for mu and pi, closure, and the plan's rows for comparison |
| `/work/submit/jinboz1/hnl/pileup_design/hp_dryrun/` | dry-run work directory (gun, packed sim hits, Fatras log, CSV tar) |
