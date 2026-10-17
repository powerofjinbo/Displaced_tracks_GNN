# Preregistration addendum H: the TrackML pileup version of the displaced gun sample

Status: written 2026-10-14 and revised 2026-10-15, before any H-test event exists. training_part03 has not been streamed
and no overlay of an H-test host has been made.

The frozen model has run on overlay events twice, both times as a plumbing test of its stages and only on the five
pilot overlay events (decision D6, DECISIONS 2026-10-14). The first run made the graphs, the event store and the edge
scores in C-full, C-1GeV and C-hs in A30 GPU jobs. The second run computed the edge scores again, in 16-CPU jobs on the
express CPU node, from a copy of the first run's graphs and event store. Each output directory was sealed in a scratch
directory right after scoring. From them only files, schemas, row counts, provenance, timing and memory were read. No
score value was read, no efficiency, fake rate, matched count or per-target number was computed, and no track was built
from them (`pileup_design/h_fix/plumb/plumb_check_v1.json` and `plumb_check_v2.json`).

No stored model output on a gun target has been read. The addendum binds from the commit that adds it, and that hash
goes into RESULTS.md. It adds claim H1 and changes nothing in `docs/preregistration.md` or in addendum G
(`docs/preregistration_new_sample.md`).

**What exists before this commit.** Everything below is truth only, except the plumbing of decision D6 described
above and the forward check of section 5. The val projections of section 6 and that forward check read stored model
outputs of native val events, never of a gun target.

| Item | Commits | Record |
|---|---|---|
| Step H-P: patched detector build, the truth gates of section 9.1, the overlay chain, and an overlay of five tp_v1 val hosts | 9c070f3, abf1b0e, f070086, d27ed1a, 17dc854, 2e48c42, 3d80f02 | `results/pileup_hp/hp_summary.md` |
| Sets, H-test hash rule, production seeds checked by enumeration, plan text updated after H-P | 607955f, c078cb2 | `configs/trackml/split_h_v1.json`, `results/pileup_hp/h_seed_check_v1.json` |
| V1 closure machinery and its dry run on the five pilot hosts | 7c94251 | `results/pileup_hp/v1_machinery_v1.md` |
| Navigator losses in the no-pileup generic samples | ad94518 | `results/generated_nav_losses_v1.md` |
| The code that section 10 requires before any production overlay or model run, and its integration on the pilot set: the pilot overlay and the pilot closure set made again with the committed seed and numbering code, every truth gate and content check, and truth-made placeholder outputs through the blinding, the guard and the H1 code | b39e222, 5e62320, d4a5c4f, 481771e | `pileup_design/h_integrate/` |
| Second to fourth code rounds: gate rules of section 9.2 (decision D5), grants and launch pins (section 10), the model-stage chain and its job scripts, their integration, and the fourth-round rules (grants bound to the code an overlay job runs, the score placement of decision E2 and its job layout, the fixed decisive output and its committed inputs) | cbfac51, e1c92fa, 1dee6b8, 02fa773, c9bac9d, ffc35c4, 6b1aa22, e6f09b4, 837d309, 61c12af, 7604541, 021c1b8, 1a716f4 | `pileup_design/h_fix/round2_reports.md`; refusal table `pileup_design/h_fix/guard_cases_round4b.json` |
| Plumbing run of the frozen model stages on the five pilot overlay events, C-full, C-1GeV and C-hs, in A30 GPU jobs (decision D6; outputs sealed in a scratch directory; structure, provenance, timing and memory only) | c9bac9d (jobs 6822169, 6822171, 6822173); 6b1aa22 (record job 6824689) | `pileup_design/h_fix/plumb/plumb_check_v1.json` |
| Wiring run with truth-made placeholder scores of the pilot set: model-stage jobs of kind dry, track-building grids, blinded evaluation, seal and V2, H1 | 8d832f1, the code of ffc35c4 (jobs 6822147-6822149, 6822311-6822313, 6824569, 6824570) | `pileup_design/h_fix/chain_b/dry/` |
| Memory-bounded against plain forward on the native val events of the five pilot hosts (section 5) | 8d832f1 (jobs 6822303-6822307, 6822310) | `pileup_design/h_fix/chunked_check_v1.json` |

The second plumbing run belongs to this list as well: the edge-score stage again on the five pilot overlay events, in
C-full, C-1GeV and C-hs, in 16-CPU jobs on submit77-1 (jobs 6822170, 6822172 and 6822174, run at c4c44c9), each
output sealed right after scoring and checked for structure only. Record job 6824690 (at c4c44c9) verified every seal
again and wrote `pileup_design/h_fix/plumb/plumb_check_v2.json`. The seal records (`SEALED.json`) of both runs carry the
rule text of the sealed H-val directory, because the code that sealed them (c9bac9d, c4c44c9) had no rule for plumbing
outputs. `hnl.overlay.chain finish` writes the plumbing rule from 61c12af on, and the existing seals were left as they
are.

Definitions, matching and conditions are those of the model card (`docs/model_card_canto_l_trackml.md`) and of
`docs/pileup_overlay_plan.md` (design v2 with its H-T0 update, as committed in 607955f), unless this file says
otherwise. Every result holds within this simulation: TrackML throughput-phase events at <mu> = 200, with a displaced
gun simulated in a patched ACTS v15 GenericDetector.

## 1. Purpose

Addendum G tests the frozen models on a displaced gun sample in the HNL benchmark setup, which has no pileup. The route
agreed with the supervisor (DECISIONS 2026-10-14) adds a later version in which the same kind of displaced tracks sit
inside TrackML pileup events. This addendum covers that version.

The only model of this project trained on pileup events is the frozen Canto-L (TrackML). Native displaced tracks are
rare in its data: the 1,399 training events hold 352 targets with |d0| >= 100 mm. The overlay places a known displaced
population into real pileup at <mu> = 200.

Claim H1 compares the gun targets in two conditions of the model card. C-full keeps every hit. C-1GeV keeps, by truth,
only the hits of particles with pT > 1 GeV. It removes the noise hits and every particle at or below 1 GeV, from the
pileup and from the hard scatter alike, and it keeps the pileup particles above 1 GeV. H1 therefore measures how the
efficiency changes when noise and soft hits are added to the event. It is not the full effect of pileup. The model was
trained in C-full only (model card, section 3), so C-1GeV is also an input the model never saw: on val its fake rate
is 0.07 % against 1.70 % in C-full. The condition without pileup, C-hs, keeps the host hard scatter and the gun. It is
reported without a claim, and in this addendum only the comparison of C-full with C-hs is called the effect of pileup.

## 2. The sample

The route is decision D1 of the plan. The gun content of spec v1 is simulated again in a setup that reproduces the
TrackML detector, and its hits are written into TrackML host events. The hits of the existing HNL-setup gun samples
are not reused. The plan (sections 2-7, with its H-T0 update) and the H-P records specify each step.

| Step | Setting | Record |
|---|---|---|
| Hosts | stored rows of TrackML throughput-phase events, unchanged: `derived/tp_v1` (H-val, closure) or `derived/tp03_v1` (H-test) | plan sections 5.4 and 6 |
| Gun | content of spec v1 section 3 without Pythia: 8 isolated LLPs (index 1-8) and 2 dense LLPs (index 9-10) per host, at the host's hard-scatter vertex; r_v up to 800 mm, \|z_v\| < 2,900 mm. LLP directions are kept at least ΔR = 0.8 apart by azimuth redraws. After 1,000 failed azimuths, eta and r_v are redrawn as well (13 of 30,000 LLPs in a test of 3,000 hosts) | `hnl.overlay.gun`; plan section 7; `results/pileup_hp/emulation_overlay_v1.md` |
| Detector | ACTS v15 GenericDetector with the z-mirror patch of the -z endcaps (patch 0001, sha256 c7b22d68...), build `build-v15-zmirror`; module table `configs/trackml/module_map_acts_zmirror_v1.csv` (sha256 88178fd6...) | `docs/pileup_overlay_hp_g0.md` |
| Field | RZ map `trackml_like_rz_map_v1.txt` (md5 b5ad1cc5...), fitted to TrackML truth, not symmetric in z | plan section 4.1 |
| Simulation | `ActsExampleFatrasGeneric` (md5 eebe8fae...); pMin 0.1 GeV (D6); bremsstrahlung off (D7); no decays, which v15 Fatras and TrackML truth both lack; hits on sensitive surfaces; v15 material kept (D5) | `configs/trackml/overlay_sim_v1.json` |
| Emulation | per-layer hit inefficiency beyond the geometric holes, with per-module rates on the first pixel layer; hadronic stops of pi, K and p with per-layer hazards binned in momentum; measured on 949 training events of training_part02. The long-strip disks (volumes 16 and 18) could not be measured and take the pooled strip rate of 0.98 %. There the emulation misses 0.7-1.0 % of the Fatras crossings, against 2.2-3.0 % for TrackML | `configs/trackml/overlay_emulation_v1.json`; `hnl.overlay.emulate`; `emulation_overlay_v1.md` |
| Digitiser | cells on TrackML module frames and pitches; pixel charge = path length times a per-cell factor from N(1, 0.100); strip hit at the plain mean of its strip centres; no resolution smear | `hnl.overlay.digitise`; `results/pileup_hp/c1_v1.md` |
| Overlay | gun-only event written as TrackML's four CSV members and converted by the frozen converter; gun rows appended after the host rows; gun particle ids with vertex field 1001-1010; gun hit weights 0 | `hnl.overlay.writer`; plan sections 5.3 and 5.4 |

Patch 0002 (a navigator fallback) is not used.

**Version hash.** `overlay_hash` (`hnl.overlay.writer.overlay_hash`) is a sha256 over the emulation hash (tables and
`emulate.py`), `digitise.py`, the pixel charge spread CELL_SIGMA and the seed rules. It covers the steps after Fatras.
The plan fixed the emulation hash alone. Gate C1 moved the charge spread into the digitiser, which that hash does not
cover, so this addendum fixes `overlay_hash` instead (DECISIONS 2026-10-14). At this commit it is 3e23b363... (emulation
hash 2be64cc0...), the version of the H-P dry run. It may change during H-T0.5 (section 10).

**Provenance manifest.** overlay_hash does not cover the gun, Fatras or the merge. A manifest fixes them
(`hnl.overlay.writer.provenance_manifest`; code list `MANIFEST_CODE`). Each set's meta records the manifest of the run
that made it. The manifest is committed with overlay_hash at the end of H-T0.5, in
`configs/trackml/overlay_version_h_v1.json`, and holds:

- the sha256 of the production code: `src/hnl/overlay/{gun,writer,fatras_io,closure}.py`,
  `src/hnl/trackml/{overlay_sim,convert,throughput,events}.py`, `src/hnl/geometry/ids.py` and
  `scripts/acts/{gun,overlay_sim}.py`;
- the sha256 of `configs/trackml/overlay_sim_v1.json`, the emulation tables and the module table;
- the md5 of the field map and of the Fatras executable;
- the container directory. The tag `atlas/centos7-atlasos:latest-gcc8` on unpacked.cern.ch can move. On 2026-10-14 it
  resolves to `.flat/ab/ab242818...826a0`, and the manifest records that directory and the LCG view.

The H-test overlay is built with the committed overlay_hash and manifest. The Version check of section 9.4 compares
every value the H-test set records with them.

**Seeds (fixed now).** Each set has one base seed. The rules are those of the H-P chain:

- gun: `default_rng([base + 7, overlay id])`;
- Fatras: seed base + 13, so ACTS draws from (base + 13) + Cantor(algorithm, event);
- digitiser cell factors: `default_rng([base + 13, overlay id])`;
- emulation: `default_rng([base + 13, overlay id, 1])`.

| Set | k | Base | Gun | Fatras, digitiser, emulation |
|---|---|---|---|---|
| H-val | 0 | 600,000,001 | 600,000,008 | 600,000,014 |
| closure | 1 | 610,000,001 | 610,000,008 | 610,000,014 |
| H-test | 2 | 620,000,001 | 620,000,008 | 620,000,014 |
| pilot (dry runs only) | 9 | 990,101 | 990,108 | 990,114 |

The drivers read each set's base seed from `split_h_v1.json` (`hnl.overlay.split_h.base_seed`): the gun driver
`hnl.overlay.gun`, `hnl.overlay.closure make`, and `slurm/overlay/overlay_set.sbatch` and `closure_set.sbatch`.
`hnl.overlay.writer.SETS` is `hnl.overlay.split_h.SETS` and has no H-train entry. A base seed given on the command
line (`BASE`, `--base-seed`) is for dry runs only. The Version check (`hnl.overlay.writer.version_check`, section 9.2)
fails a set whose recorded base seed differs from `split_h_v1.json`.

**Fatras event numbers (fixed now).** The Fatras event number of a host is fixed before its set is simulated:

- H-val and closure: the position of the host in the val list of `split_h_v1.json` (increasing id), 0-149;
- H-test: the rank of the host in the hash order of all eligible part03 events (section 3), from 0. The 600 H-test
  hosts take 0-599 and the spares 600 onward;
- pilot: the position of the host in the pilot list (increasing id), 0-4, the numbers of the H-P dry run.

A host keeps its number when a set is split over several jobs and when another host is dropped. A spare that replaces
a dropped host keeps its own rank. The seed check covers Fatras events 0-999, so a spare of rank 1,000 or more needs the
check extended before it is simulated, and the code refuses it until then. The numbers come from
`hnl.overlay.split_h.fatras_event_numbers` (`python -m hnl.overlay.split_h fatras --set SET --hosts ...` prints them).
`hnl.overlay.gun.make_gun_set` and `hnl.overlay.closure.make_closure_set` take them explicitly, and
`scripts/acts/overlay_sim.py sim` simulates every event under its own number, with one sequencer call per run of
consecutive numbers.

The plan's base seeds are not used. Two of them, 650001 and 750001, are the bases of the HNL points hnl5_50 and
hnl15_500, so the H-val and H-test Fatras runs would have repeated their streams event by event. Below about 2.9 x 10^8
the earlier ACTS streams are too dense for a set of 16,000 new streams to avoid them. A first try with bases of 2-4
million gave 4,376-6,797 coincidences per set (job 6820083). The streams of generic_iso_x1 reach 288,656,133, and the
new bases lie above them.

We checked the bases by enumeration, as for generic_iso_x1. For algorithm numbers 1-16 and Fatras events 0-999, no
ACTS stream of the three sets equals a stream of another new set or of the pilot set. None equals one of the 57.6 M
earlier stream values recorded in this project, from 907 families: every sample of spec v1, generic_iso_x1, the HNL
points, the H-P runs and the HNL pilot ranges. The check is `python -m hnl.overlay.split_h seeds`, with its output in
`results/pileup_hp/h_seed_check_v1.json` (job 6820091). The numpy streams carry the overlay id as their second word.
No earlier numpy tuple shares its first word with a production set (`numpy_first_word_clashes` is empty for all
three), so no numpy stream repeats. One earlier run did use production-range ids: the V1 closure dry run (job
6820120) used the closure ids 51,021,289-51,021,835 with the pilot base, so its tuples begin with 990,114 and not
with 610,000,014. Its tables are in `pileup_design/hp_dryrun/closure_v1` and enter no result. The integration dry run
(job 6820567) made closure events of the same hosts and ids again, with the pilot base and the Fatras event numbers of
the closure rule (36, 42, 80, 102 and 118). Its ACTS streams lie below 1,000,000, far below every production stream,
and its tables are in `pileup_design/h_integrate/root`; they enter no result either.

## 3. Splits (fixed now)

The sets are recorded in `configs/trackml/split_h_v1.json`, written by `python -m hnl.overlay.split_h make`. An overlay
event has the id 50,000,000 + 1,000,000 k + host id.

| Set | k | Hosts | Role |
|---|---|---|---|
| H-val | 0 | the 150 tp_v1 val events (sha256 of the list c4aab801...) | development; the C1 confirmation, V2 and V3; after the overlay_hash commit, the val estimate of H1; no claim |
| closure | 1 | the same 150 hosts | gates V1 and V1-Δ: the native val targets of the V1 populations are removed and simulated again by the overlay chain; no gun LLPs |
| H-test | 2 | 600 training_part03 events chosen by the hash rule below | claim H1, read once at H-T2 |
| pilot | 9 | tp_v1 val events 21289, 21332, 21610, 21753 and 21835 | H-P dry runs (the gun pilot, and the closure dry run with k = 1 ids in a work directory), the integration dry run of the section 10 code in a scratch root, and the plumbing run of the frozen model stages (decision D6; outputs sealed in a scratch directory, no number read); enters no result |

**Hash rule for H-test.** We take every complete training_part03 event, with all four CSV members present and
non-empty, that passes the data-quality rule of the frozen converter (`hnl.trackml.convert.event_tables`, at the sha256
the manifest records). These events are ordered by key = int(sha256("canto-trackml-overlay-v1:<id>")[:8], 16), with
ties broken by id. Here <id> is the decimal event id without leading zeros, not the zero-padded file name. The first
600 are H-test and the rest are spares, in that order (`hnl.overlay.split_h.htest_lists`). No model reads a spare.
Section 9.4 lists the per-event checks after which a host is dropped and replaced by the next spare; the drops are
applied by `hnl.overlay.split_h.htest_hosts`, which the guard, the blinding code and the H1 code share. At H-T1 the file
records `htest` (the first 600, in increasing id), `htest_spares` (the rest, in hash order) and `htest_dropped` (one
record of host, reason and replacement per drop, in the order of the drops). Every eligible id stays in `htest` or
`htest_spares`. The evaluated H-test hosts are `htest` after the replacements (`hnl.overlay.split_h.htest_hosts`),
and a replacement must be the first spare in hash order that has not joined the list before. A spare that replaces
no dropped host is never evaluated.

The part03 ids exist only after the part is streamed at H-T1, so until then the rule alone fixes them. No model of this
project has read a part03 event, and no part03 directory exists. `configs/trackml/split_tp_v1.json` does not cover
part03. Its rule text reads "train = every other event of the nine parts", but its scope field limits the rule to the
two streamed parts, part01 and part02. Section 9.4 adds two checks at H-T1 that no H-test host was seen before.

**Excluded sets.** tp_v1 train gives emulation inputs from truth only and no evaluation. tp_v1 test and T2 were read
once for the model card. training_part04-09 are not used, and part04 is kept for a later retraining study. There is
no H-train set.

**Collision reuse.** Throughput events reuse generator-level collisions: 76.4 % of the val collisions have a twin in the
training events. The part03 hosts are probably similar. At H-T1 their signatures are computed from truth and matched
against the training signatures. The twin fractions are committed then, and the host-integrity and C-hs numbers are
reported split by twin status. The gun particles are new, so the gun targets are unaffected.

## 4. Population and metric

- **Gun particles** are the particles with vertex field (particle_id >> 52) in 1001-1010: 1001-1008 isolated, 1009-1010
  dense. They are selected by this field alone. The converter's category label is not used, since it may call a few
  gun pairs K_S.
- **Targets** as in the model card: pT > 1 GeV, |eta| < 4, at least 5 hits, double-majority (DM) matching. The hits are
  those written in the overlay event, after Fatras, emulation and digitisation.
- **P_P**: isolated gun targets with truth |d0| >= 10 mm or r_v >= 100 mm. This is the definition of P_G in addendum G
  section 4. d0 is the converter's: the helix through the production vertex with |B| = 2 T, taken about the z axis.
  r_v is the transverse distance of the production vertex from the z axis.
- **P_D**: the same cuts on the dense component.
- **Navigator losses.** The v15 navigator loses about 2 % of gun particles on every build ("Interpolation out of
  bounds"), and 2.9 % of isolated ones with pT > 1 GeV and r_v >= 100 mm. Fatras removes such a particle with all its
  hits. The chain keeps its particle row, with no hits, and records it with sim_status "failed" in the sidecar and the
  gun truth table. Without hits it is not a target. P_P and P_D are therefore defined on what was simulated, and the
  losses enter neither condition of H1 (DECISIONS 2026-10-14). The losses are not random. The two mechanisms found
  pick out tracks that start inside a volume and move inward, and shallow crossings of volume boundaries
  (`results/pileup_hp/f1_containment_v1.md`). P_P is short of such tracks, and section 7 bounds the effect on Δ.
- **Silent ends.** In the gun probe of the patched build, 36 gun particles that Fatras simulated end outside the
  detector world without a log line (`f1_containment_build-v15-zmirror.md`). The chain classes them as simulated. The
  content check of section 9.2 counts them per set.
- **Metric.** ε(P, C) is the number of DM-matched targets of population P in condition C over the number of targets,
  pooled over the events of a set, in percent. The track fake rate (tracks with at least 5 hits that DM-match no
  particle), the number of fake tracks containing a gun hit per LLP and the duplicate rate are reported with it.

## 5. Model and conditions

- **Model.** The frozen Canto-L (TrackML), run `canto_l_v1/seed0_x1`, selection `configs/trackml/selection_v1.json`
  (sha256 a2b30f4f...). `hnl.trackml.freeze.verify` checks its files at the start of every job that runs a model
  stage, builds tracks or evaluates outputs on overlay events (`python -m hnl.overlay.chain preflight` in
  `slurm/overlay/chain.sbatch`, `chain_score.sbatch`, `grid.sbatch`, `blind.sbatch`, `h1.sbatch` and `v1.sbatch`,
  before the guard of section 10 and before any stage). Edge scores use the memory-bounded forward over the whole event
  (`hnl.trackml.chunked`).
- **Forward check.** The stored native val outputs that V1, V1-Δ and V2 compare with were scored with the plain
  forward. On the native val events of the five pilot hosts, the memory-bounded forward differs from the stored scores
  by at most 1.41e-5 in C-full (9,797,297 edges; 4,881 differ by more than 1e-6 and 16 by more than 1e-5) and by at
  most 2.4e-6 in C-1GeV (265,842 edges). One C-full edge, in event 21332, crosses the connected-components cut 0.8 of
  WP-R-P3. No C-full edge crosses 0.1, 0.2 (the first pass of WP-A and WP-S), 0.3 or 0.5, and no C-1GeV edge
  crosses any of the five cuts. In C-full, 5 stored scores lie
  within their event's largest difference of 0.1, 1 of 0.2, 1 of 0.5 and 2 of 0.8. The effect on the edge order of
  the Mutex Watershed was not checked. Record: `pileup_design/h_fix/chunked_check_v1.json` (jobs 6822303-6822307 and
  6822310, 6 threads on submit77-1).
- **No other model.** `canto_l_v1/seed0` has no edge classifier. acorn is not trained on TrackML in this project. The
  HNL-benchmark models were built without pileup. In a zero-shot test on accuracy-phase events, the edge classifier of
  Canto-B seed 0 kept 4-7 % of the true edges at its HNL cut (`docs/trackml_feasibility.md` section 3).
- **Working points.** WP-A is primary: Mutex Watershed with learned merging at tau_merge = 0.3, chosen on val at
  FR* = 0.029368. WP-S (tau_merge = 0.1) and WP-R-P3 (connected components at edge score 0.8) are reported. All three
  are frozen, and none is chosen on an overlay event.

| Condition | Hits kept | Use |
|---|---|---|
| C-full | every hit, noise included | primary; H1 |
| C-1GeV | hits of particles with pT > 1 GeV, noise removed (truth; as in the model card) | H1; outside the training density |
| C-hs | hits of the host hard scatter (vertex field 1) and of the gun; other pileup particles and noise removed (truth) | reported only; outside the training density |

C-nonoise is not run in H.

Every gun target has pT > 1 GeV, so C-1GeV keeps every hit of every gun target, and P_P is the same list of targets in
both conditions. C-1GeV does remove the gun's own particles at or below 1 GeV. Each isolated LLP decays to two charged
prongs. For 10.5 % of the generator-level P_P-like particles of the gun probe (6,943 particles), the other prong has
0.1 < pT <= 1 GeV. In the pilot overlay 2 of the 18 P_P targets have such a sibling with hits, and gun particles at or
below 1 GeV carry 157 of the 610 gun hits. For these targets Δ includes the removal of their sister track, which is not
pileup. C-1GeV is kept as in the model card, so that H1 and the native val projection use one condition. Section 7
reports Δ without these targets.

The C-hs filter is new truth-filter code, committed before it runs (`hnl.overlay.conditions`, d4a5c4f). It keeps
the particles with vertex field 1, the host hard scatter with its simulation secondaries, and those with 1001-1010, cuts
the particles table to them, and refuses closure events. C-full and C-1GeV are the frozen conditions of
`hnl.trackml.events`. The model stages (graph, event store, edge scores) run on overlay events through
`hnl.overlay.chain`, which calls the frozen stages unchanged and applies C-hs through `hnl.overlay.conditions` inside
those calls only. On the production sets the graph additions and the event store run in GPU jobs
(`slurm/overlay/chain.sbatch`, one block of up to 50 events per job) and the edge scores in CPU jobs on the express CPU
nodes (`slurm/overlay/chain_score.sbatch`, 6 CPUs, at most 5 events of a block per job in C-full and 50 in C-1GeV and
C-hs; decision E2, section 10). `chain.sbatch` refuses a production stage list
that holds the scores, and `hnl.overlay.chain run` refuses the scores together with another stage in one job on a
production set. Track building and matching use the frozen builders and `hnl.trackml.evaluate.event_counts` at WP-A,
WP-S and WP-R-P3: through `hnl.overlay.tracks` (`slurm/overlay/grid.sbatch`) for the closure set, for H-val in C-1GeV
and C-hs, and for H-test in all three conditions, and through `hnl.overlay.blind` alone (`slurm/overlay/blind.sbatch`)
for H-val in C-full. No model code changes.

## 6. Claim H1

**Δ = ε(P_P, C-full) − ε(P_P, C-1GeV)** on the same targets, for the frozen Canto-L (TrackML) at WP-A on H-test, in
percentage points. It is the change of efficiency when noise and the hits of particles at or below 1 GeV are added
(section 1).

- **Uncertainty.** σ is the standard deviation, with ddof = 1, of Δ over B = 500 paired event-bootstrap resamplings.
  The index table is that of `hnl.trackml.evaluate.bootstrap`: `default_rng(12345).integers(0, n, size=(500, n))` over
  the n H-test events, ordered by overlay id. Each resampling recomputes both pooled efficiencies on the same events.
  The outcome inequalities use the unrounded Δ and σ.
- **No seed term.** One model with one training seed appears in both conditions.
- **Margin δ = 1.0 percentage point** (section 8; decision D13).
- **No fake-rate guard.** The two conditions share the model and the working point, and they differ only in the input
  hits. The fake rate is far lower in C-1GeV by construction (1.70 % against 0.07 % on val, model card section 5.1).
  Both fake rates are reported.
- **Code.** The H1 code (`hnl.overlay.h1` with `hnl.overlay.hstats`, run by `slurm/overlay/h1.sbatch`) selects the
  gun targets by vertex field, sets the P_P flags, pairs the targets across the two conditions, runs the bootstrap and
  states the outcome. It checks its index table against `hnl.trackml.evaluate.bootstrap` on every run, and a
  resampling with no P_P target would be skipped. On H-val it reads C-full only from the sealed directory of section
  10, through `hnl.overlay.blind.sealed_per_target`, and C-1GeV and C-hs from the grids of `hnl.overlay.tracks`. On
  H-test it reads all three conditions from those grids. Every run reads C-full, C-1GeV and C-hs; a run given fewer
  stops before any other check. `h1.sbatch` takes on H-val and H-test only the grids that `slurm/overlay/grid.sbatch`
  writes (`python -m hnl.overlay.chain paths --key grid`), and the H1 code reads each only after the check that V1
  also makes (`hnl.overlay.tracks.grid_provenance`): every event of the set, the frozen classifier's scores in that
  condition in every block, the learned merge model of `selection_v1.json`, the production overlay tables and no
  placeholder. On H-val and H-test a run writes one fixed file per working point,
  `results/pileup_hp/h1_<set>_<WP>_full-1gev-hs_v1.json`, and refuses any other output path. It takes the restriction
  of P_P (section 9.3), the failed flag gates, the navigator-loss share (section 7) and the descriptive state only from
  a committed record under `results/pileup_hp/` (`--restriction-record`; on H-val an H-val record `h_t05_*`, on
  H-test an H-T1 record that R15 names). The set's launch pins that record, and the run refuses a record that its
  launch record does not pin. It refuses these four inputs as command-line arguments whenever they are given, with any
  value, an empty or zero one included (decision D7, section 7). The pilot keeps them as arguments. The H1 code was
  committed before this addendum (d4a5c4f, 481771e; the grant, launch and write-once checks of section 10 and the
  navigator-loss minimum of section 7 in 1dee6b8; the fixed output path, the pending record and the committed record in
  61c12af; the launch pin of the H-val record and the refusal of empty or zero arguments in 021c1b8) and tested on
  truth-made placeholder outputs of the pilot set, as addendum G's T2 code was (7a68ffd, e0c4307). It runs unchanged at
  H-T2.

### Outcomes

H1 is reported as one of:

- **holds**: Δ − 2σ > −δ;
- **does not hold, resolved below the margin**: Δ + 2σ < −δ;
- **does not hold, inconclusive**: otherwise.

If the fallback of section 9.3 restricts P_P to detector regions, H1 is evaluated on the restricted population, as
committed before H-T1. If a flag gate of section 9.3 failed, H1 is reported with that flag.

**Statistics check.** If H-test holds fewer than 1,000 P_P targets, after any restriction, H1 is still evaluated as
written and is labelled underpowered. The count is committed at H-T1, before any model reads an H-test event. Two
yields bracket it. The five-host dry run gave 18 P_P targets, 0.45 per isolated LLP, which scales to about 2,160 on 600
hosts (Poisson uncertainty about 25 %). The gun probe of the patched build (600 events, 4,800 isolated LLPs) has 0.69
P_P-like particles per isolated LLP with at least 5 Fatras hits before the emulation, about 3,330 on 600 hosts. The
plan's 2,890 lies between. Under the fallback of section 9.3, central targets alone (48 % of P_P in the plan) would
number about 1,040-1,600. The H-val overlay measures the yield from truth at H-T0.5.

### Val projection (stated in advance)

The stored val outputs of the frozen model (job 6812139, `study_design/val_discordance.json`) give the paired change of
native targets between C-1GeV and C-full. No native population resembles the gun targets in every respect, so all five
rows of that file are given, with equal standing, together with a reweighted projection.

| Native val targets | n | Δ [points] | σ, val | σ, projected (2,894 / 2,160 targets) | holds / inconclusive / fails, 2,894 targets | the same, 2,160 targets |
|---|---|---|---|---|---|---|
| gen-0, \|d0\| >= 10 mm or r_v >= 100 mm (the P_P cut) | 3,276 | -0.61 | 0.16 | 0.17 / 0.19 | 0.59 / 0.41 / 0.00 | 0.50 / 0.50 / 0.00 |
| gen-0, \|d0\| >= 10 mm | 1,562 | -0.70 | 0.25 | 0.19 / 0.22 | 0.41 / 0.58 / 0.02 | 0.34 / 0.64 / 0.01 |
| all, \|d0\| >= 10 mm | 2,010 | -1.39 | 0.37 | 0.31 / 0.36 | 0.02 / 0.66 / 0.32 | 0.02 / 0.72 / 0.26 |
| gen >= 1, \|d0\| >= 10 mm | 448 | -3.79 | 1.37 | 0.54 / 0.62 | 0.00 / 0.12 / 0.88 | 0.00 / 0.15 / 0.85 |
| all, \|d0\| >= 10 mm and \|d0\|/r_first >= 0.2 (not pointing back) | 363 | -4.41 | 1.38 | 0.49 / 0.57 | 0.00 / 0.05 / 0.95 | 0.00 / 0.06 / 0.94 |

Sources: `pileup_design/plan_revision/sigma_projection.json` (2,894 targets) and
`results/pileup_hp/addendum_h_inputs/rw_projection.json` (2,160 targets and the reweighting below; job 6820262;
committed in c0d689a). The projected σ scales the val σ to the target count. The outcome odds take the H-test Δ as
normal around the val Δ, with the val and projected variances added.

The first row applies the P_P cut, but its targets are of another kind. 91 % of them are K_S and Lambda daughters,
which point back to the beam line: 0.7 % have |d0|/r_first >= 0.2. The ratio reaches 0.2 for 10 of the 18 P_P
targets of the pilot overlay and for 45 % of the gun probe's P_P-like particles with at least 5 Fatras hits. The
frozen model finds the first row's targets at 98.7 % in C-full. The last row holds the native targets that do not
point back. The model finds them at 45 % in C-full, and their Δ is about seven times that of the first row. The rows
with all displaced targets mix both kinds. Particles produced in the detector simulation (gen >= 1) are 22 % of the
|d0| >= 10 mm row but carry 17 of its net 28 discordant pairs.

For the reweighted projection, the native targets with |d0| >= 10 mm or r_v >= 100 mm take the gun's distribution in
r_first (below 200 mm, 200-700 mm, above 700 mm) and |d0|/r_first (below 0.05, 0.05-0.2, 0.2 and above). The gun side
is the probe of the patched build, with r_first from its Fatras hits before the emulation. The natives cover 99.6 % of
the gun's weight. Generation-0 natives leave the cells that do not point back almost empty (24 targets carry 45 % of
the weight), so the version with all generations is the better-supported one.

| Reweighted native val targets | n | Δ [points] | σ, val | σ, projected (2,894 / 2,160) | holds / inconclusive / fails, 2,894 | the same, 2,160 |
|---|---|---|---|---|---|---|
| all generations | 3,747 | -3.4 | 0.86 | 0.50 / 0.57 | 0.00 / 0.07 / 0.93 | 0.00 / 0.10 / 0.90 |
| generation 0 | 3,276 | -1.8 | 1.35 | 0.22 / 0.26 | 0.19 / 0.22 / 0.60 | 0.17 / 0.25 / 0.58 |

Each projection assumes that a gun target behaves like a native target of the same kind. If gun targets behave like
the natives that do not point back, Δ lies several points below −δ and 2σ is about δ. H1 would then most likely be
resolved below the margin, and "holds" would need Δ near zero or above. The margin is kept (section 8), and this outcome
is accepted in advance. At H-T0.5, before the H-val estimate, the reweighting is repeated with the H-val gun truth
after the emulation and stated beside the val estimate (section 10).

## 7. Reported without a claim

- ε(P_P), ε(P_D) and the efficiency of all gun targets in C-full, C-1GeV and C-hs, at WP-A, WP-S and WP-R-P3.
- The same in the bins of addendum G section 7 (|d0|, r_v, r_first, |d0|/r_first, pT and dense ΔR neighbours), in signed
  eta and species, and in local pileup hit density. The density of a target is the mean number of host hits, noise
  included, per module over the modules that hold its hits, in C-full. Its bins are the tertiles over the H-val P_P
  targets, computed from truth and committed with overlay_hash.
- Δ on the P_P targets whose LLP has no other charged daughter with pT <= 1 GeV and at least one hit, and the number
  of P_P targets that have one (section 5).
- The stage at which each lost target is lost (graph, scores or building; `hnl.trackml.displaced_diag`), split by the
  sign of eta.
- Fake rate, fake tracks containing a gun hit per LLP, and duplicate rate.
- Host prompt efficiency, fake rate and TrackML score on H-test, beside the model card values, split by collision-twin
  status.
- The val estimate of H1 and the gun results on H-val, computed once per working point after the overlay_hash commit,
  on the population H1 uses (any restriction of section 9.3 included) and on all of P_P.
- Navigator losses per set: the number of lost gun particles and their share among generator-level P_P-like particles
  (isolated, pT > 1 GeV, |eta| < 4, |d0| >= 10 mm or r_v >= 100 mm). They are binned in r_v, eta, |d0|, |d0|/r_v, the
  direction at the vertex (inward or outward) and the bins of addendum G section 7. Beside H1 we give a bound on their
  effect on Δ: the lost share times the largest |Δ_bin − Δ| over the bins of addendum G section 7 that hold at least
  20 P_P targets. The bins with fewer P_P targets are listed with their Δ_bin and do not enter the bound (decision
  D5; `hnl.overlay.gun_content` for the losses, `hnl.overlay.h1.nav_loss_bound` with `NAV_MIN_PP` for the bound).
- The P_P and P_D counts with the long-strip disk drop rate raised until the emulation misses 2.6 % of the Fatras
  crossings there, the middle of TrackML's 2.2-3.0 % (truth only, H-val at H-T0.5 and H-test at H-T1).
- The gate results of section 9, including the descriptive checks M1, V3 and content, the first C1 evaluation, the
  split-half emulation check and every flag gate.
- G beside H for the same gun content. The detector and the model differ, so this is not a comparison.

The read-outs of this section that the H1 code does not compute at this commit, and the code of H-T1, are written after
this commit (decision D7). Each is committed and tested on truth or placeholder outputs before the step that uses it.
R15 names their files, and the H-test launch pins every file R15 names (section 10). One part of D7 is a precondition of
any H-val or H-test run of the H1 code (DECISIONS 2026-10-15). The D7 code writes, for each of the two sets, the record
that gives the run its P_P restriction, its failed flag gates, the navigator-loss share and whether H is descriptive: a
tracked file under `results/pileup_hp/`: for H-val an H-val record (`h_t05_*`), committed with the version record of
H-T0.5 step 4 and pinned by the H-val launch; for H-test an H-T1 record (`h_t1_*`), which R15 names and the H-test
launch pins. Until it is committed the H1 code refuses the set, since it takes these four inputs from no
other source (section 6, Code).

The same navigator losses were checked in the no-pileup generic samples of addendum G (ad94518,
`results/generated_nav_losses_v1.md`). In the constant 2 T field of spec v1, 6.6 % (isolated) and 5.2 % (dense) of the
simulated gun particles fly out of the world at the loop protection without a log line. Counting every missed
crossing, 596 of 8,543 would-be isolated targets fall below 5 hits. The patched build with the RZ map logs its
losses instead, and the content check of section 9.2 looks for the silent kind as well.

## 8. Why δ = 1.0 point

- **Same margin as S1 and G1.** It is the margin of claims S1 and G1, so H1 states its non-inferiority on the same scale
  as the other claims of the paper.
- **What it means for a vertex.** A displaced vertex with two tracks needs both of them. For efficiencies near one, a
  loss of δ per track becomes a loss of about 2δ, two points, in the efficiency of a two-track vertex.
- **Scale.** Between C-1GeV and C-full the frozen model loses 0.11 points of all val targets (model card, section 5.1).
  Training acorn on the signal gains (1.48 ± 0.35) points on HNL validation (addendum G section 8). That is another
  model, sample and question, and it is given only as a scale.
- **What was known when δ was set.** Design v1 of the plan, where 1.0 point was first proposed, was never committed, so
  when it was written cannot be checked. The paired val rows (`study_design/val_discordance.json`, written 2026-10-14
  at 11:58 EDT) were on disk two hours before design v2 wrote δ down (bd7e357, 13:56 EDT) and decision D13 adopted it
  (f0cf4a2, 14:15 EDT). They include the P_P-cut row
  (−0.61), the generation-0 |d0| >= 10 mm row (−0.70), all |d0| >= 10 mm (−1.39) and the targets that do not point back
  (−4.41). δ was not moved after the projections of section 6 (decision D13).
- **Uncertainty.** For native targets like the first two rows of section 6, 2σ_proj is 0.33-0.43 points and stays below
  δ. For targets that do not point back, σ_proj is about 0.5-0.6 points, so 2σ_proj reaches δ. H1 can then come out
  only as resolved below the margin or inconclusive, unless Δ is near zero.

## 9. Gates

Every threshold below is fixed by this commit. Checks marked descriptive have no pass rule and are reported as tables.
Section 9.3 sorts the gates into decisive gates and flag gates.

### 9.1 Step H-P (done before this commit; truth only)

| Gate | Rule | Result | Record |
|---|---|---|---|
| G0, geometry | centres, planes and corners within 4 µm for every module; bijective id table | pass: 18,728 of 18,728 modules; largest difference 3.75 µm, the rounding of TrackML's file | `docs/pileup_overlay_hp_g0.md` |
| F1, field | field probe of the patched build: no particle failure; \|mean bending residual\| <= 0.1 MeV in every \|z\| band, on each side | pass: 0 of 100,000 particles failed; largest band 0.033 MeV | `results/pileup_hp/f1_containment_v1.md` |
| Containment | every gun hit on its TrackML module plane and inside its bounds; every module in TrackML's `detectors.csv` | pass: 0 of 84,269 gun-probe hits fail | same |
| M1, material | descriptive | reported: the patched build scatters 1.2-1.7 times more than TrackML in the pixel barrel above \|eta\| = 1, and 3-12 times less in the strip endcaps | `hp_summary.md` section 1.1 |
| C1, cells and resolution | per volume, at matched local incidence, bins with at least 200 hits on each side: mean span0 and span1 within 10 %; effective thickness within 0.04 mm; pixel charge x cos(incidence) within 5 %; mean u bias within 1 µm; residual MAD within 20 % (u and v in the pixels, u in the strips) | pass: 514 counted bins, none fails | `results/pileup_hp/c1_v1.md` |
| Emulation | full-sample closure on part02: total hole rate per barrel layer and per disk within 0.2 points of TrackML; share of pi, K and p reaching the last barrel layer within 1 point | pass on the 26 layers and disks it can measure: hole rates within 0.06 points; reach within 0.33 points. The long-strip disks (seven interior crossings) are untested, and their rate is assumed | `results/pileup_hp/emulation_overlay_v1.md` |
| Converter | as in section 9.2 | pass on 3 and on 5 val hosts | `hp_summary.md` section 3.4 |

The emulation gate does not cover the miss rates relative to Fatras. On the first pixel layer the emulation misses 8.7 %
of the crossings against TrackML's 10.0 %. On the long-strip disks it misses 0.7-1.0 % against 2.2-3.0 %. V1 lists the
long-strip endcaps as a region of their own, and section 7 reports the effect of the disk rate on the counts.

**Readings settled after a first result.** Four choices of step H-P were made after a first result was seen. The
thresholds did not change, and every earlier result is kept and reported.

1. **C1.** The first evaluation used plain bins of the driving incidence angle. It failed 7 of 525 bins, all in the
   pixel barrel. The gun fills the second incidence angle differently from TrackML. We then compared at matched
   incidence: within each bin the gun hits are reweighted to TrackML's joint distribution of both angles. All 514
   counted bins pass. Because of this order, the C1 confirmation on H-val (section 9.2) is added, with its rule fixed
   here before the H-val overlay exists.
2. **Emulation.** The plan did not say on which events the gate runs. The full-sample gate shares its particles with
   the tables and passes. A split-half check, with tables from the even events and the gate on the odd ones, fails four
   of 26 disk hole rates by 0.21-0.37 points (chi2/ndf = 0.91 over the 26 differences). The gate was read as the
   full-sample closure after both results were known (DECISIONS 2026-10-14).
3. **F1 scoring.** The first F1 reading integrated the field with 6 samples per segment and found +0.100 MeV in one
   band. A closure test showed that this scoring alone errs by more than the threshold, and the gate uses 96 samples.
   The record of this choice is `f1_containment_v1.md`; it is not among the five H-P choices in DECISIONS.
4. **Navigator losses.** They were found by applying the F1 rule to the gun probe. The decision to keep and record them
   (section 4) came after that.

### 9.2 Gates of H-T0.5

**Per set, truth only.** These run on H-val and closure at H-T0.5. Section 9.4 gives their form on H-test at H-T1.
The code is `python -m hnl.overlay.truth_checks run` for a gun set and `python -m hnl.overlay.closure_checks run` for
the closure set, and `hnl.overlay.writer verify` and `hnl.overlay.closure verify` re-read the stored shards.

- **Converter, gun sets.** Every gun-only event passes `event_tables`. In the stored tables, read back as int64, every
  particle id is unique and positive. Every gun vertex field lies in 1001-1010, and every gun weight is 0.
- **Converter, closure set.** The closure id is the native id + 2^62 (`hnl.overlay.closure` at 7c94251). Every closure
  vertex field lies in 1024-2024, and no particle has a field in 1001-1010. Each closure id minus 2^62 is a native of
  the population table, with one closure row per replaced native, and every closure weight is 0. The converter sets
  `is_hard_scatter` to False for closure rows and drops the K_S or Lambda label of a daughter whose partner stays in
  the host. Population, region, hard-scatter flag and category of a closure target are therefore taken from its native,
  through the closure truth table.
- **Weight sums.** In H-val and H-test events the host weights sum to one within 1e-6. In closure events the sum equals
  one minus the weights of the removed native hits, with the same tolerance.
- **Containment.** Every gun hit lies on a module of TrackML's `detectors.csv`. The digitiser moves no crossing by more
  than 4 µm onto its module plane, and no crossing lies more than 4 µm outside the module bounds. The two 4 µm rules
  are read from the digitiser record of every event in the set's sidecar (`hnl.overlay.truth_checks.containment_checks`,
  for the gun and the closure sets).
- **Host rows (V2, part 1).** In every event the host rows equal the stored rows of the host in every column, and the
  host true edges are unchanged. Read through `hnl.trackml.events.load_events`, the event without its gun rows equals
  the event read from the host root, array by array, true edges and X1 cluster features included. The host root is the
  one the set's meta records, checked against the set's rule: `derived/tp_v1` for H-val, closure and pilot, and
  `derived/tp03_v1` for H-test (`hnl.overlay.hosts`). For closure events the same holds with the closure rows removed
  on one side and the replaced natives on the other. Every host belongs to the set's list in `split_h_v1.json`.
- **Version** (`hnl.overlay.writer.version_check`). The set's meta and sidecar carry its overlay_hash and its base
  seed, and the base seed equals that of `split_h_v1.json`. The Fatras seed is the base seed + 13, and each host's
  Fatras event number follows the rule of section 2. The meta records the manifest of section 2. In the last pass the
  set's recorded code hashes, executable and map md5 and container define the manifest. Once
  `configs/trackml/overlay_version_h_v1.json` is committed, overlay_hash and every manifest value of a set must equal
  it.
- **Content (descriptive).** Hits per LLP; P_P and P_D yield per LLP; species shares against the generic train split;
  gun-gun ΔR; gun-host hit pairs on one module (`hnl.overlay.truth_checks`); navigator losses in the bins of section 7;
  simulated gun particles that end outside the world without a log line, from the Fatras final states (r > 1,030 mm
  or |z| > 3,002 mm); premature ends and holes, by the method of ad94518 (crossings on a module that a particle's
  helix still makes after its last hit, and layers skipped between hits), on the Fatras hits before the emulation and
  on the stored hits (`python -m hnl.overlay.gun_content run`). For the misses each particle is propagated without
  material through the RZ map until it leaves the world (r = 1,028.538 mm or |z| = 3,001 mm) or reaches the loop
  limit of the v15 propagator, a path of π p / (0.2998 |B(vertex)|).

**C1 confirmation (H-val gun hits, truth only).** The code is `hnl.overlay.c1` at abf1b0e (bin edges `_ABS`, at least
200 hits per side, four reference sub-bins for the matching) and `hnl.overlay.truth_checks.at_test_incidence` at
2e48c42, run by `python -m hnl.overlay.c1_confirm run`. The reference is the same 40 calibration events as in C1. The
confirmation has three parts:

1. **Matched bins.** The per-bin rules of C1 at matched incidence, in the bins that count (spans 10 %, pixel charge x
   cos 5 %, MAD(du) and pixel MAD(dv) 20 %), and the mean u bias within 1 µm over the counted bins of a volume. The
   effective-thickness fit of C1 is reported and decides nothing here. With no counted bin, part 1 is "not
   informative"; with one or two counted bins it is reported as "few bins".
2. **Pooled per volume.** For every volume and C1 quantity with at least 200 H-val gun hits on that quantity's common
   support (it depends on the angles the quantity is binned in), the reference is reweighted to the gun's joint
   distribution of both incidence angles. The quantity must then lie within its C1 tolerance: spans 10 %, pixel
   charge x cos 5 %, MADs 20 %. The mean u bias fails only if the difference exceeds max(1 µm, 2 σ_diff), with
   σ_diff = sqrt(σ_gun² + σ_ref²): σ_gun is the bootstrap standard deviation of the gun mean (B = 200,
   `default_rng(1)`, as `at_test_incidence` reports it), and σ_ref is the standard error of the reweighted reference
   mean with its weights held fixed, sqrt(Σ w² (x − mean)²) / Σ w (`hnl.overlay.c1_confirm.ref_mean_se`). Each row
   reports its rule, its limit, σ_ref and σ_diff.
3. **Support.** Per volume, the share of gun hits and of P_P-target hits outside the common support is reported. C1
   does not validate these hits. In C1, the matched comparison in the strip endcaps carried about 80 % of the gun hits,
   and in the pixel barrel the gun reaches |tan_v| = 16 while TrackML stops near 10.

The confirmation passes if no counted bin and no volume's mean u bias fails part 1 and no counted volume and quantity
fails part 2. It fails if either part fails. If neither part counts anything (no counted bin in part 1 and no volume and
quantity with 200 gun hits in part 2), its outcome is "not informative": it neither passes nor fails and blocks
nothing (ratified in the third code round, DECISIONS 2026-10-15).

The u-bias rule of part 2 was set after a first result. In the integration dry run on the pilot set, volume 13 failed
the plain 1 µm rule with 204 gun hits, a difference of −1.59 µm and a gun standard error σ_gun = 1.16 µm: at the 200-hit
floor the standard error is about the size of the tolerance (`pileup_design/h_integrate/out/c1_confirm_pilot.json`). The
rule max(1 µm, 2 σ_diff) was fixed then (decision D5, DECISIONS 2026-10-14), before any H-val event existed. Under it
pilot volume 13 passes, with σ_diff = 1.16 µm and a limit of 2.32 µm
(`pileup_design/h_fix/gates/out/c1_confirm_pilot.json`, job 6821095). No other C1 tolerance changed.

Scaled from the dry run, H-val holds about 18,300 gun hits: about 1,290 in the pixel barrel and about 450 and 150 in the
pixel endcaps. We counted the dry-run gun hits per C1 bin and multiplied by 30
(`results/pileup_hp/addendum_h_inputs/c1_hval_quick.json`). At most two bins per quantity then count in the pixel
barrel, at most one in volume 7 and none in volume 9. The pixel-barrel bins that failed the first C1 evaluation, at
|tan_u| = 0.8-2.4 and |tan_v| = 8-16, are not among them. In the strips, 1-9 bins per quantity and volume should count.
Part 1 therefore tests little in the pixels, and part 2 carries the confirmation there. A volume and quantity with fewer
than 200 gun hits on its common support is "not informative" and is listed. For the pixel volumes the same comparison
runs on the H-test gun hits at H-T1 and is reported beside H1. Neither changes anything.

**Closure truth gate (before the closure model run).** The code is `python -m hnl.overlay.closure_gate run`. The
cells are those of `hnl.overlay.closure_checks.hits_per_layer`: pixel, short-strip and long-strip detector, barrel and
endcap, times the |eta| bands 0-1, 1-2, 2-3 and 3-4 of the native, with the natives of populations (a)-(c) pooled. A
crossed layer is a layer with at least one hit of the track. In every cell below |eta| = 3 with at least 2,000 crossed
layers on each side, the mean number of hits per crossed layer of the re-simulated tracks lies within 0.05 of the
natives'. With no counted cell the gate is "not informative": it neither passes nor fails and blocks nothing; one or
two counted cells are reported as "few cells". The pilot run of the gate counted no cell (job 6821095). The cells at
|eta| = 3-4 are reported without the rule. Reported per cell with it: hits and layers per track,
the share of natives still a target, and the Fatras hits before the emulation, which separate Fatras from the
emulation. In the dry run, the cells below |eta| = 3 with at least 30 crossed layers differ by at most 0.05, and
scaled to 150 hosts nine of them would count. In the pixel endcaps at |eta| = 3-4 the natives have 1.99 hits per
crossed disk and their Fatras hits 1.63. That gap is present before the emulation, and its cause is unknown.

**V1, closure.** The code is `hnl.overlay.v1` (`python -m hnl.overlay.v1 run`, run by `slurm/overlay/v1.sbatch`; V1-Δ in
`v1_delta`). The frozen model runs on the closure set in C-full at WP-A. V1 reads only the closure grids that
`slurm/overlay/grid.sbatch` writes, under a grant of kind eval for each condition, and each only after
`hnl.overlay.tracks.grid_provenance` (every closure event, the frozen classifier's scores in that condition in every
block, the learned merge model of `selection_v1.json`, built on the production overlay tables, no placeholder; the check
the H1 code makes on H-val and H-test). Each native target of the populations below is compared with itself in the
stored val outputs (C-full, WP-A) under `runs/canto_l_v1/seed0_x1/`: `grid/full/per_target.parquet` (sha256 622e0d7f...,
job 6763371) and the WP-A diagnostic tables `diag/displaced/diag_full_b*.parquet` (jobs 6773817-6773855). The sha256 of
the 30 diagnostic tables are listed in `results/pileup_hp/addendum_h_inputs/val_diag_sha256.txt` (sha256 036ef4d6...). A
re-simulated target is paired with its native original by the closure id rule above. A pair enters V1 when both versions
are targets. Beside V1 we report the share of native targets whose re-simulated version has fewer than 5 hits, and an
unconditional V1 that counts such a re-simulated version as not found. "Forward" means a native hit at |z| > 1 m, split
by the sign of eta. Counts are from `pileup_design/plan_revision/val_regions.json` (job 6812736).

| V1 population | Central | Forward +z | Forward -z | All | Pass rule on the paired Δ | Kind |
|---|---|---|---|---|---|---|
| (a) gen-0, \|d0\| >= 10 mm or r_v >= 100 mm (the P_P cut) | 1,250 | 1,050 | 976 | 3,276 | \|Δ\| <= 1.0 point in all; <= 1.5 points in each region | decisive |
| (b) gen >= 1, \|d0\| >= 10 mm, without the natives produced at a sensor | 44 | 213 | 191 | 448 before the cut | \|Δ\| <= max(3 points, 2 σ_paired) in all | flag |
| (c) prompt subset: 3,000 gen-0 targets with \|d0\| < 1 mm and r_v < 1 mm; the rule uses the members with \|eta\| < 3 (about 2,410) | - | - | - | 3,000 | \|Δ\| <= max(0.2 points, 2 σ_paired) | decisive |
| (d) all generations, \|d0\| >= 10 mm and \|d0\|/r_first >= 0.2 (not pointing back; each is a member of (a) or (b)) | - | - | - | 363 | \|Δ\| <= max(3 points, 2 σ_paired) in all | flag |

The prompt subset is the first 3,000 of the 153,603 prompt val targets in increasing
int(sha256("canto-trackml-overlay-v1-prompt:<event>:<particle_id>")[:8], 16), with ties broken by event and then by
particle id. Its members at |eta| = 3-4 (19.6 %) are reported without a rule, because of the disk gap above. Population
(d) needs no extra closure event: all its natives are already replaced. σ_paired comes from an event bootstrap over the
150 events (B = 500, `default_rng(12345)`, as for H1).

What σ_paired to expect: the discordance between a native and its re-simulation is not known. For a discordance of
1-5 %, σ_paired for (a) is about 0.17-0.39 points in all and 0.3-0.7 points per region, against rules of 1.0 and 1.5
points. For (b) and (d), with 10-30 % discordance over about 400 pairs, it is about 1.5-2.7 points, so their rule is
3-5.4 points.

A native of (b) is "produced at a sensor" when its first hit lies within 1 mm of its production vertex, measured along
the normal of the hit's module. Such a native starts with a hit that Fatras, starting at the vertex, does not record.
Gun decay vertices are spread through the volume and do not share this. On truth, 15 % of the (b) natives have a first
hit within 1 mm in r of the vertex, against 1.6 % of (a) and none of (c). In the dry run, 72.7 % of the (b) re-simulated
tracks started on the native's first layer, against 98.2 % for (a). The natives produced at a sensor are reported apart,
and the first-hit layer agreement is reported for every population.

V1 is also reported in |eta| bins and by detector region: pixel, short-strip and long-strip barrel and endcaps, with the
long-strip endcaps listed on their own. A pair counts in every region that holds one of its native hits. For (a), (b)
and (d) it is reported in bins of r_first (pixel, short strip, long strip) and |d0|/r_first, with counts. A gun-weighted
V1 is reported as well: (a) and (b) reweighted to the H-val gun's P_P distribution in the cells of section 6. None of
these has a pass rule.

**V1-Δ, closure in C-1GeV.** The frozen model also runs on the closure set in C-1GeV at WP-A. For the pairs of V1, let
Δ_closure be the C-full minus C-1GeV change of the re-simulated targets and Δ_native the same change of their natives in
the stored val outputs (C-1GeV: `grid/1gev/per_target.parquet`, sha256 df64d65b..., job 6763372, and
`diag/displaced/diag_1gev_b*.parquet`). The rule on (a) is |Δ_closure − Δ_native| <= max(0.5 points, 2 σ_paired),
with σ_paired from the same event bootstrap; it is a decisive gate. The same difference is reported for (b), (c) and
(d). H1 is a difference within one sample, so a bias that does not depend on the condition cancels in it. A mismodelled
hit that makes a track more or less exposed to the soft hits does not cancel, and V1-Δ tests for it.

**V2, host change with the gun present (flag).** The frozen model runs on the H-val overlay in C-full at WP-A, and the
host numbers are compared with the stored val outputs (`grid/full/per_event.parquet`, sha256 dd86b756..., job 6763371).
The comparison is `python -m hnl.overlay.blind v2`, run by `slurm/overlay/blind.sbatch` after the sealed directory of
section 10 is sealed. It writes `results/pileup_hp/v2_hval_v1.json` with every frozen C-full configuration, and the gate
is decided at WP-A. The host prompt-target efficiency and the fake rate should agree within 0.05 points, and the TrackML
score within 0.001. While the blinding of section 10 holds, the fake rate counts only tracks with more than half host
hits. Gun hits carry weight 0, so they add no weight to the score. They still count in the hit totals that decide
whether a track is good (both purity conditions of `trackml.score.score_event`), so a gun hit on a host track can lower
the score. The integrity of the host rows is the truth gate "Host rows" above. A change of the host numbers beyond these
tolerances can be a real effect of the 18,300 gun hits. It is reported and flagged, and it never leads to an emulation
or digitiser change. The expected change is small: in the dry run 1.0 % of gun hits lie within 1 mm of a host hit, about
180 on H-val, while 0.05 points of the 153,603 prompt val targets are about 77 targets.

**V3, features (descriptive).** Cluster features and residuals of gun hits against native tracks, in incidence bins and
signed eta, and same-module near-coincidences. tp_v1 stores truth kinematics for the 200 calibration events only, so
the native residuals and incidence come from those events. The hosts' own targets enter with cluster features only.

### 9.3 When a gate fails

**Decisive gates:** the per-set truth checks, the C1 confirmation, V1 (a), V1 (c) and V1-Δ (a). **Flag gates:** the
closure truth gate, V1 (b), V1 (d) and V2. A C1 confirmation or closure truth gate that counts nothing is "not
informative" (section 9.2): it is reported, neither passes nor fails, and neither blocks nor flags.

- **A per-set truth check.** The set does not go to the model. The cause is found, fixed and logged in RESULTS.md. At
  H-T0.5 a fix outside overlay_hash, for example in the writer or the gun driver, is not a pass, and every gate runs
  again on the remade set. A fix inside overlay_hash is a change under the rules below. Section 9.4 covers H-T1.
- **Changes.** One emulation or digitiser change may be made per pass: after a failed decisive gate, or after a failed
  closure truth gate before the closure model run. It obeys these rules:
  1. It names the failing quantity and the truth-level discrepancy it answers, measured against TrackML on training
     events (for the C1 confirmation: on the C1 sample or on H-val gun truth). It is committed with its derivation
     before the rerun.
  2. Its size is not tuned on V1, V1-Δ or V2 values. They may show that a change is needed, never how large it is. It
     is never decided on a model result for a gun target.
  3. A failure traced to Fatras, the v15 material or the geometry, which is present before the emulation, is not fixed
     through the emulation or the digitiser. It leads to the fallback below where that applies, and otherwise to the
     outcome of the last two bullets.
  4. Each change gives a new overlay_hash or manifest, and the new version goes through every val gate. Each version
     made is one pass, whether or not the model ran on it, so a C1 confirmation that fails before any model run uses
     one. There are at most three passes.
  5. The emulation draws one uniform per crossing and one per hit, so a change of table values keeps those draws. It
     can still shift the digitiser's per-cell draws, since it changes which hits survive, and a change of code may
     re-roll more. Such a re-roll is allowed only as a side effect of a change under rule 1. The Fatras streams are the
     same in every pass, so a failure caused by Fatras fluctuations persists.
  6. The first version that passes every decisive gate is committed with no further change.
- **Forward fallback (decision D12).** It applies to the first version that passes every other decisive gate and in
  which V1 (a) passes in all but fails in one or two regions. P_P is then restricted to the regions that pass, by truth
  flags of the gun targets. A gun target is central when none of its hits lies at |z| > 1 m, and forward +z or -z
  otherwise, by the sign of its eta. The restriction is committed with overlay_hash, before H-T1, and is never combined
  with a further change. Central targets alone are 48 % of P_P in the plan.
- **Descriptive.** If a decisive gate still fails after the third pass, and the fallback does not apply, H becomes
  descriptive. H1 is then not decided, and every number of sections 6 and 7 is reported as descriptive. This is
  decided and logged before H-T1.
- **Flag gates.** A flag gate that fails in the committed version is reported, and H1 is decided and carries the flag.
  V1 (b), V1 (d) and V2 never lead to a change by themselves.
- **G0, F1 and containment** passed at H-P. If a new build were needed, decision D1 would be taken again before any
  overlay of a production set.

### 9.4 Checks at H-T1 (truth only)

**Per event.** A host is dropped with its reason, and replaced by the next spare, only for a property of the host event
itself:

- no hard-scatter vertex by the rule of `hnl.overlay.gun`;
- host weights that do not sum to one within 1e-6 in the host's own stored rows;
- host particles that break a precondition of the merge, such as a vertex field above 1000 or a repeated particle id.

Where possible these are checked before the gun of that host is simulated. The spare keeps its own Fatras event number
(section 2), and the drop is recorded in `htest_dropped` of `split_h_v1.json` (section 3).

**Per set.** Every other check stops H-T1 when it fails:

- the per-set truth checks of section 9.2 (converter, weight sums after the merge, containment, host rows), and any
  Fatras failure other than a logged navigator loss;
- **Version:** overlay_hash and every value of the manifest equal the committed ones, the base seed equals that of
  `split_h_v1.json`, and every Fatras event number is the host's rank (`hnl.overlay.writer.version_check`);
- **No earlier read:** the part03 ids are disjoint from every id in `split_tp_v1.json` and `split_v1.json`, and no
  part03 event has the same hits (a hash of x, y, z and module over all hits) as a tp_v1 event;
- the H-test overlay reads its hosts from `derived/tp03_v1`, named explicitly (`HOST_ROOT` of
  `slurm/overlay/overlay_set.sbatch`, `--host-root` of the gun driver and the writer). `hnl.overlay.hosts` refuses any
  other root for H-test, the writer refuses a root other than the gun's, and the meta records it.

**After a per-set failure.** The cause is found on truth and logged in RESULTS.md. A fix that touches neither
overlay_hash nor a file of the manifest (a driver or a check) rebuilds the H-test set and is not a pass. A fix that
touches either one returns to H-T0.5 once, if a pass remains. H-val and closure are then made again with the fixed
version, and every val gate runs again. The new overlay_hash and manifest are committed and marked "made after the H-val
estimate". The H-val estimate is computed again, and both are reported. If no pass remains, H becomes descriptive. The
same rule is written into `split_h_v1.json` next to the H-test rule (section 10).

## 10. Order of work

The steps carry the prefix H- so that they are not confused with the cross-phase test set T2.

- **H-P (done; truth only).** Patched build, G0, F1, containment, M1, C1, emulation tables and gate, overlay writer,
  and a dry run on the five pilot hosts (`results/pileup_hp/hp_summary.md`). After it, and before this commit, the V1
  closure machinery (7c94251) and the navigator-loss check of the no-pileup samples (ad94518).
- **H-T0 (this commit).** This addendum. `configs/trackml/split_h_v1.json`, the seed check and the plan's H-T0 update
  were committed before it (607955f, c078cb2). Nothing is streamed from part03, and no production overlay is made.
- **Before any production overlay or model run on an overlay id.** Committed and tested (all five items are done;
  items 1-4 on truth and truth-made placeholder outputs only, item 5 also in the plumbing of decision D6):
  1. The drivers pass the Fatras event numbers of section 2 explicitly and read the base seeds from `split_h_v1.json`.
     `writer.SETS` carries the addendum's seeds and no `htrain` entry. Each set's meta records every value of the
     manifest, and the Version check compares the base seed and the Fatras event numbers with `split_h_v1.json`
     (b39e222).
  2. `split_h_v1.json` gains, from its own code (`hnl.overlay.split_h make`): the Fatras numbering, the per-event and
     per-set rules of section 9.4 with the return rule, the id format of the hash key, the record format of the
     H-test lists and drops (section 3), and the closure dry run in the pilot note (b39e222, 481771e).
  3. The blinding mechanism below, with the grants, launch pins and write-once outputs, tested on the pilot set with
     truth-made placeholder outputs and no model run (5e62320, 481771e, 1dee6b8, e6f09b4, 837d309; in the fourth
     code round the binding of the grants to the code an overlay job runs, the working-tree and Slurm-job checks, the
     pending record, the fixed decisive output and its committed record, 61c12af; in its second pass the check for
     shadow modules, the unreadable pending record, the H-val record pinned at the launch and the refusal of empty or
     zero H1 arguments, 021c1b8).
  4. The C-hs filter, the content additions of section 9.2, the pooled C1 part, the closure truth gate, V1 and V1-Δ,
     the H1 code, and the explicit `derived/tp03_v1` host root of the gun driver and the writer, tested on a dry root
     (d4a5c4f, 481771e; the gate rules of decision D5 and the dry host root for the pilot only in cbfac51 and 1dee6b8).
  5. The model stages of the frozen chain on overlay ids (graph, event store, edge scores) and the jobs that build
     tracks and evaluate them: `hnl.overlay.chain` (`src/hnl/overlay/chain.py`) with `slurm/overlay/chain.sbatch`
     (graph additions and event store, GPU, one block of up to 50 events per job) and
     `slurm/overlay/chain_score.sbatch` (edge scores on the express CPU nodes, decision E2, each block in parts of at
     most 5 events in C-full and 50 in C-1GeV and C-hs), `slurm/overlay/grid.sbatch` for `hnl.overlay.tracks`, and
     `slurm/overlay/v1.sbatch` for V1 and V1-Δ. Commits: 1dee6b8 (v1.sbatch), 02fa773, c9bac9d, ffc35c4 and 6b1aa22
     (third round), e6f09b4 and 837d309 (its integration), 61c12af (fourth round) and 021c1b8 (its second pass: the
     score jobs sized to the 1 h limit). Each job runs the freeze check, then `slurm/overlay/guard_h.sh` with
     a grant, writes only into the directories of `python -m hnl.overlay.chain paths` (the H-val C-full scores into
     the sealed directory) and applies C-hs through `hnl.overlay.conditions`. No model code changes. They were tested
     on stand-in networks, in the wiring run with placeholder scores, and in the plumbing of decision D6.

  The integration dry run made the pilot overlay (k = 9, base 990,101) and a pilot closure set (k = 1 ids, base
  990,101) again with this code in a scratch root. The pilot overlay is identical, shard by shard, to the H-P one. It
  ran every truth gate and content check on both and passed truth-made placeholder outputs through
  `hnl.overlay.blind`, `slurm/overlay/guard_h.sh`, `hnl.overlay.tracks`, `hnl.overlay.h1` and `hnl.overlay.v1` (jobs
  6820566-6820567 and 6820886-6820920; `pileup_design/h_integrate/`). A test runs the guard through every stage of
  the protocol in a copy of the repository and checks that each forbidden case is refused, with its reason
  (`tests/test_overlay_blind.py`).

  The second and third code rounds added the grants, the launch pins and the write-once outputs, the gate rules of
  decision D5 and the model-stage chain. The fourth round bound the grants to the code an overlay job runs (decision
  E1 as narrowed), fixed the score placement (decision E2) and the decisive H1 output, and closed the review's
  remaining holes (the labels rebuild, grants used outside their Slurm job, a rerun to another output path after a
  crash). Its second pass cut each block's scores into CPU jobs that end within the 1 h limit and closed the review's
  minor holes (a module shadowed by an untracked file during a job, an unreadable pending record, empty or zero H1
  arguments, the H-val record of decision D7 not pinned by the launch). The gates were run again on the pilot set (job
  6821095; `pileup_design/h_fix/gates/out/`). A wiring run passed truth-made placeholder scores of the pilot set through
  the model-stage jobs (kind dry), the grid jobs, the blinded evaluation with its seal and V2, and the H1 job, each
  under the guard's grant (jobs 6822147-6822149, 6822311-6822313, 6824569 and 6824570; `pileup_design/h_fix/chain_b/`).
  The plumbing run of decision D6 ran the frozen model stages on the five pilot overlay events in C-full, C-1GeV and
  C-hs on A30 GPUs (jobs 6822169, 6822171 and 6822173). Each output directory was sealed right after scoring and checked
  for files, schemas, row counts, provenance and the hits each condition keeps; the record job (6824689) verified every
  seal again and found no writable path (`pileup_design/h_fix/plumb/plumb_check_v1.json`).
  A second plumbing run computed the edge scores again in 16-CPU jobs on submit77-1, from a copy of the first run's
  graphs and event store (jobs 6822170, 6822172 and 6822174), sealed and checked each output the same way, and record
  job 6824690 verified every seal again (`pileup_design/h_fix/plumb/plumb_check_v2.json`).
  No track was built and no number was computed from these outputs. The test suite at the final code ran in one job
  (6825646 on submit77-1, at 1a716f4; the commits of other sessions during the job, 5ba7a3a, 28d7e87 and cf7aa7e,
  changed only results, RESULTS.md and the paper, and the bound code stayed at digest ec1e32a5bebe): the 122 tests of the
  enforcement and model-stage files, the 162 of the other overlay tests and the navigator-loss tests, and the 112 of
  `src/hnl/trackml/tests` all passed. In the same job every module of `hnl.overlay` and `hnl.trackml` was imported, and
  each of the 91 `hnl` modules then loaded is a file of the bound set; the frozen builders on native val event 21289
  reproduced the stored val grid rows exactly at WP-A, WP-S, WP-R-P2 and WP-R-P3, and `python -m hnl.overlay.chain
  cost` wrote the cost record of section 10 (`pileup_design/h_fix/round4/pass2/suite_summary_6825646.json`). The
  refusal table `pileup_design/h_fix/guard_cases_round4b.json` lists 337 cases from that job: 276 forbidden cases,
  each refused with its expected reason (69 of them run through the guard at the protocol stages in a copy of the
  repository), 46 allowed controls that pass, 14 checks of the job scripts and of the score-job size, and one record of
  the bound set; none went the wrong way. The independent verify ran the three suites again at the final code (job
  6826438 at cf7aa7e, no commit during the run, bound code at digest ec1e32a5bebe): all passed, and its 161 probes
  came out as expected (`pileup_design/h_fix/round4/verify2/probes_6826438.json`).
- **H-T0.5 (val).**
  1. Produce the H-val and closure overlays and run their per-set truth checks, the C1 confirmation, V3, content and
     the closure truth gate. From H-val truth, state the P_P yield and the reweighted projection of section 6.
  2. Run the frozen model on the closure set in C-full and C-1GeV (V1, V1-Δ) and on the H-val overlay in C-full (V2),
     under the blinding below.
  3. Apply the rules of section 9.3, with at most three passes.
  4. Commit `configs/trackml/overlay_version_h_v1.json` with overlay_hash, the manifest, any D12 restriction or the
     decision to go descriptive, and the density bins of section 7, together with the H-val record of decision D7
     (section 7), `results/pileup_hp/h_t05_*`. Any change to the H1 code is committed with it, tested on placeholder
     outputs of the pilot set.
  5. Launch H-val once (`bash slurm/overlay/guard_h.sh --launch hval`, which needs the committed version record, the
     committed H-val record of decision D7, which the launch pins, and a sealed directory that verifies). Run H-val in
     C-1GeV and C-hs. Compute the gun results on H-val once per working point, as the val estimate of H1
     (`slurm/overlay/h1.sbatch` with SET=hval, the launch token and RESTRICTION_RECORD, the committed record of decision
     D7 (section 7); the job reads the manifest sha256 from RESULTS.md and the C-1GeV and C-hs grids of the grid jobs,
     and writes `results/pileup_hp/h1_hval_<WP>_full-1gev-hs_v1.json`). They change nothing.
- **Blinding on H-val.** Until the commit of step 4, the model's results on the gun targets of H-val stay blinded. The
  blinded evaluation, the sealed directory and the guard with its grants enforce it:
  - `hnl.overlay.blind` is the only evaluation of k = 0 ids before that commit (`slurm/overlay/blind.sbatch`). Before
    `hnl.trackml.evaluate.event_counts` it removes the gun particles (vertex field 1001-1010) from the particles table,
    gives the gun hits particle id 0 and removes every track with half or more gun hits. A gun hit therefore still
    counts in the size of a host track, the matching of host tracks is the unblinded one, and the TrackML score equals
    the unblinded score. It computes the V2 numbers and writes no row and no count for a gun particle or for a track
    with half or more gun hits.
  - The frozen chain's raw outputs on H-val in C-full are written to the sealed directory
    `runs/overlay_h_v1/sealed/hval_full` of the TrackML data root: edge scores with truth labels (`scores/`), the raw
    track labels of every hit (`tracks/`), and the blinded per_event and per_target tables. `python -m hnl.overlay.blind
    seal` writes `MANIFEST.sha256` and `SEALED.json`, makes the directory read-only and prints the sha256 of the
    manifest. It is recorded in RESULTS.md as one line, `Addendum H sealed H-val manifest sha256: <64 hex>` (decision
    D4). Only `hnl.overlay.blind` and, after the commit of step 4, the step-5 job read the directory. The step-5 job
    enters through `hnl.overlay.blind.open_sealed`, which requires the committed record and a manifest that verifies
    against the sha256 of that line in RESULTS.md at HEAD (a value the caller gives must agree with it), and recomputes
    the unblinded per-target rows from the raw track labels (`sealed_per_target`), after checking that the same labels
    give the sealed blinded counts.
  - `slurm/overlay/guard_h.sh`, modelled on `slurm/addendum_g/guard.sh`, is called by every job that runs a model
    stage, builds tracks or evaluates outputs on overlay ids (`slurm/overlay/chain.sbatch`, `chain_score.sbatch`,
    `grid.sbatch`, `blind.sbatch`, `h1.sbatch` and `v1.sbatch`), after the freeze check of section 5. It refuses the
    job if a tracked file under `src`, `slurm` or `configs` has an uncommitted change or an untracked `.py` file lies
    under `src` (decision D2). It also refuses unless the blinding code, the guard, the evaluation code of this
    addendum (`hnl.overlay.split_h`, `conditions`, `tracks`, `h1`, `hstats`, `v1` and `slurm/overlay/h1.sbatch`), the
    frozen chain's evaluation code, `selection_v1.json` and `split_h_v1.json` are committed unchanged and, for every
    kind but `dry` and `plumb`, this addendum too. Every id must be an overlay id of the named set. It allows k = 1 in
    C-full and C-1GeV only. Until the commit of step 4 exists, it allows on k = 0 only the C-full model run, with
    outputs to the sealed directory and refused once the directory is sealed, and `hnl.overlay.blind` on that
    directory. After that commit, and only if the H-val meta carries the committed overlay_hash, it allows k = 0 in
    C-1GeV and C-hs and the step-5 evaluation, once: `bash slurm/overlay/guard_h.sh --launch hval` writes a launch
    record with a token exactly once (`runs/overlay_h_v1/launch/hval.json`), and these jobs run only with that token.
    The C-full model run on H-val is not repeated. It allows k = 2 only under the H-T2 rule below, never a spare. On
    k = 9 it allows two kinds, each with outputs outside the production runs root and the overlay tables: `dry`, code
    tests on truth-made placeholder outputs, and `plumb`, the plumbing run of decision D6.
  - **Grants (decisions D1 and E1).** With `--grant` the guard writes a read-only grant, which the job exports
    (`HNL_H_GRANT`). It names the set, the condition, the kind, the ids with their sha256, the Slurm job, the output
    directory and, after a launch, the launch token. The kinds are `model` (a network or learned stage of the frozen
    chain), `plumb` (the same on the pilot, decision D6), `blind` (`hnl.overlay.blind`), `eval` (any other evaluation
    of model outputs) and `dry` (placeholder tests).
    - *Binding (decision E1, narrowed in the fourth code round, DECISIONS 2026-10-15).* A grant binds the git blob
      hashes at HEAD of exactly the code an overlay job runs, and the blob hash of this addendum at HEAD (none while it
      is not committed). The bound files are computed from the code at HEAD (`hnl.overlay.grant.bound_paths`): every
      file of `src/hnl/overlay`, `src/hnl/trackml` and `slurm/overlay`; every other `src/hnl` module in the static
      import closure of the modules of `hnl.overlay` and `hnl.trackml` (their tests excepted), imports inside functions
      included; and the files under `configs` that these modules name in their code, a named directory standing for
      every file below it. At this code that is 157 paths: 25 in `src/hnl/overlay`, 47 in `src/hnl/trackml`, 13 in
      `slurm/overlay`, 52 files of the other `src/hnl` packages with their `__init__.py` (six modules of `src/hnl/eval`,
      among them `matching.py` and `trackbuild_val.py`), the 17 files of `configs/trackml` with the path of the version
      record, bound as absent until it is committed, and the two templates of `configs/baseline`. A commit that changes
      no bound file and leaves this addendum as it was (paper, docs, results, DECISIONS, the rest of `src/hnl/eval`,
      `slurm/addendum_g*`) leaves a running job's grant valid. A commit that changes, adds or removes a bound file, or
      adds a module or a configs file to the bound set, or a commit of this addendum, stops the job at its next check,
      and the job is submitted again. So does an uncommitted change to a bound file, an untracked python file under
      `src/hnl/overlay` or `src/hnl/trackml`, or an untracked file under `src` that python would import in place of a
      bound module (a package directory or a compiled module of its name, ignored files included), since a later
      process of the job would run it. The guard itself still
      refuses to start a job while a tracked file under `src`, `slurm` or `configs` has an uncommitted change or an
      untracked `.py` file lies under `src` (decision D2). A test
      computes the bound set from the source independently and compares it with the grant's, and another imports every
      module of `hnl.overlay` and `hnl.trackml` and checks that every `hnl` module then loaded is a bound file.
    - *Check.* Every entry point that runs a network or a learned component, or evaluates model outputs, on overlay
      ids refuses them unless a grant covers the request (`hnl.overlay.grant.check`): its kind, its condition, every
      id, the code state, the Slurm job and, when the grant names an output directory, every path the entry point
      writes. A grant made inside a Slurm job holds in that job only; it is refused in another job and outside Slurm,
      and a grant made outside Slurm is refused inside a job. The check sits in the frozen chain's command-line tools
      (`hnl.trackml.graphs`, `infer`, `trackbuild`, `displaced_diag`, `evaluate` and `labels`), in its development
      tools (`g0`, `probe`, `support_gpu`, `chunked` and `g3check`), at two choke points that every caller passes
      (`hnl.trackml.events.Embedder.__call__`, any embedding network on an event, kinds `model` and `plumb`;
      `hnl.trackml.trackbuild.read_scores`, any read of stored edge scores, kinds `eval` and `dry`, and `blind` only
      where `hnl.overlay.blind` asks for it), and in `hnl.overlay.chain`, `tracks`, `blind`, `h1` and `v1`. Stored
      scores of pilot events are read only if they are truth-made placeholders, so no track is built from the
      plumbing outputs.
    - *Launch.* For H-test ids, and for H-val ids under a grant of kind `eval` or of kind `model` in C-1GeV or C-hs,
      the grant must carry the token of the set's launch record, and every file pinned at the launch must be
      unchanged; `hnl.overlay.tracks` grid and `hnl.overlay.h1` run check the launch again.
    - *Out.* When the grant names an output directory, every path the entry point writes must lie under it (resolved
      paths, so a symbolic link does not lead out of it).

    On success the check writes one line naming the grant and the bound code to the job log, so every job log records
    its grant. The guard and the grants stop accidental runs. A deliberate bypass through direct Python calls, or a
    `SLURM_JOB_ID` set by hand, stays possible.
  - **Launch records (decisions D3 and D4).** `bash slurm/overlay/guard_h.sh --launch hval|htest` writes
    `runs/overlay_h_v1/launch/<set>.json` exactly once. It holds the token and the sha256 of every file the decisive
    runs depend on: the guard's code list, every tracked file of `slurm/overlay`, this addendum, the version record,
    `selection_v1.json`, `split_h_v1.json`, `pipeline_v1.yaml` and `res_v1.json`, and every `hnl` module that the entry
    points import, directly or not (the import closure). For H-test it also pins every file R15 names and the text of
    R15; for H-val it records the manifest line of RESULTS.md and pins every committed H-val record of decision D7
    (`results/pileup_hp/h_t05_*`). `--launch hval` needs the committed version record, at least one committed H-val
    record of decision D7, and a sealed directory that verifies against the one line `Addendum H sealed H-val manifest
    sha256: <64 hex>` of RESULTS.md at HEAD; `--launch htest` needs R15 (H-T2). A job under the token is refused once a
    pinned file, the R15 text or that RESULTS.md line has changed.

  Nobody inspects tracks of gun particles before that commit. Truth-level features of gun hits (C1 confirmation, V3,
  content, the yield and the reweighted projection) are not blinded. The closure set holds no gun particle, so V1 and
  V1-Δ read it in full.
- **What freezes at the commit of step 4.** The H1 estimand, P_P with any D12 restriction, the conditions, WP-A, the
  statistics, the outcome rules, δ and the H1 code. After the H-val estimate, a deviation is allowed only for a
  technical failure that leaves these unchanged, and each such deviation is marked "made after the H-val estimate".
- **H-T1 (truth only).**
  1. Stream training_part03 twice over HTTP from the tar, which is never unpacked (md5
     b0c407c72e2753ed4e1e619c6c030e62), and convert it to `derived/tp03_v1`.
  2. Apply the hash rule, fix the Fatras event numbers, and add the H-test list, the spares and the drop records to
     `split_h_v1.json` (section 3).
  3. Build the H-test overlay with the committed overlay_hash, manifest and base seed 620,000,001, and run the checks of
     section 9.4.
  4. Run the collision-reuse check.
  5. Commit checksums, dropped events, twin fractions, navigator losses, the P_P and P_D counts, the long-strip
     sensitivity of section 7 and the H-test record of decision D7 (P_P restriction, failed flag gates, navigator-loss
     share, descriptive state; section 7). These are the H-T1 records: the tracked files `results/pileup_hp/h_t1_*`, at
     least one, which R15 names (`hnl.overlay.blind.HT1_RECORDS`).
- **H-T2.** First, a ledger entry R15 in `docs/feedback/test_use_ledger.md` is committed. It names H-test, the first
  8 hex digits (at least) of the sha256 of its id list (the evaluated hosts, `hnl.overlay.split_h.htest_sha256`) and of
  overlay_hash, `split_h_v1.json`, the version record with the manifest, this addendum, the H-T1 records (every
  tracked file `results/pileup_hp/h_t1_*`) and the H1 code (`hnl.overlay.h1` and `hnl.overlay.hstats`). Each prefix is
  compared with the current file (`hnl.overlay.blind.ledger_r15`). `guard_h.sh` then allows a model or evaluation job
  on k = 2 ids only if R15 and every repo file it names are committed unchanged, only for the overlay ids of the
  evaluated H-test hosts, and only with the token of `bash slurm/overlay/guard_h.sh --launch htest`, which is issued
  once and pins the files listed under "Launch records" above. One run of the frozen model on H-test in C-full,
  C-1GeV and C-hs decides H1 (`slurm/overlay/h1.sbatch` with SET=htest). WP-S and WP-R-P3 come from the same edge
  scores. On H-test, as on H-val in step 5, every H1 run reads all three conditions and the committed record of
  decision D7 (section 7), and the output of each working point (WP-A, WP-S, WP-R-P3) is written once, to one fixed
  path, `results/pileup_hp/h1_<set>_<WP>_full-1gev-hs_v1.json`; any other output path is refused.
  `hnl.overlay.blind.write_once` writes the complete output to a temporary file, then records a pending record
  `runs/overlay_h_v1/launch/<set>.pending.<WP>.json`, then places the output, and then records the consumed marker
  `runs/overlay_h_v1/launch/<set>.consumed.<WP>.json`, each exclusively. A later run at that working point is refused
  before it reads any input. A run that stops before its pending record may be repeated. From the pending record on,
  the working point is consumed: the next attempt records the marker and writes nothing, whether or not the output was
  placed and whatever output path it names. An empty or unreadable pending record, left by a run that stopped while
  writing it, counts as consumed in the same way.
- **No retraining arm.** H3 is not part of this addendum (decision D4). If a retraining study is wanted after H-T2, it
  is preregistered separately on hash-ordered training_part04 hosts, with its own budget approval (about 57 GPU-h per
  seed).
- **Budget.** One H-T0.5 pass and H-T2 together need about 750-1,090 CPU-h and 10.4 GPU-h (plan section 11). The closure
  run in C-1GeV adds a small part of that, since C-1GeV events hold far fewer hits. Each extra pass adds about 235-350
  CPU-h and 3.4 GPU-h. A stage that would exceed 1.5 times its estimate, or 75 GPU-h, goes to the user first.

  The plumbing run measured the model stages on the pilot overlay events
  (`pileup_design/h_fix/plumb/plumb_check_v1.json`). A C-full event holds on average 117,165 hits and 1.96 M edges.
  Per event, the graph additions took 44.1 s on an A30, the event store 4.0 s and the whole-event scores 318 s on 16
  CPUs inside the GPU job, at a peak memory of 11.5 GB; in C-1GeV 2.1 s, 0.19 s and 17.6 s; in C-hs 1.2 s, 0.08 s and
  13.4 s. For H-val (150 events in three conditions), the closure set (150 in two) and H-test (600 in three) the record
  gives two placements of the scores:
  - scores in CPU jobs on the express CPU nodes left after the exclusions (submit77-1, and submit37 once it returns),
    with the GPU jobs running the additions and the event store only: 13.4 GPU-h, 1.29 times the 10.4 GPU-h above;
  - every stage inside the GPU jobs: 101.4 GPU-h, of which 62.0 for H-test in C-full, more than 1.5 times the GPU
    estimate and more than 75 GPU-h.

  **Decision E2 (DECISIONS 2026-10-15).** The scores run on the CPU nodes: the GPU jobs (`slurm/overlay/chain.sbatch`)
  run the additions and the event store, and CPU jobs (`slurm/overlay/chain_score.sbatch`) the edge scores. The
  placement inside the GPU jobs is not used, and `chain.sbatch` and `hnl.overlay.chain run` refuse it on the production
  sets. `chain_score.sbatch` asks for 6 CPUs, the configuration whose time the estimate uses: the CPU time per event
  comes from the forward check of section 5, on the native val events of the pilot hosts with 6 threads on submit77-1
  (472 s in C-full, 23.2 s in C-1GeV; the gun adds about 0.1 % of the hits). C-hs was not timed on a CPU node at 6
  threads, and the GPU job's 13.4 s per event stands in. The stand-in is probably low: scaled by the ratio of the
  16-CPU to the 6-thread time in C-1GeV, C-hs would take about 17-19 s per event, which adds about 1 wall-h and 6
  core-h, and a 50-event C-hs score job would take about 16 min.

  Both kinds of job work on the same blocks. Each GPU job makes the additions and the event store of one block of up
  to 50 events, about 41 min in C-full. Each CPU job scores a part of one block: at most 5 events in C-full, and a
  whole block of 50 in C-1GeV and C-hs (`hnl.overlay.chain.score_parts`). Five C-full events take about 40 min at the
  mean time, and 53 min if each took as long as the slowest timed event (632 s), so every score job ends within its
  1 h limit. `chain_score.sbatch` maps its array task to a block and a part, and `hnl.overlay.chain run` refuses a
  score job that names no part when its block has several, or that runs fewer than 6 threads; `python -m hnl.overlay.chain layout` prints both arrays (for
  H-val in C-full, 3 GPU jobs and 30 score jobs). In this layout the record's times give 13.4 GPU-h, 1.29 times the
  10.4 GPU-h estimate, which stays under the 1.5-times rule, and 213 score jobs with 128 wall-h and about 770 core-h of
  scoring (`pileup_design/h_fix/plumb/cost_e2_v1.json`, written by `python -m hnl.overlay.chain cost` in job
  6825646, at 1a716f4).
  The second plumbing run timed the scores of the pilot overlay events on 16 CPUs of submit77-1
  (`pileup_design/h_fix/plumb/plumb_check_v2.json`): 352 s per event in C-full, 16.7 s in C-1GeV and 12.0 s in C-hs,
  at peak memories of 9.8, 2.5 and 1.4 GB. These times do not enter the estimate, since the score jobs run 6 threads.
  On the same hosts, 16 CPUs took 0.74 times the wall time of 6 threads per C-full event (352 s against 472 s) and
  0.72 times per C-1GeV event, at about twice the core time.

  Track building adds about 96 CPU-h, from the per-event times of the native test grid. The plumbing jobs used about
  0.61 A30 GPU-h, including a first attempt that stopped before any network ran on an event (jobs 6822140, 6822142
  and 6822144); the GPU-hour ledger of RESULTS.md records them.

Every deviation from this file is logged with its reason in RESULTS.md before the affected read. For a val gate that
read is the closure or H-val model run, for the H-test rule or population it is H-T1, and for anything else H-T2.
Every H-test number is reported, whatever it is.

## 11. Where this addendum departs from the plan

| Plan, design v2 (607955f) | This addendum | Reason |
|---|---|---|
| "this document plus a spec v2 section" (section 9) | no spec v2 section; the plan and this addendum carry the content | the addendum states every setting the section would have held |
| emulation hash fixed at the end of H-T0.5 | overlay_hash and a provenance manifest fixed | the charge spread moved into the digitiser at C1; the hash does not cover the gun, Fatras or the merge |
| base seeds 650001, 950001, 750001, 850001 | 600,000,001 + 10,000,000 k; Fatras event numbers fixed by rank | two plan seeds are HNL-point bases; earlier streams are dense below 2.9 x 10^8; spares and split jobs must not shift streams |
| emulation inputs from 700 part02 events (section 6) | 949 part02 events | the tables used every stored part02 event |
| gun content of spec v1 section 3 | the same, plus an eta and r_v redraw after 1,000 failed azimuths | 13 of 30,000 LLPs needed it |
| F1 scored as in the earlier probe | 96 samples per segment | the 6-sample scoring alone erred by more than the threshold (`f1_containment_v1.md`) |
| C1 in bins of local incidence | matched incidence, plus a confirmation on H-val with a pooled part | the first, one-angle reading failed 7 of 525 bins; few H-val bins reach 200 hits |
| a failed C1 MAD rule is fixed by the resolution smear | any digitiser change, the smear included, follows section 9.3 | one rule for every change |
| emulation gate, sample not stated | full-sample closure; split-half reported; long-strip disks untested | the split-half check failed 4 of 26 disk rates |
| H3 decided at H-T0 | not part of H | decision D4 |
| navigator losses not foreseen | kept, recorded and reported; targets defined on the simulated hits | found at F1 on the gun probe |
| gun content of closure events not stated | no gun LLPs in closure events | V1 compares with stored val outputs that have no gun |
| V1 "target by target" | a pair counts when both versions are targets; unconditional V1 and the share below 5 hits reported; closure truth gate | a re-simulated target can lose hits to the emulation |
| V1 in C-full only | V1-Δ adds the closure set in C-1GeV | H1 is a difference between the two conditions |
| V1 (b) with a fixed 3-point rule | (b) without natives produced at a sensor, rule max(3 points, 2 σ_paired), a flag gate; (d) added | the fixed rule lies within the noise of about 430 pairs |
| V1 (c) on all 3,000 members | rule on the members with \|eta\| < 3 | the disk gap at \|eta\| > 3 exists before the emulation |
| V2 as one gate | host-row integrity as a truth gate; the host change as a flag | the gun hits can change host tracks for real, and no emulation change would fix that |
| V2 fake rate | over tracks with more than half host hits while blinded | blinding of gun-target results |
| a failed V2 or C1 confirmation: consequence not stated | section 9.3 | one rule for every val gate |
| C-nonoise optional on val | not run | fewer forks after unblinding |
