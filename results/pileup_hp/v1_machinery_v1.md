# V1 closure machinery: re-simulated native val targets (truth only)

Status: 2026-10-14. This note defines how the closure set of gate V1 is made (`docs/pileup_overlay_plan.md`, design v2,
sections 5.4, 9 and 10) and reports a dry run on the five pilot hosts of step H-P. It closes the open point of
`hp_summary.md` section 4: "the id scheme for re-simulated native targets is not defined". All work was CPU only on
`submit-gpu-express`, with the A30 and GTX 1080 nodes excluded. No model ran and no stored model output was read.
Nothing from `training_part03` and no test event was read. Numbers are in `v1_machinery_v1.json`.

## 1. Which natives are replaced

The plan has one closure set (set index 1), with one overlay event per tp_v1 val host. Each closure event therefore
replaces the three V1 populations of its host at once. They are native targets (stored `is_target`: pT > 1 GeV,
|eta| < 4, at least 5 hits), with generation = (particle_id >> 24) & 0xFFF.

| Population | Definition | Selected over val | Central / fwd +z / fwd -z |
|---|---|---|---|
| a | generation 0, \|d0\| >= 10 mm or r_v >= 100 mm (the P_P definition) | 3,276 | 1,250 / 1,050 / 976 |
| b | generation >= 1, \|d0\| >= 10 mm | 448 | 44 / 213 / 191 |
| c | generation 0, \|d0\| < 1 mm and r_v < 1 mm; the first 3,000 of 153,603 by the hash order | 3,000 | 48,336 / 52,682 / 52,585 before the cut |

The counts per population and region equal `pileup_design/plan_revision/val_regions.json` (job 6812736) exactly. The
prompt subset uses the key int(sha256("canto-trackml-overlay-v1-prompt:<event>:<particle_id>")[:8], 16), with ties
broken by event and then by particle id, as `configs/trackml/split_h_v1.json` states. The 153,603 keys contain two
repeats, and no tie sits at the cut. A region is "central" without a native hit at |z| > 1 m, otherwise
forward with the sign of eta.

**Removal.** A replaced native loses its particles row, its ptype row and all its hits. Its secondaries and every
other particle stay, and the remaining host rows keep their order. Each replaced native gets exactly one closure row,
also when Fatras drops it. Closure hits carry weight 0. The host weights of a closure event then sum to one minus the
weights of the removed hits, which is the plan's rule. The closure set holds no gun LLPs, so the replacement is the
only change against the stored val event.

## 2. Ids

The closure particle id is the native id plus 2^62, which sets bit 62. The TrackML vertex field becomes 1024 plus the
native vertex field. Primary, generation, secondary and process keep their native values.

| Property | Why it holds |
|---|---|
| no collision with host ids | host vertex fields are at most 1000 (the writer's host rule); the largest in tp_v1 val is 251, over 1,614,234 particles |
| no collision with gun ids | closure vertex fields lie in 1024-2024; gun fields are 1001-1010 |
| int64 safe | bit 63 stays clear, so ids are positive int64 and pandas reads the CSV ids as int64 (above 2^63 - 1 it would switch to uint64) |
| reversible | native = closure - 2^62; the stored rows pair with the stored val outputs through this rule |

The process field cannot carry a marker, since TrackML uses it (356,712 val particles have a nonzero value). In
Fatras a closure particle is the ACTS barcode (vertex primary 2, secondary vertex 0, particle i, generation 0,
sub-particle 0), with i the rank of its native id within the host. The closure truth table maps barcodes to ids, and
barcodes never reach the stored tables.

## 3. What the frozen converter sees

The re-simulated particles of a host form one closure-only event. It is written as the four TrackML CSV members in the
gun-only layout of `hnl.overlay.writer`: hits sorted by module, hit ids 1..M, weight 0, nhits counted after
digitisation. The particle rows carry the closure ids, the native type and charge, and the native's stored float32
vertex and momentum. The CSV format reproduces these float32 values exactly. The frozen `event_tables_tp` therefore
recomputes pt, eta, r_v and d0 bit-identical to the native row. The closure-only event is then appended to the host
rows without the natives, as a gun event is.

Some derived columns differ from the native's by construction. n_hits, n_pixel_hits, r_first, the hit ranks and the
target flags come from the re-simulated hits. vprim is 1024 plus the native vertex, so `is_hard_scatter` is False. The
K_S and Lambda tagging runs inside the closure-only event, so a daughter keeps its label only if its partner is
replaced too. The model reads hits only. Evaluations should take population, region, hard-scatter flag and category
from the native, through the closure truth table.

## 4. Code

| File | Content |
|---|---|
| `src/hnl/overlay/closure.py` | ids, populations and the prompt subset, Fatras input, closure members and event (the gun chain with the closure id rule), the closure converter gate, assembly, verify |
| `src/hnl/overlay/closure_checks.py` | truth checks of a stored closure set (section 5) |
| `slurm/overlay/closure_set.sbatch` | select, make, Fatras, pack, assemble, verify and checks as one job |
| `tests/test_overlay_closure.py` | 16 unit tests; the overlay suite passes (111 tests, job 6820121) |

Seeds follow the gun sets: Fatras base + 13, emulation `default_rng([base + 13, overlay id, 1])` and digitiser
`default_rng([base + 13, overlay id])`. The production base seed is the closure entry of `split_h_v1.json`
(610,000,001). One test checks that the closure chain, given the gun's ids, produces CSV bytes identical to
`writer.gun_event`.

## 5. Dry run on the five pilot hosts

Hosts 21289, 21332, 21610, 21753 and 21835 became closure events 51,021,289 to 51,021,835. We used the pilot seed
990101 (Fatras 990114), the patched build `build-v15-zmirror` and the RZ map. The emulation and overlay hashes equal
the pilot's (2be64cc0... and 3e23b363...). The tables were written to the work directory, so the production root keeps
the closure ids free. The Fatras streams equal those of the gun pilot, since both use base 990101 and events 0-4.
Neither run feeds a result. Job 6820120 ran the chain and the checks in 24 s.

| Check | Result |
|---|---|
| Population counts against `val_regions.json`; prompt subset 3,000, no tie at the cut | equal; pass |
| Closure truth of each host = the population table's rows of that host | 5 of 5 |
| Ids: rule, inverse, unique positive int64, vertex fields 1025-1241, no host or gun collision, CSV read as int64 | 246 of 246 |
| `verify_closure_set` (sha256, schema, row groups, closure gate) | 0 problems |
| Rows = host - removed + closure; closure hits last; host true edges after removal unchanged; no mixed edge | 5 of 5 events |
| Closure true edges = sum of (hits - 1) | 401, 467, 372, 402, 399, as expected |
| Weight rule: closure weights 0; host sum = 1 - removed weight | largest deviation 1.4e-7; the gate flags the sum 1 in every event |
| X1 cluster features finite; every hit on a module of TrackML's `detectors.csv` | all events |
| Converter view: q, vertex, momentum, pt, eta, r_v, d0 and type equal to the native | bit-identical for 246 of 246 |
| vprim = native + 1024; `is_hard_scatter` | all; False for all 246 (8 natives were hard scatter) |
| Production replay from the packed sim hits | CSV bytes identical; conversions equal the stored rows |
| Fatras | 246 of 246 simulated, no failure |

| Host | a | b | c | removed hits | removed weight | closure hits | stopped / dropped by the emulation |
|---|---|---|---|---|---|---|---|
| 21289 | 23 | 4 | 22 | 489 | 1.12 % | 450 | 34 / 8 |
| 21332 | 24 | 7 | 24 | 539 | 1.14 % | 522 | 32 / 5 |
| 21610 | 20 | 2 | 22 | 424 | 1.14 % | 416 | 21 / 10 |
| 21753 | 22 | 5 | 23 | 493 | 1.02 % | 452 | 28 / 8 |
| 21835 | 22 | 4 | 22 | 451 | 1.37 % | 447 | 5 / 4 |

The category check shows the expected loss of pair labels. In population a, 14 K_S daughters (seven pairs) keep their
label, while 48 K_S and 42 Lambda daughters become "other" because their partner is not replaced.

## 6. Truth-level closure (descriptive)

Each native track is compared with its re-simulated version. A native was selected as a target on its own hits, so a
re-simulation can fall below 5 hits even with a perfect chain.

| Population | natives | hits, native | hits, re-simulated | difference | still a target | first hit on the same layer |
|---|---|---|---|---|---|---|
| a | 111 | 7.66 | 7.23 | -0.42 +- 0.17 | 93.7 % | 98.2 % |
| b | 22 | 9.36 | 9.00 | -0.36 +- 0.22 | 95.5 % | 72.7 % |
| c | 113 | 11.86 | 11.38 | -0.48 +- 0.30 | 94.7 % | 94.7 % |
| all | 246 | 9.74 | 9.30 | -0.44 +- 0.16 | 94.3 % | 94.3 % |

Where both first hits lie on one layer, they are a median 8 um apart. Hits of a native and its re-simulated track on
the same layer have the same mean cluster size: 2.57 cells against 2.59 over 1,267 pairs. The pixel charge ratio is
1.007. Their median distance grows outward, from 0.09 mm in the pixel barrel to 3.2 mm in the long-strip endcaps.

Below |eta| = 3, the natives and their Fatras hits agree within 0.04 hits per crossed layer in all 11 region and
|eta| bins with more than 15 tracks. In the pixel endcaps at |eta| = 3-4, the natives have 1.99 hits per crossed disk
and their Fatras hits 1.63, with 123 crossed disks in both (20 tracks). The gap is already present before the
emulation, and its cause was not established. V1 (c) in |eta| bins will show whether the model sees it. The gun covers
|eta| < 3.

## 7. Points for addendum H

1. The draft of addendum H (`docs/preregistration_pileup.md`, section 9.2) pairs a native with its re-simulated track
   only when both are targets, and reports the share that drops below 5 hits. Here that share is 5.7 %.
2. The converter gate of that draft names the gun vertex range. For closure events the matching rule is the closure
   range 1024-2024, with no particle in 1001-1010.
3. Production uses the closure base seed of `split_h_v1.json` and writes to `derived/overlay_v1` (decision D9).
   `hnl.overlay.writer.SETS` still lists the plan's older seeds; the closure code does not read them.
4. A replaced native's secondaries stay in the host. v15 Fatras makes no secondaries, so the re-simulated track has
   none.

## 8. Files outside the repository

`/work/submit/jinboz1/hnl/pileup_design/hp_dryrun/closure_v1/` holds the population table (`population_v1.parquet`,
sha256 33e29d7e..., and its record), the Fatras input and output, the closure-only CSV tar, and the stored tables of
the dry run (`tables/`, 12 MB in total).
