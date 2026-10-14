# Step H-P: emulation, overlay gun and overlay writer

Status: done on 2026-10-14. The emulation gate of plan section 10 passes on the measurement sample, and its split-half
check passes for the stop rates but not for four disk hole rates (see below). The writer passes the converter checks
on three val hosts. This step belongs to step H-P of `docs/pileup_overlay_plan.md` (sections 4.4, 5.3, 5.4, 6 and 7).
It read truth and hits of TrackML training events of `training_part02` and of three tp_v1 val hosts. No model ran, no
stored model output was read, and nothing of `training_part03` was streamed. Scratch work and logs are in
`/work/submit/jinboz1/hnl/pileup_design/hp_overlay/`.

## Code

| Module | Content |
|---|---|
| `src/hnl/overlay/emulate.py` | measurement of the emulation tables, the emulation gate, and `emulate` for gun sim hits |
| `src/hnl/overlay/gun.py` | the overlay gun: spec-v1 content without Pythia, 8 isolated and 2 dense LLPs per host, TrackML particle ids |
| `src/hnl/overlay/writer.py` | gun-only TrackML members, frozen conversion, merge with the host rows, storage, checks, `assemble` |
| `src/hnl/overlay/fatras_io.py` | Fatras particle CSVs and packed sim hits |
| `src/hnl/trackml/events.py` | `table_root` sends ids from 50,000,000 to `derived/overlay_v1` (decision D9) |
| `configs/trackml/overlay_emulation_v1.json` | the measured tables (block `model`) and their diagnostics |
| `slurm/overlay/fatras.sbatch`, `run_py.sbatch` | Fatras with the field map (pMin 0.1 GeV, no bremsstrahlung), small CPU steps |

The emulation hash (`emulate.emulation_hash`, tables plus code) is 2be64cc0... for this version. The plan fixes the
hash only at the end of H-T0.5.

## Emulation measurement

**Sample.** We used every stored event of `training_part02` (949 events, ids 22000-22999). All of them are training
events of `configs/trackml/split_tp_v1.json`, and none is a calibration, val or test event. We selected prompt charged
particles (production radius below 1 mm, pT above 1 GeV, |eta| below 3.5, PDG e, mu, pi, K or p, at least one hit).
That gives 931,817 particles with 10.4 M TrackML hits. Their initial kinematics went through Fatras in the patched
build (`build-v15-zmirror`) with the field map, so that TrackML and Fatras describe the same particles (jobs 6819567,
6819579). Seven particles failed in Fatras on the field-map bounds or the step limit.

**Hole rates and drop rates.** Hole rates use the definition of `field_material/track_stats.py`. A hole is a layer
between the first and the last hit layer of a track without a hit, counted per barrel layer for central tracks and
per disk within each endcap volume. The same code runs on TrackML and on Fatras. The Fatras rate is the geometric part
(zero in the barrel and in the strip disks, 1.5-2.7 % in the pixel disks). Each layer's per-hit drop rate eps then
solves (1 - g) [eps (1 - f2) + eps^2 f2] = h - g. Here h is the TrackML rate, g the Fatras rate, and f2 the share of
layer crossings with two hits, since a layer with two hits becomes a hole only when both are dropped.

| Volume | Measured layers | Interior crossings | TrackML hole rate [%] | Fatras hole rate [%] | Drop rate eps [%] |
|---|---|---|---|---|---|
| 7, pixel disks -z | 5 | 96,179 | 2.01-3.34 | 1.62-2.67 | 0.73-1.15 |
| 8, pixel barrel | 3 | 156,171 | 0.78-0.89 | 0 | 0.98-1.10 |
| 9, pixel disks +z | 5 | 91,540 | 2.01-3.35 | 1.54-2.65 | 0.75-1.00 |
| 12, short-strip disks -z | 4 | 60,553 | 0.84-0.96 | 0 | 0.94-1.08 |
| 13, short-strip barrel | 4 | 198,239 | 0.73-0.78 | 0 | 0.92-0.99 |
| 14, short-strip disks +z | 4 | 59,315 | 0.87-1.03 | 0 | 0.97-1.15 |
| 16, 18, long-strip disks | 0 | 7 | - | - | 0.98 (pooled strip layers) |
| 17, long-strip barrel | 1 | 47,572 | 0.83 | 0 | 0.97 |

Source: `configs/trackml/overlay_emulation_v1.json`, `diagnostics.holes_per_layer` (job 6819646). Layers that are
never interior (first and last barrel layer, outermost disks) take the pooled rate of their volume. Prompt tracks cross
at most two long-strip disks, so those disks cannot be measured. They take the rate pooled over the measured strip
layers. Every measured layer lies near 1 %, which suggests a uniform per-hit inefficiency in TrackML.

**Modules without hits.** A second measure compares each TrackML particle with the Fatras crossings of the same
particle. It counts a crossing as missed when TrackML has no hit there although it has a later one. This miss rate is
about 1 % on every layer except the first pixel layer (8, 2), where it is 10 %. The missing hits sit at the ends of
that layer. TrackML misses 0.7 % of the Fatras crossings at |z| below 200 mm and 97 % at |z| above 400 mm
(`emulation_first_layer_v1.json`, job 6819605). Modules whose miss rate exceeds their layer's by at
least 5 points (at least 100 crossings, binomial z of at least 5) get their own drop rate (`model.eps_module`).

| Layer | Modules overridden | Their miss rate [%] |
|---|---|---|
| 8, 2 at \|z\| = 315 mm (32 modules) | 32 | 42 (mean) |
| 8, 2 at \|z\| = 385 mm (32 modules) | 32 | 84 (mean) |
| 8, 2 at \|z\| = 455 mm (32 modules) | 32 | 99 (mean) |
| 13, 6 and 13, 8 | 3 | 6.0-7.6 |

The plan's hole definition requires a hit on the first barrel layer, so it could not see this.

**Hadronic stops.** The Fatras hits of a particle list the layers it would cross without interactions. A TrackML
particle stops after crossing k when its last TrackML hit is at k and Fatras continues. The hazard first looked like a
rising function of momentum. At fixed pT above 1 GeV, high momentum means large |eta|, and so long paths through the
barrel layers. We therefore model the stop probability of a crossing as 1 - exp(-lambda s). Here s is the path factor
1 / |cos(incidence)| on the layer and lambda a hazard per unit path. Lambda is binned in species (pi, K, p), momentum
(0-2, 2-4, 4-8 and above 8 GeV) and layer, as the plan asks. Sparse cells fall back to coarser ones
(`HAZARD_LEVELS`). A particle that loses its last hit to inefficiency looks like a stop. The expected number of such
cases is subtracted, using the miss rate of the last layer.

| Pions, lambda [% per unit path] | 8,2 | 8,4 | 8,6 | 8,8 | 13,2 | 13,4 | 13,6 | 13,8 | 17,2 |
|---|---|---|---|---|---|---|---|---|---|
| p 0-2 GeV | 0.61 | 0.73 | 0.76 | 2.31 | 0.96 | 1.07 | 1.02 | 1.09 | 1.54 |
| p 4-8 GeV | 0.63 | 0.90 | 1.26 | 2.31 | 0.83 | 1.08 | 0.83 | 0.97 | 1.50 |

The step after the last pixel layer carries the most material. Kaons and protons give the same pattern within
0.1-0.3 points. Pooled over the barrel, lambda falls from 1.11 % at a mean path factor of 1.2 to 0.57 % at 6.5. The
path model therefore overstates the stops of very inclined crossings, and the momentum bins absorb part of this. The
disk volumes give 1.1-2.3 % (pixel), 1.1-2.1 % (short strip) and 2.6-3.9 % (long strip). As a check of the
method, muons show 16 stops against 15.5 expected from inefficiency, and electrons 43 against 25.

**Pixel charge.** The plan's per-hit charge factor is not part of the emulation. Gate C1 replaced it by a per-cell
factor inside the digitiser (`results/pileup_hp/c1_v1.md`), and the digitiser refuses the two together. The per-hit
spread measured here on part02 serves as a cross-check. Charge times cos(incidence) over the effective thickness has
5-95 % quantiles of 0.909-1.087 (barrel) and 0.855-1.145 (endcaps), as the calibration events gave for the plan
(`diagnostics.charge_per_hit_over_path`).

## Emulation gate

The emulation runs on the Fatras hits of the selected particles (10 draws), and the result is compared with TrackML
by the code above. The plan's rule: the total hole rate per barrel layer and per disk within 0.2 points of TrackML,
and the share of pi, K and p reaching the last barrel layer within 1 point. Barrel holes use central tracks (|eta|
below 0.8, pT above 2 GeV, a hit on the first layer). Disk holes use tracks with pT above 2 GeV and two hit disks of
the volume.

| Check | TrackML | Emulated | Difference [points] | TrackML stat. uncertainty [points] | Pass |
|---|---|---|---|---|---|
| reach of the last barrel layer, pi (35,553 tracks) | 88.84 % | 88.58 % | -0.26 | 0.17 | yes |
| reach, K (10,219) | 88.96 % | 88.63 % | -0.33 | 0.31 | yes |
| reach, p (7,375) | 88.18 % | 88.34 % | +0.16 | 0.38 | yes |
| hole rate, 26 measurable barrel layers and disks | - | - | largest \|difference\| 0.06 | 0.04-0.17 | yes, all |

Source: `emulation_gate_v1.json` (job 6819648). Fatras without emulation reaches the last layer for 99.98 % of pions.
The gate shares its particles with the tables, so it is a closure test of the procedure. The long-strip disks have
seven interior crossings in all and are not tested.

**Split-half check.** Tables from the even events, gate on the odd events (`emulation_gate_half_v1.json`, job
6819649). The reach differences are -0.46, -0.67 and -0.16 points, within the rule. Four of the 26 hole rates miss the
0.2-point rule: disks 7,6, 9,12, 14,4 and 14,6, by 0.21-0.37 points. In this check both halves fluctuate. The
uncertainty of each difference is 0.08-0.34 points, and the 26 differences give chi2 / ndf = 0.91, with the largest
at 2.2 standard deviations. At half the statistics the rule is at or below the resolution of a single disk. We report
this as a fail of the split-half check and did not change the rule.

**Not covered by the gate.** The miss rates relative to Fatras (`miss_rate_vs_fatras_descriptive` in the gate file)
show what the hole definition cannot. On the first pixel layer the emulation gives 8.7 % against TrackML's 10.0 %.
The module average flattens the edge near |z| of 300 mm. On the long-strip disks TrackML misses 2.2-3.0 % of the Fatras
crossings and the emulation 0.7-1.0 %. Part of the gap is path mismatch, because TrackML scatters far more than v15 in
the strip endcaps (plan section 4.3). The rest may be inefficiency that prompt tracks cannot isolate.

## Overlay gun

`hnl.overlay.gun` places 8 isolated (LLP index 1-8) and 2 dense (index 9-10) LLPs at the host's hard-scatter vertex.
That vertex is the production vertex shared by most vertex-1 particles within 1 mm of the beam line. Each LLP draws
exactly as `scripts/acts/gun.py` (imported, not copied). Its azimuth is then redrawn until its direction is at least
Delta R = 0.8 from the LLPs placed before it. After 1,000 failed azimuths, eta and r_v are redrawn too. In 3,000
simulated hosts this happened for 13 of 30,000 LLPs. Without those cases the gun needs 0.58 azimuth redraws per LLP.
The study's 0.88 (plan section 7) includes cases where its Monte Carlo kept eta fixed and gave up.

TrackML particle ids carry vertex field 1000 + LLP index, the ACTS particle index, generation and sub-particle
(plan section 5.3). The encoder refuses any vertex outside 1001-1010, so no id reaches the int64 sign bit. A unit test
reads the ids through the converter's CSV reader and gets positive int64 values. Gun hits have weight 0 (D10).

## Writer and dry run

The gun-only event is written as TrackML's four CSV members, with hits sorted by module as TrackML sorts them, and
converted by the frozen `throughput.event_tables_tp`. Its rows follow the host's stored rows unchanged, so gun hit ids
continue after the host's. A unit test shows that this merge equals the frozen conversion of host and gun written as
one event, table by table. The stored layout is the converter's, and `convert.read_table` and `events.load_events` read
it without change.

Dry run: three tp_v1 val hosts in the pilot set (k = 9, base seed 990001, scratch only), through the gun, Fatras of
the patched build, the emulation, the digitiser with the patched module table, the writer and `verify_set` (jobs
6819653, 6819654, 6819667, 6819670).

| Overlay id | Host hits | Gun hits | Gun particles (with hits) | P_P targets | P_D targets | True edges: host + gun | Host weight sum |
|---|---|---|---|---|---|---|---|
| 59,021,117 | 106,225 | 115 | 26 (25) | 4 | 4 | 79,672 + 90 | 1.0000000 |
| 59,021,124 | 121,140 | 135 | 24 (24) | 7 | 3 | 92,880 + 111 | 1.0000000 |
| 59,021,127 | 120,242 | 104 | 24 (21) | 2 | 4 | 92,203 + 83 | 1.0000001 |

Source: `overlay_dryrun_v1.json`. In all three events the converter accepted the gun-only event (largest distance
outside a module 0 mm). Host rows and host true edges are identical to tp_v1, and particle ids are unique and positive
as int64. Gun vertex fields span 1001-1010 and gun weights are 0. `verify_set` found no problem. Three of 74 gun
particles failed in Fatras on the field-map bounds. The F1 probe of the gun found the same at 2.1 %
(`f1_containment_build-v15.md`), and the gun sample needs a fix there before production.

## Tests

`tests/test_overlay_emulate.py`, `test_overlay_gun.py` and `test_overlay_writer.py`, together with the existing
TrackML converter, event and overlay tests: 86 passed (job 6819674).
