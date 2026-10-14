# H-P gates F1 and containment (2026-10-14)

This note records the field gate F1 and the containment gate of step H-P (`docs/pileup_overlay_plan.md` section 10).
The work is truth only. No model ran and no stored model output was read. The gun probe takes its hard-scatter
vertices from the truth particles of the tp_v1 val hosts. The B_eff comparison reads TrackML truth of the calib
training events. No test event and nothing from training_part03 was read.

## Production flow

`scripts/acts/overlay_sim.py sim` runs `ActsExampleFatrasGeneric` of a given build directory on gun particle CSVs.
It uses the settings of `configs/trackml/overlay_sim_v1.json`: the RZ map `trackml_like_rz_map_v1.txt` (md5 checked),
pMin 0.1 GeV (D6), bremsstrahlung off (D7) and hits on sensitive surfaces. v15 Fatras has no decays, so nothing is
set for them. The command writes the sim-hit CSVs, `fatras.log` and `sim_summary.json`. It exits with code 4 when a
particle fails, because v15 only logs such failures. `slurm/acts/overlay_sim.sbatch` wraps one run, and
`slurm/acts/hp_field_gates.sbatch` runs both probes and the gates for one build.

We used two probes. The field probe has 1,000 events of 100 particles with the mix of the field study's probe (seed
777). The gun probe has four gun events per val host, 600 in all, in the layout of plan section 7. All three builds
saw the same particles.

## Results

| Build | F1 (field probe) | particle failures, field / gun probe | largest band \|mean\| [MeV] | containment, gun hits failing |
|---|---|---|---|---|
| `build-v15` | FAIL | 1 / 337 | 0.029 | 6,083 of 84,290 (volumes 12 and 16 only) |
| `build-v15-zmirror` (G0 patch) | pass | 0 / 337 | 0.033 | 0 of 84,269 |
| `build-v15-zmirror-nav` (G0 patch and patch 0002) | pass | 0 / 286 | 0.033 | 0 of 85,071 |

The full tables are in `f1_containment_<build>.md` and `.json`. The thresholds are those of the plan.

**F1 on the field probe.** With the patched build no particle fails, and every |z| band on each side has a mean
bending residual within 0.034 MeV of zero. On `build-v15` a single particle of 100,000 failed, so F1 fails there.
The per-region means stay within 0.03 MeV. The settings acted as intended: particles between 0.1 and 0.5 GeV were
simulated, no secondary appeared, and electrons lose a median 5e-5 of their energy per crossing.

**Scoring of F1.** The earlier probe integrated the loop field with 6 samples along each segment chord. We first ran
F1 that way and found +0.100 MeV in the -z band at |z| = 2.5-3.1 m, with a small error. To test the scoring itself,
we propagated tracks through the same field with RK4 and no material and scored them with the same code. With 6
samples, segments of 800 mm in that band read +0.145 MeV, so the scoring error alone exceeds the threshold. With 96
samples the largest closure mean is 0.0024 MeV. The gate therefore uses 96 samples, and the threshold is unchanged.
The reports keep the 6-sample column next to the gate column.

**Containment.** On `build-v15` every failing hit lies in the -z strip endcaps, where 624 modules are 3 mm off the
TrackML plane and the trapezoids are inverted (plan section 3). With the z-mirror patch and the committed table
`configs/trackml/module_map_acts_zmirror_v1.csv`, every gun hit lies on its TrackML module plane and inside its
bounds. The tolerance is 4 um, as in G0. The largest distance from the plane is 3.8 um, in volume 17, which matches
the rounding of the TrackML file seen by G0. All 1.2 million field-probe hits also pass.

## Gun particles lost by the v15 navigator

The gun probe fails F1's rule on every build. 337 of 16,174 gun particles (2.1 %) fail with "Interpolation out of
bounds". Among isolated gun particles with pT > 1 GeV and r_v >= 100 mm the share is 2.9 %. Fatras removes a failed
particle together with all its hits.

The map is not the cause. Verbose logs show the navigator losing these particles, which then fly without
navigation until they leave the map at r = 3 m or |z| = 6 m. We found two mechanisms:

1. A particle that starts inside a volume, moving inwards on a line that misses the inner layers, gets no layer
   candidates. `TrackingVolume::compatibleLayers` walks the layers only in the sign of the radial direction, so the
   outer layer the line meets later is never tested. The navigator then aims at the volume boundary.
2. On a shallow crossing the step ends more than 0.1 mm past the volume boundary, which is the EigenStepper's
   overstep tolerance. The navigator reports "Boundary navigation lost" and finds no further target.

In a constant 2 T field, the field of spec v1, the same particles do not fail. Instead, 1,001 of 16,153 (6.2 %) end
outside the detector world, with 1.6 hits on average against 5.3 for the others, and nothing is logged. The generic
samples of spec v1 were simulated by the same v15 navigator and probably carry the same losses. We did not check
them.

Patch `patches/acts_v15/0002-TrackingVolume-compatibleLayers-full-scan-fallback.patch` addresses the first
mechanism: when the directional walk finds no layer, it tests every layer of the volume. Built with the G0 patch
into `acts/build-v15-zmirror-nav` (`slurm/acts/build_v15_zmirror_nav.sbatch`), it lowers the failures to 286 (1.8 %).
The field probe gives the same hits as without it. The second mechanism remains. We did not try a change of the
overstep tolerance.

## B_eff against TrackML truth (descriptive)

The median bending field of pointing segments agrees with TrackML truth within 0.02 T below |z| = 1.5 m on both
sides. Between 1.5 and 2 m the simulation is 0.03-0.08 T higher on both sides. Beyond 2 m the differences reach
0.2 T, where the fit has little data (plan risk 3). Up to 2.5 m the (+z) - (-z) asymmetry has TrackML's sign and a
similar size. The two samples differ in material and particle mix, so these medians are context and carry no rule.

## Decision needed before H-T0

The plan's F1 passes on the patched build, but production would lose about 2-3 % of the gun targets to navigation.
The options are:

- keep the losses and record the dropped gun particles in the sidecar (the targets are then defined on what remains);
- adopt patch 0002, and test a larger overstep tolerance for the second mechanism, before the production build;
- check the spec v1 generic samples for the same losses, since they enter addendum G.

`slurm/overlay/fatras.sbatch` runs the same Fatras settings but does not read the log, so it would not notice
failed particles. `scripts/acts/overlay_sim.py sim` stops on them.

Jobs (submit-gpu-express, CPU only, A30 and GTX 1080 nodes excluded): probes, simulation and gates 6819591 and
6819822-6819827, final gate step 6819851-6819853; diagnosis 6819663, 6819675, 6819682, 6819684, 6819687, 6819838;
build 6819764. Together about 5 allocated core-hours. The
scratch is in `/work/submit/jinboz1/hnl/pileup_design/hp_field/`.
