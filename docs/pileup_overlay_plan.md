# TrackML pileup version of the displaced gun sample (design v2)

Status: design only (2026-10-14). Nothing has been produced or run. Version 2 revises version 1 after a physics
review and a hygiene review. Section 16 lists what changed and which review points were checked and not adopted.

The plan rests on four studies made on 2026-10-14: geometry, field and material, hit format, and study design. They
have no written reports. Their scripts, JSON outputs and logs are in `/work/submit/jinboz1/hnl/pileup_design/<topic>/`,
and every number below cites a file and its JSON key, or a job id (section 15). The physics reviewer's checks are in
`pileup_design/review_physics/`, and the checks made for this version are in `pileup_design/plan_revision/`. All jobs
ran CPU-only on `submit-gpu-express`, with the A30 and GTX 1080 nodes excluded. They read truth of TrackML training
and validation events and of the generic train split. One study read the frozen model's stored outputs on val events
(job 6812139). No model was run, and no output on a test event was read.

## 1. Purpose

- **Route agreed with the supervisor** (2026-10-14, on the proposal relayed by the user). The first step is a displaced
  gun sample in the HNL benchmark setup (ACTS v15, GenericDetector, Fatras, no pileup), so that the frozen HNL models
  apply directly. That sample exists (`docs/new_samples_spec.md`, addendum G). A later version places the same kind of
  displaced gun tracks into TrackML pileup events. This document designs that later version.
- **Why TrackML events.** They carry pileup at mu = 200 and noise hits. Our only model trained on such events is the
  frozen Canto-L (TrackML), `runs/canto_l_v1/seed0_x1` (`docs/model_card_canto_l_trackml.md`). Native displaced
  tracks are rare there: all training events together hold 352 targets with |d0| >= 100 mm (model card, section 2).
  The overlay adds a known displaced population to real pileup.
- **What it measures.** The one claim, H1 (section 9), is a paired difference for the frozen Canto-L (TrackML) on the
  gun targets P_P: Delta = eps(P_P, C-full) - eps(P_P, C-1GeV). C-full keeps every hit. C-1GeV keeps only the hits of
  particles with truth pT above 1 GeV (model card, section 3). It removes noise and soft particles but keeps the
  pileup particles above that threshold. The condition without any pileup, C-hs, is reported without a claim.
- **What waits.** Production, the streaming of new TrackML data and every model run wait for the decisions of
  section 13 and for the commit of addendum H (section 9).

## 2. Route

### 2.1 Chosen route

The gun particles are simulated again in a setup that reproduces the TrackML detector description. Their hits are
written in the TrackML format into TrackML host events. The hits of the existing HNL-setup gun sample are not reused.

1. **Gun.** The content of spec v1 section 3 (`scripts/acts/gun.py`, isolated and dense components) without Pythia.
   The LLPs start at the hard-scatter vertex of the host event (section 7). Output: one particle CSV per event.
2. **Simulation.** `ActsExampleFatrasGeneric` from a new ACTS v15 build with a one-line geometry patch (section 3).
   It reads the TrackML-like field map (`--bf-map-file trackml_like_rz_map_v1.txt --bf-map-type rz`, section 4.1) and
   writes sim hits with position and momentum (`--output-csv`).
3. **Emulation.** Hit inefficiency beyond the geometric holes, hadronic stops and the pixel charge spread are applied
   to the sim hits. Their rates are measured on TrackML training events (section 4.4).
4. **Digitisation.** A Python TrackML-style digitiser places each sim hit in its TrackML module frame. It produces
   cells, a cluster-centroid position and TrackML module ids (section 5.2).
5. **Conversion.** The gun-only event is written as the four TrackML CSV members and converted by the frozen
   `hnl.trackml.convert.event_tables` and `throughput.ptype_table`. Its rows are appended to the host's rows
   (section 5.4).

Each simpler route fails a measured property of the frozen model or of the TrackML detector:
- The X1 embedding reads cluster features (ncells, charge, span0, span1), which smeared hits do not have (model card,
  section 1.1).
- The TrackML field is neither uniform nor symmetric in z (section 4.1). More than half of the gun targets have a hit
  at |z| > 1 m (`study_design/yield_overlap_train.json`, `field_exposure_train`; job 6812128).
- The 1,608 modules of the ACTS v15 negative-z strip endcaps differ from TrackML's in shape and position (section 3).

The patch and the map share one new build. The full v15 build took 10 min on 5 cores (job 6810588). Simulation and
digitisation of the gun add a few CPU-h in total (section 11).

### 2.2 Alternatives rejected

| Route | Reason for rejection | Source |
|---|---|---|
| A. Copy the hits of `generic_iso_v1` / `generic_dense_v1` into TrackML events after relabelling the module ids | Smeared positions (14.4 um in the pixels, spec v1 section 2; TrackML pixel barrel 3.8 / 4.0 um, `configs/trackml/res_v1.json`). No cells, so the X1 features would be set to 1. Uniform 2 T. The npz writer drops a second hit of a particle on one module. The barcode reads as hard-scatter, with generation = particle index. 4.6-5.3 % of all hits fail the TrackML -z strip-endcap modules; 18.6 % (iso) and 21.7 % (dense) of gun particles have at least one such hit | `geometry/out/impact.json` (job 6812183); hit_format jobs 6812130, 6812131 |
| B. As A, plus re-digitising the original sim hits and moving -z strip-endcap hits 3 mm along their direction | Needs a production rerun with KEEP_CSV=1 and about a day of code. Keeps the 2 T field. Cannot create the hits that the TrackML module outlines would record but the inverted ACTS modules miss | geometry study, `geometry/out/footprint.json` |
| C. Simulate again in the unpatched v15 GenericDetector with the field map (prebuilt executable, no rebuild) | Same -z strip-endcap problem as B: 5.9 % (iso) and 7.0 % (dense) of gun hits fail. Removing it costs one rebuild | `geometry/out/impact.json`, `frac_gun_hits_bad` |
| D. Uniform 2 T, or `acts.SolenoidBField` | With 2 T, gun-like displaced tracks sit a median 1-12 mm off in r-phi at \|eta\| > 1 (p90 up to 34 mm), against 17-26 um strip resolution. SolenoidBField passes k^2 where boost expects the modulus k (also on ACTS main, 2026-10-14), so it is wrong off the axis, and it costs 13 times the Fatras CPU | `field_material/prop_closure.json` (`part_b_gun_like`), `acts_model_test.json` (jobs 6812187, 6812193) |
| E. ACTS `PlanarSteppingAlgorithm` for the cells | Reproduces the barrel. In the endcaps it gives half the TrackML cluster size: v15 uses half the module thickness as half-thickness everywhere (ProtoLayerCreatorT.hpp L267, L471), while TrackML endcap clusters match twice the module thickness. Strip values also need resetting to 1 | `hit_format/out/resid.json` (`acts_planar`), job 6812130 |
| F. Patch the Python bindings to accept a field map | Untested; 0.5-1 day plus a rebuild. The executable already takes the map (job 6812220) | field_material job 6812138 (TypeError) |
| G. A full TrackML detector description with TrackML material | TrackML never published its field or material (arXiv:1904.06778, Sec. 5.2.2). A thickness change of the GenericDetector layers cannot match TrackML in every region (section 4.3, decision D5) | `field_material/material.json` |

## 3. Geometry and module ids

The geometry study compared the module tables of TrackML (`external/trackml/detectors.csv`, md5 440dbb93...) and of
the unpatched v15 build (jobs 6812101 and 6812102).

| Property | Finding | Source |
|---|---|---|
| Module ids | 18,728 sensitive modules in both, with the same (volume, layer, module) id sets, counts per layer and maximum ids. Volumes 7/8/9 are pixels, 12/13/14 short strips, 16/17/18 long strips | `geometry/out/modules_summary.json` (`id_join`, `modules_per_layer`) |
| Barrel (volumes 8, 13, 17; 14,000 modules) | The same id names the same module. Centres agree within 0.65 um (3.7 um in volume 17); rotations, bounds, thickness and pitch agree | `modules_summary.json` (`geometric_per_volume`); `hit_format/out/modmap.json` (`barrel_identity_frac`) |
| +z endcaps and -z pixel endcap (3,120 modules) | Centres, rotations and outlines agree within 0.65 um. Volume 7 has u and v turned by 180 degrees about w, which leaves its rectangles unchanged | `modules_summary.json`; `modmap.json` (`axis_dot_u`, `axis_dot_v`) |
| Bounds, thickness, pitch | Bounds agree within 3.1e-6 mm. Thicknesses are 0.15 / 0.25 / 0.35 mm. TrackML pitches equal the readout bins of the v15 builder (pixel 50 x 56.25 um; the v15 `default-geometric-config-generic.json` has 50 um in v and is not used) | `modules_summary.json` (`tml_per_volume_specs`, `acts_per_volume_bounds`) |
| Endcap ids (4,728 modules) | Permuted: 1.6 % keep their id. For 96.8 % the TrackML module with the same id is more than 50 mm away. The permutation is the same in every layer of a volume and bijective | `plan_revision/same_id_distance.txt`; `modmap.json` (`endcap_permutation`) |
| -z strip endcaps (volumes 12 and 16, 1,608 modules) | All trapezoids inverted, corners 6.0-10.2 mm off. 324 modules of volume 12 (ring r = 318 mm) and 300 of volume 16 (ring r = 942 mm) sit 3.000 mm off the TrackML plane | `modules_summary.json` (`endcap_patterns`, `endcap_ring_frac_dz_nonzero`); `plan/map_agreement.txt` |
| Effect on a relabel-only overlay | 9.5-10.2 % of hits fall in these volumes, and 4.6-5.3 % of all hits fail | `geometry/out/impact.json` (job 6812183) |

Unmapped ACTS endcap ids would fail the converter's 50 mm data-quality rule, so they go through a table.

**Why the -z strip endcaps differ.** ACTS v15 builds the -z endcaps as a point reflection of the +z side
(`ProtoLayerCreatorT.hpp` L489, `moduleCenter(side * ringModulePosition)`). TrackML's -z centres are the exact z-mirror
of its +z centres. Line L491 takes the local y axis from the azimuth of the +z position, so the local v axis of each
ACTS -z module points inward and the trapezoid is inverted. The 3 mm sub-plane stagger alternates with iphi parity
(`BuildGenericDetector.cpp` L130). It flips in rings with an odd number of modules per half-turn, which moves those two
rings off the TrackML plane.

**Mapping in the chosen route.**
1. Copy `src-v15` to a new source tree and place the module centre at (x, y, side * z) of `ringModulePosition` in L489.
   Build it in a new build directory, so that `src-v15` and `build-v15` stay untouched. With z-mirrored centres, L491
   gives a radially outward v axis, w = side * z and u = v x w. TrackML's endcap rotations follow this construction
   within 1.5e-6 (geometry study, login-node check).
2. Dump `detectors.csv` from the patched build and rerun `geometry/compare_modules.py` and `footprint.py` (gate G0,
   section 10). That the patch reproduces TrackML exactly is inferred from the source and the measured symmetries. It
   has not been built or dumped. G0 runs before addendum H is committed (section 9).
3. Regenerate the endcap id table from the patched dump. The present tables, `geometry/out/module_map.csv` and
   `hit_format/out/module_map_acts_v15_to_trackml.csv`, agree on all modules (`plan/map_agreement.txt`) but hold for
   the unpatched build only. The order of ACTS sensitive ids on the patched -z side is not known yet.
4. Barrel ids pass unchanged, and endcap ids go through the regenerated table. The digitiser takes every module frame
   from the TrackML `detectors.csv`.

After the patch the -z strip endcaps differ from those of the Zenodo samples and of the generic samples of spec v1.
This does not affect the frozen TrackML model, which was trained on the TrackML layout. The no-pileup condition of
this version is C-hs (section 8), made from the same simulated gun hits, so no separate no-pileup sample on the
patched detector is needed.

## 4. Field, material and simulation settings

### 4.1 Field

TrackML used "an inhomogeneous magnetic field similar to the one in the ATLAS experiment" (arXiv:1904.06778, Sec.
2.2) and did not publish it (Sec. 5.2.2). The field_material study measured it from truth. It used the transverse
momentum kick between consecutive hits of 1.885 M segments in the calib training events (non-electrons, pT > 1 GeV).
An axisymmetric, curl-free model of 67 current loops outside the tracker fits as well as a free grid of 2,898 nodes
(`field_material/loops_fit.json`, `model_comparison_600k_kept_segments`; job 6812150).

**The field is not symmetric in z.** The loop fit has no symmetry constraint, and the data place the field centre off
the origin. Bz on the axis is symmetric about z0 = -0.15 m. About that point the two sides differ by at most 0.008 T
on the axis. About the origin they differ by up to 0.25 T (`plan_revision/field_centre_scan.txt`, within 2.5 m of the
centre). The table gives the delivered map on both sides.

| Delivered map (`field_material/trackml_like_rz_map_v1.txt`, job 6812219) [T] | z = 0 | \|z\| = 0.5 m | 1.0 m | 1.5 m | 2.0 m | 2.5 m | 3.0 m |
|---|---|---|---|---|---|---|---|
| Bz on the axis, +z | 1.993 | 1.964 | 1.888 | 1.737 | 1.463 | 1.064 | 0.715 |
| Bz on the axis, -z | 1.993 | 1.985 | 1.939 | 1.838 | 1.642 | 1.310 | 0.761 |
| Bz at r = 1 m, +z | 2.018 | 2.015 | 1.941 | 1.872 | 1.636 | 1.132 | - |
| Bz at r = 1 m, -z | 2.018 | 2.032 | 1.970 | 1.940 | 1.770 | 1.547 | - |
| Br at r = 1 m, +z | 0.007 | 0.042 | 0.085 | 0.170 | 0.359 | 0.548 | - |
| Br at r = 1 m, -z | 0.007 | -0.021 | -0.051 | -0.117 | -0.227 | -0.516 | - |

The corner of the table at large r and |z| lies beyond the data and is not quoted (risk 3).

A model-free check gives the same sign and a similar size. Origin-pointing tracks bend in a weaker field on the +z side
(`review_physics/beff_sign.json`, job 6812558: truth segments of the calib events, primaries with pT > 2 GeV, no
electrons). A z-symmetric field would give equal bending on the two sides.

| \|z\| band [m] | 0-0.5 | 0.5-1.0 | 1.0-1.5 | 1.5-2.0 | 2.0-2.5 |
|---|---|---|---|---|---|
| bending field of pointing tracks, (+z) - (-z) [T] | -0.011 to -0.013 | -0.042 to -0.048 | -0.108 to -0.118 | -0.267 to -0.271 | -0.24 to -0.54 |
| segments per side | 7,000-45,000 | 5,300-12,000 | 2,600-9,800 | 3,300-8,500 | 510-4,300 |

The ideal finite solenoid fitted in job 6812156 (`field_material/solenoid_fit.json`, `ideal_formula`) is centred at
z = 0. It matches only the mean of the two sides, so it is not used to describe the field.

**Stability.** The fit was repeated on the even and odd event halves and with the coil radius moved to 1,400 mm
(`field_material/loops_syst.json`, job 6812227). The table gives the spread over these variants.

| \|z\| band [m] | 0-1 | 1-2 | 2-2.5 | 2.5-3 |
|---|---|---|---|---|
| p95 \|dBz\| [T] | 0.003-0.010 | 0.003-0.010 | 0.012-0.016 | 0.066-0.077 |
| max \|dBz\| [T] | 0.015-0.050 | 0.024-0.070 | 0.054-0.089 | 0.18-0.26 |

Beyond |z| = 2.5 m the data hardly constrain the asymmetry. Replacing one half of the map by the mirror of the other
changes the fit there by less than 10 %, while at 1.5-2 m it triples the width of the bending residual
(`review_physics/field_sym.json`, job 6812468).

| Width of the bending residual [MeV] | -z, map as delivered | -z, +z half mirrored | +z, map as delivered | +z, -z half mirrored |
|---|---|---|---|---|
| \|z\| = 1.0-1.5 m | 1.57 | 2.26 | 1.95 | 2.59 |
| \|z\| = 1.5-2.0 m | 3.26 | 11.1 | 4.45 | 11.5 |
| \|z\| = 2.0-2.5 m | 8.1 | 15.4 | 11.95 | 18.0 |
| \|z\| = 2.5-3.1 m | 11.8 | 12.5 | 11.7 | 10.6 |

- **Setting.** The RZ text map `field_material/trackml_like_rz_map_v1.txt`, read by `ActsExampleFatrasGeneric
  --bf-map-file ... --bf-map-type rz`. A probe of 200 events with 50 particles each had no failures (job 6812220). It
  took as long as a run with a constant field. Its Fatras truth matched the loop field within 0.085 MeV mean bending
  residual in every |z| band (job 6812221). A uniform field missed by up to 64 MeV.
- **Map format.** r must run fastest in the file (global bin = iz * nR + ir). A z-fastest file is read without error
  and scrambled (job 6812216). The map clamps the loop model beyond r = 1,150 mm and |z| = 3,100 mm. An unclamped
  first version produced 3,408 particle failures (job 6812205).
- **Direction.** Bz > 0, as in TrackML, where 98.3 % of segments of positive particles turn clockwise
  (`configs/trackml/calibration_v1.md`).
- **The pipeline's own map is not a simulation field.** `configs/trackml/field_map_v1.json` holds the effective
  bending field of origin-pointing tracks, Bz only, binned in |z| (`absz_edges`). It averages two sides that differ,
  so it is off with opposite signs on the two sides (`plan_revision/frozen_map_sides.txt`). Below |z| = 1.5 m the -z
  side carries most of the bias, up to 0.12 T. At 1.5-2 m either side is off by up to 0.2 T. The two estimators
  differ, so these values are approximate. The map belongs to the frozen model (risk 6).

### 4.2 Fatras settings

| Setting | Value | Reason or source |
|---|---|---|
| Program | `ActsExampleFatrasGeneric` from the patched build, `--input-dir` (particle CSVs), `--output-csv` | takes the map; the v15 Python bindings reject field maps (TypeError, job 6812138) |
| Field | the RZ map of section 4.1 | |
| pMin | `--fatras-pmin-gev 0.1` (decision D6) | TrackML particles with hits start at pT = 0.100 GeV (`field_material/particle_stats.json`, job 6812137); spec v1 uses 0.5 GeV. Fatras cuts on p, TrackML on pT |
| Multiple scattering, ionisation | on | as in spec v1; the material differs from TrackML's (section 4.3) |
| Bremsstrahlung | off (decision D7) | TrackML electrons above 2 GeV lose 0.02-0.07 % per crossing, against about 3 % in v15 (`material.json`, `brem`; job 6812190) |
| Photon conversions | no effect | the gun produces no photons; TrackML has 65 conversions with >= 5 hits per 100 events (job 6812137) |
| Nuclear interactions | not available | present in ActsFatras but not wired into the v15 `FatrasSimulation`; emulated (section 4.4) |
| Decays in flight | absent in v15 Fatras, and absent in TrackML truth | v15 simulates charged particles with `ActsFatras::NoDecay` (`FatrasSimulation.cpp` L84). TrackML truth has no decay products: 20 training events hold 15,629 prompt charged kaons, and only 5 muons are produced at 5-1,000 mm, all from the generator (`plan_revision/decay_check2.json`, job 6812742). With decays simulated, several per cent of these kaons would give a muon inside the tracker, hundreds in all. Central K and pions also reach the last barrel layer equally often (section 4.4). Nothing to emulate |
| Seeds | `seed + Cantor(algorithm, event)` as in spec v1 section 4; base seeds in section 7 | |

### 4.3 Material

The v15 GenericDetector material is set in `BuildGenericDetector.hpp`: a Be beam pipe and support tube, Si modules,
and homogeneous Si layer material. The probes show that not all of the configured layer material acts in the Fatras
runs. This section therefore describes v15 by what the probes measure, and compares it with TrackML per region and
|eta| bin.

W is the width (1.4826 times the median absolute deviation) of beta * p * theta in the polar plane per crossing. dE is
the median ionisation loss per crossing. Both are for pions with p = 0.7-5 GeV (`field_material/material.json`, list
`ms`, rows `trackml_tp` and `acts_const_before`; jobs 6812157, 6812190 and 6812228).

| Region | \|eta\| | W, TrackML [MeV] | W, v15 [MeV] | dE, TrackML [MeV] | dE, v15 [MeV] |
|---|---|---|---|---|---|
| pixel barrel | 0-0.5 | 1.67 | 1.42 | -0.62 | -1.05 |
| | 0.5-1.0 | 1.66 | 1.55 | -0.62 | -1.37 |
| | 1.0-1.5 | 1.66 | 1.98 | -0.62 | -2.03 |
| | 1.5-2.0 | 1.70 | 2.36 | -0.63 | -3.26 |
| | 2.0-2.5 | 1.76 | 2.55 | -0.79 | -4.63 |
| | 2.5-3.0 | 2.01 | 3.56 | -1.11 | -7.41 |
| short-strip barrel | 0-0.5 | 1.82 | 1.50 | -0.84 | -1.39 |
| | 0.5-1.0 | 1.85 | 1.64 | -0.84 | -1.80 |
| | 1.0-1.5 | 2.05 | 1.56 | -0.83 | -2.29 |
| | 1.5-2.0 | 2.47 | 1.44 | -1.08 | -2.02 |
| | 2.0-2.5 | 2.25 | - | -0.95 | - |
| long-strip barrel | 0-0.5 | 2.02 | 1.52 | -0.99 | -1.70 |
| | 0.5-1.0 | 2.19 | 1.84 | -1.01 | -2.18 |
| | 1.0-1.5 | 3.40 | - | -2.68 | - |
| pixel endcap | 1.5-2.0 | 2.27 | - | -1.57 | - |
| | 2.0-2.5 | 1.59 | 0.87 | -0.60 | -0.92 |
| | 2.5-3.0 | 1.45 | 0.52 | -0.52 | -0.11 |
| short-strip endcap | 1.0-1.5 | 3.72 | - | -1.09 | - |
| | 1.5-2.0 | 4.62 | 0.63 | -0.92 | -0.17 |
| | 2.0-2.5 | 6.24 | 0.80 | -0.97 | -0.77 |
| | 2.5-3.0 | 9.48 | 0.74 | -1.00 | -1.00 |
| long-strip endcap | 1.0-1.5 | 4.50 | - | -2.62 | - |
| | 1.5-2.0 | 11.6 | 1.11 | -2.42 | -1.75 |

A dash marks a bin without v15 crossings in the probe.

- **Pixel barrel.** In TrackML, W and dE per crossing stay flat in |eta|, although the path through a layer grows with
  it. In v15 both grow with the path, as expected. Above |eta| = 1, v15 scatters 1.2-1.5 times more than TrackML. It
  also loses 3-6 times more energy. Here the gun tracks are less straight than native ones.
- **Short-strip barrel.** TrackML scatters more than v15. The ratio of the widths grows with |eta| and reaches 1.7 at
  |eta| = 1.5-2.0. This barrel reaches |z| = 1,084 mm (`detectors.csv`, volume 13), so most of this mismatch lies inside
  |z| < 1 m. It holds 69 % of the barrel crossings of gun targets (`review_physics/gun_incidence.json`, job 6812557).
- **Strip endcaps.** The TrackML width rises with |eta|, while the path through a disk falls towards the disk
  thickness. v15 stays flat and far lower. The excess is the same with and without field subtraction and at every pT
  (`field_material/ms_diag.json`, job 6812190), so it is not a field effect. Truth cannot tell whether it is material
  or a feature of TrackML's scattering model.
- **Gun angles.** Gun crossings in the barrel are not more inclined than prompt ones (median path factor 1.24 against
  1.40, job 6812557). The mismatch therefore comes from the |eta| dependence of the material effects.

The last-hit position closure shows the combined effect (`field_material/closure_material.json`, job 6812228). It is
the median |r-phi| distance between the true last hit and a material-free propagation in the right field.

| Pions and muons, pT 1-2 GeV, \|eta\| | 0-1 | 1-1.5 | 1.5-2 | 2-2.5 | 2.5-3 |
|---|---|---|---|---|---|
| TrackML [mm] | 2.00 | 1.60 | 2.47 | 2.37 | 1.80 |
| v15 with the map [mm] | 2.28 | 2.66 | 2.45 | 1.41 | 0.88 |

**Configured against acting material.** The table compares the endcap Si configured in `BuildGenericDetector.hpp`
(L213-224, L434-446, L631-641) with a Highland inversion of the measured v15 widths (`material.json`,
`t_normal_incidence_pct`, rows `acts_const_before` and `acts_const_after`).

| Endcap disks | configured Si per crossing (module + layer) | configured, radiation lengths | Highland inversion of v15 W |
|---|---|---|---|
| pixel | 0.15 + 1.5 mm | 1.7 % | 0.24-0.62 % |
| short strip | 0.25 + 2.5 mm | 2.9 % | 0.33-0.56 % |
| long strip | 0.35 + 3.5 mm | 4.0 % | 0.75-0.91 % |

The measured values are 12-36 % of the configured ones. In the outer pixel-endcap bin a crossing loses only
-0.11 MeV, a tenth of a central pixel-barrel crossing with the same configured Si. The next bin loses -0.92 MeV, so
the deficit is not uniform. Part of the configured endcap layer
material therefore does not act in the Fatras runs. The cause was not established. The Highland inversion is not
calibrated on v15 either: in the central pixel barrel it returns 1.4-1.5 % for a configured 1.72 %. This plan
therefore quotes W and dE and does not convert them to radiation lengths.

**Consequences.**
- No change of a homogeneous layer thickness in v15 can match TrackML in every region. A thicker layer raises W in
  every |eta| bin of that layer. TrackML would need less material at large |eta| in the pixel barrel and more in the
  strip endcaps. A material map binned in z or r, or a per-crossing change of the Fatras scattering, might do it.
  Neither was tried, and an edit of the endcap layer material may not act at all (decision D5).
- **Proposal.** Keep the v15 material. The truth gate M1 measures the mismatch per region and |eta| bin on the
  patched build, and V1 measures its effect on the model (section 10). It is reported as a systematic.

### 4.4 Emulation after simulation

| Effect | TrackML training events | v15 | Emulation |
|---|---|---|---|
| Hit inefficiency, barrel | 0.78-0.93 % holes per interior layer (\|eta\| < 0.8, pT > 2 GeV) | no holes | drop each gun hit with the TrackML rate of its layer |
| Hit inefficiency, pixel disks | 1.00-1.35 % holes per interior disk (1.5 < \|eta\| < 2.5) | 0.57 % per interior disk, from the gap between the two rings | drop with the excess over the v15 geometric rate (about 0.3-0.8 %), both measured with the same code on the patched build |
| Hit inefficiency, strip disks | not measured | not measured | as for the pixel disks, after measuring both |
| Hadronic stops | 88.9 % of central pions with pT > 2 GeV reach the last barrel layer (K 88.8 %, p 88.0 %, mu 100 %); per-layer stop probability 0.6-2.6 % for pions | 100 % reach it | truncate gun hadron tracks with the per-layer hazards, measured in bins of p; endcap hazards measured first |
| Noise | 16.07 % of hits | - | none added; the host has its own |
| Pixel charge | 5-95 % spread of +-5 % (barrel) to +-15 % (endcap) around the path length | deterministic | multiply by a factor resampled from TrackML's normal-incidence ratio (barrel 0.909-1.087, endcap 0.851-1.149) |
| Hit resolution | pixel residual MAD u / v: 3.9 / 3.9 um (barrel), 17.2 / 9.4 um (endcap) | prototype digitiser 3.1 / 1.8 um and 5.0 / 2.5 um | none yet; if gate C1 fails, a Gaussian smear per volume and incidence bin, fitted on training events |

**The geometric part of the pixel-disk holes.** Each TrackML pixel disk has two rings, at r = 30-102 mm and
104-176 mm (`detectors.csv`, volumes 7 and 9). The radial gap between them makes holes without any inefficiency. The
v15 samples, whose digitisation has no inefficiency, show this rate on the same disks. The barrel shows none. Dropping
gun hits at the full TrackML rate would count the geometric part twice. The emulation gate therefore compares the
total hole rate, geometry plus dropping, with TrackML (section 10).

Sources:
- TrackML holes and hazards: `field_material/track_stats.json` (job 6812189). It used 700 training events of part02
  (22000-22750).
- v15 holes: `review_physics/acts_holes.json` and the log of job 6812530 (600 `generic_iso_v1` train events, the hole
  definition of `track_stats.py`).
- Charge spread: `hit_format/out/tables.json` (`charge_over_path_mm`, calib events).
- Residuals: `hit_format/out/resid.json` (calib events) and `pydigi.json` (`resid_uv_mm`, job 6812142).
- Noise share: `field_material/particle_stats.json` (job 6812137).

The emulation version is fixed by a hash at the end of H-T0.5 (section 9).

## 5. Hit record and features

### 5.1 What the frozen pipeline reads

From the frozen code (`src/hnl/trackml/`; hit_format jobs 6812130 and 6812131):
- **Embedding (X1).** r, phi, z and the logs of ncells, charge, span0 and span1. NaN becomes 0, and
  `cluster_feature` raises if ncells or a span is below 1.
- **Graph, support and edge classifier.** r, phi and z, the volume (for the resolution of `res_v1.json`), layer_id,
  the region one-hot and the B_eff map for the turning-pT feature.
- **Track building.** Module phi_c from the TrackML `detectors.csv`. An unknown (volume, layer, module) raises a
  KeyError.
- **Converter checks** (`convert.event_tables`):
  - hit ids run from 1 to N, with one truth row per hit and nhits equal to the truth count;
  - particle ids are unique and positive, and every charge is +-1;
  - a particle has at most 254 hits;
  - every hit lies within 50 mm of its module's bounds.
  The particle-id check runs on the CSV dtype, which pandas silently turns into uint64 for large ids (section 5.3).
- **Evaluation.** The particle id of each hit, the particle's production vertex, momentum, charge and nhits, and the
  true hit points for the hit ranks. The weights serve the TrackML score only.

### 5.2 Digitisation

The production digitiser grows from the prototype `hit_format/analysis.py cmd_pydigi` (job 6812142).
- Each sim hit is placed in the frame of its TrackML module. It is stepped through a slab of half-thickness
  module_t / 2 in the barrel and module_t in the endcaps. These match the effective thicknesses measured in TrackML,
  which are about the module thickness in the barrel and twice it in the endcaps (hit_format job 6812142).
- The drift is tilted by the GenericDetector Lorentz angle towards the readout side (pixels -1, strips +1). Cells sit
  on the TrackML pitch grid of the table below.
- Pixel cell value = path length in mm times the charge factor of section 4.4. Strip cell value = 1.

| Sensor | pitch u x v |
|---|---|
| pixel | 50 um x 56.25 um |
| short strip | 80 um x 1.2 mm |
| long strip | 120 um x 10.8 mm (barrel), 10.4 mm (endcap) |

- Hit position = value-weighted centroid of the cell centres plus the Lorentz shift, rounded to 6 significant digits
  as in the TrackML CSV.
- Every module crossing becomes its own hit, including a second hit on one module. nhits is counted after
  digitisation. In the prototype 14 of 92,240 sim hits produced no cell.
- With the patched geometry the sim hits lie on the TrackML planes already, so no 3 mm move is needed. The digitiser
  still checks containment (section 10).

**Validation of the prototype.** It reproduces the effective thickness within 0.04 mm in every
volume, the pixel-endcap charge times cos(incidence), and the mean u bias within 0.4 um (`pydigi.json`). It takes
about 0.15 ms per hit on one core. It does not reproduce the hit resolution: its residuals are narrower than
TrackML's, most of all in the pixel endcaps.

| Volume | TrackML MAD u / v [um] | prototype MAD u / v [um] |
|---|---|---|
| 7, pixel endcap -z | 17.2 / 9.4 | 5.0 / 2.5 |
| 8, pixel barrel | 3.9 / 3.9 | 3.1 / 1.8 |
| 9, pixel endcap +z | 17.2 / 9.4 | 4.1 / 2.0 |
| 13, short-strip barrel | 17.0 / - | 16.3 / - |
| 17, long-strip barrel | 25.8 / - | 23.8 / - |

Sources: `hit_format/out/resid.json` (`trackml`, calib events) and `pydigi.json` (`resid_uv_mm`). The strip v
residual spans the strip length and is not compared. The prototype ran on a muon gun whose angles differ from TrackML's,
and the cause of the gap was not checked. Gate C1 compares the widths in matched incidence bins before any model run.

The features follow each particle's true local direction. In pixel endcap 7, displaced targets with pT above 1 GeV
have larger clusters than prompt ones. Their mean ncells is 2.41 against 1.53 (`hit_format/out/tables.json`,
`prompt_vs_displaced`).

### 5.3 Particle ids, weights and hit order

- **particle_id** in the TrackML layout, with the gun values of the table below. The gun of spec v1 uses the same
  1000 + LLP index for its secondary vertices. The current gun barcode would read as hard-scatter, with generation
  equal to the particle index, so it is re-encoded.

| Field | Bits | Gun value |
|---|---|---|
| vertex | 52-63 | 1000 + LLP index (1001-1010) |
| primary | 36-51 | ACTS particle index |
| generation | 24-35 | ACTS generation |
| secondary | 12-23 | ACTS sub-particle index |
| process | 0-11 | 0 |

- **Why not 4000 + LLP index (design v1).** A vertex value of 2048 or more sets the int64 sign bit. The converter asks
  pandas for int64, gets uint64 without a warning, and passes its `particle_id <= 0` check (`convert.py` L138,
  L377-378). It then stores the ids with `astype(np.int64)` (L468), so every gun id would be negative in the tables. A
  signed decode of vertex 4001 then gives -95 (login-node test, pandas 3.0.1 in the project env). Vertex values
  1001-1010 keep every id positive.
- **No collision with host vertices.** TrackML vertex fields reach 224 (`hit_format/out/tables.json`, `particle_ids`)
  and 225 (job 6812531, 20 training events of part02).
- **Particle rows.** Charged particles only, since the converter needs q = +-1. Production vertex and momentum at
  production; nhits = hits written. Rows with nhits = 0 are allowed, as in TrackML (13.9 % of rows, job 6812531).
- **Weights.** 0 for every gun hit. The host weights still sum to 1, so the TrackML score keeps its definition. Gun
  performance is reported as efficiency. A gun-inclusive score would be a new metric and not comparable to published
  scores.
- **Hit order.** Gun rows are appended after the host rows, and row numbers continue from the host's. Host rows keep
  their indices, which keeps the comparison with stored val outputs simple (gate V2). TrackML sorts hit ids by
  module, and appending breaks that order. The frozen code uses the order only to break ties (hit_format job
  6812131).
- **Gun flag.** The sidecar lists every gun particle, so that the K_S / Lambda mass-hypothesis tagging never labels
  gun pairs.

### 5.4 Files, event ids and code changes

- **Host tables.** training_part03 is streamed twice over HTTP from the tar and never unpacked, as parts 01 and 02
  were (`external/trackml/README.md`). Its md5 is b0c407c72e2753ed4e1e619c6c030e62 (Zenodo listing,
  `study_design/zenodo_4730157.json`). It is converted to `derived/tp03_v1/{hits,particles,ptype}/training_part03/`,
  one row group per event, with score weights for every event. That is about 1.7 GB in about 9 files.
- **Gun events.** For each overlay event the gun-only CSV members (hits, cells, particles, truth) are written to
  job-local scratch and converted with the frozen `event_tables` and `ptype_table`. The CSVs are kept as one tar per
  set on /work.
- **Overlay tables.** Host rows, followed by the gun rows, in
  `derived/overlay_v1/{hits,particles,ptype}/<set>/shard_*.parquet`, one row group per event. Closure events lack the
  removed native rows. These are full copies of about 2.1 MB per event. Deltas joined at read time would save space
  but add join code to the read path. With copies, the H-T1 checksums cover exactly what the model reads.
- **Event ids.** `overlay id = 50,000,000 + 1,000,000 * k + host id`, with the set index k of the table below.
  TrackML ids are below 100,000, and part03's are known after streaming. The overlay ids stay below 10^8 and are
  passed as explicit lists to `config.event_set`.

| Set | H-val | closure | H-test | H-train (H3 only) | pilots |
|---|---|---|---|---|---|
| set index k | 0 | 1 | 2 | 3 | 9 |
| base seed (section 7) | 650001 | 950001 | 750001 | 850001 | - |

- **Code changes** (committed, small). `events.table_root` sends ids >= 50,000,000 to `derived/overlay_v1` (one line). A
  C-hs truth filter sits next to C-1GeV and C-nonoise. `collision_reuse` gets an argument for its signature file, and a
  truth-only signature step is added for part03 (section 6). The digitiser and overlay builder are new modules. No model
  code changes.
- **Sidecar** per set. It holds the map from overlay id to host id, gun event and seed, the host-to-overlay row map,
  the removed rows (closure), the gun truth (LLP, r_v, prongs, mass, boost, decay vertex, PDG) and the emulation
  hash.

## 6. Host events and hygiene

| Set | Events | Use by the frozen Canto-L (TrackML) | Role here |
|---|---|---|---|
| tp_v1 train: 1,400 ids, dev included, from part01 outside val and test, plus part02 | 1,399 stored (event 22500 has an empty cells member) | all training stages; the calib events gave the field map and the resolution | no evaluation; H-train for H3 only; emulation inputs (truth only) |
| tp_v1 val | 150 (valmon = the first 50) | checkpoint choice (valmon), working points (val) | H-val and closure: development and gates V1-V3; no claim |
| tp_v1 test | 300 | read once for the model card | excluded |
| Accuracy phase (train_1), the cross-phase test set T2 | - | read once for the model card | excluded |
| training_part03 | about 896 (6,932,060,160 bytes at 7.74 MB per event) | never streamed | H-test hosts |
| training_part04-09 | about 6,000 | never streamed | not used (part04 is the reserve for a later H3, decision D4) |

- **Split file.** `configs/trackml/split_tp_v1.json` does not cover part03. Its rule text reads "train = every other
  event of the nine parts", but its `scope` field limits the rule to the two streamed parts. At H-T0 a new file,
  `configs/trackml/split_h_v1.json`, records this, the hash rule below and the H-val and closure sets. At H-T1 it
  receives the H-test list and the spares.
- **Hash rule.** Every complete part03 event that passes the converter's data-quality rule is ordered by
  `int(sha256("canto-trackml-overlay-v1:<id>")[:8], 16)`. The first 600 are H-test. The rest are spares. No model
  reads them, and they replace dropped hosts in hash order at H-T1.
- **Collision reuse.** Throughput events reuse generator-level collisions. On val, 76.4 % of the collisions have a twin
  in the training events (`results/trackml_collision_reuse_v1.txt`). Only 10 of the 150 hard scatters have none. On val
  gen-0 targets the twins changed nothing. Part03 is probably similar, so H-test hosts and the host hard scatter kept in
  C-hs are not unseen at the collision level. At H-T1, part03 signatures are computed from truth with the definition of
  the gnn-displaced signatures (one hash per vertex of the generation-0 particles' type, charge and momentum). They are
  written under `external/trackml/derived/tp03_v1/`, outside the read-only gnn-displaced tree, and matched against the
  training signatures by the twin count of `hnl.trackml.collision_reuse`. Whether the gnn-displaced signature code runs
  on a new part was not checked. The twin fractions are committed, and host-integrity and C-hs numbers are reported
  split by twin status. The gun particles are new, so the gun targets are unaffected.
- **Emulation inputs.** The field map, resolution, charge factors and material come from calib training events. The
  inefficiency and hadronic hazards come from 700 part02 training events (section 4.4). Nothing comes from test,
  T2 or H-test events.
- **Gun origin.** The hard-scatter vertex of each host (particles with vertex field 1) is read from host truth at
  H-T1. Vertex 1 is the hard scatter in all 20 events checked (job 6812531). This is a truth-only read.

## 7. Gun content

Per LLP the content is that of spec v1 section 3: species, masses, boosts, the eta range, decay radii up to 800 mm and
|z_v| < 2,900 mm. The TrackML barrel radii and disk positions equal the GenericDetector's (`docs/trackml_feasibility.md`
section 1), so no range changes. The changes come from the study_design jobs 6812128 and 6812140.

1. **No Pythia.** The LLPs start at the host's hard-scatter vertex.
2. **8 isolated and 2 dense LLPs per host.** Azimuths are drawn in sequence so that LLP directions are at least
   Delta R = 0.8 apart. This costs 0.88 redraws per LLP. Then 0.16 % of isolated targets have another LLP's target
   within Delta R < 0.05, as in G's two-LLP layout (`yield_overlap_train.json`, `overlap_mc`). The gun adds about 126
   hits per host (`study_design/gun_hits.json`), about 0.1 % of the hits of a host event.
3. **Barcodes** as in section 5.3, with score weight 0.
4. **Seeds.** One base seed per set (table in section 5.4). Fatras and digitisation use base + 13, as in spec v1
   section 4.
5. **pMin 0.1 GeV** (decision D6). At generator level 5.3-5.5 % of gun daughters have p between the new cut and
   0.5 GeV (`plan/pmin_check.txt`). That is about 1.5 soft charged particles per host, none of which can be a target.
6. **r_v up to 800 mm is kept.** LLPs that decay beyond 500 mm are 32 % of the isolated LLPs. They give 4.4 % of the
   P_P targets (`study_design/yield_bins_train.csv`). Keeping them keeps the content equal to G's.

**Expected yields** (generic train split, truth only; `yield_overlap_train.json`, `yields_train`).

| Quantity | Per LLP | 600 hosts (H-test) | 150 hosts (H-val) |
|---|---|---|---|
| P_P targets (isolated LLPs) | 0.603 | about 2,890 | about 720 |
| P_D targets (dense LLPs) | 1.687 | about 2,020 | about 510 |

These yields come from the 2 T HNL setup without inefficiency or hadronic stops. The real counts are fixed at H-T1.
Of the isolated targets, 52 % have a hit beyond |z| = 1 m (`field_exposure_train`). About a third reach beyond 2 m.

## 8. Models, populations and conditions

- **Model.** The frozen Canto-L (TrackML), `runs/canto_l_v1/seed0_x1`. WP-A is primary: Mutex Watershed with learned
  merging at tau_merge = 0.3, chosen on val at FR* = 0.029368. WP-S and WP-R-P3 come from the same networks. Edge
  scores use the memory-bounded forward (`hnl.trackml.chunked`, about 10-13 GB). It gives the same scores as the
  one-shot forward (model card, section 6).
- **No other model.** `canto_l_v1/seed0` (without X1) has no edge classifier. acorn is not trained on TrackML in this
  project (DECISIONS 2026-10-05). The HNL-benchmark models were not built for mu = 200. In a zero-shot test, the
  ttbar-trained graph stage of Canto-B seed 0 built 17.8 M pairs per event on accuracy-phase events. Its classifier
  kept 4-7 % of the true edges at its HNL cut. This was measured in four narrow phi wedges of accuracy-phase event
  1000 (`docs/trackml_feasibility.md` section 3). acorn (HNL), Canto-H and Canto-L were not measured there.
- **Targets** as in the model card: pT > 1 GeV and |eta| < 4. A target needs at least 5 hits, and matching is
  double-majority.
- **P_P.** Isolated gun targets with truth |d0| >= 10 mm or r_v >= 100 mm. This is the definition of P_G in
  addendum G section 4. d0 is the converter's: the helix through the production vertex with |B| = 2 T. **P_D**
  applies the same cuts to the dense component.
- **Metrics.** DM efficiency at WP-A, the event fake rate (tracks with at least 5 hits), fake tracks containing a gun
  hit per LLP, and the duplicate rate. The per-target rows of `evaluate.event_counts` give gun efficiencies without code
  changes.
- **Conditions.** C-full keeps all hits and is primary. C-1GeV is truth-filtered, as in the model card. C-hs (new) keeps
  the host hard-scatter hits and the gun hits, without pileup and noise. C-hs lies outside the training density and is
  reported only. C-nonoise: optional, val only.
- **Bins.** Those of addendum G section 7: |d0|, r_v, r_first, |d0|/r_first, pT and dense Delta R neighbours. Also
  signed eta, species and local pileup hit density. Signed eta matters because the field and the pipeline's map
  differ between the two sides (section 4.1). The loss stage (graph, scores, building) comes from
  `hnl.trackml.displaced_diag`, also split by the sign of eta.

## 9. Addendum H (proposed)

We propose to preregister an addendum H before part03 is streamed, with one claim (decision D3). The sample is made in
this project and has free choices: emulation, LLPs per host, hosts. These should be fixed before any H-test event
exists. The skeleton mirrors addendum G.

- **Section 1, purpose.** The pileup version of the generic gun (supervisor, 2026-10-14).
- **Section 2, sample.** This document plus a spec v2 section. The emulation hash is fixed at the end of H-T0.5.
- **Section 3, splits.** H-val = tp_v1 val. Closure = tp_v1 val with native targets simulated again (gate V1). H-test
  = the first 600 part03 events by the hash rule. All are recorded in `configs/trackml/split_h_v1.json`.
- **Section 4, population and metric.** As in section 8 here.
- **Section 5, models.** The frozen Canto-L (TrackML). Canto-L (TrackML+H) only if decision D4 places H3 in H.
- **Section 6, claim H1.** Delta = eps(P_P, C-full) - eps(P_P, C-1GeV) on the same targets, frozen Canto-L, WP-A,
  H-test. sigma comes from a paired event bootstrap (B = 500, seed 12345). There is no seed term, since one model
  appears in both conditions. The margin is delta = 1.0 point (decision D13). H1 holds if Delta - 2 sigma > -delta. It
  does not hold, resolved below the margin, if Delta + 2 sigma < -delta, and is inconclusive otherwise.
- **Section 7, reported without a claim.** eps(P_P), eps(P_D) and all gun targets in C-full, C-1GeV and C-hs at WP-A,
  WP-S and WP-R-P3, with the bins and loss stages. Host prompt efficiency, fake rate and TrackML score on H-test,
  beside the model card values. G beside H for the same gun content. The detector and model differ, so this is no
  comparison.
- **Section 8, why delta = 1.0 point.** See below.
- **Section 9, order of work.** See below.

**Why delta = 1.0 point.**
- **A physics reference.** Training acorn on the signal gains (1.48 +- 0.35) points on HNL validation (addendum G
  section 8). A pileup loss inside delta is smaller than the effect of the training choice that the paper studies on
  samples without pileup.
- **Context on val.** Between C-1GeV and C-full the frozen model loses 0.11 points of all val targets (model card,
  section 5.1). The margin asks whether gun targets lose less than about nine times that.
- **Set before the matched projection.** Design v1 proposed 1.0 point before the P_P-matched val row was examined.
  The value is not moved after that projection, and it is committed at H-T0.
- **Uncertainty.** The projected sigma is 0.17-0.31 points for the three val populations closest to P_P
  (`plan_revision/sigma_projection.json`). It scales the val bootstrap sigma to the expected count of P_P targets.

**Val projection, to be stated in the addendum.** The stored val outputs (job 6812139,
`study_design/val_discordance.json`) give the paired loss of native targets between C-1GeV and C-full. The row whose
definition matches P_P is the primary projection.

| Native val targets | n | Delta [points] | sigma, val | sigma, projected | Delta - 2 sigma, projected | holds / inconclusive / fails |
|---|---|---|---|---|---|---|
| gen-0, \|d0\| >= 10 mm or r_v >= 100 mm (matches P_P) | 3,276 | -0.61 | 0.16 | 0.17 | -0.95 | 0.59 / 0.41 / 0.00 |
| gen-0, \|d0\| >= 10 mm | 1,562 | -0.70 | 0.25 | 0.19 | -1.08 | 0.41 / 0.58 / 0.02 |
| all, \|d0\| >= 10 mm | 2,010 | -1.39 | 0.37 | 0.31 | -2.01 | 0.02 / 0.66 / 0.32 |
| gen >= 1, \|d0\| >= 10 mm | 448 | -3.79 | 1.37 | 0.54 | -4.87 | 0.00 / 0.12 / 0.88 |

The projected sigma scales the val sigma to about 2,890 P_P targets. The outcome odds take the H-test Delta as normal
around the val Delta, with the val and projected variances added (`plan_revision/sigma_projection.py`). They assume
that gun targets behave like native ones. At the matched value H1 sits at the margin: it holds or is inconclusive
with similar odds. The rows with all displaced targets are dominated by secondaries from the detector simulation
(gen >= 1), found with 46 % efficiency. These do not resemble gun targets and are context only.

**Order of work.** The steps carry the prefix H- so that they are not confused with the cross-phase test set T2.
- **H-P (before the commit; truth only).** Build the patched ACTS and dump its geometry. Run G0, F1, containment and
  M1 (section 10). The outcome and the chosen route go into the addendum. If G0 fails, decision D1 is taken again
  before H-T0.
- **H-T0.** Commit addendum H, a spec v2 section and `split_h_v1.json`: gun changes, seeds, hash rule, formats,
  budget, gate thresholds, the margin, the H3 decision (D4) and the H-T0.5 rules below. Nothing is streamed from
  part03 and no overlay is made.
- **H-T0.5 (val).** Produce the H-val and closure overlays and run the frozen model on them in C-full. The rules:
  1. Emulation and digitiser changes are decided only on the truth gates and on V1, V2 and V3. Each change is logged.
  2. The model's results on the gun targets of H-val stay blinded until the emulation hash is committed. The
     evaluation at this step skips them in every condition, and nobody inspects tracks of gun particles. Truth-level
     features of gun hits (V3) are not blinded.
  3. At most three passes, each with one emulation version through all val gates. If V1 still fails after the third
     pass, H becomes descriptive, decided before H-T1.
  4. Once the hash is committed, H-val runs in C-1GeV and C-hs, and the gun results on H-val are computed once. They
     are reported as the val estimate of H1 and change nothing.
- **H-T1 (truth only).** Stream and convert part03, fix the host list, build the H-test overlay, and run the
  collision-reuse check. Commit checksums, dropped events, twin fractions and the P_P and P_D counts. Fewer than
  1,000 P_P targets means "underpowered", as in G.
- **H-T2.** One run of the frozen model on H-test (C-full, C-1GeV, C-hs), which decides H1.
- **H-T3 / H-T4.** Only if H3 is part of H (decision D4).

**H3, a retraining arm (decision D4).** Canto-L (TrackML+H) would retrain every learned stage on H-train (tp_v1
train with gun overlays) with the frozen recipe, select on H-val and run once on H-test. It would hold if Delta >= 3
sigma and the new model's fake rate is not higher (point estimates). The working point would follow G's rule on H-val
(FR* = 2.9368 %, one-standard-error ties), the rule used for WP-A.
- **When it is decided.** At H-T0, before any H-test event exists, as G's retraining arm was approved before its T2
  (DECISIONS 2026-10-14, "Addendum G T3 approved"). If H-T0 passes without a decision, H3 leaves addendum H. It could
  then be preregistered later on fresh hosts, for example hash-ordered training_part04 events.
- **Seed term.** With one seed per model, `sigma^2 = sigma_boot^2 + 2 x (0.2 point)^2`. The 0.2-point spread is
  assumed: it is the three-seed spread of Canto-L (ttbar) on HNL validation (addendum G section 8). The TrackML
  Canto-L has one seed and no measured spread (model card, section 4). H3 would also be reported with a 0.4-point
  seed term. Three seeds per model measure the term instead.

## 10. Checks before any model run

All thresholds below are fixed at H-T0. Checks marked descriptive have no pass rule and are reported as tables.

**Before H-T0 (truth only, step H-P).**
- **G0, geometry.** The patched `detectors.csv` against TrackML's: centres, planes and corners within 4 um for every
  module, and a bijective id table. The largest barrel difference now is 3.8 um. If G0 fails, decision D1 is taken
  again: a fix of the patch or route C.
- **F1, field.** A probe in the patched build with the map: no particle failures, and a mean bending residual against
  the loop field within 0.1 MeV in every |z| band, separately for +z and -z. The earlier probe reached 0.085 MeV.
- **Containment.** All gun hits lie on their TrackML module plane and inside its bounds. Every (volume, layer,
  module) is present in TrackML's `detectors.csv`.
- **M1, material (descriptive).** W and dE per crossing of the patched build against TrackML, per region and |eta|
  bin as in section 4.3, plus the last-hit closure. It documents the material systematic.

**Before H-T0.5 (truth only).**
- **C1, cells and resolution.** Pions and muons simulated over the TrackML angular range, compared per volume with
  TrackML training hits in bins of local incidence. Bins with at least 200 hits on each side count. Pass rules:
  - mean span0 and span1 within 10 % of TrackML;
  - effective thickness within 0.04 mm;
  - pixel charge times cos(incidence) within 5 % of TrackML's, in the barrel and in the endcaps;
  - mean u bias within 1 um;
  - residual MAD within 20 % of TrackML, in u and v for the pixels and in u for the strips.
  A failure of the last rule is fixed by the smear of section 4.4, which counts as an emulation change.
- **Converter.** Every gun-only event passes `event_tables`. In the stored tables, read back as int64, every particle id
  is unique and positive. Every gun vertex field lies in 1001-1010, and gun weights are 0. For H-val and H-test events
  the host weights sum to one within 1e-6. For closure events the sum equals one minus the weights of the removed native
  hits, with the same tolerance.
- **Emulation.** The total hole rate per barrel layer and per disk (geometry plus dropping) lies within 0.2
  percentage points of the TrackML rate. The fraction of pions, kaons and protons reaching the last barrel layer lies
  within 1 point of TrackML.
- **Content (descriptive).** Hits per LLP, P_P yield per LLP and species shares against the generic train split,
  gun-gun Delta R, and gun-host hit pairs on one module.

**Val gates (H-T0.5; the frozen model runs on val events only).**
- **V1, closure.** Native val targets lose their hits and are simulated again by the overlay chain from their true
  vertex and momentum. Their efficiency is compared, target by target, with the stored val outputs (C-full, WP-A).
  The populations and counts come from `plan_revision/val_regions.json` (job 6812736). "Forward" means a hit at
  |z| > 1 m, split by the sign of eta.

| V1 population | Central | Forward +z | Forward -z | All | Pass rule on the paired Delta |
|---|---|---|---|---|---|
| (a) gen-0, \|d0\| >= 10 mm or r_v >= 100 mm (the P_P definition) | 1,250 | 1,050 | 976 | 3,276 | \|Delta\| <= 1.0 point in all; <= 1.5 points in each region |
| (b) gen >= 1, \|d0\| >= 10 mm | 44 | 213 | 191 | 448 | \|Delta\| <= 3 points in all |
| (c) prompt subset: 3,000 gen-0 targets with \|d0\| < 1 mm and r_v < 1 mm | - | - | - | 3,000 | \|Delta\| <= max(0.2 points, 2 sigma_paired) |

  The prompt subset is the first 3,000 of the 153,603 prompt val targets in the order of
  `int(sha256("canto-trackml-overlay-v1-prompt:<event>:<particle_id>")[:8], 16)`. V1 is also reported in |eta| bins
  and by detector region (pixel, short-strip and long-strip barrel and endcaps), without a pass rule.
  - **Fallback (decision D12, fixed at H-T0).** If (a) passes in all but fails in a region, P_P is restricted to the
    regions that pass, by truth flags of the gun targets. The restriction is committed before H-T1. Central targets
    alone are 48 % of P_P, about 1,400 targets on H-test.
  - H1 is a difference within one sample, so an emulation bias common to both conditions cancels at first order. V1
    mainly protects the absolute efficiencies.
- **V2, host integrity.** Overlaid val events against the stored val outputs: prompt efficiency and fake rate within
  0.05 points, TrackML score within 0.001.
- **V3, features (descriptive).** Cluster features and residuals of gun hits against native tracks in incidence bins
  and signed eta, and same-module near-coincidences. C1 carries the pass rules for the same quantities.

## 11. Budget

| Item | CPU-h | GPU-h | Storage | Files | Wall time | Source |
|---|---|---|---|---|---|---|
| H-P: patched build, dump, G0, F1, containment, M1 | about 1 | 0 | about 0.15 GB on /work | - | about 15 min | job 6810588 (10 min on 5 cores); `build-v15` 118 MB, `src-v15` 31 MB |
| Emulation inputs on training events | below 1 (estimate) | 0 | small | - | minutes | estimate |
| Stream and convert part03 (two passes) | 8.4 | 0 | 1.7 GB on /ceph | about 9 | about 0.5-1 h | `study_design/budget.json`, `totals`; scaled from parts 01+02 (jobs 6684878, 6684895) |
| Collision reuse on part03 | not measured, small | 0 | small | 1-2 | minutes | truth tables only |
| Gun simulation (H-val, closure, H-test) | at most 1.6 | 0 | small | - | minutes | `budget.json`, `cpu_sim_upper` |
| Digitisation and conversion of gun events | below 1 | 0 | about 0.1 GB on /work | 3-4 tars | minutes | 0.15 ms per hit (job 6812142) |
| Overlay tables (full copies) | about 5 (estimate) | 0 | about 1.9 GB on /ceph | about 15 | hours | 2.1 MB per event (`external/trackml/README.md`) |
| Model runs H-T0.5 and H-T2, one pass | 730-1,070 (one-shot forward: about 1,350) | 10.4 | 13.9 GB (9.2 GB if event stores are deleted) | 550-600 if packed | 0.5-1 day each | `budget.json`, from sacct 6771374-6771386 and the model card |
| Each extra H-T0.5 pass (at most two) | 235-350 | 3.4 | about 4.6 GB | about 200 (estimate) | 0.5-1 day | `budget.json` rows |
| H3, per seed (only if D4 approves it) | about 680-970, plus the preparation of training events | about 57 | about 10 GB (estimate) | - | days | `plan_revision/h3_budget.json` |

**Totals without H3.** With one H-T0.5 pass: about 750-1,090 CPU-h and 10.4 GPU-h. With three passes: about 1,220-1,790
CPU-h and 17.2 GPU-h. About 17.5 GB on /ceph, the sum of the storage rows, with about 166 GB free. `hnl_lrt` grows to
about 66,600 files, below the 80,000 stop line. About 0.3 GB on /work. Wall time is about 3-5 working days if one pass
suffices and CPUs are free. That includes 2.5-3 days of code (field wiring, hit-format layer, emulation, patched build
and checks).

**When to ask the user.** The TrackML G0 rule applies (`docs/trackml_plan.md` section 8; used again in DECISIONS for
later TrackML stages). It asks before a stage above 75 GPU-h, or before any stage that would take more than 1.5 times
its estimate. The brief's older 50 GPU-h line was lifted by the user on 2026-10-03. No step without H3 comes near
75 GPU-h. If a measured H-T0.5 or H-T2 run implies more than 1.5 times its row above, the user is asked before
it continues. H3 needs about 57 GPU-h for one seed and about 170 GPU-h for three (`h3_budget.json`). Three seeds
exceed 75 GPU-h in total, so H3 needs the user's approval in either form.

## 12. Risks

1. **H1 sits at the margin** (section 9). At the matched val value it holds or is inconclusive with similar odds. The
   projection is stated in advance, and the result is reported in any case.
2. **The geometry patch is not verified.** One build settles it (G0, before H-T0). If it fails, route C remains, with
   5.9-7.0 % of gun hits moved or dropped, reported as a systematic.
3. **Field asymmetry and large |z|.** The field centre sits near z = -0.15 m, and the map follows the data on both
   sides. The map is constrained by data up to |z| of about 2.95 m and r of about 1.03 m, and clamped beyond. At
   |z| = 2.5-3 m the fit variants differ by up to 0.26 T (section 4.1). Mirroring one half there does not worsen the
   fit, so the asymmetry beyond 2.5 m is not measured.
4. **Material.** The v15 mismatch changes sign and size with region and |eta| (section 4.3). In the pixel barrel at
   |eta| > 1, v15 scatters and loses energy more than TrackML. In the short-strip barrel, mostly inside |z| < 1 m, and
   in the strip endcaps it scatters less. A forward fallback alone therefore does not remove it, and no thickness
   change in v15 can. V1 measures it per region and the material is reported as a systematic.
5. **Emulation gaps.** Strip-disk inefficiency, endcap hadronic hazards and hazards below pT = 2 GeV are not measured
   yet. The origin of the pixel charge spread is unknown. The prototype digitiser gives narrower residuals than
   TrackML, and the cause was not checked (C1 decides).
6. **The model's own field map.** The frozen pipeline uses a Bz-only effective field binned in |z| (section 4.1). It
   is off with opposite signs on the two sides, by up to 0.2 T below |z| = 2 m. Beyond 1.5 m Br
   also biases it for displaced tracks. This belongs to the frozen model, and H measures it as part of the model's
   response, split by the sign of eta.
7. **Host count.** The part03 event count is known only after streaming (about 896 expected). The 600 hosts plus
   spares allow for about 30 % of events missing or dropped.
8. **C-hs** lies outside the training density, so its numbers are descriptive.
9. **Queue time.** The model runs need 16-core jobs at 10-13 GB each. On 2026-10-14 the `submit` partition had about
   4,600 pending jobs (jobs 6812099 and 6812117 were cancelled after waiting).
10. **Collision twins.** Part03 hosts probably share collisions with training events (section 6). The gun targets are
   new, but host-integrity and C-hs numbers carry the twin split.

## 13. Decisions for the user and the supervisor

| # | Decision | Recommendation |
|---|---|---|
| D1 | Route: simulate the same gun content again in a TrackML-equivalent setup (patched GenericDetector, TrackML-like field, TrackML-style digitiser) instead of copying the hits of the existing gun sample. The gun content is the same; the simulated events are new | Yes. Copying fails on cells, field, -z endcaps and barcodes (section 2.2). G0 runs before H-T0; if it fails, decide again between fixing the patch and route C |
| D2 | Hosts: stream training_part03 (6.93 GB, never read) and use 600 hash-ordered events as H-test; tp_v1 val for development and closure | Yes. Val-only hosts save the streaming but leave no unread evaluation set. The collision-twin check runs at H-T1 |
| D3 | Addendum H: preregister with H1 as the only claim, or descriptive with fixed estimands | Preregister with H1 and the val projection. A pileup loss outside the margin is a measured result and worth reporting |
| D4 | Retraining arm H3: decide at H-T0. One seed per model (about 57 GPU-h, assumed seed term with a 0.4-point sensitivity) or three seeds (about 170 GPU-h, above the G0 rule's 75 GPU-h in total) | Leave H3 out of addendum H. If it is wanted, preregister it later on hash-ordered part04 hosts, so that H-test is read once |
| D5 | Material: keep the v15 GenericDetector material, or emulate TrackML's | Keep it and report it as a systematic (M1, V1). A thickness change cannot match every region. A binned material map or a per-crossing scattering change is costed only after a probe shows that a changed endcap layer material changes W and dE |
| D6 | pMin: 0.1 GeV (TrackML particle content) or 0.5 GeV (spec v1) | 0.1 GeV. It adds about 1.5 soft gun particles per host and changes no target |
| D7 | Bremsstrahlung for gun electrons: off (as TrackML truth) or on (as spec v1 and the Zenodo HNL sample) | Off. The model was trained on TrackML electrons, which show no brem loss |
| D8 | Emulate hit inefficiency (the excess over the geometric holes), hadronic stops and the pixel charge spread, or leave them out and state the difference. About 6 % of central TrackML tracks have a hole, and 11-12 % of central hadrons stop | Emulate. Under half a day of code. Add the resolution smear only if C1 fails |
| D9 | Overlay tables: a separate root `derived/overlay_v1` with a one-line committed `table_root` change, or a new archive inside the frozen `derived/tp_v1` | Separate root. The frozen data directory stays untouched |
| D10 | Weights of gun hits: 0, or a new gun-inclusive TrackML score | 0. The host score keeps its published meaning |
| D11 | Gun density: 8 isolated + 2 dense LLPs per host with Delta R >= 0.8 | Yes. It gives about 2,900 P_P targets at 600 hosts, with gun-gun overlap equal to G's |
| D12 | V1 fallback: restrict P_P to the detector regions where V1 passes (central, forward +z, forward -z), fixed at H-T0 | Yes |
| D13 | Margin of H1: delta = 1.0 point, with the reasons of section 9, committed at H-T0 | Yes. It was proposed before the matched projection and is not moved after it |

Side findings for the main session, not needed for this version:
- `configs/trackml/field_map_v1.json` is binned in |z| and therefore off with opposite signs on the two sides. It is
  also biased beyond |z| = 1.5 m and for displaced tracks. The (Bz, Br) loop map could replace it in the helix-based
  pieces of a future model. The frozen model keeps its map.
- `acts.SolenoidBField` evaluates the elliptic integrals with k^2 as the modulus, in v15 and on ACTS main as fetched
  on 2026-10-14. It could be reported upstream.
- `hnl.trackml.convert` reads particle ids as int64, which pandas turns into uint64 above the int64 range without a
  warning. The `particle_id <= 0` check then cannot fire, and such ids are stored negative. TrackML ids never reach
  that range, so the frozen tables are unaffected.

## 14. Consistency notes (checked for this plan)

1. **Field values.** `docs/trackml_feasibility.md` section 1 and study_design quote "Bz" values that come from the
   transverse curvature of hit triplets. They are the bending field of origin-pointing tracks, which includes a
   -cot(theta) * Br term. On the axis at |z| = 2.5 m the loop fit gives 1.06 T on +z. On -z it gives 1.31 T
   (section 4.1). The target fractions beyond |z| = 1 m and 2 m are unaffected.
2. **-z strip endcaps.** The hit_format study treats them as a 3 mm sub-plane offset that a straight move fixes. The
   geometry study shows that all 1,608 trapezoids are also inverted. The source lines were confirmed in `src-v15`:
   `ProtoLayerCreatorT.hpp` L489 and L491, `BuildGenericDetector.cpp` L130.
3. **Off-plane fractions.** The geometry study, the hit_format study and the feasibility study report different
   off-plane fractions for volumes 12 and 16. The samples and definitions differ, and with the patch none of them
   applies.
4. **CPU of the model runs.** The hit_format study's estimate for the memory-bounded forward is too low. The model card
   gives 0.33-0.71 CPU-h per event, and staging, held CPU and the grid add 0.32 CPU-h per event (sacct
   6771374-6771386). `budget.json` uses the sum.
5. **Module maps.** `geometry/out/module_map.csv` and `hit_format/out/module_map_acts_v15_to_trackml.csv` agree on all
   18,728 modules (`plan/map_agreement.txt`).
6. **Diverging proposals.** pMin: field_material proposes 0.1 GeV, study_design 0.5 GeV (D6). Hit order: hit_format
   prefers module order, study_design appends (section 5.3 appends). Storage: study_design proposes deltas joined at
   read time, hit_format full CSV events. Section 5.4 uses gun-only CSVs through the frozen converter and full table
   copies.
7. **Table size per event.** The hit_format study estimates 2.6 MB. The stored tp_v1 tables give 2.1 MB per event
   (`external/trackml/README.md`), which this plan uses.
8. **Rules reused from G.** FR* = 0.029368 also chose the TrackML WP-A (model card section 1.2), so H3's rule would
   match the frozen model. P_P uses the P_G definition of addendum G section 4.
9. **Emulation inputs.** Version 1 said that all emulation inputs came from calib events. The inefficiency and hazard
   inputs came from part02 training events (section 4.4).
10. **Endcap id distance.** The 96.8 % of section 3 had no recorded source in version 1. It was recomputed from the
   module map and `detectors.csv` (`plan_revision/same_id_distance.txt`).

## 15. Sources

No study has a written report. Each row lists the outputs and jobs that the numbers above cite.

| Topic | Directory (under `/work/submit/jinboz1/hnl/pileup_design/`) | Main outputs | Jobs |
|---|---|---|---|
| geometry | `geometry/` | `out/modules_summary.json`, `module_map.csv`, `footprint.json`, `npz_check_<sample>.json`, `impact.json` | 6812101, 6812102, 6812170-6812174, 6812183 |
| field and material | `field_material/` | `loops_fit.json`, `loops_syst.json`, `solenoid_fit.json`, `trackml_like_rz_map_v1.txt`, `prop_closure.json`, `material.json`, `ms_diag.json`, `closure_material.json`, `track_stats.json`, `particle_stats.json`, `acts_model_test.json` | 6812125-6812228; one log per job in `logs/` |
| hit format | `hit_format/` | `out/tables.json`, `raw.json`, `acts.json`, `pydigi.json`, `resid.json`, `modmap.json`, `module_map_acts_v15_to_trackml.csv` | 6812130, 6812131, 6812142, 6812144 |
| study design | `study_design/` | `yield_overlap_train.json`, `yield_bins_train.csv`, `val_discordance.json`, `gun_hits.json`, `budget.json`, `zenodo_4730157.json` | 6812128, 6812139, 6812140 |
| physics review | `review_physics/` | `beff_sign.json`, `field_sym.json`, `acts_holes.json`, `gun_incidence.json`; `host_truth.py` (log only) | 6812468, 6812530, 6812531, 6812557, 6812558 |
| version 2 checks | `plan_revision/` | `val_regions.json`, `decay_check.json`, `decay_check2.json` (jobs); `sigma_projection.json`, `h3_budget.json`, `field_centre_scan.txt`, `frozen_map_sides.txt`, `same_id_distance.txt` (login node, under a second each); `style_scan.py` | 6812736, 6812738, 6812742 |
| version 1 plan checks | `plan/` | `pmin_check.txt`, `map_agreement.txt` | login node, under 2 s each |

## 16. Changes from version 1

Every review point was checked against the files before it was adopted.

| Review point | Check | Change |
|---|---|---|
| Field not symmetric in z | confirmed: centre near z = -0.15 m (`field_centre_scan.txt`); per-side bending (job 6812558) | 4.1 gives both signs, the stability maxima and the mirroring test; F1, bins and diagnostics split by sign; risks 3 and 6 |
| Material mismatch depends on \|eta\| | confirmed in `material.json` | 4.3 shows W and dE per region and \|eta\|; risk 4 and D5 rewritten; V1 per region. Not adopted: "no geometric material can do that", since a binned material map was not tested |
| v15 material as configured, not as measured | confirmed; the pixel-endcap dE is -0.92 MeV at \|eta\| = 2-2.5 but -0.11 MeV at 2.5-3 | 4.3 quotes measured W and dE; D5 needs a probe first |
| Geometric pixel-disk holes counted twice | confirmed (job 6812530) | 4.4 emulates the excess; the gate compares total hole rates |
| Gun particle ids negative in int64 | confirmed (login-node test, `convert.py`) | vertex field 1001-1010; converter gate reads the stored ids |
| Decays absent in v15 Fatras | confirmed in the source; TrackML truth has no decay products either (job 6812742) | 4.2 row added. Not adopted: a K decay emulation or a separate K systematic, since TrackML has no decays to match |
| Val projection ignored the matched row | confirmed (`val_discordance.json`) | 9 uses the matched row as primary; risk 1 |
| Purpose did not state H1's estimand | confirmed | section 1 rewritten |
| Margin justified only by precedent; two sigma ranges | confirmed | reasons for delta and one cited sigma range (`sigma_projection.json`); D13 |
| H-T0.5 open to forking paths | confirmed | blinding of H-val gun targets, at most three passes, V1 per region with fixed thresholds and a fixed prompt subset |
| H3 decided after the test read; borrowed seed term | confirmed | D4 decided at H-T0, else H3 leaves H; seed term stated as an assumption with a 0.4-point variant |
| No resolution gate; gates without thresholds | confirmed (`resid.json`, `pydigi.json`) | C1 adds residual widths; all thresholds fixed; descriptive checks marked; closure weight sum defined |
| Study sections cited that do not exist | confirmed (no report on disk) | files, keys and jobs cited; H3 budget derived in `h3_budget.json` (the study's `budget.json` belongs to another study and was not edited) |
| 50 GPU-h line superseded | confirmed (DECISIONS 2026-10-03; trackml_plan section 8) | G0 rule cited; H3 totals stated against 75 GPU-h; 1.5x trigger for H-T0.5 and H-T2 |
| G0 scheduled after the commit | confirmed | step H-P runs G0, F1, containment and M1 before H-T0 |
| Part03 collision twins unchecked | confirmed | collision-reuse run at H-T1 with signatures outside gnn-displaced |
| Split file and host table inaccurate | confirmed (`split_tp_v1.json` rule and counts; `track_stats.py`) | `split_h_v1.json` at H-T0; train includes dev; provenance of emulation inputs corrected |
| "acorn never trained on TrackML"; zero-shot attribution | confirmed | wording scoped to this project; numbers attributed to Canto-B seed 0 |
| T2 used for two things | confirmed | steps renamed H-P, H-T0 ... H-T4 |
| Style of `docs/paper_rewrite_brief.md` | confirmed by a sentence scan | numbers moved into tables, long sentences split, semicolon chains broken, the brief's banned coined term for the gun targets removed (`style_scan.py`) |
