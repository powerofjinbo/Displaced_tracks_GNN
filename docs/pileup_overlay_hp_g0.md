# Step H-P: the patched GenericDetector and gate G0

Status: done on 2026-10-14. G0 passes. This step used no event data and ran no model. It belongs to step H-P of
`docs/pileup_overlay_plan.md` (sections 3 and 10). Scratch work, logs and the controls are in
`/work/submit/jinboz1/hnl/pileup_design/hp_geometry/`.

## Patch and build

ACTS v15 builds the -z endcap modules as the point reflection of the +z modules. We changed one line of
`ProtoLayerCreatorT.hpp` (L489) so that the module centre is the z-mirror of the +z position:

    Acts::Vector3 moduleCenter(ringModulePosition.x(), ringModulePosition.y(), side * ringModulePosition.z());

The patch is `patches/acts_v15/0001-GenericDetector-z-mirror-negative-endcaps.patch` (sha256 c7b22d68...). It applies
to v15.0.0 (commit 0fef9e0). The source copy is `/work/submit/jinboz1/hnl/acts/src-v15-zmirror`; `src-v15` and
`build-v15` were not touched.

Line L491 needs no change. It takes the local y axis from the azimuth of `ringModulePosition`, and after the patch
that is also the azimuth of the -z module centre. The local y axis therefore points radially outward, as in TrackML.
The dump below confirms this: the u, v and w axes of every module agree with TrackML's to within 6.4e-7 in their dot
product.

`slurm/acts/build_v15_zmirror.sbatch` builds into `/work/submit/jinboz1/hnl/acts/build-v15-zmirror`, with the container,
LCG view and CMake options of `build_v15.sbatch`. It builds four targets: `ActsExampleFatrasGeneric`,
`ActsExampleGeometryGeneric`, `ActsExamplePropagationGeneric` and the Python bindings. The build took 4 min on
8 cores (job 6819521) and takes 111 MB. The script checks that the source differs from v15.0.0 by exactly the patch.

## Geometry dump

`scripts/acts/dump_generic_modules.cpp` builds the GenericDetector and writes one row per sensitive surface. Each row
holds the geometry id, the element id, the centre and rotation, the outline in TrackML terms, the thickness and the
readout pitch of the digitisation module. `slurm/acts/dump_generic_modules.sbatch` compiles it against a build tree
and also runs `ActsExampleGeometryGeneric --output-csv` from the same build.

Three cross-checks tie the dumps together. On the unpatched build, the writer output is byte-identical to the dump
the geometry study used (`acts/geometry/detectors.csv`). On the patched build, a dump through the Python bindings is
byte-identical to the writer output (job 6819550). The C++ dump and the writer output agree to 0.03 um, the float
precision of the writer.

## Gate G0

`python -m hnl.overlay.modules g0` compares every module with TrackML's `detectors.csv` (md5 440dbb93...). A module
passes when its centre, its plane and its four corners lie within 4 um of TrackML's. The plane distance is the
largest distance of either module's corners from the other module's plane. The corners are compared at the same local
position, so a frame turned by 180 degrees fails even when the outline is unchanged.

| Volume | Modules | Centre, max [um] | Plane, max [um] | Corner, max [um] | Failures |
|---|---|---|---|---|---|
| 7, pixel endcap -z | 756 | 0.49 | 0.00 | 0.51 | 0 |
| 8, pixel barrel | 2,492 | 0.64 | 0.64 | 0.64 | 0 |
| 9, pixel endcap +z | 756 | 0.49 | 0.00 | 0.51 | 0 |
| 12, short-strip endcap -z | 1,020 | 0.63 | 0.00 | 0.65 | 0 |
| 13, short-strip barrel | 5,796 | 0.62 | 0.62 | 0.63 | 0 |
| 14, short-strip endcap +z | 1,020 | 0.63 | 0.00 | 0.65 | 0 |
| 16, long-strip endcap -z | 588 | 0.62 | 0.00 | 0.65 | 0 |
| 17, long-strip barrel | 5,712 | 3.74 | 3.73 | 3.75 | 0 |
| 18, long-strip endcap +z | 588 | 0.62 | 0.00 | 0.65 | 0 |

Source: `hp_geometry/g0_zmirror_final/g0.json` (job 6819551). All 18,728 modules pass, and the id table is a
bijection. The -z volumes now give the same numbers as the +z volumes.

The largest difference, 3.75 um in volume 17, is the rounding of the TrackML file. TrackML writes six significant
digits, which at r = 1 m means steps of 10 um. Rounded to six significant digits, every patched ACTS centre equals
the TrackML centre exactly (`hp_geometry/rounding_check.json`, job 6819558). Bounds and pitches agree exactly, and the
thicknesses agree to 1e-16 mm. The pitch differs by 6e-9 mm because ACTS stores its bin edges as floats.

Two controls support the comparison code. On the unpatched dump it fails 2,364 modules (job 6819531). These are all
1,608 modules of volumes 12 and 16 with inverted trapezoids, 624 of them also 3 mm off the TrackML plane. The other
756 are the volume-7 rectangles, whose frames are turned by 180 degrees: their corner set agrees within 0.51 um.
This reproduces the geometry study (`geometry/out/modules_summary.json`, `footprint.json`). We also reran the study's
own `compare_modules.py` and the corner part of `footprint.py` on the patched writer output (job 6819544). They find
dz = 0 and all axis dot products +1, and their module table agrees with ours on all 18,728 modules.

## What the patch changed, and the order of the ACTS ids

Only the 2,364 -z endcap modules of volumes 7, 12 and 16 moved, and the other 16,364 rows are unchanged. Each moved
module keeps its sensitive id and its element id; its centre goes from (-x, -y, z) to (x, y, z)
(`hp_geometry/id_order.json`, job 6819545).

In the patched build, the -z module with id m in layer L is the exact z-mirror of the +z module with id m in the
mirrored layer (for the pixels, layer 2 mirrors layer 14). TrackML numbers its modules the same way. The map from
ACTS to TrackML ids is therefore the same on both sides and in every layer of a subsystem. Every -z entry differs from
the unpatched tables, while the barrel and +z entries are unchanged.

## Module-id table

`configs/trackml/module_map_acts_zmirror_v1.csv` holds one row per module: volume_id, layer_id, acts_module_id and
trackml_module_id (18,728 rows, sha256 88178fd6590e1077814489047ab3cb8d91723df9be20fa9cd05a460b151af9de). The sidecar
`module_map_acts_zmirror_v1.json` records the sha256, the patch, the build, the dump and the G0 result. Each ACTS
module is paired with the TrackML module of nearest centre in its (volume, layer). The pairs lie within 3.74 um,
while the next TrackML module is at least 11.1 mm away, so no pair is ambiguous.

Barrel ids pass unchanged. In the endcaps, 2.2 % of the modules keep their id. `hnl.overlay.modules.ModuleMap` reads
the table, checks its sha256 and translates the geometry ids of the Fatras sim-hit CSV
(`ModuleMap.from_geometry_id`). An unknown module raises a KeyError. The unit tests are in
`tests/test_overlay_modules.py`; one of them checks the committed table.

## Smoke test of the simulation

`ActsExampleFatrasGeneric` from each build simulated 100 events of 20 muons with |eta| < 3 and pT of 1-10 GeV in a
constant 2 T field (jobs 6819553, 6819554). This is not the containment check of section 10, which uses the gun
content. We mapped each sim hit through its build's table and placed it in the TrackML module frame
(`hp_geometry/smoke_check.json`, job 6819556).

| Build | Sim hits | Off the plane by more than 10 um | Outside the outline by more than 10 um | Largest plane distance [um] |
|---|---|---|---|---|
| patched, new table | 24,453 | 0 | 0 | 3.8 |
| unpatched, design-study table | 24,466 | 5.6 % | 0.8 % | 3,000 |

With the patch every hit lies on its TrackML module and inside its outline. In the unpatched build, 6.0 % of the
hits fail either rule. That is the size the plan quotes for route C (5.9-7.0 % of gun hits).

## Using the build

Run the executables inside the v15 container with the LCG_97apython3 view, as in `build_v15.sbatch`. For Python,
source `build-v15-zmirror/python/setup.sh`. The executables and the bindings carry an RPATH to
`build-v15-zmirror/lib64`. An entry for `build-v15/lib64` in `LD_LIBRARY_PATH` does not override it; we checked this
with ldd (job 6819550).
