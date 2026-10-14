# Verification of the hit-label exchange (2026-10-14; adversarial check of results/dense_jet_check_v1.md)

**Verdict: confirmed in substance; the quoted rates depend on the method.** In the authors' npz files, where two
particles cross one module, the positions and the particle labels of their hits are permuted in a fraction of the
modules. It is not an artefact of the straight-line test: a robust helix (RANSAC circle) per particle, built only
from its own labelled hits, finds the particle's true position on the same module under another particle's label,
within 0.05 mm (median) of its trajectory. The labels are correct on modules that one particle crosses alone.

## Evidence

- **Independent method (section 1).** Each particle with pT >= 2 GeV and >= 6 hits gets a RANSAC circle in the
  transverse plane from its own hits (tolerance 1 mm), with no chain order or reference line.
  - On modules that the particle crosses alone, 0.06-0.20 % of its hits are off its trajectory.
  - On modules shared with another particle, 16.7-23.8 % are off, and 95.5-97.5 % of these have another hit on the
    same module, labelled as another particle, lying on the trajectory.
  - Those positions sit at a median of 0.047 mm from the trajectory (90 % quantile 0.34 mm), pixel resolution. The
    labelled hit sits at a median of 6.7 mm.
- **Module level.** On modules whose hits all belong to such particles, and where every particle has exactly one
  position on its trajectory, the labels do not match the positions in:
  - 11.2 % of the two-hit modules (1961/17448, ttbar val);
  - 24.6 % of the three-hit modules;
  - 26 % of the four-hit modules.
- **Who is involved.** 98.6 % of the particles in mismatched modules are generator particles (gen 0), not secondaries.
- **Separation.** The exchanged positions are 2 to more than 50 mm apart (2791 pairs above 50 mm in 1000 ttbar val
  events). Below 2 mm the 1 mm tolerance cannot tell the two hits apart. The rate does not fall with separation as
  merged or shared clusters would.
- **Local-fit pair test, all particles (section 3).** A quadratic transverse fit plus an (r, z) line through up to
  two chain neighbours on each side, not the straight line of dense_jet_check.
  - Same-module pairs: 9.8 % (ttbar val) and 11.6 % (hnl10_100 val) exchanged.
  - Control, different modules of one layer within 50 mm: 0.12-0.19 %.
  - The rate rises from 6-9 % below 2 mm to 14-15 % above 50 mm, as the test's sensitivity grows with separation.
- **Examples checked by hand (pT >= 2 GeV particles; position (r, phi, z) in mm, rad; distances transverse to the
  RANSAC circle).**
  - ttbar event 99, endcap volume 18, layer 8, one module, two hits 61.9 mm apart.
    - Hit 2194 is labelled Sigma-bar (pT 11.2 GeV) but lies 16.4 mm off its trajectory. It lies 0.19 mm from the
      trajectory of the particle labelled on hit 2195 (pT 8.1 GeV).
    - Hit 2195 lies 11.8 mm off its label's trajectory and 0.018 mm from the trajectory of hit 2194's label.
  - ttbar event 234, volume 12, layer 6, two hits 82.8 mm apart (pi-, pT 2.7 GeV; pi+, pT 5.4 GeV). The distances
    are 53.9 vs 0.047 mm and 41.4 vs 0.46 mm.
  - ttbar event 8: a proton with pT 11.8 GeV whose last labelled hit (r = 955 mm) is 88 mm off its trajectory. On a
    four-hit endcap module, the proton's trajectory passes within 0.9 mm of the position labelled as a 0.75 GeV
    electron. Two other labels on that module are also displaced by one position.

## Where it comes from

- **Not our conversion.** Every number above reads the unpacked Zenodo npz files directly (np.load). The derived hit
  tables of hnl.data.convert equal the npz arrays row by row (pid, hid, geometry id, r, phi, z) in all 1000 ttbar val
  and 1000 hnl10_100 val events (0 differ).
- **Upstream of the npz files: an ordering mismatch within modules.**
  - Within every module, the labels are in strict particle-id order of the hit ids. In all 201,029 (ttbar val) and
    138,931 (hnl10_100 val) modules holding exactly two hits of two particles, the lower hit id carries the lower
    particle id. Hit ids follow the geometry id.
  - The positions follow that order in only about 89 % of the two-hit modules.
  - Labels attached to positions by order within a module, with the two orders disagreeing, reproduce what is seen:
    - permutations only within one module;
    - a rate independent of separation;
    - mostly generator particles involved;
    - the same rate in every sample and split.
- **Why the second ordering differs.** One example is an unstable sort of the simulated hits by module, against a
  truth list ordered by particle. We could not check this without the authors' production code (ACTS version and
  writer not stated in Wang et al.).
- **Known before, in part.** DECISIONS.md 2026-10-03 records outlier hits (30 % of W muons have at least one; "likely
  secondaries recorded under the parent's id") and "label swaps with secondaries". The cause is now identified: an
  exchange with any particle on the same module, mostly generator particles.

## Rates per sample (truth only; test splits read for hit truth only, no model output)

Section 1 table: 13.1-14.2 % of the resolved multi-particle modules are mismatched in every HNL sample and split, and
14.1-14.7 % in ttbar. Shared-module hits off the trajectory: 16.7-17.5 % (HNL samples), 23.5-23.8 % (ttbar).

## Against the dense-jet check's numbers

- **"about 15 % of pairs, flat from 5 to > 50 mm".** The phenomenon holds; the number depends on the test. Here:
  - same-module pairs, local fit: 9.8 % (ttbar) and 11.6 % (HNL);
  - two-hit modules: 11.2 %;
  - shared-module hits of pT >= 2 GeV particles: 23.5 % (ttbar val).
  The local test's rate rises mildly with separation and the RANSAC rate falls mildly above 10 mm; neither shows a
  merging-like fall-off.
- **"same in ttbar and HNL".** Confirmed at module level (13-15 % everywhere).
- **"88 % of displaced ttbar targets have a swapped hit".** 82.8 +- 0.5 % here (pT >= 1 GeV RANSAC; the 75 % of
  displaced targets with a fit).
- **"41 % drop below |d0| 1 mm without them".** Here 49.6 +- 0.6 % of the fitted displaced targets have |d0| < 1 mm
  for the circle through their inlier hits. The direction agrees: about half of the "displaced" ttbar targets
  (1 <= |d0| < 100 mm) are prompt particles made displaced by an exchanged hit. The 25 % of displaced targets
  without a fit (too many off-trajectory hits) are not counted.
- **HNL daughters and W muons.** HNL daughters with an exchanged hit: 14-19 % (dense check 17 %). W muons: 24-27 %
  (dense check 26-30 %).

## Jobs, code, outputs

- **Jobs:** CPU, submit-gpu-express / submit, A30 and GTX1080 nodes excluded.
  - 6810581, 6810587: smoke tests.
  - 6810590-6810600, 6810604, 6810605: RANSAC.
  - 6810601, 6810602: local-fit test and derived check.
  - 6810619: summary.
- **Code (new files):** scripts/diagnostics/label_swap_ransac.py, label_swap_verify.py, label_swap_summary.py;
  slurm/diagnostics/label_swap_verify.sbatch.
- **Outputs:** /work/submit/jinboz1/hnl/tmp/label_swap/full (parquet per sample and split).

The generated summary follows.

---


Truth only; the authors' npz files read directly (hnl.data.npz.load_event). Scripts: scripts/diagnostics/label_swap_{ransac,verify,summary}.py; job wrapper slurm/diagnostics/label_swap_verify.sbatch.

## 1. Robust trajectories (RANSAC circle, pT >= 2 GeV, >= 6 hits, tolerance 1 mm), per sample and split

| sample | split | events | clean particles | outlier hits, modules without another hit | outlier hits, modules shared with another particle | of these, another hit on the module lies on the trajectory | resolved multi-particle modules with labels not matching positions |
|---|---|---|---|---|---|---|---|
| hnl10_100 | test | 1000 | 6484 | 0.08 +- 0.01 % (40/50081) | 16.65 +- 0.23 % (4223/25366) | 97.23 +- 0.25 % (4106/4223) | 13.94 +- 0.93 % (194/1392) |
| hnl10_100 | train | 1000 | 6843 | 0.06 +- 0.01 % (29/52431) | 16.77 +- 0.23 % (4513/26906) | 97.36 +- 0.24 % (4394/4513) | 14.22 +- 0.88 % (222/1561) |
| hnl10_100 | val | 1000 | 7076 | 0.07 +- 0.01 % (38/54659) | 16.87 +- 0.23 % (4611/27334) | 97.25 +- 0.24 % (4484/4611) | 13.67 +- 0.87 % (214/1565) |
| hnl15_100 | external | 1000 | 7048 | 0.07 +- 0.01 % (40/54865) | 17.46 +- 0.23 % (4746/27184) | 97.32 +- 0.23 % (4619/4746) | 13.78 +- 0.85 % (225/1633) |
| hnl15_200 | test | 1000 | 6399 | 0.07 +- 0.01 % (37/49606) | 16.75 +- 0.24 % (4208/25119) | 96.89 +- 0.27 % (4077/4208) | 13.68 +- 0.95 % (180/1316) |
| hnl15_200 | train | 1000 | 6729 | 0.07 +- 0.01 % (37/51929) | 16.74 +- 0.23 % (4396/26261) | 97.18 +- 0.25 % (4272/4396) | 13.06 +- 0.90 % (182/1394) |
| hnl15_200 | val | 999 | 7000 | 0.07 +- 0.01 % (38/54038) | 16.99 +- 0.23 % (4652/27385) | 97.53 +- 0.23 % (4537/4652) | 13.62 +- 0.89 % (203/1491) |
| ttbar | test | 1000 | 30592 | 0.19 +- 0.01 % (286/151814) | 23.76 +- 0.09 % (51230/215615) | 95.74 +- 0.09 % (49048/51230) | 14.69 +- 0.24 % (3287/22372) |
| ttbar | train | 1000 | 30612 | 0.20 +- 0.01 % (299/152160) | 23.52 +- 0.09 % (50791/215957) | 95.54 +- 0.09 % (48524/50791) | 14.08 +- 0.23 % (3177/22556) |
| ttbar | val | 1000 | 30129 | 0.20 +- 0.01 % (300/149169) | 23.52 +- 0.09 % (50104/213058) | 95.64 +- 0.09 % (47917/50104) | 14.06 +- 0.23 % (3087/21955) |

## 2. ttbar val (pT >= 2 GeV): the exchanged positions

- Distance of the position labelled as another particle from the particle's trajectory, for the explained outliers: median 0.047 mm, 90 % quantile 0.336 mm; distance of the labelled hit: median 6.7 mm.
- Transverse separation of the two positions: (0.0, 1.0]: 72, (1.0, 2.0]: 2099, (2.0, 5.0]: 12252, (5.0, 10.0]: 13113, (10.0, 20.0]: 9582, (20.0, 50.0]: 8008, (50.0, 100000.0]: 2791
- Fraction of shared-module hits exchanged, by separation to the nearest other hit on the module: (0.0, 2.0]: 4.2 %, (2.0, 5.0]: 30.7 %, (5.0, 10.0]: 34.1 %, (10.0, 20.0]: 28.1 %, (20.0, 50.0]: 23.3 %, (50.0, 100000.0]: 19.5 %
- By volume: 7: 19.6 %, 8: 26.1 %, 9: 19.9 %, 12: 19.3 %, 13: 21.5 %, 14: 20.1 %, 16: 16.9 %, 17: 12.1 %, 18: 16.3 %
- Resolved modules with n hits, fraction whose labels do not match the positions: n=2: 11.24 +- 0.24 % (1961/17448), n=3: 24.62 +- 0.72 % (872/3542), n=4: 26.00 +- 1.57 % (202/777), n=5: 27.92 +- 3.62 % (43/154), n=6: 23.33 +- 7.72 % (7/30), n=7: 50.00 +- 25.00 % (2/4)
- Particles in modules with mismatched labels: 7617, of which generator particles (gen 0) 98.60 +- 0.13 % (7510/7617).

## 3. Local-fit pair test on every same-module pair (quadratic transverse fit + (r, z) line through up to two chain neighbours on each side), all particles

- hnl10_100_val: same-module pairs 11.61 +- 0.06 % (39118/336880) exchanged; control (different modules of one layer, <= 50 mm) 0.12 +- 0.00 % (3854/3160000). By separation: (0.0, 2.0] 9.3 %, (2.0, 5.0] 9.3 %, (5.0, 10.0] 9.9 %, (10.0, 20.0] 11.4 %, (20.0, 50.0] 13.6 %, (50.0, 100000.0] 14.7 %; control: (0, 2] 0.14 %, (2, 5] 0.19 %, (5, 10] 0.25 %, (10, 20] 0.20 %, (20, 50] 0.08 %
  Both particles pT >= 2 GeV: 8.72 +- 0.34 % (591/6781); any pT below 1 GeV: 11.72 +- 0.06 % (36143/308323).
  Modules holding exactly two hits of two particles: 138931; the hit with the lower hit id carries the lower particle id in 100.00 +- 0.00 % (138931/138931).
  Derived hit tables (hnl.data.convert) against the npz arrays, row by row (pid, hid, geometry id, r, phi, z): 0 of 1000 events differ.
- ttbar_val: same-module pairs 9.80 +- 0.03 % (90396/922451) exchanged; control (different modules of one layer, <= 50 mm) 0.19 +- 0.00 % (7682/3958108). By separation: (0.0, 2.0] 6.2 %, (2.0, 5.0] 7.6 %, (5.0, 10.0] 9.3 %, (10.0, 20.0] 11.1 %, (20.0, 50.0] 12.9 %, (50.0, 100000.0] 13.9 %; control: (0, 2] 0.30 %, (2, 5] 0.52 %, (5, 10] 0.48 %, (10, 20] 0.35 %, (20, 50] 0.08 %
  Both particles pT >= 2 GeV: 7.50 +- 0.05 % (17839/237919); any pT below 1 GeV: 11.02 +- 0.04 % (57703/523663).
  Modules holding exactly two hits of two particles: 201029; the hit with the lower hit id carries the lower particle id in 100.00 +- 0.00 % (201029/201029).
  Derived hit tables (hnl.data.convert) against the npz arrays, row by row (pid, hid, geometry id, r, phi, z): 0 of 1000 events differ.

## 4. Targets (pT >= 1 GeV RANSAC; d0 as in docs/definitions.md section 3)

### hnl10_100 val (1000 events)

- HNL daughters: 990 targets, RANSAC fit 89.60 +- 0.97 % (887/990); with an exchanged hit 18.83 +- 1.31 % (167/887); |d0| of the circle through the inliers < 1 mm: 8.23 +- 0.92 % (73/887).
- W muons: 935 targets, RANSAC fit 99.36 +- 0.26 % (929/935); with an exchanged hit 24.00 +- 1.40 % (223/929); |d0| of the circle through the inliers < 1 mm: 95.05 +- 0.71 % (883/929).
- other, displaced (1 <= |d0| < 100 mm): 2171 targets, RANSAC fit 91.29 +- 0.61 % (1982/2171); with an exchanged hit 54.29 +- 1.12 % (1076/1982); |d0| of the circle through the inliers < 1 mm: 33.05 +- 1.06 % (655/1982).

### hnl15_200 val (999 events)

- HNL daughters: 835 targets, RANSAC fit 87.07 +- 1.16 % (727/835); with an exchanged hit 14.44 +- 1.30 % (105/727); |d0| of the circle through the inliers < 1 mm: 7.15 +- 0.96 % (52/727).
- W muons: 934 targets, RANSAC fit 99.46 +- 0.24 % (929/934); with an exchanged hit 27.45 +- 1.46 % (255/929); |d0| of the circle through the inliers < 1 mm: 94.40 +- 0.75 % (877/929).
- other, displaced (1 <= |d0| < 100 mm): 2153 targets, RANSAC fit 91.08 +- 0.61 % (1961/2153); with an exchanged hit 52.07 +- 1.13 % (1021/1961); |d0| of the circle through the inliers < 1 mm: 34.17 +- 1.07 % (670/1961).

### ttbar val (1000 events)

- prompt (|d0| < 1 mm): 52702 targets, RANSAC fit 98.24 +- 0.06 % (51772/52702); with an exchanged hit 63.45 +- 0.21 % (32849/51772); |d0| of the circle through the inliers < 1 mm: 95.24 +- 0.09 % (49310/51772).
- displaced (1 <= |d0| < 100 mm): 8411 targets, RANSAC fit 75.14 +- 0.47 % (6320/8411); with an exchanged hit 82.75 +- 0.48 % (5230/6320); |d0| of the circle through the inliers < 1 mm: 49.65 +- 0.63 % (3138/6320).

