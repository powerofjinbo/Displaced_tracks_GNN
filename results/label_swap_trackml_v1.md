# Control of the hit-label exchange test on TrackML (truth only, no model output)

Script: scripts/diagnostics/label_swap_trackml.py (the RANSAC test of label_swap_ransac.py: pT >= 2 GeV, >= 6 hits,
transverse circle, tolerance 1 mm; module = volume, layer, module). Jobs 6810638 (throughput phase, our tables, the
150 validation events of configs/trackml/split_tp_v1.json) and 6810639 (accuracy phase, the first 50 validation events
of configs/trackml/split_v1.json read from train_1.zip directly). CPU, express; A30 and GTX1080 nodes excluded.

| set | events | clean particles | off-trajectory hits, modules without another particle | off-trajectory hits, shared modules | of these, another hit on the module on the trajectory |
|---|---|---|---|---|---|
| throughput val | 150 | 33686 | 1.15 % (196/17031) | 1.09 % (4160/380220) | 6.1 % (255/4160) |
| accuracy val | 50 | 11561 | 1.24 % (75/6030) | 1.04 % (1363/130584) | 6.8 % (93/1363) |

- TrackML control: off-trajectory hits 1.09 % on shared modules against 1.15 % on unshared modules (throughput val).
- Accuracy phase, truth positions: pixel hits (volumes 7, 8, 9) farther than 1 mm from the true crossing (tx, ty, tz)
  of the particle they are labelled with: 0 of 2684539 on shared modules and 0 of 7333 on unshared modules (median
  distance on shared modules 0.009 mm).

In the Zenodo HNL and ttbar files the same RANSAC test gives 0.06-0.20 % on unshared modules against 16.7-23.8 % on
shared modules, 95.5-97.5 % of them with the particle's position on the module under another label
(results/label_swap_verify_v1.md). On TrackML the shared and unshared rates agree, and only 6-7 % of the shared-module
outliers have another hit on the trajectory, the rate expected by chance in pileup-dense modules: no exchange. The
outlier rate of about 1 % on both kinds of modules comes from the inhomogeneous TrackML field and multiple scattering,
which a circle does not describe. The exchange is a property of the production of the Zenodo files.
