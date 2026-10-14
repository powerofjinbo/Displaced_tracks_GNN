# Addendum G, step T1: samples, splits and populations (truth only)

Written 2026-10-14 12:39 UTC by `hnl.eval.addendum_g_t1` (git dcba763), before any model has read an event of these samples. Definitions in the module docstring; addendum G: `docs/preregistration_new_sample.md`; samples: `docs/new_samples_spec.md`.

## Files and events

| sample | directory | files (npz + sidecars) | events | dropped | checksums |
|---|---|---|---|---|---|
| generic_iso | generated/generic_iso_v1 | 8032 | 8000 | 0 | `results/addendum_g_t1/generic_iso_v1.sha256` |
| generic_dense | generated/generic_dense_v1 | 8032 | 8000 | 0 | `results/addendum_g_t1/generic_dense_v1.sha256` |
| hss15_100 | generated/hss15_100_v1 | 1503 | 1500 | 0 | `results/addendum_g_t1/hss15_100_v1.sha256` |
| hss40_300 | generated/hss40_300_v1 | 1503 | 1500 | 0 | `results/addendum_g_t1/hss40_300_v1.sha256` |

Event checks: the format invariants of `hnl.data.scans.event_checks` (hid a permutation, every pid in the table, unique ids, E = hits - particles with hits, edges within one particle and between modules, one chain per particle, valid geometry ids, finite positions) and the corruption rule of docs/definitions.md section 1 (a hit more than 50 mm outside its module's padded envelope, envelopes from the Zenodo scan derived/scans/v1).

| sample | largest envelope excess (mm) | hits outside the padded envelope | hits on modules unseen in Zenodo | truth checks |
|---|---|---|---|---|
| generic_iso | 6.25 | 243 | 0 | gun_particles: 32000, gun_particles_with_hits: 28374, barcode_vs_sidecar_mismatch: 0, gun_pdg_mismatch: 0, gun_initial_vs_sidecar_max_dp: 6.86e-05, gun_vertex_vs_sidecar_max_dr: 4.01e-05, gun_d0_vs_sidecar_max_diff_mm: 0.000105, particles_with_hits: 1649655, without_initial_state: 0, sidecar_pdg_mismatch: 0 |
| generic_dense | 6.25 | 243 | 0 | gun_particles: 88076, gun_particles_with_hits: 79840, barcode_vs_sidecar_mismatch: 0, gun_pdg_mismatch: 0, gun_initial_vs_sidecar_max_dp: 4.55e-05, gun_vertex_vs_sidecar_max_dr: 3.88e-05, gun_d0_vs_sidecar_max_diff_mm: 0.00016, particles_with_hits: 1691044, without_initial_state: 0, sidecar_pdg_mismatch: 0 |
| hss15_100 | 4.40 | 25 | 0 | particles_with_hits: 255335, without_initial_state: 0, sidecar_pdg_mismatch: 0 |
| hss40_300 | 3.87 | 30 | 0 | particles_with_hits: 266434, without_initial_state: 0, sidecar_pdg_mismatch: 0 |

Truth tables (one row per particle with hits): `derived/generated_v1/t1/truth_generic_iso.parquet` sha256 `ce2e0d9b6a25d76b`; `derived/generated_v1/t1/truth_generic_dense.parquet` sha256 `4ea77651363f92da`; `derived/generated_v1/t1/truth_hss15_100.parquet` sha256 `02dbbb6610facf26`; `derived/generated_v1/t1/truth_hss40_300.parquet` sha256 `941a005e8723fb7d`.

## Splits and populations

Split lists: `splits/generic_iso_{train,val,test}.txt`, `splits/generic_dense_{train,val,test}.txt` (addendum G section 3), `splits/hss15_100_all.txt`, `splits/hss40_300_all.txt`. Training events for T3 (section 5.2): `splits/generic_train_g7999.txt`, the first 7999 of the 9590 train events of both components ordered by event number, iso before dense at equal number (4005 iso, 3994 dense).

| sample | split | events | targets | population (P_G / dense / P_H) | prompt reference | gun targets | pop. r_v >= 100 mm | pop. r_v < 100 mm | pop. abs(d0) >= 300 mm |
|---|---|---|---|---|---|---|---|---|---|
| generic_iso | train | 4810 | 311128 | 5799 | 295037 | 6836 | 4898 | 901 | 163 |
| generic_iso | val | 806 | 52280 | 956 | 49557 | 1111 | 826 | 130 | 38 |
| generic_iso | test | 2384 | 155019 | 2914 | 146985 | 3428 | 2514 | 400 | 96 |
| generic_dense | train | 4780 | 320682 | 16123 | 292098 | 19377 | 13985 | 2138 | 279 |
| generic_dense | val | 835 | 55891 | 2747 | 51122 | 3249 | 2406 | 341 | 50 |
| generic_dense | test | 2385 | 161604 | 7782 | 147619 | 9290 | 6768 | 1014 | 163 |

| sample | events | targets | P_H | electrons | muons | hadrons | s -> ee daughters | V0-tagged | prompt |
|---|---|---|---|---|---|---|---|---|---|
| hss15_100 | 1500 | 55167 | 8294 | 785 | 313 | 7196 | 531 | 1167 | 41381 |
| hss40_300 | 1500 | 56856 | 9719 | 758 | 329 | 8632 | 513 | 1265 | 42147 |

P_G on the test split: 2914 targets (at least 1,000: the claims are not labelled underpowered, addendum G section 6).
