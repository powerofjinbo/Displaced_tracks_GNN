# Addendum G, step T1 on the extension generic_iso_x1 (truth only)

Written 2026-10-14 20:02 UTC by `hnl.eval.addendum_g_t1_x1` (git 2d61fcf), before any model has read an event of the extension. The extension is T1 deviation 13 (RESULTS.md): events 8000-23999 of the isolated component, the settings of `docs/new_samples_spec.md` section 3, base seed 260001 (gun 260008, Fatras and digitisation 260014), section 9 of the spec. Definitions in the module docstrings of `hnl.eval.addendum_g_t1` and `hnl.eval.addendum_g_t1_x1`; the T1 outputs of `results/addendum_g_t1_v1.md` are unchanged.

## Production

- Reproduction check before production (job 6819943, git 3d80f02): chunk 7500-7999 of generic_iso_v1 (base seed 250001) regenerated with the extension's code path (`slurm/acts/produce.sbatch`, PACK=tar) into scratch on /work: 500 of 500 events identical to the stored npz in every array (dtype, shape and bytes), sidecars `truth_7500.npz` 19 arrays, identical, `gun_truth_7500.npz` 19 arrays, identical. PASS.
- Reader identity (sufficiency report 6.1): the 500 stored npz files of that chunk packed as tar members read back through `hnl.data.shards`: bytes identical 500, arrays identical 500 (`load_event_from`) and 500 (`iter_events_from`).
- Jobs (sacct; CPU-h = CPUs x elapsed):

| job | name | CPUs | elapsed (s) | state | node | CPU-h |
|---|---|---|---|---|---|---|
| 6819942 | acts_repro_x1 | 4 | 56 | COMPLETED | submit37 | 0.06 |
| 6819943 | g_t1x1_repro | 4 | 100 | COMPLETED | submit37 | 0.11 |
| 6819969_0 | acts_x1 | 4 | 76 | COMPLETED | submit37 | 0.08 |
| 6819969_1 | acts_x1 | 4 | 76 | COMPLETED | submit37 | 0.08 |
| 6819969_2 | acts_x1 | 4 | 76 | COMPLETED | submit37 | 0.08 |
| 6819969_3 | acts_x1 | 4 | 76 | COMPLETED | submit37 | 0.08 |
| 6819969_4 | acts_x1 | 4 | 76 | COMPLETED | submit37 | 0.08 |
| 6819969_5 | acts_x1 | 4 | 74 | COMPLETED | submit37 | 0.08 |
| 6819969_6 | acts_x1 | 4 | 77 | COMPLETED | submit37 | 0.09 |
| 6819969_7 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_8 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_9 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_10 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_11 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_12 | acts_x1 | 4 | 74 | COMPLETED | submit37 | 0.08 |
| 6819969_13 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_14 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_15 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_16 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_17 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_18 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_19 | acts_x1 | 4 | 74 | COMPLETED | submit37 | 0.08 |
| 6819969_20 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_21 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_22 | acts_x1 | 4 | 73 | COMPLETED | submit37 | 0.08 |
| 6819969_23 | acts_x1 | 4 | 74 | COMPLETED | submit37 | 0.08 |
| 6819969_24 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_25 | acts_x1 | 4 | 76 | COMPLETED | submit37 | 0.08 |
| 6819969_26 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_27 | acts_x1 | 4 | 75 | COMPLETED | submit37 | 0.08 |
| 6819969_28 | acts_x1 | 4 | 72 | COMPLETED | submit37 | 0.08 |
| 6819969_29 | acts_x1 | 4 | 72 | COMPLETED | submit37 | 0.08 |
| 6819969_30 | acts_x1 | 4 | 55 | COMPLETED | submit37 | 0.06 |
| 6819969_31 | acts_x1 | 4 | 55 | COMPLETED | submit37 | 0.06 |
| 6820016 | g_t1x1_scan | 24 | 33 | COMPLETED | submit37 | 0.22 |
| 6820017 | g_t1x1_summary | 4 | 30 | COMPLETED | submit37 | 0.03 |
| 6820079 | g_t1x1_summary | 4 | 24 | COMPLETED | submit37 | 0.03 |

  Total 3.1 CPU-h; no GPU.

## Files and events

- 32 tars `generated/generic_iso_x1/events_<first>_<last>.tar` (500 members `<event>.npz` each, 16000 in all, ranges contiguous 8000-23999) and 64 sidecars in `generated/generic_iso_x1_truth/`; nothing else in either directory. Checksums: `results/addendum_g_t1/generic_iso_x1.sha256` (96 files, sha256sum format).
- Every event read through `hnl.data.shards` with the format invariants and the corruption rule of T1: 16000 events, 0 dropped, 0 unreadable. Largest excess over the padded Zenodo module envelopes 7.56 mm (rule 50 mm), 481 hits outside them, 0 hits on modules unseen in the Zenodo scan (generic_iso_v1: 6.25 mm, 243 hits).
- Truth checks: gun_particles: 64000, gun_particles_with_hits: 56643, barcode_vs_sidecar_mismatch: 0, gun_pdg_mismatch: 0, gun_initial_vs_sidecar_max_dp: 4.61e-05, gun_vertex_vs_sidecar_max_dr: 4.09e-05, gun_d0_vs_sidecar_max_diff_mm: 0.000109, particles_with_hits: 3291744, without_initial_state: 0, sidecar_pdg_mismatch: 0, sidecar_xmode: ['measured'], gun_component: ['iso'], sidecar_events: 16000, sidecar_event_range: [8000, 23999], gun_events: 16000, gun_particles_per_event: [4], dropped_duplicate_hits: 0, measurements: 33224521, pull_abs_gt5: 42.
- Shared code: on generic_iso_v1 events 0-499 and 7500-7999 the T1 functions reproduce the committed T1 truth table (207781 rows, identical: True).
- Truth table: `derived/generated_v1/t1/truth_generic_iso_x1.parquet` sha256 `1cf9df9da8c3edab`.

## Splits and populations

Split lists (addendum G section 3, component `iso`): `splits/generic_iso_x1_{train,val,test}.txt`. Union test list of the claims (deviation 13a): `splits/generic_iso_union_test.txt` = `splits/generic_iso_test.txt` (T1, unchanged) + `splits/generic_iso_x1_test.txt`, sorted by event number. The extension's train and val events are not read by any model under addendum G (deviation 13c); the T3 training list `splits/generic_train_g7999.txt` is unchanged (13b).

| sample | split | events | targets | P_G | prompt reference | gun targets | P_G r_v >= 100 mm | P_G r_v < 100 mm | P_G abs(d0) >= 300 mm |
|---|---|---|---|---|---|---|---|---|---|
| generic_iso_x1 | train | 9568 | 620885 | 11233 | 588964 | 13289 | 9621 | 1612 | 319 |
| generic_iso_x1 | val | 1583 | 101930 | 1836 | 96610 | 2174 | 1603 | 233 | 64 |
| generic_iso_x1 | test | 4849 | 312494 | 5796 | 296236 | 6968 | 4862 | 934 | 166 |
| generic_iso_v1 | test | 2384 | 155019 | 2914 | 146985 | 3428 | 2514 | 400 | 96 |
| generic_iso (union) | test | 7233 | 467513 | 8710 | 443221 | 10396 | 7376 | 1334 | 262 |

P_G on the union test split: 8710 targets (2914 from T1 + 5796 from the extension). Expected before production (sufficiency report 6.1): about 4,770 extension test events and about 8,700 P_G targets on the union; here 4849 extension test events (30 % of 16,000: 4,800 +- 58), and 5927 extension P_G targets at the T1 test rate of 1.222 per event.

## Comparison with generic_iso_v1 (all events, truth only)

Rule fixed in the module docstring before the extension existed: event-permutation tests (B = 10,000, default_rng(20261014)); a p-value below 0.01 / 9 = 0.00111 is a difference beyond expectation and stops the work. Means with event-clustered standard errors; species shares e / mu / pi / K / p.

| quantity | statistic | T | p (permutation) | generic_iso_v1 | generic_iso_x1 |
|---|---|---|---|---|---|
| P_G targets per event | |mean difference| | 0.02956 | 0.0611 | 1.209 +- 0.013 | 1.179 +- 0.0088 |
| r_v of P_G targets | KS distance | 0.01644 | 0.2495 | 243.2 +- 1.7 mm | 242.3 +- 1.2 mm |
| exact |d0| of P_G targets | KS distance | 0.01128 | 0.4721 | 64.81 +- 0.97 mm | 64.35 +- 0.68 mm |
| pT of P_G targets | KS distance | 0.00811 | 0.8327 | 15.14 +- 0.35 GeV | 14.86 +- 0.24 GeV |
| eta of P_G targets | KS distance | 0.01191 | 0.4656 | 0.06381 +- 0.015 | 0.07275 +- 0.011 |
| species shares of P_G targets (e, mu, pi, K, p) | chi-square (4 dof) | 12.79 | 0.0974 | 0.131 / 0.204 / 0.198 / 0.227 / 0.241 | 0.134 / 0.188 / 0.196 / 0.231 / 0.251 |
| hits per P_G target | KS distance | 0.0052 | 0.8248 | 7.171 +- 0.024 | 7.172 +- 0.017 |
| targets per event (all) | |mean difference| | 0.09656 | 0.7279 | 64.8 +- 0.23 | 64.71 +- 0.16 |
| hits per target (all) | KS distance | 0.001335 | 0.2961 | 11.91 +- 0.0033 | 11.91 +- 0.0023 |

Smallest p-value 0.0611 (threshold 0.00111); 0 of 9 below 0.05 (0.45 expected). **Verdict: PASS** (no difference beyond expectation; the extension joins the generic_iso test split as fixed by deviation 13).

Control (not part of the rule): the same tests between generic_iso_v1 events 0-3999 and 4000-7999 give p-values 0.019, 0.605, 0.163, 0.547, 0.031, 0.666, 0.339, 0.401, 0.107 (smallest 0.0193; 2 below 0.05).

## Shared ttbar collisions (T1 deviation 14)

Events whose ttbar collision vertex (the most common production vertex of the generator particles with hits) equals that of another event, among generic_iso_v1, generic_iso_x1 and generic_dense_v1 (generic_iso_v1 8000, generic_iso_x1 16000, generic_dense_v1 8000 events): 4 groups in all (generic_dense_v1 150, generic_iso_v1 470; generic_dense_v1 3985, generic_iso_v1 4010; generic_dense_v1 735, generic_iso_v1 860; generic_dense_v1 1528, generic_iso_v1 1592); 0 involve an extension event.
