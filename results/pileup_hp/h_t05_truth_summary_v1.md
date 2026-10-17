# Addendum H, step H-T0.5.1: H-val and closure overlays, truth gates, P_P yield and projection (pass 1)

Date: 2026-10-16 (UTC). Truth only. No model ran on an overlay event, and no model output of a gun target was read.
No H-test event was made, and training_part03 was not streamed. Rules: `docs/preregistration_pileup.md` sections 2, 6,
9.2, 9.3 and 10 (addendum H, 82fa6cc).

**Outcome.** Both sets were made and pass every per-set truth check. The closure truth gate passes. The C1
confirmation, a decisive gate, fails: in part 1 the mean u bias of volume 12 is -1.77 µm (rule 1 µm), and in part 2
the u bias of volume 9 is -1.013 µm against a limit of 1.0 µm. Section 9.3 does not settle what follows mechanically,
so the work stops here (see the last section). This is pass 1 (overlay_hash 3e23b363...).

## Production

| Item | H-val | closure |
|---|---|---|
| job | 6826774 (overlay_set.sbatch, 4 min 39 s) | 6826775 (closure_set.sbatch with the closure checks, 3 min 13 s) |
| code | 7dacde0 | 951f21c (differs from 7dacde0 only in `docs/feedback/test_use_ledger.md`) |
| hosts | the 150 tp_v1 val events of `split_h_v1.json`, Fatras events 0-149 | the same |
| base seed, Fatras seed | 600,000,001, 600,000,014 | 610,000,001, 610,000,014 |
| overlay_hash | 3e23b3630f6b... (emulation hash 2be64cc0...) | the same |
| stored | `derived/overlay_v1/{hits,particles,ptype}/hval`, meta sha256 8670092c... | `.../closure`, meta sha256 0048d136... |
| Fatras | 3,994 gun particles: 3,913 simulated, 77 failed (navigator), 4 below pMin | 6,724 replaced natives: 6,714 simulated, 10 failed (all in population (b)) |

Work directories (gun inputs, packed Fatras hits, CSV tars, the V1 population table):
`/work/submit/jinboz1/hnl/pileup_design/h_t05/{hval_work,closure_work}`. Check jobs: truth checks 6826786, C1
confirmation 6826787, gun content 6826788, reweighted projection 6826789, species shares 6826826, closure truth gate
6827073 (all at 951f21c).

## Gates of section 9.2

| Gate | Kind | Result | Record |
|---|---|---|---|
| Per-set truth checks, H-val (format, weights, containment, host rows and production, version) | decisive | pass; largest \|host weight sum - 1\| 2.2e-7; largest move onto a module plane 3.79 µm, none outside bounds | `h_t05_truth_checks_hval_v1.json` |
| Per-set truth checks, closure (population, ids, format, containment, converter view, production, version) | decisive | pass; V1 populations equal `val_regions.json` ((a) 3,276, (b) 448, (c) 3,000) | `h_t05_truth_closure_checks_v1.json` |
| C1 confirmation | decisive | **fail** (below) | `h_t05_truth_c1_confirm_hval_v1.json` |
| Closure truth gate | flag | pass: 9 counted cells, largest \|difference\| 0.010 hits per crossed layer (long-strip endcap, \|eta\| 1-2), rule 0.05 | `h_t05_truth_closure_gate_v1.json` |

**C1 confirmation** (20,039 gun hits; 5,713 on P_P targets; reference the 40 calibration events of C1).

- Part 1, matched bins: 81 counted bins and none fails. The mean u bias over the counted bins of a volume fails in
  volume 12 (short-strip endcap, -z): -1.77 µm over 4 bins, against the plain 1 µm rule of part 1. Volumes 8, 13, 14
  and 17 pass (0.06, -0.70, -0.60 and -0.19 µm).
- Part 2, pooled per volume: every span, charge and MAD row passes. The u bias fails in volume 9 (pixel endcap, +z):
  gun -0.991 µm, reference 0.022 µm, difference -1.013 µm, σ_diff 0.442 µm, limit max(1 µm, 0.884 µm) = 1.0 µm, with 483
  gun hits on the support. The pooled u bias of volume 12 passes (difference -0.075 µm, σ_diff 0.415 µm).
- No volume or quantity is "not informative". Part 3 (support) is in the record.

The descriptive C1 of the truth checks (`c1_overlay`, the same 81 bins) shows the same volume-12 bias.

## Content, V3 and navigator losses (descriptive)

| Quantity | H-val |
|---|---|
| gun particles, with hits | 3,994, 3,793 |
| gun hits | 20,039 (13.4 per LLP); 20,902 Fatras hits before the emulation, 636 removed by stops, 227 by inefficiency |
| targets | 1,517 |
| P_P targets | 793, i.e. 0.661 per isolated LLP (plan 0.603; pilot 0.45) |
| P_D targets | 508, i.e. 1.69 per dense LLP (plan 1.687) |
| smallest ΔR between targets of different LLPs | 0.037; 2 targets within 0.05 |
| modules holding gun and host hits | 16,972 |
| gun hits with another particle's hit on the module within 1 mm / 0.1 mm | 1.73 % / 0.10 % (host hits only: 0.92 % / 0.025 %); native targets 5.49 % / 0.10 % |
| navigator losses | 77 of 3,994 gun particles (1.93 %); P_P-like lost share 41 / 1,768 = 2.32 % |
| silent world exits | 22 of 3,913 simulated (0.56 %) |
| premature ends, P_P-like (Fatras hits / stored hits) | 0.29 % / 4.34 % |
| particles with a hole, P_P-like (Fatras / stored) | 8.2 % / 10.6 % |

Records: `h_t05_truth_checks_hval_v1.json` (content, V3, near coincidences) and `h_t05_truth_gun_content_hval_v1.json`
(losses in the bins of section 7, silent exits, premature ends and holes).

V3, mean cells per hit by region (truth checks, key `v3.by_region`; native side the C1 reference events):

| Region | gun | gun P_P targets | native targets | native displaced targets |
|---|---|---|---|---|
| pixel barrel | 5.92 | 5.71 | 7.09 | 5.37 |
| pixel endcap | 2.84 | 2.88 | 1.54 | 2.27 |
| short-strip barrel | 1.84 | 1.69 | 1.52 | 1.50 |
| short-strip endcap | 1.90 | 1.52 | 1.22 | 1.22 |
| long-strip barrel | 1.67 | 1.39 | 1.57 | 1.50 |
| long-strip endcap | 1.82 | 1.37 | 1.43 | 1.35 |

Species shares of gun particles with hits, against the generic train split of spec v1 (2 T HNL setup, no emulation;
`h_t05_truth_species_v1.json`):

| Component | Sample | n | e | mu | pi | K | p |
|---|---|---|---|---|---|---|---|
| isolated | H-val | 2,271 | 0.200 | 0.189 | 0.198 | 0.192 | 0.222 |
| isolated | generic train | 17,058 | 0.181 | 0.197 | 0.205 | 0.206 | 0.211 |
| dense | H-val | 1,522 | 0.050 | 0.047 | 0.704 | 0.145 | 0.053 |
| dense | generic train | 47,697 | 0.045 | 0.048 | 0.697 | 0.156 | 0.055 |

Among isolated targets the electron share is 0.178 on H-val and 0.129 in the generic train split.

## P_P yield and the reweighted projection of section 6

From the stored H-val truth (`hnl.overlay.h1.gun_targets`; `h_t05_truth_rw_projection_v1.json`):

- 793 P_P targets on 150 hosts. Scaled to the 600 H-test hosts: about 3,172 (Poisson σ about 113), above the 1,000 of
  the statistics check. The H-test count itself is committed at H-T1.
- By region: central 373 (47.0 %), forward +z 222, forward -z 198. Central targets alone would number about 1,492 on
  600 hosts.
- 69 P_P targets have a sibling of their LLP with pT <= 1 GeV and at least one hit.
- \|d0\|/r_first >= 0.2 for 45.9 % (gun probe 45.2 %); \|d0\| >= 100 mm for 21.2 % (probe 21.1 %).

Gun shares of the reweighting cells (r_first by \|d0\|/r_first), H-val against the gun probe of the addendum:

| r_first [mm] | \|d0\|/r_first < 0.05 | 0.05-0.2 | >= 0.2 |
|---|---|---|---|
| < 200 | 0.049 (0.052) | 0.092 (0.080) | 0.202 (0.193) |
| 200-700 | 0.172 (0.179) | 0.226 (0.235) | 0.255 (0.257) |
| >= 700 | 0.001 (0.000) | 0.001 (0.001) | 0.003 (0.002) |

The reweighted projection, repeated with these shares (native side and method unchanged from
`addendum_h_inputs/rw_projection.py`; natives cover 99.5 % of the gun weight):

| Reweighted native val targets | n | Δ [points] | σ, val | σ, projected (3,172 / 2,894 / 2,160) | holds / inconclusive / fails, 3,172 targets |
|---|---|---|---|---|---|
| all generations | 3,747 | -3.44 | 0.86 | 0.47 / 0.50 / 0.57 | 0.00 / 0.06 / 0.94 |
| generation 0 | 3,276 | -1.75 | 1.34 | 0.20 / 0.21 / 0.25 | 0.20 / 0.20 / 0.60 |

The addendum's gun-probe shares gave -3.4 (all generations) and -1.8 (generation 0). The H-val gun truth after the
emulation does not move the projection.

## Where this leaves H-T0.5

The C1 confirmation is a decisive gate, and it failed on pass 1, before any model run. Under section 9.3 this pass
counts, and one emulation or digitiser change may be made, under rules 1-6 (the change names the failing quantity and
its truth-level discrepancy against TrackML on the C1 sample or the H-val gun truth, is committed with its derivation
before the rerun, and gives a new overlay_hash; a failure present before the emulation, from Fatras, the v15 material or
the geometry, is not fixed through the digitiser). Whether to make such a change, or whether the cause lies before the
digitiser, is a decision that section 9.3 does not take mechanically. It was not taken here. The closure model run
(step 2) was not started, and the H-val record of decision D7 cannot be written for this pass (`hnl.overlay.d7record`
refuses a failed decisive gate before the third pass).
