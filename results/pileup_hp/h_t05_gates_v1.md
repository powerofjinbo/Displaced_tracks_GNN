# Addendum H, H-T0.5 steps 3 and 4: the gates of pass 1 and section 9.3

Date: 2026-10-16 (UTC). Rules: `docs/preregistration_pileup.md` sections 7, 9.2, 9.3 and 10 (82fa6cc). Version: pass 1,
overlay_hash 3e23b3630f6b7ec54e0b3872e6ab25909fa4cd4f0f2c810ac1b9a434263df1f4 (emulation hash 2be64cc0...). Inputs:
the truth outputs of step 1 (cb79cdf), the V1, V1-Δ and V2 outputs of step 2 (1417d7f; sealed H-val manifest
a88f9e31..., 06c4560) and the logged section 9.3 decision after the C1 confirmation (DECISIONS 2026-10-16, 5f8ee56;
`configs/trackml/overlay_h_c1_flag_v1.json`, a74ac79). No model output of a gun target was read, the sealed directory
was not opened, and nothing of H-test exists.

**Result.** Every decisive gate passes except the C1 confirmation. Its only failed rows are the two that the logged
decision names, so it is carried as a flag. V1 (a) passes in all three regions, so the forward fallback (D12) does not
apply. Every flag gate passes. Pass 1 is the committed version, P_P is not restricted, H is not descriptive, and H1
carries one flag, "C1 confirmation (u bias, volumes 12 and 9)". The navigator-loss share is 41 / 1,768 = 0.0232. The
H-val record of decision D7 (`h_t05_d7_v1.json`, written by `hnl.overlay.d7record`) holds these four values, and the
step-5 runs read them from it alone. This file is a summary and is not a record of D7.

## Decisive gates

| Gate | Rule | Value | Outcome |
|---|---|---|---|
| Per-set truth checks, H-val | converter, weight sums (1e-6), containment (4 µm), host rows, version | all hold on 150 events; largest \|weight sum − 1\| 2.2e-7; largest move 3.79 µm | pass |
| Per-set truth checks, closure | the same, plus ids, populations and the converter view | all hold; populations (a) 3,276, (b) 448, (c) 3,000 | pass |
| C1 confirmation, part 1 | per-bin rules at matched incidence; \|mean u bias\| <= 1 µm per volume | 81 counted bins, none fails; volume 12 −1.77 µm (4 bins) | fail (volume 12) |
| C1 confirmation, part 2 | pooled per volume; u bias \|diff\| <= max(1 µm, 2 σ_diff) | 42 counted rows; volume 9 u bias −1.013 µm, limit 1.0 µm | fail (volume 9) |
| V1 (a) | \|Δ\| <= 1.0 point in all; <= 1.5 in each region | +0.165 (σ 0.174; 3,036 pairs); central −0.256, forward +z +0.104, forward −z +0.778 | pass |
| V1 (c) | \|Δ\| <= max(0.2, 2 σ_paired), \|eta\| < 3 | −0.045 (σ 0.082; limit 0.2; 2,241 pairs) | pass |
| V1-Δ (a) | \|Δ_closure − Δ_native\| <= max(0.5, 2 σ_paired) | 0.000 (both −0.560; σ 0.162; limit 0.5) | pass |

The two failed C1 rows are those named in the logged decision (part 1, mean u bias of volume 12; part 2, u bias of
volume 9). The investigation of 2026-10-16 traced both to where fixed Fatras crossings fall within the cells, a
statistical fluctuation, and found no admissible change under rules 1, 3 and 5 of section 9.3. Under reading A the C1
confirmation is reported as a flag. Any other failed C1 row would have been a decisive failure; there is none.

## Flag gates

| Gate | Rule | Value | Outcome |
|---|---|---|---|
| Closure truth gate | \|Δ hits per crossed layer\| <= 0.05 in cells below \|eta\| = 3 with >= 2,000 crossed layers per side | 9 counted cells, largest 0.010 | pass |
| V1 (b) | \|Δ\| <= max(3, 2 σ_paired) | −0.274 (σ 2.106; limit 4.21; 365 pairs) | pass |
| V1 (d) | \|Δ\| <= max(3, 2 σ_paired) | −0.929 (σ 2.505; limit 5.01; 323 pairs) | pass |
| V2 at WP-A | prompt efficiency and fake rate within 0.05 points, TrackML score within 0.001 | −0.0013 points, +0.0053 points, −1.7e-5 | pass |

## Section 9.3

- Restriction of P_P: none (D12 does not apply).
- Descriptive: no; H1 is decided.
- Flags carried by H1: "C1 confirmation (u bias, volumes 12 and 9)".
- Passes made: one. Rule 6: the first version that passes every decisive gate is committed with no further change.
- Navigator-loss share (section 7): 0.02319.

## Reported without a rule

- V1-Δ (a) and V1-Δ (d) are exactly zero. Their value equals V1 in C-full minus V1 in C-1GeV, and the net discordance
  between closure and native targets (+5 of 3,036 pairs in (a)) is the same in both conditions.
- Unconditional V1 (a), with a re-simulated target below 5 hits counted as not found: −7.05 points (σ 0.48). The
  re-simulated version has fewer than 5 hits for 7.3 % of the (a) natives and 7.1 % of the (c) natives.
- The density bins of section 7 are committed in `configs/trackml/overlay_version_h_v1.json`
  (`h_t05_truth_density_bins_v1.json`).
