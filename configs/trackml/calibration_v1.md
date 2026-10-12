# TrackML calibrations v1 (plan sections 6.5 and 6.6)

- Made by `python -m hnl.trackml.calibrate` (`src/hnl/trackml/calibrate.py`; its docstring defines every
  estimator), Slurm job 6688288, 2026-10-05T07:47:49+00:00.
- Events: the 200 calibration events of `configs/trackml/split_tp_v1.json` (the 100 throughput dev ids + the next 100
  training ids), ids 21000 ... 21339, sha256 of the comma-joined ids
  `1e05a332b630`. Tables: `derived/tp_v1`. Only these training events were read; no val or test
  event.

Summary:
- **Field map.** The median B_eff per |z| band agrees with the accuracy-phase feasibility values (table below):
  2.00 T at |z| < 250 mm, falling to 0.37 T at 2.5-3 m.
- **RES.** The TrackML sigma is below the ttbar table's in 14 of 18 entries; above in 9 u, 16 v, 17 v, 18 v. Pixel
  barrel u: 0.00384 against 0.02034 mm. The barrel volumes show u offsets (medians) of 9-15 um.
- **Slope window.** At most 0.016% of the true
  target triplets fall outside it in any |eta| bin (limit 0.5 %). No bin is widened.
- **Merge cuts.** The union fit rejects 0.1%-95.7% of true unions per |eta| bin
  (more than 1 % in 8 of 10 bins). In every bin above 1 %, the outlier cut (n_out <= 1) accounts for at least
  99% of the rejections. They fall steeply with pT (multiple scattering; table at the end), and in
  1.2 < |eta| < 2.4 the rms cut adds to them (the field). The plan's rule changes only max_rms / max_zrms, so it does
  not restore the acceptance (accept_plan). This needs a decision.

## Field map (`field_map_v1.json`)

- 277,100 truth triplets (pT at the middle point > 2 GeV, transverse legs > 5 mm).
- 50 mm x 100 mm bins in (r, |z|), r 0-1100 mm, |z| 0-3100 mm; median per bin; empty bins take the nearest filled
  bin. `hnl.trackml.events.FieldMap` reads the file.
- Positive charges turn clockwise in 98.3% of the triplets (Bz > 0, as the
  converter assumes).
- Split-half stability (even against odd calibration events, 185 bins with >= 50 triplets in
  each half): max |dB| 0.1091 T, 95th percentile 0.0438 T.

Median B_eff per |z| band (all r), against the accuracy-phase feasibility values:

| absz_mm | n | B_median_T | B_p16_T | B_p84_T | feasibility_accuracy_phase_T |
|---|---|---|---|---|---|
| 0-250 | 92920 | 2 | 1.925 | 2.069 | 2.00 |
| 250-500 | 49601 | 1.983 | 1.918 | 2.043 | 1.98 |
| 500-750 | 36519 | 1.947 | 1.849 | 2.032 | 1.94 |
| 750-1000 | 26815 | 1.881 | 1.726 | 2 | 1.88 |
| 1000-1250 | 22015 | 1.78 | 1.618 | 1.914 | 1.78 |
| 1250-1500 | 14511 | 1.584 | 1.319 | 1.798 | 1.6 |
| 1500-2000 | 14966 | 1.332 | 1.022 | 1.564 | 1.3 |
| 2000-2500 | 10349 | 0.743 | 0.293 | 1.087 | 0.7-0.8 |
| 2500-3000 | 9404 | 0.371 | 0.125 | 0.712 | 0.24-0.50 |

## Resolution table RES (`res_v1.json`)

Hit minus true intersection in the module frame of `hnl.geometry.resolution` (u: the tangent at the module-centre
azimuth; v: z in the barrel, radial in the endcaps). sigma = 1.4826 MAD per volume [mm]. The ttbar values
(`hnl.geometry.resolution.RES`, what the code uses now) are shown for comparison, with the median (bias) and the
standard deviation. The detectors.csv local axes give the same sigmas within 0.25 um (`local_*` in
`res_v1.json`).

| volume | n_hits | s_u_mm | s_v_mm | ttbar_s_u_mm | ttbar_s_v_mm | median_u_mm | median_v_mm | std_u_mm | std_v_mm |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 2936216 | 0.01724 | 0.00944 | 0.01826 | 0.02 | 0 | 0 | 0.01363 | 0.00941 |
| 8 | 4842442 | 0.00384 | 0.004 | 0.02034 | 0.02467 | 0.00897 | 0 | 0.00616 | 0.00944 |
| 9 | 2930104 | 0.01724 | 0.00943 | 0.01594 | 0.01599 | -0.00001 | 0 | 0.01362 | 0.00941 |
| 12 | 1217035 | 0.01828 | 0.35966 | 0.02661 | 0.39927 | 0.00002 | 0.00038 | 0.01709 | 0.30073 |
| 13 | 3720755 | 0.0168 | 0.29201 | 0.02608 | 0.39965 | -0.0149 | 0 | 0.01746 | 0.26617 |
| 14 | 1209127 | 0.01872 | 0.35925 | 0.02722 | 0.38988 | 0.00001 | 0.00009 | 0.01744 | 0.30032 |
| 16 | 394279 | 0.02561 | 3.67566 | 0.0347 | 3.19191 | 0 | 0.0286 | 0.02428 | 2.8958 |
| 17 | 849857 | 0.02591 | 3.86812 | 0.0346 | 3.12244 | -0.01388 | 0 | 0.02659 | 3.03842 |
| 18 | 400800 | 0.02571 | 3.67452 | 0.0347 | 3.21425 | 0.00007 | 0.04199 | 0.02451 | 2.89689 |

## Candidate slope window (`slope_window_v1.tsv`)

- True target triplets with usable legs: 1,128,152. The table gives the
  fraction outside the window of `hnl.graphs.helix_union.candidate_triplets` per |eta| bin, with the ttbar resolutions
  the code uses and with RES v1.
- c_995: the constant term that would leave exactly 0.5 % outside. It is negative where the window holds 99.5 % even
  without the constant term.
- c0: the constant term to use (0.05 in the code), widened only where more than 0.5 % fall outside.
- `n_excluded_*`: true triplets the support never forms (a same-layer leg; the middle hit at an extremum of R).

| abs_eta | n_triplets | n_excluded_same_layer | n_excluded_R_extremum | frac_outside_ttbar_res | frac_outside_trackml_res | c_995_ttbar_res | c0_ttbar_res | c0_trackml_res |
|---|---|---|---|---|---|---|---|---|
| 0.0-0.4 | 158519 | 94917 | 103 | 0.00002 | 0.00002 | -0.00141 | 0.05 | 0.05 |
| 0.4-0.8 | 157680 | 94360 | 2125 | 0.00012 | 0.00012 | -0.0013 | 0.05 | 0.05 |
| 0.8-1.2 | 145008 | 90199 | 1557 | 0.00009 | 0.00009 | -0.00129 | 0.05 | 0.05 |
| 1.2-1.6 | 139697 | 83070 | 748 | 0.00016 | 0.00016 | -0.00135 | 0.05 | 0.05 |
| 1.6-2.0 | 145200 | 73211 | 210 | 0.00008 | 0.00008 | -0.00143 | 0.05 | 0.05 |
| 2.0-2.4 | 144084 | 69890 | 24 | 0.00006 | 0.00006 | -0.00144 | 0.05 | 0.05 |
| 2.4-2.8 | 125329 | 85682 | 1 | 0.00002 | 0.00002 | -0.00165 | 0.05 | 0.05 |
| 2.8-3.2 | 80404 | 105026 | 0 | 0 | 0 | -0.0011 | 0.05 | 0.05 |
| 3.2-3.6 | 27975 | 147578 | 0 | 0 | 0 | -0.00149 | 0.05 | 0.05 |
| 3.6-4.0 | 4256 | 168216 | 0 | 0 | 0 | -0.00156 | 0.05 | 0.05 |

Bins widened: none.

## Fragment-merging union fit (`merge_cuts_v1.tsv`)

- True target tracks: 221,593, split at every gap with >= 2 hits on each side:
  1,923,476 unions. The union of the two fragments is the whole track, so each track is
  fitted once (`hnl.vertex.pairing.fit_tracks`, as `hnl.trackbuild.merge_fragments._merge` does).
- frac_fail counts unions; fail_* splits it by condition: rms > 1 mm, zrms > 5 mm, n_out > 1 (hits beyond 3 Huber
  scales), R span < 50 mm.
- Where frac_fail > 1 %:
  - accept_plan: both cuts set to the bin's 99th percentiles (the plan's rule);
  - accept_widen_only: the cuts only raised (max of the default and the 99th percentile).
- diag_*: a diagnostic beyond the plan. diag_max_out_99 is the smallest max_out that keeps 99 % of the unions under
  it; diag_accept_* is the acceptance with that max_out and the widened cuts.

| abs_eta | n_tracks | n_unions | frac_fail | fail_rms | fail_zrms | fail_n_out | fail_span | p99_rms_mm | p99_zrms_mm | accept_plan | accept_widen_only | p99_n_out | diag_max_out_99 | diag_accept_widen_and_max_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.4 | 26243 | 227296 | 0.13867 | 0.00122 | 0.00112 | 0.13859 | 0.00007 | 0.286 | 2.4077 | 0.85328 | 0.86133 | 5 | 5 | 0.99136 |
| 0.4-0.8 | 26303 | 227862 | 0.12504 | 0.00187 | 0.00187 | 0.12488 | 0.00008 | 0.2736 | 2.3432 | 0.86782 | 0.87496 | 5 | 5 | 0.99297 |
| 0.8-1.2 | 25866 | 210898 | 0.27818 | 0.00407 | 0.00161 | 0.27814 | 0.00003 | 0.6287 | 2.9499 | 0.71745 | 0.72182 | 8 | 8 | 0.98917 |
| 1.2-1.6 | 24738 | 198777 | 0.78 | 0.14431 | 0.01215 | 0.77955 | 0 | 2.4679 | 5.1912 | 0.22009 | 0.22009 | 10 | 10 | 0.97987 |
| 1.6-2.0 | 23274 | 195347 | 0.95668 | 0.78588 | 0.28771 | 0.95502 | 0 | 11.9176 | 10.9492 | 0.0448 | 0.0448 | 11 | 11 | 0.97647 |
| 2.0-2.4 | 22342 | 191656 | 0.94136 | 0.4391 | 0.10009 | 0.9405 | 0 | 4.1961 | 8.1728 | 0.05925 | 0.05925 | 11 | 11 | 0.97686 |
| 2.4-2.8 | 20908 | 190104 | 0.89664 | 0.04995 | 0.02503 | 0.89612 | 0 | 1.2702 | 5.7207 | 0.10341 | 0.10341 | 10 | 10 | 0.97169 |
| 2.8-3.2 | 18640 | 166790 | 0.30597 | 0.00012 | 0.01022 | 0.30245 | 0 | 0.6665 | 5.0137 | 0.69406 | 0.69406 | 10 | 10 | 0.98069 |
| 3.2-3.6 | 17140 | 158413 | 0.00114 | 0.00001 | 0.00114 | 0.00042 | 0 | 0.026 | 1.675 | 0.99886 | 0.99886 | 0 | 1 | 0.99886 |
| 3.6-4.0 | 16139 | 156333 | 0.00052 | 0 | 0.00052 | 0.00009 | 0 | 0.0176 | 0.5821 | 0.99948 | 0.99948 | 0 | 1 | 0.99948 |

Bins above 1 %: 0.0-0.4, 0.4-0.8, 0.8-1.2, 1.2-1.6, 1.6-2.0, 2.0-2.4, 2.4-2.8, 2.8-3.2.

The failures come from the outlier count, not from the rms cuts that the plan calibrates. Fail fraction by pT and
|eta| region (unions; diagnostic):

| abs_eta | pt_gev | n_tracks | frac_fail | fail_n_out | median_n_out |
|---|---|---|---|---|---|
| 0.0-1.2 | 1-1.5 | 42682 | 0.267 | 0.2669 | 0 |
| 0.0-1.2 | 1.5-2 | 17603 | 0.1141 | 0.1141 | 0 |
| 0.0-1.2 | 2-3 | 11908 | 0.0453 | 0.045 | 0 |
| 0.0-1.2 | 3-5 | 4180 | 0.0059 | 0.0059 | 0 |
| 0.0-1.2 | 5-10 | 1297 | 0 | 0 | 0 |
| 0.0-1.2 | 10-inf | 742 | 0 | 0 | 0 |
| 1.2-2.8 | 1-1.5 | 52183 | 0.9271 | 0.9263 | 6 |
| 1.2-2.8 | 1.5-2 | 20209 | 0.8971 | 0.8962 | 5 |
| 1.2-2.8 | 2-3 | 12894 | 0.8569 | 0.8561 | 4 |
| 1.2-2.8 | 3-5 | 4487 | 0.744 | 0.7425 | 3 |
| 1.2-2.8 | 5-10 | 1071 | 0.4992 | 0.498 | 1 |
| 1.2-2.8 | 10-inf | 418 | 0.1525 | 0.1406 | 0 |
| 2.8-4.0 | 1-1.5 | 30820 | 0.1342 | 0.1319 | 0 |
| 2.8-4.0 | 1.5-2 | 11335 | 0.0888 | 0.0882 | 0 |
| 2.8-4.0 | 2-3 | 7006 | 0.0513 | 0.0508 | 0 |
| 2.8-4.0 | 3-5 | 2259 | 0.0128 | 0.0123 | 0 |
| 2.8-4.0 | 5-10 | 427 | 0.0035 | 0.0023 | 0 |
| 2.8-4.0 | 10-inf | 72 | 0 | 0 | 0 |
