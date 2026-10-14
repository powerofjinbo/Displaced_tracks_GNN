# Addendum G: val inputs for a power estimate of claim G1 (generic_iso val; nothing chosen)

Written 2026-10-14 16:18 UTC by `hnl.eval.addendum_g_summary.val_sanity` (git 996c734), from the frozen models on the generic_iso val split (806 events, 956 P_G targets). Requested before any test read, for a check of the sample size; no model, configuration or definition is chosen from these numbers. Statistics as in the module docstring (B = 500, val bootstrap table default_rng(12349)).

## Per model and seed (P_G, generic_iso val)

| model | seed | k / n | efficiency [%] | track fake rate [%] |
|---|---|---|---|---|
| Canto-L (ttbar) | 0 | 924 / 956 | 96.65 | 0.48 |
| Canto-L (ttbar) | 1 | 925 / 956 | 96.76 | 0.47 |
| Canto-L (ttbar) | 2 | 924 / 956 | 96.65 | 0.61 |
| acorn (HNL) + LM | 0 | 897 / 956 | 93.83 | 1.28 |
| acorn (HNL) + LM | 1 | 899 / 956 | 94.04 | 1.52 |
| acorn (HNL) + LM | 2 | 897 / 956 | 93.83 | 1.29 |
| acorn (ttbar) + LM | 0 | 879 / 956 | 91.95 | 1.14 |
| acorn (ttbar) + LM | 1 | 856 / 956 | 89.54 | 1.08 |
| acorn (ttbar) + LM | 2 | 883 / 956 | 92.36 | 1.14 |
| acorn (HNL) + CC | 0 | 790 / 956 | 82.64 | 4.07 |
| acorn (HNL) + CC | 1 | 816 / 956 | 85.36 | 4.80 |
| acorn (HNL) + CC | 2 | 809 / 956 | 84.62 | 3.83 |

| model | 3-seed mean [%] | sigma_boot | s_seeds | sigma | fake rate [%] |
|---|---|---|---|---|---|
| Canto-L (ttbar) | 96.69 | 0.56 | 0.06 | 0.56 | 0.52 +- 0.06 |
| acorn (HNL) + LM | 93.90 | 0.76 | 0.12 | 0.76 | 1.37 +- 0.09 |
| acorn (ttbar) + LM | 91.28 | 0.83 | 1.52 | 1.21 | 1.12 +- 0.04 |
| acorn (HNL) + CC | 84.21 | 1.18 | 1.41 | 1.43 | 4.23 +- 0.30 |

## Paired differences (points)

| difference | Delta | sigma_boot | sigma_seeds | sigma | per seed | fake-rate difference | projected sigma_boot / sigma at n = 2914 |
|---|---|---|---|---|---|---|---|
| Canto-L (ttbar) - acorn (HNL) + LM | +2.79 | 0.57 | 0.08 | 0.58 | 2.824;2.720;2.824 | -0.85 | 0.33 / 0.34 |
| Canto-L (ttbar) - acorn (ttbar) + LM | +5.40 | 0.66 | 0.88 | 1.10 | 4.707;7.218;4.289 | -0.60 | 0.38 / 0.96 |
| Canto-L (ttbar) - acorn (HNL) + CC | +12.48 | 1.07 | 0.81 | 1.34 | 14.017;11.402;12.029 | -3.72 | 0.61 / 1.02 |

Projection: the bootstrap term scaled by sqrt(n_val / n_test) with n_test the P_G count of the generic_iso test split (T1), the seed term unchanged; it assumes the test split behaves like val and is not a result.
