# Wang et al.'s displaced denominator: candidate selections of our daughter targets (truth only)

Sample hnl15_100 (external, 4973 events with a daughter in the table; 4,999 events); daughter targets 5623; W muons with >= 5 hits 4744 (paper 4,744). No model output read. Script scripts/diagnostics/wang_displaced_selection.py, job 6810500, 2026-10-14 09:25 UTC.

## Totals (paper: 2,882 in the |d0| panel plus two bins drawn at 1.0 with unknown totals)

| candidate | n | events | n_minus_2882 |
|---|---|---|---|
| all daughter targets | 5623 | 3152 | 2741 |
| mu+ only | 2803 | 2803 | -79 |
| mu- only | 2820 | 2820 | -62 |
| higher-pT daughter (target) | 3022 | 3022 | 140 |
| lower-pT daughter (target) | 2601 | 2601 | -281 |
| highest-pT target per event | 3152 | 3152 | 270 |
| same charge as the W muon | 2817 | 2817 | -65 |
| opposite charge to the W muon | 2806 | 2806 | -76 |
| first daughter in barcode order | 2785 | 2785 | -97 |
| second daughter in barcode order | 2838 | 2838 | -44 |

## chi2 per panel and reading (bins with a known paper total; vertex panels scaled by the fraction of the candidate's daughters with a fitted vertex)

| candidate /  | pt / pt | d0 / truth |d0| | d0 / r_v | z0 / truth z0 | z0 / vertex z | radius / r_v | radius / 3D vertex distance |
|---|---|---|---|---|---|---|---|
| all daughter targets | 850.5 | 2369.9 | 825.7 | 1675.2 | 827.6 | 1998.8 | 574.3 |
| mu+ only | 4.0 | 1070.9 | 11.2 | 553.8 | 16.0 | 471.2 | 5.7 |
| mu- only | 6.5 | 1085.8 | 13.9 | 592.7 | 17.6 | 477.6 | 6.9 |
| higher-pT daughter (target) | 513.9 | 1575.7 | 13.3 | 898.6 | 15.9 | 527.1 | 11.4 |
| lower-pT daughter (target) | 675.9 | 682.4 | 26.7 | 335.8 | 32.8 | 429.4 | 7.5 |
| highest-pT target per event | 403.0 | 1479.2 | 20.9 | 795.1 | 19.2 | 552.4 | 13.4 |
| same charge as the W muon | 4.0 | 1075.5 | 14.4 | 577.7 | 15.4 | 472.2 | 5.9 |
| opposite charge to the W muon | 7.2 | 1077.4 | 10.1 | 569.0 | 18.0 | 476.4 | 6.6 |
| first daughter in barcode order | 22.2 | 1076.1 | 13.2 | 551.8 | 15.7 | 468.0 | 5.5 |
| second daughter in barcode order | 6.3 | 1075.6 | 11.5 | 593.6 | 18.3 | 481.0 | 7.6 |

## Prompt check (axis readings)

| population | candidate | panel | reading | bins | chi2 | ours_in_bins | paper_in_bins_scaled | scale | n_candidate | n_without_value |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | W muons, >= 5 hits | pt | pt | 7 | 0.0 | 1922 | 1919.0 | 1.0 | 4744 | 0 |
| prompt | W muons, >= 5 hits | z0 | truth z0 | 2 | 0.2 | 4699 | 4744.0 | 1.0 | 4744 | 0 |
| prompt | W muons, >= 5 hits | radius | |z0| | 6 | 0.5 | 4662 | 4712.0 | 1.0 | 4744 | 0 |
| prompt | W muons, >= 5 hits | radius | sqrt(d0^2 + z0^2) | 6 | 0.4 | 4662 | 4712.0 | 1.0 | 4744 | 0 |

## All chi2 rows

| population | candidate | panel | reading | bins | chi2 | ours_in_bins | paper_in_bins_scaled | scale | n_candidate | n_without_value |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | W muons, >= 5 hits | pt | pt | 7 | 0.0 | 1922 | 1919.0 | 1.0 | 4744 | 0 |
| prompt | W muons, >= 5 hits | z0 | truth z0 | 2 | 0.2 | 4699 | 4744.0 | 1.0 | 4744 | 0 |
| prompt | W muons, >= 5 hits | radius | |z0| | 6 | 0.5 | 4662 | 4712.0 | 1.0 | 4744 | 0 |
| prompt | W muons, >= 5 hits | radius | sqrt(d0^2 + z0^2) | 6 | 0.4 | 4662 | 4712.0 | 1.0 | 4744 | 0 |
| displaced | all daughter targets | pt | pt | 9 | 850.5 | 5424 | 2786.0 | 1.0 | 5623 | 0 |
| displaced | all daughter targets | d0 | truth |d0| | 13 | 2369.9 | 5622 | 2882.0 | 1.0 | 5623 | 0 |
| displaced | all daughter targets | d0 | r_v | 13 | 825.7 | 5203 | 2672.4 | 0.9273 | 5623 | 409 |
| displaced | all daughter targets | z0 | truth z0 | 19 | 1675.2 | 5614 | 2887.0 | 1.0 | 5623 | 0 |
| displaced | all daughter targets | z0 | vertex z | 19 | 827.6 | 5203 | 2677.0 | 0.9273 | 5623 | 409 |
| displaced | all daughter targets | radius | r_v | 12 | 1998.8 | 4907 | 1544.8 | 0.9273 | 5623 | 409 |
| displaced | all daughter targets | radius | 3D vertex distance | 12 | 574.3 | 3183 | 1544.8 | 0.9273 | 5623 | 409 |
| displaced | mu+ only | pt | pt | 9 | 4.0 | 2700 | 2786.0 | 1.0 | 2803 | 0 |
| displaced | mu+ only | d0 | truth |d0| | 13 | 1070.9 | 2803 | 2882.0 | 1.0 | 2803 | 0 |
| displaced | mu+ only | d0 | r_v | 13 | 11.2 | 2606 | 2684.6 | 0.9315 | 2803 | 192 |
| displaced | mu+ only | z0 | truth z0 | 19 | 553.8 | 2800 | 2887.0 | 1.0 | 2803 | 0 |
| displaced | mu+ only | z0 | vertex z | 19 | 16.0 | 2606 | 2689.2 | 0.9315 | 2803 | 192 |
| displaced | mu+ only | radius | r_v | 12 | 471.2 | 2456 | 1551.9 | 0.9315 | 2803 | 192 |
| displaced | mu+ only | radius | 3D vertex distance | 12 | 5.7 | 1591 | 1551.9 | 0.9315 | 2803 | 192 |
| displaced | mu- only | pt | pt | 9 | 6.5 | 2724 | 2786.0 | 1.0 | 2820 | 0 |
| displaced | mu- only | d0 | truth |d0| | 13 | 1085.8 | 2819 | 2882.0 | 1.0 | 2820 | 0 |
| displaced | mu- only | d0 | r_v | 13 | 13.9 | 2597 | 2660.2 | 0.923 | 2820 | 217 |
| displaced | mu- only | z0 | truth z0 | 19 | 592.7 | 2814 | 2887.0 | 1.0 | 2820 | 0 |
| displaced | mu- only | z0 | vertex z | 19 | 17.6 | 2597 | 2664.8 | 0.923 | 2820 | 217 |
| displaced | mu- only | radius | r_v | 12 | 477.6 | 2451 | 1537.8 | 0.923 | 2820 | 217 |
| displaced | mu- only | radius | 3D vertex distance | 12 | 6.9 | 1592 | 1537.8 | 0.923 | 2820 | 217 |
| displaced | higher-pT daughter (target) | pt | pt | 9 | 513.9 | 2823 | 2786.0 | 1.0 | 3022 | 0 |
| displaced | higher-pT daughter (target) | d0 | truth |d0| | 13 | 1575.7 | 3022 | 2882.0 | 1.0 | 3022 | 0 |
| displaced | higher-pT daughter (target) | d0 | r_v | 13 | 13.3 | 2715 | 2595.9 | 0.9007 | 3022 | 300 |
| displaced | higher-pT daughter (target) | z0 | truth z0 | 19 | 898.6 | 3021 | 2887.0 | 1.0 | 3022 | 0 |
| displaced | higher-pT daughter (target) | z0 | vertex z | 19 | 15.9 | 2715 | 2600.4 | 0.9007 | 3022 | 300 |
| displaced | higher-pT daughter (target) | radius | r_v | 12 | 527.1 | 2558 | 1500.6 | 0.9007 | 3022 | 300 |
| displaced | higher-pT daughter (target) | radius | 3D vertex distance | 12 | 11.4 | 1643 | 1500.6 | 0.9007 | 3022 | 300 |
| displaced | lower-pT daughter (target) | pt | pt | 9 | 675.9 | 2601 | 2786.0 | 1.0 | 2601 | 0 |
| displaced | lower-pT daughter (target) | d0 | truth |d0| | 13 | 682.4 | 2600 | 2882.0 | 1.0 | 2601 | 0 |
| displaced | lower-pT daughter (target) | d0 | r_v | 13 | 26.7 | 2488 | 2761.2 | 0.9581 | 2601 | 109 |
| displaced | lower-pT daughter (target) | z0 | truth z0 | 19 | 335.8 | 2593 | 2887.0 | 1.0 | 2601 | 0 |
| displaced | lower-pT daughter (target) | z0 | vertex z | 19 | 32.8 | 2488 | 2766.0 | 0.9581 | 2601 | 109 |
| displaced | lower-pT daughter (target) | radius | r_v | 12 | 429.4 | 2349 | 1596.2 | 0.9581 | 2601 | 109 |
| displaced | lower-pT daughter (target) | radius | 3D vertex distance | 12 | 7.5 | 1540 | 1596.2 | 0.9581 | 2601 | 109 |
| displaced | highest-pT target per event | pt | pt | 9 | 403.0 | 2953 | 2786.0 | 1.0 | 3152 | 0 |
| displaced | highest-pT target per event | d0 | truth |d0| | 13 | 1479.2 | 3151 | 2882.0 | 1.0 | 3152 | 0 |
| displaced | highest-pT target per event | d0 | r_v | 13 | 20.9 | 2805 | 2574.8 | 0.8934 | 3152 | 336 |
| displaced | highest-pT target per event | z0 | truth z0 | 19 | 795.1 | 3148 | 2887.0 | 1.0 | 3152 | 0 |
| displaced | highest-pT target per event | z0 | vertex z | 19 | 19.2 | 2806 | 2579.2 | 0.8934 | 3152 | 336 |
| displaced | highest-pT target per event | radius | r_v | 12 | 552.4 | 2610 | 1488.4 | 0.8934 | 3152 | 336 |
| displaced | highest-pT target per event | radius | 3D vertex distance | 12 | 13.4 | 1651 | 1488.4 | 0.8934 | 3152 | 336 |
| displaced | same charge as the W muon | pt | pt | 9 | 4.0 | 2718 | 2786.0 | 1.0 | 2817 | 0 |
| displaced | same charge as the W muon | d0 | truth |d0| | 13 | 1075.5 | 2817 | 2882.0 | 1.0 | 2817 | 0 |
| displaced | same charge as the W muon | d0 | r_v | 13 | 14.4 | 2600 | 2666.1 | 0.9251 | 2817 | 211 |
| displaced | same charge as the W muon | z0 | truth z0 | 19 | 577.7 | 2815 | 2887.0 | 1.0 | 2817 | 0 |
| displaced | same charge as the W muon | z0 | vertex z | 19 | 15.4 | 2603 | 2670.8 | 0.9251 | 2817 | 211 |
| displaced | same charge as the W muon | radius | r_v | 12 | 472.2 | 2452 | 1541.2 | 0.9251 | 2817 | 211 |
| displaced | same charge as the W muon | radius | 3D vertex distance | 12 | 5.9 | 1588 | 1541.2 | 0.9251 | 2817 | 211 |
| displaced | opposite charge to the W muon | pt | pt | 9 | 7.2 | 2706 | 2786.0 | 1.0 | 2806 | 0 |
| displaced | opposite charge to the W muon | d0 | truth |d0| | 13 | 1077.4 | 2805 | 2882.0 | 1.0 | 2806 | 0 |
| displaced | opposite charge to the W muon | d0 | r_v | 13 | 10.1 | 2603 | 2678.6 | 0.9294 | 2806 | 198 |
| displaced | opposite charge to the W muon | z0 | truth z0 | 19 | 569.0 | 2799 | 2887.0 | 1.0 | 2806 | 0 |
| displaced | opposite charge to the W muon | z0 | vertex z | 19 | 18.0 | 2600 | 2683.3 | 0.9294 | 2806 | 198 |
| displaced | opposite charge to the W muon | radius | r_v | 12 | 476.4 | 2455 | 1548.4 | 0.9294 | 2806 | 198 |
| displaced | opposite charge to the W muon | radius | 3D vertex distance | 12 | 6.6 | 1595 | 1548.4 | 0.9294 | 2806 | 198 |
| displaced | first daughter in barcode order | pt | pt | 9 | 22.2 | 2705 | 2786.0 | 1.0 | 2785 | 0 |
| displaced | first daughter in barcode order | d0 | truth |d0| | 13 | 1076.1 | 2785 | 2882.0 | 1.0 | 2785 | 0 |
| displaced | first daughter in barcode order | d0 | r_v | 13 | 13.2 | 2587 | 2682.3 | 0.9307 | 2785 | 193 |
| displaced | first daughter in barcode order | z0 | truth z0 | 19 | 551.8 | 2777 | 2887.0 | 1.0 | 2785 | 0 |
| displaced | first daughter in barcode order | z0 | vertex z | 19 | 15.7 | 2589 | 2686.9 | 0.9307 | 2785 | 193 |
| displaced | first daughter in barcode order | radius | r_v | 12 | 468.0 | 2437 | 1550.5 | 0.9307 | 2785 | 193 |
| displaced | first daughter in barcode order | radius | 3D vertex distance | 12 | 5.5 | 1579 | 1550.5 | 0.9307 | 2785 | 193 |
| displaced | second daughter in barcode order | pt | pt | 9 | 6.3 | 2719 | 2786.0 | 1.0 | 2838 | 0 |
| displaced | second daughter in barcode order | d0 | truth |d0| | 13 | 1075.6 | 2837 | 2882.0 | 1.0 | 2838 | 0 |
| displaced | second daughter in barcode order | d0 | r_v | 13 | 11.5 | 2616 | 2662.7 | 0.9239 | 2838 | 216 |
| displaced | second daughter in barcode order | z0 | truth z0 | 19 | 593.6 | 2837 | 2887.0 | 1.0 | 2838 | 0 |
| displaced | second daughter in barcode order | z0 | vertex z | 19 | 18.3 | 2614 | 2667.3 | 0.9239 | 2838 | 216 |
| displaced | second daughter in barcode order | radius | r_v | 12 | 481.0 | 2470 | 1539.2 | 0.9239 | 2838 | 216 |
| displaced | second daughter in barcode order | radius | 3D vertex distance | 12 | 7.6 | 1604 | 1539.2 | 0.9239 | 2838 | 216 |
