One-time test, frozen models (post-freeze summaries of the one-time evaluation, not claims; nothing chosen on test). Primary / HNL-daughter / W-muon efficiencies and fake rate on the pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events), ttbar prompt on ttbar_test (1,000 events); every cut and configuration fixed on val (FR* = 0.029368, pooled HNL val); mean ± seed sd over the listed seeds.

| rank | model | seeds | primary eff. | HNL daughters | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Canto-H (full budget): hand-cut helix union, walkthrough + merging (rule R6) | 3 | 0.9918 ± 0.0012 | 0.9930 ± 0.0010 | 0.984 | 99/108 | 0.9976 ± 0.0001 | 0.9622 ± 0.0014 | 0.0142 ± 0.0006 |
| 2 | Canto-L (full budget): learned triplet union, learned merging (rule R6) | 3 | 0.9908 ± 0.0017 | 0.9916 ± 0.0016 | 0.986 | 94/108 | 0.9977 ± 0.0001 | 0.9632 ± 0.0034 | 0.0102 ± 0.0003 |
| 3 | Canto-B (full budget): base graph, learned merging (rule R6) | 3 | 0.9843 ± 0.0038 | 0.9869 ± 0.0026 | 0.970 | 97/108 | 0.9978 ± 0.0001 | 0.9658 ± 0.0004 | 0.0129 ± 0.0007 |
| 4 | acorn IGNN, HNL-trained + CC (best cut <= FR*) | 3 | 0.9448 ± 0.0136 | 0.9488 ± 0.0112 | 0.931 | 99/108 | 0.9851 ± 0.0033 | 0.8448 ± 0.0229 | 0.0290 ± 0.0002 |
| 5 | acorn IGNN, HNL-trained + CC (prereg. cut rule) | 3 | 0.9257 ± 0.0057 | 0.9306 ± 0.0067 | 0.917 | 98/108 | 0.9762 ± 0.0047 | 0.8167 ± 0.0032 | 0.0286 ± 0.0007 |
| 6 | acorn IGNN, ttbar-trained + CC (prereg. cut rule) | 3 | 0.9038 ± 0.0164 | 0.9183 ± 0.0123 | 0.849 | 86/108 | 0.9844 ± 0.0016 | 0.8741 ± 0.0085 | 0.0295 ± 0.0005 |
| 7 | acorn IGNN, ttbar-trained + CC (best cut <= FR*) | 3 | 0.9038 ± 0.0164 | 0.9183 ± 0.0123 | 0.849 | 86/108 | 0.9844 ± 0.0016 | 0.8741 ± 0.0085 | 0.0295 ± 0.0005 |

Working points (seeds 0 / 1 / 2, all fixed on val; "best cut <= FR*" = the cut chosen on val by that rule, applied unchanged to test):

- Canto-L: mwslmerge 0.05 (trackbuild_test, configuration 0); mwslmerge 0.03 (trackbuild_test, configuration 0); mwslmerge 0.1 (trackbuild_test, configuration 0)
- Canto-H: walkmerge 0.2,0.7,1.0 (trackbuild_test, configuration 0); walkmerge 0.2,0.5,1.0 (trackbuild_test, configuration 0); walkmerge 0.2,0.5,1.0 (trackbuild_test, configuration 0)
- Canto-B: mwslmerge 0.05 (trackbuild_test, configuration 0); mwslmerge 0.2 (trackbuild_test, configuration 0); mwslmerge 0.1 (trackbuild_test, configuration 0)
- acorn (HNL) CC: CC 0.26; CC 0.36 (FR* unreachable on val: minimum-rate cut); CC 0.27
- acorn (HNL) CC*: CC 0.38 (post-freeze CC on the stored test scores); CC 0.36 (FR* unreachable on val: minimum-rate cut); CC 0.44 (post-freeze CC on the stored test scores)
- acorn (ttbar) CC: CC 0.34 (FR* unreachable on val: minimum-rate cut); CC 0.33; CC 0.31 (FR* unreachable on val: minimum-rate cut)
- acorn (ttbar) CC*: CC 0.34 (FR* unreachable on val: minimum-rate cut); CC 0.33; CC 0.31 (FR* unreachable on val: minimum-rate cut)

Primary efficiency with the preregistered total sigma (event bootstrap of the test sets + seed term, hnl.eval.bootstrap) and the seed-pooled count k / n (every seed has the same particles, so k / n equals the seed mean):

- Canto-H: 0.9918 ± 0.0013 (bootstrap 0.0011, seed sd 0.0012); k / n = 14916/15039 = 0.9918
- Canto-L: 0.9908 ± 0.0015 (bootstrap 0.0012, seed sd 0.0017); k / n = 14900/15039 = 0.9908
- Canto-B: 0.9843 ± 0.0026 (bootstrap 0.0015, seed sd 0.0038); k / n = 14803/15039 = 0.9843
- acorn (HNL) CC*: 0.9448 ± 0.0083 (bootstrap 0.0028, seed sd 0.0136); k / n = 14209/15039 = 0.9448
- acorn (HNL) CC: 0.9257 ± 0.0047 (bootstrap 0.0034, seed sd 0.0057); k / n = 13922/15039 = 0.9257
- acorn (ttbar) CC: 0.9038 ± 0.0102 (bootstrap 0.0037, seed sd 0.0164); k / n = 13592/15039 = 0.9038
- acorn (ttbar) CC*: 0.9038 ± 0.0102 (bootstrap 0.0037, seed sd 0.0164); k / n = 13592/15039 = 0.9038

DeZoort IN / IN-L were never evaluated on test and have no row. The preregistered claims (results/claims_test_v1.json) stay as they are.

Identity checks against results/claims_test_v1.json (|difference| <= 1e-12):

- ok primary: eff_new (seed mean): 0.9907573641864484 (table) vs 0.9907573641864484 (claims)
- ok primary: eff_base (seed mean): 0.9257264445774321 (table) vs 0.9257264445774321 (claims)
- ok primary: difference: 0.0650309196090163 (table) vs 0.0650309196090163 (claims)
- ok primary guard: d eff W muons: 0.0214736624236780 (table) vs 0.0214736624236780 (claims)
- ok primary guard: d eff ttbar prompt: 0.1465186300338728 (table) vs 0.1465186300338728 (claims)
- ok S1: new - acorn (ttbar): 0.0869738679433473 (table) vs 0.0869738679433473 (claims)
- ok S1: new - reference: 0.0650309196090163 (table) vs 0.0650309196090163 (claims)
- ok S1 guard: d eff W muons: 0.0133498913380938 (table) vs 0.0133498913380938 (claims)
- ok S1 guard: d eff ttbar prompt: 0.0891230711328568 (table) vs 0.0891230711328568 (claims)
- ok S1b: fake rate of acorn (ttbar): 0.0294600470309288 (table) vs 0.0294600470309288 (claims)
