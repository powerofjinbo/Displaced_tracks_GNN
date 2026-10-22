# HNL-daughter efficiency in |d0| and r_v bins, four models of paper Fig. 5 (test_binned_v1)

`python -m hnl.eval.test_binned --split test` (module docstring for the method). Ledger R17 (docs/feedback/test_use_ledger.md): post-freeze, descriptive read-out of stored test outputs; no network run, no track building, nothing chosen on test, no claim.

- Set: pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external); 7999 events read (with ttbar_test for the identity checks); replicate tables: preregistered test tables (B = 500, seed 12345).
- Population: HNL-daughter targets (pT > 1 GeV, |eta| < 4, >= 5 space points); |d0| and r_v bins of docs/definitions.md section 4 (truth-hit fit; r_v = two-track vertex radius, "no vertex" when the pair has none).
- value = mean of the three seeds; combined = sqrt(boot sd^2 + seed sd^2 / 3) (Eq. (2) of the paper); 68 % CP = Clopper-Pearson on the per-seed count.
- Models: Canto-L = Canto-L (frozen: learned merging); acorn (ttbar) LM = acorn (ttbar) + MWS-layer + learned merging (Canto-L's builder); acorn (ttbar+HNL) LM = acorn (ttbar+HNL) + MWS-layer + learned merging (Canto-L's builder); acorn (HNL) LM = acorn (HNL) + MWS-layer + learned merging (Canto-L's builder); working points per seed in the CSV.
- Run: job 6848945, host submit23.mit.edu, 2026-10-17T23:10:17Z to 2026-10-17T23:10:54Z; git 9f5f8e4f618c81bd6b3b93e38f12906b8ddc05bc (src/slurm: 2 changed or new files); src/hnl/eval/test_binned.py sha256 35138178b9f59985a18408da8fdb6feb4d227e71d4590a99c6f34cf3c3e70881; slurm/feedback/test_binned.sbatch sha256 4d7960b2bf6059d67b4fbffcddbdecb0551afaee6fbc11b5066ac68983b09787.

## Efficiency [%] in the |d0| bins [mm]

Cells: value [68 % CP] (combined sigma, points); n = HNL daughters per seed.

| bin | n | Canto-L | acorn (ttbar) LM | acorn (ttbar+HNL) LM | acorn (HNL) LM |
|---|---|---|---|---|---|
| 0-0.1 | 95 | 99.65 [97.55, 100.00] (0.50) | 100.00 [98.08, 100.00] (0.00) | 100.00 [98.08, 100.00] (0.00) | 100.00 [98.08, 100.00] (0.00) |
| 0.1-1 | 506 | 99.21 [98.59, 99.59] (0.37) | 98.81 [98.11, 99.28] (0.43) | 98.75 [98.03, 99.23] (0.48) | 98.75 [98.03, 99.23] (0.46) |
| 1-3 | 650 | 99.33 [98.84, 99.64] (0.30) | 98.56 [97.93, 99.02] (0.44) | 98.56 [97.93, 99.02] (0.48) | 98.72 [98.11, 99.15] (0.45) |
| 3-10 | 1403 | 99.36 [99.07, 99.57] (0.26) | 99.05 [98.71, 99.31] (0.26) | 99.07 [98.74, 99.33] (0.24) | 99.14 [98.82, 99.39] (0.25) |
| 10-30 | 2013 | 99.45 [99.23, 99.62] (0.15) | 98.10 [97.74, 98.40] (0.28) | 98.36 [98.03, 98.64] (0.27) | 98.72 [98.42, 98.97] (0.21) |
| 30-100 | 2009 | 99.04 [98.77, 99.25] (0.22) | 95.67 [95.17, 96.12] (0.80) | 97.03 [96.60, 97.41] (0.44) | 97.69 [97.31, 98.03] (0.34) |
| 100-300 | 584 | 98.57 [97.90, 99.06] (0.39) | 91.84 [90.54, 92.98] (1.80) | 95.49 [94.46, 96.35] (1.01) | 96.75 [95.84, 97.48] (0.59) |
| >= 300 | 36 | 87.04 [78.85, 92.70] (6.28) | 88.89 [80.97, 94.13] (4.50) | 90.74 [83.13, 95.50] (4.40) | 93.52 [86.45, 97.43] (3.43) |

## Efficiency [%] in the r_v bins [mm]

Cells: value [68 % CP] (combined sigma, points); n = HNL daughters per seed.

| bin | n | Canto-L | acorn (ttbar) LM | acorn (ttbar+HNL) LM | acorn (HNL) LM |
|---|---|---|---|---|---|
| 0-10 | 599 | 99.55 [99.08, 99.81] (0.27) | 99.67 [99.23, 99.88] (0.24) | 99.67 [99.23, 99.88] (0.24) | 99.67 [99.23, 99.88] (0.24) |
| 10-30 | 1050 | 99.46 [99.13, 99.68] (0.23) | 99.11 [98.72, 99.40] (0.24) | 99.14 [98.75, 99.42] (0.26) | 99.30 [98.94, 99.55] (0.21) |
| 30-100 | 2278 | 99.33 [99.11, 99.50] (0.18) | 97.70 [97.34, 98.02] (0.40) | 98.45 [98.14, 98.71] (0.27) | 98.68 [98.40, 98.92] (0.24) |
| 100-200 | 1494 | 99.40 [99.12, 99.59] (0.21) | 97.19 [96.69, 97.62] (0.60) | 97.79 [97.34, 98.17] (0.48) | 98.53 [98.15, 98.84] (0.37) |
| 200-300 | 848 | 99.41 [99.01, 99.66] (0.22) | 95.72 [94.90, 96.41] (0.78) | 97.05 [96.35, 97.63] (0.70) | 97.92 [97.30, 98.40] (0.45) |
| >= 300 | 468 | 99.64 [99.11, 99.89] (0.20) | 95.66 [94.50, 96.60] (1.27) | 97.22 [96.24, 97.98] (0.72) | 97.72 [96.80, 98.40] (0.59) |
| no vertex | 559 | 96.12 [95.12, 96.94] (0.88) | 92.25 [90.95, 93.39] (1.18) | 93.98 [92.80, 94.99] (0.93) | 94.57 [93.44, 95.54] (0.93) |

## Identity checks

- 622 of 622 passed (per-seed entries and model rows of results/strong_baselines_v3_entries.csv and results/strong_baselines_v3.csv; Canto-L against results/bootstrap_v1.csv (test_release_card); bin sums; the same particles for every model).

## Stored outputs read (sha256)

- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final_learned/l1tl/armi_s0/trackbuild_test/per_particle.parquet 3909ccc2bbb414db64d6788f59932137c358aadf6fda60e29ca17d11bd0b9ee0
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final_learned/l1tl/armi_s0/trackbuild_test/per_event.parquet 3c51d397dbc9f0439401c9fb396b1327964bb170facc80d6e708e27d01e1458d
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final_learned/l1tl/armi_s1/trackbuild_test/per_particle.parquet 7cfb3aad2e3af2b1f5299cdbdea449c828039ed916121b5d3583b9bb5fd9bb8e
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final_learned/l1tl/armi_s1/trackbuild_test/per_event.parquet 4586b7133f8b55df27a11cff9f3c1b3663bb837c444532edfc3ef27f88dde410
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final_learned/l1tl/armi_s2/trackbuild_test/per_particle.parquet 239a042debf4c0475a5a8e42a6f7a8de8143777163cb6e0c61827acf5b6ec6b9
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final_learned/l1tl/armi_s2/trackbuild_test/per_event.parquet b7e67d980bf88bcd049875624649610041e187d2c57b3b42e444d6c2c771bf56
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armi_s0/lmerge_test/per_particle.parquet 2b310c0f3fd39b15ab325ebcbaf85f8dd137aa5939e7c3826e2b4a1738334841
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armi_s0/lmerge_test/per_event.parquet f53c68e9b88aff43a799bf3845eb632ec034850fb60a16e7c11ee61f9b8e9437
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armi_s1/lmerge_test/per_particle.parquet fdea6570594b3e712a2ed852ecefc6ae9995791ec02317d1efb73e0df53508a3
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armi_s1/lmerge_test/per_event.parquet 9bba8e0f611e59567e8fdad1312f4346491527b2d6f2d6d8d563361c40819e54
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armi_s2/lmerge_test/per_particle.parquet 59c4782ade6b21494150ffe9e97159e550970d30fc2c3c9f41d74fc77349379c
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armi_s2/lmerge_test/per_event.parquet aaaf3bfa553c3a2742509f8f041b886934557c2283063e0913ada331744f49bf
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armiii_s0/lmerge_test/per_particle.parquet 7c2b5e579b5069dbd1c95f827f28e2c35829c74859218b10576c1b2ae92a906f
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armiii_s0/lmerge_test/per_event.parquet bfb514a79676230098f81fc02a50fa9ab20c96cb10b818841abb0bfd118af9c9
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armiii_s1/lmerge_test/per_particle.parquet cf314ce37f9d03e3623d64b08775e68ea72598dc03de35d681b186ed538e404c
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armiii_s1/lmerge_test/per_event.parquet 032c1f38b4a27caec0394eabe920e9237b4ad06b4ed8b1f0e5d70867e8d0760b
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armiii_s2/lmerge_test/per_particle.parquet 41007a406823efcf426e93e1993c1f106d04c6e66e512662802f03f80dcf84a1
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armiii_s2/lmerge_test/per_event.parquet f2fd7f3b74d1d0086950030e5c88cc0557877b915f5068fbd587470d35fe5550
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armii_s0/lmerge_test/per_particle.parquet c3069530bbd08f4c3db72cd72daef85ce2e7ff6ea23cd4cc05faadc9f3fb2640
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armii_s0/lmerge_test/per_event.parquet bb6cfd9cbb41ccd1b3b8fba4e69a4fa2205b26e9e6678e8ec2180aeca39bcd9c
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armii_s1/lmerge_test/per_particle.parquet f2fd43f66403084423d285b6d6c11dec0eff32fcd7044682c6bc42f1614006bd
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armii_s1/lmerge_test/per_event.parquet 0f0155b1ed2f344442dbc04e2a3b3133462b61005cdc841f3660c4c288c755bf
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armii_s2/lmerge_test/per_particle.parquet dab2ea4fb884febefd40b53315801bf466dff790eb7994ac02d779da2cd737f5
- /ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v3/strong_baselines_a2/armii_s2/lmerge_test/per_event.parquet 3a557bb107dcbf671d49f379179ec38df5edd5f622d6a0f0663119ab9758f1a8
