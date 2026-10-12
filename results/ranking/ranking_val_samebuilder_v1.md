Pooled HNL val, fake-rate target FR* = 0.029368; mean ± seed sd over the listed seeds.

| rank | model | seeds | primary eff. | HNL daughters | \|d0\| 100-300 mm | \|d0\| >= 300 mm | W muons | ttbar prompt | fake rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | L3 triplets, HNL-trained (full budget, v1), MWS-layer + merging | 1 | 0.9940 | 0.9940 | 0.982 | 18/19 | 0.9973 | 0.9680 | 0.0122 |
| 2 | Canto-H (preview): hand-cut helix union, MWS-layer + merging | 1 | 0.9940 | 0.9940 | 0.988 | 17/19 | 0.9979 | 0.9606 | 0.0144 |
| 3 | Canto-L (preview): learned triplet union, MWS-layer + merging | 1 | 0.9917 | 0.9923 | 0.982 | 17/19 | 0.9979 | 0.9602 | 0.0114 |
| 4 | Canto-B (preview): base graph, MWS-layer + merging | 1 | 0.9872 | 0.9885 | 0.964 | 15/19 | 0.9979 | 0.9612 | 0.0145 |
| 5 | ladder: L3 triplets (absolute coords), ttbar, MWS-layer + merging | 1 | 0.9865 | 0.9885 | 0.958 | 15/19 | 0.9979 | 0.9638 | 0.0159 |
| 6 | acorn IGNN, HNL-trained + Canto builder (MWS-layer + merging) | 3 | 0.9857 ± 0.0020 | 0.9879 ± 0.0020 | 0.968 | 52/57 | 0.9975 ± 0.0006 | 0.9463 ± 0.0015 | 0.0147 ± 0.0004 |
| 7 | ladder: L0 (exactly invariant), ttbar, MWS-layer + merging | 1 | 0.9857 | 0.9879 | 0.964 | 15/19 | 0.9973 | 0.9615 | 0.0148 |
| 8 | ladder: blind triplets (no position), ttbar, MWS-layer + merging | 1 | 0.9857 | 0.9879 | 0.964 | 15/19 | 0.9979 | 0.9624 | 0.0158 |
| 9 | ladder: L1 (symmetric + regions), ttbar, MWS-layer + merging | 1 | 0.9857 | 0.9879 | 0.958 | 15/19 | 0.9973 | 0.9717 | 0.0138 |
| 10 | ladder: L0 + trunc., ttbar, MWS-layer + merging | 1 | 0.9850 | 0.9868 | 0.964 | 15/19 | 0.9979 | 0.9555 | 0.0150 |
| 11 | ladder: rz triplets, ttbar, MWS-layer + merging | 1 | 0.9850 | 0.9874 | 0.958 | 15/19 | 0.9979 | 0.9729 | 0.0147 |
| 12 | ladder: L3 + trunc., ttbar, MWS-layer + merging | 1 | 0.9827 | 0.9852 | 0.964 | 14/19 | 0.9979 | 0.9634 | 0.0152 |
| 13 | Canto-B (full budget): base graph, MWS-layer + merging | 2 | 0.9827 ± 0.0021 | 0.9855 ± 0.0019 | 0.952 | 28/38 | 0.9976 ± 0.0004 | 0.9632 ± 0.0005 | 0.0150 ± 0.0002 |
| 14 | acorn IGNN, ttbar+HNL-trained + Canto builder (MWS-layer + merging) | 3 | 0.9782 ± 0.0020 | 0.9816 ± 0.0026 | 0.956 | 47/57 | 0.9971 ± 0.0003 | 0.9570 ± 0.0007 | 0.0150 ± 0.0019 |
| 15 | acorn IGNN, ttbar-trained + Canto builder (MWS-layer + merging) | 3 | 0.9689 ± 0.0034 | 0.9755 ± 0.0025 | 0.932 | 42/57 | 0.9979 ± 0.0000 | 0.9552 ± 0.0021 | 0.0175 ± 0.0008 |
| 16 | ladder: IGNN (acorn arch.), screening budget, ttbar, MWS-layer + merging | 1 | 0.9632 | 0.9710 | 0.916 | 13/19 | 0.9968 | 0.9512 | 0.0192 |
| 17 | ladder: IGNN + trunc., ttbar, MWS-layer + merging | 1 | 0.9632 | 0.9704 | 0.916 | 13/19 | 0.9963 | 0.9428 | 0.0198 |
| 18 | acorn IGNN, ttbar+HNL-trained + CC (prereg. cut rule) | 3 | 0.9394 ± 0.0227 | 0.9463 ± 0.0241 | 0.916 | 42/57 | 0.9888 ± 0.0065 | 0.8579 ± 0.0084 | 0.0289 ± 0.0002 |
| 19 | acorn IGNN, HNL-trained + CC (prereg. cut rule) | 3 | 0.9384 ± 0.0026 | 0.9388 ± 0.0046 | 0.934 | 46/57 | 0.9815 ± 0.0048 | 0.8210 ± 0.0034 | 0.0293 ± 0.0005 |
| 20 | acorn IGNN, ttbar-trained + CC (prereg. cut rule) | 3 | 0.9231 ± 0.0128 | 0.9330 ± 0.0119 | 0.892 | 37/57 | 0.9870 ± 0.0031 | 0.8723 ± 0.0094 | 0.0297 ± 0.0003 |
| 21 | DeZoort IN-L (127k par.) + CC [FR* unreachable] | 1 | 0.8798 | 0.8926 | 0.826 | 10/19 | 0.9663 | 0.7817 | 0.0609 |
| 22 | DeZoort IN (6.4k par.) + CC [FR* unreachable] | 1 | 0.8482 | 0.8482 | 0.802 | 9/19 | 0.9117 | 0.6263 | 0.1251 |
