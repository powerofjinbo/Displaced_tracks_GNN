# Mirror truncation probe: start radius or track endpoints? (supervisor feedback round 1, point 4a)

Validation only (300 hnl10_100 val events, the events of the original probe); a diagnostic of the edge classifiers, not a claim; nothing here is chosen on or evaluated on test. Produced by `python -m hnl.eval.probe_outer` (one job per model, `slurm/stage1/probe_outer.sbatch`) and `python -m hnl.eval.probe_outer_report`; per-edge outputs in `/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v1/probe_outer_v1/<model>/edges.parquet`.

## Method

- Particles and edits as the truncation probe (`hnl.eval.truncation_probe`, paper Sec. IV B): prompt particles (truth |d0| < 1 mm, >= 8 hits, pT >= 2 GeV, at most 5 per event; 1145 particles) of 300 HNL val events (hnl10_100, the first 300 HNL members of the val graph tars); hits removed together with every graph edge touching them, no edge added (`hnl.eval.truncation_probe.truncate`; for the metric-learning radius graph this is the graph the graph stage would build without those hits, as its neighbour cap never binds; for the Canto-L union graphs, union edges whose triplet support used a removed hit are kept, i.e. the graph stage is not re-run and the probe tests the edge classifier), the edited graph re-scored by the model's GNN (best checkpoint of the finished training); edge TPR at the fixed score cut 0.35 of the chain-true edges whose two hits survive at every depth m = 0, 2, 4 (the same edges at every depth; an edge missing from the graph counts as lost, as in the original probe; column in_graph of the csv); dT = TPR(m=4) - TPR(m=0) in points.
- **inner** (the original probe): remove the m innermost hits (R = sqrt(r^2 + z^2)); measured: chain-true edges with both hits at R rank >= 4 (the outer edges). Reproduces the stored RUN/truncation_probe/summary.csv.
- **outer** (the mirror): remove the m outermost hits of the same particles in the same events; measured: chain-true edges with both hits at reversed rank n - 1 - rank >= 4 (the inner edges). The track still starts at the beam line and points to it; it only ends early.
- **middle**: remove m hits in the middle (R ranks [n//2 - m/2, n//2 + m/2)); measured: chain-true edges with both hits outside the m = 4 window: the inner piece (ends at the gap) and the outer piece (starts after the gap, at a larger radius).
- Distance to the cut: for the fixed edges of inner (outer), d = rank of the edge's inner hit - 4 (reversed rank of its outer hit - 4): d = 0 is the edge adjacent to the new start (new end) at m = 4, 2 hops from it at m = 2, 4 hops at m = 0.
- Uncertainties: event bootstrap, B = 2000 resamples of the 300 events with replacement (all edges of an event move together, so the correlation of the edges of one track is included); the same resamples for every model, mode and depth (paired differences); +- is the bootstrap standard deviation; 'sig.' = the 95 % percentile interval excludes 0. Seed means: mean of the seeds' dT, bootstrap of the mean with the same resamples; the seed sd is given separately. dT is at a fixed score cut, not at a matched false-positive rate (as in the paper).
- Verdict rule (fixed before any number was read): significance = the 95 % event-bootstrap interval excludes 0. Inner drop significant and significantly larger than the outer change (dT_inner - dT_outer < 0): 'start radius', with 'an endpoint part' if the outer drop is significant too; inner and outer drops both significant and not significantly different: 'endpoint'; inner drop not significant: 'no significant inner drop'. Labels of models whose two changes are both below 1 point carry the prefix 'both below 1 point' (added for reading; the rule is unchanged).

## Summary: dT_inner against dT_outer [points], +- event-bootstrap sd

| model | dT_inner | dT_outer | dT_inner - dT_outer | verdict |
|---|---|---|---|---|
| acorn (ttbar) s0 | -7.05 +- 0.47 | -0.04 +- 0.04 | -7.01 +- 0.47 | start radius (no significant outer drop) |
| acorn (ttbar) s1 | -8.10 +- 0.48 | -0.06 +- 0.04 | -8.04 +- 0.48 | start radius (no significant outer drop) |
| acorn (ttbar) s2 | -8.21 +- 0.47 | -0.06 +- 0.03 | -8.15 +- 0.47 | start radius with an endpoint part (outer drop 1% of the inner drop) |
| acorn (ttbar), mean of seeds 0-2 (seed sd 0.64 / 0.01) | -7.78 +- 0.46 | -0.05 +- 0.02 | -7.73 +- 0.46 | start radius with an endpoint part (outer drop 1% of the inner drop) |
| acorn (HNL) s0 | -0.35 +- 0.09 | -0.15 +- 0.05 | -0.21 +- 0.09 | both below 1 point; start radius with an endpoint part (outer drop 41% of the inner drop) |
| equal-budget IGNN | -5.25 +- 0.37 | -0.08 +- 0.04 | -5.17 +- 0.37 | start radius (no significant outer drop) |
| L1 without truncation | -7.44 +- 0.47 | -0.12 +- 0.06 | -7.32 +- 0.47 | start radius with an endpoint part (outer drop 2% of the inner drop) |
| Canto-L s0 | -0.13 +- 0.04 | -0.15 +- 0.06 | +0.01 +- 0.07 | both below 1 point; endpoint (outer drop not significantly smaller than the inner drop) |
| Canto-L s1 | -0.06 +- 0.05 | -0.02 +- 0.03 | -0.04 +- 0.05 | both below 1 point; no significant inner drop |
| Canto-L s2 | -0.11 +- 0.06 | -0.10 +- 0.04 | -0.01 +- 0.05 | both below 1 point; no significant inner drop |
| Canto-L, mean of seeds 0-2 (seed sd 0.04 / 0.06) | -0.10 +- 0.04 | -0.09 +- 0.03 | -0.01 +- 0.04 | both below 1 point; endpoint (outer drop not significantly smaller than the inner drop) |
| IGNN + truncation | -0.02 +- 0.05 | -0.05 +- 0.02 | +0.02 +- 0.05 | both below 1 point; no significant inner drop |
| L1 + truncation (Canto classifier preview) | -0.10 +- 0.06 | -0.04 +- 0.05 | -0.06 +- 0.07 | both below 1 point; no significant inner drop |

## Verdict

**Classifiers trained on ttbar without truncation augmentation** (acorn (ttbar) s0, acorn (ttbar) s1, acorn (ttbar) s2, equal-budget IGNN, L1 without truncation): removing the 4 innermost hits lowers the TPR of the remaining outer edges by dT_inner = -8.2 to -5.3 points (event-bootstrap sd 0.4 to 0.5); removing the 4 outermost hits of the same particles changes the TPR of the remaining inner edges by dT_outer = -0.12 to -0.04 points (sd 0.03 to 0.06). The paired difference dT_inner - dT_outer is -8.1 to -5.2 points, at least 14 sd from zero for every model. The cut-independent mean score of the same edges moves by -0.099 to -0.073 (inner) against -0.0033 to -0.0005 (outer).

**Verdict: start radius, not endpoints.** For every one of these classifiers the inner drop is significant and significantly larger than the outer change (95 % event-bootstrap intervals). A new track end at mid radius moves the TPR by at most 0.12 points (at most 2% of the same model's inner drop); a late start lowers it by 5.3 to 8.2 points. The outer change is statistically significant for acorn (ttbar) s2, L1 without truncation and the acorn (ttbar) seed mean (-0.05 +- 0.02), i.e. an endpoint part of that size exists; for the others it is consistent with zero. The scores depend on where a track starts (a learned start-radius prior), not on where it ends: the network did not learn a generic distrust of edges near a track endpoint.

**Middle gap.** Removing 4 hits in the middle changes the TPR of the outer piece, which now starts after the gap, by -7.6 to -5.7 points (sd 0.5), and that of the inner piece, which now ends at the gap, by -0.89 to +0.03 points (sd 0.10 to 0.19). The piece that starts late loses; the piece that ends early loses at most 12% of that: the same reading.

**Adjacent against far edges.** Inner probe: the edge adjacent to the new start at m = 4 (d = 0) changes by -5.7 to -3.7 points, edges at least 4 hops from it (d >= 4) by -7.7 to -3.5 points. Outer probe: the edge adjacent to the new end changes by -0.44 to -0.17 points (-0.26 +- 0.10 for the acorn (ttbar) seed mean). The start effect is not confined to the edges next to the new start: edges at least 4 hops away lose too, so a late start lowers the scores along the remaining track. A local endpoint effect, if any, is at most 0.44 points on the edge next to the new end, at most 9% of the change of the edge next to the new start.

**acorn (HNL) s0** (the same architecture trained on the HNL signal): dT_inner = -0.35 +- 0.09, dT_outer = -0.15 +- 0.05 points. Both changes are below 0.36 points, the inner one 15 times smaller than the smallest ttbar-trained inner drop (the verdict rule labels it 'both below 1 point; start radius with an endpoint part (outer drop 41% of the inner drop)', on effects of this size).

**Truncation-augmented classifiers** (Canto-L s0, Canto-L s1, Canto-L s2, IGNN + truncation, L1 + truncation (Canto classifier preview)): dT_inner = -0.13 to -0.02 points, dT_outer = -0.15 to -0.02 points (event-bootstrap sd at most 0.06). The augmentation, which only removes inner hits in training, removes the start-radius sensitivity; the response to a new track end stays below 0.5 points, as without it.

For Canto-L the two changes are equal within the uncertainties (seed mean dT_inner -0.10 +- 0.04, dT_outer -0.09 +- 0.03, difference -0.01 +- 0.04 points; the verdict rule therefore reads 'both below 1 point; endpoint (outer drop not significantly smaller than the inner drop)'): what is left after the augmentation is a change of about 0.1 point for a new start or a new end, 52 or more times smaller than the start-radius drop of the classifiers without augmentation. It sits on the edge next to the new endpoint (d = 0, seed mean: inner -0.41 +- 0.15, outer -0.49 +- 0.17 points; d >= 4: inner +0.01 +- 0.03, outer -0.01 +- 0.02): a small local endpoint effect, the same at both ends.

A 4-hit gap in the middle is a different edit: for these classifiers the outer piece changes by -2.1 to -0.2 points and the inner piece by -1.7 to -0.5 points (largest change of the two pieces together: Canto-L s1, -1.6 +- 0.2), against -7.6 to -5.7 points for the outer piece of the classifiers without augmentation.

## Main table: dT_inner against dT_outer [points]

| model | budget, graph | edges (inner / outer) | TPR inner m=0 / 2 / 4 | dT_inner | TPR outer m=0 / 2 / 4 | dT_outer | dT_inner - dT_outer | verdict (rule in the text) |
|---|---|---|---|---|---|---|---|---|
| acorn (ttbar) s0 | full budget, metric-learning graph | 8261 / 8267 | 0.914 / 0.870 / 0.844 | -7.05 +- 0.47 | 0.852 / 0.852 / 0.851 | -0.04 +- 0.04 | -7.01 +- 0.47 | start radius (no significant outer drop) |
| acorn (ttbar) s1 | full budget, metric-learning graph | 8261 / 8267 | 0.914 / 0.866 / 0.833 | -8.10 +- 0.48 | 0.849 / 0.849 / 0.848 | -0.06 +- 0.04 | -8.04 +- 0.48 | start radius (no significant outer drop) |
| acorn (ttbar) s2 | full budget, metric-learning graph | 8261 / 8267 | 0.913 / 0.871 / 0.831 | -8.21 +- 0.47 | 0.848 / 0.847 / 0.847 | -0.06 +- 0.03 | -8.15 +- 0.47 | start radius with an endpoint part (outer drop 1% of the inner drop) |
| acorn (ttbar), mean of seeds 0-2 (seed sd: 0.64 / 0.01) | full budget, metric-learning graph | 8261 / 8267 | 0.914 / 0.869 / 0.836 | -7.78 +- 0.46 | 0.849 / 0.849 / 0.849 | -0.05 +- 0.02 | -7.73 +- 0.46 | start radius with an endpoint part (outer drop 1% of the inner drop) |
| acorn (HNL) s0 | full budget, metric-learning graph (HNL-trained) | 8261 / 8267 | 0.914 / 0.913 / 0.911 | -0.35 +- 0.09 | 0.851 / 0.851 / 0.850 | -0.15 +- 0.05 | -0.21 +- 0.09 | both below 1 point; start radius with an endpoint part (outer drop 41% of the inner drop) |
| equal-budget IGNN | screening budget, acorn (ttbar) s0 graph | 8261 / 8267 | 0.913 / 0.903 / 0.861 | -5.25 +- 0.37 | 0.848 / 0.848 / 0.847 | -0.08 +- 0.04 | -5.17 +- 0.37 | start radius (no significant outer drop) |
| L1 without truncation | screening budget, acorn (ttbar) s0 graph | 8261 / 8267 | 0.914 / 0.895 / 0.840 | -7.44 +- 0.47 | 0.848 / 0.848 / 0.847 | -0.12 +- 0.06 | -7.32 +- 0.47 | start radius with an endpoint part (outer drop 2% of the inner drop) |
| Canto-L s0 | full budget, own learned union graph | 8261 / 8267 | 0.915 / 0.915 / 0.913 | -0.13 +- 0.04 | 0.848 / 0.848 / 0.847 | -0.15 +- 0.06 | +0.01 +- 0.07 | both below 1 point; endpoint (outer drop not significantly smaller than the inner drop) |
| Canto-L s1 | full budget, own learned union graph | 8261 / 8267 | 0.916 / 0.915 / 0.915 | -0.06 +- 0.05 | 0.850 / 0.850 / 0.850 | -0.02 +- 0.03 | -0.04 +- 0.05 | both below 1 point; no significant inner drop |
| Canto-L s2 | full budget, own learned union graph | 8261 / 8267 | 0.915 / 0.915 / 0.914 | -0.11 +- 0.06 | 0.849 / 0.848 / 0.848 | -0.10 +- 0.04 | -0.01 +- 0.05 | both below 1 point; no significant inner drop |
| Canto-L, mean of seeds 0-2 (seed sd: 0.04 / 0.06) | full budget, own learned union graph | 8261 / 8267 | 0.915 / 0.915 / 0.914 | -0.10 +- 0.04 | 0.849 / 0.849 / 0.848 | -0.09 +- 0.03 | -0.01 +- 0.04 | both below 1 point; endpoint (outer drop not significantly smaller than the inner drop) |
| IGNN + truncation | screening budget, acorn (ttbar) s0 graph | 8261 / 8267 | 0.913 / 0.913 / 0.913 | -0.02 +- 0.05 | 0.845 / 0.845 / 0.844 | -0.05 +- 0.02 | +0.02 +- 0.05 | both below 1 point; no significant inner drop |
| L1 + truncation (Canto classifier preview) | screening budget, acorn (ttbar) s0 graph | 8261 / 8267 | 0.915 / 0.914 / 0.914 | -0.10 +- 0.06 | 0.846 / 0.846 / 0.846 | -0.04 +- 0.05 | -0.06 +- 0.07 | both below 1 point; no significant inner drop |

## Middle gap [points]

| model | dT middle, all | inner piece (ends at the gap) | outer piece (starts after the gap) | outer piece - inner piece |
|---|---|---|---|---|
| acorn (ttbar) s0 | -3.3 +- 0.3 (7119 edges) | -0.40 +- 0.12 (3275) | -5.7 +- 0.5 (3844) | -5.4 +- 0.5 |
| acorn (ttbar) s1 | -3.8 +- 0.3 (7119 edges) | -0.15 +- 0.10 (3275) | -7.0 +- 0.5 (3844) | -6.8 +- 0.5 |
| acorn (ttbar) s2 | -3.7 +- 0.3 (7119 edges) | -0.40 +- 0.12 (3275) | -6.5 +- 0.5 (3844) | -6.1 +- 0.5 |
| acorn (ttbar), mean of seeds 0-2 | -3.6 +- 0.3 (7119 edges) | -0.32 +- 0.08 (3275) | -6.4 +- 0.5 (3844) | -6.1 +- 0.5 |
| acorn (HNL) s0 | -1.0 +- 0.1 (7119 edges) | -0.55 +- 0.17 (3275) | -1.4 +- 0.2 (3844) | -0.9 +- 0.3 |
| equal-budget IGNN | -3.2 +- 0.3 (7119 edges) | +0.03 +- 0.10 (3275) | -6.0 +- 0.5 (3844) | -6.0 +- 0.5 |
| L1 without truncation | -4.5 +- 0.3 (7119 edges) | -0.89 +- 0.19 (3275) | -7.6 +- 0.5 (3844) | -6.7 +- 0.6 |
| Canto-L s0 | -1.1 +- 0.1 (7119 edges) | -1.74 +- 0.27 (3275) | -0.5 +- 0.1 (3844) | +1.2 +- 0.3 |
| Canto-L s1 | -1.6 +- 0.2 (7119 edges) | -1.07 +- 0.20 (3275) | -2.1 +- 0.3 (3844) | -1.0 +- 0.3 |
| Canto-L s2 | -1.0 +- 0.1 (7119 edges) | -1.04 +- 0.21 (3275) | -1.0 +- 0.2 (3844) | +0.0 +- 0.3 |
| Canto-L, mean of seeds 0-2 | -1.2 +- 0.1 (7119 edges) | -1.28 +- 0.18 (3275) | -1.2 +- 0.2 (3844) | +0.1 +- 0.2 |
| IGNN + truncation | -0.4 +- 0.1 (7119 edges) | -0.49 +- 0.13 (3275) | -0.2 +- 0.1 (3844) | +0.3 +- 0.2 |
| L1 + truncation (Canto classifier preview) | -0.7 +- 0.1 (7119 edges) | -1.01 +- 0.19 (3275) | -0.5 +- 0.1 (3844) | +0.5 +- 0.2 |

## Where along the track: TPR change m=0 -> 4 by distance d to the new start (inner) or new end (outer) [points]

| model | probe | d=0 (adjacent) | d=1 | d=2-3 | d>=4 | adjacent - far (d>=4) |
|---|---|---|---|---|---|---|
| acorn (ttbar) s0 | inner | -3.7 +- 0.6 (1140) | -6.9 +- 0.8 (1144) | -9.0 +- 0.7 (2270) | -6.9 +- 0.6 (3707) | +3.2 +- 0.7 |
| acorn (ttbar) s0 | outer | -0.35 +- 0.21 (1145) | +0.00 +- 0.12 (1145) | +0.04 +- 0.04 (2270) | +0.00 +- 0.00 (3707) | -0.35 +- 0.21 |
| acorn (ttbar) s1 | inner | -5.2 +- 0.7 (1140) | -9.4 +- 0.9 (1144) | -9.6 +- 0.6 (2270) | -7.7 +- 0.6 (3707) | +2.5 +- 0.8 |
| acorn (ttbar) s1 | outer | -0.17 +- 0.12 (1145) | +0.00 +- 0.00 (1145) | -0.18 +- 0.09 (2270) | +0.03 +- 0.03 (3707) | -0.20 +- 0.13 |
| acorn (ttbar) s2 | inner | -5.7 +- 0.7 (1140) | -9.6 +- 0.9 (1144) | -10.7 +- 0.7 (2270) | -7.0 +- 0.6 (3707) | +1.3 +- 0.8 |
| acorn (ttbar) s2 | outer | -0.26 +- 0.15 (1145) | -0.09 +- 0.09 (1145) | +0.00 +- 0.00 (2270) | -0.03 +- 0.05 (3707) | -0.24 +- 0.15 |
| acorn (ttbar), mean of seeds 0-2 | inner | -4.9 +- 0.6 (1140) | -8.7 +- 0.8 (1144) | -9.8 +- 0.6 (2270) | -7.2 +- 0.6 (3707) | +2.3 +- 0.7 |
| acorn (ttbar), mean of seeds 0-2 | outer | -0.26 +- 0.10 (1145) | -0.03 +- 0.06 (1145) | -0.04 +- 0.03 (2270) | +0.00 +- 0.02 (3707) | -0.26 +- 0.10 |
| acorn (HNL) s0 | inner | -1.1 +- 0.3 (1140) | -0.3 +- 0.2 (1144) | -0.4 +- 0.1 (2270) | -0.1 +- 0.1 (3707) | -1.0 +- 0.3 |
| acorn (HNL) s0 | outer | -0.61 +- 0.22 (1145) | -0.44 +- 0.19 (1145) | -0.04 +- 0.04 (2270) | +0.03 +- 0.03 (3707) | -0.64 +- 0.23 |
| equal-budget IGNN | inner | -4.9 +- 0.7 (1140) | -7.5 +- 0.8 (1144) | -7.1 +- 0.6 (2270) | -3.5 +- 0.4 (3707) | -1.4 +- 0.7 |
| equal-budget IGNN | outer | -0.44 +- 0.26 (1145) | +0.00 +- 0.12 (1145) | -0.04 +- 0.04 (2270) | -0.03 +- 0.03 (3707) | -0.41 +- 0.26 |
| L1 without truncation | inner | -4.1 +- 0.6 (1140) | -7.7 +- 0.8 (1144) | -9.3 +- 0.7 (2270) | -7.3 +- 0.6 (3707) | +3.1 +- 0.7 |
| L1 without truncation | outer | -0.26 +- 0.23 (1145) | -0.09 +- 0.15 (1145) | -0.18 +- 0.11 (2270) | -0.05 +- 0.05 (3707) | -0.21 +- 0.24 |
| Canto-L s0 | inner | -0.6 +- 0.2 (1140) | -0.3 +- 0.2 (1144) | -0.0 +- 0.0 (2270) | +0.0 +- 0.0 (3707) | -0.6 +- 0.2 |
| Canto-L s0 | outer | -0.79 +- 0.35 (1145) | -0.17 +- 0.12 (1145) | +0.00 +- 0.06 (2270) | -0.03 +- 0.03 (3707) | -0.76 +- 0.36 |
| Canto-L s1 | inner | -0.2 +- 0.2 (1140) | -0.1 +- 0.1 (1144) | -0.1 +- 0.1 (2270) | +0.0 +- 0.0 (3707) | -0.2 +- 0.2 |
| Canto-L s1 | outer | -0.17 +- 0.17 (1145) | +0.00 +- 0.00 (1145) | +0.00 +- 0.00 (2270) | +0.00 +- 0.04 (3707) | -0.17 +- 0.18 |
| Canto-L s2 | inner | -0.4 +- 0.2 (1140) | -0.2 +- 0.1 (1144) | -0.1 +- 0.1 (2270) | +0.0 +- 0.1 (3707) | -0.4 +- 0.2 |
| Canto-L s2 | outer | -0.52 +- 0.21 (1145) | +0.00 +- 0.12 (1145) | -0.09 +- 0.06 (2270) | +0.00 +- 0.00 (3707) | -0.52 +- 0.21 |
| Canto-L, mean of seeds 0-2 | inner | -0.4 +- 0.1 (1140) | -0.2 +- 0.1 (1144) | -0.1 +- 0.0 (2270) | +0.0 +- 0.0 (3707) | -0.4 +- 0.1 |
| Canto-L, mean of seeds 0-2 | outer | -0.49 +- 0.17 (1145) | -0.06 +- 0.06 (1145) | -0.03 +- 0.03 (2270) | -0.01 +- 0.02 (3707) | -0.49 +- 0.17 |
| IGNN + truncation | inner | +0.0 +- 0.2 (1140) | -0.2 +- 0.1 (1144) | +0.0 +- 0.1 (2270) | +0.0 +- 0.0 (3707) | +0.0 +- 0.2 |
| IGNN + truncation | outer | -0.26 +- 0.15 (1145) | -0.09 +- 0.09 (1145) | +0.00 +- 0.00 (2270) | +0.00 +- 0.00 (3707) | -0.26 +- 0.15 |
| L1 + truncation (Canto classifier preview) | inner | -0.3 +- 0.2 (1140) | -0.2 +- 0.2 (1144) | -0.1 +- 0.1 (2270) | +0.0 +- 0.0 (3707) | -0.3 +- 0.2 |
| L1 + truncation (Canto classifier preview) | outer | -0.17 +- 0.17 (1145) | +0.00 +- 0.18 (1145) | -0.04 +- 0.10 (2270) | +0.00 +- 0.04 (3707) | -0.17 +- 0.18 |

## Reproduction of the stored inner probe

| model | stored TPR m=0 / 2 / 4 (RUN/truncation_probe) | this run | max abs difference |
|---|---|---|---|
| acorn (ttbar) s0 | 0.9142 / 0.8704 / 0.8437 (8261 edges) | 0.9142 / 0.8704 / 0.8437 (8261 edges) | 0.0e+00 |
| acorn (ttbar) s1 | 0.9141 / 0.8659 / 0.8331 (8261 edges) | 0.9141 / 0.8659 / 0.8331 (8261 edges) | 1.1e-16 |
| acorn (HNL) s0 | 0.9141 / 0.9132 / 0.9105 (8261 edges) | 0.9141 / 0.9132 / 0.9105 (8261 edges) | 1.1e-16 |
| equal-budget IGNN | 0.9134 / 0.9029 / 0.8609 (8261 edges) | 0.9134 / 0.9029 / 0.8609 (8261 edges) | 0.0e+00 |
| L1 without truncation | 0.9142 / 0.8946 / 0.8397 (8261 edges) | 0.9142 / 0.8946 / 0.8397 (8261 edges) | 0.0e+00 |
| IGNN + truncation | 0.9133 / 0.9133 / 0.9131 (8261 edges) | 0.9133 / 0.9133 / 0.9131 (8261 edges) | 0.0e+00 |
| L1 + truncation (Canto classifier preview) | 0.9147 / 0.9144 / 0.9137 (8261 edges) | 0.9147 / 0.9144 / 0.9137 (8261 edges) | 1.1e-16 |

## Files

- `results/probe_outer_v1.csv`: one row per model (seed means: `<group>_mean`), mode (inner, outer, middle, difference) and subset (all; distance groups d; middle pieces; paired differences): n_edges, in_graph, TPR at m = 0 / 2 / 4, dT_pts (m=4 - m=0) and dT_m2_pts (m=2 - m=0) in points, dT_boot_sd and the 95 % percentile interval (dT_boot_lo95, dT_boot_hi95), the mean score of the same in-graph edges at m = 0 / 2 / 4 and its change (dscore, dscore_boot_sd), dT_seed_sd for seed means, and the verdict of `classify` on the 'inner - outer' rows.
- `results/probe_outer_v1_meta.json`: per-model run metadata (checkpoint, graphs, job, host, timings) and the reproduction check against RUN/truncation_probe/summary.csv.
- Per-edge outputs: `/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v1/probe_outer_v1/<model>/edges.parquet` (columns mode, depth, event_id, particle_id, r_lo, r_hi, n_hits, in_graph, score; every chain-true edge of the chosen particles that survives the edit), `/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/feedback_v1/probe_outer_v1/events.json` (the 300 event ids).

## Runs

| model | run | graphs | checkpoint | job | host | elapsed |
|---|---|---|---|---|---|---|
| acorn (ttbar) s0 | baseline_v1/armi_s0 | baseline_v1/armi_s0/ml_graphs/valset.tar | best-epoch=23-val_loss=0.058278.ckpt | 6692295 | submit21.mit.edu | 6032.4 s |
| acorn (ttbar) s1 | baseline_v1/armi_s1 | baseline_v1/armi_s1/ml_graphs/valset.tar | best-epoch=29-val_loss=0.056590.ckpt | 6692296 | submit22.mit.edu | 933.4 s |
| acorn (ttbar) s2 | baseline_v1/armi_s2 | baseline_v1/armi_s2/ml_graphs/valset.tar | best-epoch=28-val_loss=0.046033.ckpt | 6692297 | submit37.mit.edu | 2452.9 s |
| acorn (HNL) s0 | baseline_v1/armii_s0 | baseline_v1/armii_s0/ml_graphs/valset.tar | best-epoch=24-val_loss=0.058465.ckpt | 6692400 | submit22.mit.edu | 1036.4 s |
| equal-budget IGNN | stage1_screen_ttbar/ignn/armi_s0 | baseline_v1/armi_s0/ml_graphs/valset.tar | best-epoch=14-val_loss=0.062376.ckpt | 6692401 | submit37.mit.edu | 2268.4 s |
| L1 without truncation | stage1_screen_ttbar/l1/armi_s0 | baseline_v1/armi_s0/ml_graphs/valset.tar | best-epoch=14-val_loss=0.060345.ckpt | 6692403 | submit22.mit.edu | 2947.0 s |
| Canto-L s0 | stage1_final_learned/l1tl/armi_s0 | baseline_v1/armi_s0/ltf_union_graphs/valset.tar | best-epoch=28-val_loss=0.052177.ckpt | 6692405 + 6692747 | submit37.mit.edu (both jobs) | 6900 + 4539 s |
| Canto-L s1 | stage1_final_learned/l1tl/armi_s1 | baseline_v1/armi_s1/ltf_union_graphs/valset.tar | best-epoch=29-val_loss=0.051051.ckpt | 6692406 | submit20.mit.edu | 3354.9 s |
| Canto-L s2 | stage1_final_learned/l1tl/armi_s2 | baseline_v1/armi_s2/ltf_union_graphs/valset.tar | best-epoch=28-val_loss=0.044359.ckpt | 6692889 | submit21.mit.edu | 3981.3 s |
| IGNN + truncation | stage1_screen_ttbar/ignn_trunc/armi_s0 | baseline_v1/armi_s0/ml_graphs/valset.tar | best-epoch=14-val_loss=0.063254.ckpt | 6692402 | submit21.mit.edu | 1008.7 s |
| L1 + truncation (Canto classifier preview) | stage1_screen_ttbar/symregion_trunc/armi_s0 | baseline_v1/armi_s0/ml_graphs/valset.tar | best-epoch=14-val_loss=0.061297.ckpt | 6692404 | submit22.mit.edu | 2603.1 s |

- Canto-L s0: none_0, inner_2/4 and outer_2/4 from job 6692405 (8 CPUs on submit37, ~1,400 s per inference; cancelled after outer_4 because it would have hit its 2.5 h limit before the middle depths), middle_2/4 from the middle-only helper job 6692747 (canto_l_s0_midhelper, same node, checkpoint, graphs and events); their depth-0 rows are identical (max score difference 0.0).
- CPU only (8-16 cores per job, no GPU); inference time per depth 90-1,500 s depending on the model and node.

