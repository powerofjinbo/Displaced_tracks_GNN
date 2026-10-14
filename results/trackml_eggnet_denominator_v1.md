# Canto-L under the EggNet code with the target denominator of Chan et al.'s particle count (v1)

Ledger R13 (docs/feedback/test_use_ledger.md, committed in 1f68a57 before this read-out). Post-freeze, descriptive;
nothing is chosen. Read: the stored post-freeze test tracks of the configurations already evaluated under the EggNet
code in Table VI (C-full: WP-A, WP-R-P3; C-1GeV: WP-A, WP-R-P4), 300 throughput test events of run canto_l_v1/seed0_x1.
No network or track builder was run. Code: hnl.trackml.eggnet_denominator, slurm/trackml/eggnet_denominator.sbatch;
job 6810578 (CPU, express, submit37; 6810562 gave the same numbers but compared with refmetrics.csv parsed without
round-trip precision and flagged 11 of 12 rows unequal in the last bit). Per-event counts and the full summary:
$RUN/test_post/eggnet_denominator/. All numbers: trackml_eggnet_denominator_v1.csv.

Why: the EggNet code counts as targets the particles with pT >= 1 GeV and at least 3 hits, but Chan et al.'s "About 467K
particles" over 400 test events matches pT >= 1 GeV with at least 1 hit (results/trackml_port_closure_v1.md). On our 300
test events the same counts are 468.0K per 400 events with >= 1 hit, 459.5K with >= 2 and 451.7K with >= 3. Only the
efficiency denominator (and with it the matched-target count) changes; matching, candidates (tracks with >= 3 hits),
duplication and fake rate are the code's. Errors: event bootstrap, B = 500, seed 12345, as in hnl.trackml.refmetrics.

Check: with >= 3 hits the recount equals the Table VI numbers ($RUN/test_post/refmetrics, 2026-10-11) exactly: all 12
values and bootstrap standard deviations, and every per-event EggNet count (0 differing rows in either condition).

## Efficiency (percent)

| Condition | Canto-L | >= 1 hit | >= 2 hits | >= 3 hits (Table VI) | EggNet, published |
|---|---|---|---|---|---|
| All hits | LM, WP-A | 95.99 +- 0.04 | 97.77 +- 0.03 | 99.41 +- 0.01 | 95.87 +- 0.03 (Chan et al.) |
| All hits | CC, WP-R-P3 | 95.92 +- 0.04 | 97.70 +- 0.03 | 99.37 +- 0.01 | |
| TrackML-1GeV | LM, WP-A | 96.09 +- 0.04 | 97.88 +- 0.03 | 99.56 +- 0.01 | 99.56 +- 0.03 (Calafiura et al.) |
| TrackML-1GeV | WT+M, WP-R-P4 | 96.20 +- 0.04 | 97.98 +- 0.03 | 99.67 +- 0.01 | |

Targets (300 events): 350,996 with >= 1 hit, 344,592 with >= 2, 338,753 with >= 3. Canto-L matches none of the
one-hit particles (a >= 3-hit track cannot hold a majority of one particle's single hit) and 166 (WP-A, all hits) of
the 5,839 two-hit particles.

The duplication rate changes by at most 0.0002 points with the denominator and the fake rate not at all: all hits,
WP-A 0.168 +- 0.007 % and 2.40 +- 0.01 %; WP-R-P3 1.70 +- 0.04 % and 0.173 +- 0.003 % (EggNet 2.76 +- 0.02 % and
0.19 +- 0.01 %).

## Reading

- **All hits.** With Chan et al.'s population (>= 1 hit), Canto-L at its working point finds 95.99 +- 0.04 %
  against EggNet's 95.87 +- 0.03 %: 0.12 +- 0.05 points higher, 2.6 standard deviations, at 12.6 times EggNet's fake
  rate. Connected components at EggNet's fake rate (WP-R-P3) find 95.92 +- 0.04 %: 0.05 +- 0.05 points, not resolved.
  The 3.5-point lead of Table VI (99.41 % against 95.87 %) comes from the denominator, if Chan et al.'s 467K particles
  are the denominator of their efficiency.
- **TrackML-1GeV.** The >= 3-hit comparison of Table VI stands. A denominator with one-hit particles cannot give
  Calafiura et al.'s 99.56 %: no >= 3-hit candidate can be matched to a one-hit particle, so with >= 1 hit the
  efficiency is at most the fraction of particles with >= 2 hits, 98.18 % on our test events.
- What is not known: which population Chan et al.'s efficiency used. The quoted count is the only evidence; the
  repository's evaluation configuration (configs/trackML/eval_trackml.yaml, e2bd117) says >= 3 hits.
