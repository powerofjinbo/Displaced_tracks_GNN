# GPU-hour ledger recount v1 (sacct, 2026-10-03 to 2026-10-15 00:35 UTC)

Snapshot: `sacct` pulled at 2026-10-15 00:35:38 UTC. Scope: every job of user jinboz1 that ran from 2026-10-03 00:00
UTC on, all partitions, all projects (the non-project jobs in that window, e.g. `/work/submit/jinboz1/tmp/caltech_cleanup`,
held no GPU). Rule (DECISIONS 2026-10-02): Slurm accounting has no `gres/gpu` TRES, so GPU-h = elapsed × GPUs requested.
Per-job rows: `results/gpu_ledger_recount_v1.csv`.

## Result

| | GPU job runs | GPU-h |
|---|---|---|
| Up to 2026-10-05 07:10 UTC (time of the last cumulative ledger row) | 103 | 318.00 |
| After 2026-10-05 07:10 UTC | 472 | 255.86 |
| **Total, 2026-10-03 to 2026-10-15 00:35 UTC** | **572** | **573.86** |

The three jobs running at 07:10 on 2026-10-05 are split at that time and appear in both rows (103 + 472 = 575).
The 7 jobs still running at the snapshot are included with their elapsed time so far (3.32 GPU-h, table below).
Every GPU job requested exactly one GPU. The 572 runs belong to 569 job IDs: 3 runs are earlier instances of requeued
jobs (6712799, 6812162 requeued; 6820071 NODE_FAIL, then rerun), 0.31 GPU-h together.

## Method

Commands (login node, read-only):

```
export TZ=UTC
sacct -u jinboz1 -S 2026-10-03T00:00:00 -E now -X -D -P --delimiter=$'\x1f' \
  -o JobID,JobIDRaw,JobName%60,Partition,State,Submit,Start,End,Elapsed,ElapsedRaw,NodeList,AllocTRES%200,ReqTRES%200,WorkDir%300,Constraints%100,SubmitLine%5000
sacct -B -j 6652941                  # "NONE": the site does not store batch scripts
scontrol show config | grep -E 'AccountingStorageTRES|AccountingStoreFlags'   # no gres/gpu; no job_script flag
sinfo -N -o "%N|%P|%G|%f|%c|%m"; scontrol show node <node>                    # GPU nodes and types
scontrol show job <id>               # running jobs: TresPerNode=gres/gpu:1 (all 7 checked)
```

- `TZ=UTC` makes `-S` and all printed times UTC (the login node runs in America/New_York).
- `-X` gives one row per job allocation and per array element (pending array ranges such as `6820144_[0-23]` appear
  as one row with no node and are skipped). `-D` adds the earlier instances of requeued jobs, which used GPU time too.
- The `\x1f` delimiter is needed because `SubmitLine` loses quoting and 14 submit lines (`--wrap` scripts) contain
  newlines and `|`.
- Rows: 3,213, of which 2,989 started; 2,376 ran on a GPU node; 572 of those requested a GPU.

GPU type per node (`sinfo` GRES and features; the project calls the `nvidia_gtx1080` cards GTX 1080 Ti, 11 GB, see
docs/site_facts.md):

| GPU type | Nodes (feature) | GPUs per node |
|---|---|---|
| A30 | submit20-1, 21-1, 22-1, 23-1 (`nvidia_a30`) | 2 |
| RTX A6000 | submit37 (`nvidia_rtx6000`) | 2 |
| V100 | submit77-1 (`Tesla_v100`) | 4 |
| GTX 1080 Ti | submit60, 62-68, 71-73 (`nvidia_gtx1080`) | 4 |
| GTX 1080 Ti | t3btch042 (tier3, `nvidia_gtx1080ti`) | 1 (no job of ours ran there) |

Number of GPUs per run, in this order:

1. A run on a node without GPU GRES has 0 GPUs. No run with a parsed GPU request ran on such a node (0 cases).
2. Submit line: the last `--gres=gpu[:type]:N`, `--gpus[-per-node|-per-task]=N` or `-G N` before any `--wrap`.
   **409 runs, 238.64 GPU-h.**
3. `--wrap` submissions without a GPU flag: 0 GPUs (85 runs on GPU nodes).
4. Otherwise the batch script named on the submit line (relative paths resolved against `WorkDir`; when an unquoted
   `--export=...,CMD=python x.py` value hides the script, the last `*.sbatch`/`*.sh` token is used), read up to the
   first command line as `sbatch` does: `#SBATCH --gres=gpu:N` etc., else 0. **163 GPU runs, 335.22 GPU-h, from 25
   scripts** (largest: slurm/stage1/variant.sbatch 40 runs, slurm/addendum_g/acorn_infer.sbatch 21, slurm/baseline/*
   54). The same rule gave 0 GPUs for 1,709 runs on GPU nodes.
5. Script no longer on disk: 10 runs on GPU nodes (1.47 elapsed h), set to 0 GPUs (caveats).

Elapsed = `ElapsedRaw` (seconds, wall time from start to end; for running jobs, to the snapshot). The CSV column
`classified_by` gives rule 2 ("submit line") or rule 4 (the script path) per run.

Cross-check with the job logs (`/work/submit/jinboz1/hnl/logs/*_<jobid>.out`): 549 of the 572 GPU runs print a GPU
(Lightning "GPU available: True", the nvidia-smi header of slurm/*/_common.sh, or a CUDA device line); none prints
"GPU available: False"; 23 print nothing either way (probes, early failures). Of the CPU-classified runs on GPU nodes,
two matched, both through a results table that names the A30 of an earlier scoring job (fts_dry 6712913,
fts_analyze 6713002), not through GPU use.

## Consistency check with the 2026-10-05 07:10 UTC row (cumulative 306.80)

That row counted "jobs on the A30 nodes, 104 jobs, 44.4 GPU-h of them still running" at 1 GPU each. The same count
reproduces it exactly at a snapshot time of about 05:57 UTC: 104 jobs, 306.80 GPU-h, 44.43 GPU-h in running jobs.
So the query behind that row ran about 1 h 13 min before the time written in the row. The recount for the period up
to 07:10 UTC is 318.00 GPU-h. The difference of +11.20 GPU-h is:

| Item | GPU-h |
|---|---|
| Ledger row 2026-10-05 07:10 UTC | 306.80 |
| GPU time between 05:57 and 07:10 of the four A30 jobs running then (6671111 s1_ignn60_ii_s1, until 06:50; 6680785-6680787 s1_l1tl_i_s0..s2) | +4.54 |
| Four CPU-only jobs that ran on A30 nodes and were counted as GPU jobs: 6665472 s1_trunc_probe_armi_s1, 6665473 s1_attach, 6665474 s1_score_train_cpu_ii_s0, 6665475 s1_score_train_cpu_i_s1 (scripts have no `--gres`) | −5.96 |
| Two Stage-1 variant jobs on the RTX A6000s (submit37; slurm/stage1/variant.sbatch, `--gres=gpu:1`) left out by the A30-node filter: 6665445 s1_rz_i_s0 5.51, 6665505 s1_sym_i_s0 7.11 | +12.62 |
| V100 probe 6652958 (1 s) | +0.00 |
| **Recount up to 07:10 UTC** | **318.00** |

Two further checks against the ledger: the Stage-1 figures of that row (Canto-B 40.3, Canto-H 32.9) match
`s1_l1t_` 40.33 and `s1_l1tu_` 32.91 below, and the 2026-10-14 18:45 UTC row (secondary-particle upweighting, 8.30
GPU-h: 4.05 + 3.56 on A30, 0.38 on V100, 0.32 on RTX A6000) matches the recount of those 12 runs to 0.01 GPU-h. That
8.30 is part of the 255.86 after 2026-10-05 07:10 and must not be added again.

## By GPU type

| GPU type | GPU job runs | GPU-h | Share |
|---|---|---|---|
| A30 | 425 | 470.22 | 81.9% |
| GTX 1080 Ti | 87 | 67.78 | 11.8% |
| V100 | 50 | 21.34 | 3.7% |
| RTX A6000 | 10 | 14.52 | 2.5% |
| **Total** | **572** | **573.86** | 100% |

Per node: submit20-1 118.11, submit21-1 117.23, submit22-1 146.76, submit23-1 88.12, submit37 14.52, submit77-1 21.34,
submit62 38.31, submit63 11.54, submit68 8.57, submit72 3.66, submit65 2.60, submit60 1.86, submit73 1.24.

## By project phase (job-name prefix)

Groups are assigned by job-name prefix, first match wins, in this order: `probe_`, `gpu_tests`, `env_build`,
`smoke_baseline` (setup); `bl_`; `s1_secw`, `secw_`; `s1_l1tl_ii_`, `s1_l1tl_iii_`; other `s1_`; `fts_`; `tml_`;
`g_`; `g3_`. No GPU run fell outside these prefixes.

| Group | Job runs | GPU-h | Up to 2026-10-05 07:10 | After |
|---|---|---|---|---|
| Stage 1: Canto variants, screening ladder, scoring (other `s1_`) | 74 | 238.26 | 221.07 | 17.19 |
| Training-sample study, Canto-L arms ii / iii seeds (`s1_l1tl_ii_`, `s1_l1tl_iii_`) | 122 | 103.21 | 0.00 | 103.21 |
| TrackML (`tml_`) | 240 | 100.59 | 0.21 | 100.39 |
| Stage 0 baseline, acorn (`bl_`) | 54 | 96.02 | 96.02 | 0.00 |
| Addendum G T3 retraining (`g3_`) | 31 | 23.03 | 0.00 | 23.03 |
| Secondary-particle upweighting test (`s1_secw*`, `secw_`) | 12 | 8.30 | 0.00 | 8.30 |
| Addendum G T2 frozen-model scoring (`g_`) | 27 | 3.05 | 0.00 | 3.05 |
| Setup, probes and GPU tests | 8 | 0.70 | 0.70 | 0.00 |
| Far train-split check (`fts_`) | 4 | 0.70 | 0.00 | 0.70 |
| **Total** | **572** | **573.86** | **318.00** | **255.86** |

Two-token prefixes (all with ≥ 1 GPU-h):

| Prefix | Job runs | GPU-h | Prefix | Job runs | GPU-h |
|---|---|---|---|---|---|
| `s1_l1tl_` (Canto-L, all arms) | 125 | 148.44 | `g3_gnn_` | 9 | 6.96 |
| `bl_gnn_` | 18 | 61.24 | `tml_ml_` | 11 | 6.81 |
| `tml_gnn_` | 58 | 45.65 | `s1_rz_` | 1 | 5.51 |
| `tml_additions_` | 144 | 43.95 | `s1_secw18_` | 5 | 4.05 |
| `s1_l1t_` (Canto-B) | 3 | 40.33 | `s1_l1_` | 1 | 3.97 |
| `s1_ignn60_` | 6 | 35.83 | `s1_secwctrl_` | 5 | 3.56 |
| `s1_l1tu_` (Canto-H) | 2 | 32.91 | `s1_l3_` | 1 | 3.19 |
| `bl_ml_` | 19 | 32.72 | `s1_blind_` | 1 | 2.97 |
| `s1_triplet_` | 7 | 21.28 | `s1_score_` | 14 | 2.64 |
| `s1_symregion_` | 4 | 18.53 | `g_acorn_` | 21 | 2.30 |
| `g3_ml_` | 21 | 16.06 | `s1_ignn_` | 3 | 2.25 |
| `s1_sym_` | 5 | 11.84 | `tml_stage_` | 9 | 1.56 |
| `s1_ignn147_` | 14 | 10.08 | `bl_test_` | 11 | 1.49 |
| | | | `tml_g4_` | 10 | 1.25 |

The remaining 17 prefixes add up to 6.51 GPU-h (each < 1 GPU-h; see the CSV).

## By day (UTC)

GPU time is split at midnight UTC; "job runs active" counts runs with any GPU time that day.

| Day (UTC) | Job runs active | A30 | GTX 1080 Ti | V100 | RTX A6000 | Total | Running sum |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | 61 | 86.53 | 0.00 | 0.00 | 4.50 | 91.03 | 91.03 |
| 2026-10-04 | 51 | 176.28 | 0.00 | 0.00 | 8.12 | 184.39 | 275.42 |
| 2026-10-05 | 22 | 52.66 | 0.00 | 0.00 | 0.00 | 52.66 | 328.08 |
| 2026-10-06 | 72 | 10.14 | 32.15 | 3.37 | 0.08 | 45.73 | 373.82 |
| 2026-10-07 | 115 | 41.25 | 11.45 | 0.00 | 0.00 | 52.70 | 426.51 |
| 2026-10-08 | 30 | 19.95 | 0.00 | 0.00 | 0.89 | 20.84 | 447.35 |
| 2026-10-09 | 27 | 23.99 | 0.00 | 0.00 | 0.00 | 23.99 | 471.33 |
| 2026-10-10 | 22 | 3.80 | 0.00 | 0.54 | 0.00 | 4.34 | 475.67 |
| 2026-10-11 | 84 | 22.47 | 0.00 | 6.75 | 0.00 | 29.22 | 504.90 |
| 2026-10-12 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 504.90 |
| 2026-10-13 | 2 | 0.41 | 0.00 | 0.00 | 0.00 | 0.41 | 505.30 |
| 2026-10-14 | 107 | 31.57 | 22.62 | 9.30 | 0.93 | 64.42 | 569.72 |
| 2026-10-15 (to 00:35) | 11 | 1.19 | 1.56 | 1.39 | 0.00 | 4.14 | 573.86 |

## Running at the snapshot (2026-10-15 00:35:38 UTC)

Counted with elapsed time so far; they keep accruing after this recount.

| Job | Name | Node | GPU type | Start (UTC) | GPU-h so far |
|---|---|---|---|---|---|
| 6820662 | g3_gnn_train_s0 | submit65 | GTX 1080 Ti | 2026-10-14 23:50:30 | 0.75 |
| 6820664 | s1_l1tl_ii_s1 | submit21-1 | A30 | 2026-10-14 23:51:29 | 0.74 |
| 6820698 | s1_l1tl_iii_s1 | submit77-1 | V100 | 2026-10-14 23:55:29 | 0.67 |
| 6820808 | g3_gnn_train_s2 | submit77-1 | V100 | 2026-10-15 00:05:29 | 0.50 |
| 6820872 | s1_l1tl_iii_s2 | submit20-1 | A30 | 2026-10-15 00:20:27 | 0.25 |
| 6820879 | s1_l1tl_ii_s2 | submit62 | GTX 1080 Ti | 2026-10-15 00:23:29 | 0.20 |
| 6820882 | g3_gnn_train_s1 | submit77-1 | V100 | 2026-10-15 00:23:30 | 0.20 |
| | | | | **Total** | **3.32** |

40 sacct rows were pending (Addendum G union arrays, `g3_union_*`, `g3_gbdt_*`, `ov_py`); they are not counted.

## Caveats

- **GPUs from batch scripts (163 runs, 335.22 GPU-h).** The site stores no batch scripts (`sacct -B` gives "NONE"),
  so these runs take the GPU count from the script as it is on disk now. 10 of the 25 scripts were modified after
  their first job, but the git history of `slurm/` never removes or changes a `--gres` line: in each of the 24
  scripts in the repo, the one commit touching it is the script's first commit. Five scripts were used before their first commit and one is outside the repo
  (`/work/submit/jinboz1/hnl/tmp/g3_smoke/scan_smoke.sbatch`): 12 runs, 1.69 GPU-h, all in GPU-only partitions, 11 of
  them with a GPU line in their log (the twelfth, g3_smoke 6812085, 56 s, prints none).
- **Missing scripts (10 runs on GPU nodes, set to 0 GPUs).** Scratchpad and `logs/vE2_job*.sh` scripts are gone. Two
  logs show no GPU (6692747 po_canto_l_s0_mid, 1.26 h: "GPU available: False"; 6696366 vE2_verify2: "no GPU is
  available"). The other eight (s1cap_reporttest, vE2_verify*, d5_fig07, obl_check) are table and figure jobs with
  no GPU line in their logs; if they had held a GPU, the total would rise by at most 0.18 GPU-h.
- **CPU-only jobs on GPU nodes are not GPU-hours.** 1,804 runs with no GPU request ran on GPU nodes (498.36 elapsed
  hours: RTX A6000 node 207.37, V100 node 130.54, A30 nodes 148.74, GTX nodes 11.71), mostly via `-p submit,submit-gpu`
  or `submit-gpu-express`. They held CPUs and memory there, not GPUs. Counting them at 1 GPU each is the error in the
  2026-10-05 row (four such jobs).
- **GPU visibility without a request.** On the hosts split into a GPU node and a CPU node (submit20-23 `-1`/`-2`,
  submit77 `-1`/`-2`), jobs on the CPU node can see the host's GPUs: e.g. tml_test_grid 6771382_4 on submit21-2
  prints "NVIDIA A30" in its nvidia-smi header. The ledger rule counts requested GPUs only; the jobs concerned
  (track building, labels, diagnostics) run CPU code.
- **Elapsed includes all states.** Failed (5.17 GPU-h), timed-out (13.06), cancelled (0.98), requeued (0.07) and
  node-fail (0.24) runs are counted, as they held the GPU.
- **Start bound.** `-S 2026-10-03T00:00:00` UTC; the first GPU job of the project (6652941) started 01:46 UTC.
