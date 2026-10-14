#!/bin/bash
# Addendum G step T1 (truth only): checksums, event checks and the per-particle truth table of one generated sample
# (hnl.eval.addendum_g_t1 scan). CPU; A30 and GTX 1080 nodes excluded.
# Usage: sbatch --array=0-3 slurm/addendum_g/t1_scan.sbatch
#SBATCH -J g_t1_scan
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
SAMPLES=(generic_iso generic_dense hss15_100 hss40_300)
S=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; sample $S"
OMP_NUM_THREADS=1 python -I -m hnl.eval.addendum_g_t1 scan --sample "$S" --workers "$SLURM_CPUS_PER_TASK"
