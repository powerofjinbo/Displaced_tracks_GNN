#!/bin/bash
# Addendum G: event tables and acorn feature-store tars of the generated samples (truth only; hnl.eval.addendum_g
# prepare), after the T1 split lists exist. CPU; A30 and GTX 1080 nodes excluded.
#SBATCH -J g_prepare
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 24
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID"
OMP_NUM_THREADS=1 python -I -m hnl.eval.addendum_g prepare --workers "$SLURM_CPUS_PER_TASK"
