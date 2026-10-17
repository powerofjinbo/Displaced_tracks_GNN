#!/bin/bash
# Addendum G T4: one CPU step of a phase (gather-scored, summary) after the guard. CPU.
# Usage: sbatch [-c N --mem=XG -J name] slurm/addendum_g_t4/cpu.sbatch PHASE SET <module> [args...]
#SBATCH -J g4_cpu
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
PHASE=$1; SET=$2; MOD=$3
shift 3
bash slurm/addendum_g_t4/guard.sh "$PHASE" "$SET"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}"
echo "python -m $MOD $*"
OMP_NUM_THREADS=1 python -I -m "$MOD" "$@"
