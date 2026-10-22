#!/bin/bash
# slurm/addendum_g/cpu.sbatch with g_env.sh (ADDG_OUT, ADDG_TS_SEEDS): one CPU step of the T2 code (gather-scored,
# the summaries of hnl.eval.addendum_g_seeds12).
# Usage: sbatch [-c N --mem=XG -t HH:MM:SS -J name] slurm/ts_seeds12/g_cpu.sbatch <module> [args...]
#SBATCH -J g12_cpu
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
source slurm/ts_seeds12/g_env.sh
MOD=$1
shift
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}"
echo "python -m $MOD $*"
OMP_NUM_THREADS=1 python -I -m "$MOD" "$@"
