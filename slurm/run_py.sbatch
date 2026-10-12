#!/bin/bash
# Generic CPU job: run one python module of this repo with the given arguments.
#   sbatch [-c N --mem=XG -t HH:MM:SS -J name] slurm/run_py.sbatch <module> [args...]
# Records the git hash and any uncommitted changes under src/, env/, slurm/, configs/.
#SBATCH -J run_py
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
MOD=$1
shift
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src env slurm configs | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}"
echo "python -m $MOD $*"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} python -I -m "$MOD" "$@"
