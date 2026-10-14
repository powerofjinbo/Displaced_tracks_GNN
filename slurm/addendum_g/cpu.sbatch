#!/bin/bash
# Addendum G: one CPU step of the T2 code with the generated samples registered (HNL_GENERATED=1): truth-only inputs
# (hnl.eval.addendum_g prepare-inputs / shard-identity, hnl.eval.addendum_g_truth), the packing of scored parts
# (gather-scored), summaries and read-outs. A30 and GTX 1080 nodes excluded.
# Usage: sbatch [-c N --mem=XG -t HH:MM:SS -J name] slurm/addendum_g/cpu.sbatch <module> [args...]
#SBATCH -J g_cpu
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
MOD=$1
shift
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}"
echo "python -m $MOD $*"
OMP_NUM_THREADS=1 python -I -m "$MOD" "$@"
