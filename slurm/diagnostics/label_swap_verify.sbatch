#!/bin/bash
# Independent check of the hit-label exchange (scripts/diagnostics/label_swap_verify.py); truth only.
#   sbatch [-c N --mem=XG -t HH:MM:SS] slurm/diagnostics/label_swap_verify.sbatch <script in scripts/diagnostics> <args>
#SBATCH -J lsv
#SBATCH -p submit-gpu-express,submit
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); host $(hostname); job ${SLURM_JOB_ID:-none}"
SCRIPT=$1
shift
python "scripts/diagnostics/$SCRIPT" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
