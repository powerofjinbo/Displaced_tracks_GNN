#!/bin/bash
# Small CPU step of the pileup overlay (hnl.overlay.*): truth only, no model.
#   sbatch --export=ALL,CMD="python -u -m hnl.overlay.emulate select --work DIR" run_py.sbatch
#SBATCH -J ov_py
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:50:00
#SBATCH -o /work/submit/jinboz1/hnl/pileup_design/hp_overlay/logs/%x_%j.out
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
source env/activate.sh > /dev/null 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "host $(hostname) start $(date -u) git $(git rev-parse --short HEAD)"
eval "$CMD"
echo "end $(date -u)"
