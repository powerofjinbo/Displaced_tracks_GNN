#!/bin/bash
# Causal inner-hit truncation probe (M4 diagnostic, CPU) for each listed run: writes RUN/truncation_probe/.
# Usage: sbatch slurm/stage1/truncation_probe.sbatch "RUN1 RUN2 ..." [N_EVENTS]
#SBATCH -J s1_trunc_probe
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID"
for RUN in $1; do
    echo "== $RUN"
    mkdir -p "$LOCAL/w"
    G=$RUN/ml_graphs/valset.tar                            # Stage-1 runs: their baseline run's graphs
    [ -e "$G" ] || G=$HNL_DATA/runs/baseline_v1/$(basename "$RUN")/ml_graphs/valset.tar
    OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.eval.truncation_probe --run "$RUN" --out "$RUN/truncation_probe" \
        --work "$LOCAL/w" --n-events "${2:-300}" --graphs "$G"
    rm -rf "$LOCAL/w"
done
