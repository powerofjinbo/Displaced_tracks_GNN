#!/bin/bash
# Run a python module on a run's scored val graphs (CPU): extracts RUN/gnn_scores/valset.tar to node-local disk and
# calls python -m MODULE --graphs DIR ARGS... (e.g. hnl.eval.loss_classes).
# Usage: sbatch -J NAME slurm/stage1/val_py.sbatch RUN_DIR MODULE [ARGS...]
#SBATCH -J s1_val_py
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; MOD=$2; shift 2
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; $MOD $*"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
python -m "$MOD" --graphs "$LOCAL/s" "$@"
