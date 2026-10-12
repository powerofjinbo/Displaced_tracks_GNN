#!/bin/bash
# Learned dendrogram cut (CPU): node data from a run's scored TRAINING graphs (gnn_scores/trainset.tar), the
# node-quality model, then the val track-building grid with the 'ldc' family next to CC and walkthrough + merging.
# Usage: sbatch slurm/stage1/ldc.sbatch RUN_DIR [BUILDERS]     (outputs RUN_DIR/ldc/, RUN_DIR/trackbuild_val_ldc/)
#SBATCH -J s1_ldc
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
BUILDERS=${2:-cc,ldc,walkmerge}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/train" "$LOCAL/val" "$RUN/ldc"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN"
tar -xf "$RUN/gnn_scores/trainset.tar" -C "$LOCAL/train"
python -m hnl.trackbuild.ldc data --graphs "$LOCAL/train" --out "$RUN/ldc/nodes_train.npz"
python -m hnl.trackbuild.ldc train --data "$RUN/ldc/nodes_train.npz" --out "$RUN/ldc/model.joblib"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/trackbuild_val_ldc" --builders "$BUILDERS" \
    --ldc-model "$RUN/ldc/model.joblib"
touch "$RUN/trackbuild_val_ldc.done"
