#!/bin/bash
# Learned fragment merging (CPU): fragment-pair data after the same-layer Mutex Watershed (s0 0.2) on a run's scored
# TRAINING graphs (gnn_scores/trainset.tar), the merge model, then the val grid with mwslmerge next to mwslayermerge.
# Usage: sbatch slurm/stage1/learned_merge.sbatch RUN_DIR [BUILDERS]   (outputs RUN_DIR/lmerge/, RUN_DIR/trackbuild_val_lmerge/)
#SBATCH -J s1_lmerge
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
BUILDERS=${2:-cc,mwslayer,mwslayermerge,mwslmerge}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/train" "$LOCAL/val" "$RUN/lmerge"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN"
tar -xf "$RUN/gnn_scores/trainset.tar" -C "$LOCAL/train"
python -m hnl.trackbuild.learned_merge data --graphs "$LOCAL/train" --out "$RUN/lmerge/pairs_train.npz"
python -m hnl.trackbuild.learned_merge train --data "$RUN/lmerge/pairs_train.npz" --out "$RUN/lmerge/model.joblib"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/trackbuild_val_lmerge" --builders "$BUILDERS" \
    --merge-model "$RUN/lmerge/model.joblib"
touch "$RUN/trackbuild_val_lmerge.done"
