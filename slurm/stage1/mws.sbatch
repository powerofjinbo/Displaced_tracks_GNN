#!/bin/bash
# Mutex Watershed track building (CPU): pair data (graph edges + 2-hop pairs) from a run's scored TRAINING graphs (gnn_scores/trainset.tar), the
# same-particle pair head, then the val grid with the mws families next to CC and walkthrough + merging.
# Usage: sbatch slurm/stage1/mws.sbatch RUN_DIR [BUILDERS]     (outputs RUN_DIR/mws/, RUN_DIR/trackbuild_val_mws/)
#SBATCH -J s1_mws
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
BUILDERS=${2:-cc,walkmergeS,mws,mwslift,mwsliftmerge,mwslayer,mwslayermerge,mwsrep}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/train" "$LOCAL/val" "$RUN/mws"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN"
tar -xf "$RUN/gnn_scores/trainset.tar" -C "$LOCAL/train"
python -m hnl.trackbuild.mutex_ws data --graphs "$LOCAL/train" --out "$RUN/mws/pairs_train.npz" --max-events 3000 --row-frac 0.2
python -m hnl.trackbuild.mutex_ws train --data "$RUN/mws/pairs_train.npz" --out "$RUN/mws/model.joblib"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/trackbuild_val_mws" --builders "$BUILDERS" \
    --mws-model "$RUN/mws/model.joblib"
touch "$RUN/trackbuild_val_mws.done"
