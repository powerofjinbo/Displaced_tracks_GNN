#!/bin/bash
# Stage 1 C (CPU): alternative track builders on a run's scored val graphs (no retraining).
# Usage: sbatch slurm/stage1/trackbuild.sbatch RUN_DIR [BUILDERS] [OUTNAME]   (RUN_DIR holds gnn_scores/valset.tar)
# TB_EXTRA: extra arguments of hnl.eval.trackbuild_val, e.g. "--ldc-model RUN/ldc/model.joblib" (sbatch --export=ALL,TB_EXTRA=...)
#SBATCH -J s1_trackbuild
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
BUILDERS=${2:-cc,jr,walk,ccwalk,ccmerge,walkmerge,ccmergeany}
OUTNAME=${3:-trackbuild_val}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); host $(hostname); job $SLURM_JOB_ID; run $RUN; builders $BUILDERS"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/s" --out "$RUN/$OUTNAME" --builders "$BUILDERS" ${TB_EXTRA:-}
touch "$RUN/$OUTNAME.done"
