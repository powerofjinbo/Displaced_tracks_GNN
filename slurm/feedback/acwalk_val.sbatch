#!/bin/bash
# Feedback round 1 (strongest baselines, CPU): acorn's own CC + walkthrough builder (CCandWalk, plain and score-guided;
# hnl.trackbuild.acorn_walk) on a run's scored val graphs, the acwalk / acwalkg grid of hnl.eval.trackbuild_val.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/acwalk_val.sbatch RUN_DIR [BUILDERS] [OUTNAME]
#   (RUN_DIR holds gnn_scores/valset.tar; writes RUN_DIR/OUTNAME, default trackbuild_val_acwalk, a new sub-directory)
#SBATCH -J fb_acwalk_val
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
BUILDERS=${2:-acwalk,acwalkg}
OUTNAME=${3:-trackbuild_val_acwalk}
[ -e "$RUN/$OUTNAME/per_particle.parquet" ] && { echo "$RUN/$OUTNAME exists: refusing to overwrite"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; builders $BUILDERS; $(date -u +%FT%TZ)"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
echo "$(ls "$LOCAL/s" | wc -l) val graphs"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/s" --out "$RUN/$OUTNAME" --builders "$BUILDERS"
echo "done $(date -u +%FT%TZ)"
