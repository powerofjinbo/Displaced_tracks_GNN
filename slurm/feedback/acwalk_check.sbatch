#!/bin/bash
# Feedback round 1: hnl.trackbuild.acorn_walk against acorn's own CCandWalk._build_tracks_one_evt on every K-th val graph
# of a run, every acwalk / acwalkg configuration (identical track partitions required).
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/acwalk_check.sbatch RUN_DIR K OUT_JSON
#   TAR (env, default gnn_scores/valset.tar; test: gnn_scores_test/testset.tar), CONFIGS (env, optional: only these
#   configurations, "fam,cut,min,add;fam,cut,min,add")
#SBATCH -J fb_acwalk_check
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; K=$2; OUT=$3
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); host $(hostname); job $SLURM_JOB_ID; run $RUN; every ${K}-th val graph; $(date -u +%FT%TZ)"
TAR=${TAR:-gnn_scores/valset.tar}
tar -tf "$RUN/$TAR" | sort | awk -v k="$K" 'NR % k == 1' > "$LOCAL/list.txt"
tar -xf "$RUN/$TAR" -C "$LOCAL/s" -T "$LOCAL/list.txt"
echo "$(ls "$LOCAL/s" | wc -l) graphs of $TAR"
python -m hnl.trackbuild.acorn_walk check --graphs "$LOCAL/s" --out "$OUT" ${CONFIGS:+--configs "$CONFIGS"}
echo "done $(date -u +%FT%TZ)"
