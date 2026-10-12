#!/bin/bash
# Vertex efficiency (Stage 1 D, also for the baseline): both HNL daughters DM-matched and paired into one two-track
# vertex, per builder configuration, on a run's scored val graphs (CPU). Usage:
#   sbatch slurm/stage1/vertex_eff.sbatch RUN_DIR OUTNAME BUILDER [BUILDER...]   (BUILDER e.g. cc:0.40, walkmerge:0.1:0.5:1.0)
#SBATCH -J s1_vertex_eff
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
OUTNAME=$2
shift 2
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; builders $*"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
args=()
for b in "$@"; do args+=(--builder "$b"); done
python -m hnl.eval.vertex_eff --graphs "$LOCAL/s" --out "$RUN/$OUTNAME" "${args[@]}"
touch "$RUN/$OUTNAME.done"
