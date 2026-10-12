#!/bin/bash
# Edge TPR vs truth |d0| at each listed run's own cut giving a target edge FPR on its pooled HNL val fake edges, and dD
# (preregistration section 6c; read by hnl.eval.ladder_report). TARGET: a working_points.json (its e_star) or a number.
# Usage: sbatch slurm/stage1/edge_tpr_estar.sbatch "RUN1 RUN2 ..." TARGET
#   (writes RUN/edge_tpr_estar.csv and RUN/edge_tpr_estar_efpr.json)
#SBATCH -J s1_edge_tpr_estar
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUNS=$1
if [ -f "$2" ]; then target=(--wp "$2"); else target=(--efpr "$2"); fi
LOCAL=/tmp/$USER/$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); host $(hostname); job $SLURM_JOB_ID; runs $RUNS; target $2"
for RUN in $RUNS; do
    mkdir -p "$LOCAL/s"
    tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
    echo "== $RUN"
    python -m hnl.eval.edge_tpr --graphs "$LOCAL/s" --out "$RUN/edge_tpr_estar.csv" "${target[@]}"
    rm -rf "$LOCAL/s"
done
