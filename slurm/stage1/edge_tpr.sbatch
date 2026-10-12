#!/bin/bash
# Edge TPR and graph efficiency vs truth |d0| and pT (M4 diagnostic) for each listed run's scored val graphs, at the
# given cuts. Usage: sbatch slurm/stage1/edge_tpr.sbatch "RUN1 RUN2 ..." CUT [CUT...]   (writes RUN/edge_tpr_val.csv)
#SBATCH -J s1_edge_tpr
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
shift
cuts=()
for c in "$@"; do cuts+=(--cut "$c"); done
LOCAL=/tmp/$USER/$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); host $(hostname); job $SLURM_JOB_ID; runs $RUNS; cuts $*"
for RUN in $RUNS; do
    mkdir -p "$LOCAL/s"
    tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
    echo "== $RUN"
    python -m hnl.eval.edge_tpr --graphs "$LOCAL/s" --out "$RUN/edge_tpr_val.csv" "${cuts[@]}"
    rm -rf "$LOCAL/s"
done
