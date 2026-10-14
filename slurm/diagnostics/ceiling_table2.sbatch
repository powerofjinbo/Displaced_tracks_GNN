#!/bin/bash
# Ceilings of the training-sample comparison (Table 2), val only: one block of one run's scored val graphs through
# hnl.eval.ceiling_table2 run (perfect edge classifier: graph ceiling with connected components, and learned merging with
# the run's own merge model). CPU only; never on the A30 or GTX1080 nodes.
# Usage: sbatch -J ceil_<TAG>_<K> slurm/diagnostics/ceiling_table2.sbatch TAG RUN MERGE_MODEL TAU BLOCK NBLOCKS
#    or: sbatch --array=0-13%5 --export=ALL,PLAN=FILE slurm/diagnostics/ceiling_table2.sbatch  (one run per task: line
#        TASK+1 of FILE, "TAG RUN MERGE_MODEL TAU" as printed by python -m hnl.eval.ceiling_table2 plan; one block)
#   writes $CEIL_BASE/TAG/block_<BLOCK>of<NBLOCKS>/ (default CEIL_BASE: $HNL_DATA/runs/ceiling_table2_v1)
#SBATCH -J ceil_t2
#SBATCH -p submit-gpu,submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
if [ -n "${PLAN:-}" ]; then
    read -r TAG RUN MODEL TAU < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PLAN"); K=0; N=1
else
    TAG=$1; RUN=$2; MODEL=$3; TAU=$4; K=$5; N=$6
fi
BASE=${CEIL_BASE:-/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/ceiling_table2_v1}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/val"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; tag $TAG; run $RUN; model $MODEL; tau $TAU; block $K of $N; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
echo "$(ls "$LOCAL/val" | wc -l) val graphs"
python -m hnl.eval.ceiling_table2 run --graphs "$LOCAL/val" --merge-model "$MODEL" --tau "$TAU" --out "$BASE/$TAG" \
    --block "$K" --nblocks "$N" ${MAX_EVENTS:+--max-events $MAX_EVENTS}
echo "done $(date -u +%FT%TZ)"
