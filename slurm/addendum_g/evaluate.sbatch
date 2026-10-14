#!/bin/bash
# Addendum G: frozen builder and double-majority matching of one model, seed and set, one block of events
# (hnl.eval.addendum_g evaluate); then, if every block is done, the merge. CPU; A30 and GTX 1080 nodes excluded.
# Usage: sbatch [--array=0-(N-1)] slurm/addendum_g/evaluate.sbatch MODEL SEED SET [N_BLOCKS]
#SBATCH -J g_eval
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
MODEL=$1; SEED=$2; SET=$3; NB=${4:-1}
BLK=${SLURM_ARRAY_TASK_ID:-0}
[ "$SET" = val ] || bash slurm/addendum_g/guard.sh "$SET"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $MODEL s$SEED $SET block $BLK of $NB"
OMP_NUM_THREADS=1 python -I -m hnl.eval.addendum_g evaluate --model "$MODEL" --seed "$SEED" --set "$SET" \
    --block "$BLK" --n-blocks "$NB" --workers "$SLURM_CPUS_PER_TASK"
D=$HNL_DATA/runs/addendum_g_v1/eval/${MODEL}_s${SEED}/$SET/blocks
if [ "$(ls "$D"/*/done 2>/dev/null | wc -l)" = "$NB" ] && [ ! -e "$D/../per_particle.parquet" ]; then
    python -I -m hnl.eval.addendum_g merge --model "$MODEL" --seed "$SEED" --set "$SET" || true
fi
