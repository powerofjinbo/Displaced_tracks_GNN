#!/bin/bash
# slurm/addendum_g/evaluate.sbatch with the output root ADDG_OUT (g_env.sh) and the R20 guard: frozen builder and
# double-majority matching of one model, seed and set, one block of events; the last finished block merges.
# Usage: sbatch [--array=0-(N-1)] slurm/ts_seeds12/g_eval.sbatch MODEL SEED SET [N_BLOCKS]
#SBATCH -J g12_eval
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
source slurm/ts_seeds12/g_env.sh
MODEL=$1; SEED=$2; SET=$3; NB=${4:-1}
BLK=${SLURM_ARRAY_TASK_ID:-0}
[ "$SET" = val ] || bash slurm/ts_seeds12/g_guard.sh "$SET"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $MODEL s$SEED $SET block $BLK of $NB"
OMP_NUM_THREADS=1 python -I -m hnl.eval.addendum_g evaluate --model "$MODEL" --seed "$SEED" --set "$SET" \
    --block "$BLK" --n-blocks "$NB" --workers "$SLURM_CPUS_PER_TASK"
D=$ADDG_OUT/eval/${MODEL}_s${SEED}/$SET/blocks
if [ "$(ls "$D"/*/done 2>/dev/null | wc -l)" = "$NB" ] && [ ! -e "$D/../per_particle.parquet" ]; then
    python -I -m hnl.eval.addendum_g merge --model "$MODEL" --seed "$SEED" --set "$SET" || true
fi
