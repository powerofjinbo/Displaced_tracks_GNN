#!/bin/bash
# Addendum G T4 (post): builder and double-majority matching of one evaluation run (acorn_g, canto_g, acorn_g_alt,
# canto_g_alt), seed and set, one block of events (hnl.eval.addendum_g_t4 evaluate); the merge after the last block. CPU.
# Usage: sbatch [--array=0-(N-1)] slurm/addendum_g_t4/evaluate.sbatch RUN SEED SET [N_BLOCKS]   (SELECTION: val tests)
#SBATCH -J g4_eval
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
RUN=$1; SEED=$2; SET=$3; NB=${4:-1}
BLK=${SLURM_ARRAY_TASK_ID:-0}
bash slurm/addendum_g_t4/guard.sh post "$SET"
[ "$SET" != val ] && [ -n "${SELECTION:-}" ] && { echo "SELECTION is for val code tests only"; exit 1; }
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $RUN s$SEED $SET block $BLK of $NB"
OMP_NUM_THREADS=1 python -I -m hnl.eval.addendum_g_t4 evaluate --run "$RUN" --seed "$SEED" --set "$SET" \
    --block "$BLK" --n-blocks "$NB" --workers "$SLURM_CPUS_PER_TASK" ${SELECTION:+--selection "$SELECTION"}
D=$HNL_DATA/runs/addendum_g_t4_v1/eval/${RUN}_s${SEED}/$SET/blocks
if [ "$(ls "$D"/*/done 2>/dev/null | wc -l)" = "$NB" ] && [ ! -e "$D/../per_particle.parquet" ]; then
    python -I -m hnl.eval.addendum_g_t4 merge --run "$RUN" --seed "$SEED" --set "$SET" || true
fi
