#!/bin/bash
# Addendum G T4 (pre): Canto-L (ttbar+G) seed S, union graphs of one set, block K of N (array task): helix_union build
# --no-metrics on the acorn (ttbar+G) graphs of the set (returned to their metric-learning form, as the frozen test
# chain) with the run's final learned triplet support; packed with a sha256 manifest. No metric. CPU.
# Usage: sbatch --array=0-(N-1) slurm/addendum_g_t4/canto_union.sbatch SEED SET N
#SBATCH -J g4_union
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
SEED=$1; SET=$2; NB=${3:-1}
BLK=${SLURM_ARRAY_TASK_ID:-0}
OUT=$HNL_DATA/runs/addendum_g_t4_v1/canto_g_s${SEED}/$SET/union/blocks/$(printf %03d_of_%03d "$BLK" "$NB")
[ -e "$OUT/done" ] && { echo "$OUT is done: refusing to repeat it"; exit 1; }
bash slurm/addendum_g_t4/guard.sh pre "$SET"
read -r GRUN TM TAU K < <(python -I -m hnl.eval.addendum_g_t4 union-args --seed "$SEED")
SRC=$HNL_DATA/runs/addendum_g_t4_v1/acorn_g_s${SEED}/$SET
[ -e "$SRC/done" ] || { echo "no finished acorn (ttbar+G) graphs at $SRC"; exit 1; }
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; s$SEED $SET block $BLK of $NB; graphs run $GRUN; triplet $TM tau $TAU k $K"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/u" "$LOCAL/x" "$OUT"
trap 'rm -rf "$LOCAL"' EXIT
python -I -m hnl.eval.addendum_g block-tar --tar "$SRC/testset.tar" --block "$BLK" --n-blocks "$NB" --dest "$LOCAL/x/testset.tar"
export OMP_NUM_THREADS=1
python -I -m hnl.graphs.helix_union build --ml-run "$GRUN" --split test --tar "$LOCAL/x/testset.tar" --no-metrics \
    --out "$LOCAL/u" --report "$OUT/report" --triplet-model "$TM" --tau "$TAU" --k "$K" --workers "${WORKERS:-10}"
python -I -m hnl.train.acorn_runner pack --src "$LOCAL/u" --dest "$OUT"
python -I -m hnl.eval.addendum_g_t4 manifest --what union --seed "$SEED" --set "$SET" --block "$BLK" --n-blocks "$NB"
touch "$OUT/done"
