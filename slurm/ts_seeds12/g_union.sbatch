#!/bin/bash
# slurm/addendum_g/canto_union.sbatch with the output root ADDG_OUT (g_env.sh) and the R20 guard: the Canto-L union
# graphs of one block of a set from the acorn arm's scored graphs of that set, with the run's learned triplet filter.
# Usage: sbatch --array=0-(N-1) slurm/ts_seeds12/g_union.sbatch MODEL SEED SET N
#SBATCH -J g12_union
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
source slurm/ts_seeds12/g_env.sh
MODEL=$1; SEED=$2; SET=$3; NB=${4:-1}
BLK=${SLURM_ARRAY_TASK_ID:-0}
OUT=$ADDG_OUT/${MODEL}_s${SEED}/$SET/union/blocks/$(printf %03d_of_%03d "$BLK" "$NB")
[ -e "$OUT/done" ] && { echo "$OUT is done: refusing to repeat it"; exit 1; }
[ "$SET" = val ] || bash slurm/ts_seeds12/g_guard.sh "$SET"
read -r GRUN TM TAU K GNCK ARM < <(python -I -m hnl.eval.addendum_g canto-args --model "$MODEL" --seed "$SEED")
SRC=$ADDG_OUT/acorn_${ARM}_s${SEED}/$SET
[ -e "$SRC/done" ] || { echo "no finished acorn graphs at $SRC"; exit 1; }
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $MODEL s$SEED $SET block $BLK of $NB; graphs run $GRUN; triplet $TM tau $TAU k $K; source $SRC"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/u" "$LOCAL/x" "$OUT"
trap 'rm -rf "$LOCAL"' EXIT
python -I -m hnl.eval.addendum_g block-tar --tar "$SRC/testset.tar" --block "$BLK" --n-blocks "$NB" --dest "$LOCAL/x/testset.tar"
export OMP_NUM_THREADS=1
python -I -m hnl.graphs.helix_union build --ml-run "$GRUN" --split test --tar "$LOCAL/x/testset.tar" --no-metrics \
    --out "$LOCAL/u" --report "$OUT/report" --triplet-model "$TM" --tau "$TAU" --k "$K"
python -I -m hnl.train.acorn_runner pack --src "$LOCAL/u" --dest "$OUT"
touch "$OUT/done"
