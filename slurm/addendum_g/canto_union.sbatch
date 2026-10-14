#!/bin/bash
# Addendum G: the Canto-L union graphs of one set (hnl.graphs.helix_union build --no-metrics on the frozen acorn arm's
# scored graphs of that set, with the run's learned triplet filter; as slurm/stage1/union_graphs.sbatch RUN test), one
# block of the sorted graphs per array task (each block a tar of the same members, so the build reads it exactly as
# the frozen test read RUN/gnn_scores_test/testset.tar). CPU; A30 and GTX 1080 nodes excluded.
# Usage: sbatch --array=0-(N-1) slurm/addendum_g/canto_union.sbatch MODEL SEED SET N
#SBATCH -J g_union
#SBATCH -p submit-gpu-express,submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
MODEL=$1; SEED=$2; SET=$3; NB=${4:-1}
BLK=${SLURM_ARRAY_TASK_ID:-0}
OUT=$HNL_DATA/runs/addendum_g_v1/${MODEL}_s${SEED}/$SET/union/blocks/$(printf %03d_of_%03d "$BLK" "$NB")
[ -e "$OUT/done" ] && { echo "$OUT is done: refusing to repeat it"; exit 1; }
[ "$SET" = val ] || bash slurm/addendum_g/guard.sh "$SET"
read -r GRUN TM TAU K GNCK ARM < <(python -I -m hnl.eval.addendum_g canto-args --model "$MODEL" --seed "$SEED")
SRC=$HNL_DATA/runs/addendum_g_v1/acorn_${ARM}_s${SEED}/$SET
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
