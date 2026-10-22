#!/bin/bash
# slurm/addendum_g/acorn_infer.sbatch with the output root ADDG_OUT (g_env.sh) and the R20 guard: one frozen acorn run
# (here arm iii, seeds 1 and 2) on one set: metric-learning graphs with the checkpoint and (r_infer, knn_infer) of the
# run's one-time test inference, then its edge classifier, packed to ADDG_OUT/acorn_<arm>_s<seed>/<set>/testset.tar.
# Usage: sbatch slurm/ts_seeds12/g_acorn.sbatch ARM SEED SET
#SBATCH -J g12_acorn
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|nvidia_rtx6000|Tesla_v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
source slurm/ts_seeds12/g_env.sh
ARM=$1; SEED=$2; SET=$3
OUT=$ADDG_OUT/acorn_${ARM}_s${SEED}/$SET
[ -e "$OUT/done" ] && { echo "$OUT is done: refusing to repeat it"; exit 1; }
[ "$SET" = val ] || bash slurm/ts_seeds12/g_guard.sh "$SET"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(nvidia-smi --query-gpu=name --format=csv,noheader)"
read -r MLCK RINF KINF GNCK < <(python -I -m hnl.eval.addendum_g acorn-args --arm "$ARM" --seed "$SEED")
echo "acorn arm $ARM seed $SEED set $SET: ML $MLCK (r $RINF, knn $KINF); GNN $GNCK"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
python -I -m hnl.eval.addendum_g stage --set "$SET" --dest "$LOCAL/fs/testset"
python -I -m hnl.train.acorn_runner infer --stage ml --ckpt "$MLCK" --input-dir "$LOCAL/fs" --out-dir "$LOCAL/ml" \
    --set r_infer=$RINF --set knn_infer=$KINF
rm -rf "$LOCAL/fs"
python -I -m hnl.train.acorn_runner infer --stage gnn --ckpt "$GNCK" --input-dir "$LOCAL/ml" --out-dir "$LOCAL/gnn"
mkdir -p "$OUT"
python -I -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$OUT"
cp "$LOCAL"/gnn/infer_*.json "$OUT/"
for f in "$LOCAL"/ml/infer_*.json; do cp "$f" "$OUT/ml_$(basename "$f")"; done
touch "$OUT/done"
