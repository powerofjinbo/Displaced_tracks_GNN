#!/bin/bash
# Addendum G T4 (pre): acorn (ttbar+G) seed S on one set: metric-learning graphs with the run's checkpoint and
# (r_infer, knn_infer) of its val inference, then its edge classifier (the checkpoint of its val scores); packed to
# DATA/runs/addendum_g_t4_v1/acorn_g_s<S>/<set>/testset.tar with a sha256 manifest. No metric. GPU with FRNN.
# Usage: sbatch slurm/addendum_g_t4/acorn_infer.sbatch SEED SET
#SBATCH -J g4_acorn
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|nvidia_rtx6000|Tesla_v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
SEED=$1; SET=$2
OUT=$HNL_DATA/runs/addendum_g_t4_v1/acorn_g_s${SEED}/$SET
[ -e "$OUT/done" ] && { echo "$OUT is done: refusing to repeat it"; exit 1; }
bash slurm/addendum_g_t4/guard.sh pre "$SET"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(nvidia-smi --query-gpu=name --format=csv,noheader)"
read -r MLCK RINF KINF GNCK < <(python -I -m hnl.eval.addendum_g_t4 acorn-args --seed "$SEED")
echo "acorn (ttbar+G) seed $SEED set $SET: ML $MLCK (r $RINF, knn $KINF); GNN $GNCK"
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
python -I -m hnl.eval.addendum_g_t4 manifest --what acorn --seed "$SEED" --set "$SET"
touch "$OUT/done"
