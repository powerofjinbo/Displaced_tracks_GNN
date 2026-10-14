#!/bin/bash
# Addendum G: the frozen Canto-L edge classifier on its union graphs of one set (every block of canto_union.sbatch; as
# slurm/stage1/score_split.sbatch on the union test graphs), packed to
# DATA/runs/addendum_g_v1/<model>_s<seed>/<set>/scored/testset.tar. Usage: sbatch slurm/addendum_g/canto_score.sbatch MODEL SEED SET
#SBATCH -J g_score
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
export HNL_GENERATED=1
MODEL=$1; SEED=$2; SET=$3
OUT=$HNL_DATA/runs/addendum_g_v1/${MODEL}_s${SEED}/$SET
[ -e "$OUT/scored/done" ] && { echo "$OUT/scored is done: refusing to repeat it"; exit 1; }
[ "$SET" = val ] || bash slurm/addendum_g/guard.sh "$SET"
read -r GRUN TM TAU K GNCK ARM < <(python -I -m hnl.eval.addendum_g canto-args --model "$MODEL" --seed "$SEED")
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(nvidia-smi --query-gpu=name --format=csv,noheader); $MODEL s$SEED $SET; GNN $GNCK"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/in/testset"
trap 'rm -rf "$LOCAL"' EXIT
python -I -m hnl.eval.addendum_g gather-union --model "$MODEL" --seed "$SEED" --set "$SET" --dest "$LOCAL/in/testset"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -I -m hnl.train.acorn_runner infer --stage gnn --ckpt "$GNCK" \
    --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" --accelerator gpu
mkdir -p "$OUT/scored"
python -I -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$OUT/scored"
cp "$LOCAL"/gnn/infer_*.json "$OUT/scored/"
touch "$OUT/scored/done"
