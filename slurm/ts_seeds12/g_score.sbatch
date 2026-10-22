#!/bin/bash
# slurm/addendum_g/canto_score.sbatch with the output root ADDG_OUT (g_env.sh) and the R20 guard: the frozen Canto-L
# edge classifier on part k of its union blocks of one set, packed to ADDG_OUT/<model>_s<seed>/<set>/scored/parts.
# Usage: sbatch [--array=0-(N-1)] slurm/ts_seeds12/g_score.sbatch MODEL SEED SET [N]
#SBATCH -J g12_score
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|nvidia_rtx6000|Tesla_v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
source slurm/ts_seeds12/g_env.sh
MODEL=$1; SEED=$2; SET=$3; NP=${4:-1}
K=${SLURM_ARRAY_TASK_ID:-0}
OUT=$ADDG_OUT/${MODEL}_s${SEED}/$SET
PART=$OUT/scored/parts/$(printf %03d_of_%03d "$K" "$NP")
[ -e "$OUT/scored/done" ] && { echo "$OUT/scored is done: refusing to repeat it"; exit 1; }
[ -e "$PART/done" ] && { echo "$PART is done: refusing to repeat it"; exit 1; }
[ "$SET" = val ] || bash slurm/ts_seeds12/g_guard.sh "$SET"
read -r GRUN TM TAU K_ GNCK ARM < <(python -I -m hnl.eval.addendum_g canto-args --model "$MODEL" --seed "$SEED")
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(nvidia-smi --query-gpu=name --format=csv,noheader); $MODEL s$SEED $SET part $K of $NP; GNN $GNCK"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/in/testset"
trap 'rm -rf "$LOCAL"' EXIT
python -I -m hnl.eval.addendum_g gather-union --model "$MODEL" --seed "$SEED" --set "$SET" --dest "$LOCAL/in/testset" \
    --part "$K" --n-parts "$NP"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -I -m hnl.train.acorn_runner infer --stage gnn --ckpt "$GNCK" \
    --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" --accelerator gpu
mkdir -p "$PART"
python -I -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$PART"
cp "$LOCAL"/gnn/infer_*.json "$PART/"
touch "$PART/done"
