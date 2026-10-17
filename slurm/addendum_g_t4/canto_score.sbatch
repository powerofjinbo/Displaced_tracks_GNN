#!/bin/bash
# Addendum G T4 (post): the Canto-L (ttbar+G) classifier of seed S (checkpoint from the selection file) on part K of N
# of its union blocks of one set (array task); packed to .../canto_g_s<S>/<set>/scored/parts/<k>_of_<N>/. GPU.
# Usage: sbatch [--array=0-(N-1)] slurm/addendum_g_t4/canto_score.sbatch SEED SET [N]   (CANTO_CKPT, SELECTION: val tests)
#SBATCH -J g4_score
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
SEED=$1; SET=$2; NP=${3:-1}
K=${SLURM_ARRAY_TASK_ID:-0}
OUT=$HNL_DATA/runs/addendum_g_t4_v1/canto_g_s${SEED}/$SET
PART=$OUT/scored/parts/$(printf %03d_of_%03d "$K" "$NP")
[ -e "$OUT/scored/done" ] && { echo "$OUT/scored is done: refusing to repeat it"; exit 1; }
[ -e "$PART/done" ] && { echo "$PART is done: refusing to repeat it"; exit 1; }
bash slurm/addendum_g_t4/guard.sh post "$SET"
if [ "$SET" != val ] && { [ -n "${CANTO_CKPT:-}" ] || [ -n "${SELECTION:-}" ]; }; then echo "overrides are for val code tests only"; exit 1; fi
GNCK=$(python -I -m hnl.eval.addendum_g_t4 canto-args --seed "$SEED" ${SELECTION:+--selection "$SELECTION"} ${CANTO_CKPT:+--canto-ckpt "$CANTO_CKPT"})
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(nvidia-smi --query-gpu=name --format=csv,noheader); s$SEED $SET part $K of $NP; GNN $GNCK"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/in/testset"
trap 'rm -rf "$LOCAL"' EXIT
python -I -m hnl.eval.addendum_g_t4 gather-union --seed "$SEED" --set "$SET" --dest "$LOCAL/in/testset" --part "$K" --n-parts "$NP"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -I -m hnl.train.acorn_runner infer --stage gnn --ckpt "$GNCK" \
    --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" --accelerator gpu
mkdir -p "$PART"
python -I -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$PART"
cp "$LOCAL"/gnn/infer_*.json "$PART/"
touch "$PART/done"
