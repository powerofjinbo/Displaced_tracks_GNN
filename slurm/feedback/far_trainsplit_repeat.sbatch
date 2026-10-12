#!/bin/bash
# Repeatability of the GPU edge scores behind the identity check of hnl.eval.far_trainsplit (validation events only):
# the stored val graphs of the 49 identity events (acorn (ttbar): RUN/ml_graphs; Canto-L: the learned-triplet union) are
# scored twice with the same checkpoint on an A30, as the val scores were, and both scorings are compared with each
# other and with the stored val scores. Output $HNL_DATA/runs/feedback_v4/far_trainsplit/repeat_s<SEED>.json.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/far_trainsplit_repeat.sbatch [SEED]
#SBATCH -J fts_repeat
#SBATCH -p submit-gpu-express,submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
# nodes without AVX2 (the GTX 1080 Ti nodes): Polars' compat runtime (as slurm/baseline/_common.sh)
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
SEED=${1:-0}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; seed $SEED; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo no-gpu); $(date -u +%FT%TZ)"
eval "$(python -m hnl.eval.far_trainsplit params --seed "$SEED")"
python -m hnl.eval.far_trainsplit repeat --seed "$SEED" --work "$LOCAL" --phase extract
for k in 1 2; do
    python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$ACORN_GNN_CKPT" --input-dir "$LOCAL/acorn_ttbar" \
        --out-dir "$LOCAL/acorn_ttbar_run$k" > /dev/null
    python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CANTO_GNN_CKPT" --input-dir "$LOCAL/canto_l" \
        --out-dir "$LOCAL/canto_l_run$k" > /dev/null
done
python -m hnl.eval.far_trainsplit repeat --seed "$SEED" --work "$LOCAL" --phase compare
echo "done $(date -u +%FT%TZ)"
