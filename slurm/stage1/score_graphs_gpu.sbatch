#!/bin/bash
# slurm/stage1/score_graphs.sbatch with the scoring on one GPU (acorn_runner infer, as variant.sbatch scores the val
# graphs) instead of the CPU, for previews whose CPU scoring is too slow on a busy node. GRAPHS_TAR (an acorn
# <split>set.tar) -> OUT/gnn_scores/<split>set.tar -> OUT/trackbuild_val[ with BUILDERS].
# Usage: sbatch slurm/stage1/score_graphs_gpu.sbatch GNN_RUN GRAPHS_TAR OUT_DIR [BUILDERS]
#SBATCH -J s1_score_graphs_gpu
#SBATCH -p submit-gpu-express
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
GNN_RUN=$1; TAR=$2; OUT=$3
BUILDERS=${4:-mwslayermerge}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
SPLIT=$(basename "$TAR" .tar)
mkdir -p "$LOCAL/in/$SPLIT" "$OUT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1); gnn $GNN_RUN; graphs $TAR"
tar -xf "$TAR" -C "$LOCAL/in/$SPLIT"
CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$GNN_RUN" --stage gnn)
python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn"
python -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$OUT/gnn_scores"
cp "$LOCAL"/gnn/infer_*.json "$OUT/gnn_scores/" 2>/dev/null || true
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.eval.trackbuild_val --graphs "$LOCAL/gnn/$SPLIT" --out "$OUT/trackbuild_val" --builders "$BUILDERS"
touch "$OUT/done"
