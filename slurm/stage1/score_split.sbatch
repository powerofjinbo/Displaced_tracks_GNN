#!/bin/bash
# GNN edge scores of a Stage-1 run on one split of stored graphs (CPU by default; on an A30 with
# `sbatch -p submit-gpu --constraint=nvidia_a30 --gres=gpu:1 --exclude= --export=ALL,ACCEL=gpu ...`): the first N graphs
# of GRAPHS_DIR/<split>set.tar scored with GNN_RUN's best checkpoint, packed to GNN_RUN/gnn_scores/<split>set.tar
# (e.g. the training graphs for learned track building, or the base / union test graphs for the one-time test).
# MLFORM_LIKE=RUN/ml_graphs/valset.tar: the input graphs are GNN-scored ones (e.g. a baseline's gnn_scores_test/), returned
# to the metric-learning form first (hnl.graphs.helix_union mlform).
# Usage: [N_EVENTS=3000] [ACCEL=cpu|gpu] [MLFORM_LIKE=TAR] sbatch slurm/stage1/score_split.sbatch GNN_RUN GRAPHS_DIR SPLIT
#SBATCH -J s1_score_split
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
GNN_RUN=$1; GRAPHS=$2; SPLIT=${3:-train}
N=${N_EVENTS:-0}                                   # 0: all graphs of the split
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/all" "$LOCAL/in/${SPLIT}set"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; gnn $GNN_RUN; graphs $GRAPHS/${SPLIT}set.tar; N $N; accel ${ACCEL:-cpu}"
tar -xf "$GRAPHS/${SPLIT}set.tar" -C "$LOCAL/all"
files=($(ls "$LOCAL/all" | sort))
[ "$N" -gt 0 ] && files=("${files[@]:0:$N}")
for f in "${files[@]}"; do mv "$LOCAL/all/$f" "$LOCAL/in/${SPLIT}set/"; done
rm -rf "$LOCAL/all"
[ -n "${MLFORM_LIKE:-}" ] && python -m hnl.graphs.helix_union mlform --dir "$LOCAL/in/${SPLIT}set" --like "$MLFORM_LIKE"
CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$GNN_RUN" --stage gnn)
echo "scoring $(ls "$LOCAL/in/${SPLIT}set" | wc -l) graphs with $CKPT"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" \
    --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" --accelerator "${ACCEL:-cpu}"
python -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$GNN_RUN/gnn_scores"
mkdir -p "$GNN_RUN/gnn_scores/${SPLIT}_infer"
cp "$LOCAL"/gnn/infer_*.json "$GNN_RUN/gnn_scores/${SPLIT}_infer/"
touch "$GNN_RUN/score_${SPLIT}.done"
