#!/bin/bash
# Score stored graphs with a run's GNN on CPU and run the val builder grid on them (Stage-1 previews, e.g. a screening
# classifier on HelixFilter union graphs): GRAPHS_TAR (an acorn <split>set.tar) -> OUT/gnn_scores/<split>set.tar ->
# OUT/trackbuild_val[_NAME].
# Usage: sbatch slurm/stage1/score_graphs.sbatch GNN_RUN GRAPHS_TAR OUT_DIR [BUILDERS]
#SBATCH -J s1_score_graphs
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
GNN_RUN=$1; TAR=$2; OUT=$3
BUILDERS=${4:-cc,walkmergeS,mwslayer,mwslayermerge,mws2passmerge}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
SPLIT=$(basename "$TAR" .tar)
mkdir -p "$LOCAL/in/$SPLIT" "$OUT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; gnn $GNN_RUN; graphs $TAR"
tar -xf "$TAR" -C "$LOCAL/in/$SPLIT"
CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$GNN_RUN" --stage gnn)
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" \
    --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" --accelerator cpu
python -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$OUT/gnn_scores"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/gnn/$SPLIT" --out "$OUT/trackbuild_val" --builders "$BUILDERS"
touch "$OUT/done"
