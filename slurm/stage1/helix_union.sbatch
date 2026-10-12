#!/bin/bash
# HelixFilter decisive test (Stage 1 A, CPU; design synthesis step 5b): a run's metric-learning val graphs united with
# the support-filtered wider-radius candidates of the same embedding (hnl.graphs.helix_union), their ceilings and the
# rescued set, the run's GNN scores on them (CPU inference), the val builder grid (cc, walkthrough + merging,
# mwslayer) and how many rescued particles are DM-matched at the FR* configuration (also on the base graphs' grid if
# RUN/trackbuild_val_mwslayer exists).
# Usage: sbatch slurm/stage1/helix_union.sbatch RUN [ML_RUN] [FR_STAR]      (outputs RUN/helix_union/)
#   BASE_TB_NAME (env): the base graphs' builder grid to compare with (default trackbuild_val_mwslayer)
#   RUN: the run whose GNN scores the union graphs; ML_RUN: the baseline run holding ml_graphs/ and the
#   metric-learning checkpoint (default RUN; for a Stage-1 run, the baseline run of its arm and seed)
#SBATCH -J s1_helix_union
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
ML_RUN=${2:-$1}
FR=${3:-0.031402365961525895}
OUT=$RUN/helix_union
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL" "$OUT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; ml run $ML_RUN"
python -m hnl.graphs.helix_union build --ml-run "$ML_RUN" --split val --out "$LOCAL/union" --report "$OUT/ceiling"
CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$RUN" --stage gnn)
echo "scoring $(ls "$LOCAL/union/valset" | wc -l) union graphs on CPU with $CKPT"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" \
    --input-dir "$LOCAL/union" --out-dir "$LOCAL/gnn" --accelerator cpu
python -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$OUT/gnn_scores"
cp "$LOCAL"/gnn/infer_*.json "$OUT/gnn_scores/"
rm -rf "$LOCAL/union"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/gnn/valset" --out "$OUT/trackbuild_val" \
    --builders cc,walkmergeS,mwslayer,mwslayermerge
BASE_TB=()
B=$RUN/${BASE_TB_NAME:-trackbuild_val_mwslayer}
[ -e "$B/per_particle.parquet" ] && BASE_TB=(--base-trackbuild "$B")
python -m hnl.graphs.helix_union rescued --ceiling "$OUT/ceiling" --trackbuild "$OUT/trackbuild_val" --fr-star "$FR" \
    "${BASE_TB[@]}"
touch "$OUT/done"
