#!/bin/bash
# Learned HelixFilter (CPU; hnl.graphs.learned_filter): candidate-pair data from a run's TRAINING graphs (ttbar for
# arm i), the GBDT filter, then the scan of val graph sizes and ceilings over tau against the hand-cut union.
# Usage: sbatch slurm/stage1/learned_filter.sbatch RUN_DIR [samepid|chain]   (outputs RUN_DIR/learned_filter[_chain]/)
#SBATCH -J s1_learned_filter
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
LABEL=${2:-samepid}
OUT=$RUN/learned_filter$([ "$LABEL" = samepid ] || echo "_$LABEL")
mkdir -p "$OUT"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN"
python -m hnl.graphs.learned_filter data --ml-run "$RUN" --split train --out "$OUT/train_pairs.npz" --max-events 2000 \
    --label "$LABEL"
python -m hnl.graphs.learned_filter train --data "$OUT/train_pairs.npz" --out "$OUT/model.joblib"
python -m hnl.graphs.learned_filter scan --ml-run "$RUN" --split val --model "$OUT/model.joblib" --out "$OUT/scan_val"
touch "$OUT.done"
