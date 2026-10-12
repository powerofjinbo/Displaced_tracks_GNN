#!/bin/bash
# Learned triplet support (CPU; hnl.graphs.learned_triplet_filter, graph booster v3): triplet data from a run's TRAINING
# graphs (ttbar for arm i), the GBDT triplet probability, then the scan of val graph sizes and ceilings over (k, tau)
# against the hand-cut union (rule R7, second part). Usage: sbatch slurm/stage1/learned_triplet_filter.sbatch RUN_DIR
#   (training data RUN_DIR/learned_triplet_filter/train_triplets.npz, shared; model and scan in
#   RUN_DIR/learned_triplet_filter[_TAG]/; N_EVENTS training graphs, default 400; SCAN_EVENTS val graphs, default all;
#   TRAIN_EXTRA, e.g. --weighted)
#SBATCH -J s1_learned_tfilter
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
DATA=$RUN/learned_triplet_filter/train_triplets.npz
OUT=$RUN/learned_triplet_filter${TAG:+_$TAG}
mkdir -p "$OUT" "$(dirname "$DATA")"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN"
# one thread per worker process in data / scan, all cores for the boosting (OpenMP otherwise sees every core of the node)
[ -e "$DATA" ] || OMP_NUM_THREADS=1 python -m hnl.graphs.learned_triplet_filter data --ml-run "$RUN" --split train \
    --out "$DATA" --max-events "${N_EVENTS:-400}"
[ -e "$OUT/model.joblib" ] || OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.graphs.learned_triplet_filter train \
    --data "$DATA" --out "$OUT/model.joblib" ${TRAIN_EXTRA:-}
OMP_NUM_THREADS=1 python -m hnl.graphs.learned_triplet_filter scan --ml-run "$RUN" --split val --model "$OUT/model.joblib" \
    --out "$OUT/${SCAN_NAME:-scan_val}" ${SCAN_EVENTS:+--max-events $SCAN_EVENTS} ${SCAN_EXTRA:-}
touch "$OUT.done"
