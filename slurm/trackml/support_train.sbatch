#!/bin/bash
# S2b (plan section 6.2): learned triplet support retrained on TrackML. Training rows from the gbdt events (the 100
# smallest training ids outside the calibration set): candidates within support.data_rw_factor x r*(largest density),
# the 16 invariant features (log_pt_turn at the field map's B when FIELD_MAP / support.field_map exists), label = both
# legs chain edges, every positive and negatives subsampled to at most support.max_negative_rows; then the unchanged
# GBDT training of hnl.graphs.learned_triplet_filter (300 iterations, lr 0.1, 63 leaves, unweighted; holdout AUC).
# Usage: [SEED=0] [RUN_SUFFIX=_x1] [FIELD_MAP=configs/trackml/field_map_v1.json | none] sbatch slurm/trackml/support_train.sbatch
#SBATCH -J tml_support
#SBATCH -p submit
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
require "$RUN/ml_train.done" "$RUN/graph/radii.json"
FM=${FIELD_MAP:-$(cfg support.field_map)}
if [ "$FM" != none ] && [ ! -e "$FM" ]; then
    echo "field map $FM missing (plan section 6.5 calibration step); FIELD_MAP=none trains with the 2 T feature"; exit 1
fi
FMARG=()
if [ "$FM" != none ]; then FMARG=(--field-map "$FM"); fi
mkdir -p "$RUN/support"
[ -e "$RUN/support/train_triplets.npz" ] || storage_gate 2.5e9      # the training rows (S1: 1.79 GB) + model
[ -e "$RUN/support/train_triplets.npz" ] || python -m hnl.trackml.graphs ltf-data --events gbdt --ml-ckpt "$(best_ckpt ml)" \
    --radii "$RUN/graph/radii.json" --out "$RUN/support/train_triplets.npz" --device cpu \
    "${FMARG[@]}"
OMP_NUM_THREADS=$(nproc) python -m hnl.graphs.learned_triplet_filter train --data "$RUN/support/train_triplets.npz" \
    --out "$RUN/support/model.joblib" --seed "$SEED"
echo "$FM" > "$RUN/support/field_map.txt"
touch "$RUN/support_train.done"
