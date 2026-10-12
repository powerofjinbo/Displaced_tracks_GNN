#!/bin/bash
# Post-freeze track labels of the frozen TrackML configurations (hnl.trackml.labels; DECISIONS 2026-10-11, the
# user's decision): rebuild a test set's tracks from the test chain's stored scores with the frozen configurations
# ($RUN/test/frozen_<COND>.json), so that Canto-L can be recounted under the published references' own evaluation
# code. Nothing is chosen. One array task per block of 10 events; output $RUN/test_post/labels/<SET>_<COND>.
# Usage: RUN_SUFFIX=_x1 TEST_CHAIN=1 sbatch --array=0-29 slurm/trackml/labels_post.sbatch SET COND
#SBATCH -J tml_labels
#SBATCH -p submit
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 06:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
SET=$1
COND=${2:-full}
case $SET in test|t2) [ "${TEST_CHAIN:-0}" = 1 ] || { echo "test sets only with TEST_CHAIN=1"; exit 1; } ;; esac
require "$RUN/test/frozen_$COND.json" "$(scores_dir "$SET" "$COND").done"
python -m hnl.trackml.labels rebuild --events "$SET" --condition "$COND" --scores "$(scores_dir "$SET" "$COND")" \
    --configs "$RUN/test/frozen_$COND.json" --merge-model "$RUN/lmerge/model.joblib" \
    --out "$RUN/test_post/labels/${SET}_$COND" --block "${SLURM_ARRAY_TASK_ID:-0}" --block-size 10 --allow-test
