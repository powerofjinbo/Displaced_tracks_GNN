#!/bin/bash
# Diagnosis of Canto-L's displaced-target losses on TrackML validation events (hnl.trackml.displaced_diag; post-freeze,
# nothing is chosen, test events are not read): WP-A tracks rebuilt from the stored val scores, graph and edge-score
# stages per target. One array task per block of 10 val events -> $RUN/diag/displaced/.
# Usage: RUN_SUFFIX=_x1 sbatch --array=0-14 slurm/trackml/displaced_diag.sbatch [COND]
#SBATCH -J tml_ddiag
#SBATCH -p submit
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
COND=${1:-full}
python -m hnl.trackml.displaced_diag run --events val --condition "$COND" --scores "$(scores_dir val "$COND")" \
    --configs "$RUN/test/frozen_$COND.json" --merge-model "$RUN/lmerge/model.joblib" \
    --out "$RUN/diag/displaced" --block "${SLURM_ARRAY_TASK_ID:-0}" --block-size 10
