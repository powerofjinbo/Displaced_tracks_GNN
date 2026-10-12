#!/bin/bash
# Mirror / endpoint checks of the truncation probe (supervisor feedback round 1, point 4a; CPU, validation only):
# hnl.eval.probe_outer for one model, modes inner, outer and middle at depths 0, 2, 4 on the fixed 300 HNL val events
# of the original probe. Writes $HNL_DATA/runs/feedback_v1/probe_outer_v1/TAG/{edges.parquet,meta.json} (small);
# per-event graphs only on node-local /tmp. Peak RSS ~4 GB (acorn) to ~8 GB (Canto-L union graphs); 16G lets the job
# fit the memory left on busy nodes (the v1 jobs were submitted with 32G and lowered to 16G while pending).
# Usage: env -u OMP_NUM_THREADS sbatch -J po_TAG slurm/stage1/probe_outer.sbatch TAG RUN GRAPHS_VALSET_TAR
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
TAG=$1; RUN=$2; GRAPHS=$3
BASE=$HNL_DATA/runs/feedback_v1/probe_outer_v1
OUT=$BASE/$TAG
LOCAL=/tmp/$USER/$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
mkdir -p "$LOCAL/w"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID"
echo "tag $TAG run $RUN graphs $GRAPHS"
if [ -e "$OUT/edges.parquet" ]; then echo "exists: $OUT/edges.parquet"; exit 1; fi
python -m hnl.eval.probe_outer --run "$RUN" --graphs "$GRAPHS" --out "$OUT" --events "$BASE/events.json" \
    --work "$LOCAL/w" --modes inner,outer,middle --depths 0,2,4
