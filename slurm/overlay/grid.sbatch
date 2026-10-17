#!/bin/bash
# Addendum H: track building on overlay events (hnl.overlay.tracks grid: the frozen builders and
# hnl.trackml.evaluate.event_counts at WP-A, WP-S and WP-R-P3; on the pilot the dry configurations), one array task per
# block of the set's events, CPU; the task that finds every block written pools them (hnl.overlay.tracks merge) and
# prints GRID_<COND>=<dir>. Order: hnl.overlay.chain preflight (hnl.trackml.freeze verify), then
# slurm/overlay/guard_h.sh --grant (kind eval; dry on the pilot; out OUT) over every id of the set, exported and checked
# by hnl.overlay.tracks (on H-val and H-test also the launch record, HNL_H_LAUNCH). The grids that slurm/overlay/v1.sbatch
# and slurm/overlay/h1.sbatch read, both from `hnl.overlay.chain paths --key grid` (they refuse any other directory):
#   closure full, closure 1gev   V1 and V1-Delta (v1.sbatch)
#   hval 1gev, hval hs           step 5 after `guard_h.sh --launch hval` (h1.sbatch SET=hval)
#   htest full, 1gev, hs         H-T2 (h1.sbatch SET=htest)
# H-val C-full has no grid: hnl.overlay.blind evaluates its sealed directory (blind.sbatch). Production scores and grids:
# hnl.overlay.chain paths (RUNS/model/<set>_<cond>/scores -> RUNS/grid/<set>_<cond>); SCORES and OUT are not accepted
# for production sets. SET=pilot (kind dry, truth-made placeholder scores of `chain.sbatch pilot COND 1` with KIND=dry):
# SCORES and OUT required (scratch), ROOT optional (a scratch table root).
# Blocks: C-full WP-A takes about 255 s per event (native test grid), 6 GB per worker: 16 events per block with 8 workers;
# C-1GeV 30 events per block; C-hs one block.
# Usage: sbatch [--dependency=afterok:<chain job>] --array=0-(N-1) slurm/overlay/grid.sbatch SET COND N
#        sbatch --array=0-1 --export=ALL,SCORES=DIR,OUT=DIR slurm/overlay/grid.sbatch pilot 1gev 2
#SBATCH -J ov_grid
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 24
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
unset HNL_TML_CONFIG HNL_TML_SPLIT
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SET=${1:?SET}; COND=${2:?COND}; NB=${3:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
W=${WORKERS:-8}
echo "git $(git rev-parse HEAD); $(python -I -m hnl.overlay.grant state); host $(hostname); job ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}; $SET $COND block $K of $NB; workers $W"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
if [ "$SET" = pilot ]; then
    KIND=dry
    : "${SCORES:?pilot grids need SCORES=<placeholder scores>}" "${OUT:?pilot grids need OUT=<scratch directory>}"
else
    KIND=eval
    [ -z "${SCORES:-}${OUT:-}${ROOT:-}" ] || { echo "SCORES, OUT and ROOT are for pilot dry runs only"; exit 1; }
    SCORES=$(python -I -m hnl.overlay.chain paths --set "$SET" --cond "$COND" --key scores)
    OUT=$(python -I -m hnl.overlay.chain paths --set "$SET" --cond "$COND" --key grid)
    [ -n "$OUT" ] || { echo "$SET $COND has no grid (hnl.overlay.blind evaluates it)"; exit 1; }
fi
IDS=$(python -I -m hnl.overlay.blind ids --set "$SET")
G=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$COND" --kind "$KIND" --ids "$IDS" --out "$OUT" --grant) || exit 1
export "$G"
echo "grant: $HNL_H_GRANT"
OMP_NUM_THREADS=1 python -I -m hnl.overlay.tracks grid --set "$SET" --cond "$COND" --scores "$SCORES" --out "$OUT" \
    --block "$K" --n-blocks "$NB" --workers "$W" ${ROOT:+--root "$ROOT"}
if [ "$NB" -gt 1 ]; then
    DONE=$(find "$OUT/blocks" -mindepth 2 -maxdepth 2 -name grid.json 2> /dev/null | wc -l)
    echo "blocks written: $DONE of $NB"
    { [ "$DONE" -eq "$NB" ] && mkdir "$OUT/.merging" 2> /dev/null; } || exit 0
    python -I -m hnl.overlay.tracks merge --out "$OUT"
fi
[ -e "$OUT/grid.json" ] || { echo "no merged grid in $OUT"; exit 1; }
echo "GRID_$(echo "$COND" | tr '[:lower:]' '[:upper:]')=$OUT"
