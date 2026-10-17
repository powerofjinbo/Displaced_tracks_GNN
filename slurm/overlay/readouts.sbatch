#!/bin/bash
# Addendum H section 7: the read-outs that the H1 code does not compute (hnl.overlay.readouts: fake and duplicate rates
# per condition, fake tracks with a gun hit per LLP, the stage at which each lost gun target is lost, the host prompt
# efficiency, fake rate and TrackML score by collision-twin status), one array task per block of the set's events, CPU;
# the task that finds every block written pools them (hnl.overlay.readouts merge). Order as slurm/overlay/grid.sbatch:
# hnl.overlay.chain preflight, then slurm/overlay/guard_h.sh --grant (kind eval, out OUT; dry on the pilot) over every id
# of the set, exported and checked by hnl.overlay.readouts (on H-val and H-test also the launch record, HNL_H_LAUNCH).
# Production sets read the scores and grids of `hnl.overlay.chain paths` (H-val C-full: the sealed directory) and write
# to RUNS/readouts/<set>_<cond>; SCORES, GRID, SEALED, EXPECT and OUT are for the pilot (placeholders, scratch) only.
# TWINS: the host twin file of hnl.overlay.signatures twins (optional).
# Usage: sbatch --export=ALL,HNL_H_LAUNCH=<token>[,TWINS=FILE] --array=0-(N-1) slurm/overlay/readouts.sbatch SET COND N
#        sbatch --export=ALL,OUT=DIR,SCORES=DIR,GRID=DIR slurm/overlay/readouts.sbatch pilot 1gev 1
#        sbatch --export=ALL,OUT=DIR,SEALED=DIR,EXPECT=SHA slurm/overlay/readouts.sbatch pilot full 1
#SBATCH -J ov_readouts
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
unset HNL_TML_CONFIG HNL_TML_SPLIT
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SET=${1:?SET}; COND=${2:?COND}; NB=${3:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
W=${WORKERS:-4}
echo "git $(git rev-parse HEAD); $(python -I -m hnl.overlay.grant state); host $(hostname); job ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}; $SET $COND block $K of $NB; workers $W"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
ARGS=()
if [ "$SET" = pilot ]; then
    KIND=dry
    : "${OUT:?pilot read-outs need OUT=<scratch directory>}"
    for v in SCORES:scores GRID:grid SEALED:sealed EXPECT:expect ROOT:root; do
        n=${v%%:*}; [ -n "${!n:-}" ] && ARGS+=(--"${v#*:}" "${!n}")
    done
else
    KIND=eval
    [ -z "${SCORES:-}${GRID:-}${SEALED:-}${EXPECT:-}${ROOT:-}${OUT:-}" ] || { echo "SCORES, GRID, SEALED, EXPECT, ROOT and OUT are for the pilot only"; exit 1; }
    OUT=$(python -I -c "import os; from hnl.overlay import blind as B; print(os.path.join(B.RUNS_H, 'readouts', '${SET}_${COND}'))")
fi
IDS=$(python -I -m hnl.overlay.blind ids --set "$SET")
G=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$COND" --kind "$KIND" --ids "$IDS" --out "$OUT" --grant) || exit 1
export "$G"
echo "grant: $HNL_H_GRANT"
OMP_NUM_THREADS=1 python -I -m hnl.overlay.readouts run --set "$SET" --cond "$COND" --out "$OUT" --block "$K" \
    --n-blocks "$NB" --workers "$W" ${TWINS:+--twins "$TWINS"} "${ARGS[@]}"
if [ "$NB" -gt 1 ]; then
    DONE=$(find "$OUT/blocks" -mindepth 2 -maxdepth 2 -name summary.json 2> /dev/null | wc -l)
    echo "blocks written: $DONE of $NB"
    { [ "$DONE" -eq "$NB" ] && mkdir "$OUT/.merging" 2> /dev/null; } || exit 0
    python -I -m hnl.overlay.readouts merge --out "$OUT" ${TWINS:+--twins "$TWINS"}
fi
[ -e "$OUT/summary.json" ] || { echo "no summary in $OUT"; exit 1; }
echo "READOUTS_$(echo "$COND" | tr '[:lower:]' '[:upper:]')=$OUT"
