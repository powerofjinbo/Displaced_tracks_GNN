#!/bin/bash
# Addendum H: the score stage of hnl.overlay.chain alone, as a CPU job (whole-event edge scores of the frozen classifier
# with the memory-bounded forward hnl.trackml.chunked), for production blocks whose GPU stages ran in
# slurm/overlay/chain.sbatch with STAGES=additions,store. Decision E2: the production edge scores run here, on the
# express CPU nodes, with 6 CPUs, the thread count of the score time that the cost estimate and the job layout use.
# Layout: the GPU array makes the additions and the store of N blocks of up to 50 events; each block is scored by S jobs
# of this array (S = `python -m hnl.overlay.chain score-parts`, at most hnl.overlay.chain.SCORE_EVENTS_PER_JOB events
# per job, so that a job ends within the 1 h limit: 5 in C-full, 50 in C-1GeV and C-hs), task K = block K/S, part K%S;
# `python -m hnl.overlay.chain layout --set SET --cond COND` prints N, S and both sbatch lines. Same order as
# chain.sbatch: hnl.overlay.chain preflight (freeze verify), then slurm/overlay/guard_h.sh --grant (kind model, the
# part's ids, out the set's model directory), exported and checked by the stage (decision E1: the grant binds the code
# an overlay job runs, not the commit). hnl.trackml.infer imports polars: submit-gpu-express only, without the A30 and
# GTX 1080 nodes. The block's store is removed once all its events are scored and checked, by the last of its score
# jobs, unless KEEP_STORE=1. Production sets only (the pilot plumbing run scores inside chain.sbatch).
# Usage: sbatch --dependency=afterok:<chain.sbatch array> --array=0-(N*S-1) slurm/overlay/chain_score.sbatch SET COND N
#SBATCH -J ov_chain_score
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
unset HNL_TML_CONFIG HNL_TML_SPLIT
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SET=${1:?SET}; COND=${2:?COND}; NB=${3:?N}; K=${SLURM_ARRAY_TASK_ID:-0}
case $SET in hval|closure|htest) ;; *) echo "chain_score.sbatch runs production sets only"; exit 1 ;; esac
NP=$(python -I -m hnl.overlay.chain score-parts --set "$SET" --cond "$COND" --n-blocks "$NB") || exit 1
if [ "$K" -ge $((NB * NP)) ]; then
    echo "refusing: task $K, but $SET $COND in $NB blocks has $((NB * NP)) score jobs ($NP per block)"; exit 1
fi
BLK=$((K / NP)); PART=$((K % NP))
echo "git $(git rev-parse HEAD); $(python -I -m hnl.overlay.grant state); host $(hostname); job ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}; $SET $COND block $BLK of $NB, part $PART of $NP; cpus $(nproc)"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
OUT=$(python -I -m hnl.overlay.chain paths --set "$SET" --cond "$COND" --key out)
IDS=$(python -I -m hnl.overlay.chain score-ids --set "$SET" --cond "$COND" --block "$BLK" --n-blocks "$NB" --part "$PART")
if [ -z "$IDS" ]; then echo "block $BLK part $PART holds no event (a short last block): nothing to score"; exit 0; fi
G=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$COND" --kind model --ids "$IDS" --out "$OUT" --grant) || exit 1
export "$G"
echo "grant: $HNL_H_GRANT"
ARGS=(--stages score --part "$PART")
if [ "${KEEP_STORE:-0}" = 1 ]; then ARGS+=(--keep-store); fi
python -I -m hnl.overlay.chain run --set "$SET" --cond "$COND" --block "$BLK" --n-blocks "$NB" "${ARGS[@]}"
echo "done: $SET $COND block $BLK of $NB, part $PART of $NP scored -> $OUT/scores"
