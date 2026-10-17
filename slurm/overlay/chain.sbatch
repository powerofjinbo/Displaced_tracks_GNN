#!/bin/bash
# Addendum H (docs/preregistration_pileup.md section 10): the frozen chain's model stages on overlay ids
# (hnl.overlay.chain): union additions and the event store on the GPU (FRNN, torch support, as the native val and test
# events), then the whole-event edge scores of the frozen classifier on this node's CPUs (memory-bounded forward
# hnl.trackml.chunked), for one block (at most 50 events) of a set in one condition. C-hs through
# hnl.overlay.conditions.with_condition, applied to the input tables before graph building. Order: hnl.overlay.chain
# preflight (hnl.trackml.freeze verify; refused if anything differs), then slurm/overlay/guard_h.sh --grant (kind model;
# plumb on the pilot; out OUT), whose grant this job exports and every stage checks (hnl.overlay.grant: kind, condition,
# ids, this job, every write under OUT); the log records it. The grant binds the code an overlay job runs, not the
# commit (decision E1, hnl.overlay.grant.bound_paths): a commit of paper, docs, results, DECISIONS or code outside the
# bound files leaves the job valid; a commit to a bound file (or of the addendum) stops its later stages.
#   SET=hval     before the overlay_hash commit: C-full only, into RUNS/sealed/hval_full (scores/ for
#                slurm/overlay/blind.sbatch, which then evaluates, merges and seals it); after it: C-1GeV and C-hs, with
#                HNL_H_LAUNCH (the token of `guard_h.sh --launch hval`)
#   SET=closure  C-full and C-1GeV (V1, V1-Delta)
#   SET=htest    C-full, C-1GeV and C-hs, with HNL_H_LAUNCH
#   SET=pilot    KIND=plumb (default): the plumbing test of decision D6, one block, OUT required (a scratch directory
#                outside the TrackML data root); the store is kept; the job seals OUT right after the scores are written
#                and checks its structure only (hnl.overlay.chain finish -> CHECK, default <OUT>/../checks/check_COND.json).
#                KIND=dry: truth-made placeholder scores into OUT/scores (no model; code tests of grid.sbatch / h1.sbatch)
# Production outputs: hnl.overlay.chain paths (RUNS/model/<set>_<cond>; H-val C-full: the sealed directory); OUT is not
# accepted for production sets. Score placement (decision E2): production sets run STAGES=additions,store here (the
# default) and the edge scores in slurm/overlay/chain_score.sbatch on the express CPU nodes, S score jobs per block of
# this array (at most 5 events each in C-full, so that each ends within its 1 h limit; `python -m hnl.overlay.chain
# layout --set SET --cond COND` prints N, S and both sbatch lines); a production STAGES that holds score is refused.
# The pilot plumbing run keeps STAGES=additions,store,score (its default). KEEP_STORE=1 keeps the store after scoring
# (production default: removed by the last of the block's score jobs once all its scores are written and checked).
# Usage: sbatch --array=0-(N-1) slurm/overlay/chain.sbatch SET COND N
#        sbatch --export=ALL,OUT=DIR slurm/overlay/chain.sbatch pilot COND 1
#        sbatch --export=ALL,KIND=dry,OUT=DIR slurm/overlay/chain.sbatch pilot COND 1
#SBATCH -J ov_chain
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100|nvidia_rtx6000
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
unset HNL_TML_CONFIG HNL_TML_SPLIT
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SET=${1:?SET}; COND=${2:?COND}; NB=${3:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
KIND=${KIND:-$([ "$SET" = pilot ] && echo plumb || echo model)}
case $SET:$KIND in pilot:plumb|pilot:dry|hval:model|closure:model|htest:model) ;; *) echo "no chain job for $SET kind $KIND"; exit 1 ;; esac
if [ "$SET" = pilot ]; then STAGES=${STAGES:-additions,store,score}; else STAGES=${STAGES:-additions,store}; fi
if [ "$SET" != pilot ] && [[ ",$STAGES," == *,score,* ]]; then
    echo "refusing: decision E2 places the production edge scores on the CPU nodes (slurm/overlay/chain_score.sbatch);"\
         "STAGES=$STAGES on $SET"; exit 1
fi
echo "git $(git rev-parse HEAD); $(python -I -m hnl.overlay.grant state); host $(hostname); job ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}; $SET $COND block $K of $NB; kind $KIND; stages $STAGES; cpus $(nproc); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo no-gpu)"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
if [ "$SET" = pilot ]; then
    : "${OUT:?pilot runs need OUT=<scratch directory>}"
    [ "$NB" = 1 ] || { echo "pilot runs are one block"; exit 1; }
else
    [ -z "${OUT:-}" ] || { echo "OUT is for pilot runs only; production outputs: hnl.overlay.chain paths"; exit 1; }
    OUT=$(python -I -m hnl.overlay.chain paths --set "$SET" --cond "$COND" --key out)
fi
IDS=$(python -I -m hnl.overlay.chain block-ids --set "$SET" --block "$K" --n-blocks "$NB")
G=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$COND" --kind "$KIND" --ids "$IDS" --out "$OUT" --grant) || exit 1
export "$G"
echo "grant: $HNL_H_GRANT"
if [ "$KIND" = dry ]; then
    python -I -m hnl.overlay.chain dry --set pilot --cond "$COND" --out "$OUT" ${ROOT:+--root "$ROOT"}
    echo "placeholder scores (truth-made, no model): $OUT/scores"
    exit 0
fi
[ -z "${ROOT:-}" ] || { echo "ROOT is for dry runs only: the model stages read the production overlay tables"; exit 1; }
ARGS=(--stages "$STAGES")
if [ "$SET" = pilot ]; then ARGS+=(--out "$OUT" --keep-store); elif [ "${KEEP_STORE:-0}" = 1 ]; then ARGS+=(--keep-store); fi
python -I -m hnl.overlay.chain run --set "$SET" --cond "$COND" --block "$K" --n-blocks "$NB" "${ARGS[@]}"
if [ "$SET" = pilot ] && [[ ",$STAGES," == *,score,* ]]; then
    python -I -m hnl.overlay.chain finish --set pilot --cond "$COND" --out "$OUT" \
        --json "${CHECK:-$(dirname "$OUT")/checks/check_$COND.json}"
fi
echo "done: $SET $COND block $K of $NB -> $OUT"
