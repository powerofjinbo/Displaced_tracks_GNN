#!/bin/bash
# Addendum H: the blinded evaluation of H-val in C-full (hnl.overlay.blind; docs/preregistration_pileup.md section 10),
# one block of events per array task, CPU. The edge scores must already be in the sealed directory's scores/ (the model
# job writes them there). The task that finds every block done merges the blocks, seals the directory (MANIFEST.sha256;
# its sha256 goes into RESULTS.md as the line 'Addendum H sealed H-val manifest sha256: <64 hex>') and writes the V2
# summary outside it. Order: hnl.overlay.chain preflight (hnl.trackml.freeze verify: `evaluate` runs the learned merge
# model on H-val), then slurm/overlay/guard_h.sh --grant; its grant (kind blind, or dry on the pilot; C-full; out the
# sealed directory) is exported as HNL_H_GRANT, and hnl.overlay.blind evaluate and v2 refuse overlay ids without it.
#   SET=hval:  sealed directory RUNS/sealed/hval_full (hnl.overlay.blind sealed-dir), kind blind; V2 summary to V2
#              (default results/pileup_hp/v2_hval_v1.json)
#   SET=pilot: code test on truth-made placeholder scores (hnl.overlay.blind dry-scores) in a scratch directory DRY,
#              kind dry; V2 required (a scratch path); ROOT: the pilot tables (default the production overlay root; a
#              scratch root for dry runs of the chain)
# Usage: sbatch --array=0-(N-1) [--export=ALL,DRY=DIR,V2=JSON[,ROOT=DIR]] slurm/overlay/blind.sbatch SET N_BLOCKS
#SBATCH -J ov_blind
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
SET=$1; NB=${2:-1}
K=${SLURM_ARRAY_TASK_ID:-0}
if [ "$SET" = pilot ]; then
    SEALED=${DRY:?pilot runs need DRY=<scratch directory>}; KIND=dry; V2=${V2:?pilot runs need V2=<scratch json>}
else
    [ -z "${ROOT:-}" ] || { echo "ROOT is for pilot dry runs only"; exit 1; }
    SEALED=$(python -I -m hnl.overlay.blind sealed-dir --set "$SET" --cond full); KIND=blind
    V2=${V2:-$HNL_REPO/results/pileup_hp/v2_${SET}_v1.json}
fi
IDS=$(python -I -m hnl.overlay.blind ids --set "$SET")
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
G=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond full --kind "$KIND" --ids "$IDS" --out "$SEALED" --grant) || exit 1
export "${G:?}"
echo "git $(git rev-parse HEAD); grant $HNL_H_GRANT; host $(hostname); job ${SLURM_JOB_ID:-}; $SET block $K of $NB; $SEALED"
OMP_NUM_THREADS=1 python -I -m hnl.overlay.blind evaluate --set "$SET" --sealed "$SEALED" --block "$K" \
    --n-blocks "$NB" --workers "${SLURM_CPUS_PER_TASK:-1}" ${ROOT:+--root "$ROOT"}
read -r DONE ALL < <(python -I -c "from hnl.overlay.blind import blocks_done; print(*blocks_done('$SEALED'))")
echo "blocks done: $DONE of $ALL"
{ [ "$DONE" = "$NB" ] && [ "$ALL" = "$NB" ] && mkdir "$SEALED/.merging" 2> /dev/null; } || exit 0
python -I -m hnl.overlay.blind merge --sealed "$SEALED"
python -I -m hnl.overlay.blind seal --sealed "$SEALED"
python -I -m hnl.overlay.blind v2 --sealed "$SEALED" --out "$V2"
