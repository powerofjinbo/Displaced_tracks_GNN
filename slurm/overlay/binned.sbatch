#!/bin/bash
# Addendum H section 7: the efficiencies in bins (hnl.overlay.binned: eps(P_P), eps(P_D) and all gun targets in C-full,
# C-1GeV and C-hs at WP-A, WP-S and WP-R-P3, in the bins of addendum G section 7, signed eta, species and local pileup
# hit density), CPU. Order as slurm/overlay/h1.sbatch: hnl.overlay.chain preflight, then slurm/overlay/guard_h.sh
# --grant for every condition the job reads (kind eval, out OUT; dry on the pilot), exported together in HNL_H_GRANT
# (joined by ':'); on H-val and H-test also the launch record (HNL_H_LAUNCH). Production sets read the grids of
# `hnl.overlay.chain paths` (H-val C-full: its sealed directory) and write to RUNS/binned/<set>.
# Usage: sbatch --export=ALL,HNL_H_LAUNCH=<token> slurm/overlay/binned.sbatch hval|htest
#        sbatch --export=ALL,DRY=<pilot dry directory>,OUT=<scratch> slurm/overlay/binned.sbatch pilot
#SBATCH -J ov_binned
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
unset HNL_TML_CONFIG HNL_TML_SPLIT
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SET=${1:?SET}
echo "git $(git rev-parse HEAD); $(python -I -m hnl.overlay.grant state); host $(hostname); job ${SLURM_JOB_ID:-}; set $SET"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
IDS=$(python -I -m hnl.overlay.blind ids --set "$SET")
ARGS=()
if [ "$SET" = pilot ]; then
    KIND=dry
    : "${DRY:?pilot runs need DRY=<the dry directory of the pilot>}" "${OUT:?pilot runs need OUT=<scratch directory>}"
    EXPECT=$(python -I -c "from hnl.overlay.blind import manifest_sha256; print(manifest_sha256('$DRY/pilot_full'))")
    ARGS=(--sealed "$DRY/pilot_full" --expect "$EXPECT" --grid-1gev "$DRY/grid_1gev" --grid-hs "$DRY/grid_hs")
else
    KIND=eval
    [ -z "${DRY:-}${OUT:-}" ] || { echo "DRY and OUT are for the pilot only"; exit 1; }
    OUT=$(python -I -c "import os; from hnl.overlay import blind as B; print(os.path.join(B.RUNS_H, 'binned', '$SET'))")
fi
GRANTS=""
for c in full 1gev hs; do
    g=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$c" --kind "$KIND" --ids "$IDS" --out "$OUT" --grant) \
        || { echo "guard_h refused $SET $c $KIND"; exit 1; }
    GRANTS=${GRANTS:+$GRANTS:}${g#HNL_H_GRANT=}
done
export HNL_H_GRANT=$GRANTS
echo "grants: $HNL_H_GRANT"
OMP_NUM_THREADS=1 python -I -m hnl.overlay.binned run --set "$SET" --out "$OUT" "${ARGS[@]}"
echo "BINNED=$OUT"
