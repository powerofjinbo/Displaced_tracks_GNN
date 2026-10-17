#!/bin/bash
# Addendum H: gates V1 and V1-Delta on the closure set (hnl.overlay.v1 run; docs/preregistration_pileup.md section
# 9.2), CPU; no network. Order: hnl.overlay.chain preflight (hnl.trackml.freeze verify), then slurm/overlay/guard_h.sh
# --grant (kind eval, out OUT) for the closure set (k = 1) in C-full and C-1GeV; the grants are exported together in
# HNL_H_GRANT (joined by ':'), and hnl.overlay.v1 run refuses the closure ids without one of each condition. The grids
# are those of slurm/overlay/grid.sbatch, `python -m hnl.overlay.chain paths --set closure --cond C --key grid`
# (GRID_FULL / GRID_1GEV, if given, must name the same directories; each must hold grid.json); hnl.overlay.v1 run checks
# them with hnl.overlay.tracks.grid_provenance before anything is read. The closure tables are read from the production
# overlay root only. GUN_CELLS: optional parquet of the H-val gun's P_P targets (r_first, d0) for the gun-weighted V1.
# OUT: output JSON, default results/pileup_hp/v1_closure_v1.json.
# Usage: sbatch [--export=ALL[,GUN_CELLS=PQ][,OUT=JSON]] slurm/overlay/v1.sbatch
#SBATCH -J ov_v1
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SET=closure
IDS=$(python -I -m hnl.overlay.blind ids --set "$SET")
ROOT=$(python -I -c "from hnl.overlay.blind import DERIVED_OVERLAY; print(DERIVED_OVERLAY)")
OUT=${OUT:-$HNL_REPO/results/pileup_hp/v1_closure_v1.json}
echo "git $(git rev-parse HEAD); host $(hostname); job ${SLURM_JOB_ID:-}; set $SET; tables $ROOT"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
production_grid() {  # production_grid COND [GIVEN]: the grid job's directory of the condition (stdout); refuses others
    local d
    d=$(python -I -m hnl.overlay.chain paths --set "$SET" --cond "$1" --key grid) && [ -n "$d" ] \
        || { echo "refusing: no grid directory of $SET in $1 (hnl.overlay.chain paths)" >&2; return 1; }
    if [ -n "${2:-}" ] && [ "$(realpath -m "$2")" != "$(realpath -m "$d")" ]; then
        echo "refusing: $SET in $1 reads the grid of slurm/overlay/grid.sbatch, $d, not $2" >&2; return 1
    fi
    [ -e "$d/grid.json" ] || { echo "refusing: no grid in $1 at $d (slurm/overlay/grid.sbatch $SET $1 first)" >&2; return 1; }
    echo "$d"
}
GFULL=$(production_grid full "${GRID_FULL:-}")
G1GEV=$(production_grid 1gev "${GRID_1GEV:-}")
GRANTS=""
for c in full 1gev; do
    g=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$c" --kind eval --ids "$IDS" --out "$OUT" --grant) \
        || { echo "guard_h refused $SET $c eval"; exit 1; }
    GRANTS=${GRANTS:+$GRANTS:}${g#HNL_H_GRANT=}
done
export HNL_H_GRANT=$GRANTS
echo "grants: $HNL_H_GRANT; grids $GFULL $G1GEV"
python -I -m hnl.overlay.v1 run --root "$ROOT" --closure-full "$GFULL" --closure-1gev "$G1GEV" \
    ${GUN_CELLS:+--gun-cells "$GUN_CELLS"} --out-json "$OUT"
echo "wrote $OUT"
