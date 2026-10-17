#!/bin/bash
# Addendum H: claim H1 and the gun read-outs of section 7 (hnl.overlay.h1), CPU; no graph or classifier network (the
# pilot's dry builders have no learned component). Order: hnl.overlay.chain preflight (hnl.trackml.freeze verify), then
# slurm/overlay/guard_h.sh --grant for every condition the job reads (docs/preregistration_pileup.md section 10); the
# grants (out: OUT, on the pilot DRY) are exported together in HNL_H_GRANT (joined by ':'), and hnl.overlay.tracks grid
# and hnl.overlay.h1 run refuse overlay ids without one of the condition. Every run reads C-full, C-1GeV and C-hs
# (hnl.overlay.h1 run refuses fewer before any other check, so a decisive run never goes without C-hs).
#   SET=pilot  code test on truth-made placeholder outputs (kind dry): DRY = the scratch directory whose pilot_full is
#              the sealed placeholder directory of slurm/overlay/blind.sbatch (C-full); ROOT = the pilot tables
#              (default the production overlay root). GRID_1GEV, GRID_HS: placeholder grids (default DRY/grid_<c>, made
#              here when absent from placeholder scores, hnl.overlay.tracks dry-scores / grid, builders without a
#              learned component); then h1 run at WP-R-P3.
#   SET=hval   the val estimate of H1 (H-T0.5 step 5; kind eval, after `guard_h.sh --launch hval`): the manifest sha256
#              of RUNS/sealed/hval_full is read from RESULTS.md (EXPECT, if given, must agree with it); the C-1GeV and
#              C-hs grids are those of slurm/overlay/grid.sbatch, `python -m hnl.overlay.chain paths --set hval --cond C
#              --key grid` (GRID_1GEV / GRID_HS, if given, must name the same directories; each must hold grid.json);
#              HNL_H_LAUNCH = the launch token.
#   SET=htest  claim H1 (H-T2; kind eval, after `guard_h.sh --launch htest`): the C-full, C-1GeV and C-hs grids of
#              `python -m hnl.overlay.chain paths --set htest` (GRID_FULL / GRID_1GEV / GRID_HS, if given, must name
#              them) and HNL_H_LAUNCH.
# On hval and htest: RESTRICTION_RECORD (required) names the committed record under results/pileup_hp/ (written by the
# code of decision D7; on H-val an H-val record h_t05_*, on H-test an H-T1 record that R15 names; the set's launch
# record pins it) from which the run takes its P_P restriction (section 9.3), failed flag gates, navigator-loss share
# (section 7) and descriptive state; RESTRICTION, FLAGS and LOST_SHARE are refused there whenever they are set, even
# empty (they stay free on the pilot). The output path is fixed,
# results/pileup_hp/h1_<set>_<WP>_full-1gev-hs_v1.json (`python -m hnl.overlay.h1 out-path`); an OUT naming any other
# path is refused. The output of a working point is written once (hnl.overlay.blind.write_once: a pending record, the
# output placed, then a consumed marker next to the launch record; all are checked before any input is read); a run
# that stops before its pending record may be repeated, and from the pending record on the working point is consumed.
# WP: working point (default WP-A; the pilot WP-R-P3). OUT: the pilot's output JSON, default
# DRY/h1_pilot_<WP>_full-1gev-hs.json.
# Usage: sbatch --export=ALL,SET=pilot,DRY=DIR[,ROOT=DIR][,GRID_1GEV=DIR,GRID_HS=DIR][,WP=WP-R-P3] slurm/overlay/h1.sbatch
#        sbatch --export=ALL,SET=hval,HNL_H_LAUNCH=TOKEN,RESTRICTION_RECORD=results/pileup_hp/h_t05_..[,WP=WP-S] \
#            slurm/overlay/h1.sbatch
#        sbatch --export=ALL,SET=htest,HNL_H_LAUNCH=TOKEN,RESTRICTION_RECORD=results/pileup_hp/h_t1_..[,WP=..] \
#            slurm/overlay/h1.sbatch
#SBATCH -J ov_h1
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
: "${SET:?}"
IDS=$(python -I -m hnl.overlay.blind ids --set "$SET")
W=${SLURM_CPUS_PER_TASK:-1}
if [ "$SET" = pilot ]; then WPN=${WP:-WP-R-P3}; else WPN=${WP:-WP-A}; fi
echo "git $(git rev-parse HEAD); host $(hostname); job ${SLURM_JOB_ID:-}; set $SET; working point $WPN"
python -I -m hnl.overlay.chain preflight || { echo "refusing: the frozen inputs do not verify (hnl.trackml.freeze verify)"; exit 1; }
GRANTS=""
grant() {          # grant COND KIND OUT: the guard's grant for one condition, added to GRANTS
    local g
    g=$(bash slurm/overlay/guard_h.sh --set "$SET" --cond "$1" --kind "$2" --ids "$IDS" --out "$3" --grant) \
        || { echo "guard_h refused $SET $1 $2"; exit 1; }
    GRANTS=${GRANTS:+$GRANTS:}${g#HNL_H_GRANT=}
}
decisive() {        # the fixed output of hval / htest at the working point, the committed record, no free inputs
    [ -z "${ROOT:-}" ] || { echo "ROOT is for pilot dry runs only"; exit 1; }
    if [ -n "${RESTRICTION+x}${FLAGS+x}${LOST_SHARE+x}" ]; then       # set at all, even empty
        echo "refusing: RESTRICTION, FLAGS and LOST_SHARE come from RESTRICTION_RECORD on $SET (decision D7)"; exit 1
    fi
    : "${RESTRICTION_RECORD:?$SET needs RESTRICTION_RECORD=<committed record under results/pileup_hp/> (decision D7)}"
    FIXED=$(python -I -m hnl.overlay.h1 out-path --set "$SET" --wp "$WPN")
    if [ -n "${OUT:-}" ] && [ "$(realpath -m "$OUT")" != "$(realpath -m "$FIXED")" ]; then
        echo "refusing: $SET at $WPN writes $FIXED only, not $OUT"; exit 1
    fi
    OUT=$FIXED
}
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
case $SET in
    pilot)
        : "${DRY:?pilot runs need DRY=<scratch directory>}"
        OUT=${OUT:-$DRY/h1_pilot_${WPN}_full-1gev-hs.json}
        for c in full 1gev hs; do grant $c dry "$DRY"; done
        export HNL_H_GRANT=$GRANTS
        echo "grants: $HNL_H_GRANT"
        EXPECT=$(python -I -c "from hnl.overlay.blind import manifest_sha256; print(manifest_sha256('$DRY/pilot_full'))")
        G1GEV=${GRID_1GEV:-$DRY/grid_1gev}; GHS=${GRID_HS:-$DRY/grid_hs}
        for c in 1gev hs; do
            if [ $c = 1gev ]; then GD=$G1GEV; GIVEN=${GRID_1GEV:-}; else GD=$GHS; GIVEN=${GRID_HS:-}; fi
            [ -e "$GD/grid.json" ] && continue
            [ -z "$GIVEN" ] || { echo "refusing: GRID_${c^^}=$GIVEN holds no grid.json"; exit 1; }
            [ -e "$DRY/scores_$c" ] || python -I -m hnl.overlay.tracks dry-scores --set pilot --cond $c --out "$DRY/scores_$c" ${ROOT:+--root "$ROOT"}
            OMP_NUM_THREADS=1 python -I -m hnl.overlay.tracks grid --set pilot --cond $c --scores "$DRY/scores_$c" \
                --out "$GD" --workers "$W" ${ROOT:+--root "$ROOT"}
        done
        python -I -m hnl.overlay.h1 run --set pilot --sealed "$DRY/pilot_full" --expect "$EXPECT" \
            --grid-1gev "$G1GEV" --grid-hs "$GHS" --wp "$WPN" ${ROOT:+--root "$ROOT"} --out-json "$OUT" ;;
    hval)
        decisive
        G1GEV=$(production_grid 1gev "${GRID_1GEV:-}")
        GHS=$(production_grid hs "${GRID_HS:-}")
        SEALED=$(python -I -m hnl.overlay.blind sealed-dir --set hval --cond full)
        for c in full 1gev hs; do grant $c eval "$OUT"; done
        export HNL_H_GRANT=$GRANTS
        echo "grants: $HNL_H_GRANT; grids $G1GEV $GHS; record $RESTRICTION_RECORD"
        python -I -m hnl.overlay.h1 run --set hval --sealed "$SEALED" ${EXPECT:+--expect "$EXPECT"} \
            --grid-1gev "$G1GEV" --grid-hs "$GHS" --wp "$WPN" --restriction-record "$RESTRICTION_RECORD" \
            --out-json "$OUT" ;;
    htest)
        decisive
        GFULL=$(production_grid full "${GRID_FULL:-}")
        G1GEV=$(production_grid 1gev "${GRID_1GEV:-}")
        GHS=$(production_grid hs "${GRID_HS:-}")
        for c in full 1gev hs; do grant $c eval "$OUT"; done
        export HNL_H_GRANT=$GRANTS
        echo "grants: $HNL_H_GRANT; grids $GFULL $G1GEV $GHS; record $RESTRICTION_RECORD"
        python -I -m hnl.overlay.h1 run --set htest --grid-full "$GFULL" --grid-1gev "$G1GEV" --grid-hs "$GHS" \
            --wp "$WPN" --restriction-record "$RESTRICTION_RECORD" --out-json "$OUT" ;;
    *)  echo "no H1 job for set $SET"; exit 1 ;;
esac
echo "wrote $OUT"
