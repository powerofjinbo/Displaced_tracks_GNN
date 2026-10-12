#!/bin/bash
# Extended builder grid (DECISIONS 2026-10-04 "Builder grids extended at the upper edge"): for every Canto run and
# preview, and at lower priority every acorn baseline run, once its val scores exist, the val grid of the extended
# walkthrough + merging, MWS-layer + merging and two-pass + merging families (RUN/trackbuild_val_ext, a superset of
# the original grid points of these families). Login node, light: waits and submits. DRY_RUN=1 prints instead.
# Usage: setsid nohup bash slurm/stage1/ext_grid_chain.sh >> $HNL_LOGS/ext_grid_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
D=$HNL_DATA/runs
BUILDERS=walkmergeX,mwslayermergeX,mws2passmergeX
OUT=trackbuild_val_ext
log() { echo "$(date -u +%FT%TZ) $*"; }
sub() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; else sbatch --parsable "$@"; fi; }
P=stage1_screen_ttbar/symregion_trunc/armi_s0
TODO=""
for s in 0 1 2; do
    TODO="$TODO stage1_final/l1t/armi_s$s:l1t_s$s:0 stage1_final_union/l1tu/armi_s$s:l1tu_s$s:0"
    TODO="$TODO stage1_final_learned/l1tl/armi_s$s:l1tl_s$s:0"
done
TODO="$TODO $P:prevB:5 $P/union_preview:prevH:5 $P/ltf_union_preview:prevL:5"
for a in i ii iii; do for s in 0 1 2; do TODO="$TODO baseline_v1/arm${a}_s$s:acorn${a}_s$s:20"; done; done
log "ext grid chain start pid $$ on $(hostname); builders $BUILDERS"
while [ -n "$TODO" ]; do
    left=""
    for item in $TODO; do
        IFS=: read -r rel tag nice <<< "$item"
        F=$D/$rel
        if [ -e "$F/$OUT.done" ] || [ -e "$F/$OUT.submitted" ]; then log "$tag: already done or submitted"; continue; fi
        ready=0
        case $rel in
            */union_preview|*/ltf_union_preview) [ -e "$F/gnn_scores/valset.tar" ] && ready=1 ;;
            *) [ -e "$F/eval.done" ] && [ -e "$F/gnn_scores/valset.tar" ] && ready=1 ;;
        esac
        if [ $ready = 1 ]; then
            j=$(sub --nice=$nice -J "s1_tbx_$tag" slurm/stage1/trackbuild.sbatch "$F" "$BUILDERS" "$OUT")
            [ "${DRY_RUN:-0}" = 1 ] || echo "$j $(date -u +%FT%TZ)" > "$F/$OUT.submitted"
            log "$tag ready -> $OUT job $j (nice $nice)"
        else
            left="$left $item"
        fi
    done
    TODO=$(echo $left)
    [ -n "$TODO" ] && sleep "${POLL:-300}"
done
log "ext grid chain: all submitted"
