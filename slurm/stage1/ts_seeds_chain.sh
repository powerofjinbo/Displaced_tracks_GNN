#!/bin/bash
# Training-sample study, seeds 1 and 2 (docs/training_sample_study.md section 9). Login node, light: waits and submits.
# The val chain of slurm/stage1/ts_chain.sh (seed 0), unchanged, for every run in RUNS (ARM:SEED):
#   per run runs/stage1_final_learned_ts/l1tl/arm<A>_s<S>, once its val scores exist (eval.done):
#      (a) trackbuild_val with walkmergeS,mwslayer,mwslayermerge
#      (b) GNN scores of 2000 training graphs drawn at random, rng 0 (score_split_sample.sbatch)
#      (c) after (b): learned merging + its val grid (learned_merge.sbatch)
#      (d) the extended grid trackbuild_val_ext, nice 0
#      (e) the e* edge TPR at results/working_points_v1.json
#      (f) oblique crossings on val (hnl.eval.oblique_val), as for seed 0 (section 8)
# The classifiers themselves are trained by slurm/stage1/express_driver.sh with SPEC=slurm/stage1/express_runs_ts_seeds.txt
# (its own STATE directory), the GBDT and union-graph steps are submitted by hand (section 9).
# Differences from ts_chain.sh: RUNS instead of seed 0 only; the CPU exclude list also has the A30 nodes
# (submit20-1..23-1), per the current rule for CPU jobs; step (f).
# CPU jobs: -p submit,submit-gpu --exclude=$EXC, submitted with OMP_NUM_THREADS unset. Every job id goes to this log, to
# $STATE/jobs.txt and to RUN/<step>.submitted; a step is never submitted twice (delete its .submitted file to redo it).
# One instance at a time (heartbeat $STATE/heartbeat, FORCE=1 overrides); stop with touch $STATE/STOP; DRY_RUN=1 prints.
# Usage: setsid nohup bash slurm/stage1/ts_seeds_chain.sh >> /work/submit/jinboz1/hnl/logs/ts_seeds_chain.log 2>&1 < /dev/null &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
S1TAG=final_learned_ts
G=$HNL_DATA/runs/baseline_v1_ts
P=$HNL_DATA/runs/stage1_$S1TAG
WP=$HNL_REPO/results/working_points_v1.json
RUNS=${RUNS:-ii:1 ii:2 iii:1 iii:2}
POLL=${POLL:-300}
EXC=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
CPU=(-p submit,submit-gpu "--exclude=$EXC")
TB_BUILDERS=walkmergeS,mwslayer,mwslayermerge
LMERGE_BUILDERS=cc,mwslayer,mwslayermerge,mws2pass,mws2passmerge,mwslmerge
EXT_BUILDERS=walkmergeX,mwslayermergeX,mws2passmergeX
STATE=$P/_chain_seeds
mkdir -p "$STATE"
log() { echo "$(date -u +%FT%TZ) $*"; }
sub() {   # sbatch --parsable with OMP_NUM_THREADS unset (the job's activate.sh then follows its CPU allocation)
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; return 0; fi
    env -u OMP_NUM_THREADS sbatch --parsable "$@"
}
okid() { [[ "$1" =~ ^[0-9]+$ ]] || [[ "${DRY_RUN:-0}" = 1 && "$1" =~ ^DRY ]]; }
record() {   # MARKER_FILE TEXT: the .submitted marker and the jobs list
    [ "${DRY_RUN:-0}" = 1 ] && return 0
    echo "$2 $(date -u +%FT%TZ)" > "$1"
    echo "$(date -u +%FT%TZ) $2 $(basename "$(dirname "$1")") $(basename "$1" .submitted)" >> "$STATE/jobs.txt"
}
HB=$STATE/heartbeat
if [ -e "$HB" ] && [ "${FORCE:-0}" != 1 ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
    if [ "$age" -lt $(( 2 * POLL + 60 )) ]; then
        echo "$(date -u +%FT%TZ) another ts_seeds_chain is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi
log "ts seeds chain start pid $$ on $(hostname); runs $RUNS; graphs $G; runs $P/l1tl"
while true; do
    [ -e "$STATE/STOP" ] && { log "STOP file found, exiting"; rm -f "$HB"; exit 0; }
    echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    left=0
    for AS in $RUNS; do
        A=${AS%%:*}; S=${AS##*:}
        F=$P/l1tl/arm${A}_s${S}
        R=$G/arm${A}_s${S}
        if ! { [ -e "$F/eval.done" ] && [ -e "$F/gnn_scores/valset.tar" ]; }; then left=1; continue; fi
        if [ ! -e "$F/trackbuild_val.done" ] && [ ! -e "$F/trackbuild_val.submitted" ]; then
            j=$(sub "${CPU[@]}" -J "s1tb_l1tl_${A}_s${S}" slurm/stage1/trackbuild.sbatch "$F" "$TB_BUILDERS" trackbuild_val)
            if okid "$j"; then record "$F/trackbuild_val.submitted" "$j"; log "arm $A s$S trackbuild_val ($TB_BUILDERS) -> $j"
            else log "sbatch failed (arm $A s$S trackbuild_val): $j"; left=1; fi
        fi
        if [ ! -e "$F/trackbuild_val_lmerge.done" ] && [ ! -e "$F/trackbuild_val_lmerge.submitted" ]; then
            j1=$(N_EVENTS=2000 sub "${CPU[@]}" -J "s1_scoretr_l1tl_${A}_s${S}" slurm/stage1/score_split_sample.sbatch "$F" \
                "$R/ltf_union_graphs" train)
            if okid "$j1"; then
                record "$F/trackbuild_val_lmerge.submitted" "$j1"          # never a second train-score job
                j2=$(sub "${CPU[@]}" --dependency=afterok:$j1 -J "s1_lmerge_l1tl_${A}_s${S}" slurm/stage1/learned_merge.sbatch \
                    "$F" "$LMERGE_BUILDERS")
                if okid "$j2"; then
                    record "$F/trackbuild_val_lmerge.submitted" "$j1 $j2"
                    log "arm $A s$S train scores (2000 at random) $j1 -> learned merging + val grid ($LMERGE_BUILDERS) $j2"
                else log "sbatch failed (arm $A s$S learned merging after train scores $j1; submit it by hand): $j2"; fi
            else log "sbatch failed (arm $A s$S train scores): $j1"; left=1; fi
        fi
        if [ ! -e "$F/trackbuild_val_ext.done" ] && [ ! -e "$F/trackbuild_val_ext.submitted" ]; then
            j=$(sub "${CPU[@]}" --nice=0 -J "s1_tbx_l1tl_${A}_s${S}" slurm/stage1/trackbuild.sbatch "$F" "$EXT_BUILDERS" trackbuild_val_ext)
            if okid "$j"; then record "$F/trackbuild_val_ext.submitted" "$j"; log "arm $A s$S trackbuild_val_ext ($EXT_BUILDERS) -> $j"
            else log "sbatch failed (arm $A s$S trackbuild_val_ext): $j"; left=1; fi
        fi
        if [ ! -e "$F/edge_tpr_estar.submitted" ]; then
            j=$(sub "${CPU[@]}" -J "s1_estar_l1tl_${A}_s${S}" slurm/stage1/edge_tpr_estar.sbatch "$F" "$WP")
            if okid "$j"; then record "$F/edge_tpr_estar.submitted" "$j"; log "arm $A s$S e* edge TPR -> $j"
            else log "sbatch failed (arm $A s$S e*): $j"; left=1; fi
        fi
        if [ ! -e "$F/oblique_val.submitted" ]; then
            j=$(sub "${CPU[@]}" -J "s1_oblique_l1tl_${A}_s${S}" slurm/feedback/oblique_val.sbatch "oblique_val_${A}_s${S}_v1" \
                "Canto-L (${A}) s${S}=$F:${A}")
            if okid "$j"; then record "$F/oblique_val.submitted" "$j"; log "arm $A s$S oblique crossings -> $j"
            else log "sbatch failed (arm $A s$S oblique): $j"; left=1; fi
        fi
    done
    if [ "$left" = 0 ]; then log "ts seeds chain: all submitted"; rm -f "$HB"; exit 0; fi
    [ "${DRY_RUN:-0}" = 1 ] && { log "dry run: one pass done"; rm -f "$HB"; exit 0; }
    sleep "$POLL"
done
