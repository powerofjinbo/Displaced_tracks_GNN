#!/bin/bash
# Training-sample study chain (docs/training_sample_study.md). Login node, light: waits and submits.
# Canto-L with the arm (i) recipe on the training samples of arm (ii) (HNL) and arm (iii) (ttbar + HNL), seed 0.
#  (1) As soon as an arm's learned-support union graphs exist (runs/baseline_v1_ts/arm<A>_s0/ltf_union_graphs.done), its
#      classifier's first attempt, exactly as the Stage-1 orchestrator submits it (same job name, script, arguments,
#      environment S1TAG=final_learned_ts, TAG=v1_ts so that variant.sbatch reads runs/baseline_v1_ts/arm<A>_s0/
#      ltf_union_graphs, GRAPHS_SUBDIR=ltf_union_graphs, nice 5; recorded in the orchestrator's <run>.jobs file); once every
#      arm has been submitted, the orchestrator itself on slurm/stage1/plan_final_learned_ts.txt (it resubmits a run whose
#      job ends without eval.done), started as canto_l_chain.sh started the arm (i) one but without a builders file: the
#      arm (i) orchestrator's builders step is step (2a) here, so that it gets the CPU flags below. Per-arm starts let one
#      arm train while the other's union graphs still wait (e.g. for storage).
#  (2) Per run (runs/stage1_final_learned_ts/l1tl/arm<A>_s0), once its val scores exist (eval.done), the arm (i) val chain:
#      (a) trackbuild_val with walkmergeS,mwslayer,mwslayermerge              (arm (i): the orchestrator's builders file)
#      (b) GNN scores of 2000 training graphs drawn at random, rng 0         (arm (i): val_chain.sh, the first 2000 sorted)
#      (c) after (b): learned merging + its val grid, val_chain.sh's builders (learned_merge.sbatch)
#      (d) the extended grid trackbuild_val_ext, ext_grid_chain.sh's builders, nice 0 as for the Canto runs
#      (e) the e* edge TPR at results/working_points_v1.json                 (night_waiter.sh)
# CPU jobs: -p submit,submit-gpu --exclude=$EXC, submitted with OMP_NUM_THREADS unset. Every job id goes to this log, to
# $STATE/jobs.txt and to RUN/<step>.submitted; a step is never submitted twice (delete its .submitted file to redo it).
# One instance at a time (heartbeat $STATE/heartbeat, FORCE=1 overrides); stop with touch $STATE/STOP; DRY_RUN=1 prints.
# Usage: setsid nohup bash slurm/stage1/ts_chain.sh >> /work/submit/jinboz1/hnl/logs/ts_chain.log 2>&1 < /dev/null &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
S1TAG=final_learned_ts
G=$HNL_DATA/runs/baseline_v1_ts
P=$HNL_DATA/runs/stage1_$S1TAG
PLAN=slurm/stage1/plan_final_learned_ts.txt
WP=$HNL_REPO/results/working_points_v1.json
ARMS=${ARMS:-ii iii}
POLL=${POLL:-300}
EXC=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
CPU=(-p submit,submit-gpu "--exclude=$EXC")
TB_BUILDERS=walkmergeS,mwslayer,mwslayermerge
LMERGE_BUILDERS=cc,mwslayer,mwslayermerge,mws2pass,mws2passmerge,mwslmerge
EXT_BUILDERS=walkmergeX,mwslayermergeX,mws2passmergeX
STATE=$P/_chain
mkdir -p "$STATE"
log() { echo "$(date -u +%FT%TZ) $*"; }
sub() {   # sbatch --parsable with OMP_NUM_THREADS unset (the job's activate.sh then follows its CPU allocation)
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; return 0; fi
    env -u OMP_NUM_THREADS sbatch --parsable "$@"
}
sub_gpu() {   # arm A's classifier, as orchestrate_s1.sh submits it (same environment: the orchestrator inherits ours)
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "DRY$RANDOM"
        echo "  would: S1TAG=$S1TAG TAG=v1_ts GRAPHS_SUBDIR=ltf_union_graphs sbatch --nice=5 -J s1_l1tl_$1_s0 slurm/stage1/variant.sbatch l1tl $HNL_REPO/configs/stage1/triplet_symregion_trunc.yaml $1 0 30" >&2
        return 0
    fi
    S1TAG=$S1TAG TAG=v1_ts GRAPHS_SUBDIR=ltf_union_graphs sbatch --parsable --nice=5 -J "s1_l1tl_$1_s0" \
        "$HNL_REPO/slurm/stage1/variant.sbatch" l1tl "$HNL_REPO/configs/stage1/triplet_symregion_trunc.yaml" "$1" 0 30
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
        echo "$(date -u +%FT%TZ) another ts_chain is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi
log "ts chain start pid $$ on $(hostname); arms $ARMS; graphs $G; runs $P/l1tl"
warned=""
while true; do
    [ -e "$STATE/STOP" ] && { log "STOP file found, exiting"; rm -f "$HB"; exit 0; }
    echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    # (1) per arm, the classifier's first attempt once its union graphs exist; the orchestrator once all are submitted
    if [ ! -e "$STATE/orchestrator.started" ]; then
        n=0; na=0
        q=$(squeue -h -u "$USER" -o "%j" 2>/dev/null)
        for A in $ARMS; do
            na=$((na + 1))
            F=$P/l1tl/arm${A}_s0
            SF=$P/_orchestrator/l1tl_${A}_s0.jobs
            if [ -e "$G/arm${A}_s0/ltf_union_graphs.done" ]; then
                if [ -s "$SF" ] || [ -e "$F/eval.done" ] || grep -qx "s1_l1tl_${A}_s0" <<< "$q"; then
                    n=$((n + 1))
                else
                    j=$(sub_gpu "$A")
                    if okid "$j"; then
                        if [ "${DRY_RUN:-0}" != 1 ]; then
                            mkdir -p "$P/_orchestrator"
                            echo "$j submitted $(date -u +%FT%TZ) l1tl configs/stage1/triplet_symregion_trunc.yaml $A 0 30" >> "$SF"
                            echo "$(date -u +%FT%TZ) $j arm${A}_s0 classifier (variant.sbatch, attempt 1)" >> "$STATE/jobs.txt"
                        fi
                        log "arm $A union graphs ready -> classifier s1_l1tl_${A}_s0 job $j (A30, nice 5, 30 epochs)"
                        n=$((n + 1))
                    else
                        log "sbatch failed (arm $A classifier): $j"
                    fi
                fi
            elif ! grep -qx "s1_ltf_union_${A}_s0" <<< "$q" && [[ "$warned" != *"u$A "* ]]; then
                log "WARNING arm $A: no ltf_union_graphs.done and no s1_ltf_union_${A}_s0 job queued (failed or cancelled?)"
                warned="$warned u$A "
            fi
        done
        if [ "$n" = "$na" ]; then
            if [ "${DRY_RUN:-0}" = 1 ]; then
                log "would start the orchestrator (S1TAG=$S1TAG TAG=v1_ts GRAPHS_SUBDIR=ltf_union_graphs S1_NICE=5 $PLAN)"
            else
                S1TAG=$S1TAG TAG=v1_ts GRAPHS_SUBDIR=ltf_union_graphs S1_NICE=5 setsid nohup bash slurm/stage1/orchestrate_s1.sh \
                    "$PLAN" >> "$HNL_LOGS/orchestrate_s1_$S1TAG.log" 2>&1 < /dev/null &
                echo "$! $(hostname) $(date -u +%FT%TZ)" > "$STATE/orchestrator.started"
                log "every arm submitted -> Stage-1 orchestrator started for resubmissions (pid $!, log $HNL_LOGS/orchestrate_s1_$S1TAG.log)"
            fi
        fi
    fi
    # (2) the val chain of every run with val scores
    left=0
    for A in $ARMS; do
        F=$P/l1tl/arm${A}_s0
        R=$G/arm${A}_s0
        if ! { [ -e "$F/eval.done" ] && [ -e "$F/gnn_scores/valset.tar" ]; }; then left=1; continue; fi
        if [ ! -e "$F/trackbuild_val.done" ] && [ ! -e "$F/trackbuild_val.submitted" ]; then
            j=$(sub "${CPU[@]}" -J "s1tb_l1tl_${A}_s0" slurm/stage1/trackbuild.sbatch "$F" "$TB_BUILDERS" trackbuild_val)
            if okid "$j"; then record "$F/trackbuild_val.submitted" "$j"; log "arm $A trackbuild_val ($TB_BUILDERS) -> $j"
            else log "sbatch failed (arm $A trackbuild_val): $j"; left=1; fi
        fi
        if [ ! -e "$F/trackbuild_val_lmerge.done" ] && [ ! -e "$F/trackbuild_val_lmerge.submitted" ]; then
            j1=$(N_EVENTS=2000 sub "${CPU[@]}" -J "s1_scoretr_l1tl_${A}_s0" slurm/stage1/score_split_sample.sbatch "$F" \
                "$R/ltf_union_graphs" train)
            if okid "$j1"; then
                record "$F/trackbuild_val_lmerge.submitted" "$j1"          # never a second train-score job
                j2=$(sub "${CPU[@]}" --dependency=afterok:$j1 -J "s1_lmerge_l1tl_${A}_s0" slurm/stage1/learned_merge.sbatch \
                    "$F" "$LMERGE_BUILDERS")
                if okid "$j2"; then
                    record "$F/trackbuild_val_lmerge.submitted" "$j1 $j2"
                    log "arm $A train scores (2000 at random) $j1 -> learned merging + val grid ($LMERGE_BUILDERS) $j2"
                else log "sbatch failed (arm $A learned merging after train scores $j1; submit it by hand): $j2"; fi
            else log "sbatch failed (arm $A train scores): $j1"; left=1; fi
        fi
        if [ ! -e "$F/trackbuild_val_ext.done" ] && [ ! -e "$F/trackbuild_val_ext.submitted" ]; then
            j=$(sub "${CPU[@]}" --nice=0 -J "s1_tbx_l1tl_${A}_s0" slurm/stage1/trackbuild.sbatch "$F" "$EXT_BUILDERS" trackbuild_val_ext)
            if okid "$j"; then record "$F/trackbuild_val_ext.submitted" "$j"; log "arm $A trackbuild_val_ext ($EXT_BUILDERS) -> $j"
            else log "sbatch failed (arm $A trackbuild_val_ext): $j"; left=1; fi
        fi
        if [ ! -e "$F/edge_tpr_estar.submitted" ]; then
            j=$(sub "${CPU[@]}" -J "s1_estar_l1tl_${A}_s0" slurm/stage1/edge_tpr_estar.sbatch "$F" "$WP")
            if okid "$j"; then record "$F/edge_tpr_estar.submitted" "$j"; log "arm $A e* edge TPR -> $j"
            else log "sbatch failed (arm $A e*): $j"; left=1; fi
        fi
    done
    if [ "$left" = 0 ] && [ -e "$STATE/orchestrator.started" ]; then log "ts chain: all submitted"; rm -f "$HB"; exit 0; fi
    [ "${DRY_RUN:-0}" = 1 ] && { log "dry run: one pass done"; rm -f "$HB"; exit 0; }
    sleep "$POLL"
done
