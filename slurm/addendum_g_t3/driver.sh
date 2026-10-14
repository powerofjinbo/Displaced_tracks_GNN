#!/bin/bash
# Addendum G step T3 driver (login node, light; docs/addendum_g_t3_plan.md). Per seed S, submits each stage of
#   acorn (ttbar+G)   A = runs/baseline_g1/armg_s<S>:     ml_train (chunks) -> ml_infer -> gnn_train (chunks) -> gnn_infer
#                     -> train scores (2000 at random) -> lmerge -> grid blocks -> grid-merge -> prompt guard
#   Canto-L (ttbar+G) G = runs/baseline_g1_ts/armg_s<S>:  (after A/ml_infer.done) GBDT -> union blocks -> gather
#                     C = runs/stage1_g1/l1tl/armg_s<S>:  classifier (chunks; scores val_all, eval.done) -> train scores
#                     -> lmerge -> grid blocks -> grid-merge -> prompt guard
# once its inputs exist. GPU training stages run as resumable 1-h chunks on submit-gpu-express: one chunk queued or
# running plus one successor (afterany); a stage is given up (STATE/<name>.gaveup, logged) after MAX_CHUNKS chunks or
# NOPROG consecutive finished chunks without a larger global step. Other jobs are submitted at most 3 times each
# (resubmitted if they end without their marker). Every submission goes to STATE/<name>.jobs and STATE/jobs.txt.
# GPU-hour stop: no new GPU job once the T3 GPU jobs (sacct, elapsed x 1 GPU) reach GPU_STOP hours (default 230).
# One instance (heartbeat), stop with touch STATE/STOP; DRY_RUN=1 prints one pass.
# Usage: setsid nohup bash slurm/addendum_g_t3/driver.sh >> /work/submit/jinboz1/hnl/logs/addendum_g_t3_driver.log 2>&1 < /dev/null &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
D=$HNL_REPO/slurm/addendum_g_t3
A_=$HNL_DATA/runs/baseline_g1
G_=$HNL_DATA/runs/baseline_g1_ts
C_=$HNL_DATA/runs/stage1_g1/l1tl
STATE=$HNL_DATA/runs/_addendum_g_t3
SEEDS=${SEEDS:-0 1 2}
POLL=${POLL:-120}
MAX_CHUNKS=${MAX_CHUNKS:-80}
NOPROG=${NOPROG:-3}
NBLK=${NBLK:-4}
NU_TRAIN=${NU_TRAIN:-23}     # union blocks (<= 700 events each, 1-h jobs)
NU_VAL=${NU_VAL:-7}
GPU_STOP=${GPU_STOP:-230}
EXC=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
# CPU jobs in 1-h pieces on the express tier first (the CPU partition is saturated; DECISIONS 2026-10-03)
CPU=(-p submit-gpu-express,submit-gpu,submit -t 01:00:00 "--exclude=$EXC")
EXPRESS=(-p submit-gpu-express -t 01:00:00 --gres=gpu:1)
mkdir -p "$STATE"
log() { echo "$(date -u +%FT%TZ) $*"; }

HB=$STATE/heartbeat
if [ -e "$HB" ] && [ "${FORCE:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
    if [ "$age" -lt $(( 2 * POLL + 60 )) ]; then
        log "another driver is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi

sub() {   # NAME SBATCH-ARGS... -> job id (env without OMP_NUM_THREADS, so activate.sh follows the allocation)
    local name=$1; shift
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY"; log "DRY would submit $name: sbatch -J $name $*" >&2; return 0; fi
    local out
    out=$(env -u OMP_NUM_THREADS -u RUN_OVERRIDE -u TAG sbatch --parsable -J "$name" "$@" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
        echo "$out submitted $(date -u +%FT%TZ)" >> "$STATE/$name.jobs"
        echo "$(date -u +%FT%TZ) $out $name" >> "$STATE/jobs.txt"
        echo "$out"
    else
        log "sbatch failed for $name: $out" >&2
    fi
}
queued() { squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | wc -l; }
attempts() { [ -e "$STATE/$1.jobs" ] && wc -l < "$STATE/$1.jobs" || echo 0; }

gpu_hours() {   # elapsed hours of every GPU job of T3 (names g3_ml_*, g3_gnn_*, g3_canto_*) since the driver's first job
    sacct -n -X -u "$USER" -S 2026-10-14 -o JobName%40,ElapsedRaw -P 2>/dev/null \
        | awk -F'|' '$1 ~ /^g3_(ml_|gnn_|canto_|scoretr_|smoke_scan)/ {s += $2} END {printf "%.2f", s / 3600}'
}

max_step() {   # largest global_step over STAGE_DIR/run_*.json
    python -I -c "
import glob, json, sys
s = [json.load(open(p)).get('global_step') or 0 for p in glob.glob(sys.argv[1] + '/run_*.json')]
print(max(s) if s else 0)" "$1"
}

once() {   # NAME DONE_FILE SBATCH-ARGS... : submit unless done, queued, or 3 attempts used
    local name=$1 done=$2; shift 2
    [ -e "$done" ] && return 0
    [ -e "$STATE/$name.gaveup" ] && return 0
    [ "$(queued "$name")" -gt 0 ] && return 0
    if [ "$(attempts "$name")" -ge 3 ]; then
        touch "$STATE/$name.gaveup"; log "GIVING UP $name after 3 attempts (no $done)"; return 0
    fi
    local j; j=$(sub "$name" "$@")
    [ -n "$j" ] && log "submitted $name -> $j"
}

chunked() {   # NAME DONE_FILE STAGE_DIR CONSTRAINT SCRIPT ARGS... : one chunk + one successor while not done
    local name=$1 done=$2 sdir=$3 cons=$4; shift 4
    [ -e "$done" ] && return 0
    [ -e "$STATE/$name.gaveup" ] && return 0
    [ "$GPU_OK" = 1 ] || return 0
    local n; n=$(queued "$name")
    [ "$n" -ge 2 ] && return 0
    local steps last noprog
    steps=$(max_step "$sdir")
    last=$(cat "$STATE/$name.laststep" 2>/dev/null || echo -1)
    noprog=$(cat "$STATE/$name.noprog" 2>/dev/null || echo 0)
    if [ "$n" -lt 2 ] && [ -e "$STATE/$name.jobs" ]; then      # a chunk finished since the last submission
        if [ "$steps" -gt "$last" ]; then noprog=0; else noprog=$((noprog + 1)); fi
        echo "$steps" > "$STATE/$name.laststep"; echo "$noprog" > "$STATE/$name.noprog"
    fi
    if [ "$noprog" -ge "$NOPROG" ] || [ "$(attempts "$name")" -ge "$MAX_CHUNKS" ]; then
        touch "$STATE/$name.gaveup"; log "GIVING UP $name: $(attempts "$name") chunks, $noprog without progress (step $steps)"
        return 0
    fi
    local dep="" j
    [ "$n" -eq 1 ] && dep=$(squeue -h -u "$USER" -n "$name" -o %i | head -n 1)
    j=$(sub "$name" "${EXPRESS[@]}" --constraint="$cons" --signal=B:USR1@300 ${dep:+--dependency=afterany:$dep} "$@")
    [ -n "$j" ] && log "submitted $name chunk -> $j (step $steps${dep:+, after $dep})"
    if [ "$n" -eq 0 ] && [ -n "$j" ]; then
        local j2; j2=$(sub "$name" "${EXPRESS[@]}" --constraint="$cons" --signal=B:USR1@300 --dependency=afterany:$j "$@")
        [ -n "$j2" ] && log "submitted $name successor -> $j2"
    fi
}

grid_stage() {   # MODEL SEED RUN
    local m=$1 s=$2 run=$3 k
    [ -e "$run/grid_val_iso/done" ] && return 0
    [ -e "$run/lmerge.done" ] && [ -e "$run/gnn_scores/valset.tar" ] || return 0
    local all=1
    for ((k = 0; k < NBLK; k++)); do
        local b; b=$(printf '%s/grid_val_iso/blocks/%03d_of_%03d/done' "$run" "$k" "$NBLK")
        [ -e "$b" ] && continue
        all=0
        once "g3_grid_${m}_s${s}_b$k" "$b" "${CPU[@]}" "$D/grid.sbatch" "$m" "$s" "$k" "$NBLK"
    done
    if [ "$all" = 1 ]; then
        once "g3_gridmerge_${m}_s${s}" "$run/grid_val_iso/done" "${CPU[@]}" "$D/gridmerge.sbatch" "$m" "$s"
    fi
}

guard_stage() {   # MODEL SEED RUN: the prompt guard of every family's per-run choice (after the grid)
    [ -e "$3/grid_val_iso/done" ] || return 0
    once "g3_guard_${1}_s${2}" "$3/guard_val.json" "${CPU[@]}" "$D/guard.sbatch" "$1" "$2"
}

log "addendum G T3 driver start pid $$ on $(hostname); seeds $SEEDS"
while true; do
    [ -e "$STATE/STOP" ] && { log "STOP file found, exiting"; rm -f "$HB"; exit 0; }
    [ "${DRY_RUN:-0}" = 1 ] || echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    gh=$(gpu_hours)
    GPU_OK=1
    if awk -v g="$gh" -v s="$GPU_STOP" 'BEGIN {exit !(g >= s)}'; then
        GPU_OK=0
        [ -e "$STATE/gpu_stop.logged" ] || { log "GPU STOP: T3 GPU jobs at $gh h >= $GPU_STOP h; no new GPU job"; touch "$STATE/gpu_stop.logged"; }
    fi
    left=0
    for S in $SEEDS; do
        A=$A_/armg_s$S; G=$G_/armg_s$S; C=$C_/armg_s$S
        mkdir -p "$A" "$G" "$C"
        # ---- acorn (ttbar+G)
        chunked "g3_ml_train_s$S" "$A/ml_train.done" "$A/ml" "nvidia_a30|Tesla_v100" "$D/ml_train.sbatch" "$S"
        if [ -e "$A/ml_train.done" ] && [ "$GPU_OK" = 1 ]; then
            once "g3_ml_infer_s$S" "$A/ml_infer.done" "${EXPRESS[@]}" --constraint="nvidia_a30|Tesla_v100" "$D/ml_infer.sbatch" "$S"
        fi
        if [ -e "$A/ml_infer.done" ]; then
            chunked "g3_gnn_train_s$S" "$A/gnn_train.done" "$A/gnn" "nvidia_a30|Tesla_v100|nvidia_rtx6000|nvidia_gtx1080" \
                "$D/gnn_train.sbatch" "$S"
            # Canto-L pieces on the same embedding
            [ -e "$G/ml" ] || { ln -s "$A/ml" "$G/ml"; ln -s "$A/ml_graphs" "$G/ml_graphs"; log "seed $S: $G/ml, ml_graphs -> $A"; }
            once "g3_gbdt_s$S" "$G/learned_triplet_filter_v2.done" "${CPU[@]}" "$D/gbdt.sbatch" "$S"
        fi
        if [ -e "$A/gnn_train.done" ] && [ "$GPU_OK" = 1 ]; then
            once "g3_gnn_infer_s$S" "$A/gnn_infer.done" "${EXPRESS[@]}" \
                --constraint="nvidia_a30|Tesla_v100|nvidia_rtx6000|nvidia_gtx1080" "$D/gnn_infer.sbatch" "$S"
        fi
        if [ -e "$A/gnn_infer.done" ] && [ "$GPU_OK" = 1 ]; then
            N_EVENTS=2000 ACCEL=gpu once "g3_scoretr_acorn_s$S" "$A/score_train.done" "${EXPRESS[@]}" -c 8 --mem=48G \
                --constraint="nvidia_a30|Tesla_v100|nvidia_rtx6000|nvidia_gtx1080" \
                "$HNL_REPO/slurm/stage1/score_split_sample.sbatch" "$A" "$A/ml_graphs" train
        fi
        [ -e "$A/score_train.done" ] && once "g3_lmerge_acorn_s$S" "$A/lmerge.done" "${CPU[@]}" "$D/lmerge.sbatch" "$A"
        grid_stage acorn "$S" "$A"
        guard_stage acorn "$S" "$A"
        # ---- Canto-L (ttbar+G)
        if [ -e "$G/learned_triplet_filter_v2.done" ]; then
            for SP in train val; do
                [ -e "$G/ltf_union_graphs_$SP.done" ] && continue
                if [ "$SP" = train ]; then NU=$NU_TRAIN; else NU=$NU_VAL; fi
                all=1
                for ((k = 0; k < NU; k++)); do
                    b=$(printf '%s/ltf_union_graphs/blocks/%s_%03d_of_%03d/done' "$G" "$SP" "$k" "$NU")
                    [ -e "$b" ] && continue
                    all=0
                    once "g3_union_${SP}_s${S}_b$k" "$b" "${CPU[@]}" "$D/union_block.sbatch" "$S" "$SP" "$k" "$NU"
                done
                [ "$all" = 1 ] && once "g3_uniongather_${SP}_s$S" "$G/ltf_union_graphs_$SP.done" "${CPU[@]}" \
                    "$D/union_gather.sbatch" "$S" "$SP"
            done
            if [ -e "$G/ltf_union_graphs_train.done" ] && [ -e "$G/ltf_union_graphs_val.done" ] && [ ! -e "$G/ltf_union_graphs.done" ]; then
                touch "$G/ltf_union_graphs.done"; log "seed $S: union graphs complete"
            fi
        fi
        if [ -e "$G/ltf_union_graphs.done" ]; then
            chunked "g3_canto_train_s$S" "$C/eval.done" "$C/gnn" "nvidia_a30|Tesla_v100|nvidia_rtx6000" "$D/canto_train.sbatch" "$S"
        fi
        if [ -e "$C/eval.done" ] && [ "$GPU_OK" = 1 ]; then
            N_EVENTS=2000 ACCEL=gpu once "g3_scoretr_canto_s$S" "$C/score_train.done" "${EXPRESS[@]}" -c 8 --mem=48G \
                --constraint="nvidia_a30|Tesla_v100|nvidia_rtx6000" \
                "$HNL_REPO/slurm/stage1/score_split_sample.sbatch" "$C" "$G/ltf_union_graphs" train
        fi
        [ -e "$C/score_train.done" ] && once "g3_lmerge_canto_s$S" "$C/lmerge.done" "${CPU[@]}" "$D/lmerge.sbatch" "$C"
        grid_stage canto "$S" "$C"
        guard_stage canto "$S" "$C"
        { [ -e "$A/guard_val.json" ] && [ -e "$C/guard_val.json" ]; } || left=1
    done
    if [ "$left" = 0 ]; then log "every grid and prompt guard of every seed is done: the selection is next (hnl.eval.addendum_g_t3 select)"; rm -f "$HB"; exit 0; fi
    [ "${DRY_RUN:-0}" = 1 ] && { log "dry run: one pass done"; rm -f "$HB"; exit 0; }
    sleep "$POLL"
done
