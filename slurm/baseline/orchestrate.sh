#!/bin/bash
# Submit the baseline pipelines stage by stage within the brief's limits (<= 4 GPU jobs and <= 2 CPU jobs of this user
# queued at any time, counting jobs submitted by hand), resubmitting a stage whose job ended without its marker
# (preempted or failed) at most MAX_ATTEMPTS times. Light: a few squeue / sacct calls every POLL seconds.
#   setsid nohup bash slurm/baseline/orchestrate.sh slurm/baseline/plan_v1.txt >> $HNL_LOGS/orchestrate_v1.log 2>&1 &
# Plan lines: ARM SEED ML_EPOCHS GNN_EPOCHS (order = priority). Stop it with: touch $STATE_DIR/STOP
# A stage script that needs a human decision writes $RUN/HOLD; that pipeline is skipped until the file is removed.
# A stage that gave up (MAX_ATTEMPTS) is re-armed by deleting $STATE_DIR/<key>_<stage>.jobs and $STATE_DIR/<key>.gaveup.
# PAIR_GNN=1: two pipelines that both wait for gnn_train share one GPU job (gnn_train_pair.sbatch).
# DRY_RUN=1: one pass, print what would be submitted, submit nothing.
# Every submitter of this project (both orchestrators, slurm/submit_when_slot.sh) counts the queue and submits
# while holding one flock on a file on /work (NFS 4.1, server-side locks: shared by all login nodes), so two of
# them never take the same free slot; if the lock cannot be opened or taken, nothing is submitted.
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
PLAN=$1
TAG=${TAG:-v1}
POLL=${POLL:-300}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
MAX_GPU=${MAX_GPU:-64}          # limits lifted by the user (2026-10-03); 64 = sanity bound
MAX_CPU=${MAX_CPU:-64}
MIN_CEPH_FREE_GB=${MIN_CEPH_FREE_GB:-40}
PAIR_GNN=${PAIR_GNN:-0}
DRY_RUN=${DRY_RUN:-0}
BASE=$HNL_DATA/runs/baseline_${TAG}
STATE_DIR=$BASE/_orchestrator
SB=$HNL_REPO/slurm/baseline
mkdir -p "$STATE_DIR"
SUBMIT_LOCK=${SUBMIT_LOCK:-/work/submit/jinboz1/hnl/locks/hnl_submit.lock}
exec 9>> "$SUBMIT_LOCK" || { echo "cannot open the submission lock $SUBMIT_LOCK" >&2; exit 1; }
# single instance across login nodes: every pass writes "host pid time" to $STATE_DIR/heartbeat; a new orchestrator
# refuses to start while another one's heartbeat is fresh (younger than 2 POLL + 60 s) unless FORCE=1; a STOP exit
# removes it (DECISIONS.md: two orchestrators on submit00 and submit01 ran together for 30 min on 2026-10-03)
HB=$STATE_DIR/heartbeat
if [ -e "$HB" ] && [ "${FORCE:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
    if [ "$age" -lt $(( 2 * POLL + 60 )) ] && [ "$(cut -d' ' -f1,2 < "$HB")" != "$(hostname) $$" ]; then
        echo "$(date -u +%FT%TZ) another orchestrator is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi
[ "$DRY_RUN" = 1 ] || echo $$ > "$STATE_DIR/pid"
STAGES=(ml_train ml_infer gnn_train gnn_infer eval)
log() { echo "$(date -u +%FT%TZ) $*"; }

# slot limits can be lowered while running, never raised (e.g. to keep one GPU free for a short test job):
#   echo 3 > $STATE_DIR/max_gpu   (delete the file to restore the default); max_cpu likewise
limit() {  # NAME DEFAULT -> the number in $STATE_DIR/NAME if it is below DEFAULT, else DEFAULT
    local v=""
    [ -e "$STATE_DIR/$1" ] && v=$(tr -dc 0-9 < "$STATE_DIR/$1")
    if [ -n "$v" ] && [ "$v" -lt "$2" ]; then echo "$v"; else echo "$2"; fi
}

count_attempts() { local n; n=$( { grep -c submitted "$1" 2>/dev/null || true; } | tail -n 1); echo "${n:-0}"; }

# is a job for pipeline ARM_SEED at stage ST queued? (single-stage name or a GNN pair containing it)
in_flight() {
    local key=$1 st=$2
    grep -qx "bl_${st}_${key}" <<< "$names" && return 0
    [ "$st" = gnn_train ] && grep -qE "^bl_gnn_pair__(.*__)?${key}(__|$)" <<< "$names" && return 0
    return 1
}

submit() {  # NAME SCRIPT ARGS... -> prints job id or nothing
    local name=$1 script=$2
    shift 2
    if [ "$DRY_RUN" = 1 ]; then log "DRY RUN would submit $name: $script $*" >&2; echo "dry"; return; fi
    local out
    out=$(TAG=$TAG sbatch --parsable -J "$name" "$script" "$@" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then echo "$out"; else log "sbatch failed for $name: $out" >&2; fi
}

log "orchestrator start pid $$ plan $PLAN tag $TAG pair_gnn $PAIR_GNN dry_run $DRY_RUN"
while true; do
    if [ -e "$STATE_DIR/STOP" ]; then log "STOP file found, exiting"; rm -f "$HB"; exit 0; fi
    [ "$DRY_RUN" = 1 ] || echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    max_gpu=$(limit max_gpu "$MAX_GPU"); max_cpu=$(limit max_cpu "$MAX_CPU")
    [ "$max_gpu/$max_cpu" = "${prev_lim:-$MAX_GPU/$MAX_CPU}" ] || log "slot limits now gpu $max_gpu cpu $max_cpu"
    prev_lim=$max_gpu/$max_cpu
    flock 9 || { log "submission lock failed"; sleep "$POLL"; continue; }
    names=$(squeue -h -u "$USER" -o "%j" 2>/dev/null) || { log "squeue failed"; flock -u 9; sleep "$POLL"; continue; }
    parts=$(squeue -h -u "$USER" -o "%P" 2>/dev/null) || { log "squeue failed"; flock -u 9; sleep "$POLL"; continue; }
    ngpu=$(awk 'NF && /gpu/' <<< "$parts" | wc -l); ncpu=$(awk 'NF && !/gpu/' <<< "$parts" | wc -l)   # (empty queue: 0)
    # ---- pass 1: state of every pipeline
    A=(); S=(); MLE=(); GNE=(); NEXT=(); READY=()
    all_done=1
    while read -r ARM SEED ml_e gn_e; do
        [ -z "${ARM:-}" ] && continue
        case "$ARM" in \#*) continue;; esac
        run=$BASE/arm${ARM}_s${SEED}
        key=${ARM}_s${SEED}
        # side job (does not block the pipeline): graph-stage ceilings of the metric-learning val graphs (M2)
        side=bl_ceiling_ml_${key}
        if [ -e "$run/ml_infer.done" ] && [ ! -e "$run/ceiling_ml.done" ] && ! grep -qx "$side" <<< "$names" \
            && [ "$ncpu" -lt "$max_cpu" ] && [ "$(count_attempts "$STATE_DIR/${key}_ceiling_ml.jobs")" -lt 2 ]; then
            jid=$(submit "$side" "$SB/ceiling_ml.sbatch" "$ARM" "$SEED")
            if [ -n "$jid" ] && [ "$jid" != dry ]; then
                echo "$jid submitted $(date -u +%FT%TZ)" >> "$STATE_DIR/${key}_ceiling_ml.jobs"
                log "submitted $side -> $jid"; ncpu=$((ncpu + 1)); names=$(printf '%s\n%s' "$names" "$side")
            fi
        fi
        next=""
        for st in "${STAGES[@]}"; do
            if [ ! -e "$run/$st.done" ]; then next=$st; break; fi
        done
        [ -z "$next" ] && continue
        all_done=0
        if [ -e "$run/HOLD" ]; then
            [ -e "$STATE_DIR/${key}.held" ] || { log "HOLD on arm $ARM seed $SEED before $next (decide by hand, then rm $run/HOLD)"; [ "$DRY_RUN" = 1 ] || touch "$STATE_DIR/${key}.held"; }
            continue
        fi
        [ "$DRY_RUN" = 1 ] || rm -f "$STATE_DIR/${key}.held"
        in_flight "$key" "$next" && continue
        sfile=$STATE_DIR/${key}_${next}.jobs
        if [ -s "$sfile" ] && ! tail -n 1 "$sfile" | grep -q ended; then
            jid=$(tail -n 1 "$sfile" | cut -d' ' -f1)
            info=$(sacct -n -X -j "$jid" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1)
            [ "$DRY_RUN" = 1 ] || echo "$jid ended $info" >> "$sfile"
            log "bl_${next}_${key} job $jid ended without marker: $info"
        fi
        if [ "$(count_attempts "$sfile")" -ge "$MAX_ATTEMPTS" ]; then
            [ -e "$STATE_DIR/${key}.gaveup" ] || { log "GIVING UP bl_${next}_${key} after $MAX_ATTEMPTS attempts"; [ "$DRY_RUN" = 1 ] || touch "$STATE_DIR/${key}.gaveup"; }
            continue
        fi
        A+=("$ARM"); S+=("$SEED"); MLE+=("$ml_e"); GNE+=("$gn_e"); NEXT+=("$next"); READY+=(1)
    done < "$PLAN"
    # ---- pass 2: submit in priority order
    for i in "${!A[@]}"; do
        [ "${READY[$i]}" = 1 ] || continue
        ARM=${A[$i]}; SEED=${S[$i]}; next=${NEXT[$i]}; key=${ARM}_s${SEED}; run=$BASE/arm${key}
        attempts=$(count_attempts "$STATE_DIR/${key}_${next}.jobs")
        case $next in
            eval) [ "$ncpu" -ge "$max_cpu" ] && continue ;;
            *) [ "$ngpu" -ge "$max_gpu" ] && continue ;;
        esac
        if [ "$next" = ml_infer ]; then
            free=$(df -BG --output=avail "$HNL_DATA" | tail -n 1 | tr -dc 0-9)
            if [ "$free" -lt "$MIN_CEPH_FREE_GB" ]; then log "ceph free ${free} GB < $MIN_CEPH_FREE_GB: holding bl_ml_infer_${key}"; continue; fi
        fi
        partner=""
        if [ "$next" = gnn_train ] && [ "$PAIR_GNN" = 1 ]; then
            for j in "${!A[@]}"; do
                if [ "$j" -gt "$i" ] && [ "${READY[$j]}" = 1 ] && [ "${NEXT[$j]}" = gnn_train ] && [ "${GNE[$j]}" = "${GNE[$i]}" ]; then
                    partner=$j; break
                fi
            done
        fi
        if [ -n "$partner" ]; then
            key2=${A[$partner]}_s${S[$partner]}
            name=bl_gnn_pair__${key}__${key2}
            jid=$(submit "$name" "$SB/gnn_train_pair.sbatch" "$ARM" "$SEED" "${A[$partner]}" "${S[$partner]}" "${GNE[$i]}")
            [ -z "$jid" ] && continue
            READY[$partner]=0
            if [ "$jid" != dry ]; then
                for k in "$key" "$key2"; do
                    echo "$jid submitted $(date -u +%FT%TZ) pair $name" >> "$STATE_DIR/${k}_gnn_train.jobs"
                done
                log "submitted $name -> $jid"
            fi
        else
            case $next in
                ml_train) args=("$ARM" "$SEED" "${MLE[$i]}") ;;
                gnn_train) args=("$ARM" "$SEED" "${GNE[$i]}") ;;
                *) args=("$ARM" "$SEED") ;;
            esac
            name=bl_${next}_${key}
            jid=$(submit "$name" "$SB/$next.sbatch" "${args[@]}")
            [ -z "$jid" ] && continue
            if [ "$jid" != dry ]; then
                echo "$jid submitted $(date -u +%FT%TZ) ${args[*]}" >> "$STATE_DIR/${key}_${next}.jobs"
                log "submitted $name -> $jid (attempt $((attempts + 1)))"
            fi
        fi
        if [ "$next" = eval ]; then ncpu=$((ncpu + 1)); else ngpu=$((ngpu + 1)); fi
        names=$(printf '%s\n%s' "$names" "$name")
    done
    if [ "$all_done" = 1 ]; then log "all pipelines done"; exit 0; fi
    [ "$DRY_RUN" = 1 ] && exit 0
    flock -u 9
    sleep "$POLL"
done
