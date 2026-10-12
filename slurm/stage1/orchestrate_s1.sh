#!/bin/bash
# Stage-1 variant runs (slurm/stage1/variant.sbatch) within the brief's limits: <= MAX_GPU GPU jobs of this user
# queued at any time (all jobs counted), resubmitting runs whose job ended without eval.done (preempted / failed) at
# most MAX_ATTEMPTS times. Plan lines: NAME CONFIG ARM SEED EPOCHS (order = priority). Stop: touch $STATE_DIR/STOP
# S1_NICE: Slurm nice value of this orchestrator's training jobs (> 0 lets the Stage-0 baselines go first in the queue).
# S1_CONSTRAINT: GPU type constraint of the training jobs (default: the job script's nvidia_a30); exploratory screening
# only may use e.g. "nvidia_a30|nvidia_rtx6000" (compared numbers stay on A30, DECISIONS.md).
# After a run's eval.done, if $STATE_DIR/builders holds a builder list (e.g. "cc,walkmerge"), the val track-building
# step slurm/stage1/trackbuild.sbatch RUN BUILDERS trackbuild_val follows as a CPU job (<= MAX_CPU CPU jobs, 2 tries).
#   setsid nohup bash slurm/stage1/orchestrate_s1.sh slurm/stage1/plan_X.txt >> $HNL_LOGS/orchestrate_s1.log 2>&1 &
# Every submitter of this project (both orchestrators, slurm/submit_when_slot.sh) counts the queue and submits
# while holding one flock on a file on /work (NFS 4.1, server-side locks: shared by all login nodes), so two of
# them never take the same free slot; if the lock cannot be opened or taken, nothing is submitted.
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
PLAN=$1
S1TAG=${S1TAG:-v1}
POLL=${POLL:-300}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
MAX_GPU=${MAX_GPU:-64}          # limits lifted by the user (2026-10-03); 64 = sanity bound
MAX_CPU=${MAX_CPU:-64}
BASE=$HNL_DATA/runs/stage1_${S1TAG}
STATE_DIR=$BASE/_orchestrator
mkdir -p "$STATE_DIR"
SUBMIT_LOCK=${SUBMIT_LOCK:-/work/submit/jinboz1/hnl/locks/hnl_submit.lock}
exec 9>> "$SUBMIT_LOCK" || { echo "cannot open the submission lock $SUBMIT_LOCK" >&2; exit 1; }
# single instance across login nodes: every pass writes "host pid time" to $STATE_DIR/heartbeat; a new orchestrator
# refuses to start while another one's heartbeat is fresh (younger than 2 POLL + 60 s) unless FORCE=1; a STOP exit
# removes it (DECISIONS.md: two orchestrators on submit00 and submit01 ran together for 30 min on 2026-10-03)
HB=$STATE_DIR/heartbeat
if [ -e "$HB" ] && [ "${FORCE:-0}" != 1 ] && true; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
    if [ "$age" -lt $(( 2 * POLL + 60 )) ] && [ "$(cut -d' ' -f1,2 < "$HB")" != "$(hostname) $$" ]; then
        echo "$(date -u +%FT%TZ) another orchestrator is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi
echo $$ > "$STATE_DIR/pid"
log() { echo "$(date -u +%FT%TZ) $*"; }
# slot limits can be lowered while running, never raised (e.g. to keep one GPU free for a short test job):
#   echo 3 > $STATE_DIR/max_gpu   (delete the file to restore the default); same file name as the baseline orchestrator's, in this STATE_DIR
#   max_gpu bounds ALL GPU jobs of the user; max_own_gpu bounds this orchestrator's own GPU jobs (names s1_*), e.g.
#   echo 1 > $STATE_DIR/max_own_gpu keeps Stage 1 to one A30 while the Stage-0 baselines still run
limit() {  # NAME DEFAULT -> the number in $STATE_DIR/NAME if it is below DEFAULT, else DEFAULT
    local v=""
    [ -e "$STATE_DIR/$1" ] && v=$(tr -dc 0-9 < "$STATE_DIR/$1")
    if [ -n "$v" ] && [ "$v" -lt "$2" ]; then echo "$v"; else echo "$2"; fi
}
log "stage-1 orchestrator start pid $$ plan $PLAN"
while true; do
    [ -e "$STATE_DIR/STOP" ] && { log "STOP file found, exiting"; rm -f "$HB"; exit 0; }
    echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    max_gpu=$(limit max_gpu "$MAX_GPU"); max_cpu=$(limit max_cpu "$MAX_CPU"); max_own=$(limit max_own_gpu "$MAX_GPU")
    builders=""
    [ -e "$STATE_DIR/builders" ] && builders=$(tr -d ' \n' < "$STATE_DIR/builders")
    flock 9 || { log "submission lock failed"; sleep "$POLL"; continue; }
    q=$(squeue -h -u "$USER" -o "%j %P" 2>/dev/null) || { log "squeue failed"; flock -u 9; sleep "$POLL"; continue; }
    names=$(awk 'NF {print $1}' <<< "$q")
    ngpu=$(awk 'NF && $2 ~ /gpu/' <<< "$q" | wc -l); ncpu=$(awk 'NF && $2 !~ /gpu/' <<< "$q" | wc -l)
    nown=$(awk 'NF && $1 ~ /^s1_/ && $2 ~ /gpu/' <<< "$q" | wc -l)      # this orchestrator's own GPU jobs
    all_done=1
    while read -r NAME CONFIG ARM SEED EPOCHS; do
        [ -z "${NAME:-}" ] && continue
        case "$NAME" in \#*) continue;; esac
        run=$BASE/$NAME/arm${ARM}_s${SEED}
        if [ -e "$run/eval.done" ]; then
            { [ -z "$builders" ] || [ -e "$run/trackbuild_val.done" ]; } && continue
            all_done=0
            tname=s1tb_${NAME}_${ARM}_s${SEED}
            grep -qx "$tname" <<< "$names" && continue
            tfile=$STATE_DIR/${NAME}_${ARM}_s${SEED}_tb.jobs
            tries=$( { grep -c submitted "$tfile" 2>/dev/null || true; } | tail -n 1)
            if [ "${tries:-0}" -ge 2 ]; then
                [ -e "$STATE_DIR/$tname.gaveup" ] || { log "GIVING UP $tname after $tries attempts"; touch "$STATE_DIR/$tname.gaveup"; }
                continue
            fi
            [ "$ncpu" -ge "$max_cpu" ] && continue
            out=$(sbatch --parsable -J "$tname" "$HNL_REPO/slurm/stage1/trackbuild.sbatch" "$run" "$builders" trackbuild_val 2>&1)
            if [[ "$out" =~ ^[0-9]+$ ]]; then
                echo "$out submitted $(date -u +%FT%TZ) $builders" >> "$tfile"
                log "submitted $tname -> $out ($builders)"
                ncpu=$((ncpu + 1)); names=$(printf '%s\n%s' "$names" "$tname")
            else
                log "sbatch failed for $tname: $out"
            fi
            continue
        fi
        all_done=0
        jname=s1_${NAME}_${ARM}_s${SEED}
        grep -qx "$jname" <<< "$names" && continue
        sfile=$STATE_DIR/${NAME}_${ARM}_s${SEED}.jobs
        if [ -s "$sfile" ] && ! tail -n 1 "$sfile" | grep -q ended; then
            jid=$(tail -n 1 "$sfile" | cut -d' ' -f1)
            info=$(sacct -n -X -j "$jid" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1)
            echo "$jid ended $info" >> "$sfile"
            log "$jname job $jid ended without eval.done: $info"
        fi
        attempts=$( { grep -c submitted "$sfile" 2>/dev/null || true; } | tail -n 1); attempts=${attempts:-0}
        if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
            [ -e "$STATE_DIR/$jname.gaveup" ] || { log "GIVING UP $jname after $attempts attempts"; touch "$STATE_DIR/$jname.gaveup"; }
            continue
        fi
        [ "$ngpu" -ge "$max_gpu" ] && continue
        [ "$nown" -ge "$max_own" ] && continue
        out=$(S1TAG=$S1TAG sbatch --parsable --nice="${S1_NICE:-0}" ${S1_CONSTRAINT:+--constraint=$S1_CONSTRAINT} -J "$jname" "$HNL_REPO/slurm/stage1/variant.sbatch" "$NAME" "$HNL_REPO/$CONFIG" "$ARM" "$SEED" "$EPOCHS" 2>&1)
        if [[ "$out" =~ ^[0-9]+$ ]]; then
            echo "$out submitted $(date -u +%FT%TZ) $NAME $CONFIG $ARM $SEED $EPOCHS" >> "$sfile"
            log "submitted $jname -> $out (attempt $((attempts + 1)))"
            ngpu=$((ngpu + 1)); nown=$((nown + 1)); names=$(printf '%s\n%s' "$names" "$jname")
        else
            log "sbatch failed for $jname: $out"
        fi
    done < "$PLAN"
    [ "$all_done" = 1 ] && { log "all runs done"; exit 0; }
    flock -u 9
    sleep "$POLL"
done
