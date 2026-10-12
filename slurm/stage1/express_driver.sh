#!/bin/bash
# Express driver (login node, light): runs Stage-1 trainings (slurm/stage1/variant.sbatch) on the submit-gpu-express
# partition in resumable chunks of at most 1 h. That partition schedules ahead of submit-gpu (priority tier 2000 against
# 1000), so a chunk starts on the next GPU that frees up. Approved by the user on 2026-10-05 (PDT) for the supervisor
# feedback runs and the training-sample study (DECISIONS.md, same date).
#
# For each run in $SPEC (JOBNAME RUN_DIR EXPORTS NAME CONFIG ARM SEED EPOCHS [NEEDS], one per line), while RUN_DIR/eval.done
# is missing:
#  - a pending job of that name in another partition (submitted by a chain or by the Stage-1 orchestrator) is moved in
#    place to submit-gpu-express with a 1 h limit;
#  - one chunk is kept running or pending, plus one successor that depends on it (afterany), so that the GPU a finished
#    chunk releases goes to its successor;
#  - a chunk is variant.sbatch behind a guard that exits at once when eval.done exists. variant.sbatch resumes from the
#    newest checkpoint (preempt / periodic / last); the pre-timeout SIGUSR1 (300 s) writes preempt.ckpt and the chunk
#    exits 0 without eval.done. Lightning resumes a mid-epoch checkpoint by finishing that epoch's step count from a new
#    iterator over the same epoch order (training_epoch_loop.py), so the batches of an interrupted epoch can differ
#    slightly from an uninterrupted run.
#  - NEEDS (optional) is a file that must exist before the first chunk is submitted (e.g. the union graphs' .done).
# A run is given up (STATE/<jobname>.gaveup, logged) after MAX_CHUNKS chunks or NOPROG consecutive chunks without
# progress (no larger global_step in RUN_DIR/gnn/run_<job>.json and no new eval.done).
# Other chains look for a queued job with the run's name before resubmitting, so they leave these runs alone while the
# driver keeps one queued, and take over the read-out once eval.done exists.
# Per-run GPU types: $STATE/constraint.<JOBNAME> overrides the global constraint for that run.
# Per-run priority: lines "JOBNAME NICE" in $STATE/nice (e.g. the training-sample study behind the supervisor's
# runs) are passed as --nice to new chunks.
# The allowed GPU types (CONSTRAINT) can be changed while it runs by writing them to $STATE/constraint; pending
# chunks and adopted jobs are updated to it (types are added only after a smoke test passed on them).
# One instance (heartbeat), stop with touch $STATE/STOP, DRY_RUN=1 prints one pass.
# Usage: setsid nohup bash slurm/stage1/express_driver.sh >> /work/submit/jinboz1/hnl/logs/express_driver.log 2>&1 < /dev/null &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SPEC=${SPEC:-slurm/stage1/express_runs.txt}
STATE=${STATE:-$HNL_DATA/runs/_express_driver}
PART=submit-gpu-express
TLIM=01:00:00
CONSTRAINT=${CONSTRAINT:-nvidia_a30|Tesla_v100}
MAX_CHUNKS=${MAX_CHUNKS:-120}
NOPROG=${NOPROG:-2}
POLL=${POLL:-60}
LOGS=$HNL_LOGS
mkdir -p "$STATE"
log() { echo "$(date -u +%FT%TZ) $*"; }

HB=$STATE/heartbeat
if [ -e "$HB" ] && [ "${FORCE:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
    if [ "$age" -lt $(( 3 * POLL + 60 )) ]; then
        log "another express driver is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi

# largest global_step over a run's training records (0 if none)
steps() {
    python -I -c "
import glob, json, sys
s = [json.load(open(p)).get('global_step') or 0 for p in glob.glob(sys.argv[1] + '/gnn/run_*.json')]
print(max(s) if s else 0)" "$1"
}

# submit one chunk; $1 = dependency job id or empty; prints the job id
chunk() {
    local dep=$1
    local nice cons=$CONSTRAINT
    nice=$(awk -v j="$JOB" '$1 == j {print $2}' "$STATE/nice" 2>/dev/null | tail -n 1)
    [ -s "$STATE/constraint.$JOB" ] && cons=$(tr -d ' \n' < "$STATE/constraint.$JOB")
    local guard="[ -e '$RUN/eval.done' ] && { echo 'eval.done exists: nothing to do'; exit 0; }; "
    guard+="exec bash '$HNL_REPO/slurm/stage1/variant.sbatch' $NAME '$HNL_REPO/$CONFIG' $ARM $SEED $EPOCHS"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "DRY$RANDOM"
        echo "  would: sbatch -J $JOB -p $PART -t $TLIM --constraint=$CONSTRAINT --export=ALL,$EXPORTS ${dep:+--dependency=afterany:$dep} --wrap \"$guard\"" >&2
        return 0
    fi
    env -u OMP_NUM_THREADS -u NUMBA_NUM_THREADS -u LIMIT_TRAIN -u S1TAG -u TAG -u TB_EXTRA -u N_EVENTS -u RUN_OVERRIDE \
        -u GRAPHS_SUBDIR sbatch --parsable -J "$JOB" -p "$PART" -t "$TLIM" --constraint="$cons" --gres=gpu:1 -c 8 \
        --mem=32G --signal=B:USR1@300 -o "$LOGS/%x_%j.out" --export="ALL,$EXPORTS" ${nice:+--nice=$nice} \
        ${dep:+--dependency=afterany:$dep} --wrap "$guard"
}

log "express driver start pid $$ on $(hostname); spec $SPEC; constraint $CONSTRAINT; max chunks $MAX_CHUNKS; no-progress limit $NOPROG"
while true; do
    [ -e "$STATE/STOP" ] && { log "STOP file found, exiting"; rm -f "$HB"; exit 0; }
    [ "${DRY_RUN:-0}" = 1 ] || echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    if [ -s "$STATE/constraint" ]; then
        c=$(tr -d ' \n' < "$STATE/constraint")
        [ "$c" != "$CONSTRAINT" ] && { log "GPU constraint $CONSTRAINT -> $c (from $STATE/constraint)"; CONSTRAINT=$c; }
    fi
    q=$(squeue -h -u "$USER" -o "%i|%j|%T|%P|%E|%f" 2>/dev/null) || { log "squeue failed"; sleep "$POLL"; continue; }
    left=0
    while read -r JOB RUN EXPORTS NAME CONFIG ARM SEED EPOCHS NEEDS; do
        [ -z "${JOB:-}" ] && continue
        case "$JOB" in \#*) continue;; esac
        RUN=$(eval echo "$RUN"); NEEDS=$(eval echo "${NEEDS:-}")
        jf=$STATE/$JOB.jobs                       # one line per chunk: <id> <submitted|adopted> <utc> [dep]
        [ -e "$RUN/eval.done" ] && { [ -e "$STATE/$JOB.done" ] || { touch "$STATE/$JOB.done"; log "$JOB eval.done ($(wc -l < "$jf" 2>/dev/null || echo 0) chunks)"; }; continue; }
        [ -e "$STATE/$JOB.gaveup" ] && continue
        left=1
        if [ -n "$NEEDS" ] && [ ! -e "$NEEDS" ]; then continue; fi
        mine=$(awk -F'|' -v n="$JOB" '$2 == n' <<< "$q")
        # move pending jobs of this name from other partitions (chains, orchestrator) to express, 1 h
        while IFS='|' read -r id _ st part _ feat; do
            [ -z "$id" ] && continue
            if [ "$st" = PENDING ] && [ "$part" != "$PART" ]; then
                if [ "${DRY_RUN:-0}" = 1 ]; then log "would move $JOB $id from $part to $PART"
                elif scontrol update JobId="$id" Partition="$PART" TimeLimit="$TLIM" Features="$CONSTRAINT"; then
                    grep -q "^$id " "$jf" 2>/dev/null || echo "$id adopted $(date -u +%FT%TZ) from_$part" >> "$jf"
                    log "$JOB job $id moved from $part to $PART ($TLIM, $CONSTRAINT)"
                else log "scontrol update of $JOB job $id failed"; fi
            elif [ "$st" = PENDING ] && [ "${DRY_RUN:-0}" != 1 ]; then
                want=$CONSTRAINT; [ -s "$STATE/constraint.$JOB" ] && want=$(tr -d ' \n' < "$STATE/constraint.$JOB")
                [ "$feat" != "$want" ] && scontrol update JobId="$id" Features="$want" && log "$JOB job $id GPU constraint $feat -> $want"
            fi
        done <<< "$mine"
        n=$(awk 'NF' <<< "$mine" | wc -l)
        # account for chunks that have ended: progress or not
        pf=$STATE/$JOB.progress                  # <last checked job id> <global_step> <noprog count>
        { read -r lastid laststep bad < "$pf"; } 2>/dev/null || { lastid=0; laststep=0; bad=0; }
        newest_ended=$(awk '{print $1}' "$jf" 2>/dev/null | while read -r id; do grep -q "^$id|" <<< "$q" || echo "$id"; done | tail -n 1)
        if [ -n "$newest_ended" ] && [ "$newest_ended" != "$lastid" ]; then
            s=$(steps "$RUN")
            info=$(sacct -n -X -j "$newest_ended" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1)
            progress=0
            [ "$s" -gt "$laststep" ] && progress=1
            if [ -e "$RUN/gnn_train.done" ] && [ ! -e "$STATE/$JOB.trained_seen" ]; then
                progress=1; touch "$STATE/$JOB.trained_seen"
            fi
            if [ "$progress" = 1 ]; then bad=0; else bad=$((bad + 1)); fi
            log "$JOB chunk $newest_ended ended ($info): global_step $laststep -> $s; no-progress count $bad"
            echo "$newest_ended $s $bad" > "$pf"
            if [ "$bad" -ge "$NOPROG" ]; then
                log "GIVING UP $JOB: $bad chunks without progress (last $newest_ended); queued chunks left as they are"
                touch "$STATE/$JOB.gaveup"; continue
            fi
        fi
        total=$(awk 'NF' "$jf" 2>/dev/null | wc -l)
        if [ "$n" -ge 2 ]; then continue; fi
        if [ "$total" -ge "$MAX_CHUNKS" ]; then
            [ -e "$STATE/$JOB.gaveup" ] || { log "GIVING UP $JOB after $total chunks"; touch "$STATE/$JOB.gaveup"; }
            continue
        fi
        dep=""
        [ "$n" = 1 ] && dep=$(awk -F'|' 'NF {print $1}' <<< "$mine" | head -n 1)
        out=$(chunk "$dep" 2>&1)
        id=${out%%$'\n'*}
        if [[ "$id" =~ ^[0-9]+$ ]]; then
            echo "$id submitted $(date -u +%FT%TZ) ${dep:+after_$dep}" >> "$jf"
            log "$JOB chunk $id submitted${dep:+ after $dep} ($((total + 1)) of at most $MAX_CHUNKS)"
        elif [[ "$id" =~ ^DRY ]]; then
            log "$JOB dry run: ${out#*$'\n'}"
        else
            log "sbatch failed for $JOB: $out"
        fi
    done < <(grep -v '^#' "$SPEC" | awk 'NF')
    [ "$left" = 0 ] && { log "express driver: every run has eval.done or was given up"; rm -f "$HB"; exit 0; }
    [ "${DRY_RUN:-0}" = 1 ] && { log "dry run: one pass only"; exit 0; }
    sleep "$POLL"
done
