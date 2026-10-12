#!/bin/bash
# Capacity-matched control chain (supervisor feedback round 1, point 4b; val only). Login node, light: waits, submits.
# Runs (S1TAG=capacity; trained on runs/baseline_v1/armi_s0/ml_graphs, the base graphs of Canto-B):
#   ignn147         slurm/stage1/plan_capacity.txt         full budget: 30 epochs x all 7999 arm (i) training graphs
#   ignn147_screen  slurm/stage1/plan_capacity_screen.txt  screening budget: 15 epochs x the first 4000 (LIMIT_TRAIN)
# 1. Training, val scoring and the CC evaluation: slurm/stage1/variant.sbatch on one GPU (the GPU that ran it is in
#    RUN/gnn/run_<job>.json "gpu"). GPU types: the A30 (every earlier run) always; the V100 and the RTX A6000 once their
#    smoke test (slurm/stage1/smoke_variants.sbatch, jobs SMOKE_V100 / SMOKE_A6000) has trained, scored and evaluated
#    the capacity IGNN ("SMOKE OK ignn147" in its log): a pending training job is then widened in place (scontrol
#    update Features), never narrowed; a failed or missing smoke test keeps that type out. GPU_CONSTRAINT, if set,
#    replaces the gate; GPU_CONSTRAINT_FILE (e.g. the express driver's constraint file) does the same, re-read every
#    pass. A job that ends without eval.done is resubmitted (variant.sbatch resumes from the newest checkpoint), at
#    most MAX_ATTEMPTS submissions per run.
# 2. Once RUN/eval.done exists, the CPU read-out as for the other Stage-1 runs:
#    dD   hnl.eval.edge_tpr on the val scores: screening run at the ladder rungs' target edge FPR 0.0122
#         (RUN/edge_tpr_estar.*); full run at 0.0122 (RUN/edge_tpr_e0122.*, the target of acorn arm (i)) and then at
#         e* of results/working_points_v1.json (RUN/edge_tpr_estar.*, as Canto-B; the newest file, read by default);
#    dT   slurm/stage1/truncation_probe.sbatch (300 HNL val events, depths 0 / 2 / 4, cut 0.35);
#    grid slurm/stage1/trackbuild.sbatch RUN walkmergeS,mwslayer,mwslayermerge trackbuild_val (the ladder's grid);
#         full run also walkmergeX,mwslayermergeX,mws2passmergeX -> RUN/trackbuild_val_ext (as acorn and Canto-B).
# 3. After a run's read-out jobs (afterany), the read-out next to the references (CPU job; "--report KIND" below):
#    screen: hnl.eval.ladder_report with the ladder rungs at the ladder's fake-rate target 0.031402, dD target 0.0122
#            (as runs/stage1_screen_ttbar/_reports/ladder_fr0314, canto_facts 6.2), and hnl.eval.ranking at FR* with the
#            one-standard-error tie rule (results/ranking/spec_val_capacity_screen_v1.json);
#    full:   hnl.eval.ladder_report with acorn arm (i) s0-s2 and Canto-B s0-s2 at FR* (dD at e* and at 0.0122), and
#            hnl.eval.ranking at FR* with the tie rule (results/ranking/spec_val_capacity_v1.json).
#    Ladder reports -> runs/stage1_capacity/_reports/; rankings -> results/ranking/ranking_val_capacity*_v1.*
# Single instance (heartbeat in $STATE); stop: touch $STATE/STOP. DRY_RUN=1 prints instead of submitting.
# Usage: setsid nohup bash slurm/stage1/capacity_chain.sh >> $HNL_LOGS/capacity_chain.log 2>&1 < /dev/null &
#        (SMOKE_V100, SMOKE_A6000, GPU_CONSTRAINT, TL_FULL, TL_SCREEN, MAX_ATTEMPTS, POLL from the environment)
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO" || exit 1
D=$HNL_DATA/runs
BASE=${BASE_OVERRIDE:-$D/stage1_capacity}   # BASE_OVERRIDE: tests only
STATE=${STATE_OVERRIDE:-$BASE/_chain}       # STATE_OVERRIDE: e.g. a scratch directory for DRY_RUN=1
REP=$BASE/_reports
WP=$HNL_REPO/results/working_points_v1.json
LOGS=${LOGS_OVERRIDE:-/work/submit/jinboz1/hnl/logs}   # LOGS_OVERRIDE: tests only
CPU_EXCLUDE=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73

# ---------------------------------------------------------------------------------------- report mode (in a CPU job)
if [ "${1:-}" = "--report" ]; then
    set -e
    echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm configs | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}; report $2"
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    FR=$(python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['FR_star'])" "$WP")
    mkdir -p "$REP"
    case $2 in
        screen)
            S=$D/stage1_screen_ttbar
            python -m hnl.eval.ladder_report "$BASE/ignn147_screen/armi_s0" $S/ignn/armi_s0 $S/ignn_trunc/armi_s0 \
                $S/l1/armi_s0 $S/symregion_trunc/armi_s0 $S/l3/armi_s0 $S/triplet_trunc/armi_s0 $S/sym/armi_s0 \
                $S/sym_trunc/armi_s0 $S/rz/armi_s0 $S/blind/armi_s0 \
                --fr-star 0.031402 --e-star 0.0122 --names trackbuild_val --out "$REP/ladder_screen_fr0314"
            python -m hnl.eval.ranking --spec results/ranking/spec_val_capacity_screen_v1.json --fr "$FR" --tie-se \
                --out results/ranking --name ranking_val_capacity_screen_v1 --fig-dir capacity
            ;;
        full)
            RUNS="$BASE/ignn147/armi_s0"
            for s in 0 1 2; do RUNS="$RUNS $D/baseline_v1/armi_s$s $D/stage1_final/l1t/armi_s$s"; done
            python -m hnl.eval.ladder_report $RUNS --wp "$WP" --names trackbuild_val,trackbuild_val_ext \
                --out "$REP/ladder_full_wp"
            python -m hnl.eval.ladder_report $RUNS --wp "$WP" --e-star 0.0122 --names trackbuild_val,trackbuild_val_ext \
                --out "$REP/ladder_full_wp_e0122"
            python -m hnl.eval.ranking --spec results/ranking/spec_val_capacity_v1.json --fr "$FR" --tie-se \
                --out results/ranking --name ranking_val_capacity_v1 --fig-dir capacity
            ;;
        *) echo "unknown report $2" >&2; exit 2 ;;
    esac
    exit 0
fi

# ---------------------------------------------------------------------------------------- chain mode (login node)
SMOKE_V100=${SMOKE_V100:-}
SMOKE_A6000=${SMOKE_A6000:-}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
POLL=${POLL:-300}
PLANS=(slurm/stage1/plan_capacity.txt slurm/stage1/plan_capacity_screen.txt)
KINDS=(full screen)
LIMITS=("" 4000)
TLIMITS=("${TL_FULL:-16:00:00}" "${TL_SCREEN:-06:00:00}")
mkdir -p "$STATE" "$REP"
log() { echo "$(date -u +%FT%TZ) $*"; }
# sbatch without the login node's thread settings (activate.sh sets OMP / NUMBA threads to 4 outside Slurm) and
# without stray stage variables; every job writes its log to $LOGS
sub() {
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; return 0; fi
    env -u OMP_NUM_THREADS -u NUMBA_NUM_THREADS -u LIMIT_TRAIN -u S1TAG -u TB_EXTRA -u N_EVENTS -u RUN_OVERRIDE \
        -u GRAPHS_SUBDIR sbatch --parsable -o "$LOGS/%x_%j.out" "$@"
}
cpu() { sub -p submit,submit-gpu --exclude=$CPU_EXCLUDE -c 16 --mem=32G "$@"; }
HB=$STATE/heartbeat
if [ -e "$HB" ] && [ "${FORCE:-0}" != 1 ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
    if [ "$age" -lt $(( 2 * POLL + 60 )) ]; then
        log "another capacity chain is alive ($(cat "$HB"), ${age}s ago): not starting (FORCE=1 overrides)" >&2
        exit 1
    fi
fi
SUBMIT_LOCK=${SUBMIT_LOCK:-/work/submit/jinboz1/hnl/locks/hnl_submit.lock}
exec 9>> "$SUBMIT_LOCK" || { log "cannot open the submission lock $SUBMIT_LOCK"; exit 1; }
smoke_ok() {  # JOBID -> 0 if that smoke job trained, scored and evaluated the capacity IGNN
    [ -n "$1" ] && grep -qx "SMOKE OK ignn147" "$LOGS"/s1_smoke_cap_*_"$1".out 2>/dev/null
}
gpu_constraint() {  # the allowed GPU types now (A30 always; V100 / RTX A6000 after a passed smoke test)
    if [ -n "${GPU_CONSTRAINT:-}" ]; then echo "$GPU_CONSTRAINT"; return; fi
    if [ -n "${GPU_CONSTRAINT_FILE:-}" ] && [ -s "$GPU_CONSTRAINT_FILE" ]; then tr -d ' \n' < "$GPU_CONSTRAINT_FILE"; return; fi
    local c=nvidia_a30
    smoke_ok "$SMOKE_V100" && c="$c|Tesla_v100"
    smoke_ok "$SMOKE_A6000" && c="$c|nvidia_rtx6000"
    echo "$c"
}
log "capacity chain start pid $$ on $(hostname); smoke jobs V100 ${SMOKE_V100:-none} A6000 ${SMOKE_A6000:-none}; GPU constraint now $(gpu_constraint); time limits ${TLIMITS[*]}"

step() {  # NAME STEP SBATCH-ARGS... : submit once (job id kept in $STATE/NAME.STEP), print the job id
    local n=$1 s=$2 f j
    shift 2
    f=$STATE/$n.$s
    if [ -s "$f" ]; then cat "$f"; return 0; fi
    j=$(cpu "$@") && [[ "$j" =~ ^[0-9]+$|^DRY ]] || { log "sbatch failed for $n $s: $j" >&2; return 1; }
    [ "${DRY_RUN:-0}" = 1 ] || echo "$j" > "$f"
    echo "$j"
}

readout() {  # KIND RUN NAME -> job ids (colon list) of the read-out jobs; steps already submitted are not repeated
    local kind=$1 F=$2 n=$3 j0 j1 j2 j3 j4 ids
    if [ "$kind" = full ]; then
        j0=$(step "$n" e0122 -t 01:00:00 -J "s1cap_e0122_$n" slurm/stage1/val_py.sbatch "$F" hnl.eval.edge_tpr \
            --out "$F/edge_tpr_e0122.csv" --efpr 0.0122) || return 1
        j1=$(step "$n" estar -t 01:00:00 --dependency=afterany:$j0 -J "s1cap_estar_$n" slurm/stage1/edge_tpr_estar.sbatch \
            "$F" "$WP") || return 1
        ids="$j0:$j1"
    else
        j1=$(step "$n" estar -t 01:00:00 -J "s1cap_estar_$n" slurm/stage1/edge_tpr_estar.sbatch "$F" 0.0122) || return 1
        ids="$j1"
    fi
    j2=$(step "$n" probe -t 04:00:00 -J "s1cap_probe_$n" slurm/stage1/truncation_probe.sbatch "$F") || return 1
    j3=$(step "$n" tb -t 06:00:00 -J "s1cap_tb_$n" slurm/stage1/trackbuild.sbatch "$F" walkmergeS,mwslayer,mwslayermerge \
        trackbuild_val) || return 1
    ids="$ids:$j2:$j3"
    if [ "$kind" = full ]; then
        j4=$(step "$n" tbx -t 10:00:00 -J "s1cap_tbx_$n" slurm/stage1/trackbuild.sbatch "$F" \
            walkmergeX,mwslayermergeX,mws2passmergeX trackbuild_val_ext) || return 1
        ids="$ids:$j4"
    fi
    echo "$ids"
}

while true; do
    [ -e "$STATE/STOP" ] && { log "STOP file found, exiting"; rm -f "$HB"; exit 0; }
    echo "$(hostname) $$ $(date -u +%FT%TZ)" > "$HB"
    q=$(squeue -h -u "$USER" -o "%i %j %T %f" 2>/dev/null) || { log "squeue failed"; sleep "$POLL"; continue; }
    want=$(gpu_constraint)
    left=0
    for idx in 0 1; do
        read -r NAME CONFIG ARM SEED EPOCHS < <(grep -v '^#' "${PLANS[$idx]}" | awk 'NF' | head -n 1)
        kind=${KINDS[$idx]}
        F=$BASE/$NAME/arm${ARM}_s$SEED
        jname=s1_${NAME}_${ARM}_s${SEED}
        tfile=$STATE/$NAME.train.jobs
        if [ ! -e "$F/eval.done" ]; then
            [ -e "$STATE/$NAME.gaveup" ] && continue
            left=1
            row=$(awk -v n="$jname" '$2 == n' <<< "$q" | head -n 1)
            if [ -n "$row" ]; then                                              # queued or running
                read -r qid _ qstate qfeat <<< "$row"
                if [ "$qstate" = PENDING ] && [ "$qfeat" != "$want" ] && [ "${DRY_RUN:-0}" != 1 ]; then
                    if scontrol update JobId="$qid" Features="$want"; then
                        echo "$qid features $want $(date -u +%FT%TZ)" >> "$tfile"
                        log "$jname job $qid pending: GPU constraint $qfeat -> $want"
                    else
                        log "scontrol update of $qid failed"
                    fi
                fi
                continue
            fi
            last=$(grep -E ' (submitted|ended) ' "$tfile" 2>/dev/null | tail -n 1)
            if [ -n "$last" ] && ! grep -q ' ended ' <<< "$last"; then
                jid=$(cut -d' ' -f1 <<< "$last")
                info=$(sacct -n -X -j "$jid" -o State,Elapsed,NodeList -P 2>/dev/null | head -n 1)
                echo "$jid ended $info" >> "$tfile"
                log "$jname job $jid ended without eval.done: $info"
            fi
            n=$( { grep -c ' submitted ' "$tfile" 2>/dev/null || true; } | tail -n 1); n=${n:-0}
            if [ "$n" -ge "$MAX_ATTEMPTS" ]; then
                log "GIVING UP $jname after $n submissions"; touch "$STATE/$NAME.gaveup"; continue
            fi
            exp="ALL,S1TAG=capacity"
            [ -n "${LIMITS[$idx]}" ] && exp="$exp,LIMIT_TRAIN=${LIMITS[$idx]}"
            flock 9 || { log "submission lock failed"; continue; }
            out=$(sub --export="$exp" -J "$jname" -p submit-gpu --constraint="$want" --gres=gpu:1 -c 8 \
                --mem=16G -t "${TLIMITS[$idx]}" slurm/stage1/variant.sbatch "$NAME" "$HNL_REPO/$CONFIG" "$ARM" "$SEED" "$EPOCHS" 2>&1)
            flock -u 9
            if [[ "$out" =~ ^[0-9]+$ ]] || [[ "$out" =~ ^DRY ]]; then
                out=${out%%$'\n'*}
                [ "${DRY_RUN:-0}" = 1 ] || echo "$out submitted $(date -u +%FT%TZ) $NAME $CONFIG $ARM $SEED $EPOCHS limit=${LIMITS[$idx]:-all} constraint=$want tl=${TLIMITS[$idx]}" >> "$tfile"
                log "submitted $jname -> $out (submission $((n + 1)))"
            else
                log "sbatch failed for $jname: $out"
            fi
            continue
        fi
        if [ ! -e "$STATE/$NAME.readout.submitted" ]; then
            left=1
            info=$(python -I -c "
import glob, json, sys
rs = [json.load(open(p)) for p in glob.glob(sys.argv[1] + '/gnn/run_*.json')]
print('; '.join(f\"job {r.get('slurm_job')} {r.get('gpu')} on {r.get('host')}: {r.get('state')}, {r.get('epochs_completed')} epochs, \"
                f\"{r.get('fit_seconds')} s, {r.get('n_parameters')} parameters, peak {r.get('peak_gpu_mem_gib')} GiB\" for r in rs))" "$F")
            log "$NAME eval.done; training: $info"
            flock 9 || { log "submission lock failed"; continue; }
            ids=$(readout "$kind" "$F" "$NAME")
            rc=$?
            flock -u 9
            if [ $rc = 0 ] && [ -n "$ids" ]; then
                echo "$ids" > "$STATE/$NAME.readout.jobs"
                touch "$STATE/$NAME.readout.submitted"
                log "$NAME read-out jobs $ids"
            else
                log "read-out submission failed for $NAME (partial: ${ids:-none}); retrying next pass"
            fi
            continue
        fi
        if [ ! -e "$STATE/$NAME.report.submitted" ]; then
            left=1
            ids=$(cat "$STATE/$NAME.readout.jobs")
            j=$(step "$NAME" report -t 02:00:00 --dependency=afterany:$ids -J "s1cap_report_$kind" \
                --wrap "bash $HNL_REPO/slurm/stage1/capacity_chain.sh --report $kind") \
                && touch "$STATE/$NAME.report.submitted" && log "$NAME report job $j (after $ids)"
        fi
    done
    [ "$left" = 0 ] && { log "capacity chain: all submitted"; rm -f "$HB"; exit 0; }
    [ "${DRY_RUN:-0}" = 1 ] && { log "dry run: one pass only"; rm -f "$HB"; exit 0; }
    sleep "$POLL"
done
