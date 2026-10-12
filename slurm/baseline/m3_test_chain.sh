#!/bin/bash
# The test part of the M3 chain (slurm/baseline/m3_chain.sh), for a restart after the working points are committed:
# the nine ONE-TIME test evaluations (runs without test_eval.done; nice 0) submitted directly (no slot wait: the
# chain's GPU-slot count also counted CPU jobs queued on submit,submit-gpu), then the test statistics.
# Refuses unless the analysis working points equal the committed results/working_points_v1.json.
# Usage: setsid nohup bash slurm/baseline/m3_test_chain.sh >> $HNL_LOGS/m3_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
B=$HNL_DATA/runs/baseline_v1
WP=$B/analysis_val/working_points.json
log() { echo "$(date -u +%FT%TZ) $*"; }
git ls-files --error-unmatch results/working_points_v1.json > /dev/null 2>&1 && git diff --quiet HEAD -- results/working_points_v1.json \
    || { log "results/working_points_v1.json is not committed as is: stopping"; exit 1; }
[ "$(sha256sum < "$WP")" = "$(sha256sum < results/working_points_v1.json)" ] \
    || { log "$WP differs from the committed working points: stopping"; exit 1; }
log "m3 test chain start pid $$ on $(hostname); working points $(sha256sum "$WP" | cut -c1-16)"
JOBS=""
for a in i ii iii; do for s in 0 1 2; do
    if [ -e "$B/arm${a}_s${s}/test_eval.done" ]; then log "arm $a seed $s already evaluated on test: skipped"; continue; fi
    if squeue -h -u "$USER" -o "%j" | grep -qx "bl_test_eval_${a}_s${s}"; then log "bl_test_eval_${a}_s${s} already queued: skipped"; continue; fi
    j=$(sbatch --parsable -J "bl_test_eval_${a}_s${s}" slurm/baseline/test_eval.sbatch "$a" "$s" "$WP")
    log "test evaluation arm $a seed $s -> $j"
    JOBS="$JOBS $j"
done; done
for j in $JOBS; do
    s=$(POLL=120 bash slurm/wait_job.sh "$j")
    [ "$s" = COMPLETED ] || log "test evaluation $j ended $s (see its log)"
done
J=$(sbatch --parsable -c 4 --mem=32G -t 04:00:00 -J test_stats slurm/run_py.sbatch \
    hnl.eval.test_stats --runs "$B" --wp "$WP" --out "$B/analysis_test")
log "test statistics job $J"
s=$(POLL=60 bash slurm/wait_job.sh "$J")
log "test statistics $J ended $s"
log "m3 chain done"
