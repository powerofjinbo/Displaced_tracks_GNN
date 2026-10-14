#!/bin/bash
# M3 chain, run on a login node once (light: waits and submits): when all nine Stage-0 runs are evaluated on val, the
# official val analysis (working points from 3 arms x 3 seeds, headroom tables, M3/M4 figures); the working points are
# copied into the repo and committed (preregistration: fixed before any test number); the chain check; the nine
# ONE-TIME test evaluations (nice 0: first in the queue); then the test statistics incl. the paper comparison.
# Usage: setsid nohup bash slurm/baseline/m3_chain.sh >> $HNL_LOGS/m3_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
B=$HNL_DATA/runs/baseline_v1
log() { echo "$(date -u +%FT%TZ) $*"; }
RUNS=""
for a in i ii iii; do for s in 0 1 2; do RUNS="$RUNS arm${a}_s${s}"; done; done
log "m3 chain start pid $$ on $(hostname)"
while true; do
    n=0
    for r in $RUNS; do
        [ -e "$B/$r/HOLD" ] && { log "HOLD in $r: stopping the chain (decide by hand)"; exit 1; }
        [ -e "$B/$r/eval.done" ] && n=$((n + 1))
    done
    [ "$n" = 9 ] && break
    sleep 300
done
log "all 9 runs evaluated on val"
J=$(KIND=cpu POLL=60 bash slurm/submit_when_slot.sh slurm/baseline/val_analysis.sbatch i,ii,iii 0,1,2)
log "val analysis job $J"
s=$(POLL=60 bash slurm/wait_job.sh "$J")
[ "$s" = COMPLETED ] || { log "val analysis $J ended $s: stopping"; exit 1; }
WP=$B/analysis_val/working_points.json
mkdir -p results
cp "$WP" results/working_points_v1.json
git add results/working_points_v1.json
git commit -q -m "Working points v1: the preregistered val working points from all 3 arms x 3 seeds (before any test evaluation)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
" || { log "git commit failed: stopping"; exit 1; }
log "working points committed as $(git rev-parse HEAD) (sha256 $(sha256sum "$WP" | cut -c1-16))"
python -m hnl.train.chain_check --run "$B/armi_s0" --wp "$WP" || { log "chain check failed: stopping"; exit 1; }
JOBS=""
for a in i ii iii; do for s in 0 1 2; do
    if [ -e "$B/arm${a}_s${s}/test_eval.done" ]; then log "arm $a seed $s already evaluated on test: skipped"; continue; fi
    j=$(MAX_GPU=16 POLL=60 bash slurm/submit_when_slot.sh -J "bl_test_eval_${a}_s${s}" slurm/baseline/test_eval.sbatch "$a" "$s" "$WP")
    log "test evaluation arm $a seed $s -> $j"
    JOBS="$JOBS $j"
done; done
for j in $JOBS; do
    s=$(POLL=120 bash slurm/wait_job.sh "$j")
    [ "$s" = COMPLETED ] || log "test evaluation $j ended $s (see its log)"
done
J=$(KIND=cpu POLL=60 bash slurm/submit_when_slot.sh -c 4 --mem=32G -t 04:00:00 -J test_stats slurm/run_py.sbatch \
    hnl.eval.test_stats --runs "$B" --wp "$WP" --out "$B/analysis_test")
log "test statistics job $J"
s=$(POLL=60 bash slurm/wait_job.sh "$J")
log "test statistics $J ended $s"
log "m3 chain done"
