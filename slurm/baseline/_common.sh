# Sourced by the baseline stage scripts (ARM SEED as $1 $2): environment, node-local scratch, run directory, helpers.
# Each stage writes $RUN/<stage>.done on success; slurm/baseline/orchestrate.sh submits the next stage.
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
# nodes without AVX2 (the GTX 1080 Ti nodes): Polars' compat runtime from a side directory; elsewhere unchanged
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
cd "$HNL_REPO"
ARM=$1
SEED=$2
TAG=${TAG:-v1}
RUN=${RUN_OVERRIDE:-$HNL_DATA/runs/baseline_${TAG}/arm${ARM}_s${SEED}}
LOCAL=/tmp/$USER/$SLURM_JOB_ID${SUBDIR:+/$SUBDIR}
mkdir -p "$LOCAL" "$RUN"
trap 'rm -rf "$LOCAL"' EXIT
R="python -m hnl.train.acorn_runner"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src env configs slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; arm $ARM seed $SEED; run $RUN; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo no-gpu)"

# run a command in the background so that the pre-timeout SIGUSR1 (sbatch --signal=B:USR1@300) reaches it;
# returns the command's own exit status
run_fwd() {
    "$@" &
    FWD_PID=$!
    trap 'kill -USR1 $FWD_PID 2>/dev/null' USR1
    local rc
    while true; do
        rc=0
        wait $FWD_PID || rc=$?
        kill -0 $FWD_PID 2>/dev/null || break          # still alive: wait was interrupted by the trap
    done
    trap - USR1
    return $rc
}

# checkpoint to resume a training stage from, empty if none: the one with the largest global step among preempt /
# periodic / last, and on a tie the first of these. After a pre-timeout stop (PreemptCallback) Lightning validates and
# writes last.ckpt at the same step with the epoch marked as ended, so resuming from last.ckpt would skip the rest of
# that epoch; preempt.ckpt (saved mid-epoch) finishes it.
resume_ckpt() {
    local d="$RUN/$1/artifacts"
    python -I - "$d/preempt.ckpt" "$d/periodic/periodic.ckpt" "$d/last.ckpt" <<'PYEOF' 2>/dev/null || true
import os, sys
import torch
best = None
for p in sys.argv[1:]:
    if os.path.exists(p):
        step = torch.load(p, map_location="cpu", weights_only=False).get("global_step", -1)
        if best is None or step > best[0]:
            best = (step, p)
if best is not None:
    print(best[1])
PYEOF
}

# train a stage and write $RUN/<stage>_train.done if it finished (not preempted); extra arguments go to the runner
train_stage() {
    local stage=$1 epochs=$2 input=$3
    shift 3
    local resume
    resume=$(resume_ckpt "$stage")
    echo "resume from: ${resume:-none}"
    # GPU utilisation / memory once a minute (to size later runs; harmless if nvidia-smi is missing)
    mkdir -p "$RUN/$stage"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
        > "$RUN/$stage/gpu_util_$SLURM_JOB_ID.csv" 2>/dev/null &
    local smi=$!
    run_fwd $R train --stage "$stage" --arm "$ARM" --seed "$SEED" --run-dir "$RUN" --input-dir "$input" \
        --max-epochs "$epochs" ${resume:+--resume "$resume"} "$@"
    kill $smi 2>/dev/null || true
    local state
    state=$(python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['state'])" "$RUN/$stage/run_$SLURM_JOB_ID.json")
    echo "training state: $state"
    if [ "$state" = done ]; then
        touch "$RUN/${stage}_train.done"
    fi
}

# best checkpoint of the latest finished training run of a stage (from its run_<job>.json)
best_ckpt() {
    # RUN/<stage>/best_ckpt_override.json {"ckpt": ..., "reason": ...} replaces the recorded best checkpoint (written by
    # hand when that file was lost; DECISIONS 2026-10-07 ~21:15 UTC)
    python -I -c "
import glob, json, os, sys
o = os.path.join(sys.argv[1], 'best_ckpt_override.json')
if os.path.exists(o):
    d = json.load(open(o))
    if not os.path.exists(d['ckpt']):
        sys.exit('missing override ' + d['ckpt'])
    print('best checkpoint override: ' + d['ckpt'] + ' (' + d['reason'] + ')', file=sys.stderr)
    print(d['ckpt'])
    sys.exit(0)
runs = [json.load(open(p)) for p in glob.glob(os.path.join(sys.argv[1], 'run_*.json'))]
runs = [r for r in runs if r.get('state') == 'done' and r.get('best_ckpt')]
if not runs:
    sys.exit('no finished training run in ' + sys.argv[1])
r = max(runs, key=lambda r: r['end_utc'])
if not os.path.exists(r['best_ckpt']):
    sys.exit('missing ' + r['best_ckpt'])
print(r['best_ckpt'])" "$RUN/$1"
}
