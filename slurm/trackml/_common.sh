# Sourced by slurm/trackml/*.sbatch (Canto-L on TrackML, docs/trackml_plan.md): environment, the pipeline config
# (configs/trackml/pipeline_v1.yaml through hnl.trackml.config; HNL_TML_CONFIG / HNL_TML_SPLIT override), the run
# directory of a seed, node-local scratch and helpers. Each stage writes $RUN/<stage>.done on success.
#
# Stage order (plan section 8; a gate failing stops the chain and is reported to the user):
#   S1  ml_train.sbatch                     metric learning (A30)                           -> ml_train.done
#   S2  graph_radii.sbatch                  r* per density target on valmon                  -> graph/radii.json
#       support_train.sbatch                learned-support GBDT rows (gbdt events) + model  -> support/model.joblib
#       graph_scan.sbatch (array)           valmon scan density x r_w x (tau, k)             -> graph/scan/
#       graph_choose.sbatch                 choice + gate G1                                 -> graph/graph_choice.json
#   S3  additions.sbatch SET [COND] (array) union additions (train, valmon, val, dev, ...)  -> additions/<SET>[_COND]/
#   S4  classifier_train.sbatch             windows on A30, resumable (2-day limit: resubmit) -> gnn_train.done
#       gate_g2.sbatch                      stitched vs full-event scores (first-epoch ckpt)  -> gates/gate_g2_*.json
#   S5  score.sbatch SET [COND]             8-sector scores, stitched (A30)                   -> scores/<SET>[_COND]/
#       gate_g3.sbatch                      builder runtime + exact fast merging (dev scores) -> gates/gate_g3.json
#   S6  lmerge_train.sbatch                 learned merging on merge_train scores             -> lmerge/model.joblib
#       trackbuild_grid.sbatch COND         val builder grid (both metric families)          -> grid/<COND>/
#       select.sbatch                       WP-A / WP-S / WP-R (then the freeze is committed by the main session)
#   S7  test_chain.sh                       the one-time test (refuses without a committed selection)
#   smoke_probe.sbatch                      CPU sizing and checks without trained networks
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
# nodes without AVX2 (the GTX 1080 Ti nodes): Polars' compat runtime from a side directory, as slurm/baseline/_common.sh
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
cd "$HNL_REPO"
cfg() { python -m hnl.trackml.config get "$1" 2>/dev/null | tail -n 1; }
events() { python -m hnl.trackml.config events "$@" 2>/dev/null | tail -n 1; }
SEED=${SEED:-$(cfg seed)}
RUN_ROOT=${RUN_ROOT:-$(cfg run_root)}
# RUN_SUFFIX: a sibling run of the same seed (X1, plan D3: RUN_SUFFIX=_x1 -> seed0_x1); empty for the main run
RUN=$RUN_ROOT/seed$SEED${RUN_SUFFIX:-}
LOCAL=/tmp/$USER/${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}
mkdir -p "$RUN" "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
# the CPUs of this allocation (not an OMP_NUM_THREADS inherited from the submitting shell, which nproc would echo)
export OMP_NUM_THREADS=${OMP_NUM_THREADS_OVERRIDE:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}
TML_CONFIG=${HNL_TML_CONFIG:-$HNL_REPO/configs/trackml/pipeline_v1.yaml}
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src env configs slurm | wc -l); host $(hostname); job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}; seed $SEED; run $RUN; config $TML_CONFIG ($(sha256sum "$TML_CONFIG" | cut -c1-12)); split ${HNL_TML_SPLIT:-$(cfg data.split)}; cpus $(nproc); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo no-gpu)"

# a command in the background so that the pre-timeout SIGUSR1 (sbatch --signal=B:USR1@300) reaches it
run_fwd() {
    "$@" &
    FWD_PID=$!
    trap 'kill -USR1 $FWD_PID 2>/dev/null' USR1
    local rc
    while true; do
        rc=0
        wait $FWD_PID || rc=$?
        kill -0 $FWD_PID 2>/dev/null || break
    done
    trap - USR1
    return $rc
}

# checkpoint to resume a training stage from, empty if none: the one with the largest global step among preempt /
# periodic / last, and on a tie the first of these (the rule of slurm/baseline/_common.sh). After a pre-timeout stop
# Lightning used to validate and write last.ckpt at the same step with the epoch marked as ended, so resuming from
# last.ckpt (the newest file) would skip the rest of that epoch; preempt.ckpt (saved mid-epoch) finishes it (with
# hnl.trackml.ml's sampler, on the same events as an uninterrupted run). Since 2026-10-08 the stop
# (hnl.train.resume_sampler.PreemptStop) skips that validation in the middle of an epoch; the rule is unchanged. A checkpoint that cannot be read (a
# chunk killed while writing it) is skipped; if checkpoints exist but none can be read, the job stops instead of
# starting from scratch.
resume_ckpt() {
    local d="$RUN/$1/artifacts"
    python -I - "$d/preempt.ckpt" "$d/periodic/periodic.ckpt" "$d/last.ckpt" <<'PYEOF'
import os, sys
import torch
best, bad = None, []
for p in sys.argv[1:]:
    if os.path.exists(p):
        try:
            step = torch.load(p, map_location="cpu", weights_only=False).get("global_step", -1)
        except Exception as e:
            bad.append(p)
            print(f"unreadable checkpoint skipped: {p} ({type(e).__name__})", file=sys.stderr)
            continue
        if best is None or step > best[0]:
            best = (step, p)
if best is not None:
    print(best[1])
elif bad:
    sys.exit("no readable checkpoint among " + ", ".join(bad))
PYEOF
}

# plan section 7 stop rule before a job writes to /ceph (hnl.trackml.convert.storage_check): projected bytes
storage_gate() {
    python -c "import json, sys; from hnl.trackml.convert import storage_check; print('storage', json.dumps(storage_check(float(sys.argv[1]))))" "$1"
}

# best checkpoint of the latest finished training of a stage (its run_<job>.json)
best_ckpt() {
    python -I -c "
import glob, json, os, sys
runs = [json.load(open(p)) for p in glob.glob(os.path.join(sys.argv[1], 'run_*.json'))]
runs = [r for r in runs if r.get('state') == 'done' and r.get('best_ckpt')]
if not runs:
    sys.exit('no finished training run in ' + sys.argv[1])
r = max(runs, key=lambda r: r['end_utc'])
if not os.path.exists(r['best_ckpt']):
    sys.exit('missing ' + r['best_ckpt'])
print(r['best_ckpt'])" "$RUN/$1"
}

# checkpoint of the classifier after its first virtual epoch (gates G2 / G3): TrackMLCanto saves
# artifacts/epochs/virtual_epoch0.ckpt itself (acorn_runner keeps only every 5th epoch); EPOCH1_CKPT overrides
first_epoch_ckpt() {
    if [ -n "${EPOCH1_CKPT:-}" ]; then echo "$EPOCH1_CKPT"; return; fi
    local f=$RUN/gnn/artifacts/epochs/virtual_epoch0.ckpt
    [ -e "$f" ] || { echo "no first-epoch checkpoint $f yet" >&2; return 1; }
    echo "$f"
}

# train a stage (acorn_runner) and write $RUN/<stage>_train.done if it finished (not preempted)
train_stage() {
    local stage=$1 epochs=$2 input=$3 config=$4
    shift 4
    local resume
    resume=$(resume_ckpt "$stage") || { echo "resume_ckpt failed: stop"; return 1; }
    echo "resume from: ${resume:-none}"
    mkdir -p "$RUN/$stage"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
        > "$RUN/$stage/gpu_util_$SLURM_JOB_ID.csv" 2>/dev/null &
    local smi=$!
    run_fwd python -m hnl.train.acorn_runner train --stage "$stage" --arm trackml --seed "$SEED" --run-dir "$RUN" \
        --input-dir "$input" --max-epochs "$epochs" --config "$config" ${resume:+--resume "$resume"} "$@"
    kill $smi 2>/dev/null || true
    local state
    state=$(python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['state'])" "$RUN/$stage/run_$SLURM_JOB_ID.json")
    echo "training state: $state"
    if [ "$state" = done ]; then touch "$RUN/${stage}_train.done"; fi
}

# events of a set, sliced for a Slurm array task (every n-th event from the task index)
array_slice() {
    python -I -c "
import sys
ev = [e for e in sys.argv[1].split(',') if e]
i, n = int(sys.argv[2]), int(sys.argv[3])
print(','.join(ev[i::n]))" "$1" "${SLURM_ARRAY_TASK_ID:-0}" "${SCAN_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"   # SCAN_TASK_COUNT: a resubmitted single task of an n-task array
}

require() { for f in "$@"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done; }

# directory of the union additions that hold an event set in a condition: valmon reads val's; the subsets of train
# (dev, gbdt, calib, merge_train) read train's
adds_dir() {
    local s=$1 c=${2:-full}
    case $s in valmon) s=val ;; dev|gbdt|calib|merge_train) s=train ;; esac
    if [ "$c" = full ]; then echo "$RUN/additions/$s"; else echo "$RUN/additions/${s}_$c"; fi
}

# directory of the scores of an event set, condition and tag
scores_dir() {
    local d=$RUN/scores/$1
    if [ "${2:-full}" != full ]; then d=${d}_$2; fi
    if [ -n "${3:-}" ]; then d=${d}_$3; fi
    echo "$d"
}
