#!/bin/bash
# One resumable chunk of the TrackML metric learning (slurm/trackml/ml_train.sbatch) for slurm/trackml/ml_chunks.sh:
# exits at once when $RUN/ml_train.done exists; gives up (and cancels the pending chunks of its own chain, same job name)
# when the run's global step did not grow over the last NOPROG chunks; otherwise execs ml_train.sbatch, which resumes
# from the checkpoint with the largest global step (preempt.ckpt on ties) and, on the pre-timeout SIGUSR1, writes
# preempt.ckpt and exits 0. Environment as ml_train.sbatch (RUN_SUFFIX, ML_CONFIG, SEED); NOPROG default 2.
#SBATCH -p submit-gpu-express
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
SEED=${SEED:-$(python -m hnl.trackml.config get seed 2>/dev/null | tail -n 1)}
RUN=${RUN_ROOT:-$(python -m hnl.trackml.config get run_root 2>/dev/null | tail -n 1)}/seed$SEED${RUN_SUFFIX:-}
[ -e "$RUN/ml_train.done" ] && { echo "chunk $SLURM_JOB_ID: $RUN/ml_train.done exists, nothing to do"; exit 0; }
mkdir -p "$RUN/ml"
STEP=$(python -I - "$RUN/ml" <<'PYEOF' 2>/dev/null
import glob, json, os, sys
import torch
d = sys.argv[1]
s = [json.load(open(p)).get("global_step") or 0 for p in glob.glob(os.path.join(d, "run_*.json"))]
for p in ("artifacts/preempt.ckpt", "artifacts/periodic/periodic.ckpt", "artifacts/last.ckpt"):
    if os.path.exists(os.path.join(d, p)):
        try:
            s.append(torch.load(os.path.join(d, p), map_location="cpu", weights_only=False).get("global_step", 0))
        except Exception:
            pass
print(max(s) if s else 0)
PYEOF
)
PF=$RUN/ml/chunk_progress.txt
echo "$SLURM_JOB_ID ${STEP:-0} $(date -u +%FT%TZ) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)" >> "$PF"
NOPROG=${NOPROG:-2}
if [ "$(tail -n $((NOPROG + 1)) "$PF" | awk '{print $2}' | sort -u | wc -l)" = 1 ] && [ "$(wc -l < "$PF")" -gt "$NOPROG" ]; then
    echo "chunk $SLURM_JOB_ID: global step ${STEP:-0} unchanged over the last $NOPROG chunks: giving up, cancelling the pending chunks of $SLURM_JOB_NAME"
    scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING
    exit 1
fi
echo "chunk $SLURM_JOB_ID: run $RUN at global step ${STEP:-0}"
exec bash "$HNL_REPO/slurm/trackml/ml_train.sbatch"
