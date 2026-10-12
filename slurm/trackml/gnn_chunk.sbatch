#!/bin/bash
# One resumable chunk of the TrackML classifier training S4 (slurm/trackml/classifier_train.sbatch with
# STORE=$RUN/store) for slurm/trackml/gnn_chunks.sh: exits at once when $RUN/gnn_train.done exists; gives up (and
# cancels the pending chunks of its own chain, same job name) when neither the staged event store nor the global step
# grew over the last NOPROG chunks; otherwise execs classifier_train.sbatch, which first fills the persistent store
# (event files already there are kept) and, once $RUN/store.done exists, resumes the training from the checkpoint with
# the largest global step; on the pre-timeout SIGUSR1 it writes preempt.ckpt and exits 0. NOPROG default 2.
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
SEED=${SEED:-$(python -m hnl.trackml.config get seed 2>/dev/null | tail -n 1)}
RUN=${RUN_ROOT:-$(python -m hnl.trackml.config get run_root 2>/dev/null | tail -n 1)}/seed$SEED${RUN_SUFFIX:-}
[ -e "$RUN/gnn_train.done" ] && { echo "chunk $SLURM_JOB_ID: $RUN/gnn_train.done exists, nothing to do"; exit 0; }
mkdir -p "$RUN/gnn"
STEP=$(python -I - "$RUN/gnn" <<'PYEOF' 2>/dev/null
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
NST=$(ls "$RUN/store/trainset" "$RUN/store/valset" 2>/dev/null | grep -c '\.pyg\.gz$')
PF=$RUN/gnn/chunk_progress.txt
echo "$SLURM_JOB_ID ${STEP:-0}+${NST} $(date -u +%FT%TZ) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)" >> "$PF"
NOPROG=${NOPROG:-2}
if [ "$(tail -n $((NOPROG + 1)) "$PF" | awk '{print $2}' | sort -u | wc -l)" = 1 ] && [ "$(wc -l < "$PF")" -gt "$NOPROG" ]; then
    echo "chunk $SLURM_JOB_ID: store and global step unchanged over the last $NOPROG chunks: giving up, cancelling the pending chunks of $SLURM_JOB_NAME"
    scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING
    exit 1
fi
echo "chunk $SLURM_JOB_ID: run $RUN at global step ${STEP:-0}, $NST event files in the store"
export STORE=$RUN/store
exec bash "$HNL_REPO/slurm/trackml/classifier_train.sbatch"
