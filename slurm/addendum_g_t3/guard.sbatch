#!/bin/bash
# Addendum G T3: every family's per-run choice on the ttbar and HNL val events and the paper's prompt guard against
# acorn arm (i) (hnl.eval.addendum_g_t3 guard -> RUN/guard_val.json). Usage: sbatch guard.sbatch acorn|canto S
#SBATCH -J g3_guard
#SBATCH -p submit-gpu-express,submit-gpu,submit
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 24
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1 OMP_NUM_THREADS=1
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $*"
python -m hnl.eval.addendum_g_t3 guard --model "$1" --seed "$2" --workers "$SLURM_CPUS_PER_TASK"
