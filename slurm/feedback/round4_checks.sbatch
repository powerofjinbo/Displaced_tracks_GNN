#!/bin/bash
# Supervisor feedback, second round (validation only; no test file or test output is read): hnl.eval.round4_checks,
# check 1 (primary efficiency on the acorn (HNL) checkpoint-monitoring events "valmon" and on the other HNL val events)
# and check 2 (where the far-displaced daughters are lost). Outputs results/<NAME>_*.
# Usage: env -u OMP_NUM_THREADS sbatch -J fbR2_checks slurm/feedback/round4_checks.sbatch [CMD=all|valmon|losses] [NAME]
#SBATCH -J fbR2_checks
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
CMD=${1:-all}; NAME=${2:-round4_checks}
export LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # one thread per worker process
python -m hnl.eval.round4_checks "$CMD" --out results --name "$NAME" --workers "$SLURM_CPUS_PER_TASK" --tmp "$LOCAL"
echo "done $(date -u +%FT%TZ)"
