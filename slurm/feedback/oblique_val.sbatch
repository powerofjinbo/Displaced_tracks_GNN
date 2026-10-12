#!/bin/bash
# Training-sample study, val only (no test file or test output is read): hnl.eval.oblique_val, graph- and
# classifier-stage losses of the primary-population HNL daughters on val versus |d0| / r_first. Outputs results/<NAME>_*.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/oblique_val.sbatch NAME LABEL=RUN_DIR:ARM [LABEL=RUN_DIR:ARM ...]
#SBATCH -J obl_val
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
NAME=$1; shift
RUNS=(); for r in "$@"; do RUNS+=(--run "$r"); done
export LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # one thread per worker process
python -m hnl.eval.oblique_val "${RUNS[@]}" --out results --name "$NAME" --workers "$SLURM_CPUS_PER_TASK" --tmp "$LOCAL"
echo "done $(date -u +%FT%TZ)"
