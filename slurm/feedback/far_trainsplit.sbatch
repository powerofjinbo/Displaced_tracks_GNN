#!/bin/bash
# Far-displaced HNL daughters on the HNL TRAINING split (diagnostic only; nothing is chosen, no test file or test output
# is read): hnl.eval.far_trainsplit population (the daughters with |d0| >= 200 mm in the HNL train events) or analyze
# (loss classes, identity check on val, merge decisions, truncation start radii; after far_trainsplit_graphs.sbatch for
# seeds 0-2). Outputs under $HNL_DATA/runs/feedback_v4/far_trainsplit and results/<NAME>*.
# Usage: env -u OMP_NUM_THREADS sbatch [-c N --mem M] slurm/feedback/far_trainsplit.sbatch population|analyze [NAME]
#   (RESULTS=DIR: another output directory than results/; EXTRA: further arguments of analyze, e.g. "--seeds 0")
#SBATCH -J fts_cpu
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
# nodes without AVX2 (the GTX 1080 Ti nodes): Polars' compat runtime (as slurm/baseline/_common.sh)
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
CMD=${1:-population}; NAME=${2:-far_trainsplit_v1}
export LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; cmd $CMD; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # one thread per worker process
if [ "$CMD" = population ]; then
    python -m hnl.eval.far_trainsplit population
else
    python -m hnl.eval.far_trainsplit analyze --results "${RESULTS:-results}" --name "$NAME" \
        --workers "$SLURM_CPUS_PER_TASK" --tmp "$LOCAL" ${EXTRA:-}
fi
echo "done $(date -u +%FT%TZ)"
