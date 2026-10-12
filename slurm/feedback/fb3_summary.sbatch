#!/bin/bash
# Feedback round 3, task A2 (test, post-freeze descriptive): the test tables, differences, identity checks and the
# markdown of results/strong_baselines_v3 from the val-fixed spec and its test outputs (provenance and checksums checked).
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/fb3_summary.sbatch [--notes FILE]
#SBATCH -J fb3_summary
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
python -I -m hnl.eval.strong_baselines_v3 test "$@"
echo "done $(date -u +%FT%TZ)"
