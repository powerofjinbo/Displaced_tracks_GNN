#!/bin/bash
# Feedback round 3, task A2 (val): unite every grid part of the nine acorn runs (RUN/trackbuild_val_ext,
# RUN/trackbuild_val_acwalk_all, RUN/trackbuild_val_fb3_lmerge and the extension parts -> RUN/trackbuild_val_fb3_all, a
# data copy), then the val tables, the per-seed full-menu choices, the edge table, the differences, the identity checks
# and the val-fixed test spec (results/strong_baselines_v3_*), then print the test job commands.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/fb3_val.sbatch [SKIP_MERGE=1 in the environment: no merge]
#SBATCH -J fb3_val
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
[ "${SKIP_MERGE:-0}" = 1 ] || python -I -m hnl.eval.strong_baselines_v3 merge
python -I -m hnl.eval.strong_baselines_v3 val
python -I -m hnl.eval.strong_baselines_v3 jobs
echo "done $(date -u +%FT%TZ)"
