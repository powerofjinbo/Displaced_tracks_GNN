#!/bin/bash
# Feedback round 2, task E (acorn (ttbar+HNL), val): unite the CCandWalk grid parts of the three arm (iii) runs
# (RUN/trackbuild_val_acwalk + RUN/trackbuild_val_acwalk_ext -> RUN/trackbuild_val_acwalk_all, a data copy), then the
# val tables, differences, identity checks and the final val-fixed test spec, then print the test job commands.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/armiii_val.sbatch [extra arguments of `val`]
#SBATCH -J fb2E_val
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
python -I -m hnl.eval.strong_baselines_armiii merge
python -I -m hnl.eval.strong_baselines_armiii val "$@"
python -I -m hnl.eval.strong_baselines_armiii jobs
echo "done $(date -u +%FT%TZ)"
