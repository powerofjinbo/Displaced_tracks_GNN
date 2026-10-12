#!/bin/bash
# Training-sample study, val only: paired event-bootstrap uncertainties of the training-sample differences
# (hnl.eval.training_sample_paired) -> results/training_sample_paired_v1.csv
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/training_sample_paired.sbatch
#SBATCH -J ts_paired
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
python -m hnl.eval.training_sample_paired --out results/training_sample_paired_v1.csv
