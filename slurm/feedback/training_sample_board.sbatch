#!/bin/bash
# Training-sample study, val only: hnl.eval.training_sample_board for one builder family -> results/training_sample_val_<FAMILY>.csv
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/training_sample_board.sbatch FAMILY   (mwslmerge | mwslayermerge)
#SBATCH -J ts_board
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
python -m hnl.eval.training_sample_board --family "$1" --out "results/training_sample_val_$1${2:+_$2}.csv" ${3:+--new-grids "$3"}
