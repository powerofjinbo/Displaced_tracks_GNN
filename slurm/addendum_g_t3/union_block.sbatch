#!/bin/bash
# Addendum G T3, Canto-L (ttbar+G) seed S: block K of N of the learned-support union graphs of one split
# (hnl.eval.addendum_g_t3 union-block). Usage: sbatch union_block.sbatch S train|val K N
#SBATCH -J g3_union_block
#SBATCH -p submit-gpu-express,submit-gpu,submit
#SBATCH --exclude=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 24
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1 OMP_NUM_THREADS=1
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $*"
python -m hnl.eval.addendum_g_t3 union-block --seed "$1" --split "$2" --block "$3" --n-blocks "$4" --workers "${WORKERS:-14}"
