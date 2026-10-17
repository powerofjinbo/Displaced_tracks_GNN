#!/bin/bash
# Addendum G T3: merge the grid blocks of one run (hnl.eval.addendum_g_t3 grid-merge). Usage: sbatch gridmerge.sbatch acorn|canto S
#SBATCH -J g3_gridmerge
#SBATCH -p submit-gpu-express,submit-gpu,submit
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
python -m hnl.eval.addendum_g_t3 grid-merge --model "$1" --seed "$2"
