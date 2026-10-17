#!/bin/bash
# Addendum G T3, Canto-L (ttbar+G) seed S: the union blocks of one split into ltf_union_graphs/<split>set.tar
# (hnl.eval.addendum_g_t3 union-gather). Usage: sbatch union_gather.sbatch S train|val
#SBATCH -J g3_union_gather
#SBATCH -p submit-gpu-express,submit-gpu,submit
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
python -m hnl.eval.addendum_g_t3 union-gather --seed "$1" --split "$2"
