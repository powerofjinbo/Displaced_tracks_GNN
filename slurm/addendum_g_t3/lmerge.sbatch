#!/bin/bash
# Addendum G T3: learned fragment merging model of one run (acorn (ttbar+G) or Canto-L (ttbar+G)), from that run's GNN
# scores of 2000 training graphs drawn at random (slurm/stage1/score_split_sample.sbatch, rng 0), as
# slurm/stage1/learned_merge.sbatch without its val grid (hnl.eval.addendum_g_t3 grid evaluates the menus).
# Usage: sbatch slurm/addendum_g_t3/lmerge.sbatch RUN_DIR
#SBATCH -J g3_lmerge
#SBATCH -p submit-gpu-express,submit-gpu,submit
#SBATCH --exclude=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 24
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
RUN=$1
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/train" "$RUN/lmerge"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN"
tar -xf "$RUN/gnn_scores/trainset.tar" -C "$LOCAL/train"
echo "scored training graphs: $(ls "$LOCAL/train" | wc -l); per sample code: $(ls "$LOCAL/train" | cut -c6 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
python -m hnl.trackbuild.learned_merge data --graphs "$LOCAL/train" --out "$RUN/lmerge/pairs_train.npz"
python -m hnl.trackbuild.learned_merge train --data "$RUN/lmerge/pairs_train.npz" --out "$RUN/lmerge/model.joblib"
touch "$RUN/lmerge.done"
