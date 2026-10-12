#!/bin/bash
# S6a (plan sections 6.5-6.6): learned fragment merging retrained on TrackML: fragment pairs after the same-layer MWS
# (s0 0.2) on the classifier scores of the merge_train events (the 200 smallest training ids outside the calibration and
# GBDT sets; scored by score.sbatch merge_train), module azimuths from detectors.csv (TrackML numbering), the exact fast
# candidate search; then the unchanged hnl.trackbuild.learned_merge training (holdout AUC) -> $RUN/lmerge/model.joblib.
# Usage: [SEED=0] sbatch slurm/trackml/lmerge_train.sbatch
#SBATCH -J tml_lmerge
#SBATCH -p submit
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
S=$(scores_dir merge_train)
require "$S.done"
mkdir -p "$RUN/lmerge"
python -m hnl.trackml.trackbuild lmerge-data --scores "$S" --events merge_train \
    --out "$RUN/lmerge/pairs_train.npz" --workers "$(nproc)"
OMP_NUM_THREADS=$(nproc) python -m hnl.trackbuild.learned_merge train --data "$RUN/lmerge/pairs_train.npz" \
    --out "$RUN/lmerge/model.joblib" --seed "$SEED"
touch "$RUN/lmerge_train.done"
