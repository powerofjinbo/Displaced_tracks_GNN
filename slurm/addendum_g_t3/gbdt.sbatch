#!/bin/bash
# Addendum G T3, Canto-L (ttbar+G) seed S: learned triplet support (graph booster v3) on arm g's metric-learning
# graphs, as slurm/stage1/learned_triplet_filter.sbatch with TAG=v2: triplet data from 400 training graphs drawn at
# random (rng 0), the unweighted GBDT. The operating point is fixed (k 5, tau 0.005, r_w 0.24;
# docs/training_sample_study.md section 4.1), so the diagnostic val scan of that script is not run.
# G = runs/baseline_g1_ts/armg_s<S>, with ml and ml_graphs symlinked to the acorn (ttbar+G) run.
# Usage: sbatch slurm/addendum_g_t3/gbdt.sbatch S
#SBATCH -J g3_gbdt
#SBATCH -p submit-gpu-express,submit-gpu,submit
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
S=$1
G=$HNL_DATA/runs/baseline_g1_ts/armg_s$S
DATA=$G/learned_triplet_filter/train_triplets.npz
OUT=$G/learned_triplet_filter_v2
mkdir -p "$OUT" "$(dirname "$DATA")"
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $G"
[ -e "$DATA" ] || OMP_NUM_THREADS=1 python -m hnl.graphs.learned_triplet_filter data --ml-run "$G" --split train \
    --out "$DATA" --max-events 400 --workers "${WORKERS:-8}"
[ -e "$OUT/model.joblib" ] || OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.graphs.learned_triplet_filter train \
    --data "$DATA" --out "$OUT/model.joblib"
touch "$OUT.done"
