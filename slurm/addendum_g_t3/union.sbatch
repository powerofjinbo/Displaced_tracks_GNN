#!/bin/bash
# Addendum G T3, Canto-L (ttbar+G) seed S: learned-support union graphs of one split (train or val_all) of arm g, as
# slurm/stage1/union_graphs.sbatch with TRIPLET_MODEL=G/learned_triplet_filter_v2/model.joblib TAU=0.005 KTOP=5 and
# SUBDIR=ltf_union_graphs; --no-metrics (the graph-level ceilings of that script read HNL truth tables; the graphs
# themselves are the same). Packed to G/ltf_union_graphs/<split>set.tar. Usage: sbatch union.sbatch S train|val
#SBATCH -J g3_union
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 28
#SBATCH --mem=56G
#SBATCH -t 1-00:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export HNL_GENERATED=1
S=$1; SP=$2
G=$HNL_DATA/runs/baseline_g1_ts/armg_s$S
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/u" "$G/ltf_union_graphs"
trap 'rm -rf "$LOCAL"' EXIT
[ -e "$G/ltf_union_graphs_$SP.done" ] && { echo "done already"; exit 0; }
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $G; split $SP"
export OMP_NUM_THREADS=1
python -m hnl.graphs.helix_union build --ml-run "$G" --split "$SP" --out "$LOCAL/u" --report "$G/ltf_union_graphs/report_$SP" \
    --triplet-model "$G/learned_triplet_filter_v2/model.joblib" --tau 0.005 --k 5 --no-metrics --workers "${WORKERS:-18}"
python -m hnl.train.acorn_runner pack --src "$LOCAL/u" --dest "$G/ltf_union_graphs"
ls -la "$G/ltf_union_graphs"
touch "$G/ltf_union_graphs_$SP.done"
