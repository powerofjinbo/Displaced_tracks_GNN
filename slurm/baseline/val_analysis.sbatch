#!/bin/bash
# Stage-0 val analysis (CPU): working points (s*, FR*, reference arm, per-seed cuts, e*), the 5g headroom report and
# the M3 (val) / M4 figures. Usage: [OUT=dir FIGS=0 ALLOW_MISSING=1] sbatch slurm/baseline/val_analysis.sbatch [ARMS] [SEEDS]
# (ALLOW_MISSING only for previews: the preregistered working points need every arm and seed)
# ACCEPT_UNREACHABLE=ARM:SEED[,..]: reference-arm seeds whose unreachable FR* was reviewed by hand (DECISIONS.md)
#SBATCH -J bl_val_analysis
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
ARMS=${1:-i,ii,iii}
SEEDS=${2:-0,1,2}
BASE=$HNL_DATA/runs/baseline_${TAG:-v1}
OUT=${OUT:-$BASE/analysis_val}
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); job $SLURM_JOB_ID; arms $ARMS seeds $SEEDS"
if [ "${ALLOW_MISSING:-0}" = 1 ]; then ALLOW=--allow-missing; else ALLOW=; fi
python -m hnl.eval.baseline_val --runs "$BASE" --out "$OUT" --arms "$ARMS" --seeds "$SEEDS" ${ALLOW:+"$ALLOW"} \
    ${ACCEPT_UNREACHABLE:+--accept-unreachable "$ACCEPT_UNREACHABLE"}
python -m hnl.eval.headroom --runs "$BASE" --wp "$OUT/working_points.json" --out "$OUT/headroom"
if [ "${FIGS:-1}" = 1 ]; then
    python -m hnl.plots.m3 --val "$OUT"
    python -m hnl.plots.m4 --headroom "$OUT/headroom"
fi
