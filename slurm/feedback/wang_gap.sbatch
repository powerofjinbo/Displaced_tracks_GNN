#!/bin/bash
# Wang et al.'s evaluation sample, post-freeze (ledger R12): `cc` = CC of the Canto-L classifiers on their stored scored
# test graphs at the validation cuts (once); `summary` = the stage decomposition and the Fig. 3 displaced muons
# (hnl.eval.wang_gap). Usage: sbatch -p submit-gpu-express slurm/feedback/wang_gap.sbatch cc|summary
#SBATCH -J wang_gap
#SBATCH --exclude=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); dirty: $(git status --porcelain -- src/hnl/eval/wang_gap.py slurm/feedback/wang_gap.sbatch | wc -l); host $(hostname); job $SLURM_JOB_ID; step $1"
case "$1" in
  cc) python -m hnl.eval.wang_gap cc ;;
  summary) python -m hnl.eval.wang_gap summary --out results ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
