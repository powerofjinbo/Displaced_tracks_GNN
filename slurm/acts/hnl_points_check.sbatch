#!/bin/bash
# Truth-only checks of the produced HNL points (scripts/acts/hnl_points_check.py; docs/hnl_points_spec.md section 5).
#   sbatch slurm/acts/hnl_points_check.sbatch scan --sample hnl5_50 --out DIR --workers 8
#   sbatch slurm/acts/hnl_points_check.sbatch swapref --zenodo hnl10_100 --split val --out DIR --workers 8
#SBATCH -J hnl_pcheck
#SBATCH -p submit-gpu-express,submit
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:59:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
REPO=/work/submit/jinboz1/hnl/hnl-displaced
source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
t0=$(date +%s)
python3 $REPO/scripts/acts/hnl_points_check.py "$@"
echo "check done in $(( $(date +%s) - t0 )) s"
