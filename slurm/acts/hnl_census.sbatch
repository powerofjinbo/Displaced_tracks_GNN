#!/bin/bash
# Truth census for the HNL closure (scripts/acts/hnl_census.py; no model is run). CPU only.
#   sbatch slurm/acts/hnl_census.sbatch zenodo --sample hnl10_100 --splits train,val --out OUT
#   sbatch slurm/acts/hnl_census.sbatch shards --dir DIR --out OUT
#SBATCH -J hnl_census
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
REPO=/work/submit/jinboz1/hnl/hnl-displaced
source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
t0=$(date +%s)
python3 $REPO/scripts/acts/hnl_census.py "$@"
echo "census done in $(( $(date +%s) - t0 )) s"
