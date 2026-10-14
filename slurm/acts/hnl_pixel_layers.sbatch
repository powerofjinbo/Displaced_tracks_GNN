#!/bin/bash
# Barrel-pixel layers of the N daughters (scripts/acts/hnl_pixel_layers.py), Zenodo train+val and our pilots; truth only.
#   sbatch --array=0-3 --export=ALL,OUT=DIR slurm/acts/hnl_pixel_layers.sbatch
#SBATCH -J hnl_pixlay
#SBATCH -p submit-gpu-express,submit
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:59:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
: "${OUT:?}"
REPO=/work/submit/jinboz1/hnl/hnl-displaced
PI=/work/submit/jinboz1/hnl/samples_v2/hnl_points/pilot
source $REPO/env/activate.sh > /dev/null 2>&1
S=$REPO/scripts/acts/hnl_pixel_layers.py
case ${SLURM_ARRAY_TASK_ID} in
  0) python3 $S zenodo --sample hnl10_100 --splits train,val --out $OUT/zenodo_hnl10_100.parquet ;;
  1) python3 $S zenodo --sample hnl15_200 --splits train,val --out $OUT/zenodo_hnl15_200.parquet ;;
  2) python3 $S shards --dir $PI/F_hnl10_100 --out $OUT/F_hnl10_100.parquet ;;
  3) python3 $S shards --dir $PI/F_hnl15_200 --out $OUT/F_hnl15_200.parquet ;;
esac
