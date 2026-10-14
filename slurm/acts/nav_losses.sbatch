#!/bin/bash
# Truth-only check of Fatras navigation losses (hnl.truth.nav_losses; results/generated_nav_losses_v1.md), CPU only,
# no model and no model output. Generated samples: train and val splits only; Zenodo ttbar: train split.
#   sbatch --export=ALL,SAMPLE=generic_iso|generic_dense|ttbar slurm/acts/nav_losses.sbatch    (one scan)
#   sbatch --export=ALL,SAMPLE=summary,TABLES=<md> slurm/acts/nav_losses.sbatch               (json + tables)
#SBATCH -J nav_losses
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:50:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${SAMPLE:?}"
cd /work/submit/jinboz1/hnl/hnl-displaced
source env/activate.sh > /dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
echo "host $(hostname) start $(date -u) git $(git rev-parse --short HEAD) sample $SAMPLE"
if [ "$SAMPLE" = summary ]; then
  python -u -m hnl.truth.nav_losses_report ${TABLES:+--tables $TABLES}
else
  python -u -m hnl.truth.nav_losses scan --sample "$SAMPLE" --workers "${SLURM_CPUS_PER_TASK:-8}"
fi
echo "end $(date -u)"
