#!/bin/bash
# Addendum G step T1 on the extension generic_iso_x1 (truth only; hnl.eval.addendum_g_t1_x1). CPU; A30 and GTX 1080
# nodes excluded.
# Usage: sbatch slurm/addendum_g/t1_x1.sbatch scan        (checksums, structure, event checks, truth table)
#        sbatch -c 4 --mem=32G slurm/addendum_g/t1_x1.sbatch summary --jobs <ids>   (splits, P_G, comparison)
#SBATCH -J g_t1x1
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 24
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
CMD=$1
shift
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $CMD $*"
if [ "$CMD" = scan ]; then
    OMP_NUM_THREADS=1 python -I -m hnl.eval.addendum_g_t1_x1 scan --workers "$SLURM_CPUS_PER_TASK" "$@"
else
    OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -I -m hnl.eval.addendum_g_t1_x1 "$CMD" "$@"
fi
