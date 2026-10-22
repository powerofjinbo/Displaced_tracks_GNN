#!/bin/bash
# Ledger R17 (docs/feedback/test_use_ledger.md): efficiency of the HNL daughters in the |d0| and r_v bins on the pooled
# HNL test sets for the four models of paper Fig. 5 (hnl.eval.test_binned), from stored outputs; post-freeze,
# descriptive: no network run, no track building, nothing chosen on test, no claim. Outputs results/<NAME>.csv and .md
# (the module refuses to overwrite them).
# Usage: env -u OMP_NUM_THREADS -u NUMBA_NUM_THREADS sbatch slurm/feedback/test_binned.sbatch [NAME]
#SBATCH -J test_binned
#SBATCH -p submit
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
NAME=${1:-test_binned_v1}
echo "git $(git rev-parse HEAD); changed or new code files: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
sha256sum src/hnl/eval/test_binned.py slurm/feedback/test_binned.sbatch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python -I -m hnl.eval.test_binned --split test --out results --name "$NAME" --sbatch slurm/feedback/test_binned.sbatch
echo "done $(date -u +%FT%TZ)"
