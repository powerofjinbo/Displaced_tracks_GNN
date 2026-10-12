#!/bin/bash
# Primary efficiency per sample, validation and test read out separately (hnl.eval.per_sample): stored val and test
# outputs at committed per-seed working points; the test part is a post-freeze descriptive read-out, nothing is chosen
# on test. Outputs results/<NAME>.csv, _differences.csv, _changes.csv, _checks.csv, .md, _meta.json.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/per_sample.sbatch CC_REF [NAME]
#   CC_REF: reference read-out of the Canto-L classifier + CC val entries (CSV, on /work)
#SBATCH -J per_sample
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
CC_REF=$1; NAME=${2:-per_sample_v1}
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python -I -m hnl.eval.per_sample --out results --name "$NAME" --cc-ref "$CC_REF"
echo "done $(date -u +%FT%TZ)"
