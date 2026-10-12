#!/bin/bash
# Brief §5b: per-particle helix fits of every particle's own true hits (fixed-R and free-R perigee fits, d0, z0,
# HNL two-track vertices) for all samples, field sign -1 (sigma = -sign(q), fixed on W muons 2026-10-03).
#SBATCH -J fit_truth
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
OUT=$HNL_DATA/derived/truth_fits/${TAG:-v1}
if [ -e "$OUT/meta.json" ]; then echo "$OUT holds a finished fit; refusing to overwrite"; exit 1; fi
mkdir -p "$OUT"
echo "git $(git rev-parse HEAD) $(git status --porcelain | wc -l) uncommitted; out $OUT; host $(hostname)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$HNL_ENV/venv/bin/python" -I -c \
    "import sys; sys.path.insert(0, '$HNL_REPO/src'); from hnl.truth.fit_events import main; main(['--out', '$OUT', '--field-sign', '-1', '--workers', '$SLURM_CPUS_PER_TASK'])"
