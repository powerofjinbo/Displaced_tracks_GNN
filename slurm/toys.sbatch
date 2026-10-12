#!/bin/bash
# Brief §5b resolution estimate 1: toy helices through the generic-detector layers with residuals measured on W muons.
#SBATCH -J toys
#SBATCH -p submit
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
OUT=$HNL_DATA/derived/truth_fits/toys_${TAG:-v1}
mkdir -p "$OUT"
echo "git $(git rev-parse HEAD) $(git status --porcelain | wc -l) uncommitted; out $OUT; host $(hostname)"
OMP_NUM_THREADS=2 "$HNL_ENV/venv/bin/python" -I -c "import sys; sys.path.insert(0, '$HNL_REPO/src'); from hnl.truth.toys import main; main(['--envelopes', '$HNL_DATA/derived/scans/v1/module_envelopes.parquet', '--out', '$OUT', '--calib-events', '2000', '--n', '5000'])"
