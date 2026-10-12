#!/bin/bash
# Brief §2.3: validate the HNL-daughter and W-muon barcode rules against column 12 on every labeled HNL event.
#SBATCH -J validate_labels
#SBATCH -p submit
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
OUT=$HNL_DATA/derived/labels/${TAG:-v1}
mkdir -p "$OUT"
echo "git $(git rev-parse HEAD) $(git status --porcelain | wc -l) uncommitted; out $OUT; host $(hostname)"
"$HNL_ENV/venv/bin/python" -I -c "import sys; sys.path.insert(0, '$HNL_REPO/src'); from hnl.truth.validate_labels import main; main(['--out', '$OUT'])"
