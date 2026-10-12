#!/bin/bash
# Brief §5b resolution estimate 2: split-track d0 consistency on real hits (first 2000 events of hnl10_100 and ttbar).
#SBATCH -J split_tracks
#SBATCH -p submit
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
OUT=$HNL_DATA/derived/truth_fits/split_${TAG:-v1}
mkdir -p "$OUT"
echo "git $(git rev-parse HEAD) $(git status --porcelain | wc -l) uncommitted; out $OUT; host $(hostname)"
OMP_NUM_THREADS=2 "$HNL_ENV/venv/bin/python" -I -c "import sys; sys.path.insert(0, '$HNL_REPO/src'); from hnl.truth.split_tracks import main; main(['--out', '$OUT', '--events', '2000'])"
