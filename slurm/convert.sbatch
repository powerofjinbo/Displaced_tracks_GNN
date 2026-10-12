#!/bin/bash
# Brief §5a: npz -> sharded per-event tables (hits, particles, edges), 500 events per shard, dropped events omitted.
#SBATCH -J convert
#SBATCH -p submit
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
OUT=$HNL_DATA/derived/events/${TAG:-v1}
if [ -e "$OUT/meta.json" ]; then echo "$OUT holds a finished conversion; refusing to overwrite"; exit 1; fi
echo "git $(git rev-parse HEAD) $(git status --porcelain | wc -l) uncommitted; out $OUT; host $(hostname)"
OMP_NUM_THREADS=2 "$HNL_ENV/venv/bin/python" -I -c "import sys; sys.path.insert(0, '$HNL_REPO/src'); from hnl.data.convert import main; main(['--out', '$OUT'])"
echo "files: $(find "$OUT" -type f | wc -l)"
