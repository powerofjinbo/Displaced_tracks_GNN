#!/bin/bash
# Brief §2.6: format checks, corrupted-event scan (hard ranges + per-module envelopes) and event-overlap scan over
# all 35,000 events. Output: $HNL_DATA/derived/scans/<tag>/{event_scan,module_envelopes,overlap_pairs}.parquet + meta.
# ~55M hits held in memory (~6 GB peak expected); MaxMemPerCPU = 2 GB, hence -c 8 for 16 GB.
#SBATCH -J scan_events
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
cd "$HNL_REPO"
TAG=${TAG:-v1}
OUT=$HNL_DATA/derived/scans/$TAG
if [ -e "$OUT/meta.json" ]; then echo "$OUT already holds a finished scan; refusing to overwrite"; exit 1; fi
echo "git $(git rev-parse HEAD) $(git status --porcelain | wc -l) uncommitted; out $OUT; host $(hostname)"
PYTHONPATH=$HNL_REPO/src OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK /usr/bin/time -v \
    "$HNL_ENV/venv/bin/python" -I -c "import sys; sys.path.insert(0, '$HNL_REPO/src'); from hnl.data.scans import main; main(['--out', '$OUT'])" \
    2> >(tee "$OUT.time.txt" >&2)
