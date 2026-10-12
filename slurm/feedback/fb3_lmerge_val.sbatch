#!/bin/bash
# Feedback round 3, task A2 (val): the val grid of the learned-merging family (hnl.eval.trackbuild_val mwslmerge,
# tau 0.005-0.5) on an acorn run's stored val scores with the run's own merge model, for a run whose
# fb3_lmerge.sbatch job stopped after the model was written (provenance.json present) but before the grid; the same
# command and environment as the last step of fb3_lmerge.sbatch.
# Usage: env -u OMP_NUM_THREADS sbatch -J fb3_lmval_arm<A>_s<S> slurm/feedback/fb3_lmerge_val.sbatch RUN_DIR OUT_DIR [GRID]
#SBATCH -J fb3_lmerge_val
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; OUT=$2; GRID=${3:-trackbuild_val_fb3_lmerge}
[ -e "$RUN/$GRID/per_particle.parquet" ] && { echo "$RUN/$GRID exists: refusing to overwrite"; exit 1; }
[ -e "$OUT/provenance.json" ] || { echo "$OUT has no finished merge model (provenance.json)"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/val"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; model $OUT/model.joblib; grid $GRID; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # one thread per worker process (outputs identical; decisions 12:00 UTC)
python -I -c "
import hashlib, json, sys
p = json.load(open(sys.argv[1] + '/provenance.json'))
h = hashlib.sha256(open(sys.argv[1] + '/model.joblib', 'rb').read()).hexdigest()
sys.exit(0 if h == p['model_sha256'] else f'merge model checksum mismatch: {h} != {p[\"model_sha256\"]}')" "$OUT"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
echo "$(ls "$LOCAL/val" | wc -l) val graphs"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/$GRID" --builders mwslmerge \
    --merge-model "$OUT/model.joblib"
python -I -c "
import json, sys, datetime
json.dump({'run': sys.argv[1], 'grid': sys.argv[2], 'merge_model': sys.argv[3], 'builders': 'mwslmerge',
           'slurm_job': sys.argv[4], 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'feedback round 3 task A2 (val only); hnl.eval.strong_baselines_v3 (fb3_lmerge_val.sbatch)'},
          open(sys.argv[5], 'w'), indent=1)
" "$RUN" "$GRID" "$OUT/model.joblib" "$SLURM_JOB_ID" "$RUN/$GRID/provenance.json"
echo "done $(date -u +%FT%TZ)"
