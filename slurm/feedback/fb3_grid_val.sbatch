#!/bin/bash
# Feedback round 3, task A2 (val only): one grid-extension part of an acorn baseline run, i.e. hnl.eval.trackbuild_val's
# evaluation (hnl.eval.strong_baselines_v3 grid) of an explicit list of configurations on the run's stored val scores
# (RUN/gnn_scores/valset.tar). The list comes from the step's plan (hnl.eval.strong_baselines_v3 inventory --step K,
# ext-jobs --step K); learned-merging configurations need the run's merge model.
# Usage: env -u OMP_NUM_THREADS [WORKERS=N] sbatch -J fb3_ext<K><tag>_arm<A>_s<S> slurm/feedback/fb3_grid_val.sbatch RUN_DIR OUTNAME CONFIG_FILE [MERGE_MODEL]
#   WORKERS (default: the CPUs): fewer worker processes than CPUs leave more memory per worker (acorn's CCandWalk on
#   ttbar events needs more than the 2 GB per CPU of the partition)
#   writes RUN_DIR/OUTNAME (a new sub-directory, refused if it exists) with provenance.json
#SBATCH -J fb3_grid_val
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # one thread per worker process (outputs identical; decisions 12:00 UTC)
RUN=$1; OUTNAME=$2; CFG=$3; MM=${4:-}
[ -e "$RUN/$OUTNAME/per_particle.parquet" ] && { echo "$RUN/$OUTNAME exists: refusing to overwrite"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; out $OUTNAME; configs $CFG ($(tr ';' '\n' < "$CFG" | wc -l)); merge model ${MM:-none}; $(date -u +%FT%TZ)"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s"
echo "$(ls "$LOCAL/s" | wc -l) val graphs"
python -m hnl.eval.strong_baselines_v3 grid --graphs "$LOCAL/s" --out "$RUN/$OUTNAME" --configs "@$CFG" ${MM:+--merge-model "$MM"} \
    --workers "${WORKERS:-$SLURM_CPUS_PER_TASK}"
python -I -c "
import hashlib, json, sys, datetime
json.dump({'run': sys.argv[1], 'grid': sys.argv[2], 'configs_file': sys.argv[3],
           'configs_sha256': hashlib.sha256(open(sys.argv[3], 'rb').read()).hexdigest(), 'merge_model': sys.argv[4] or None,
           'slurm_job': sys.argv[5], 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'feedback round 3 task A2, grid extension on val (hnl.eval.strong_baselines_v3)'},
          open(sys.argv[6], 'w'), indent=1)
" "$RUN" "$OUTNAME" "$CFG" "$MM" "$SLURM_JOB_ID" "$RUN/$OUTNAME/provenance.json"
echo "done $(date -u +%FT%TZ)"
