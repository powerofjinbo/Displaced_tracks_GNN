#!/bin/bash
# Feedback round 1 (strongest baselines, post-freeze descriptive): track building on an acorn baseline run's STORED test
# scores (RUN/gnn_scores_test/testset.tar, from the one-time test chain) at configurations fixed on val
# (hnl.eval.strong_baselines spec), one hnl.eval.trackbuild_val --config per configuration; nothing is chosen on test.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/baseline_test.sbatch RUN_DIR OUT_ROOT NAME=FAMILY,P1,P2,... [...]
#   SCORES (env, default gnn_scores_test/testset.tar): the scored test graphs under RUN_DIR (a Canto run: gnn_scores/testset.tar)
#   writes OUT_ROOT/NAME (per_particle / per_event / configs.json) and OUT_ROOT/NAME/provenance.json
#SBATCH -J fb_baseline_test
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; OUTROOT=$2; shift 2
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/t" "$OUTROOT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; $(date -u +%FT%TZ)"
SCORES=${SCORES:-gnn_scores_test/testset.tar}
tar -xf "$RUN/$SCORES" -C "$LOCAL/t"
echo "$(ls "$LOCAL/t" | wc -l) test graphs"
for item in "$@"; do
    NAME=${item%%=*}; CFG=${item#*=}
    OUT=$OUTROOT/$NAME
    [ -e "$OUT/provenance.json" ] && { echo "$OUT exists: skipping"; continue; }
    echo "$NAME: $CFG $(date -u +%FT%TZ)"
    python -m hnl.eval.trackbuild_val --graphs "$LOCAL/t" --out "$OUT" --config "$CFG"
    python -I -c "
import json, sys, datetime
json.dump({'run': sys.argv[1], 'config': sys.argv[2], 'scores': sys.argv[1] + '/' + sys.argv[7],
           'git': sys.argv[3], 'slurm_job': sys.argv[4], 'host': sys.argv[5],
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'post-freeze descriptive; configuration fixed on val (hnl.eval.strong_baselines spec)'},
          open(sys.argv[6], 'w'), indent=1)" "$RUN" "$CFG" "$(git rev-parse HEAD)" "$SLURM_JOB_ID" "$(hostname)" "$OUT/provenance.json" "$SCORES"
done
echo "done $(date -u +%FT%TZ)"
