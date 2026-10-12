#!/bin/bash
# Feedback round 2, task E (acorn (ttbar+HNL), post-freeze descriptive): track building on an acorn baseline run's STORED
# test scores (RUN/gnn_scores_test/testset.tar, from the one-time test chain) at one configuration fixed on val
# (results/strong_baselines_v2_armiii_spec_test.json, hnl.eval.strong_baselines_armiii val), one
# hnl.eval.trackbuild_val --config per configuration; nothing is chosen on test. Same steps as baseline_test.sbatch of
# round 1; the provenance also records the spec file and its sha256 at run time (env SPEC, default the final spec;
# rows complete before the last val grid finished run from the first-stage spec of `val --skip-missing`).
# Usage: env -u OMP_NUM_THREADS sbatch [-c N --mem=XG] slurm/feedback/armiii_test.sbatch RUN_DIR OUT_ROOT NAME=FAMILY,P1,...
#   writes OUT_ROOT/NAME (per_particle / per_event / configs.json) and OUT_ROOT/NAME/provenance.json
#SBATCH -J fb2E_test
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; OUTROOT=$2; shift 2
SPEC=${SPEC:-$HNL_REPO/results/strong_baselines_v2_armiii_spec_test.json}   # env SPEC: the val-fixed spec of this job
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/t" "$OUTROOT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; $(date -u +%FT%TZ)"
for item in "$@"; do
    CFG=${item#*=}
    python -I - "$SPEC" "$RUN" "$CFG" <<'EOF'
import json, sys
spec, run, cfg = sys.argv[1:]
ok = any(e.get("config") == cfg and e["run"] == run for m in json.load(open(spec))["models"] for e in m["entries"])
sys.exit(0 if ok else f"{cfg} for {run} is not in the val-fixed spec {spec}")
EOF
done
SCORES=gnn_scores_test/testset.tar
tar -xf "$RUN/$SCORES" -C "$LOCAL/t"
echo "$(ls "$LOCAL/t" | wc -l) test graphs"
for item in "$@"; do
    NAME=${item%%=*}; CFG=${item#*=}
    OUT=$OUTROOT/$NAME
    [ -e "$OUT/provenance.json" ] && { echo "$OUT exists: skipping"; continue; }
    echo "$NAME: $CFG $(date -u +%FT%TZ)"
    python -m hnl.eval.trackbuild_val --graphs "$LOCAL/t" --out "$OUT" --config "$CFG"
    python -I -c "
import hashlib, json, sys, datetime
json.dump({'run': sys.argv[1], 'config': sys.argv[2], 'scores': sys.argv[1] + '/' + sys.argv[7],
           'git': sys.argv[3], 'slurm_job': sys.argv[4], 'host': sys.argv[5],
           'spec': sys.argv[8], 'spec_sha256': hashlib.sha256(open(sys.argv[8], 'rb').read()).hexdigest(),
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'post-freeze descriptive; configuration fixed on val (hnl.eval.strong_baselines_armiii spec)'},
          open(sys.argv[6], 'w'), indent=1)" "$RUN" "$CFG" "$(git rev-parse HEAD)" "$SLURM_JOB_ID" "$(hostname)" "$OUT/provenance.json" "$SCORES" "$SPEC"
done
echo "done $(date -u +%FT%TZ)"
