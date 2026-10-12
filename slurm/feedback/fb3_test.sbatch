#!/bin/bash
# Feedback round 3, task A2 (post-freeze descriptive): track building on an acorn baseline run's STORED test scores
# (RUN/gnn_scores_test/testset.tar, from the one-time test chain) at one configuration fixed on val
# (results/strong_baselines_v3_spec_test.json, hnl.eval.strong_baselines_v3 val), one hnl.eval.trackbuild_val --config
# per configuration; nothing is chosen on test. As armiii_test.sbatch of round 2; a learned-merging configuration uses
# the run's own merge model (env MM; its sha256 is recorded). The provenance records the spec file and its sha256.
# Usage: env -u OMP_NUM_THREADS SPEC=SPEC.json [MM=MODEL.joblib] [WORKERS=N] sbatch -J fb3_test_<key>_arm<A>_s<S> \
#            slurm/feedback/fb3_test.sbatch RUN_DIR OUT_ROOT NAME=FAMILY,P1,...
#   writes OUT_ROOT/NAME (per_particle / per_event / configs.json) and OUT_ROOT/NAME/provenance.json
#SBATCH -J fb3_test
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # one thread per worker process (outputs identical; decisions 12:00 UTC)
RUN=$1; OUTROOT=$2; ITEM=$3
SPEC=${SPEC:?SPEC (the val-fixed test spec) is required}
MM=${MM:-}
NAME=${ITEM%%=*}; CFG=${ITEM#*=}
OUT=$OUTROOT/$NAME
[ -e "$OUT/provenance.json" ] && { echo "$OUT exists: the test evaluation is done once"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/t" "$OUTROOT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; $NAME = $CFG; merge model ${MM:-none}; spec $SPEC; $(date -u +%FT%TZ)"
python -I - "$SPEC" "$RUN" "$CFG" "$OUT" "$MM" <<'EOF'
import json, os, sys
spec, run, cfg, out, mm = sys.argv[1:]
ok = [e for m in json.load(open(spec))["models"] for e in m["entries"]
      if e.get("config") == cfg and os.path.normpath(e["run"]) == os.path.normpath(run)
      and os.path.normpath(e["sub"]) == os.path.normpath(out) and (e.get("merge_model") or "") == mm]
sys.exit(0 if ok else f"{cfg} for {run} -> {out} (merge model {mm or 'none'}) is not in the val-fixed spec {spec}")
EOF
SCORES=gnn_scores_test/testset.tar
tar -xf "$RUN/$SCORES" -C "$LOCAL/t"
echo "$(ls "$LOCAL/t" | wc -l) test graphs"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/t" --out "$OUT" --config "$CFG" ${MM:+--merge-model "$MM"} \
    --workers "${WORKERS:-$SLURM_CPUS_PER_TASK}"
python -I -c "
import hashlib, json, sys, datetime
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest() if p else None
json.dump({'run': sys.argv[1], 'config': sys.argv[2], 'scores': sys.argv[1] + '/' + sys.argv[7],
           'git': sys.argv[3], 'slurm_job': sys.argv[4], 'host': sys.argv[5],
           'spec': sys.argv[8], 'spec_sha256': sha(sys.argv[8]), 'merge_model': sys.argv[9] or None,
           'merge_model_sha256': sha(sys.argv[9]),
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'post-freeze descriptive; configuration fixed on val (hnl.eval.strong_baselines_v3 spec)'},
          open(sys.argv[6], 'w'), indent=1)" "$RUN" "$CFG" "$(git rev-parse HEAD)" "$SLURM_JOB_ID" "$(hostname)" \
    "$OUT/provenance.json" "$SCORES" "$SPEC" "$MM"
echo "done $(date -u +%FT%TZ)"
