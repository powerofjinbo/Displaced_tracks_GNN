#!/bin/bash
# Final ONE-TIME test chain (login node, light: waits and submits), started only after the selection
# (results/final_selection_v1.json, hnl.eval.final_select select) is committed. Per pipeline of the selection and seed,
# once the matching baseline's one-time Stage-0 test evaluation exists (test_eval.done): the union test graphs (union
# pipelines; slurm/stage1/union_graphs.sbatch RUN test, no truth-derived numbers), the run's GNN scores on its test
# graphs (A30), then slurm/stage1/final_test.sbatch; per seed the acorn arm (i) dD counts on test
# (slurm/stage1/edge_dd_test.sbatch). When all are done: the claims spec of the R8 candidate and hnl.eval.claims.
# Usage: setsid nohup bash slurm/stage1/final_test_chain.sh SELECTION_JSON >> $HNL_LOGS/final_test_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SEL=${1:-results/final_selection_v1.json}
B=$HNL_DATA/runs/baseline_v1
OUT=$HNL_DATA/runs/final_test_v1
GPU_EXCL=submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
log() { echo "$(date -u +%FT%TZ) $*"; }
git ls-files --error-unmatch "$SEL" > /dev/null 2>&1 && git diff --quiet HEAD -- "$SEL" \
    || { log "$SEL is not committed as is: stopping"; exit 1; }
WP=$(python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['working_points'])" "$SEL")
ITEMS=$(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))
for p, d in s['pipelines'].items():
    if d.get('family'):
        for seed in sorted(d['seeds']):
            print(f\"{p}:{seed}:{d['graphs']}:{d['seeds'][seed]['run']}\")" "$SEL")
SEEDS=$(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))
print(' '.join(sorted({k for d in s['pipelines'].values() for k in d['seeds']})))" "$SEL")
log "final test chain start pid $$; selection $SEL ($(sha256sum "$SEL" | cut -c1-16)) at $(git rev-parse --short HEAD); working points $WP"
mkdir -p "$OUT"
declare -A UJ
todo="$ITEMS"; dd_todo="$SEEDS"; JOBS=""
while [ -n "$todo$dd_todo" ]; do
    left=""
    for it in $todo; do
        IFS=: read -r P S G RUN <<< "$it"
        BR=$B/armi_s$S
        if [ ! -e "$BR/test_eval.done" ]; then left="$left $it"; continue; fi
        if [ -e "$RUN/final_test.done" ]; then log "$P s$S: already evaluated on test"; continue; fi
        dep=""
        if [ "$G" != ml_graphs ]; then
            # union test graphs built exactly as the run's train / val union graphs (filter settings of report_val)
            if [ ! -e "$BR/${G}_test.done" ] && [ -z "${UJ[$G$S]:-}" ]; then
                fenv=$(python -I -c "
import json, sys
m = json.load(open(sys.argv[1]))
if m.get('triplet_model'):
    print(f\"TRIPLET_MODEL={m['triplet_model']} TAU={m['tau']} KTOP={m['k']}\")" "$BR/$G/report_val/build.json") \
                    || { log "no build record $BR/$G/report_val/build.json: stopping"; exit 1; }
                UJ[$G$S]=$(env SUBDIR=$G $fenv sbatch --parsable -J "s1_${G}_test_s$S" slurm/stage1/union_graphs.sbatch "$BR" test)
                log "$G test graphs seed $S ($fenv) -> ${UJ[$G$S]}"
            fi
            [ -n "${UJ[$G$S]:-}" ] && dep="--dependency=afterok:${UJ[$G$S]}"
            GD=$BR/$G; LIKE=""
        else
            GD=$BR/gnn_scores_test; LIKE=$BR/ml_graphs/valset.tar
        fi
        j1=$(MLFORM_LIKE=$LIKE ACCEL=gpu sbatch --parsable $dep -p submit-gpu --constraint=nvidia_a30 --gres=gpu:1 \
             -c 8 --mem=48G --exclude=$GPU_EXCL -J "s1_score_test_${P}_s$S" slurm/stage1/score_split.sbatch "$RUN" "$GD" test)
        j2=$(sbatch --parsable --dependency=afterok:$j1 -J "s1_final_test_${P}_s$S" slurm/stage1/final_test.sbatch "$SEL" "$P" "$S")
        log "$P s$S: test scores $j1 ($GD), final test $j2"
        JOBS="$JOBS $j2"
    done
    dleft=""
    for S in $dd_todo; do
        BR=$B/armi_s$S
        if [ ! -e "$BR/test_eval.done" ]; then dleft="$dleft $S"; continue; fi
        if [ -e "$BR/edge_tpr_test_events.parquet" ]; then continue; fi
        j=$(sbatch --parsable -J "s1_edge_dd_test_i_s$S" slurm/stage1/edge_dd_test.sbatch "$BR" "$WP")
        log "acorn arm (i) seed $S dD counts on test -> $j"
        JOBS="$JOBS $j"
    done
    todo=$(echo $left); dd_todo=$(echo $dleft)
    [ -n "$todo$dd_todo" ] && sleep 300
done
for j in $JOBS; do
    s=$(POLL=120 bash slurm/wait_job.sh "$j")
    [ "$s" = COMPLETED ] || log "job $j ended $s (see its log)"
done
python -m hnl.eval.final_select spec --selection "$SEL" --baselines "$B" --out "$OUT/claims_spec.json" > /dev/null \
    || { log "claims spec failed: stopping"; exit 1; }
J=$(sbatch --parsable -c 4 --mem=32G -t 04:00:00 -J final_claims slurm/run_py.sbatch hnl.eval.claims \
    --spec "$OUT/claims_spec.json" --out "$OUT/claims")
log "claims job $J"
s=$(POLL=60 bash slurm/wait_job.sh "$J")
log "claims $J ended $s; final test chain done"
