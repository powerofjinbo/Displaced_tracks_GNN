#!/bin/bash
# HelixFilter union graphs of a baseline run for training (CPU): the run's metric-learning graphs of the listed splits
# united with the support-filtered wider-radius candidates of its own embedding (hnl.graphs.helix_union, hand-cut
# filter, ttbar-train resolutions), packed like ml_graphs/ (RUN/union_graphs/<split>set.tar) so that
# slurm/stage1/variant.sbatch can train on them (GRAPHS_SUBDIR=union_graphs). The test split (only after the run's
# one-time Stage-0 test evaluation) is built from RUN/gnn_scores_test/testset.tar without any truth-derived number.
# Learned triplet support instead of the hand cuts (hnl.graphs.learned_triplet_filter): TRIPLET_MODEL=MODEL.joblib
# TAU=t KTOP=k, written to RUN/$SUBDIR (default union_graphs; e.g. SUBDIR=ltf_union_graphs).
# Usage: [TRIPLET_MODEL=.. TAU=.. KTOP=.. SUBDIR=..] sbatch slurm/stage1/union_graphs.sbatch RUN_DIR "train val" | RUN_DIR test
#SBATCH -J s1_union_graphs
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
SPLITS=${2:-train val}
SUB=${SUBDIR:-union_graphs}
FILTER=()
if [ -n "${TRIPLET_MODEL:-}" ]; then
    [ -n "${TAU:-}" ] && [ -n "${KTOP:-}" ] || { echo "TRIPLET_MODEL needs TAU and KTOP"; exit 1; }
    [ "$SUB" != union_graphs ] || { echo "learned-support graphs need their own SUBDIR"; exit 1; }
    FILTER=(--triplet-model "$TRIPLET_MODEL" --tau "$TAU" --k "$KTOP")
fi
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/u" "$RUN/$SUB"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; splits $SPLITS; out $RUN/$SUB; filter ${FILTER[*]:-hand cuts}"
export OMP_NUM_THREADS=1
for SP in $SPLITS; do
    if [ "$SP" = test ]; then
        [ -e "$RUN/test_eval.done" ] || { echo "no one-time Stage-0 test evaluation in $RUN yet"; exit 1; }
        python -m hnl.graphs.helix_union build --ml-run "$RUN" --split test --tar "$RUN/gnn_scores_test/testset.tar" \
            --no-metrics --out "$LOCAL/u" --report "$RUN/$SUB/report_test" "${FILTER[@]}"
    else
        python -m hnl.graphs.helix_union build --ml-run "$RUN" --split "$SP" --out "$LOCAL/u" --report "$RUN/$SUB/report_$SP" \
            "${FILTER[@]}"
    fi
done
python -m hnl.train.acorn_runner pack --src "$LOCAL/u" --dest "$RUN/$SUB"
ls -la "$RUN/$SUB"
if [ "$SPLITS" = test ]; then touch "$RUN/${SUB}_test.done"; else touch "$RUN/$SUB.done"; fi
