#!/bin/bash
# Per-event dD counts of a baseline on the test sets (preregistration section 6c), after its one-time test evaluation:
# edge_tpr --per-event on RUN/gnn_scores_test/testset.tar at the run's e* cut fixed on val (working points' cuts_at_e_star).
# Usage: sbatch slurm/stage1/edge_dd_test.sbatch BASELINE_RUN WORKING_POINTS_JSON    (RUN = .../arm<A>_s<S>)
#SBATCH -J s1_edge_dd_test
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; WP=$2
[ -e "$RUN/test_eval.done" ] || { echo "no one-time test evaluation in $RUN"; exit 1; }
CUT=$(python -I -c "
import json, re, sys
m = re.search(r'arm(i{1,3})_s(\d+)/?$', sys.argv[2])
print(f\"{json.load(open(sys.argv[1]))['cuts_at_e_star'][m.group(1)][m.group(2)]:.4f}\")" "$WP" "$RUN")
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/t"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); host $(hostname); job $SLURM_JOB_ID; run $RUN; e* cut $CUT ($WP)"
tar -xf "$RUN/gnn_scores_test/testset.tar" -C "$LOCAL/t"
python -m hnl.eval.edge_tpr --graphs "$LOCAL/t" --out "$RUN/edge_tpr_test.csv" --cut "$CUT" \
    --per-event "$RUN/edge_tpr_test_events.parquet"
