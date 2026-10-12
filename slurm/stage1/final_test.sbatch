#!/bin/bash
# ONE-TIME test evaluation of one seed of a final Stage-1 pipeline, only with the committed selection
# (results/final_selection_v1.json, hnl.eval.final_select): on the run's scored test graphs (RUN/gnn_scores/testset.tar,
# slurm/stage1/score_split.sbatch) the frozen FR* and LLP-blind builder configurations (hnl.eval.trackbuild_val --config,
# the run's learned merging model), the classifier at its e* cut (evaluate_gnn) and the per-event dD counts at that cut
# (edge_tpr --per-event). Refuses to run twice. Usage: sbatch slurm/stage1/final_test.sbatch SELECTION_JSON PIPELINE SEED
#SBATCH -J s1_final_test
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
SEL=$1; P=$2; S=$3
git ls-files --error-unmatch "$SEL" > /dev/null 2>&1 && git diff --quiet HEAD -- "$SEL" \
    || { echo "$SEL is not committed as is: refusing the test evaluation"; exit 1; }
read -r RUN CFG LLP CUTE MERGE < <(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))
p = s['pipelines'][sys.argv[2]]
d = p['seeds'][sys.argv[3]]
print(d['run'], d['config'], d['llp_blind_config'], f\"{d['cut_e']:.4f}\", d['merge_model'])" "$SEL" "$P" "$S")
[ -e "$RUN/final_test.done" ] && { echo "test evaluation already done for $RUN: refusing to repeat it"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/t"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID"
echo "selection $SEL ($(sha256sum "$SEL" | cut -c1-16)); pipeline $P seed $S; run $RUN; FR* config $CFG; LLP-blind $LLP; e* cut $CUTE"
tar -xf "$RUN/gnn_scores/testset.tar" -C "$LOCAL/t"
echo "$(ls "$LOCAL/t" | wc -l) test graphs"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/t" --out "$RUN/trackbuild_test" --config "$CFG" --merge-model "$MERGE"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/t" --out "$RUN/trackbuild_test_llpblind" --config "$LLP" --merge-model "$MERGE"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/t" --out "$RUN/eval_test" --cuts "$CUTE"
python -m hnl.eval.edge_tpr --graphs "$LOCAL/t" --out "$RUN/edge_tpr_test.csv" --cut "$CUTE" \
    --per-event "$RUN/edge_tpr_test_events.parquet"
cp "$SEL" "$RUN/final_selection_used.json"
touch "$RUN/final_test.done"
