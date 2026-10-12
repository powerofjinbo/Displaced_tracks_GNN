#!/bin/bash
# Baseline ONE-TIME test evaluation (preregistration §3, §7), run only after the val working points are committed:
# metric-learning graphs (the run's checkpoint and its val-selected r, knn) and GNN scores for the test splits of the
# three labeled samples and the external hnl15_100 set, then evaluate_gnn at the cuts fixed on val for this arm and
# seed (s*, cut at FR*, cut at e*; from $WP). Usage: sbatch slurm/baseline/test_eval.sbatch ARM SEED WP_JSON
#SBATCH -J bl_test_eval
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
WP=$3
[ -e "$RUN/test_eval.done" ] && { echo "test evaluation already done for this run: refusing to repeat it"; exit 1; }
# the working points must be the preregistered 3 x 3 ones, and the checkpoints the ones behind the val numbers
python -m hnl.train.chain_check --run "$RUN" --wp "$WP"
CUTS=$(python -I -c "
import json, sys
wp = json.load(open(sys.argv[1])); arm, seed = sys.argv[2], sys.argv[3]
c = {wp['s_star'][arm], wp['cuts_at_FR_star'][arm][seed], wp['cuts_at_e_star'][arm][seed]}
print(','.join(f'{x:.4f}' for x in sorted(c)))" "$WP" "$ARM" "$SEED")
read -r RINF KINF < <(python -I -c "
import glob, json, sys
m = json.load(open(sorted(glob.glob(sys.argv[1] + '/infer_*.json'))[-1]))['overrides']
print(m['r_infer'], m['knn_infer'])" "$RUN/ml_graphs")
echo "cuts $CUTS; r_infer $RINF knn_infer $KINF; working points $WP ($(sha256sum "$WP" | cut -c1-16))"
MLCK=$(best_ckpt ml)
GNCK=$(best_ckpt gnn)
$R stage --arm "$ARM" --what test_all,external --dest "$LOCAL/fs"
$R infer --stage ml --ckpt "$MLCK" --input-dir "$LOCAL/fs" --out-dir "$LOCAL/ml" --set r_infer=$RINF --set knn_infer=$KINF
rm -rf "$LOCAL/fs"
$R infer --stage gnn --ckpt "$GNCK" --input-dir "$LOCAL/ml" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores_test"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores_test/"
for f in "$LOCAL"/ml/infer_*.json; do cp "$f" "$RUN/gnn_scores_test/ml_$(basename "$f")"; done
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/gnn/testset" --out "$RUN/eval_test" --cuts "$CUTS"
cp "$WP" "$RUN/eval_test/working_points_used.json"
touch "$RUN/test_eval.done"
