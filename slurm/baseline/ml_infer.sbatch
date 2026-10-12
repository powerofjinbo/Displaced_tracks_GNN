#!/bin/bash
# Baseline: (r, knn) scan on the arm's own val mix (preregistration §1b: >= 99 % of all true edges, knn cap never
# binding on val), then metric-learning graphs for the arm's train split and for the val splits of all three labeled
# samples, packed to $RUN/ml_graphs. Usage: sbatch slurm/baseline/ml_infer.sbatch ARM SEED
#SBATCH -J bl_ml_infer
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
CKPT=$(best_ckpt ml)
echo "ml checkpoint $CKPT"
$R stage --arm "$ARM" --what val --dest "$LOCAL/scan"
python -m hnl.graphs.ml_scan --ckpt "$CKPT" --input-dir "$LOCAL/scan" --out "$RUN/ml_scan"
SEL=$(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))['selection']
r = s['r_star_all_edges']
k = s.get('knn_all_edges_never_binds') or s.get('knn_all_edges_binds_lt5pct')
if r is None or k is None:
    sys.exit('no (r, knn) on the grid meets the preregistered rule: stop and decide by hand')
if s['r_star_all_edges_pairs_per_event'] > 50000:
    sys.exit('graph above 50k pairs per event: filter decision needed (DECISIONS.md), stop and decide by hand')
print(r, k)" "$RUN/ml_scan/scan.json") || { echo "HOLD: see the message above"; touch "$RUN/HOLD"; exit 1; }
read -r RSTAR KSTAR <<< "$SEL"
echo "selected r_infer=$RSTAR knn_infer=$KSTAR"
rm -rf "$LOCAL/scan"
$R stage --arm "$ARM" --what train,val_all --dest "$LOCAL/inf"
$R infer --stage ml --ckpt "$CKPT" --input-dir "$LOCAL/inf" --out-dir "$LOCAL/ml" --set r_infer=$RSTAR --set knn_infer=$KSTAR
$R pack --src "$LOCAL/ml" --dest "$RUN/ml_graphs"
cp "$LOCAL"/ml/infer_*.json "$RUN/ml_graphs/"
touch "$RUN/ml_infer.done"
