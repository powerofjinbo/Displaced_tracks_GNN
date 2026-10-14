#!/bin/bash
# Addendum G T3, acorn (ttbar+G) seed S: (r, knn) scan on arm g's own val mix (ttbar + generic_iso + generic_dense val;
# the rule of hnl.graphs.ml_scan, hnl.eval.addendum_g_t3 scan), then metric-learning graphs of the train events and of
# val_all (val of ttbar, both HNL samples and both generic components), packed to RUN/ml_graphs. A30 or V100 (FRNN).
# Usage: sbatch slurm/addendum_g_t3/ml_infer.sbatch S
#SBATCH -J g3_ml_infer
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/addendum_g_t3/_g.sh "$1"
[ -e "$RUN/ml_infer.done" ] && { echo "ml_infer.done exists: nothing to do"; exit 0; }
CKPT=$(best_ckpt ml)
echo "ml checkpoint $CKPT"
if [ ! -e "$RUN/ml_scan/scan.json" ] || [ "$(python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['ckpt'])" "$RUN/ml_scan/scan.json")" != "$CKPT" ]; then
    $RG stage --what val --dest "$LOCAL/scan"
    $G3 scan --ckpt "$CKPT" --input-dir "$LOCAL/scan" --out "$RUN/ml_scan"
    rm -rf "$LOCAL/scan"
fi
SEL=$($G3 rk --scan "$RUN/ml_scan/scan.json") || { echo "HOLD: see the message above"; touch "$RUN/HOLD"; exit 1; }
read -r RSTAR KSTAR <<< "$SEL"
echo "selected r_infer=$RSTAR knn_infer=$KSTAR"
$RG stage --what train,val_all --dest "$LOCAL/inf"
$R infer --stage ml --ckpt "$CKPT" --input-dir "$LOCAL/inf" --out-dir "$LOCAL/ml" --set r_infer=$RSTAR --set knn_infer=$KSTAR
$R pack --src "$LOCAL/ml" --dest "$RUN/ml_graphs"
cp "$LOCAL"/ml/infer_*.json "$RUN/ml_graphs/"
touch "$RUN/ml_infer.done"
