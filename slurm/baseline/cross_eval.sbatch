#!/bin/bash
# Headroom crossed comparison (brief §5g): the val metric-learning graphs of arm GRAPH_ARM scored by the GNN of arm
# GNN_ARM (same seed), evaluated with our definitions. Usage: sbatch slurm/baseline/cross_eval.sbatch GRAPH_ARM GNN_ARM SEED
#SBATCH -J bl_cross_eval
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$2" "$3"
GRAPH_RUN=$HNL_DATA/runs/baseline_${TAG}/arm$1_s$3
OUT=$HNL_DATA/runs/baseline_${TAG}/cross/graph$1_gnn$2_s$3
mkdir -p "$OUT"
GNCK=$(best_ckpt gnn)
echo "graphs of $GRAPH_RUN scored by $GNCK"
$R stage --arm "$1" --what val_all --source tar --tar-dir "$GRAPH_RUN/ml_graphs" --dest "$LOCAL/in"
$R infer --stage gnn --ckpt "$GNCK" --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn"
cp "$LOCAL"/gnn/infer_*.json "$OUT/"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/gnn/valset" --out "$OUT/eval_val" --cuts 0.01:0.99:0.01
touch "$OUT/eval.done"
