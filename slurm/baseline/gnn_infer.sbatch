#!/bin/bash
# Baseline: GNN edge scores on the val graphs of all three labeled samples, packed to $RUN/gnn_scores.
# Usage: sbatch slurm/baseline/gnn_infer.sbatch ARM SEED
#SBATCH -J bl_gnn_infer
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
python -m hnl.train.chain_check --run "$RUN" --stages ml || { echo "HOLD: see the message above"; touch "$RUN/HOLD"; exit 1; }   # the graphs come from the current ML checkpoint
CKPT=$(best_ckpt gnn)
echo "gnn checkpoint $CKPT"
$R stage --arm "$ARM" --what val_all --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/gnn_in"
$R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/gnn_in" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores/"
touch "$RUN/gnn_infer.done"
