#!/bin/bash
# Baseline (CPU): our metrics on the GNN val scores (docs/definitions.md). Usage: sbatch slurm/baseline/eval.sbatch ARM SEED
#SBATCH -J bl_eval
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
python -m hnl.train.chain_check --run "$RUN" --stages ml,gnn || { echo "HOLD: see the message above"; touch "$RUN/HOLD"; exit 1; }   # val scores from the current best checkpoints
mkdir -p "$LOCAL/scores"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/scores"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/scores" --out "$RUN/eval_val" --cuts 0.01:0.99:0.01
touch "$RUN/eval.done"
