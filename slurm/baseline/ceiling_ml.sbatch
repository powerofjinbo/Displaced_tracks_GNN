#!/bin/bash
# M2 (CPU): graph-stage ceilings of a baseline run's own metric-learning val graphs (no GNN needed): perfect-classifier
# CC + DM and graph statistics on the val splits of all three labeled samples. Usage: sbatch slurm/baseline/ceiling_ml.sbatch ARM SEED
#SBATCH -J bl_ceiling_ml
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
mkdir -p "$LOCAL/g"
tar -xf "$RUN/ml_graphs/valset.tar" -C "$LOCAL/g"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/g" --out "$RUN/ceiling_ml" --cuts none
touch "$RUN/ceiling_ml.done"
