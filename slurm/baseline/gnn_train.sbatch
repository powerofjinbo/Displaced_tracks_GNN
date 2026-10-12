#!/bin/bash
# Baseline: InteractionGNN on the metric-learning graphs (resumes automatically).
# Usage: sbatch slurm/baseline/gnn_train.sbatch ARM SEED EPOCHS
#SBATCH -J bl_gnn_train
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
$R stage --arm "$ARM" --what train,valmon --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/gnn_in"
train_stage gnn "$3" "$LOCAL/gnn_in"
