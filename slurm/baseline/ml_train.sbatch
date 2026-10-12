#!/bin/bash
# Baseline: metric-learning training (resumes automatically). Usage: sbatch slurm/baseline/ml_train.sbatch ARM SEED EPOCHS
#SBATCH -J bl_ml_train
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
$R stage --arm "$ARM" --what train,valmon --dest "$LOCAL/fs"
train_stage ml "$3" "$LOCAL/fs"
