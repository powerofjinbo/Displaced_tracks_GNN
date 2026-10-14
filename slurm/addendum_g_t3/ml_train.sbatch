#!/bin/bash
# Addendum G T3, acorn (ttbar+G) seed S: metric learning on arm g (45 epochs, the arm-(iii) budget), resumable 1-h
# express chunks (FRNN: A30 or V100 only). Usage: sbatch slurm/addendum_g_t3/ml_train.sbatch S
#SBATCH -J g3_ml_train
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/addendum_g_t3/_g.sh "$1"
[ -e "$RUN/ml_train.done" ] && { echo "ml_train.done exists: nothing to do"; exit 0; }
$RG stage --what train,valmon --source fs --dest "$LOCAL/fs"
train_stage ml 45 "$LOCAL/fs"
