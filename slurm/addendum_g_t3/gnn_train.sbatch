#!/bin/bash
# Addendum G T3, acorn (ttbar+G) seed S: InteractionGNN on the metric-learning graphs (30 epochs, the arm-(iii) budget),
# resumable 1-h express chunks. Usage: sbatch slurm/addendum_g_t3/gnn_train.sbatch S
#SBATCH -J g3_gnn_train
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100|nvidia_rtx6000|nvidia_gtx1080
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/addendum_g_t3/_g.sh "$1"
[ -e "$RUN/gnn_train.done" ] && { echo "gnn_train.done exists: nothing to do"; exit 0; }
python -m hnl.train.chain_check --run "$RUN" --stages ml || { echo "HOLD: see the message above"; touch "$RUN/HOLD"; exit 1; }
$RG stage --what train,valmon --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/gnn_in"
train_stage gnn 30 "$LOCAL/gnn_in"
