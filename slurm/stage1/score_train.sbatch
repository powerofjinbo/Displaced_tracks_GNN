#!/bin/bash
# GNN edge scores on a baseline run's TRAINING graphs (input for learned second-stage track building; the GNN's
# train and val losses agree at the end of training, DECISIONS.md), packed to $RUN/gnn_scores/trainset.tar.
# Usage: sbatch slurm/stage1/score_train.sbatch ARM SEED
#SBATCH -J s1_score_train
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
python -m hnl.train.chain_check --run "$RUN" --stages ml,gnn || exit 1
CKPT=$(best_ckpt gnn)
$R stage --arm "$ARM" --what train --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/in"
$R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
mkdir -p "$RUN/gnn_scores/train_infer"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores/train_infer/"
touch "$RUN/score_train.done"
