#!/bin/bash
# Addendum G T3, acorn (ttbar+G) seed S: GNN edge scores of the val_all graphs, packed to RUN/gnn_scores.
# Usage: sbatch slurm/addendum_g_t3/gnn_infer.sbatch S
#SBATCH -J g3_gnn_infer
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100|nvidia_rtx6000|nvidia_gtx1080
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/addendum_g_t3/_g.sh "$1"
[ -e "$RUN/gnn_infer.done" ] && { echo "gnn_infer.done exists: nothing to do"; exit 0; }
python -m hnl.train.chain_check --run "$RUN" --stages ml || { echo "HOLD: see the message above"; touch "$RUN/HOLD"; exit 1; }
CKPT=$(best_ckpt gnn)
echo "gnn checkpoint $CKPT"
$RG stage --what val_all --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/gnn_in"
$R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/gnn_in" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores/"
touch "$RUN/gnn_infer.done"
