#!/bin/bash
# GNN edge scores on the first N training graphs of a baseline run, on CPU (input for learned track building when
# no GPU slot is free; the GNN's train and val losses agree, DECISIONS.md), packed to $RUN/gnn_scores/trainset.tar.
# Usage: [N_EVENTS=2000] sbatch slurm/stage1/score_train_cpu.sbatch ARM SEED
#SBATCH -J s1_score_train_cpu
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
N=${N_EVENTS:-2000}
python -m hnl.train.chain_check --run "$RUN" --stages ml,gnn || exit 1
CKPT=$(best_ckpt gnn)
$R stage --arm "$ARM" --what train --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/all"
mkdir -p "$LOCAL/in/trainset"
files=("$LOCAL/all/trainset"/*.pyg.gz)
mv "${files[@]:0:$N}" "$LOCAL/in/trainset/"
rm -rf "$LOCAL/all"
echo "scoring $(ls "$LOCAL/in/trainset" | wc -l) training graphs on CPU with $CKPT"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" \
    --accelerator cpu
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
mkdir -p "$RUN/gnn_scores/train_infer"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores/train_infer/"
touch "$RUN/score_train.done"
