#!/bin/bash
# Stage 1: train one edge-classifier variant on a baseline run's metric-learning graphs (same arm and seed), then score
# and evaluate the val splits of all three labeled samples. Resumes automatically; preempted -> exits 0 without marker.
# Usage: [LIMIT_TRAIN=N] sbatch slurm/stage1/variant.sbatch NAME CONFIG ARM SEED EPOCHS
# LIMIT_TRAIN: training graphs per epoch, for reduced-cost screening runs only (run them under a separate S1TAG): the
# first N positions of each epoch's shuffled order (the runner's DDPStrategy makes Lightning reshuffle every training
# epoch; DECISIONS 2026-10-06 ~18:45 UTC), the same subsets for every run with the same seed.
#SBATCH -J s1_variant
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
NAME=$1; CONFIG=$2
export RUN_OVERRIDE=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_${S1TAG:-v1}/$NAME/arm$3_s$4
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$3" "$4"
# one job per run (chains, the orchestrator and slurm/stage1/express_driver.sh can each queue one): leave if another job
# of this name is already running, or if the run is finished
others=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -o %i 2>/dev/null | grep -vx "$SLURM_JOB_ID" || true)
[ -n "$others" ] && { echo "job(s) $others of the same name already running: exiting"; exit 0; }
[ -e "$RUN/eval.done" ] && { echo "eval.done exists: nothing to do"; exit 0; }
GRAPHS=$HNL_DATA/runs/baseline_${TAG}/arm${ARM}_s${SEED}/${GRAPHS_SUBDIR:-ml_graphs}   # GRAPHS_SUBDIR=union_graphs: HelixFilter union
echo "variant $NAME config $CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)) graphs $GRAPHS"
if [ ! -e "$RUN/gnn_train.done" ]; then
    $R stage --arm "$ARM" --what train,valmon --source tar --tar-dir "$GRAPHS" --dest "$LOCAL/in"
    train_stage gnn "$5" "$LOCAL/in" --config "$CONFIG" ${LIMIT_TRAIN:+--limit-train-batches "$LIMIT_TRAIN"}
    [ -e "$RUN/gnn_train.done" ] || exit 0
    rm -rf "$LOCAL/in"
fi
CKPT=$(best_ckpt gnn)
$R stage --arm "$ARM" --what val_all --source tar --tar-dir "$GRAPHS" --dest "$LOCAL/val"
$R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/val" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores/"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/gnn/valset" --out "$RUN/eval_val" --cuts 0.01:0.99:0.01
touch "$RUN/eval.done"
