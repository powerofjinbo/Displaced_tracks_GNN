#!/bin/bash
# Baseline: two InteractionGNN trainings sharing one A30 (each is overhead bound and uses < 1 GiB), each exactly as
# gnn_train.sbatch (own staging, own run directory and marker, automatic resume, SIGUSR1 forwarded to both).
# Usage: sbatch slurm/baseline/gnn_train_pair.sbatch ARM1 SEED1 ARM2 SEED2 EPOCHS
#SBATCH -J bl_gnn_pair
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -uo pipefail
SB=/work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline
EPOCHS=$5
one() {   # ARM SEED: a full gnn_train in a subshell with its own scratch directory
    ( export SUBDIR=$1_s$2; source "$SB/_common.sh" "$1" "$2"
      $R stage --arm "$ARM" --what train,valmon --source tar --tar-dir "$RUN/ml_graphs" --dest "$LOCAL/gnn_in"
      train_stage gnn "$EPOCHS" "$LOCAL/gnn_in" ) > "/work/submit/jinboz1/hnl/logs/bl_gnn_pair_${SLURM_JOB_ID}_$1_s$2.out" 2>&1
}
one "$1" "$2" & P1=$!
one "$3" "$4" & P2=$!
trap 'kill -USR1 $P1 $P2 2>/dev/null' USR1
rc1=0; rc2=0
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
    wait $P1 2>/dev/null; wait $P2 2>/dev/null
done
wait $P1 || rc1=$?
wait $P2 || rc2=$?
echo "pair done: $1_s$2 rc=$rc1, $3_s$4 rc=$rc2"
