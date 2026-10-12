#!/bin/bash
# Submits the TrackML metric learning as a chain of N resumable chunks of at most 1 h on submit-gpu-express
# (slurm/trackml/ml_chunk.sbatch, each depending on the previous one with afterany), the pattern of
# slurm/stage1/express_driver.sh (DECISIONS 2026-10-06) without a polling driver: the express driver runs only
# slurm/stage1/variant.sbatch. Chunks after ml_train.done exit at once; a chain without progress cancels itself.
# Light (login node, seconds). Prints the job ids.
# Usage: RUN_SUFFIX=_x1 ML_CONFIG=configs/trackml/ml_train_x1.yaml [N=10] [NAME=tml_ml_x1] \
#        [CONSTRAINT='nvidia_a30|Tesla_v100|nvidia_rtx6000'] [NICE=100] [CPUS=6] [MEM=12G] [AFTER=<job id>] [HOLD=1] \
#        bash slurm/trackml/ml_chunks.sh
# CPUS / MEM: the X1 smoke on a V100 (job 6705177, 6 CPUs) peaked at 3.0 GB RSS; 6 CPUs / 12 GB fit next to the CPU
# jobs that share the GPU nodes (MaxMemPerCPU 2 GB: a larger --mem raises the CPU count).
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
N=${N:-10}
NAME=${NAME:-tml_ml_x1}
CONSTRAINT=${CONSTRAINT:-nvidia_a30}
NICE=${NICE:-100}
CPUS=${CPUS:-6}
MEM=${MEM:-12G}
: "${ML_CONFIG:?set ML_CONFIG}"
: "${RUN_SUFFIX?set RUN_SUFFIX (empty for the main run)}"
dep=${AFTER:-}
for i in $(seq 1 "$N"); do
    id=$(env -u OMP_NUM_THREADS sbatch --parsable -J "$NAME" -p submit-gpu-express -t 01:00:00 --constraint="$CONSTRAINT" \
        --gres=gpu:1 -c "$CPUS" --mem="$MEM" --signal=B:USR1@300 --nice="$NICE" ${HOLD:+--hold} \
        --export="ALL,RUN_SUFFIX=$RUN_SUFFIX,ML_CONFIG=$ML_CONFIG" ${dep:+--dependency=afterany:$dep} \
        slurm/trackml/ml_chunk.sbatch)
    echo "$id"
    dep=$id
done
