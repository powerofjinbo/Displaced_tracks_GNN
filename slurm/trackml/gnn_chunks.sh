#!/bin/bash
# Submits the TrackML classifier training S4 as a chain of N resumable chunks of at most 1 h on submit-gpu-express
# (slurm/trackml/gnn_chunk.sbatch, each depending on the previous one with afterany), as slurm/trackml/ml_chunks.sh does
# for the metric learning. The event store of the training and valmon events is built once into $RUN/store on /ceph
# (the first chunk(s)) instead of node-local disk at every job start. GPUs with at least 24 GB (the training windows
# peak at 15 GB, G0). Chunks after gnn_train.done exit at once; a chain without progress cancels itself.
# Light (login node, seconds). Prints the job ids.
# Usage: RUN_SUFFIX=_x1 [N=40] [NAME=tml_gnn_x1] [CONSTRAINT='nvidia_a30|Tesla_v100|nvidia_rtx6000'] [NICE=0]
#        [CPUS=8] [MEM=32G] [AFTER=<job id>] [HOLD=1] bash slurm/trackml/gnn_chunks.sh
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
N=${N:-40}
NAME=${NAME:-tml_gnn_x1}
CONSTRAINT=${CONSTRAINT:-nvidia_a30|Tesla_v100|nvidia_rtx6000}
NICE=${NICE:-0}
CPUS=${CPUS:-8}
MEM=${MEM:-32G}
: "${RUN_SUFFIX?set RUN_SUFFIX (empty for the main run)}"
dep=${AFTER:-}
for i in $(seq 1 "$N"); do
    id=$(env -u OMP_NUM_THREADS sbatch --parsable -J "$NAME" -p submit-gpu-express -t 01:00:00 --constraint="$CONSTRAINT" \
        --gres=gpu:1 -c "$CPUS" --mem="$MEM" --signal=B:USR1@300 --nice="$NICE" ${HOLD:+--hold} \
        -o /work/submit/jinboz1/hnl/logs/%x_%j.out \
        --export="ALL,RUN_SUFFIX=$RUN_SUFFIX" ${dep:+--dependency=afterany:$dep} \
        slurm/trackml/gnn_chunk.sbatch)
    echo "$id"
    dep=$id
done
