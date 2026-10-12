#!/bin/bash
# Run an existing CPU job script of this repo unchanged inside this allocation, with WORKERS worker processes instead of
# one per allocated CPU: the scripts' Python stages default --workers to $SLURM_CPUS_PER_TASK, and MaxMemPerCPU = 2 GB on
# this cluster ties the memory of a job to its CPU count, so a stage that needs more than 2 GB per worker cannot get it
# by asking for more memory. Here SLURM_CPUS_PER_TASK is set to WORKERS before the inner script runs (its #SBATCH lines
# are comments when run with bash; resources come from this job's own sbatch options). Results do not depend on the
# worker count (per-event work, deterministic per event). Written for the training-sample study
# (docs/training_sample_study.md), where the arm (iii) embedding gives denser candidate sets than arm (i)'s.
# Usage: WORKERS=16 [script's env vars] sbatch -c 28 --mem=56G -t 03:00:00 -J NAME slurm/stage1/with_workers.sbatch \
#            slurm/stage1/<script>.sbatch ARGS...
#SBATCH -J s1_with_workers
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 28
#SBATCH --mem=56G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
INNER=$1; shift
W=${WORKERS:?WORKERS (number of worker processes) is required}
[ "$W" -le "$SLURM_CPUS_PER_TASK" ] || { echo "WORKERS $W > allocated CPUs $SLURM_CPUS_PER_TASK"; exit 1; }
echo "with_workers: $INNER $*; allocation $SLURM_CPUS_PER_TASK CPUs, ${SLURM_MEM_PER_NODE:-?} MB; workers $W"
export SLURM_CPUS_PER_TASK=$W
exec bash "/work/submit/jinboz1/hnl/hnl-displaced/$INNER" "$@"
