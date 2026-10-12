#!/bin/bash
# Environment build, stages "build" + "smoke" on one A30: FRNN / prefix_sum wheels, acorn (pin + patches, CUDA
# extension), hnl (editable), then env/smoke_gpu.py and env/lock.txt. An A30 node has 48 CPUs, ~123 GiB and 2 GPUs;
# 8 CPUs / 16 GB leave the second GPU usable (MaxMemPerCPU = 2 GB).
#SBATCH -J env_build_smoke
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
bash env/make_env.sh --stage build,smoke
