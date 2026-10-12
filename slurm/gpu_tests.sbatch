#!/bin/bash
# GPU-marked tests (acorn pitfalls: FRNN first-K, filter chunking, ...) on one A30, express partition.
#SBATCH -J gpu_tests
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); host $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
python -m pytest -q -p no:cacheprovider -m gpu tests "$@"
