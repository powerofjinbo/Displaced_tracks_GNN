#!/bin/bash
# Environment build, stage "deps" (internet, no GPU): micromamba, Python 3.12, CUDA 12.6 toolchain env, the venv
# and its wheels, FRNN + acorn sources. CPU nodes reach pypi / pytorch / pyg / github / gitlab (checked 2026-10-02).
#SBATCH -J env_deps
#SBATCH -p submit
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
bash env/make_env.sh --stage deps
