#!/bin/bash
# S2a (plan section 6.2): the radii r* of the trained embedding that give the density targets (graph.densities, uncapped
# pairs per hit) on the first graph.radii_events valmon events (bisection per event, radii averaged) -> graph/radii.json.
# CPU (embedding and cKDTree on CPU; FRNN is not needed for counting).
# Usage: [SEED=0] sbatch slurm/trackml/graph_radii.sbatch
#SBATCH -J tml_radii
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
require "$RUN/ml_train.done"
mkdir -p "$RUN/graph"
python -m hnl.trackml.graphs radii --events valmon --ml-ckpt "$(best_ckpt ml)" --out "$RUN/graph/radii.json" --device cpu
