#!/bin/bash
# M2 / Stage-1 studies (CPU) on a baseline run's metric-learning val graphs: triplet counts (Stage-1 B sizing),
# true-edge efficiency per layer pair, and the ceiling of the graph united with displaced physics windows. Usage: sbatch slurm/baseline/ml_graph_studies.sbatch ARM SEED
#SBATCH -J bl_ml_studies
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
mkdir -p "$LOCAL/g"
tar -xf "$RUN/ml_graphs/valset.tar" -C "$LOCAL/g"
python -m hnl.models.triplet_stats --graphs "$LOCAL/g" --max-events 300 > "$RUN/triplet_stats.txt" 2>&1 || true
python -m hnl.models.triplet_stats --graphs "$LOCAL/g" --max-events 300 --pattern "event2*.pyg.gz" \
    > "$RUN/triplet_stats_hnl.txt" 2>&1 || true
python -m hnl.graphs.layer_pairs --out "$HNL_DATA/derived/graph_ceiling/layer_pairs_v1" --ml-graphs "$LOCAL/g" \
    --ml-label "ml_arm${ARM}_s${SEED}"
python -m hnl.graphs.union_ceiling --graphs "$LOCAL/g" --out "$RUN/union_ceiling" \
    --windows "d0=10,D=inf,s=1;d0=30,D=inf,s=1;d0=100,D=inf,s=1"
touch "$RUN/ml_studies.done"
