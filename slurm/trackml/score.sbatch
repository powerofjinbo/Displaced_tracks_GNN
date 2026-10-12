#!/bin/bash
# S5 (plan section 6.3): stitched edge scores of an event set: the event store is built on the A30 (embedding + FRNN
# base + stored additions of the same condition), each event is scored by its 8 INFERENCE sectors and every edge keeps
# its owner sector's score (hnl.trackml.infer score) -> $RUN/scores/<SET>[_<COND>][_<TAG>]/ (zstd parquet).
# CKPT defaults to the best classifier checkpoint; CKPT=epoch0 uses the first-epoch weights (gates G2 / G3, TAG=epoch0).
# SECTORS=g2_fallback scores with +-45 deg halos. Test sets only through test_chain.sh.
# Usage: [SEED=0] [CKPT=path|epoch0] [TAG=..] [SECTORS=inference] sbatch slurm/trackml/score.sbatch SET [COND]
#SBATCH -J tml_score
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
SET=$1
COND=${2:-full}
TEST=()
case $SET in test|t2) [ "${TEST_CHAIN:-0}" = 1 ] || { echo "test sets only through test_chain.sh"; exit 1; }; TEST=(--allow-test) ;; esac
ADD=$(adds_dir "$SET" "$COND")
if [ "${CKPT:-best}" = best ]; then C=$(best_ckpt gnn); elif [ "$CKPT" = epoch0 ]; then C=$(first_epoch_ckpt); else C=$CKPT; fi
OUT=$(scores_dir "$SET" "$COND" "${TAG:-}")
python -m hnl.trackml.graphs stage --events "$SET" --graph-choice "$RUN/graph/graph_choice.json" --additions "$ADD" \
    --dest "$LOCAL/store/scoreset" --condition "$COND" --device cuda --allow-backend-change "${TEST[@]}"
python -m hnl.trackml.infer score --store "$LOCAL/store/scoreset" --ckpt "$C" --out "$OUT" --condition "$COND" \
    --sectors "${SECTORS:-inference}"
touch "$OUT.done"
