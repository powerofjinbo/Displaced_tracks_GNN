#!/bin/bash
# S3 (plan section 6.2): the union additions (learned-support pairs not in the base graph, int32, 50 events per zstd
# parquet file) of an event set, for the chosen graph (graph/graph_choice.json), under $RUN/additions/<SET>[_<COND>]/.
# CPU by default (cKDTree base and candidates, numpy support: the exact reference, about 2-3 min per event at the ttbar
# operating point on 16 cores). Once gate G4 has passed, run it on an A30 (FRNN graphs, torch support):
#   DEVICE=cuda sbatch -p submit-gpu --constraint=nvidia_a30 --gres=gpu:1 -c 8 --mem=32G --exclude= ... additions.sbatch SET
# (S4 / S5 then rebuild the same FRNN base graph). Resumable: events already stored are skipped. The storage stop rule
# is checked before every write. Test sets (test, t2) only from test_chain.sh.
# Usage: [SEED=0] sbatch --array=0-3 slurm/trackml/additions.sbatch SET [COND]
#   SET: train | val | test | t2 (the subsets valmon, dev, gbdt, merge_train read val's / train's);
#   COND: full (default) | nonoise | 1gev
#SBATCH -J tml_additions
#SBATCH -p submit
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
SET=$1
COND=${2:-full}
require "$RUN/graph_choice.done"
TEST=()
case $SET in test|t2) [ "${TEST_CHAIN:-0}" = 1 ] || { echo "test sets only through test_chain.sh"; exit 1; }; TEST=(--allow-test) ;; esac
OUT=$(adds_dir "$SET" "$COND")
EV=$(array_slice "$(events "$SET" "${TEST[@]}")")
python -m hnl.trackml.graphs additions --events "$EV" --graph-choice "$RUN/graph/graph_choice.json" --out "$OUT" \
    --condition "$COND" --device "${DEVICE:-cpu}" "${TEST[@]}"
