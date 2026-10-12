#!/bin/bash
# score_full.sbatch with the memory-bounded forward (hnl.trackml.chunked, DECISIONS 2026-10-11): the same model,
# inputs and output files, computed in ranges of middle hits and blocks of rows, so that the accuracy-phase T2
# events (140-255 GB in the one-shot forward) fit in an ordinary node. Validated on val events against the stored
# whole-event scores before use.
# S5 after gate G2 failed with both halos (DECISIONS 2026-10-08): edge scores of whole events, one forward per event on
# the CPU (hnl.trackml.infer score --sectors full, gate G2's reference; a whole event needs more than 48 GB on a GPU).
# Reads a persistent event store built on the GPU like the training store (FRNN base graph + stored additions): the S4
# store for the training subsets (dev, merge_train, gbdt, calib) and valmon; $RUN/store_s5/<SET>[_<COND>] for the other
# sets. One array task per block of 50 events (sorted store order, the files of an unsplit run); the task that finds
# every block's file present writes OUT.done. Test sets only through test_chain.sh (TEST_CHAIN=1).
# Usage: [SEED=0] [RUN_SUFFIX=_x1] [CKPT=best|epoch0|path] [TAG=..] sbatch --array=0-K slurm/trackml/score_full.sbatch SET [COND]
#SBATCH -J tml_scorefull_ch
#SBATCH -p submit
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH -t 12:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
SET=$1
COND=${2:-full}
case $SET in test|t2) [ "${TEST_CHAIN:-0}" = 1 ] || { echo "test sets only through test_chain.sh"; exit 1; } ;; esac
EV=()
case $SET in
    dev|merge_train|gbdt|calib) STORE=$RUN/store/trainset; EV=(--events "$SET")
                                N=$(events "$SET" | tr ',' '\n' | grep -c .) ;;
    valmon) STORE=$RUN/store/valset; N=$(ls "$STORE" | grep -c '\.pyg\.gz$') ;;
    *) STORE=$RUN/store_s5/$SET; [ "$COND" != full ] && STORE=${STORE}_$COND
       require "$STORE.done"; N=$(ls "$STORE" | grep -c '\.pyg\.gz$') ;;
esac
case $SET in dev|merge_train|gbdt|calib|valmon) [ "$COND" = full ] || { echo "$SET: condition full only"; exit 1; } ;; esac
if [ "${CKPT:-best}" = best ]; then C=$(best_ckpt gnn); elif [ "$CKPT" = epoch0 ]; then C=$(first_epoch_ckpt); else C=$CKPT; fi
OUT=$(scores_dir "$SET" "$COND" "${TAG:-}")
echo "set $SET ($N events), condition $COND, block ${SLURM_ARRAY_TASK_ID:-0}, store $STORE, ckpt $C, out $OUT"
python -m hnl.trackml.infer score --store "$STORE" "${EV[@]}" --block "${SLURM_ARRAY_TASK_ID:-0}" --ckpt "$C" \
    --out "$OUT" --condition "$COND" --sectors full --device cpu --chunked
NB=$(( (N + 49) / 50 ))
NF=$(ls "$OUT" 2>/dev/null | grep -c '^scores_.*\.parquet$')
echo "blocks written: $NF of $NB"
if [ "$NF" -eq "$NB" ]; then touch "$OUT.done"; echo "all blocks: $OUT.done"; fi
