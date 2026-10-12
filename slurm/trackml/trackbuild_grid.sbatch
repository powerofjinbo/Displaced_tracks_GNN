#!/bin/bash
# S6b (plan section 5): the val builder grid in one condition (builders.families: cc, walkthrough + merging, MWS-layer
# with and without merging, two-pass MWS + merging, learned merging; extended grids), every configuration matched with
# all metrics of hnl.trackml.evaluate (ours and P0-P4) -> $RUN/grid/<COND>/ (per_event, per_target, timing, summary).
# Merge-cut sets (DECISIONS 2026-10-05): every configuration of the hand-cut merging families (walkmerge,
# mwslayermerge, mws2passmerge) runs under both builders.merge_cut_sets (H = HNL cuts, C = per |eta| bin of the
# union), the set being a trailing grid parameter (flag 0 / 1) that the builder rule chooses like any other.
# G3 (2026-10-05, accuracy dev events 1000-1004, oracle-noise scores): both merging passes go through
# hnl.trackml.seqmerge.merge_exact, labels identical to the unfiltered passes. One build at a time on 16 CPUs
# (HNL_TML_PREFIT_PROCS=16): heaviest family mws2passmerge H 17-37 s, C 34-80 s per event and configuration; with 4
# builds sharing 16 CPUs (4 processes each, earlier any-order code) H <= 117 s and C <= 165 s except mws2passmerge C
# (<= 294 s). ~2-3 GB per worker. Here each build is one process (HNL_TML_PREFIT_PROCS default 1, the workers fill
# the CPUs); C costs about 2-3 times H (its loose |eta| bins keep ~85 % of the 1-5 M sequential candidates).
# Usage: [SEED=0] [WORKERS=32] sbatch slurm/trackml/trackbuild_grid.sbatch [COND]
#SBATCH -J tml_grid
#SBATCH -p submit
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
COND=${1:-full}
S=$(scores_dir val "$COND")
require "$S.done" "$RUN/lmerge_train.done"
python -m hnl.trackml.trackbuild grid --scores "$S" --events val --out "$RUN/grid/$COND" --condition "$COND" \
    --merge-model "$RUN/lmerge/model.joblib" --workers "${WORKERS:-$(nproc)}"
touch "$RUN/grid_$COND.done"
