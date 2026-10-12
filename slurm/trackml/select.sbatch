#!/bin/bash
# S6c (plan section 5): working points on val: WP-A (builder rule with the one-standard-error tie at FR_T = FR*, per
# family, then across families), WP-S (highest val TrackML score) and one WP-R per reference (P1 in C-nonoise, P2 / P3 in
# C-full, P4 in C-1GeV) -> $RUN/selection_v1.json and configs/trackml/selection_v1.json. The main session reviews it,
# adds the checkpoint / calibration sha256 values, and commits it before any test job (test_chain.sh refuses otherwise).
# GRID_FULL: the C-full grid of the event blocks (trackbuild_grid_blocks.sbatch, its <dir>.done) instead of $RUN/grid/full.
# Usage: [SEED=0] [GRID_FULL=dir] sbatch slurm/trackml/select.sbatch
#SBATCH -J tml_select
#SBATCH -p submit
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
if [ -n "${GRID_FULL:-}" ]; then require "$GRID_FULL.done"; GF=$GRID_FULL; else require "$RUN/grid_full.done"; GF=$RUN/grid/full; fi
echo "C-full grid $GF"
python -m hnl.trackml.trackbuild select --grid "$GF" \
    $( [ -e "$RUN/grid_nonoise.done" ] && echo --grid-nonoise "$RUN/grid/nonoise" ) \
    $( [ -e "$RUN/grid_1gev.done" ] && echo --grid-1gev "$RUN/grid/1gev" ) --out "$RUN/selection_v1.json"
python -m hnl.trackml.freeze make --selection "$RUN/selection_v1.json" --run "$RUN" \
    --out "$HNL_REPO/configs/trackml/selection_v1.json"
