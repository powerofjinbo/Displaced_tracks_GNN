#!/bin/bash
# S6b split into event blocks (DECISIONS 2026-10-11): the val builder grid of trackbuild_grid.sbatch, one array task per
# block of BS events (hnl.trackml.trackbuild grid --block) -> $RUN/grid_blocks/<COND>/b<K>; the task that finds every
# block present pools them (grid-merge) into $DEST (default $RUN/grid_blocks/<COND>/merged) and writes $DEST.done.
# Same configurations, code and scores as the unsplit grid; a build is deterministic, so the pooled summary equals
# the unsplit one. Usage: [SEED=0] [BS=10] [DEST=dir] sbatch --array=0-14 slurm/trackml/trackbuild_grid_blocks.sbatch [COND]
#SBATCH -J tml_gridb
#SBATCH -p submit
#SBATCH -c 10
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
COND=${1:-full}
BS=${BS:-10}
K=${SLURM_ARRAY_TASK_ID:-0}
S=$(scores_dir val "$COND")
require "$S.done" "$RUN/lmerge_train.done"
G=$RUN/grid_blocks/$COND
DEST=${DEST:-$G/merged}
python -m hnl.trackml.trackbuild grid --scores "$S" --events val --out "$G/b$K" --condition "$COND" \
    --merge-model "$RUN/lmerge/model.joblib" --block "$K" --block-size "$BS" --chunk 1 --workers "$(nproc)"
touch "$G/b$K.done"
N=$(events val | tr ',' '\n' | grep -c .)
NB=$(( (N + BS - 1) / BS ))
NF=$(ls "$G" | grep -c '^b[0-9]*\.done$')
echo "blocks done: $NF of $NB"
if [ "$NF" -eq "$NB" ] && mkdir "$G/.merging" 2>/dev/null; then
    python -m hnl.trackml.trackbuild grid-merge --blocks $(for k in $(seq 0 $((NB - 1))); do echo "$G/b$k"; done) \
        --events val --out "$DEST" && touch "$DEST.done"
fi
