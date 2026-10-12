#!/bin/bash
# S7 part (only through test_chain.sh, TEST_CHAIN=1): the frozen configurations of one condition on the test scores of
# SET (test or t2), every metric of hnl.trackml.evaluate, then the report (event bootstrap B = 500, seed 12345; the
# comparison rows with the published numbers of that condition) -> $RUN/test/<SET>[_COND]/ and $RUN/test/report/.
# As an array (DECISIONS 2026-10-11), one task per block of BS events (grid --block) -> <OUT>.blocks/b<K>; the task that
# finds every block present pools them (grid-merge; identical to the unsplit grid) into <OUT>, then the report.
# Usage (test_chain.sh): [BS=20] sbatch [--array=0-K] slurm/trackml/test_grid.sbatch SET COND FROZEN_CONFIGS_JSON
#SBATCH -J tml_test_grid
#SBATCH -p submit
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
SET=$1; COND=$2; FROZEN=$3
[ "${TEST_CHAIN:-0}" = 1 ] || { echo "only through test_chain.sh"; exit 1; }
python -m hnl.trackml.freeze check --file configs/trackml/selection_v1.json
S=$(scores_dir "$SET" "$COND")
OUT=$RUN/test/$SET$([ "$COND" = full ] || echo "_$COND")
GRID=(python -m hnl.trackml.trackbuild grid --scores "$S" --events "$SET" --allow-test --condition "$COND"
      --configs "$FROZEN" --merge-model "$RUN/lmerge/model.joblib")
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    "${GRID[@]}" --out "$OUT" --workers "${WORKERS:-16}"   # about 6 GB per worker (S6)
else
    BS=${BS:-20}; K=$SLURM_ARRAY_TASK_ID; G=$OUT.blocks
    "${GRID[@]}" --out "$G/b$K" --block "$K" --block-size "$BS" --chunk 1 --workers "$(nproc)"
    touch "$G/b$K.done"
    N=$(events "$SET" --allow-test | tr ',' '\n' | grep -c .)
    NB=$(( (N + BS - 1) / BS ))
    NF=$(ls "$G" | grep -c '^b[0-9]*\.done$')
    echo "blocks done: $NF of $NB"
    { [ "$NF" -eq "$NB" ] && mkdir "$G/.merging" 2>/dev/null; } || exit 0
    python -m hnl.trackml.trackbuild grid-merge --blocks $(for k in $(seq 0 $((NB - 1))); do echo "$G/b$k"; done) \
        --events "$SET" --allow-test --out "$OUT"
fi
python -m hnl.trackml.evaluate report --grid "$OUT" --condition "$COND" --out "$RUN/test/report_$SET"
touch "$OUT.done"
