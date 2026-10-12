#!/bin/bash
# S7 (plan sections 5 "Freeze" and 10): the one-time TrackML test of the frozen Canto-L pipeline. Login node, light:
# checks and submits. Refuses unless configs/trackml/selection_v1.json is committed and unchanged
# (python -m hnl.trackml.freeze check). Then, with TEST_CHAIN=1 (the only way the stage scripts accept test sets):
#   for SET in test (this phase) and t2 (the cross-phase set), for COND in full, nonoise, 1gev:
#     additions (GPU array, frnn_cuda as for val) -> GPU-built event store (stage_store.sbatch) -> whole-event CPU
#     scores (score_full.sbatch; gate G2 failed with both halos, DECISIONS 2026-10-08) -> the frozen configurations of
#     that condition only (trackbuild grid
#     --configs, blocks of 20 events pooled by grid-merge) -> report (event bootstrap B = 500, seed 12345; comparison rows with the published numbers).
# Every test number is reported whatever it is; a bug found afterwards is fixed, the test re-run, and both reported.
# Guards (2026-10-11 review): variables that steer the stage scripts are unset; the run directory, every frozen input's
# sha256 and the classifier checkpoint are verified (freeze verify); the frozen classifier is passed as CKPT; a second
# launch is refused once $RUN/test/launched exists.
# Usage: [SEED=0] [DRY_RUN=1] RUN_SUFFIX=_x1 bash slurm/trackml/test_chain.sh
set -euo pipefail
unset CKPT TAG SCAN_TASK_COUNT BS WORKERS EPOCH1_CKPT HNL_TML_SPLIT HNL_TML_CONFIG RUN_ROOT OMP_NUM_THREADS_OVERRIDE
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
SEL=configs/trackml/selection_v1.json
python -m hnl.trackml.freeze check --file "$SEL" || { echo "refusing: $SEL is not committed unchanged"; exit 1; }
SEED=${SEED:-$(python -m hnl.trackml.config get seed 2>/dev/null | tail -n 1)}
RUN=$(python -m hnl.trackml.config get run_root 2>/dev/null | tail -n 1)/seed$SEED${RUN_SUFFIX:-}   # RUN_SUFFIX=_x1: the X1 run
sub() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; else sbatch --parsable "$@"; fi; }
python -m hnl.trackml.freeze verify --file "$SEL" --run "$RUN" || { echo "refusing: frozen inputs do not verify"; exit 1; }
CKPT=$(python -I -c "import json, sys; print(json.load(open(sys.argv[1]))['inputs']['classifier_checkpoint']['path'])" "$SEL")
export TEST_CHAIN=1 SEED CKPT
mkdir -p "$RUN/test"
if [ "${DRY_RUN:-0}" != 1 ]; then
    [ -e "$RUN/test/launched" ] && { echo "refusing: the test chain was launched before ($RUN/test/launched)"; exit 1; }
    echo "$(date -u +%FT%TZ) git $(git rev-parse HEAD)" > "$RUN/test/launched"
fi
for SET in test t2; do
    for COND in full nonoise 1gev; do
        FROZEN=$RUN/test/frozen_$COND.json
        python -m hnl.trackml.freeze configs --file "$SEL" --condition "$COND" > "$FROZEN"
        if [ "$(cat "$FROZEN")" = "[]" ]; then echo "$SET $COND: no frozen configuration"; continue; fi
        j1=$(sub --array=0-7 -p submit-gpu-express -t 01:00:00 --constraint='nvidia_a30|Tesla_v100|nvidia_rtx6000' \
            --gres=gpu:1 -c 8 --mem=32G --export=ALL,DEVICE=cuda slurm/trackml/additions.sbatch "$SET" "$COND")
        j2a=$(sub --dependency=afterok:$j1 --export=ALL slurm/trackml/stage_store.sbatch "$SET" "$COND")
        n=$(python -m hnl.trackml.config events "$SET" --allow-test 2>/dev/null | tail -n 1 | tr ',' '\n' | grep -c .)
        nb=$(( (n + 49) / 50 ))
        j2=$(sub --dependency=afterok:$j2a --array=0-$((nb - 1)) --export=ALL slurm/trackml/score_full.sbatch "$SET" "$COND")
        ng=$(( (n + 19) / 20 ))   # blocks of 20 events (BS default of test_grid.sbatch) on 10 CPUs
        j3=$(sub --dependency=afterok:$j2 --array=0-$((ng - 1)) -c 10 --mem=48G -t 12:00:00 --export=ALL \
            slurm/trackml/test_grid.sbatch "$SET" "$COND" "$FROZEN")
        echo "$SET $COND: additions $j1 -> store $j2a -> whole-event scores $j2 ($nb blocks) -> frozen configurations ($ng blocks) + report $j3"
    done
done
