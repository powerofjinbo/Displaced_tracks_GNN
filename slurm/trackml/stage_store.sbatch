#!/bin/bash
# S5 after gate G2 (DECISIONS 2026-10-08): the persistent event store of an event set on /ceph for the whole-event CPU
# scores (score_full.sbatch), built on the GPU as the training store (hnl.trackml.graphs stage: FRNN base graph and the
# stored additions), resumable (--skip-existing; files written to .tmp and renamed), into $RUN/store_s5/<SET>[_<COND>];
# <dir>.done once every event of the set is there. Test sets only through test_chain.sh (TEST_CHAIN=1).
# Usage: [SEED=0] [RUN_SUFFIX=_x1] sbatch slurm/trackml/stage_store.sbatch SET [COND]
#SBATCH -J tml_stage
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100|nvidia_rtx6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
SET=$1
COND=${2:-full}
TEST=()
case $SET in test|t2) [ "${TEST_CHAIN:-0}" = 1 ] || { echo "test sets only through test_chain.sh"; exit 1; }; TEST=(--allow-test) ;; esac
require "$RUN/graph_choice.done"
D=$RUN/store_s5/$SET
[ "$COND" != full ] && D=${D}_$COND
python -m hnl.trackml.graphs stage --events "$SET" --graph-choice "$RUN/graph/graph_choice.json" \
    --additions "$(adds_dir "$SET" "$COND")" --dest "$D" --condition "$COND" --device cuda --allow-backend-change \
    --skip-existing "${TEST[@]}"
n=$(ls "$D" | grep -c '\.pyg\.gz$')
e=$(events "$SET" "${TEST[@]}" | tr ',' '\n' | grep -c .)
echo "store $D: $n of $e events"
if [ "$n" -eq "$e" ]; then touch "$D.done"; else echo "incomplete: run again"; exit 1; fi
