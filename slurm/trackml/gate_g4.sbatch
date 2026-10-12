#!/bin/bash
# Gate G4 (plan section 6.2), before the union additions run on the GPU: the torch port of the learned triplet support
# (hnl.trackml.support_gpu) against the numpy reference (hnl.trackml.events.support_select_multi) on the first 20 dev
# events, with the run's own embedding, support GBDT, field map and candidate radius r_wide from graph/graph_choice.json
# (the operating points 0.005:5, the chosen tau:k and 0.3:1 in one pass). Pass: identical kept pairs at the chosen
# operating point on every event; differing feature values are listed with their distance to a split threshold.
# A Slurm array on the express partition (1 h): task i takes every n-th of the 20 events, writing
# $RUN/gates/g4/task<i>/support_gpu_compare_cuda.json.
# Usage: RUN_SUFFIX=_x1 sbatch --array=0-9 slurm/trackml/gate_g4.sbatch
#SBATCH -J tml_g4
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
require "$RUN/graph_choice.done"
CH=$RUN/graph/graph_choice.json
read -r ML LTF FM RW OP < <(python -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['ml_ckpt'], d['ltf_model'], d.get('field_map') or 'none', d['r_wide'], f\"{d['tau']:g}:{d['k']}\")" "$CH")
FMARG=(); [ "$FM" != none ] && FMARG=(--field-map "$FM")
ALL=$(python -I -c "import sys; print(','.join(sorted(sys.argv[1].split(','), key=int)[:20]))" "$(events dev)")
EV=$(array_slice "$ALL")
OUT=$RUN/gates/g4/task${SLURM_ARRAY_TASK_ID:-0}
python -m hnl.trackml.support_gpu compare --events "$EV" --ml-ckpt "$ML" --r-wide "$RW" --model "$LTF" \
    --ops "0.005:5,$OP,0.3:1" --out "$OUT" --device cuda "${FMARG[@]}"
