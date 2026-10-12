#!/bin/bash
# S2 (plan section 6.2) for a finished metric learning, as one dependency chain of CPU jobs: radii -> support GBDT ->
# valmon scan (array) -> choice at the plan's budget (gate G1) and, for D3 X2, the same rule at 20 pairs per hit
# (graph/graph_choice_x2.json). The stage scripts are unchanged apart from the run directory (RUN_SUFFIX) and the
# optional budget of graph_choose.sbatch. Light (login node, seconds); prints the job ids.
# Usage: RUN_SUFFIX=_x1 [NTASKS=25] [NICE=200] bash slurm/trackml/s2_chain.sh
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
: "${RUN_SUFFIX?set RUN_SUFFIX (empty for the main run)}"
NTASKS=${NTASKS:-25}
NICE=${NICE:-200}
S=(env -u OMP_NUM_THREADS sbatch --parsable --nice="$NICE" --export="ALL,RUN_SUFFIX=$RUN_SUFFIX")
r=$("${S[@]}" slurm/trackml/graph_radii.sbatch)
s=$("${S[@]}" --dependency=afterok:"$r" slurm/trackml/support_train.sbatch)
c=$("${S[@]}" --dependency=afterok:"$s" --array=0-$((NTASKS - 1)) slurm/trackml/graph_scan.sbatch)
h=$("${S[@]}" --dependency=afterok:"$c" slurm/trackml/graph_choose.sbatch)
RUN=$(source env/activate.sh > /dev/null; python -m hnl.trackml.config get run_root 2>/dev/null | tail -n 1)/seed$(source env/activate.sh > /dev/null; python -m hnl.trackml.config get seed 2>/dev/null | tail -n 1)$RUN_SUFFIX
x=$(env -u OMP_NUM_THREADS sbatch --parsable --nice="$NICE" --export="ALL,RUN_SUFFIX=$RUN_SUFFIX,MAX_PAIRS=20,OUT=$RUN/graph/graph_choice_x2.json" \
    -J tml_choose_x2 --dependency=afterany:"$h" slurm/trackml/graph_choose.sbatch)
echo "radii $r support $s scan $c choose $h choose_x2 $x (run $RUN)"
