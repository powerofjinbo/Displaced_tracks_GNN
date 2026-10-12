#!/bin/bash
# S2d (plan section 6.2): the graph configuration from the valmon scan (highest union target ceiling at <= 16.5 pairs per
# hit; ceilings within 0.05 point tied -> fewer pairs), FRNN-cap check and gate G1 (ceiling >= 0.990; otherwise D3:
# X1, then X2) -> graph/graph_choice.json. Light (seconds). D3 X2: MAX_PAIRS=20 OUT=<file> applies the same rule with
# a budget of 20 pairs per hit and writes OUT only (graph_choice.done is written only for graph/graph_choice.json).
# Usage: [SEED=0] [RUN_SUFFIX=_x1] [MAX_PAIRS=20 OUT=...] sbatch slurm/trackml/graph_choose.sbatch
#SBATCH -J tml_choose
#SBATCH -p submit
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
mkdir -p "$RUN/graph/scan/all"
python -I -c "
import glob, sys, pandas as pd, shutil, os
parts = sorted(glob.glob(os.path.join(sys.argv[1], 'part*', 'per_event.parquet')))
pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(os.path.join(sys.argv[1], 'all', 'per_event.parquet'))
shutil.copy(os.path.join(os.path.dirname(parts[0]), 'scan.json'), os.path.join(sys.argv[1], 'all', 'scan.json'))
print(len(parts), 'parts')" "$RUN/graph/scan"
OUT=${OUT:-$RUN/graph/graph_choice.json}
python -m hnl.trackml.graphs choose --scan "$RUN/graph/scan/all" --out "$OUT" ${MAX_PAIRS:+--max-pairs-per-hit "$MAX_PAIRS"}
if python -I -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['gate_g1_pass'] and d['cap_ok'] else 3)" "$OUT"; then
    if [ "$OUT" = "$RUN/graph/graph_choice.json" ]; then touch "$RUN/graph_choice.done"; fi
    echo "gate G1 passed at <= ${MAX_PAIRS:-$(cfg graph.union_max_pairs_per_hit)} pairs per hit ($OUT)"
else
    echo "GATE G1 or FRNN cap check failed: stop and report"
fi
