#!/bin/bash
# Canto-L chain (login node, light: waits and submits): once seed 0's learned-support union graphs exist, the preview
# (the screening L1 + trunc classifier scoring them, val builder grid; against the hand-cut union preview); once all
# three seeds' graphs exist, the full-budget Canto-L orchestrator (plan_final_learned.txt, S1TAG=final_learned,
# GRAPHS_SUBDIR=ltf_union_graphs, nice 5) and its val chain (slurm/stage1/val_chain.sh).
# Usage: setsid nohup bash slurm/stage1/canto_l_chain.sh >> $HNL_LOGS/canto_l_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
B=$HNL_DATA/runs/baseline_v1
SCREEN=$HNL_DATA/runs/stage1_screen_ttbar/symregion_trunc/armi_s0
log() { echo "$(date -u +%FT%TZ) $*"; }
log "canto-L chain start pid $$ on $(hostname)"
preview=0
while true; do
    if [ "$preview" = 0 ] && [ -e "$B/armi_s0/ltf_union_graphs.done" ]; then
        j=$(sbatch --parsable -J s1_ltf_preview slurm/stage1/score_graphs.sbatch "$SCREEN" \
            "$B/armi_s0/ltf_union_graphs/valset.tar" "$SCREEN/ltf_union_preview")
        log "seed 0 learned-support union graphs ready -> preview $j"
        preview=1
    fi
    n=0
    for s in 0 1 2; do [ -e "$B/armi_s$s/ltf_union_graphs.done" ] && n=$((n + 1)); done
    if [ "$n" = 3 ]; then
        mkdir -p "$HNL_DATA/runs/stage1_final_learned/_orchestrator"
        echo "walkmergeS,mwslayer,mwslayermerge" > "$HNL_DATA/runs/stage1_final_learned/_orchestrator/builders"
        S1TAG=final_learned GRAPHS_SUBDIR=ltf_union_graphs S1_NICE=5 setsid nohup bash slurm/stage1/orchestrate_s1.sh \
            slurm/stage1/plan_final_learned.txt >> "$HNL_LOGS/orchestrate_s1_final_learned.log" 2>&1 < /dev/null &
        log "all three seeds' learned-support union graphs ready -> Canto-L orchestrator started"
        setsid nohup bash slurm/stage1/val_chain.sh final_learned:l1tl:0:ltf_union_graphs \
            final_learned:l1tl:1:ltf_union_graphs final_learned:l1tl:2:ltf_union_graphs \
            >> "$HNL_LOGS/val_chain_l1tl.log" 2>&1 < /dev/null &
        log "Canto-L val chain started"
        break
    fi
    sleep 120
done
log "canto-L chain done"
