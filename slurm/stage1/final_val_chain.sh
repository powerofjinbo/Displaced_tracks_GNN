#!/bin/bash
# Final val chain (login node, light: waits and submits). For each full-budget run of the final plans (S1TAG final:
# base graphs; final_union: HelixFilter union graphs), once its val scores exist (eval.done): GNN scores of its first
# 2000 training graphs (the graphs it was trained on: ttbar only for arm (i)), then the learned fragment-merging model
# from them and the val grid of every builder family with that model (RUN/trackbuild_val_lmerge). No selection here:
# that needs the committed working points (rules R6 / R8, DECISIONS.md). DRY_RUN=1 prints instead of submitting.
# Usage: setsid nohup bash slurm/stage1/final_val_chain.sh >> $HNL_LOGS/final_val_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
D=${RUNS_DIR:-$HNL_DATA/runs}
B=$HNL_DATA/runs/baseline_v1
BUILDERS=${BUILDERS:-cc,mwslayer,mwslayermerge,mws2pass,mws2passmerge,mwslmerge}
N_TRAIN=${N_TRAIN:-2000}
log() { echo "$(date -u +%FT%TZ) $*"; }
sub() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; else sbatch --parsable "$@"; fi; }
TODO=""
for s in 0 1 2; do TODO="$TODO final:l1t:$s:ml_graphs final_union:l1tu:$s:union_graphs"; done
log "final val chain start pid $$ on $(hostname); builders $BUILDERS; $N_TRAIN training graphs"
while [ -n "$TODO" ]; do
    left=""
    for item in $TODO; do
        IFS=: read -r tag name seed graphs <<< "$item"
        F=$D/stage1_$tag/$name/armi_s$seed
        if [ -e "$F/eval.done" ] && [ -e "$F/gnn_scores/valset.tar" ]; then
            if [ -e "$F/trackbuild_val_lmerge.done" ]; then log "$name s$seed: val grid already done"; continue; fi
            j1=$(N_EVENTS=$N_TRAIN sub -J "s1_scoretr_${name}_s$seed" slurm/stage1/score_split.sbatch "$F" "$B/armi_s$seed/$graphs" train)
            j2=$(sub --dependency=afterok:$j1 -J "s1_lmerge_${name}_s$seed" slurm/stage1/learned_merge.sbatch "$F" "$BUILDERS")
            log "$name s$seed eval.done -> train scores $j1, learned merging + val grid $j2"
        else
            left="$left $item"
        fi
    done
    TODO=$(echo $left)
    [ -n "$TODO" ] && sleep "${POLL:-300}"
done
log "final val chain: all submitted"
