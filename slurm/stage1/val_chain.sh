#!/bin/bash
# Val chain for listed Stage-1 runs (login node, light: waits and submits; the generic form of final_val_chain.sh).
# Per item TAG:NAME:SEED:GRAPHS_SUBDIR (run RUNS/stage1_TAG/NAME/armi_sSEED, trained on baseline_v1/armi_sSEED/GRAPHS_SUBDIR),
# once its val scores exist (eval.done): GNN scores of its first N_TRAIN (2000) training graphs, then the learned
# fragment-merging model and the val grid of every builder family with it (RUN/trackbuild_val_lmerge).
# Usage: setsid nohup bash slurm/stage1/val_chain.sh ITEM [ITEM ...] >> $HNL_LOGS/val_chain_<name>.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
D=${RUNS_DIR:-$HNL_DATA/runs}
B=$HNL_DATA/runs/baseline_v1
BUILDERS=${BUILDERS:-cc,mwslayer,mwslayermerge,mws2pass,mws2passmerge,mwslmerge}
N_TRAIN=${N_TRAIN:-2000}
log() { echo "$(date -u +%FT%TZ) $*"; }
sub() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY$RANDOM"; echo "  would: sbatch $*" >&2; else sbatch --parsable "$@"; fi; }
TODO="$*"
[ -n "$TODO" ] || { echo "usage: val_chain.sh TAG:NAME:SEED:GRAPHS ..."; exit 1; }
log "val chain start pid $$ on $(hostname); items $TODO; builders $BUILDERS; $N_TRAIN training graphs"
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
log "val chain: all submitted"
