#!/bin/bash
# Overnight waiter (login node, light): (1) when baseline arm (ii) seed 2 has finished its 30 GNN epochs, seed its
# acorn continuation run from epochs/epoch29.ckpt and enable the plan line (plan_acorn_long.txt is re-read by its
# orchestrator); (2) once the official working points are committed (results/working_points_v1.json), the e* edge-TPR
# job (edge_tpr_estar.sbatch) for every final run with eval.done (Canto-B / -H / -L), once per run.
# Usage: setsid nohup bash slurm/stage1/night_waiter.sh >> $HNL_LOGS/night_waiter.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO"
D=$HNL_DATA/runs
WP=$HNL_REPO/results/working_points_v1.json
log() { echo "$(date -u +%FT%TZ) $*"; }
log "night waiter start pid $$ on $(hostname)"
acorn_done=0
while true; do
    if [ "$acorn_done" = 0 ] && [ -e "$D/baseline_v1/armii_s2/gnn_train.done" ]; then
        src=$D/baseline_v1/armii_s2/gnn/artifacts/epochs/epoch29.ckpt
        dst=$D/stage1_acorn_long/ignn60/armii_s2/gnn/artifacts
        if [ -e "$src" ]; then
            mkdir -p "$dst" && cp -p "$src" "$dst/last.ckpt"
            sed -i 's/^# ignn60 configs\/baseline\/gnn_train.yaml ii 2 60 .*$/ignn60 configs\/baseline\/gnn_train.yaml ii 2 60/' \
                slurm/stage1/plan_acorn_long.txt
            log "acorn arm (ii) seed 2 finished: continuation seeded from epoch29.ckpt and enabled"
        else
            log "acorn arm (ii) seed 2 finished but $src is missing: not enabled"
        fi
        acorn_done=1
    fi
    if [ -e "$WP" ] && git ls-files --error-unmatch "$WP" > /dev/null 2>&1; then
        for r in stage1_final/l1t stage1_final_union/l1tu stage1_final_learned/l1tl; do
            for s in 0 1 2; do
                F=$D/$r/armi_s$s
                [ -e "$F/eval.done" ] || continue
                [ -e "$F/edge_tpr_estar.submitted" ] && continue
                j=$(sbatch --parsable -J "s1_estar_$(basename $r)_s$s" slurm/stage1/edge_tpr_estar.sbatch "$F" "$WP") \
                    && touch "$F/edge_tpr_estar.submitted" && log "e* edge TPR for $r s$s -> $j"
            done
        done
    fi
    sleep 300
done
