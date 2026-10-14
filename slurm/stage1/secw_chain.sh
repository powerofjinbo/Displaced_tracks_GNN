#!/bin/bash
# Read-out chain of the secondary-particle upweighting test (docs/secondary_upweight_plan.md, section 4; val only).
# Login node, light: waits and submits. For secw18 and secwctrl, once RUN/eval.done exists (training by
# slurm/stage1/express_driver.sh with SPEC=slurm/stage1/express_runs_secw.txt):
#   base graphs  slurm/stage1/trackbuild.sbatch RUN walkmergeS,mwslayer,mwslayermerge trackbuild_val
#   union graphs slurm/stage1/score_graphs.sbatch RUN <arm i s0 ltf_union_graphs valset.tar> RUN/ltf_union_preview
#                mwslayermerge (CPU scoring, as the reference's preview), then trackbuild.sbatch on it with
#                walkmergeX,mwslayermergeX,mws2passmergeX -> trackbuild_val_ext
#   dD           slurm/stage1/edge_tpr_estar.sbatch RUN 0.0122
# (the reference's rebuild under runs/stage1_secw/ref: jobs in runs/stage1_secw/ref_jobs.txt). When every output
# exists: hnl.eval.oblique_val on the base and on the union scores of the three runs, then hnl.eval.secondary_upweight.
# State: $HNL_DATA/runs/stage1_secw/_chain. Usage: setsid nohup bash slurm/stage1/secw_chain.sh >> $HNL_LOGS/secw_chain.log 2>&1 &
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd "$HNL_REPO" || exit 1
B=$HNL_DATA/runs/stage1_secw
ST=$B/_chain
LTF=$HNL_DATA/runs/baseline_v1/armi_s0/ltf_union_graphs/valset.tar
mkdir -p "$ST"
log() { echo "$(date -u +%FT%TZ) $*"; }
# CPU jobs: never on the A30 or GTX 1080 nodes (the --exclude here replaces each script's own list)
EXCL=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
sub() { env -u OMP_NUM_THREADS -u NUMBA_NUM_THREADS -u LIMIT_TRAIN -u S1TAG sbatch --parsable -p submit-gpu,submit --exclude=$EXCL "$@"; }
once() {  # KEY SBATCH-ARGS... : submit once, job id kept in $ST/KEY
    local k=$1; shift
    [ -s "$ST/$k" ] && { cat "$ST/$k"; return 0; }
    local j; j=$(sub "$@") && [[ "$j" =~ ^[0-9]+$ ]] || { log "sbatch failed for $k: $j" >&2; return 1; }
    echo "$j" > "$ST/$k"; log "$k -> $j" >&2; echo "$j"
}
log "secw chain start pid $$ on $(hostname)"
while true; do
    [ -e "$ST/STOP" ] && { log "STOP"; exit 0; }
    for n in secw18 secwctrl; do
        R=$B/$n/armi_s0
        [ -e "$R/eval.done" ] || continue
        once "$n.tb" -J "secw_tb_$n" slurm/stage1/trackbuild.sbatch "$R" walkmergeS,mwslayer,mwslayermerge trackbuild_val > /dev/null
        s=$(once "$n.score" -J "secw_score_$n" slurm/stage1/score_graphs.sbatch "$R" "$LTF" "$R/ltf_union_preview" mwslayermerge) || continue
        once "$n.tbx" -J "secw_tbx_$n" --dependency=afterok:$s slurm/stage1/trackbuild.sbatch "$R/ltf_union_preview" \
            walkmergeX,mwslayermergeX,mws2passmergeX trackbuild_val_ext > /dev/null
        once "$n.etpr" -J "secw_etpr_$n" -t 01:30:00 slurm/stage1/edge_tpr_estar.sbatch "$R" 0.0122 > /dev/null
    done
    ready=1
    for n in ref secw18 secwctrl; do
        R=$B/$n/armi_s0
        for f in trackbuild_val.done ltf_union_preview/trackbuild_val_ext.done edge_tpr_estar_efpr.json; do
            [ -e "$R/$f" ] || ready=0
        done
    done
    if [ "$ready" = 1 ]; then
        a=$(once obl_base -J secw_obl_base slurm/feedback/oblique_val.sbatch secondary_upweight_v1_oblique_base \
            ref=$B/ref/armi_s0:i secwctrl=$B/secwctrl/armi_s0:i secw18=$B/secw18/armi_s0:i)
        b=$(once obl_union -J secw_obl_union slurm/feedback/oblique_val.sbatch secondary_upweight_v1_oblique_union \
            ref=$B/ref/armi_s0/ltf_union_preview:i secwctrl=$B/secwctrl/armi_s0/ltf_union_preview:i \
            secw18=$B/secw18/armi_s0/ltf_union_preview:i)
        once readout -J secw_readout --dependency=afterok:$a:$b -c 8 --mem=16G -t 01:00:00 -o "$HNL_LOGS/%x_%j.out" \
            --wrap "source $HNL_REPO/env/activate.sh && cd $HNL_REPO && python -m hnl.eval.secondary_upweight --out results --name secondary_upweight_v1" > /dev/null
        log "all read-out jobs submitted (oblique $a $b)"
        exit 0
    fi
    sleep 300
done
