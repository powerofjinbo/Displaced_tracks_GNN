#!/bin/bash
# Addendum G: submit the whole frozen-model chain of one set with Slurm dependencies (login node, light: sbatch only).
#   acorn arms i (s0-2), ii (s0-2), iii (s0): metric learning + edge classifier          slurm/addendum_g/acorn_infer.sbatch
#   Canto-L (ttbar) s0-2, Canto-L (HNL) s0, Canto-L (ttbar+HNL) s0: union graphs (blocks), scores  canto_union / canto_score
#   evaluation of every model and seed, in blocks of at most 1 h                                 evaluate.sbatch
#   summary                                                                                        hnl.eval.addendum_g summary
# Steps whose output is complete (done marker) are not resubmitted.
# Usage: bash slurm/addendum_g/chain.sh SET [AFTER_JOB]     (SET = val | test | hss)
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
SET=$1; AFTER=${2:-}
O=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/addendum_g_v1
case $SET in   # blocks: union for Canto-L (ttbar) / for Canto-L (HNL, ttbar+HNL) (val: 410 events took 12 / 25-53 min);
               # evaluation for Canto-L (ttbar), acorn + LM, acorn + CC, Canto-L (HNL / ttbar+HNL)
    val)  UB=4; UB_TS=4; EB_C=2; EB_A=2; EB_CC=1; EB_TS=8 ;;
    test) UB=12; UB_TS=24; EB_C=8; EB_A=8; EB_CC=2; EB_TS=16 ;;
    hss)  UB=8; UB_TS=16; EB_C=6; EB_A=6; EB_CC=2; EB_TS=12 ;;
esac
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }
declare -A AJ
for spec in i:0 i:1 i:2 ii:0 ii:1 ii:2 iii:0; do
    IFS=: read -r ARM S <<< "$spec"
    if [ -e "$O/acorn_${ARM}_s${S}/$SET/done" ]; then AJ[$ARM$S]=""; continue; fi
    AJ[$ARM$S]=$(sbatch --parsable $(dep "$AFTER") -J "g_acorn_${ARM}_s${S}_$SET" slurm/addendum_g/acorn_infer.sbatch "$ARM" "$S" "$SET")
    echo "acorn $ARM s$S -> ${AJ[$ARM$S]}"
done
EV=""
ev() {   # MODEL SEED DEPJOB NBLOCKS
    local nb=$4 j
    j=$(sbatch --parsable --array=0-$((nb - 1)) $(dep "$3") -J "g_eval_${1}_s${2}_$SET" slurm/addendum_g/evaluate.sbatch "$1" "$2" "$SET" "$nb")
    EV="$EV:$j"
    echo "eval $1 s$2 ($nb blocks) -> $j"
}
for spec in canto_l_ttbar:0:i:$EB_C:$UB canto_l_ttbar:1:i:$EB_C:$UB canto_l_ttbar:2:i:$EB_C:$UB \
            canto_l_hnl:0:ii:$EB_TS:$UB_TS canto_l_tthnl:0:iii:$EB_TS:$UB_TS; do
    IFS=: read -r M S ARM NB UBM <<< "$spec"
    u=""; c=""
    if [ ! -e "$O/${M}_s${S}/$SET/scored/done" ]; then
        u=$(sbatch --parsable --array=0-$((UBM - 1)) $(dep "${AJ[$ARM$S]}") -J "g_union_${M}_s${S}_$SET" slurm/addendum_g/canto_union.sbatch "$M" "$S" "$SET" "$UBM")
        c=$(sbatch --parsable --dependency=afterok:$u -J "g_score_${M}_s${S}_$SET" slurm/addendum_g/canto_score.sbatch "$M" "$S" "$SET")
        echo "$M s$S: union $u ($UBM blocks), score $c"
    fi
    ev "$M" "$S" "$c" "$NB"
done
for S in 0 1 2; do
    ev acorn_hnl_lm "$S" "${AJ[ii$S]}" "$EB_A"
    ev acorn_hnl_cc "$S" "${AJ[ii$S]}" "$EB_CC"
    ev acorn_ttbar_lm "$S" "${AJ[i$S]}" "$EB_A"
done
# evaluation arrays merge in their last block; the summary waits for every evaluation job (afterany: a failed block
# leaves its model out of the summary, which says so)
J=$(sbatch --parsable --dependency=afterany$EV -p submit-gpu-express,submit,submit-gpu -c 4 --mem=32G -t 01:00:00 \
    -J "g_summary_$SET" --export=ALL,HNL_GENERATED=1 slurm/run_py.sbatch hnl.eval.addendum_g summary --set "$SET")
echo "summary -> $J"
