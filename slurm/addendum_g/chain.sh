#!/bin/bash
# Addendum G: submit the whole frozen-model chain of one set with Slurm dependencies (login node, light: sbatch only).
#   acorn arms i (s0-2), ii (s0-2), iii (s0): metric learning + edge classifier          slurm/addendum_g/acorn_infer.sbatch
#   Canto-L (ttbar) s0-2, Canto-L (HNL) s0, Canto-L (ttbar+HNL) s0: union graphs (blocks), scores in parts, packed
#     (gather-scored, which then removes the union-block tars)              canto_union / canto_score / cpu.sbatch
#   evaluation of every model and seed, in blocks of at most 1 h, tracks stored                 evaluate.sbatch
#   summary (hnl.eval.addendum_g summary) and, for val and test, the deviation-15 read-outs (hnl.eval.addendum_g_readouts)
# Steps whose output is complete (done marker) are not resubmitted. Sets other than val: nothing is submitted unless
# slurm/addendum_g/guard.sh passes (ledger entry R14 committed and naming the set's inputs), and every model job checks
# the guard again. The truth-only inputs must exist first: prepare / prepare-inputs (feature-store tars and event
# tables), hnl.eval.addendum_g_truth points (hnl) and fits --list union_test (test read-outs).
# Usage: bash slurm/addendum_g/chain.sh SET [AFTER_JOB]     (SET = val | test | hss | hnl)
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
SET=$1; AFTER=${2:-}
O=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/addendum_g_v1
FS=/ceph/submit/data/user/j/jinboz1/hnl_lrt/derived/acorn_fs/generated_v1
# blocks: union for Canto-L (ttbar) / for Canto-L (HNL, ttbar+HNL) (val: 410 events took 12-30 / 13-53 min); scoring
# parts (val: 1,641 events took 5-9 min of prediction on one GPU); evaluation for Canto-L (ttbar), acorn + LM,
# acorn + CC, Canto-L (HNL / ttbar+HNL); peak storage on /ceph before the union tars are removed (GB, from the val
# outputs: about 0.24 MB per event and acorn run, 0.28 + 0.26 MB per event and Canto-L run)
case $SET in
    val)  UB=4;  UB_TS=4;  NS=1; EB_C=2;  EB_A=2;  EB_CC=1; EB_TS=8;  PEAK=0 ;;
    test) UB=24; UB_TS=48; NS=4; EB_C=8;  EB_A=8;  EB_CC=2; EB_TS=16; PEAK=45 ;;
    hss)  UB=8;  UB_TS=16; NS=1; EB_C=6;  EB_A=6;  EB_CC=2; EB_TS=12; PEAK=12 ;;
    hnl)  UB=24; UB_TS=64; NS=4; EB_C=12; EB_A=12; EB_CC=3; EB_TS=16; PEAK=52 ;;
    *)    echo "unknown set $SET"; exit 1 ;;
esac
if [ "$SET" != val ]; then
    bash slurm/addendum_g/guard.sh "$SET"
    Q=$(getfattr --only-values -n ceph.quota.max_bytes /ceph/submit/data/user/j/jinboz1 2>/dev/null)
    U=$(getfattr --only-values -n ceph.dir.rbytes /ceph/submit/data/user/j/jinboz1 2>/dev/null)
    FREE=$(( (Q - U) / 1000000000 ))
    [ $((FREE - PEAK)) -ge 15 ] || { echo "storage: $FREE GB free on /ceph, the $SET chain needs up to $PEAK GB and the floor is 15 GB"; exit 1; }
    echo "storage: $FREE GB free on /ceph; the $SET chain needs up to $PEAK GB"
fi
case $SET in
    test) NEED="generic_iso_test.tar generic_iso_x1_test.tar generic_dense_test.tar" ;;
    hss)  NEED="hss15_100_all.tar hss40_300_all.tar" ;;
    hnl)  NEED="hnl5_50_all.tar hnl20_200_all.tar hnl15_500_all.tar hnl10_100_ref_all.tar hnl15_200_ref_all.tar" ;;
    *)    NEED="generic_iso_val.tar generic_dense_val.tar" ;;
esac
for t in $NEED; do [ -e "$FS/$t" ] || { echo "missing feature-store tar $FS/$t (prepare / prepare-inputs)"; exit 1; }; done
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
    p=""
    if [ ! -e "$O/${M}_s${S}/$SET/scored/done" ]; then
        u=""
        if ! ls "$O/${M}_s${S}/$SET/union/blocks"/*/done > /dev/null 2>&1 \
                || [ "$(ls "$O/${M}_s${S}/$SET/union/blocks"/*/done | wc -l)" != "$UBM" ]; then
            u=$(sbatch --parsable --array=0-$((UBM - 1)) $(dep "${AJ[$ARM$S]}") -J "g_union_${M}_s${S}_$SET" slurm/addendum_g/canto_union.sbatch "$M" "$S" "$SET" "$UBM")
        fi
        c=$(sbatch --parsable --array=0-$((NS - 1)) $(dep "$u") -J "g_score_${M}_s${S}_$SET" slurm/addendum_g/canto_score.sbatch "$M" "$S" "$SET" "$NS")
        p=$(sbatch --parsable --dependency=afterok:$c -c 2 --mem=8G -J "g_gather_${M}_s${S}_$SET" slurm/addendum_g/cpu.sbatch \
            hnl.eval.addendum_g gather-scored --model "$M" --seed "$S" --set "$SET" $([ "$SET" = val ] || echo --prune-union))
        echo "$M s$S: union ${u:-done} ($UBM blocks), score $c ($NS parts), gather $p"
    fi
    ev "$M" "$S" "$p" "$NB"
done
for S in 0 1 2; do
    ev acorn_hnl_lm "$S" "${AJ[ii$S]}" "$EB_A"
    ev acorn_hnl_cc "$S" "${AJ[ii$S]}" "$EB_CC"
    ev acorn_ttbar_lm "$S" "${AJ[i$S]}" "$EB_A"
done
# evaluation arrays merge in their last block; the summary waits for every evaluation job (afterany: a failed block
# leaves its model out of the summary, which says so); the read-outs (val, test) after the summary
J=$(sbatch --parsable --dependency=afterany$EV -c 4 --mem=32G -J "g_summary_$SET" slurm/addendum_g/cpu.sbatch \
    hnl.eval.addendum_g summary --set "$SET")
echo "summary -> $J"
if [ "$SET" = val ] || [ "$SET" = test ]; then
    R=$(sbatch --parsable --dependency=afterok:$J -c 16 --mem=48G -J "g_readouts_$SET" slurm/addendum_g/cpu.sbatch \
        hnl.eval.addendum_g_readouts --set "$SET" --workers 16)
    echo "read-outs -> $R"
fi
