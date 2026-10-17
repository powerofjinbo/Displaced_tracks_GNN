#!/bin/bash
# Addendum G step T4: submit one phase of the retrained-model chain on one set with Slurm dependencies (login node,
# light: sbatch only). PHASE pre: acorn (ttbar+G) graphs and edge scores (acorn_infer.sbatch), then the Canto-L
# (ttbar+G) union blocks (canto_union.sbatch) - no metric. PHASE post: Canto-L scores in parts (canto_score.sbatch),
# gather-scored (cpu.sbatch; test: union tars removed after), the evaluation of every run of the selection file
# (evaluate.sbatch: acorn_g, canto_g and the guarded alternatives) and the summary (claims G2, G3). Steps whose output is
# complete are not resubmitted. Test: nothing is submitted unless slurm/addendum_g_t4/guard.sh passes for the phase,
# and every job checks it again. Storage: refused unless /ceph keeps 15 GB after the phase's peak; files: refused if
# hnl_lrt would pass 79,000. Jobs at --nice=$NICE (default 50: the T3 tail goes first).
# Usage: bash slurm/addendum_g_t4/chain.sh SET PHASE [AFTER_JOB] [SEEDS]   (SET val|test; val: SELECTION, CANTO_CKPT)
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
source env/activate.sh > /dev/null
SET=$1; PHASE=$2; AFTER=${3:-}; SEEDS=${4:-0 1 2}
NICE=${NICE:-50}
O=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/addendum_g_t4_v1
# blocks per seed: union (about 320 events per 1-h block at 10 workers), scoring parts, evaluation blocks; peak
# storage of the phase in GB (about 0.3 MB per event for each of the acorn graphs, the union graphs and the scored
# union graphs; the union tars are removed after gather-scored on test) and new files
case $SET in
    val)  UB=6;  NS=1; EB=2;  PEAK_PRE=4;  PEAK_POST=3;  FILES_NEW=300 ;;
    test) UB=32; NS=4; EB=12; PEAK_PRE=18; PEAK_POST=10; FILES_NEW=1500 ;;
    *)    echo "unknown set $SET"; exit 1 ;;
esac
EB=${EB_OVERRIDE:-$EB}           # evaluation blocks per run (EB_OVERRIDE=3: fewer files under hnl_lrt; same outputs)
[ "$PHASE" = post ] && FILES_NEW=$((60 + 4 * 3 * (5 * EB + 3)))   # post: scoring and gather, then 5 files per block + 3 merged per run, up to 4 runs x 3 seeds
bash slurm/addendum_g_t4/guard.sh "$PHASE" "$SET"
PEAK=$([ "$PHASE" = pre ] && echo $PEAK_PRE || echo $PEAK_POST)
Q=$(getfattr --only-values -n ceph.quota.max_bytes /ceph/submit/data/user/j/jinboz1 2>/dev/null)
U=$(getfattr --only-values -n ceph.dir.rbytes /ceph/submit/data/user/j/jinboz1 2>/dev/null)
FREE=$(( (Q - U) / 1000000000 ))
[ $((FREE - PEAK)) -ge 15 ] || { echo "storage: $FREE GB free on /ceph, $SET $PHASE needs up to $PEAK GB, floor 15 GB"; exit 1; }
NF=$(getfattr --only-values -n ceph.dir.rfiles /ceph/submit/data/user/j/jinboz1/hnl_lrt 2>/dev/null)
[ $((NF + FILES_NEW)) -le 79000 ] || { echo "files: hnl_lrt holds $NF files, $SET would add about $FILES_NEW"; exit 1; }
echo "storage: $FREE GB free, peak $PEAK GB; hnl_lrt files $NF (+~$FILES_NEW)"
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }
SB=(sbatch --parsable --nice="$NICE")
ENV=()
[ -n "${SELECTION:-}" ] && ENV+=(SELECTION="$SELECTION")
[ -n "${CANTO_CKPT:-}" ] && ENV+=(CANTO_CKPT="$CANTO_CKPT")
if [ "$SET" != val ] && [ ${#ENV[@]} -gt 0 ]; then echo "SELECTION / CANTO_CKPT are for val code tests only"; exit 1; fi
if [ "$PHASE" = pre ]; then
    for S in $SEEDS; do
        a=""
        if [ ! -e "$O/acorn_g_s$S/$SET/done" ]; then
            a=$("${SB[@]}" $(dep "$AFTER") -J "g4_acorn_s${S}_$SET" slurm/addendum_g_t4/acorn_infer.sbatch "$S" "$SET")
        fi
        n=$( (ls "$O/canto_g_s$S/$SET/union/blocks"/*/done 2>/dev/null || true) | wc -l)
        u=""
        if [ "$n" != "$UB" ]; then
            u=$("${SB[@]}" --array=0-$((UB - 1)) $(dep "${a:-$AFTER}") -J "g4_union_s${S}_$SET" slurm/addendum_g_t4/canto_union.sbatch "$S" "$SET" "$UB")
        fi
        echo "seed $S: acorn ${a:-done}, union ${u:-done} ($UB blocks)"
    done
    exit 0
fi
RUNS=$(env "${ENV[@]}" HNL_GENERATED=1 python -I -m hnl.eval.addendum_g_t4 runs ${SELECTION:+--selection "$SELECTION"})
echo "evaluation runs: $RUNS"
EV=""
for S in $SEEDS; do
    p=""
    if [ ! -e "$O/canto_g_s$S/$SET/scored/done" ]; then
        c=$(env "${ENV[@]}" "${SB[@]}" --array=0-$((NS - 1)) $(dep "$AFTER") -J "g4_score_s${S}_$SET" slurm/addendum_g_t4/canto_score.sbatch "$S" "$SET" "$NS")
        p=$("${SB[@]}" --dependency=afterok:$c -c 2 --mem=8G -J "g4_gather_s${S}_$SET" slurm/addendum_g_t4/cpu.sbatch post "$SET" \
            hnl.eval.addendum_g_t4 gather-scored --seed "$S" --set "$SET" $([ "$SET" = val ] || echo --prune-union))
        echo "seed $S: score $c ($NS parts), gather $p"
    fi
    for R in $RUNS; do
        d=$AFTER; case $R in canto*) d=${p:-$AFTER} ;; esac
        j=$(env "${ENV[@]}" "${SB[@]}" --array=0-$((EB - 1)) $(dep "$d") -J "g4_eval_${R}_s${S}_$SET" slurm/addendum_g_t4/evaluate.sbatch "$R" "$S" "$SET" "$EB")
        EV="$EV:$j"
        echo "seed $S: eval $R ($EB blocks) -> $j"
    done
done
SEEDARG=$(echo $SEEDS | tr ' ' ',')
J=$("${SB[@]}" --dependency=afterany$EV -c 4 --mem=48G -J "g4_summary_$SET" slurm/addendum_g_t4/cpu.sbatch post "$SET" \
    hnl.eval.addendum_g_t4 summary --set "$SET" --seeds "$SEEDARG" ${SELECTION:+--selection "$SELECTION"})
echo "summary -> $J"
