#!/bin/bash
# Seeds 1 and 2 of Canto-L (HNL) and Canto-L (ttbar+HNL) on the generated samples of addendum G step T2 (ledger R20;
# post-freeze, descriptive): the T2 chain of slurm/addendum_g/chain.sh for these four runs (and acorn (ttbar+HNL) s1,
# s2, which Canto-L (ttbar+HNL) s1, s2 build their union graphs from), with the output root ADDG_OUT on /work
# (g_env.sh) and the removal of each intermediate file once the next stage has passed its check
# (hnl.eval.addendum_g_seeds12 cleanup; the first author's permission of 2026-10-17, scope in ledger R20). Login node,
# light: links and sbatch only. Per (model, seed, set): [acorn (ttbar+HNL) inference] -> union blocks -> scores in parts
# -> gather (packs the parts, removes the part tars and the union-block tars, as R14) -> evaluation blocks (last one
# merges) -> cleanup of the scored tar [and of the acorn tar after the gather]. Items run in waves (each wave after the
# cleanups of the one before) so that the peak on /work stays below about 20 GB; wave 1 waits for the removal of the
# HNL-test intermediates of slurm/ts_seeds12/chain.sh (after the R19 read-outs). After the last wave: the summary
# (hnl.eval.addendum_g_seeds12 summarize). Dependent jobs carry --kill-on-invalid-dep=yes.
# Usage: [NICE=20] bash slurm/ts_seeds12/g_chain.sh            (test, hss, hnl; ledger R20 must be committed)
#        bash slurm/ts_seeds12/g_chain.sh val                  (pipeline check on the generic val split, Canto-L (HNL)
#                                                               s2 only; no guard needed for val)
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
source env/activate.sh > /dev/null
source slurm/ts_seeds12/g_env.sh
MODE=${1:-all}
R=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/addendum_g_v1
NICE=${NICE:-20}
CPU_EXC=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
SB() { env -u OMP_NUM_THREADS -u NUMBA_NUM_THREADS "$@"; }
log() { echo "$(date -u +%FT%TZ) $*" >&2; }
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1 --kill-on-invalid-dep=yes" || true; }
link() {
    if [ -L "$2" ]; then [ "$(readlink "$2")" = "$1" ] || { log "$2 points elsewhere"; exit 1; }
    elif [ -e "$2" ]; then log "$2 exists and is not a link"; exit 1
    else ln -s "$1" "$2"; fi
}
mkdir -p "$ADDG_OUT/eval"
for s in 1 2; do link "$R/acorn_ii_s$s" "$ADDG_OUT/acorn_ii_s$s"; done
for d in "$R"/eval/*_s[0-9]; do link "$d" "$ADDG_OUT/eval/$(basename "$d")"; done
blocks() { case $1 in val) echo "4 1 8" ;; test) echo "48 4 16" ;; hss) echo "16 1 12" ;; hnl) echo "64 4 16" ;; esac; }
item() {   # MODEL SEED SET WAVE_DEP -> prints the ids that the next wave waits for
    local M=$1 S=$2 SET=$3 D=$4 A="" U C G E CS CA="" ARM=ii MEM=32G UB NS EB
    read -r UB NS EB <<< "$(blocks "$SET")"
    if [ "$M" = canto_l_tthnl ]; then ARM=iii; MEM=40G; fi
    if [ "$ARM" = iii ] && [ ! -e "$ADDG_OUT/acorn_iii_s$S/$SET/done" ]; then
        A=$(SB sbatch --parsable --nice="$NICE" $(dep "$D") -J "g12_acorn_iii_s${S}_$SET" slurm/ts_seeds12/g_acorn.sbatch iii "$S" "$SET")
    fi
    U=$(SB sbatch --parsable --nice="$NICE" $(dep "${A:-$D}") --array=0-$((UB - 1)) -c 16 --mem=$MEM --exclude=$CPU_EXC \
        -J "g12_union_${M}_s${S}_$SET" slurm/ts_seeds12/g_union.sbatch "$M" "$S" "$SET" "$UB")
    C=$(SB sbatch --parsable --nice="$NICE" $(dep "$U") --array=0-$((NS - 1)) -J "g12_score_${M}_s${S}_$SET" \
        slurm/ts_seeds12/g_score.sbatch "$M" "$S" "$SET" "$NS")
    G=$(SB sbatch --parsable --nice="$NICE" $(dep "$C") -c 2 --mem=8G --exclude=$CPU_EXC -J "g12_gather_${M}_s${S}_$SET" \
        slurm/ts_seeds12/g_cpu.sbatch hnl.eval.addendum_g gather-scored --model "$M" --seed "$S" --set "$SET" --prune-union)
    E=$(SB sbatch --parsable --nice="$NICE" $(dep "$G") --array=0-$((EB - 1)) --exclude=$CPU_EXC -J "g12_eval_${M}_s${S}_$SET" \
        slurm/ts_seeds12/g_eval.sbatch "$M" "$S" "$SET" "$EB")
    CS=$(SB sbatch --parsable --nice="$NICE" $(dep "$E") -c 2 --mem=8G --exclude=$CPU_EXC -J "g12_clean_${M}_s${S}_$SET" \
         slurm/ts_seeds12/g_cpu.sbatch hnl.eval.addendum_g_seeds12 cleanup scored --model "$M" --seed "$S" --set "$SET")
    if [ "$ARM" = iii ]; then
        CA=$(SB sbatch --parsable --nice="$NICE" $(dep "$G") -c 2 --mem=8G --exclude=$CPU_EXC -J "g12_clean_acorn_s${S}_$SET" \
             slurm/ts_seeds12/g_cpu.sbatch hnl.eval.addendum_g_seeds12 cleanup acorn --seed "$S" --set "$SET")
    fi
    log "$M s$S $SET: acorn ${A:--} -> union $U ($UB) -> scores $C ($NS) -> gather $G -> eval $E ($EB) -> cleanup $CS ${CA:+/ acorn cleanup $CA}"
    echo "$CS${CA:+:$CA}"
}
if [ "$MODE" = val ]; then
    item canto_l_hnl 2 val "" > /dev/null
    exit 0
fi
for s in test hss hnl; do bash slurm/ts_seeds12/g_guard.sh "$s" >&2; done
log "g_chain start at $(git rev-parse --short HEAD); ADDG_OUT $ADDG_OUT; nice $NICE"
# wave 0: the HNL-test intermediates of chain.sh, after the R19 read-outs (6849677-6849679)
W=""
for spec in ii:1 ii:2 iii:1 iii:2; do
    IFS=: read -r A S <<< "$spec"
    j=$(SB sbatch --parsable --nice="$NICE" $(dep "${R19_READOUTS:-6849677:6849678:6849679}") -c 2 --mem=8G --exclude=$CPU_EXC \
        -J "g12_clean_testrun_${A}_s$S" slurm/ts_seeds12/g_cpu.sbatch hnl.eval.addendum_g_seeds12 cleanup testrun --arm "$A" --seed "$S")
    W="$W:$j"
done
W=${W#:}; log "wave 0 (HNL-test intermediates): $W"
WAVES=("test:canto_l_hnl:1 test:canto_l_hnl:2"
       "test:canto_l_tthnl:1 hss:canto_l_hnl:1 hss:canto_l_hnl:2"
       "test:canto_l_tthnl:2 hss:canto_l_tthnl:1 hss:canto_l_tthnl:2"
       "hnl:canto_l_hnl:1 hnl:canto_l_hnl:2"
       "hnl:canto_l_tthnl:1"
       "hnl:canto_l_tthnl:2")
k=1
for wave in "${WAVES[@]}"; do
    N=""
    for it in $wave; do
        IFS=: read -r SET M S <<< "$it"
        N="$N:$(item "$M" "$S" "$SET" "$W")"
    done
    W=${N#:}; log "wave $k done when $W"; k=$((k + 1))
done
J=$(SB sbatch --parsable --nice="$NICE" $(dep "$W") -c 8 --mem=48G -t 01:00:00 --exclude=$CPU_EXC -J g12_summarize \
    slurm/ts_seeds12/g_cpu.sbatch hnl.eval.addendum_g_seeds12 summarize --out results --name addendum_g_canto_hnl_seeds12_v1)
log "summary -> $J"
