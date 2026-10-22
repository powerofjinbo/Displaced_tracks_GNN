#!/bin/bash
# Seeds 1 and 2 of Canto-L (HNL) and Canto-L (ttbar+HNL) on the test sets of seed 0's post-freeze read-out
# (docs/feedback/test_use_ledger.md R19; seed 0: R11): seed 0's chain, unchanged scripts, with every output under the
# selection's out_root (/work/submit/jinboz1/hnl/runs_extra/ts_seeds12; the hnl_lrt file budget). Login node, light:
# symbolic links and sbatch only. Per run (arm A in ii, iii; seed S in 1, 2), as seed 0 (DECISIONS 2026-10-12;
# jobs 6794043-6794046, 6802510-6802515, 6808893-6808894):
#   mirrors   graphs_run = OUT/baseline_v1_ts/armA_sS: links ml, ml_graphs, gnn_scores_test, test_eval.done to
#             runs/baseline_v1/armA_sS; run = OUT/l1tl/armA_sS: links gnn, lmerge to the trained run (read only)
#   union     slurm/stage1/union_graphs.sbatch graphs_run test with the run's learned triplet filter (TRIPLET_MODEL,
#             TAU, KTOP of the selection = its report_val); arm (iii) through with_workers.sbatch (WORKERS 18, 28 CPUs,
#             56 GB), arm (ii) 24 CPUs and 48 GB, as seed 0
#   scores    slurm/stage1/score_split.sbatch run graphs_run/ltf_union_graphs test on an A30 (ACCEL=gpu, as seed 0)
#   build     slurm/stage1/final_test_shard.sbatch SEL l1tl_A S fr|llp NB (NB contiguous blocks of the sorted scored
#             graphs, 16 CPUs, 1 h), then slurm/stage1/final_test_merge.sbatch SEL l1tl_A S NB NB
#   read-out  after the four merges: hnl.eval.training_sample_seeds12 readout; hnl.eval.bootstrap_tables
#             --extra-test-spec SPEC (pooled HNL rows, |d0| and r_v bins); then hnl.eval.training_sample_paired_test
#             with SPEC and those bootstrap rows
# Guard (nothing is submitted otherwise): the selection file, the spec and the ledger committed unchanged, ledger entry
# R19 committed and naming the selection file; the checkpoints, triplet filters and merge models match the selection's
# sha256. Dependent jobs carry --kill-on-invalid-dep=yes (no job can be left waiting on a failed one). Steps whose
# done marker exists are not resubmitted.
# Usage: [NICE=20] [NB=24] bash slurm/ts_seeds12/chain.sh            (log: /work/submit/jinboz1/hnl/logs/ts12_chain.log)
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
source env/activate.sh > /dev/null
SEL=results/training_sample_test_selection_seeds12_v1.json
SPEC=results/training_sample_spec_test_3seed_v1.json
LEDGER=docs/feedback/test_use_ledger.md
NICE=${NICE:-20}; NB=${NB:-24}
CPU_EXC=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
GPU_EXC=submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
SB() { env -u OMP_NUM_THREADS -u NUMBA_NUM_THREADS "$@"; }
log() { echo "$(date -u +%FT%TZ) $*"; }
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1 --kill-on-invalid-dep=yes" || true; }
for f in "$SEL" "$SPEC" "$LEDGER"; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
        || { log "$f is not committed as is: nothing submitted"; exit 1; }
done
R19=$(git show "HEAD:$LEDGER" | awk '/^### R19\. /{on=1} on && /^(### R[0-9]+\.|## [0-9])/ && !/^### R19\. /{on=0} on')
grep -qF "$SEL" <<< "$R19" || { log "ledger entry R19 not committed or not naming $SEL: nothing submitted"; exit 1; }
python -I - "$SEL" <<'EOF' || { log "selection inputs differ from their sha256: nothing submitted"; exit 1; }
import hashlib, json, sys
s = json.load(open(sys.argv[1]))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for p in s["pipelines"].values():
    for d in p["seeds"].values():
        for k in ("gnn_ckpt", "triplet_model", "merge_model"):
            if sha(d[k]) != d[f"{k}_sha256"]:
                sys.exit(f"{d[k]}: sha256 differs from the selection")
EOF
log "chain start at $(git rev-parse --short HEAD); selection $SEL ($(sha256sum "$SEL" | cut -c1-16)); NB $NB; nice $NICE"
ITEMS=$(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))
for p, d in s['pipelines'].items():
    for seed, e in sorted(d['seeds'].items()):
        print(p, seed, e['run'], e['source_run'], e['graphs_run'], e['ml_run'], e['triplet_model'], e['tau'], e['k'])" "$SEL")
link() {   # TARGET LINK: a symbolic link, or the same one already there
    if [ -L "$2" ]; then [ "$(readlink "$2")" = "$1" ] || { log "$2 points to $(readlink "$2"), not $1"; exit 1; }
    elif [ -e "$2" ]; then log "$2 exists and is not a link"; exit 1
    else ln -s "$1" "$2"; fi
}
MERGES=""
while read -r P S RUN SRC GRUN MLRUN TM TAU K; do
    ARM=${P#l1tl_}
    mkdir -p "$GRUN" "$RUN"
    for x in ml ml_graphs gnn_scores_test test_eval.done; do link "$MLRUN/$x" "$GRUN/$x"; done
    for x in gnn lmerge; do link "$SRC/$x" "$RUN/$x"; done
    if [ -e "$RUN/final_test.done" ]; then log "$P s$S: test evaluation done already"; continue; fi
    U=""
    if [ ! -e "$GRUN/ltf_union_graphs_test.done" ]; then
        if [ "$ARM" = iii ]; then
            U=$(SB TRIPLET_MODEL="$TM" TAU="$TAU" KTOP="$K" SUBDIR=ltf_union_graphs WORKERS=18 sbatch --parsable \
                --nice="$NICE" -p submit-gpu,submit -c 28 --mem=56G -t 08:00:00 --exclude=$CPU_EXC \
                -J "ts12_union_${ARM}_s$S" slurm/stage1/with_workers.sbatch slurm/stage1/union_graphs.sbatch "$GRUN" test)
        else
            U=$(SB TRIPLET_MODEL="$TM" TAU="$TAU" KTOP="$K" SUBDIR=ltf_union_graphs sbatch --parsable \
                --nice="$NICE" -p submit-gpu,submit -c 24 --mem=48G -t 06:00:00 --exclude=$CPU_EXC \
                -J "ts12_union_${ARM}_s$S" slurm/stage1/union_graphs.sbatch "$GRUN" test)
        fi
    fi
    C=""
    if [ ! -e "$RUN/score_test.done" ]; then
        C=$(SB ACCEL=gpu sbatch --parsable --nice="$NICE" $(dep "$U") -p submit-gpu-express,submit-gpu \
            --constraint=nvidia_a30 --gres=gpu:1 -c 8 --mem=48G -t 01:00:00 --exclude=$GPU_EXC -J "ts12_score_${ARM}_s$S" \
            slurm/stage1/score_split.sbatch "$RUN" "$GRUN/ltf_union_graphs" test)
    fi
    F=$(SB sbatch --parsable --nice="$NICE" $(dep "$C") --array=0-$((NB - 1)) -p submit-gpu-express,submit-gpu -c 16 \
        --mem=32G -t 01:00:00 --exclude=$CPU_EXC -J "ts12_shard_${ARM}_s${S}_fr" \
        slurm/stage1/final_test_shard.sbatch "$SEL" "$P" "$S" fr "$NB")
    L=$(SB sbatch --parsable --nice="$NICE" $(dep "$C") --array=0-$((NB - 1)) -p submit-gpu-express,submit-gpu -c 16 \
        --mem=32G -t 01:00:00 --exclude=$CPU_EXC -J "ts12_shard_${ARM}_s${S}_llp" \
        slurm/stage1/final_test_shard.sbatch "$SEL" "$P" "$S" llp "$NB")
    M=$(SB sbatch --parsable --nice="$NICE" $(dep "$F:$L") -p submit-gpu-express,submit-gpu -c 8 --mem=24G \
        -t 01:00:00 --exclude=$CPU_EXC -J "ts12_merge_${ARM}_s$S" \
        slurm/stage1/final_test_merge.sbatch "$SEL" "$P" "$S" "$NB" "$NB")
    log "$P s$S: union ${U:-done} -> scores ${C:-done} -> shards fr $F, llp $L ($NB blocks each) -> merge $M"
    MERGES="$MERGES:$M"
done <<< "$ITEMS"
MERGES=${MERGES#:}
R1=$(SB sbatch --parsable --nice="$NICE" $(dep "$MERGES") -p submit-gpu-express,submit-gpu -c 16 --mem=32G -t 01:00:00 \
     --exclude=$CPU_EXC -J ts12_readout slurm/run_py.sbatch hnl.eval.training_sample_seeds12 readout --out results \
     --name training_sample_test_seeds12_v1)
R2=$(SB sbatch --parsable --nice="$NICE" $(dep "$MERGES") -p submit-gpu-express,submit-gpu -c 16 --mem=32G -t 01:00:00 \
     --exclude=$CPU_EXC -J ts12_bootstrap slurm/run_py.sbatch hnl.eval.bootstrap_tables --out results \
     --name bootstrap_ts_test_3seed_v1 --extra-test-spec "$SPEC")
R3=$(SB sbatch --parsable --nice="$NICE" $(dep "$R2") -p submit-gpu-express,submit-gpu -c 16 --mem=32G -t 01:00:00 \
     --exclude=$CPU_EXC -J ts12_paired slurm/run_py.sbatch hnl.eval.training_sample_paired_test \
     --out results/training_sample_paired_test_3seed_v1.csv --ts-spec "$SPEC" \
     --ts-bootstrap results/bootstrap_ts_test_3seed_v1.csv)
log "read-outs: per sample $R1, bootstrap rows $R2, paired differences $R3 (after the merges $MERGES)"
