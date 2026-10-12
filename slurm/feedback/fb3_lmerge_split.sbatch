#!/bin/bash
# Feedback round 3, task A2 (val): fb3_lmerge.sbatch split into smaller jobs (for a run whose single 16-CPU job could not
# be scheduled): MODE score K N scores part K of N of the 2000 drawn training graphs (the same draw, every N-th graph
# from K) with the run's checkpoint and keeps the scored graphs in OUT/scored_tmp/part<K>; MODE finish (after all parts)
# runs the remaining steps of fb3_lmerge.sbatch on their union (learned_merge data / train, provenance, the val grid of
# the learned-merging family) and removes OUT/scored_tmp.
# Usage: env -u OMP_NUM_THREADS sbatch -J fb3_lmsplit_<mode>_arm<A>_s<S> slurm/feedback/fb3_lmerge_split.sbatch RUN_DIR OUT_DIR score K N
#        env -u OMP_NUM_THREADS sbatch --dependency=afterok:J1:J2:... slurm/feedback/fb3_lmerge_split.sbatch RUN_DIR OUT_DIR finish N
#SBATCH -J fb3_lmerge_split
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; OUT=$2; MODE=$3
GRID=trackbuild_val_fb3_lmerge
N_DRAW=2000
[ -e "$OUT/model.joblib" ] && { echo "$OUT/model.joblib exists: refusing to overwrite"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL" "$OUT/scored_tmp"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; out $OUT; mode $MODE ${4:-} ${5:-}; $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ "$MODE" = score ]; then
    K=$4; N=$5
    python -m hnl.train.chain_check --run "$RUN" --stages ml,gnn
    CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$RUN" --stage gnn)
    mkdir -p "$LOCAL/all" "$LOCAL/in/trainset"
    tar -xf "$RUN/ml_graphs/trainset.tar" -C "$LOCAL/all"
    python -I -m hnl.eval.strong_baselines_v3 pick --dir "$LOCAL/all" --n "$N_DRAW" > "$LOCAL/picked_graphs.txt"
    awk -v k="$K" -v n="$N" '(NR - 1) % n == k' "$LOCAL/picked_graphs.txt" > "$LOCAL/part.txt"
    while read -r f; do mv "$LOCAL/all/$f" "$LOCAL/in/trainset/"; done < "$LOCAL/part.txt"
    rm -rf "$LOCAL/all"
    echo "part $K of $N: $(ls "$LOCAL/in/trainset" | wc -l) graphs; checkpoint $CKPT"
    python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" \
        --accelerator cpu > "$LOCAL/infer.log" 2>&1 || { tail -50 "$LOCAL/infer.log"; exit 1; }
    D="$OUT/scored_tmp/part$K"
    rm -rf "$D.tmp"; mkdir -p "$D.tmp"
    cp "$LOCAL"/gnn/trainset/*.pyg.gz "$D.tmp/"
    cp "$LOCAL"/gnn/infer_*.json "$LOCAL/picked_graphs.txt" "$LOCAL/part.txt" "$D.tmp/"
    rm -rf "$D"; mv "$D.tmp" "$D"
    echo "scored $(ls "$D"/*.pyg.gz | wc -l) graphs -> $D $(date -u +%FT%TZ)"
elif [ "$MODE" = finish ]; then
    N=$4
    [ -e "$RUN/$GRID/per_particle.parquet" ] && { echo "$RUN/$GRID exists: refusing to overwrite"; exit 1; }
    T0=$(date -u +%FT%TZ)
    CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$RUN" --stage gnn)
    mkdir -p "$LOCAL/scored"
    for k in $(seq 0 $((N - 1))); do
        D="$OUT/scored_tmp/part$k"
        cmp "$D/picked_graphs.txt" "$OUT/scored_tmp/part0/picked_graphs.txt"
        cp "$D"/*.pyg.gz "$LOCAL/scored/"
        cp "$D"/infer_*.json "$OUT/"
    done
    cp "$OUT/scored_tmp/part0/picked_graphs.txt" "$OUT/picked_graphs.txt"
    echo "union: $(ls "$LOCAL/scored" | wc -l) scored graphs"
    python -m hnl.trackbuild.learned_merge data --graphs "$LOCAL/scored" --out "$LOCAL/pairs_train.npz"
    python -m hnl.trackbuild.learned_merge train --data "$LOCAL/pairs_train.npz" --out "$OUT/model.joblib"
    python -I -m hnl.eval.strong_baselines_v3 provenance-lmerge --run "$RUN" --out "$OUT" --ckpt "$CKPT" \
        --pairs "$LOCAL/pairs_train.npz" --scored "$LOCAL/scored" --started "$T0"
    echo "model done $(date -u +%FT%TZ)"
    mkdir -p "$LOCAL/val"
    tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # val grid: one thread per worker process (decisions 12:00 UTC)
    python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/$GRID" --builders mwslmerge \
        --merge-model "$OUT/model.joblib"
    python -I -c "
import json, sys, datetime
json.dump({'run': sys.argv[1], 'grid': sys.argv[2], 'merge_model': sys.argv[3], 'builders': 'mwslmerge',
           'slurm_job': sys.argv[4], 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'feedback round 3 task A2 (val only); hnl.eval.strong_baselines_v3 (fb3_lmerge_split.sbatch)'},
          open(sys.argv[5], 'w'), indent=1)
" "$RUN" "$GRID" "$OUT/model.joblib" "$SLURM_JOB_ID" "$RUN/$GRID/provenance.json"
    rm -rf "$OUT/scored_tmp"
    echo "done $(date -u +%FT%TZ)"
fi
