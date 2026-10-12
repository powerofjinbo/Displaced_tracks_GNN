#!/bin/bash
# Feedback round 3, task A2 (full builder menu for the acorn baselines, val): learned fragment merging on an acorn
# baseline run's OWN GNN scores, made as Canto's learned merging (slurm/stage1/val_chain.sh -> score_split*.sbatch ->
# learned_merge.sbatch): 2000 training graphs of the run (RUN/ml_graphs/trainset.tar, the graphs its GNN was trained
# on) drawn at random with numpy default_rng(0) over the sorted graph names (the rule of score_split_sample.sbatch, the
# training-sample study's D1), scored on CPU with the run's best GNN checkpoint (the checkpoint of its stored val and
# test scores; sha256 checked), fragment-pair data after the same-layer Mutex Watershed (s0 0.2) on all of them
# (hnl.trackbuild.learned_merge data), the GBDT (learned_merge train: its fixed settings, seed 0), then the val grid of
# the mwslmerge family (hnl.eval.trackbuild_val, tau 0.005-0.5) on the run's stored val scores with that model.
# The scored training graphs stay on the node (not stored); the pair data are stored compressed (np.savez_compressed,
# the same arrays).
# Usage: env -u OMP_NUM_THREADS sbatch -J fb3_lmerge_arm<A>_s<S> slurm/feedback/fb3_lmerge.sbatch RUN_DIR OUT_DIR [GRID]
#   writes OUT_DIR/{model.joblib,model.joblib.json,pairs_train.npz,picked_graphs.txt,infer_*.json,provenance.json} and
#   RUN_DIR/GRID (default trackbuild_val_fb3_lmerge; a new sub-directory, refused if it exists)
#SBATCH -J fb3_lmerge
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; OUT=$2; GRID=${3:-trackbuild_val_fb3_lmerge}
N=2000
[ -e "$RUN/$GRID/per_particle.parquet" ] && { echo "$RUN/$GRID exists: refusing to overwrite"; exit 1; }
[ -e "$OUT/model.joblib" ] && { echo "$OUT/model.joblib exists: refusing to overwrite"; exit 1; }
mkdir -p "$OUT"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/all" "$LOCAL/in/trainset" "$LOCAL/val"
trap 'rm -rf "$LOCAL"' EXIT
T0=$(date -u +%FT%TZ)
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; out $OUT; grid $GRID; N $N; $T0"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# the checkpoint behind the run's stored val scores (chain_check: sha256 of the best checkpoint = the val inference's)
python -m hnl.train.chain_check --run "$RUN" --stages ml,gnn
CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$RUN" --stage gnn)
echo "checkpoint $CKPT"
tar -xf "$RUN/ml_graphs/trainset.tar" -C "$LOCAL/all"
python -I -m hnl.eval.strong_baselines_v3 pick --dir "$LOCAL/all" --n "$N" > "$LOCAL/picked_graphs.txt"
while read -r f; do mv "$LOCAL/all/$f" "$LOCAL/in/trainset/"; done < "$LOCAL/picked_graphs.txt"
rm -rf "$LOCAL/all"
echo "drawn $(ls "$LOCAL/in/trainset" | wc -l) of the run's training graphs; per sample code: $(ls "$LOCAL/in/trainset" | cut -c6 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" \
    --accelerator cpu > "$LOCAL/infer.log" 2>&1 || { tail -50 "$LOCAL/infer.log"; exit 1; }
tail -3 "$LOCAL/infer.log"
echo "scored $(ls "$LOCAL/gnn/trainset" | wc -l) training graphs $(date -u +%FT%TZ)"
python -m hnl.trackbuild.learned_merge data --graphs "$LOCAL/gnn/trainset" --out "$LOCAL/pairs_train.npz"
python -m hnl.trackbuild.learned_merge train --data "$LOCAL/pairs_train.npz" --out "$OUT/model.joblib"
cp "$LOCAL/picked_graphs.txt" "$OUT/picked_graphs.txt"
cp "$LOCAL"/gnn/infer_*.json "$OUT/"
python -I -m hnl.eval.strong_baselines_v3 provenance-lmerge --run "$RUN" --out "$OUT" --ckpt "$CKPT" \
    --pairs "$LOCAL/pairs_train.npz" --scored "$LOCAL/gnn/trainset" --started "$T0"
echo "model done $(date -u +%FT%TZ)"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
echo "$(ls "$LOCAL/val" | wc -l) val graphs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1   # val grid: one thread per worker process (outputs identical; decisions 12:00 UTC)
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/$GRID" --builders mwslmerge \
    --merge-model "$OUT/model.joblib"
python -I -c "
import json, sys, datetime
json.dump({'run': sys.argv[1], 'grid': sys.argv[2], 'merge_model': sys.argv[3], 'builders': 'mwslmerge',
           'slurm_job': sys.argv[4], 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'feedback round 3 task A2 (val only); hnl.eval.strong_baselines_v3'}, open(sys.argv[5], 'w'), indent=1)
" "$RUN" "$GRID" "$OUT/model.joblib" "$SLURM_JOB_ID" "$RUN/$GRID/provenance.json"
echo "done $(date -u +%FT%TZ)"
