#!/bin/bash
# Learned graph-free attachment v2 (CPU; module-frame pulls, hnl.trackbuild.attach): candidate data from a run's scored
# TRAINING graphs (gnn_scores/trainset.tar) after the first pass, the attach model, then the val grid with the
# matching family ('mwsattach' after the same-layer Mutex Watershed, 'wmattach' after walkthrough + merging) next to
# CC, walkthrough + merging and mwslayer.
# Usage: [FIRST_PASS=mwslayer|mwslayermerge|mwslmerge|walkmerge] [MERGE_MODEL=...] sbatch slurm/stage1/attach.sbatch RUN_DIR [BUILDERS]
#   outputs RUN_DIR/attach_v2_<first pass>/, RUN_DIR/trackbuild_val_attach_v2_<first pass>/ (v1: RUN_DIR/attach/)
#SBATCH -J s1_attach
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1
FP=${FIRST_PASS:-mwslayer}
case "$FP" in
    mwslayer) FAMILY=mwsattach; FLAG=--mwsattach-model ;;
    walkmerge) FAMILY=wmattach; FLAG=--attach-model ;;
    mwslayermerge) FAMILY=mwsmattach; FLAG=--mwsmattach-model ;;
    mwslmerge) FAMILY=mwslmattach; FLAG=--mwslmattach-model ;;
    *) echo "FIRST_PASS must be mwslayer, mwslayermerge, mwslmerge or walkmerge" >&2; exit 1 ;;
esac
BUILDERS=${2:-cc,walkmergeS,mwslayer,mwslayermerge,$FAMILY}
MERGE=()                                  # the learned-merge model of the mwslmerge first pass (default: the run's)
[ "$FP" = mwslmerge ] && MERGE=(--merge-model "${MERGE_MODEL:-$RUN/lmerge/model.joblib}")
OUT=$RUN/attach_v2_$FP
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/train" "$LOCAL/val" "$OUT"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; run $RUN; first pass $FP"
tar -xf "$RUN/gnn_scores/trainset.tar" -C "$LOCAL/train"
python -m hnl.trackbuild.attach data --graphs "$LOCAL/train" --out "$OUT/candidates_train.npz" --first-pass "$FP" \
    "${MERGE[@]}"
python -m hnl.trackbuild.attach train --data "$OUT/candidates_train.npz" --out "$OUT/model.joblib"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/val"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/val" --out "$RUN/trackbuild_val_attach_v2_$FP" --builders "$BUILDERS" \
    "$FLAG" "$OUT/model.joblib" "${MERGE[@]}"
touch "$RUN/trackbuild_val_attach_v2_$FP.done"
