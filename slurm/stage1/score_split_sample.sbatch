#!/bin/bash
# GNN edge scores of a Stage-1 run on N graphs of one split, drawn uniformly at random (numpy default_rng(0).choice
# over the sorted graph names, as hnl.graphs.learned_filter._graphs and `learned_merge data --max-events` draw theirs).
# Otherwise identical to slurm/stage1/score_split.sbatch, which takes the FIRST N sorted graphs. Written for the
# training-sample study (docs/training_sample_study.md): graph names are event<sample code><index>, so for a mixed
# training split the first N sorted graphs are a single sample (arm (iii): ttbar only; arm (ii): hnl10_100 only), while
# for arm (i) (ttbar only) both rules give an arbitrary ttbar subset. Packed to GNN_RUN/gnn_scores/<split>set.tar.
# Usage: N_EVENTS=2000 sbatch slurm/stage1/score_split_sample.sbatch GNN_RUN GRAPHS_DIR SPLIT
#SBATCH -J s1_score_split_sample
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
GNN_RUN=$1; GRAPHS=$2; SPLIT=${3:-train}
N=${N_EVENTS:?N_EVENTS (number of graphs to draw) is required}
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/all" "$LOCAL/in/${SPLIT}set"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; gnn $GNN_RUN; graphs $GRAPHS/${SPLIT}set.tar; N $N drawn at random (rng 0); accel ${ACCEL:-cpu}"
tar -xf "$GRAPHS/${SPLIT}set.tar" -C "$LOCAL/all"
python -I - "$LOCAL/all" "$N" > "$LOCAL/pick.txt" <<'EOF'
import os, sys
import numpy as np
names = sorted(f for f in os.listdir(sys.argv[1]) if f.endswith(".pyg.gz"))
n = int(sys.argv[2])
if n < len(names):
    pick = np.sort(np.random.default_rng(0).choice(len(names), n, replace=False))
    names = [names[k] for k in pick]
print("\n".join(names))
EOF
while read -r f; do mv "$LOCAL/all/$f" "$LOCAL/in/${SPLIT}set/"; done < "$LOCAL/pick.txt"
rm -rf "$LOCAL/all"
echo "drawn $(ls "$LOCAL/in/${SPLIT}set" | wc -l) graphs; per sample code: $(ls "$LOCAL/in/${SPLIT}set" | cut -c6 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
[ -n "${MLFORM_LIKE:-}" ] && python -m hnl.graphs.helix_union mlform --dir "$LOCAL/in/${SPLIT}set" --like "$MLFORM_LIKE"
CKPT=$(python -m hnl.graphs.helix_union ckpt --run "$GNN_RUN" --stage gnn)
echo "scoring $(ls "$LOCAL/in/${SPLIT}set" | wc -l) graphs with $CKPT"
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -m hnl.train.acorn_runner infer --stage gnn --ckpt "$CKPT" \
    --input-dir "$LOCAL/in" --out-dir "$LOCAL/gnn" --accelerator "${ACCEL:-cpu}"
python -m hnl.train.acorn_runner pack --src "$LOCAL/gnn" --dest "$GNN_RUN/gnn_scores"
mkdir -p "$GNN_RUN/gnn_scores/${SPLIT}_infer"
cp "$LOCAL"/gnn/infer_*.json "$GNN_RUN/gnn_scores/${SPLIT}_infer/"
cp "$LOCAL/pick.txt" "$GNN_RUN/gnn_scores/${SPLIT}_infer/picked_graphs.txt"
touch "$GNN_RUN/score_${SPLIT}.done"
