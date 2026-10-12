#!/bin/bash
# Brief §9 smoke test of the acorn baseline on one A30: metric-learning step time / memory (500 training events of
# arm ii), graph size of the metric-learning radius graph on 500 val events, GNN step time / memory on those graphs.
#SBATCH -J smoke_baseline
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
RUN=$HNL_DATA/runs/smoke_$SLURM_JOB_ID
echo "git $(git rev-parse HEAD); host $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); local $LOCAL; run $RUN"
R="python -m hnl.train.acorn_runner"

t0=$(date +%s)
$R stage --arm ii --what train,valmon --dest "$LOCAL/fs"
echo "staging: $(( $(date +%s) - t0 )) s; local usage $(du -sh "$LOCAL" | cut -f1)"

$R train --stage ml --arm ii --seed 0 --run-dir "$RUN" --input-dir "$LOCAL/fs" --max-epochs 1 --limit-train-batches 500
CKPT=$(ls "$RUN"/ml/artifacts/last*.ckpt | head -1)
echo "ml checkpoint $CKPT"

mkdir -p "$LOCAL/fs_val"
mv "$LOCAL/fs/valset" "$LOCAL/fs_val/valset"
$R infer --stage ml --ckpt "$CKPT" --input-dir "$LOCAL/fs_val" --out-dir "$LOCAL/ml" --set r_infer=0.1 --set knn_infer=1000

python -I - "$LOCAL/ml/valset" <<'EOF'
import glob, sys, numpy as np
from acorn.utils.loading_utils import load_pyg
E, T, K = [], [], []
for f in sorted(glob.glob(sys.argv[1] + "/*.pyg.gz"))[:200]:
    g = load_pyg(f[:-3], weights_only=False)
    E.append(g.edge_index.shape[1]); T.append(g.track_edges.shape[1]); K.append(int((g.truth_map >= 0).sum()))
E, T, K = map(np.array, (E, T, K))
print(f"ML graph (1-epoch model, r 0.1, knn 1000): edges per event mean {E.mean():.0f} median {np.median(E):.0f} max {E.max()}; true-edge eff {K.sum()/T.sum():.4f}")
EOF

mkdir -p "$LOCAL/gnn_in/trainset" "$LOCAL/gnn_in/valset"
files=("$LOCAL/ml/valset"/*.pyg.gz)          # (no `ls | head`: SIGPIPE + pipefail would end the job)
mv "${files[@]:0:400}" "$LOCAL/gnn_in/trainset/"
mv "${files[@]:400}" "$LOCAL/gnn_in/valset/"
$R train --stage gnn --arm ii --seed 0 --run-dir "$RUN" --input-dir "$LOCAL/gnn_in" --max-epochs 1
echo "== total $(( $(date +%s) - t0 )) s"
