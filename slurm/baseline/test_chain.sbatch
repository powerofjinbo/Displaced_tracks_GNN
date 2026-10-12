#!/bin/bash
# End-to-end check of the baseline stage machinery on the express queue with the smoke checkpoints (6653703):
# ml_scan on 200 val events; ML inference + pack on 500 events (400 "train" / 100 "val"); GNN training with a
# simulated pre-timeout SIGUSR1 (preempt checkpoint, state "preempted", no marker), automatic resume to completion
# (marker); GNN inference + pack; evaluate_gnn. Writes under runs/baseline_chaintest (deleted afterwards by hand).
#SBATCH -J bl_test_chain
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
export TAG=chaintest
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh ii 0
SMOKE=$HNL_DATA/runs/smoke_6653703
t0=$(date +%s)
$R stage --arm ii --what val --dest "$LOCAL/scan"
python -m hnl.graphs.ml_scan --ckpt "$SMOKE/ml/artifacts/last.ckpt" --input-dir "$LOCAL/scan" --out "$RUN/ml_scan" --max-events 200
python -I -c "import json,sys; print('selection', json.load(open(sys.argv[1]))['selection'])" "$RUN/ml_scan/scan.json"
echo "== scan done $(( $(date +%s) - t0 )) s"

$R stage --arm ii --what valmon --dest "$LOCAL/inf"
mkdir -p "$LOCAL/inf/trainset"
files=("$LOCAL/inf/valset"/*.pyg.gz)
mv "${files[@]:0:400}" "$LOCAL/inf/trainset/"
$R infer --stage ml --ckpt "$SMOKE/ml/artifacts/last.ckpt" --input-dir "$LOCAL/inf" --out-dir "$LOCAL/ml" --set r_infer=0.1 --set knn_infer=1000
$R pack --src "$LOCAL/ml" --dest "$RUN/ml_graphs"
mkdir -p "$LOCAL/gnn_in/trainset" "$LOCAL/gnn_in/valset"
tar -xf "$RUN/ml_graphs/trainset.tar" -C "$LOCAL/gnn_in/trainset"
tar -xf "$RUN/ml_graphs/valset.tar" -C "$LOCAL/gnn_in/valset"
echo "== ml graphs done $(( $(date +%s) - t0 )) s"

(sleep 45; echo "sending USR1 to the batch shell"; kill -USR1 $$) &
train_stage gnn 3 "$LOCAL/gnn_in"
ls -la "$RUN/gnn/artifacts" "$RUN"/*.done 2>&1 || true
[ -e "$RUN/gnn_train.done" ] && { echo "FAIL: marker written after preemption"; exit 1; }
train_stage gnn 3 "$LOCAL/gnn_in"
[ -e "$RUN/gnn_train.done" ] || { echo "FAIL: no marker after resume"; exit 1; }
echo "== gnn train (preempt + resume) done $(( $(date +%s) - t0 )) s"

rm -rf "$LOCAL/gnn_in/trainset"
CKPT=$(best_ckpt gnn)
$R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/gnn_in" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
mkdir -p "$LOCAL/scores"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/scores"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/scores" --out "$RUN/eval_val"
python -I - "$RUN/eval_val" <<'PY'
import sys, pandas as pd
pe = pd.read_parquet(sys.argv[1] + "/per_event.parquet"); pp = pd.read_parquet(sys.argv[1] + "/per_particle.parquet")
print(pe.groupby("cut")[["tp", "fp", "fn", "n_tracks_ge5", "n_fake", "n_dup_pairs"]].sum().iloc[::3].to_string())
print("events", pe.event_id.nunique(), "particles", len(pp) // pe.cut.nunique(), "dm eff at 0.5",
      round(pp[pp.cut == 0.5].dm_matched.mean(), 4))
PY
echo "== total $(( $(date +%s) - t0 )) s"
