#!/bin/bash
# Far-displaced daughters on the HNL TRAINING split (diagnostic only; no test file is opened): graphs and edge scores of
# one seed for acorn (ttbar) (runs/baseline_v1/armi_s<S>) and Canto-L (runs/stage1_final_learned/l1tl/armi_s<S>), made as
# their val graphs were (hnl.eval.far_trainsplit docstring): the population's train events and the 49 identity val
# events staged from the feature store as "valset" -> metric-learning graphs (acorn MetricLearning on the GPU, the
# run's r_infer / knn_infer) -> Canto-L's learned-triplet union (CPU, the seed's GBDT, k 5, tau 0.005) -> GNN scores of
# both runs (GPU, best checkpoints) -> $OUT/s<S>/{acorn_ttbar,canto_l}/valset.tar.
# Usage: env -u OMP_NUM_THREADS sbatch slurm/feedback/far_trainsplit_graphs.sbatch SEED   (after the population job)
#SBATCH -J fts_graphs
#SBATCH -p submit-gpu-express,submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:59:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
# nodes without AVX2 (the GTX 1080 Ti nodes): Polars' compat runtime (as slurm/baseline/_common.sh)
grep -qw avx2 /proc/cpuinfo || export PYTHONPATH=/work/submit/jinboz1/hnl/env/compat_polars${PYTHONPATH:+:$PYTHONPATH}
SEED=$1
OUT=$HNL_DATA/runs/feedback_v4/far_trainsplit
S=$OUT/s$SEED
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL" "$S"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; seed $SEED; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo no-gpu); $(date -u +%FT%TZ)"
eval "$(python -m hnl.eval.far_trainsplit params --seed "$SEED")"
echo "ml $ML_CKPT r_infer $R_INFER knn_infer $KNN_INFER; triplet model $TRIPLET_MODEL k $KTOP tau $TAU r_w $R_WIDE"
echo "gnn acorn (ttbar) $ACORN_GNN_CKPT; gnn Canto-L $CANTO_GNN_CKPT"
R="python -m hnl.train.acorn_runner"
python -m hnl.eval.far_trainsplit stage --dest "$LOCAL/fs"
echo "staged $(ls "$LOCAL/fs/valset" | wc -l) events $(date -u +%FT%TZ)"
$R infer --stage ml --ckpt "$ML_CKPT" --input-dir "$LOCAL/fs" --out-dir "$LOCAL/ml" --set r_infer="$R_INFER" \
    --set knn_infer="$KNN_INFER"
echo "metric-learning graphs $(ls "$LOCAL/ml/valset" | wc -l) $(date -u +%FT%TZ)"
OMP_NUM_THREADS=1 python -m hnl.graphs.helix_union build --ml-run "$ML_RUN" --split val --graphs "$LOCAL/ml/valset" \
    --out "$LOCAL/u" --report "$S/union_report" --triplet-model "$TRIPLET_MODEL" --tau "$TAU" --k "$KTOP" \
    --r-wide "$R_WIDE" --nsig "$NSIG" --pt-min "$PT_MIN" --workers "$SLURM_CPUS_PER_TASK" | tail -5
echo "union graphs $(ls "$LOCAL/u/valset" | wc -l) $(date -u +%FT%TZ)"
$R infer --stage gnn --ckpt "$ACORN_GNN_CKPT" --input-dir "$LOCAL/ml" --out-dir "$LOCAL/gnn_a"
$R infer --stage gnn --ckpt "$CANTO_GNN_CKPT" --input-dir "$LOCAL/u" --out-dir "$LOCAL/gnn_c"
echo "scored $(ls "$LOCAL/gnn_a/valset" | wc -l) / $(ls "$LOCAL/gnn_c/valset" | wc -l) $(date -u +%FT%TZ)"
$R pack --src "$LOCAL/gnn_a" --dest "$S/acorn_ttbar"
$R pack --src "$LOCAL/gnn_c" --dest "$S/canto_l"
cp "$LOCAL"/ml/infer_*.json "$S/ml_infer.json"
cp "$LOCAL"/gnn_a/infer_*.json "$S/acorn_ttbar/infer.json"
cp "$LOCAL"/gnn_c/infer_*.json "$S/canto_l/infer.json"
du -sh "$S"
touch "$S/graphs.done"
echo "done $(date -u +%FT%TZ)"
