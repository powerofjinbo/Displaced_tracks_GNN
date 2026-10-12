#!/bin/bash
# Gate G0 on an A30 (plan section 8, before S1; untrained networks, about 1 GPU-h): (1) the FRNN-on-CUDA base graph
# against the CPU cKDTree build of the same events (S3 makes the additions on CPU, S4 / S5 rebuild the base with FRNN);
# (2) peak GPU memory and time of the classifier per INFERENCE sector (forward) and per training window (forward +
# backward) on event-store files (default: the smoke test's ttbar-embedding union graphs at 12 pairs per hit).
# (3) gate G4: the torch port of the learned triplet support on CUDA against the numpy reference (identical triplets,
# features, p and kept pairs; differences listed with their distance to a split threshold) on the smoke events with the
# ttbar-trained embedding and GBDT (G4_EVENTS / G4_MODEL / G4_RW to repeat it with the TrackML-trained ones, 20 dev events
# per the plan). The metric-learning step time is measured by S1 itself.
# Usage: STORE=/work/.../trackml_smoke/<job>/store/valset [ML_CKPT=..] [R=0.0686] sbatch slurm/trackml/g0_gpu.sbatch
#SBATCH -J tml_g0
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
OUT=$RUN/gates/g0
mkdir -p "$OUT"
ML=${ML_CKPT:-/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/baseline_v1/armi_s0/ml/artifacts/best-epoch=20-f1=0.356576.ckpt}
python -m hnl.trackml.g0 base-check --events "$(events smoke)" --ml-ckpt "$ML" --r "${R:-0.0686}" --out "$OUT"
python -m hnl.trackml.g0 sectors --store "${STORE:?STORE=<event-store split dir>}" --out "$OUT"
G4M=${G4_MODEL:-/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/baseline_v1/armi_s0/learned_triplet_filter_v2/model.joblib}
python -m hnl.trackml.support_gpu compare --events "${G4_EVENTS:-$(events smoke)}" --ml-ckpt "$ML" \
    --r-wide "${G4_RW:-0.1026}" --model "$G4M" --ops 0.005:5,0.05:3,0.3:1 --out "$OUT" --device cuda
