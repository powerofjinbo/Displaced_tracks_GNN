#!/bin/bash
# S1 (plan section 6.1): metric learning on TrackML (configs/trackml/ml_train.yaml; hnl.trackml.ml.TrackMLMetricLearning:
# 30 virtual epochs x 1,000 training events drawn from seeded permutations cycling through all training events; best
# valmon f1). The acorn feature store of the training and valmon events is written to node-local disk at job start
# (hnl.trackml.graphs fs; about 3-4 MB per event, never on /ceph). Resumes automatically (preempt / periodic / last);
# a preempted or timed-out run exits 0 without the marker: resubmit.
# X1 (plan D3): ML_CONFIG=configs/trackml/ml_train_x1.yaml RUN_SUFFIX=_x1 (run seed0_x1; the feature store then carries
# the config's cluster features). Chunked on submit-gpu-express: slurm/trackml/ml_chunks.sh.
# Usage: [SEED=0] [ML_CONFIG=...] [RUN_SUFFIX=...] sbatch slurm/trackml/ml_train.sbatch
#SBATCH -J tml_ml
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
[ -e "$RUN/ml_train.done" ] && { echo "S1 already done: $RUN/ml_train.done"; exit 0; }
ML_CONFIG=${ML_CONFIG:-$(cfg metric_learning.config)}
echo "metric-learning config $ML_CONFIG ($(sha256sum "$ML_CONFIG" | cut -c1-12))"
storage_gate 1e9                        # checkpoints of the run (about 50 MB each; best, last, preempt, periodic, epochs)
NP=$(( OMP_NUM_THREADS < 8 ? OMP_NUM_THREADS : 8 ))
python -m hnl.trackml.graphs fs --events train --dest "$LOCAL/fs/trainset" --ml-config "$ML_CONFIG" --nproc "$NP"
python -m hnl.trackml.graphs fs --events valmon --dest "$LOCAL/fs/valset" --ml-config "$ML_CONFIG" --nproc "$NP"
du -sh "$LOCAL/fs"
train_stage ml "$(cfg metric_learning.virtual_epochs)" "$LOCAL/fs" "$ML_CONFIG" --set "ml_seed=$SEED"
if [ -e "$RUN/ml_train.done" ]; then echo "best checkpoint: $(best_ckpt ml)"; else echo "not finished: resubmit to resume"; fi
