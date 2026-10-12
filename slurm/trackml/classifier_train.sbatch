#!/bin/bash
# S4 (plan sections 6.3-6.4): the Canto-L classifier on TrackML windows (configs/trackml/classifier_train.yaml,
# hnl.trackml.classifier.TrackMLCanto: 30 virtual epochs x 4,000 windows of 22.5 deg +-11.25 deg, truncation
# augmentation, loss on core-owned edges; valmon = the 8 inference sectors of the 50 valmon events, checkpoint on the
# lowest val_loss; the weights after virtual epoch 0 are kept for gates G2 / G3). At job start the full-event union
# graphs of the training and valmon events are built on the A30 (embedding + FRNN base graph + stored additions) into a
# node-local event store (about 10-20 MB per event, never on /ceph). Resumes automatically; the 2-day partition limit
# ends with a pre-timeout checkpoint: resubmit the same command (about 2.1 days in total, plan section 8).
# Usage: [SEED=0] sbatch slurm/trackml/classifier_train.sbatch
#SBATCH -J tml_gnn
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
[ -e "$RUN/gnn_train.done" ] && { echo "S4 already done"; exit 0; }
require "$RUN/graph_choice.done" "$(adds_dir train)" "$(adds_dir val)"
G=$RUN/graph/graph_choice.json
# additions were made on CPU (cKDTree); the base graph is rebuilt here with FRNN: pairs at distance ~ r* may differ
# (measured by g0_gpu.sbatch base-check before S3 is accepted)
# STORE (optional): a persistent event store (e.g. $RUN/store on /ceph) filled with --skip-existing over several
# chunks of at most 1 h (slurm/trackml/gnn_chunks.sh, DECISIONS 2026-10-08); training starts once STORE.done exists.
# allocator segments that grow and shrink: a validation after training steps failed on fragmentation (13.3 GB reserved
# but unallocated, S4 chunk 6727230; DECISIONS 2026-10-08); allocation only, the numerics are unchanged
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# periodic.ckpt every 10 min of training (default 30): a chunk whose stop signal falls in an epoch-end validation
# (about 6.5 min, longer than the 5 min before the limit) is killed and resumes from the last periodic state
export HNL_PERIODIC_MIN=10
S=${STORE:-$LOCAL/store}
if [ ! -e "$S.done" ]; then
    python -m hnl.trackml.graphs stage --events train --graph-choice "$G" --additions "$(adds_dir train)" \
        --dest "$S/trainset" --device cuda --allow-backend-change ${STORE:+--skip-existing}
    python -m hnl.trackml.graphs stage --events valmon --graph-choice "$G" --additions "$(adds_dir valmon)" \
        --dest "$S/valset" --device cuda --allow-backend-change ${STORE:+--skip-existing}
    if [ -n "${STORE:-}" ]; then
        nt=$(ls "$S/trainset" | grep -c '\.pyg\.gz$'); nv=$(ls "$S/valset" | grep -c '\.pyg\.gz$')
        et=$(events train | tr ',' '\n' | grep -c .); ev=$(events valmon | tr ',' '\n' | grep -c .)
        echo "store: $nt of $et training and $nv of $ev valmon events"
        if [ "$nt" -eq "$et" ] && [ "$nv" -eq "$ev" ]; then touch "$S.done"; else echo "store incomplete: next chunk"; exit 0; fi
    fi
fi
du -sh "$S"
train_stage gnn "$(cfg classifier.virtual_epochs)" "$S" "$(cfg classifier.config)" \
    --set "window_seed=$SEED" --set "windows_per_epoch=$(cfg classifier.windows_per_epoch)"
if [ -e "$RUN/gnn_train.done" ]; then echo "best checkpoint: $(best_ckpt gnn)"; else echo "not finished: resubmit to resume"; fi
