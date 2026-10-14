#!/bin/bash
# Addendum G T3, Canto-L (ttbar+G) seed S: the Canto-L classifier (configs/stage1/triplet_symregion_trunc.yaml, 30
# epochs over arm g's training graphs, best checkpoint by valmon loss) on the learned-support union graphs, then the
# scores of the val_all union graphs (RUN/gnn_scores/valset.tar) and eval.done. slurm/stage1/variant.sbatch with arm g's
# staging; the val CC curves of evaluate_gnn are not computed here (they read HNL truth tables); the val read-out is
# hnl.eval.addendum_g_t3. Resumable 1-h express chunks. RUN = runs/stage1_g1/l1tl/armg_s<S>.
# Usage: sbatch slurm/addendum_g_t3/canto_train.sbatch S
#SBATCH -J g3_canto_train
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30|Tesla_v100|nvidia_rtx6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
export RUN_OVERRIDE=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_g1/l1tl/armg_s$1
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/addendum_g_t3/_g.sh "$1"
CONFIG=$HNL_REPO/configs/stage1/triplet_symregion_trunc.yaml
others=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -o %i 2>/dev/null | grep -vx "$SLURM_JOB_ID" || true)
[ -n "$others" ] && { echo "job(s) $others of the same name already running: exiting"; exit 0; }
[ -e "$RUN/eval.done" ] && { echo "eval.done exists: nothing to do"; exit 0; }
GRAPHS=$HNL_DATA/runs/baseline_g1_ts/armg_s$SEED/ltf_union_graphs
echo "Canto-L (ttbar+G) config $CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)) graphs $GRAPHS"
if [ ! -e "$RUN/gnn_train.done" ]; then
    $RG stage --what train,valmon --source tar --tar-dir "$GRAPHS" --dest "$LOCAL/in"
    train_stage gnn 30 "$LOCAL/in" --config "$CONFIG"
    [ -e "$RUN/gnn_train.done" ] || exit 0
    rm -rf "$LOCAL/in"
fi
CKPT=$(best_ckpt gnn)
$RG stage --what val_all --source tar --tar-dir "$GRAPHS" --dest "$LOCAL/val"
$R infer --stage gnn --ckpt "$CKPT" --input-dir "$LOCAL/val" --out-dir "$LOCAL/gnn"
$R pack --src "$LOCAL/gnn" --dest "$RUN/gnn_scores"
cp "$LOCAL"/gnn/infer_*.json "$RUN/gnn_scores/"
touch "$RUN/eval.done"
