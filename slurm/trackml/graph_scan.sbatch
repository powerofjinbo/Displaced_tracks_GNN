#!/bin/bash
# S2c (plan section 6.2): valmon graph scan, density x r_w factor x (tau, k) (configs/trackml/pipeline_v1.yaml graph.*):
# union pairs per hit, chain-edge efficiency, FRNN cap binding and graph-stage ceilings (targets, displaced targets,
# >= 5-hit particles). A Slurm array: task i takes every n-th valmon event; then graph_choose.sbatch.
# CPU by default (cKDTree + numpy support; the cost per event grows steeply with r_w: smoke_probe.sbatch measured about
# 2-3 min per event at 1.5 r* on 16 cores with the ttbar models). After gate G4: DEVICE=cuda on an A30 (FRNN, torch).
# Usage: [SEED=0] sbatch --array=0-9 slurm/trackml/graph_scan.sbatch
#SBATCH -J tml_scan
#SBATCH -p submit
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2, acorn imports die with SIGILL (DECISIONS 2026-10-03 21:45 UTC)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
require "$RUN/support_train.done"
FM=$(cat "$RUN/support/field_map.txt")
FMARG=()
if [ "$FM" != none ]; then FMARG=(--field-map "$FM"); fi
EV=$(array_slice "$(events valmon)")
python -m hnl.trackml.graphs scan --events "$EV" --ml-ckpt "$(best_ckpt ml)" --radii "$RUN/graph/radii.json" \
    --ltf-model "$RUN/support/model.joblib" --out "$RUN/graph/scan/part${SLURM_ARRAY_TASK_ID:-0}" --device "${DEVICE:-cpu}" \
    "${FMARG[@]}"
touch "$RUN/graph/scan/part${SLURM_ARRAY_TASK_ID:-0}.done"
