#!/bin/bash
# TrackML calibrations of docs/trackml_plan.md section 6.5, on the 200 calibration events of the throughput TRAINING
# set only (section 6.6: the 100 dev ids + the next 100 training ids of configs/trackml/split_tp_v1.json; never val or
# test): field map, resolution table RES, slope-window check and merge-cut acceptance per |eta| bin
# (hnl.trackml.calibrate) -> configs/trackml/{field_map_v1.json,res_v1.json,slope_window_v1.tsv,merge_cuts_v1.tsv,
# calibration_v1.md}. Needs the tables of slurm/trackml/tp_convert.sbatch.
#   env -u OMP_NUM_THREADS sbatch slurm/trackml/tp_calibrate.sbatch
#SBATCH -J tml_tp_calib
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm configs | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}; cpus $(env -u OMP_NUM_THREADS nproc); $(date -u +%FT%TZ)"
echo "code sha256: $(sha256sum src/hnl/trackml/calibrate.py | awk '{print $1}')"
echo "split: $(sha256sum configs/trackml/split_tp_v1.json)"
OMP_NUM_THREADS=2 python -I -m pytest -q -p no:cacheprovider src/hnl/trackml/tests/test_calibrate.py
# the pipeline's event sets (hnl.trackml.config, from the stored tables of data.tables) against the split file's lists
python -I - <<'EOF'
import json
from hnl.trackml import config as CFG
cfg = CFG.load()
s = CFG._sets(cfg)
sp = json.load(open(CFG.repo_path(cfg["data"]["split"])))
print("config", cfg["_path"], "tables", cfg["data"]["tables"], "phase", s["_phase"])
for k in ("dev", "val", "valmon", "train", "calib", "gbdt", "merge_train"):
    print(f"  {k}: {len(s[k])} events, {'equal to' if list(s[k]) == sp[k] else 'DIFFERENT FROM'} the split file")
EOF
OMP_NUM_THREADS=$(env -u OMP_NUM_THREADS nproc) python -I -m hnl.trackml.calibrate \
    --summary "/work/submit/jinboz1/hnl/logs/tml_tp_calib_${SLURM_JOB_ID:-local}.json"
ls -l configs/trackml/field_map_v1.json configs/trackml/res_v1.json configs/trackml/slope_window_v1.tsv \
    configs/trackml/merge_cuts_v1.tsv configs/trackml/calibration_v1.md
date -u +%FT%TZ
