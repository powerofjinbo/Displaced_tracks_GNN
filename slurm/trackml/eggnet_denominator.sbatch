#!/bin/bash
# Ledger R13 (post-freeze, descriptive): Canto-L's stored TrackML test tracks under the EggNet code with the target
# denominator pT >= 1 GeV and >= 1, >= 2, >= 3 hits (hnl.trackml.eggnet_denominator). No network or builder is run.
#   sbatch slurm/trackml/eggnet_denominator.sbatch
#SBATCH -J eggnet_denominator
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/canto_l_v1/seed0_x1
OUT=${OUT:-$RUN/test_post/eggnet_denominator}
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname)"
python -I -m hnl.trackml.eggnet_denominator --run "$RUN" --out "$OUT"
