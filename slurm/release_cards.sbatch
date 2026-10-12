#!/bin/bash
# Render the release cards on a compute node: the login node submit05 has no FreeSerif (the card titles' font), so
# matplotlib gets a private font cache built on the node that runs the job.
#   sbatch slurm/release_cards.sbatch
#SBATCH -J release_cards
#SBATCH -p submit
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export MPLCONFIGDIR=$TMPDIR/mpl_${SLURM_JOB_ID:-local}
mkdir -p "$MPLCONFIGDIR"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname)"
python - <<'PY'
from matplotlib.font_manager import findfont, FontProperties
f = findfont(FontProperties(family="FreeSerif"), fallback_to_default=False)
print("FreeSerif:", f)
PY
python -I -m hnl.plots.release_card_trackml --out figures/release/canto_l_trackml_card \
    --out-accuracy figures/release/canto_l_trackml_accuracy_card
rm -rf "$MPLCONFIGDIR"
