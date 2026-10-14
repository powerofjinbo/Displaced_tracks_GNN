#!/bin/bash
# Render the release cards on a compute node: the login node submit05 has no FreeSerif (the card titles' font), so
# matplotlib gets a private font cache built on the node that runs the job.
#   sbatch slurm/release_cards.sbatch
#   sbatch --export=ALL,CARDS="full_dump",DUMP_DIR=<dir> slurm/release_cards.sbatch   (only the cards named;
#   default: trackml full; full_dump also writes every cell of the full card to DIR)
#SBATCH -J release_cards
#SBATCH -p submit
#SBATCH --exclude=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
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
for card in ${CARDS:-trackml full}; do
    case $card in
        trackml) python -I -m hnl.plots.release_card_trackml --out figures/release/canto_l_trackml_card ;;
        full) python -I -m hnl.plots.release_card_full --out figures/release/canto_card_full ;;
        full_dump) python -I -m hnl.plots.release_card_full --out figures/release/canto_card_full \
            --dump "${DUMP_DIR:?}/card_full_cells.csv" ;;
        *) echo "unknown card $card" >&2; exit 1 ;;
    esac
done
rm -rf "$MPLCONFIGDIR"
