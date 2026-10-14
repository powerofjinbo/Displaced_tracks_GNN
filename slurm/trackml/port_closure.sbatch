#!/bin/bash
# Closure tests of the reference-code ports (hnl.trackml.port_closure): the EggNet code's target count on our
# non-test throughput and accuracy events, and Truth-CC on Liu et al.'s released TrackML-1GeV events against their
# Table 3. Evaluation code only; no Canto-L output and no test event of ours is read.
#   sbatch slurm/trackml/port_closure.sbatch            (after slurm/trackml/port_closure_fetch.sbatch)
#   STEPS="liu" sbatch slurm/trackml/port_closure.sbatch
#SBATCH -J port_closure
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
OUT=${OUT:-/work/submit/jinboz1/hnl/tmp/port_closure/out}
EV=${EV:-/work/submit/jinboz1/hnl/tmp/port_closure/liu_1gev/ev}
mkdir -p "$OUT"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm | wc -l); host $(hostname)"
for step in ${STEPS:-liu eggnet}; do
    case $step in
        liu) python -I -m hnl.trackml.port_closure liu-truthcc --events "$EV" --out "$OUT" --draws 20 --workers ${SLURM_CPUS_PER_TASK:-8} ;;
        eggnet) python -I -m hnl.trackml.port_closure eggnet-counts --out "$OUT" ;;
    esac
done
