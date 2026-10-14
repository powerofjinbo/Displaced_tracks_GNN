#!/bin/bash
# Gun simulation of the TrackML pileup overlay (docs/pileup_overlay_plan.md sections 2.1 step 2, 4.1, 4.2), CPU only:
# gun particle CSVs (event<N>-particles.csv) -> ActsExampleFatrasGeneric of BUILD with the TrackML-like RZ field map,
# pMin 0.1 GeV and bremsstrahlung off (configs/trackml/overlay_sim_v1.json) -> sim-hit and particle CSVs in OUT, with
# fatras.log and sim_summary.json. BUILD defaults to build-v15; production takes the patched acts/build-v15-zmirror.
# The job fails if a particle fails to simulate (scripts/acts/overlay_sim.py sim, exit code 4).
#   sbatch --export=ALL,IN=<particle dir>,OUT=<empty dir>,SEED=<set base seed + 13>[,BUILD=<build dir>][,FIRST=<n>,N=<n>] slurm/acts/overlay_sim.sbatch
#SBATCH -J overlay_sim
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${IN:?}" "${OUT:?}" "${SEED:?}"
REPO=/work/submit/jinboz1/hnl/hnl-displaced
BUILD=${BUILD:-/work/submit/jinboz1/hnl/acts/build-v15}
source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
echo "host $(hostname) start $(date -u) build $BUILD"
python3 -u $REPO/scripts/acts/overlay_sim.py sim "$IN" "$OUT" --build "$BUILD" --seed "$SEED" \
  ${FIRST:+--first $FIRST} ${N:+--events $N} --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "end $(date -u)"
