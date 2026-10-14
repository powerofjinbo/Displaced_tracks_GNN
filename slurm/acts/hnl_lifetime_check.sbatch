#!/bin/bash
# Lifetime check of the HNL card, generator level only (scripts/acts/hnl_lifetime_check.cc; docs/hnl_points_spec.md
# section 10). One array task = CHUNK events starting at task * CHUNK, written to OUTDIR/tau_<first>.csv.
#   sbatch --array=0-39 --export=ALL,MASS=20,CTAU=200,SEED=10710001,CHUNK=1000,OUTDIR=DIR[,EXTRA="a = b;c = d"][,RNG=pythia] \
#       slurm/acts/hnl_lifetime_check.sbatch
#SBATCH -J hnl_tau
#SBATCH -p submit-gpu-express,submit
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:40:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
: "${MASS:?}" "${CTAU:?}" "${SEED:?}" "${CHUNK:?}" "${OUTDIR:?}"
RNG=${RNG:-engine}; ECM=${ECM:-14000}
REPO=/work/submit/jinboz1/hnl/hnl-displaced
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
FIRST=$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK ))
W=$(mktemp -d /work/submit/jinboz1/hnl/acts/work/tau_XXXXXX)
mkdir -p "$OUTDIR"
python3 $REPO/scripts/acts/hnl_card.py "$W/card.cmnd" --mass $MASS --ctau $CTAU --extra "${EXTRA:-}" > /dev/null
t0=$(date +%s)
apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
  source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
  g++ -O2 -std=c++17 $REPO/scripts/acts/hnl_lifetime_check.cc -I\$PYTHIA8/include -L\$PYTHIA8/lib \
      -Wl,-rpath,\$PYTHIA8/lib -lpythia8 -ldl -o $W/check
  $W/check $W/card.cmnd $FIRST $CHUNK $SEED $ECM $OUTDIR/tau_$(printf %07d $FIRST).csv $RNG"
cp "$W/card.cmnd" "$OUTDIR/card.cmnd"
rm -rf "$W"
echo "m=$MASS ctau=$CTAU seed=$SEED rng=$RNG extra='${EXTRA:-}' events $FIRST+$CHUNK done in $(( $(date +%s) - t0 )) s"
