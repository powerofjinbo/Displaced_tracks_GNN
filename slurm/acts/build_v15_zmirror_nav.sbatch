#!/bin/bash
# Candidate build for gate F1 of the pileup overlay (docs/pileup_overlay_plan.md section 10): the z-mirror build of
# build_v15_zmirror.sbatch plus patches/acts_v15/0002-TrackingVolume-compatibleLayers-full-scan-fallback.patch.
# Without 0002 the v15 navigator loses a few per cent of the displaced gun particles: a particle that starts inside a
# volume moving inwards on a line that misses the inner layers gets no layer candidates, targets the volume boundary
# and is often lost; with the field map it then fails with "Interpolation out of bounds" once it leaves the map
# (results/pileup_hp/f1_containment_*.md). 0002 tests every layer of the volume when the directional walk finds none.
# Only ActsExampleFatrasGeneric is built. src-v15, build-v15 and the z-mirror build are not touched.
#   sbatch slurm/acts/build_v15_zmirror_nav.sbatch
#SBATCH -J acts_build_v15_zmirror_nav
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
A=/work/submit/jinboz1/hnl/acts
S=$A/src-v15-zmirror-nav
B=$A/build-v15-zmirror-nav
P=/work/submit/jinboz1/hnl/hnl-displaced/patches/acts_v15
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
TARGETS="ActsExampleFatrasGeneric"
[ -f $B/build.done ] && { echo "already built"; exit 0; }
J=${BUILD_JOBS:-$(( ${SLURM_CPUS_PER_TASK:-8} ))}
if [ ! -d $S ]; then
  cp -a $A/src-v15 $S
  git -C $S apply $P/0001-GenericDetector-z-mirror-negative-endcaps.patch
  git -C $S apply $P/0002-TrackingVolume-compatibleLayers-full-scan-fallback.patch
fi
# the source must differ from v15.0.0 by exactly the two patches (git diff lists Core/ before Examples/)
git -C $S diff | cmp - <(cat $P/0002-TrackingVolume-compatibleLayers-full-scan-fallback.patch \
  $P/0001-GenericDetector-z-mirror-negative-endcaps.patch) || { echo "source tree $S does not match the patches"; exit 1; }
apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
set -e
export PATH=$A/shim:\$PATH
source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
if [ ! -f $B/CMakeCache.txt ]; then
  cmake -S $S -B $B -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DACTS_BUILD_FATRAS=ON -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_PYTHIA8=ON \
    -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DACTS_BUILD_UNITTESTS=OFF -DACTS_BUILD_INTEGRATIONTESTS=OFF \
    -DACTS_BUILD_BENCHMARKS=OFF 2>&1 | tail -15
fi
timeout 3000 ninja -C $B -j $J $TARGETS 2>&1 | grep -E 'FAILED|error:|^\\[[0-9]+/[0-9]+\\]' | tail -30
" || true
if apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "export PATH=$A/shim:\$PATH; source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1; timeout 300 ninja -C $B -j $J $TARGETS > /dev/null 2>&1"; then
  { date; for f in $P/0001-*.patch $P/0002-*.patch; do echo "patch $(basename $f) $(sha256sum < $f | cut -d' ' -f1)"; done; echo "targets $TARGETS"; } > $B/build.done
  echo "BUILD DONE"
else
  echo "build not finished (continue with another job)"
fi
