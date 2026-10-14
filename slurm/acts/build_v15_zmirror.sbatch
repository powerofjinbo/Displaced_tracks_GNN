#!/bin/bash
# Build the patched ACTS v15.0.0 GenericDetector for the TrackML pileup version of the gun sample
# (docs/pileup_overlay_plan.md section 3, step H-P). Same container, LCG view and CMake options as build_v15.sbatch.
# The source is a copy of src-v15 with patches/acts_v15/0001-GenericDetector-z-mirror-negative-endcaps.patch applied
# (the -z endcap module centres become the z-mirror of the +z centres instead of their point reflection). src-v15 and
# build-v15 are never touched. Only the targets later steps need are built: the Fatras, geometry and propagation
# examples for the GenericDetector and the Python bindings (ninja directly: the LCG_97 CMake 3.14 takes one --target).
# Resumable like build_v15.sbatch.
#   sbatch slurm/acts/build_v15_zmirror.sbatch
#SBATCH -J acts_build_v15_zmirror
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
A=/work/submit/jinboz1/hnl/acts
S=$A/src-v15-zmirror
B=$A/build-v15-zmirror
P=/work/submit/jinboz1/hnl/hnl-displaced/patches/acts_v15/0001-GenericDetector-z-mirror-negative-endcaps.patch
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
TARGETS="ActsExampleFatrasGeneric ActsExampleGeometryGeneric ActsExamplePropagationGeneric ActsPythonBindings"
[ -f $B/build.done ] && { echo "already built"; exit 0; }
J=${BUILD_JOBS:-$(( ${SLURM_CPUS_PER_TASK:-8} ))}
if [ ! -d $S ]; then
  cp -a $A/src-v15 $S
  git -C $S apply $P
fi
# the source must differ from v15.0.0 by exactly the patch
git -C $S diff | cmp - $P || { echo "source tree $S does not match $P"; exit 1; }
echo "patch sha256 $(sha256sum < $P | cut -d' ' -f1)"
apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
set -e
export PATH=$A/shim:\$PATH
source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
echo CXX=\$CXX; which ninja
if [ ! -f $B/CMakeCache.txt ]; then
  cmake -S $S -B $B -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DACTS_BUILD_FATRAS=ON -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_PYTHIA8=ON \
    -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DACTS_BUILD_UNITTESTS=OFF -DACTS_BUILD_INTEGRATIONTESTS=OFF \
    -DACTS_BUILD_BENCHMARKS=OFF 2>&1 | tail -15
fi
timeout 3000 ninja -C $B -j $J $TARGETS 2>&1 | grep -E 'FAILED|error:|^\\[[0-9]+/[0-9]+\\]' | tail -30
" || true
if apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "export PATH=$A/shim:\$PATH; source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1; timeout 300 ninja -C $B -j $J $TARGETS > /dev/null 2>&1"; then
  { date; echo "patch $(sha256sum < $P | cut -d' ' -f1)"; echo "targets $TARGETS"; } > $B/build.done
  echo "BUILD DONE"
else
  echo "build not finished (continue with another job)"
fi
