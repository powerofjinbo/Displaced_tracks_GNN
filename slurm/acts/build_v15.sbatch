#!/bin/bash
# Build ACTS v15.0.0 (Fatras, examples, Pythia8, Python bindings) in the CentOS7 container with the LCG_97apython3
# view, the environment of the v15 CI (docs/displaced_sample_search.md section 3). Resumable: each job continues the
# build where the previous one stopped; the last one installs and writes build.done.
# Base OS: atlas/centos7-atlasos (CentOS 7.9 with glibc-devel and make; the bare centos:centos7 image lacks both);
# compilers and libraries come from the LCG view. The `which` shim is kept for images without it.
#   sbatch slurm/acts/build_v15.sbatch     (chain several with --dependency=afterany:<id>)
#SBATCH -J acts_build_v15
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 5
#SBATCH --mem=10G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
A=/work/submit/jinboz1/hnl/acts
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8   # CentOS 7.9 with devel packages
[ -f $A/build.done ] && { echo "already built"; exit 0; }
J=${BUILD_JOBS:-4}
mkdir -p $A/shim     # `which` for images without it (the LCG gcc setup.sh calls it)
[ -x $A/shim/which ] || { printf '%s\n' '#!/bin/bash' 'rc=0; for a in "$@"; do case $a in -*) continue;; esac; p=$(command -v "$a") && echo "$p" || rc=1; done; exit $rc' > $A/shim/which; chmod +x $A/shim/which; }
apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
set -e
export PATH=$A/shim:\$PATH
source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
echo CXX=\$CXX; which ninja
cd $A
if [ ! -f build-v15/CMakeCache.txt ]; then
  cmake -S src-v15 -B build-v15 -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$A/install-v15 \
    -DACTS_BUILD_FATRAS=ON -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_PYTHIA8=ON \
    -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DACTS_BUILD_UNITTESTS=OFF -DACTS_BUILD_INTEGRATIONTESTS=OFF \
    -DACTS_BUILD_BENCHMARKS=OFF 2>&1 | tail -40
fi
timeout 2700 cmake --build build-v15 -j $J 2>&1 | grep -E 'FAILED|error:|^\\[[0-9]+/[0-9]+\\]' | tail -30
" || true
# done when the build finishes inside the time limit
if apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "export PATH=$A/shim:\$PATH; source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1; cd $A; timeout 600 cmake --build build-v15 -j $J > /dev/null 2>&1 && cmake --install build-v15 > install.log 2>&1"; then
  date > $A/build.done; echo "BUILD DONE"
else
  echo "build not finished (continue with another job)"
fi
