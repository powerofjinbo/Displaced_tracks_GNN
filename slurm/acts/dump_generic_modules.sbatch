#!/bin/bash
# Compile scripts/acts/dump_generic_modules.cpp against one ACTS v15 build tree and dump the GenericDetector's sensitive
# modules (centre, rotation, bounds, thickness, pitch, element id), plus the standard CsvTrackingGeometryWriter
# detectors.csv of ActsExampleGeometryGeneric from the same build. The tool is compiled into OUT; the build and source
# trees are only read. Gate G0 of docs/pileup_overlay_plan.md (section 10).
#   sbatch --export=ALL,SRC=src-v15-zmirror,BUILD=build-v15-zmirror,OUT=<dir> slurm/acts/dump_generic_modules.sbatch
#   sbatch --export=ALL,SRC=src-v15,BUILD=build-v15,OUT=<dir> slurm/acts/dump_generic_modules.sbatch   (control)
#SBATCH -J acts_dump_modules
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${SRC:?}" "${BUILD:?}" "${OUT:?}"
A=/work/submit/jinboz1/hnl/acts
S=$A/$SRC
B=$A/$BUILD
R=/work/submit/jinboz1/hnl/hnl-displaced
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt
mkdir -p $OUT
apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
set -e
export PATH=$A/shim:\$PATH
source $LCG/setup.sh > /dev/null 2>&1
\$CXX -std=gnu++17 -O2 -Wall -Wextra \
  -I$S/Examples/Detectors/GenericDetector/include -I$S/Core/include -I$B/Core -I$S/Plugins/Identification/include \
  -I$S/Plugins/Digitization/include -I$S/Examples/Framework/include -I$S/Fatras/include \
  -I$S/Examples/Detectors/Common/include -isystem $LCG/include/eigen3 -isystem $LCG/include \
  $R/scripts/acts/dump_generic_modules.cpp -o $OUT/dump_generic_modules \
  -L$B/lib64 -Wl,-rpath,$B/lib64:$LCG/lib \
  -lActsExamplesDetectorGeneric -lActsExamplesFramework -lActsPluginDigitization -lActsCore
$OUT/dump_generic_modules $OUT/modules.csv
mkdir -p $OUT/geometry_example
cd $OUT/geometry_example && LD_LIBRARY_PATH=$B/lib64:\$LD_LIBRARY_PATH $B/bin/ActsExampleGeometryGeneric -n 1 \
  --output-csv --output-dir $OUT/geometry_example -l 3 2>&1 | tail -5
"
ls -la $OUT $OUT/geometry_example
sha256sum $OUT/modules.csv $OUT/geometry_example/*detectors.csv
