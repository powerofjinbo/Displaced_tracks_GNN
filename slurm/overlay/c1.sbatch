#!/bin/bash
# Gate C1 of the pileup overlay (docs/pileup_overlay_plan.md section 10): pions and muons over TrackML's angular range
# (hnl.overlay.c1 particles), Fatras of the patched build with the TrackML-like field map (settings of
# configs/trackml/overlay_sim_v1.json), then the digitiser against TrackML calibration (training) hits
# (hnl.overlay.c1 run). Truth only, CPU only. The sim CSVs stay in the scratch directory W.
#   sbatch [--export=ALL,NEV=2000,SEED=910001,STEPS="particles fatras run"] slurm/overlay/c1.sbatch
#SBATCH -J ov_c1
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/pileup_design/hp_digitiser/logs/%x_%j.out
set -euo pipefail
R=/work/submit/jinboz1/hnl/hnl-displaced
A=/work/submit/jinboz1/hnl/acts
W=${W:-/work/submit/jinboz1/hnl/pileup_design/hp_digitiser/c1_sim}
BUILD=${BUILD:-$A/build-v15-zmirror}
LOOKUP=${LOOKUP:-$R/configs/trackml/module_map_acts_zmirror_v1.csv}
MAP=${MAP:-/work/submit/jinboz1/hnl/pileup_design/field_material/trackml_like_rz_map_v1.txt}
OUTJ=${OUTJ:-$R/results/pileup_hp/c1_v1.json}
NEV=${NEV:-2000}
SEED=${SEED:-910001}
REF=${REF:-40}
STEPS=${STEPS:-particles fatras run}
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
echo "host $(hostname) job ${SLURM_JOB_ID:-} start $(date -u) steps: $STEPS build $BUILD lookup $LOOKUP"
source $R/env/activate.sh > /dev/null 2>&1 || exit 1
export PYTHONPATH=$R/src
for st in $STEPS; do
  t0=$(date +%s)
  case $st in
    particles) rm -rf $W/in; python -u -m hnl.overlay.c1 particles --out $W/in --events $NEV --seed $SEED ;;
    fatras)
      rm -rf $W/out; mkdir -p $W/out
      [ -f $BUILD/build.done ] || { echo "no finished build in $BUILD"; exit 1; }
      apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
        export PATH=$A/shim:\$PATH
        source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
        export LD_LIBRARY_PATH=$BUILD/lib:\$LD_LIBRARY_PATH
        $BUILD/bin/ActsExampleFatrasGeneric -n $NEV -j ${SLURM_CPUS_PER_TASK:-8} --rnd-seed $(( SEED + 13 )) \
          --input-dir $W/in --output-dir $W/out --output-csv --bf-map-file $MAP --bf-map-type rz \
          --fatras-pmin-gev 0.1 --fatras-em-radiation 0 --fatras-hits sensitive" | tail -5 ;;
    run) python -u -m hnl.overlay.c1 run --sim-dir $W/out --lookup $LOOKUP --out-json $OUTJ --ref-events $REF \
           --seed $SEED ;;
  esac
  echo "step $st done in $(( $(date +%s) - t0 )) s"
done
echo "end $(date -u)"
