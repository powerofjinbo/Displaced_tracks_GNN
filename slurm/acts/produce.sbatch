#!/bin/bash
# One chunk of the new-sample production (docs/new_samples_spec.md): Pythia8 (ACTS v15 container) -> optional gun
# (project env) -> Fatras + smearing digitisation (container) -> npz in the Zenodo format + truth sidecar (project
# env). Scratch CSVs live under /work and are deleted once converted.
#   sbatch --export=ALL,SAMPLE=<name>,FIRST=<n>,N=<n>[,XMODE=measured|truth][,OUT=<dir>][,KEEP_CSV=1] slurm/acts/produce.sbatch
#   sbatch --array=0-15%4 --export=ALL,SAMPLE=<name>,CHUNK=500 slurm/acts/produce.sbatch
# SAMPLE: ttbar_val (Pythia8 ttbar only), generic_iso, generic_dense (ttbar + gun), hss15, hss40 (Pythia8 h->ss).
#SBATCH -J acts_produce
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
# as an array: --array=0-K with CHUNK=<events per task> gives FIRST = task id x CHUNK and N = CHUNK
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${CHUNK:-}" ]; then FIRST=$((SLURM_ARRAY_TASK_ID * CHUNK)); N=$CHUNK; fi
: "${SAMPLE:?}" "${FIRST:?}" "${N:?}"
XMODE=${XMODE:-measured}
REPO=/work/submit/jinboz1/hnl/hnl-displaced
A=/work/submit/jinboz1/hnl/acts
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
DET=$A/geometry/detectors.csv
DATA=/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated   # hnl_lrt/unpacked (the Zenodo data) is read-only
# seeds: one base per sample (docs/new_samples_spec.md section 4); event numbers are global within a sample
case $SAMPLE in
  ttbar_val)     PROC=ttbar; GUN="";    SEED=150001; DIRNAME=${OUT:-$A/validation/ttbar_v15_$XMODE} ;;
  generic_iso)   PROC=ttbar; GUN=iso;   SEED=250001; DIRNAME=${OUT:-$DATA/generic_iso_v1} ;;
  generic_dense) PROC=ttbar; GUN=dense; SEED=350001; DIRNAME=${OUT:-$DATA/generic_dense_v1} ;;
  hss15)         PROC=hss15; GUN="";    SEED=450001; DIRNAME=${OUT:-$DATA/hss15_100_v1} ;;
  hss40)         PROC=hss40; GUN="";    SEED=550001; DIRNAME=${OUT:-$DATA/hss40_300_v1} ;;
  *) echo "unknown SAMPLE $SAMPLE"; exit 2 ;;
esac
W=$A/work/$SAMPLE/$FIRST
if [ -e "$DIRNAME/${FIRST}.npz" ] && [ -z "${OVERWRITE:-}" ]; then echo "$DIRNAME/${FIRST}.npz exists: not overwriting"; exit 3; fi
rm -rf "$W"; mkdir -p "$W" "$DIRNAME" "${DIRNAME}_truth"
T=${SLURM_CPUS_PER_TASK:-4}
in_box() {
  apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
    export PATH=$A/shim:\$PATH
    source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
    source $A/build-v15/python/setup.sh > /dev/null 2>&1
    export ACTS_SRC=$A/src-v15
    python3 $REPO/scripts/acts/acts_sim.py $*"
}
t0=$(date +%s)
in_box pythia "$W" --process $PROC --first $FIRST --events $N --seed $SEED --threads $T
source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
STEM=particles_gen
if [ -n "$GUN" ]; then
  python3 $REPO/scripts/acts/gun.py "$W" --component $GUN --first $FIRST --events $N --seed $((SEED + 7))
  mv "$W/gun_truth_${FIRST}.npz" "${DIRNAME}_truth/"
  rm -f "$W"/event*-particles_gen.csv
  STEM=particles_input
fi
in_box sim "$W" --stem $STEM --first $FIRST --events $N --seed $((SEED + 13)) --threads $T
python3 $REPO/scripts/acts/csv_to_npz.py "$W" --detectors $DET --out "$DIRNAME" --first $FIRST --events $N \
  --x $XMODE --sidecar "${DIRNAME}_truth/truth_${FIRST}.npz" ${KEEP_CSV:+--keep-csv}
[ -n "${KEEP_CSV:-}" ] || rm -rf "$W"
echo "chunk $SAMPLE $FIRST+$N done in $(( $(date +%s) - t0 )) s"
