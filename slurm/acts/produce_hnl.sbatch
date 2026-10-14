#!/bin/bash
# One chunk of an HNL point (docs/hnl_points_spec.md): card (hnl_card.py) -> Pythia8 244 (hnl_pythia.cc, built in the
# ACTS v15 container) -> Fatras + smearing digitisation (acts_sim.py sim) -> one tar shard of Zenodo-format npz events
# + one truth sidecar (hnl_to_npz.py). Scratch CSVs live under /work and are deleted once converted.
#   sbatch --export=ALL,MASS=<GeV>,CTAU=<mm>,SEED=<base>,OUT=<sample dir>,FIRST=<n>,N=<n> slurm/acts/produce_hnl.sbatch
#   sbatch --array=0-5%3 --export=ALL,MASS=..,CTAU=..,SEED=..,OUT=..,CHUNK=500 slurm/acts/produce_hnl.sbatch
# Optional: ECM (GeV, default 14000), BEAM_T_NS (sigma of the vertex time, default 5 ns), DECAY (N decay form of
# hnl_card.DECAYS, default majorana_va), EXTRA (further Pythia settings separated by ';', pilot studies only; the
# driver only), GEN=driver|acts (acts: the ACTS v15
# Pythia8Generator, generator level only, for the cross-check), SIMULATE=0 (generator level only: the census of
# hnl_census.py gen in OUT/gen_<GEN>_<first>_*; the CSVs are kept only with KEEP_CSV=1), KEEP_CSV=1.
#SBATCH -J hnl_produce
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${CHUNK:-}" ]; then FIRST=$((SLURM_ARRAY_TASK_ID * CHUNK)); N=$CHUNK; fi
: "${MASS:?}" "${CTAU:?}" "${SEED:?}" "${OUT:?}" "${FIRST:?}" "${N:?}"
ECM=${ECM:-14000}; BEAM_T_NS=${BEAM_T_NS:-5}; DECAY=${DECAY:-majorana_va}; GEN=${GEN:-driver}; SIMULATE=${SIMULATE:-1}
# production into hnl_lrt/generated: only the settings of docs/hnl_points_spec.md (no pilot overrides)
case "$OUT" in
  */hnl_lrt/generated/*)
    if [ -n "${EXTRA:-}" ] || [ "$ECM" != 14000 ] || [ "$BEAM_T_NS" != 5 ] || [ "$DECAY" != majorana_va ] || \
       [ "$GEN" != driver ] || [ "$SIMULATE" != 1 ] || [ "$N" != 500 ] || [ $((FIRST % 500)) != 0 ]; then
      echo "production into $OUT takes only the spec settings (500-event chunks)"; exit 2; fi ;;
esac
REPO=/work/submit/jinboz1/hnl/hnl-displaced
A=/work/submit/jinboz1/hnl/acts
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
DET=$A/geometry/detectors.csv
LAST=$((FIRST + N - 1))
TAR=$OUT/events_$(printf %06d $FIRST)_$(printf %06d $LAST).tar
SIDE=${OUT}_truth/truth_${FIRST}.npz
if [ "$SIMULATE" = 1 ] && { [ -e "$TAR" ] || [ -e "$SIDE" ]; }; then echo "$TAR or $SIDE exists: not overwriting"; exit 3; fi
W=$A/work/hnl_$(basename "$OUT")_${GEN}/$FIRST
rm -rf "$W"; mkdir -p "$W" "$OUT"
T=${SLURM_CPUS_PER_TASK:-4}
in_box() {
  apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
    export PATH=$A/shim:\$PATH
    source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
    source $A/build-v15/python/setup.sh > /dev/null 2>&1
    export ACTS_SRC=$A/src-v15
    $*"
}
t0=$(date +%s)
python3 $REPO/scripts/acts/hnl_card.py "$W/card.cmnd" --mass $MASS --ctau $CTAU --decay $DECAY --extra "${EXTRA:-}"
if [ "$GEN" = driver ]; then
  in_box "g++ -O2 -std=c++17 $REPO/scripts/acts/hnl_pythia.cc -I\$PYTHIA8/include -L\$PYTHIA8/lib -Wl,-rpath,\$PYTHIA8/lib -lpythia8 -ldl -o $W/hnl_pythia"
  in_box "$W/hnl_pythia $W/card.cmnd $W $FIRST $N $SEED $ECM 0.0125 55.5 $BEAM_T_NS"
else
  in_box "python3 $REPO/scripts/acts/acts_sim.py pythia $W --process hnl --mass $MASS --ctau $CTAU --ecm $ECM --decay $DECAY --first $FIRST --events $N --seed $SEED --threads $T"
fi
echo "generation done in $(( $(date +%s) - t0 )) s"
if [ "$SIMULATE" != 1 ]; then
  # generator level: summarise with the census (gen mode) and drop the CSVs unless KEEP_CSV
  source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
  python3 $REPO/scripts/acts/hnl_census.py gen --dir "$W" --out "$OUT/gen_${GEN}_${FIRST}"
  if [ -n "${KEEP_CSV:-}" ]; then D=$OUT/gen_${GEN}_$FIRST; rm -rf "$D"; mv "$W" "$D"; else rm -rf "$W"; fi
  rmdir --ignore-fail-on-non-empty "$(dirname "$W")"
  echo "generator-level chunk $FIRST+$N done in $(( $(date +%s) - t0 )) s"; exit 0
fi
in_box "python3 $REPO/scripts/acts/acts_sim.py sim $W --stem particles_gen --first $FIRST --events $N --seed $((SEED + 13)) --threads $T"
source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
python3 $REPO/scripts/acts/hnl_to_npz.py "$W" --detectors $DET --tar "$TAR" --sidecar "$SIDE" --first $FIRST --events $N \
  ${KEEP_CSV:+--keep-csv}
[ -n "${KEEP_CSV:-}" ] || { rm -rf "$W"; rmdir --ignore-fail-on-non-empty "$(dirname "$W")"; }
echo "chunk m=$MASS ctau=$CTAU $FIRST+$N done in $(( $(date +%s) - t0 )) s"
