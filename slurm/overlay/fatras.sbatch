#!/bin/bash
# Fatras of the pileup overlay (docs/pileup_overlay_plan.md section 4.2) on particle CSVs, CPU only, then the sim hits
# packed into one parquet file and the CSV outputs removed (hnl.overlay.fatras_io pack).
# Settings of section 4.2: the TrackML-like RZ field map, pMin 0.1 GeV (D6), bremsstrahlung off (D7), multiple
# scattering and ionisation on, hits on sensitive surfaces only.
#   sbatch --export=ALL,IN=DIR,OUT=DIR,NEV=N,SEED=S[,BUILD=/work/.../build-v15-zmirror][,MAP=file] fatras.sbatch
# BUILD defaults to the unpatched build-v15 (correct everywhere except the -z strip endcaps, plan section 3).
#SBATCH -J ov_fatras
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:50:00
#SBATCH -o /work/submit/jinboz1/hnl/pileup_design/hp_overlay/logs/%x_%j.out
set -euo pipefail
A=/work/submit/jinboz1/hnl/acts
BUILD=${BUILD:-$A/build-v15}
MAP=${MAP:-/work/submit/jinboz1/hnl/pileup_design/field_material/trackml_like_rz_map_v1.txt}
PMIN=${PMIN:-0.1}
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/atlas/centos7-atlasos:latest-gcc8
: "${IN:?}" "${OUT:?}" "${NEV:?}" "${SEED:?}"
mkdir -p "$OUT"
echo "host $(hostname) start $(date -u) build $BUILD map $MAP sha256 $(sha256sum "$MAP" | cut -c1-16) pmin $PMIN seed $SEED"
t0=$(date +%s)
apptainer exec --cleanenv -B /cvmfs -B /work $IMG bash -c "
  export PATH=$A/shim:\$PATH
  source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc9-opt/setup.sh > /dev/null 2>&1
  export LD_LIBRARY_PATH=$BUILD/lib64:$BUILD/lib:\$LD_LIBRARY_PATH
  $BUILD/bin/ActsExampleFatrasGeneric -n $NEV -j ${SLURM_CPUS_PER_TASK:-8} --rnd-seed $SEED \
    --input-dir $IN --output-dir $OUT --output-csv \
    --bf-map-file $MAP --bf-map-type rz \
    --fatras-pmin-gev $PMIN --fatras-em-radiation 0 --fatras-hits sensitive"
echo "fatras done in $(( $(date +%s) - t0 )) s"
case "$BUILD" in *zmirror*) touch "$OUT/PATCHED" ;; esac
printf '{"build": "%s", "map": "%s", "map_sha256": "%s", "pmin_gev": %s, "rnd_seed": %s, "events": %s, "radiation": 0, "hits": "sensitive", "slurm_job": "%s", "host": "%s", "seconds": %s}\n' \
  "$BUILD" "$MAP" "$(sha256sum "$MAP" | cut -d' ' -f1)" "$PMIN" "$SEED" "$NEV" "${SLURM_JOB_ID:-}" "$(hostname)" \
  "$(( $(date +%s) - t0 ))" > "$OUT/fatras.json"
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null 2>&1
python -u -m hnl.overlay.fatras_io pack --sim-dir "$OUT" --out "$OUT/simhits.parquet" --remove-csv
echo "end $(date -u)"
