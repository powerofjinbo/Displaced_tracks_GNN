#!/bin/bash
# The V1 closure set of the TrackML pileup version (docs/pileup_overlay_plan.md section 10, gate V1; hnl.overlay.closure),
# end to end, CPU only, truth only (no model, no model output):
#   1. select     hnl.overlay.closure select: the three V1 populations over all 150 tp_v1 val events and the prompt
#                 subset -> WORK/population_v1.parquet (+ .json with the counts against val_regions.json); skipped if
#                 the table exists
#   2. make       hnl.overlay.closure make: Fatras input of the replaced natives of HOSTS (barcodes (2, 0, i, 0, 0)),
#                 closure truth and closure_events.json -> WORK/closure
#   3. Fatras     scripts/acts/overlay_sim.py sim: the z-mirror build with the TrackML-like RZ map, pMin 0.1 GeV, no
#                 bremsstrahlung, seed = base + 13
#   4. pack       hnl.overlay.fatras_io pack
#   5. assemble   hnl.overlay.closure assemble: emulation, digitiser, frozen converter, host rows without the natives
#                 + closure rows -> ROOT/{hits,particles,ptype}/closure, meta, sidecar, closure_truth; CSV tar on /work
#   6. verify     hnl.overlay.closure verify
#   7. checks     hnl.overlay.closure_checks run -> OUT (if given)
# sbatch splits --export at commas, so HOSTS separates the host ids by colons:
#   sbatch --export=ALL,HOSTS=21289:21332,BASE=990101,WORK=<empty dir on /work>,ROOT=<table root>[,OUT=<json>]
#          [,BUILD=<build dir>][,OVERWRITE=1] slurm/overlay/closure_set.sbatch
# BASE defaults to the closure set's production base seed (configs/trackml/split_h_v1.json); dry runs pass the pilot
# seed 990101.
#SBATCH -J ov_closure
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${HOSTS:?}" "${WORK:?}" "${ROOT:?}"
REPO=/work/submit/jinboz1/hnl/hnl-displaced
BUILD=${BUILD:-/work/submit/jinboz1/hnl/acts/build-v15-zmirror}
cd $REPO
source env/activate.sh > /dev/null 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
if [ -z "${BASE:-}" ]; then
  BASE=$(python -c "from hnl.overlay.closure import default_base_seed; print(default_base_seed('closure'))")
fi
if [ -e "$WORK/closure" ]; then echo "$WORK/closure exists; use an empty WORK" >&2; exit 2; fi
mkdir -p "$WORK" "$ROOT"
echo "host $(hostname) start $(date -u) git $(git rev-parse --short HEAD) hosts $HOSTS base $BASE build $BUILD root $ROOT"
t0=$(date +%s)
if [ ! -e "$WORK/population_v1.parquet" ]; then
  python -u -m hnl.overlay.closure select --out "$WORK/population_v1.parquet"
fi
python -u -m hnl.overlay.closure make --hosts "$HOSTS" --set closure --population "$WORK/population_v1.parquet" \
  --base-seed "$BASE" --out "$WORK/closure"
python3 -u scripts/acts/overlay_sim.py sim "$WORK/closure/fatras_in" "$WORK/closure/fatras_out" --build "$BUILD" \
  --seed $((BASE + 13)) --threads "${SLURM_CPUS_PER_TASK:-4}" --allow-failures
python -u -m hnl.overlay.fatras_io pack --sim-dir "$WORK/closure/fatras_out" \
  --out "$WORK/closure/fatras_out/simhits.parquet" --remove-csv
python -u -m hnl.overlay.closure assemble --set closure --dir "$WORK/closure" \
  --sim "$WORK/closure/fatras_out/simhits.parquet" --root "$ROOT" --csv-tar "$WORK/closure_csv_closure.tar" \
  $([ "${OVERWRITE:-0}" = 1 ] && echo --overwrite)
python -u -m hnl.overlay.closure verify --set closure --root "$ROOT"
if [ -n "${OUT:-}" ]; then
  python -u -m hnl.overlay.closure_checks run --work "$WORK" --root "$ROOT" --out-json "$OUT"
fi
echo "end $(date -u) ($(( $(date +%s) - t0 )) s)"
