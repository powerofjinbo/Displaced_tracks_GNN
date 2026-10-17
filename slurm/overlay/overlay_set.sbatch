#!/bin/bash
# One overlay set of the TrackML pileup version, end to end (docs/pileup_overlay_plan.md sections 2.1, 5.4, 7), CPU
# only, truth only (no model):
#   1. gun        hnl.overlay.gun: 8 isolated + 2 dense LLPs per host at its hard-scatter vertex -> WORK/gun; the
#                 Fatras input of a host is event<N>-particles.csv with N its Fatras event number by the rule of
#                 configs/trackml/split_h_v1.json (hnl.overlay.split_h.fatras_event_numbers), whichever hosts the job
#                 holds
#   2. Fatras     scripts/acts/overlay_sim.py sim: ActsExampleFatrasGeneric of BUILD (default: the z-mirror build that
#                 passed G0) with the TrackML-like RZ map, pMin 0.1 GeV, no bremsstrahlung (overlay_sim_v1.json),
#                 seed = base + 13, every event under its own number (one call per run of consecutive numbers);
#                 keeps fatras.log and sim_summary.json
#   3. pack       hnl.overlay.fatras_io pack: sim hits and the simulated-particle list, CSVs removed
#   4. assemble   hnl.overlay.writer assemble: emulation, digitiser, frozen converter, host rows + gun rows ->
#                 ROOT/{hits,particles,ptype}/SET, meta_SET.json, sidecar_SET.json, gun_truth_SET.parquet, and the
#                 gun-only CSV members as one tar on /work
#   5. verify     hnl.overlay.writer verify: the converter gate on the stored shards
# Particles that Fatras fails to simulate are recorded (sidecar, gun_truth sim_status) and the run continues
# (ALLOW_FAILURES=1, default); ALLOW_FAILURES=0 stops at the first failure (overlay_sim.py exit code 4).
# sbatch splits --export at commas, so HOSTS separates the host ids by colons:
#   sbatch --export=ALL,SET=pilot,HOSTS=21332:21610,WORK=<empty dir on /work>[,ROOT=<table root>]
#          [,BUILD=<build dir>][,ALLOW_FAILURES=0][,OVERWRITE=1][,BASE=<seed>][,FATRAS_EVENTS=<n1:n2:...>]
#          slurm/overlay/overlay_set.sbatch
# HOST_ROOT: the hosts' tables (hnl.overlay.hosts.host_root): derived/tp_v1 for hval, closure and pilot (the default);
# the H-test set names derived/tp03_v1 explicitly (HOST_ROOT is required for SET=htest; addendum H section 9.4).
# ROOT defaults to external/trackml/derived/overlay_v1 (decision D9). The base seed and the Fatras event numbers come
# from configs/trackml/split_h_v1.json (addendum H section 2); BASE and FATRAS_EVENTS override them for dry runs only,
# and the Version check (hnl.overlay.writer.version_check) then fails the set.
#SBATCH -J ov_set
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${SET:?}" "${HOSTS:?}" "${WORK:?}"
HOSTS=${HOSTS//:/,}
REPO=/work/submit/jinboz1/hnl/hnl-displaced
BUILD=${BUILD:-/work/submit/jinboz1/hnl/acts/build-v15-zmirror}
ROOT=${ROOT:-/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/derived/overlay_v1}
ALLOW_FAILURES=${ALLOW_FAILURES:-1}
cd $REPO
source env/activate.sh > /dev/null 2>&1
if [ -z "${BASE:-}" ]; then
  BASE=$(python -c "from hnl.overlay.split_h import base_seed; print(base_seed('$SET'))")
fi
: "${BASE:?no base seed for set $SET}"
if [ "$SET" = htest ] && [ -z "${HOST_ROOT:-}" ]; then echo "SET=htest needs HOST_ROOT (derived/tp03_v1)" >&2; exit 2; fi
HOST_ROOT=${HOST_ROOT:-$(python -c "from hnl.overlay.hosts import host_root; print(host_root('$SET'))")}
if [ -e "$WORK/gun" ]; then echo "$WORK/gun exists; use an empty WORK" >&2; exit 2; fi
mkdir -p "$WORK"
echo "host $(hostname) start $(date -u) git $(git rev-parse --short HEAD) set $SET hosts $HOSTS base $BASE build $BUILD root $ROOT host root $HOST_ROOT"
t0=$(date +%s)
python -u -m hnl.overlay.gun --hosts "$HOSTS" --set "$SET" --base-seed "$BASE" --out "$WORK/gun" \
  --host-root "$HOST_ROOT" ${FATRAS_EVENTS:+--fatras-events "$FATRAS_EVENTS"}
python3 -u scripts/acts/overlay_sim.py sim "$WORK/gun/fatras_in" "$WORK/gun/fatras_out" --build "$BUILD" \
  --seed $((BASE + 13)) --threads "${SLURM_CPUS_PER_TASK:-4}" $([ "$ALLOW_FAILURES" = 1 ] && echo --allow-failures)
python -u -m hnl.overlay.fatras_io pack --sim-dir "$WORK/gun/fatras_out" --out "$WORK/gun/fatras_out/simhits.parquet" \
  --remove-csv
python -u -m hnl.overlay.writer assemble --set "$SET" --gun-dir "$WORK/gun" \
  --sim "$WORK/gun/fatras_out/simhits.parquet" --root "$ROOT" --csv-tar "$WORK/gun_csv_$SET.tar" \
  --host-root "$HOST_ROOT" $([ "${OVERWRITE:-0}" = 1 ] && echo --overwrite)
python -u -m hnl.overlay.writer verify --set "$SET" --root "$ROOT" --host-root "$HOST_ROOT"
echo "end $(date -u) ($(( $(date +%s) - t0 )) s)"
