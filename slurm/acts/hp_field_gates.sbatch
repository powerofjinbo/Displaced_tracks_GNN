#!/bin/bash
# Step H-P, gates F1 and containment (docs/pileup_overlay_plan.md section 10). Truth only and CPU only: no model runs.
# 1. field probe: 1000 events x 100 particles of the field study's probe mix (seed 777, Fatras seed 790);
# 2. gun probe: 4 gun events per tp_v1 val host in the layout of plan section 7 (pilot base seed 990001, Fatras
#    seed 990014), at the hosts' hard-scatter vertices (val truth particles only);
# 3. both simulated with BUILD through scripts/acts/overlay_sim.py sim (the production path); steps whose output
#    exists are skipped, so a rerun repeats only what is missing;
# 4. gates: results/pileup_hp/f1_containment_<TAG>.{json,md}.
# The module map is MODULE_MAP (a volume_id, layer_id, acts_module_id, trackml_module_id table, e.g.
# configs/trackml/module_map_acts_zmirror_v1.csv for build-v15-zmirror) or built from ACTS_DET, a detectors.csv
# dumped from the same build. INPUTS (default W) holds field_in/ and gun_in/, so several builds share one probe.
#   sbatch --export=ALL,BUILD=<build dir>,TAG=<name>,MODULE_MAP=<csv>|ACTS_DET=<csv>[,W=<scratch>][,INPUTS=<dir>][,GATE_EXTRA="..."] slurm/acts/hp_field_gates.sbatch
#SBATCH -J hp_field_gates
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${BUILD:?}" "${TAG:?}"
REPO=/work/submit/jinboz1/hnl/hnl-displaced
W=${W:-/work/submit/jinboz1/hnl/pileup_design/hp_field/$TAG}
I=${INPUTS:-$W}
if [ -n "${MODULE_MAP:-}" ]; then MAPARG="--module-map $MODULE_MAP"; else MAPARG="--acts-detectors ${ACTS_DET:?}"; fi
SEG=/work/submit/jinboz1/hnl/pileup_design/field_material/segments_calib200.parquet   # calib training events
S="python3 -u $REPO/scripts/acts/overlay_sim.py"
T=${SLURM_CPUS_PER_TASK:-4}
source $REPO/env/activate.sh > /dev/null 2>&1 || exit 1
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
echo "host $(hostname) start $(date -u) build $BUILD"
mkdir -p "$W" "$REPO/results/pileup_hp"
[ -e "$I/field_in/event000000999-particles.csv" ] || $S field-probe "$I/field_in" --events 1000 --per-event 100 --seed 777
[ -e "$I/gun_in/hosts.json" ] || $S gun-probe "$I/gun_in" --per-host 4 --seed 990001
[ -e "$W/field_sim/sim_summary.json" ] || $S sim "$I/field_in" "$W/field_sim" --build "$BUILD" --seed 790 --threads $T --allow-failures
[ -e "$W/gun_sim/sim_summary.json" ] || $S sim "$I/gun_in" "$W/gun_sim" --build "$BUILD" --seed 990014 --threads $T --allow-failures
$S gates --field-probe "$W/field_sim" --gun-probe "$W/gun_sim" --gun-probe-input "$I/gun_in" \
  $MAPARG --trackml-segments "$SEG" ${GATE_EXTRA:-} --out "$REPO/results/pileup_hp/f1_containment_$TAG"
echo "end $(date -u)"
