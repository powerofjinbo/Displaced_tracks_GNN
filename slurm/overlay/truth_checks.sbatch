#!/bin/bash
# Truth-only checks of the pileup overlay before any model run (docs/pileup_overlay_plan.md section 10), CPU only, no
# model and no model output:
#   STEP=m1      hnl.overlay.m1 run: material check M1 of a Fatras probe directory or tar (SIM) against TrackML truth
#   STEP=overlay hnl.overlay.truth_checks run: format and id checks, weight sums, C1 on the overlaid gun hits, V3 and
#                content of one stored overlay set (SET, ROOT, WORK of slurm/overlay/overlay_set.sbatch)
#   STEP=pytest  the overlay unit tests (TESTS, default tests/test_overlay_*.py)
#   sbatch --export=ALL,STEP=m1,SIM=<dir|tar>,OUT=<json>[,LABEL=<name>] slurm/overlay/truth_checks.sbatch
#   sbatch --export=ALL,STEP=overlay,SET=pilot,WORK=<dir>,OUT=<json>[,ROOT=<table root>] slurm/overlay/truth_checks.sbatch
#SBATCH -J ov_checks
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
: "${STEP:?}"
cd /work/submit/jinboz1/hnl/hnl-displaced
source env/activate.sh > /dev/null 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "host $(hostname) start $(date -u) git $(git rev-parse --short HEAD) step $STEP"
case "$STEP" in
  m1)
    : "${SIM:?}" "${OUT:?}"
    python -u -m hnl.overlay.m1 run --sim "$SIM" --out-json "$OUT" ${LABEL:+--label $LABEL} ;;
  overlay)
    : "${SET:?}" "${WORK:?}" "${OUT:?}"
    python -u -m hnl.overlay.truth_checks run --set "$SET" --work "$WORK" --out-json "$OUT" ${ROOT:+--root $ROOT} ;;
  pytest)
    python -m pytest -q ${TESTS:-tests/test_overlay_*.py} ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
echo "end $(date -u)"
