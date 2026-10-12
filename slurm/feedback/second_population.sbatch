#!/bin/bash
# Feedback round 2, task F (CPU): a second displaced population (V0 daughters; photon-conversion e+- as a separate
# group) at the frozen working points, hnl.eval.second_population. Reads the stored per-particle tables only (no track
# building, no GPU). The test step runs once and refuses to overwrite its output.
# Usage: env -u OMP_NUM_THREADS sbatch -J fb2F_val  slurm/feedback/second_population.sbatch census,checks,val OUT_DIR
#        env -u OMP_NUM_THREADS sbatch -J fb2F_test slurm/feedback/second_population.sbatch test OUT_DIR
#        env -u OMP_NUM_THREADS sbatch -J fb2F_test_add slurm/feedback/second_population.sbatch test OUT_DIR --add
#        env -u OMP_NUM_THREADS sbatch -J fbF2_val slurm/feedback/second_population.sbatch val OUT_DIR --add   (round 3, F2:
#          OUT_DIR = runs/feedback_v3/second_population_v2, a copy of the round-2 read-out; likewise test --add)
#   (extra arguments after OUT_DIR are passed to every command)
#SBATCH -J fb2F_secpop
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:50:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
CMDS=$1; OUT=$2; shift 2
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID; $CMDS $* -> $OUT; $(date -u +%FT%TZ)"
sha256sum src/hnl/eval/second_population.py
for c in ${CMDS//,/ }; do
    echo "== $c $* $(date -u +%FT%TZ)"
    python -m hnl.eval.second_population "$c" --out "$OUT" "$@"
done
echo "done $(date -u +%FT%TZ)"
