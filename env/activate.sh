# Source (bash) in every job script and interactive shell:
#     source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
# = env/env.sh (all caches / TMPDIR under /work, refuses /home) + the venv built by env/make_env.sh.
# Thread counts follow the Slurm allocation; outside Slurm (login node) they default to 4.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || return 1
if [ ! -x "$HNL_ENV/venv/bin/python" ]; then
    echo "activate.sh: no venv at $HNL_ENV/venv (run env/make_env.sh)" >&2
    return 1
fi
# shellcheck disable=SC1091
source "$HNL_ENV/venv/bin/activate"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-$OMP_NUM_THREADS}
