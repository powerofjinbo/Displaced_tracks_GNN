# Source (bash) before any work on subMIT, interactive or in a Slurm job:
#     source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
# Points every cache, temporary, config and data directory that Python, pip, micromamba, torch, triton, numba,
# matplotlib, wandb and git might write to under /work/submit/jinboz1/hnl. The home directory (10 GB quota) must
# never receive anything: every location is resolved (symbolic links followed) and checked BEFORE anything is
# exported or created; if any resolves under /home or the home directory, nothing is exported and this returns 1.
# activate.sh sources this file and then activates the venv.

_hnl_root=${HNL_ROOT:-/work/submit/jinboz1/hnl}
_hnl_cache=$_hnl_root/cache
# NAME=value, one per line; values must not contain whitespace
_hnl_pairs="
HNL_ROOT=$_hnl_root
HNL_REPO=$_hnl_root/hnl-displaced
HNL_ENV=$_hnl_root/env
HNL_CACHE=$_hnl_cache
HNL_TMP=$_hnl_root/tmp
HNL_LOGS=$_hnl_root/logs
HNL_BUILD_HOME=$_hnl_root/build_home
MAMBA_ROOT_PREFIX=$_hnl_root/env/mamba
TMPDIR=$_hnl_root/tmp
XDG_CACHE_HOME=$_hnl_cache/xdg-cache
PIP_CACHE_DIR=$_hnl_cache/pip
MPLCONFIGDIR=$_hnl_cache/matplotlib
TORCH_HOME=$_hnl_cache/torch
TRITON_CACHE_DIR=$_hnl_cache/triton
CUDA_CACHE_PATH=$_hnl_cache/nv
NUMBA_CACHE_DIR=$_hnl_cache/numba
WANDB_DIR=$_hnl_root/logs/wandb
CONDA_PKGS_DIRS=$_hnl_cache/conda-pkgs
XDG_CONFIG_HOME=$_hnl_cache/xdg-config
XDG_DATA_HOME=$_hnl_cache/xdg-data
PYTHONUSERBASE=$_hnl_cache/python-userbase
IPYTHONDIR=$_hnl_cache/ipython
TORCHINDUCTOR_CACHE_DIR=$_hnl_cache/torchinductor
TORCH_EXTENSIONS_DIR=$_hnl_cache/torch_extensions
HF_HOME=$_hnl_cache/huggingface
TRITON_HOME=$_hnl_cache/triton-home
WANDB_CACHE_DIR=$_hnl_cache/wandb-cache
WANDB_CONFIG_DIR=$_hnl_cache/wandb-config
WANDB_DATA_DIR=$_hnl_cache/wandb-data
"
# The first 29 lines are the brief's 4.3 lists (TRITON_HOME and the WANDB_*_DIR are extra: those tools also
# default to $HOME). HNL_DATA is on /ceph and is not a cache; it is exported below.

(return 0 2>/dev/null) && _hnl_sourced=1 || _hnl_sourced=0
_hnl_forbidden=""
for _h in /home "${HOME:-}" "$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"; do
    if [ -n "$_h" ]; then
        _h=$(realpath -m "$_h")
        case " $_hnl_forbidden " in *" $_h "*) ;; *) _hnl_forbidden="$_hnl_forbidden $_h" ;; esac
    fi
done
_hnl_bad=0
case "$_hnl_root" in *[[:space:]]*) echo "env.sh: refusing: HNL_ROOT contains whitespace" >&2; _hnl_bad=1 ;; esac
for _p in $_hnl_pairs; do
    _real=$(realpath -m "${_p#*=}")
    for _h in $_hnl_forbidden; do
        case "$_real/" in
            "$_h"/*) echo "env.sh: refusing: ${_p%%=*}=${_p#*=} resolves to $_real, under $_h" >&2; _hnl_bad=1 ;;
        esac
    done
done
if [ "$_hnl_bad" != 0 ]; then
    echo "env.sh: nothing exported, nothing created" >&2
    unset _hnl_root _hnl_cache _hnl_pairs _hnl_forbidden _hnl_bad _h _p _real
    if [ "$_hnl_sourced" = 1 ]; then
        unset _hnl_sourced
        return 1
    fi
    exit 1
fi
for _p in $_hnl_pairs; do
    export "$_p"
    case "${_p%%=*}" in HNL_REPO) ;; *) mkdir -p "${_p#*=}" ;; esac
done
unset _hnl_root _hnl_cache _hnl_pairs _hnl_forbidden _hnl_bad _hnl_sourced _h _p _real

export HNL_DATA=${HNL_DATA:-/ceph/submit/data/user/j/jinboz1/hnl_lrt}
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export GIT_TERMINAL_PROMPT=0
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLBACKEND=Agg

# micromamba must never see the real home (it writes ~/.conda/environments.txt and reads ~/.mambarc)
hnl_mm() {
    env -u CONDARC -u MAMBARC HOME="$HNL_BUILD_HOME" XDG_CONFIG_HOME="$HNL_BUILD_HOME/.config" \
        "$MAMBA_ROOT_PREFIX/bin/micromamba" "$@"
}
