#!/bin/bash
# Build the hnl-displaced Python environment on subMIT (brief 4.4), adapted from the main project's
# env/make_submit_env.sh (gnn-displaced repo dc603528). Stages:
#   deps   internet, no GPU (CPU job, slurm/env_deps.sbatch): micromamba; Python 3.12 env; CUDA 12.6 toolchain env
#          (nvcc + headers for the extension builds; /usr/local/cuda on the nodes is 13.4 and there is no 12.6);
#          the venv with torch 2.11.0+cu126, PyG 2.7.0 + pt211cu126 extension wheels, lightning 2.6.1 and the rest;
#          FRNN and acorn sources at their pinned commits. Purges the pip cache at the end.
#   build  no internet needed, A30 node (slurm/env_build_smoke.sbatch): FRNN + prefix_sum wheels, acorn = pin +
#          patches/acorn (editable, CUDA extension), hnl (editable).
#   smoke  A30: env/smoke_gpu.py, then env/lock.txt.
#
#   bash env/make_env.sh --stage deps
#   bash env/make_env.sh --stage build,smoke
#
# Every location comes from env/env.sh (under /work/submit/jinboz1/hnl; it refuses anything under /home), and
# micromamba runs with HOME pointing at $HNL_BUILD_HOME and without CONDARC / MAMBARC.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/env/env.sh" || exit 1

STAGES=deps,build,smoke
while [ $# -gt 0 ]; do
    case "$1" in
        --stage|--stages) STAGES=$2; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

PYTHON_VERSION=3.12
CUDA_VARIANT=cu126
CUDA_VERSION=12.6
TORCH_VERSION=2.11.0
PYG_VERSION=2.7.0
PYG_WHEELS="pyg_lib==0.6.0+pt211$CUDA_VARIANT torch_scatter==2.1.2+pt211$CUDA_VARIANT torch_sparse==0.6.18+pt211$CUDA_VARIANT torch_cluster==1.6.3+pt211$CUDA_VARIANT"
LIGHTNING_VERSION=2.6.1
FRNN_URL=https://github.com/murnanedaniel/FRNN.git
FRNN_COMMIT=c3d7bc3fedc67d24929829411eae4ba31f881d76
ACORN_URL=https://gitlab.cern.ch/gnn4itkteam/acorn.git
ACORN_COMMIT=99d153418981cba9018e70b476fc7ea0f17bacd4
# Same commit in the main project's checkout on /ceph; used only if gitlab.cern.ch cannot serve the commit.
ACORN_LOCAL_MIRROR=/ceph/submit/data/user/j/jinboz1/gnn-displaced/repo/third_party/acorn
TRACKML_URL="https://github.com/LAL/trackml-library/archive/53a165e15a2c885f54c2bef1bd1ed53db6ed9648.tar.gz"
MICROMAMBA_URL=https://micro.mamba.pm/api/micromamba/linux-64/latest
ARCH_LIST="7.0;8.0;8.6"     # A30 = 8.0 (all compared numbers), RTX A6000 = 8.6, V100 = 7.0 (fallback only)

MICROMAMBA=$MAMBA_ROOT_PREFIX/bin/micromamba
PYENV=$MAMBA_ROOT_PREFIX/envs/py312
TOOLCHAIN=$MAMBA_ROOT_PREFIX/envs/cuda126
VENV=$HNL_ENV/venv
SRC=$HNL_ENV/src
WHEELS=$HNL_ENV/wheels
MARK=$HNL_ENV/.stages
CONSTRAINTS=$HNL_ENV/constraints.txt
mkdir -p "$SRC" "$WHEELS" "$MARK"

# what hnl, acorn and the tests import; versions come from the Perlmutter freeze where it has them
PIP_PACKAGES=(
    "lightning==$LIGHTNING_VERSION" "pytorch-lightning==$LIGHTNING_VERSION" torchmetrics lightning-utilities
    numpy scipy pandas h5py pyarrow matplotlib mplhep hist pytest
    scikit-learn numba llvmlite networkx PyYAML tqdm click class-resolver atlasify seaborn polars uproot awkward
    wandb psutil ninja setuptools wheel packaging
    "trackml @ $TRACKML_URL"
)

log() { echo "[make_env $(date -u +%FT%TZ)] $*"; }
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
space() { log "space ($1): $(quota -s 2>/dev/null | awk '/submit_filer$/{getline; print "home " $1 "/" $2} /submit_filer_2$/{getline; print "work " $1 "/" $2}' | paste -sd' ') | du home $(du -sh "$HOME" 2>/dev/null | cut -f1)"; }
log "repo $REPO | venv $VENV | mamba $MAMBA_ROOT_PREFIX | stages $STAGES | host $(hostname) | job ${SLURM_JOB_ID:-none}"

# ------------------------------------------------------------------------------------------------ deps (internet)
stage_deps() {
    space "before deps"
    if [ -e "$VENV/bin/python" ]; then
        echo "$VENV exists; remove it first to rebuild (the build is not incremental)" >&2
        exit 1
    fi
    if [ ! -x "$MICROMAMBA" ]; then
        log "micromamba -> $MICROMAMBA"
        mkdir -p "$MAMBA_ROOT_PREFIX"
        curl -fsSL "$MICROMAMBA_URL" | tar -xj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
    fi
    log "micromamba $(hnl_mm --version)"
    if [ ! -x "$PYENV/bin/python" ]; then
        log "Python $PYTHON_VERSION -> $PYENV"
        hnl_mm create -y -p "$PYENV" -c conda-forge --override-channels "python=$PYTHON_VERSION" pip
    fi
    if [ ! -x "$TOOLCHAIN/bin/nvcc" ]; then
        log "CUDA $CUDA_VERSION toolchain -> $TOOLCHAIN"
        # headers that torch's ATen/cuda/CUDAContext.h (included by FRNN and acorn's kernels) needs, plus thrust/cub
        hnl_mm create -y -p "$TOOLCHAIN" -c conda-forge --override-channels "cuda-version=$CUDA_VERSION" cuda-nvcc \
            cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev
    fi
    "$TOOLCHAIN/bin/nvcc" --version | tail -2
    hnl_mm clean -a -y > /dev/null || true

    log "venv $VENV"
    "$PYENV/bin/python" -m venv "$VENV"
    # constraints = the main project's Perlmutter freeze without CUDA-specific wheels, URL references and NERSC-only
    # packages (torch / PyG / nvidia-* / triton come from the cu126 indexes below)
    grep -v -E '^#|@ |^(torch|torchvision|triton|pyg_lib|torch_scatter|torch_sparse|torch_cluster|torch-geometric)==|^nvidia-|^cuda-|^(apex|nersc-|mpi4py|deepspeed)' \
        "$REPO/env/constraints/perlmutter-pip-freeze.txt" > "$CONSTRAINTS"
    local pip="$VENV/bin/python -m pip"
    $pip install --upgrade pip
    log "torch $TORCH_VERSION+$CUDA_VARIANT"
    $pip install --prefer-binary --index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
        --extra-index-url https://pypi.org/simple -c "$CONSTRAINTS" "torch==$TORCH_VERSION+$CUDA_VARIANT"
    log "PyG $PYG_VERSION + pt211$CUDA_VARIANT extension wheels"
    # shellcheck disable=SC2086
    $pip install --prefer-binary -f "https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VARIANT.html" \
        -c "$CONSTRAINTS" "torch-geometric==$PYG_VERSION" $PYG_WHEELS
    log "the rest"
    $pip install --prefer-binary -c "$CONSTRAINTS" "${PIP_PACKAGES[@]}"
    $pip check || log "WARNING: pip check reports conflicts (see above)"
    "$VENV/bin/python" -I - <<'EOF'
import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster, pyg_lib, lightning, numpy, pandas
print("torch", torch.__version__, "cuda", torch.version.cuda, "| pyg", torch_geometric.__version__, "| lightning",
      lightning.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
assert torch.__version__ == "2.11.0+cu126" and torch.version.cuda.startswith("12.6"), "unexpected torch build"
EOF

    log "FRNN sources @ $FRNN_COMMIT"
    [ -d "$SRC/FRNN/.git" ] || git clone --recursive "$FRNN_URL" "$SRC/FRNN"
    git -C "$SRC/FRNN" checkout -q "$FRNN_COMMIT"
    git -C "$SRC/FRNN" submodule update --init -q
    git -C "$SRC/FRNN" log -1 --format='FRNN %H %ci %s'

    log "acorn sources @ $ACORN_COMMIT"
    [ -d "$SRC/acorn/.git" ] || git clone -q "$ACORN_URL" "$SRC/acorn"
    if ! git -C "$SRC/acorn" cat-file -e "$ACORN_COMMIT^{commit}" 2>/dev/null; then
        log "commit not in the gitlab clone; fetching the same commit from $ACORN_LOCAL_MIRROR"
        git -C "$SRC/acorn" fetch -q "$ACORN_LOCAL_MIRROR" "$ACORN_COMMIT"
    fi
    git -C "$SRC/acorn" checkout -q --detach "$ACORN_COMMIT"
    [ "$(git -C "$SRC/acorn" rev-parse HEAD)" = "$ACORN_COMMIT" ] || { echo "acorn HEAD is not the pin" >&2; exit 1; }
    git -C "$SRC/acorn" log -1 --format='acorn %H %ci %s'
    git -C "$SRC/acorn" branch -a --contains "$ACORN_COMMIT" | head -5 | sed 's/^/    contained in: /'

    $pip cache purge > /dev/null 2>&1 || true
    date -u +%FT%TZ > "$MARK/deps.done"
    space "after deps"
    log "deps done"
}

# ------------------------------------------------------------------------------------------------ build (A30 node)
stage_build() {
    [ -f "$MARK/deps.done" ] || { echo "run --stage deps first" >&2; exit 1; }
    space "before build"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export CUDA_HOME=$TOOLCHAIN
    export CUDA_INC_PATH=$TOOLCHAIN/targets/x86_64-linux/include      # conda-forge's CUDA 12 header layout
    export LIBRARY_PATH=$TOOLCHAIN/targets/x86_64-linux/lib:$TOOLCHAIN/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
    export PATH=$TOOLCHAIN/bin:$PATH
    export TORCH_CUDA_ARCH_LIST=$ARCH_LIST
    export MAX_JOBS=${SLURM_CPUS_PER_TASK:-4}
    export CC=${CC:-gcc} CXX=${CXX:-g++}
    local gccmaj
    gccmaj=$("$CXX" -dumpversion | cut -d. -f1)
    if [ "$gccmaj" -lt 9 ] || [ "$gccmaj" -gt 13 ]; then
        echo "host compiler $CXX is GCC $gccmaj; nvcc 12.6 + torch 2.11 need GCC 9-13" >&2
        exit 1
    fi
    for h in cuda_runtime_api.h cusparse.h cublas_v2.h cublasLt.h cusolverDn.h cub/cub.cuh; do
        [ -e "$CUDA_INC_PATH/$h" ] || [ -e "$CUDA_HOME/include/$h" ] || { echo "missing CUDA header $h" >&2; exit 1; }
    done
    log "CUDA_HOME $CUDA_HOME ($(nvcc --version | grep -o 'release [0-9.]*')), $CXX $gccmaj, arch $TORCH_CUDA_ARCH_LIST, $MAX_JOBS jobs"

    # 1) acorn = pin + patches/acorn (git apply --check first, inside the script)
    bash "$REPO/env/apply_acorn_patches.sh"

    # 2) FRNN and its prefix_sum (torch >= 2.1 headers need C++17)
    if ! ls "$WHEELS"/frnn-*.whl > /dev/null 2>&1 || ! ls "$WHEELS"/prefix_sum-*.whl > /dev/null 2>&1; then
        sed -i 's/-std=c++14/-std=c++17/' "$SRC/FRNN/setup.py"
        pip wheel --no-build-isolation --no-deps --no-index -w "$WHEELS" "$SRC/FRNN/external/prefix_sum"
        pip wheel --no-build-isolation --no-deps --no-index -w "$WHEELS" "$SRC/FRNN"
    fi
    pip install --no-deps --no-index "$WHEELS"/prefix_sum-*.whl "$WHEELS"/frnn-*.whl
    # the pinned commit prints the grid size on every call (one line per training step): drop that print (pure
    # Python file, so patching the installed copy is enough; DECISIONS.md 2026-10-03)
    sed -i 's/^\( *\)print(grid_params_cuda\[i, grid_delta_idx + 1:grid_total_idx\])$/\1pass  # upstream debug print removed (hnl env\/make_env.sh)/' \
        "$VENV/lib/python3.12/site-packages/frnn/frnn.py"
    if grep -q "print(grid_params_cuda" "$VENV/lib/python3.12/site-packages/frnn/frnn.py"; then
        echo "FRNN debug-print patch did not apply" >&2; exit 1
    fi

    # 3) acorn, editable, with its CUDA kernels (connected components, path tracing)
    rm -rf "$SRC/acorn/build"
    ACORN_BUILD_CUDA_EXT=1 pip install --no-deps --no-build-isolation --no-index -e "$SRC/acorn"

    # 4) this repo's package, editable
    pip install --no-deps --no-build-isolation --no-index -e "$REPO"
    python -I -c "import torch, torch_geometric, lightning, torch_scatter, torch_cluster, frnn, prefix_sum, trackml, acorn, hnl; \
import acorn.cuda_ext as ce; print('OK torch', torch.__version__, '| pyg', torch_geometric.__version__, '| lightning', \
lightning.__version__, '| acorn', acorn.__file__, '| cuda_ext', ce.__file__, '| hnl', hnl.__file__)"
    date -u +%FT%TZ > "$MARK/build.done"
    space "after build"
    log "build done"
}

# ------------------------------------------------------------------------------------------------ smoke (A30)
stage_smoke() {
    [ -f "$MARK/build.done" ] || { echo "run --stage build first" >&2; exit 1; }
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    python -I "$REPO/env/smoke_gpu.py"
    local lock=$REPO/env/lock.txt
    {
        echo "# hnl-displaced environment lock, written by env/make_env.sh --stage smoke"
        echo "# date_utc $(date -u +%FT%TZ) host $(hostname) slurm_job ${SLURM_JOB_ID:-none}"
        echo "# gpu $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1)"
        echo "# micromamba $(hnl_mm --version); python $(python -I -c 'import sys; print(sys.version.split()[0])'); venv $VENV"
        echo "# torch_cuda_arch_list $ARCH_LIST; nvcc $("$TOOLCHAIN/bin/nvcc" --version | grep -o 'release [0-9.]*'); host gcc $(gcc -dumpfullversion)"
        echo "# acorn $ACORN_URL @ $(git -C "$SRC/acorn" rev-parse HEAD) + patches/acorn:"
        sed 's/^/#     /' "$REPO/patches/acorn/SHA256SUMS"
        echo "# acorn working tree: $(bash "$REPO/env/apply_acorn_patches.sh" --verify)"
        echo "# FRNN $FRNN_URL @ $(git -C "$SRC/FRNN" rev-parse HEAD) (setup.py: -std=c++14 -> -std=c++17; frnn.py: grid debug print removed)"
        echo "# ---- micromamba env $PYENV (explicit)"
        hnl_mm list -p "$PYENV" --explicit --md5 | sed 's/^/# /'
        echo "# ---- micromamba env $TOOLCHAIN (explicit)"
        hnl_mm list -p "$TOOLCHAIN" --explicit --md5 | sed 's/^/# /'
        echo "# ---- pip freeze (editable installs excluded: acorn, hnl)"
        pip freeze --exclude-editable
    } > "$lock"
    date -u +%FT%TZ > "$MARK/smoke.done"
    log "smoke done: lock in $lock (commit it)"
}

has_stage deps && stage_deps
has_stage build && stage_build
has_stage smoke && stage_smoke
log "finished stages $STAGES"
