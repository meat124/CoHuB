#!/usr/bin/env bash
# Install GauDP's environment and pretrained weights:  bash install.sh
# Creates the conda env `gaudp` (COHUB_GAUDP_ENV), python 3.10.
if [ "${COHUB_INSTALL_IN:-}" != GauDP ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install GauDP "$@"; exit
fi
# Package steps, run inside the active conda/virtual environment.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICIES_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CUROPE_DIR="${POLICY_DIR}/gaudp/third_party/noposplat/model/encoder/backbone/croco/curope"

TORCH_VERSION="${GAUDP_TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${GAUDP_TORCHVISION_VERSION:-0.16.2}"
TORCHAUDIO_VERSION="${GAUDP_TORCHAUDIO_VERSION:-2.1.2}"
TORCH_INDEX_URL="${GAUDP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
RASTERIZER_URL="${GAUDP_RASTERIZER_URL:-git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git}"

echo "[GauDP] python=$(command -v python)"
echo "[GauDP] environment=${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-<not activated>}}"
if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[GauDP] WARNING: no conda/virtual environment is active; a dedicated environment is recommended." >&2
fi
python - <<'PY'
import sys

if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit(
        f"GauDP requires Python 3.10 or 3.11; current interpreter is {sys.version.split()[0]}"
    )
PY

# PyTorch 2.1's cpp_extension needs pkg_resources, so setuptools stays below 81.
python -m pip install --upgrade pip "setuptools<81" wheel packaging ninja

if [[ "${GAUDP_SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
    echo "[GauDP] installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
    python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"
else
    echo "[GauDP] keeping the environment's existing PyTorch installation"
fi

# Building cuRoPE and the Gaussian rasterizer needs nvcc. Order: GAUDP_NVCC,
# CUDA_HOME, PATH, then a conda install of the matching toolkit.
if [[ "${GAUDP_SKIP_CUDA_EXTENSIONS:-0}" != "1" ]]; then
    NVCC_CANDIDATE="${GAUDP_NVCC:-}"
    if [[ -z "${NVCC_CANDIDATE}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        NVCC_CANDIDATE="${CUDA_HOME}/bin/nvcc"
    fi
    if [[ -z "${NVCC_CANDIDATE}" ]]; then
        NVCC_CANDIDATE="$(command -v nvcc || true)"
        # An nvcc that merely happens to be on PATH (a distribution's
        # nvidia-cuda-toolkit, say) is used only when its CUDA major version is
        # PyTorch's. Otherwise it counts as absent, and the matching toolkit goes
        # into this conda environment below; the system's one is left alone. An
        # nvcc named through GAUDP_NVCC or CUDA_HOME is never set aside: the
        # version check further down reports a mismatch there.
        if [[ -n "${NVCC_CANDIDATE}" && -x "${NVCC_CANDIDATE}" ]]; then
            PATH_NVCC_MAJOR="$("${NVCC_CANDIDATE}" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | sed -n 1p)"
            TORCH_CUDA_MAJOR="$(python -c 'import torch; print((torch.version.cuda or "").split(".")[0])')"
            if [[ -n "${TORCH_CUDA_MAJOR}" && "${PATH_NVCC_MAJOR}" != "${TORCH_CUDA_MAJOR}" ]]; then
                echo "[GauDP] ${NVCC_CANDIDATE} is CUDA ${PATH_NVCC_MAJOR:-unknown}, PyTorch is built for CUDA ${TORCH_CUDA_MAJOR}; not using it"
                NVCC_CANDIDATE=""
            fi
        fi
    fi

    if [[ -z "${NVCC_CANDIDATE}" || ! -x "${NVCC_CANDIDATE}" ]]; then
        if [[ "${GAUDP_AUTO_INSTALL_CUDA_TOOLKIT:-1}" == "1" \
            && -n "${CONDA_PREFIX:-}" \
            && -n "$(command -v conda || true)" ]]; then
            TORCH_CUDA_VERSION="$(python - <<'PY'
import torch

if torch.version.cuda is None:
    raise SystemExit("installed PyTorch is CPU-only; GauDP requires a CUDA build")
print(torch.version.cuda)
PY
)"
            CUDA_TOOLKIT_CHANNEL="${GAUDP_CUDA_TOOLKIT_CHANNEL:-nvidia/label/cuda-${TORCH_CUDA_VERSION}.0}"
            echo "[GauDP] nvcc not found; installing CUDA toolkit ${TORCH_CUDA_VERSION} into ${CONDA_PREFIX}"
            echo "[GauDP] CUDA toolkit channel=${CUDA_TOOLKIT_CHANNEL}"
            # Use the versioned channel so every CUDA component matches.
            conda install -y --override-channels \
                -c "${CUDA_TOOLKIT_CHANNEL}" \
                -c defaults \
                "cuda-toolkit=${TORCH_CUDA_VERSION}"
            NVCC_CANDIDATE="${CONDA_PREFIX}/bin/nvcc"
            if [[ ! -x "${NVCC_CANDIDATE}" ]]; then
                echo "[GauDP] CUDA toolkit installation completed but nvcc was not found at ${NVCC_CANDIDATE}." >&2
                exit 2
            fi
            export GAUDP_NVCC="${NVCC_CANDIDATE}"
            export CUDA_HOME="${CONDA_PREFIX}"
        else
            echo "[GauDP] nvcc is required to build cuRoPE and the Gaussian rasterizer." >&2
            echo "[GauDP] Activate a conda environment for automatic toolkit installation," >&2
            echo "[GauDP] install a toolkit matching torch.version.cuda manually, or set" >&2
            echo "[GauDP] GAUDP_SKIP_CUDA_EXTENSIONS=1 for data/schema-only usage." >&2
            exit 2
        fi
    fi
fi

# Pinned policy and NoPoSplat dependencies, then the baselines package itself.
python -m pip install -r "${POLICY_DIR}/requirements.txt"
python -m pip install -e "${POLICIES_ROOT}"

if [[ "${GAUDP_SKIP_CUDA_EXTENSIONS:-0}" != "1" ]]; then
    NVCC_BIN="${GAUDP_NVCC:-}"
    if [[ -z "${NVCC_BIN}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        NVCC_BIN="${CUDA_HOME}/bin/nvcc"
    fi
    if [[ -z "${NVCC_BIN}" ]]; then
        NVCC_BIN="$(command -v nvcc || true)"
    fi
    if [[ -z "${NVCC_BIN}" || ! -x "${NVCC_BIN}" ]]; then
        echo "[GauDP] nvcc is required to build cuRoPE and the Gaussian rasterizer." >&2
        echo "[GauDP] Install a CUDA toolkit matching torch.version.cuda, or set" >&2
        echo "[GauDP] GAUDP_SKIP_CUDA_EXTENSIONS=1 for data/schema-only usage." >&2
        exit 2
    fi
    export GAUDP_NVCC_BIN="${NVCC_BIN}"
    export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)}"

    # Fail early on a CUDA major-version mismatch between nvcc and PyTorch.
    python - <<'PY'
import re
import os
import subprocess
import torch

nvcc = os.environ["GAUDP_NVCC_BIN"]
output = subprocess.check_output([nvcc, "--version"], text=True)
match = re.search(r"release\s+(\d+)\.(\d+)", output)
if match is None:
    raise SystemExit("could not determine CUDA version from 'nvcc --version'")
toolkit = f"{match.group(1)}.{match.group(2)}"
runtime = torch.version.cuda
if runtime is None:
    raise SystemExit("installed PyTorch is CPU-only; GauDP requires a CUDA build")
if toolkit.split(".")[0] != runtime.split(".")[0]:
    raise SystemExit(
        "CUDA major-version mismatch: "
        f"nvcc={toolkit}, PyTorch={runtime}. Install a matching toolkit/wheel; "
        "override GAUDP_TORCH_* variables when using CUDA 12."
    )
print(f"[GauDP] CUDA toolchain: {nvcc}={toolkit}, PyTorch={runtime}")
PY

    # nvcc refuses a host gcc newer than its toolkit knows: CUDA 11.8 stops at
    # gcc 11, and Ubuntu 24.04 ships 13. The ceiling is read from the toolkit's
    # own header. A newer system gcc is left alone and the matching one goes into
    # this conda environment, where torch's extension build picks it up through
    # CC and CXX. A CC the caller set is never replaced.
    GCC_LIMIT="$(sed -n 's/^#if __GNUC__ > \([0-9][0-9]*\).*/\1/p' "${CUDA_HOME}/include/crt/host_config.h" 2>/dev/null | sed -n 1p)"
    HOST_GCC_MAJOR="$("${CC:-gcc}" -dumpversion 2>/dev/null | cut -d. -f1 || true)"
    if [[ -n "${GCC_LIMIT}" && ( -z "${HOST_GCC_MAJOR}" || "${HOST_GCC_MAJOR}" -gt "${GCC_LIMIT}" ) ]]; then
        if [[ -z "${CC:-}" \
            && "${GAUDP_AUTO_INSTALL_CUDA_TOOLKIT:-1}" == "1" \
            && -n "${CONDA_PREFIX:-}" \
            && -n "$(command -v conda || true)" ]]; then
            echo "[GauDP] host gcc is ${HOST_GCC_MAJOR:-missing}, this CUDA toolkit builds with gcc ${GCC_LIMIT} at most; installing gcc ${GCC_LIMIT} into ${CONDA_PREFIX}"
            conda install -y --override-channels -c conda-forge "gxx_linux-64=${GCC_LIMIT}.*"
            export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
            export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
            if [[ ! -x "${CC}" || ! -x "${CXX}" ]]; then
                echo "[GauDP] gcc ${GCC_LIMIT} was installed but ${CC} is missing." >&2
                exit 2
            fi
        else
            echo "[GauDP] ${CC:-gcc} is gcc ${HOST_GCC_MAJOR:-missing}; ${NVCC_BIN} builds with gcc ${GCC_LIMIT} at most." >&2
            echo "[GauDP] Point CC and CXX at a gcc ${GCC_LIMIT} or older, or run inside a conda" >&2
            echo "[GauDP] environment so that one can be installed there." >&2
            exit 2
        fi
    fi
    echo "[GauDP] host compiler: ${CC:-gcc} ($("${CC:-gcc}" -dumpversion 2>/dev/null || echo unknown))"

    # GPU architectures the extensions are built for. An extension run on a
    # card outside this list segfaults. Default: A100 (8.0), RTX 3090 / A6000
    # (8.6), RTX 4090 / RTX 6000 Ada (8.9), H100 (9.0).
    export TORCH_CUDA_ARCH_LIST="${GAUDP_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
    echo "[GauDP] building for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

    echo "[GauDP] building vendored cuRoPE (MAX_JOBS=${MAX_JOBS:-4})"
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
        --no-build-isolation --no-cache-dir --force-reinstall --no-deps "${CUROPE_DIR}"

    echo "[GauDP] building NoPoSplat Gaussian rasterizer"
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
        --no-build-isolation --no-cache-dir --force-reinstall --no-deps "${RASTERIZER_URL}"
fi

verify_args=()
if [[ "${GAUDP_SKIP_CUDA_EXTENSIONS:-0}" == "1" ]]; then
    verify_args+=(--skip-cuda-extensions)
fi
if [[ "${GAUDP_SKIP_PIP_CHECK:-0}" != "1" ]]; then
    python -m pip check
fi
python "${POLICY_DIR}/verify_install.py" "${verify_args[@]}"

echo "[GauDP] installation complete"
