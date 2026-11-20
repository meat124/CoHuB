#!/usr/bin/env bash
# Install GR00T_N17's environment and pretrained weights:  bash install.sh
# Creates the uv venv at gr00t_n17/.venv and downloads the GR00T N1.7 base weights.
if [ "${COHUB_INSTALL_IN:-}" != GR00T_N17 ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install GR00T_N17 "$@"; exit
fi
# Package steps, run inside the environment.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
POLICIES_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[GR00T_N17] GR00T_ROOT=${GR00T_ROOT}"
echo "[GR00T_N17] POLICIES_ROOT=${POLICIES_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${GR00T_ROOT}"
# `uv sync` and `uv run` fail on x86_64: pyproject requires aarch64 wheels that are
# not shipped. Use `uv pip install -e .` and the venv's python directly.
# Every `uv pip install` names this venv's python: without it uv installs into
# whichever venv the calling shell has activated (the simulation .venv, after the
# main README's `source .venv/bin/activate`) and leaves this one empty.
PYTHON_BIN="${GR00T_ROOT}/.venv/bin/python"
uv venv --clear --python 3.10 "${GR00T_ROOT}/.venv"
uv pip install --python "${PYTHON_BIN}" -e .
"${PYTHON_BIN}" -c "import gr00t; print('GR00T ok')"

uv pip install --python "${PYTHON_BIN}" -e "${POLICIES_ROOT}"
uv pip install --python "${PYTHON_BIN}" h5py pyyaml
"${PYTHON_BIN}" -c "import baselines; print('baselines ok')"

echo "[GR00T_N17] Installation finished."
echo "[GR00T_N17] Policy server env: source ${GR00T_ROOT}/.venv/bin/activate"
