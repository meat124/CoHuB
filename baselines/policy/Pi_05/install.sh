#!/usr/bin/env bash
# Pi_05's environment and pretrained weights:  bash install.sh
# Creates the uv venv at openpi/.venv unless it exists. openpi downloads pi05_base itself on the first run.
if [ "${COHUB_INSTALL_IN:-}" != Pi_05 ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install Pi_05 "$@"; exit
fi
# Package steps, run inside the environment.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${POLICY_DIR}/openpi"
POLICIES_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[Pi_05] OPENPI_ROOT=${OPENPI_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${OPENPI_ROOT}"
# `uv sync` builds openpi/.venv whatever the calling shell has activated. Every
# `uv pip install` names that venv's python: without it uv installs into the
# activated venv (the simulation .venv, after the main README's
# `source .venv/bin/activate`) and this one never gets the baselines package.
PYTHON_BIN="${OPENPI_ROOT}/.venv/bin/python"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "${PYTHON_BIN}" -e .

uv pip install --python "${PYTHON_BIN}" -e "${POLICIES_ROOT}"
# From outside the checkout, so the import finds the installed package and not a folder beside it.
(cd / && "${PYTHON_BIN}" -c "import baselines; print('baselines ok')")

echo "[Pi_05] Installation finished."
echo "[Pi_05] Activate: source ${OPENPI_ROOT}/.venv/bin/activate"
