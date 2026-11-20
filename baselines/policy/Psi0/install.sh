#!/usr/bin/env bash
# Psi0's environment and pretrained weights:  bash install.sh
# Creates the uv venv at psi0/.venv unless it exists, then fetches the Psi0 pretrained weights.
if [ "${COHUB_INSTALL_IN:-}" != Psi0 ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install Psi0 "$@"; exit
fi
# Package steps, run inside the environment.
#   PSI0_TORCH_BACKEND   CUDA build of torch (default cu128)
#   PSI0_FLASH_ATTN      0 skips flash-attn; Psi0 then uses sdpa
#   PSI0_CLEAN_VENV      1 rebuilds the venv from scratch
#   PSI0_KEEP_ENV        1 keeps an existing psi0/.env
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI_ROOT="${POLICY_DIR}/psi0"
POLICIES_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
TORCH_BACKEND="${PSI0_TORCH_BACKEND:-cu128}"

echo "[Psi0] PSI_ROOT=${PSI_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
[ -f "${PSI_ROOT}/pyproject.toml" ] || {
  echo "no Psi0 checkout at ${PSI_ROOT} -- clone https://github.com/physical-superintelligence-lab/Psi0" >&2
  exit 1; }

cd "${PSI_ROOT}"
export UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1
venv_flag=--allow-existing
[ "${PSI0_CLEAN_VENV:-0}" = 1 ] && venv_flag=--clear
uv venv .venv --python 3.11 "$venv_flag"
uv sync --group serve --group psi --index-strategy unsafe-best-match

# Reinstall torch when its CUDA build tag differs from TORCH_BACKEND.
have=$(.venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)
case "${have}" in
  *"+${TORCH_BACKEND}") echo "[Psi0] torch ${have} already matches ${TORCH_BACKEND}" ;;
  *)
    echo "[Psi0] torch is ${have}; installing the ${TORCH_BACKEND} build"
    uv pip install --python .venv/bin/python --torch-backend="${TORCH_BACKEND}" \
      --reinstall-package torch --reinstall-package torchvision \
      "torch==2.7.0" "torchvision==0.22.0" ;;
esac

# flash-attn from the release wheel matching this interpreter, torch build and C++ ABI.
if [ "${PSI0_FLASH_ATTN:-1}" = 1 ]; then
  abi=$(.venv/bin/python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
  wheel="https://github.com/Dao-AILab/flash-attention/releases/download/v${PSI0_FLASH_ATTN_VERSION:-2.7.4.post1}/flash_attn-${PSI0_FLASH_ATTN_VERSION:-2.7.4.post1}+cu12torch2.7cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"
  uv pip install --python .venv/bin/python "${wheel}" \
    || echo "[Psi0] flash-attn did not install -- training and serving fall back to sdpa" >&2
fi

# The policy server imports the baselines package from this environment.
uv pip install --python .venv/bin/python -e "${POLICIES_ROOT}"

# scripts/train.py requires psi0/.env to exist.
if [ -f .env ] && [ "${PSI0_KEEP_ENV:-0}" = 1 ]; then
  echo "[Psi0] keeping the existing .env"
else
  cat > .env <<ENVEOF
${WANDB_ENTITY:+WANDB_ENTITY=$WANDB_ENTITY}
PSI_HOME=${PSI_ROOT}
DATA_HOME=${POLICY_DIR}/data
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
TOKENIZERS_PARALLELISM=false
DEEPSPEED_LOG_LEVEL=warning
TF_CPP_MIN_LOG_LEVEL=3
AV_LOG_LEVEL=quiet
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENVEOF
  echo "[Psi0] wrote ${PSI_ROOT}/.env"
fi

.venv/bin/python - <<'PY'
import torch, psi
print(f"[Psi0] psi {psi.__version__}, torch {torch.__version__} "
      f"(cuda {torch.version.cuda}, arch {torch.cuda.get_arch_list()})")
import baselines; print("[Psi0] baselines ok")
try:
    from transformers.utils import is_flash_attn_2_available
    print(f"[Psi0] attention backend: "
          f"{'flash_attention_2' if is_flash_attn_2_available() else 'sdpa'}")
except Exception:
    print("[Psi0] attention backend: sdpa")
PY

echo "[Psi0] Installation finished."
echo "[Psi0] Activate: source ${PSI_ROOT}/.venv/bin/activate"
