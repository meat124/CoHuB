#!/bin/bash
# LatentToM's environment:  bash install.sh
# conda env `latenttom` (COHUB_LATENTTOM_ENV), python 3.10. A no-op when the interpreter is already there.
if [ "${COHUB_INSTALL_IN:-}" != LatentToM ]; then   # called from outside the environment
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install LatentToM "$@"; exit
fi
# Package steps, run inside the environment.
set -euo pipefail

# Version pins follow policy/DP/install.sh. pytorch3d is not needed.
pip install torch==2.4.1 torchvision
pip install \
  zarr==2.12.0 wandb ipdb gpustat \
  omegaconf hydra-core==1.2.0 dill==0.3.5.1 \
  einops==0.4.1 diffusers==0.11.1 numba==0.56.4 \
  moviepy imageio av imagecodecs matplotlib termcolor sympy \
  h5py opencv-python numpy==1.23.5 threadpoolctl \
  huggingface_hub==0.25.2 pandas pyarrow

# Install the baselines package itself.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICIES_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${POLICIES_ROOT}"

# The install above replaces numpy. Pin back the version numba 0.56.4 imports.
pip install "numpy==1.23.5"
