#!/usr/bin/env bash
# DP's environment:  bash install.sh
# conda env `cohub_dp` (COHUB_DP_ENV), python 3.10. A no-op when the interpreter is already there.
if [ "${COHUB_INSTALL_IN:-}" != DP ]; then   # called from outside the environment
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install DP "$@"; exit
fi
# Package steps, run inside the environment.
set -euo pipefail
pip install torch==2.4.1 torchvision
pip install \
  zarr==2.12.0 wandb ipdb gpustat dm_control \
  omegaconf hydra-core==1.2.0 dill==0.3.5.1 \
  einops==0.4.1 diffusers==0.11.1 numba==0.56.4 \
  moviepy imageio av matplotlib termcolor sympy \
  h5py opencv-python numpy==1.23.5 \
  huggingface_hub==0.25.2 pandas pyarrow imageio-ffmpeg

pip install -e .

# install the baselines package
cd ../../
pip install -e .
# The editable install replaces numpy. Pin back the version numba 0.56.4 imports.
pip install "numpy==1.23.5" "opencv-python-headless<5"
