#!/bin/bash
# FastWAM's environment and pretrained weights:  bash install.sh
# Creates the conda env `fastwam` (COHUB_FASTWAM_ENV) unless it exists, then fetches the Wan2.2 weights and builds the ActionDiT backbone.
if [ "${COHUB_INSTALL_IN:-}" != FastWAM ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install FastWAM "$@"; exit
fi
# Package steps, run inside the environment.
set -euo pipefail

ENV_NAME=${1:-fastwam}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

conda install -c conda-forge "ffmpeg>=6,<7" -y
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install "opencv-python<5"   # 5.0 needs numpy>=2; fastwam pins 1.26.4
pip install -e "${FASTWAM_DIR}"
pip install -e "${ROOT_DIR}"
