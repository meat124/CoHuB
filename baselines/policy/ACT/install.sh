#!/usr/bin/env bash
# ACT's environment:  bash install.sh
# conda env `cohub_act` (COHUB_ACT_ENV), python 3.10. A no-op when the interpreter is already there.
if [ "${COHUB_INSTALL_IN:-}" != ACT ]; then   # called from outside the environment
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry install ACT "$@"; exit
fi
# Package steps, run inside the environment.
set -euo pipefail
pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython torch==2.4.1 torchvision numpy wandb

cd detr && pip install -e .

# install the baselines package
cd ../../..
pip install -e .