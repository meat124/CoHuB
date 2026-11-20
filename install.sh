#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[install] simulation environment (.venv, Python 3.12)"
[ -d .venv ] || uv venv --python 3.12 .venv
grep -q OMNI_KIT_ACCEPT_EULA .venv/bin/activate || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

echo "[install] Isaac Sim 6.0.1"
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

echo "[install] Isaac Lab 3.0"
[ -d external/IsaacLab ] || git clone -b release/3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git external/IsaacLab
git -C external/IsaacLab checkout 6a7acb0320a0bdc15b13e44e83b575e00797faf4
uv pip install --prerelease=allow \
    -e external/IsaacLab/source/isaaclab -e external/IsaacLab/source/isaaclab_physx \
    -e external/IsaacLab/source/isaaclab_assets -e external/IsaacLab/source/isaaclab_tasks \
    -e external/IsaacLab/source/isaaclab_teleop -e external/IsaacLab/source/isaaclab_visualizers

echo "[install] Isaac Teleop"
uv pip install --prerelease=allow "isaacteleop[retargeters,ui,cloudxr]~=1.3.0"

echo "[install] CoHuB"
uv pip install -e . "msgpack>=1.0.8" "msgpack-numpy>=0.4.8"

echo "[install] assets"
.venv/bin/python scripts/setup_assets.py rooms
.venv/bin/python scripts/setup_assets.py isaac --dest isaac_assets/Isaac

echo "[install] done"
