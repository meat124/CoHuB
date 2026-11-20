#!/bin/bash
# Build DP's training data:  bash process_data.sh <task>
# Downloads the dataset when it is missing and writes the task's zarr replay
# buffers, train and val, under the DP environment; no GPU. Finished zarrs are
# kept; COHUB_DATA_REBUILD=1 converts again.
if [ "${COHUB_DATA_IN:-}" != DP ]; then   # called from outside: load the config first
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data DP "$@"; exit
fi
set -euo pipefail

TASK=${1:?usage: process_data.sh <task>}
MODE=${2:-decentralized}

WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; export COHUB_WT=$WT
# shellcheck source=../../common.sh
source "$WT/baselines/common.sh"
TASK=$(mh_task_word "$TASK")
ENV_CFG=$(mh_env_cfg_type "$MODE") || exit 2
PY=$(cohub_policy_python DP)
OUT=$WT/baselines/data
DP_DATA=$WT/baselines/policy/DP/data

# The LeRobot export: datasets/<task>/lerobot, else datasets/<task> itself.
cohub_fetch_task "$TASK"
SRC=$WT/datasets/$TASK/lerobot
[ -f "$SRC/meta/info.json" ] || SRC=$WT/datasets/$TASK
[ -f "$SRC/meta/info.json" ] || {
    echo "no LeRobot export under $WT/datasets/$TASK (looked for lerobot/meta/info.json and meta/info.json)" >&2
    exit 1
}

[ -x "$PY" ] || { echo "no DP python at $PY -- baselines/policy/DP/install.sh builds it (or set COHUB_DP_PY)" >&2; exit 1; }
mkdir -p "$OUT" "$DP_DATA"

# A zarr is a directory: link it with `ln -sfn`.
link() {  # <src.zarr> <dest name>
    local src=$1 dest=$DP_DATA/$2
    [ -d "$src" ] || { echo "converter did not write $src" >&2; exit 1; }
    ln -sfn "$src" "$dest"
    echo "  linked $2 -> ${src#$WT/}"
}

case "$MODE" in
  decentralized)
    # One camera (the robot's own ego view), state 43, action 35, for every
    # robot listed in meta/modality.json.
    ROBOTS=$("$PY" -c 'import json, sys; v = json.load(open(sys.argv[1]))["video"]; print(" ".join(r for r, c in (("robot_a", "ego_a"), ("robot_b", "ego_b"), ("robot_c", "ego_c")) if c in v))' "$SRC/meta/modality.json")
    built=1   # the converter writes .zattrs last
    for r in $ROBOTS; do
      [ -f "$OUT/${TASK}_diffusion_policy_${r}.zarr/.zattrs" ] && [ -f "$OUT/${TASK}_diffusion_policy_${r}_val.zarr/.zattrs" ] || built=0
    done
    if [ "$built" = 1 ] && [ "${COHUB_DATA_REBUILD:-0}" != 1 ]; then
      echo "[prepare] DP $TASK: reusing ${OUT#$WT/}/${TASK}_diffusion_policy_* (COHUB_DATA_REBUILD=1 to rebuild)"
    else
      for split in train val; do
        # The DP environment does not install cohub: the converter imports it from source/.
        PYTHONPATH="$WT/source${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$WT/scripts/export/convert_lerobot.py" --source-path "$SRC" \
            --output-path "$OUT" --format diffusion_policy --robot each \
            --split "$split" --overwrite
      done
    fi
    for r in $ROBOTS; do
      link "$OUT/${TASK}_diffusion_policy_${r}.zarr"     "$(mh_data_id "${TASK}_${r}" "$ENV_CFG").zarr"
      link "$OUT/${TASK}_diffusion_policy_${r}_val.zarr" "$(mh_data_id "${TASK}_${r}" "$ENV_CFG")_val.zarr"
    done
    ;;
  centralized)
    echo "DP's centralized (one policy, both robots) set is not a CoHuB setting; the benchmark trains one checkpoint per robot" >&2
    exit 2 ;;
esac
echo "[prepare] $TASK $MODE done"
