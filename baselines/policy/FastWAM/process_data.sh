#!/usr/bin/env bash
# Build the training data from the Hub dataset (no GPU):  bash process_data.sh
# A three-robot task as the argument, or COHUB_TASK_SET=all3, builds the three-robot set.
#
# A single task's own dataset, for a run on that task alone:
#   COHUB_DATA_IN=FastWAM bash process_data.sh <task> <centralized|decentralized>
if [ "${COHUB_DATA_IN:-}" != FastWAM ]; then   # called from outside: load the config first
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data FastWAM "$@"; exit
fi
set -euo pipefail

TASK=${1:?usage: COHUB_DATA_IN=FastWAM process_data.sh <task> <centralized|decentralized>}
MODE=${2:?usage: COHUB_DATA_IN=FastWAM process_data.sh <task> <centralized|decentralized>}

WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; export COHUB_WT=$WT
# shellcheck source=../../common.sh
source "$WT/baselines/common.sh"
TASK=$(mh_task_word "$TASK")
PY=${COHUB_CONVERT_PY:-$WT/.venv/bin/python}
FASTWAM_PY=$(cohub_policy_python FastWAM)
OUT=$WT/baselines/data/fastwam
FW_DATA=$WT/baselines/policy/FastWAM/data

SRC=$WT/datasets/$TASK/lerobot
[ -f "$SRC/meta/info.json" ] || SRC=$WT/datasets/$TASK
[ -f "$SRC/meta/info.json" ] || {
    echo "no LeRobot export under $WT/datasets/$TASK" >&2
    exit 1
}
[ -x "$PY" ] || { echo "no converter python at $PY -- set COHUB_CONVERT_PY" >&2; exit 1; }
[ -x "$FASTWAM_PY" ] || { echo "no fastwam python at $FASTWAM_PY -- run policy/FastWAM/install.sh first" >&2; exit 1; }
mkdir -p "$OUT" "$FW_DATA"

link() {  # <src dir> <dataset_id>
    ln -sfn "$1" "$FW_DATA/$2"
    echo "  linked $2 -> ${1#$WT/}"
}

case "$MODE" in
  decentralized)
    for split in train val; do
      "$PY" "$WT/scripts/export/convert_lerobot.py" --source-path "$SRC" \
          --output-path "$OUT" --format fastwam --robot both \
          --split "$split" --overwrite
    done
    for r in robot_a robot_b; do
      root="$OUT/${TASK}_fastwam_${r}"
      "$FASTWAM_PY" "$WT/baselines/policy/FastWAM/dataset_stats.py" \
          --task cohub_uncond_1cam_192_1e-4 \
          --train-dir "$root/lerobot" --out "$root/dataset_stats.json" \
          --action-dim 35 --state-dim 43
      link "$root" "$(mh_data_id "${TASK}_${r}" "$(mh_env_cfg_type decentralized)")"
    done
    ;;
  centralized)
    for split in train val; do
      "$PY" "$WT/scripts/export/convert_lerobot.py" --source-path "$SRC" \
          --output-path "$OUT" --format fastwam --robot duo \
          --cameras ego_a,ego_b --split "$split" --overwrite
    done
    root="$OUT/${TASK}_fastwam"
    "$FASTWAM_PY" "$WT/baselines/policy/FastWAM/dataset_stats.py" \
        --task cohub_uncond_2cam_384_1e-4 \
        --train-dir "$root/lerobot" --out "$root/dataset_stats.json" \
        --action-dim 70 --state-dim 86
    link "$root" "$(mh_data_id "$TASK" "$(mh_env_cfg_type centralized)")"
    ;;
  *) echo "mode must be decentralized or centralized; got $MODE" >&2; exit 2 ;;
esac
echo "[prepare-fastwam] $TASK $MODE done"
