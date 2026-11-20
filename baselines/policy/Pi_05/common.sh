# Shared setup for the pi0.5 train/eval hooks and the data builder. Sourced,
# not executed.

# WT is set by the caller. This is the standalone fallback.
WT=${WT:-${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
POLICY_DIR="$WT/baselines/policy/Pi_05"
if ! declare -f mh_policy_python >/dev/null 2>&1; then
  # shellcheck source=common.sh
  source "$WT/baselines/common.sh"
fi

# Where the LeRobot v3.0 views live, one per task, built by
# baselines/policy/Pi_05/convert_v30.py. Fast local storage is best.
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${HF_HOME:-$HOME/.cache/huggingface}/lerobot}

# The v2.1 exports the views point at.
export COHUB_DATASETS=${COHUB_DATASETS:-$WT/datasets}

# Cache for pi05_base and the PaliGemma tokenizer.
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

# The flattened all-task dataset the shared decentralized policy trains on,
# built by baselines/utils/prepare_data.sh.
export COHUB_MULTITASK_DATASET=${COHUB_MULTITASK_DATASET:-$WT/baselines/data/multitask/lerobot}

mh_pi05_source() {    # <task|multitask> -> the v2.1 export a view is built from
  case "$1" in
    multitask)  echo "$COHUB_MULTITASK_DATASET" ;;
    multitask3) echo "$WT/baselines/data/multitask3/lerobot" ;;   # the three-robot set
    *)          echo "$COHUB_DATASETS/$1/lerobot" ;;
  esac
}

mh_pi05_video_keys() {  # <task|multitask> -> the camera columns the view keeps
  # The flattened datasets hold one camera per row under a plain name.
  case "$1" in multitask|multitask3) echo "observation.images.ego" ;; esac
}

require_dataset() {   # <task|multitask>
  local view="$HF_LEROBOT_HOME/cohub-$1" src keys
  src=$(mh_pi05_source "$1")
  keys=$(mh_pi05_video_keys "$1")
  if [ ! -f "$view/meta/info.json" ]; then
    echo "[pi05] building the v3.0 view for $1" >&2
    ( cd "$POLICY_DIR/openpi" && uv run python "$WT/baselines/policy/Pi_05/convert_v30.py" \
        --src "$src" --dst "$view" --no-check ${keys:+--video-keys "$keys"} ) || return 1
  fi
}
