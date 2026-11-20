#!/usr/bin/env bash
# pi0.5: train the shared multitask policy. Knobs: baselines/configs/Pi_05.yaml.
#
#   baselines/policy/Pi_05/train_hook.sh                               train the shared multitask policy
#   baselines/policy/Pi_05/train_hook.sh data                          build the flattened all-task dataset it reads (CPU only)
#   PI05_NUM_TRAIN_STEPS=20000 baselines/policy/Pi_05/train_hook.sh    a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/Pi_05/train_hook.sh               resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot Pi_05
  # COHUB_TASK_SET=all3 trains (and builds) the three-robot set, checkpoint `multitask3`.
  cohub_train_main Pi_05 multitask "$@"; exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# policy/Pi_05/train.sh picks the TrainConfig from ckpt_name and env_cfg_type,
# computes norm stats when missing and resumes from an existing step directory.
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_Pi05}

train_env() {
  export WANDB_NAME=${WANDB_NAME:-$(cohub_wandb_name "$CKPT_NAME" Pi_05 train)}   # openpi's display name; its id is in wandb_id.txt
  # Shared with serve_Pi_05 (scripts/eval/policies.sh).
  # shellcheck source=common.sh
  source "$WT/baselines/policy/Pi_05/common.sh"
}

# The export this run reads: the task's, or the flattened one for the shared policy.
pi05_dataset_name() {
  case "$CKPT_NAME" in multitask|multitask3) echo "$CKPT_NAME" ;; *) echo "$TASK_WORD" ;; esac
}

require_data() {
  # Only the v2.1 export is checked. prepare_data builds the v3.0 view from it.
  local name src
  name=$(pi05_dataset_name)
  src=$(mh_pi05_source "$name")
  [ -f "$src/meta/info.json" ] && return 0
  echo "missing LeRobot v2.1 export at $src"
  if [ "$name" = multitask ] || [ "$name" = multitask3 ]; then
    echo "build it with:  bash baselines/utils/prepare_data.sh pi05"
  else
    echo "materialise datasets/$TASK_WORD first (scripts/export/export_lerobot.py)"
  fi
  return 1
}

prepare_data() {
  # Builds $HF_LEROBOT_HOME/cohub-<name> if missing.
  require_dataset "$(pi05_dataset_name)"
}
}
