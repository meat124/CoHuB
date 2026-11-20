#!/usr/bin/env bash
# LatentToM: train one task, both arms in one run. Knobs: baselines/configs/LatentToM.yaml.
#
#   baselines/policy/LatentToM/train_hook.sh handover                          one run for the task
#   baselines/policy/LatentToM/train_hook.sh data handover                     convert that task's LeRobot export to its replay buffer
#   LATENTTOM_EPOCHS=1200 baselines/policy/LatentToM/train_hook.sh handover    a config key set in the shell wins over the file
#   LATENTTOM_EXTRA=training.debug=true baselines/policy/LatentToM/train_hook.sh handover   further hydra overrides
#   TRAIN_DRY_RUN=1 baselines/policy/LatentToM/train_hook.sh handover          resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot LatentToM
  cohub_train_main LatentToM centralized "$@"; exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
cohub_two_robot_only LatentToM "$TASK" || return 2
# `centralized` names the run directory: both arms train in one process and
# are saved in one directory named for the task.
MODES_SUPPORTED=centralized
ACTION_TYPE=joint
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_LatentToM}

# train.sh resumes from the last checkpoint.

# Only the LeRobot export is checked. train.sh stops when the replay buffer is missing.
require_data() {
  [ -f "$WT/datasets/$TASK_WORD/lerobot/meta/info.json" ] && return 0
  [ -f "$WT/datasets/$TASK_WORD/meta/info.json" ] && return 0
  echo "no LeRobot export at datasets/$TASK_WORD/lerobot"
  echo "prepare it with:  bash baselines/utils/prepare_data.sh fetch   (the Hub snapshot, linked into datasets/<task>/lerobot)"
  return 1
}

train_env() {
  export LATENTTOM_PYTHON; LATENTTOM_PYTHON=$(cohub_policy_python LatentToM)
  export LATENTTOM_RUN_TAG=${CKPT_TAG:-}
}

# The run directory is cohub-<task>-<task>-joint-<seed>: LatentToM's
# env_cfg_type is the scene. serve_LatentToM (scripts/eval/policies.sh) uses the same name.
run_env_cfg_type() { echo "$TASK_WORD"; }
_latenttom_args() {
  # Name the wandb run and give it the run directory's name as id.
  echo training.num_epochs="${LATENTTOM_EPOCHS:-1000}" \
       training.checkpoint_every="${LATENTTOM_CKPT_EVERY:-10}" \
       n_obs_steps="${LATENTTOM_OBS_STEPS:-1}" \
       checkpoint.topk.k=0 \
       checkpoint.snapshot_every="${LATENTTOM_SNAPSHOT_EVERY:-250}" \
       training.state_noise="${LATENTTOM_STATE_NOISE:-0.02}" \
       dataloader.batch_size="${LATENTTOM_BATCH_SIZE:-32}" \
       val_dataloader.batch_size="${LATENTTOM_BATCH_SIZE:-32}" \
       logging.project="$WANDB_PROJECT" \
       ${WANDB_ENTITY:+logging.entity="$WANDB_ENTITY"} \
       logging.id="$RUN_NAME" \
       logging.name="$(cohub_wandb_name "$TASK_WORD" LatentToM train)" \
       ${LATENTTOM_EXTRA:-}
}
train_cmd() {
  cd "$POLICY_DIR"
  exec bash train.sh cohub "$TASK_WORD" "$TASK_WORD" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}" $(_latenttom_args)
}
train_cmd_preview() {
  printf 'cd %q && bash train.sh cohub %q %q %q %q %q %s\n' \
    "$POLICY_DIR" "$TASK_WORD" "$TASK_WORD" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}" "$(_latenttom_args)"
}
}
