#!/usr/bin/env bash
# Diffusion Policy: train one task's per-robot policies. Knobs: baselines/configs/DP.yaml.
#
#   baselines/policy/DP/train_hook.sh cocarry                       one run per robot the task has
#   baselines/policy/DP/train_hook.sh data cocarry                  build that task's dataset first
#   DP_NUM_EPOCHS=300 baselines/policy/DP/train_hook.sh cocarry     a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/DP/train_hook.sh cocarry       resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot DP
  cohub_train_main DP per_robot "$@"; exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# The trainer resumes from latest.ckpt.
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_DP}

# DP takes no language input: one checkpoint per (task, robot).
DECENTRALIZED_STYLE=per_robot
MODES_SUPPORTED=decentralized

require_data() {
  local zarr="$POLICIES/policy/DP/data/$(mh_data_id "$CKPT_NAME" "$ENV_CFG_TYPE").zarr"
  [ -d "$zarr" ] && return 0
  echo "missing $zarr"
  echo "prepare it with:  bash baselines/policy/DP/process_data.sh $TASK_WORD"
  return 1
}

# policy/DP/train.sh reads the run's identity and knobs from DP_* variables.
train_env() {
  export DP_PYTHON; DP_PYTHON=$(cohub_policy_python DP)
  export DP_NUM_EPOCHS=${DP_NUM_EPOCHS:-600} DP_CKPT_EVERY=${DP_CKPT_EVERY:-150} DP_BATCH_SIZE=${DP_BATCH_SIZE:-256} DP_LR=${DP_LR:-1e-4}
  export DP_HORIZON DP_N_ACTION_STEPS
  export DP_RUN_TAG=${DP_RUN_TAG:-$(cohub_budget DP)}   # e.g. b256-e600-h40-a20
  local role=; [ "$MODE" = decentralized ] && role=-$ROBOT
  # obs1: robot_dp.yaml's n_obs_steps=1
  export DP_RUN_NAME="dp-${MODE}-${TASK_WORD}${role}-obs1-${DP_RUN_TAG}-seed${SEED}"
  export DP_WANDB_PROJECT=$WANDB_PROJECT DP_WANDB_ENTITY=${WANDB_ENTITY:-}
  export DP_WANDB_TAGS="[dp,${MODE},${TASK_WORD}${role:+,$ROBOT},obs1,${DP_RUN_TAG},seed${SEED}]"
  export DP_WANDB_NAME=${DP_WANDB_NAME:-$(cohub_wandb_name "${TASK_WORD}${role}" DP train)}   # the display name; the id stays DP_RUN_NAME
}
train_cmd_preview() {
  cd "$POLICY_DIR" && DP_DRY_RUN=1 bash train.sh cohub "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}"
}
}
