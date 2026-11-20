#!/usr/bin/env bash
# ACT: train one task's per-robot policies. Knobs: baselines/configs/ACT.yaml.
#
#   baselines/policy/ACT/train_hook.sh cocarry                        one run per robot the task has
#   baselines/policy/ACT/train_hook.sh data cocarry                   build that task's dataset first
#   ACT_NUM_EPOCHS=300 baselines/policy/ACT/train_hook.sh cocarry     a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/ACT/train_hook.sh cocarry        resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot ACT
  case "${1:-}" in
    data)  # fetch, convert, register in policy/ACT/TASK_CONFIGS.json
           task=$(mh_task_word "${2:?task}") || exit 2
           sim_py=${COHUB_SIM_PY:-$WT/.venv/bin/python}
           cohub_fetch_task "$task"
           robots=$(mh_robots "$(mh_scene "$task")"); built=1
           for robot in $robots; do
             [ -f "$WT/baselines/data/${task}_act_${robot}/.complete" ] && [ -f "$WT/baselines/data/${task}_act_${robot}_val/.complete" ] || built=0
           done
           if [ "$built" = 1 ] && [ "${COHUB_DATA_REBUILD:-0}" != 1 ]; then
             cohub_say "ACT $task: reusing baselines/data/${task}_act_* (COHUB_DATA_REBUILD=1 to rebuild)"
           else
             for split in train val; do
               cohub_step "$sim_py" "$WT/scripts/export/convert_lerobot.py" --source-path "$WT/datasets/$task/lerobot" \
                 --output-path "$WT/baselines/data" --format act --robot each --split "$split" --overwrite
             done
             for robot in $robots; do
               cohub_step touch "$WT/baselines/data/${task}_act_${robot}/.complete" "$WT/baselines/data/${task}_act_${robot}_val/.complete"
             done
           fi
           for robot in $robots; do
             (cd "$WT/baselines/policy/ACT" && cohub_step "$sim_py" register_task_config.py \
                --bench_name cohub --ckpt_name "${task}_${robot}" --env_cfg_type "$(mh_env_cfg_type decentralized)" \
                --action_type joint --dataset_dir "$WT/baselines/data/${task}_act_${robot}" \
                --val_dataset_dir "$WT/baselines/data/${task}_act_${robot}_val")
           done ;;
    *)     cohub_train_main ACT per_robot "$@" ;;
  esac
  exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# imitate_episodes.py has no resume: an interrupted run starts over.
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_ACT}

# ACT takes no language input: one checkpoint per (task, robot).
DECENTRALIZED_STYLE=per_robot
MODES_SUPPORTED=decentralized

train_env() {
  export WANDB_PROJECT
  # The wandb run id is the checkpoint directory's name, so eval uploads
  # land in the training run.
  export WANDB_RUN_ID=$RUN_NAME WANDB_RESUME=allow
  export WANDB_NAME=${WANDB_NAME:-$(cohub_wandb_name "$CKPT_NAME" ACT train)}
}

require_data() {
  local ok=1
  local py; py=$(cohub_policy_python ACT)
  local data="$WT/baselines/data/${TASK_WORD}_act_${ROBOT}"   # what convert_lerobot.py --format act --robot each writes
  if [ ! -x "$py" ]; then
    echo "no ACT interpreter at $py -- create the env first (policy/ACT/install.sh), or set COHUB_ACT_PY"
    ok=0
  fi
  if [ ! -d "$data" ]; then
    echo "missing ACT episodes at $data"
    echo "build them with:  baselines/policy/ACT/train_hook.sh data $TASK_WORD"
    ok=0
  fi
  if [ ! -f "$POLICIES/policy/ACT/TASK_CONFIGS.json" ]; then
    echo "missing policy/ACT/TASK_CONFIGS.json -- baselines/policy/ACT/train_hook.sh data $TASK_WORD registers the dataset"
    ok=0
  fi
  [ "$ok" = 1 ]
}

# train.sh calls python3 off PATH, so the ACT env's bin goes first.
train_cmd() {
  local py; py=$(cohub_policy_python ACT)
  cd "$POLICY_DIR"
  PATH="$(dirname "$py"):$PATH" exec bash train.sh cohub "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}"
}
train_cmd_preview() {
  local py; py=$(cohub_policy_python ACT)
  printf 'cd %q && PATH=%q:$PATH bash train.sh cohub %q %q %q %q %q\n' \
    "$POLICY_DIR" "$(dirname "$py")" "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}"
}
}
