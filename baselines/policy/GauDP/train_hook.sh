#!/usr/bin/env bash
# GauDP: one task's features and policy, or the encoder every task shares. Knobs: baselines/configs/GauDP.yaml.
#
#   baselines/policy/GauDP/train_hook.sh handover           the chain: features -> policy (GAUDP_SHARED_GAUSSIAN=0: gaussian first)
#   baselines/policy/GauDP/train_hook.sh handover policy    one stage (gaussian | features | policy | all)
#   baselines/policy/GauDP/train_hook.sh shared             one encoder over GAUDP_SHARED_TASKS, for every task to reuse
#   baselines/policy/GauDP/train_hook.sh shared all3        the three-robot set's encoder (GAUDP_SHARED3_*)
#   baselines/policy/GauDP/train_hook.sh movehouse policy   a stage of a three-robot task
#   baselines/policy/GauDP/train_hook.sh data handover      convert that task's LeRobot export to GauDP's hdf5
#   GAUDP_POLICY_EPOCHS=200 baselines/policy/GauDP/train_hook.sh handover policy   a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/GauDP/train_hook.sh handover    resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot GauDP
  run_stage() {                   # <gaussian|features|policy|shared_gaussian|all> <task>
    GAUDP_STAGE=$1 cohub_train GauDP "$2" centralized - "${SEED:-0}"
  }
  case "${1:-}" in
    ""|-h|--help|data) cohub_train_main GauDP centralized "$@" ;;
    shared|shared_gaussian)
      # The task picks the set: `all3` or a three-robot task is the three-robot
      # encoder, anything else the two-robot one.
      shared_task=${2:-handover}
      if [ "$shared_task" = all3 ]; then shared_task=${GAUDP_SHARED3_TASKS:-$MH_ALL3_TASKS}; shared_task=${shared_task%% *}; fi
      run_stage shared_gaussian "$shared_task" ;;
    *)
      task=$(mh_task_word "$1") || exit 2
      case "${2:-all}" in
        all|gaussian|features|policy|shared_gaussian) run_stage "${2:-all}" "$task" ;;
        *) echo "unknown stage '$2': gaussian, features, policy, shared_gaussian or all" >&2; exit 2 ;;
      esac ;;
  esac
  exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# GauDP is centralized only: one policy reads every robot's state and ego
# view and predicts all of their actions.
MODES_SUPPORTED=centralized
ACTION_TYPE=joint
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_GauDP}

# One stage, or `all` for the three in order.
GAUDP_STAGE=${GAUDP_STAGE:-all}
case "$GAUDP_STAGE" in
  gaussian|features|policy|shared_gaussian|all) ;;
  *) echo "[train-hook] GAUDP_STAGE=$GAUDP_STAGE: expected gaussian, features, policy, shared_gaussian or all" >&2
     return 2 ;;
esac

# Three-robot tasks use their own encoder, task list and batch (GAUDP_*3_*).
GAUDP_THREE_ROBOTS=0
if mh_robots "$(mh_scene "$TASK")" 2>/dev/null | grep -qx robot_c; then
  GAUDP_THREE_ROBOTS=1
  GAUDP_SHARED_TASKS=${GAUDP_SHARED3_TASKS:-$MH_ALL3_TASKS}
  GAUDP_SHARED_EPOCHS=${GAUDP_SHARED3_EPOCHS:-8}
  GAUDP_GAUSSIAN_BATCH_SIZE=${GAUDP_GAUSSIAN3_BATCH_SIZE:-4}
  GAUDP_GAUSSIAN_GRAD_ACCUM=${GAUDP_GAUSSIAN3_GRAD_ACCUM:-2}
  GAUDP_SHARED_VARIANT=${GAUDP_SHARED3_VARIANT:-}
  GAUDP_SHARED_EARLY_STOP=${GAUDP_SHARED3_EARLY_STOP:-}
  GAUDP_SHARED_LR=${GAUDP_SHARED3_LR:-}
  GAUDP_SHARED_BACKBONE_LR_MULT=${GAUDP_SHARED3_BACKBONE_LR_MULT:-}
fi


# Only the LeRobot export is checked. A stage stops when its converted hdf5 is missing.
require_data() {
  local tasks=("$TASK_WORD") missing=() t src
  [ "$GAUDP_STAGE" = shared_gaussian ] && read -r -a tasks <<<"$GAUDP_SHARED_TASKS"
  for t in "${tasks[@]}"; do
    src=$(mh_task_word "$t") || return 1
    [ -f "$WT/datasets/$src/lerobot/meta/info.json" ] || [ -f "$WT/datasets/$src/meta/info.json" ] || missing+=("$src")
  done
  [ ${#missing[@]} -eq 0 ] && return 0
  echo "no LeRobot export for: ${missing[*]}"
  local task_set=; [ "$GAUDP_THREE_ROBOTS" = 1 ] && task_set="COHUB_TASK_SET=all3 "
  echo "prepare it with:  ${task_set}bash baselines/utils/prepare_data.sh fetch   (the Hub snapshot, linked into datasets/<task>/lerobot)"
  return 1
}

train_env() {
  export GAUDP_PYTHON; GAUDP_PYTHON=$(cohub_policy_python GauDP)
  # The run directory, named as mh_run_name does, so serve_GauDP (scripts/eval/policies.sh) finds it.
  export GAUDP_BENCH=cohub GAUDP_CKPT=$CKPT_NAME GAUDP_ENV_CFG=$ENV_CFG_TYPE GAUDP_SEED=$SEED GAUDP_TAG=${CKPT_TAG:-}
  export GAUDP_FINETUNE_MODE=${GAUDP_FINETUNE_MODE:-full}
  export GAUDP_SHARED_GAUSSIAN=${GAUDP_SHARED_GAUSSIAN:-1}
  export GAUDP_SHARED_TASKS=${GAUDP_SHARED_TASKS:-$MH_ALL_TASKS}
  export GAUDP_SHARED_VARIANT=${GAUDP_SHARED_VARIANT:-}
}

_gaudp_encoder_args() {         # all but the epoch budget
  echo --batch-size "${GAUDP_GAUSSIAN_BATCH_SIZE:-8}" \
       --gradient-accumulation-steps "${GAUDP_GAUSSIAN_GRAD_ACCUM:-1}" \
       --num-workers "${GAUDP_GAUSSIAN_NUM_WORKERS:-5}" \
       --early-stop-patience "${GAUDP_GAUSSIAN_EARLY_STOP:-3}" \
       --finetune-mode "$GAUDP_FINETUNE_MODE" \
       ${GAUDP_GAUSSIAN_VAL_EVERY:+--val-every-steps "$GAUDP_GAUSSIAN_VAL_EVERY"} \
       ${GAUDP_GAUSSIAN_VAL_SUBSET:+--val-subset-frames "$GAUDP_GAUSSIAN_VAL_SUBSET"}
}
_gaudp_gaussian_args() { echo --epochs "${GAUDP_GAUSSIAN_EPOCHS:-20}" $(_gaudp_encoder_args); }
_gaudp_shared_args()   {        # a later --early-stop-patience wins in argparse
  echo --epochs "${GAUDP_SHARED_EPOCHS:-10}" ${GAUDP_STOP_AT_EPOCH:+--stop-at-epoch "$GAUDP_STOP_AT_EPOCH"} $(_gaudp_encoder_args) \
       ${GAUDP_SHARED_EARLY_STOP:+--early-stop-patience "$GAUDP_SHARED_EARLY_STOP"} \
       ${GAUDP_SHARED_LR:+--lr "$GAUDP_SHARED_LR"} \
       ${GAUDP_SHARED_BACKBONE_LR_MULT:+--backbone-lr-multiplier "$GAUDP_SHARED_BACKBONE_LR_MULT"}
}
_gaudp_policy_args() {
  echo --epochs "${GAUDP_POLICY_EPOCHS:-150}" \
       --batch-size "${GAUDP_POLICY_BATCH_SIZE:-128}" \
       --num-workers "${GAUDP_POLICY_NUM_WORKERS:-6}" \
       --grad-accum "${GAUDP_POLICY_GRAD_ACCUM:-1}" \
       --lr "${GAUDP_POLICY_LR:-1e-4}" \
       --warmup "${GAUDP_POLICY_WARMUP:-500}" \
       --state-noise "${GAUDP_POLICY_STATE_NOISE:-0.02}" \
       --normalizer-range-eps "${GAUDP_POLICY_RANGE_EPS:-1e-4}" \
       --obs-steps "${GAUDP_POLICY_OBS_STEPS:-1}" \
       --horizon "${GAUDP_POLICY_HORIZON:-40}" \
       --action-steps "${GAUDP_POLICY_ACTION_STEPS:-20}"
}

_gaudp_gpu() {               # the first card of CUDA_VISIBLE_DEVICES: the stages run on one
  local gpu=${CUDA_VISIBLE_DEVICES:-0}; gpu=${gpu%%,*}
  [[ "$gpu" =~ ^[0-9]+$ ]] && echo "$gpu" || echo 0
}
train_cmd() {
  cd "$POLICY_DIR"
  local tag=${CKPT_TAG:+-$CKPT_TAG} gpu; gpu=$(_gaudp_gpu)
  if [ "$GAUDP_STAGE" = shared_gaussian ]; then
    # The task argument is ignored: one encoder over GAUDP_SHARED_TASKS.
    exec bash train_gaussian_shared.sh "$SEED" "$gpu" $GAUDP_SHARED_TASKS -- $(_gaudp_shared_args)
  fi
  # The chain fine-tunes an encoder only for a task of its own (GAUDP_SHARED_GAUSSIAN=0).
  if [ "$GAUDP_STAGE" = gaussian ] || { [ "$GAUDP_STAGE" = all ] && [ "${GAUDP_SHARED_GAUSSIAN:-1}" = 0 ]; }; then
    echo "[gaudp] === Gaussian encoder of ${TASK_WORD} ==="
    # An existing encoder is kept. GAUDP_FORCE_GAUSSIAN=1 finetunes anyway.
    bash train_gaussian.sh "$TASK_WORD" "$SEED" "$gpu" $(_gaudp_gaussian_args) \
      --wandb-run-name "${TASK_WORD}-gaussian-seed${SEED}${tag}" \
      --wandb-tags "gaussian,${TASK_WORD},seed-${SEED}"
  fi
  if [ "$GAUDP_STAGE" = all ] || [ "$GAUDP_STAGE" = features ]; then
    echo "[gaudp] === Gaussian features ==="
    # Resumable.
    bash extract_gaussian_features.sh "$TASK_WORD" "$SEED" "$gpu"
  fi
  if [ "$GAUDP_STAGE" = all ] || [ "$GAUDP_STAGE" = policy ]; then
    echo "[gaudp] === policy ==="
    # The wandb run id is the run directory's name, so an evaluation is logged into it.
    WANDB_RUN_ID=$RUN_NAME WANDB_RESUME=allow bash train.sh "$TASK_WORD" "$SEED" "$gpu" $(_gaudp_policy_args) \
      --wandb-run-name "$(cohub_wandb_name "$CKPT_NAME" GauDP train)" \
      --wandb-tags "policy,${TASK_WORD},seed-${SEED}"
  fi
}
train_cmd_preview() {
  printf 'cd %q\n' "$POLICY_DIR"
  case "$GAUDP_STAGE" in
    shared_gaussian) printf 'bash train_gaussian_shared.sh %q %q %s -- %s\n' "$SEED" "$(_gaudp_gpu)" "$GAUDP_SHARED_TASKS" "$(_gaudp_shared_args)" ;;
    *)
      if [ "$GAUDP_STAGE" = gaussian ] || { [ "$GAUDP_STAGE" = all ] && [ "${GAUDP_SHARED_GAUSSIAN:-1}" = 0 ]; }; then
        printf 'bash train_gaussian.sh %q %q %q %s\n' "$TASK_WORD" "$SEED" "$(_gaudp_gpu)" "$(_gaudp_gaussian_args)"
      fi
      case "$GAUDP_STAGE" in all|features) printf 'bash extract_gaussian_features.sh %q %q %q\n' "$TASK_WORD" "$SEED" "$(_gaudp_gpu)" ;; esac
      case "$GAUDP_STAGE" in all|policy)   printf 'bash train.sh %q %q %q %s\n' "$TASK_WORD" "$SEED" "$(_gaudp_gpu)" "$(_gaudp_policy_args)" ;; esac ;;
  esac
}
}
