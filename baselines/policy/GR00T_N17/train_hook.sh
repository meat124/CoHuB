#!/usr/bin/env bash
# GR00T N1.7: train the shared multitask policy. Knobs: baselines/configs/GR00T_N17.yaml.
#
#   baselines/policy/GR00T_N17/train_hook.sh                       train the shared multitask policy
#   baselines/policy/GR00T_N17/train_hook.sh data                  build the flattened all-task dataset it reads (CPU only)
#   MAX_STEPS=20000 baselines/policy/GR00T_N17/train_hook.sh       a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/GR00T_N17/train_hook.sh       resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot GR00T_N17
  # COHUB_TASK_SET=all3 trains (and builds) the three-robot set, checkpoint `multitask3`.
  cohub_train_main GR00T_N17 multitask "$@"; exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# Decentralized fits a 24 GB card. Centralized, TUNE_VISUAL and TUNE_LLM want
# 48 GB. An interrupted run resumes from its newest checkpoint.
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_GR00T}

train_env() {
  # 40: the released N1.7-3B checkpoint is a 40-step model.
  export COHUB_ACTION_HORIZON=${COHUB_ACTION_HORIZON:-40}
  case "$MODE" in
    centralized)
      # The room camera is off by default.
      export COHUB_SCENE_CAMERA=${COHUB_SCENE_CAMERA:-0} ;;
    decentralized)
      # The robot unitree_g1x2_decentralized_config.py registers. The shared
      # multitask policy registers `shared`.
      case "$CKPT_NAME" in
        multitask|multitask3) export COHUB_ROBOT=shared ;;
        *) export COHUB_ROBOT="$ROBOT" ;;
      esac ;;
  esac

  export GR00T_LEROBOT_HOME="$WT/baselines/data/gr00t"
  # Local snapshots. `baselines/policy/GR00T_N17/install.sh` fetches them.
  for snap in GR00T_BASE_MODEL GR00T_COSMOS_MODEL; do
    [ -e "${!snap}" ] || { echo "[GR00T_N17 hook] $snap=${!snap} is missing -- baselines/policy/GR00T_N17/install.sh fetches it" >&2; return 1; }
  done
  export GR00T_BASE_MODEL GR00T_COSMOS_MODEL

  # The benchmark's budget: global batch 32, 40k optimizer steps.
  export MAX_STEPS=${MAX_STEPS:-40000}
  # finetune.sh keeps 5 checkpoints (--save_total_limit 5). At 10000 the
  # evaluated 10k/20k/30k/40k all survive.
  export SAVE_STEPS=${SAVE_STEPS:-10000}
  export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
  export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
  export LORA_RANK=${LORA_RANK:-32}
  export LORA_ALPHA=${LORA_ALPHA:-16}
  export LORA_DROPOUT=${LORA_DROPOUT:-0.1}

  # The validation split comes from the dataset's meta/info.json. eval/loss
  # is computed on a fixed subset.
  export EVAL_STRATEGY=${EVAL_STRATEGY:-steps}
  export EVAL_STEPS=${EVAL_STEPS:-1000}
  export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
  export EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-512}

  export USE_WANDB=${USE_WANDB:-1}
  # The wandb run id is the run directory's name, so a restart continues the same run.
  export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
  export WANDB_NAME=${WANDB_NAME:-$(cohub_wandb_name "$CKPT_NAME" GR00T_N17 train)}
  export WANDB_RESUME=${WANDB_RESUME:-allow}

  # Tags for filtering in wandb.
  export TUNE_LLM=${TUNE_LLM:-0}
  export TUNE_VISUAL=${TUNE_VISUAL:-0}
  local tags="${TASK_WORD},${MODE}"
  [ "$MODE" = decentralized ] && [ "$ROBOT" != - ] && tags="${tags},${ROBOT}"
  case "$CKPT_NAME" in multitask|multitask3) tags="${tags},${CKPT_NAME}" ;; esac
  if [ "${LORA_RANK}" = 0 ]; then tags="${tags},full"; else tags="${tags},lora-r${LORA_RANK}"; fi
  [ "${LORA_FULL_MODEL:-0}" = 1 ] && tags="${tags},lora-backbone"
  [ -n "${COHUB_LORA_TARGETS:-}" ] && tags="${tags},lora-${COHUB_LORA_TARGETS}"
  if [ "${TUNE_LLM}" = 1 ]; then tags="${tags},llm-tuned"; else tags="${tags},llm-frozen"; fi
  if [ "${TUNE_VISUAL}" = 1 ]; then tags="${tags},vision-tuned"; else tags="${tags},vision-frozen"; fi
  [ "${COHUB_SCENE_CAMERA:-0}" != 0 ] && tags="${tags},scene"
  export WANDB_TAGS=${WANDB_TAGS:-$tags}
}

require_data() {
  command -v ffprobe >/dev/null 2>&1 || { echo "no ffprobe on PATH -- GR00T's loader reads the videos' lengths with it (install ffmpeg)"; return 1; }
  local dataset="$WT/baselines/data/gr00t/$(mh_data_id "$CKPT_NAME" "$ENV_CFG_TYPE")"
  if [ -d "$dataset" ] && [ -f "$dataset/meta/stats.json" ]; then return 0; fi
  echo "missing processed GR00T dataset (or its meta/stats.json): $dataset"
  if [ "$CKPT_NAME" = multitask ] || [ "$CKPT_NAME" = multitask3 ]; then
    echo "build it with:  bash baselines/utils/prepare_data.sh gr00t"
  else
    echo "build and link it with baselines/policy/GR00T_N17/train_hook.sh data"
  fi
  return 1
}

# Uses the runner's default train_cmd.
}
