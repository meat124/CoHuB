#!/usr/bin/env bash
# Psi0: train the shared multitask policy. Knobs: baselines/configs/Psi0.yaml.
#
#   baselines/policy/Psi0/train_hook.sh                       train the shared multitask policy
#   baselines/policy/Psi0/train_hook.sh data                  build the flattened all-task dataset it reads (CPU only)
#   MAX_STEPS=20000 baselines/policy/Psi0/train_hook.sh       a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/Psi0/train_hook.sh       resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot Psi0
  # COHUB_TASK_SET=all3 trains (and builds) the three-robot set, checkpoint `multitask3`.
  cohub_train_main Psi0 multitask "$@"; exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# Only the action expert is trained (--model.no-tune-vlm). Psi0 drives one
# robot from one head camera, so it has no centralized mode.
MODES_SUPPORTED=decentralized
DECENTRALIZED_STYLE=shared

WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_Psi0}

train_env() {
  # Local snapshots of the pretrained pieces. `baselines/policy/Psi0/install.sh` fetches them.
  for snap in PSI0_BASE_MODEL PSI0_ACTION_HEADER; do
    [ -e "${!snap}" ] || { echo "[Psi0 hook] $snap=${!snap} is missing -- baselines/policy/Psi0/install.sh fetches it" >&2; return 1; }
  done
  export PSI0_BASE_MODEL PSI0_ACTION_HEADER

  # The benchmark's budget: global batch 32, 40k optimizer steps.
  export MAX_STEPS=${MAX_STEPS:-40000}
  export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
  # Per device. train.sh derives the accumulation that makes the global batch.
  export PSI0_MICRO_BATCH=${PSI0_MICRO_BATCH:-8}
  # One checkpoint per evaluated step (10k/20k/30k/40k).
  export SAVE_STEPS=${SAVE_STEPS:-10000}
  # Validation loss on the held-out episodes.
  export PSI0_EVAL_STEPS=${PSI0_EVAL_STEPS:-1000}
  export PSI0_EVAL_BATCHES=${PSI0_EVAL_BATCHES:-20}

  # The wandb run id is the run directory's name, so a restart continues the same run.
  export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
  export WANDB_NAME=${WANDB_NAME:-$(cohub_wandb_name "$CKPT_NAME" Psi0 train)}
  export WANDB_RESUME=${WANDB_RESUME:-allow}
  local tags="${TASK_WORD},${MODE},${CKPT_NAME},rtc"   # multitask, or multitask3 for the three-robot set
  if [ "${PSI0_LORA_DIT_RANK:-0}" != 0 ]; then tags="${tags},dit-lora-r${PSI0_LORA_DIT_RANK}"; else tags="${tags},dit-full"; fi
  if [ "${PSI0_LORA_LLM_RANK:-0}" != 0 ]; then tags="${tags},llm-lora-r${PSI0_LORA_LLM_RANK}"; else tags="${tags},llm-frozen"; fi
  if [ "${PSI0_TUNE_VISION:-0}" = 1 ]; then tags="${tags},vision-tuned"; else tags="${tags},vision-frozen"; fi
  export WANDB_TAGS=${WANDB_TAGS:-$tags}
}

require_data() {
  local dataset="$POLICIES/policy/Psi0/data/$(mh_data_id "$CKPT_NAME" "$ENV_CFG_TYPE")"
  if [ -f "$dataset/meta/stats_psi0.json" ] && [ -d "${dataset}_val" ]; then return 0; fi
  echo "missing Psi0 dataset (or its meta/stats_psi0.json, or the held-out ${dataset##*/}_val): $dataset"
  echo "build it with:  bash baselines/utils/prepare_data.sh psi0"
  return 1
}

# Uses the runner's default train_cmd.
}
