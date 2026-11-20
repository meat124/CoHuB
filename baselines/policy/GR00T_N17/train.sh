#!/usr/bin/env bash
# Train GR00T_N17:  bash train.sh [task] [--tag T]   (a three-robot task, or COHUB_TASK_SET=all3, picks the three-robot set)
# Runs baselines/policy/GR00T_N17/train_hook.sh (settings: baselines/configs/GR00T_N17.yaml). With `cohub` as the first argument it runs the six-argument chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train GR00T_N17 "$@"; exit
fi

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
DATA_ROOT="${GR00T_LEROBOT_HOME:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set GR00T_LEROBOT_HOME to the LeRobot datasets root." >&2
  exit 1
fi

base_model="${GR00T_BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
cosmos_model="${GR00T_COSMOS_MODEL:-nvidia/Cosmos-Reason2-2B}"

# CKPT_TAG names the run. Runs over the same data share data_setting.
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
# COHUB_RUN_SUFFIX stands in for everything after the checkpoint name (common.sh: mh_run_name).
ckpt_setting="${bench_name}-${ckpt_name}-${COHUB_RUN_SUFFIX:-${env_cfg_type}-${action_type}-${seed}${CKPT_TAG:+-${CKPT_TAG}}}"
dataset_path="${GR00T_DATASET_PATH:-${DATA_ROOT}/${data_setting}}"
modality_config="${GR00T_MODALITY_CONFIG:-${POLICY_DIR}/configs/${env_cfg_type}_config.py}"
output_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

export SEED="${seed}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
# launch_finetune.py goes offline whenever GR00T_COSMOS_MODEL is set, so export
# it only when the caller set it.
if [[ -n "${GR00T_COSMOS_MODEL:-}" ]]; then
  export GR00T_COSMOS_MODEL="${cosmos_model}"
fi
export GR00T_VIDEO_BACKEND="${GR00T_VIDEO_BACKEND:-pyav}"

if [[ ! -d "${dataset_path}" ]]; then
  echo "Processed dataset not found: ${dataset_path}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${modality_config}" ]]; then
  echo "Modality config not found: ${modality_config}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ -f "${base_model}/config.json" ]]; then
  :
elif [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "GR00T base model not found locally: ${base_model}" >&2
  echo "Unset HF_HUB_OFFLINE or set GR00T_BASE_MODEL to a local directory." >&2
  exit 1
fi

if [[ -f "${cosmos_model}/config.json" ]]; then
  :
elif [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "Cosmos backbone not found locally: ${cosmos_model}" >&2
  echo "Unset HF_HUB_OFFLINE or set GR00T_COSMOS_MODEL to a local directory." >&2
  exit 1
fi

mkdir -p "${output_dir}"

# Optimiser overrides. Unset ones keep the defaults of examples/finetune.sh.
[[ -n "${GR00T_LR:-}" ]]            && export LEARNING_RATE="${GR00T_LR}"
[[ -n "${GR00T_MIN_LR:-}" ]]        && export MIN_LR="${GR00T_MIN_LR}"
[[ -n "${GR00T_LR_SCHEDULER:-}" ]]  && export LR_SCHEDULER_TYPE="${GR00T_LR_SCHEDULER}"
[[ -n "${GR00T_WARMUP_STEPS:-}" ]]  && export WARMUP_STEPS="${GR00T_WARMUP_STEPS}"
[[ -n "${GR00T_WARMUP_RATIO:-}" ]]  && export WARMUP_RATIO="${GR00T_WARMUP_RATIO}"
[[ -n "${GR00T_WEIGHT_DECAY:-}" ]]  && export WEIGHT_DECAY="${GR00T_WEIGHT_DECAY}"
[[ -n "${GR00T_ADAM_BETA1:-}" ]]    && export ADAM_BETA1="${GR00T_ADAM_BETA1}"
[[ -n "${GR00T_ADAM_BETA2:-}" ]]    && export ADAM_BETA2="${GR00T_ADAM_BETA2}"
[[ -n "${GR00T_ADAM_EPS:-}" ]]      && export ADAM_EPSILON="${GR00T_ADAM_EPS}"
[[ -n "${GR00T_MAX_GRAD_NORM:-}" ]] && export MAX_GRAD_NORM="${GR00T_MAX_GRAD_NORM}"

MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-640}"
USE_WANDB="${USE_WANDB:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

export MAX_STEPS SAVE_STEPS GLOBAL_BATCH_SIZE USE_WANDB DATALOADER_NUM_WORKERS

echo "[GR00T_N17] dataset_path=${dataset_path}"
echo "[GR00T_N17] base_model=${base_model}"
echo "[GR00T_N17] cosmos_model=${cosmos_model}"
echo "[GR00T_N17] video_backend=${GR00T_VIDEO_BACKEND}"
echo "[GR00T_N17] output_dir=${output_dir}"
echo "[GR00T_N17] num_gpus=${NUM_GPUS}"
echo "[GR00T_N17] global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "[GR00T_N17] per_gpu_batch_size=$((GLOBAL_BATCH_SIZE / NUM_GPUS))"
echo "[GR00T_N17] max_steps=${MAX_STEPS}"
echo "[GR00T_N17] save_steps=${SAVE_STEPS}"
echo "[GR00T_N17] lr=${LEARNING_RATE:-1e-4} min_lr=${MIN_LR:-none} scheduler=${LR_SCHEDULER_TYPE:-cosine} warmup_steps=${WARMUP_STEPS:-0} warmup_ratio=${WARMUP_RATIO:-0.05} wd=${WEIGHT_DECAY:-1e-5} betas=${ADAM_BETA1:-0.9}/${ADAM_BETA2:-0.999}"
echo "[GR00T_N17] lora: rank=${LORA_RANK:-0} alpha=${LORA_ALPHA:-16} dropout=${LORA_DROPOUT:-0.1} full_model=${LORA_FULL_MODEL:-0} targets=${COHUB_LORA_TARGETS:-attn} backbone_rank=${COHUB_LORA_BACKBONE_RANK:-same} backbone_alpha=${COHUB_LORA_BACKBONE_ALPHA:-same} tune_visual=${TUNE_VISUAL:-0} tune_llm=${TUNE_LLM:-0}"

cd "${GR00T_ROOT}"
source .venv/bin/activate

uv run --no-sync bash examples/finetune.sh \
  --base-model-path "${base_model}" \
  --dataset-path "${dataset_path}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${modality_config}" \
  --output-dir "${output_dir}" \
  --experiment-name "${ckpt_setting}"