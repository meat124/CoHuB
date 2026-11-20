#!/bin/bash
# On CoHuB:  bash train.sh <task> [--tag T]
# Runs baselines/policy/ACT/train_hook.sh (settings: baselines/configs/ACT.yaml). With `cohub` first it runs the six-argument chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train ACT "$@"; exit
fi

bench_name=${1}
ckpt_name=${2} # run name
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/utils"
action_dim=$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" action_dim); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
state_dim=${ACT_STATE_DIM:-$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" state_dim)}; echo -e "\033[33m[INFO] State dim: ${state_dim}\033[0m"
export ACT_ACTION_DIM=${action_dim}
export ACT_STATE_DIM=${state_dim}
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
# CKPT_TAG separates runs over the same data.
# COHUB_RUN_SUFFIX stands in for everything after the checkpoint name (common.sh: mh_run_name).
ckpt_dir="${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${COHUB_RUN_SUFFIX:-${env_cfg_type}-${action_type}-${seed}${CKPT_TAG:+-${CKPT_TAG}}}"
# ACT_DATA_SETTING names another TASK_CONFIGS.json entry to read the episodes from.
ckpt_setting="${ACT_DATA_SETTING:-${ckpt_setting}}"

# Defaults: batch 256 for 600 epochs. An epoch is one pass over every frame.
BATCH_SIZE=${ACT_BATCH_SIZE:-256}
NUM_EPOCHS=${ACT_NUM_EPOCHS:-600}
# One checkpoint per quarter of the run.
SAVE_FREQ=${ACT_SAVE_FREQ:-150}
LR=${ACT_LR:-1e-4}

python3 imitate_episodes.py \
    --bench_name ${bench_name} \
    --task_name ${ckpt_name} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir "${ckpt_dir}" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size ${BATCH_SIZE} \
    --dim_feedforward 3200 \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --save_freq ${SAVE_FREQ} \
    --seed ${seed} ${ACT_GRAD_ACCUM:+--grad_accum ${ACT_GRAD_ACCUM}}
