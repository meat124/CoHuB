#!/bin/bash
# On CoHuB:  bash train.sh [--tag T]   (COHUB_TASK_SET=all3, or a three-robot task as the argument, picks the three-robot set)
# Runs baselines/policy/FastWAM/train_hook.sh (settings: baselines/configs/FastWAM.yaml). With `cohub` as the first argument it runs the chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train FastWAM "$@"; exit
fi
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi
train_seed=${seed}
if [[ "${train_seed}" -le 0 ]]; then
    train_seed=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
UTILS_DIR="${ROOT_DIR}/utils"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_DIR}/checkpoints}"
# expandable_segments on a single GPU only; it is unstable with multi-GPU runs.
if [[ "${num_gpus}" -le 1 ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
export PYTHONPATH="${ROOT_DIR}/..:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}"
# Per-host Triton cache: a helper compiled on one node may not load on another.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${USER:-$(id -un)}-$(hostname -s)}"
mkdir -p "${TRITON_CACHE_DIR}"

action_dim=$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" action_dim)
# FASTWAM_DATASET_ID overrides the dataset name (default: data_key).
data_key="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

# CoHuB differs from the robotwin defaults:
#   - its own task yaml per mode (2 ego views stacked / 1 ego view);
#   - state is the raw joint-angle vector (86/43), not the action vector;
#   - a separate lerobot_val dataset instead of a random validation split;
#   - wandb on, with the run id set to the checkpoint run-dir name.
if [[ "${bench_name}" == "cohub" ]]; then
    state_dim=$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" state_dim)
    case "${env_cfg_type}" in
        unitree_g1x2_centralized)   task_name="cohub_uncond_2cam_384_1e-4" ;;
        unitree_g1x2_decentralized) task_name="cohub_uncond_1cam_192_1e-4" ;;
        *) echo "[ERROR] unknown cohub env_cfg_type: ${env_cfg_type}" >&2; exit 2 ;;
    esac
else
    state_dim="${action_dim}"
    task_name="robotwin_uncond_3cam_384_1e-4"
fi
dataset_id="${FASTWAM_DATASET_ID:-${data_key}}"
converted_root="${POLICY_DIR}/data/${dataset_id}"
dataset_dir="${converted_root}/lerobot"
stats_path="${converted_root}/dataset_stats.json"
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/XPolicyLab/${dataset_id}"
ckpt_setting="${FASTWAM_CKPT_SETTING:-${data_key}-${seed}}"
action_dit="${FASTWAM_DIR}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
batch_size="${FASTWAM_BATCH_SIZE:-8}"
gradient_accumulation_steps="${FASTWAM_GRADIENT_ACCUMULATION_STEPS:-1}"
num_workers="${FASTWAM_NUM_WORKERS:-8}"
# The upstream trainer needs an integer num_epochs even when max_steps sets the schedule.
num_epochs_override="${FASTWAM_NUM_EPOCHS:-16}"
# Optional step budget. Unset: the schedule follows num_epochs and the dataset length.
max_steps_override="${FASTWAM_MAX_STEPS:-}"
# DeepSpeed ZeRO stage: 1 (upstream default), 2 (shards optimizer and gradients),
# or 2off (stage 2 plus CPU offload of the optimizer, for 24 GB cards).
zero_stage="${FASTWAM_ZERO:-1}"
case "${zero_stage}" in 1|2|2off) ;; *) echo "[ERROR] FASTWAM_ZERO must be 1, 2 or 2off, got ${zero_stage}" >&2; exit 2 ;; esac

if [[ ! -d "${dataset_dir}/meta" ]]; then
    echo "[ERROR] LeRobot dataset not found: ${dataset_dir}/meta"
    echo "Prepare a LeRobot v2.1 dataset under ${converted_root}/lerobot/ and dataset_stats.json."
    if [[ -n "${FASTWAM_DATASET_ID:-}" ]]; then
        echo "FASTWAM_DATASET_ID=${FASTWAM_DATASET_ID}"
    fi
    exit 1
fi

if [[ ! -f "${action_dit}" ]]; then
    echo "[ERROR] Missing ActionDiT backbone: ${action_dit}"
    echo "bash install.sh derives it from the Wan2.2 weights."
    exit 1
fi

if [[ ! -d "${text_cache_dir}" || -z "$(find "${text_cache_dir}" -name '*.pt' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] Missing T5 text embedding cache: ${text_cache_dir}"
    echo "Precompute it with the upstream script in the FastWAM policy environment:"
    echo "  cd ${FASTWAM_DIR}"
    echo "  python scripts/precompute_text_embeds.py \\"
    echo "    task=${task_name} \\"
    echo "    data.train.dataset_dirs=[${dataset_dir}] \\"
    echo "    data.val.dataset_dirs=[${dataset_dir}] \\"
    echo "    data.train.text_embedding_cache_dir=${text_cache_dir} \\"
    echo "    data.val.text_embedding_cache_dir=${text_cache_dir}"
    exit 1
fi

cd "${FASTWAM_DIR}"

# CoHuB validates on its own lerobot_val dataset.
val_dataset_dir="${dataset_dir}"
if [[ "${bench_name}" == "cohub" ]]; then
    val_dataset_dir="${converted_root}/lerobot_val"
    if [[ ! -d "${val_dataset_dir}/meta" ]]; then
        echo "[ERROR] LeRobot val dataset not found: ${val_dataset_dir}/meta"
        echo "Build both splits: bash process_data.sh"
        exit 1
    fi
fi

train_common=(
    "task=${task_name}"
    "seed=${train_seed}"
    "batch_size=${batch_size}"
    "gradient_accumulation_steps=${gradient_accumulation_steps}"
    "num_workers=${num_workers}"
    "num_epochs=${num_epochs_override}"
    "data.train.dataset_dirs=[${dataset_dir}]"
    "data.val.dataset_dirs=[${val_dataset_dir}]"
    "data.train.text_embedding_cache_dir=${text_cache_dir}"
    "data.val.text_embedding_cache_dir=${text_cache_dir}"
    "data.train.pretrained_norm_stats=${stats_path}"
    "data.val.pretrained_norm_stats=${stats_path}"
    "data.train.shape_meta.action.0.raw_shape=${action_dim}"
    "data.train.shape_meta.action.0.shape=${action_dim}"
    "data.train.shape_meta.state.0.raw_shape=${state_dim}"
    "data.train.shape_meta.state.0.shape=${state_dim}"
    "data.val.shape_meta.action.0.raw_shape=${action_dim}"
    "data.val.shape_meta.action.0.shape=${action_dim}"
    "data.val.shape_meta.state.0.raw_shape=${state_dim}"
    "data.val.shape_meta.state.0.shape=${state_dim}"
    "data.train.processor.action_output_dim=${action_dim}"
    "data.train.processor.proprio_output_dim=${state_dim}"
    "data.val.processor.action_output_dim=${action_dim}"
    "data.val.processor.proprio_output_dim=${state_dim}"
    "output_dir=${POLICY_DIR}/checkpoints/${ckpt_setting}"
)

if [[ -n "${max_steps_override}" ]]; then
    train_common+=("max_steps=${max_steps_override}")
fi

# Fine-tuning options:
#   FASTWAM_INIT_WEIGHTS      a .pt to start the weights from (no optimizer state; `resume` wins over it).
#   FASTWAM_FREEZE_VIDEO_DIT  1 -> train the action expert only.
#   FASTWAM_LORA_RANK         >0 -> LoRA on the video expert.
#   FASTWAM_GRAD_CKPT         0/1 -> gradient checkpointing on both experts (the cohub yamls default to on).
if [[ -n "${FASTWAM_INIT_WEIGHTS:-}" ]]; then
    if [[ ! -f "${FASTWAM_INIT_WEIGHTS}" ]]; then
        echo "[ERROR] FASTWAM_INIT_WEIGHTS is not a file: ${FASTWAM_INIT_WEIGHTS}" >&2
        exit 2
    fi
    train_common+=("init_weights=${FASTWAM_INIT_WEIGHTS}")
fi
if [[ "${FASTWAM_FREEZE_VIDEO_DIT:-0}" == "1" ]]; then
    train_common+=("freeze_video_dit=true")
fi
# Empty keeps the task yaml's value.
case "${FASTWAM_GRAD_CKPT:-}" in
    0) train_common+=("model.mot_checkpoint_mixed_attn=false") ;;
    1) train_common+=("model.mot_checkpoint_mixed_attn=true") ;;
    "") ;;
    *) echo "[ERROR] FASTWAM_GRAD_CKPT must be 0 or 1, got ${FASTWAM_GRAD_CKPT}" >&2; exit 2 ;;
esac
if [[ -n "${FASTWAM_LORA_RANK:-}" && "${FASTWAM_LORA_RANK}" != "0" ]]; then
    train_common+=(
        "lora_rank=${FASTWAM_LORA_RANK}"
        "lora_alpha=${FASTWAM_LORA_ALPHA:-16}"
        "lora_dropout=${FASTWAM_LORA_DROPOUT:-0.0}"
    )
fi

# Training-set image augmentation: FASTWAM_AUGMENT_SHIFT=24x16 -> augment_shift_px=[24,16] (dy x dx).
if [[ -n "${FASTWAM_AUGMENT_SHIFT:-}" ]]; then
    if [[ ! "${FASTWAM_AUGMENT_SHIFT}" =~ ^[0-9]+x[0-9]+$ ]]; then
        echo "[ERROR] FASTWAM_AUGMENT_SHIFT must look like 24x16, got ${FASTWAM_AUGMENT_SHIFT}" >&2
        exit 2
    fi
    train_common+=("data.train.augment_shift_px=[${FASTWAM_AUGMENT_SHIFT%x*},${FASTWAM_AUGMENT_SHIFT#*x}]")
fi
if [[ -n "${FASTWAM_AUGMENT_COLOR:-}" ]]; then
    train_common+=("data.train.augment_color=${FASTWAM_AUGMENT_COLOR}")
fi

# FASTWAM_EXTRA: extra hydra overrides, space-separated.
# shellcheck disable=SC2206
if [[ -n "${FASTWAM_EXTRA:-}" ]]; then
    train_common+=(${FASTWAM_EXTRA})
fi

if [[ "${bench_name}" == "cohub" ]]; then
    # The wandb run id is the run-dir name, so a requeue resumes the same run.
    export WANDB_RUN_ID="${WANDB_RUN_ID:-${ckpt_setting}}"
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
    # Keep only the newest trainer state by default; each is tens of GB.
    export FASTWAM_KEEP_STATES="${FASTWAM_KEEP_STATES:-1}"
    train_common+=(
        "wandb.enabled=true"
        ${WANDB_ENTITY:+"wandb.workspace=${WANDB_ENTITY}"}
        "wandb.project=${WANDB_PROJECT:-CoHuB-FastWAM}"
        "wandb.name=${WANDB_NAME:-${ckpt_setting}}"
        "wandb.mode=${WANDB_MODE:-online}"
    )
    # Resume from the newest saved trainer state, if any.
    latest_state=$(ls -1d "${POLICY_DIR}/checkpoints/${ckpt_setting}/checkpoints/state/step_"* 2>/dev/null | sort -V | tail -1) || latest_state=
    if [[ -n "${latest_state}" ]]; then
        echo "[FastWAM] resuming from ${latest_state}"
        train_common+=("resume=${latest_state}")
    fi
fi

bash "scripts/train_zero${zero_stage}.sh" "${num_gpus}" "${train_common[@]}"
