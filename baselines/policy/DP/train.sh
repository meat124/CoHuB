#!/bin/bash
# On CoHuB:  bash train.sh <task> [--tag T]
# Runs baselines/policy/DP/train_hook.sh (settings: baselines/configs/DP.yaml). With `cohub` first it runs the six-argument chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train DP "$@"; exit
fi

bench_name=${1}
ckpt_name=${2} # run name
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

addition_info=train
# DP_RUN_NAME names the run, its wandb id and its Hydra output directory.
exp_name=${DP_RUN_NAME:-${ckpt_name}-robot_dp-${addition_info}}
run_dir="data/outputs/${DP_RUN_NAME:-${exp_name}_seed${seed}}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/utils"
action_dim=$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" action_dim); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
state_dim=${DP_STATE_DIM:-$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" state_dim)}; echo -e "\033[33m[INFO] State dim: ${state_dim}\033[0m"
num_cameras=${DP_NUM_CAMERAS:-$(bash "${UTILS_DIR}/robot_info.sh" "${ROOT_DIR}" "${env_cfg_type}" num_cameras)}; echo -e "\033[33m[INFO] Num cameras: ${num_cameras}\033[0m"
alg_name=robot_dp

# Camera keys: 1 camera -> head_cam, 2 -> left_cam + right_cam. model.py reads
# the same names.
if [ "${num_cameras}" -ge 2 ]; then
    EXTRA_CAMERA_ARGS=(
        "+task.shape_meta.obs.left_cam.shape=[3,240,320]"
        "+task.shape_meta.obs.left_cam.type=rgb"
        "+task.shape_meta.obs.right_cam.shape=[3,240,320]"
        "+task.shape_meta.obs.right_cam.type=rgb"
    )
else
    EXTRA_CAMERA_ARGS=(
        "+task.shape_meta.obs.head_cam.shape=[3,240,320]"
        "+task.shape_meta.obs.head_cam.type=rgb"
    )
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
# DP_ZARR_PATH names the zarr to train on; its held-out split is <name>_val.zarr beside it.
zarr_path="${DP_ZARR_PATH:-data/${data_setting}.zarr}"
val_zarr_path="${zarr_path%.zarr}_val.zarr"

if [ ! -d "${zarr_path}" ] && [ "${DP_DRY_RUN:-0}" != 1 ]; then
    echo "[DP] no data at ${zarr_path} -- build it first: bash process_data.sh <task>" >&2
    exit 2
fi

val_override=()
if [ -d "${val_zarr_path}" ]; then
    echo -e "\033[33m[INFO] Held-out validation set found: ${val_zarr_path}\033[0m"
    val_override=(task.dataset.val_zarr_path="${val_zarr_path}")
fi

# Optional overrides. Empty keeps robot_dp.yaml's value.
# DP_RUN_TAG keeps a run's checkpoints out of an existing run's directory.
OVERRIDES=()
if [ -n "${DP_RUN_NAME:-}" ]; then
    OVERRIDES+=("logging.name=${DP_WANDB_NAME:-$DP_RUN_NAME}" "logging.id=${DP_RUN_NAME}" "logging.resume=allow" "hydra.run.dir=${run_dir}")
fi
[ -n "${DP_WANDB_PROJECT:-}" ] && OVERRIDES+=("logging.project=${DP_WANDB_PROJECT}")
[ -n "${DP_WANDB_ENTITY:-}" ]  && OVERRIDES+=("+logging.entity=${DP_WANDB_ENTITY}")
[ -n "${DP_WANDB_TAGS:-}" ]    && OVERRIDES+=("logging.tags=${DP_WANDB_TAGS}")
[ -n "${DP_GRAD_CLIP:-}" ]     && OVERRIDES+=("training.max_grad_norm=${DP_GRAD_CLIP}")
[ -n "${DP_ACCUM:-}" ]         && OVERRIDES+=("training.gradient_accumulate_every=${DP_ACCUM}")
[ -n "${DP_HORIZON:-}" ]        && OVERRIDES+=("horizon=${DP_HORIZON}")
[ -n "${DP_N_OBS_STEPS:-}" ]    && OVERRIDES+=("n_obs_steps=${DP_N_OBS_STEPS}")
[ -n "${DP_N_ACTION_STEPS:-}" ] && OVERRIDES+=("n_action_steps=${DP_N_ACTION_STEPS}")
[ -n "${DP_NUM_EPOCHS:-}" ]     && OVERRIDES+=("training.num_epochs=${DP_NUM_EPOCHS}")
[ -n "${DP_CKPT_EVERY:-}" ]     && OVERRIDES+=("training.checkpoint_every=${DP_CKPT_EVERY}")
[ -n "${DP_RESUME_EVERY:-}" ]   && OVERRIDES+=("training.resume_every=${DP_RESUME_EVERY}")
[ -n "${DP_RESUME:-}" ]         && OVERRIDES+=("training.resume=${DP_RESUME}")
# task.dataset.batch_size interpolates dataloader.batch_size, so it follows.
[ -n "${DP_BATCH_SIZE:-}" ]     && OVERRIDES+=("dataloader.batch_size=${DP_BATCH_SIZE}" "val_dataloader.batch_size=${DP_BATCH_SIZE}")
[ -n "${DP_RUN_TAG:-}" ]        && OVERRIDES+=("checkpoint.run_tag=${DP_RUN_TAG}")
# COHUB_RUN_SUFFIX stands in for everything after the checkpoint name (common.sh: mh_run_name).
[ -n "${COHUB_RUN_SUFFIX:-}" ]  && OVERRIDES+=("checkpoint.run_dir=${bench_name}-${ckpt_name}-${COHUB_RUN_SUFFIX}")
[ -n "${DP_WANDB_MODE:-}" ]     && OVERRIDES+=("logging.mode=${DP_WANDB_MODE}")
[ -n "${DP_LR:-}" ]             && OVERRIDES+=("optimizer.lr=${DP_LR}")
[ -n "${DP_NUM_WORKERS:-}" ]    && OVERRIDES+=("dataloader.num_workers=${DP_NUM_WORKERS}" "val_dataloader.num_workers=${DP_NUM_WORKERS}")
[ -n "${DP_OBS_NOISE:-}" ]      && OVERRIDES+=("training.obs_noise=${DP_OBS_NOISE}")
# Crop as HxW, e.g. 200x264. Unset keeps the config's default.
if [ -n "${DP_CROP_SHAPE:-}" ]; then
    OVERRIDES+=("policy.obs_encoder.crop_shape=[${DP_CROP_SHAPE%x*},${DP_CROP_SHAPE#*x}]")
fi
[ -n "${DP_EXTRA:-}" ]         && OVERRIDES+=(${DP_EXTRA})   # any further Hydra override, space-separated
[ ${#OVERRIDES[@]} -gt 0 ] && echo -e "\033[33m[INFO] overrides: ${OVERRIDES[*]}\033[0m"

CMD=("${DP_PYTHON:-python}" train.py --config-name="${alg_name}.yaml"
                bench_name="${bench_name}"
                task.name="${ckpt_name}"
                "task.shape_meta.action.shape=[${action_dim}]"
                "task.shape_meta.obs.agent_pos.shape=[${state_dim}]"
                "${EXTRA_CAMERA_ARGS[@]}"
                task.dataset.zarr_path="${zarr_path}"
                "${val_override[@]}"
                training.debug=$DEBUG
                training.seed=${seed}
                training.device="cuda:0"
                exp_name=${exp_name}
                logging.mode=${wandb_mode}
                setting=${env_cfg_type}
                ${OVERRIDES[@]+"${OVERRIDES[@]}"})
# DP_DRY_RUN=1 prints the command instead of running it.
if [ "${DP_DRY_RUN:-0}" = 1 ]; then
    printf 'cd %q &&' "$(pwd)"; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi
exec "${CMD[@]}"
