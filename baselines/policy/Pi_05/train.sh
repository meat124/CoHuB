#!/usr/bin/env bash
# On CoHuB:  bash train.sh [--tag T]   (COHUB_TASK_SET=all3, or a three-robot task as the argument, picks the three-robot set)
# Runs baselines/policy/Pi_05/train_hook.sh (settings: baselines/configs/Pi_05.yaml). With `cohub` as the first argument it runs the chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train Pi_05 "$@"; exit
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
# ckpt_setting is the run directory name. CKPT_TAG separates runs over the same data.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}${CKPT_TAG:+-${CKPT_TAG}}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"

# CoHuB derives the config name and the LeRobot repo from ckpt_name:
#   centralized:   ckpt_name = <task>
#   decentralized: ckpt_name = multitask | multitask3 (shared policy)
#                  | <task>_robot_a | <task>_robot_b (single-task)
cohub_repo_id=""
if [[ "${bench_name}" == "cohub" ]]; then
  if [[ "${ckpt_name}" == "multitask" || "${ckpt_name}" == "multitask3" ]]; then
    # multitask3 is the three-robot set's shared policy.
    cohub_task="${ckpt_name}"; cohub_target="decentralized"
  else
    case "${env_cfg_type}" in
      unitree_g1x2_centralized)   cohub_task="${ckpt_name}";            cohub_target="centralized" ;;
      unitree_g1x2_decentralized) cohub_task="${ckpt_name%_robot_*}";   cohub_target="robot_${ckpt_name##*_robot_}" ;;
      *) echo "unknown cohub env_cfg_type ${env_cfg_type}" >&2; exit 2 ;;
    esac
  fi
  train_config_name="pi05_cohub_${cohub_task}_${cohub_target}"
  cohub_repo_id="cohub-${cohub_task}"
  # The wandb entity comes from WANDB_ENTITY.
  export WANDB_TAGS="${WANDB_TAGS:-pi05,lora,${cohub_task},${cohub_target}}"
else
  train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
  lerobot_repo_id="${OPENPI_LEROBOT_REPO_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
fi

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# Per-user, per-host local cache for HuggingFace datasets and JAX, to avoid
# lock contention on shared storage.
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-${USER:-$(id -un)}-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

# Resume when the run already has a step directory. --overwrite deletes the run directory.
resume_flag="--overwrite"
if compgen -G "${ckpt_dir}/[0-9]*" > /dev/null; then
  resume_flag="--resume"
fi

echo "[Pi_05] train_config_name=${train_config_name}"
echo "[Pi_05] checkpoint_dir=${ckpt_dir} (${resume_flag})"
echo "[Pi_05] local_cache_root=${LOCAL_CACHE_ROOT}"

cd "${POLICY_DIR}/openpi/"

if [[ "${bench_name}" == "cohub" ]]; then
  echo "[Pi_05] lerobot_repo_id=${cohub_repo_id} (from the config; not overridden)"
  # Training needs norm stats at assets/<config>/<repo_id>/.
  if [[ ! -f "assets/${train_config_name}/${cohub_repo_id}/norm_stats.json" ]]; then
    echo "[Pi_05] norm stats missing; computing them first"
    uv run scripts/compute_norm_stats.py --config-name "${train_config_name}"
  fi
  # Do not pass --data.repo-id: it changes asset_id, and the norm stats are then not found.
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "${train_config_name}" \
      --exp-name="${ckpt_setting}" \
      --fsdp-devices="${fsdp_devices}" \
      --checkpoint-dir-override="${ckpt_dir}" \
      --seed="${seed}" \
      "${resume_flag}"
else
  echo "[Pi_05] lerobot_repo_id=${lerobot_repo_id}"
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "${train_config_name}" \
      --exp-name="${ckpt_setting}" \
      --data.repo-id="${lerobot_repo_id}" \
      --fsdp-devices="${fsdp_devices}" \
      --checkpoint-dir-override="${ckpt_dir}" \
      --seed="${seed}" \
      "${resume_flag}"
fi
