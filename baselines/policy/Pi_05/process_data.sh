#!/usr/bin/env bash
# Build the training data from the Hub dataset (no GPU):  bash process_data.sh
# A three-robot task as the argument, or COHUB_TASK_SET=all3, builds the three-robot set. With `cohub` as the first argument it runs the conversion below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data Pi_05 "$@"; exit
fi
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num_or_raw_task_dirs=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CoHuB: pi0.5 reads a LeRobot v3.0 view of the export, plus normalization statistics.
if [[ "${bench_name}" == "cohub" ]]; then
  BASELINES_DIR="$(cd "${POLICY_DIR}/../.." && pwd)"
  COHUB_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
  case "${env_cfg_type}" in
    unitree_g1x2_centralized)   task="${ckpt_name}";          target="centralized" ;;
    unitree_g1x2_decentralized) task="${ckpt_name%_robot_*}"; target="robot_${ckpt_name##*_robot_}" ;;
    *) echo "unknown cohub env_cfg_type ${env_cfg_type}" >&2; exit 2 ;;
  esac
  train_config_name="pi05_cohub_${task}_${target}"
  repo_id="cohub-${task}"

  : "${HF_LEROBOT_HOME:?set HF_LEROBOT_HOME to the LeRobot datasets root}"
  src="${COHUB_DATASETS:-${COHUB_ROOT}/datasets}/${task}/lerobot"
  dst="${HF_LEROBOT_HOME}/${repo_id}"

  cd "${POLICY_DIR}/openpi"
  if [[ ! -f "${dst}/meta/info.json" ]]; then
    echo "[Pi_05] building the v3.0 view: ${src} -> ${dst}"
    uv run python "${POLICY_DIR}/convert_v30.py" --src "${src}" --dst "${dst}"
  else
    echo "[Pi_05] v3.0 view already present: ${dst}"
  fi

  # Norm stats are per target.
  if [[ ! -f "assets/${train_config_name}/${repo_id}/norm_stats.json" ]]; then
    echo "[Pi_05] computing norm stats for ${train_config_name}"
    uv run scripts/compute_norm_stats.py --config-name "${train_config_name}"
  else
    echo "[Pi_05] norm stats already present for ${train_config_name}"
  fi
  echo "[Pi_05] process_data done."
  exit 0
fi

# RoboDojo (upstream)
mode="${OPENPI_DATA_MODE:-image}"
py_args=("${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" --mode "${mode}")
if [[ -n "${expert_data_num_or_raw_task_dirs}" ]]; then
  py_args+=("${expert_data_num_or_raw_task_dirs}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  py_args+=("${raw_task_dirs}")
fi

cd "${POLICY_DIR}/openpi"
python scripts/process_data.py "${py_args[@]}"
