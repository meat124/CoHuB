#!/usr/bin/env bash
# Train a GauDP policy:  bash train.sh <task> [--tag T]
# Extracts the task's Gaussian features first when they are missing. The shared encoder: train_gaussian.sh.
# Runs baselines/policy/GauDP/train_hook.sh (settings: baselines/configs/GauDP.yaml). `bash train.sh <task> <seed> <gpu> ...` runs the policy stage below.
if ! { [[ "${2:-}" =~ ^[0-9]+$ ]] || [[ "${4:-}" == joint || "${4:-}" == ee ]]; }; then
  case "${1:-}" in
    shared|shared_gaussian) echo "[GauDP] the shared encoder is trained by: bash train_gaussian.sh${2:+ $2}" >&2; exit 2 ;;
  esac
  case "${2:-}" in
    gaussian|features|policy|all)
      echo "[GauDP] train.sh <task> extracts the features and trains the policy; the features alone: bash extract_gaussian_features.sh <task>" >&2; exit 2 ;;
  esac
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train GauDP "$@"; exit
fi
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
# Reuse an existing feature cache when one is found.
if [[ -n "${GAUDP_GAUSSIAN_FEATURES:-}" ]]; then
    gaussian_features="${GAUDP_GAUSSIAN_FEATURES}"
elif ! gaussian_features="$(gaudp_find_gaussian_artifact features.hdf5)"; then
    gaussian_features="${run}/gaussian/features.hdf5"
fi
python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
if [[ ! -f "${data}" ]]; then echo "[GauDP] no data at ${data} -- build it first: bash process_data.sh ${task}" >&2; exit 2; fi
if [[ ! -f "${gaussian_features}" ]]; then
    echo "[GauDP] Offline Gaussian features are required: ${gaussian_features}" >&2
    echo "[GauDP] Run extract_gaussian_features.sh before policy training" >&2
    exit 2
fi
if [[ -n "${GAUDP_GAUSSIAN_CKPT:-}" ]]; then
    gaussian="${GAUDP_GAUSSIAN_CKPT}"
elif gaussian="$(gaudp_find_gaussian_artifact best.ckpt)"; then
    :
else
    gaussian="$(gaudp_cache_checkpoint "${gaussian_features}")"
fi
if [[ ! -s "${gaussian}" ]]; then
    echo "[GauDP] Gaussian checkpoint not found or empty: ${gaussian}" >&2
    echo "[GauDP] Set GAUDP_GAUSSIAN_CKPT or re-run extract_gaussian_features.sh" >&2
    exit 2
fi
echo "[GauDP] data     ${data}"
echo "[GauDP] run      ${run}"
echo "[GauDP] gaussian ${gaussian}"
echo "[GauDP] features ${gaussian_features}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_policy.py" \
    --data "${data}" --output "${run}/policy" --gaussian "${gaussian}" \
    --gaussian-features "${gaussian_features}" --seed "${seed}" "${extra[@]}"
