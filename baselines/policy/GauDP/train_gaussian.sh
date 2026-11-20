#!/usr/bin/env bash
# Train the shared Gaussian encoder:  bash train_gaussian.sh [all3] [--tag T]
# Runs baselines/policy/GauDP/train_hook.sh shared. `bash train_gaussian.sh <task> <seed> <gpu> ...` fine-tunes an encoder of that task's own (the stage below).
if ! { [[ "${2:-}" =~ ^[0-9]+$ ]] || [[ "${4:-}" == joint || "${4:-}" == ee ]]; }; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train GauDP shared "$@"; exit
fi
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
# An existing fine-tuned encoder ends the stage. GAUDP_FORCE_GAUSSIAN=1 fine-tunes anyway.
if [[ "${GAUDP_FORCE_GAUSSIAN:-0}" != "1" ]] && existing="$(gaudp_find_gaussian_artifact best.ckpt)"; then
    echo "[GauDP] reusing the fine-tuned Gaussian encoder: ${existing}"
    echo "[GauDP] set GAUDP_FORCE_GAUSSIAN=1 to fine-tune it again"
    exit 0
fi

if [[ ! -f "${data}" ]]; then echo "[GauDP] no data at ${data} -- build it first: bash process_data.sh ${task}" >&2; exit 2; fi
if [[ "${GAUDP_SHARED_GAUSSIAN:-1}" == "1" && "${GAUDP_FORCE_GAUSSIAN:-0}" != "1" ]]; then
    gaudp_no_shared_encoder
fi

source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
require_gaussian_supervision "${data}"
pretrained="$(resolve_noposplat_checkpoint "${data}" "${POLICY_DIR}")"
python_bin="${GAUDP_PYTHON:-python}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_gaussian.py" \
    --data "${data}" --output "${run}/gaussian" --pretrained "${pretrained}" --seed "${seed}" "${extra[@]}"
