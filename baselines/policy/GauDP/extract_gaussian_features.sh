#!/usr/bin/env bash
# Extract one task's Gaussian features:  bash extract_gaussian_features.sh <task> [--tag T]
# train.sh does this first when the cache is missing; run it alone to build the cache on another GPU.
# Runs baselines/policy/GauDP/train_hook.sh <task> features. `bash extract_gaussian_features.sh <task> <seed> <gpu> ...` runs the stage below.
if ! { [[ "${2:-}" =~ ^[0-9]+$ ]] || [[ "${4:-}" == joint || "${4:-}" == ee ]]; }; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train GauDP "$@" features; exit
fi
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
if [[ ! -f "${data}" ]]; then echo "[GauDP] no data at ${data} -- build it first: bash process_data.sh ${task}" >&2; exit 2; fi
python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi

# Reuse an existing feature cache when one is found.
if [[ -n "${GAUDP_GAUSSIAN_FEATURES:-}" ]]; then
    features="${GAUDP_GAUSSIAN_FEATURES}"
elif ! features="$(gaudp_find_gaussian_artifact features.hdf5)"; then
    features="${run}/gaussian/features.hdf5"
fi

# The encoder: the one named, a fine-tuned one, the one an existing cache was
# extracted with, or with GAUDP_SHARED_GAUSSIAN=0 the pretrained NoPoSplat.
set -- "${extra[@]}"
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    gaussian=$1
    shift
elif [[ -n "${GAUDP_GAUSSIAN_CKPT:-}" ]]; then
    gaussian="${GAUDP_GAUSSIAN_CKPT}"
elif gaussian="$(gaudp_find_gaussian_artifact best.ckpt)"; then
    :
elif [[ -f "${features}" ]] && gaussian="$(gaudp_cache_checkpoint "${features}")" && [[ -n "${gaussian}" ]]; then
    :
elif [[ "${GAUDP_SHARED_GAUSSIAN:-1}" == "1" ]]; then
    gaudp_no_shared_encoder
else
    source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
    gaussian="$(resolve_noposplat_checkpoint "${data}" "${POLICY_DIR}")"
fi
if [[ ! -s "${gaussian}" ]]; then
    echo "[GauDP] Gaussian checkpoint not found or empty: ${gaussian}" >&2
    exit 2
fi

echo "[GauDP] gaussian ${gaussian}"
echo "[GauDP] features ${features}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/extract_gaussian_features.py" \
    --data "${data}" --checkpoint "${gaussian}" --output "${features}" --seed "${seed}" "$@"
