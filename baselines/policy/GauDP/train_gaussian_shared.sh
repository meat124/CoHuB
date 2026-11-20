#!/usr/bin/env bash
# Fine-tune one NoPoSplat encoder on every task's frames.
#
#   bash train_gaussian_shared.sh [seed] [gpu] [task...] [-- extra args]
#   bash train_gaussian_shared.sh 0 0 cocarry handover framehang pouring
#
# The checkpoint lands in the shared run directory, which launcher_args.sh
# searches before the per-task ones.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"

seed=${1:-0}
gpu=${2:-0}
shift $(( $# >= 2 ? 2 : $# ))
tasks=()
extra=()
while (( $# )); do
    if [[ "$1" == "--" ]]; then shift; extra=("$@"); break; fi
    tasks+=("$1"); shift
done
[[ ${#tasks[@]} -gt 0 ]] || tasks=(cocarry handover framehang pouring copouring trashcollection cartservice tablealign)

bench=${GAUDP_BENCH:-cohub}
action_type=$GAUDP_ACTION_TYPE
env_cfg=${GAUDP_ENV_CFG:-$GAUDP_DEFAULT_ENV_CFG}

# Collect the datasets to train over.
datasets=()
for t in "${tasks[@]}"; do
    gaudp_task_config "$t"
    ckpt=$task
    data="$(gaudp_data_path)"
    if [[ ! -f "${data}" ]]; then
        echo "[GauDP][shared] no data at ${data} -- build it first: bash process_data.sh ${task}" >&2; exit 2
    fi
    datasets+=("${data}")
done

source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
for d in "${datasets[@]}"; do require_gaussian_supervision "${d}"; done
# Every dataset must share one camera_order, so the first one names the views.
views="$(gaudp_view_suffix "${datasets[0]}")"
out="$(gaudp_shared_gaussian_dir "${views}")/gaussian"
pretrained="$(resolve_noposplat_checkpoint "${datasets[0]}" "${POLICY_DIR}")"
python_bin="${GAUDP_PYTHON:-python}"

echo "[GauDP][shared] tasks    ${tasks[*]}"
echo "[GauDP][shared] datasets ${#datasets[@]}"
echo "[GauDP][shared] output   ${out}"
wandb_id="shared-gaussian-seed${seed}${GAUDP_TAG:+-${GAUDP_TAG}}${GAUDP_SHARED_VARIANT:+-${GAUDP_SHARED_VARIANT}}${views}-baseline"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_gaussian.py" \
    --data "${datasets[@]}" --output "${out}" --pretrained "${pretrained}" --seed "${seed}" \
    --wandb-run-name "${wandb_id}" \
    --wandb-id "${wandb_id}" \
    --wandb-tags "gaussian,shared,seed-${seed}" \
    ${extra[@]+"${extra[@]}"}
