#!/bin/bash
# On CoHuB:  bash train.sh <task> [--tag T]
# Runs baselines/policy/LatentToM/train_hook.sh (settings: baselines/configs/LatentToM.yaml). With `cohub` first it runs the six-argument chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train LatentToM "$@"; exit
fi
set -e

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Output: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
#   checkpoints/{arm1,arm2}_latest.ckpt, one file per arm.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" || -z "${seed}" || -z "${gpu_id}" ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LATENTTOM_RUN_TAG gives the run its own directory and wandb id. The converted
# data stays untagged.
run_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}${LATENTTOM_RUN_TAG:+-${LATENTTOM_RUN_TAG}}"
run_dir="${POLICY_DIR}/checkpoints/${run_setting}"
# LATENTTOM_DATA_DIR lets several runs share one converted copy.
dataset_path="${LATENTTOM_DATA_DIR:-${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"

if [ ! -d "${dataset_path}" ]; then
    if [[ -n "${LATENTTOM_DATA_DIR:-}" ]]; then
        echo "[LatentToM] LATENTTOM_DATA_DIR=${LATENTTOM_DATA_DIR} is not a directory" >&2
        exit 1
    fi
    echo "[LatentToM] no data at ${dataset_path} -- build it first: bash process_data.sh <task>" >&2
    exit 2
fi

# Trailing arguments (from $7 on) are hydra overrides. `training.debug=true`
# logs offline.
wandb_args=("logging.tags=[latenttom,${env_cfg_type},${action_type},${ckpt_name}]")
for arg in "${@:7}"; do
    case "${arg,,}" in
        training.debug=true|training.debug=1)
            wandb_args+=(logging.mode=offline)
            break
            ;;
    esac
done

export HYDRA_FULL_ERROR=1

if [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[LatentToM] slurm allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; ignoring gpu_id=${gpu_id}"
else
    export CUDA_VISIBLE_DEVICES=${gpu_id}
fi

# One BLAS/OpenMP thread per process. DataLoader workers inherit it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Use the policy env's interpreter. LATENTTOM_PYTHON names it outright.
PY="${LATENTTOM_PYTHON:-${CONDA_PREFIX:-}/bin/python}"
[ -x "$PY" ] || PY=python

# One task config serves every CoHuB two-G1 scene. LATENTTOM_TASK_CONFIG
# selects another.
"$PY" "${POLICY_DIR}/train.py" \
    --config-name=sheaf_xarm_split_diffusion_workspace \
    task="${LATENTTOM_TASK_CONFIG:-cohub_sheaf_split}" \
    task.dataset_path="${dataset_path}" \
    task_name="${ckpt_name}" \
    training.seed="${seed}" \
    hydra.run.dir="${run_dir}" \
    "${wandb_args[@]}" \
    "${@:7}"

echo "[LatentToM] checkpoints written to ${run_dir}/checkpoints/{arm1,arm2}_latest.ckpt"
