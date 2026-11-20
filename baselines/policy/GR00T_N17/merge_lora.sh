#!/usr/bin/env bash
# Fold a LoRA finetune into a standalone model the policy server can load.
# Needs a GPU allocated and ~160 GB of host RAM.
#
#   bash baselines/policy/GR00T_N17/merge_lora.sh <task> <robot_a|robot_b|both|centralized|multitask> [step...] [seed]
#   bash baselines/policy/GR00T_N17/merge_lora.sh all multitask "10000 20000 30000 40000"
#
# `step` defaults to the newest checkpoint, `seed` to 0. CKPT_TAG selects a
# tagged run. Already-merged steps are skipped. `multitask` is the shared
# decentralized policy and ignores the task argument (pass `all`).
# GR00T_BASE_MODEL / GR00T_COSMOS_MODEL are the snapshots
# `baselines/policy/GR00T_N17/install.sh` fetches.

set -euo pipefail

TASK=${1:?usage: merge_lora.sh <task> <multitask|robot_a|robot_b|both|centralized> [step] [seed]}
TARGET=${2:?usage: merge_lora.sh <task> <multitask|robot_a|robot_b|both|centralized> [step] [seed]}
STEP=${3:-}
SEED=${4:-0}

WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
# shellcheck source=../../common.sh
source "$WT/baselines/common.sh"
GR="${WT}/baselines/policy/GR00T_N17"
PY=$(cohub_policy_python GR00T_N17)

for snap in GR00T_BASE_MODEL GR00T_COSMOS_MODEL; do
  [ -e "${!snap}" ] || { echo "$snap=${!snap} is missing -- baselines/policy/GR00T_N17/install.sh fetches it" >&2; exit 1; }
done

case "${TARGET}" in
  robot_a|robot_b)    targets=("${TARGET}") ;;
  both)               targets=(robot_a robot_b) ;;   # what decentralized eval needs
  centralized)        targets=(centralized) ;;
  multitask|shared)   targets=(multitask) ;;         # the shared decentralized policy
  multitask3)         targets=(multitask3) ;;        # the three-robot shared policy
  *) echo "unknown target ${TARGET}" >&2; exit 2 ;;
esac

merge_one() {   # <target> [step]
  local target=$1 step=${2:-} mode=decentralized ckpt_name run_name run ckpt out
  case "${target}" in
    centralized) mode=centralized; ckpt_name=$(mh_ckpt_name "$(mh_task_word "$TASK")" centralized -) ;;
    multitask)   ckpt_name=$(mh_ckpt_name all  decentralized -) ;;
    multitask3)  ckpt_name=$(mh_ckpt_name all3 decentralized -) ;;
    *)           ckpt_name=$(mh_ckpt_name "$(mh_task_word "$TASK")" decentralized "${target}") ;;
  esac
  run_name=$(mh_run_name "${ckpt_name}" "$(mh_env_cfg_type "${mode}")" joint "${SEED}" "${CKPT_TAG:-}")
  # train.sh nests the run directory under itself (output_dir/experiment_name)
  run="${GR}/checkpoints/${run_name}/${run_name}"
  [ -d "${run}" ] || { echo "no run directory: ${run}" >&2; exit 1; }

  if [ -z "${step}" ]; then
    step=$(ls -d "${run}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    [ -n "${step}" ] || { echo "no checkpoint-* under ${run}" >&2; exit 1; }
  fi
  ckpt="${run}/checkpoint-${step}"
  out="${run}/merged-${step}"
  # A finished merge needs no checkpoint beside it: a released model is the merged one alone.
  if ls "${out}"/*.safetensors >/dev/null 2>&1; then
    echo "[merge] ${target}: ${out} exists, skipping"
    return
  fi
  [ -d "${ckpt}" ] || { echo "missing ${ckpt}" >&2; exit 1; }
  # Skip only a finished merge. A directory without weights is a partial one.
  if [ -d "${out}" ]; then
    if ls "${out}"/*.safetensors >/dev/null 2>&1; then
      echo "[merge] ${target}: ${out} exists, skipping"
      return
    fi
    echo "[merge] ${target}: ${out} has no weights -- removing a partial merge and redoing it"
    rm -rf "${out}"
  fi

  # A full finetune has no adapter: its checkpoint already is the whole model.
  if [ ! -f "${ckpt}/adapter_config.json" ]; then
    echo "[merge] ${target}: ${ckpt} is a full model already, nothing to merge"
    return
  fi

  echo "[merge] ${target}: ${ckpt} -> ${out}"
  "${PY}" "${WT}/baselines/policy/GR00T_N17/merge_lora.py" \
    --base "${GR00T_BASE_MODEL}" --cosmos "${GR00T_COSMOS_MODEL}" \
    --checkpoint "${ckpt}" --out "${out}"
}

for target in "${targets[@]}"; do
  if [ -z "${STEP}" ]; then
    merge_one "${target}"                       # newest checkpoint
  else
    for st in ${STEP//,/ }; do
      merge_one "${target}" "${st}"
    done
  fi
done
echo "[merge] done"
