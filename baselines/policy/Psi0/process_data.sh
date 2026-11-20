#!/bin/bash
# Build the training data from the Hub dataset (no GPU):  bash process_data.sh
# A three-robot task as the argument, or COHUB_TASK_SET=all3, builds the three-robot set. With `cohub` as the first argument it runs the chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data Psi0 "$@"; exit
fi
#   bash process_data.sh cohub multitask unitree_g1x2_decentralized joint
# Runs baselines/utils/prepare_data.sh psi0. Output: data/<bench>-<ckpt>-<env_cfg>-<action>{,_val}.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${COHUB_WT:-$(cd "${POLICY_DIR}/../../.." && pwd)}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [[ "${bench_name}" != "cohub" || "${ckpt_name}" != "multitask" ]]; then
  echo "policy/Psi0 has only the CoHuB shared multitask dataset wired up;" >&2
  echo "asked for ${data_setting}" >&2
  exit 2
fi

[ -f "${BENCH_ROOT}/baselines/utils/prepare_data.sh" ] || {
  echo "no CoHuB workspace at ${BENCH_ROOT} -- set COHUB_WT" >&2; exit 1; }

exec bash "${BENCH_ROOT}/baselines/utils/prepare_data.sh" psi0
