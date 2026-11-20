#!/bin/bash
# On CoHuB:  bash process_data.sh <task>
# Builds the training data from the dataset on the Hub; no GPU. With `cohub` first it runs the conversion below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data ACT "$@"; exit
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4

python detr/process_data.py "$bench_name" "$ckpt_name" "$env_cfg_type" "$action_type"
