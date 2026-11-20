#!/usr/bin/env bash
# Build the T5 prompt-embedding cache FastWAM trains and serves from.
#
#   bash baselines/policy/FastWAM/prepare_text_cache.sh                # decentralized
#   bash baselines/policy/FastWAM/prepare_text_cache.sh centralized
#
# Run `baselines/policy/FastWAM/train_hook.sh data` first. Runs on the CPU unless
# FASTWAM_CACHE_GPU names a card. An existing cache is kept unless
# FASTWAM_REBUILD_CACHE=1.
set -euo pipefail
WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; export COHUB_WT=$WT
# shellcheck source=../../common.sh
source "$WT/baselines/common.sh"
POLICIES=$WT/baselines
FW=$POLICIES/policy/FastWAM/FastWAM

MODE=${1:-decentralized}
case "$MODE" in
  decentralized) TASK_YAML=cohub_uncond_1cam_192_1e-4 ;;
  centralized)   TASK_YAML=cohub_uncond_2cam_384_1e-4 ;;
  *) echo "mode must be decentralized or centralized; got $MODE" >&2; exit 2 ;;
esac
# Same names as train_hook.sh derives.
CKPT=$(cohub_multitask_word)
ENV_CFG=$(mh_env_cfg_type "$MODE")
DS=$(mh_data_id "$CKPT" "$ENV_CFG")
ROOT=$POLICIES/policy/FastWAM/data/$DS
CACHE=$FW/data/text_embeds_cache/XPolicyLab/$DS

PY=$(cohub_policy_python FastWAM)
[ -x "$PY" ] || { echo "no python at $PY -- install the fastwam env (baselines/policy/FastWAM/install.sh)" >&2; exit 1; }
[ -d "$ROOT/lerobot" ] && [ -d "$ROOT/lerobot_val" ] || {
  echo "no dataset at $ROOT/{lerobot,lerobot_val} -- run baselines/policy/FastWAM/train_hook.sh data first" >&2; exit 1; }
if ls "$CACHE"/*.pt >/dev/null 2>&1 && [ "${FASTWAM_REBUILD_CACHE:-0}" != 1 ]; then
  echo "[fastwam-cache] present: $CACHE"; exit 0
fi

echo "[fastwam-cache] $TASK_YAML -> $CACHE"
cd "$FW"
CUDA_VISIBLE_DEVICES=${FASTWAM_CACHE_GPU:-} \
DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-$FW/checkpoints} \
"$PY" scripts/precompute_text_embeds.py \
  "task=$TASK_YAML" \
  "data.train.dataset_dirs=[$ROOT/lerobot]" "data.val.dataset_dirs=[$ROOT/lerobot_val]" \
  "data.train.text_embedding_cache_dir=$CACHE" "data.val.text_embedding_cache_dir=$CACHE"
echo "[fastwam-cache] done"
