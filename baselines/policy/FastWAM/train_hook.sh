#!/usr/bin/env bash
# FastWAM: train the shared multitask policy. Knobs: baselines/configs/FastWAM.yaml.
#
#   baselines/policy/FastWAM/train_hook.sh                            train the shared multitask policy
#   baselines/policy/FastWAM/train_hook.sh data                       build the flattened all-task dataset it reads (CPU only)
#   FASTWAM_MAX_STEPS=20000 baselines/policy/FastWAM/train_hook.sh    a config key set in the shell wins over the file
#   TRAIN_DRY_RUN=1 baselines/policy/FastWAM/train_hook.sh            resolve everything and stop

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_boot FastWAM
  # COHUB_TASK_SET=all3 trains (and builds) the three-robot set, checkpoint `multitask3`.
  cohub_train_main FastWAM multitask "$@"; exit
fi

# ---- the hook utils/train_runner.sh sources ----
train_hook() {
# The benchmark's recipe is LoRA on the video expert with the action expert
# trained in full (configs/FastWAM.yaml). Other arms:
#   FASTWAM_LORA_RANK=0             full fine-tune
#   FASTWAM_FREEZE_VIDEO_DIT=1      action expert only
#   FASTWAM_INIT_WEIGHTS=<file.pt>  start from a checkpoint instead of the Wan2.2 weights
# train_env picks the ZeRO stage and the micro-batch from the memory of the
# cards CUDA_VISIBLE_DEVICES grants. FASTWAM_ZERO (1, 2, 2off) overrides the stage.
WANDB_PROJECT=${WANDB_PROJECT:-CoHuB_FastWAM}

# `none` starts from the Wan2.2 video weights and the derived ActionDiT.
[ "${FASTWAM_INIT_WEIGHTS:-none}" != none ] || FASTWAM_INIT_WEIGHTS=
export FASTWAM_INIT_WEIGHTS

train_env() {
  # train.sh calls `accelerate launch` off PATH, so the policy env's bin goes first.
  local env_bin
  env_bin=$(dirname "$(cohub_policy_python FastWAM)")
  # A dry run continues without the env.
  [ -x "$env_bin/accelerate" ] || { echo "[FastWAM hook] no accelerate in $env_bin -- is the fastwam env installed?" >&2
                                    [ "${TRAIN_DRY_RUN:-0}" = 1 ] || return 1; }
  case ":$PATH:" in *":$env_bin:"*) ;; *) export PATH="$env_bin:$PATH" ;; esac

  # The trainer needs a seed above 0, so train.sh runs seed 0 as 1.
  [ "$SEED" != 1 ] || { echo "[FastWAM hook] ERROR: SEED=1 is the training SEED=0 already runs; use SEED=2 or above" >&2; return 1; }

  # The Wan2.2 weights (DiT, VAE, T5). The ActionDiT backbone stays in FastWAM/checkpoints.
  export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-$POLICIES/policy/FastWAM/FastWAM/checkpoints}
  export FASTWAM_CKPT_SETTING=${FASTWAM_CKPT_SETTING:-$RUN_NAME}
  export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
  export WANDB_NAME=${WANDB_NAME:-$(cohub_wandb_name "$CKPT_NAME" FastWAM train)}
  export WANDB_RESUME=${WANDB_RESUME:-allow}
  # The benchmark's budget: global batch 32, 40k steps. The task yamls leave
  # max_steps null, so this sets it.
  export FASTWAM_MAX_STEPS=${FASTWAM_MAX_STEPS:-40000}
  # Global batch = micro-batch x GPUs x accumulation.
  local global_batch=${FASTWAM_GLOBAL_BATCH:-32}
  # Gradient checkpointing. FASTWAM_GRAD_CKPT=0 turns it off where memory allows.
  export FASTWAM_GRAD_CKPT=${FASTWAM_GRAD_CKPT:-1}
  if [ "${FASTWAM_FREEZE_VIDEO_DIT:-0}" = "1" ] && [ -n "${FASTWAM_LORA_RANK:-}" ] \
     && [ "${FASTWAM_LORA_RANK}" != "0" ]; then
    echo "[FastWAM hook] ERROR: FASTWAM_FREEZE_VIDEO_DIT and FASTWAM_LORA_RANK are" \
         "alternatives -- a frozen video expert leaves the adapters nothing to adapt." >&2
    return 1
  fi

  local n_gpus vram_mb vram_gb micro free cap
  n_gpus=$(($(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr -cd , | wc -c) + 1))
  # Assumes 48 GB cards when nvidia-smi gives no number. FASTWAM_VRAM_MB
  # overrides the probe.
  vram_mb=${FASTWAM_VRAM_MB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n 1p)} || true
  [[ "$vram_mb" =~ ^[0-9]+$ ]] || vram_mb=49140
  vram_gb=$((vram_mb / 1024))

  # Memory model per GPU, in GiB: 11 for the bf16 weights plus 18 bytes per
  # trainable parameter, sharded over the GPUs. Trainable parameters in
  # millions: 6021 full, 1021 action expert only, 2.519 more per LoRA rank.
  local trainable_m=6021
  if [ "${FASTWAM_FREEZE_VIDEO_DIT:-0}" = "1" ]; then
    trainable_m=1021
  elif [ -n "${FASTWAM_LORA_RANK:-}" ] && [ "${FASTWAM_LORA_RANK}" != "0" ]; then
    trainable_m=$(( 1021 + 2519040 * FASTWAM_LORA_RANK / 1000000 ))
  fi
  # /1074 turns M-params x bytes into GiB (1e6 / 2^30).
  local sharded=$(( 18 * trainable_m / 1074 ))
  local grads=$(( 2 * trainable_m / 1074 ))
  local share_z2=$(( (sharded + n_gpus - 1) / n_gpus ))
  local share_off=$(( (grads + n_gpus - 1) / n_gpus ))
  # ZeRO-1 keeps the full bf16 gradient on every rank and shards only the
  # 16-byte optimizer block.
  local share_z1=$(( grads + (16 * trainable_m / 1074 + n_gpus - 1) / n_gpus ))

  # Stage 1 when it leaves 6 GiB for activations, else stage 2, else stage 2
  # with CPU offload.
  if [ -z "${FASTWAM_ZERO:-}" ]; then
    if [ $(( vram_gb - 11 - share_z1 )) -ge 6 ]; then
      FASTWAM_ZERO=1
    elif [ $(( vram_gb - 11 - share_z2 )) -ge 6 ]; then
      FASTWAM_ZERO=2
    else
      FASTWAM_ZERO=2off
    fi
  fi
  export FASTWAM_ZERO
  case "$FASTWAM_ZERO" in
    1)    free=$(( vram_gb - 11 - share_z1 )) ;;
    2)    free=$(( vram_gb - 11 - share_z2 )) ;;
    2off) free=$(( vram_gb - 11 - share_off ))
          # The CPU runs the AdamW step: leave it cores.
          export FASTWAM_NUM_WORKERS=${FASTWAM_NUM_WORKERS:-4} ;;
  esac

  if [ "$free" -lt 5 ]; then
    echo "[FastWAM hook] ERROR: ${n_gpus}x${vram_gb}GiB leaves ${free} GiB after weights" \
         "and optimizer state -- not enough for activations. Give the run more GPUs" \
         "(or bigger ones): 24 GiB cards need at least 4, 48 GiB at least 1 with offload." >&2
    return 1
  fi

  # Map the leftover memory to a micro-batch. Accumulation makes up the
  # global batch.
  if   [ "$free" -ge 24 ]; then cap=8
  elif [ "$free" -ge 14 ]; then cap=4
  elif [ "$free" -ge 8 ];  then cap=2
  else                          cap=1
  fi
  # Without gradient checkpointing activations roughly double: halve the cap.
  [ "$FASTWAM_GRAD_CKPT" = 0 ] && cap=$(( cap > 1 ? cap / 2 : 1 ))
  micro=$(( global_batch / n_gpus ))
  [ "$micro" -le "$cap" ] || micro=$cap
  # The largest micro-batch that divides the global batch over the cards.
  while [ "$micro" -gt 1 ] && [ $(( global_batch % (micro * n_gpus) )) -ne 0 ]; do micro=$(( micro - 1 )); done
  [ "$micro" -ge 1 ] || micro=1

  export FASTWAM_BATCH_SIZE=${FASTWAM_BATCH_SIZE:-$micro}
  local accum=$((global_batch / (FASTWAM_BATCH_SIZE * n_gpus)))
  [ "$accum" -ge 1 ] || accum=1
  export FASTWAM_GRADIENT_ACCUMULATION_STEPS=${FASTWAM_GRADIENT_ACCUMULATION_STEPS:-$accum}
  local effective=$(( FASTWAM_BATCH_SIZE * n_gpus * FASTWAM_GRADIENT_ACCUMULATION_STEPS ))
  if [ "$effective" -ne "$global_batch" ]; then
    echo "[FastWAM hook] ERROR: micro-batch ${FASTWAM_BATCH_SIZE} x ${n_gpus} GPUs x accumulation" \
         "${FASTWAM_GRADIENT_ACCUMULATION_STEPS} = ${effective}, not FASTWAM_GLOBAL_BATCH=${global_batch}." \
         "Use a GPU count and micro-batch that divide it, or set FASTWAM_GLOBAL_BATCH=${effective}." >&2
    return 1
  fi

  # The CPU AdamW step needs more than one OMP thread.
  if [ "${FASTWAM_ZERO}" = "2off" ]; then
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / n_gpus ))}
    [ "${OMP_NUM_THREADS}" -ge 1 ] || export OMP_NUM_THREADS=1
  fi

  # Dataloader workers per rank: the cores per rank minus one, at most 8.
  local cpus; cpus=$(nproc) workers=
  workers=$(( cpus / n_gpus ))
  workers=$(( workers > 1 ? workers - 1 : 1 ))
  [ "$workers" -le 8 ] || workers=8
  export FASTWAM_NUM_WORKERS=${FASTWAM_NUM_WORKERS:-$workers}

  echo "[FastWAM hook] gpus=${n_gpus} vram=${vram_mb}MiB zero=${FASTWAM_ZERO}" \
       "trainable=${trainable_m}M sharded=${sharded}GiB free=${free}GiB" \
       "micro_batch=${FASTWAM_BATCH_SIZE} accum=${FASTWAM_GRADIENT_ACCUMULATION_STEPS}" \
       "global_batch=$((FASTWAM_BATCH_SIZE * n_gpus * FASTWAM_GRADIENT_ACCUMULATION_STEPS))" \
       "max_steps=${FASTWAM_MAX_STEPS} workers=${FASTWAM_NUM_WORKERS}" \
       "init_weights=${FASTWAM_INIT_WEIGHTS:-none} freeze=${FASTWAM_FREEZE_VIDEO_DIT:-0}" \
       "lora_rank=${FASTWAM_LORA_RANK:-0} grad_ckpt=${FASTWAM_GRAD_CKPT} omp=${OMP_NUM_THREADS:-default}"
}

train_cmd() {
  # train.sh derives num_gpus from CUDA_VISIBLE_DEVICES.
  cd "$POLICY_DIR"
  exec bash train.sh cohub "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}"
}

train_cmd_preview() {
  printf 'cd %q && bash train.sh cohub %q %q %q %q ${CUDA_VISIBLE_DEVICES:-0}\n' \
    "$POLICY_DIR" "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED"
}

require_data() {
  local ok=1 id; id=$(mh_data_id "$CKPT_NAME" "$ENV_CFG_TYPE")
  local data_root="$POLICIES/policy/FastWAM/data/$id"
  local base="$POLICIES/policy/FastWAM/FastWAM/checkpoints"
  local cache="$POLICIES/policy/FastWAM/FastWAM/data/text_embeds_cache/XPolicyLab/$id"
  for want in "$data_root/lerobot/meta/info.json" "$data_root/lerobot_val/meta/info.json" "$data_root/dataset_stats.json"; do
    [ -e "$want" ] || { echo "missing $want"; ok=0; }
  done
  if [ "$ok" = 0 ]; then
    if [ "$CKPT_NAME" = multitask ] || [ "$CKPT_NAME" = multitask3 ]; then
      echo "prepare the dataset with:  bash baselines/utils/prepare_data.sh fastwam"
    else
      echo "prepare the dataset with:  COHUB_DATA_IN=FastWAM bash baselines/policy/FastWAM/process_data.sh $TASK_WORD $MODE"
    fi
  fi
  if [ ! -f "$base/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" ]; then
    echo "missing ActionDiT backbone under $base -- baselines/policy/FastWAM/install.sh derives it (CPU, ~2 min)"
    ok=0
  fi
  if [ -n "${FASTWAM_INIT_WEIGHTS:-}" ] && [ ! -f "$FASTWAM_INIT_WEIGHTS" ]; then
    echo "missing init weights $FASTWAM_INIT_WEIGHTS (leave FASTWAM_INIT_WEIGHTS unset to train from the Wan2.2 video weights, the paper's start)"
    ok=0
  fi
  # The Wan2.2 files the model is built from.
  local wan_base=${DIFFSYNTH_MODEL_BASE_PATH:-$base}
  local wan="$wan_base/Wan-AI/Wan2.2-TI2V-5B"
  for want in "$wan/diffusion_pytorch_model-00001-of-00003.safetensors" \
              "$wan/models_t5_umt5-xxl-enc-bf16.pth" "$wan/Wan2.2_VAE.pth" \
              "$wan_base/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"; do
    [ -e "$want" ] || { echo "missing base weight $want -- baselines/policy/FastWAM/install.sh fetches it"; ok=0; }
  done
  if ! ls "$cache"/*.pt >/dev/null 2>&1; then
    echo "missing T5 text-embedding cache at $cache -- build it with:  bash baselines/policy/FastWAM/prepare_text_cache.sh"
    ok=0
  fi
  [ "$ok" = 1 ]
}

}
