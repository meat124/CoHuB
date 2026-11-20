#!/usr/bin/env bash
# On CoHuB:  bash train.sh [--tag T]   (COHUB_TASK_SET=all3, or a three-robot task as the argument, picks the three-robot set)
# Runs baselines/policy/Psi0/train_hook.sh (settings: baselines/configs/Psi0.yaml). With `cohub` as the first argument it runs the chain below.
if [ "${1:-}" != cohub ]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry train Psi0 "$@"; exit
fi
# Fine-tune Psi0 on CoHuB: the action expert trains, the VLM backbone stays frozen.
#
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
#   PSI0_MICRO_BATCH=8 GLOBAL_BATCH_SIZE=32 bash train.sh cohub multitask \
#       unitree_g1x2_decentralized joint 0 0
#
# Defaults follow upstream's scripts/train/psi0/finetune-real-psi0.sh, with a global batch of 32.
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI_ROOT="${POLICY_DIR}/psi0"
PSI_PY="${COHUB_PSI0_PY:-${PSI_ROOT}/.venv/bin/python}"

# CKPT_TAG separates runs over the same task, mode and seed.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}${CKPT_TAG:+-${CKPT_TAG}}"
run_root="${POLICY_DIR}/checkpoints/${ckpt_setting}"
# The dataset name carries no seed and no tag.
data_id="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
data_root="${PSI0_DATA_ROOT:-${POLICY_DIR}/data}"

[ -x "${PSI_PY}" ] || { echo "no Psi0 python at ${PSI_PY} -- run policy/Psi0/install.sh" >&2; exit 1; }
[ -f "${PSI_ROOT}/.env" ] || {
  echo "no ${PSI_ROOT}/.env -- scripts/train.py requires one; install.sh writes it" >&2; exit 1; }
[ -f "${data_root}/${data_id}/meta/stats_psi0.json" ] || {
  echo "missing Psi0 dataset ${data_root}/${data_id} (or its meta/stats_psi0.json)" >&2
  echo "build it with:  bash baselines/policy/Psi0/train_hook.sh data" >&2
  exit 1; }
: "${PSI0_BASE_MODEL:?set it to the pretrained Psi0 VLM directory (site.env)}"
: "${PSI0_ACTION_HEADER:?set it to the pretrained Psi0 action header directory (site.env)}"

val_repo="${data_id}_val"
[ -d "${data_root}/${val_repo}" ] || {
  # Without it Psi0 would validate on the training set.
  echo "missing held-out dataset ${data_root}/${val_repo} -- rebuild with prepare_multitask_data.sh psi0" >&2
  exit 1; }

export CUDA_VISIBLE_DEVICES="${gpu_id}"
NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"

# global batch = micro batch x accumulation x GPUs
MAX_STEPS="${MAX_STEPS:-40000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
# Per-device batch. 8 fits a 48 GB card.
MICRO_BATCH="${PSI0_MICRO_BATCH:-8}"
per_step=$(( MICRO_BATCH * NUM_GPUS ))
if (( GLOBAL_BATCH_SIZE % per_step != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not a multiple of PSI0_MICRO_BATCH x GPUs (${per_step})" >&2
  exit 1
fi
GRAD_ACCUM="${PSI0_GRAD_ACCUM:-$(( GLOBAL_BATCH_SIZE / per_step ))}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
EVAL_STEPS="${PSI0_EVAL_STEPS:-1000}"
EVAL_BATCHES="${PSI0_EVAL_BATCHES:-20}"
LR="${PSI0_LR:-1e-4}"
WARMUP="${PSI0_WARMUP_STEPS:-1000}"
# Optimizer defaults are upstream's; baselines/policy/Psi0/train_hook.sh overrides them.
WEIGHT_DECAY="${PSI0_WEIGHT_DECAY:-1e-6}"
BETA1="${PSI0_BETA1:-0.95}"
BETA2="${PSI0_BETA2:-0.999}"
EPS="${PSI0_EPS:-1e-8}"
MIN_LR="${PSI0_MIN_LR:-}"
LR_SCHEDULER="${PSI0_LR_SCHEDULER:-$([ -n "${MIN_LR}" ] && echo cosine_with_min_lr || echo cosine)}"
# LoRA rank 0 turns LoRA off for that part. PSI0_TUNE_VISION=1 trains the vision
# tower and projector in full. PSI0_GRAD_CKPT=1 turns on gradient checkpointing.
LORA_LLM_RANK="${PSI0_LORA_LLM_RANK:-0}"
LORA_LLM_ALPHA="${PSI0_LORA_LLM_ALPHA:-16}"
LORA_DIT_RANK="${PSI0_LORA_DIT_RANK:-0}"
LORA_DIT_ALPHA="${PSI0_LORA_DIT_ALPHA:-32}"
LORA_DROPOUT="${PSI0_LORA_DROPOUT:-0}"
TUNE_VISION="${PSI0_TUNE_VISION:-0}"
TUNE_DIT_FULL="${PSI0_TUNE_DIT_FULL:-0}"
GRAD_CKPT="${PSI0_GRAD_CKPT:-0}"

# Fixed timestamp: the run directory stays the same across launches, so a requeue resumes.
TIMESTAMP="${PSI0_TIMESTAMP:-cohub}"
TRAIN_NAME="${PSI0_TRAIN_NAME:-finetune}"

# Resume from the newest checkpoint under the run root (PSI0_RESUME=0 turns it off).
resume_args=()
if [ "${PSI0_RESUME:-1}" = 1 ]; then
  latest=$(ls -d "${run_root}/${TRAIN_NAME}"/*/checkpoints/ckpt_* 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "${latest}" ]; then
    resume_dir=$(dirname "$(dirname "${latest}")")
    resume_args=(--train.resume_from_checkpoint="${resume_dir}")
    echo "[Psi0] resuming from ${latest#${run_root}/}"
  fi
fi

mkdir -p "${run_root}"
WANDB_PROJECT="${WANDB_PROJECT:-CoHuB-Psi0}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${ckpt_setting}}"

cd "${PSI_ROOT}"

args=(
  finetune_real_psi0_config
  --seed="${seed}"
  --exp="${ckpt_setting}"
  --timestamp="${TIMESTAMP}"
  --train.name="${TRAIN_NAME}"
  --train.output_dir="${run_root}"
  --train.data_parallel=ddp
  --train.mixed_precision=bf16
  --train.train_batch_size="${MICRO_BATCH}"
  --train.gradient_accumulation_steps="${GRAD_ACCUM}"
  --train.max_checkpoints_to_keep=5
  --train.learning_rate="${LR}"
  --train.max_training_steps="${MAX_STEPS}"
  --train.warmup_ratio=None
  --train.warmup_steps="${WARMUP}"
  --train.checkpointing_steps="${SAVE_STEPS}"
  --train.validation_steps="${EVAL_STEPS}"
  --train.val_num_batches="${EVAL_BATCHES}"
  --train.max_grad_norm=1.0
  --train.lr_scheduler_type="${LR_SCHEDULER}"
  --train.lr_scheduler_kwargs.weight_decay="${WEIGHT_DECAY}"
  --train.lr_scheduler_kwargs.betas "${BETA1}" "${BETA2}"
  --train.lr_scheduler_kwargs.eps="${EPS}"
  --log.report_to=wandb
  --wandb.project="${WANDB_PROJECT}"
  ${WANDB_ENTITY:+--wandb.entity="${WANDB_ENTITY}"}
  --wandb.id="${WANDB_RUN_ID}"
  --wandb.name="${WANDB_NAME:-$WANDB_RUN_ID}"
  --wandb.resume="${WANDB_RESUME:-allow}"
  --data.root_dir="${data_root}"
  --data.train_repo_ids="${data_id}"
  --data.val_repo_ids="${val_repo}"
  --data.transform.repack.image-keys observation.images.ego
  --data.transform.repack.state-key=states
  --data.transform.repack.action-key=action
  --data.transform.repack.instruction-key=task
  --data.transform.repack.pad-action-dim=36
  --data.transform.repack.pad-state-dim=36
  --data.transform.field.stat-path=meta/stats_psi0.json
  --data.transform.field.stat-action-key=action
  --data.transform.field.stat-state-key=states
  --data.transform.field.action_norm_type=bounds
  --data.transform.field.no-use-norm-mask
  --data.transform.field.normalize-state
  --data.transform.field.pad-action-dim=36
  --data.transform.field.pad-state-dim=36
  --data.transform.model.img-aug
  --data.transform.model.resize.size 240 320
  --data.transform.model.center_crop.size 240 320
  --model.model_name_or_path="${PSI0_BASE_MODEL}"
  --model.pretrained-action-header-path="${PSI0_ACTION_HEADER}"
  --model.noise-scheduler=flow
  --model.train-diffusion-steps=1000
  --model.n_conditions=0
  --model.action-chunk-size=30
  --model.action-dim=36
  --model.action-exec-horizon=30
  --model.observation-horizon=1
  --model.odim=36
  --model.view_feature_dim=2048
  --model.no-tune-vlm
  # One learning rate for every parameter group.
  --model.lang_backbone_lr="${LR}"
  --model.vision_tower_lr="${LR}"
  --model.mm_projector_lr="${LR}"
  --model.lora_llm_rank="${LORA_LLM_RANK}"
  --model.lora_llm_alpha="${LORA_LLM_ALPHA}"
  --model.lora_dit_rank="${LORA_DIT_RANK}"
  --model.lora_dit_alpha="${LORA_DIT_ALPHA}"
  --model.lora_dropout="${LORA_DROPOUT}"
  --model.no-use_film
  --model.no-combined_temb
  --model.rtc
  --model.max-delay=8
  "${resume_args[@]}"
)
[ -n "${MIN_LR}" ] && args+=(--train.scheduler_specific_kwargs.min_lr="${MIN_LR}")
[ "${TUNE_VISION}" = 1 ] && args+=(--model.tune-mm-vision --model.tune-mm-mlp)
[ "${TUNE_DIT_FULL}" = 1 ] && args+=(--model.tune-dit-full)
[ "${GRAD_CKPT}" = 1 ] && args+=(--model.gradient_checkpointing)
[ "${PSI0_FROZEN_VLM_BF16:-1}" = 0 ] && args+=(--model.no-frozen-vlm-bf16)
# PSI0_EXTRA: extra flags, split on whitespace.
if [ -n "${PSI0_EXTRA:-}" ]; then
  read -r -a extra <<< "${PSI0_EXTRA}"
  args+=("${extra[@]}")
fi

echo "[Psi0] run dir : ${run_root}/${TRAIN_NAME}/${ckpt_setting}.b${GLOBAL_BATCH_SIZE}.gpus${NUM_GPUS}.${TIMESTAMP}"
echo "[Psi0] batch   : ${MICRO_BATCH} x ${GRAD_ACCUM} accum x ${NUM_GPUS} gpu = ${GLOBAL_BATCH_SIZE}"
echo "[Psi0] steps   : ${MAX_STEPS} (save every ${SAVE_STEPS}, validate every ${EVAL_STEPS})"
echo "[Psi0] optim   : lr=${LR} min_lr=${MIN_LR:-none} sched=${LR_SCHEDULER} warmup=${WARMUP} wd=${WEIGHT_DECAY} betas=${BETA1}/${BETA2} eps=${EPS}"
echo "[Psi0] lora    : llm r=${LORA_LLM_RANK} a=${LORA_LLM_ALPHA}; dit r=${LORA_DIT_RANK} a=${LORA_DIT_ALPHA}; dropout=${LORA_DROPOUT}; tune_vision=${TUNE_VISION} tune_dit_full=${TUNE_DIT_FULL} grad_ckpt=${GRAD_CKPT}"
echo "[Psi0] wandb   : ${WANDB_ENTITY:-<account default>}/${WANDB_PROJECT} id=${WANDB_RUN_ID}"
echo "[Psi0] data    : ${data_root}/${data_id} (+ ${val_repo})"

if [ "${PSI0_PRINT_ONLY:-0}" = 1 ]; then
  printf '%q ' torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" scripts/train.py "${args[@]}"; echo
  exit 0
fi

# Pick a free port so two runs on one host do not collide on torchrun's default.
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=$(bash "${POLICY_DIR}/../../utils/get_free_port.sh" 2>/dev/null) || MASTER_PORT=""
  [ -n "${MASTER_PORT}" ] || MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
fi
echo "[Psi0] master_port: ${MASTER_PORT}"

exec "${PSI_ROOT}/.venv/bin/torchrun" \
  --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py "${args[@]}"
