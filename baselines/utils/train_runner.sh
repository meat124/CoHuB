#!/usr/bin/env bash
# The training runner for every policy.
#
#   bash baselines/utils/train_runner.sh <POLICY> <task> <centralized|decentralized> <robot_a|robot_b|-> [seed]
#
# baselines/policy/<POLICY>/train_hook.sh is the entry point that calls this. The runner
# loads baselines/configs/<POLICY>.yaml (a variable already set in the shell
# wins over the file) and sources policy/<POLICY>/train_hook.sh, which may define:
#   MODES_SUPPORTED        optional  default "centralized decentralized"
#   DECENTRALIZED_STYLE    optional  shared (default) | per_robot. The serving
#                                    hook must agree.
#   ACTION_TYPE            optional  joint | ee, default joint
#   run_env_cfg_type()     optional  the middle field of the run-directory
#                                    name, default the mode's profile
#   WANDB_PROJECT          required  the project this policy's runs log to
#   train_env()            optional  exports evaluated at start
#   require_data()         optional  check only: prints the preparing command
#                                    and returns non-zero when data is missing
#                                    (TRAIN_FORCE=1 runs anyway)
#   prepare_data()         optional  idempotent build of derived artifacts
#   train_cmd()            optional  the training process, default the
#                                    policy's 6-arg train.sh chain
#   train_cmd_preview()    optional  print what train_cmd will do, for
#                                    TRAIN_DRY_RUN
#
# TRAIN_DRY_RUN=1 resolves everything and exits before anything is created or
# trained. Resuming is each policy's train.sh's job.

set -euo pipefail

USAGE="usage: utils/train_runner.sh <POLICY> <task> <centralized|decentralized> <robot_a|robot_b|-> [seed]"
POLICY=${1:?$USAGE}
TASK=${2:?$USAGE}
MODE=${3:?$USAGE}
ROBOT=${4:--}
SEED=${5:-0}

WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}; export COHUB_WT=$WT
# shellcheck source=../common.sh
source "$WT/baselines/common.sh"
cohub_runner_boot train "$POLICY"   # sets POLICIES, POLICY_DIR, HOOK and loads the config

# Defaults the hook may override.
MODES_SUPPORTED="centralized decentralized"
# shared: one set of weights drives every robot, told apart by the
# instruction. per_robot: one checkpoint per (task, robot).
DECENTRALIZED_STYLE=shared
ACTION_TYPE=joint
# The middle field of the run directory name `cohub-<ckpt>-<HERE>-<action>-<seed>`.
run_env_cfg_type() { echo "$ENV_CFG_TYPE"; }
train_env()    { :; }
require_data() { return 0; }
prepare_data() { :; }
train_cmd() {
  cd "$POLICY_DIR"
  exec bash train.sh cohub "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}"
}
train_cmd_preview() {
  printf 'cd %q && bash train.sh cohub %q %q %q %q %q\n' \
    "$POLICY_DIR" "$CKPT_NAME" "$ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "${CUDA_VISIBLE_DEVICES:-0}"
}
# shellcheck source=/dev/null
source "$HOOK"
declare -f train_hook >/dev/null || { echo "$HOOK defines no train_hook()" >&2; exit 2; }
train_hook || exit 2
: "${WANDB_PROJECT:?policy/$POLICY/train_hook.sh must set WANDB_PROJECT}"

# COHUB_CKPT_NAME names the run in place of the name built from the task, the
# mode and the robot. The mode and the robot are then the caller's to choose.
[ -z "${COHUB_CKPT_NAME:-}" ] || MODES_SUPPORTED="centralized decentralized"
case " $MODES_SUPPORTED " in
  *" $MODE "*) ;;
  *) echo "[train-runner] $POLICY supports '$MODES_SUPPORTED', not '$MODE'" >&2; exit 2 ;;
esac
case "$DECENTRALIZED_STYLE" in
  shared|per_robot) ;;
  *) echo "[train-runner] policy/$POLICY/train_hook.sh set DECENTRALIZED_STYLE=$DECENTRALIZED_STYLE; expected shared or per_robot" >&2; exit 2 ;;
esac
case "$MODE" in
  centralized)
    [ "$ROBOT" = - ] || { echo "[train-runner] centralized takes no robot (got '$ROBOT'); pass -" >&2; exit 2; } ;;
  decentralized)
    case "$ROBOT" in
      robot_a|robot_b) ;;
      robot_c)
        # Only a scene that has a third robot.
        mh_robots "$(mh_scene "$TASK")" | grep -qx robot_c || {
          echo "[train-runner] $TASK has no robot_c" >&2; exit 2; } ;;
      -)
        [ "$DECENTRALIZED_STYLE" = shared ] || {
          echo "[train-runner] $POLICY trains one checkpoint per robot: pass robot_a or robot_b" >&2; exit 2; }
        case "$TASK" in
          all|multitask|all3|multitask3) ;;
          *) echo "[train-runner] $POLICY's shared decentralized policy trains on every task at once:" >&2
             echo "[train-runner]   pass 'all' (or 'all3' for the three-robot set), or name" >&2
             echo "[train-runner]   robot_a/robot_b for a single-task per-robot run" >&2
             exit 2 ;;
        esac ;;
      *) [ -n "${COHUB_CKPT_NAME:-}" ] || { echo "[train-runner] decentralized needs robot_a, robot_b, robot_c (three-robot tasks) or - (shared), got '$ROBOT'" >&2; exit 2; } ;;
    esac ;;
esac

TASK_WORD=$(mh_task_word "$TASK")
CKPT_NAME=${COHUB_CKPT_NAME:-$(mh_ckpt_name "$TASK_WORD" "$MODE" "$ROBOT")}
ENV_CFG_TYPE=$(mh_env_cfg_type "$MODE")
CKPT_TAG=${CKPT_TAG:-}
RUN_ENV_CFG_TYPE=$(run_env_cfg_type)
RUN_NAME=$(mh_run_name "$CKPT_NAME" "$RUN_ENV_CFG_TYPE" "$ACTION_TYPE" "$SEED" "$CKPT_TAG")

mh_export_common
export WANDB_PROJECT
train_env
# COHUB_TRAIN_ENV: KEY=value lines exported over the hook's.
while read -r pair; do [ -z "$pair" ] || export "$pair"; done <<< "${COHUB_TRAIN_ENV:-}"
# COHUB_TRAIN_DATA: the path the run reads, checked in place of the hook's require_data.
[ -z "${COHUB_TRAIN_DATA:-}" ] || require_data() { [ -e "$COHUB_TRAIN_DATA" ] || { echo "missing $COHUB_TRAIN_DATA"; return 1; }; }

DATA_OK=1
DATA_MSG=$(require_data 2>&1) || DATA_OK=0

if [ "${TRAIN_DRY_RUN:-0}" = 1 ]; then
  echo "[dry-run] policy=$POLICY hook=$HOOK"
  echo "[dry-run] task=$TASK task_word=$TASK_WORD mode=$MODE robot=$ROBOT seed=$SEED"
  echo "[dry-run] decentralized_style=$DECENTRALIZED_STYLE"
  echo "[dry-run] ckpt_name=$CKPT_NAME env_cfg=$ENV_CFG_TYPE action_type=$ACTION_TYPE run=$RUN_NAME"
  echo "[dry-run] wandb_project=$WANDB_PROJECT data_ok=$DATA_OK"
  [ -n "$DATA_MSG" ] && printf '%s\n' "$DATA_MSG" | sed 's/^/[dry-run] data: /'
  # The exports the run would train under.
  for v in COHUB_ROBOT COHUB_ACTION_HORIZON COHUB_SCENE_CAMERA \
           GR00T_LEROBOT_HOME GR00T_BASE_MODEL GR00T_COSMOS_MODEL \
           MAX_STEPS SAVE_STEPS GLOBAL_BATCH_SIZE DATALOADER_NUM_WORKERS \
           LORA_RANK LORA_ALPHA LORA_DROPOUT \
           EVAL_STRATEGY EVAL_STEPS EVAL_BATCH_SIZE EVAL_MAX_SAMPLES \
           USE_WANDB WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RESUME WANDB_TAGS WANDB_NAME \
           TUNE_LLM TUNE_VISUAL PYTORCH_CUDA_ALLOC_CONF \
           HF_LEROBOT_HOME COHUB_DATASETS OPENPI_DATA_HOME XLA_PYTHON_CLIENT_MEM_FRACTION \
           DP_RUN_NAME DP_RUN_TAG DP_NUM_EPOCHS DP_CKPT_EVERY DP_BATCH_SIZE DP_LR DP_HORIZON DP_N_ACTION_STEPS DP_WANDB_TAGS \
           LATENTTOM_RUN_TAG \
           GAUDP_STAGE GAUDP_BENCH GAUDP_CKPT GAUDP_ENV_CFG GAUDP_SEED GAUDP_TAG GAUDP_FINETUNE_MODE GAUDP_SHARED_GAUSSIAN GAUDP_SHARED_TASKS \
           PSI0_BASE_MODEL PSI0_ACTION_HEADER PSI0_MICRO_BATCH PSI0_EVAL_STEPS PSI0_EVAL_BATCHES \
           DIFFSYNTH_MODEL_BASE_PATH FASTWAM_CKPT_SETTING FASTWAM_MAX_STEPS FASTWAM_ZERO FASTWAM_BATCH_SIZE \
           FASTWAM_GRADIENT_ACCUMULATION_STEPS FASTWAM_NUM_WORKERS FASTWAM_GRAD_CKPT FASTWAM_INIT_WEIGHTS; do
    [ -n "${!v:-}" ] && echo "[dry-run] env $v=${!v}"
  done
  echo "[dry-run] cmd:"
  train_cmd_preview | sed 's/^/[dry-run]   /'
  exit 0
fi

if [ "$DATA_OK" != 1 ] && [ "${TRAIN_FORCE:-0}" != 1 ]; then
  echo "[train-runner] $POLICY $TASK_WORD $MODE: training data is not there yet:" >&2
  printf '%s\n' "$DATA_MSG" >&2
  echo "[train-runner] build it first (bash baselines/policy/$POLICY/process_data.sh), or TRAIN_FORCE=1 to run anyway" >&2
  exit 1
fi

echo "[train-runner] policy=$POLICY task=$TASK_WORD mode=$MODE robot=$ROBOT seed=$SEED tag=${CKPT_TAG:-none}"
echo "[train-runner] run=$RUN_NAME wandb=${WANDB_ENTITY:-<account default>}/$WANDB_PROJECT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

prepare_data
train_cmd
