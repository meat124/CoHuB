#!/usr/bin/env bash
# Evaluation runner for every policy: one policy server, one Isaac Sim client
# process per episode, results folded into results.json and uploaded to wandb.
#
#   bash scripts/eval/runner.sh DP framehang [episodes] [seed]
#   (defaults: 100 episodes, seeds 1..100)
#   EVAL_CKPT_NUM=40000 bash scripts/eval/runner.sh Pi_05 handover
#
# scripts/eval/eval.py is the entry point that loops this over tasks and checkpoints.
# serve_<POLICY> in policies.sh, sourced below, sets what differs per policy:
#   POLICY_PY              interpreter that can import the policy's deps (required)
#   resolve_model_dir RUN  -> the servable checkpoint directory; the default is
#                             checkpoints/RUN. Printing nothing leaves
#                             resolution to the policy
#   model_dir_guard RUN SRC -> 0 to leave this run to the policy
#   server_env             -> exports the server needs
#   SERVER_ENV_CFG         -> scene (default) | profile: what the server gets as
#                             env_cfg_type, the Isaac scene or the mode's
#                             unitree_g1x2_* profile
#   MODEL_DIR_KEY          -> deploy.yml key for the checkpoint dir
#   ACTION_TYPE            -> joint, what the policy predicts
#   DECENTRALIZED_STYLE    -> shared (default) | per_robot
#   run_names              -> "<run-dir>:<deploy.yml key>" per line
#   SERVER_TIMEOUT_S       -> seconds to wait for the server's port, 3600
#   RPC_TIMEOUT_S          -> seconds the client waits per inference call;
#                             empty = the client's own default
#   WANDB_UPLOAD           -> 0 to skip the wandb upload
#   EVAL_SERVER_OVERRIDES  -> extra key=value words for the server's --overrides
#
# Environment variables:
#   EVAL_DRY_RUN=1           resolve everything and stop before anything needs a GPU
#   STEP_LIMIT               per-episode step budget, over the task's own (mh_step_limit)
#   EVAL_CKPT_NUM            which saved checkpoint the server loads
#   EVAL_WANDB=0             skip the wandb upload
#   EVAL_OBS_CAMERAS, EVAL_OBS_STRIDE, EVAL_VIDEO=0, EVAL_VIDEO_CAMERAS, EVAL_VIDEO_STRIDE,
#   EVAL_VIDEO_SCALE, EVAL_SMOOTH_VIDEO, EVAL_SCENE_CAM, EVAL_FOLLOW_DISTANCE,
#   EVAL_FOLLOW_MAX_DOLLY, EVAL_EGO_DEPTH, EVAL_DROP_UNUSED_CAMERAS   rendering, see below
#   EVAL_SERVER_TIMEOUT, EVAL_RPC_TIMEOUT_S                          override the hook's timeouts
#   EVAL_EXEC_HORIZON, EVAL_SAMPLER, EVAL_INFERENCE_STEPS, EVAL_ACTION_CLIP, EVAL_OBS_CLIP
#                            server-side overrides, read by the DP-family adapters only
#   EVAL_CLIENT_RETRIES (2)  replays of a client chunk that crashed
#   EVAL_MAX_MISSING (3)     client processes in a row with no results file before the run stops
#   EVAL_MIN_FREE_VRAM_MB (6144)  the GPU memory Isaac must have left once the server is up
#   EVAL_EPISODES_PER_PROC (1)    episodes per Isaac process; >1 is untested
#   EVAL_RUN_ID, POLICY_NAME       override the results directory's name
#   EVAL_KEEP_GPU_CSV=1      keep the nvidia-smi samples beside results.json
#   EVAL_POLICY_SEED_SALT, COHUB_SEED_SERVER, CKPT_SEED, COHUB_MODE,
#   COHUB_DECENTRALIZED_STYLE, COHUB_EVAL_LABEL                     see their lines below
#   COHUB_SIM_PY             the Isaac client's interpreter (default: this checkout's .venv)
#   COHUB_SIM_GPU, COHUB_POLICY_GPU   the client's card and the server's: the first of
#                            CUDA_VISIBLE_DEVICES unless named

set -euo pipefail

USAGE="usage: eval/runner.sh <POLICY> <cocarry|handover|framehang> [episodes=100] [seed=1]"
POLICY=${1:?$USAGE}
TASK=${2:?$USAGE}
EPISODES=${3:-100}
SEED=${4:-1}
# SEED is the evaluation seed: episode i uses SEED+i for its scene and its
# policy noise. CKPT_SEED is the training seed in the checkpoint run-dir name.
CKPT_SEED=${CKPT_SEED:-0}
# Salt for repeats of a stochastic policy. Not part of checkpoint resolution.
POLICY_SEED_SALT=${EVAL_POLICY_SEED_SALT:-0}
POLICY_SEED_SCHEDULE=none
[ "$POLICY" = FastWAM ] && POLICY_SEED_SCHEDULE=blake2b-v1
# 1 = the policy server seeds its generators at start and per episode.
SEED_SERVER=${COHUB_SEED_SERVER:-1}

WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}; export COHUB_WT=$WT
# shellcheck source=../common.sh
source "$WT/baselines/common.sh"
cohub_runner_boot eval "$POLICY"    # sets POLICIES, POLICY_DIR, HOOK and loads the config
# The Isaac Sim client's interpreter.
SIM_PY=${COHUB_SIM_PY:-$WT/.venv/bin/python}
model_dir_guard()   { return 1; }     # defaults the hook may override; 0 = do not pass this run's dir
server_env()        { :; }
WANDB_UPLOAD=${WANDB_UPLOAD:-1}       # 0 = skip the wandb upload
resolve_model_dir() {                 # <run-name> -> the run directory itself, when it exists
  # A policy with its own layout overrides this. Printing nothing leaves
  # resolution to the policy.
  local base="$CKPT_DIR/$1"
  [ -d "$base" ] && echo "$base"
  return 0
}
SERVER_ENV_CFG=scene                  # scene | profile: what the server gets as env_cfg_type
MODEL_DIR_KEY=model_dir               # deploy.yml key that names a servable dir
# What the policy predicts: joint targets, which is all the evaluation env takes.
ACTION_TYPE=joint
# Seconds to wait for the server's port, and per inference call.
# EVAL_SERVER_TIMEOUT / EVAL_RPC_TIMEOUT_S override them.
SERVER_TIMEOUT_S=3600
RPC_TIMEOUT_S=
# What "decentralized" means for this policy's checkpoints:
#   shared     one multitask policy drives every robot, told apart by the instruction
#   per_robot  one checkpoint per (task, robot)
# COHUB_DECENTRALIZED_STYLE overrides it.
DECENTRALIZED_STYLE=shared
# Views the env client renders for the policy. A different list changes the
# rendered frames, so compare results only under the same list.
OBS_CAMERAS=scene,ego_a,ego_b
SCENE_CAM=follow
# Checkpoint run directories to serve, one "<run-dir-name>:<deploy.yml key>" per line.
run_names() {
  if [ "$COHUB_MODE" = centralized ] || [ "$DECENTRALIZED_STYLE" = shared ]; then
    echo "$(mh_run_name "$CKPT_NAME" "$(mh_env_cfg_type "$COHUB_MODE")" joint "$CKPT_SEED" "$CKPT_TAG"):${MODEL_DIR_KEY}"
  else
    # One checkpoint per robot in the scene.
    local env_cfg robot
    env_cfg=$(mh_env_cfg_type decentralized)
    while read -r robot; do
      echo "$(mh_run_name "${CKPT_NAME}_${robot}" "$env_cfg" joint "$CKPT_SEED" "$CKPT_TAG"):model_dir_${robot}"
    done < <(mh_robots "$ENV_CFG_TYPE")
  fi
}
# shellcheck source=/dev/null
source "$HOOK"
declare -f "serve_$POLICY" >/dev/null || { echo "$HOOK defines no serve_$POLICY()" >&2; exit 2; }
"serve_$POLICY" || exit 2
: "${POLICY_PY:?serve_$POLICY must set POLICY_PY}"
[ "$ACTION_TYPE" = joint ] || { echo "[eval-runner] serve_$POLICY set ACTION_TYPE=$ACTION_TYPE; the evaluation env takes joint targets only" >&2; exit 2; }
[ -x "$POLICY_PY" ] || { echo "[eval-runner] POLICY_PY is not executable: $POLICY_PY" >&2; exit 2; }

# TASK_WORD is the dataset word in checkpoint names, ENV_CFG_TYPE the Isaac
# scene. Either spelling is accepted on the command line.
# The step budget is per task (mh_step_limit) and below the env's own
# episode_length_s, so an episode that runs out reports step_limit.
TASK_WORD=$(mh_task_word "$TASK") || exit 2
ENV_CFG_TYPE=$(mh_scene "$TASK") || exit 2
# The task, not the caller's COHUB_TASK_SET, picks the shared policy's set.
export COHUB_TASK_SET; COHUB_TASK_SET=$(cohub_task_set "$TASK_WORD")
DEF_STEP_LIMIT=$(mh_step_limit "$ENV_CFG_TYPE")
STEP_LIMIT=${STEP_LIMIT:-$DEF_STEP_LIMIT}

export PATH="$WT/baselines/bin:$PATH"   # ffprobe
export OMNI_KIT_ACCEPT_EULA=YES

# Isaac assets come from a local mirror. A missing asset does not raise: the
# prim spawns empty and the scene fails later.
export ISAAC_ASSETS; ISAAC_ASSETS=$(mh_find_isaac_assets)
[ -d "$ISAAC_ASSETS" ] || { echo "[eval-runner] no Isaac asset mirror at $ISAAC_ASSETS -- set ISAAC_ASSETS (site.env)" >&2; exit 1; }
export COHUB_G1_USD=${COHUB_G1_USD:-$ISAAC_ASSETS/Robots/Unitree/G1/g1.usd}
export COHUB_GRID_USD=${COHUB_GRID_USD:-$ISAAC_ASSETS/Environments/Grid/default_environment.usd}
export COHUB_G1_IK_URDF=${COHUB_G1_IK_URDF:-$ISAAC_ASSETS/IsaacLab/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf}
export COHUB_AGILE_POLICY=${COHUB_AGILE_POLICY:-$ISAAC_ASSETS/IsaacLab/Policies/Agile/agile_locomotion.pt}
for asset in "$COHUB_G1_USD" "$COHUB_GRID_USD" "$COHUB_G1_IK_URDF" "$COHUB_AGILE_POLICY"; do
  [ -e "$asset" ] || { echo "[eval-runner] missing Isaac asset: $asset" >&2; exit 1; }
done


# Per-episode JSONs live in a dot-dir and are deleted once folded in.
# COHUB_MODE: decentralized (default), or centralized for one policy that
# emits every robot's action.
# COHUB_SERVE_MODE sets it past a hook that refuses the mode.
COHUB_MODE=${COHUB_SERVE_MODE:-${COHUB_MODE:-decentralized}}
CKPT_TAG=${CKPT_TAG:-}
DECENTRALIZED_STYLE=${COHUB_DECENTRALIZED_STYLE:-$DECENTRALIZED_STYLE}
case "$DECENTRALIZED_STYLE" in
  shared|per_robot) ;;
  *) echo "[eval-runner] DECENTRALIZED_STYLE=$DECENTRALIZED_STYLE; expected shared or per_robot" >&2; exit 2 ;;
esac
# Checkpoint name: the task word, or the multitask word for a policy trained on
# every task. The adapter appends _robot_<x> itself in the per-robot case.
if [ "$COHUB_MODE" = decentralized ] && [ "$DECENTRALIZED_STYLE" = shared ]; then
  # multitask3 for a three-robot task
  CKPT_NAME=$(cohub_multitask_word)
else
  CKPT_NAME=$TASK_WORD
fi
# COHUB_CKPT_NAME names it instead; COHUB_SERVE_RUNS lists the run directories
# to serve, in run_names' format.
CKPT_NAME=${COHUB_CKPT_NAME:-$CKPT_NAME}
[ -z "${COHUB_SERVE_RUNS:-}" ] || run_names() { echo "$COHUB_SERVE_RUNS"; }
# The server's env_cfg_type: the scene, or the mode's unitree_g1x2_* profile for
# an adapter that keys off it.
case "$SERVER_ENV_CFG" in
  scene)   SERVER_ENV_CFG_TYPE=$ENV_CFG_TYPE ;;
  profile) SERVER_ENV_CFG_TYPE=$(mh_env_cfg_type "$COHUB_MODE") ;;
  *) echo "[eval-runner] serve_$POLICY set SERVER_ENV_CFG=$SERVER_ENV_CFG; expected scene or profile" >&2; exit 2 ;;
esac
# Cameras the episode videos come from, one directory each under videos/.
# A three-robot scene adds ego_c to the default lists. An explicit EVAL_* list wins.
if mh_robots "$ENV_CFG_TYPE" | grep -qx robot_c; then
  case ",$(printf '%s' "$OBS_CAMERAS" | tr ':+ \t' ',,,,')," in
    *,ego_b,*) OBS_CAMERAS=$OBS_CAMERAS,ego_c ;;
  esac
  DEF_VIDEO_CAMERAS=scene,ego_a,ego_b,ego_c
fi
# Task-declared extra video cameras join the default list. They are never policy inputs.
EXTRA_VIDEO_CAMERAS=$(mh_extra_video_cameras "$ENV_CFG_TYPE")
if [ -n "$EXTRA_VIDEO_CAMERAS" ]; then
  DEF_VIDEO_CAMERAS=${DEF_VIDEO_CAMERAS:-scene,ego_a,ego_b},$EXTRA_VIDEO_CAMERAS
fi
EVAL_OBS_CAMERAS=${EVAL_OBS_CAMERAS:-$OBS_CAMERAS}
EVAL_VIDEO_CAMERAS=${EVAL_VIDEO_CAMERAS:-${DEF_VIDEO_CAMERAS:-scene,ego_a,ego_b}}
# Upsampling factor ffmpeg applies to the saved videos.
EVAL_VIDEO_SCALE=${EVAL_VIDEO_SCALE:-2}
# 1 = record from twin sensors at the recording rate (one extra render per
# recorded frame). 0 = hold the frame the policy was given.
EVAL_SMOOTH_VIDEO=${EVAL_SMOOTH_VIDEO:-1}
case "$EVAL_SMOOTH_VIDEO" in
  0|1) ;;
  *) echo "[eval-runner] EVAL_SMOOTH_VIDEO=$EVAL_SMOOTH_VIDEO; expected 0 or 1" >&2; exit 2 ;;
esac
if [ "$EVAL_SMOOTH_VIDEO" = 1 ]; then SMOOTH_VIDEO=true; else SMOOTH_VIDEO=false; fi
# follow: the scene camera tracks the robots from EVAL_FOLLOW_DISTANCE metres.
# fixed: the dataset pose. When the policy observes the scene view, follow needs
# the twin sensors (EVAL_SMOOTH_VIDEO=1), so the default falls back to fixed.
case ",$(printf '%s' "$EVAL_OBS_CAMERAS" | tr ':+ \t' ',,,,')," in
  *,scene,*) SCENE_IN_OBS=1 ;;
  *) SCENE_IN_OBS=0 ;;
esac
if [ "$SCENE_IN_OBS" = 1 ] && [ "$EVAL_SMOOTH_VIDEO" = 0 ]; then SCENE_CAM=fixed; fi
EVAL_SCENE_CAM=${EVAL_SCENE_CAM:-$SCENE_CAM}
EVAL_FOLLOW_DISTANCE=${EVAL_FOLLOW_DISTANCE:-2.0}
# Maximum distance (metres) the follow camera moves from the authored pose.
EVAL_FOLLOW_MAX_DOLLY=${EVAL_FOLLOW_MAX_DOLLY:-1.5}
case "$EVAL_SCENE_CAM" in
  follow|fixed) ;;
  *) echo "[eval-runner] EVAL_SCENE_CAM=$EVAL_SCENE_CAM; expected follow or fixed" >&2; exit 2 ;;
esac
POLICY_NAME=${POLICY_NAME:-$POLICY}
EVAL_EXEC_HORIZON=${EVAL_EXEC_HORIZON:-}
# Which saved checkpoint to serve. Empty = deploy.yml's checkpoint_num.
EVAL_CKPT_NUM=${EVAL_CKPT_NUM:-}
EVAL_EPISODES_PER_PROC=${EVAL_EPISODES_PER_PROC:-1}

# ---- rendering ----
# Steps one observation is reused for. Empty = the policy's own OBS_STRIDE.
EVAL_OBS_STRIDE=${EVAL_OBS_STRIDE:-}
# Record one frame every N steps; the video fps is divided to match.
EVAL_VIDEO_STRIDE=${EVAL_VIDEO_STRIDE:-2}
EVAL_EGO_DEPTH=${EVAL_EGO_DEPTH:-true}
EVAL_DROP_UNUSED_CAMERAS=${EVAL_DROP_UNUSED_CAMERAS:-false}
EVAL_VIDEO=${EVAL_VIDEO:-1}

# Every episode runs in this process. aggregate.py folds .episodes/shard*/.
EP_LO=0
EP_HI=$EPISODES
# The results directory names what was evaluated. The checkpoint seed, policy
# salt and step budget appear only when they are not the default.
CKPT_SEED_TAG=""
[ "$CKPT_SEED" = 0 ] || CKPT_SEED_TAG="-ckptseed${CKPT_SEED}"
POLICY_SEED_TAG=""
[ "$POLICY_SEED_SALT" = 0 ] || POLICY_SEED_TAG="-policysalt${POLICY_SEED_SALT}"
STEPS_TAG=""
[ "$STEP_LIMIT" = "$DEF_STEP_LIMIT" ] || STEPS_TAG="-steps${STEP_LIMIT}"
RUN_ID=${EVAL_RUN_ID:-${POLICY_NAME}-${COHUB_MODE}${CKPT_TAG:+-${CKPT_TAG}}-seed${SEED}${CKPT_SEED_TAG}${POLICY_SEED_TAG}${EVAL_EXEC_HORIZON:+-exec${EVAL_EXEC_HORIZON}}${EVAL_CKPT_NUM:+-ckpt${EVAL_CKPT_NUM}}${STEPS_TAG}}
# Results are filed by scene name. COHUB_EVAL_LABEL files them under
# eval_results_<label>/.
OUT_DIR=$WT/eval_results${COHUB_EVAL_LABEL:+_${COHUB_EVAL_LABEL}}/${ENV_CFG_TYPE}/${RUN_ID}
RESULTS=$OUT_DIR/results.json
EP_ROOT=$OUT_DIR/.episodes
EP_DIR=$EP_ROOT/shard0
VIDEO_ROOT=$OUT_DIR/videos
STATUS_TSV=$EP_DIR/client_status.tsv
VRAM_CSV=$OUT_DIR/.gpu_samples.csv   # diagnostics, kept only if EVAL_KEEP_GPU_CSV=1
VIDEO_ARGS=""
if [ "$EVAL_VIDEO" = 1 ]; then
  VIDEO_ARGS="--video_dir '$VIDEO_ROOT' --video_cameras '$EVAL_VIDEO_CAMERAS' --video_stride '$EVAL_VIDEO_STRIDE' --video_scale '$EVAL_VIDEO_SCALE'"
fi
RENDER_ARGS="--obs_cameras '$EVAL_OBS_CAMERAS' --ego_depth '$EVAL_EGO_DEPTH' --drop_unused_cameras '$EVAL_DROP_UNUSED_CAMERAS'"
RENDER_ARGS="$RENDER_ARGS --scene_camera '$EVAL_SCENE_CAM' --follow_distance '$EVAL_FOLLOW_DISTANCE'"
RENDER_ARGS="$RENDER_ARGS --follow_max_dolly '$EVAL_FOLLOW_MAX_DOLLY' --smooth_video '$SMOOTH_VIDEO'"
[ -n "$EVAL_OBS_STRIDE" ] && RENDER_ARGS="$RENDER_ARGS --obs_stride '$EVAL_OBS_STRIDE'"

# Set before the dry run: a hook's run_names may look there.
CKPT_DIR=$POLICY_DIR/checkpoints
SIM_GPU=${COHUB_SIM_GPU:-${CUDA_VISIBLE_DEVICES:-0}}; SIM_GPU=${SIM_GPU%%,*}
POLICY_GPU=${COHUB_POLICY_GPU:-$SIM_GPU}

# EVAL_DRY_RUN=1 resolves everything and stops before anything needs a GPU, a
# free port or a checkpoint.
if [ "${EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "[dry-run] policy=$POLICY hook=$HOOK python=$POLICY_PY"
  echo "[dry-run] task=$TASK task_word=$TASK_WORD env_cfg=$ENV_CFG_TYPE server_env_cfg=$SERVER_ENV_CFG_TYPE mode=$COHUB_MODE"
  echo "[dry-run] decentralized_style=$DECENTRALIZED_STYLE ckpt_name=$CKPT_NAME"
  echo "[dry-run] action_type=$ACTION_TYPE"
  echo "[dry-run] server_timeout_s=${EVAL_SERVER_TIMEOUT:-$SERVER_TIMEOUT_S} rpc_timeout_s=${EVAL_RPC_TIMEOUT_S:-${RPC_TIMEOUT_S:-client-default}}"
  echo "[dry-run] episodes=$EPISODES per_proc=$EVAL_EPISODES_PER_PROC seed=$SEED ckpt_seed=$CKPT_SEED policy_seed_salt=$POLICY_SEED_SALT server_seed=$SEED_SERVER"
  echo "[dry-run] out=$OUT_DIR ep_dir=$EP_DIR model_dir_key=$MODEL_DIR_KEY"
  echo "[dry-run] sim_gpu=$SIM_GPU policy_gpu=$POLICY_GPU"
  # The run directories this eval would serve.
  echo "[dry-run] runs=$(run_names | cut -d: -f1 | paste -sd, -)"
  echo "[dry-run] client: --episode_step_limit $STEP_LIMIT $VIDEO_ARGS $RENDER_ARGS"
  exit 0
fi

mkdir -p "$EP_DIR" "$VIDEO_ROOT"
# A rerun replaces the previous results.
rm -rf "${EP_DIR:?}"/*
[ -n "${COHUB_EVAL_CLIENT:-}" ] || rm -rf "${VIDEO_ROOT:?}"/*
: > "$STATUS_TSV"

# ---- served checkpoints ----
# One "<deploy.yml key>=<dir>" server override per run the hook names. A run the
# hook cannot resolve or guards is left to the policy's own resolution.
MODEL_DIR_ARGS=()
mapfile -t runs < <(run_names)
for entry in "${runs[@]}"; do
  run=${entry%:*}; key=${entry#*:}
  src=$(resolve_model_dir "$run")
  if [ -z "$src" ]; then
    echo "[eval-runner] no model dir for $run -- leaving resolution to the policy" >&2
    continue
  fi
  model_dir_guard "$run" "$src" && continue
  MODEL_DIR_ARGS+=("${key}=$src")
done

PORT=$(bash "$POLICIES/utils/get_free_port.sh")
echo "[eval-runner] task=$TASK_WORD env_cfg=$ENV_CFG_TYPE server_env_cfg=$SERVER_ENV_CFG_TYPE mode=$COHUB_MODE/$DECENTRALIZED_STYLE ckpt=$CKPT_NAME tag=${CKPT_TAG:-none} episodes=$EPISODES seed=$SEED ckpt_seed=$CKPT_SEED policy_seed_salt=$POLICY_SEED_SALT server_seed=$SEED_SERVER port=$PORT step_limit=$STEP_LIMIT exec_horizon=${EVAL_EXEC_HORIZON:-deploy.yml} action_type=$ACTION_TYPE"
# Host and GPU state at the start, for diagnosing CUDA failures.
echo "[eval-runner] host=$(hostname -s) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} obs_codec=${COHUB_OBS_CODEC:-pyav}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/[eval-runner]   gpu: /' || echo "[eval-runner]   gpu: nvidia-smi failed"
# Fail early when CUDA is unusable on this machine.
CUDA_PROBE=$("$SIM_PY" -c "
import torch
torch.zeros(1, device='cuda').sum().item()
print('ok')
" 2>&1) || true
case "$CUDA_PROBE" in
  *ok*) echo "[eval-runner] cuda probe: a context opens on the granted GPU" ;;
  *)    echo "[eval-runner] FATAL: CUDA is unusable on $(hostname -s); nothing was evaluated." >&2
        printf '%s\n' "$CUDA_PROBE" | tail -3 | sed 's/^/[eval-runner]   /' >&2
        echo "[eval-runner] This is the node, not the policy -- report the host above." >&2
        exit 3 ;;
esac
echo "[eval-runner] results -> $OUT_DIR"

# GPU memory/utilization sampler.
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 10 > "$VRAM_CSV" 2>/dev/null &
SAMPLER_PID=$!

cleanup() {
  # stop_server is defined further down.
  declare -F stop_server >/dev/null && stop_server || kill "$SERVER_PID" 2>/dev/null || true
  kill "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- provenance: written into every results file and into results.json ----
prov_sha()   { git -C "$1" rev-parse --short=12 HEAD 2>/dev/null || echo unknown; }
prov_dirty() { git -C "$1" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' '; }
COHUB_EVAL_PROVENANCE=$(python3 -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:])))' \
  "cohub_sha=$(prov_sha "$WT")" "cohub_dirty_files=$(prov_dirty "$WT")" \
  "baselines_sha=$(prov_sha "$POLICIES")" "baselines_dirty_files=$(prov_dirty "$POLICIES")" \
  "runner_host=$(hostname -s)" "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n 1p)" \
  ${EVAL_PROVENANCE_EXTRA:-} \
  "policy=$POLICY" "task=$TASK_WORD" "env_cfg=$ENV_CFG_TYPE" "mode=$COHUB_MODE" "decentralized_style=$DECENTRALIZED_STYLE" \
  "ckpt_name=$CKPT_NAME" "ckpt_tag=${CKPT_TAG:-}" "ckpt_num=${EVAL_CKPT_NUM:-}" "ckpt_seed=$CKPT_SEED" \
  "runs=$(run_names | cut -d: -f1 | paste -sd, -)" "model_dirs=$(printf '%s,' ${MODEL_DIR_ARGS[@]+"${MODEL_DIR_ARGS[@]}"})" \
  "seed=$SEED" "policy_seed_schedule=$POLICY_SEED_SCHEDULE" "policy_seed_salt=$POLICY_SEED_SALT" "episodes=$EPISODES" "episodes_per_proc=$EVAL_EPISODES_PER_PROC" \
  "server_seeded=$SEED_SERVER" "exec_horizon=${EVAL_EXEC_HORIZON:-}" "sampler=${EVAL_SAMPLER:-}" "inference_steps=${EVAL_INFERENCE_STEPS:-}" \
  "step_limit=$STEP_LIMIT" "isaac_assets=$ISAAC_ASSETS")
export COHUB_EVAL_PROVENANCE

# ---- policy server: one process for the whole run ----
server_env

start_server() {                # -> $SERVER_PID serving on $PORT
  local log="$OUT_DIR/server.log"
  # PYTHONPATH: the server imports the adapter from this checkout, not from an
  # editable install of another one.
  # PYTHONUNBUFFERED: server.log shows progress as it happens.
  # eval_seed seeds the server; `seed` stays the checkpoint's.
  # setsid: the server gets its own process group, so stop_server reaches its children.
  COHUB_SEED_SERVER="$SEED_SERVER" \
  CUDA_VISIBLE_DEVICES="$POLICY_GPU" PYTHONUNBUFFERED=1 PYTHONPATH="${EVAL_PYTHONPATH:+$EVAL_PYTHONPATH:}$WT${PYTHONPATH:+:$PYTHONPATH}" \
    setsid "$POLICY_PY" "$POLICIES/client_server/setup_policy_server.py" \
    --config_path "$POLICY_DIR/deploy.yml" \
    --overrides \
      port="$PORT" host=localhost \
      bench_name=cohub task_name="$TASK_WORD" ckpt_name="$CKPT_NAME" \
      env_cfg_type="$SERVER_ENV_CFG_TYPE" action_type="$ACTION_TYPE" seed="$CKPT_SEED" eval_seed="$SEED" policy_seed_salt="$POLICY_SEED_SALT" \
      policy_name="$POLICY" cohub_mode="$COHUB_MODE" ckpt_tag="$CKPT_TAG" \
      cohub_decentralized_style="$DECENTRALIZED_STYLE" \
      ${EVAL_EXEC_HORIZON:+exec_horizon=$EVAL_EXEC_HORIZON} \
      ${EVAL_CKPT_NUM:+checkpoint_num=$EVAL_CKPT_NUM} \
      ${EVAL_SAMPLER:+inference_scheduler=$EVAL_SAMPLER} \
      ${EVAL_INFERENCE_STEPS:+num_inference_steps=$EVAL_INFERENCE_STEPS} \
      ${EVAL_ACTION_CLIP:+action_clip=$EVAL_ACTION_CLIP} \
      ${EVAL_OBS_CLIP:+obs_clip=$EVAL_OBS_CLIP} \
      ${EVAL_SERVER_OVERRIDES:-} \
      ${MODEL_DIR_ARGS[@]+"${MODEL_DIR_ARGS[@]}"} \
    > "$log" 2>&1 &
  SERVER_PID=$!

  # The timeout is a ceiling, not a sleep.
  bash "$POLICIES/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER_PID" "$POLICY policy server" "${EVAL_SERVER_TIMEOUT:-$SERVER_TIMEOUT_S}"

  # The server and Isaac's renderer share one card. With too little memory left
  # Isaac renders undenoised frames without failing, so stop here.
  # sed -n 1p, not head -1: head closes the pipe early, which fails under pipefail.
  if command -v nvidia-smi > /dev/null 2>&1; then
    # The client's card; the first listed when nvidia-smi does not know it by that index.
    free_mb=$({ nvidia-smi -i "$SIM_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
                || nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null; } | sed -n 1p | tr -d ' ') || true
    min_free=${EVAL_MIN_FREE_VRAM_MB:-6144}
    if [ -n "$free_mb" ] && [ "$free_mb" -lt "$min_free" ] 2>/dev/null; then
      echo "[eval-runner] only ${free_mb} MiB of GPU memory left after the $POLICY server started;" >&2
      echo "[eval-runner] Isaac needs about ${min_free} and will render undenoised speckle below it." >&2
      echo "[eval-runner] cap the server's share (pi0.5: XLA_PYTHON_CLIENT_PREALLOCATE/MEM_FRACTION)" >&2
      echo "[eval-runner] or give the run a bigger card. EVAL_MIN_FREE_VRAM_MB overrides this floor." >&2
      stop_server
      exit 3
    fi
    echo "[eval-runner] ${free_mb} MiB of GPU memory left for Isaac (floor ${min_free})"
  fi
}

stop_server() {                 # tear down $SERVER_PID and everything it started
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -0 "$SERVER_PID" 2>/dev/null || { SERVER_PID=""; return 0; }
  # Signal the whole process group, so the server's children exit too.
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  local waited=0
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep 1; waited=$((waited + 1))
    [ "$waited" -lt 30 ] || { kill -KILL -- "-$SERVER_PID" 2>/dev/null || true; break; }
  done
  SERVER_PID=""
}

start_server

# ---- Isaac Sim env client: one fresh process per episode ----
# Without /usr/share/nvidia/nvoptix.bin the OptiX denoiser is off and the policy
# sees noisy frames.
[ -f /usr/share/nvidia/nvoptix.bin ] || echo "[eval-runner] warning: /usr/share/nvidia/nvoptix.bin is missing -- frames will not be denoised" >&2

# COHUB_EVAL_CLIENT: a client script run once against the server in place of
# the episode loop. It takes the loop's arguments and COHUB_EVAL_CLIENT_ARGS,
# writes results.json itself and resumes it when run again. Nothing is
# aggregated or uploaded.
if [ -n "${COHUB_EVAL_CLIENT:-}" ]; then
  rm -rf "$EP_ROOT"
  for ((attempt = 0; attempt <= ${EVAL_CLIENT_RETRIES:-2}; attempt++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    rc=0
    (
      cd "$WT"
      export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
      export PYTHONPATH="$WT/source:$WT/external/IsaacLab/source/isaaclab_teleop"
      export EVAL_RPC_TIMEOUT_S="${EVAL_RPC_TIMEOUT_S:-$RPC_TIMEOUT_S}"
      sim_gpu=${COHUB_SIM_GPU:-${CUDA_VISIBLE_DEVICES:-0}}; export CUDA_VISIBLE_DEVICES=${sim_gpu%%,*}
      eval exec "'$SIM_PY'" "'$COHUB_EVAL_CLIENT'" \
        --bench_name cohub --task_name "'$TASK_WORD'" \
        --env_cfg_type "'$ENV_CFG_TYPE'" --policy_name "'$POLICY'" \
        --host localhost --port "'$PORT'" --root_dir "'$WT/baselines'" \
        --device cpu --seed "'$SEED'" \
        --results_path "'$RESULTS'" "$VIDEO_ARGS" "$RENDER_ARGS" "${COHUB_EVAL_CLIENT_ARGS:-}"
    ) || rc=$?
    [ "$rc" != 0 ] || break
    echo "[eval-runner] the client stopped (rc=$rc); it resumes from $RESULTS"
  done
  exit "$rc"
fi

missing=0
for ((i = EP_LO; i < EP_HI; i += EVAL_EPISODES_PER_PROC)); do
  chunk=$((EP_HI - i < EVAL_EPISODES_PER_PROC ? EP_HI - i : EVAL_EPISODES_PER_PROC))
  printf -v idx '%03d' "$i"
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[eval-runner] policy server died before episode $idx; stopping the loop"
    break
  fi
  ep_json=$EP_DIR/ep${idx}.json
  echo "[eval-runner] === episodes $idx..$((i + chunk - 1)) ($((i + 1))-$((i + chunk))/$EPISODES), seed $((SEED + i)) ==="
  ep_t0=$SECONDS
  # A client crash says nothing about the policy: the chunk is replayed on the
  # same seeds up to EVAL_CLIENT_RETRIES times before it counts as a client_crash.
  attempt=0
  while :; do
  rm -f "$ep_json"
  rc=0
  # The client starts from $WT. $VIDEO_ARGS/$RENDER_ARGS carry their own
  # quoting, hence eval.
  (
    cd "$WT"
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
    export PYTHONPATH="$WT/source:$WT/external/IsaacLab/source/isaaclab_teleop"
    export EVAL_RPC_TIMEOUT_S="${EVAL_RPC_TIMEOUT_S:-$RPC_TIMEOUT_S}"
    export CUDA_VISIBLE_DEVICES=$SIM_GPU
    eval exec "'$SIM_PY'" scripts/eval/eval_policy_client.py \
      --bench_name cohub --task_name "'$TASK_WORD'" \
      --env_cfg_type "'$ENV_CFG_TYPE'" --policy_name "'$POLICY'" \
      --host localhost --port "'$PORT'" --root_dir "'$WT/baselines'" \
      --device cpu --seed "'$((SEED + i))'" \
      --eval_episode_num "'$chunk'" --episode_step_limit "'$STEP_LIMIT'" \
      --results_path "'$ep_json'" "$VIDEO_ARGS" "$RENDER_ARGS"
  ) || rc=$?
  written=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("episodes") or []))' "$ep_json" 2>/dev/null || echo 0)
  [ "$written" -ge "$chunk" ] && break
  [ "$attempt" -ge "${EVAL_CLIENT_RETRIES:-2}" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || break
  attempt=$((attempt + 1))
  echo "[eval-runner] episodes $idx..$((i + chunk - 1)) crashed (rc=$rc, $written/$chunk written); retry $attempt/${EVAL_CLIENT_RETRIES:-2}"
  done
  printf '%s\t%s\t%s\t%s\n' "$idx" "$rc" "$((SECONDS - ep_t0))" "$chunk" >> "$STATUS_TSV"
  echo "[eval-runner] episodes $idx..$((i + chunk - 1)) done rc=$rc (${SECONDS}s elapsed total)"

  # A client that dies inside Isaac can still exit 0, so a missing results file
  # is the failure signal.
  if [ ! -f "$ep_json" ]; then
    missing=$((missing + 1))
    if [ "$missing" -ge "${EVAL_MAX_MISSING:-3}" ]; then
      echo "[eval-runner] $missing client processes in a row wrote no results; stopping." >&2
      echo "[eval-runner] check the client log above -- scene assets and the policy server are the usual causes." >&2
      break
    fi
  else
    missing=0
  fi
done

# ---- aggregate per-episode JSONs into the run summary, then wandb ----
python3 "$WT/scripts/eval/aggregate.py" "$RESULTS" "$EP_ROOT"
rm -rf "$EP_ROOT"   # folded into results.json
# Every episode a client crash: nothing was evaluated.
evaluated=$(python3 -c 'import json,sys; print(sum(not str(e.get("reason", "")).startswith("client_crash") for e in json.load(open(sys.argv[1]))["episodes"]))' "$RESULTS" 2>/dev/null || echo 0)
if [ "$evaluated" = 0 ]; then
  echo "[eval-runner] every episode crashed in the Isaac client; nothing was evaluated -- see the client's output above" >&2
  exit 1
fi
# A failed upload does not fail the evaluation; logging/log.sh redoes it.
if [ "${EVAL_WANDB:-1}" = 1 ] && [ "$WANDB_UPLOAD" = 1 ]; then
  cohub_upload "$POLICY" "$OUT_DIR" "$EPISODES" || echo "[eval-runner] wandb upload failed; retry with baselines/logging/log.sh $POLICY ${OUT_DIR#$WT/}" >&2
elif [ "$WANDB_UPLOAD" != 1 ]; then
  echo "[eval-runner] wandb: skipped -- $POLICY's training run has no id to log the rollout into"
fi
[ "${EVAL_KEEP_GPU_CSV:-0}" = 1 ] || rm -f "$VRAM_CSV"
echo "[eval-runner] done"
echo "[eval-runner]   results: $RESULTS"
echo "[eval-runner]   videos : $VIDEO_ROOT/{$(echo "$EVAL_VIDEO_CAMERAS" | tr ':+ ' ',,,')}/"
