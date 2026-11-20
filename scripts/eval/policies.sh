# What differs per policy when it is served: sourced by runner.sh, which calls serve_<POLICY>.

serve_ACT() {
# The benchmark's ACT setting is one checkpoint per robot; a centralized policy is not served.
[ "${COHUB_MODE:-decentralized}" = decentralized ] || {
  echo "[eval] ACT serves the decentralized setting only (one checkpoint per robot); COHUB_MODE=$COHUB_MODE is not a CoHuB setting" >&2
  return 2; }

POLICY_PY=$(cohub_policy_python ACT)

# The server gets the mode's profile as env_cfg_type. It names the checkpoint
# run directory and selects the two-checkpoint mode.
SERVER_ENV_CFG=profile

# ACT takes no language input, so it serves one checkpoint per (task, robot).
DECENTRALIZED_STYLE=per_robot

# ACT's centralized path reads ckpt_dir, not model_dir.
MODEL_DIR_KEY=ckpt_dir

# The model is sized from ACT_ACTION_DIM / ACT_STATE_DIM, for the served profile.
server_env() {
  local profile=${SERVER_ENV_CFG_TYPE:-unitree_g1x2_decentralized}
  ACT_ACTION_DIM=$(bash "$POLICIES/utils/robot_info.sh" "$WT/baselines" "$profile" action_dim) || return 2
  ACT_STATE_DIM=$(bash "$POLICIES/utils/robot_info.sh" "$WT/baselines" "$profile" state_dim) || return 2
  export ACT_ACTION_DIM ACT_STATE_DIM
  # detr/act_policy.py also reads the dimension off the server config.
  EVAL_SERVER_OVERRIDES="action_dim=$ACT_ACTION_DIM${EVAL_SERVER_OVERRIDES:+ $EVAL_SERVER_OVERRIDES}"
}
}

serve_DP() {
# The benchmark's DP setting is one checkpoint per robot; a centralized policy is not served.
[ "${COHUB_MODE:-decentralized}" = decentralized ] || {
  echo "[eval] DP serves the decentralized setting only (one checkpoint per robot); COHUB_MODE=$COHUB_MODE is not a CoHuB setting" >&2
  return 2; }

POLICY_PY=$(cohub_policy_python DP)

# The server gets the mode's profile as env_cfg_type. It names the checkpoint
# run directory and selects the two-checkpoint mode.
SERVER_ENV_CFG=profile

# DP takes no language input, so it serves one checkpoint per (task, robot).
DECENTRALIZED_STYLE=per_robot

}

serve_LatentToM() {
cohub_two_robot_only LatentToM "$TASK" || return 2
# Both arms are saved in one run directory (checkpoints/{arm1,arm2}_latest.ckpt),
# so there is one directory to serve.

POLICY_PY=$(cohub_policy_python LatentToM)

ACTION_TYPE=joint

# The server's env_cfg_type stays the scene name.

run_names() {
  # One entry regardless of COHUB_MODE. ENV_CFG_TYPE is what train.sh writes.
  local ckpt="${COHUB_LATENTTOM_CKPT:-$ENV_CFG_TYPE}"
  echo "cohub-${ckpt}-${ENV_CFG_TYPE}-joint-${CKPT_SEED}${CKPT_TAG:+-${CKPT_TAG}}:model_dir"
}
}

serve_GauDP() {
# GauDP reads both robots' 86D joint state and the two ego views, and predicts
# the pair's 70D absolute joint-target action.

POLICY_PY=$(cohub_policy_python GauDP)

ACTION_TYPE=joint

# One policy drives both robots, so there is no decentralized mode.
COHUB_MODE=centralized

# The server gets the training profile as env_cfg_type.
SERVER_ENV_CFG=profile

# Prefer the standard run name. For handover, fall back to the handover_easy spelling.
run_names() {
  local canonical legacy
  canonical=$(mh_run_name "$CKPT_NAME" "$SERVER_ENV_CFG_TYPE" joint "$CKPT_SEED" "$CKPT_TAG")
  if [ -s "$POLICY_DIR/checkpoints/$canonical/policy/best.ckpt" ] || \
     [ -s "$POLICY_DIR/checkpoints/$canonical/policy/last.ckpt" ]; then
    echo "$canonical:${MODEL_DIR_KEY}"
    return
  fi
  if [ "$CKPT_NAME" = handover ]; then
    legacy=$(mh_run_name handover_easy "$SERVER_ENV_CFG_TYPE" joint "$CKPT_SEED" "$CKPT_TAG")
    if [ -s "$POLICY_DIR/checkpoints/$legacy/policy/best.ckpt" ] || \
       [ -s "$POLICY_DIR/checkpoints/$legacy/policy/last.ckpt" ]; then
      echo "$legacy:${MODEL_DIR_KEY}"
      return
    fi
  fi
  echo "$canonical:${MODEL_DIR_KEY}"
}
}

serve_Pi_05() {
POLICY_PY=$(cohub_policy_python Pi_05)

server_env() {
  # Read before sourcing: policy/Pi_05/common.sh sets its own training default.
  local requested=${XLA_PYTHON_CLIENT_MEM_FRACTION:-}

  # Sets the dataset path and OPENPI_DATA_HOME, which the server needs.
  source "$WT/baselines/policy/Pi_05/common.sh"

  # Allocate GPU memory on demand, so Isaac keeps its share of the card.
  export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
  # A ceiling, used only when preallocation is on.
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${requested:-0.5}
}

resolve_model_dir() {                 # <run-name> -> step EVAL_CKPT_NUM, else the newest
  # openpi writes checkpoints/<run>/<step>/, one directory per saved step.
  local run=$1
  local base="$CKPT_DIR/$run"
  local step
  [ -d "$base" ] || return 0
  step=$(ls -1 "$base" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  # openpi saves a run's last step as N-1 (39999 of 40k).
  if [ -n "${EVAL_CKPT_NUM:-}" ]; then
    case "$EVAL_CKPT_NUM" in
      *[!0-9]*) echo "[eval-runner] EVAL_CKPT_NUM=$EVAL_CKPT_NUM is not a training step" >&2; return 1 ;;
    esac
    [ -d "$base/$EVAL_CKPT_NUM" ] && { echo "$base/$EVAL_CKPT_NUM"; return 0; }
    [ "$step" = "$((10#$EVAL_CKPT_NUM - 1))" ] && { echo "$base/$step"; return 0; }
    echo "[eval-runner] $run has no step $EVAL_CKPT_NUM under $base (newest: ${step:-none})" >&2
    return 1
  fi
  [ -n "$step" ] && echo "$base/$step"
  return 0
}
}

serve_GR00T_N17() {
POLICY_PY=$(cohub_policy_python GR00T_N17)

resolve_model_dir() {                 # <run-name> -> merged-<N> for EVAL_CKPT_NUM, else newest merged-* / checkpoint-*
  local run=$1
  local base="$CKPT_DIR/$run/$run"
  [ -d "$base" ] || base="$CKPT_DIR/$run"
  local dir
  # A requested step that is missing stops the evaluation.
  if [ -n "${EVAL_CKPT_NUM:-}" ]; then
    for dir in "$base/merged-$EVAL_CKPT_NUM" "$base/checkpoint-$EVAL_CKPT_NUM"; do
      [ -d "$dir" ] && { echo "$dir"; return 0; }
    done
    echo "[eval-runner] $run has no merged-$EVAL_CKPT_NUM or checkpoint-$EVAL_CKPT_NUM under $base" >&2
    return 1
  fi
  dir=$(ls -d "$base"/merged-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [ -n "$dir" ] || dir=$(ls -d "$base"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  echo "$dir"
}

model_dir_guard() {                   # <run> <src> -- 0 = do not hand this run's dir to the server
  # An unmerged LoRA checkpoint holds adapters, not a model. Leave it to the policy.
  if [ -f "$2/adapter_config.json" ]; then
    echo "[eval-runner] $1 has only LoRA adapters -- leaving resolution to the policy" >&2
    return 0
  fi
  return 1
}
}

serve_Psi0() {
POLICY_PY=$(cohub_policy_python Psi0)
DECENTRALIZED_STYLE=shared

# SERVER_ENV_CFG stays scene: policy/Psi0/model.py substitutes the training
# profile itself.

resolve_model_dir() {                 # <run-name> -> the Psi0 run directory
  # The run directory is the one holding run_config.json, up to two levels
  # below checkpoints/<run>.
  local run=$1 base="$CKPT_DIR/$run" found
  [ -d "$base" ] || return 0
  if [ -f "$base/run_config.json" ]; then echo "$base"; return 0; fi
  found=$(ls -d "$base"/*/*/run_config.json "$base"/*/run_config.json 2>/dev/null | sort | tail -1)
  [ -n "$found" ] && dirname "$found"
  return 0
}

}

serve_FastWAM() {
# The adapter loads one weights file per saved step:
# checkpoints/<run>/checkpoints/weights/step_XXXXXX.pt.

POLICY_PY=$(cohub_policy_python FastWAM)

# The server gets the training profile as env_cfg_type, not the scene.
SERVER_ENV_CFG=profile

resolve_model_dir() {                 # <run-name> -> the servable weights file
  local base="$CKPT_DIR/$1/checkpoints/weights"
  [ -d "$base" ] || return 0
  if [ -n "${EVAL_CKPT_NUM:-}" ] && [ "$EVAL_CKPT_NUM" != latest ]; then
    printf '%s/step_%06d.pt\n' "$base" "$EVAL_CKPT_NUM"
  else
    ls -1 "$base"/step_*.pt 2>/dev/null | sort | tail -1
  fi
  return 0
}

server_env() {
  # The serving model loads the Wan2.2 text encoder and VAE from this path.
  export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-$POLICIES/policy/FastWAM/FastWAM/checkpoints}
}
}
