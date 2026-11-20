# Shared naming and derivation rules for the baselines train/eval runners.
# Sourced, never executed.
#
# mh_task_word also accepts the older spellings frame_hang, movehouse_g1x3 and
# bigtable_g1x3 and maps them to the current task word.

mh_wt() {                 # -> the checkout: the caller's COHUB_WT, else this file's repo
  if [ -n "${COHUB_WT:-}" ]; then echo "$COHUB_WT"; else (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); fi
}

# Site values (asset mirror, caches, interpreters, model snapshots) come from
# baselines/site.env when that file exists. A variable already set in the
# environment wins.
_mh_site_env=${COHUB_SITE_ENV:-${WT:-$(mh_wt)}/baselines/site.env}
if [ -f "$_mh_site_env" ]; then
  # shellcheck disable=SC1090
  source "$_mh_site_env"
fi

# The pretrained snapshots GR00T and Psi0 start from.
# `baselines/policy/<POLICY>/install.sh` fetches a missing one.
_mh_policy_dir=${WT:-$(mh_wt)}/baselines/policy
export GR00T_BASE_MODEL=${GR00T_BASE_MODEL:-$_mh_policy_dir/GR00T_N17/weights/GR00T-N1.7-3B}
export GR00T_COSMOS_MODEL=${GR00T_COSMOS_MODEL:-$_mh_policy_dir/GR00T_N17/weights/Cosmos-Reason2-2B}
export PSI0_BASE_MODEL=${PSI0_BASE_MODEL:-$_mh_policy_dir/Psi0/weights/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k}
export PSI0_ACTION_HEADER=${PSI0_ACTION_HEADER:-$_mh_policy_dir/Psi0/weights/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k}

mh_load_recipe() {        # <file.yaml> -- export its `KEY: value` lines, environment first
  # Flat `KEY: value` lines. A variable already set in the shell wins.
  local file=$1 key value
  [ -f "$file" ] || { echo "[recipe] no such file: $file" >&2; return 1; }
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}                      # drop comments, then surrounding blanks
    line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}
    [ -z "$line" ] && continue
    case "$line" in *:*) ;; *) echo "[recipe] $file: not a KEY: value line: $line" >&2; return 1 ;; esac
    key=${line%%:*}; value=${line#*:}
    key=${key%"${key##*[![:space:]]}"}; value=${value#"${value%%[![:space:]]*}"}
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
      || { echo "[recipe] $file: not an environment variable name: $key" >&2; return 1; }
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}



mh_find_isaac_assets() {  # -> the local mirror of the Isaac asset tree
  local wt c
  wt=${WT:-$(mh_wt)}
  for c in "${ISAAC_ASSETS:-}" "$wt/isaac_assets/Isaac"; do
    [ -n "$c" ] && [ -d "$c" ] && { echo "$c"; return 0; }
  done
  echo "${ISAAC_ASSETS:-$wt/isaac_assets/Isaac}"
}

# The task table is read from source/cohub/task_registry.py.
eval "$(python3 - "$(mh_wt)/source/cohub/task_registry.py" <<'REGISTRY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
rows = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "TASKS")).values()
print("COHUB_TASK_WORDS=%r" % " ".join(dict.fromkeys(r["slug"] for r in rows)))
print("MH_ALL_TASKS=%r" % " ".join(r["slug"] for r in rows if r["multitask"]))
print("MH_ALL3_TASKS=%r" % " ".join(r["slug"] for r in rows if r["robots"] == 3))
print("COHUB_STEP_LIMITS=%r" % " ".join("%s=%d" % (r["scene"], r["step_limit"]) for r in rows))
REGISTRY
)"
export COHUB_TASK_WORDS MH_ALL_TASKS MH_ALL3_TASKS COHUB_STEP_LIMITS

mh_task_word() {          # <task> -> canonical task word (== slug == scene)
  local word=$1
  case "$word" in
    # older spellings
    frame_hang)      word=framehang ;;
    movehouse_g1x3)  word=movehouse ;;
    bigtable_g1x3)   word=bigtable ;;
    all|multitask)   echo all;  return 0 ;;
    all3|multitask3) echo all3; return 0 ;;
  esac
  case " $COHUB_TASK_WORDS " in
    *" $word "*) echo "$word" ;;
    *) echo "unknown task '$1' (${COHUB_TASK_WORDS// /|}|all)" >&2; return 2 ;;
  esac
}

# `all` is the two-robot multitask set (MH_ALL_TASKS), `all3` the three-robot
# set (MH_ALL3_TASKS). Eval always names one task.

mh_scene() {              # <task> -> Isaac scene spelling (the identity map)
  # `all` is not a scene and is rejected here, unlike in mh_task_word.
  local word
  word=$(mh_task_word "$1") || return 2
  [ "$word" = all ] && { echo "'all' is a training set, not a scene" >&2; return 2; }
  echo "$word"
}

mh_robot_profile() {      # <scene> -> the robot profile its env_cfg names (g1x2, g1x3)
  # Read from config.robot in baselines/env_cfg/<scene>.yml.
  local yml
  yml=$(mh_wt)/baselines/env_cfg/$1.yml
  [ -f "$yml" ] || { echo "no baselines/env_cfg/$1.yml" >&2; return 2; }
  sed -n 's/^[[:space:]]*robot:[[:space:]]*\([^[:space:]]*\).*/\1/p' "$yml" | head -1
}

mh_extra_video_cameras() {  # <scene> -> env_cfg's extra_video_cameras (comma list), or nothing
  local yml
  yml=$(mh_wt)/baselines/env_cfg/$1.yml
  [ -f "$yml" ] || return 0
  sed -n 's/^extra_video_cameras:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$yml" | head -1
}

mh_robots() {             # <scene> -> the robot names, one per line
  local profile count
  profile=$(mh_robot_profile "$1") || return 2
  count=${profile#g1x}
  case "$count" in
    2) printf 'robot_a\nrobot_b\n' ;;
    3) printf 'robot_a\nrobot_b\nrobot_c\n' ;;
    *) echo "unknown robot profile '$profile' for scene '$1'" >&2; return 2 ;;
  esac
}

mh_env_cfg_type() {       # <centralized|decentralized> -> checkpoint profile
  # `g1x2` is the adapter profile name. It stays the same on three-robot tasks.
  case "$1" in
    centralized|decentralized) echo "unitree_g1x2_$1" ;;
    *) echo "unknown mode '$1' (centralized|decentralized)" >&2; return 2 ;;
  esac
}

mh_ckpt_name() {          # <task_word> <mode> <robot_a|robot_b|-> -> ckpt_name
  local task=$1 mode=$2 robot=${3:--}
  case "$mode" in
    centralized)
      [ "$task" = all ] && { echo "centralized trains one task at a time, not 'all'" >&2; return 2; }
      echo "$task" ;;
    decentralized)
      # The shared multitask policy takes no robot.
      if [ "$task" = all ] || [ "$task" = all3 ]; then
        [ "$robot" = - ] || { echo "shared decentralized takes no robot (got '$robot'); pass -" >&2; return 2; }
        [ "$task" = all3 ] && { echo multitask3; return 0; }
        echo multitask
        return 0
      fi
      case "$robot" in
        robot_a|robot_b|robot_c) echo "${task}_${robot}" ;;
        *) echo "decentralized needs robot_a, robot_b or robot_c (or task 'all' for the shared policy), got '$robot'" >&2; return 2 ;;
      esac ;;
    *) echo "unknown mode '$mode'" >&2; return 2 ;;
  esac
}

mh_run_name() {           # <ckpt_name> <env_cfg_type> <action_type> <seed> [tag]
  # Must match utils/checkpoint_resolver.build_run_dir_name.
  local ckpt=$1 env_cfg=$2 action=$3 seed=$4 tag=${5:-}
  # COHUB_RUN_SUFFIX stands in for everything after the checkpoint name.
  [ -z "${COHUB_RUN_SUFFIX:-}" ] || { echo "cohub-${ckpt}-${COHUB_RUN_SUFFIX}"; return; }
  echo "cohub-${ckpt}-${env_cfg}-${action}-${seed}${tag:+-${tag}}"
}
mh_data_id() {            # <ckpt_name> <env_cfg_type> -> the dataset id a policy's data lives under
  # The run name without seed and tag.
  echo "cohub-$1-$2-joint"
}

# ---- the policies ----
# Policy names, their COHUB_<TAG>_PY / COHUB_<TAG>_ENV override words and
# interpreters.
COHUB_POLICIES="ACT DP LatentToM GauDP Pi_05 GR00T_N17 Psi0 FastWAM"
cohub_policy_tag() {      # <POLICY> -> the word in its COHUB_<TAG>_PY / COHUB_<TAG>_ENV overrides
  case "$1" in
    ACT) echo ACT ;; DP) echo DP ;; LatentToM) echo LATENTTOM ;; GauDP) echo GAUDP ;;
    FastWAM) echo FASTWAM ;; Pi_05) echo PI05 ;; GR00T_N17) echo GROOT ;; Psi0) echo PSI0 ;;
    *) echo "unknown policy '$1' (one of: $COHUB_POLICIES)" >&2; return 2 ;;
  esac
}
cohub_policy_env() {      # <POLICY> -> its conda env name; nothing for the uv-managed three
  local tag var; tag=$(cohub_policy_tag "$1") || return 2; var=COHUB_${tag}_ENV
  case "$1" in
    ACT) echo "${!var:-cohub_act}" ;; DP) echo "${!var:-cohub_dp}" ;; LatentToM) echo "${!var:-latenttom}" ;;
    GauDP) echo "${!var:-gaudp}" ;; FastWAM) echo "${!var:-fastwam}" ;;
  esac
}
cohub_policy_python() {   # <POLICY> -> the interpreter its hooks run under (may not exist yet)
  local tag var env venv; tag=$(cohub_policy_tag "$1") || return 2; var=COHUB_${tag}_PY
  env=$(cohub_policy_env "$1")
  if [ -n "$env" ]; then mh_policy_python "$env" "${!var:-}"; return; fi
  case "$1" in Pi_05) venv=openpi ;; GR00T_N17) venv=gr00t_n17 ;; Psi0) venv=psi0 ;; esac
  echo "${!var:-$(mh_wt)/baselines/policy/$1/$venv/.venv/bin/python}"
}

mh_policy_python() {      # <conda env name> [explicit python] -> interpreter path
  # Order: the explicit override, then the conda installs on this machine.
  local env=$1 explicit=${2:-} base candidate
  if [ -n "$explicit" ]; then echo "$explicit"; return 0; fi
  local prefix=${CONDA_PREFIX:-}
  for base in "${CONDA_ROOT:-}" \
              "$( { command -v conda >/dev/null 2>&1 && conda info --base; } 2>/dev/null || true)" \
              "${prefix%/envs/*}" \
              "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
              /opt/conda /opt/anaconda3 /opt/miniconda3; do
    [ -n "$base" ] || continue
    candidate="$base/envs/$env/bin/python"
    [ -x "$candidate" ] && { echo "$candidate"; return 0; }
  done
  # Nothing found: return a concrete path for the caller's error message.
  echo "$HOME/anaconda3/envs/$env/bin/python"
}

mh_export_common() {      # exports every training run wants, idempotently
  local wt
  wt=$(mh_wt)
  # expandable_segments on a single GPU only. An explicit
  # PYTORCH_CUDA_ALLOC_CONF in the environment is kept.
  local n_visible
  n_visible=$(( $(printf '%s' "${CUDA_VISIBLE_DEVICES:-0}" | tr -cd , | wc -c) + 1 ))
  if [ "$n_visible" -le 1 ]; then
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  fi
  # ffprobe/ffmpeg (episode loaders) may not be on PATH.
  case ":$PATH:" in
    *":$wt/baselines/bin:"*) ;;
    *) export PATH="$wt/baselines/bin:$PATH" ;;
  esac
}



mh_step_limit() {         # <scene> -> the eval's per-episode step budget (50 Hz)
  # The registry's step_limit column. STEP_LIMIT overrides it.
  local kv
  for kv in $COHUB_STEP_LIMITS; do
    [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return 0; }
  done
  echo 1000
}


# ---- shared by the entry points ----
# A policy's knobs are in baselines/configs/<POLICY>.yaml. A variable already
# set in the shell wins. TRAIN_DRY_RUN=1 / EVAL_DRY_RUN=1 make the runners
# resolve everything and stop.

cohub_init() {              # <POLICY> -- load its config (WT is set by the caller)
  local config=$WT/baselines/configs/$1.yaml
  [ -f "$config" ] || { echo "no config for $1 at $config" >&2; exit 2; }
  mh_load_recipe "$config" || exit 1
}
cohub_boot() {              # <POLICY> -- an entry point's first line: the checkout, then its config
  WT=$(mh_wt); export COHUB_WT=$WT
  cohub_init "$1"
}
cohub_runner_boot() {       # <train|eval> <POLICY> -- both runners' prologue: POLICIES, POLICY_DIR, HOOK, the config
  POLICIES=$WT/baselines
  POLICY_DIR=$POLICIES/policy/$2
  HOOK=$POLICY_DIR/train_hook.sh; [ "$1" != eval ] || HOOK=$WT/scripts/eval/policies.sh
  [ -f "$HOOK" ] || { echo "no ${HOOK#$WT/} -- add one (see the runner's header)" >&2; exit 2; }
  mh_load_recipe "$WT/baselines/configs/$2.yaml" || exit 1
}
cohub_multitask_word() {    # -> the task word a shared run trains on: multitask, or multitask3
  case "${COHUB_TASK_SET:-all}" in
    all)  echo multitask ;;
    all3) echo multitask3 ;;
    *) echo "unknown COHUB_TASK_SET=${COHUB_TASK_SET} (all|all3)" >&2; return 2 ;;
  esac
}
cohub_task_set() {          # <task> -> the set its shared policy trains on: all, or all3
  local word; word=$(mh_task_word "$1") || return 2
  case " all3 $MH_ALL3_TASKS " in *" $word "*) echo all3 ;; *) echo all ;; esac
}
cohub_fetch_task() {        # <task> -- link its LeRobot export from the Hub when it is not there
  local wt=${WT:-$(mh_wt)} task_set
  [ -f "$wt/datasets/$1/lerobot/meta/info.json" ] || [ -f "$wt/datasets/$1/meta/info.json" ] && return 0
  task_set=$(cohub_task_set "$1") || return 2
  cohub_step env COHUB_TASK_SET="$task_set" bash "$wt/baselines/utils/prepare_data.sh" fetch
}
cohub_two_robot_only() {    # <POLICY> <task> -- refuse a three-robot task
  local task_set; task_set=$(cohub_task_set "$2") || return 2
  [ "$task_set" = all ] && return 0
  echo "[$1] $2 is a three-robot task; $1 is built for two-robot tasks only" >&2; return 2
}
cohub_tasks() {             # <task[,task...]|all> -> space-separated task words
  case "${1:-all}" in all) echo "$MH_ALL_TASKS" ;; all3) echo "$MH_ALL3_TASKS" ;; *) echo "${1//,/ }" ;; esac
}
cohub_usage() {             # the calling script's header comment, then exit 2
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2
}
cohub_budget() {            # <POLICY> -> the run's training budget as a label: b32-40k (batch, steps) or b256-e600 (batch, epochs)
  local b n
  case "$1" in
    Pi_05)      b=${PI05_BATCH_SIZE:-32};           n=${PI05_NUM_TRAIN_STEPS:-40000} ;;
    GR00T_N17|Psi0) b=${GLOBAL_BATCH_SIZE:-32};     n=${MAX_STEPS:-40000} ;;
    FastWAM)    b=${FASTWAM_GLOBAL_BATCH:-32};      n=${FASTWAM_MAX_STEPS:-40000} ;;
    # DP's label also carries its chunking (horizon, action steps).
    DP)  echo "b${DP_BATCH_SIZE:-256}-e${DP_NUM_EPOCHS:-600}${DP_HORIZON:+-h$DP_HORIZON}${DP_N_ACTION_STEPS:+-a$DP_N_ACTION_STEPS}"; return ;;
    ACT) echo "b${ACT_BATCH_SIZE:-256}-e${ACT_NUM_EPOCHS:-600}"; return ;;
    GauDP) echo "b${GAUDP_POLICY_BATCH_SIZE:-128}-e${GAUDP_POLICY_EPOCHS:-150}"; return ;;
    LatentToM) echo "b${LATENTTOM_BATCH_SIZE:-32}-e${LATENTTOM_EPOCHS:-1000}"; return ;;
    *)          b=${GLOBAL_BATCH_SIZE:-};           n=${MAX_STEPS:-} ;;
  esac
  if [ -n "$n" ] && [ $((n % 1000)) -eq 0 ]; then n="$((n / 1000))k"; fi
  echo "${b:+b$b-}$n"
}
cohub_wandb_name() {        # <ckpt name> <POLICY> <train|eval> -> the run's display name in wandb
  # <ckpt>-<budget>[-<CKPT_TAG>][-<COHUB_EVAL_LABEL>]-<train|eval>, e.g.
  # multitask-b32-40k-train. A CKPT_TAG of the form cohub<N> is left out.
  # COHUB_EVAL_LABEL applies to eval runs only.
  local budget tag=${CKPT_TAG:-}; budget=$(cohub_budget "$2")
  tag=${tag#"$budget"}; tag=${tag#-}   # DP's tag starts with its budget
  [[ -z "$tag" || "$tag" =~ ^cohub[0-9]*$ ]] && tag="" || tag="-$tag"
  [ "$3" = eval ] && tag="$tag${COHUB_EVAL_LABEL:+-$COHUB_EVAL_LABEL}"
  echo "$1-$budget$tag-$3"
}

cohub_step() {              # <command...> -- print it, then run it unless COHUB_DRY_RUN=1
  echo "[cohub] \$ $*" >&2
  [ "${COHUB_DRY_RUN:-0}" = 1 ] || "$@"
}

# ---- the steps an entry point is made of ----
# Used by policy/<POLICY>/install.sh, process_data.sh and train.sh.
cohub_say() { echo "[cohub] $*" >&2; }
# Runs also under COHUB_DRY_RUN=1.
cohub_entry_run() { cohub_say "\$ $*"; "$@"; }
cohub_is_multitask() { case "$1" in Pi_05|GR00T_N17|Psi0|FastWAM) return 0 ;; *) return 1 ;; esac; }
cohub_sim_python() { echo "${COHUB_SIM_PY:-$WT/.venv/bin/python}"; }
cohub_runner_dry() { [ "${TRAIN_DRY_RUN:-0}" = 1 ] || [ "${EVAL_DRY_RUN:-0}" = 1 ]; }
cohub_conda_bin() {         # -> a conda executable, or nothing
  local c
  c=$(command -v conda 2>/dev/null || true); [ -n "$c" ] && { echo "$c"; return; }
  for c in "${CONDA_ROOT:-}" "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda /opt/anaconda3 /opt/miniconda3; do
    [ -n "$c" ] && [ -x "$c/bin/conda" ] && { echo "$c/bin/conda"; return; }
  done
}
cohub_require_sim() {
  local py; py=$(cohub_sim_python)
  [ -x "$py" ] || [ "${COHUB_DRY_RUN:-0}" = 1 ] || { cohub_say "no simulation python at $py -- README, Installation (or set COHUB_SIM_PY)"; return 1; }
}

cohub_install() {           # <POLICY> -- its environment, then the pretrained weights it starts from
  # Builds the conda env or uv venv when the interpreter is missing, then runs
  # policy/<POLICY>/install.sh inside it (COHUB_INSTALL_IN=<POLICY>).
  local policy=$1 py conda env dir=$WT/baselines/policy/$1
  # Marks an install that did not finish: the interpreter exists before its packages do.
  local unfinished=$dir/.install_unfinished given; given=COHUB_$(cohub_policy_tag "$policy")_PY
  py=$(cohub_policy_python "$policy")
  if [ -x "$py" ] && { [ ! -e "$unfinished" ] || [ -n "${!given:-}" ]; }; then cohub_say "$policy: environment present ($py)"
  elif cohub_runner_dry; then cohub_say "$policy: install skipped (runner dry run)"; return 0
  else
    if [ -x "$py" ]; then cohub_say "$policy: an earlier install did not finish -- running install.sh again"
    else cohub_say "$policy: no interpreter at $py -- building the environment"; fi
    [ "${COHUB_DRY_RUN:-0}" = 1 ] || : > "$unfinished"
    case "$policy" in
      ACT|DP|LatentToM|GauDP)
        conda=$(cohub_conda_bin); [ -n "$conda" ] || { cohub_say "$policy needs conda (or set COHUB_$(cohub_policy_tag "$policy")_PY to a prepared interpreter)"; return 1; }
        env=$(cohub_policy_env "$policy")
        "$conda" env list 2>/dev/null | awk '{print $1}' | grep -qx "$env" || cohub_step "$conda" create -y -n "$env" python=3.10 || return
        # install.sh uses paths relative to its own directory.
        cohub_step "$conda" run --no-capture-output -n "$env" bash -c "cd '$dir' && COHUB_INSTALL_IN=$policy bash install.sh" || return ;;
      FastWAM) cohub_step env COHUB_INSTALL_IN=FastWAM bash "$dir/install.sh" "$(cohub_policy_env FastWAM)" || return ;;
      Pi_05|GR00T_N17|Psi0)
        command -v uv >/dev/null 2>&1 || { cohub_say "$policy needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; return 1; }
        cohub_step env COHUB_INSTALL_IN="$policy" bash "$dir/install.sh" || return ;;
    esac
    [ "${COHUB_DRY_RUN:-0}" = 1 ] || [ -x "$(cohub_policy_python "$policy")" ] || { cohub_say "$policy: install.sh finished but no interpreter at $(cohub_policy_python "$policy")"; return 1; }
    rm -f "$unfinished"
  fi
  # Pretrained weights. pi0.5 is absent: openpi fetches pi05_base itself.
  cohub_runner_dry && return 0
  local fw=$dir/FastWAM wan=${DIFFSYNTH_MODEL_BASE_PATH:-$dir/FastWAM/checkpoints}
  case "$policy" in
    GR00T_N17|Psi0) cohub_step "$(cohub_policy_python "$policy")" "$WT/baselines/utils/download_weights.py" "$policy" ;;
    FastWAM)
      cohub_step "$(cohub_policy_python FastWAM)" "$WT/baselines/utils/download_weights.py" FastWAM || return
      # Derive the ActionDiT backbone from the Wan weights once, on the CPU.
      # The model configs read it from FastWAM/checkpoints.
      if [ "${COHUB_DRY_RUN:-0}" = 1 ] || [ ! -f "$fw/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" ]; then
        (cd "$fw" && DIFFSYNTH_MODEL_BASE_PATH=$wan cohub_step "$(cohub_policy_python FastWAM)" scripts/preprocess_action_dit_backbone.py \
           --model-config configs/model/fastwam_noredirect.yaml \
           --output "$fw/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
           --device cpu --dtype bfloat16)
      fi ;;
  esac
}

cohub_pick_task_set() {     # <all|all3|task> -> export COHUB_TASK_SET for the shared policies
  local task_set; task_set=$(cohub_task_set "$1") || return 2
  export COHUB_TASK_SET=$task_set
}
cohub_data_step() {         # <POLICY> [task]
  local policy=$1 task=${2:-}
  cohub_runner_dry && { cohub_say "data: skipped (runner dry run)"; return 0; }
  # The others need the simulation python only to fetch; prepare_data.sh checks it.
  if cohub_is_multitask "$policy" || [ "$policy" = ACT ]; then cohub_require_sim || return; fi
  if cohub_is_multitask "$policy"; then
    [ -z "$task" ] || cohub_pick_task_set "$task" || return
    cohub_entry_run bash "$WT/baselines/policy/$policy/train_hook.sh" data || return
    case "$policy" in
      FastWAM)    cohub_step bash "$WT/baselines/policy/FastWAM/prepare_text_cache.sh" || return ;;
    esac
  else
    [ -n "$task" ] || { cohub_say "$policy trains one run per task: bash baselines/policy/$policy/process_data.sh <task>"; return 2; }
    case "$policy:$task" in
      GauDP:shared|GauDP:shared_gaussian) cohub_step bash "$WT/baselines/utils/prepare_data.sh" fetch ;;
      *) cohub_entry_run bash "$WT/baselines/policy/$policy/train_hook.sh" data "$task" ;;
    esac
  fi
}
cohub_train_step() {        # <POLICY> [task] [args...]  (GauDP: [stage]; `shared [all3]`)
  local policy=$1 task=${2:-}
  if [ $# -ge 2 ]; then shift 2; else shift; fi
  if cohub_is_multitask "$policy"; then
    [ -z "$task" ] || cohub_pick_task_set "$task" || return
    cohub_say "$policy trains one shared policy on the ${COHUB_TASK_SET:-all} task set"
    cohub_entry_run bash "$WT/baselines/policy/$policy/train_hook.sh"
  else
    [ -n "$task" ] || { cohub_say "$policy trains one run per task: bash baselines/policy/$policy/train.sh <task>"; return 2; }
    cohub_entry_run bash "$WT/baselines/policy/$policy/train_hook.sh" "$task" "$@"
  fi
}
cohub_parse_args() {        # <POLICY> [args...] -> COHUB_ARGS, with --tag <TAG> taken out and applied
  # --tag sets CKPT_TAG. For DP the tag is appended to its budget label.
  local policy=$1 tag=; shift; COHUB_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --tag)   [ $# -ge 2 ] || { cohub_say "--tag needs a value"; return 2; }; tag=$2; shift 2 ;;
      --tag=*) tag=${1#--tag=}; shift ;;
      *)       COHUB_ARGS+=("$1"); shift ;;
    esac
  done
  [ -n "$tag" ] || return 0
  export CKPT_TAG=$tag
  if [ "$policy" = DP ]; then export DP_RUN_TAG="$(cohub_budget DP)-$tag" CKPT_TAG="$(cohub_budget DP)-$tag"; fi
}
cohub_entry() {             # <install|data|train> <POLICY> [args...] -- policy/<POLICY>/<step>.sh, called the CoHuB way
  #   install.sh                                    the environment and the weights
  #   process_data.sh [task]                        the data its training reads
  #   train.sh [task] [--tag T]                     train
  local step=$1 policy=$2; shift 2
  WT=$(mh_wt); export COHUB_WT=$WT; cd "$WT"
  cohub_init "$policy"
  cohub_parse_args "$policy" "$@" || return; set -- ${COHUB_ARGS[@]+"${COHUB_ARGS[@]}"}
  case "$step" in
    install) cohub_install "$policy" ;;
    data)    cohub_data_step "$policy" "${1:-}" ;;
    train)   [ "${1:-}" != data ] || { cohub_say "the data is built by process_data.sh: bash process_data.sh ${2:-[task]}"; return 2; }
             cohub_train_step "$policy" "$@" ;;
  esac
}

# ---- training ---------------------------------------------------------------
cohub_train() {             # <POLICY> <task|multitask> <mode> [robot|-] [seed] -- run the training here, now
  local policy=$1 task=$2 mode=$3 robot=${4:--} seed=${5:-0}
  echo "[train] $policy $task $mode ${robot/-/} seed=$seed" >&2
  cohub_step bash "$WT/baselines/utils/train_runner.sh" "$policy" "$task" "$mode" "$robot" "$seed"
}
cohub_data() {              # <gr00t|pi05|psi0|fastwam|all> -- build that view
  # Views share the flattened dataset, so two of them must not run at once.
  cohub_step bash "$WT/baselines/utils/prepare_data.sh" "$1"
}
cohub_train_main() {        # <POLICY> <multitask|per_robot|centralized> [args...] -- a train_hook.sh's dispatch
  # multitask    `train` (the default) is the shared run over the task set
  #              (COHUB_TASK_SET), `data` builds that set's view.
  # per_robot    <task> is one run per robot the scene has; `data <task>` is
  #              policy/<POLICY>/process_data.sh <task>.
  # centralized  <task> is one run named for the task; `data <task>` likewise.
  local policy=$1 shape=$2 view task robot; shift 2
  if [ "$shape" = multitask ]; then
    case "${1:-train}" in
      train) cohub_train "$policy" "$(cohub_multitask_word)" decentralized - "${SEED:-0}" ;;
      data)  case "$policy" in Pi_05) view=pi05 ;; GR00T_N17) view=gr00t ;; Psi0) view=psi0 ;; FastWAM) view=fastwam ;; esac
             cohub_data "$view" ;;
      *)     cohub_usage ;;
    esac
    return
  fi
  case "${1:-}" in
    ""|-h|--help) cohub_usage ;;
    data) task=$(mh_task_word "${2:?task}") || return 2
          cohub_step env COHUB_DATA_IN="$policy" bash "$WT/baselines/policy/$policy/process_data.sh" "$task" ;;
    *)    task=$(mh_task_word "$1") || return 2
          if [ "$shape" = centralized ]; then
            cohub_train "$policy" "$task" centralized - "${SEED:-0}"
          else
            for robot in $(mh_robots "$(mh_scene "$task")"); do
              cohub_train "$policy" "$task" decentralized "$robot" "${SEED:-0}" || return $?
            done
          fi ;;
  esac
}

# ---- evaluation -------------------------------------------------------------
cohub_eval() {              # <POLICY> <task[,task...]|all> ["ckpt ..."] -- one evaluation per (task, checkpoint), in order
  # Episodes and seed come from COHUB_EVAL_EPISODES and COHUB_EVAL_SEED, the
  # checkpoints from the argument or COHUB_EVAL_CKPTS. EVAL_WANDB=0 skips the
  # wandb upload. The first failure stops the sweep.
  local policy=$1 tasks=$2 ckpts=${3:-${COHUB_EVAL_CKPTS:-}} task ckpt
  for ckpt in ${ckpts:-""}; do for task in $(cohub_tasks "$tasks"); do
    echo "[eval] $policy $task${ckpt:+ ckpt $ckpt}  (${COHUB_EVAL_EPISODES:-100} episodes, seed ${COHUB_EVAL_SEED:-1})" >&2
    EVAL_CKPT_NUM=$ckpt cohub_step bash "$WT/scripts/eval/runner.sh" "$policy" "$task" "${COHUB_EVAL_EPISODES:-100}" "${COHUB_EVAL_SEED:-1}" || return $?
  done; done
}
cohub_eval_main() {         # <POLICY> [task[,task...]|all] [ckpt ...] -- scripts/eval/eval.py's dispatch
  local policy=$1 tasks; shift
  tasks=${1:-all}; [ $# -gt 0 ] && shift
  cohub_eval "$policy" "$tasks" "${*:-${COHUB_EVAL_CKPTS:-}}"
}
cohub_groot_merge() {       # ["ckpt ..."] -- the LoRA merge GR00T's server needs first
  # Merges every step (policy/GR00T_N17/merge_lora.sh: a GPU and ~160 GB of host RAM).
  local ckpts=${1:-${COHUB_EVAL_CKPTS:-}}
  cohub_step bash "$WT/baselines/policy/GR00T_N17/merge_lora.sh" all "$(cohub_multitask_word)" "$ckpts" 0
}

# ---- logging ----------------------------------------------------------------
cohub_wandb_python() {      # [POLICY] -> an interpreter that can import wandb; the policy's own first
  local cand
  [ -n "${COHUB_WANDB_PY:-${COHUB_DP_PY:-}}" ] && { echo "${COHUB_WANDB_PY:-$COHUB_DP_PY}"; return; }
  for cand in ${1:+"$(cohub_policy_python "$1")"} "$(cohub_policy_python FastWAM)" python3; do
    command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import wandb; wandb.init" >/dev/null 2>&1 && { echo "$cand"; return; }
  done
  echo "no interpreter here can import wandb; set COHUB_WANDB_PY" >&2; return 1
}
cohub_upload() {            # <POLICY> <run dir> <min episodes> -- write results.json (and its mp4s) into wandb
  # upload.py refuses fewer than <min episodes> or a run of crashes.
  local policy=$1 dir=$2 min=$3 py
  py=$(cohub_wandb_python "$policy") || return 1
  "$py" "${COHUB_UPLOAD_PY:-$WT/baselines/logging/upload.py}" "$dir" --worktree "$WT" --policy "$policy" --seed "${CKPT_SEED:-0}" \
    --min-episodes "$min" --eval-run-name "$(cohub_wandb_name "$(cohub_multitask_word)" "$policy" eval)" \
    ${CKPT_TAG:+--ckpt-tag "$CKPT_TAG"} ${COHUB_EVAL_LABEL:+--label "$COHUB_EVAL_LABEL"}
}
