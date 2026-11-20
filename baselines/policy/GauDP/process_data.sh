#!/usr/bin/env bash
# Build GauDP's training data:  bash process_data.sh <task>
# No GPU needed. With COHUB_DATA_IN set, or in the long form `<bench> <ckpt> <env_cfg> joint [task] [max_demos]`, it runs the conversion below.
if [ -z "${COHUB_DATA_IN:-}" ] && ! [[ "${4:-}" == joint || "${4:-}" == ee ]]; then
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data GauDP "$@"; exit
fi
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${COHUB_DATA_IN:-}" = GauDP ]; then
  # Called with the task alone: fetch the dataset and keep a finished hdf5.
  WT=${COHUB_WT:-$(cd "$POLICY_DIR/../../.." && pwd)}; export COHUB_WT=$WT
  # shellcheck source=../../common.sh
  source "$WT/baselines/common.sh"
  TASK=$(mh_task_word "${1:?usage: process_data.sh <task>}")
  export GAUDP_PYTHON; GAUDP_PYTHON=$(cohub_policy_python GauDP)
  export GAUDP_BENCH=cohub
  GAUDP_ENV_CFG=$(mh_env_cfg_type centralized); export GAUDP_ENV_CFG
  [ -n "${COHUB_DATASET_PATH:-}" ] || cohub_fetch_task "$TASK"
  DATA=$POLICY_DIR/data/$(mh_data_id "$TASK" "$GAUDP_ENV_CFG").hdf5
  if [ -f "$DATA" ] && [ "${COHUB_DATA_REBUILD:-0}" != 1 ]; then
    echo "[prepare] GauDP $TASK: reusing ${DATA#$WT/} (COHUB_DATA_REBUILD=1 to rebuild)"; exit 0
  fi
  echo "[prepare] GauDP $TASK  python=$GAUDP_PYTHON"
  set -- "$TASK"
fi
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_data_args "$@"
gaudp_require_joint_action_type

COHUB_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"
DATASETS_ROOT="${GAUDP_DATASETS_ROOT:-${COHUB_ROOT}/datasets}"

is_lerobot_root() {
    [[ -f "$1/meta/info.json" && -d "$1/data" && -d "$1/videos" ]]
}

# Accept the LeRobot root itself or a task directory with a `lerobot/` child.
resolve_lerobot_root() {
    local candidate=$1
    if is_lerobot_root "${candidate}"; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if is_lerobot_root "${candidate}/lerobot"; then
        printf '%s\n' "${candidate}/lerobot"
        return 0
    fi
    return 1
}

# Dataset folders may use either spelling of a task name.
task_names=("${task}")
case "${task}" in
  door_passage) task_names+=(doorpassage) ;;
  frame_hang)   task_names+=(framehang) ;;
  doorpassage)  task_names+=(door_passage) ;;
  framehang)    task_names+=(frame_hang) ;;
  handover_easy) task_names+=(handover) ;;
esac

source_path=""
if [[ -n "${COHUB_DATASET_PATH:-}" ]]; then
    source_path="$(resolve_lerobot_root "${COHUB_DATASET_PATH}" || true)"
    searched=("${COHUB_DATASET_PATH}" "${COHUB_DATASET_PATH}/lerobot")
else
    searched=()
    for task_name in "${task_names[@]}"; do
        candidate="${DATASETS_ROOT}/${task_name}"
        searched+=("${candidate}" "${candidate}/lerobot")
        source_path="$(resolve_lerobot_root "${candidate}" || true)"
        if [[ -n "${source_path}" ]]; then
            break
        fi
    done
fi

if [[ -z "${source_path}" ]]; then
    echo "[GauDP] no LeRobot dataset found for task '${task}'" >&2
    echo "[GauDP] searched:" >&2
    printf '  - %s\n' "${searched[@]}" >&2
    echo "[GauDP] Expected meta/info.json, data/, and videos/ under the dataset root." >&2
    echo "[GauDP] Default location: ${DATASETS_ROOT}/<task>" >&2
    echo "[GauDP] Available task folders:" >&2
    find "${DATASETS_ROOT}" -mindepth 2 -maxdepth 3 -path '*/meta/info.json' \
        -printf '  - %h\n' 2>/dev/null | sed 's#/meta$##' >&2 || true
    echo "[GauDP] Pass the task as argument 5 or set COHUB_DATASET_PATH." >&2
    exit 2
fi
echo "[GauDP] task     ${task}"
echo "[GauDP] source   ${source_path}"
# Print which export this is (date, commit, size, task sentence).
prov=cohub_provenance.json; [[ -f "${source_path}/meta/$prov" ]] || prov=mhbench_provenance.json
if [[ -f "${source_path}/meta/$prov" ]]; then
  python3 - "${source_path}" "$prov" <<'PROV'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
prov = json.loads((root / "meta" / sys.argv[2]).read_text())
info = json.loads((root / "meta" / "info.json").read_text())
written = prov.get("env_args", {}).get("provenance", {})
print(f"[GauDP] exported {written.get('written_at', '?')[:19]} @ {str(written.get('commit', '?'))[:7]}"
      f"  {info.get('total_episodes')} episodes / {info.get('total_frames')} frames")
sentences = prov.get("task", {}).get("sentences") or ["?"]
print(f"[GauDP] task     {sentences[0]}")
PROV
fi

output="$(gaudp_data_path)"
extra=()
if [[ -n "${max_demos}" ]]; then extra+=(--max-demos "${max_demos}"); fi
if [[ "${GAUDP_USE_SCENE:-0}" == "1" ]]; then extra+=(--use-scene); fi
# GAUDP_CAMERAS overrides the view list, e.g. "ego_a,ego_b".
if [[ -n "${GAUDP_CAMERAS:-}" ]]; then extra+=(--cameras "${GAUDP_CAMERAS}"); fi

python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/process_data.py" "${source_path}" "${output}" "${extra[@]}"
[ "${COHUB_DATA_IN:-}" != GauDP ] || echo "[prepare] GauDP $TASK done"
