#!/bin/bash
# Build LatentToM's training data:  bash process_data.sh <task> [max_demos]
# Downloads the dataset when it is missing and writes the task's replay buffer
# (data/cohub-<task>-<task>-joint/{replay_buffer.zarr,videos/}); no GPU. A
# finished one is kept; COHUB_DATA_REBUILD=1 converts again.
if [ "${COHUB_DATA_IN:-}" != LatentToM ]; then   # called from outside: load the config first
  set -euo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"; cohub_entry data LatentToM "$@"; exit
fi
set -euo pipefail

TASK=${1:?usage: process_data.sh <task> [max_demos]}
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT=${COHUB_WT:-$(cd "$POLICY_DIR/../../.." && pwd)}; export COHUB_WT=$WT
# shellcheck source=../../common.sh
source "$WT/baselines/common.sh"
TASK=$(mh_task_word "$TASK")
cohub_two_robot_only LatentToM "$TASK"

# The LeRobot export to convert. COHUB_DATASET_PATH names one kept elsewhere.
[ -n "${COHUB_DATASET_PATH:-}" ] || cohub_fetch_task "$TASK"
COHUB_DATASET_PATH=${COHUB_DATASET_PATH:-$WT/datasets/$TASK/lerobot}
if [[ ! -d "${COHUB_DATASET_PATH}/meta" ]]; then
  echo "[LatentToM] no LeRobot export at ${COHUB_DATASET_PATH} (needs meta/ beside data/)." >&2
  exit 1
fi

# ckpt_name and env_cfg_type are both the task.
OUT=$POLICY_DIR/data/cohub-$TASK-$TASK-joint
if [ -f "$OUT/.complete" ] && [ "${COHUB_DATA_REBUILD:-0}" != 1 ]; then
  echo "[prepare] LatentToM $TASK: reusing ${OUT#$WT/} (COHUB_DATA_REBUILD=1 to rebuild)"; exit 0
fi
PY=$(cohub_policy_python LatentToM)
[ -x "$PY" ] || PY=python
echo "[prepare] LatentToM $TASK  python=$PY"

# Print which export this is. Older exports name the file mhbench_provenance.json.
prov=cohub_provenance.json; [[ -f "${COHUB_DATASET_PATH}/meta/$prov" ]] || prov=mhbench_provenance.json
if [[ -f "${COHUB_DATASET_PATH}/meta/$prov" ]]; then
  python3 - "${COHUB_DATASET_PATH}" "$prov" <<'PROV'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
prov = json.loads((root / "meta" / sys.argv[2]).read_text())
info = json.loads((root / "meta" / "info.json").read_text())
written = prov.get("env_args", {}).get("provenance", {})
print(f"[LatentToM] source   {root}")
print(f"[LatentToM] exported {written.get('written_at', '?')[:19]} @ {str(written.get('commit', '?'))[:7]}"
      f"  {info.get('total_episodes')} episodes / {info.get('total_frames')} frames")
sentences = prov.get("task", {}).get("sentences") or ["?"]
print(f"[LatentToM] task     {sentences[0]}")
PROV
fi

"$PY" "$POLICY_DIR/convert_to_replay_buffer.py" "$COHUB_DATASET_PATH" "$OUT" ${2:+--max-demos "$2"}
: > "$OUT/.complete"
echo "[prepare] LatentToM $TASK done: ${OUT#$WT/}/{replay_buffer.zarr,videos/}"
