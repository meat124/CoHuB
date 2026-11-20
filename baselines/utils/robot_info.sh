#!/bin/bash
# Print one number of a robot profile in robot/_robot_info.json.
#   robot_info.sh <baselines dir> <profile> <action_dim|state_dim|num_cameras>
set -e
python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys

root, profile, field = sys.argv[1:4]
info = json.load(open(os.path.join(root, "utils", "robot", "_robot_info.json"), encoding="utf-8"))[profile]
action_dim = sum(info["arm_dim"]) + sum(info["ee_dim"])
print({
    "action_dim": action_dim,
    "state_dim": info.get("state_dim", action_dim),
    "num_cameras": info.get("num_cameras", 1),
}[field])
PY
