#!/usr/bin/env bash

# Shared argument parsing for the GauDP launchers. Sourced, not run.
#
# Artifacts:
#     data/cohub-<task>-unitree_g1x2_centralized-joint.hdf5
#     checkpoints/cohub-<task>-unitree_g1x2_centralized-joint-<seed>/
#
# Task names use the compact spelling (`framehang`, not `frame_hang`).

GAUDP_ACTION_TYPE=joint
GAUDP_DEFAULT_ENV_CFG=unitree_g1x2_centralized

gaudp_task_config() {
    task=$1
    case "${task}" in
        cocarry) scene=cocarry ;;
        handover|handover_easy|handovereasy) task=handover; scene=handover ;;
        door_passage|doorpassage) task=doorpassage; scene=doorpassage ;;
        frame_hang|framehang) task=framehang; scene=framehang ;;
        *) scene=${task} ;;
    esac
    env_cfg=${GAUDP_ENV_CFG:-${GAUDP_DEFAULT_ENV_CFG}}
}

gaudp_parse_stage_args() {
    if (( $# >= 6 )) && [[ "$4" == "joint" || "$4" == "ee" ]]; then
        # Long form: <bench> <ckpt> <env_cfg> <action_type> <seed> <gpu> [extra...]
        bench=$1
        ckpt=$2
        env_cfg=$3
        action_type=$4
        seed=$5
        gpu=$6
        task=${GAUDP_TASK:-${ckpt}}
        scene=${GAUDP_SCENE:-${task}}
        extra=("${@:7}")
        return
    fi

    # Short form: <task> [seed] [gpu] [extra...]
    task=${1:?usage: $0 <task> [seed] [gpu] [extra args]}
    shift
    gaudp_task_config "${task}"
    bench=${GAUDP_BENCH:-cohub}
    ckpt=${GAUDP_CKPT:-${scene}}
    action_type=${GAUDP_ACTION_TYPE}
    seed=${GAUDP_SEED:-0}
    gpu=${GAUDP_GPU:-0}
    if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then seed=$1; shift; fi
    if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then gpu=$1; shift; fi
    extra=("$@")
}

gaudp_parse_data_args() {
    if (( $# >= 4 )) && [[ "$4" == "joint" || "$4" == "ee" ]]; then
        # Long form: <bench> <ckpt> <env_cfg> <action_type> [task] [max_demos]
        bench=$1
        ckpt=$2
        env_cfg=$3
        action_type=$4
        task=${5:-${GAUDP_TASK:-${ckpt}}}
        max_demos=${6:-}
        # A numeric argument 5 is max_demos.
        if [[ "${task}" =~ ^[0-9]+$ ]]; then
            max_demos=${task}
            task=${GAUDP_TASK:-${ckpt}}
        fi
        scene=${GAUDP_SCENE:-${task}}
        return
    fi

    # Short form: <task> [max_demos]
    task=${1:?usage: $0 <task> [max_demos]}
    max_demos=${2:-}
    gaudp_task_config "${task}"
    bench=${GAUDP_BENCH:-cohub}
    ckpt=${GAUDP_CKPT:-${scene}}
    action_type=${GAUDP_ACTION_TYPE}
}

gaudp_require_joint_action_type() {
    if [[ "${action_type}" != "joint" ]]; then
        echo "[GauDP] action_type=${action_type} is no longer supported." >&2
        echo "[GauDP] GauDP trains and serves the centralized GR00T joint contract" >&2
        echo "[GauDP] (86D joint state / 70D absolute joint targets). The EE datasets and" >&2
        echo "[GauDP] policy checkpoints under *-ee-* are kept but cannot be resumed into" >&2
        echo "[GauDP] joint space -- convert with process_data.sh and retrain." >&2
        exit 2
    fi
}

# The run directory this stage writes into. GAUDP_TAG (CKPT_TAG) is appended.
gaudp_run_dir() {
    printf '%s\n' "${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${env_cfg}-${action_type}-${seed}${GAUDP_TAG:+-${GAUDP_TAG}}"
}

gaudp_data_path() {
    printf '%s\n' "${POLICY_DIR}/data/${bench}-${ckpt}-${env_cfg}-${action_type}.hdf5"
}

# Shared encoder directory (train_gaussian_shared.sh). Its name carries the
# camera_order unless that is [ego_a, ego_b], and GAUDP_SHARED_VARIANT when set.
gaudp_view_suffix() {   # [converted hdf5] -> "" or "-<view>-<view>..."
    local data=${1:-$(gaudp_data_path)}
    [[ -f "${data}" ]] || return 0
    PYTHONNOUSERSITE=1 "${GAUDP_PYTHON:-python}" - "${data}" <<'PY'
import json
import sys

import h5py

with h5py.File(sys.argv[1], "r") as source:
    order = json.loads(source.attrs["camera_order"])
print("" if order == ["ego_a", "ego_b"] else "-" + "-".join(order))
PY
}

gaudp_shared_gaussian_dir() {   # [view suffix]; default: this task's converted hdf5
    local views
    if (( $# )); then views=$1; else views="$(gaudp_view_suffix)"; fi
    printf '%s\n' "${GAUDP_SHARED_GAUSSIAN_DIR:-${POLICY_DIR}/checkpoints/${bench:-cohub}-shared-${env_cfg}-${action_type}-${seed}${GAUDP_TAG:+-${GAUDP_TAG}}${GAUDP_SHARED_VARIANT:+-${GAUDP_SHARED_VARIANT}}${views}}"
}

gaudp_gaussian_run_dirs() {
    if [[ "${GAUDP_SHARED_GAUSSIAN:-1}" == "1" ]]; then
        printf '%s\n' "$(gaudp_shared_gaussian_dir)"
    fi
    printf '%s\n' "$(gaudp_run_dir)"
    # Tagged runs stop here. Untagged runs also search these directories.
    [[ -n "${GAUDP_TAG:-}" ]] && return 0
    printf '%s\n' \
        "${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${scene}-ee-${seed}" \
        "${POLICY_DIR}/checkpoints/${scene}-experiment-${scene}-ee-${seed}"
    if [[ "${task}" == "handover" ]]; then
        printf '%s\n' \
            "${POLICY_DIR}/checkpoints/${bench}-handover_easy-handover_easy-ee-${seed}"
    fi
}

# First existing `gaussian/<name>` across gaudp_gaussian_run_dirs.
gaudp_find_gaussian_artifact() {
    local name=$1 candidate
    while read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        if [[ -s "${candidate}/gaussian/${name}" ]]; then
            printf '%s\n' "${candidate}/gaussian/${name}"
            return 0
        fi
    done < <(gaudp_gaussian_run_dirs)
    return 1
}

# The encoder a feature cache was extracted with.
gaudp_cache_checkpoint() {   # <features.hdf5>
    PYTHONNOUSERSITE=1 "${GAUDP_PYTHON:-python}" - "$1" <<'PY'
import sys

import h5py

with h5py.File(sys.argv[1], "r") as source:
    print(str(source.attrs.get("gaussian_checkpoint", "")))
PY
}

gaudp_no_shared_encoder() {
    echo "[GauDP] no shared encoder at $(gaudp_shared_gaussian_dir)/gaussian/best.ckpt" >&2
    echo "[GauDP] train it first: bash train_gaussian.sh   (three-robot tasks: bash train_gaussian.sh all3)" >&2
    echo "[GauDP] or name one with GAUDP_SHARED_GAUSSIAN_DIR or GAUDP_GAUSSIAN_CKPT, or GAUDP_SHARED_GAUSSIAN=0 for an encoder of this task's own" >&2
    exit 2
}
