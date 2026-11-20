from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
BASE = WT / "baselines"
POLICY_ROOT = BASE / "policy"
BUILT = HERE / "data"
MODALITY = HERE / "policy" / "GR00T_N17" / "modality.py"
ROBOTS = ("robot_a", "robot_b")

BATCH = {
    "ACT": ("ACT_BATCH_SIZE", "ACT_MICRO_BATCH", "ACT_GRAD_ACCUM"),
    "DP": ("DP_BATCH_SIZE", "DP_MICRO_BATCH", "DP_ACCUM"),
    "GR00T_N17": ("GLOBAL_BATCH_SIZE", "GR00T_MICRO_BATCH", "GRAD_ACCUM"),
}
CONVERT = {
    "local-shared": ("convert.py", ["--robot", "shared"]),
    "global-separate": ("convert.py", ["--robot", "each", "--view", "global"]),
    "global-shared": ("convert_lerobot.py", ["--robot", "duo", "--cameras", "ego_a,ego_b"]),
}
GROOT = {
    "local-separate": ("shared", ["--role", "{robot}"]),
    "global-separate": ("shared_global", ["--view", "global", "--role", "{robot}"]),
    "global-shared": ("duo", ["--rows", "pair"]),
}


def env_cfg(mode: str) -> str:
    return f"unitree_g1x2_{mode}"


def data_id(ckpt: str, mode: str) -> str:
    return f"cohub-{ckpt}-{env_cfg(mode)}-joint"


def runs(policy: str, cell: str, task: str) -> list[tuple[str, str, str]]:
    sharing = cell.split("-")[1]
    if policy == "GR00T_N17":
        if sharing == "shared":
            return [("decentralized", "-", "multitask")]
        return [("decentralized", robot, f"multitask_{robot}") for robot in ROBOTS]
    if cell == "local-shared":
        return [("decentralized", "shared", f"{task}_shared")]
    if cell == "global-shared":
        return [("centralized", "-", task)]
    return [("decentralized", robot, f"{task}_{robot}") for robot in ROBOTS]


def converted(policy: str, cell: str, task: str, robot: str) -> Path:
    fmt = {"ACT": "act", "DP": "diffusion_policy"}[policy]
    suffix = {"local-shared": "_shared", "global-separate": f"_global_{robot}", "global-shared": ""}[cell]
    return BUILT / cell / f"{task}_{fmt}{suffix}"


def is_baseline(policy: str, cell: str) -> bool:
    return cell == ("local-shared" if policy == "GR00T_N17" else "local-separate")


def train_data(policy: str, cell: str, task: str, mode: str, robot: str, ckpt: str) -> Path:
    if policy == "ACT":
        return BASE / "data" / f"{task}_act_{robot}" if is_baseline(policy, cell) else converted("ACT", cell, task, robot)
    if policy == "DP":
        if is_baseline(policy, cell):
            return POLICY_ROOT / "DP" / "data" / f"{data_id(ckpt, mode)}.zarr"
        return Path(f"{converted('DP', cell, task, robot)}.zarr")
    if is_baseline(policy, cell):
        return BASE / "data" / "gr00t" / data_id(ckpt, mode)
    return BUILT / cell / ckpt / "lerobot"


def run_name(ckpt: str) -> str:
    return f"cohub-{ckpt}-{os.environ['COHUB_RUN_SUFFIX']}"
