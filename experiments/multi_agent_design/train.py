#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from design_config import BASE, BATCH, GROOT, MODALITY, env_cfg, is_baseline, runs, train_data
from utils import say, setup, step


def state_dim(mode: str) -> int:
    return int(subprocess.check_output(
        ["bash", str(BASE / "utils" / "robot_info.sh"), str(BASE), env_cfg(mode), "state_dim"], text=True))


def train_env(policy: str, cell: str, mode: str, ckpt: str, data: Path) -> list[str]:
    env = []
    if is_baseline(policy, cell):
        pass
    elif policy == "ACT":
        env += [f"ACT_DATA_SETTING=cohub-{ckpt}-{cell}"]
    elif policy == "DP":
        env += [f"DP_ZARR_PATH={data}"]
    else:
        env += [f"COHUB_ROBOT={GROOT[cell][0]}", f"GR00T_DATASET_PATH={data}", f"GR00T_MODALITY_CONFIG={MODALITY}"]
    if policy == "ACT" and cell == "global-separate":
        env += [f"ACT_STATE_DIM={2 * state_dim(mode)}"]
    if policy == "DP" and cell == "global-separate":
        env += [f"DP_STATE_DIM={2 * state_dim(mode)}", "DP_NUM_CAMERAS=2"]
    batch, micro, accum = BATCH[policy]
    size = os.environ.get(micro) or (os.environ.get("MAD_GLOBAL_MICRO_BATCH") if cell.startswith("global") else None)
    if size:
        if not size.isdigit() or int(size) == 0 or int(os.environ[batch]) % int(size):
            sys.exit(f"{micro}={size} has to be a positive divisor of {batch}={os.environ[batch]}")
        env += [f"{batch}={size}", f"{accum}={int(os.environ[batch]) // int(size)}"]
    return env


def train(policy: str, cell: str, task: str) -> None:
    word = "multitask" if policy == "GR00T_N17" else task
    seed = os.environ.get("SEED", "0")
    for mode, robot, ckpt in runs(policy, cell, task):
        say(f"[train] {policy} {word} {mode} {robot.replace('-', '')} seed={seed} cell={cell}")
        data = train_data(policy, cell, task, mode, robot, ckpt)
        step("bash", BASE / "utils" / "train_runner.sh", policy, word, mode, robot, seed, env={
            "COHUB_CKPT_NAME": ckpt,
            "COHUB_TRAIN_DATA": str(data / "meta" / "stats.json" if policy == "GR00T_N17" else data),
            "COHUB_TRAIN_ENV": "\n".join(train_env(policy, cell, mode, ckpt, data)),
        })


if __name__ == "__main__":
    args = setup()
    train(args.policy, args.cell, args.task)
