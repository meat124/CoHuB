#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines" / "logging"))
import upload as core  # noqa: E402

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--policy", default="DP")
cell, rest = pre.parse_known_args()
SHARING = os.environ.get("COHUB_SHARING")
cell.project = os.environ.get("WANDB_PROJECT")

_core_wandb_runs_for = core.wandb_runs_for


def run_dir_names(task: str, mode: str, seed: int, ckpt_tag: str | None, style: str = "shared") -> list[str]:
    ckpt_task = core.CKPT_TASK_ALIASES.get(task, task)
    suffix = os.environ["COHUB_RUN_SUFFIX"]
    if mode == "centralized":
        return [f"cohub-{ckpt_task}-{suffix}"]
    if style == "shared":
        robots = ("",) if SHARING == "shared" else ("_robot_a", "_robot_b")
        return [f"cohub-{core.MULTITASK_CKPT}{robot}-{suffix}" for robot in robots]
    robots = ("_shared",) if SHARING == "shared" else ("_robot_a", "_robot_b")
    return [f"cohub-{ckpt_task}{robot}-{suffix}" for robot in robots]


def wandb_runs_for(mode: str, task: str, seed: int, variant: str = "obs1") -> list[str]:
    if mode == "decentralized" and SHARING == "shared":
        return [f"dp-decentralized-{task}-shared-{variant}-seed{seed}"]
    return _core_wandb_runs_for(mode, task, seed, variant)


core.run_dir_names = run_dir_names
core.wandb_runs_for = wandb_runs_for
if cell.project:
    projects = core.POLICY_PROJECTS[cell.policy]
    if cell.project not in projects:
        projects.insert(0, cell.project)

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *rest, "--policy", cell.policy]
    raise SystemExit(core.main())
